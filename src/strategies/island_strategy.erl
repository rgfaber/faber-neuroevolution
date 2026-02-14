%% @doc Island model evolution strategy implementation.
%%
%% The island model runs multiple isolated subpopulations (islands) in parallel,
%% with periodic migration of individuals between islands. This maintains
%% diversity and enables exploration of multiple fitness peaks simultaneously.
%%
%% Key features:
%% - Multiple islands, each running its own sub-strategy
%% - Configurable migration topology (ring, full, random, custom)
%% - Various migrant selection methods (best, random, diverse)
%% - Periodic migration events based on evaluation count
%%
%% Migration topologies:
%% - ring: Each island sends to the next (circular)
%% - full: Every island can send to every other island
%% - random: Random destination for each migration
%% - custom: User-specified connections
%%
%% @author R.G. Lefever
%% @copyright 2024-2026 R.G. Lefever
-module(island_strategy).
-behaviour(evolution_strategy).

-include("neuroevolution.hrl").
-include("evolution_strategy.hrl").
-include("lifecycle_events.hrl").

%% evolution_strategy callbacks
-export([
    init/1,
    handle_evaluation_result/3,
    tick/1,
    get_population_snapshot/1,
    get_meta_inputs/1,
    apply_meta_params/2,
    terminate/2
]).

%%% ============================================================================
%%% State Records
%%% ============================================================================

%% State for a single island
-record(island, {
    id :: island_id(),
    strategy_module :: module(),
    strategy_state :: term(),
    evaluations_since_migration = 0 :: non_neg_integer()
}).

%% Main strategy state
-record(island_state, {
    %% Configuration
    config :: neuro_config(),
    params :: island_params(),
    network_factory :: module(),

    %% Islands (map from island_id to island record)
    islands :: #{island_id() => #island{}},

    %% Migration topology (precomputed connections)
    %% Maps source island to list of destination islands
    topology :: #{island_id() => [island_id()]},

    %% Total evaluations across all islands
    total_evaluations = 0 :: non_neg_integer(),

    %% Best fitness ever seen (across all islands)
    best_fitness_ever = 0.0 :: float()
}).

%%% ============================================================================
%%% Behaviour Callbacks
%%% ============================================================================

%% @doc Initialize the island model strategy.
-spec init(Config :: map()) -> {ok, #island_state{}, [lifecycle_event()]}.
init(Config) ->
    NeuroConfig = maps:get(neuro_config, Config),
    Params = parse_params(maps:get(strategy_params, Config, #{})),
    NetworkFactory = maps:get(network_factory, Config, network_factory),

    %% Create islands
    IslandCount = Params#island_params.island_count,
    PopPerIsland = Params#island_params.population_per_island,
    SubStrategy = Params#island_params.island_strategy,
    SubParams = Params#island_params.island_strategy_params,

    %% Create sub-config for each island
    IslandConfig = fun(IslandId) ->
        %% Modify neuro_config for island population size
        IslandNeuroConfig = NeuroConfig#neuro_config{
            population_size = PopPerIsland
        },
        #{
            neuro_config => IslandNeuroConfig,
            strategy_params => SubParams,
            network_factory => NetworkFactory,
            island_id => IslandId
        }
    end,

    %% Initialize all islands and collect events
    {Islands, AllEvents} = lists:foldl(
        fun(IslandId, {AccIslands, AccEvents}) ->
            {ok, SubState, SubEvents} = SubStrategy:init(IslandConfig(IslandId)),
            Island = #island{
                id = IslandId,
                strategy_module = SubStrategy,
                strategy_state = SubState
            },
            %% Tag events with island_id in metadata
            TaggedEvents = tag_events_with_island(SubEvents, IslandId),
            {AccIslands#{IslandId => Island}, AccEvents ++ TaggedEvents}
        end,
        {#{}, []},
        lists:seq(1, IslandCount)
    ),

    %% Build topology
    Topology = build_topology(Params),

    State = #island_state{
        config = NeuroConfig,
        params = Params,
        network_factory = NetworkFactory,
        islands = Islands,
        topology = Topology
    },

    {ok, State, AllEvents}.

%% @doc Handle an individual evaluation result.
%%
%% Routes the result to the appropriate island based on individual ID.
-spec handle_evaluation_result(IndividualId, FitnessResult, State) -> Result when
    IndividualId :: individual_id(),
    FitnessResult :: #{fitness := float(), metrics => map()},
    State :: #island_state{},
    Result :: {[strategy_action()], [lifecycle_event()], #island_state{}}.
handle_evaluation_result(IndividualId, FitnessResult, State) ->
    Fitness = maps:get(fitness, FitnessResult),

    %% Find which island owns this individual
    case find_island_for_individual(IndividualId, State) of
        {ok, IslandId, Island} ->
            SubModule = Island#island.strategy_module,
            SubState = Island#island.strategy_state,

            %% Delegate to sub-strategy
            {SubActions, SubEvents, NewSubState} = SubModule:handle_evaluation_result(
                IndividualId, FitnessResult, SubState
            ),

            %% Update island state
            NewIsland = Island#island{
                strategy_state = NewSubState,
                evaluations_since_migration = Island#island.evaluations_since_migration + 1
            },

            %% Tag events with island_id
            TaggedEvents = tag_events_with_island(SubEvents, IslandId),

            %% Update best fitness
            BestFitness = max(Fitness, State#island_state.best_fitness_ever),

            %% Per-individual evaluations (PRIMARY progress metric)
            Config = State#island_state.config,
            EvalsPerIndividual = Config#neuro_config.evaluations_per_individual,

            NewState = State#island_state{
                islands = maps:put(IslandId, NewIsland, State#island_state.islands),
                total_evaluations = State#island_state.total_evaluations + EvalsPerIndividual,
                best_fitness_ever = BestFitness
            },

            %% Check if migration should occur
            {MigrationActions, MigrationEvents, FinalState} = maybe_migrate(IslandId, NewState),

            {SubActions ++ MigrationActions, TaggedEvents ++ MigrationEvents, FinalState};

        not_found ->
            %% Individual not found in any island
            {[], [], State}
    end.

%% @doc Periodic tick for maintenance operations.
-spec tick(State :: #island_state{}) -> {[strategy_action()], [lifecycle_event()], #island_state{}}.
tick(State) ->
    %% Tick all islands
    {AllActions, AllEvents, NewIslands} = maps:fold(
        fun(IslandId, Island, {AccActions, AccEvents, AccIslands}) ->
            SubModule = Island#island.strategy_module,
            {SubActions, SubEvents, NewSubState} = SubModule:tick(Island#island.strategy_state),
            TaggedEvents = tag_events_with_island(SubEvents, IslandId),
            NewIsland = Island#island{strategy_state = NewSubState},
            {AccActions ++ SubActions, AccEvents ++ TaggedEvents, AccIslands#{IslandId => NewIsland}}
        end,
        {[], [], #{}},
        State#island_state.islands
    ),

    {AllActions, AllEvents, State#island_state{islands = NewIslands}}.

%% @doc Get a snapshot of the population across all islands.
-spec get_population_snapshot(State :: #island_state{}) -> population_snapshot().
get_population_snapshot(State) ->
    %% Collect snapshots from all islands
    IslandSnapshots = maps:fold(
        fun(IslandId, Island, Acc) ->
            SubModule = Island#island.strategy_module,
            Snapshot = SubModule:get_population_snapshot(Island#island.strategy_state),
            [{IslandId, Snapshot} | Acc]
        end,
        [],
        State#island_state.islands
    ),

    %% Aggregate statistics
    AllIndividuals = lists:flatmap(
        fun({IslandId, Snapshot}) ->
            %% Add island_id to each individual summary
            [maps:put(island_id, IslandId, Ind) || Ind <- maps:get(individuals, Snapshot)]
        end,
        IslandSnapshots
    ),

    TotalSize = length(AllIndividuals),
    AllFitnesses = [maps:get(fitness, Ind) || Ind <- AllIndividuals],

    {BestFitness, AvgFitness, WorstFitness} = case AllFitnesses of
        [] -> {0.0, 0.0, 0.0};
        _ -> {lists:max(AllFitnesses),
              lists:sum(AllFitnesses) / length(AllFitnesses),
              lists:min(AllFitnesses)}
    end,

    #{
        size => TotalSize,
        individuals => AllIndividuals,
        best_fitness => BestFitness,
        avg_fitness => AvgFitness,
        worst_fitness => WorstFitness,
        species_count => maps:size(State#island_state.islands),  % Islands as "species"
        generation => State#island_state.total_evaluations,
        extra => #{
            island_count => maps:size(State#island_state.islands),
            best_fitness_ever => State#island_state.best_fitness_ever,
            island_snapshots => IslandSnapshots
        }
    }.

%% @doc Get normalized inputs for meta-controller.
-spec get_meta_inputs(State :: #island_state{}) -> [float()].
get_meta_inputs(State) ->
    Snapshot = get_population_snapshot(State),

    BestFitness = maps:get(best_fitness, Snapshot),
    AvgFitness = maps:get(avg_fitness, Snapshot),

    %% Calculate diversity (variance across island best fitnesses)
    IslandSnapshots = maps:get(island_snapshots, maps:get(extra, Snapshot)),
    IslandBests = [maps:get(best_fitness, S) || {_, S} <- IslandSnapshots],
    Diversity = case length(IslandBests) > 1 of
        true -> calculate_variance(IslandBests);
        false -> 0.0
    end,
    NormalizedDiversity = min(1.0, Diversity / 100.0),

    %% Improvement gap
    ImprovementGap = case AvgFitness == 0.0 of
        true -> 0.0;
        false -> min(1.0, (BestFitness - AvgFitness) / max(1.0, AvgFitness))
    end,

    %% Migration pressure (how often migrations occur)
    Params = State#island_state.params,
    MigrationInterval = Params#island_params.migration_interval,
    NormalizedMigration = min(1.0, 50.0 / max(1.0, to_float(MigrationInterval))),

    [
        NormalizedDiversity,     % Inter-island diversity
        ImprovementGap,          % Gap between best and average
        NormalizedMigration,     % Migration frequency
        min(1.0, BestFitness / 1000.0)  % Normalized best fitness
    ].

%% @doc Apply parameter updates from meta-controller.
-spec apply_meta_params(MetaParams :: map(), State :: #island_state{}) -> #island_state{}.
apply_meta_params(MetaParams, State) ->
    %% Apply params to all islands
    NewIslands = maps:map(
        fun(_IslandId, Island) ->
            SubModule = Island#island.strategy_module,
            NewSubState = SubModule:apply_meta_params(MetaParams, Island#island.strategy_state),
            Island#island{strategy_state = NewSubState}
        end,
        State#island_state.islands
    ),

    %% Update migration interval if provided
    Params = State#island_state.params,
    NewMigrationInterval = case maps:get(migration_interval, MetaParams, undefined) of
        undefined -> Params#island_params.migration_interval;
        Interval -> max(10, min(500, round(Interval)))
    end,

    NewParams = Params#island_params{migration_interval = NewMigrationInterval},

    State#island_state{
        islands = NewIslands,
        params = NewParams
    }.

%% @doc Clean up when strategy terminates.
-spec terminate(Reason :: term(), State :: #island_state{}) -> ok.
terminate(Reason, State) ->
    %% Terminate all islands
    maps:foreach(
        fun(_IslandId, Island) ->
            SubModule = Island#island.strategy_module,
            SubModule:terminate(Reason, Island#island.strategy_state)
        end,
        State#island_state.islands
    ),
    ok.

%%% ============================================================================
%%% Internal Functions - Initialization
%%% ============================================================================

%% @private Parse strategy params from map or record.
parse_params(Params) when is_map(Params) ->
    #island_params{
        island_count = maps:get(island_count, Params, 4),
        population_per_island = maps:get(population_per_island, Params, 25),
        migration_interval = maps:get(migration_interval, Params, 50),
        migration_count = maps:get(migration_count, Params, 2),
        migration_selection = maps:get(migration_selection, Params, best),
        topology = maps:get(topology, Params, ring),
        custom_connections = maps:get(custom_connections, Params, []),
        island_strategy = maps:get(island_strategy, Params, generational_strategy),
        island_strategy_params = maps:get(island_strategy_params, Params, #{})
    };
parse_params(Params) when is_record(Params, island_params) ->
    Params;
parse_params(_) ->
    #island_params{}.

%% @private Build migration topology.
build_topology(Params) ->
    IslandCount = Params#island_params.island_count,
    Islands = lists:seq(1, IslandCount),

    case Params#island_params.topology of
        ring ->
            %% Each island sends to the next one (circular)
            maps:from_list([
                {I, [((I rem IslandCount) + 1)]} || I <- Islands
            ]);

        full ->
            %% Every island can send to every other
            maps:from_list([
                {I, [J || J <- Islands, J =/= I]} || I <- Islands
            ]);

        random ->
            %% Will pick random destination at migration time
            maps:from_list([{I, Islands -- [I]} || I <- Islands]);

        custom ->
            %% Use custom connections
            Connections = Params#island_params.custom_connections,
            lists:foldl(
                fun({From, To}, Acc) ->
                    Existing = maps:get(From, Acc, []),
                    maps:put(From, [To | Existing], Acc)
                end,
                #{},
                Connections
            )
    end.

%%% ============================================================================
%%% Internal Functions - Migration
%%% ============================================================================

%% @private Check if migration should occur for an island.
maybe_migrate(IslandId, State) ->
    Island = maps:get(IslandId, State#island_state.islands),
    Params = State#island_state.params,
    MigrationInterval = Params#island_params.migration_interval,

    case Island#island.evaluations_since_migration >= MigrationInterval of
        true ->
            perform_migration(IslandId, State);
        false ->
            {[], [], State}
    end.

%% @private Perform migration from an island.
perform_migration(SourceIslandId, State) ->
    Params = State#island_state.params,
    Topology = State#island_state.topology,

    %% Get destination islands
    Destinations = maps:get(SourceIslandId, Topology, []),

    case Destinations of
        [] ->
            %% No destinations, just reset counter
            SourceIsland = maps:get(SourceIslandId, State#island_state.islands),
            NewSourceIsland = SourceIsland#island{evaluations_since_migration = 0},
            NewState = State#island_state{
                islands = maps:put(SourceIslandId, NewSourceIsland, State#island_state.islands)
            },
            {[], [], NewState};

        _ ->
            %% Pick destination (random for random topology, first for others)
            DestIslandId = case Params#island_params.topology of
                random -> lists:nth(rand:uniform(length(Destinations)), Destinations);
                _ -> hd(Destinations)
            end,

            %% Select migrants from source
            MigrationCount = Params#island_params.migration_count,
            {Migrants, NewSourceState} = select_migrants(SourceIslandId, MigrationCount, State),

            case Migrants of
                [] ->
                    {[], [], State};
                _ ->
                    %% Insert migrants into destination
                    {NewDestState, InsertEvents} = insert_migrants(DestIslandId, Migrants, NewSourceState),

                    %% Create migration events
                    MigrationEvents = [
                        #island_migration{
                            individual_id = Ind#individual.id,
                            from_island = SourceIslandId,
                            to_island = DestIslandId,
                            fitness = Ind#individual.fitness,
                            timestamp = erlang:timestamp()
                        } || Ind <- Migrants
                    ],

                    %% Reset source island's migration counter
                    SourceIsland = maps:get(SourceIslandId, NewDestState#island_state.islands),
                    FinalSourceIsland = SourceIsland#island{evaluations_since_migration = 0},
                    FinalState = NewDestState#island_state{
                        islands = maps:put(SourceIslandId, FinalSourceIsland, NewDestState#island_state.islands)
                    },

                    {[], MigrationEvents ++ InsertEvents, FinalState}
            end
    end.

%% @private Select migrants from an island.
select_migrants(IslandId, Count, State) ->
    Params = State#island_state.params,
    Island = maps:get(IslandId, State#island_state.islands),
    SubModule = Island#island.strategy_module,
    Snapshot = SubModule:get_population_snapshot(Island#island.strategy_state),

    Individuals = maps:get(individuals, Snapshot),
    SelectionMethod = Params#island_params.migration_selection,

    %% Get actual individual records (need to extract from sub-strategy state)
    %% For now, we'll create copies based on snapshot info
    %% In a real implementation, we'd need access to actual networks

    SelectedSummaries = case SelectionMethod of
        best ->
            %% Sort by fitness, take top N
            Sorted = lists:sort(
                fun(A, B) -> maps:get(fitness, A) >= maps:get(fitness, B) end,
                Individuals
            ),
            lists:sublist(Sorted, min(Count, length(Sorted)));

        random ->
            %% Random selection
            random_sample(Individuals, min(Count, length(Individuals)));

        diverse ->
            %% Select individuals with diverse fitness values
            %% Simple implementation: sort by fitness, take evenly spaced
            Sorted = lists:sort(
                fun(A, B) -> maps:get(fitness, A) >= maps:get(fitness, B) end,
                Individuals
            ),
            select_diverse(Sorted, Count)
    end,

    %% Convert summaries to individuals (simplified - would need network access)
    Migrants = [summary_to_individual(S) || S <- SelectedSummaries],

    {Migrants, State}.

%% @private Insert migrants into destination island.
insert_migrants(DestIslandId, Migrants, State) ->
    Island = maps:get(DestIslandId, State#island_state.islands),

    %% Create birth events for migrants
    BirthEvents = [
        #individual_born{
            id = Ind#individual.id,
            parent_ids = [],
            origin = migration,
            timestamp = erlang:timestamp(),
            metadata = #{from_island => undefined, fitness => Ind#individual.fitness}
        } || Ind <- Migrants
    ],

    %% Tag with destination island
    TaggedEvents = tag_events_with_island(BirthEvents, DestIslandId),

    %% Note: In a full implementation, we would need to actually insert
    %% the migrants into the sub-strategy's population. This would require
    %% the sub-strategy to support an insert_individual callback.
    %% For now, we just emit the events.

    NewIsland = Island,  % Would update sub-strategy state here

    NewState = State#island_state{
        islands = maps:put(DestIslandId, NewIsland, State#island_state.islands)
    },

    {NewState, TaggedEvents}.

%% @private Convert a snapshot summary to an individual record.
summary_to_individual(Summary) ->
    #individual{
        id = maps:get(id, Summary),
        fitness = maps:get(fitness, Summary),
        network = undefined,  % Would need actual network
        is_survivor = maps:get(is_survivor, Summary, false),
        is_offspring = maps:get(is_offspring, Summary, false)
    }.

%%% ============================================================================
%%% Internal Functions - Utilities
%%% ============================================================================

%% @private Find which island owns an individual.
find_island_for_individual(IndividualId, State) ->
    Result = maps:fold(
        fun(IslandId, Island, Acc) ->
            case Acc of
                {ok, _, _} -> Acc;  % Already found
                not_found ->
                    SubModule = Island#island.strategy_module,
                    Snapshot = SubModule:get_population_snapshot(Island#island.strategy_state),
                    Individuals = maps:get(individuals, Snapshot),
                    case lists:any(fun(Ind) -> maps:get(id, Ind) =:= IndividualId end, Individuals) of
                        true -> {ok, IslandId, Island};
                        false -> not_found
                    end
            end
        end,
        not_found,
        State#island_state.islands
    ),
    Result.

%% @private Tag events with island_id in metadata.
tag_events_with_island(Events, IslandId) ->
    lists:map(
        fun(Event) ->
            case Event of
                #individual_born{metadata = M} ->
                    Event#individual_born{metadata = M#{island_id => IslandId}};
                #individual_died{metadata = M} ->
                    Event#individual_died{metadata = M#{island_id => IslandId}};
                #individual_evaluated{metadata = M} ->
                    Event#individual_evaluated{metadata = M#{island_id => IslandId}};
                _ ->
                    Event
            end
        end,
        Events
    ).

%% @private Calculate variance.
calculate_variance([]) -> 0.0;
calculate_variance([_]) -> 0.0;
calculate_variance(Values) ->
    Mean = lists:sum(Values) / length(Values),
    SumSquaredDiffs = lists:sum([(V - Mean) * (V - Mean) || V <- Values]),
    SumSquaredDiffs / length(Values).

%% @private Random sample from list.
random_sample(List, N) when N >= length(List) ->
    List;
random_sample(List, N) ->
    random_sample(List, N, []).

random_sample(_List, 0, Acc) ->
    Acc;
random_sample(List, N, Acc) ->
    Index = rand:uniform(length(List)),
    Item = lists:nth(Index, List),
    RemainingList = lists:delete(Item, List),
    random_sample(RemainingList, N - 1, [Item | Acc]).

%% @private Select evenly spaced diverse individuals.
select_diverse(Sorted, Count) when length(Sorted) =< Count ->
    Sorted;
select_diverse(Sorted, Count) ->
    Len = length(Sorted),
    Step = Len / Count,
    Indices = [round(I * Step) || I <- lists:seq(0, Count - 1)],
    [lists:nth(min(Len, max(1, Idx + 1)), Sorted) || Idx <- Indices].

%% @private Convert integer to float.
to_float(N) when is_integer(N) -> N * 1.0;
to_float(F) when is_float(F) -> F.
