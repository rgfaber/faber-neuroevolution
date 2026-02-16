%% @doc Novelty Search evolution strategy.
%%
%% Novelty Search replaces fitness-based selection with novelty-based selection.
%% Instead of selecting for the fittest individuals, it selects for those with
%% the most novel behaviors - behaviors that are different from both the current
%% population and an archive of previously seen behaviors.
%%
%% This approach is particularly effective for:
%% - Deceptive fitness landscapes where fitness gradients lead to local optima
%% - Open-ended exploration where diverse solutions are valuable
%% - Problems where the path to the solution is not clear
%%
%% == Behavior Descriptors ==
%%
%% The evaluator must return a behavior descriptor in the metrics map:
%% #{fitness => F, metrics => #{behavior => [float(), ...]}}
%%
%% The behavior descriptor is a vector characterizing the individual's behavior.
%% For example:
%% - For a maze robot: final (x, y) position
%% - For a game AI: action frequencies, states visited
%% - For neural networks: activation patterns
%%
%% == Novelty Calculation ==
%%
%% Novelty is the average distance to the k-nearest neighbors in behavior space.
%% Neighbors come from both the current population and the archive.
%%
%% novelty(ind) = avg(distance(ind, neighbor_i)) for i in 1..k
%%
%% == Hybrid Mode ==
%%
%% When include_fitness=true and fitness_weight > 0, selection is based on:
%% score = (1 - fitness_weight) * novelty + fitness_weight * fitness
%%
%% @author R.G. Lefever
%% @copyright 2024-2026 R.G. Lefever
-module(novelty_strategy).
-behaviour(evolution_strategy).

-include("evolution_strategy.hrl").
-include("lifecycle_events.hrl").
-include("neuroevolution.hrl").

%% Behaviour callbacks
-export([
    init/1,
    handle_evaluation_result/3,
    tick/1,
    get_population_snapshot/1,
    get_meta_inputs/1,
    apply_meta_params/2
]).

%% Internal state record
-record(novelty_state, {
    %% Configuration
    config :: neuro_config(),
    params :: novelty_params(),

    %% Network factory module
    network_factory = network_factory :: module(),

    %% Population management
    population = [] :: [individual()],
    population_map = #{} :: #{individual_id() => individual()},  %% O(1) lookup
    population_size :: pos_integer(),

    %% Generation tracking
    generation = 1 :: pos_integer(),

    %% Evaluation state
    evaluated_count = 0 :: non_neg_integer(),

    %% Novelty archive - list of behavior descriptors
    archive = [] :: [behavior_descriptor()],

    %% Statistics
    best_novelty = 0.0 :: float(),
    avg_novelty = 0.0 :: float(),
    best_fitness = 0.0 :: float(),
    archive_adds = 0 :: non_neg_integer()
}).

%% Behavior descriptor type
-type behavior_descriptor() :: {individual_id(), [float()]}.

%%% ============================================================================
%%% Behaviour Callbacks
%%% ============================================================================

%% @doc Initialize the novelty search strategy.
%%
%% Expects config map with:
%% - neuro_config - the full neuroevolution config
%% - strategy_params - optional novelty_params record or map
%% - network_factory - optional module for network creation
-spec init(Config :: map()) -> {ok, #novelty_state{}, [lifecycle_event()]} | {error, term()}.
init(Config) ->
    NeuroConfig = maps:get(neuro_config, Config),

    %% Parse strategy params
    Params = parse_params(maps:get(strategy_params, Config, #{})),

    %% Get network factory
    NetworkFactory = maps:get(network_factory, Config, network_factory),

    %% Create initial population
    Population = create_initial_population(NeuroConfig, NetworkFactory),
    PopulationMap = build_population_map(Population),

    State = #novelty_state{
        config = NeuroConfig,
        params = Params,
        network_factory = NetworkFactory,
        population = Population,
        population_map = PopulationMap,
        population_size = NeuroConfig#neuro_config.population_size
    },

    %% Emit individual_born events for initial population
    BirthEvents = [create_birth_event(Ind, initial) || Ind <- Population],

    {ok, State, BirthEvents}.

%% @doc Handle an individual evaluation result.
%%
%% Accumulates behavior descriptors, computes novelty scores when all are evaluated,
%% then performs novelty-based selection and breeding.
-spec handle_evaluation_result(
    IndividualId :: individual_id(),
    FitnessResult :: map(),
    State :: #novelty_state{}
) -> strategy_result().
handle_evaluation_result(IndividualId, FitnessResult, State) ->
    Fitness = maps:get(fitness, FitnessResult),
    Metrics = maps:get(metrics, FitnessResult, #{}),
    Behavior = maps:get(behavior, Metrics, undefined),

    %% O(1) lookup in population map
    PopMap = State#novelty_state.population_map,
    case maps:find(IndividualId, PopMap) of
        {ok, Individual} ->
            %% Update individual with fitness and behavior
            NewMetrics = case Behavior of
                undefined -> Individual#individual.metrics;
                _ -> maps:put(behavior, Behavior, Individual#individual.metrics)
            end,
            UpdatedInd = Individual#individual{fitness = Fitness, metrics = NewMetrics},

            %% Update both map and list
            NewPopMap = maps:put(IndividualId, UpdatedInd, PopMap),
            UpdatedPop = update_individual_in_list(UpdatedInd, State#novelty_state.population),

            %% Track evaluation progress
            NewEvaluatedCount = State#novelty_state.evaluated_count + 1,

            %% Create individual_evaluated event
            EvalEvent = #individual_evaluated{
                id = IndividualId,
                fitness = Fitness,
                metrics = Metrics,
                timestamp = erlang:timestamp(),
                metadata = #{generation => State#novelty_state.generation}
            },

            %% Check if all individuals have been evaluated
            PopSize = State#novelty_state.population_size,
            case NewEvaluatedCount >= PopSize of
                true ->
                    %% All evaluated - compute novelty and breed
                    handle_cohort_complete(State#novelty_state{
                        population = UpdatedPop,
                        population_map = NewPopMap,
                        evaluated_count = NewEvaluatedCount
                    }, [EvalEvent]);
                false ->
                    %% Still evaluating
                    NewState = State#novelty_state{
                        population = UpdatedPop,
                        population_map = NewPopMap,
                        evaluated_count = NewEvaluatedCount
                    },
                    {[], [EvalEvent], NewState}
            end;
        error ->
            %% Individual not found (shouldn't happen)
            {[], [], State}
    end.

%% @doc Periodic tick - not heavily used in novelty strategy.
-spec tick(State :: #novelty_state{}) -> strategy_result().
tick(State) ->
    {[], [], State}.

%% @doc Get a snapshot of the current population state.
-spec get_population_snapshot(State :: #novelty_state{}) -> population_snapshot().
get_population_snapshot(State) ->
    Population = State#novelty_state.population,

    %% Sort by novelty score (stored in metrics)
    Sorted = lists:sort(
        fun(A, B) ->
            NoveltyA = get_novelty_score(A),
            NoveltyB = get_novelty_score(B),
            NoveltyA >= NoveltyB
        end,
        Population
    ),

    %% Calculate fitness statistics
    {BestFitness, AvgFitness, WorstFitness} = calculate_fitness_stats(Sorted),

    %% Create individual summaries with novelty
    Summaries = [summarize_individual(Ind) || Ind <- Sorted],

    #{
        size => length(Population),
        individuals => Summaries,
        best_fitness => BestFitness,
        avg_fitness => AvgFitness,
        worst_fitness => WorstFitness,
        generation => State#novelty_state.generation,
        extra => #{
            archive_size => length(State#novelty_state.archive),
            best_novelty => State#novelty_state.best_novelty,
            avg_novelty => State#novelty_state.avg_novelty,
            archive_adds => State#novelty_state.archive_adds
        }
    }.

%% @doc Get inputs for the meta-controller.
-spec get_meta_inputs(State :: #novelty_state{}) -> meta_inputs().
get_meta_inputs(State) ->
    Params = State#novelty_state.params,

    %% Archive fullness (0-1)
    ArchiveFullness = min(1.0, length(State#novelty_state.archive) /
                          Params#novelty_params.archive_size),

    %% Novelty metrics (normalized)
    AvgNovelty = normalize_value(State#novelty_state.avg_novelty, 0.0, 100.0),
    BestNovelty = normalize_value(State#novelty_state.best_novelty, 0.0, 100.0),

    %% Fitness if hybrid mode
    BestFitness = normalize_value(State#novelty_state.best_fitness, 0.0, 1000.0),

    [
        ArchiveFullness,
        AvgNovelty,
        BestNovelty,
        BestFitness
    ].

%% @doc Apply meta-controller parameter adjustments.
-spec apply_meta_params(Params :: meta_params(), State :: #novelty_state{}) -> #novelty_state{}.
apply_meta_params(MetaParams, State) ->
    NoveltyParams = State#novelty_state.params,

    %% Update novelty-specific params
    NewParams = NoveltyParams#novelty_params{
        archive_probability = bound_value(
            maps:get(archive_probability, MetaParams, NoveltyParams#novelty_params.archive_probability),
            0.01, 1.0
        ),
        fitness_weight = bound_value(
            maps:get(fitness_weight, MetaParams, NoveltyParams#novelty_params.fitness_weight),
            0.0, 1.0
        ),
        novelty_threshold = bound_value(
            maps:get(novelty_threshold, MetaParams, NoveltyParams#novelty_params.novelty_threshold),
            0.0, 100.0
        )
    },

    State#novelty_state{params = NewParams}.

%%% ============================================================================
%%% Internal Functions - Cohort Complete / Breeding
%%% ============================================================================

%% @private Handle completion of evaluating entire cohort.
handle_cohort_complete(State, AccEvents) ->
    Population = State#novelty_state.population,
    Generation = State#novelty_state.generation,
    Params = State#novelty_state.params,
    Archive = State#novelty_state.archive,

    %% Extract behavior descriptors from population
    PopBehaviors = extract_behaviors(Population),

    %% Compute novelty scores for each individual
    ScoredPop = compute_novelty_scores(Population, PopBehaviors, Archive, Params),

    %% Update archive with novel individuals
    {NewArchive, ArchiveAdds} = update_archive(ScoredPop, Archive, Params),

    %% Calculate novelty statistics
    {BestNovelty, AvgNovelty} = calculate_novelty_stats(ScoredPop),
    {BestFitness, _, _} = calculate_fitness_stats(ScoredPop),

    %% Create cohort_evaluated event with novelty info
    CohortEvent = #cohort_evaluated{
        generation = Generation,
        best_fitness = BestFitness,
        avg_fitness = AvgNovelty,  % Use avg_novelty as "fitness" in novelty search
        worst_fitness = 0.0,
        population_size = length(ScoredPop),
        timestamp = erlang:timestamp()
    },

    %% Perform selection based on novelty (or hybrid score)
    Config = State#novelty_state.config,
    NumSurvivors = max(2, round(length(ScoredPop) * Config#neuro_config.selection_ratio)),

    %% Sort by selection score (novelty or hybrid)
    Sorted = sort_by_selection_score(ScoredPop, Params),

    Survivors = lists:sublist(Sorted, NumSurvivors),
    Eliminated = lists:nthtail(NumSurvivors, Sorted),

    %% Create death events
    DeathEvents = [create_death_event(Ind, selection_pressure) || Ind <- Eliminated],

    %% Breed offspring
    NumOffspring = State#novelty_state.population_size - NumSurvivors,
    NetworkFactory = State#novelty_state.network_factory,
    {Offspring, BirthEvents} = breed_offspring(
        Survivors, Config, Generation, NumOffspring, NetworkFactory
    ),

    %% Create breeding_complete event
    BreedingEvent = #breeding_complete{
        generation = Generation,
        survivor_count = NumSurvivors,
        eliminated_count = length(Eliminated),
        offspring_count = length(Offspring),
        timestamp = erlang:timestamp()
    },

    %% Reset survivors for next generation
    ResetSurvivors = [reset_individual(Ind) || Ind <- Survivors],

    %% Form next population
    NextPopulation = ResetSurvivors ++ Offspring,

    %% Rebuild population map for next generation
    NextPopulationMap = build_population_map(NextPopulation),

    %% Update state
    NewState = State#novelty_state{
        population = NextPopulation,
        population_map = NextPopulationMap,
        generation = Generation + 1,
        evaluated_count = 0,
        archive = NewArchive,
        best_novelty = BestNovelty,
        avg_novelty = AvgNovelty,
        best_fitness = BestFitness,
        archive_adds = State#novelty_state.archive_adds + ArchiveAdds
    },

    %% Request evaluation of new population
    EvalAction = {evaluate_batch, [Ind#individual.id || Ind <- NextPopulation]},

    AllEvents = AccEvents ++ [CohortEvent] ++ DeathEvents ++ BirthEvents ++ [BreedingEvent],

    {[EvalAction], AllEvents, NewState}.

%%% ============================================================================
%%% Internal Functions - Novelty Computation
%%% ============================================================================

%% @private Extract behavior descriptors from population.
extract_behaviors(Population) ->
    lists:filtermap(
        fun(Ind) ->
            case maps:get(behavior, Ind#individual.metrics, undefined) of
                undefined -> false;
                Behavior -> {true, {Ind#individual.id, Behavior}}
            end
        end,
        Population
    ).

%% @private Compute novelty scores for all individuals.
%%
%% Uses tweann_nif:knn_novelty/4 for NIF-accelerated computation when available.
%% Falls back to pure Erlang implementation if NIF not loaded.
compute_novelty_scores(Population, PopBehaviors, Archive, Params) ->
    K = Params#novelty_params.k_nearest,

    %% Extract just the behavior vectors (without IDs) for NIF call
    PopBehaviorVecs = [B || {_Id, B} <- PopBehaviors],
    ArchiveBehaviorVecs = [B || {_Id, B} <- Archive],

    lists:map(
        fun(Ind) ->
            Behavior = maps:get(behavior, Ind#individual.metrics, undefined),
            Novelty = case Behavior of
                undefined ->
                    %% No behavior descriptor - assign zero novelty
                    0.0;
                _ ->
                    %% Use NIF-accelerated K-NN novelty computation
                    %% tweann_nif handles fallback internally if NIF not loaded
                    try
                        tweann_nif:knn_novelty(Behavior, PopBehaviorVecs, ArchiveBehaviorVecs, K)
                    catch
                        _:_ ->
                            %% Dimension mismatch or other error — degrade gracefully
                            0.0
                    end
            end,
            %% Store novelty in metrics
            NewMetrics = maps:put(novelty, Novelty, Ind#individual.metrics),
            Ind#individual{metrics = NewMetrics}
        end,
        Population
    ).

%% @private Update the archive with novel individuals.
update_archive(ScoredPop, Archive, Params) ->
    ArchiveSize = Params#novelty_params.archive_size,
    ArchiveProb = Params#novelty_params.archive_probability,
    Threshold = Params#novelty_params.novelty_threshold,

    %% Filter individuals that pass threshold and probability check
    NewEntries = lists:filtermap(
        fun(Ind) ->
            Novelty = get_novelty_score(Ind),
            Behavior = maps:get(behavior, Ind#individual.metrics, undefined),
            case Behavior of
                undefined ->
                    false;
                _ ->
                    PassThreshold = Novelty >= Threshold,
                    PassProb = rand:uniform() < ArchiveProb,
                    case PassThreshold andalso PassProb of
                        true -> {true, {Ind#individual.id, Behavior}};
                        false -> false
                    end
            end
        end,
        ScoredPop
    ),

    %% Add new entries to archive
    UpdatedArchive = NewEntries ++ Archive,

    %% Trim archive if too large (remove oldest)
    TrimmedArchive = case length(UpdatedArchive) > ArchiveSize of
        true -> lists:sublist(UpdatedArchive, ArchiveSize);
        false -> UpdatedArchive
    end,

    {TrimmedArchive, length(NewEntries)}.

%% @private Sort population by selection score.
%% Pure novelty or hybrid (novelty + fitness).
sort_by_selection_score(Population, Params) ->
    IncludeFitness = Params#novelty_params.include_fitness,
    FitnessWeight = Params#novelty_params.fitness_weight,

    lists:sort(
        fun(A, B) ->
            ScoreA = selection_score(A, IncludeFitness, FitnessWeight),
            ScoreB = selection_score(B, IncludeFitness, FitnessWeight),
            ScoreA >= ScoreB
        end,
        Population
    ).

%% @private Compute selection score (novelty or hybrid).
selection_score(Ind, false, _FitnessWeight) ->
    %% Pure novelty
    get_novelty_score(Ind);
selection_score(Ind, true, FitnessWeight) ->
    %% Hybrid: weighted combination
    Novelty = get_novelty_score(Ind),
    Fitness = Ind#individual.fitness,
    (1.0 - FitnessWeight) * Novelty + FitnessWeight * Fitness.

%% @private Get novelty score from individual.
get_novelty_score(Ind) ->
    maps:get(novelty, Ind#individual.metrics, 0.0).

%% @private Calculate novelty statistics.
calculate_novelty_stats([]) -> {0.0, 0.0};
calculate_novelty_stats(Population) ->
    Novelties = [get_novelty_score(Ind) || Ind <- Population],
    Best = lists:max(Novelties),
    Avg = lists:sum(Novelties) / length(Novelties),
    {Best, Avg}.

%%% ============================================================================
%%% Internal Functions - Population Management
%%% ============================================================================

%% @private Create initial random population.
%%
%% When topology_mutation_config is set, creates NEAT genomes for each
%% individual. Otherwise creates fixed-topology networks.
create_initial_population(Config, NetworkFactory) ->
    PopSize = Config#neuro_config.population_size,
    UseNeat = Config#neuro_config.topology_mutation_config =/= undefined,

    lists:map(
        fun(Index) ->
            case UseNeat of
                true ->
                    %% NEAT mode: create minimal genome and derive network
                    Genome = genome_factory:create_minimal(Config),
                    Network = genome_factory:to_network(Genome),
                    #individual{
                        id = {initial, Index},
                        network = Network,
                        genome = Genome,
                        generation_born = 1
                    };
                false ->
                    %% Legacy mode: create fixed-topology network
                    Topology = Config#neuro_config.network_topology,
                    Network = NetworkFactory:create_feedforward(Topology),
                    #individual{
                        id = {initial, Index},
                        network = Network,
                        generation_born = 1
                    }
            end
        end,
        lists:seq(1, PopSize)
    ).

%% @private Build population map for O(1) lookup.
build_population_map(Population) ->
    lists:foldl(
        fun(Ind, Acc) -> maps:put(Ind#individual.id, Ind, Acc) end,
        #{},
        Population
    ).

%% @private Update individual in list (used to keep list in sync with map).
update_individual_in_list(UpdatedInd, Population) ->
    Id = UpdatedInd#individual.id,
    lists:map(
        fun(Ind) ->
            case Ind#individual.id =:= Id of
                true -> UpdatedInd;
                false -> Ind
            end
        end,
        Population
    ).

%% @private Reset individual for next generation.
reset_individual(Ind) ->
    Ind#individual{
        fitness = 0.0,
        metrics = #{},
        is_survivor = true,
        is_offspring = false
    }.

%% @private Breed offspring from survivors.
%%
%% When LC is enabled, gets current mutation rates from L0 controller.
%% This enables dynamic adaptation of layer-specific mutation rates.
breed_offspring(Survivors, Config, Generation, Count, NetworkFactory) ->
    %% Get L0-controlled params if LC is running
    DynamicConfig = neuro_config:with_l0_params(Config),
    breed_offspring(Survivors, DynamicConfig, Generation, Count, NetworkFactory, [], []).

breed_offspring(_Survivors, _Config, _Generation, 0, _NetworkFactory, Offspring, Events) ->
    {lists:reverse(Offspring), lists:reverse(Events)};
breed_offspring(Survivors, Config, Generation, Remaining, NetworkFactory, Offspring, Events) ->
    %% Select two parents
    {Parent1, Parent2} = select_parents(Survivors),

    %% Create offspring
    Child = create_offspring_with_factory(
        Parent1, Parent2, Config, Generation + 1, NetworkFactory
    ),

    %% Create birth event
    BirthEvent = create_birth_event(Child, crossover, [Parent1#individual.id, Parent2#individual.id]),

    breed_offspring(Survivors, Config, Generation, Remaining - 1, NetworkFactory,
                   [Child | Offspring], [BirthEvent | Events]).

%% @private Create offspring using the network factory.
%%
%% When parents have genomes, uses neuroevolution_genetic for NEAT crossover.
%% Otherwise uses the NetworkFactory for legacy weight-only evolution.
create_offspring_with_factory(Parent1, Parent2, Config, Generation, NetworkFactory) ->
    %% Check if NEAT mode (parents have genomes)
    case {Parent1#individual.genome, Parent2#individual.genome} of
        {Genome1, Genome2} when Genome1 =/= undefined, Genome2 =/= undefined ->
            %% NEAT mode: use genetic operators module
            neuroevolution_genetic:create_offspring(Parent1, Parent2, Config, Generation);
        _ ->
            %% Legacy mode: use network factory
            ChildNetwork = NetworkFactory:crossover(
                Parent1#individual.network,
                Parent2#individual.network
            ),
            MutatedNetwork = NetworkFactory:mutate(
                ChildNetwork,
                Config#neuro_config.mutation_strength
            ),
            #individual{
                id = make_ref(),
                network = MutatedNetwork,
                parent1_id = Parent1#individual.id,
                parent2_id = Parent2#individual.id,
                generation_born = Generation,
                is_offspring = true
            }
    end.

%% @private Select two parents for breeding (tournament selection).
select_parents(Survivors) when length(Survivors) >= 2 ->
    P1 = tournament_select(Survivors, 2),
    P2 = tournament_select(Survivors, 2),
    {P1, P2};
select_parents([Single]) ->
    {Single, Single}.

%% @private Tournament selection based on novelty.
tournament_select(Population, TournamentSize) ->
    Candidates = random_sample(Population, TournamentSize),
    lists:foldl(
        fun(Ind, Best) ->
            case get_novelty_score(Ind) > get_novelty_score(Best) of
                true -> Ind;
                false -> Best
            end
        end,
        hd(Candidates),
        tl(Candidates)
    ).

%% @private Random sample from list.
random_sample(List, N) when N >= length(List) -> List;
random_sample(List, N) ->
    random_sample(List, N, []).

random_sample(_List, 0, Acc) -> Acc;
random_sample(List, N, Acc) ->
    Index = rand:uniform(length(List)),
    Item = lists:nth(Index, List),
    random_sample(List, N - 1, [Item | Acc]).

%%% ============================================================================
%%% Internal Functions - Events
%%% ============================================================================

%% @private Create individual_born event.
create_birth_event(Individual, Origin) ->
    create_birth_event(Individual, Origin, []).

create_birth_event(Individual, Origin, ParentIds) ->
    #individual_born{
        id = Individual#individual.id,
        parent_ids = ParentIds,
        timestamp = erlang:timestamp(),
        origin = Origin,
        metadata = #{
            generation => Individual#individual.generation_born
        }
    }.

%% @private Create individual_died event.
create_death_event(Individual, Reason) ->
    #individual_died{
        id = Individual#individual.id,
        reason = Reason,
        final_fitness = Individual#individual.fitness,
        timestamp = erlang:timestamp(),
        metadata = #{novelty => get_novelty_score(Individual)}
    }.

%%% ============================================================================
%%% Internal Functions - Utilities
%%% ============================================================================

%% @private Parse strategy params from map or record.
parse_params(Params) when is_record(Params, novelty_params) ->
    Params;
parse_params(Params) when is_map(Params) ->
    #novelty_params{
        archive_size = maps:get(archive_size, Params, 1000),
        archive_probability = maps:get(archive_probability, Params, 0.10),
        k_nearest = maps:get(k_nearest, Params, 15),
        include_fitness = maps:get(include_fitness, Params, false),
        fitness_weight = maps:get(fitness_weight, Params, 0.0),
        novelty_threshold = maps:get(novelty_threshold, Params, 0.0),
        behavior_dimensions = maps:get(behavior_dimensions, Params, undefined)
    };
parse_params(_) ->
    #novelty_params{}.

%% @private Calculate fitness statistics from population.
calculate_fitness_stats([]) ->
    {0.0, 0.0, 0.0};
calculate_fitness_stats(Population) ->
    Fitnesses = [Ind#individual.fitness || Ind <- Population],
    Best = lists:max(Fitnesses),
    Worst = lists:min(Fitnesses),
    Avg = lists:sum(Fitnesses) / length(Fitnesses),
    {Best, Avg, Worst}.

%% @private Summarize individual for snapshot.
summarize_individual(Ind) ->
    #{
        id => Ind#individual.id,
        fitness => Ind#individual.fitness,
        novelty => get_novelty_score(Ind),
        is_survivor => Ind#individual.is_survivor,
        is_offspring => Ind#individual.is_offspring
    }.

%% @private Normalize value to 0-1 range.
normalize_value(Value, Min, Max) ->
    Clamped = max(Min, min(Max, Value)),
    (Clamped - Min) / (Max - Min).

%% @private Bound value to range.
bound_value(Value, Min, Max) ->
    max(Min, min(Max, Value)).
