%% @doc MAP-Elites quality-diversity evolution strategy.
%%
%% MAP-Elites is a quality-diversity algorithm that maintains a map (grid)
%% of elite solutions. Each cell in the grid corresponds to a region of
%% behavior space, and contains the highest-fitness individual whose
%% behavior maps to that region.
%%
%% This approach simultaneously optimizes for:
%% - Quality: Each cell contains the best-performing solution for that behavior
%% - Diversity: The grid covers a wide range of behaviors
%%
%% == Behavior Space ==
%%
%% The behavior space is divided into discrete bins:
%% - behavior_dimensions: number of dimensions (e.g., 2 for a 2D grid)
%% - bins_per_dimension: discretization resolution (e.g., 10 means 10x10=100 cells)
%% - behavior_bounds: min/max for each dimension (for normalization)
%%
%% The evaluator must return behavior descriptors in metrics:
%% #{fitness => F, metrics => #{behavior => [float(), ...]}}
%%
%% == Algorithm ==
%%
%% 1. Initialize: Generate random individuals, evaluate, place in grid
%% 2. Select: Choose parents from existing elites
%% 3. Mutate: Create offspring through mutation
%% 4. Evaluate: Get fitness and behavior for offspring
%% 5. Update: Place offspring in grid if better than current occupant
%% 6. Repeat from step 2
%%
%% == Key Properties ==
%%
%% - Grid cells act as niches preventing competition between behaviors
%% - Elites are never deleted, only replaced by better solutions
%% - Coverage metric: fraction of cells with elites
%% - QD-score: sum of all elite fitnesses (quality × diversity)
%%
%% @author R.G. Lefever
%% @copyright 2024-2026 R.G. Lefever
-module(map_elites_strategy).
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
-record(me_state, {
    %% Configuration
    config :: neuro_config(),
    params :: map_elites_params(),

    %% Network factory module
    network_factory = network_factory :: module(),

    %% The elite grid: map from cell_index to individual
    %% cell_index is a tuple like {Bin1, Bin2, ...}
    grid = #{} :: #{cell_index() => individual()},

    %% Grid dimensions
    dimensions :: pos_integer(),
    bins :: pos_integer(),
    total_cells :: pos_integer(),

    %% Behavior bounds for normalization
    bounds = [] :: [{float(), float()}],

    %% Current batch being evaluated
    batch = [] :: [individual()],
    batch_map = #{} :: #{individual_id() => individual()},  %% O(1) lookup
    batch_size :: pos_integer(),
    evaluated_count = 0 :: non_neg_integer(),

    %% Generation/iteration tracking
    iteration = 1 :: pos_integer(),

    %% Statistics
    cells_filled = 0 :: non_neg_integer(),
    total_evaluations = 0 :: non_neg_integer(),
    best_fitness = 0.0 :: float(),
    qd_score = 0.0 :: float()
}).

-type cell_index() :: tuple().

%%% ============================================================================
%%% Behaviour Callbacks
%%% ============================================================================

%% @doc Initialize the MAP-Elites strategy.
-spec init(Config :: map()) -> {ok, #me_state{}, [lifecycle_event()]} | {error, term()}.
init(Config) ->
    NeuroConfig = maps:get(neuro_config, Config),

    %% Parse strategy params
    Params = parse_params(maps:get(strategy_params, Config, #{})),

    %% Get network factory
    NetworkFactory = maps:get(network_factory, Config, network_factory),

    %% Calculate grid size with safety limit
    Dimensions = Params#map_elites_params.behavior_dimensions,
    Bins = Params#map_elites_params.bins_per_dimension,
    TotalCells = round(math:pow(Bins, Dimensions)),

    %% Validate grid size to prevent unbounded memory growth
    %% Max 10,000 cells (e.g., 10 bins × 4 dimensions, or 100 bins × 2 dimensions)
    MaxCells = 10000,
    case TotalCells > MaxCells of
        true ->
            error({grid_too_large, #{
                dimensions => Dimensions,
                bins => Bins,
                total_cells => TotalCells,
                max_allowed => MaxCells,
                hint => <<"Reduce behavior_dimensions or bins_per_dimension">>
            }});
        false ->
            ok
    end,

    %% Get or generate behavior bounds
    Bounds = case Params#map_elites_params.behavior_bounds of
        [] -> [{0.0, 1.0} || _ <- lists:seq(1, Dimensions)];
        B -> B
    end,

    %% Determine batch size
    BatchSize = Params#map_elites_params.batch_size,

    %% Create initial batch of random individuals
    InitialBatch = create_initial_batch(NeuroConfig, NetworkFactory, BatchSize),
    InitialBatchMap = build_batch_map(InitialBatch),

    State = #me_state{
        config = NeuroConfig,
        params = Params,
        network_factory = NetworkFactory,
        dimensions = Dimensions,
        bins = Bins,
        total_cells = TotalCells,
        bounds = Bounds,
        batch = InitialBatch,
        batch_map = InitialBatchMap,
        batch_size = BatchSize
    },

    %% Emit birth events for initial batch
    BirthEvents = [create_birth_event(Ind, initial) || Ind <- InitialBatch],

    {ok, State, BirthEvents}.

%% @doc Handle an individual evaluation result.
%%
%% When an individual is evaluated:
%% 1. Compute which cell it belongs to
%% 2. If cell is empty or new individual is better, update the grid
%% 3. When batch is complete, generate new batch from elites
-spec handle_evaluation_result(
    IndividualId :: individual_id(),
    FitnessResult :: map(),
    State :: #me_state{}
) -> strategy_result().
handle_evaluation_result(IndividualId, FitnessResult, State) ->
    Fitness = maps:get(fitness, FitnessResult),
    Metrics = maps:get(metrics, FitnessResult, #{}),
    Behavior = maps:get(behavior, Metrics, undefined),

    %% O(1) lookup in batch map
    BatchMap = State#me_state.batch_map,
    case maps:find(IndividualId, BatchMap) of
        {ok, Individual} ->
            %% Update individual with fitness and behavior
            NewMetrics = case Behavior of
                undefined -> Individual#individual.metrics;
                _ -> maps:put(behavior, Behavior, Individual#individual.metrics)
            end,
            UpdatedInd = Individual#individual{fitness = Fitness, metrics = NewMetrics},

            %% Update both map and list
            NewBatchMap = maps:put(IndividualId, UpdatedInd, BatchMap),
            UpdatedBatch = update_individual_in_list(UpdatedInd, State#me_state.batch),

            %% Create evaluation event
            EvalEvent = #individual_evaluated{
                id = IndividualId,
                fitness = Fitness,
                metrics = Metrics,
                timestamp = erlang:timestamp(),
                metadata = #{iteration => State#me_state.iteration}
            },

            %% Try to place in grid
            {UpdatedGrid, CellEvents, Replaced} = try_place_in_grid(UpdatedInd, State),

            %% Update cells filled count
            NewCellsFilled = case Replaced of
                new_cell -> State#me_state.cells_filled + 1;
                _ -> State#me_state.cells_filled
            end,

            %% Update QD score
            NewQDScore = calculate_qd_score(UpdatedGrid),
            NewBestFitness = max(State#me_state.best_fitness, Fitness),

            %% Track evaluation progress (per-individual evaluations)
            Config = State#me_state.config,
            EvalsPerIndividual = Config#neuro_config.evaluations_per_individual,
            NewEvaluatedCount = State#me_state.evaluated_count + 1,
            NewTotalEvals = State#me_state.total_evaluations + EvalsPerIndividual,

            %% Check if batch is complete
            BatchSize = State#me_state.batch_size,
            case NewEvaluatedCount >= BatchSize of
                true ->
                    %% Generate new batch from elites
                    handle_batch_complete(State#me_state{
                        batch = UpdatedBatch,
                        batch_map = NewBatchMap,
                        grid = UpdatedGrid,
                        evaluated_count = NewEvaluatedCount,
                        total_evaluations = NewTotalEvals,
                        cells_filled = NewCellsFilled,
                        best_fitness = NewBestFitness,
                        qd_score = NewQDScore
                    }, [EvalEvent | CellEvents]);
                false ->
                    %% Still evaluating
                    NewState = State#me_state{
                        batch = UpdatedBatch,
                        batch_map = NewBatchMap,
                        grid = UpdatedGrid,
                        evaluated_count = NewEvaluatedCount,
                        total_evaluations = NewTotalEvals,
                        cells_filled = NewCellsFilled,
                        best_fitness = NewBestFitness,
                        qd_score = NewQDScore
                    },
                    {[], [EvalEvent | CellEvents], NewState}
            end;
        error ->
            %% Individual not found (shouldn't happen)
            {[], [], State}
    end.

%% @doc Periodic tick - not heavily used.
-spec tick(State :: #me_state{}) -> strategy_result().
tick(State) ->
    {[], [], State}.

%% @doc Get a snapshot of the current population state.
-spec get_population_snapshot(State :: #me_state{}) -> population_snapshot().
get_population_snapshot(State) ->
    Grid = State#me_state.grid,
    Elites = maps:values(Grid),

    %% Sort by fitness
    Sorted = lists:sort(
        fun(A, B) -> A#individual.fitness >= B#individual.fitness end,
        Elites
    ),

    %% Calculate fitness statistics
    {BestFitness, AvgFitness, WorstFitness} = calculate_fitness_stats(Sorted),

    %% Create individual summaries
    Summaries = [summarize_individual(Ind, State) || Ind <- Sorted],

    %% Coverage metrics
    Coverage = State#me_state.cells_filled / max(1, State#me_state.total_cells),

    #{
        size => length(Elites),
        individuals => Summaries,
        best_fitness => BestFitness,
        avg_fitness => AvgFitness,
        worst_fitness => WorstFitness,
        generation => State#me_state.iteration,
        extra => #{
            cells_filled => State#me_state.cells_filled,
            total_cells => State#me_state.total_cells,
            coverage => Coverage,
            qd_score => State#me_state.qd_score,
            total_evaluations => State#me_state.total_evaluations
        }
    }.

%% @doc Get inputs for the meta-controller.
-spec get_meta_inputs(State :: #me_state{}) -> meta_inputs().
get_meta_inputs(State) ->
    %% Coverage (0-1)
    Coverage = State#me_state.cells_filled / max(1, State#me_state.total_cells),

    %% Normalized QD score
    MaxPossibleQD = State#me_state.total_cells * 1000.0,  % Assume max fitness 1000
    NormQDScore = min(1.0, State#me_state.qd_score / max(1.0, MaxPossibleQD)),

    %% Best fitness normalized
    BestFitnessNorm = normalize_value(State#me_state.best_fitness, 0.0, 1000.0),

    %% Iteration progress
    IterProgress = min(1.0, State#me_state.iteration / 1000.0),

    [
        Coverage,
        NormQDScore,
        BestFitnessNorm,
        IterProgress
    ].

%% @doc Apply meta-controller parameter adjustments.
-spec apply_meta_params(Params :: meta_params(), State :: #me_state{}) -> #me_state{}.
apply_meta_params(MetaParams, State) ->
    MEParams = State#me_state.params,

    NewParams = MEParams#map_elites_params{
        mutation_rate = bound_value(
            maps:get(mutation_rate, MetaParams, MEParams#map_elites_params.mutation_rate),
            0.01, 1.0
        ),
        mutation_strength = bound_value(
            maps:get(mutation_strength, MetaParams, MEParams#map_elites_params.mutation_strength),
            0.01, 2.0
        ),
        random_probability = bound_value(
            maps:get(random_probability, MetaParams, MEParams#map_elites_params.random_probability),
            0.0, 1.0
        )
    },

    State#me_state{params = NewParams}.

%%% ============================================================================
%%% Internal Functions - Grid Management
%%% ============================================================================

%% @private Try to place an individual in the grid.
%% Returns {UpdatedGrid, Events, Replaced} where Replaced is:
%% - new_cell: placed in previously empty cell
%% - replaced: replaced existing (lower fitness)
%% - rejected: existing was better
try_place_in_grid(Individual, State) ->
    Behavior = maps:get(behavior, Individual#individual.metrics, undefined),

    case Behavior of
        undefined ->
            %% No behavior - can't place in grid
            {State#me_state.grid, [], rejected};
        _ ->
            CellIndex = behavior_to_cell(Behavior, State),
            Grid = State#me_state.grid,
            Fitness = Individual#individual.fitness,

            case maps:get(CellIndex, Grid, undefined) of
                undefined ->
                    %% Empty cell - place individual
                    NewGrid = maps:put(CellIndex, Individual, Grid),
                    BirthEvent = create_elite_event(Individual, CellIndex),
                    {NewGrid, [BirthEvent], new_cell};

                Existing when Fitness > Existing#individual.fitness ->
                    %% Better than existing - replace
                    NewGrid = maps:put(CellIndex, Individual, Grid),
                    DeathEvent = create_death_event(Existing, replaced_by_better),
                    ReplaceEvent = create_elite_event(Individual, CellIndex),
                    {NewGrid, [DeathEvent, ReplaceEvent], replaced};

                _Existing ->
                    %% Not better - reject
                    DeathEvent = create_death_event(Individual, lower_fitness),
                    {Grid, [DeathEvent], rejected}
            end
    end.

%% @private Convert behavior vector to cell index.
behavior_to_cell(Behavior, State) when is_list(Behavior) ->
    Bounds = State#me_state.bounds,
    Bins = State#me_state.bins,

    %% Normalize and discretize each dimension
    Indices = lists:zipwith(
        fun(Value, {Min, Max}) ->
            Normalized = (Value - Min) / max(0.0001, Max - Min),
            Clamped = max(0.0, min(1.0, Normalized)),
            %% Convert to bin index (0 to Bins-1)
            min(Bins - 1, trunc(Clamped * Bins))
        end,
        Behavior,
        Bounds
    ),

    list_to_tuple(Indices);
behavior_to_cell(_, _) ->
    %% Invalid behavior
    {0}.

%% @private Calculate QD-score (sum of all elite fitnesses).
calculate_qd_score(Grid) ->
    Elites = maps:values(Grid),
    lists:sum([Ind#individual.fitness || Ind <- Elites]).

%%% ============================================================================
%%% Internal Functions - Batch Generation
%%% ============================================================================

%% @private Handle completion of evaluating a batch.
handle_batch_complete(State, AccEvents) ->
    Grid = State#me_state.grid,
    Params = State#me_state.params,
    Config = State#me_state.config,
    NetworkFactory = State#me_state.network_factory,
    BatchSize = State#me_state.batch_size,

    %% Get elites for parent selection
    Elites = maps:values(Grid),

    %% Generate new batch
    {NewBatch, BirthEvents} = case Elites of
        [] ->
            %% No elites yet - generate random
            Batch = create_initial_batch(Config, NetworkFactory, BatchSize),
            Events = [create_birth_event(Ind, initial) || Ind <- Batch],
            {Batch, Events};
        _ ->
            %% Generate from elites (some random, some mutations)
            generate_batch_from_elites(Elites, Config, Params, NetworkFactory, BatchSize)
    end,

    %% Build batch map for O(1) lookup
    NewBatchMap = build_batch_map(NewBatch),

    NewState = State#me_state{
        batch = NewBatch,
        batch_map = NewBatchMap,
        evaluated_count = 0,
        iteration = State#me_state.iteration + 1
    },

    %% Request evaluation of new batch
    EvalAction = {evaluate_batch, [Ind#individual.id || Ind <- NewBatch]},

    {[EvalAction], AccEvents ++ BirthEvents, NewState}.

%% @private Generate new batch from elites.
generate_batch_from_elites(Elites, Config, Params, NetworkFactory, BatchSize) ->
    RandomProb = Params#map_elites_params.random_probability,

    lists:foldl(
        fun(_, {AccBatch, AccEvents}) ->
            {Ind, Event} = case rand:uniform() < RandomProb of
                true ->
                    %% Generate random individual
                    RandomInd = create_random_individual(Config, NetworkFactory),
                    {RandomInd, create_birth_event(RandomInd, initial)};
                false ->
                    %% Select elite and mutate
                    Parent = random_elite(Elites),
                    Offspring = mutate_individual(Parent, Config, Params, NetworkFactory),
                    {Offspring, create_birth_event(Offspring, mutation, [Parent#individual.id])}
            end,
            {[Ind | AccBatch], [Event | AccEvents]}
        end,
        {[], []},
        lists:seq(1, BatchSize)
    ).

%% @private Select a random elite.
random_elite(Elites) ->
    Index = rand:uniform(length(Elites)),
    lists:nth(Index, Elites).

%% @private Mutate an individual.
%%
%% When parent has a genome, uses genome_factory for NEAT mutation.
%% Otherwise uses the NetworkFactory for legacy weight-only mutation.
mutate_individual(Parent, Config, Params, NetworkFactory) ->
    case Parent#individual.genome of
        Genome when Genome =/= undefined ->
            %% NEAT mode: use genome_factory for mutation
            MutConfig = case Config#neuro_config.topology_mutation_config of
                undefined ->
                    #mutation_config{
                        weight_mutation_rate = Params#map_elites_params.mutation_rate,
                        weight_perturb_strength = Params#map_elites_params.mutation_strength
                    };
                MC -> MC
            end,
            MutatedGenome = genome_factory:mutate(Genome, MutConfig),
            MutatedNetwork = genome_factory:to_network(MutatedGenome),
            #individual{
                id = make_ref(),
                network = MutatedNetwork,
                genome = MutatedGenome,
                parent1_id = Parent#individual.id,
                generation_born = 0,
                is_offspring = true
            };
        _ ->
            %% Legacy mode: use network factory
            MutatedNetwork = NetworkFactory:mutate(
                Parent#individual.network,
                Params#map_elites_params.mutation_strength
            ),
            #individual{
                id = make_ref(),
                network = MutatedNetwork,
                parent1_id = Parent#individual.id,
                generation_born = 0,
                is_offspring = true
            }
    end.

%% @private Create a random individual.
%%
%% When topology_mutation_config is set, creates NEAT genomes.
%% Otherwise creates fixed-topology networks.
create_random_individual(Config, NetworkFactory) ->
    UseNeat = Config#neuro_config.topology_mutation_config =/= undefined,
    case UseNeat of
        true ->
            %% NEAT mode: create minimal genome and derive network
            Genome = genome_factory:create_minimal(Config),
            Network = genome_factory:to_network(Genome),
            #individual{
                id = make_ref(),
                network = Network,
                genome = Genome,
                generation_born = 0
            };
        false ->
            %% Legacy mode: create fixed-topology network
            Topology = Config#neuro_config.network_topology,
            Network = NetworkFactory:create_feedforward(Topology),
            #individual{
                id = make_ref(),
                network = Network,
                generation_born = 0
            }
    end.

%% @private Create initial batch of random individuals.
%%
%% When topology_mutation_config is set, creates NEAT genomes for each
%% individual. Otherwise creates fixed-topology networks.
create_initial_batch(Config, NetworkFactory, BatchSize) ->
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
        lists:seq(1, BatchSize)
    ).

%%% ============================================================================
%%% Internal Functions - Population Management
%%% ============================================================================

%% @private Build batch map for O(1) lookup.
build_batch_map(Batch) ->
    lists:foldl(
        fun(Ind, Acc) -> maps:put(Ind#individual.id, Ind, Acc) end,
        #{},
        Batch
    ).

%% @private Update individual in list (used to keep list in sync with map).
update_individual_in_list(UpdatedInd, Batch) ->
    Id = UpdatedInd#individual.id,
    lists:map(
        fun(Ind) ->
            case Ind#individual.id =:= Id of
                true -> UpdatedInd;
                false -> Ind
            end
        end,
        Batch
    ).

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
        metadata = #{}
    }.

%% @private Create event for elite placement.
create_elite_event(Individual, CellIndex) ->
    #individual_born{
        id = Individual#individual.id,
        parent_ids = [],
        timestamp = erlang:timestamp(),
        origin = elite_placement,
        metadata = #{
            cell_index => CellIndex,
            fitness => Individual#individual.fitness
        }
    }.

%% @private Create individual_died event.
create_death_event(Individual, Reason) ->
    #individual_died{
        id = Individual#individual.id,
        reason = Reason,
        final_fitness = Individual#individual.fitness,
        timestamp = erlang:timestamp(),
        metadata = #{}
    }.

%%% ============================================================================
%%% Internal Functions - Utilities
%%% ============================================================================

%% @private Parse strategy params from map or record.
parse_params(Params) when is_record(Params, map_elites_params) ->
    Params;
parse_params(Params) when is_map(Params) ->
    #map_elites_params{
        behavior_dimensions = maps:get(behavior_dimensions, Params, 2),
        bins_per_dimension = maps:get(bins_per_dimension, Params, 10),
        behavior_bounds = maps:get(behavior_bounds, Params, []),
        batch_size = maps:get(batch_size, Params, 10),
        random_probability = maps:get(random_probability, Params, 0.10),
        mutation_rate = maps:get(mutation_rate, Params, 0.10),
        mutation_strength = maps:get(mutation_strength, Params, 0.3)
    };
parse_params(_) ->
    #map_elites_params{}.

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
summarize_individual(Ind, State) ->
    Behavior = maps:get(behavior, Ind#individual.metrics, undefined),
    CellIndex = case Behavior of
        undefined -> undefined;
        _ -> behavior_to_cell(Behavior, State)
    end,

    #{
        id => Ind#individual.id,
        fitness => Ind#individual.fitness,
        cell_index => CellIndex,
        is_elite => true
    }.

%% @private Normalize value to 0-1 range.
normalize_value(Value, Min, Max) ->
    Clamped = max(Min, min(Max, Value)),
    (Clamped - Min) / (Max - Min).

%% @private Bound value to range.
bound_value(Value, Min, Max) ->
    max(Min, min(Max, Value)).
