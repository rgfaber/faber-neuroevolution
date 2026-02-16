%% @doc Generational evolution strategy - the default (mu,lambda) batch evolution.
%%
%% This strategy implements traditional generational evolution:
%% 1. Evaluate entire population
%% 2. Select survivors based on fitness
%% 3. Breed offspring via crossover and mutation
%% 4. Form next generation from survivors + offspring
%% 5. Repeat
%%
%% This preserves all existing behavior from neuroevolution_server including:
%% - Fixed population size
%% - Tournament/top-n selection
%% - Sexual reproduction with crossover
%% - Weight mutation
%% - Optional NEAT-style speciation
%%
%% == Lifecycle Events ==
%%
%% Universal events (all strategies emit):
%% - individual_born - when offspring are created
%% - individual_died - when individuals are eliminated
%% - individual_evaluated - when fitness is computed
%%
%% Strategy-specific events:
%% - cohort_evaluated - all individuals in generation evaluated
%% - breeding_complete - selection and breeding finished
%% - generation_advanced - new generation begins
%%
%% @author R.G. Lefever
%% @copyright 2024-2026 R.G. Lefever
-module(generational_strategy).
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
-record(gen_state, {
    %% Configuration
    config :: neuro_config(),
    params :: generational_params(),

    %% Network factory module (default: network_factory, can override for testing)
    network_factory = network_factory :: module(),

    %% Population management
    population = [] :: [individual()],
    population_map = #{} :: #{individual_id() => individual()},  % O(1) lookup
    population_size :: pos_integer(),

    %% Evaluation-centric progress tracking (PRIMARY metric)
    total_evaluations = 0 :: non_neg_integer(),

    %% Cohort tracking (SECONDARY, for lineage - formerly "generation")
    generation = 1 :: pos_integer(),

    %% Evaluation state
    evaluating = false :: boolean(),
    evaluated_count = 0 :: non_neg_integer(),
    evaluated_individuals = [] :: [individual()],

    %% Statistics
    best_fitness_ever = 0.0 :: float(),
    last_best_fitness = 0.0 :: float(),
    last_avg_fitness = 0.0 :: float(),
    stagnation_count = 0 :: non_neg_integer(),

    %% Speciation (optional)
    species = [] :: [species()],
    next_species_id = 1 :: pos_integer()
}).

%%% ============================================================================
%%% Behaviour Callbacks
%%% ============================================================================

%% @doc Initialize the generational strategy.
%%
%% Expects config map with:
%% - neuro_config - the full neuroevolution config
%% - strategy_params - optional generational_params record or map
%% - network_factory - optional module for network creation (default: network_factory)
-spec init(Config :: map()) -> {ok, #gen_state{}} | {error, term()}.
init(Config) ->
    NeuroConfig = maps:get(neuro_config, Config),

    %% Parse strategy params (defaults if not provided)
    Params = parse_params(maps:get(strategy_params, Config, #{})),

    %% Get network factory (for testability)
    NetworkFactory = maps:get(network_factory, Config, network_factory),

    %% Create initial population
    Population = create_initial_population(NeuroConfig, NetworkFactory),

    %% Build population map for O(1) lookup
    PopulationMap = build_population_map(Population),

    State = #gen_state{
        config = NeuroConfig,
        params = Params,
        network_factory = NetworkFactory,
        population = Population,
        population_map = PopulationMap,
        population_size = NeuroConfig#neuro_config.population_size
    },

    %% Emit individual_born events for initial population
    BirthEvents = [create_birth_event(Ind, initial) || Ind <- Population],

    %% Return initial state with birth events
    %% Note: The server will handle emitting these events
    {ok, State, BirthEvents}.

%% @doc Handle an individual evaluation result.
%%
%% Accumulates results until all individuals are evaluated,
%% then triggers selection and breeding.
-spec handle_evaluation_result(
    IndividualId :: individual_id(),
    FitnessResult :: map(),
    State :: #gen_state{}
) -> strategy_result().
handle_evaluation_result(IndividualId, FitnessResult, State) ->
    Fitness = maps:get(fitness, FitnessResult),
    Metrics = maps:get(metrics, FitnessResult, #{}),

    %% O(1) lookup and update via population map
    PopMap = State#gen_state.population_map,
    case maps:find(IndividualId, PopMap) of
        {ok, Individual} ->
            UpdatedInd = Individual#individual{fitness = Fitness, metrics = Metrics},

            %% Update both map and list
            NewPopMap = maps:put(IndividualId, UpdatedInd, PopMap),
            UpdatedPop = update_individual_in_list(UpdatedInd, State#gen_state.population),

            %% Track evaluation progress
            NewEvaluatedCount = State#gen_state.evaluated_count + 1,
            NewEvaluatedInds = [UpdatedInd | State#gen_state.evaluated_individuals],

            %% Create individual_evaluated event
            EvalEvent = #individual_evaluated{
                id = IndividualId,
                fitness = Fitness,
                metrics = Metrics,
                timestamp = erlang:timestamp(),
                metadata = #{generation => State#gen_state.generation}
            },

            %% Check if all individuals have been evaluated
            PopSize = State#gen_state.population_size,
            case NewEvaluatedCount >= PopSize of
                true ->
                    %% All evaluated - trigger breeding
                    handle_cohort_complete(State#gen_state{
                        population = UpdatedPop,
                        population_map = NewPopMap,
                        evaluated_count = NewEvaluatedCount,
                        evaluated_individuals = NewEvaluatedInds
                    }, [EvalEvent]);
                false ->
                    %% Still evaluating - just update state
                    NewState = State#gen_state{
                        population = UpdatedPop,
                        population_map = NewPopMap,
                        evaluated_count = NewEvaluatedCount,
                        evaluated_individuals = NewEvaluatedInds
                    },
                    {[], [EvalEvent], NewState}
            end;
        error ->
            %% Individual not found (shouldn't happen)
            {[], [], State}
    end.

%% @doc Periodic tick - not heavily used in generational strategy.
%%
%% Could be used for timeout handling or periodic statistics.
-spec tick(State :: #gen_state{}) -> strategy_result().
tick(State) ->
    %% Generational strategy doesn't need periodic ticks
    %% Everything is driven by evaluation results
    {[], [], State}.

%% @doc Get a snapshot of the current population state.
-spec get_population_snapshot(State :: #gen_state{}) -> population_snapshot().
get_population_snapshot(State) ->
    Population = State#gen_state.population,

    %% Sort by fitness for summary
    Sorted = lists:sort(
        fun(A, B) -> A#individual.fitness >= B#individual.fitness end,
        Population
    ),

    %% Calculate fitness statistics
    {BestFitness, AvgFitness, WorstFitness} = calculate_fitness_stats(Sorted),

    %% Create individual summaries (include age = current_generation - generation_born)
    CurrentGen = State#gen_state.generation,
    Summaries = [summarize_individual(Ind, CurrentGen) || Ind <- Sorted],

    #{
        size => length(Population),
        individuals => Summaries,
        best_fitness => BestFitness,
        avg_fitness => AvgFitness,
        worst_fitness => WorstFitness,
        species_count => length(State#gen_state.species),
        generation => State#gen_state.generation,
        extra => #{
            stagnation_count => State#gen_state.stagnation_count,
            best_fitness_ever => State#gen_state.best_fitness_ever
        }
    }.

%% @doc Get inputs for the meta-controller.
%%
%% Returns normalized values representing evolution progress.
-spec get_meta_inputs(State :: #gen_state{}) -> meta_inputs().
get_meta_inputs(State) ->
    Population = State#gen_state.population,

    %% Calculate current stats
    {BestFitness, AvgFitness, WorstFitness} = calculate_fitness_stats(Population),

    %% Fitness spread (diversity indicator)
    FitnessSpread = case BestFitness of
        +0.0 -> 0.0;
        _ -> (BestFitness - WorstFitness) / BestFitness
    end,

    %% Improvement since last generation
    Improvement = BestFitness - State#gen_state.last_best_fitness,
    NormImprovement = normalize_value(Improvement, -100.0, 100.0),

    %% Stagnation indicator (0 = not stagnant, 1 = very stagnant)
    StagnationNorm = min(1.0, State#gen_state.stagnation_count / 20.0),

    %% Generation progress (normalized to max 1000 generations)
    GenProgress = min(1.0, State#gen_state.generation / 1000.0),

    %% Diversity (approximate from fitness variance)
    Variance = calculate_fitness_variance(Population, AvgFitness),
    DiversityNorm = normalize_value(Variance, 0.0, 1000.0),

    [
        normalize_value(BestFitness, 0.0, 1000.0),   % Best fitness
        normalize_value(AvgFitness, 0.0, 1000.0),   % Avg fitness
        FitnessSpread,                               % Diversity
        NormImprovement,                             % Improvement rate
        StagnationNorm,                              % Stagnation
        GenProgress,                                 % Generation progress
        DiversityNorm                                % Population diversity
    ].

%% @doc Apply meta-controller parameter adjustments.
-spec apply_meta_params(Params :: meta_params(), State :: #gen_state{}) -> #gen_state{}.
apply_meta_params(Params, State) ->
    Config = State#gen_state.config,
    StrategyParams = State#gen_state.params,

    %% Update config with bounded values
    NewConfig = Config#neuro_config{
        mutation_rate = bound_value(
            maps:get(mutation_rate, Params, Config#neuro_config.mutation_rate),
            0.01, 1.0
        ),
        mutation_strength = bound_value(
            maps:get(mutation_strength, Params, Config#neuro_config.mutation_strength),
            0.01, 2.0
        ),
        selection_ratio = bound_value(
            maps:get(selection_ratio, Params, Config#neuro_config.selection_ratio),
            0.05, 0.50
        )
    },

    %% Update strategy params if provided
    NewParams = StrategyParams#generational_params{
        mutation_rate = NewConfig#neuro_config.mutation_rate,
        mutation_strength = NewConfig#neuro_config.mutation_strength,
        selection_ratio = NewConfig#neuro_config.selection_ratio
    },

    State#gen_state{
        config = NewConfig,
        params = NewParams
    }.

%%% ============================================================================
%%% Internal Functions - Cohort Complete / Breeding
%%% ============================================================================

%% @private Handle completion of evaluating entire cohort.
handle_cohort_complete(State, AccEvents) ->
    Population = State#gen_state.population,
    Generation = State#gen_state.generation,
    Config = State#gen_state.config,
    Params = State#gen_state.params,

    %% Calculate batch evaluations for this cohort
    PopSize = State#gen_state.population_size,
    EvalsPerIndividual = Config#neuro_config.evaluations_per_individual,
    BatchEvaluations = PopSize * EvalsPerIndividual,
    NewTotalEvaluations = State#gen_state.total_evaluations + BatchEvaluations,

    %% Sort population by fitness
    Sorted = lists:sort(
        fun(A, B) -> A#individual.fitness >= B#individual.fitness end,
        Population
    ),

    %% Calculate statistics
    {BestFitness, AvgFitness, WorstFitness} = calculate_fitness_stats(Sorted),

    %% Calculate improvement for velocity tracking
    Improvement = BestFitness - State#gen_state.last_best_fitness,

    %% Create cohort_evaluated event (legacy, for backward compat)
    CohortEvent = #cohort_evaluated{
        generation = Generation,
        best_fitness = BestFitness,
        avg_fitness = AvgFitness,
        worst_fitness = WorstFitness,
        population_size = length(Sorted),
        timestamp = erlang:timestamp()
    },

    %% Create progress_checkpoint event (evaluation-centric, PRIMARY)
    %% This enables the meta-controller and dashboard to track progress
    %% in evaluation units rather than generation/cohort units
    ProgressEvent = #progress_checkpoint{
        total_evaluations = NewTotalEvaluations,
        evaluations_since_last = BatchEvaluations,
        cohort = Generation,
        best_fitness = BestFitness,
        avg_fitness = AvgFitness,
        worst_fitness = WorstFitness,
        population_size = PopSize,
        improvement = Improvement,
        timestamp = erlang:timestamp()
    },

    %% Perform selection
    NumSurvivors = round(length(Sorted) * Params#generational_params.selection_ratio),
    NumSurvivors2 = max(2, NumSurvivors),  % At least 2 for breeding

    Survivors = lists:sublist(Sorted, NumSurvivors2),
    Eliminated = lists:nthtail(NumSurvivors2, Sorted),

    %% Create death events for eliminated individuals
    DeathEvents = [create_death_event(Ind, selection_pressure) || Ind <- Eliminated],

    %% Breed offspring
    NumOffspring = State#gen_state.population_size - NumSurvivors2,
    NetworkFactory = State#gen_state.network_factory,
    {Offspring, BirthEvents} = breed_offspring(Survivors, Config, Generation, NumOffspring, NetworkFactory),

    %% Create breeding_complete event
    BreedingEvent = #breeding_complete{
        generation = Generation,
        survivor_count = NumSurvivors2,
        eliminated_count = length(Eliminated),
        offspring_count = length(Offspring),
        timestamp = erlang:timestamp()
    },

    %% Reset survivors for next generation
    ResetSurvivors = [reset_individual(Ind, true) || Ind <- Survivors],

    %% Form next population
    NextPopulation = ResetSurvivors ++ Offspring,

    %% Rebuild population map for next generation
    NextPopMap = build_population_map(NextPopulation),

    %% Update stagnation tracking
    Improved = BestFitness > State#gen_state.best_fitness_ever,
    NewStagnation = case Improved of
        true -> 0;
        false -> State#gen_state.stagnation_count + 1
    end,

    %% Create generation_advanced event
    GenAdvancedEvent = #generation_advanced{
        generation = Generation + 1,
        previous_best_fitness = BestFitness,
        previous_avg_fitness = AvgFitness,
        population_size = length(NextPopulation),
        species_count = length(State#gen_state.species),
        timestamp = erlang:timestamp()
    },

    %% Update state for next generation
    NewState = State#gen_state{
        population = NextPopulation,
        population_map = NextPopMap,
        total_evaluations = NewTotalEvaluations,  %% PRIMARY: Evaluation counter
        generation = Generation + 1,
        evaluating = false,
        evaluated_count = 0,
        evaluated_individuals = [],
        best_fitness_ever = max(State#gen_state.best_fitness_ever, BestFitness),
        last_best_fitness = BestFitness,
        last_avg_fitness = AvgFitness,
        stagnation_count = NewStagnation
    },

    %% Combine all events - ProgressEvent is the PRIMARY progress indicator
    AllEvents = AccEvents ++ [ProgressEvent, CohortEvent] ++ DeathEvents ++ BirthEvents ++
                [BreedingEvent, GenAdvancedEvent],

    %% Actions: request evaluation of new population
    EvalAction = {evaluate_batch, [Ind#individual.id || Ind <- NextPopulation]},

    {[EvalAction], AllEvents, NewState}.

%%% ============================================================================
%%% Internal Functions - Population Management
%%% ============================================================================

%% @private Create initial population, optionally seeded with existing networks.
%%
%% When seed_networks is non-empty, the population is composed of:
%% - ~25% exact copies of seed networks
%% - ~25% mutated variants of seeds
%% - ~50% random individuals (ensures exploration)
%%
%% When seed_networks is empty (default), creates all-random population.
%% When topology_mutation_config is set, creates NEAT genomes for each
%% random individual. Otherwise creates fixed-topology networks.
create_initial_population(Config, NetworkFactory) ->
    Seeds = Config#neuro_config.seed_networks,
    case Seeds of
        [_ | _] ->
            create_seeded_population(Config, Seeds, NetworkFactory);
        _ ->
            create_random_population(Config, NetworkFactory)
    end.

%% @private Create all-random population (original behavior).
create_random_population(Config, NetworkFactory) ->
    PopSize = Config#neuro_config.population_size,
    UseNeat = Config#neuro_config.topology_mutation_config =/= undefined,
    lists:map(
        fun(Index) ->
            create_random_individual(Config, NetworkFactory, UseNeat, Index)
        end,
        lists:seq(1, PopSize)
    ).

%% @private Create a single random individual.
create_random_individual(Config, _NetworkFactory, true = _UseNeat, Index) ->
    Genome = genome_factory:create_minimal(Config),
    Network = genome_factory:to_network(Genome),
    #individual{
        id = {initial, Index},
        network = Network,
        genome = Genome,
        generation_born = 1
    };
create_random_individual(Config, NetworkFactory, false = _UseNeat, Index) ->
    Topology = Config#neuro_config.network_topology,
    Network = NetworkFactory:create_feedforward(Topology),
    #individual{
        id = {initial, Index},
        network = Network,
        generation_born = 1
    }.

%% @private Create population seeded with existing networks.
%% ~25% exact copies, ~25% mutated variants, ~50% random.
create_seeded_population(Config, Seeds, NetworkFactory) ->
    PopSize = Config#neuro_config.population_size,
    SeedCount = min(length(Seeds), max(1, PopSize div 4)),
    MutantCount = min(SeedCount, max(1, PopSize div 4)),
    RandomCount = PopSize - SeedCount - MutantCount,

    %% Exact copies of seed networks
    SeedIndividuals = lists:map(
        fun({SeedNet, Idx}) ->
            #individual{
                id = {seed, Idx},
                network = SeedNet,
                generation_born = 1,
                is_survivor = true
            }
        end,
        lists:zip(lists:sublist(Seeds, SeedCount), lists:seq(1, SeedCount))
    ),

    %% Mutated variants of seed networks
    MutantIndividuals = lists:map(
        fun(Idx) ->
            SeedIdx = ((Idx - 1) rem SeedCount) + 1,
            SeedNet = lists:nth(SeedIdx, Seeds),
            MutatedNet = NetworkFactory:mutate(
                SeedNet,
                Config#neuro_config.mutation_strength
            ),
            #individual{
                id = {seed_mutant, Idx},
                network = MutatedNet,
                generation_born = 1,
                is_offspring = true
            }
        end,
        lists:seq(1, MutantCount)
    ),

    %% Random individuals (ensures exploration)
    UseNeat = Config#neuro_config.topology_mutation_config =/= undefined,
    RandomIndividuals = lists:map(
        fun(Idx) ->
            create_random_individual(Config, NetworkFactory, UseNeat, Idx)
        end,
        lists:seq(1, RandomCount)
    ),

    SeedIndividuals ++ MutantIndividuals ++ RandomIndividuals.

%% @private Reset individual for next generation.
reset_individual(Ind, IsSurvivor) ->
    Ind#individual{
        fitness = 0.0,
        metrics = #{},
        is_survivor = IsSurvivor,
        is_offspring = false
    }.

%% @private Build a map from individual ID to individual for O(1) lookup.
build_population_map(Population) ->
    lists:foldl(
        fun(Ind, Acc) -> maps:put(Ind#individual.id, Ind, Acc) end,
        #{},
        Population
    ).

%% @private Update an individual in the population list.
%% Uses the individual's ID to find and replace.
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

    %% Create offspring via crossover and mutation using the factory
    Child = create_offspring_with_factory(
        Parent1, Parent2, Config, Generation + 1, NetworkFactory
    ),

    %% Create birth event
    BirthEvent = create_birth_event(Child, crossover, [Parent1#individual.id, Parent2#individual.id]),

    breed_offspring(Survivors, Config, Generation, Remaining - 1, NetworkFactory,
                   [Child | Offspring], [BirthEvent | Events]).

%% @private Create offspring using the network factory.
%% This allows dependency injection for testing.
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
            ChildId = make_ref(),
            #individual{
                id = ChildId,
                network = MutatedNetwork,
                parent1_id = Parent1#individual.id,
                parent2_id = Parent2#individual.id,
                generation_born = Generation,
                is_offspring = true
            }
    end.

%% @private Select two parents for breeding.
select_parents(Survivors) when length(Survivors) >= 2 ->
    %% Tournament selection with size 2
    P1 = tournament_select(Survivors, 2),
    P2 = tournament_select(Survivors, 2),
    {P1, P2};
select_parents([Single]) ->
    {Single, Single}.

%% @private Tournament selection.
tournament_select(Population, TournamentSize) ->
    %% Select random individuals for tournament
    Candidates = random_sample(Population, TournamentSize),
    %% Return the one with highest fitness
    lists:foldl(
        fun(Ind, Best) ->
            case Ind#individual.fitness > Best#individual.fitness of
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
        metadata = #{}
    }.

%%% ============================================================================
%%% Internal Functions - Utilities
%%% ============================================================================

%% @private Parse strategy params from map or record.
parse_params(Params) when is_record(Params, generational_params) ->
    Params;
parse_params(Params) when is_map(Params) ->
    #generational_params{
        selection_method = maps:get(selection_method, Params, top_n),
        selection_ratio = maps:get(selection_ratio, Params, 0.20),
        tournament_size = maps:get(tournament_size, Params, 3),
        mutation_rate = maps:get(mutation_rate, Params, 0.10),
        mutation_strength = maps:get(mutation_strength, Params, 0.3),
        crossover_rate = maps:get(crossover_rate, Params, 0.75),
        elitism = maps:get(elitism, Params, true),
        elite_count = maps:get(elite_count, Params, 1)
    };
parse_params(_) ->
    #generational_params{}.

%% @private Calculate fitness statistics from population.
calculate_fitness_stats([]) ->
    {0.0, 0.0, 0.0};
calculate_fitness_stats(Population) ->
    Fitnesses = [Ind#individual.fitness || Ind <- Population],
    Best = lists:max(Fitnesses),
    Worst = lists:min(Fitnesses),
    Avg = lists:sum(Fitnesses) / length(Fitnesses),
    {Best, Avg, Worst}.

%% @private Calculate fitness variance.
calculate_fitness_variance([], _Avg) -> 0.0;
calculate_fitness_variance(Population, Avg) ->
    Fitnesses = [Ind#individual.fitness || Ind <- Population],
    SumSquaredDiffs = lists:sum([(F - Avg) * (F - Avg) || F <- Fitnesses]),
    SumSquaredDiffs / length(Fitnesses).

%% @private Summarize individual for snapshot.
summarize_individual(Ind, CurrentGeneration) ->
    GenBorn = Ind#individual.generation_born,
    Age = CurrentGeneration - GenBorn,
    #{
        id => Ind#individual.id,
        fitness => Ind#individual.fitness,
        is_survivor => Ind#individual.is_survivor,
        is_offspring => Ind#individual.is_offspring,
        generation_born => GenBorn,
        age => Age
    }.

%% @private Normalize value to 0-1 range.
normalize_value(Value, Min, Max) ->
    Clamped = max(Min, min(Max, Value)),
    (Clamped - Min) / (Max - Min).

%% @private Bound value to range.
bound_value(Value, Min, Max) ->
    max(Min, min(Max, Value)).
