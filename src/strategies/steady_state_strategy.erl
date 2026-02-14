%% @doc Steady-state evolution strategy implementation.
%%
%% Unlike generational evolution which replaces the entire population each
%% generation, steady-state evolution replaces only a few individuals at a time.
%% This provides a continuous evolutionary pressure with no distinct generations.
%%
%% Key characteristics:
%% - After each evaluation, 1-N individuals may be replaced
%% - No distinct generations - continuous replacement
%% - Maintains population diversity through gradual change
%% - Age tracking for victim selection
%%
%% @author R.G. Lefever
%% @copyright 2024-2026 R.G. Lefever
-module(steady_state_strategy).
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
%%% State Record
%%% ============================================================================

-record(ss_state, {
    %% Neuroevolution config
    config :: neuro_config(),

    %% Strategy-specific parameters
    params :: steady_state_params(),

    %% Network factory module
    network_factory :: module(),

    %% Current population (list of individuals)
    population :: [individual()],

    %% Population map for O(1) lookup
    population_map = #{} :: #{individual_id() => individual()},

    %% Population size (constant)
    population_size :: pos_integer(),

    %% Number of individuals that have been evaluated
    evaluated_count = 0 :: non_neg_integer(),

    %% Total evaluations performed (for tracking)
    total_evaluations = 0 :: non_neg_integer(),

    %% Best fitness ever seen
    best_fitness_ever = 0.0 :: float(),

    %% Age of each individual (evaluations since birth)
    %% Stored separately for efficiency
    ages :: #{individual_id() => non_neg_integer()},

    %% Progress checkpoint configuration
    checkpoint_interval = 1000 :: pos_integer(),

    %% Evaluations since last checkpoint was emitted
    evals_since_checkpoint = 0 :: non_neg_integer(),

    %% Training start time for elapsed time calculation
    start_time :: erlang:timestamp() | undefined
}).

%%% ============================================================================
%%% Behaviour Callbacks
%%% ============================================================================

%% @doc Initialize the steady-state strategy.
-spec init(Config :: map()) -> {ok, #ss_state{}, [lifecycle_event()]}.
init(Config) ->
    NeuroConfig = maps:get(neuro_config, Config),

    %% Parse strategy params (defaults if not provided)
    Params = parse_params(maps:get(strategy_params, Config, #{})),

    %% Get network factory (for testability)
    NetworkFactory = maps:get(network_factory, Config, network_factory),

    %% Create initial population with default max_age from Params
    Population = create_initial_population(NeuroConfig, NetworkFactory, Params),
    PopulationMap = build_population_map(Population),

    %% Initialize ages (all start at 0)
    %% Ages are also stored on individuals as birth_evaluation, but we keep
    %% this map for backwards compatibility and efficient lookups
    Ages = maps:from_list([{Ind#individual.id, 0} || Ind <- Population]),

    %% Get checkpoint interval from strategy params (default 1000 evals)
    CheckpointInterval = maps:get(checkpoint_interval,
                                   maps:get(strategy_params, Config, #{}),
                                   1000),

    State = #ss_state{
        config = NeuroConfig,
        params = Params,
        network_factory = NetworkFactory,
        population = Population,
        population_map = PopulationMap,
        population_size = NeuroConfig#neuro_config.population_size,
        ages = Ages,
        checkpoint_interval = CheckpointInterval,
        evals_since_checkpoint = 0,
        start_time = erlang:timestamp()
    },

    %% Emit individual_born events for initial population
    BirthEvents = [create_birth_event(Ind, initial) || Ind <- Population],

    {ok, State, BirthEvents}.

%% @doc Handle an individual evaluation result.
%%
%% In steady-state, each evaluation may trigger immediate replacement.
-spec handle_evaluation_result(IndividualId, FitnessResult, State) -> Result when
    IndividualId :: individual_id(),
    FitnessResult :: #{fitness := float(), metrics => map()},
    State :: #ss_state{},
    Result :: {[strategy_action()], [lifecycle_event()], #ss_state{}}.
handle_evaluation_result(IndividualId, FitnessResult, State) ->
    Fitness = maps:get(fitness, FitnessResult),
    Metrics = maps:get(metrics, FitnessResult, #{}),

    %% O(1) lookup in population map
    PopMap = State#ss_state.population_map,
    case maps:find(IndividualId, PopMap) of
        {ok, Individual} ->
            UpdatedIndividual = Individual#individual{
                fitness = Fitness,
                metrics = Metrics
            },

            %% Update both map and list
            NewPopMap = maps:put(IndividualId, UpdatedIndividual, PopMap),
            UpdatedPopulation = update_individual_in_list(UpdatedIndividual, State#ss_state.population),

            %% Increment age for all individuals
            Ages = increment_all_ages(State#ss_state.ages),

            %% Track best fitness
            BestFitness = max(Fitness, State#ss_state.best_fitness_ever),

            %% Per-individual evaluations (PRIMARY progress metric)
            Config = State#ss_state.config,
            EvalsPerIndividual = Config#neuro_config.evaluations_per_individual,
            NewTotalEvaluations = State#ss_state.total_evaluations + EvalsPerIndividual,

            %% Create evaluation event
            EvalEvent = #individual_evaluated{
                id = IndividualId,
                fitness = Fitness,
                metrics = Metrics,
                timestamp = erlang:timestamp(),
                metadata = #{evaluation_count => NewTotalEvaluations}
            },

            NewEvalsSinceCheckpoint = State#ss_state.evals_since_checkpoint + EvalsPerIndividual,

            NewState = State#ss_state{
                population = UpdatedPopulation,
                population_map = NewPopMap,
                evaluated_count = State#ss_state.evaluated_count + 1,
                total_evaluations = NewTotalEvaluations,
                best_fitness_ever = BestFitness,
                ages = Ages,
                evals_since_checkpoint = NewEvalsSinceCheckpoint
            },

            %% Check if we should trigger replacement
            {Actions, ReplacementEvents, StateAfterReplacement} = maybe_trigger_replacement(NewState),

            %% Check if we should emit a progress checkpoint
            {CheckpointEvents, FinalState} = maybe_emit_checkpoint(StateAfterReplacement),

            {Actions, [EvalEvent | ReplacementEvents] ++ CheckpointEvents, FinalState};

        error ->
            %% Individual not in population (might have been replaced)
            {[], [], State}
    end.

%% @doc Periodic tick for continuous operations.
%%
%% Steady-state can use ticks for age-based culling.
%% Each individual has its own max_age - culling compares individual age
%% against their personal max_age threshold.
-spec tick(State :: #ss_state{}) -> {[strategy_action()], [lifecycle_event()], #ss_state{}}.
tick(State) ->
    %% Check for individuals that exceed their own max age
    cull_old_individuals(State).

%% @doc Get a snapshot of the current population state.
-spec get_population_snapshot(State :: #ss_state{}) -> population_snapshot().
get_population_snapshot(State) ->
    Population = State#ss_state.population,
    Ages = State#ss_state.ages,

    {BestFitness, AvgFitness, WorstFitness} = calculate_fitness_stats(Population),

    Summaries = [#{
        id => Ind#individual.id,
        fitness => Ind#individual.fitness,
        is_survivor => true,  % All are survivors in steady-state
        is_offspring => Ind#individual.is_offspring,
        age => maps:get(Ind#individual.id, Ages, 0),
        max_age => Ind#individual.max_age,
        birth_evaluation => Ind#individual.birth_evaluation
    } || Ind <- Population],

    #{
        size => length(Population),
        individuals => Summaries,
        best_fitness => BestFitness,
        avg_fitness => AvgFitness,
        worst_fitness => WorstFitness,
        species_count => 1,  % Steady-state without speciation = single species
        generation => State#ss_state.total_evaluations,  % Use total evals as "generation"
        extra => #{
            evaluated_count => State#ss_state.evaluated_count,
            total_evaluations => State#ss_state.total_evaluations,
            best_fitness_ever => State#ss_state.best_fitness_ever
        }
    }.

%% @doc Get normalized inputs for meta-controller.
-spec get_meta_inputs(State :: #ss_state{}) -> [float()].
get_meta_inputs(State) ->
    Population = State#ss_state.population,
    Ages = State#ss_state.ages,
    Params = State#ss_state.params,

    {BestFitness, AvgFitness, _WorstFitness} = calculate_fitness_stats(Population),

    %% Calculate diversity (fitness variance)
    Fitnesses = [Ind#individual.fitness || Ind <- Population],
    Variance = calculate_variance(Fitnesses),
    NormalizedVariance = min(1.0, Variance / 100.0),

    %% Calculate average age
    AgeValues = maps:values(Ages),
    AvgAge = case AgeValues of
        [] -> 0.0;
        _ -> lists:sum(AgeValues) / length(AgeValues)
    end,
    NormalizedAge = min(1.0, AvgAge / 100.0),

    %% Calculate improvement rate (best vs average)
    ImprovementGap = case AvgFitness == 0.0 of
        true -> 0.0;
        false -> min(1.0, (BestFitness - AvgFitness) / max(1.0, AvgFitness))
    end,

    %% Current mutation rate (normalized)
    MutationRate = Params#steady_state_params.mutation_rate,

    [
        NormalizedVariance,     % Population diversity
        NormalizedAge,          % Average age
        ImprovementGap,         % Gap between best and average
        MutationRate            % Current mutation rate
    ].

%% @doc Apply parameter updates from meta-controller.
-spec apply_meta_params(MetaParams :: map(), State :: #ss_state{}) -> #ss_state{}.
apply_meta_params(MetaParams, State) ->
    Params = State#ss_state.params,
    Config = State#ss_state.config,

    %% Update mutation rate if provided
    NewMutationRate = maps:get(mutation_rate, MetaParams, Params#steady_state_params.mutation_rate),
    BoundedMutationRate = max(0.01, min(0.5, NewMutationRate)),

    %% Update mutation strength if provided
    NewMutationStrength = maps:get(mutation_strength, MetaParams, Params#steady_state_params.mutation_strength),
    BoundedMutationStrength = max(0.01, min(1.0, NewMutationStrength)),

    NewParams = Params#steady_state_params{
        mutation_rate = BoundedMutationRate,
        mutation_strength = BoundedMutationStrength
    },

    %% Also update the neuro_config for consistency
    NewConfig = Config#neuro_config{
        mutation_rate = BoundedMutationRate,
        mutation_strength = BoundedMutationStrength
    },

    State#ss_state{
        params = NewParams,
        config = NewConfig
    }.

%% @doc Clean up when strategy terminates.
-spec terminate(Reason :: term(), State :: #ss_state{}) -> ok.
terminate(_Reason, _State) ->
    ok.

%%% ============================================================================
%%% Internal Functions - Initialization
%%% ============================================================================

%% @private Parse strategy params from map or record.
parse_params(Params) when is_map(Params) ->
    #steady_state_params{
        replacement_count = maps:get(replacement_count, Params, 1),
        parent_selection = maps:get(parent_selection, Params, tournament),
        victim_selection = maps:get(victim_selection, Params, worst),
        tournament_size = maps:get(tournament_size, Params, 3),
        mutation_rate = maps:get(mutation_rate, Params, 0.10),
        mutation_strength = maps:get(mutation_strength, Params, 0.3),
        default_max_age = maps:get(default_max_age, Params, maps:get(max_age, Params, 5000)),
        max_age_mutation_rate = maps:get(max_age_mutation_rate, Params, 0.10),
        max_age_mutation_strength = maps:get(max_age_mutation_strength, Params, 0.10)
    };
parse_params(Params) when is_record(Params, steady_state_params) ->
    Params;
parse_params(_) ->
    #steady_state_params{}.

%% @private Create initial population.
%%
%% When topology_mutation_config is set, creates NEAT genomes for each
%% individual. Otherwise creates fixed-topology networks.
%% Each individual is initialized with birth_evaluation=0 and max_age from default.
create_initial_population(Config, NetworkFactory, Params) ->
    PopSize = Config#neuro_config.population_size,
    UseNeat = Config#neuro_config.topology_mutation_config =/= undefined,
    DefaultMaxAge = Params#steady_state_params.default_max_age,

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
                        generation_born = 1,
                        birth_evaluation = 0,
                        max_age = DefaultMaxAge
                    };
                false ->
                    %% Legacy mode: create fixed-topology network
                    Topology = Config#neuro_config.network_topology,
                    Network = NetworkFactory:create_feedforward(Topology),
                    #individual{
                        id = {initial, Index},
                        network = Network,
                        generation_born = 1,
                        birth_evaluation = 0,
                        max_age = DefaultMaxAge
                    }
            end
        end,
        lists:seq(1, PopSize)
    ).

%% @private Create birth event for an individual.
create_birth_event(Individual, Origin) ->
    create_birth_event(Individual, Origin, []).

create_birth_event(Individual, Origin, ParentIds) ->
    #individual_born{
        id = Individual#individual.id,
        parent_ids = ParentIds,
        origin = Origin,
        timestamp = erlang:timestamp(),
        metadata = #{network_complexity => estimate_complexity(Individual#individual.network)}
    }.

%% @private Estimate network complexity.
estimate_complexity(Network) when is_map(Network) ->
    case maps:get(weights, Network, []) of
        Weights when is_list(Weights) -> length(Weights);
        _ -> 0
    end;
estimate_complexity(_) ->
    0.

%%% ============================================================================
%%% Internal Functions - Replacement Logic
%%% ============================================================================

%% @private Maybe trigger replacement after evaluation.
%%
%% In steady-state, we replace after every N evaluations where N is population size.
maybe_trigger_replacement(State) ->
    EvalCount = State#ss_state.evaluated_count,
    PopSize = State#ss_state.population_size,

    case EvalCount >= PopSize of
        true ->
            %% Time to do replacement
            perform_replacement(State);
        false ->
            {[], [], State}
    end.

%% @private Perform the actual replacement.
perform_replacement(State) ->
    Params = State#ss_state.params,
    Config = State#ss_state.config,
    Population = State#ss_state.population,
    Ages = State#ss_state.ages,
    NetworkFactory = State#ss_state.network_factory,
    TotalEvaluations = State#ss_state.total_evaluations,

    ReplacementCount = Params#steady_state_params.replacement_count,

    %% Select victims to replace
    {Victims, Remaining} = select_victims(Population, Ages, Params, ReplacementCount),

    %% Create offspring to replace victims (with lifespan inheritance)
    {Offspring, BirthEvents} = create_offspring_with_lifespan(
        Remaining, Config, Params, NetworkFactory, TotalEvaluations, ReplacementCount
    ),

    %% Create death events for victims
    DeathEvents = [create_death_event(Ind, replacement) || Ind <- Victims],

    %% Update ages: remove victims, add offspring at age 0
    NewAges = lists:foldl(
        fun(Ind, Acc) -> maps:remove(Ind#individual.id, Acc) end,
        Ages,
        Victims
    ),
    FinalAges = lists:foldl(
        fun(Ind, Acc) -> maps:put(Ind#individual.id, 0, Acc) end,
        NewAges,
        Offspring
    ),

    %% New population = remaining + offspring
    NewPopulation = Remaining ++ Offspring,

    %% Rebuild population map for new population
    NewPopulationMap = build_population_map(NewPopulation),

    %% Calculate fitness stats for event
    {BestFit, AvgFit, _WorstFit} = calculate_fitness_stats(NewPopulation),

    %% Create replacement event
    ReplacementEvent = #steady_state_replacement{
        replaced_ids = [V#individual.id || V <- Victims],
        offspring_ids = [O#individual.id || O <- Offspring],
        best_fitness = BestFit,
        avg_fitness = AvgFit,
        timestamp = erlang:timestamp()
    },

    NewState = State#ss_state{
        population = NewPopulation,
        population_map = NewPopulationMap,
        evaluated_count = 0,  % Reset for next cycle
        ages = FinalAges
    },

    {[], DeathEvents ++ BirthEvents ++ [ReplacementEvent], NewState}.

%% @private Select victims for replacement.
select_victims(Population, Ages, Params, Count) ->
    Method = Params#steady_state_params.victim_selection,
    TournamentSize = Params#steady_state_params.tournament_size,

    select_victims_by_method(Population, Ages, Method, TournamentSize, Count, [], []).

select_victims_by_method(_Pop, _Ages, _Method, _TSize, 0, Victims, Remaining) ->
    {lists:reverse(Victims), Remaining};
select_victims_by_method(Population, Ages, Method, TournamentSize, Count, Victims, _) ->
    Victim = case Method of
        worst ->
            %% Select individual with lowest fitness
            lists:foldl(
                fun(Ind, Worst) ->
                    case Ind#individual.fitness < Worst#individual.fitness of
                        true -> Ind;
                        false -> Worst
                    end
                end,
                hd(Population),
                tl(Population)
            );
        oldest ->
            %% Select oldest individual
            lists:foldl(
                fun(Ind, Oldest) ->
                    IndAge = maps:get(Ind#individual.id, Ages, 0),
                    OldestAge = maps:get(Oldest#individual.id, Ages, 0),
                    case IndAge > OldestAge of
                        true -> Ind;
                        false -> Oldest
                    end
                end,
                hd(Population),
                tl(Population)
            );
        random ->
            %% Select random individual
            lists:nth(rand:uniform(length(Population)), Population);
        tournament ->
            %% Inverse tournament: winner has LOWEST fitness
            Candidates = random_sample(Population, TournamentSize),
            lists:foldl(
                fun(Ind, Worst) ->
                    case Ind#individual.fitness < Worst#individual.fitness of
                        true -> Ind;
                        false -> Worst
                    end
                end,
                hd(Candidates),
                tl(Candidates)
            )
    end,

    Remaining = lists:delete(Victim, Population),
    select_victims_by_method(Remaining, Ages, Method, TournamentSize, Count - 1, [Victim | Victims], Remaining).

%% @private Create offspring with lifespan inheritance and mutation.
%%
%% Offspring inherit max_age from parents with possible mutation, and
%% have their birth_evaluation set to the current total evaluations.
%% When LC is enabled, gets current mutation rates from L0 controller.
create_offspring_with_lifespan(Population, Config, Params, NetworkFactory, TotalEvaluations, Count) ->
    %% Get L0-controlled params if LC is running
    DynamicConfig = neuro_config:with_l0_params(Config),
    create_offspring_with_lifespan(Population, DynamicConfig, Params, NetworkFactory, TotalEvaluations, Count, [], []).

create_offspring_with_lifespan(_Pop, _Config, _Params, _Factory, _TotalEvals, 0, Offspring, Events) ->
    {lists:reverse(Offspring), lists:reverse(Events)};
create_offspring_with_lifespan(Population, Config, Params, NetworkFactory, TotalEvaluations, Count, Offspring, Events) ->
    %% Select two parents via tournament
    Parent1 = tournament_select(Population, 3),
    Parent2 = tournament_select(Population, 3),

    %% Create child with inherited max_age
    Child = create_child_with_lifespan(Parent1, Parent2, Config, Params, NetworkFactory, TotalEvaluations),

    %% Create birth event
    BirthEvent = create_birth_event(Child, crossover, [Parent1#individual.id, Parent2#individual.id]),

    create_offspring_with_lifespan(Population, Config, Params, NetworkFactory, TotalEvaluations, Count - 1,
                                   [Child | Offspring], [BirthEvent | Events]).

%% @private Create a child from two parents.
%%
%% When parents have genomes, uses neuroevolution_genetic for NEAT crossover.
%% Otherwise uses the NetworkFactory for legacy weight-only evolution.
create_child(Parent1, Parent2, Config, NetworkFactory) ->
    %% Check if NEAT mode (parents have genomes)
    case {Parent1#individual.genome, Parent2#individual.genome} of
        {Genome1, Genome2} when Genome1 =/= undefined, Genome2 =/= undefined ->
            %% NEAT mode: use genetic operators module
            neuroevolution_genetic:create_offspring(Parent1, Parent2, Config, 0);
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
                generation_born = 0,  % Not meaningful in steady-state
                is_offspring = true
            }
    end.

%% @private Create a child with inherited and mutated lifespan.
%%
%% The child inherits max_age from parents (average with possible mutation)
%% and has birth_evaluation set to the current total evaluations.
create_child_with_lifespan(Parent1, Parent2, Config, Params, NetworkFactory, TotalEvaluations) ->
    %% Create base child using existing logic
    BaseChild = create_child(Parent1, Parent2, Config, NetworkFactory),

    %% Inherit max_age from parents (average of both)
    Parent1MaxAge = Parent1#individual.max_age,
    Parent2MaxAge = Parent2#individual.max_age,
    InheritedMaxAge = (Parent1MaxAge + Parent2MaxAge) div 2,

    %% Possibly mutate max_age
    MutatedMaxAge = maybe_mutate_max_age(InheritedMaxAge, Params),

    %% Update child with lifespan fields
    BaseChild#individual{
        birth_evaluation = TotalEvaluations,
        max_age = MutatedMaxAge
    }.

%% @private Maybe mutate max_age based on configured rates.
%%
%% With probability max_age_mutation_rate, applies a relative change
%% (gaussian perturbation) controlled by max_age_mutation_strength.
maybe_mutate_max_age(MaxAge, Params) ->
    MutationRate = Params#steady_state_params.max_age_mutation_rate,
    MutationStrength = Params#steady_state_params.max_age_mutation_strength,

    case rand:uniform() < MutationRate of
        true ->
            %% Apply gaussian perturbation (relative to current value)
            %% Perturbation is in range [-strength, +strength] relative
            Perturbation = (rand:uniform() * 2 - 1) * MutationStrength,
            NewMaxAge = round(MaxAge * (1 + Perturbation)),
            %% Ensure minimum of 100 evaluations lifespan
            max(100, NewMaxAge);
        false ->
            MaxAge
    end.

%% @private Tournament selection (select best from random sample).
tournament_select(Population, TournamentSize) ->
    Candidates = random_sample(Population, TournamentSize),
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

%% @private Create death event.
create_death_event(Individual, Reason) ->
    #individual_died{
        id = Individual#individual.id,
        reason = Reason,
        final_fitness = Individual#individual.fitness,
        timestamp = erlang:timestamp(),
        metadata = #{}
    }.

%%% ============================================================================
%%% Internal Functions - Age Management
%%% ============================================================================

%% @private Increment age for all individuals.
increment_all_ages(Ages) ->
    maps:map(fun(_Id, Age) -> Age + 1 end, Ages).

%% @private Cull individuals that exceed their own max_age.
%%
%% Each individual has a personal max_age field (inherited trait).
%% An individual is culled when their age (from ages map) exceeds their max_age.
cull_old_individuals(State) ->
    Population = State#ss_state.population,
    Ages = State#ss_state.ages,
    Config = State#ss_state.config,
    Params = State#ss_state.params,
    NetworkFactory = State#ss_state.network_factory,
    TotalEvaluations = State#ss_state.total_evaluations,

    %% Find individuals over their own max age
    %% Each individual is checked against their personal max_age field
    {Old, Young} = lists:partition(
        fun(Ind) ->
            IndAge = maps:get(Ind#individual.id, Ages, 0),
            IndMaxAge = Ind#individual.max_age,
            %% Only cull if max_age > 0 (0 = immortal)
            IndMaxAge > 0 andalso IndAge > IndMaxAge
        end,
        Population
    ),

    case Old of
        [] ->
            {[], [], State};
        _ ->
            %% Create replacements with inherited max_age
            {Offspring, BirthEvents} = create_offspring_with_lifespan(
                Young, Config, Params, NetworkFactory, TotalEvaluations, length(Old)
            ),

            %% Create death events with age_limit reason
            DeathEvents = [create_death_event(Ind, age_limit) || Ind <- Old],

            %% Create aged_out lifecycle events for each culled individual
            AgedOutEvents = [#individual_aged_out{
                id = Ind#individual.id,
                final_age = maps:get(Ind#individual.id, Ages, 0),
                final_fitness = Ind#individual.fitness,
                lifetime_stats = #{
                    total_evaluations => maps:get(Ind#individual.id, Ages, 0),
                    avg_fitness => Ind#individual.fitness,
                    best_fitness => Ind#individual.fitness,
                    offspring_count => 0,
                    max_age => Ind#individual.max_age
                },
                timestamp = erlang:timestamp()
            } || Ind <- Old],

            %% Update ages
            NewAges = lists:foldl(
                fun(Ind, Acc) -> maps:remove(Ind#individual.id, Acc) end,
                Ages,
                Old
            ),
            FinalAges = lists:foldl(
                fun(Ind, Acc) -> maps:put(Ind#individual.id, 0, Acc) end,
                NewAges,
                Offspring
            ),

            NewPopulation = Young ++ Offspring,

            %% Rebuild population map
            NewPopulationMap = build_population_map(NewPopulation),

            NewState = State#ss_state{
                population = NewPopulation,
                population_map = NewPopulationMap,
                ages = FinalAges
            },

            {[], DeathEvents ++ AgedOutEvents ++ BirthEvents, NewState}
    end.

%%% ============================================================================
%%% Internal Functions - Progress Checkpoints
%%% ============================================================================

%% @private Maybe emit a progress checkpoint event.
%%
%% Progress checkpoints are emitted every N evaluations (configured via
%% checkpoint_interval parameter). This provides a strategy-agnostic way
%% to track evolution progress for continuous evolution scenarios.
-spec maybe_emit_checkpoint(State :: #ss_state{}) -> {[lifecycle_event()], #ss_state{}}.
maybe_emit_checkpoint(State) ->
    case State#ss_state.evals_since_checkpoint >= State#ss_state.checkpoint_interval of
        true ->
            emit_progress_checkpoint(State);
        false ->
            {[], State}
    end.

%% @private Emit a progress checkpoint event.
%%
%% Creates a #progress_checkpoint{} event with current population statistics
%% and resets the checkpoint counter.
-spec emit_progress_checkpoint(State :: #ss_state{}) -> {[lifecycle_event()], #ss_state{}}.
emit_progress_checkpoint(State) ->
    Population = State#ss_state.population,
    {BestFitness, AvgFitness, WorstFitness} = calculate_fitness_stats(Population),

    %% Calculate elapsed time
    StartTime = State#ss_state.start_time,
    ElapsedMs = case StartTime of
        undefined -> 0;
        _ -> timer:now_diff(erlang:timestamp(), StartTime) div 1000
    end,

    %% Calculate evaluations per second
    TotalEvals = State#ss_state.total_evaluations,
    EvalsPerSecond = case ElapsedMs > 0 of
        true -> TotalEvals / (ElapsedMs / 1000);
        false -> 0.0
    end,

    Checkpoint = #progress_checkpoint{
        total_evaluations = TotalEvals,
        evaluations_since_last = State#ss_state.evals_since_checkpoint,
        best_fitness = BestFitness,
        avg_fitness = AvgFitness,
        worst_fitness = WorstFitness,
        population_size = length(Population),
        elapsed_ms = ElapsedMs,
        evals_per_second = EvalsPerSecond,
        checkpoint_interval = State#ss_state.checkpoint_interval,
        timestamp = erlang:timestamp()
    },

    %% Reset checkpoint counter
    NewState = State#ss_state{
        evals_since_checkpoint = 0
    },

    {[Checkpoint], NewState}.

%%% ============================================================================
%%% Internal Functions - Utilities
%%% ============================================================================

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

%% @private Calculate fitness statistics.
calculate_fitness_stats([]) ->
    {0.0, 0.0, 0.0};
calculate_fitness_stats(Population) ->
    Fitnesses = [Ind#individual.fitness || Ind <- Population],
    Best = lists:max(Fitnesses),
    Worst = lists:min(Fitnesses),
    Avg = lists:sum(Fitnesses) / length(Fitnesses),
    {Best, Avg, Worst}.

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
