%% @doc EUnit tests for steady_state_strategy module.
%%
%% Tests the steady-state evolution strategy implementation including:
%% - Initialization and population creation
%% - Evaluation result handling
%% - Replacement logic
%% - Age tracking
%% - Meta-controller interface
%%
%% @author R.G. Lefever
%% @copyright 2024-2026 R.G. Lefever
-module(steady_state_strategy_tests).

-include_lib("eunit/include/eunit.hrl").
-include("neuroevolution.hrl").
-include("evolution_strategy.hrl").
-include("lifecycle_events.hrl").

%%% ============================================================================
%%% Test Fixtures
%%% ============================================================================

test_neuro_config() ->
    #neuro_config{
        population_size = 5,
        evaluations_per_individual = 1,
        selection_ratio = 0.4,
        mutation_rate = 0.1,
        mutation_strength = 0.2,
        max_generations = 100,
        network_topology = {4, [8], 2},
        evaluator_module = mock_evaluator
    }.

test_init_config() ->
    #{
        neuro_config => test_neuro_config(),
        strategy_params => #{
            replacement_count => 1,
            parent_selection => tournament,
            victim_selection => worst,
            tournament_size => 2,
            mutation_rate => 0.1,
            mutation_strength => 0.2,
            max_age => 0
        },
        network_factory => mock_network_factory
    }.

%%% ============================================================================
%%% Initialization Tests
%%% ============================================================================

init_creates_population_test() ->
    Config = test_init_config(),
    {ok, State, _Events} = steady_state_strategy:init(Config),

    Snapshot = steady_state_strategy:get_population_snapshot(State),
    ?assertEqual(5, maps:get(size, Snapshot)).

init_returns_birth_events_test() ->
    Config = test_init_config(),
    {ok, _State, Events} = steady_state_strategy:init(Config),

    %% Should have one birth event per individual
    BirthEvents = [E || E <- Events, is_record(E, individual_born)],
    ?assertEqual(5, length(BirthEvents)),

    %% All should be initial births
    lists:foreach(fun(E) ->
        ?assertEqual(initial, E#individual_born.origin)
    end, BirthEvents).

init_parses_params_from_map_test() ->
    Config = #{
        neuro_config => test_neuro_config(),
        strategy_params => #{
            replacement_count => 2,
            victim_selection => oldest,
            max_age => 50
        },
        network_factory => mock_network_factory
    },
    {ok, State, _Events} = steady_state_strategy:init(Config),

    %% Verify params were parsed (check via snapshot)
    Snapshot = steady_state_strategy:get_population_snapshot(State),
    ?assertEqual(5, maps:get(size, Snapshot)).

%%% ============================================================================
%%% Evaluation Result Tests
%%% ============================================================================

eval_updates_fitness_test() ->
    Config = test_init_config(),
    {ok, State, Events} = steady_state_strategy:init(Config),

    %% Get first individual ID
    [FirstBirth | _] = [E || E <- Events, is_record(E, individual_born)],
    IndId = FirstBirth#individual_born.id,

    %% Evaluate with fitness 100
    FitnessResult = #{fitness => 100.0, metrics => #{wins => 5}},
    {_Actions, EvalEvents, NewState} = steady_state_strategy:handle_evaluation_result(IndId, FitnessResult, State),

    %% Should emit evaluation event
    EvaluatedEvents = [E || E <- EvalEvents, is_record(E, individual_evaluated)],
    ?assertEqual(1, length(EvaluatedEvents)),

    %% Check fitness updated in snapshot
    Snapshot = steady_state_strategy:get_population_snapshot(NewState),
    BestFitness = maps:get(best_fitness, Snapshot),
    ?assertEqual(100.0, BestFitness).

eval_emits_event_test() ->
    Config = test_init_config(),
    {ok, State, Events} = steady_state_strategy:init(Config),

    [FirstBirth | _] = [E || E <- Events, is_record(E, individual_born)],
    IndId = FirstBirth#individual_born.id,

    FitnessResult = #{fitness => 50.0, metrics => #{}},
    {_Actions, EvalEvents, _NewState} = steady_state_strategy:handle_evaluation_result(IndId, FitnessResult, State),

    [EvalEvent] = [E || E <- EvalEvents, is_record(E, individual_evaluated)],
    ?assertEqual(IndId, EvalEvent#individual_evaluated.id),
    ?assertEqual(50.0, EvalEvent#individual_evaluated.fitness).

%%% ============================================================================
%%% Replacement Logic Tests
%%% ============================================================================

replacement_triggered_after_full_cycle_test() ->
    Config = test_init_config(),
    {ok, State, Events} = steady_state_strategy:init(Config),

    %% Get all individual IDs
    BirthEvents = [E || E <- Events, is_record(E, individual_born)],
    IndIds = [E#individual_born.id || E <- BirthEvents],

    %% Evaluate all individuals with different fitness values
    {FinalState, AllEvents} = lists:foldl(
        fun({Id, Fitness}, {AccState, AccEvents}) ->
            FitnessResult = #{fitness => Fitness, metrics => #{}},
            {_Actions, NewEvents, NewState} = steady_state_strategy:handle_evaluation_result(Id, FitnessResult, AccState),
            {NewState, AccEvents ++ NewEvents}
        end,
        {State, []},
        lists:zip(IndIds, [100.0, 80.0, 60.0, 40.0, 20.0])
    ),

    %% Should have a replacement event
    ReplacementEvents = [E || E <- AllEvents, is_record(E, steady_state_replacement)],
    ?assertEqual(1, length(ReplacementEvents)),

    %% Should have death and birth events for replaced individual
    DeathEvents = [E || E <- AllEvents, is_record(E, individual_died)],
    OffspringBirths = [E || E <- AllEvents, is_record(E, individual_born), E#individual_born.origin =:= crossover],
    ?assertEqual(1, length(DeathEvents)),
    ?assertEqual(1, length(OffspringBirths)),

    %% Population size should remain constant
    Snapshot = steady_state_strategy:get_population_snapshot(FinalState),
    ?assertEqual(5, maps:get(size, Snapshot)).

worst_victim_selection_test() ->
    Config = test_init_config(),
    {ok, State, Events} = steady_state_strategy:init(Config),

    BirthEvents = [E || E <- Events, is_record(E, individual_born)],
    IndIds = [E#individual_born.id || E <- BirthEvents],

    %% Evaluate all - last one has lowest fitness
    {_FinalState, AllEvents} = lists:foldl(
        fun({Id, Fitness}, {AccState, AccEvents}) ->
            FitnessResult = #{fitness => Fitness, metrics => #{}},
            {_Actions, NewEvents, NewState} = steady_state_strategy:handle_evaluation_result(Id, FitnessResult, AccState),
            {NewState, AccEvents ++ NewEvents}
        end,
        {State, []},
        lists:zip(IndIds, [100.0, 80.0, 60.0, 40.0, 20.0])
    ),

    %% The individual with fitness 20.0 should be dead
    DeathEvents = [E || E <- AllEvents, is_record(E, individual_died)],
    [DeathEvent] = DeathEvents,
    DeadId = DeathEvent#individual_died.id,

    %% Find which individual had fitness 20.0
    LowestFitnessId = lists:last(IndIds),
    ?assertEqual(LowestFitnessId, DeadId).

%%% ============================================================================
%%% Snapshot Tests
%%% ============================================================================

snapshot_has_required_fields_test() ->
    Config = test_init_config(),
    {ok, State, _Events} = steady_state_strategy:init(Config),

    Snapshot = steady_state_strategy:get_population_snapshot(State),

    ?assert(maps:is_key(size, Snapshot)),
    ?assert(maps:is_key(individuals, Snapshot)),
    ?assert(maps:is_key(best_fitness, Snapshot)),
    ?assert(maps:is_key(avg_fitness, Snapshot)),
    ?assert(maps:is_key(worst_fitness, Snapshot)),
    ?assert(maps:is_key(extra, Snapshot)).

snapshot_has_age_info_test() ->
    Config = test_init_config(),
    {ok, State, _Events} = steady_state_strategy:init(Config),

    Snapshot = steady_state_strategy:get_population_snapshot(State),
    Individuals = maps:get(individuals, Snapshot),

    %% All individuals should have age field
    lists:foreach(fun(Ind) ->
        ?assert(maps:is_key(age, Ind)),
        ?assertEqual(0, maps:get(age, Ind))  % Initial age is 0
    end, Individuals).

%%% ============================================================================
%%% Age Tracking Tests
%%% ============================================================================

age_increments_on_evaluation_test() ->
    Config = test_init_config(),
    {ok, State, Events} = steady_state_strategy:init(Config),

    [FirstBirth | _] = [E || E <- Events, is_record(E, individual_born)],
    IndId = FirstBirth#individual_born.id,

    %% Initial age should be 0
    Snapshot0 = steady_state_strategy:get_population_snapshot(State),
    [Ind0] = [I || I <- maps:get(individuals, Snapshot0), maps:get(id, I) =:= IndId],
    ?assertEqual(0, maps:get(age, Ind0)),

    %% After one evaluation, age should increment for all individuals
    FitnessResult = #{fitness => 50.0, metrics => #{}},
    {_Actions, _Events1, State1} = steady_state_strategy:handle_evaluation_result(IndId, FitnessResult, State),

    Snapshot1 = steady_state_strategy:get_population_snapshot(State1),
    [Ind1] = [I || I <- maps:get(individuals, Snapshot1), maps:get(id, I) =:= IndId],
    ?assertEqual(1, maps:get(age, Ind1)).

%%% ============================================================================
%%% Meta-Controller Interface Tests
%%% ============================================================================

meta_inputs_returns_floats_test() ->
    Config = test_init_config(),
    {ok, State, _Events} = steady_state_strategy:init(Config),

    Inputs = steady_state_strategy:get_meta_inputs(State),

    ?assert(is_list(Inputs)),
    ?assertEqual(4, length(Inputs)),  % Diversity, age, improvement gap, mutation rate
    lists:foreach(fun(V) ->
        ?assert(is_float(V))
    end, Inputs).

meta_inputs_normalized_test() ->
    Config = test_init_config(),
    {ok, State, _Events} = steady_state_strategy:init(Config),

    Inputs = steady_state_strategy:get_meta_inputs(State),

    lists:foreach(fun(V) ->
        ?assert(V >= 0.0),
        ?assert(V =< 1.0)
    end, Inputs).

apply_params_mutation_rate_test() ->
    Config = test_init_config(),
    {ok, State, _Events} = steady_state_strategy:init(Config),

    NewState = steady_state_strategy:apply_meta_params(#{mutation_rate => 0.25}, State),
    Inputs = steady_state_strategy:get_meta_inputs(NewState),

    %% Last input is mutation rate
    MutationRate = lists:last(Inputs),
    ?assertEqual(0.25, MutationRate).

apply_params_bounds_values_test() ->
    Config = test_init_config(),
    {ok, State, _Events} = steady_state_strategy:init(Config),

    %% Try extreme values
    State1 = steady_state_strategy:apply_meta_params(#{mutation_rate => 2.0}, State),
    Inputs1 = steady_state_strategy:get_meta_inputs(State1),
    MutationRate1 = lists:last(Inputs1),
    ?assert(MutationRate1 =< 0.5),  % Should be bounded

    State2 = steady_state_strategy:apply_meta_params(#{mutation_rate => -1.0}, State),
    Inputs2 = steady_state_strategy:get_meta_inputs(State2),
    MutationRate2 = lists:last(Inputs2),
    ?assert(MutationRate2 >= 0.01).  % Should be bounded

%%% ============================================================================
%%% Tick Tests
%%% ============================================================================

tick_returns_empty_without_max_age_test() ->
    Config = test_init_config(),
    {ok, State, _Events} = steady_state_strategy:init(Config),

    {Actions, Events, _NewState} = steady_state_strategy:tick(State),

    ?assertEqual([], Actions),
    ?assertEqual([], Events).

%%% ============================================================================
%%% Behaviour Callback Tests
%%% ============================================================================

behaviour_init_dispatch_test() ->
    Config = test_init_config(),
    {ok, State, Events} = evolution_strategy:init(steady_state_strategy, Config),

    ?assert(is_tuple(State)),
    ?assert(is_list(Events)),
    BirthEvents = [E || E <- Events, is_record(E, individual_born)],
    ?assertEqual(5, length(BirthEvents)).
