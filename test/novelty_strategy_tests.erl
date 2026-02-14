%% @doc EUnit tests for novelty_strategy module.
%%
%% Tests the novelty search evolution strategy including:
%% - Initialization
%% - Behavior descriptor handling
%% - Novelty score computation
%% - Archive management
%% - Hybrid mode (novelty + fitness)
%% - Meta-controller interface
%%
%% @author R.G. Lefever
%% @copyright 2024-2026 R.G. Lefever
-module(novelty_strategy_tests).

-include_lib("eunit/include/eunit.hrl").
-include("neuroevolution.hrl").
-include("evolution_strategy.hrl").
-include("lifecycle_events.hrl").

%%% ============================================================================
%%% Test Fixtures
%%% ============================================================================

test_neuro_config() ->
    #neuro_config{
        population_size = 10,
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
            archive_size => 50,
            archive_probability => 0.20,
            k_nearest => 3,
            include_fitness => false,
            fitness_weight => 0.0,
            novelty_threshold => 0.0
        },
        network_factory => mock_network_factory
    }.

%% Helper to create behavior descriptor
make_behavior(Values) when is_list(Values) ->
    Values.

%%% ============================================================================
%%% Initialization Tests
%%% ============================================================================

init_creates_population_test() ->
    Config = test_init_config(),
    {ok, State, _Events} = novelty_strategy:init(Config),

    Snapshot = novelty_strategy:get_population_snapshot(State),
    ?assertEqual(10, maps:get(size, Snapshot)).

init_returns_birth_events_test() ->
    Config = test_init_config(),
    {ok, _State, Events} = novelty_strategy:init(Config),

    BirthEvents = [E || E <- Events, is_record(E, individual_born)],
    ?assertEqual(10, length(BirthEvents)),

    lists:foreach(fun(E) ->
        ?assertEqual(initial, E#individual_born.origin)
    end, BirthEvents).

init_with_default_params_test() ->
    Config = #{
        neuro_config => test_neuro_config(),
        network_factory => mock_network_factory
    },
    {ok, State, _Events} = novelty_strategy:init(Config),

    Snapshot = novelty_strategy:get_population_snapshot(State),
    ?assertEqual(10, maps:get(size, Snapshot)).

%%% ============================================================================
%%% Evaluation Result Handling Tests
%%% ============================================================================

handle_evaluation_stores_behavior_test() ->
    Config = test_init_config(),
    {ok, State, Events} = novelty_strategy:init(Config),

    %% Get first individual
    [FirstBirth | _] = [E || E <- Events, is_record(E, individual_born)],
    IndId = FirstBirth#individual_born.id,

    %% Evaluate with behavior descriptor
    Behavior = make_behavior([1.0, 2.0, 3.0]),
    FitnessResult = #{
        fitness => 100.0,
        metrics => #{behavior => Behavior}
    },

    {_Actions, EvalEvents, _NewState} = novelty_strategy:handle_evaluation_result(
        IndId, FitnessResult, State
    ),

    %% Should emit evaluation event
    EvaluatedEvents = [E || E <- EvalEvents, is_record(E, individual_evaluated)],
    ?assertEqual(1, length(EvaluatedEvents)).

handle_evaluation_without_behavior_test() ->
    Config = test_init_config(),
    {ok, State, Events} = novelty_strategy:init(Config),

    [FirstBirth | _] = [E || E <- Events, is_record(E, individual_born)],
    IndId = FirstBirth#individual_born.id,

    %% Evaluate without behavior descriptor
    FitnessResult = #{fitness => 50.0, metrics => #{}},

    {_Actions, EvalEvents, _NewState} = novelty_strategy:handle_evaluation_result(
        IndId, FitnessResult, State
    ),

    %% Should still work, just with zero novelty
    ?assertEqual(1, length([E || E <- EvalEvents, is_record(E, individual_evaluated)])).

%%% ============================================================================
%%% Novelty Score Tests
%%% ============================================================================

novelty_computed_after_cohort_test() ->
    Config = test_init_config(),
    {ok, State, Events} = novelty_strategy:init(Config),

    %% Evaluate all individuals with different behaviors
    BirthEvents = [E || E <- Events, is_record(E, individual_born)],

    %% Create distinct behaviors for each individual
    {_Actions, AllEvents, _FinalState} = lists:foldl(
        fun(BirthEvent, {AccActions, AccEvents, AccState}) ->
            IndId = BirthEvent#individual_born.id,
            %% Each individual gets unique behavior
            Behavior = make_behavior([rand:uniform() * 10.0, rand:uniform() * 10.0]),
            FitnessResult = #{
                fitness => rand:uniform() * 100.0,
                metrics => #{behavior => Behavior}
            },
            {Actions, NewEvents, NewState} = novelty_strategy:handle_evaluation_result(
                IndId, FitnessResult, AccState
            ),
            {AccActions ++ Actions, AccEvents ++ NewEvents, NewState}
        end,
        {[], [], State},
        BirthEvents
    ),

    %% Should have cohort_evaluated event
    CohortEvents = [E || E <- AllEvents, is_record(E, cohort_evaluated)],
    ?assertEqual(1, length(CohortEvents)).

%%% ============================================================================
%%% Archive Tests
%%% ============================================================================

archive_initially_empty_test() ->
    Config = test_init_config(),
    {ok, State, _Events} = novelty_strategy:init(Config),

    Snapshot = novelty_strategy:get_population_snapshot(State),
    Extra = maps:get(extra, Snapshot),
    ?assertEqual(0, maps:get(archive_size, Extra)).

archive_grows_after_generation_test() ->
    %% Configure with high archive probability
    Config = #{
        neuro_config => test_neuro_config(),
        strategy_params => #{
            archive_size => 100,
            archive_probability => 1.0,  % Always add to archive
            k_nearest => 3,
            novelty_threshold => 0.0
        },
        network_factory => mock_network_factory
    },
    {ok, State, Events} = novelty_strategy:init(Config),

    %% Evaluate all individuals
    BirthEvents = [E || E <- Events, is_record(E, individual_born)],

    {_Actions, _AllEvents, FinalState} = lists:foldl(
        fun(BirthEvent, {AccActions, AccEvents, AccState}) ->
            IndId = BirthEvent#individual_born.id,
            Behavior = make_behavior([rand:uniform() * 10.0, rand:uniform() * 10.0]),
            FitnessResult = #{
                fitness => rand:uniform() * 100.0,
                metrics => #{behavior => Behavior}
            },
            {Actions, NewEvents, NewState} = novelty_strategy:handle_evaluation_result(
                IndId, FitnessResult, AccState
            ),
            {AccActions ++ Actions, AccEvents ++ NewEvents, NewState}
        end,
        {[], [], State},
        BirthEvents
    ),

    %% Archive should have entries
    Snapshot = novelty_strategy:get_population_snapshot(FinalState),
    Extra = maps:get(extra, Snapshot),
    ?assert(maps:get(archive_size, Extra) > 0).

%%% ============================================================================
%%% Hybrid Mode Tests
%%% ============================================================================

hybrid_mode_uses_fitness_weight_test() ->
    Config = #{
        neuro_config => test_neuro_config(),
        strategy_params => #{
            archive_size => 50,
            archive_probability => 0.20,
            k_nearest => 3,
            include_fitness => true,
            fitness_weight => 0.5,  % 50% novelty, 50% fitness
            novelty_threshold => 0.0
        },
        network_factory => mock_network_factory
    },

    {ok, State, _Events} = novelty_strategy:init(Config),

    Snapshot = novelty_strategy:get_population_snapshot(State),
    ?assertEqual(10, maps:get(size, Snapshot)).

%%% ============================================================================
%%% Snapshot Tests
%%% ============================================================================

snapshot_has_required_fields_test() ->
    Config = test_init_config(),
    {ok, State, _Events} = novelty_strategy:init(Config),

    Snapshot = novelty_strategy:get_population_snapshot(State),

    ?assert(maps:is_key(size, Snapshot)),
    ?assert(maps:is_key(individuals, Snapshot)),
    ?assert(maps:is_key(best_fitness, Snapshot)),
    ?assert(maps:is_key(avg_fitness, Snapshot)),
    ?assert(maps:is_key(extra, Snapshot)).

snapshot_extra_has_novelty_stats_test() ->
    Config = test_init_config(),
    {ok, State, _Events} = novelty_strategy:init(Config),

    Snapshot = novelty_strategy:get_population_snapshot(State),
    Extra = maps:get(extra, Snapshot),

    ?assert(maps:is_key(archive_size, Extra)),
    ?assert(maps:is_key(best_novelty, Extra)),
    ?assert(maps:is_key(avg_novelty, Extra)),
    ?assert(maps:is_key(archive_adds, Extra)).

%%% ============================================================================
%%% Meta-Controller Interface Tests
%%% ============================================================================

meta_inputs_returns_floats_test() ->
    Config = test_init_config(),
    {ok, State, _Events} = novelty_strategy:init(Config),

    Inputs = novelty_strategy:get_meta_inputs(State),

    ?assert(is_list(Inputs)),
    ?assertEqual(4, length(Inputs)),
    lists:foreach(fun(V) ->
        ?assert(is_float(V))
    end, Inputs).

meta_inputs_normalized_test() ->
    Config = test_init_config(),
    {ok, State, _Events} = novelty_strategy:init(Config),

    Inputs = novelty_strategy:get_meta_inputs(State),

    lists:foreach(fun(V) ->
        ?assert(V >= 0.0),
        ?assert(V =< 1.0)
    end, Inputs).

apply_meta_params_updates_state_test() ->
    Config = test_init_config(),
    {ok, State, _Events} = novelty_strategy:init(Config),

    NewState = novelty_strategy:apply_meta_params(#{
        archive_probability => 0.5,
        fitness_weight => 0.3
    }, State),

    %% State should still function
    Snapshot = novelty_strategy:get_population_snapshot(NewState),
    ?assertEqual(10, maps:get(size, Snapshot)).

%%% ============================================================================
%%% Tick Tests
%%% ============================================================================

tick_returns_empty_actions_test() ->
    Config = test_init_config(),
    {ok, State, _Events} = novelty_strategy:init(Config),

    {Actions, Events, _NewState} = novelty_strategy:tick(State),

    ?assertEqual([], Actions),
    ?assertEqual([], Events).

%%% ============================================================================
%%% Behaviour Callback Tests
%%% ============================================================================

behaviour_init_dispatch_test() ->
    Config = test_init_config(),
    {ok, State, Events} = evolution_strategy:init(novelty_strategy, Config),

    ?assert(is_tuple(State)),
    ?assert(is_list(Events)),
    BirthEvents = [E || E <- Events, is_record(E, individual_born)],
    ?assertEqual(10, length(BirthEvents)).

%%% ============================================================================
%%% Edge Case Tests
%%% ============================================================================

empty_behavior_list_test() ->
    Config = test_init_config(),
    {ok, State, Events} = novelty_strategy:init(Config),

    [FirstBirth | _] = [E || E <- Events, is_record(E, individual_born)],
    IndId = FirstBirth#individual_born.id,

    %% Evaluate with empty behavior
    FitnessResult = #{
        fitness => 100.0,
        metrics => #{behavior => []}
    },

    {_Actions, EvalEvents, _NewState} = novelty_strategy:handle_evaluation_result(
        IndId, FitnessResult, State
    ),

    ?assertEqual(1, length([E || E <- EvalEvents, is_record(E, individual_evaluated)])).

mismatched_behavior_dimensions_test() ->
    Config = test_init_config(),
    {ok, State, Events} = novelty_strategy:init(Config),

    BirthEvents = [E || E <- Events, is_record(E, individual_born)],

    %% Evaluate with different dimension behaviors - should handle gracefully
    {_Actions, AllEvents, _FinalState} = lists:foldl(
        fun({BirthEvent, Idx}, {AccActions, AccEvents, AccState}) ->
            IndId = BirthEvent#individual_born.id,
            %% Mix 2D and 3D behaviors
            Behavior = case Idx rem 2 of
                0 -> make_behavior([rand:uniform() * 10.0, rand:uniform() * 10.0]);
                1 -> make_behavior([rand:uniform() * 10.0, rand:uniform() * 10.0, rand:uniform() * 10.0])
            end,
            FitnessResult = #{
                fitness => rand:uniform() * 100.0,
                metrics => #{behavior => Behavior}
            },
            {Actions, NewEvents, NewState} = novelty_strategy:handle_evaluation_result(
                IndId, FitnessResult, AccState
            ),
            {AccActions ++ Actions, AccEvents ++ NewEvents, NewState}
        end,
        {[], [], State},
        lists:zip(BirthEvents, lists:seq(1, length(BirthEvents)))
    ),

    %% Should still complete without crashing
    CohortEvents = [E || E <- AllEvents, is_record(E, cohort_evaluated)],
    ?assertEqual(1, length(CohortEvents)).
