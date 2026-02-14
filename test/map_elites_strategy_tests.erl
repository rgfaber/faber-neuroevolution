%% @doc EUnit tests for map_elites_strategy module.
%%
%% Tests the MAP-Elites quality-diversity evolution strategy including:
%% - Initialization
%% - Grid cell placement
%% - Elite replacement
%% - Batch generation
%% - Coverage and QD-score metrics
%% - Meta-controller interface
%%
%% @author R.G. Lefever
%% @copyright 2024-2026 R.G. Lefever
-module(map_elites_strategy_tests).

-include_lib("eunit/include/eunit.hrl").
-include("neuroevolution.hrl").
-include("evolution_strategy.hrl").
-include("lifecycle_events.hrl").

%%% ============================================================================
%%% Test Fixtures
%%% ============================================================================

test_neuro_config() ->
    #neuro_config{
        population_size = 20,  % Not really used in MAP-Elites
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
            behavior_dimensions => 2,
            bins_per_dimension => 5,  % 5x5 = 25 cells
            behavior_bounds => [{0.0, 10.0}, {0.0, 10.0}],
            batch_size => 5,
            random_probability => 0.10,
            mutation_rate => 0.10,
            mutation_strength => 0.3
        },
        network_factory => mock_network_factory
    }.

%% Helper to create behavior descriptor
make_behavior(Values) when is_list(Values) ->
    Values.

%%% ============================================================================
%%% Initialization Tests
%%% ============================================================================

init_creates_initial_batch_test() ->
    Config = test_init_config(),
    {ok, _State, Events} = map_elites_strategy:init(Config),

    %% Should have batch_size birth events
    BirthEvents = [E || E <- Events, is_record(E, individual_born)],
    ?assertEqual(5, length(BirthEvents)).

init_grid_is_empty_test() ->
    Config = test_init_config(),
    {ok, State, _Events} = map_elites_strategy:init(Config),

    Snapshot = map_elites_strategy:get_population_snapshot(State),

    %% Grid starts empty
    ?assertEqual(0, maps:get(size, Snapshot)),
    Extra = maps:get(extra, Snapshot),
    ?assertEqual(0, maps:get(cells_filled, Extra)).

init_calculates_total_cells_test() ->
    Config = test_init_config(),
    {ok, State, _Events} = map_elites_strategy:init(Config),

    Snapshot = map_elites_strategy:get_population_snapshot(State),
    Extra = maps:get(extra, Snapshot),

    %% 5x5 = 25 total cells
    ?assertEqual(25, maps:get(total_cells, Extra)).

init_with_default_params_test() ->
    Config = #{
        neuro_config => test_neuro_config(),
        network_factory => mock_network_factory
    },
    {ok, State, _Events} = map_elites_strategy:init(Config),

    Snapshot = map_elites_strategy:get_population_snapshot(State),
    Extra = maps:get(extra, Snapshot),

    %% Default: 10x10 = 100 cells
    ?assertEqual(100, maps:get(total_cells, Extra)).

%%% ============================================================================
%%% Evaluation and Grid Placement Tests
%%% ============================================================================

evaluation_places_in_grid_test() ->
    Config = test_init_config(),
    {ok, State, Events} = map_elites_strategy:init(Config),

    %% Get first individual
    [FirstBirth | _] = [E || E <- Events, is_record(E, individual_born)],
    IndId = FirstBirth#individual_born.id,

    %% Evaluate with behavior that maps to a cell
    Behavior = make_behavior([2.5, 2.5]),  % Should map to middle cell
    FitnessResult = #{
        fitness => 100.0,
        metrics => #{behavior => Behavior}
    },

    {_Actions, _EvalEvents, NewState} = map_elites_strategy:handle_evaluation_result(
        IndId, FitnessResult, State
    ),

    Snapshot = map_elites_strategy:get_population_snapshot(NewState),
    Extra = maps:get(extra, Snapshot),

    %% Should have 1 cell filled
    ?assertEqual(1, maps:get(cells_filled, Extra)).

better_fitness_replaces_elite_test() ->
    Config = test_init_config(),
    {ok, State, Events} = map_elites_strategy:init(Config),

    BirthEvents = [E || E <- Events, is_record(E, individual_born)],

    %% Evaluate first two individuals with same behavior, different fitness
    [First, Second | _] = BirthEvents,

    Behavior = make_behavior([5.0, 5.0]),

    %% First individual with low fitness
    {_Actions1, _Events1, State1} = map_elites_strategy:handle_evaluation_result(
        First#individual_born.id,
        #{fitness => 50.0, metrics => #{behavior => Behavior}},
        State
    ),

    %% Second individual with high fitness (same cell)
    {_Actions2, _Events2, State2} = map_elites_strategy:handle_evaluation_result(
        Second#individual_born.id,
        #{fitness => 100.0, metrics => #{behavior => Behavior}},
        State1
    ),

    Snapshot = map_elites_strategy:get_population_snapshot(State2),

    %% Should still have 1 cell (replaced, not added)
    Extra = maps:get(extra, Snapshot),
    ?assertEqual(1, maps:get(cells_filled, Extra)),

    %% Best fitness should be 100
    ?assertEqual(100.0, maps:get(best_fitness, Snapshot)).

lower_fitness_rejected_test() ->
    Config = test_init_config(),
    {ok, State, Events} = map_elites_strategy:init(Config),

    BirthEvents = [E || E <- Events, is_record(E, individual_born)],
    [First, Second | _] = BirthEvents,

    Behavior = make_behavior([5.0, 5.0]),

    %% First with high fitness
    {_, _, State1} = map_elites_strategy:handle_evaluation_result(
        First#individual_born.id,
        #{fitness => 100.0, metrics => #{behavior => Behavior}},
        State
    ),

    %% Second with lower fitness (same cell) - should be rejected
    {_, Events2, State2} = map_elites_strategy:handle_evaluation_result(
        Second#individual_born.id,
        #{fitness => 50.0, metrics => #{behavior => Behavior}},
        State1
    ),

    %% Should have death event for rejected individual
    DeathEvents = [E || E <- Events2, is_record(E, individual_died)],
    ?assertEqual(1, length(DeathEvents)),

    Snapshot = map_elites_strategy:get_population_snapshot(State2),
    ?assertEqual(1, maps:get(size, Snapshot)),
    ?assertEqual(100.0, maps:get(best_fitness, Snapshot)).

different_behaviors_fill_different_cells_test() ->
    Config = test_init_config(),
    {ok, State, Events} = map_elites_strategy:init(Config),

    BirthEvents = [E || E <- Events, is_record(E, individual_born)],
    [First, Second | _] = BirthEvents,

    %% Two different behaviors -> two different cells
    {_, _, State1} = map_elites_strategy:handle_evaluation_result(
        First#individual_born.id,
        #{fitness => 100.0, metrics => #{behavior => make_behavior([1.0, 1.0])}},
        State
    ),

    {_, _, State2} = map_elites_strategy:handle_evaluation_result(
        Second#individual_born.id,
        #{fitness => 100.0, metrics => #{behavior => make_behavior([9.0, 9.0])}},
        State1
    ),

    Snapshot = map_elites_strategy:get_population_snapshot(State2),
    Extra = maps:get(extra, Snapshot),

    %% Should have 2 cells filled
    ?assertEqual(2, maps:get(cells_filled, Extra)).

%%% ============================================================================
%%% Batch Completion Tests
%%% ============================================================================

batch_complete_generates_new_batch_test() ->
    Config = test_init_config(),
    {ok, State, Events} = map_elites_strategy:init(Config),

    BirthEvents = [E || E <- Events, is_record(E, individual_born)],

    %% Evaluate all 5 individuals in batch
    {Actions, _AllEvents, _FinalState} = lists:foldl(
        fun({BirthEvent, Idx}, {AccActions, AccEvents, AccState}) ->
            Behavior = make_behavior([
                erlang:float(Idx) * 2.0,
                erlang:float(Idx) * 2.0
            ]),
            FitnessResult = #{
                fitness => erlang:float(Idx) * 10.0,
                metrics => #{behavior => Behavior}
            },
            {NewActions, NewEvents, NewState} = map_elites_strategy:handle_evaluation_result(
                BirthEvent#individual_born.id, FitnessResult, AccState
            ),
            {AccActions ++ NewActions, AccEvents ++ NewEvents, NewState}
        end,
        {[], [], State},
        lists:zip(BirthEvents, lists:seq(1, length(BirthEvents)))
    ),

    %% Should have action to evaluate new batch
    EvalActions = [A || {evaluate_batch, _} = A <- Actions],
    ?assertEqual(1, length(EvalActions)).

%%% ============================================================================
%%% Snapshot Tests
%%% ============================================================================

snapshot_has_required_fields_test() ->
    Config = test_init_config(),
    {ok, State, _Events} = map_elites_strategy:init(Config),

    Snapshot = map_elites_strategy:get_population_snapshot(State),

    ?assert(maps:is_key(size, Snapshot)),
    ?assert(maps:is_key(individuals, Snapshot)),
    ?assert(maps:is_key(best_fitness, Snapshot)),
    ?assert(maps:is_key(avg_fitness, Snapshot)),
    ?assert(maps:is_key(extra, Snapshot)).

snapshot_extra_has_coverage_metrics_test() ->
    Config = test_init_config(),
    {ok, State, _Events} = map_elites_strategy:init(Config),

    Snapshot = map_elites_strategy:get_population_snapshot(State),
    Extra = maps:get(extra, Snapshot),

    ?assert(maps:is_key(cells_filled, Extra)),
    ?assert(maps:is_key(total_cells, Extra)),
    ?assert(maps:is_key(coverage, Extra)),
    ?assert(maps:is_key(qd_score, Extra)),
    ?assert(maps:is_key(total_evaluations, Extra)).

%%% ============================================================================
%%% Meta-Controller Interface Tests
%%% ============================================================================

meta_inputs_returns_floats_test() ->
    Config = test_init_config(),
    {ok, State, _Events} = map_elites_strategy:init(Config),

    Inputs = map_elites_strategy:get_meta_inputs(State),

    ?assert(is_list(Inputs)),
    ?assertEqual(4, length(Inputs)),
    lists:foreach(fun(V) ->
        ?assert(is_float(V))
    end, Inputs).

meta_inputs_normalized_test() ->
    Config = test_init_config(),
    {ok, State, _Events} = map_elites_strategy:init(Config),

    Inputs = map_elites_strategy:get_meta_inputs(State),

    lists:foreach(fun(V) ->
        ?assert(V >= 0.0),
        ?assert(V =< 1.0)
    end, Inputs).

apply_meta_params_updates_state_test() ->
    Config = test_init_config(),
    {ok, State, _Events} = map_elites_strategy:init(Config),

    NewState = map_elites_strategy:apply_meta_params(#{
        mutation_rate => 0.5,
        mutation_strength => 0.8,
        random_probability => 0.25
    }, State),

    %% State should still function
    Snapshot = map_elites_strategy:get_population_snapshot(NewState),
    ?assertEqual(0, maps:get(size, Snapshot)).

%%% ============================================================================
%%% Tick Tests
%%% ============================================================================

tick_returns_empty_actions_test() ->
    Config = test_init_config(),
    {ok, State, _Events} = map_elites_strategy:init(Config),

    {Actions, Events, _NewState} = map_elites_strategy:tick(State),

    ?assertEqual([], Actions),
    ?assertEqual([], Events).

%%% ============================================================================
%%% Behaviour Callback Tests
%%% ============================================================================

behaviour_init_dispatch_test() ->
    Config = test_init_config(),
    {ok, State, Events} = evolution_strategy:init(map_elites_strategy, Config),

    ?assert(is_tuple(State)),
    ?assert(is_list(Events)),
    BirthEvents = [E || E <- Events, is_record(E, individual_born)],
    ?assertEqual(5, length(BirthEvents)).

%%% ============================================================================
%%% Edge Case Tests
%%% ============================================================================

no_behavior_not_placed_test() ->
    Config = test_init_config(),
    {ok, State, Events} = map_elites_strategy:init(Config),

    [FirstBirth | _] = [E || E <- Events, is_record(E, individual_born)],

    %% Evaluate without behavior
    FitnessResult = #{fitness => 100.0, metrics => #{}},

    {_Actions, _EvalEvents, NewState} = map_elites_strategy:handle_evaluation_result(
        FirstBirth#individual_born.id, FitnessResult, State
    ),

    Snapshot = map_elites_strategy:get_population_snapshot(NewState),
    Extra = maps:get(extra, Snapshot),

    %% Should have 0 cells filled
    ?assertEqual(0, maps:get(cells_filled, Extra)).

behavior_at_boundary_test() ->
    Config = test_init_config(),
    {ok, State, Events} = map_elites_strategy:init(Config),

    [FirstBirth | _] = [E || E <- Events, is_record(E, individual_born)],

    %% Behavior at boundary (max values)
    Behavior = make_behavior([10.0, 10.0]),
    FitnessResult = #{
        fitness => 100.0,
        metrics => #{behavior => Behavior}
    },

    {_Actions, _EvalEvents, NewState} = map_elites_strategy:handle_evaluation_result(
        FirstBirth#individual_born.id, FitnessResult, State
    ),

    Snapshot = map_elites_strategy:get_population_snapshot(NewState),
    Extra = maps:get(extra, Snapshot),

    %% Should handle boundary correctly
    ?assertEqual(1, maps:get(cells_filled, Extra)).

behavior_out_of_bounds_clamped_test() ->
    Config = test_init_config(),
    {ok, State, Events} = map_elites_strategy:init(Config),

    [FirstBirth | _] = [E || E <- Events, is_record(E, individual_born)],

    %% Behavior outside bounds
    Behavior = make_behavior([100.0, -50.0]),
    FitnessResult = #{
        fitness => 100.0,
        metrics => #{behavior => Behavior}
    },

    {_Actions, _EvalEvents, NewState} = map_elites_strategy:handle_evaluation_result(
        FirstBirth#individual_born.id, FitnessResult, State
    ),

    Snapshot = map_elites_strategy:get_population_snapshot(NewState),
    Extra = maps:get(extra, Snapshot),

    %% Should clamp to valid cell
    ?assertEqual(1, maps:get(cells_filled, Extra)).

%%% ============================================================================
%%% QD-Score Tests
%%% ============================================================================

qd_score_sums_elite_fitness_test() ->
    Config = test_init_config(),
    {ok, State, Events} = map_elites_strategy:init(Config),

    BirthEvents = [E || E <- Events, is_record(E, individual_born)],
    [First, Second | _] = BirthEvents,

    %% Add two elites with known fitness
    {_, _, State1} = map_elites_strategy:handle_evaluation_result(
        First#individual_born.id,
        #{fitness => 100.0, metrics => #{behavior => make_behavior([1.0, 1.0])}},
        State
    ),

    {_, _, State2} = map_elites_strategy:handle_evaluation_result(
        Second#individual_born.id,
        #{fitness => 150.0, metrics => #{behavior => make_behavior([9.0, 9.0])}},
        State1
    ),

    Snapshot = map_elites_strategy:get_population_snapshot(State2),
    Extra = maps:get(extra, Snapshot),

    %% QD score should be sum of fitnesses: 100 + 150 = 250
    ?assertEqual(250.0, maps:get(qd_score, Extra)).
