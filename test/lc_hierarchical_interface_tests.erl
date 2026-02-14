%% @doc Integration tests for L2→L1 Hierarchical Interface.
%%
%% Tests the L2 guidance mechanism in task_silo without requiring
%% the full meta_controller infrastructure. Focus on:
%% - L2 guidance record usage
%% - task_silo receiving and applying L2 guidance
%% - L0 bounds enforcement under L2 guidance
%%
%% NOTE: task_silo registers as {local, task_silo} so tests must be run
%% sequentially with proper cleanup between each.
%%
%% @author R.G. Lefever
%% @copyright 2024-2026 R.G. Lefever
-module(lc_hierarchical_interface_tests).

-include_lib("eunit/include/eunit.hrl").
-include("meta_controller.hrl").

%%% ============================================================================
%%% L2 Guidance Record Tests (no process required)
%%% ============================================================================

l2_guidance_record_test_() ->
    [
        {"L2 guidance record has aggressive default values",
         fun() ->
             Guidance = ?L2_GUIDANCE_DEFAULTS,
             %% Aggressive defaults for rule-based L1
             ?assertEqual(1.5, Guidance#l2_guidance.aggression_factor),
             ?assertEqual(0.5, Guidance#l2_guidance.exploration_step),
             ?assertEqual(0.001, Guidance#l2_guidance.stagnation_sensitivity),
             ?assertEqual(2.5, Guidance#l2_guidance.topology_aggression),
             ?assertEqual(0.3, Guidance#l2_guidance.exploitation_weight)
         end},
        {"L2 guidance bounds are properly defined",
         fun() ->
             Bounds = ?L2_GUIDANCE_BOUNDS,
             ?assert(is_map(Bounds)),
             %% Check key bounds exist
             ?assert(maps:is_key(aggression_factor, Bounds)),
             ?assert(maps:is_key(exploration_step, Bounds)),
             ?assert(maps:is_key(topology_aggression, Bounds)),
             %% Verify bound ranges
             {AFMin, AFMax} = maps:get(aggression_factor, Bounds),
             ?assertEqual(0.0, AFMin),
             ?assertEqual(2.0, AFMax),
             %% Exploration step bounds
             {ESMin, ESMax} = maps:get(exploration_step, Bounds),
             ?assertEqual(0.05, ESMin),
             ?assertEqual(0.5, ESMax)
         end},
        {"L2 guidance record can be modified",
         fun() ->
             Guidance = ?L2_GUIDANCE_DEFAULTS,
             Modified = Guidance#l2_guidance{
                 aggression_factor = 1.8,
                 exploration_step = 0.3
             },
             ?assertEqual(1.8, Modified#l2_guidance.aggression_factor),
             ?assertEqual(0.3, Modified#l2_guidance.exploration_step),
             %% Unchanged values remain default
             ?assertEqual(0.001, Modified#l2_guidance.stagnation_sensitivity)
         end},
        {"L2 guidance has threshold fields",
         fun() ->
             Guidance = ?L2_GUIDANCE_DEFAULTS,
             ?assert(is_float(Guidance#l2_guidance.warning_threshold)),
             ?assert(is_float(Guidance#l2_guidance.intervention_threshold)),
             ?assert(is_float(Guidance#l2_guidance.critical_threshold)),
             %% Thresholds should be in increasing order
             ?assert(Guidance#l2_guidance.warning_threshold <
                     Guidance#l2_guidance.intervention_threshold),
             ?assert(Guidance#l2_guidance.intervention_threshold <
                     Guidance#l2_guidance.critical_threshold)
         end},
        {"L2 guidance has resource silo fields",
         fun() ->
             Guidance = ?L2_GUIDANCE_DEFAULTS,
             ?assert(is_float(Guidance#l2_guidance.memory_high_threshold)),
             ?assert(is_float(Guidance#l2_guidance.memory_critical_threshold)),
             ?assert(is_float(Guidance#l2_guidance.cpu_high_threshold))
         end}
    ].

%%% ============================================================================
%%% Task Silo L2 Integration Sequential Tests
%%% ============================================================================

%% Run all task_silo tests sequentially in a single test to avoid registration conflicts
task_silo_l2_integration_test() ->
    %% Ensure clean state
    catch gen_server:stop(task_silo),
    catch unregister(task_silo),
    timer:sleep(100),

    try
        %% Test 1: Start with L2 enabled
        test_task_silo_l2_enabled(),

        %% Cleanup
        catch gen_server:stop(task_silo),
        timer:sleep(50),

        %% Test 2: Accept and apply L2 guidance
        test_task_silo_set_l2_guidance(),

        %% Cleanup
        catch gen_server:stop(task_silo),
        timer:sleep(50),

        %% Test 3: L2 guidance in state output
        test_task_silo_l2_guidance_in_state(),

        %% Cleanup
        catch gen_server:stop(task_silo),
        timer:sleep(50),

        %% Test 4: Different aggression factors produce different results
        test_aggression_factor_impact(),

        %% Cleanup
        catch gen_server:stop(task_silo),
        timer:sleep(50),

        %% Test 5: L0 bounds enforcement
        test_l0_bounds_enforcement(),

        %% Cleanup
        catch gen_server:stop(task_silo),
        timer:sleep(50),

        %% Test 6: Backward compatibility
        test_backward_compatibility()

    after
        catch gen_server:stop(task_silo),
        catch unregister(task_silo)
    end.

%%% ============================================================================
%%% Individual Test Functions
%%% ============================================================================

test_task_silo_l2_enabled() ->
    Config = #{
        realm => <<"test">>,
        l2_enabled => true
    },
    {ok, _Pid} = task_silo:start_link(Config),

    %% Get state - l2_guidance should have default values
    State = task_silo:get_state(task_silo),
    L2Info = maps:get(l2_guidance, State, #{}),

    %% Should have L2 guidance values
    ?assert(maps:is_key(aggression_factor, L2Info)),
    ?assertEqual(1.5, maps:get(aggression_factor, L2Info, 0.0)),
    ok.

test_task_silo_set_l2_guidance() ->
    Config = #{
        realm => <<"test">>,
        l2_enabled => true
    },
    {ok, _Pid} = task_silo:start_link(Config),

    %% Create custom L2 guidance
    Guidance = #l2_guidance{
        aggression_factor = 1.8,
        exploration_step = 0.4,
        stagnation_sensitivity = 0.002,
        topology_aggression = 2.0,
        exploitation_weight = 0.6,
        adaptation_momentum = 0.5,
        warning_threshold = 0.25,
        intervention_threshold = 0.5,
        critical_threshold = 0.8,
        velocity_window_size = 8,
        memory_high_threshold = 0.8,
        memory_critical_threshold = 0.95,
        cpu_high_threshold = 0.85,
        pressure_scale_factor = 0.85,
        min_scale_factor = 0.15,
        pressure_change_threshold = 0.08,
        generation = 0
    },

    %% Set the guidance
    ok = task_silo:set_l2_guidance(task_silo, Guidance),
    timer:sleep(50),

    %% Verify guidance was applied
    State = task_silo:get_state(task_silo),
    L2Info = maps:get(l2_guidance, State, #{}),
    ?assertEqual(1.8, maps:get(aggression_factor, L2Info, 0.0)),
    ?assertEqual(0.4, maps:get(exploration_step, L2Info, 0.0)),
    ok.

test_task_silo_l2_guidance_in_state() ->
    Config = #{realm => <<"test">>},
    {ok, _Pid} = task_silo:start_link(Config),

    State = task_silo:get_state(task_silo),

    %% Should have l2_guidance map in state
    ?assert(maps:is_key(l2_guidance, State)),
    L2Info = maps:get(l2_guidance, State),
    ?assert(is_map(L2Info)),

    %% Should have all key guidance values
    ?assert(maps:is_key(aggression_factor, L2Info)),
    ?assert(maps:is_key(exploration_step, L2Info)),
    ?assert(maps:is_key(topology_aggression, L2Info)),
    ok.

test_aggression_factor_impact() ->
    Config = #{
        realm => <<"test">>,
        l2_enabled => true
    },
    {ok, _Pid} = task_silo:start_link(Config),

    %% Stagnating stats to trigger adjustments
    Stats = #{
        max_fitness => 0.3,
        avg_fitness => 0.2,
        improvement => 0.0,
        stagnation_counter => 20
    },

    %% Conservative (low aggression)
    Defaults1 = ?L2_GUIDANCE_DEFAULTS,
    LowAggr = Defaults1#l2_guidance{aggression_factor = 0.2},
    ok = task_silo:set_l2_guidance(task_silo, LowAggr),
    timer:sleep(30),
    LowRecs = task_silo:get_recommendations(task_silo, Stats),

    %% Aggressive (high aggression)
    Defaults2 = ?L2_GUIDANCE_DEFAULTS,
    HighAggr = Defaults2#l2_guidance{aggression_factor = 2.0},
    ok = task_silo:set_l2_guidance(task_silo, HighAggr),
    timer:sleep(30),
    HighRecs = task_silo:get_recommendations(task_silo, Stats),

    %% High aggression should produce higher or equal mutation rates
    LowMR = maps:get(mutation_rate, LowRecs, 0.1),
    HighMR = maps:get(mutation_rate, HighRecs, 0.1),
    ?assert(HighMR >= LowMR),
    ok.

test_l0_bounds_enforcement() ->
    Config = #{
        realm => <<"test">>,
        l2_enabled => true
    },
    {ok, _Pid} = task_silo:start_link(Config),

    %% Set extremely aggressive L2 guidance
    AggressiveGuidance = #l2_guidance{
        aggression_factor = 2.0,
        exploration_step = 0.5,
        stagnation_sensitivity = 0.0001,
        topology_aggression = 3.0,
        exploitation_weight = 0.1,
        adaptation_momentum = 0.1,
        warning_threshold = 0.1,
        intervention_threshold = 0.2,
        critical_threshold = 0.5,
        velocity_window_size = 3,
        memory_high_threshold = 0.7,
        memory_critical_threshold = 0.9,
        cpu_high_threshold = 0.9,
        pressure_scale_factor = 0.9,
        min_scale_factor = 0.1,
        pressure_change_threshold = 0.05,
        generation = 0
    },
    ok = task_silo:set_l2_guidance(task_silo, AggressiveGuidance),
    timer:sleep(50),

    %% Extreme stagnation
    Stats = #{
        max_fitness => 0.1,
        avg_fitness => 0.05,
        improvement => 0.0,
        stagnation_counter => 100
    },
    Recs = task_silo:get_recommendations(task_silo, Stats),

    %% L0 bounds should still be enforced
    MR = maps:get(mutation_rate, Recs, 0.1),
    MS = maps:get(mutation_strength, Recs, 0.1),
    SR = maps:get(selection_ratio, Recs, 0.2),

    %% Verify within L0 bounds (from task_l0_defaults)
    ?assert(MR >= 0.01),
    ?assert(MR =< 0.50),
    ?assert(MS >= 0.05),
    ?assert(MS =< 1.0),
    ?assert(SR >= 0.05),
    ?assert(SR =< 0.50),
    ok.

test_backward_compatibility() ->
    Config = #{
        realm => <<"test">>,
        l2_enabled => false
    },
    {ok, _Pid} = task_silo:start_link(Config),

    Stats = #{
        max_fitness => 0.5,
        avg_fitness => 0.4,
        improvement => 0.01,
        stagnation_counter => 0
    },
    Recs = task_silo:get_recommendations(task_silo, Stats),

    %% Should produce valid recommendations
    ?assert(is_map(Recs)),
    ?assert(maps:is_key(mutation_rate, Recs)),
    ?assert(maps:is_key(selection_ratio, Recs)),
    ?assert(maps:is_key(mutation_strength, Recs)),

    %% Get state and verify defaults
    State = task_silo:get_state(task_silo),
    L2Info = maps:get(l2_guidance, State, #{}),

    %% Should use aggressive default values even when L2 disabled
    ?assertEqual(1.5, maps:get(aggression_factor, L2Info, -1)),
    ?assertEqual(0.5, maps:get(exploration_step, L2Info, -1)),
    ?assertEqual(2.5, maps:get(topology_aggression, L2Info, -1)),
    ok.
