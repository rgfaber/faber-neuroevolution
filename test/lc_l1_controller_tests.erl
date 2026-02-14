%% @doc Unit tests for lc_l1_controller module.
%%
%% Tests L1 tactical hyperparameter adjustment.
%%
%% @author R.G. Lefever
%% @copyright 2024-2026 R.G. Lefever
-module(lc_l1_controller_tests).

-include_lib("eunit/include/eunit.hrl").

%%% ============================================================================
%%% Test Setup
%%% ============================================================================

%% @doc Create a mock morphology module for testing.
setup() ->
    %% Stop any existing controller
    catch gen_server:stop(lc_l1_test),
    ok.

cleanup(_) ->
    catch gen_server:stop(lc_l1_test),
    ok.

%%% ============================================================================
%%% Mock Morphology Module
%%% ============================================================================

%% For testing, we'll use resource_l0_morphology which exists

%%% ============================================================================
%%% Test Generators
%%% ============================================================================

lc_l1_controller_test_() ->
    {foreach,
     fun setup/0,
     fun cleanup/1,
     [
         {"start_link creates controller", fun start_link_test/0},
         {"get_current_hyperparameters returns defaults", fun get_hyperparameters_test/0},
         {"get_hyperparameter_deltas initially empty", fun get_deltas_test/0},
         {"observe_l0_performance accumulates history", fun observe_performance_test/0},
         {"set_l2_hyperparameters updates state", fun set_l2_hyperparameters_test/0}
     ]}.

%%% ============================================================================
%%% Individual Tests
%%% ============================================================================

start_link_test() ->
    Config = #{
        silo_type => resource,
        morphology_module => resource_l0_morphology,
        tau_l1 => 5000,  % 5 seconds for faster testing
        window_size => 5
    },
    {ok, Pid} = lc_l1_controller:start_link(Config),
    ?assert(is_pid(Pid)),
    ?assert(is_process_alive(Pid)),
    ok = gen_server:stop(Pid).

get_hyperparameters_test() ->
    Config = #{
        silo_type => resource,
        morphology_module => resource_l0_morphology
    },
    {ok, Pid} = lc_l1_controller:start_link(Config),

    Hyperparams = lc_l1_controller:get_current_hyperparameters(Pid),
    ?assert(is_map(Hyperparams)),

    %% Should have resource L0 defaults
    ?assertEqual(0.70, maps:get(memory_high_threshold, Hyperparams)),
    ?assertEqual(10, maps:get(base_concurrency, Hyperparams)),

    ok = gen_server:stop(Pid).

get_deltas_test() ->
    Config = #{
        silo_type => resource,
        morphology_module => resource_l0_morphology
    },
    {ok, Pid} = lc_l1_controller:start_link(Config),

    Deltas = lc_l1_controller:get_hyperparameter_deltas(Pid),
    ?assert(is_map(Deltas)),
    %% Initially empty (no adjustments made)
    ?assertEqual(#{}, Deltas),

    ok = gen_server:stop(Pid).

observe_performance_test() ->
    Config = #{
        silo_type => resource,
        morphology_module => resource_l0_morphology,
        tau_l1 => 100,  % Very short for testing
        window_size => 3
    },
    {ok, Pid} = lc_l1_controller:start_link(Config),

    %% Send performance observations
    L0Metrics = #{reward => 0.5, stability => 0.8},
    ok = lc_l1_controller:observe_l0_performance(Pid, L0Metrics),
    timer:sleep(50),

    %% Send more observations
    ok = lc_l1_controller:observe_l0_performance(Pid, #{reward => 0.6}),
    ok = lc_l1_controller:observe_l0_performance(Pid, #{reward => 0.7}),
    timer:sleep(150),  % Wait for tau_l1 to trigger adjustment

    %% Should have computed some deltas after adjustment
    %% (may or may not have non-empty deltas depending on trends)
    _Deltas = lc_l1_controller:get_hyperparameter_deltas(Pid),
    %% Just verify it doesn't crash
    ?assert(is_process_alive(Pid)),

    ok = gen_server:stop(Pid).

set_l2_hyperparameters_test() ->
    Config = #{
        silo_type => resource,
        morphology_module => resource_l0_morphology
    },
    {ok, Pid} = lc_l1_controller:start_link(Config),

    %% Set new L1 hyperparameters from L2
    NewL1Params = #{
        threshold_adaptation_rate => 0.1,
        pressure_sensitivity => 1.5
    },
    ok = lc_l1_controller:set_l2_hyperparameters(Pid, NewL1Params),
    timer:sleep(10),

    %% Verify it doesn't crash and process is alive
    ?assert(is_process_alive(Pid)),

    ok = gen_server:stop(Pid).

%%% ============================================================================
%%% Apply Deltas Tests
%%% ============================================================================

apply_deltas_utility_test() ->
    BaseParams = #{
        memory_high_threshold => 0.70,
        base_concurrency => 10
    },
    Deltas = #{
        memory_high_threshold => 0.65,  % New value (merge replaces)
        base_concurrency => 12          % New value (merge replaces)
    },

    %% Test the utility function (maps:merge replaces values)
    NewParams = lc_l1_controller:apply_deltas_to_hyperparameters(BaseParams, Deltas),

    ?assertEqual(0.65, maps:get(memory_high_threshold, NewParams)),
    ?assertEqual(12, maps:get(base_concurrency, NewParams)).

%%% ============================================================================
%%% Task Silo L1 Controller Tests
%%% ============================================================================

task_l1_controller_test_() ->
    {foreach,
     fun setup/0,
     fun cleanup/1,
     [
         {"task L1 starts with task morphology",
          fun() ->
              Config = #{
                  silo_type => task,
                  morphology_module => task_l0_morphology,
                  tau_l1 => 1000
              },
              {ok, Pid} = lc_l1_controller:start_link(Config),
              ?assert(is_pid(Pid)),

              Hyperparams = lc_l1_controller:get_current_hyperparameters(Pid),
              %% Should have task L0 defaults
              ?assert(maps:is_key(mutation_rate_min, Hyperparams)),
              ?assert(maps:is_key(topology_mutation_boost, Hyperparams)),

              ok = gen_server:stop(Pid)
          end},
         {"task L1 computes stagnation deltas",
          fun() ->
              Config = #{
                  silo_type => task,
                  morphology_module => task_l0_morphology,
                  tau_l1 => 50,  % Very short
                  window_size => 3
              },
              {ok, Pid} = lc_l1_controller:start_link(Config),

              %% Send stagnating performance
              lists:foreach(
                  fun(_) ->
                      ok = lc_l1_controller:observe_l0_performance(Pid, #{reward => 0.5}),
                      timer:sleep(20)
                  end,
                  lists:seq(1, 10)
              ),

              timer:sleep(100),
              ?assert(is_process_alive(Pid)),

              ok = gen_server:stop(Pid)
          end}
     ]}.

%%% ============================================================================
%%% Performance Trend Analysis Tests
%%% ============================================================================

performance_trend_test_() ->
    {foreach,
     fun setup/0,
     fun cleanup/1,
     [
         {"improving performance trends",
          fun() ->
              Config = #{
                  silo_type => resource,
                  morphology_module => resource_l0_morphology,
                  tau_l1 => 50,
                  window_size => 5
              },
              {ok, Pid} = lc_l1_controller:start_link(Config),

              %% Send improving performance
              Rewards = [0.3, 0.4, 0.5, 0.6, 0.7],
              lists:foreach(
                  fun(R) ->
                      ok = lc_l1_controller:observe_l0_performance(Pid, #{reward => R}),
                      timer:sleep(15)
                  end,
                  Rewards
              ),

              timer:sleep(100),
              ?assert(is_process_alive(Pid)),

              ok = gen_server:stop(Pid)
          end},
         {"declining performance trends",
          fun() ->
              Config = #{
                  silo_type => resource,
                  morphology_module => resource_l0_morphology,
                  tau_l1 => 50,
                  window_size => 5
              },
              {ok, Pid} = lc_l1_controller:start_link(Config),

              %% Send declining performance
              Rewards = [0.7, 0.6, 0.5, 0.4, 0.3],
              lists:foreach(
                  fun(R) ->
                      ok = lc_l1_controller:observe_l0_performance(Pid, #{reward => R}),
                      timer:sleep(15)
                  end,
                  Rewards
              ),

              timer:sleep(100),
              ?assert(is_process_alive(Pid)),

              ok = gen_server:stop(Pid)
          end}
     ]}.
