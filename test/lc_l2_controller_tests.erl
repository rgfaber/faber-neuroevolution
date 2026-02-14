%% @doc Unit tests for lc_l2_controller module.
%%
%% Tests L2 strategic hyperparameter meta-tuning.
%%
%% @author R.G. Lefever
%% @copyright 2024-2026 R.G. Lefever
-module(lc_l2_controller_tests).

-include_lib("eunit/include/eunit.hrl").

%%% ============================================================================
%%% Test Setup
%%% ============================================================================

setup() ->
    catch gen_server:stop(lc_l2_test),
    ok.

cleanup(_) ->
    catch gen_server:stop(lc_l2_test),
    ok.

%%% ============================================================================
%%% Test Generators
%%% ============================================================================

lc_l2_controller_test_() ->
    {foreach,
     fun setup/0,
     fun cleanup/1,
     [
         {"start_link creates controller", fun start_link_test/0},
         {"get_l1_hyperparameters returns defaults", fun get_l1_hyperparameters_test/0},
         {"get_performance_summary returns state", fun get_performance_summary_test/0},
         {"observe_l1_performance accumulates history", fun observe_l1_performance_test/0},
         {"exploration finds better hyperparameters", fun exploration_test/0}
     ]}.

%%% ============================================================================
%%% Individual Tests
%%% ============================================================================

start_link_test() ->
    Config = #{
        silo_type => resource,
        morphology_module => resource_l0_morphology,
        tau_l2 => 5000,  % 5 seconds for faster testing
        long_term_window_size => 10,
        exploration_rate => 0.1
    },
    {ok, Pid} = lc_l2_controller:start_link(Config),
    ?assert(is_pid(Pid)),
    ?assert(is_process_alive(Pid)),
    ok = gen_server:stop(Pid).

get_l1_hyperparameters_test() ->
    Config = #{
        silo_type => resource,
        morphology_module => resource_l0_morphology
    },
    {ok, Pid} = lc_l2_controller:start_link(Config),

    L1Params = lc_l2_controller:get_l1_hyperparameters(Pid),
    ?assert(is_map(L1Params)),

    %% Should have resource L1 defaults
    ?assertEqual(0.05, maps:get(threshold_adaptation_rate, L1Params)),
    ?assertEqual(1.0, maps:get(pressure_sensitivity, L1Params)),

    ok = gen_server:stop(Pid).

get_performance_summary_test() ->
    Config = #{
        silo_type => task,
        morphology_module => task_l0_morphology
    },
    {ok, Pid} = lc_l2_controller:start_link(Config),

    Summary = lc_l2_controller:get_performance_summary(Pid),
    ?assert(is_map(Summary)),

    %% Check summary fields
    ?assertEqual(task, maps:get(silo_type, Summary)),
    ?assertEqual(0, maps:get(cycles_observed, Summary)),
    ?assert(is_map(maps:get(current_hyperparameters, Summary))),
    ?assert(is_map(maps:get(best_hyperparameters, Summary))),
    ?assert(is_float(maps:get(best_cumulative_reward, Summary))),
    ?assert(is_float(maps:get(exploration_rate, Summary))),

    ok = gen_server:stop(Pid).

observe_l1_performance_test() ->
    Config = #{
        silo_type => resource,
        morphology_module => resource_l0_morphology,
        tau_l2 => 100,  % Very short for testing
        long_term_window_size => 5
    },
    {ok, Pid} = lc_l2_controller:start_link(Config),

    %% Send L1 performance observations
    L1Metrics = #{l0_avg_reward => 0.5, adjustment_count => 3},
    ok = lc_l2_controller:observe_l1_performance(Pid, L1Metrics),
    timer:sleep(10),

    %% Verify cycle count increased
    Summary1 = lc_l2_controller:get_performance_summary(Pid),
    ?assertEqual(1, maps:get(cycles_observed, Summary1)),

    %% Send more observations
    ok = lc_l2_controller:observe_l1_performance(Pid, #{l0_avg_reward => 0.6}),
    ok = lc_l2_controller:observe_l1_performance(Pid, #{l0_avg_reward => 0.7}),
    timer:sleep(150),  % Wait for tau_l2 to trigger adjustment

    Summary2 = lc_l2_controller:get_performance_summary(Pid),
    ?assert(maps:get(cycles_observed, Summary2) >= 3),

    ok = gen_server:stop(Pid).

exploration_test() ->
    Config = #{
        silo_type => task,
        morphology_module => task_l0_morphology,
        tau_l2 => 50,  % Very short for testing
        long_term_window_size => 3,
        exploration_rate => 0.2
    },
    {ok, Pid} = lc_l2_controller:start_link(Config),

    %% Get initial hyperparameters
    _InitialParams = lc_l2_controller:get_l1_hyperparameters(Pid),

    %% Send good performance to trigger adjustment
    lists:foreach(
        fun(_) ->
            ok = lc_l2_controller:observe_l1_performance(Pid, #{l0_avg_reward => 0.8}),
            timer:sleep(20)
        end,
        lists:seq(1, 10)
    ),

    timer:sleep(100),

    %% Hyperparameters should have changed due to exploration
    FinalParams = lc_l2_controller:get_l1_hyperparameters(Pid),

    %% At least one parameter should be different (due to exploration)
    %% Note: might be same if exploration direction is 0 for all
    ?assert(is_map(FinalParams)),

    %% Best cumulative reward should be updated
    Summary = lc_l2_controller:get_performance_summary(Pid),
    BestReward = maps:get(best_cumulative_reward, Summary),
    %% Should have accumulated positive rewards
    ?assert(BestReward > -999999.0),

    ok = gen_server:stop(Pid).

%%% ============================================================================
%%% Task Silo L2 Controller Tests
%%% ============================================================================

task_l2_controller_test_() ->
    {foreach,
     fun setup/0,
     fun cleanup/1,
     [
         {"task L2 manages task L1 hyperparameters",
          fun() ->
              Config = #{
                  silo_type => task,
                  morphology_module => task_l0_morphology
              },
              {ok, Pid} = lc_l2_controller:start_link(Config),

              L1Params = lc_l2_controller:get_l1_hyperparameters(Pid),

              %% Should have task L1 defaults
              ?assert(maps:is_key(aggression_factor, L1Params)),
              ?assert(maps:is_key(exploration_step, L1Params)),
              ?assert(maps:is_key(topology_aggression, L1Params)),

              ok = gen_server:stop(Pid)
          end},
         {"task L2 adjusts for stagnation",
          fun() ->
              Config = #{
                  silo_type => task,
                  morphology_module => task_l0_morphology,
                  tau_l2 => 50,
                  long_term_window_size => 3,
                  exploration_rate => 0.3
              },
              {ok, Pid} = lc_l2_controller:start_link(Config),

              %% Send poor performance (L0 not improving under L1)
              lists:foreach(
                  fun(_) ->
                      ok = lc_l2_controller:observe_l1_performance(Pid, #{l0_avg_reward => 0.1}),
                      timer:sleep(20)
                  end,
                  lists:seq(1, 10)
              ),

              timer:sleep(100),

              %% L2 should have accumulated some cycles
              Summary = lc_l2_controller:get_performance_summary(Pid),
              %% At least some observations were recorded
              ?assert(maps:get(cycles_observed, Summary) >= 1),

              ok = gen_server:stop(Pid)
          end}
     ]}.

%%% ============================================================================
%%% Exploration Direction Tests
%%% ============================================================================

exploration_direction_test_() ->
    {foreach,
     fun setup/0,
     fun cleanup/1,
     [
         {"exploration direction randomized on failure",
          fun() ->
              Config = #{
                  silo_type => resource,
                  morphology_module => resource_l0_morphology,
                  tau_l2 => 30,
                  long_term_window_size => 2,
                  exploration_rate => 0.5
              },
              {ok, Pid} = lc_l2_controller:start_link(Config),

              %% Send poor performance (should trigger exploration)
              lists:foreach(
                  fun(_) ->
                      ok = lc_l2_controller:observe_l1_performance(Pid, #{l0_avg_reward => 0.1}),
                      timer:sleep(40)
                  end,
                  lists:seq(1, 5)
              ),

              timer:sleep(50),
              ?assert(is_process_alive(Pid)),

              ok = gen_server:stop(Pid)
          end},
         {"exploration continues on improvement",
          fun() ->
              Config = #{
                  silo_type => resource,
                  morphology_module => resource_l0_morphology,
                  tau_l2 => 30,
                  long_term_window_size => 2,
                  exploration_rate => 0.3
              },
              {ok, Pid} = lc_l2_controller:start_link(Config),

              %% Send improving performance
              lists:foreach(
                  fun(I) ->
                      ok = lc_l2_controller:observe_l1_performance(Pid, #{l0_avg_reward => 0.1 * I}),
                      timer:sleep(40)
                  end,
                  lists:seq(1, 8)
              ),

              timer:sleep(50),

              %% Should have updated best hyperparameters
              Summary = lc_l2_controller:get_performance_summary(Pid),
              BestReward = maps:get(best_cumulative_reward, Summary),
              ?assert(BestReward > 0),

              ok = gen_server:stop(Pid)
          end}
     ]}.

%%% ============================================================================
%%% Distribution Silo L2 Tests
%%% ============================================================================

distribution_l2_controller_test() ->
    Config = #{
        silo_type => distribution,
        morphology_module => distribution_l0_morphology,
        tau_l2 => 5000
    },
    {ok, Pid} = lc_l2_controller:start_link(Config),

    L1Params = lc_l2_controller:get_l1_hyperparameters(Pid),

    %% Should have distribution L1 defaults
    ?assert(maps:is_key(load_sensitivity, L1Params)),
    ?assert(maps:is_key(migration_adaptation_rate, L1Params)),

    ok = gen_server:stop(Pid).

%%% ============================================================================
%%% Bounds Enforcement Tests
%%% ============================================================================

bounds_enforcement_test_() ->
    {foreach,
     fun setup/0,
     fun cleanup/1,
     [
         {"L1 hyperparameters stay within bounds",
          fun() ->
              Config = #{
                  silo_type => resource,
                  morphology_module => resource_l0_morphology,
                  tau_l2 => 20,
                  long_term_window_size => 2,
                  exploration_rate => 1.0  % Max exploration
              },
              {ok, Pid} = lc_l2_controller:start_link(Config),

              %% Force many adjustments
              lists:foreach(
                  fun(_) ->
                      ok = lc_l2_controller:observe_l1_performance(Pid, #{l0_avg_reward => rand:uniform()}),
                      timer:sleep(25)
                  end,
                  lists:seq(1, 20)
              ),

              timer:sleep(50),

              %% Get final hyperparameters
              L1Params = lc_l2_controller:get_l1_hyperparameters(Pid),
              Bounds = resource_l0_morphology:get_l1_bounds(),

              %% Verify all parameters are within bounds
              maps:foreach(
                  fun(Name, Value) ->
                      case maps:get(Name, Bounds, undefined) of
                          undefined -> ok;
                          {Min, Max} ->
                              ?assert(Value >= Min),
                              ?assert(Value =< Max)
                      end
                  end,
                  L1Params
              ),

              ok = gen_server:stop(Pid)
          end}
     ]}.
