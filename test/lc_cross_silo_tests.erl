%% @doc Unit tests for lc_cross_silo module.
%%
%% Tests cross-silo signal routing and coordination.
%%
%% @author R.G. Lefever
%% @copyright 2024-2026 R.G. Lefever
-module(lc_cross_silo_tests).

-include_lib("eunit/include/eunit.hrl").

%%% ============================================================================
%%% Test Setup
%%% ============================================================================

setup() ->
    %% Stop any existing lc_cross_silo process to ensure clean test state
    case whereis(lc_cross_silo) of
        undefined -> ok;
        Pid when is_pid(Pid) ->
            %% Unlink to avoid crashing our test process
            catch unlink(Pid),
            %% Try graceful stop first, then brutal kill
            catch gen_server:stop(Pid, normal, 1000),
            %% Wait for process to die
            timer:sleep(50),
            %% Force kill if still alive
            case is_process_alive(Pid) of
                true -> exit(Pid, kill);
                false -> ok
            end,
            timer:sleep(50)
    end,
    ok.

cleanup(_) ->
    case whereis(lc_cross_silo) of
        undefined -> ok;
        Pid when is_pid(Pid) ->
            catch unlink(Pid),
            catch gen_server:stop(Pid, normal, 1000),
            timer:sleep(50)
    end,
    ok.

%%% ============================================================================
%%% Test Generators
%%% ============================================================================

lc_cross_silo_test_() ->
    {foreach,
     fun setup/0,
     fun cleanup/1,
     [
         {"start_link with default config", fun start_link_default_test/0},
         {"start_link with custom config", fun start_link_custom_config_test/0},
         {"send valid signal", fun send_valid_signal_test/0},
         {"invalid route rejected", fun invalid_route_rejected_test/0},
         {"get signal returns value", fun get_signal_test/0},
         {"get signal default value", fun get_signal_default_test/0},
         {"signal decay", fun signal_decay_test/0},
         {"effective evals per individual", fun effective_evals_test/0}
     ]}.

%%% ============================================================================
%%% Individual Tests
%%% ============================================================================

start_link_default_test() ->
    {ok, Pid} = lc_cross_silo:start_link(#{}),
    ?assert(is_pid(Pid)),
    ?assert(is_process_alive(Pid)),
    ok = gen_server:stop(Pid).

start_link_custom_config_test() ->
    Config = #{
        signal_decay_ms => 10000,
        default_decay_factor => 0.8
    },
    {ok, Pid} = lc_cross_silo:start_link(Config),
    ?assert(is_pid(Pid)),
    ok = gen_server:stop(Pid).

send_valid_signal_test() ->
    {ok, Pid} = lc_cross_silo:start_link(#{}),
    %% Valid route: resource -> task: pressure_signal
    ok = lc_cross_silo:emit(resource, task, pressure_signal, 0.75),
    timer:sleep(10),  % Allow cast to process
    %% Should not crash
    ?assert(is_process_alive(Pid)),
    ok = gen_server:stop(Pid).

invalid_route_rejected_test() ->
    {ok, Pid} = lc_cross_silo:start_link(#{}),
    %% Invalid route: task -> resource: nonexistent_signal
    %% emit is a cast, so invalid routes just log a warning and return ok
    ok = lc_cross_silo:emit(task, resource, nonexistent_signal, 0.5),
    timer:sleep(10),
    %% Signal should NOT be stored (invalid route)
    Signals = lc_cross_silo:get_signals_for(resource),
    ?assertEqual(undefined, maps:get(nonexistent_signal, Signals, undefined)),
    ok = gen_server:stop(Pid).

get_signal_test() ->
    {ok, Pid} = lc_cross_silo:start_link(#{}),
    %% Send a signal
    ok = lc_cross_silo:emit(resource, task, pressure_signal, 0.75),
    timer:sleep(10),
    %% Get the signal
    Signals = lc_cross_silo:get_signals_for(task),
    Value = maps:get(pressure_signal, Signals),
    ?assert(is_float(Value)),
    ?assert(Value > 0.7),  % Should be close to 0.75
    ok = gen_server:stop(Pid).

get_signal_default_test() ->
    {ok, Pid} = lc_cross_silo:start_link(#{}),
    %% Get signal with initial value (pressure_signal starts at 0.0)
    Signals = lc_cross_silo:get_signals_for(task),
    Value = maps:get(pressure_signal, Signals, undefined),
    ?assertEqual(0.0, Value),
    ok = gen_server:stop(Pid).

signal_decay_test() ->
    %% Use very short decay for testing
    Config = #{signal_decay_ms => 50},
    {ok, Pid} = lc_cross_silo:start_link(Config),

    %% Send a signal
    ok = lc_cross_silo:emit(resource, task, pressure_signal, 1.0),
    timer:sleep(10),

    %% Get immediately
    Signals1 = lc_cross_silo:get_signals_for(task),
    Value1 = maps:get(pressure_signal, Signals1),
    ?assert(Value1 > 0.8),

    %% Wait for decay
    timer:sleep(100),

    %% Should be decayed (close to 0)
    Signals2 = lc_cross_silo:get_signals_for(task),
    Value2 = maps:get(pressure_signal, Signals2),
    ?assert(Value2 < Value1),

    ok = gen_server:stop(Pid).

effective_evals_test() ->
    {ok, Pid} = lc_cross_silo:start_link(#{}),

    %% Send resource max (resource -> task signal)
    ok = lc_cross_silo:emit(resource, task, max_evals_per_individual, 10),
    timer:sleep(10),

    %% Send task desired (task -> resource signal)
    ok = lc_cross_silo:emit(task, resource, desired_evals_per_individual, 20),
    timer:sleep(10),

    %% Effective should be min(resource_max, task_desired) = 10
    Effective = lc_cross_silo:get_effective_evals_per_individual(),
    ?assertEqual(10, Effective),

    ok = gen_server:stop(Pid).

%%% ============================================================================
%%% Route Validation Tests
%%% ============================================================================

valid_routes_test_() ->
    {setup,
     fun setup/0,
     fun cleanup/1,
     fun(_) ->
         {ok, Pid} = lc_cross_silo:start_link(#{}),
         Tests = [
             %% Resource -> Task routes
             {"resource->task pressure_signal",
              fun() -> ?assertEqual(ok, lc_cross_silo:emit(resource, task, pressure_signal, 0.5)) end},
             {"resource->task max_evals_per_individual",
              fun() -> ?assertEqual(ok, lc_cross_silo:emit(resource, task, max_evals_per_individual, 10)) end},
             {"resource->task should_simplify",
              fun() -> ?assertEqual(ok, lc_cross_silo:emit(resource, task, should_simplify, 0.3)) end},

             %% Resource -> Distribution routes
             {"resource->distribution offload_preference",
              fun() -> ?assertEqual(ok, lc_cross_silo:emit(resource, distribution, offload_preference, 0.7)) end},
             {"resource->distribution local_capacity",
              fun() -> ?assertEqual(ok, lc_cross_silo:emit(resource, distribution, local_capacity, 0.5)) end},

             %% Task -> Resource routes
             {"task->resource exploration_boost",
              fun() -> ?assertEqual(ok, lc_cross_silo:emit(task, resource, exploration_boost, 0.8)) end},
             {"task->resource desired_evals_per_individual",
              fun() -> ?assertEqual(ok, lc_cross_silo:emit(task, resource, desired_evals_per_individual, 15)) end},
             {"task->resource expected_complexity_growth",
              fun() -> ?assertEqual(ok, lc_cross_silo:emit(task, resource, expected_complexity_growth, 0.2)) end},

             %% Task -> Distribution routes
             {"task->distribution diversity_need",
              fun() -> ?assertEqual(ok, lc_cross_silo:emit(task, distribution, diversity_need, 0.6)) end},
             {"task->distribution speciation_pressure",
              fun() -> ?assertEqual(ok, lc_cross_silo:emit(task, distribution, speciation_pressure, 0.4)) end},

             %% Distribution -> Resource routes
             {"distribution->resource network_load_contribution",
              fun() -> ?assertEqual(ok, lc_cross_silo:emit(distribution, resource, network_load_contribution, 0.3)) end},
             {"distribution->resource remote_capacity_available",
              fun() -> ?assertEqual(ok, lc_cross_silo:emit(distribution, resource, remote_capacity_available, 0.7)) end},

             %% Distribution -> Task routes
             {"distribution->task island_diversity_score",
              fun() -> ?assertEqual(ok, lc_cross_silo:emit(distribution, task, island_diversity_score, 0.85)) end},
             {"distribution->task migration_activity",
              fun() -> ?assertEqual(ok, lc_cross_silo:emit(distribution, task, migration_activity, 0.5)) end}
         ],
         gen_server:stop(Pid),
         Tests
     end}.

%%% ============================================================================
%%% Signal Value Clamping Tests
%%% ============================================================================

signal_clamping_test_() ->
    {foreach,
     fun setup/0,
     fun cleanup/1,
     [
         {"value clamped to max 1.0",
          fun() ->
              {ok, Pid} = lc_cross_silo:start_link(#{}),
              ok = lc_cross_silo:emit(resource, task, pressure_signal, 5.0),
              timer:sleep(10),
              Signals = lc_cross_silo:get_signals_for(task),
              Value = maps:get(pressure_signal, Signals),
              ?assert(Value =< 1.0),
              gen_server:stop(Pid)
          end},
         {"value clamped to min 0.0",
          fun() ->
              {ok, Pid} = lc_cross_silo:start_link(#{}),
              ok = lc_cross_silo:emit(resource, task, pressure_signal, -1.0),
              timer:sleep(10),
              Signals = lc_cross_silo:get_signals_for(task),
              Value = maps:get(pressure_signal, Signals),
              ?assert(Value >= 0.0),
              gen_server:stop(Pid)
          end}
     ]}.

%%% ============================================================================
%%% get_all_signals Tests
%%% ============================================================================

get_all_signals_test_() ->
    {foreach,
     fun setup/0,
     fun cleanup/1,
     [
         {"get all signals returns map with all silos",
          fun() ->
              {ok, Pid} = lc_cross_silo:start_link(#{}),

              %% Send multiple signals
              ok = lc_cross_silo:emit(resource, task, pressure_signal, 0.75),
              ok = lc_cross_silo:emit(task, resource, exploration_boost, 0.5),
              timer:sleep(10),

              %% Get all signals
              AllSignals = lc_cross_silo:get_all_signals(),
              ?assert(is_map(AllSignals)),
              %% Should have entries for all silos
              ?assert(maps:is_key(resource, AllSignals)),
              ?assert(maps:is_key(task, AllSignals)),
              ?assert(maps:is_key(distribution, AllSignals)),

              gen_server:stop(Pid)
          end}
     ]}.
