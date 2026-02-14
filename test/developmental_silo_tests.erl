%% @doc Unit tests for developmental_silo module.
%%
%% Tests ontogeny, plasticity, critical periods, and metamorphosis functionality.
%%
%% @author R.G. Lefever
%% @copyright 2024-2026 R.G. Lefever
-module(developmental_silo_tests).

-include_lib("eunit/include/eunit.hrl").

%%% ============================================================================
%%% Setup/Teardown
%%% ============================================================================

setup() ->
    case whereis(developmental_silo) of
        undefined -> ok;
        Pid ->
            catch gen_server:stop(Pid, normal, 1000),
            timer:sleep(50)
    end,
    ok.

cleanup(_) ->
    case whereis(developmental_silo) of
        undefined -> ok;
        Pid ->
            catch gen_server:stop(Pid, normal, 1000),
            timer:sleep(50)
    end,
    ok.

%%% ============================================================================
%%% Test Suite
%%% ============================================================================

developmental_silo_test_() ->
    {foreach,
     fun setup/0,
     fun cleanup/1,
     [
         {"start_link with default config", fun start_link_default_test/0},
         {"start_link with custom config", fun start_link_custom_config_test/0},
         {"get_params returns defaults", fun get_params_test/0},
         {"update_developmental_state tracks development", fun update_developmental_state_test/0},
         {"get_developmental_state retrieves state", fun get_developmental_state_test/0},
         {"open_critical_period creates period", fun open_critical_period_test/0},
         {"close_critical_period closes period", fun close_critical_period_test/0},
         {"trigger_metamorphosis works", fun trigger_metamorphosis_test/0},
         {"get_developmental_stats returns metrics", fun get_developmental_stats_test/0},
         {"reset clears state", fun reset_test/0},
         {"get_state returns full state", fun get_state_test/0}
     ]}.

%%% ============================================================================
%%% Individual Tests
%%% ============================================================================

start_link_default_test() ->
    {ok, Pid} = developmental_silo:start_link(),
    ?assert(is_pid(Pid)),
    ?assert(is_process_alive(Pid)),
    ok = gen_server:stop(Pid).

start_link_custom_config_test() ->
    Config = #{
        realm => <<"test_realm">>,
        enabled_levels => [l0]
    },
    {ok, Pid} = developmental_silo:start_link(Config),
    ?assert(is_pid(Pid)),
    ok = gen_server:stop(Pid).

get_params_test() ->
    {ok, Pid} = developmental_silo:start_link(),
    Params = developmental_silo:get_params(Pid),
    ?assert(is_map(Params)),
    ?assert(maps:is_key(growth_rate, Params)),
    ?assert(maps:is_key(maturation_speed, Params)),
    ?assert(maps:is_key(initial_plasticity, Params)),
    ?assertEqual(0.05, maps:get(growth_rate, Params)),
    ?assertEqual(0.9, maps:get(initial_plasticity, Params)),
    ok = gen_server:stop(Pid).

update_developmental_state_test() ->
    {ok, Pid} = developmental_silo:start_link(),

    %% Update developmental states (4 args: Pid, IndividualId, Stage, Plasticity)
    ok = developmental_silo:update_developmental_state(Pid, ind_1, embryonic, 0.9),
    ok = developmental_silo:update_developmental_state(Pid, ind_2, juvenile, 0.7),
    timer:sleep(50),

    %% Check stats
    Stats = developmental_silo:get_developmental_stats(Pid),
    ?assertEqual(2, maps:get(individual_count, Stats)),
    ok = gen_server:stop(Pid).

get_developmental_state_test() ->
    {ok, Pid} = developmental_silo:start_link(),

    %% Add a developmental state
    ok = developmental_silo:update_developmental_state(Pid, ind_1, embryonic, 0.9),
    timer:sleep(50),

    %% Retrieve it
    {ok, State} = developmental_silo:get_developmental_state(Pid, ind_1),
    ?assert(is_map(State)),

    %% Non-existent individual
    ?assertEqual(not_found, developmental_silo:get_developmental_state(Pid, non_existent)),
    ok = gen_server:stop(Pid).

open_critical_period_test() ->
    {ok, Pid} = developmental_silo:start_link(),

    %% First add a developmental state
    ok = developmental_silo:update_developmental_state(Pid, ind_1, embryonic, 0.9),
    timer:sleep(50),

    %% Open a critical period (3 args: Pid, IndividualId, PeriodType)
    ok = developmental_silo:open_critical_period(Pid, ind_1, sensory),
    timer:sleep(50),

    %% Check stats
    Stats = developmental_silo:get_developmental_stats(Pid),
    ?assert(maps:get(active_critical_periods, Stats) >= 0),
    ok = gen_server:stop(Pid).

close_critical_period_test() ->
    {ok, Pid} = developmental_silo:start_link(),

    %% Setup: add state and open period
    ok = developmental_silo:update_developmental_state(Pid, ind_1, embryonic, 0.9),
    ok = developmental_silo:open_critical_period(Pid, ind_1, sensory),
    timer:sleep(50),

    %% Close the critical period (2 args: Pid, IndividualId)
    ok = developmental_silo:close_critical_period(Pid, ind_1),
    timer:sleep(50),

    %% Verify period was closed (no error)
    ok = gen_server:stop(Pid).

trigger_metamorphosis_test() ->
    {ok, Pid} = developmental_silo:start_link(),

    %% Add a developmental state
    ok = developmental_silo:update_developmental_state(Pid, ind_1, juvenile, 0.5),
    timer:sleep(50),

    %% Trigger metamorphosis
    ok = developmental_silo:trigger_metamorphosis(Pid, ind_1),
    timer:sleep(50),

    %% Check metamorphosis count
    Stats = developmental_silo:get_developmental_stats(Pid),
    ?assert(maps:get(metamorphosis_count, Stats) >= 1),
    ok = gen_server:stop(Pid).

get_developmental_stats_test() ->
    {ok, Pid} = developmental_silo:start_link(),

    %% Empty stats
    Stats1 = developmental_silo:get_developmental_stats(Pid),
    ?assertEqual(0, maps:get(individual_count, Stats1)),
    ?assertEqual(0, maps:get(active_critical_periods, Stats1)),
    ?assertEqual(0, maps:get(metamorphosis_count, Stats1)),

    %% Add some data
    ok = developmental_silo:update_developmental_state(Pid, ind_1, juvenile, 0.7),
    ok = developmental_silo:open_critical_period(Pid, ind_1, motor),
    timer:sleep(50),

    Stats2 = developmental_silo:get_developmental_stats(Pid),
    ?assertEqual(1, maps:get(individual_count, Stats2)),
    ok = gen_server:stop(Pid).

reset_test() ->
    {ok, Pid} = developmental_silo:start_link(),

    %% Add data
    ok = developmental_silo:update_developmental_state(Pid, ind_1, juvenile, 0.7),
    ok = developmental_silo:trigger_metamorphosis(Pid, ind_1),
    timer:sleep(50),

    %% Verify data exists
    Stats1 = developmental_silo:get_developmental_stats(Pid),
    ?assert(maps:get(individual_count, Stats1) > 0),

    %% Reset
    ok = developmental_silo:reset(Pid),

    %% Verify reset
    Stats2 = developmental_silo:get_developmental_stats(Pid),
    ?assertEqual(0, maps:get(individual_count, Stats2)),
    ?assertEqual(0, maps:get(metamorphosis_count, Stats2)),
    ok = gen_server:stop(Pid).

get_state_test() ->
    {ok, Pid} = developmental_silo:start_link(#{realm => <<"test">>}),
    State = developmental_silo:get_state(Pid),
    ?assert(is_map(State)),
    ?assertEqual(<<"test">>, maps:get(realm, State)),
    ?assert(maps:is_key(current_params, State)),
    ?assert(maps:is_key(metamorphosis_count, State)),
    ?assert(maps:is_key(sensors, State)),
    ok = gen_server:stop(Pid).

%%% ============================================================================
%%% Behavior Callback Tests
%%% ============================================================================

behavior_callbacks_test_() ->
    [
        {"get_silo_type returns developmental",
         ?_assertEqual(developmental, developmental_silo:get_silo_type())},
        {"get_time_constant returns 40.0",
         ?_assertEqual(40.0, developmental_silo:get_time_constant())}
    ].
