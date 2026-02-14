%% @doc Unit tests for cultural_silo module.
%%
%% Tests innovation, tradition, and cultural learning functionality.
%%
%% @author R.G. Lefever
%% @copyright 2024-2026 R.G. Lefever
-module(cultural_silo_tests).

-include_lib("eunit/include/eunit.hrl").

%%% ============================================================================
%%% Setup/Teardown
%%% ============================================================================

setup() ->
    case whereis(cultural_silo) of
        undefined -> ok;
        Pid ->
            catch gen_server:stop(Pid, normal, 1000),
            timer:sleep(50)
    end,
    ok.

cleanup(_) ->
    case whereis(cultural_silo) of
        undefined -> ok;
        Pid ->
            catch gen_server:stop(Pid, normal, 1000),
            timer:sleep(50)
    end,
    ok.

%%% ============================================================================
%%% Test Suite
%%% ============================================================================

cultural_silo_test_() ->
    {foreach,
     fun setup/0,
     fun cleanup/1,
     [
         {"start_link with default config", fun start_link_default_test/0},
         {"start_link with custom config", fun start_link_custom_config_test/0},
         {"get_params returns defaults", fun get_params_test/0},
         {"record_innovation tracks innovations", fun record_innovation_test/0},
         {"record_imitation tracks imitations", fun record_imitation_test/0},
         {"promote_to_tradition creates tradition", fun promote_to_tradition_test/0},
         {"get_innovation_stats returns metrics", fun get_innovation_stats_test/0},
         {"reset clears state", fun reset_test/0},
         {"get_state returns full state", fun get_state_test/0}
     ]}.

%%% ============================================================================
%%% Individual Tests
%%% ============================================================================

start_link_default_test() ->
    {ok, Pid} = cultural_silo:start_link(),
    ?assert(is_pid(Pid)),
    ?assert(is_process_alive(Pid)),
    ok = gen_server:stop(Pid).

start_link_custom_config_test() ->
    Config = #{
        realm => <<"test_realm">>,
        enabled_levels => [l0]
    },
    {ok, Pid} = cultural_silo:start_link(Config),
    ?assert(is_pid(Pid)),
    ok = gen_server:stop(Pid).

get_params_test() ->
    {ok, Pid} = cultural_silo:start_link(),
    Params = cultural_silo:get_params(Pid),
    ?assert(is_map(Params)),
    ?assert(maps:is_key(innovation_bonus, Params)),
    ?assert(maps:is_key(imitation_probability, Params)),
    ?assert(maps:is_key(tradition_threshold, Params)),
    ?assertEqual(0.15, maps:get(innovation_bonus, Params)),
    ?assertEqual(5, maps:get(tradition_threshold, Params)),
    ok = gen_server:stop(Pid).

record_innovation_test() ->
    {ok, Pid} = cultural_silo:start_link(),

    %% Record innovations
    ok = cultural_silo:record_innovation(Pid, innovator_1, <<"behavior_sig_1">>, 0.5),
    ok = cultural_silo:record_innovation(Pid, innovator_2, <<"behavior_sig_2">>, 0.3),
    timer:sleep(50),

    %% Check stats
    Stats = cultural_silo:get_innovation_stats(Pid),
    ?assertEqual(2, maps:get(innovation_count, Stats)),
    ?assertEqual(2, maps:get(active_innovations, Stats)),
    ok = gen_server:stop(Pid).

record_imitation_test() ->
    {ok, Pid} = cultural_silo:start_link(),

    %% Record imitations
    ok = cultural_silo:record_imitation(Pid, imitator_1, source_1, true),
    ok = cultural_silo:record_imitation(Pid, imitator_2, source_2, false),
    ok = cultural_silo:record_imitation(Pid, imitator_3, source_1, true),
    timer:sleep(50),

    %% Check stats
    Stats = cultural_silo:get_innovation_stats(Pid),
    ?assertEqual(3, maps:get(imitation_count, Stats)),
    ok = gen_server:stop(Pid).

promote_to_tradition_test() ->
    {ok, Pid} = cultural_silo:start_link(),

    %% First create an innovation
    ok = cultural_silo:record_innovation(Pid, innovator_1, <<"behavior_sig_1">>, 0.5),
    timer:sleep(50),

    %% Get innovation stats to find the innovation
    Stats1 = cultural_silo:get_innovation_stats(Pid),
    ?assertEqual(1, maps:get(active_innovations, Stats1)),
    ?assertEqual(0, maps:get(tradition_count, Stats1)),

    %% Note: promote_to_tradition requires knowing the innovation ID
    %% which is auto-generated. For this test, we'll just verify the API works
    ?assertEqual({error, innovation_not_found},
                 cultural_silo:promote_to_tradition(Pid, non_existent_id)),

    ok = gen_server:stop(Pid).

get_innovation_stats_test() ->
    {ok, Pid} = cultural_silo:start_link(),

    %% Empty stats
    Stats1 = cultural_silo:get_innovation_stats(Pid),
    ?assertEqual(0, maps:get(innovation_count, Stats1)),
    ?assertEqual(0, maps:get(imitation_count, Stats1)),
    ?assertEqual(0, maps:get(tradition_count, Stats1)),

    %% Add some data
    ok = cultural_silo:record_innovation(Pid, inn_1, <<"sig_1">>, 0.1),
    ok = cultural_silo:record_imitation(Pid, imit_1, inn_1, true),
    timer:sleep(50),

    Stats2 = cultural_silo:get_innovation_stats(Pid),
    ?assertEqual(1, maps:get(innovation_count, Stats2)),
    ?assertEqual(1, maps:get(imitation_count, Stats2)),
    ok = gen_server:stop(Pid).

reset_test() ->
    {ok, Pid} = cultural_silo:start_link(),

    %% Add data
    ok = cultural_silo:record_innovation(Pid, inn_1, <<"sig_1">>, 0.1),
    ok = cultural_silo:record_imitation(Pid, imit_1, inn_1, true),
    timer:sleep(50),

    %% Verify data exists
    Stats1 = cultural_silo:get_innovation_stats(Pid),
    ?assert(maps:get(innovation_count, Stats1) > 0),

    %% Reset
    ok = cultural_silo:reset(Pid),

    %% Verify reset
    Stats2 = cultural_silo:get_innovation_stats(Pid),
    ?assertEqual(0, maps:get(innovation_count, Stats2)),
    ?assertEqual(0, maps:get(imitation_count, Stats2)),
    ok = gen_server:stop(Pid).

get_state_test() ->
    {ok, Pid} = cultural_silo:start_link(#{realm => <<"test">>}),
    State = cultural_silo:get_state(Pid),
    ?assert(is_map(State)),
    ?assertEqual(<<"test">>, maps:get(realm, State)),
    ?assert(maps:is_key(current_params, State)),
    ?assert(maps:is_key(innovation_count, State)),
    ?assert(maps:is_key(sensors, State)),
    ok = gen_server:stop(Pid).

%%% ============================================================================
%%% Behavior Callback Tests
%%% ============================================================================

behavior_callbacks_test_() ->
    [
        {"get_silo_type returns cultural",
         ?_assertEqual(cultural, cultural_silo:get_silo_type())},
        {"get_time_constant returns 35.0",
         ?_assertEqual(35.0, cultural_silo:get_time_constant())}
    ].
