%% @doc Unit tests for communication_silo module.
%%
%% Tests signaling, vocabulary, and coordination functionality.
%%
%% @author R.G. Lefever
%% @copyright 2024-2026 R.G. Lefever
-module(communication_silo_tests).

-include_lib("eunit/include/eunit.hrl").

%%% ============================================================================
%%% Setup/Teardown
%%% ============================================================================

setup() ->
    case whereis(communication_silo) of
        undefined -> ok;
        Pid ->
            catch gen_server:stop(Pid, normal, 1000),
            timer:sleep(50)
    end,
    ok.

cleanup(_) ->
    case whereis(communication_silo) of
        undefined -> ok;
        Pid ->
            catch gen_server:stop(Pid, normal, 1000),
            timer:sleep(50)
    end,
    ok.

%%% ============================================================================
%%% Test Suite
%%% ============================================================================

communication_silo_test_() ->
    {foreach,
     fun setup/0,
     fun cleanup/1,
     [
         {"start_link with default config", fun start_link_default_test/0},
         {"start_link with custom config", fun start_link_custom_config_test/0},
         {"get_params returns defaults", fun get_params_test/0},
         {"register_signal adds to vocabulary", fun register_signal_test/0},
         {"get_vocabulary returns signals", fun get_vocabulary_test/0},
         {"send_message tracks messages", fun send_message_test/0},
         {"record_coordination tracks success", fun record_coordination_test/0},
         {"get_communication_stats returns metrics", fun get_communication_stats_test/0},
         {"reset clears state", fun reset_test/0},
         {"get_state returns full state", fun get_state_test/0}
     ]}.

%%% ============================================================================
%%% Individual Tests
%%% ============================================================================

start_link_default_test() ->
    {ok, Pid} = communication_silo:start_link(),
    ?assert(is_pid(Pid)),
    ?assert(is_process_alive(Pid)),
    ok = gen_server:stop(Pid).

start_link_custom_config_test() ->
    Config = #{
        realm => <<"test_realm">>,
        enabled_levels => [l0]
    },
    {ok, Pid} = communication_silo:start_link(Config),
    ?assert(is_pid(Pid)),
    ok = gen_server:stop(Pid).

get_params_test() ->
    {ok, Pid} = communication_silo:start_link(),
    Params = communication_silo:get_params(Pid),
    ?assert(is_map(Params)),
    ?assert(maps:is_key(vocabulary_growth_rate, Params)),
    ?assert(maps:is_key(communication_cost, Params)),
    ?assert(maps:is_key(coordination_reward, Params)),
    ?assertEqual(0.02, maps:get(vocabulary_growth_rate, Params)),
    ?assertEqual(0.2, maps:get(coordination_reward, Params)),
    ok = gen_server:stop(Pid).

register_signal_test() ->
    {ok, Pid} = communication_silo:start_link(),

    %% Register signals
    ok = communication_silo:register_signal(Pid, sig_1, #{meaning => danger}),
    ok = communication_silo:register_signal(Pid, sig_2, #{meaning => food, dialect => alpha}),
    timer:sleep(50),

    %% Check vocabulary
    Vocab = communication_silo:get_vocabulary(Pid),
    ?assertEqual(2, length(Vocab)),
    ok = gen_server:stop(Pid).

get_vocabulary_test() ->
    {ok, Pid} = communication_silo:start_link(),

    %% Empty vocabulary
    Vocab1 = communication_silo:get_vocabulary(Pid),
    ?assertEqual([], Vocab1),

    %% Add some signals
    ok = communication_silo:register_signal(Pid, sig_1, #{meaning => alert}),
    timer:sleep(50),

    Vocab2 = communication_silo:get_vocabulary(Pid),
    ?assertEqual(1, length(Vocab2)),
    ok = gen_server:stop(Pid).

send_message_test() ->
    {ok, Pid} = communication_silo:start_link(),

    %% Send messages
    ok = communication_silo:send_message(Pid, sender_1, receiver_1, #{content_length => 3}),
    ok = communication_silo:send_message(Pid, sender_2, receiver_2, #{honest => false}),
    ok = communication_silo:send_message(Pid, sender_1, receiver_3, #{}),
    timer:sleep(50),

    %% Check stats
    Stats = communication_silo:get_communication_stats(Pid),
    ?assertEqual(3, maps:get(total_messages_sent, Stats)),
    ok = gen_server:stop(Pid).

record_coordination_test() ->
    {ok, Pid} = communication_silo:start_link(),

    %% Record coordinations
    ok = communication_silo:record_coordination(Pid, [a, b, c], true),
    ok = communication_silo:record_coordination(Pid, [d, e], false),
    ok = communication_silo:record_coordination(Pid, [f, g, h, i], true),
    timer:sleep(50),

    %% Check stats
    Stats = communication_silo:get_communication_stats(Pid),
    ?assertEqual(3, maps:get(coordination_count, Stats)),
    ?assertEqual(2, maps:get(successful_coordinations, Stats)),
    ok = gen_server:stop(Pid).

get_communication_stats_test() ->
    {ok, Pid} = communication_silo:start_link(),

    %% Empty stats
    Stats1 = communication_silo:get_communication_stats(Pid),
    ?assertEqual(0, maps:get(vocabulary_size, Stats1)),
    ?assertEqual(0, maps:get(total_messages_sent, Stats1)),
    ?assertEqual(0, maps:get(coordination_count, Stats1)),

    %% Add some data
    ok = communication_silo:register_signal(Pid, sig_1, #{}),
    ok = communication_silo:send_message(Pid, a, b, #{}),
    ok = communication_silo:record_coordination(Pid, [a, b], true),
    timer:sleep(50),

    Stats2 = communication_silo:get_communication_stats(Pid),
    ?assertEqual(1, maps:get(vocabulary_size, Stats2)),
    ?assertEqual(1, maps:get(total_messages_sent, Stats2)),
    ?assertEqual(1, maps:get(coordination_count, Stats2)),
    ok = gen_server:stop(Pid).

reset_test() ->
    {ok, Pid} = communication_silo:start_link(),

    %% Add data
    ok = communication_silo:register_signal(Pid, sig_1, #{}),
    ok = communication_silo:send_message(Pid, a, b, #{}),
    ok = communication_silo:record_coordination(Pid, [a, b], true),
    timer:sleep(50),

    %% Verify data exists
    Stats1 = communication_silo:get_communication_stats(Pid),
    ?assert(maps:get(vocabulary_size, Stats1) > 0),

    %% Reset
    ok = communication_silo:reset(Pid),

    %% Verify reset
    Stats2 = communication_silo:get_communication_stats(Pid),
    ?assertEqual(0, maps:get(vocabulary_size, Stats2)),
    ?assertEqual(0, maps:get(total_messages_sent, Stats2)),
    ?assertEqual(0, maps:get(coordination_count, Stats2)),
    ok = gen_server:stop(Pid).

get_state_test() ->
    {ok, Pid} = communication_silo:start_link(#{realm => <<"test">>}),
    State = communication_silo:get_state(Pid),
    ?assert(is_map(State)),
    ?assertEqual(<<"test">>, maps:get(realm, State)),
    ?assert(maps:is_key(current_params, State)),
    ?assert(maps:is_key(message_count, State)),
    ?assert(maps:is_key(sensors, State)),
    ok = gen_server:stop(Pid).

%%% ============================================================================
%%% Behavior Callback Tests
%%% ============================================================================

behavior_callbacks_test_() ->
    [
        {"get_silo_type returns communication",
         ?_assertEqual(communication, communication_silo:get_silo_type())},
        {"get_time_constant returns 55.0",
         ?_assertEqual(55.0, communication_silo:get_time_constant())}
    ].
