%% @doc Unit tests for competitive_silo module.
%%
%% Tests opponent archives, Elo ratings, and matchmaking functionality.
%%
%% @author R.G. Lefever
%% @copyright 2024-2026 R.G. Lefever
-module(competitive_silo_tests).

-include_lib("eunit/include/eunit.hrl").

%%% ============================================================================
%%% Setup/Teardown
%%% ============================================================================

setup() ->
    case whereis(competitive_silo) of
        undefined -> ok;
        Pid ->
            catch gen_server:stop(Pid, normal, 1000),
            timer:sleep(50)
    end,
    ok.

cleanup(_) ->
    case whereis(competitive_silo) of
        undefined -> ok;
        Pid ->
            catch gen_server:stop(Pid, normal, 1000),
            timer:sleep(50)
    end,
    ok.

%%% ============================================================================
%%% Test Suite
%%% ============================================================================

competitive_silo_test_() ->
    {foreach,
     fun setup/0,
     fun cleanup/1,
     [
         {"start_link with default config", fun start_link_default_test/0},
         {"start_link with custom config", fun start_link_custom_config_test/0},
         {"get_params returns defaults", fun get_params_test/0},
         {"add_to_archive stores opponent", fun add_to_archive_test/0},
         {"get_elo returns rating", fun get_elo_test/0},
         {"update_elo adjusts ratings", fun update_elo_test/0},
         {"record_match tracks matches", fun record_match_test/0},
         {"select_opponent finds opponent", fun select_opponent_test/0},
         {"get_archive_stats returns metrics", fun get_archive_stats_test/0},
         {"reset clears state", fun reset_test/0},
         {"get_state returns full state", fun get_state_test/0}
     ]}.

%%% ============================================================================
%%% Individual Tests
%%% ============================================================================

start_link_default_test() ->
    {ok, Pid} = competitive_silo:start_link(),
    ?assert(is_pid(Pid)),
    ?assert(is_process_alive(Pid)),
    ok = gen_server:stop(Pid).

start_link_custom_config_test() ->
    Config = #{
        realm => <<"test_realm">>,
        enabled_levels => [l0]
    },
    {ok, Pid} = competitive_silo:start_link(Config),
    ?assert(is_pid(Pid)),
    ok = gen_server:stop(Pid).

get_params_test() ->
    {ok, Pid} = competitive_silo:start_link(),
    Params = competitive_silo:get_params(Pid),
    ?assert(is_map(Params)),
    ?assert(maps:is_key(archive_addition_threshold, Params)),
    ?assert(maps:is_key(archive_max_size, Params)),
    ?assert(maps:is_key(matchmaking_elo_range, Params)),
    ?assert(maps:is_key(self_play_ratio, Params)),
    ?assertEqual(100, maps:get(archive_max_size, Params)),
    ?assertEqual(200, maps:get(matchmaking_elo_range, Params)),
    ok = gen_server:stop(Pid).

add_to_archive_test() ->
    {ok, Pid} = competitive_silo:start_link(),
    NetworkBinary = <<"test_network_data">>,

    %% Add opponent to archive
    ok = competitive_silo:add_to_archive(Pid, opponent_1, NetworkBinary),

    %% Verify stats
    Stats = competitive_silo:get_archive_stats(Pid),
    ?assertEqual(1, maps:get(archive_size, Stats)),
    ok = gen_server:stop(Pid).

get_elo_test() ->
    {ok, Pid} = competitive_silo:start_link(),

    %% Non-existent player
    ?assertEqual(not_found, competitive_silo:get_elo(Pid, unknown_player)),

    %% Add to archive (creates Elo)
    ok = competitive_silo:add_to_archive(Pid, player_1, <<"network">>),
    {ok, Elo} = competitive_silo:get_elo(Pid, player_1),
    ?assertEqual(1500.0, Elo),
    ok = gen_server:stop(Pid).

update_elo_test() ->
    {ok, Pid} = competitive_silo:start_link(),

    %% Update Elo for two players (win for player_1)
    {ok, Player1Elo, Player2Elo} = competitive_silo:update_elo(Pid, player_1, player_2, win),

    %% Winner should gain, loser should lose
    ?assert(Player1Elo > 1500.0),
    ?assert(Player2Elo < 1500.0),

    %% Check stored values
    {ok, StoredElo1} = competitive_silo:get_elo(Pid, player_1),
    {ok, StoredElo2} = competitive_silo:get_elo(Pid, player_2),
    ?assertEqual(Player1Elo, StoredElo1),
    ?assertEqual(Player2Elo, StoredElo2),
    ok = gen_server:stop(Pid).

record_match_test() ->
    {ok, Pid} = competitive_silo:start_link(),

    %% Record several matches
    ok = competitive_silo:record_match(Pid, player_1, player_2, win, 16.0),
    ok = competitive_silo:record_match(Pid, player_1, player_3, loss, -12.0),
    ok = competitive_silo:record_match(Pid, player_2, player_3, draw, 0.0),
    timer:sleep(50),

    %% Check stats
    Stats = competitive_silo:get_archive_stats(Pid),
    ?assertEqual(3, maps:get(match_count, Stats)),
    ok = gen_server:stop(Pid).

select_opponent_test() ->
    {ok, Pid} = competitive_silo:start_link(),

    %% No opponents yet
    ?assertEqual(no_opponents, competitive_silo:select_opponent(Pid, player_1)),

    %% Add some opponents
    ok = competitive_silo:add_to_archive(Pid, opp_1, <<"net1">>),
    ok = competitive_silo:add_to_archive(Pid, opp_2, <<"net2">>),
    ok = competitive_silo:add_to_archive(Pid, opp_3, <<"net3">>),

    %% Should find an opponent
    {ok, Opponent} = competitive_silo:select_opponent(Pid, player_1),
    ?assert(lists:member(Opponent, [opp_1, opp_2, opp_3])),
    ok = gen_server:stop(Pid).

get_archive_stats_test() ->
    {ok, Pid} = competitive_silo:start_link(),

    %% Empty stats
    Stats1 = competitive_silo:get_archive_stats(Pid),
    ?assertEqual(0, maps:get(archive_size, Stats1)),
    ?assertEqual(0, maps:get(match_count, Stats1)),

    %% Add data
    ok = competitive_silo:add_to_archive(Pid, opp_1, <<"net1">>),
    ok = competitive_silo:add_to_archive(Pid, opp_2, <<"net2">>),
    ok = competitive_silo:record_match(Pid, player_1, opp_1, win, 16.0),
    timer:sleep(50),

    Stats2 = competitive_silo:get_archive_stats(Pid),
    ?assertEqual(2, maps:get(archive_size, Stats2)),
    ?assertEqual(1, maps:get(match_count, Stats2)),
    ok = gen_server:stop(Pid).

reset_test() ->
    {ok, Pid} = competitive_silo:start_link(),

    %% Add data
    ok = competitive_silo:add_to_archive(Pid, opp_1, <<"net1">>),
    ok = competitive_silo:record_match(Pid, player_1, opp_1, win, 16.0),
    timer:sleep(50),

    %% Verify data exists
    Stats1 = competitive_silo:get_archive_stats(Pid),
    ?assertEqual(1, maps:get(archive_size, Stats1)),

    %% Reset
    ok = competitive_silo:reset(Pid),

    %% Verify reset
    Stats2 = competitive_silo:get_archive_stats(Pid),
    ?assertEqual(0, maps:get(archive_size, Stats2)),
    ?assertEqual(0, maps:get(match_count, Stats2)),
    ok = gen_server:stop(Pid).

get_state_test() ->
    {ok, Pid} = competitive_silo:start_link(#{realm => <<"test">>}),
    State = competitive_silo:get_state(Pid),
    ?assert(is_map(State)),
    ?assertEqual(<<"test">>, maps:get(realm, State)),
    ?assert(maps:is_key(current_params, State)),
    ?assert(maps:is_key(match_count, State)),
    ?assert(maps:is_key(sensors, State)),
    ok = gen_server:stop(Pid).

%%% ============================================================================
%%% Behavior Callback Tests
%%% ============================================================================

behavior_callbacks_test_() ->
    [
        {"get_silo_type returns competitive",
         ?_assertEqual(competitive, competitive_silo:get_silo_type())},
        {"get_time_constant returns 15.0",
         ?_assertEqual(15.0, competitive_silo:get_time_constant())}
    ].
