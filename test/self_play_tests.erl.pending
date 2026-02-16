%%% @doc Unit tests for self-play components (pure self-play mode).
%%%
%%% Tests opponent_archive, self_play_manager, and self_play_sup.
%%% Pure self-play: no heuristic opponents, population is the opponent pool.
%%% @end
-module(self_play_tests).

-include_lib("eunit/include/eunit.hrl").

%%====================================================================
%% Test Fixtures
%%====================================================================

archive_test_() ->
    {foreach,
     fun setup_archive/0,
     fun cleanup_archive/1,
     [
      {"create empty archive", fun test_archive_empty/0},
      {"add single opponent", fun test_archive_add_single/0},
      {"add multiple opponents", fun test_archive_add_multiple/0},
      {"sample from empty returns empty", fun test_archive_sample_empty/0},
      {"sample from archive returns opponent", fun test_archive_sample/0},
      {"archive respects max size", fun test_archive_max_size/0},
      {"archive stats computation", fun test_archive_stats/0},
      {"archive prune keeps top N", fun test_archive_prune/0},
      {"archive clear removes all", fun test_archive_clear/0},
      {"weighted sampling favors higher fitness", fun test_weighted_sampling/0}
     ]}.

manager_test_() ->
    {foreach,
     fun setup_manager/0,
     fun cleanup_manager/1,
     [
      {"empty archive returns from batch", fun test_manager_empty_archive_uses_batch/0},
      {"get opponent after archive populated", fun test_manager_after_archive_populated/0},
      {"report result adds to archive", fun test_manager_report_result/0},
      {"get stats returns metrics", fun test_manager_stats/0},
      {"sample from batch when archive empty", fun test_manager_batch_fallback/0}
     ]}.

sup_test_() ->
    {setup,
     fun setup_sup/0,
     fun cleanup_sup/1,
     [
      {"start and stop self-play for realm", fun test_sup_start_stop/0},
      {"get manager returns pid", fun test_sup_get_manager/0},
      {"list realms returns active realms", fun test_sup_list_realms/0}
     ]}.

%%====================================================================
%% Setup/Cleanup
%%====================================================================

setup_archive() ->
    %% Use a unique archive ID for each test
    ArchiveId = list_to_atom("test_archive_" ++ integer_to_list(erlang:unique_integer([positive]))),
    {ok, _Pid} = opponent_archive:start_link(ArchiveId, #{max_size => 10}),
    ArchiveId.

cleanup_archive(ArchiveId) ->
    catch opponent_archive:stop(ArchiveId),
    ok.

setup_manager() ->
    %% Use a unique realm for each test
    Realm = list_to_atom("test_realm_" ++ integer_to_list(erlang:unique_integer([positive]))),
    {ok, Pid} = self_play_manager:start_link(Realm, #{
        archive_size => 10
    }),
    {Realm, Pid}.

cleanup_manager({_Realm, Pid}) ->
    catch self_play_manager:stop(Pid),
    ok.

setup_sup() ->
    %% Start the supervisor if not already running
    case whereis(self_play_sup) of
        undefined ->
            {ok, Pid} = self_play_sup:start_link(),
            {started, Pid};
        Pid ->
            {existing, Pid}
    end.

cleanup_sup({started, _Pid}) ->
    %% Stop any test realms
    lists:foreach(
        fun(Realm) -> self_play_sup:stop_self_play(Realm) end,
        self_play_sup:list_realms()
    ),
    ok;
cleanup_sup({existing, _Pid}) ->
    %% Don't stop existing supervisor
    ok.

%%====================================================================
%% Archive Tests
%%====================================================================

test_archive_empty() ->
    ArchiveId = setup_archive(),
    ?assertEqual(0, opponent_archive:size(ArchiveId)),
    cleanup_archive(ArchiveId).

test_archive_add_single() ->
    ArchiveId = setup_archive(),
    Opponent = make_opponent(100.0),
    ?assertEqual(ok, opponent_archive:add(ArchiveId, Opponent)),
    ?assertEqual(1, opponent_archive:size(ArchiveId)),
    cleanup_archive(ArchiveId).

test_archive_add_multiple() ->
    ArchiveId = setup_archive(),
    lists:foreach(
        fun(Fitness) ->
            opponent_archive:add(ArchiveId, make_opponent(Fitness))
        end,
        [100.0, 150.0, 200.0]
    ),
    ?assertEqual(3, opponent_archive:size(ArchiveId)),
    cleanup_archive(ArchiveId).

test_archive_sample_empty() ->
    ArchiveId = setup_archive(),
    ?assertEqual(empty, opponent_archive:sample(ArchiveId)),
    cleanup_archive(ArchiveId).

test_archive_sample() ->
    ArchiveId = setup_archive(),
    opponent_archive:add(ArchiveId, make_opponent(100.0)),
    {ok, Opponent} = opponent_archive:sample(ArchiveId),
    ?assert(is_map(Opponent)),
    ?assert(maps:is_key(network, Opponent)),
    ?assert(maps:is_key(fitness, Opponent)),
    cleanup_archive(ArchiveId).

test_archive_max_size() ->
    ArchiveId = setup_archive(),
    %% Add 15 opponents to archive with max_size 10
    lists:foreach(
        fun(I) ->
            %% Use increasing fitness so newer ones replace older
            opponent_archive:add(ArchiveId, make_opponent(float(I * 10)))
        end,
        lists:seq(1, 15)
    ),
    %% Should be capped at 10
    ?assertEqual(10, opponent_archive:size(ArchiveId)),
    cleanup_archive(ArchiveId).

test_archive_stats() ->
    ArchiveId = setup_archive(),
    lists:foreach(
        fun(Fitness) ->
            opponent_archive:add(ArchiveId, make_opponent(Fitness))
        end,
        [100.0, 200.0, 300.0]
    ),
    Stats = opponent_archive:stats(ArchiveId),
    ?assertEqual(3, maps:get(count, Stats)),
    ?assertEqual(10, maps:get(max_size, Stats)),
    ?assertEqual(200.0, maps:get(avg_fitness, Stats)),
    ?assertEqual(300.0, maps:get(max_fitness, Stats)),
    ?assertEqual(100.0, maps:get(min_fitness, Stats)),
    cleanup_archive(ArchiveId).

test_archive_prune() ->
    ArchiveId = setup_archive(),
    %% Add 5 opponents with varying fitness
    lists:foreach(
        fun(Fitness) ->
            opponent_archive:add(ArchiveId, make_opponent(Fitness))
        end,
        [100.0, 200.0, 300.0, 400.0, 500.0]
    ),
    ?assertEqual(5, opponent_archive:size(ArchiveId)),

    %% Prune to keep top 2
    opponent_archive:prune(ArchiveId, 2),
    ?assertEqual(2, opponent_archive:size(ArchiveId)),

    %% Check remaining are the highest fitness
    Stats = opponent_archive:stats(ArchiveId),
    ?assert(maps:get(min_fitness, Stats) >= 400.0),
    cleanup_archive(ArchiveId).

test_archive_clear() ->
    ArchiveId = setup_archive(),
    opponent_archive:add(ArchiveId, make_opponent(100.0)),
    opponent_archive:add(ArchiveId, make_opponent(200.0)),
    ?assertEqual(2, opponent_archive:size(ArchiveId)),

    opponent_archive:clear(ArchiveId),
    ?assertEqual(0, opponent_archive:size(ArchiveId)),
    cleanup_archive(ArchiveId).

test_weighted_sampling() ->
    ArchiveId = setup_archive(),
    %% Add opponents with very different fitness
    opponent_archive:add(ArchiveId, make_opponent(1.0)),      % Low fitness
    opponent_archive:add(ArchiveId, make_opponent(1000.0)),   % High fitness

    %% Sample many times and count
    Samples = [element(2, opponent_archive:sample(ArchiveId)) || _ <- lists:seq(1, 100)],
    HighFitness = length([S || S <- Samples, maps:get(fitness, S) > 500]),

    %% High fitness should be sampled more often (at least 60% of time)
    ?assert(HighFitness >= 60),
    cleanup_archive(ArchiveId).

%%====================================================================
%% Manager Tests (Pure Self-Play)
%%====================================================================

test_manager_empty_archive_uses_batch() ->
    {_Realm, Pid} = setup_manager(),
    %% When archive is empty, should sample from batch networks
    BatchNetworks = [make_network(1), make_network(2), make_network(3)],
    {ok, Network} = self_play_manager:get_opponent(Pid, BatchNetworks),
    ?assert(is_map(Network)),
    ?assert(lists:member(Network, BatchNetworks)),
    cleanup_manager({undefined, Pid}).

test_manager_after_archive_populated() ->
    {_Realm, Pid} = setup_manager(),
    %% Add a champion to archive
    Champion = make_opponent(100.0),
    self_play_manager:add_champion(Pid, Champion),

    %% Now should get from archive (not batch)
    BatchNetworks = [make_network(999)],  % Different from champion
    {ok, Network} = self_play_manager:get_opponent(Pid, BatchNetworks),
    ?assert(is_map(Network)),
    %% Should be the champion's network, not the batch network
    ?assertEqual(maps:get(network, Champion), Network),
    cleanup_manager({undefined, Pid}).

test_manager_report_result() ->
    {_Realm, Pid} = setup_manager(),
    %% Report a high-fitness result
    Result = #{
        individual => #{network => #{test => true}},
        fitness => 1000.0,
        generation => 1
    },
    self_play_manager:report_result(Pid, Result),

    %% Give it time to process
    timer:sleep(50),

    %% Check stats show champion added
    Stats = self_play_manager:get_stats(Pid),
    ?assert(maps:get(champions_added, Stats) >= 1),
    cleanup_manager({undefined, Pid}).

test_manager_stats() ->
    {_Realm, Pid} = setup_manager(),
    Stats = self_play_manager:get_stats(Pid),
    ?assert(is_map(Stats)),
    ?assert(maps:is_key(total_evaluations, Stats)),
    ?assert(maps:is_key(archive_size, Stats)),
    ?assertEqual(pure_self_play, maps:get(mode, Stats)),
    cleanup_manager({undefined, Pid}).

test_manager_batch_fallback() ->
    {_Realm, Pid} = setup_manager(),
    %% Create distinct batch networks
    Net1 = #{weights => [1.0], id => net1},
    Net2 = #{weights => [2.0], id => net2},
    Net3 = #{weights => [3.0], id => net3},
    BatchNetworks = [Net1, Net2, Net3],

    %% Sample multiple times - should all come from batch
    Samples = [begin
        {ok, N} = self_play_manager:get_opponent(Pid, BatchNetworks),
        N
    end || _ <- lists:seq(1, 10)],

    %% All samples should be from batch
    lists:foreach(
        fun(Sample) ->
            ?assert(lists:member(Sample, BatchNetworks))
        end,
        Samples
    ),
    cleanup_manager({undefined, Pid}).

%%====================================================================
%% Supervisor Tests
%%====================================================================

test_sup_start_stop() ->
    Realm = list_to_atom("test_sup_realm_" ++ integer_to_list(erlang:unique_integer([positive]))),
    {ok, Pid} = self_play_sup:start_self_play(Realm, #{}),
    ?assert(is_pid(Pid)),

    %% Stop it
    ?assertEqual(ok, self_play_sup:stop_self_play(Realm)),

    %% Should be gone
    ?assertEqual({error, not_found}, self_play_sup:get_manager(Realm)).

test_sup_get_manager() ->
    Realm = list_to_atom("test_sup_get_" ++ integer_to_list(erlang:unique_integer([positive]))),
    {ok, Pid1} = self_play_sup:start_self_play(Realm, #{}),

    {ok, Pid2} = self_play_sup:get_manager(Realm),
    ?assertEqual(Pid1, Pid2),

    self_play_sup:stop_self_play(Realm).

test_sup_list_realms() ->
    Realm1 = list_to_atom("test_list_1_" ++ integer_to_list(erlang:unique_integer([positive]))),
    Realm2 = list_to_atom("test_list_2_" ++ integer_to_list(erlang:unique_integer([positive]))),

    {ok, _} = self_play_sup:start_self_play(Realm1, #{}),
    {ok, _} = self_play_sup:start_self_play(Realm2, #{}),

    Realms = self_play_sup:list_realms(),
    ?assert(lists:member(Realm1, Realms)),
    ?assert(lists:member(Realm2, Realms)),

    self_play_sup:stop_self_play(Realm1),
    self_play_sup:stop_self_play(Realm2).

%%====================================================================
%% Test Helpers
%%====================================================================

make_opponent(Fitness) ->
    #{
        network => make_network(Fitness),
        fitness => Fitness,
        generation => 1
    }.

make_network(Seed) ->
    #{
        weights => [float(Seed), float(Seed * 2), float(Seed * 3)],
        topology => #{inputs => 4, outputs => 2},
        seed => Seed
    }.
