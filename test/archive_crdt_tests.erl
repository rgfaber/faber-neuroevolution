%% @doc Unit tests for archive_crdt OR-Set module.
-module(archive_crdt_tests).

-include_lib("eunit/include/eunit.hrl").

%%====================================================================
%% Test Fixtures
%%====================================================================

crdt_test_() ->
    {foreach,
     fun setup/0,
     fun cleanup/1,
     [
      {"create new orset", fun test_new_orset/0},
      {"add element to orset", fun test_add_element/0},
      {"add multiple elements", fun test_add_multiple/0},
      {"remove element", fun test_remove_element/0},
      {"merge two orsets", fun test_merge_orsets/0},
      {"merge is commutative", fun test_merge_commutative/0},
      {"merge is associative", fun test_merge_associative/0},
      {"merge is idempotent", fun test_merge_idempotent/0},
      {"concurrent adds merge correctly", fun test_concurrent_adds/0},
      {"tombstones win in merge", fun test_tombstone_wins/0},
      {"export and import binary", fun test_export_import/0},
      {"import invalid binary fails", fun test_import_invalid/0},
      {"size returns live count", fun test_size/0},
      {"is_crdt predicate", fun test_is_crdt/0}
     ]}.

setup() ->
    ok.

cleanup(_) ->
    ok.

%%====================================================================
%% Tests
%%====================================================================

test_new_orset() ->
    CRDT = archive_crdt:new(),
    ?assert(archive_crdt:is_crdt(CRDT)),
    ?assertEqual([], archive_crdt:value(CRDT)),
    ?assertEqual(0, archive_crdt:size(CRDT)).

test_add_element() ->
    ActorId = <<"node1">>,
    CRDT = archive_crdt:new(ActorId),
    Entry = #{id => <<"entry1">>, fitness => 0.9},
    CRDT2 = archive_crdt:add(Entry, CRDT, ActorId),
    ?assertEqual(1, archive_crdt:size(CRDT2)),
    [Retrieved] = archive_crdt:value(CRDT2),
    ?assertEqual(Entry, Retrieved).

test_add_multiple() ->
    ActorId = <<"node1">>,
    CRDT = archive_crdt:new(ActorId),
    E1 = #{id => <<"e1">>, fitness => 0.8},
    E2 = #{id => <<"e2">>, fitness => 0.9},
    E3 = #{id => <<"e3">>, fitness => 0.7},
    CRDT2 = archive_crdt:add(E1, CRDT, ActorId),
    CRDT3 = archive_crdt:add(E2, CRDT2, ActorId),
    CRDT4 = archive_crdt:add(E3, CRDT3, ActorId),
    ?assertEqual(3, archive_crdt:size(CRDT4)),
    Values = archive_crdt:value(CRDT4),
    ?assert(lists:member(E1, Values)),
    ?assert(lists:member(E2, Values)),
    ?assert(lists:member(E3, Values)).

test_remove_element() ->
    ActorId = <<"node1">>,
    CRDT = archive_crdt:new(ActorId),
    E1 = #{id => <<"e1">>, fitness => 0.8},
    E2 = #{id => <<"e2">>, fitness => 0.9},
    CRDT2 = archive_crdt:add(E1, CRDT, ActorId),
    CRDT3 = archive_crdt:add(E2, CRDT2, ActorId),
    ?assertEqual(2, archive_crdt:size(CRDT3)),
    CRDT4 = archive_crdt:remove(E1, CRDT3),
    ?assertEqual(1, archive_crdt:size(CRDT4)),
    [Remaining] = archive_crdt:value(CRDT4),
    ?assertEqual(E2, Remaining).

test_merge_orsets() ->
    Actor1 = <<"node1">>,
    Actor2 = <<"node2">>,
    CRDT1 = archive_crdt:new(Actor1),
    CRDT2 = archive_crdt:new(Actor2),
    E1 = #{id => <<"e1">>, fitness => 0.8},
    E2 = #{id => <<"e2">>, fitness => 0.9},
    CRDT1_2 = archive_crdt:add(E1, CRDT1, Actor1),
    CRDT2_2 = archive_crdt:add(E2, CRDT2, Actor2),
    Merged = archive_crdt:merge(CRDT1_2, CRDT2_2),
    ?assertEqual(2, archive_crdt:size(Merged)),
    Values = archive_crdt:value(Merged),
    ?assert(lists:member(E1, Values)),
    ?assert(lists:member(E2, Values)).

test_merge_commutative() ->
    Actor1 = <<"node1">>,
    Actor2 = <<"node2">>,
    CRDT1 = archive_crdt:add(#{id => <<"e1">>}, archive_crdt:new(Actor1), Actor1),
    CRDT2 = archive_crdt:add(#{id => <<"e2">>}, archive_crdt:new(Actor2), Actor2),
    Merged1 = archive_crdt:merge(CRDT1, CRDT2),
    Merged2 = archive_crdt:merge(CRDT2, CRDT1),
    ?assertEqual(lists:sort(archive_crdt:value(Merged1)),
                 lists:sort(archive_crdt:value(Merged2))).

test_merge_associative() ->
    Actor1 = <<"node1">>,
    Actor2 = <<"node2">>,
    Actor3 = <<"node3">>,
    CRDT1 = archive_crdt:add(#{id => <<"e1">>}, archive_crdt:new(Actor1), Actor1),
    CRDT2 = archive_crdt:add(#{id => <<"e2">>}, archive_crdt:new(Actor2), Actor2),
    CRDT3 = archive_crdt:add(#{id => <<"e3">>}, archive_crdt:new(Actor3), Actor3),
    %% (A merge B) merge C
    MergedABC = archive_crdt:merge(archive_crdt:merge(CRDT1, CRDT2), CRDT3),
    %% A merge (B merge C)
    MergedABC2 = archive_crdt:merge(CRDT1, archive_crdt:merge(CRDT2, CRDT3)),
    ?assertEqual(lists:sort(archive_crdt:value(MergedABC)),
                 lists:sort(archive_crdt:value(MergedABC2))).

test_merge_idempotent() ->
    ActorId = <<"node1">>,
    CRDT = archive_crdt:new(ActorId),
    E1 = #{id => <<"e1">>, fitness => 0.8},
    CRDT2 = archive_crdt:add(E1, CRDT, ActorId),
    Merged = archive_crdt:merge(CRDT2, CRDT2),
    ?assertEqual(archive_crdt:value(CRDT2), archive_crdt:value(Merged)).

test_concurrent_adds() ->
    Actor1 = <<"node1">>,
    Actor2 = <<"node2">>,
    %% Both nodes start with empty CRDT
    Base = archive_crdt:new(Actor1),
    E1 = #{id => <<"champion1">>, fitness => 0.95},
    E2 = #{id => <<"champion2">>, fitness => 0.92},
    %% Concurrent adds on different nodes
    CRDT1 = archive_crdt:add(E1, Base, Actor1),
    CRDT2 = archive_crdt:add(E2, archive_crdt:new(Actor2), Actor2),
    %% Merge preserves both
    Merged = archive_crdt:merge(CRDT1, CRDT2),
    ?assertEqual(2, archive_crdt:size(Merged)),
    Values = archive_crdt:value(Merged),
    ?assert(lists:member(E1, Values)),
    ?assert(lists:member(E2, Values)).

test_tombstone_wins() ->
    ActorId = <<"node1">>,
    CRDT1 = archive_crdt:new(ActorId),
    E1 = #{id => <<"e1">>, fitness => 0.8},
    CRDT2 = archive_crdt:add(E1, CRDT1, ActorId),
    %% Create copy and remove from copy
    CRDT3 = archive_crdt:remove(E1, CRDT2),
    ?assertEqual(0, archive_crdt:size(CRDT3)),
    %% Merge: tombstone wins even if other has live entry
    Merged = archive_crdt:merge(CRDT2, CRDT3),
    ?assertEqual(0, archive_crdt:size(Merged)).

test_export_import() ->
    ActorId = <<"node1">>,
    CRDT = archive_crdt:new(ActorId),
    E1 = #{id => <<"e1">>, fitness => 0.8, network => #{weights => [0.1, 0.2]}},
    E2 = #{id => <<"e2">>, fitness => 0.9, network => #{weights => [0.3, 0.4]}},
    CRDT2 = archive_crdt:add(E1, CRDT, ActorId),
    CRDT3 = archive_crdt:add(E2, CRDT2, ActorId),
    Binary = archive_crdt:export_binary(CRDT3),
    ?assert(is_binary(Binary)),
    {ok, Imported} = archive_crdt:import_binary(Binary),
    ?assertEqual(lists:sort(archive_crdt:value(CRDT3)),
                 lists:sort(archive_crdt:value(Imported))).

test_import_invalid() ->
    ?assertEqual({error, invalid_format}, archive_crdt:import_binary(<<>>)),
    ?assertEqual({error, {unsupported_version, 99}},
                 archive_crdt:import_binary(<<99:8, 0:16, 0:64, 0:32>>)).

test_size() ->
    ActorId = <<"node1">>,
    CRDT = archive_crdt:new(ActorId),
    ?assertEqual(0, archive_crdt:size(CRDT)),
    CRDT2 = archive_crdt:add(#{id => <<"e1">>}, CRDT, ActorId),
    ?assertEqual(1, archive_crdt:size(CRDT2)),
    CRDT3 = archive_crdt:add(#{id => <<"e2">>}, CRDT2, ActorId),
    ?assertEqual(2, archive_crdt:size(CRDT3)),
    CRDT4 = archive_crdt:remove(#{id => <<"e1">>}, CRDT3),
    ?assertEqual(1, archive_crdt:size(CRDT4)).

test_is_crdt() ->
    CRDT = archive_crdt:new(),
    ?assert(archive_crdt:is_crdt(CRDT)),
    ?assertNot(archive_crdt:is_crdt(#{})),
    ?assertNot(archive_crdt:is_crdt([])),
    ?assertNot(archive_crdt:is_crdt(atom)).

%%====================================================================
%% Archive Integration Tests
%%====================================================================

%% NOTE: CRDT tracking in red_team_archive is currently disabled to prevent
%% memory leaks (full network data was accumulating without pruning).
%% See red_team_archive.erl lines 344-346.
%%
%% When Phase 2 mesh sync is implemented with proper genome-only storage,
%% re-enable these tests by uncommenting them.
%%
%% The archive_crdt module unit tests (14 tests above) verify that the CRDT
%% implementation itself works correctly. Archive integration will be
%% re-tested when CRDT tracking is re-enabled.

archive_crdt_integration_test_() ->
    {foreach,
     fun setup_archive/0,
     fun cleanup_archive/1,
     [
      {"export crdt from archive", fun test_archive_export_crdt/0}
      %% DISABLED: CRDT merge tests - waiting for Phase 2 re-enablement
      %% {"merge crdt into archive", fun test_archive_merge_crdt/0},
      %% {"merge archives from two nodes", fun test_archive_two_node_merge/0}
     ]}.

setup_archive() ->
    ok.

cleanup_archive(_) ->
    catch red_team_archive:stop(test_crdt_archive_1),
    catch red_team_archive:stop(test_crdt_archive_2),
    ok.

test_archive_export_crdt() ->
    {ok, _} = red_team_archive:start_link(test_crdt_archive_1, #{max_size => 10}),
    Opponent = #{network => #{weights => [0.1, 0.2]}, fitness => 0.9, generation => 1},
    ok = red_team_archive:add(test_crdt_archive_1, Opponent),
    Binary = red_team_archive:export_crdt(test_crdt_archive_1),
    ?assert(is_binary(Binary)),
    ?assert(byte_size(Binary) > 0),
    red_team_archive:stop(test_crdt_archive_1).

%% DISABLED: CRDT merge tests - waiting for Phase 2 re-enablement
%% CRDT tracking in red_team_archive is currently disabled to prevent memory leaks.
%% See red_team_archive.erl lines 344-346.
%%
%% test_archive_merge_crdt() ->
%%     {ok, _} = red_team_archive:start_link(test_crdt_archive_1, #{max_size => 10}),
%%     {ok, _} = red_team_archive:start_link(test_crdt_archive_2, #{max_size => 10}),
%%     %% Add to archive 1
%%     O1 = #{network => #{weights => [0.1]}, fitness => 0.9, generation => 1},
%%     ok = red_team_archive:add(test_crdt_archive_1, O1),
%%     %% Add to archive 2
%%     O2 = #{network => #{weights => [0.2]}, fitness => 0.8, generation => 2},
%%     ok = red_team_archive:add(test_crdt_archive_2, O2),
%%     %% Export from archive 1 and merge into archive 2
%%     Crdt1 = red_team_archive:export_crdt(test_crdt_archive_1),
%%     ok = red_team_archive:merge_crdt(test_crdt_archive_2, Crdt1),
%%     %% Archive 2 should now have both
%%     ?assertEqual(2, red_team_archive:size(test_crdt_archive_2)),
%%     red_team_archive:stop(test_crdt_archive_1),
%%     red_team_archive:stop(test_crdt_archive_2).
%%
%% test_archive_two_node_merge() ->
%%     {ok, _} = red_team_archive:start_link(test_crdt_archive_1, #{max_size => 10}),
%%     {ok, _} = red_team_archive:start_link(test_crdt_archive_2, #{max_size => 10}),
%%     %% Simulate two nodes adding different champions concurrently
%%     lists:foreach(
%%         fun(I) ->
%%             O = #{network => #{id => I}, fitness => 0.5 + I/10, generation => I},
%%             ok = red_team_archive:add(test_crdt_archive_1, O)
%%         end,
%%         [1, 2, 3]
%%     ),
%%     lists:foreach(
%%         fun(I) ->
%%             O = #{network => #{id => I + 100}, fitness => 0.5 + I/10, generation => I},
%%             ok = red_team_archive:add(test_crdt_archive_2, O)
%%         end,
%%         [1, 2, 3]
%%     ),
%%     ?assertEqual(3, red_team_archive:size(test_crdt_archive_1)),
%%     ?assertEqual(3, red_team_archive:size(test_crdt_archive_2)),
%%     %% Bidirectional merge
%%     Crdt1 = red_team_archive:export_crdt(test_crdt_archive_1),
%%     Crdt2 = red_team_archive:export_crdt(test_crdt_archive_2),
%%     ok = red_team_archive:merge_crdt(test_crdt_archive_1, Crdt2),
%%     ok = red_team_archive:merge_crdt(test_crdt_archive_2, Crdt1),
%%     %% Both archives should have all 6 entries
%%     ?assertEqual(6, red_team_archive:size(test_crdt_archive_1)),
%%     ?assertEqual(6, red_team_archive:size(test_crdt_archive_2)),
%%     red_team_archive:stop(test_crdt_archive_1),
%%     red_team_archive:stop(test_crdt_archive_2).
