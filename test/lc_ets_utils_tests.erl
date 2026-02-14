%% @doc Unit tests for lc_ets_utils module.
%%
%% Tests ETS table utilities for Liquid Conglomerate silos.
%%
%% @author R.G. Lefever
%% @copyright 2024-2026 R.G. Lefever
-module(lc_ets_utils_tests).

-include_lib("eunit/include/eunit.hrl").

%%% ============================================================================
%%% Table Management Tests
%%% ============================================================================

create_tables_test() ->
    Tables = lc_ets_utils:create_tables(test_silo, <<"realm1">>, [
        {items, [{keypos, 1}]},
        {cache, [{keypos, 1}, {read_concurrency, true}]}
    ]),
    ?assertEqual(2, maps:size(Tables)),
    ?assert(maps:is_key(items, Tables)),
    ?assert(maps:is_key(cache, Tables)),
    %% Tables should be valid ETS references
    ItemsTable = maps:get(items, Tables),
    ?assertEqual(0, ets:info(ItemsTable, size)),
    lc_ets_utils:delete_tables(Tables).

delete_tables_test() ->
    Tables = lc_ets_utils:create_tables(delete_test, <<"realm1">>, [
        {temp, []}
    ]),
    TempTable = maps:get(temp, Tables),
    ?assert(is_reference(TempTable) orelse is_atom(TempTable)),
    ok = lc_ets_utils:delete_tables(Tables),
    %% Table should no longer exist
    ?assertEqual(undefined, ets:info(TempTable, size)).

table_name_test() ->
    Name = lc_ets_utils:table_name(competitive, <<"world1">>, opponents),
    ?assertEqual(competitive_world1_opponents, Name).

%%% ============================================================================
%%% CRUD Operations Tests
%%% ============================================================================

insert_lookup_test() ->
    Tables = lc_ets_utils:create_tables(insert_test, <<"test">>, [{data, []}]),
    Table = maps:get(data, Tables),
    true = lc_ets_utils:insert(Table, key1, <<"value1">>),
    ?assertEqual({ok, <<"value1">>}, lc_ets_utils:lookup(Table, key1)),
    lc_ets_utils:delete_tables(Tables).

insert_timestamp_test() ->
    Tables = lc_ets_utils:create_tables(insert_ts_test, <<"test">>, [{data, []}]),
    Table = maps:get(data, Tables),
    Timestamp = 1234567890,
    true = lc_ets_utils:insert_with_timestamp(Table, key1, data, Timestamp),
    {ok, data, StoredTs} = lc_ets_utils:lookup_with_timestamp(Table, key1),
    ?assertEqual(Timestamp, StoredTs),
    lc_ets_utils:delete_tables(Tables).

lookup_not_found_test() ->
    Tables = lc_ets_utils:create_tables(lookup_nf_test, <<"test">>, [{data, []}]),
    Table = maps:get(data, Tables),
    ?assertEqual(not_found, lc_ets_utils:lookup(Table, nonexistent)),
    lc_ets_utils:delete_tables(Tables).

lookup_with_timestamp_test() ->
    Tables = lc_ets_utils:create_tables(lookup_ts_test, <<"test">>, [{data, []}]),
    Table = maps:get(data, Tables),
    Before = erlang:system_time(millisecond),
    true = lc_ets_utils:insert(Table, key1, value1),
    After = erlang:system_time(millisecond),
    {ok, value1, Timestamp} = lc_ets_utils:lookup_with_timestamp(Table, key1),
    ?assert(Timestamp >= Before),
    ?assert(Timestamp =< After),
    lc_ets_utils:delete_tables(Tables).

delete_entry_test() ->
    Tables = lc_ets_utils:create_tables(delete_entry_test, <<"test">>, [{data, []}]),
    Table = maps:get(data, Tables),
    true = lc_ets_utils:insert(Table, key1, value1),
    ?assertEqual({ok, value1}, lc_ets_utils:lookup(Table, key1)),
    true = lc_ets_utils:delete(Table, key1),
    ?assertEqual(not_found, lc_ets_utils:lookup(Table, key1)),
    lc_ets_utils:delete_tables(Tables).

update_test() ->
    Tables = lc_ets_utils:create_tables(update_test, <<"test">>, [{data, []}]),
    Table = maps:get(data, Tables),
    true = lc_ets_utils:insert(Table, counter, 0),
    ok = lc_ets_utils:update(Table, counter, fun(V) -> V + 1 end),
    ?assertEqual({ok, 1}, lc_ets_utils:lookup(Table, counter)),
    lc_ets_utils:delete_tables(Tables).

update_missing_test() ->
    Tables = lc_ets_utils:create_tables(update_missing_test, <<"test">>, [{data, []}]),
    Table = maps:get(data, Tables),
    ok = lc_ets_utils:update(Table, new_key, fun(undefined) -> initial; (V) -> V end),
    ?assertEqual({ok, initial}, lc_ets_utils:lookup(Table, new_key)),
    lc_ets_utils:delete_tables(Tables).

%%% ============================================================================
%%% Bulk Operations Tests
%%% ============================================================================

all_entries_test() ->
    Tables = lc_ets_utils:create_tables(all_test, <<"test">>, [{items, []}]),
    Table = maps:get(items, Tables),
    lc_ets_utils:insert(Table, a, 1),
    lc_ets_utils:insert(Table, b, 2),
    lc_ets_utils:insert(Table, c, 3),
    All = lc_ets_utils:all(Table),
    ?assertEqual(3, length(All)),
    Keys = [K || {K, _, _} <- All],
    ?assert(lists:member(a, Keys)),
    ?assert(lists:member(b, Keys)),
    ?assert(lists:member(c, Keys)),
    lc_ets_utils:delete_tables(Tables).

all_keys_test() ->
    Tables = lc_ets_utils:create_tables(allkeys_test, <<"test">>, [{items, []}]),
    Table = maps:get(items, Tables),
    lc_ets_utils:insert(Table, a, 1),
    lc_ets_utils:insert(Table, b, 2),
    lc_ets_utils:insert(Table, c, 3),
    Keys = lc_ets_utils:all_keys(Table),
    ?assertEqual(3, length(Keys)),
    ?assert(lists:member(a, Keys)),
    ?assert(lists:member(b, Keys)),
    ?assert(lists:member(c, Keys)),
    lc_ets_utils:delete_tables(Tables).

count_test() ->
    Tables = lc_ets_utils:create_tables(count_test, <<"test">>, [{items, []}]),
    Table = maps:get(items, Tables),
    lc_ets_utils:insert(Table, a, 1),
    lc_ets_utils:insert(Table, b, 2),
    lc_ets_utils:insert(Table, c, 3),
    ?assertEqual(3, lc_ets_utils:count(Table)),
    lc_ets_utils:delete_tables(Tables).

%%% ============================================================================
%%% Time-Based Operations Tests
%%% ============================================================================

prune_by_age_test() ->
    Tables = lc_ets_utils:create_tables(prune_age_test, <<"test">>, [{events, []}]),
    Table = maps:get(events, Tables),
    %% Insert entries with explicit old timestamps
    OldTs = erlang:system_time(millisecond) - 10000,
    NewTs = erlang:system_time(millisecond),
    lc_ets_utils:insert_with_timestamp(Table, old1, data1, OldTs),
    lc_ets_utils:insert_with_timestamp(Table, old2, data2, OldTs),
    lc_ets_utils:insert_with_timestamp(Table, new1, data3, NewTs),
    ?assertEqual(3, lc_ets_utils:count(Table)),
    %% Prune entries older than 5 seconds
    Deleted = lc_ets_utils:prune_by_age(Table, 5000),
    ?assertEqual(2, Deleted),
    ?assertEqual(1, lc_ets_utils:count(Table)),
    ?assertEqual({ok, data3}, lc_ets_utils:lookup(Table, new1)),
    lc_ets_utils:delete_tables(Tables).

prune_by_count_test() ->
    Tables = lc_ets_utils:create_tables(prune_cnt_test, <<"test">>, [{events, []}]),
    Table = maps:get(events, Tables),
    %% Insert 5 entries with different timestamps
    lists:foreach(
        fun(I) ->
            lc_ets_utils:insert_with_timestamp(Table, I, I * 10, I * 1000)
        end,
        lists:seq(1, 5)
    ),
    ?assertEqual(5, lc_ets_utils:count(Table)),
    %% Prune to max 3
    Deleted = lc_ets_utils:prune_by_count(Table, 3),
    ?assertEqual(2, Deleted),
    ?assertEqual(3, lc_ets_utils:count(Table)),
    %% Oldest (1, 2) should be gone, newest (3, 4, 5) should remain
    ?assertEqual(not_found, lc_ets_utils:lookup(Table, 1)),
    ?assertEqual(not_found, lc_ets_utils:lookup(Table, 2)),
    ?assertEqual({ok, 30}, lc_ets_utils:lookup(Table, 3)),
    lc_ets_utils:delete_tables(Tables).

get_recent_test() ->
    Tables = lc_ets_utils:create_tables(recent_test, <<"test">>, [{events, []}]),
    Table = maps:get(events, Tables),
    %% Insert entries with different timestamps
    lists:foreach(
        fun(I) ->
            lc_ets_utils:insert_with_timestamp(Table, I, I * 10, I * 1000)
        end,
        lists:seq(1, 5)
    ),
    Recent = lc_ets_utils:get_recent(Table, 2),
    ?assertEqual(2, length(Recent)),
    %% Most recent first
    [{K1, _, _}, {K2, _, _}] = Recent,
    ?assertEqual(5, K1),
    ?assertEqual(4, K2),
    lc_ets_utils:delete_tables(Tables).

get_oldest_test() ->
    Tables = lc_ets_utils:create_tables(oldest_test, <<"test">>, [{events, []}]),
    Table = maps:get(events, Tables),
    %% Insert entries with different timestamps
    lists:foreach(
        fun(I) ->
            lc_ets_utils:insert_with_timestamp(Table, I, I * 10, I * 1000)
        end,
        lists:seq(1, 5)
    ),
    Oldest = lc_ets_utils:get_oldest(Table, 2),
    ?assertEqual(2, length(Oldest)),
    %% Oldest first
    [{K1, _, _}, {K2, _, _}] = Oldest,
    ?assertEqual(1, K1),
    ?assertEqual(2, K2),
    lc_ets_utils:delete_tables(Tables).

%%% ============================================================================
%%% Aggregation Tests
%%% ============================================================================

sum_field_test() ->
    Tables = lc_ets_utils:create_tables(sum_test, <<"test">>, [{scores, []}]),
    Table = maps:get(scores, Tables),
    lc_ets_utils:insert(Table, a, #{value => 10}),
    lc_ets_utils:insert(Table, b, #{value => 20}),
    lc_ets_utils:insert(Table, c, #{value => 30}),
    Sum = lc_ets_utils:sum_field(Table, fun(M) -> maps:get(value, M) end),
    ?assertEqual(60, Sum),
    lc_ets_utils:delete_tables(Tables).

avg_field_test() ->
    Tables = lc_ets_utils:create_tables(avg_test, <<"test">>, [{scores, []}]),
    Table = maps:get(scores, Tables),
    lc_ets_utils:insert(Table, a, #{value => 10}),
    lc_ets_utils:insert(Table, b, #{value => 20}),
    lc_ets_utils:insert(Table, c, #{value => 30}),
    Avg = lc_ets_utils:avg_field(Table, fun(M) -> maps:get(value, M) end),
    ?assertEqual(20.0, Avg),
    lc_ets_utils:delete_tables(Tables).

max_field_test() ->
    Tables = lc_ets_utils:create_tables(max_test, <<"test">>, [{scores, []}]),
    Table = maps:get(scores, Tables),
    lc_ets_utils:insert(Table, a, #{value => 10}),
    lc_ets_utils:insert(Table, b, #{value => 20}),
    lc_ets_utils:insert(Table, c, #{value => 30}),
    Max = lc_ets_utils:max_field(Table, fun(M) -> maps:get(value, M) end),
    ?assertEqual(30, Max),
    lc_ets_utils:delete_tables(Tables).

min_field_test() ->
    Tables = lc_ets_utils:create_tables(min_test, <<"test">>, [{scores, []}]),
    Table = maps:get(scores, Tables),
    lc_ets_utils:insert(Table, a, #{value => 10}),
    lc_ets_utils:insert(Table, b, #{value => 20}),
    lc_ets_utils:insert(Table, c, #{value => 30}),
    Min = lc_ets_utils:min_field(Table, fun(M) -> maps:get(value, M) end),
    ?assertEqual(10, Min),
    lc_ets_utils:delete_tables(Tables).

aggregation_empty_test() ->
    Tables = lc_ets_utils:create_tables(empty_agg_test, <<"test">>, [{empty, []}]),
    Empty = maps:get(empty, Tables),
    ?assertEqual(0, lc_ets_utils:sum_field(Empty, fun(V) -> V end)),
    ?assertEqual(undefined, lc_ets_utils:avg_field(Empty, fun(V) -> V end)),
    ?assertEqual(undefined, lc_ets_utils:max_field(Empty, fun(V) -> V end)),
    ?assertEqual(undefined, lc_ets_utils:min_field(Empty, fun(V) -> V end)),
    lc_ets_utils:delete_tables(Tables).

%%% ============================================================================
%%% Iteration Tests
%%% ============================================================================

fold_test() ->
    Tables = lc_ets_utils:create_tables(fold_test, <<"test">>, [{items, []}]),
    Table = maps:get(items, Tables),
    lc_ets_utils:insert(Table, a, 1),
    lc_ets_utils:insert(Table, b, 2),
    lc_ets_utils:insert(Table, c, 3),
    Sum = lc_ets_utils:fold(
        fun({_K, V, _T}, Acc) -> Acc + V end,
        0,
        Table
    ),
    ?assertEqual(6, Sum),
    lc_ets_utils:delete_tables(Tables).

foreach_test() ->
    Tables = lc_ets_utils:create_tables(foreach_test, <<"test">>, [{items, []}]),
    Table = maps:get(items, Tables),
    lc_ets_utils:insert(Table, a, 1),
    lc_ets_utils:insert(Table, b, 2),
    lc_ets_utils:insert(Table, c, 3),
    %% Use process dictionary to track calls (side effect test)
    put(foreach_count, 0),
    ok = lc_ets_utils:foreach(
        fun(_Entry) -> put(foreach_count, get(foreach_count) + 1) end,
        Table
    ),
    ?assertEqual(3, get(foreach_count)),
    erase(foreach_count),
    lc_ets_utils:delete_tables(Tables).

filter_test() ->
    Tables = lc_ets_utils:create_tables(filter_test, <<"test">>, [{items, []}]),
    Table = maps:get(items, Tables),
    lc_ets_utils:insert(Table, a, 1),
    lc_ets_utils:insert(Table, b, 2),
    lc_ets_utils:insert(Table, c, 3),
    %% Filter entries where value > 1
    Filtered = lc_ets_utils:filter(
        fun({_K, V, _T}) -> V > 1 end,
        Table
    ),
    ?assertEqual(2, length(Filtered)),
    Values = [V || {_K, V, _T} <- Filtered],
    ?assert(lists:member(2, Values)),
    ?assert(lists:member(3, Values)),
    lc_ets_utils:delete_tables(Tables).
