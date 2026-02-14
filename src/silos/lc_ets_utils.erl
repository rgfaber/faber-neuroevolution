%% @doc Shared ETS utilities for Liquid Conglomerate Silos.
%%
%% Provides a consistent pattern for silos that need persistent collections:
%%   Competitive Silo: opponents, matches, elo_ratings
%%   Social Silo: reputations, coalitions, interactions
%%   Cultural Silo: innovations, traditions, memes
%%   Communication Silo: vocabulary, dialects, messages
%%
%% == Usage Pattern ==
%%
%% In init/1, create tables for your silo:
%%
%%   EtsTables = lc_ets_utils:create_tables(competitive, Realm, TableSpecs)
%%
%% In normal operation, use insert/lookup/delete:
%%
%%   lc_ets_utils:insert(Table, Key, Value)
%%   {ok, Data} = lc_ets_utils:lookup(Table, Key)
%%
%% Cleanup when silo terminates:
%%
%%   lc_ets_utils:delete_tables(EtsTables)
%%
%% == Time-Based Operations ==
%%
%% All entries are timestamped automatically for:
%%   Age-based pruning (remove stale entries)
%%   Recency queries (get most recent N entries)
%%   Decay calculations (reduce values over time)
%%
%% @author R.G. Lefever
%% @copyright 2024-2026 R.G. Lefever
-module(lc_ets_utils).

-export([
    %% Table management
    create_tables/3,
    delete_tables/1,
    table_name/3,
    %% CRUD operations
    insert/3,
    insert_with_timestamp/4,
    lookup/2,
    lookup_with_timestamp/2,
    delete/2,
    update/3,
    %% Bulk operations
    all/1,
    all_keys/1,
    count/1,
    %% Time-based operations
    prune_by_age/2,
    prune_by_count/2,
    get_recent/2,
    get_oldest/2,
    %% Aggregation
    sum_field/2,
    avg_field/2,
    max_field/2,
    min_field/2,
    %% Iteration
    fold/3,
    foreach/2,
    filter/2
]).

-type table_name() :: atom().
-type table_ref() :: ets:tid() | atom().
-type table_spec() :: {table_name(), [ets:table_type() | {atom(), term()}]}.
-type timestamped_entry() :: {Key :: term(), Value :: term(), Timestamp :: integer()}.

%%% ============================================================================
%%% Table Management
%%% ============================================================================

%% @doc Create ETS tables for a silo with consistent naming.
%%
%% Tables are named: {SiloType}_{Realm}_{TableName}
%% Example: competitive_realm1_opponents
%%
%% Returns a map of table_name => ets:tid().
-spec create_tables(SiloType :: atom(), Realm :: binary(), TableSpecs :: [table_spec()]) ->
    #{table_name() => table_ref()}.
create_tables(SiloType, Realm, TableSpecs) ->
    lists:foldl(
        fun({TableName, Options}, Acc) ->
            FullName = table_name(SiloType, Realm, TableName),
            %% Merge default options with provided options
            DefaultOpts = [set, public, {keypos, 1}, {read_concurrency, true}],
            MergedOpts = merge_options(DefaultOpts, Options),
            Tid = ets:new(FullName, MergedOpts),
            maps:put(TableName, Tid, Acc)
        end,
        #{},
        TableSpecs
    ).

%% @doc Delete all ETS tables in the map.
-spec delete_tables(Tables :: #{table_name() => table_ref()}) -> ok.
delete_tables(Tables) ->
    maps:foreach(
        fun(_Name, Tid) ->
            catch ets:delete(Tid)
        end,
        Tables
    ),
    ok.

%% @doc Generate consistent table name atom.
-spec table_name(SiloType :: atom(), Realm :: binary(), TableName :: atom()) -> atom().
table_name(SiloType, Realm, TableName) ->
    RealmStr = binary_to_list(Realm),
    list_to_atom(
        atom_to_list(SiloType) ++ "_" ++ RealmStr ++ "_" ++ atom_to_list(TableName)
    ).

%%% ============================================================================
%%% CRUD Operations
%%% ============================================================================

%% @doc Insert a key-value pair with automatic timestamp.
-spec insert(Table :: table_ref(), Key :: term(), Value :: term()) -> true.
insert(Table, Key, Value) ->
    Timestamp = erlang:system_time(millisecond),
    ets:insert(Table, {Key, Value, Timestamp}).

%% @doc Insert with explicit timestamp (for time-travel or replay).
-spec insert_with_timestamp(
    Table :: table_ref(),
    Key :: term(),
    Value :: term(),
    Timestamp :: integer()
) -> true.
insert_with_timestamp(Table, Key, Value, Timestamp) ->
    ets:insert(Table, {Key, Value, Timestamp}).

%% @doc Lookup a value by key. Returns {ok, Value} or not_found.
-spec lookup(Table :: table_ref(), Key :: term()) -> {ok, term()} | not_found.
lookup(Table, Key) ->
    lookup_result(ets:lookup(Table, Key)).

lookup_result([{_Key, Value, _Timestamp}]) -> {ok, Value};
lookup_result([{_Key, Value}]) -> {ok, Value};  %% Non-timestamped
lookup_result([]) -> not_found.

%% @doc Lookup a value with its timestamp.
-spec lookup_with_timestamp(Table :: table_ref(), Key :: term()) ->
    {ok, term(), integer()} | not_found.
lookup_with_timestamp(Table, Key) ->
    lookup_ts_result(ets:lookup(Table, Key)).

lookup_ts_result([{_Key, Value, Timestamp}]) -> {ok, Value, Timestamp};
lookup_ts_result([]) -> not_found.

%% @doc Delete an entry by key.
-spec delete(Table :: table_ref(), Key :: term()) -> true.
delete(Table, Key) ->
    ets:delete(Table, Key).

%% @doc Update a value using a function.
%%
%% UpdateFn(OldValue) -> NewValue
%% If key doesn't exist, UpdateFn(undefined) is called.
-spec update(Table :: table_ref(), Key :: term(), UpdateFn :: fun((term()) -> term())) -> ok.
update(Table, Key, UpdateFn) ->
    OldValue = get_value_or_undefined(lookup(Table, Key)),
    NewValue = UpdateFn(OldValue),
    insert(Table, Key, NewValue),
    ok.

get_value_or_undefined({ok, V}) -> V;
get_value_or_undefined(not_found) -> undefined.

%%% ============================================================================
%%% Bulk Operations
%%% ============================================================================

%% @doc Get all entries as a list of {Key, Value, Timestamp} tuples.
-spec all(Table :: table_ref()) -> [timestamped_entry()].
all(Table) ->
    ets:tab2list(Table).

%% @doc Get all keys in the table.
-spec all_keys(Table :: table_ref()) -> [term()].
all_keys(Table) ->
    ets:foldl(
        fun({Key, _Value, _Timestamp}, Acc) -> [Key | Acc];
           ({Key, _Value}, Acc) -> [Key | Acc]  %% Non-timestamped
        end,
        [],
        Table
    ).

%% @doc Count entries in the table.
-spec count(Table :: table_ref()) -> non_neg_integer().
count(Table) ->
    ets:info(Table, size).

%%% ============================================================================
%%% Time-Based Operations
%%% ============================================================================

%% @doc Remove entries older than MaxAgeMs milliseconds.
%%
%% Returns the number of entries deleted.
-spec prune_by_age(Table :: table_ref(), MaxAgeMs :: pos_integer()) -> non_neg_integer().
prune_by_age(Table, MaxAgeMs) ->
    Now = erlang:system_time(millisecond),
    Cutoff = Now - MaxAgeMs,
    %% Find old entries
    OldKeys = ets:foldl(
        fun({Key, _Value, Timestamp}, Acc) when Timestamp < Cutoff -> [Key | Acc];
           (_, Acc) -> Acc
        end,
        [],
        Table
    ),
    %% Delete them
    lists:foreach(fun(Key) -> ets:delete(Table, Key) end, OldKeys),
    length(OldKeys).

%% @doc Remove oldest entries to keep table at MaxCount size.
%%
%% Returns the number of entries deleted.
-spec prune_by_count(Table :: table_ref(), MaxCount :: pos_integer()) -> non_neg_integer().
prune_by_count(Table, MaxCount) ->
    CurrentCount = ets:info(Table, size),
    prune_if_over_limit(Table, CurrentCount, MaxCount).

prune_if_over_limit(_Table, CurrentCount, MaxCount) when CurrentCount =< MaxCount ->
    0;
prune_if_over_limit(Table, CurrentCount, MaxCount) ->
    Entries = ets:tab2list(Table),
    Sorted = lists:sort(fun sort_by_timestamp_asc/2, Entries),
    ToDelete = lists:sublist(Sorted, CurrentCount - MaxCount),
    delete_entries(Table, ToDelete),
    length(ToDelete).

sort_by_timestamp_asc({_, _, T1}, {_, _, T2}) -> T1 < T2.

delete_entries(Table, Entries) ->
    lists:foreach(fun({Key, _, _}) -> ets:delete(Table, Key) end, Entries).

%% @doc Get the N most recent entries.
-spec get_recent(Table :: table_ref(), N :: pos_integer()) -> [timestamped_entry()].
get_recent(Table, N) ->
    Entries = ets:tab2list(Table),
    Sorted = lists:sort(
        fun({_, _, T1}, {_, _, T2}) -> T1 > T2 end,  %% Most recent first
        Entries
    ),
    lists:sublist(Sorted, N).

%% @doc Get the N oldest entries.
-spec get_oldest(Table :: table_ref(), N :: pos_integer()) -> [timestamped_entry()].
get_oldest(Table, N) ->
    Entries = ets:tab2list(Table),
    Sorted = lists:sort(
        fun({_, _, T1}, {_, _, T2}) -> T1 < T2 end,  %% Oldest first
        Entries
    ),
    lists:sublist(Sorted, N).

%%% ============================================================================
%%% Aggregation Functions
%%% ============================================================================

%% @doc Sum a numeric field from all values.
%%
%% FieldFn extracts the numeric field from each value.
-spec sum_field(Table :: table_ref(), FieldFn :: fun((term()) -> number())) -> number().
sum_field(Table, FieldFn) ->
    ets:foldl(
        fun({_Key, Value, _Timestamp}, Acc) -> Acc + FieldFn(Value);
           ({_Key, Value}, Acc) -> Acc + FieldFn(Value)
        end,
        0,
        Table
    ).

%% @doc Average a numeric field from all values.
-spec avg_field(Table :: table_ref(), FieldFn :: fun((term()) -> number())) ->
    number() | undefined.
avg_field(Table, FieldFn) ->
    {Sum, Count} = ets:foldl(
        fun(Entry, {S, C}) -> {S + FieldFn(extract_value(Entry)), C + 1} end,
        {0, 0},
        Table
    ),
    compute_avg(Sum, Count).

compute_avg(_Sum, 0) -> undefined;
compute_avg(Sum, Count) -> Sum / Count.

%% @doc Find maximum value of a numeric field.
-spec max_field(Table :: table_ref(), FieldFn :: fun((term()) -> number())) ->
    number() | undefined.
max_field(Table, FieldFn) ->
    fold_field(Table, FieldFn, fun erlang:max/2).

%% @doc Find minimum value of a numeric field.
-spec min_field(Table :: table_ref(), FieldFn :: fun((term()) -> number())) ->
    number() | undefined.
min_field(Table, FieldFn) ->
    fold_field(Table, FieldFn, fun erlang:min/2).

%% @private Fold over table applying FieldFn and combining with CombineFn.
fold_field(Table, FieldFn, CombineFn) ->
    fold_field_if_nonempty(ets:info(Table, size), Table, FieldFn, CombineFn).

fold_field_if_nonempty(0, _Table, _FieldFn, _CombineFn) ->
    undefined;
fold_field_if_nonempty(_Size, Table, FieldFn, CombineFn) ->
    ets:foldl(
        fun(Entry, undefined) -> FieldFn(extract_value(Entry));
           (Entry, Acc) -> CombineFn(Acc, FieldFn(extract_value(Entry)))
        end,
        undefined,
        Table
    ).

%% @private Extract value from entry (supports timestamped and non-timestamped).
extract_value({_Key, Value, _Timestamp}) -> Value;
extract_value({_Key, Value}) -> Value.

%%% ============================================================================
%%% Iteration Functions
%%% ============================================================================

%% @doc Fold over all entries.
-spec fold(
    Fun :: fun((timestamped_entry(), Acc) -> Acc),
    InitAcc :: Acc,
    Table :: table_ref()
) -> Acc when Acc :: term().
fold(Fun, InitAcc, Table) ->
    ets:foldl(Fun, InitAcc, Table).

%% @doc Execute a function for each entry (side effects).
-spec foreach(Fun :: fun((timestamped_entry()) -> any()), Table :: table_ref()) -> ok.
foreach(Fun, Table) ->
    ets:foldl(
        fun(Entry, ok) -> Fun(Entry), ok end,
        ok,
        Table
    ),
    ok.

%% @doc Filter entries matching a predicate.
-spec filter(
    Pred :: fun((timestamped_entry()) -> boolean()),
    Table :: table_ref()
) -> [timestamped_entry()].
filter(Pred, Table) ->
    ets:foldl(
        fun(Entry, Acc) -> maybe_include(Pred(Entry), Entry, Acc) end,
        [],
        Table
    ).

maybe_include(true, Entry, Acc) -> [Entry | Acc];
maybe_include(false, _Entry, Acc) -> Acc.

%%% ============================================================================
%%% Internal Functions
%%% ============================================================================

%% @private Merge ETS table options, with later options taking precedence.
merge_options(DefaultOpts, OverrideOpts) ->
    %% Convert to maps for easy merging, handling both atoms and {Key,Val} tuples
    DefaultMap = opts_to_map(DefaultOpts),
    OverrideMap = opts_to_map(OverrideOpts),
    MergedMap = maps:merge(DefaultMap, OverrideMap),
    %% Convert back to list format
    maps:fold(fun opts_from_map/3, [], MergedMap).

%% @private Convert map entries back to ETS option format.
%% Bare atom options (set, public, etc.) are stored as atom => true.
opts_from_map(Key, true, Acc) ->
    maybe_bare_atom_opt(Key, Acc);
opts_from_map(Key, Val, Acc) ->
    [{Key, Val} | Acc].

%% @private Bare atom options become just the atom, others become tuples.
maybe_bare_atom_opt(set, Acc) -> [set | Acc];
maybe_bare_atom_opt(ordered_set, Acc) -> [ordered_set | Acc];
maybe_bare_atom_opt(bag, Acc) -> [bag | Acc];
maybe_bare_atom_opt(duplicate_bag, Acc) -> [duplicate_bag | Acc];
maybe_bare_atom_opt(public, Acc) -> [public | Acc];
maybe_bare_atom_opt(protected, Acc) -> [protected | Acc];
maybe_bare_atom_opt(private, Acc) -> [private | Acc];
maybe_bare_atom_opt(named_table, Acc) -> [named_table | Acc];
maybe_bare_atom_opt(Key, Acc) -> [{Key, true} | Acc].

opts_to_map(Opts) ->
    lists:foldl(fun opt_to_map_entry/2, #{}, Opts).

opt_to_map_entry(set, Acc) -> maps:put(set, true, Acc);
opt_to_map_entry(ordered_set, Acc) -> maps:put(ordered_set, true, Acc);
opt_to_map_entry(bag, Acc) -> maps:put(bag, true, Acc);
opt_to_map_entry(duplicate_bag, Acc) -> maps:put(duplicate_bag, true, Acc);
opt_to_map_entry(public, Acc) -> maps:put(public, true, Acc);
opt_to_map_entry(protected, Acc) -> maps:put(protected, true, Acc);
opt_to_map_entry(private, Acc) -> maps:put(private, true, Acc);
opt_to_map_entry(named_table, Acc) -> maps:put(named_table, true, Acc);
opt_to_map_entry({Key, Val}, Acc) -> maps:put(Key, Val, Acc);
opt_to_map_entry(_, Acc) -> Acc.
