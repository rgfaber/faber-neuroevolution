%% @doc OR-Set CRDT for opponent archive synchronization.
%%
%% Implements an Observed-Remove Set (OR-Set) CRDT for conflict-free
%% archive merging across distributed nodes. Each champion entry includes
%% a unique tag allowing concurrent adds and removes to be resolved
%% automatically.
%%
%% CRDT Operations:
%% - add/3: Add element with unique tag (actor + counter)
%% - remove/2: Remove specific element (marks tag as tombstone)
%% - merge/2: Combine two OR-Sets (union of all non-tombstoned entries)
%% - value/1: Get current set contents
%%
%% Binary Format (for network serialization):
%% - Version byte (1)
%% - Entry count (4 bytes, big-endian)
%% - For each entry:
%%   - Tag length (2 bytes) + Tag binary
%%   - Entry term_to_binary (4 bytes length + data)
%%
%% @reference https://hal.inria.fr/inria-00555588/document
-module(archive_crdt).

-export([
    new/0,
    new/1,
    add/3,
    remove/2,
    merge/2,
    value/1,
    size/1,
    export_binary/1,
    import_binary/1,
    is_crdt/1
]).

-export_type([orset/0, actor_id/0, entry/0]).

-define(CRDT_VERSION, 1).

%% OR-Set entry with unique tag
-record(entry, {
    tag :: binary(),          % Unique identifier (actor_id + counter)
    value :: term(),          % The actual champion data
    tombstone = false :: boolean()  % Removed entries become tombstones
}).

%% OR-Set state
-record(orset, {
    actor_id :: binary(),     % This node's unique ID
    counter = 0 :: non_neg_integer(),  % Monotonic counter for tags
    entries = #{} :: #{binary() => #entry{}}  % Tag -> Entry map
}).

-type orset() :: #orset{}.
-type actor_id() :: binary().
-type entry() :: #entry{}.

%%====================================================================
%% API
%%====================================================================

%% @doc Create new empty OR-Set with random actor ID.
-spec new() -> orset().
new() ->
    new(generate_actor_id()).

%% @doc Create new empty OR-Set with specific actor ID.
-spec new(actor_id()) -> orset().
new(ActorId) when is_binary(ActorId) ->
    #orset{actor_id = ActorId, counter = 0, entries = #{}}.

%% @doc Add element to OR-Set.
%% Returns updated OR-Set with new entry tagged uniquely.
-spec add(term(), orset(), actor_id()) -> orset().
add(Value, #orset{counter = Counter, entries = Entries} = ORSet, ActorId) ->
    NewCounter = Counter + 1,
    Tag = make_tag(ActorId, NewCounter),
    Entry = #entry{tag = Tag, value = Value, tombstone = false},
    ORSet#orset{
        counter = NewCounter,
        entries = Entries#{Tag => Entry}
    }.

%% @doc Remove element from OR-Set by value.
%% Marks all entries with matching value as tombstones.
-spec remove(term(), orset()) -> orset().
remove(Value, #orset{entries = Entries} = ORSet) ->
    UpdatedEntries = maps:map(
        fun(_Tag, #entry{value = V, tombstone = false} = E) when V =:= Value ->
            E#entry{tombstone = true};
           (_Tag, E) ->
            E
        end,
        Entries
    ),
    ORSet#orset{entries = UpdatedEntries}.

%% @doc Merge two OR-Sets.
%% - All unique tags are preserved
%% - Tombstones win: if any version has tombstone=true, result is tombstone
-spec merge(orset(), orset()) -> orset().
merge(#orset{entries = E1} = ORSet1, #orset{entries = E2}) ->
    %% Merge entries: tombstones win
    MergedEntries = maps:fold(
        fun(Tag, Entry2, Acc) ->
            case maps:find(Tag, Acc) of
                {ok, Entry1} ->
                    %% Tag exists in both - tombstone wins
                    Tombstone = Entry1#entry.tombstone orelse Entry2#entry.tombstone,
                    Acc#{Tag => Entry1#entry{tombstone = Tombstone}};
                error ->
                    %% New tag from E2
                    Acc#{Tag => Entry2}
            end
        end,
        E1,
        E2
    ),
    %% Use higher counter
    ORSet1#orset{entries = MergedEntries}.

%% @doc Get current set values (excludes tombstones).
-spec value(orset()) -> [term()].
value(#orset{entries = Entries}) ->
    [E#entry.value || #entry{tombstone = false} = E <- maps:values(Entries)].

%% @doc Get size (live entries only).
-spec size(orset()) -> non_neg_integer().
size(ORSet) ->
    length(value(ORSet)).

%% @doc Export OR-Set to binary for network transmission.
-spec export_binary(orset()) -> binary().
export_binary(#orset{actor_id = ActorId, counter = Counter, entries = Entries}) ->
    %% Only export non-tombstone entries (tombstones are local state)
    LiveEntries = maps:filter(
        fun(_Tag, #entry{tombstone = T}) -> not T end,
        Entries
    ),
    EntryList = maps:to_list(LiveEntries),
    EntryCount = length(EntryList),

    %% Serialize entries
    EntriesBin = lists:foldl(
        fun({Tag, #entry{value = Value}}, Acc) ->
            TagLen = byte_size(Tag),
            ValueBin = term_to_binary(Value),
            ValueLen = byte_size(ValueBin),
            <<Acc/binary,
              TagLen:16/big, Tag/binary,
              ValueLen:32/big, ValueBin/binary>>
        end,
        <<>>,
        EntryList
    ),

    ActorLen = byte_size(ActorId),
    <<?CRDT_VERSION:8,
      ActorLen:16/big, ActorId/binary,
      Counter:64/big,
      EntryCount:32/big,
      EntriesBin/binary>>.

%% @doc Import OR-Set from binary.
-spec import_binary(binary()) -> {ok, orset()} | {error, term()}.
import_binary(<<?CRDT_VERSION:8,
                ActorLen:16/big, ActorId:ActorLen/binary,
                Counter:64/big,
                EntryCount:32/big,
                EntriesBin/binary>>) ->
    case parse_entries(EntriesBin, EntryCount, #{}) of
        {ok, Entries} ->
            {ok, #orset{actor_id = ActorId, counter = Counter, entries = Entries}};
        {error, _} = Error ->
            Error
    end;
import_binary(<<Version:8, _/binary>>) when Version =/= ?CRDT_VERSION ->
    {error, {unsupported_version, Version}};
import_binary(_) ->
    {error, invalid_format}.

%% @doc Check if term is an OR-Set.
-spec is_crdt(term()) -> boolean().
is_crdt(#orset{}) -> true;
is_crdt(_) -> false.

%%====================================================================
%% Internal Functions
%%====================================================================

%% @private Generate random actor ID.
generate_actor_id() ->
    <<A:32, B:16, C:16, D:16, E:48>> = crypto:strong_rand_bytes(16),
    iolist_to_binary(io_lib:format(
        "~8.16.0b-~4.16.0b-~4.16.0b-~4.16.0b-~12.16.0b",
        [A, B, C, D, E]
    )).

%% @private Create unique tag from actor ID and counter.
make_tag(ActorId, Counter) ->
    <<ActorId/binary, $:, (integer_to_binary(Counter))/binary>>.

%% @private Parse binary entries into map.
parse_entries(<<>>, 0, Acc) ->
    {ok, Acc};
parse_entries(<<TagLen:16/big, Tag:TagLen/binary,
                ValueLen:32/big, ValueBin:ValueLen/binary,
                Rest/binary>>, Count, Acc) when Count > 0 ->
    Value = binary_to_term(ValueBin),
    Entry = #entry{tag = Tag, value = Value, tombstone = false},
    parse_entries(Rest, Count - 1, Acc#{Tag => Entry});
parse_entries(_, Count, _) when Count > 0 ->
    {error, truncated_data};
parse_entries(Extra, 0, _) when byte_size(Extra) > 0 ->
    {error, extra_data}.
