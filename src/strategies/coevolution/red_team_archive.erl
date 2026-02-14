%%% @doc ETS-based Red Team archive for competitive coevolution.
%%%
%%% == NAMING CONVENTION ==
%%%
%%% Red Team = CHAMPIONS / Hall of Fame / Elite Archive
%%% These are the "good guys" - networks that have proven themselves worthy.
%%% The naming follows the "Red Queen" hypothesis where the Red Queen
%%% (champion) sets the pace that challengers must match.
%%%
%%% == Purpose ==
%%%
%%% Maintains the Red Team (champion archive) - elite networks that the Blue Team
%%% (evolving challengers) must compete against. Creates an arms race dynamic
%%% where both teams must continuously improve.
%%%
%%% == Red Team (Champion) Strategy ==
%%%
%%% - Promotion: High-performing Blue Team members are promoted to Red Team
%%% - Fitness tracking: Champions accumulate fitness based on wins vs challengers
%%% - Age decay: Older champions have reduced selection probability
%%% - Size limit: Archive maintains maximum size, evicting weakest champions
%%% - Weighted sampling: Fitness and recency affect selection probability
%%%
%%% == Immigration ==
%%%
%%% Top Red Team champions can immigrate back to Blue Team, bringing
%%% proven strategies into the evolving challenger population.
%%%
%%% == Mesh-Native Design ==
%%%
%%% This module is designed for eventual mesh distribution:
%%% - Phase 1 (current): Local ETS storage
%%% - Phase 2: CRDT-based conflict resolution for multi-node sync
%%% - Phase 3: Macula PubSub for mesh-wide propagation
%%%
%%% @end
-module(red_team_archive).

-behaviour(gen_server).

%% API
-export([
    start_link/1,
    start_link/2,
    add/2,
    add/3,
    sample/1,
    sample/2,
    size/1,
    stats/1,
    prune/2,
    clear/1,
    stop/1,
    %% Red Team specific
    update_fitness/3,
    get_top/2,
    get_member/2
]).

%% For future mesh sync (Phase 2)
-export([
    export_crdt/1,
    merge_crdt/2
]).

%% gen_server callbacks
-export([
    init/1,
    handle_call/3,
    handle_cast/2,
    handle_info/2,
    terminate/2
]).

-include_lib("kernel/include/logger.hrl").

%% Types
-type archive_id() :: atom() | {atom(), term()}.
-type red_team_member() :: #{
    id := binary(),
    network := map(),
    fitness := float(),
    generation := non_neg_integer(),
    added_at := integer(),
    wins := non_neg_integer(),
    losses := non_neg_integer(),
    origin => blue_team | red_team | imported,
    node_id => binary()
}.

-type archive_config() :: #{
    max_size => pos_integer(),
    min_fitness_percentile => float(),
    age_decay_ms => pos_integer()
}.

-export_type([archive_id/0, red_team_member/0, archive_config/0]).

%% Default configuration
-define(DEFAULT_MAX_SIZE, 30).
-define(DEFAULT_MIN_FITNESS_PERCENTILE, 0.5).
-define(DEFAULT_AGE_DECAY_MS, 120000).  % 2 minute half-life (slower than Blue Team)

%% Records
-record(state, {
    id :: archive_id(),
    table :: ets:tid(),
    max_size :: pos_integer(),
    min_fitness_percentile :: float(),
    age_decay_ms :: pos_integer(),
    count :: non_neg_integer(),
    node_id :: binary(),
    %% CRDT state for mesh sync (Phase 2)
    crdt :: archive_crdt:orset()
}).

%%====================================================================
%% API
%%====================================================================

%% @doc Start the Red Team archive with a given ID.
-spec start_link(archive_id()) -> {ok, pid()} | {error, term()}.
start_link(Id) ->
    start_link(Id, #{}).

%% @doc Start the Red Team archive with ID and configuration.
-spec start_link(archive_id(), archive_config()) -> {ok, pid()} | {error, term()}.
start_link(Id, Config) ->
    gen_server:start_link({local, archive_name(Id)}, ?MODULE, {Id, Config}, []).

%% @doc Add a network to the Red Team.
%% Returns ok if added, rejected if it didn't meet criteria.
-spec add(archive_id(), red_team_member()) -> ok | rejected.
add(Id, Member) ->
    gen_server:call(archive_name(Id), {add, Member}).

%% @doc Add a network with explicit fitness threshold check.
-spec add(archive_id(), red_team_member(), float()) -> ok | rejected.
add(Id, Member, FitnessThreshold) ->
    gen_server:call(archive_name(Id), {add, Member, FitnessThreshold}).

%% @doc Sample a random Red Team member for competition.
%% Uses weighted sampling based on fitness and recency.
-spec sample(archive_id()) -> {ok, red_team_member()} | empty.
sample(Id) ->
    sample(Id, #{}).

%% @doc Sample with options.
%% Options:
%%   strategy - random | weighted (default: weighted)
%%   prefer_recent - boolean (default: true)
-spec sample(archive_id(), map()) -> {ok, red_team_member()} | empty.
sample(Id, Options) ->
    gen_server:call(archive_name(Id), {sample, Options}).

%% @doc Update fitness for a Red Team member after competition.
%% Called when a Red Team member competes against Blue Team.
-spec update_fitness(archive_id(), binary(), float()) -> ok | {error, not_found}.
update_fitness(Id, MemberId, FitnessDelta) ->
    gen_server:call(archive_name(Id), {update_fitness, MemberId, FitnessDelta}).

%% @doc Get top N Red Team members by fitness.
%% Used for immigration to Blue Team.
-spec get_top(archive_id(), pos_integer()) -> {ok, [red_team_member()]} | empty.
get_top(Id, Count) ->
    gen_server:call(archive_name(Id), {get_top, Count}).

%% @doc Get a specific Red Team member by ID.
-spec get_member(archive_id(), binary()) -> {ok, red_team_member()} | {error, not_found}.
get_member(Id, MemberId) ->
    gen_server:call(archive_name(Id), {get_member, MemberId}).

%% @doc Get current Red Team size.
-spec size(archive_id()) -> non_neg_integer().
size(Id) ->
    gen_server:call(archive_name(Id), size).

%% @doc Get Red Team statistics.
-spec stats(archive_id()) -> #{
    count := non_neg_integer(),
    max_size := pos_integer(),
    avg_fitness := float(),
    max_fitness := float(),
    min_fitness := float(),
    generation_range := {non_neg_integer(), non_neg_integer()}
}.
stats(Id) ->
    gen_server:call(archive_name(Id), stats).

%% @doc Prune Red Team to keep only top N by fitness.
-spec prune(archive_id(), pos_integer()) -> ok.
prune(Id, KeepCount) ->
    gen_server:call(archive_name(Id), {prune, KeepCount}).

%% @doc Clear all members from the Red Team.
-spec clear(archive_id()) -> ok.
clear(Id) ->
    gen_server:call(archive_name(Id), clear).

%% @doc Stop the Red Team archive process.
-spec stop(archive_id()) -> ok.
stop(Id) ->
    gen_server:stop(archive_name(Id)).

%% @doc Export Red Team state as CRDT-compatible format.
%% For future mesh sync.
-spec export_crdt(archive_id()) -> binary().
export_crdt(Id) ->
    gen_server:call(archive_name(Id), export_crdt).

%% @doc Merge a remote CRDT state into local Red Team.
%% For future mesh sync.
-spec merge_crdt(archive_id(), binary()) -> ok.
merge_crdt(Id, CrdtData) ->
    gen_server:call(archive_name(Id), {merge_crdt, CrdtData}).

%%====================================================================
%% gen_server callbacks
%%====================================================================

init({Id, Config}) ->
    MaxSize = maps:get(max_size, Config, ?DEFAULT_MAX_SIZE),
    MinPercentile = maps:get(min_fitness_percentile, Config, ?DEFAULT_MIN_FITNESS_PERCENTILE),
    AgeDecay = maps:get(age_decay_ms, Config, ?DEFAULT_AGE_DECAY_MS),

    %% Create ETS table for fast lookups
    Table = ets:new(red_team_table, [set, private]),

    %% Generate unique node ID for CRDT operations
    NodeId = generate_node_id(),

    %% Initialize CRDT state (Phase 2)
    CRDT = archive_crdt:new(NodeId),

    State = #state{
        id = Id,
        table = Table,
        max_size = MaxSize,
        min_fitness_percentile = MinPercentile,
        age_decay_ms = AgeDecay,
        count = 0,
        node_id = NodeId,
        crdt = CRDT
    },

    ?LOG_INFO("[red_team_archive] Started Red Team ~p with max_size=~p", [Id, MaxSize]),
    {ok, State}.

handle_call({add, Member}, _From, State) ->
    {Reply, NewState} = do_add(Member, State),
    {reply, Reply, NewState};

handle_call({add, Member, Threshold}, _From, State) ->
    Fitness = maps:get(fitness, Member, 0.0),
    case Fitness >= Threshold of
        true ->
            {Reply, NewState} = do_add(Member, State),
            {reply, Reply, NewState};
        false ->
            {reply, rejected, State}
    end;

handle_call({sample, Options}, _From, State) ->
    Reply = do_sample(Options, State),
    {reply, Reply, State};

handle_call({update_fitness, MemberId, FitnessDelta}, _From, State) ->
    Reply = do_update_fitness(MemberId, FitnessDelta, State),
    {reply, Reply, State};

handle_call({get_top, Count}, _From, State) ->
    Reply = do_get_top(Count, State),
    {reply, Reply, State};

handle_call({get_member, MemberId}, _From, State) ->
    Reply = do_get_member(MemberId, State),
    {reply, Reply, State};

handle_call(size, _From, #state{count = Count} = State) ->
    {reply, Count, State};

handle_call(stats, _From, State) ->
    Stats = compute_stats(State),
    {reply, Stats, State};

handle_call({prune, KeepCount}, _From, State) ->
    NewState = do_prune(KeepCount, State),
    {reply, ok, NewState};

handle_call(clear, _From, State) ->
    ets:delete_all_objects(State#state.table),
    {reply, ok, State#state{count = 0}};

handle_call(export_crdt, _From, State) ->
    Data = do_export_crdt(State),
    {reply, Data, State};

handle_call({merge_crdt, CrdtData}, _From, State) ->
    NewState = do_merge_crdt(CrdtData, State),
    {reply, ok, NewState};

handle_call(_Request, _From, State) ->
    {reply, {error, unknown_request}, State}.

handle_cast(_Msg, State) ->
    {noreply, State}.

handle_info(_Info, State) ->
    {noreply, State}.

terminate(_Reason, #state{table = Table}) ->
    ets:delete(Table),
    ok.

%%====================================================================
%% Internal functions
%%====================================================================

archive_name(Id) when is_atom(Id) ->
    list_to_atom("red_team_archive_" ++ atom_to_list(Id));
archive_name({Prefix, Suffix}) ->
    list_to_atom("red_team_archive_" ++ atom_to_list(Prefix) ++ "_" ++ term_to_string(Suffix)).

term_to_string(Term) when is_binary(Term) ->
    binary_to_list(Term);
term_to_string(Term) when is_atom(Term) ->
    atom_to_list(Term);
term_to_string(Term) ->
    lists:flatten(io_lib:format("~p", [Term])).

generate_node_id() ->
    <<A:32, B:16, C:16, D:16, E:48>> = crypto:strong_rand_bytes(16),
    iolist_to_binary(io_lib:format("~8.16.0b-~4.16.0b-~4.16.0b-~4.16.0b-~12.16.0b",
                                   [A, B, C, D, E])).

do_add(Member, #state{count = Count, max_size = MaxSize} = State) ->
    %% Prepare entry with metadata
    Entry = prepare_entry(Member, State),

    case Count < MaxSize of
        true ->
            %% Red Team not full - always add
            insert_entry(Entry, State),
            %% NOTE: CRDT tracking disabled to prevent memory leak.
            %% Full network data was being accumulated in the CRDT without pruning.
            %% When Phase 2 mesh sync is implemented, store only genome/weights, not full network.
            {ok, State#state{count = Count + 1}};
        false ->
            %% Red Team full - check if exceeds threshold
            case exceeds_threshold(Entry, State) of
                true ->
                    %% Evict weakest and add
                    evict_weakest(State),
                    insert_entry(Entry, State),
                    {ok, State};
                false ->
                    {rejected, State}
            end
    end.

prepare_entry(Member, #state{node_id = NodeId}) ->
    Id = case maps:get(id, Member, undefined) of
        undefined -> generate_entry_id();
        ExistingId -> ExistingId
    end,
    %% Strip compiled_ref from network to prevent NIF memory leaks.
    %% The compiled_ref holds a Rust ResourceArc that keeps native memory alive.
    %% Networks will be recompiled on-demand when evaluated as opponents.
    Network = maps:get(network, Member),
    StrippedNetwork = network_evaluator:strip_compiled_ref(Network),
    #{
        id => Id,
        network => StrippedNetwork,
        fitness => maps:get(fitness, Member, 0.0),
        generation => maps:get(generation, Member, 0),
        added_at => erlang:monotonic_time(millisecond),
        wins => maps:get(wins, Member, 0),
        losses => maps:get(losses, Member, 0),
        origin => maps:get(origin, Member, blue_team),
        node_id => NodeId
    }.

generate_entry_id() ->
    <<A:64, B:64>> = crypto:strong_rand_bytes(16),
    iolist_to_binary(io_lib:format("~16.16.0b~16.16.0b", [A, B])).

insert_entry(Entry, #state{table = Table}) ->
    Id = maps:get(id, Entry),
    ets:insert(Table, {Id, Entry}).

exceeds_threshold(Entry, #state{table = Table, min_fitness_percentile = Percentile}) ->
    Fitnesses = [F || {_, #{fitness := F}} <- ets:tab2list(Table)],
    case Fitnesses of
        [] ->
            true;
        _ ->
            Sorted = lists:sort(Fitnesses),
            ThresholdIdx = trunc(length(Sorted) * Percentile),
            Threshold = lists:nth(max(1, ThresholdIdx), Sorted),
            maps:get(fitness, Entry, 0.0) >= Threshold
    end.

%% Evict weakest member (lowest fitness)
evict_weakest(#state{table = Table}) ->
    Entries = ets:tab2list(Table),
    case Entries of
        [] ->
            ok;
        _ ->
            {WeakestId, _} = lists:foldl(
                fun({Id, #{fitness := Fitness}}, {MinId, MinFit}) ->
                    case Fitness < MinFit of
                        true -> {Id, Fitness};
                        false -> {MinId, MinFit}
                    end
                end,
                {undefined, infinity},
                Entries
            ),
            case WeakestId of
                undefined -> ok;
                _ -> ets:delete(Table, WeakestId)
            end
    end.

do_sample(_Options, #state{count = 0}) ->
    empty;
do_sample(Options, #state{table = Table, age_decay_ms = AgeDecay}) ->
    Entries = [E || {_, E} <- ets:tab2list(Table)],
    Strategy = maps:get(strategy, Options, weighted),

    Selected = case Strategy of
        random ->
            lists:nth(rand:uniform(length(Entries)), Entries);
        weighted ->
            weighted_sample(Entries, AgeDecay)
    end,

    {ok, Selected}.

weighted_sample(Entries, AgeDecay) ->
    Now = erlang:monotonic_time(millisecond),

    %% Calculate weights based on fitness and recency
    Weights = lists:map(
        fun(Entry) ->
            #{fitness := Fitness, added_at := AddedAt} = Entry,
            AgeMs = Now - AddedAt,
            %% Exponential decay: weight = e^(-age/decay)
            AgeWeight = math:exp(-AgeMs / AgeDecay),
            %% Fitness weight (ensure positive)
            FitnessWeight = max(Fitness, 0.1),
            %% Combined weight
            AgeWeight * FitnessWeight
        end,
        Entries
    ),

    TotalWeight = lists:sum(Weights),
    Target = rand:uniform() * TotalWeight,

    select_by_weight(lists:zip(Entries, Weights), Target, 0.0).

select_by_weight([{Entry, _}], _Target, _Acc) ->
    Entry;
select_by_weight([{Entry, Weight} | Rest], Target, Acc) ->
    NewAcc = Acc + Weight,
    case NewAcc >= Target of
        true -> Entry;
        false -> select_by_weight(Rest, Target, NewAcc)
    end.

do_update_fitness(MemberId, FitnessDelta, #state{table = Table}) ->
    case ets:lookup(Table, MemberId) of
        [{MemberId, Member}] ->
            CurrentFitness = maps:get(fitness, Member, 0.0),
            UpdatedMember = Member#{fitness => CurrentFitness + FitnessDelta},
            ets:insert(Table, {MemberId, UpdatedMember}),
            ok;
        [] ->
            {error, not_found}
    end.

do_get_top(_Count, #state{count = 0}) ->
    empty;
do_get_top(Count, #state{table = Table}) ->
    Entries = [E || {_, E} <- ets:tab2list(Table)],
    %% Sort by fitness descending
    Sorted = lists:reverse(lists:keysort(1, [{maps:get(fitness, E, 0.0), E} || E <- Entries])),
    Top = lists:sublist([E || {_, E} <- Sorted], Count),
    {ok, Top}.

do_get_member(MemberId, #state{table = Table}) ->
    case ets:lookup(Table, MemberId) of
        [{MemberId, Member}] -> {ok, Member};
        [] -> {error, not_found}
    end.

compute_stats(#state{table = Table, count = Count, max_size = MaxSize}) ->
    Entries = [E || {_, E} <- ets:tab2list(Table)],

    Fitnesses = [F || #{fitness := F} <- Entries],
    Generations = [G || #{generation := G} <- Entries],

    #{
        count => Count,
        max_size => MaxSize,
        avg_fitness => safe_avg(Fitnesses),
        max_fitness => safe_max(Fitnesses),
        min_fitness => safe_min(Fitnesses),
        generation_range => {safe_min(Generations), safe_max(Generations)}
    }.

do_prune(KeepCount, #state{table = Table, count = Count} = State) when Count > KeepCount ->
    %% Get all entries sorted by fitness descending
    Entries = ets:tab2list(Table),
    Sorted = lists:reverse(lists:keysort(2, [{Id, maps:get(fitness, E, 0.0)} || {Id, E} <- Entries])),

    %% Delete entries beyond KeepCount
    ToDelete = lists:nthtail(KeepCount, Sorted),
    lists:foreach(fun({Id, _}) -> ets:delete(Table, Id) end, ToDelete),

    State#state{count = KeepCount};
do_prune(_KeepCount, State) ->
    State.

%% CRDT operations (Phase 2)

do_export_crdt(#state{crdt = CRDT}) ->
    %% Export CRDT state using archive_crdt binary format
    archive_crdt:export_binary(CRDT).

do_merge_crdt(CrdtData, #state{table = Table, crdt = CRDT, max_size = MaxSize} = State) ->
    %% Import remote CRDT
    case archive_crdt:import_binary(CrdtData) of
        {ok, RemoteCRDT} ->
            %% Merge CRDTs (conflict-free)
            MergedCRDT = archive_crdt:merge(CRDT, RemoteCRDT),

            %% Get merged values and rebuild ETS
            MergedEntries = archive_crdt:value(MergedCRDT),

            %% Get existing IDs before merge
            ExistingIds = [Id || {Id, _} <- ets:tab2list(Table)],

            %% Add new entries to ETS (respect max size)
            NewCount = lists:foldl(
                fun(Entry, Acc) ->
                    Id = maps:get(id, Entry),
                    case lists:member(Id, ExistingIds) of
                        true ->
                            %% Already have this entry
                            Acc;
                        false ->
                            %% New entry - insert it
                            ets:insert(Table, {Id, Entry}),
                            Acc + 1
                    end
                end,
                0,
                MergedEntries
            ),

            %% Update count
            CurrentCount = ets:info(Table, size),

            %% Prune if over max size
            NewState = State#state{count = CurrentCount, crdt = MergedCRDT},
            FinalState = case CurrentCount > MaxSize of
                true ->
                    do_prune(MaxSize, NewState);
                false ->
                    NewState
            end,

            ?LOG_INFO("[red_team_archive] Merged CRDT data, added ~p new entries (total: ~p)",
                      [NewCount, FinalState#state.count]),
            FinalState;

        {error, Reason} ->
            ?LOG_ERROR("[red_team_archive] Failed to import CRDT: ~p", [Reason]),
            State
    end.

%% Safe math helpers

safe_avg([]) -> 0.0;
safe_avg(List) -> lists:sum(List) / length(List).

safe_max([]) -> 0;
safe_max(List) -> lists:max(List).

safe_min([]) -> 0;
safe_min(List) -> lists:min(List).
