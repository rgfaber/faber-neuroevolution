%% @doc Zero-config event emitter for Liquid Conglomerate silos.
%%
%% Provides fire-and-forget event emission that automatically uses
%% esdb_lineage_backend when available. If the backend is not installed,
%% events are silently dropped (no-op).
%%
%% Auto-detects if faber_neuroevolution_esdb is available. If the
%% esdb_lineage_backend module exists, events are persisted. Otherwise,
%% emit/3 returns ok immediately (no-op).
%%
%% Event emission never blocks the silo. Backend state is cached in
%% persistent_term for efficiency. The backend uses spawn for async writes
%% and errors are logged, not propagated.
%%
%% Events are routed to streams using the pattern: lc-REALM.SILO_TYPE
%%
%% @author R.G. Lefever
%% @copyright 2024-2026 R.G. Lefever
-module(lc_event_emitter).

%% API
-export([
    emit/3,
    emit/4,
    emit_batch/1,
    is_backend_available/0,
    get_backend_state/0
]).

-define(BACKEND_MODULE, esdb_lineage_backend).
-define(STATE_KEY, {?MODULE, backend_state}).
-define(AVAILABLE_KEY, {?MODULE, backend_available}).

%%% ============================================================================
%%% API Functions
%%% ============================================================================

%% @doc Emit a single LC silo event (fire-and-forget).
%% Uses the default realm from the event payload or "default".
-spec emit(SiloType, EventType, Payload) -> ok when
    SiloType :: atom(),
    EventType :: atom(),
    Payload :: map().
emit(SiloType, EventType, Payload) ->
    Realm = maps:get(realm, Payload, <<"default">>),
    emit(SiloType, EventType, Realm, Payload).

%% @doc Emit a single LC silo event with explicit realm (fire-and-forget).
-spec emit(SiloType, EventType, Realm, Payload) -> ok when
    SiloType :: atom(),
    EventType :: atom(),
    Realm :: binary(),
    Payload :: map().
emit(SiloType, EventType, Realm, Payload) ->
    case is_backend_available() of
        false ->
            ok;
        true ->
            Event = build_event(SiloType, EventType, Realm, Payload),
            State = get_backend_state(),
            ?BACKEND_MODULE:persist_event(Event, State)
    end.

%% @doc Emit a batch of LC silo events (fire-and-forget).
%% Each event map should have keys: silo, event_type, realm (optional), payload.
-spec emit_batch(Events) -> ok when
    Events :: [map()].
emit_batch([]) ->
    ok;
emit_batch(Events) ->
    case is_backend_available() of
        false ->
            ok;
        true ->
            BuiltEvents = [build_event_from_map(E) || E <- Events],
            State = get_backend_state(),
            ?BACKEND_MODULE:persist_batch(BuiltEvents, State)
    end.

%% @doc Check if the event store backend is available.
%%
%% Result is cached after first check.
-spec is_backend_available() -> boolean().
is_backend_available() ->
    case persistent_term:get(?AVAILABLE_KEY, undefined) of
        undefined ->
            Available = check_backend_available(),
            persistent_term:put(?AVAILABLE_KEY, Available),
            Available;
        Available ->
            Available
    end.

%% @doc Get the cached backend state.
%%
%% Initializes the backend on first call if available.
%% Returns undefined if backend is not available.
-spec get_backend_state() -> term() | undefined.
get_backend_state() ->
    case is_backend_available() of
        false ->
            undefined;
        true ->
            case persistent_term:get(?STATE_KEY, undefined) of
                undefined ->
                    init_backend_state();
                State ->
                    State
            end
    end.

%%% ============================================================================
%%% Internal Functions
%%% ============================================================================

%% @private Check if the backend module exists and is loaded.
check_backend_available() ->
    case code:which(?BACKEND_MODULE) of
        non_existing ->
            false;
        _ ->
            %% Module exists, try to ensure it's loaded
            case code:ensure_loaded(?BACKEND_MODULE) of
                {module, ?BACKEND_MODULE} -> true;
                {error, _} -> false
            end
    end.

%% @private Initialize the backend state and cache it.
init_backend_state() ->
    try
        %% Use default store_id - the backend provides sensible defaults
        {ok, State} = ?BACKEND_MODULE:init(#{}),
        persistent_term:put(?STATE_KEY, State),
        State
    catch
        _:Reason ->
            error_logger:warning_msg(
                "[lc_event_emitter] Failed to initialize backend: ~p~n",
                [Reason]
            ),
            %% Mark as unavailable to prevent repeated failures
            persistent_term:put(?AVAILABLE_KEY, false),
            undefined
    end.

%% @private Build an event map from silo type, event type, realm, and payload.
build_event(SiloType, EventType, Realm, Payload) ->
    #{
        event_type => EventType,
        silo => SiloType,
        realm => Realm,
        timestamp => erlang:system_time(millisecond),
        payload => Payload
    }.

%% @private Build an event from a map specification.
build_event_from_map(EventSpec) ->
    SiloType = maps:get(silo, EventSpec),
    EventType = maps:get(event_type, EventSpec),
    Realm = maps:get(realm, EventSpec, <<"default">>),
    Payload = maps:get(payload, EventSpec, #{}),
    build_event(SiloType, EventType, Realm, Payload).
