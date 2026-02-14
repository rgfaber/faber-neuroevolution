%%%-------------------------------------------------------------------
%%% @doc
%%% Behaviour definition for neuroevolution lineage event persistence.
%%%
%%% This behaviour defines the minimal API for lineage tracking.
%%% Implementations MUST be non-blocking to avoid impacting evolution
%%% performance.
%%%
%%% == Callbacks ==
%%%
%%%   init/1          - Initialize backend
%%%   persist_event/2 - Fire-and-forget single event
%%%   persist_batch/2 - Fire-and-forget batch of events
%%%   read_stream/3   - Read events (for recovery/replay)
%%%   subscribe/3     - Subscribe to stream (for projections)
%%%   unsubscribe/3   - Unsubscribe from stream
%%%
%%% == Performance Requirements ==
%%%
%%% Lineage tracking must NEVER block the evolution loop:
%%%
%%%   - persist_event/persist_batch should return immediately
%%%   - Use async I/O internally (spawn, cast, buffering)
%%%   - Acceptable to lose events under extreme load
%%%   - read_stream may block (only used for recovery)
%%%
%%% == Stream Design ==
%%%
%%% Events are organized into streams based on entity type:
%%%
%%%   - individual-{id} : Birth, death, fitness, mutations
%%%   - species-{id}    : Speciation, lineage events
%%%   - population-{id} : Generation, capacity events
%%%   - coalition-{id}  : Coalition lifecycle
%%%
%%% @end
%%%-------------------------------------------------------------------
-module(neuroevolution_lineage_events).

%%% ============================================================================
%%% Type Definitions
%%% ============================================================================

-type event() :: map().
-type stream_id() :: binary().
-type position() :: non_neg_integer().
-type direction() :: forward | backward.

-type read_opts() :: #{
    from => position(),
    limit => pos_integer(),
    direction => direction()
}.

-export_type([
    event/0, stream_id/0, position/0, direction/0, read_opts/0
]).

%%% ============================================================================
%%% Callbacks
%%% ============================================================================

%% Initialize the event store backend.
-callback init(Config :: map()) ->
    {ok, State :: term()} | {error, Reason :: term()}.

%% Persist a single event (fire-and-forget).
%% MUST return immediately. Use async I/O internally.
-callback persist_event(Event :: event(), State :: term()) -> ok.

%% Persist a batch of events (fire-and-forget).
%% MUST return immediately. Use async I/O internally.
-callback persist_batch(Events :: [event()], State :: term()) -> ok.

%% Read events from a stream.
%% May block. Used for recovery/replay, not during evolution.
-callback read_stream(StreamId :: stream_id(), Opts :: read_opts(), State :: term()) ->
    {ok, Events :: [event()]} | {error, Reason :: term()}.

%% Subscribe to new events on a stream.
%% Subscriber receives {lineage_event, StreamId, Event} messages.
-callback subscribe(StreamId :: stream_id(), Pid :: pid(), State :: term()) ->
    ok | {error, Reason :: term()}.

%% Unsubscribe from a stream.
-callback unsubscribe(StreamId :: stream_id(), Pid :: pid(), State :: term()) ->
    ok | {error, Reason :: term()}.
