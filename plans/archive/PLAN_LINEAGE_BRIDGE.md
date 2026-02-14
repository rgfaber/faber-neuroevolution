# PLAN_LINEAGE_BRIDGE.md

**Status:** Planning
**Created:** 2025-12-23
**Last Updated:** 2025-12-23
**Related:** PLAN_BEHAVIORAL_EVENTS.md, EVENT_DRIVEN_ARCHITECTURE.md

---

## Overview

This document specifies the bridge library architecture for persisting neuroevolution events to erl-esdb and erl-evoq. The bridge enables permanent lineage tracking, replay capability, and event-driven aggregates while keeping the core libraries (faber-tweann, faber-neuroevolution) independent of persistence concerns.

### Design Principles

1. **Separation of Concerns**: Core libraries define behaviours, bridge implements persistence
2. **Non-Blocking**: Event persistence must not slow evolution
3. **Async Batching**: Collect events, persist in batches for efficiency
4. **Dual Event Types**: Notification (pubsub) vs Persistence (stored)
5. **Producer Ownership**: Event schema owned by emitting domain

---

## Architecture

### Library Relationship

```
┌─────────────────────────────────────────────────────────────────────┐
│                     APPLICATION LAYER                                │
│                                                                      │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │                  faber-neuroevolution                       │    │
│  │                                                              │    │
│  │  ┌─────────────────┐  ┌─────────────────┐                   │    │
│  │  │ neuroevolution_ │  │ neuroevolution_ │                   │    │
│  │  │ events          │  │ lineage_events  │ ◄── BEHAVIOUR     │    │
│  │  │ (pubsub)        │  │ (behaviour)     │                   │    │
│  │  └────────┬────────┘  └────────┬────────┘                   │    │
│  │           │                    │                            │    │
│  └───────────┼────────────────────┼────────────────────────────┘    │
│              │                    │                                  │
│              │    IMPLEMENTS      │                                  │
│              │         ┌──────────┘                                  │
│              │         │                                             │
│  ┌───────────▼─────────▼───────────────────────────────────────┐    │
│  │              faber-neuroevolution-evoq                      │    │
│  │                   (BRIDGE LIBRARY)                           │    │
│  │                                                              │    │
│  │  ┌─────────────────┐  ┌─────────────────┐                   │    │
│  │  │ neuroevolution_ │  │ event_batcher   │                   │    │
│  │  │ lineage_esdb    │  │ (async batch)   │                   │    │
│  │  └────────┬────────┘  └────────┬────────┘                   │    │
│  │           │                    │                            │    │
│  └───────────┼────────────────────┼────────────────────────────┘    │
│              │                    │                                  │
│              └─────────┬──────────┘                                  │
│                        │                                             │
│  ┌─────────────────────▼───────────────────────────────────────┐    │
│  │                   PERSISTENCE LAYER                          │    │
│  │                                                              │    │
│  │  ┌─────────────────┐  ┌─────────────────┐                   │    │
│  │  │    erl-esdb     │  │   erl-evoq      │                   │    │
│  │  │  (event store)  │  │ (CQRS/ES agg)   │                   │    │
│  │  └─────────────────┘  └─────────────────┘                   │    │
│  │                                                              │    │
│  └──────────────────────────────────────────────────────────────┘    │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### Event Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│                         EVENT FLOW                                   │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│   Evolution Process                                                  │
│        │                                                             │
│        │ emit(offspring_born, Event)                                │
│        ▼                                                             │
│   ┌────────────────┐                                                │
│   │neuroevolution_ │                                                │
│   │events          │                                                │
│   └───────┬────────┘                                                │
│           │                                                          │
│           ├──────────────────────┬───────────────────────┐          │
│           │                      │                       │          │
│           ▼                      ▼                       ▼          │
│   ┌──────────────┐      ┌──────────────┐       ┌──────────────┐    │
│   │  pg (local   │      │  Lineage     │       │  Other       │    │
│   │  subscribers)│      │  Backend     │       │  Backends    │    │
│   └──────────────┘      └──────┬───────┘       └──────────────┘    │
│                                │                                    │
│                                │ (if configured)                    │
│                                ▼                                    │
│                        ┌──────────────┐                            │
│                        │ Event Batcher│                            │
│                        └──────┬───────┘                            │
│                                │                                    │
│                                │ flush (async, batch)              │
│                                ▼                                    │
│                        ┌──────────────┐                            │
│                        │  erl-esdb    │                            │
│                        │  streams     │                            │
│                        └──────────────┘                            │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Behaviour Definition

### neuroevolution_lineage_events

Defined in faber-neuroevolution, implemented by bridge libraries.

**CQRS Architecture**: Required callbacks handle event store operations. Optional callbacks provide query capabilities - implementations should use internal projections.

```erlang
-module(neuroevolution_lineage_events).

%% === Required Callbacks (Event Store) ===

-callback init(Config :: map()) ->
    {ok, State :: term()} | {error, Reason :: term()}.

-callback persist_event(Event :: event(), State :: term()) ->
    ok | {error, Reason :: term()}.

-callback persist_batch(Events :: [event()], State :: term()) ->
    ok | {error, Reason :: term()}.

-callback read_stream(StreamId :: stream_id(), Opts :: read_opts(), State :: term()) ->
    {ok, Events :: [event()]} | {error, Reason :: term()}.

-callback subscribe(StreamId :: stream_id(), Pid :: pid(), State :: term()) ->
    ok | {error, Reason :: term()}.

-callback unsubscribe(StreamId :: stream_id(), Pid :: pid(), State :: term()) ->
    ok | {error, Reason :: term()}.

%% === Optional Callbacks (Queries via Projections) ===

-optional_callbacks([
    get_breeding_tree/3,
    get_fitness_trajectory/2,
    get_mutation_history/2,
    get_knowledge_transfers/2,
    get_by_causation/2
]).

-callback get_breeding_tree(IndividualId :: binary(), Depth :: pos_integer(),
                            State :: term()) ->
    {ok, Tree :: map()} | {error, Reason :: term()}.

-callback get_fitness_trajectory(IndividualId :: binary(), State :: term()) ->
    {ok, [{integer(), float()}]} | {error, Reason :: term()}.

-callback get_mutation_history(IndividualId :: binary(), State :: term()) ->
    {ok, [event()]} | {error, Reason :: term()}.

-callback get_knowledge_transfers(IndividualId :: binary(), State :: term()) ->
    {ok, [event()]} | {error, Reason :: term()}.

-callback get_by_causation(CausationId :: binary(), State :: term()) ->
    {ok, [event()]} | {error, Reason :: term()}.
```

### Callback Responsibilities

| Callback Type | Responsibility | Implementation |
|---------------|----------------|----------------|
| Required | Event store ops | Direct erl-esdb calls |
| Optional | Query ops | Query internal projections |

**Key Insight**: The behaviour defines the complete API consumers need. The bridge library implements optional callbacks by delegating to projections that build read models.

---

## Bridge Library: faber-neuroevolution-evoq

### Directory Structure

```
faber-neuroevolution-evoq/
├── src/
│   ├── faber_neuroevolution_evoq.app.src
│   ├── faber_neuroevolution_evoq_app.erl
│   ├── faber_neuroevolution_evoq_sup.erl
│   │
│   ├── neuroevolution_lineage_esdb.erl      # Behaviour implementation
│   ├── neuroevolution_event_batcher.erl     # Async batching
│   ├── neuroevolution_stream_router.erl     # Stream routing logic
│   │
│   ├── aggregates/
│   │   ├── lineage_aggregate.erl            # erl-evoq aggregate
│   │   ├── individual_aggregate.erl         # Individual lifecycle
│   │   └── species_aggregate.erl            # Species lifecycle
│   │
│   └── projections/
│       ├── lineage_tree_projection.erl      # Builds lineage trees
│       ├── fitness_trajectory_projection.erl
│       └── mutation_history_projection.erl
│
├── include/
│   └── neuroevolution_evoq.hrl
│
├── test/
│   ├── neuroevolution_lineage_esdb_tests.erl
│   └── event_batcher_tests.erl
│
└── rebar.config
```

### Dependencies

```erlang
%% rebar.config
{deps, [
    {erl_esdb, "~> 0.4.4"},
    {erl_esdb_gater, "~> 0.6.4"},
    {erl_evoq, "~> 0.3.0"},
    {erl_evoq_esdb, "~> 0.3.1"},
    {faber_neuroevolution, "~> 0.17.0"}
]}.
```

---

## Behaviour Implementation

### Module: `neuroevolution_lineage_esdb.erl`

```erlang
-module(neuroevolution_lineage_esdb).
-behaviour(neuroevolution_lineage_events).

%% Behaviour callbacks
-export([
    init/1,
    persist_event/2,
    persist_batch/2,
    get_lineage/2,
    get_species_history/2,
    get_population_timeline/3,
    replay_from/3,
    get_breeding_tree/2,
    get_mutation_history/2,
    get_fitness_trajectory/2,
    get_knowledge_transfers/2,
    get_by_causation/2
]).

-record(state, {
    store :: atom(),
    batcher_pid :: pid(),
    stream_router :: fun((event()) -> binary()),
    config :: map()
}).

%%====================================================================
%% Behaviour Implementation
%%====================================================================

init(Config) ->
    Store = maps:get(store_name, Config, neuroevolution_store),

    %% Start event batcher
    BatcherConfig = #{
        store => Store,
        batch_size => maps:get(batch_size, Config, 100),
        flush_interval_ms => maps:get(flush_interval_ms, Config, 1000)
    },
    {ok, BatcherPid} = neuroevolution_event_batcher:start_link(BatcherConfig),

    %% Initialize stream router
    StreamRouter = maps:get(stream_router, Config, fun default_stream_router/1),

    {ok, #state{
        store = Store,
        batcher_pid = BatcherPid,
        stream_router = StreamRouter,
        config = Config
    }}.

persist_event(Event, #state{batcher_pid = Batcher} = State) ->
    %% Add to async batcher
    ok = neuroevolution_event_batcher:add_event(Batcher, Event),
    {ok, State}.

persist_batch(Events, #state{store = Store, stream_router = Router} = State) ->
    %% Group events by stream
    GroupedEvents = group_by_stream(Events, Router),

    %% Write each stream's events
    Results = maps:fold(fun(StreamId, StreamEvents, Acc) ->
        case erl_esdb_streams:append(Store, StreamId, any, StreamEvents) of
            {ok, _} -> Acc;
            {error, Reason} -> [{error, StreamId, Reason} | Acc]
        end
    end, [], GroupedEvents),

    case Results of
        [] -> {ok, State};
        Errors -> {error, Errors}
    end.

get_lineage(IndividualId, #state{store = Store} = State) ->
    StreamId = <<"individual-", IndividualId/binary>>,
    case erl_esdb_streams:read(Store, StreamId, #{direction => forward}) of
        {ok, Events} ->
            %% Also fetch parent events recursively
            FullLineage = build_lineage_chain(Events, Store),
            {ok, FullLineage};
        {error, stream_not_found} ->
            {ok, []};
        {error, Reason} ->
            {error, Reason}
    end.

get_species_history(SpeciesId, #state{store = Store} = State) ->
    StreamId = <<"species-", SpeciesId/binary>>,
    case erl_esdb_streams:read(Store, StreamId, #{direction => forward}) of
        {ok, Events} -> {ok, Events};
        {error, stream_not_found} -> {ok, []};
        {error, Reason} -> {error, Reason}
    end.

get_population_timeline(PopulationId, Opts, #state{store = Store} = State) ->
    StreamId = <<"population-", PopulationId/binary>>,
    ReadOpts = #{
        direction => forward,
        from => maps:get(from, Opts, 0),
        limit => maps:get(limit, Opts, 1000)
    },
    case erl_esdb_streams:read(Store, StreamId, ReadOpts) of
        {ok, Events} -> {ok, Events};
        {error, stream_not_found} -> {ok, []};
        {error, Reason} -> {error, Reason}
    end.

replay_from(StreamId, Position, #state{store = Store} = State) ->
    case erl_esdb_streams:read(Store, StreamId, #{
        direction => forward,
        from => Position
    }) of
        {ok, Events} -> {ok, Events};
        {error, Reason} -> {error, Reason}
    end.

%%====================================================================
%% Optional Callbacks
%%====================================================================

get_breeding_tree(IndividualId, State) ->
    %% Build full family tree from lineage events
    {ok, Lineage} = get_lineage(IndividualId, State),

    Tree = build_breeding_tree(Lineage, IndividualId),
    {ok, Tree}.

get_mutation_history(IndividualId, #state{store = Store} = State) ->
    StreamId = <<"individual-", IndividualId/binary>>,
    case erl_esdb_streams:read(Store, StreamId, #{direction => forward}) of
        {ok, Events} ->
            MutationEvents = [E || E <- Events,
                                   is_mutation_event(maps:get(event_type, E))],
            {ok, MutationEvents};
        {error, Reason} ->
            {error, Reason}
    end.

get_fitness_trajectory(IndividualId, #state{store = Store} = State) ->
    StreamId = <<"individual-", IndividualId/binary>>,
    case erl_esdb_streams:read(Store, StreamId, #{direction => forward}) of
        {ok, Events} ->
            FitnessEvents = [E || E <- Events,
                                  maps:get(event_type, E) == fitness_evaluated],
            Trajectory = [{maps:get(evaluated_at, E), maps:get(fitness, E)}
                         || E <- FitnessEvents],
            {ok, Trajectory};
        {error, Reason} ->
            {error, Reason}
    end.

get_knowledge_transfers(IndividualId, #state{store = Store} = State) ->
    StreamId = <<"individual-", IndividualId/binary>>,
    case erl_esdb_streams:read(Store, StreamId, #{direction => forward}) of
        {ok, Events} ->
            TransferEvents = [E || E <- Events,
                                   is_transfer_event(maps:get(event_type, E))],
            {ok, TransferEvents};
        {error, Reason} ->
            {error, Reason}
    end.

get_by_causation(CausationId, #state{store = Store} = State) ->
    %% Query by causation requires scanning or index
    %% Use category stream with filter
    case erl_esdb_streams:read(Store, <<"$ce-neuroevolution">>, #{
        direction => forward,
        filter => fun(E) ->
            maps:get(causation_id, maps:get(metadata, E, #{}), undefined) == CausationId
        end
    }) of
        {ok, Events} -> {ok, Events};
        {error, Reason} -> {error, Reason}
    end.

%%====================================================================
%% Internal Functions
%%====================================================================

default_stream_router(Event) ->
    EventType = maps:get(event_type, Event),
    route_by_event_type(EventType, Event).

route_by_event_type(EventType, Event) when
        EventType == offspring_born;
        EventType == individual_culled;
        EventType == lifespan_expired;
        EventType == fitness_evaluated;
        EventType == mutation_applied ->
    %% Individual lifecycle events
    IndividualId = maps:get(individual_id, Event),
    <<"individual-", IndividualId/binary>>;

route_by_event_type(EventType, Event) when
        EventType == lineage_diverged;
        EventType == species_emerged;
        EventType == lineage_ended ->
    %% Species events
    SpeciesId = case EventType of
        lineage_diverged -> maps:get(new_species_id, Event);
        _ -> maps:get(species_id, Event)
    end,
    <<"species-", SpeciesId/binary>>;

route_by_event_type(EventType, Event) when
        EventType == generation_completed;
        EventType == population_initialized ->
    %% Population events
    PopulationId = maps:get(population_id, Event),
    <<"population-", PopulationId/binary>>;

route_by_event_type(EventType, Event) when
        EventType == knowledge_transferred;
        EventType == mentor_assigned ->
    %% Knowledge transfer - route to student
    StudentId = maps:get(student_id, Event),
    <<"individual-", StudentId/binary>>;

route_by_event_type(_EventType, Event) ->
    %% Default: route to population stream
    PopulationId = maps:get(population_id, Event, <<"default">>),
    <<"population-", PopulationId/binary>>.

group_by_stream(Events, Router) ->
    lists:foldl(fun(Event, Acc) ->
        StreamId = Router(Event),
        ExistingEvents = maps:get(StreamId, Acc, []),
        maps:put(StreamId, [Event | ExistingEvents], Acc)
    end, #{}, Events).

is_mutation_event(mutation_applied) -> true;
is_mutation_event(neuron_added) -> true;
is_mutation_event(neuron_removed) -> true;
is_mutation_event(connection_added) -> true;
is_mutation_event(connection_removed) -> true;
is_mutation_event(weight_perturbed) -> true;
is_mutation_event(_) -> false.

is_transfer_event(knowledge_transferred) -> true;
is_transfer_event(skill_imitated) -> true;
is_transfer_event(behavior_cloned) -> true;
is_transfer_event(weights_grafted) -> true;
is_transfer_event(structure_seeded) -> true;
is_transfer_event(_) -> false.
```

---

## Event Batcher

### Module: `neuroevolution_event_batcher.erl`

```erlang
-module(neuroevolution_event_batcher).
-behaviour(gen_server).

%% API
-export([
    start_link/1,
    add_event/2,
    flush/1,
    get_stats/1
]).

%% gen_server callbacks
-export([init/1, handle_call/3, handle_cast/2, handle_info/2, terminate/2]).

-record(state, {
    store :: atom(),
    batch_size :: non_neg_integer(),
    flush_interval_ms :: non_neg_integer(),
    buffer :: [map()],
    buffer_count :: non_neg_integer(),
    flush_timer :: reference() | undefined,
    stats :: #{
        events_received => non_neg_integer(),
        batches_flushed => non_neg_integer(),
        events_persisted => non_neg_integer(),
        errors => non_neg_integer()
    }
}).

%%====================================================================
%% API
%%====================================================================

start_link(Config) ->
    gen_server:start_link(?MODULE, Config, []).

add_event(Pid, Event) ->
    gen_server:cast(Pid, {add_event, Event}).

flush(Pid) ->
    gen_server:call(Pid, flush).

get_stats(Pid) ->
    gen_server:call(Pid, get_stats).

%%====================================================================
%% gen_server Callbacks
%%====================================================================

init(Config) ->
    Store = maps:get(store, Config),
    BatchSize = maps:get(batch_size, Config, 100),
    FlushInterval = maps:get(flush_interval_ms, Config, 1000),

    %% Start flush timer
    Timer = erlang:send_after(FlushInterval, self(), flush_timer),

    {ok, #state{
        store = Store,
        batch_size = BatchSize,
        flush_interval_ms = FlushInterval,
        buffer = [],
        buffer_count = 0,
        flush_timer = Timer,
        stats = #{
            events_received => 0,
            batches_flushed => 0,
            events_persisted => 0,
            errors => 0
        }
    }}.

handle_cast({add_event, Event}, #state{} = State) ->
    NewBuffer = [Event | State#state.buffer],
    NewCount = State#state.buffer_count + 1,
    NewStats = maps:update_with(events_received, fun(V) -> V + 1 end,
                                State#state.stats),

    NewState = State#state{
        buffer = NewBuffer,
        buffer_count = NewCount,
        stats = NewStats
    },

    %% Check if we should flush
    case NewCount >= State#state.batch_size of
        true ->
            {noreply, do_flush(NewState)};
        false ->
            {noreply, NewState}
    end.

handle_call(flush, _From, State) ->
    NewState = do_flush(State),
    {reply, ok, NewState};

handle_call(get_stats, _From, State) ->
    {reply, State#state.stats, State}.

handle_info(flush_timer, State) ->
    %% Periodic flush
    NewState = do_flush(State),

    %% Reset timer
    Timer = erlang:send_after(State#state.flush_interval_ms, self(), flush_timer),

    {noreply, NewState#state{flush_timer = Timer}}.

terminate(_Reason, State) ->
    %% Flush remaining events on shutdown
    do_flush(State),
    ok.

%%====================================================================
%% Internal Functions
%%====================================================================

do_flush(#state{buffer = [], buffer_count = 0} = State) ->
    State;

do_flush(#state{} = State) ->
    Events = lists:reverse(State#state.buffer),

    %% Call backend to persist
    Result = neuroevolution_lineage_esdb:persist_batch(Events, #{
        store = State#state.store
    }),

    NewStats = case Result of
        {ok, _} ->
            State#state.stats#{
                batches_flushed := maps:get(batches_flushed, State#state.stats) + 1,
                events_persisted := maps:get(events_persisted, State#state.stats) +
                                   length(Events)
            };
        {error, _} ->
            State#state.stats#{
                errors := maps:get(errors, State#state.stats) + 1
            }
    end,

    State#state{
        buffer = [],
        buffer_count = 0,
        stats = NewStats
    }.
```

---

## Stream Design

### Stream Naming Convention

| Stream Pattern | Purpose | Example |
|----------------|---------|---------|
| `individual-{id}` | Single individual's complete history | `individual-abc123` |
| `species-{id}` | Species lifecycle events | `species-def456` |
| `population-{id}` | Population-level events | `population-ghi789` |
| `lineage-{id}` | Multi-generational lineage (denormalized) | `lineage-jkl012` |
| `coalition-{id}` | Coalition lifecycle | `coalition-mno345` |
| `$ce-neuroevolution` | Category stream (all events) | System stream |

### Event Metadata

All events include standard metadata for correlation and causation tracking:

```erlang
add_metadata(Event) ->
    BaseMetadata = #{
        event_id => generate_event_id(),
        timestamp => erlang:system_time(microsecond),
        version => 1
    },

    %% Add correlation from process dictionary if available
    Correlation = case get(correlation_id) of
        undefined -> #{};
        CorrId -> #{correlation_id => CorrId}
    end,

    %% Add causation if this event was triggered by another
    Causation = case get(causation_id) of
        undefined -> #{};
        CausId -> #{causation_id => CausId}
    end,

    Event#{metadata => maps:merge(BaseMetadata,
                                  maps:merge(Correlation, Causation))}.
```

---

## Integration with neuroevolution_events

### Configuration

```erlang
%% In application config
{faber_neuroevolution, [
    {event_backends, [
        %% Local pubsub (always enabled)
        {neuroevolution_events_local, #{}},

        %% Lineage persistence (optional)
        {neuroevolution_lineage_esdb, #{
            store_name => neuroevolution_store,
            batch_size => 100,
            flush_interval_ms => 1000,
            async => true
        }}
    ]}
]}.
```

### Backend Registration

```erlang
%% In neuroevolution_events.erl
-spec add_backend(Module :: module(), Config :: map()) -> ok.
add_backend(Module, Config) ->
    {ok, State} = Module:init(Config),
    Backends = get_backends(),
    put_backends([{Module, State} | Backends]),
    ok.

%% Event emission fans out to all backends
emit(EventType, Event) ->
    EventWithMeta = add_metadata(Event#{event_type => EventType}),

    lists:foreach(fun({Module, State}) ->
        case erlang:function_exported(Module, persist_event, 2) of
            true ->
                %% Lineage backend
                Module:persist_event(EventWithMeta, State);
            false ->
                %% Pubsub backend
                Module:publish(event_topic(EventType), EventWithMeta)
        end
    end, get_backends()).
```

---

## erl-evoq Aggregates

### Lineage Aggregate

```erlang
-module(lineage_aggregate).
-behaviour(erl_evoq_aggregate).

-export([
    init/1,
    execute/2,
    apply_event/2
]).

-record(lineage_state, {
    individual_id :: binary(),
    parent_ids :: [binary()],
    species_id :: binary(),
    birth_generation :: non_neg_integer(),
    mutations :: [map()],
    fitness_history :: [{integer(), float()}],
    status :: alive | dead
}).

init(IndividualId) ->
    {ok, #lineage_state{
        individual_id = IndividualId,
        parent_ids = [],
        species_id = undefined,
        birth_generation = 0,
        mutations = [],
        fitness_history = [],
        status = undefined
    }}.

%% Commands
execute({record_birth, BirthEvent}, State) ->
    {ok, [BirthEvent]};

execute({record_mutation, MutationEvent}, State) ->
    {ok, [MutationEvent]};

execute({record_fitness, FitnessEvent}, State) ->
    {ok, [FitnessEvent]};

execute({record_death, DeathEvent}, State) ->
    {ok, [DeathEvent]}.

%% Event application
apply_event(#{event_type := offspring_born} = Event, State) ->
    State#lineage_state{
        parent_ids = maps:get(parent_ids, Event, []),
        species_id = maps:get(species_id, Event),
        birth_generation = maps:get(generation, Event),
        status = alive
    };

apply_event(#{event_type := Type} = Event, State) when
        Type == individual_culled;
        Type == lifespan_expired;
        Type == individual_perished ->
    State#lineage_state{status = dead};

apply_event(#{event_type := fitness_evaluated} = Event, State) ->
    Timestamp = maps:get(evaluated_at, Event),
    Fitness = maps:get(fitness, Event),
    State#lineage_state{
        fitness_history = [{Timestamp, Fitness} | State#lineage_state.fitness_history]
    };

apply_event(#{event_type := Type} = Event, State) when
        Type == mutation_applied;
        Type == neuron_added;
        Type == connection_added ->
    State#lineage_state{
        mutations = [Event | State#lineage_state.mutations]
    };

apply_event(_Event, State) ->
    State.
```

---

## Query Patterns

### Common Queries

```erlang
%% Get full lineage tree (ancestors + descendants)
get_full_lineage_tree(IndividualId) ->
    Backend = get_lineage_backend(),
    {ok, State} = Backend:init(get_config()),

    %% Get individual's history
    {ok, IndividualEvents} = Backend:get_lineage(IndividualId, State),

    %% Extract parent IDs
    BirthEvent = find_birth_event(IndividualEvents),
    ParentIds = maps:get(parent_ids, BirthEvent, []),

    %% Recursively get ancestors
    Ancestors = lists:flatmap(fun(ParentId) ->
        {ok, ParentLineage} = get_full_lineage_tree(ParentId),
        ParentLineage
    end, ParentIds),

    %% Get offspring (query by parent_id in birth events)
    Offspring = get_offspring(IndividualId, State),

    #{
        individual_id => IndividualId,
        events => IndividualEvents,
        ancestors => Ancestors,
        offspring => Offspring
    }.

%% Get fitness trajectory with rolling average
get_fitness_trajectory_with_smoothing(IndividualId, WindowSize) ->
    Backend = get_lineage_backend(),
    {ok, State} = Backend:init(get_config()),

    {ok, Trajectory} = Backend:get_fitness_trajectory(IndividualId, State),

    %% Apply rolling average
    smooth_trajectory(Trajectory, WindowSize).

%% Get all descendants of a champion
get_champion_descendants(ChampionId, MaxDepth) ->
    Backend = get_lineage_backend(),
    {ok, State} = Backend:init(get_config()),

    get_descendants_recursive(ChampionId, State, 0, MaxDepth, #{}).
```

---

## Implementation Phases

- [ ] **Phase 1:** Create faber-neuroevolution-evoq repository
- [ ] **Phase 2:** Implement neuroevolution_lineage_esdb behaviour
- [ ] **Phase 3:** Implement event batcher with async flush
- [ ] **Phase 4:** Implement stream routing logic
- [ ] **Phase 5:** Add lineage query functions
- [ ] **Phase 6:** Create erl-evoq aggregates (Lineage, Individual, Species)
- [ ] **Phase 7:** Add projections for trees and trajectories
- [ ] **Phase 8:** Integration tests with erl-esdb
- [ ] **Phase 9:** Performance benchmarking

---

## Success Criteria

- [ ] Events persist without blocking evolution
- [ ] Batch size and flush interval are configurable
- [ ] All 70+ event types route to correct streams
- [ ] Lineage queries return complete history
- [ ] Breeding trees can be reconstructed
- [ ] Fitness trajectories are queryable
- [ ] Mutation history is trackable
- [ ] Causation chains are followable
- [ ] Performance: < 1% overhead on evolution loop

---

## References

- PLAN_BEHAVIORAL_EVENTS.md - Event catalog
- EVENT_DRIVEN_ARCHITECTURE.md - Existing event system
- erl-esdb documentation
- erl-evoq documentation
- erl-evoq-esdb adapter documentation
