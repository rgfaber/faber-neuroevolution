%% @doc Mock implementation of neuroevolution_lineage_events for testing.
%%
%% This module provides an in-memory implementation of the lineage backend
%% behaviour for use in tests. Events are stored in ETS tables.
%%
%% NOTE: This mock implements ALL callbacks including optional query callbacks.
%% The query implementations scan events directly (acceptable for testing).
%% Production backends should use projections for query callbacks.
-module(mock_lineage_backend).
-behaviour(neuroevolution_lineage_events).

%% Required callbacks - Event Store Operations
-export([
    init/1,
    persist_event/2,
    persist_batch/2,
    read_stream/3,
    subscribe/3,
    unsubscribe/3
]).

%% Optional callbacks - Query Operations
-export([
    get_breeding_tree/3,
    get_fitness_trajectory/2,
    get_mutation_history/2,
    get_knowledge_transfers/2,
    get_by_causation/2
]).

%% Test helpers
-export([
    get_all_events/1,
    clear/1
]).

-record(state, {
    events_table :: ets:tid(),
    positions_table :: ets:tid(),
    subscribers_table :: ets:tid()
}).

%%====================================================================
%% Behaviour Implementation - Required Callbacks (Event Store)
%%====================================================================

init(_Config) ->
    EventsTable = ets:new(mock_lineage_events, [bag, public]),
    PositionsTable = ets:new(mock_lineage_positions, [set, public]),
    SubscribersTable = ets:new(mock_lineage_subscribers, [bag, public]),
    {ok, #state{
        events_table = EventsTable,
        positions_table = PositionsTable,
        subscribers_table = SubscribersTable
    }}.

persist_event(Event, #state{} = State) ->
    StreamId = route_event(Event),
    Position = get_next_position(State#state.positions_table, StreamId),
    EventWithMeta = Event#{
        stream_id => StreamId,
        position => Position,
        stored_at => erlang:system_time(millisecond)
    },
    ets:insert(State#state.events_table, {StreamId, Position, EventWithMeta}),
    update_position(State#state.positions_table, StreamId, Position),
    notify_subscribers(StreamId, EventWithMeta, State),
    ok.

persist_batch(Events, State) ->
    lists:foreach(fun(E) -> persist_event(E, State) end, Events),
    ok.

read_stream(StreamId, Opts, #state{events_table = Table} = _State) ->
    FromPos = maps:get(from, Opts, 0),
    Limit = maps:get(limit, Opts, 10000),
    Direction = maps:get(direction, Opts, forward),

    AllEntries = ets:lookup(Table, StreamId),
    Events = [{Pos, E} || {_S, Pos, E} <- AllEntries, Pos >= FromPos],

    Sorted = case Direction of
        forward -> lists:keysort(1, Events);
        backward -> lists:reverse(lists:keysort(1, Events))
    end,

    Limited = lists:sublist(Sorted, Limit),
    {ok, [E || {_Pos, E} <- Limited]}.

subscribe(StreamId, Pid, #state{subscribers_table = Table} = _State) ->
    ets:insert(Table, {StreamId, Pid}),
    ok.

unsubscribe(StreamId, Pid, #state{subscribers_table = Table} = _State) ->
    ets:delete_object(Table, {StreamId, Pid}),
    ok.

%%====================================================================
%% Behaviour Implementation - Optional Callbacks (Queries)
%%====================================================================
%%
%% These implementations scan events directly. In production, use projections.

%% @doc Get breeding tree by scanning birth events recursively.
get_breeding_tree(IndividualId, Depth, State) ->
    StreamId = <<"individual-", IndividualId/binary>>,
    {ok, Events} = read_stream(StreamId, #{}, State),
    BirthEvent = find_birth_event(Events),
    Tree = build_tree(IndividualId, BirthEvent, Depth, State),
    {ok, Tree}.

%% @doc Get fitness trajectory by scanning fitness_evaluated events.
get_fitness_trajectory(IndividualId, State) ->
    StreamId = <<"individual-", IndividualId/binary>>,
    {ok, Events} = read_stream(StreamId, #{}, State),
    FitnessEvents = [E || E <- Events, maps:get(event_type, E, undefined) =:= fitness_evaluated],
    Trajectory = [{maps:get(stored_at, E), maps:get(fitness, E)} || E <- FitnessEvents],
    {ok, Trajectory}.

%% @doc Get mutation history by filtering mutation events.
get_mutation_history(IndividualId, State) ->
    StreamId = <<"individual-", IndividualId/binary>>,
    {ok, Events} = read_stream(StreamId, #{}, State),
    MutationTypes = [mutation_applied, neuron_added, neuron_removed,
                     connection_added, connection_removed, weight_perturbed],
    Mutations = [E || E <- Events,
                 lists:member(maps:get(event_type, E, undefined), MutationTypes)],
    {ok, Mutations}.

%% @doc Get knowledge transfer events involving the individual.
get_knowledge_transfers(IndividualId, State) ->
    StreamId = <<"individual-", IndividualId/binary>>,
    {ok, Events} = read_stream(StreamId, #{}, State),
    TransferTypes = [knowledge_transferred, skill_imitated, behavior_cloned,
                     weights_grafted, mentor_assigned, mentorship_concluded],
    Transfers = [E || E <- Events,
                 lists:member(maps:get(event_type, E, undefined), TransferTypes)],
    {ok, Transfers}.

%% @doc Get events by causation ID (scans all events).
get_by_causation(CausationId, #state{events_table = Table} = _State) ->
    AllEntries = ets:tab2list(Table),
    AllEvents = [E || {_Stream, _Pos, E} <- AllEntries],
    Matching = [E || E <- AllEvents,
                maps:get(causation_id, maps:get(metadata, E, #{}), undefined) =:= CausationId],
    {ok, Matching}.

%%====================================================================
%% Test Helpers
%%====================================================================

get_all_events(#state{events_table = Table} = _State) ->
    AllEntries = ets:tab2list(Table),
    [E || {_Stream, _Pos, E} <- AllEntries].

clear(#state{events_table = ET, positions_table = PT, subscribers_table = ST} = _State) ->
    ets:delete_all_objects(ET),
    ets:delete_all_objects(PT),
    ets:delete_all_objects(ST),
    ok.

%%====================================================================
%% Internal Functions
%%====================================================================

route_event(Event) ->
    EventType = maps:get(event_type, Event, unknown),
    case EventType of
        %% Individual events
        T when T == offspring_born; T == pioneer_spawned; T == fitness_evaluated;
               T == mutation_applied; T == individual_culled;
               T == weight_perturbed; T == neuron_added; T == connection_added;
               T == neuron_removed; T == connection_removed;
               T == knowledge_transferred; T == skill_imitated;
               T == behavior_cloned; T == weights_grafted;
               T == lifespan_expired; T == individual_perished;
               T == mentor_assigned; T == mentorship_concluded ->
            IndId = maps:get(individual_id, Event, <<"unknown">>),
            <<"individual-", IndId/binary>>;
        %% Species events
        T when T == lineage_diverged; T == species_emerged;
               T == lineage_ended; T == lineage_merged ->
            SpeciesId = maps:get(species_id, Event, <<"unknown">>),
            <<"species-", SpeciesId/binary>>;
        %% Population events
        T when T == generation_completed; T == population_initialized;
               T == population_terminated; T == stagnation_detected;
               T == breakthrough_achieved ->
            PopId = maps:get(population_id, Event, <<"default">>),
            <<"population-", PopId/binary>>;
        %% Coalition events
        T when T == coalition_formed; T == coalition_dissolved;
               T == coalition_joined ->
            CoalId = maps:get(coalition_id, Event, <<"unknown">>),
            <<"coalition-", CoalId/binary>>;
        %% Default to population stream
        _ ->
            PopId = maps:get(population_id, Event, <<"default">>),
            <<"population-", PopId/binary>>
    end.

get_next_position(Table, StreamId) ->
    case ets:lookup(Table, StreamId) of
        [{_, Pos}] -> Pos + 1;
        [] -> 0
    end.

update_position(Table, StreamId, Position) ->
    ets:insert(Table, {StreamId, Position}).

notify_subscribers(StreamId, Event, #state{subscribers_table = Table}) ->
    Subscribers = ets:lookup(Table, StreamId),
    lists:foreach(fun({_S, Pid}) ->
        Pid ! {lineage_event, StreamId, Event}
    end, Subscribers).

%% @private Find birth event (offspring_born or pioneer_spawned)
find_birth_event(Events) ->
    case [E || E <- Events,
               lists:member(maps:get(event_type, E, undefined),
                           [offspring_born, pioneer_spawned])] of
        [Birth | _] -> Birth;
        [] -> undefined
    end.

%% @private Build tree recursively up to Depth generations
build_tree(IndividualId, BirthEvent, Depth, State) ->
    Node = #{individual_id => IndividualId, birth_event => BirthEvent},
    case {Depth, BirthEvent} of
        {D, _} when D =< 0 ->
            Node;
        {_, undefined} ->
            Node;
        {_, #{event_type := pioneer_spawned}} ->
            Node#{parents => []};
        {_, #{event_type := offspring_born, parent_ids := ParentIds}} ->
            Parents = [build_parent_tree(PId, Depth - 1, State) || PId <- ParentIds],
            Node#{parents => Parents};
        _ ->
            Node
    end.

%% @private Build parent tree node
build_parent_tree(ParentId, Depth, State) ->
    StreamId = <<"individual-", ParentId/binary>>,
    {ok, Events} = read_stream(StreamId, #{}, State),
    BirthEvent = find_birth_event(Events),
    build_tree(ParentId, BirthEvent, Depth, State).
