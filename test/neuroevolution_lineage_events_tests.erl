%% @doc EUnit tests for neuroevolution_lineage_events behaviour.
%%
%% Tests the EVENT STORE contract:
%% - Event persistence (single and batch)
%% - Stream reading with options
%% - Subscription for projections
%%
%% NOTE: Derived queries (breeding trees, trajectories) are NOT tested here.
%% Those are the responsibility of projection modules, not the event store.
-module(neuroevolution_lineage_events_tests).

-include_lib("eunit/include/eunit.hrl").

%%====================================================================
%% Setup and Teardown
%%====================================================================

setup() ->
    {ok, State} = mock_lineage_backend:init(#{}),
    State.

%%====================================================================
%% Behaviour Contract Tests - Event Store Operations
%%====================================================================

init_test() ->
    {ok, State} = mock_lineage_backend:init(#{some => config}),
    ?assertMatch({state, _, _, _}, State).

persist_single_event_test() ->
    State = setup(),
    Event = #{
        event_type => offspring_born,
        individual_id => <<"ind-001">>,
        parent_ids => [<<"ind-parent-1">>, <<"ind-parent-2">>],
        generation => 5
    },
    Result = mock_lineage_backend:persist_event(Event, State),
    ?assertEqual(ok, Result).

persist_batch_test() ->
    State = setup(),
    Events = [
        #{event_type => offspring_born, individual_id => <<"ind-001">>, generation => 1},
        #{event_type => fitness_evaluated, individual_id => <<"ind-001">>, fitness => 0.75},
        #{event_type => mutation_applied, individual_id => <<"ind-001">>, mutation_type => weight_perturb}
    ],
    Result = mock_lineage_backend:persist_batch(Events, State),
    ?assertEqual(ok, Result).

%%====================================================================
%% Stream Reading Tests
%%====================================================================

read_stream_empty_test() ->
    State = setup(),
    {ok, Events} = mock_lineage_backend:read_stream(<<"individual-nonexistent">>, #{}, State),
    ?assertEqual([], Events).

read_stream_single_event_test() ->
    State = setup(),
    Event = #{event_type => pioneer_spawned, individual_id => <<"ind-pioneer">>},
    mock_lineage_backend:persist_event(Event, State),
    {ok, Events} = mock_lineage_backend:read_stream(<<"individual-ind-pioneer">>, #{}, State),
    ?assertEqual(1, length(Events)),
    ?assertEqual(pioneer_spawned, maps:get(event_type, hd(Events))).

read_stream_multiple_events_test() ->
    State = setup(),
    IndId = <<"ind-multi">>,
    Events = [
        #{event_type => offspring_born, individual_id => IndId, parent_ids => [<<"p1">>]},
        #{event_type => fitness_evaluated, individual_id => IndId, fitness => 0.5},
        #{event_type => fitness_evaluated, individual_id => IndId, fitness => 0.7},
        #{event_type => mutation_applied, individual_id => IndId, mutation_type => add_neuron},
        #{event_type => individual_culled, individual_id => IndId, reason => selection}
    ],
    mock_lineage_backend:persist_batch(Events, State),
    {ok, Retrieved} = mock_lineage_backend:read_stream(<<"individual-", IndId/binary>>, #{}, State),
    ?assertEqual(5, length(Retrieved)).

read_stream_with_limit_test() ->
    State = setup(),
    IndId = <<"ind-limited">>,
    Events = [#{event_type => fitness_evaluated, individual_id => IndId, fitness => F / 10}
              || F <- lists:seq(1, 10)],
    mock_lineage_backend:persist_batch(Events, State),
    {ok, Limited} = mock_lineage_backend:read_stream(<<"individual-", IndId/binary>>, #{limit => 5}, State),
    ?assertEqual(5, length(Limited)).

read_stream_with_from_position_test() ->
    State = setup(),
    IndId = <<"ind-from">>,
    Events = [#{event_type => fitness_evaluated, individual_id => IndId, fitness => F / 10}
              || F <- lists:seq(1, 10)],
    mock_lineage_backend:persist_batch(Events, State),
    {ok, FromPos5} = mock_lineage_backend:read_stream(<<"individual-", IndId/binary>>, #{from => 5}, State),
    ?assertEqual(5, length(FromPos5)).

read_stream_backward_test() ->
    State = setup(),
    IndId = <<"ind-backward">>,
    Events = [#{event_type => fitness_evaluated, individual_id => IndId, fitness => F / 10}
              || F <- lists:seq(1, 5)],
    mock_lineage_backend:persist_batch(Events, State),
    {ok, Forward} = mock_lineage_backend:read_stream(<<"individual-", IndId/binary>>, #{direction => forward}, State),
    {ok, Backward} = mock_lineage_backend:read_stream(<<"individual-", IndId/binary>>, #{direction => backward}, State),
    ?assertEqual(lists:reverse(Forward), Backward).

%%====================================================================
%% Stream Routing Tests
%%====================================================================

individual_events_route_correctly_test() ->
    State = setup(),
    IndEvents = [
        #{event_type => offspring_born, individual_id => <<"ind-route">>},
        #{event_type => pioneer_spawned, individual_id => <<"ind-route">>},
        #{event_type => fitness_evaluated, individual_id => <<"ind-route">>, fitness => 0.5},
        #{event_type => mutation_applied, individual_id => <<"ind-route">>},
        #{event_type => individual_culled, individual_id => <<"ind-route">>}
    ],
    mock_lineage_backend:persist_batch(IndEvents, State),
    {ok, Retrieved} = mock_lineage_backend:read_stream(<<"individual-ind-route">>, #{}, State),
    ?assertEqual(5, length(Retrieved)).

species_events_route_correctly_test() ->
    State = setup(),
    SpeciesEvents = [
        #{event_type => species_emerged, species_id => <<"sp-route">>},
        #{event_type => lineage_diverged, species_id => <<"sp-route">>}
    ],
    mock_lineage_backend:persist_batch(SpeciesEvents, State),
    {ok, Retrieved} = mock_lineage_backend:read_stream(<<"species-sp-route">>, #{}, State),
    ?assertEqual(2, length(Retrieved)).

population_events_route_correctly_test() ->
    State = setup(),
    PopEvents = [
        #{event_type => population_initialized, population_id => <<"pop-route">>, size => 50},
        #{event_type => generation_completed, population_id => <<"pop-route">>, generation => 1}
    ],
    mock_lineage_backend:persist_batch(PopEvents, State),
    {ok, Retrieved} = mock_lineage_backend:read_stream(<<"population-pop-route">>, #{}, State),
    ?assertEqual(2, length(Retrieved)).

coalition_events_route_correctly_test() ->
    State = setup(),
    CoalEvents = [
        #{event_type => coalition_formed, coalition_id => <<"coal-route">>, founder_ids => [<<"a">>, <<"b">>]},
        #{event_type => coalition_dissolved, coalition_id => <<"coal-route">>}
    ],
    mock_lineage_backend:persist_batch(CoalEvents, State),
    {ok, Retrieved} = mock_lineage_backend:read_stream(<<"coalition-coal-route">>, #{}, State),
    ?assertEqual(2, length(Retrieved)).

%%====================================================================
%% Subscription Tests (for projections)
%%====================================================================

subscribe_receives_events_test() ->
    State = setup(),
    StreamId = <<"individual-sub-test">>,

    %% Subscribe self to stream
    ok = mock_lineage_backend:subscribe(StreamId, self(), State),

    %% Persist an event
    Event = #{event_type => fitness_evaluated, individual_id => <<"sub-test">>, fitness => 0.5},
    mock_lineage_backend:persist_event(Event, State),

    %% Should receive the event
    receive
        {lineage_event, StreamId, ReceivedEvent} ->
            ?assertEqual(fitness_evaluated, maps:get(event_type, ReceivedEvent))
    after 1000 ->
        ?assert(false)
    end.

unsubscribe_stops_events_test() ->
    State = setup(),
    StreamId = <<"individual-unsub-test">>,

    %% Subscribe then unsubscribe
    ok = mock_lineage_backend:subscribe(StreamId, self(), State),
    ok = mock_lineage_backend:unsubscribe(StreamId, self(), State),

    %% Persist an event
    Event = #{event_type => fitness_evaluated, individual_id => <<"unsub-test">>, fitness => 0.5},
    mock_lineage_backend:persist_event(Event, State),

    %% Should NOT receive the event
    receive
        {lineage_event, _, _} ->
            ?assert(false)
    after 100 ->
        ok
    end.

%%====================================================================
%% Event Metadata Tests
%%====================================================================

events_have_metadata_test() ->
    State = setup(),
    Event = #{event_type => offspring_born, individual_id => <<"ind-meta">>},
    mock_lineage_backend:persist_event(Event, State),

    {ok, [Retrieved]} = mock_lineage_backend:read_stream(<<"individual-ind-meta">>, #{}, State),

    %% Should have stream_id, position, stored_at added
    ?assert(maps:is_key(stream_id, Retrieved)),
    ?assert(maps:is_key(position, Retrieved)),
    ?assert(maps:is_key(stored_at, Retrieved)),
    ?assertEqual(<<"individual-ind-meta">>, maps:get(stream_id, Retrieved)),
    ?assertEqual(0, maps:get(position, Retrieved)).

events_have_sequential_positions_test() ->
    State = setup(),
    IndId = <<"ind-seq">>,
    Events = [#{event_type => fitness_evaluated, individual_id => IndId, fitness => F}
              || F <- [0.1, 0.2, 0.3]],
    mock_lineage_backend:persist_batch(Events, State),

    {ok, Retrieved} = mock_lineage_backend:read_stream(<<"individual-", IndId/binary>>, #{}, State),
    Positions = [maps:get(position, E) || E <- Retrieved],
    ?assertEqual([0, 1, 2], Positions).

%%====================================================================
%% Optional Callback Tests - Query Operations
%%====================================================================

get_fitness_trajectory_test() ->
    State = setup(),
    IndId = <<"ind-trajectory">>,

    %% Persist some fitness events
    Events = [
        #{event_type => pioneer_spawned, individual_id => IndId},
        #{event_type => fitness_evaluated, individual_id => IndId, fitness => 0.5},
        #{event_type => mutation_applied, individual_id => IndId, mutation_type => add_neuron},
        #{event_type => fitness_evaluated, individual_id => IndId, fitness => 0.6},
        #{event_type => fitness_evaluated, individual_id => IndId, fitness => 0.75}
    ],
    mock_lineage_backend:persist_batch(Events, State),

    %% Get trajectory
    {ok, Trajectory} = mock_lineage_backend:get_fitness_trajectory(IndId, State),

    %% Should have 3 fitness points
    ?assertEqual(3, length(Trajectory)),
    %% Should be in order with increasing fitness
    Fitnesses = [F || {_T, F} <- Trajectory],
    ?assertEqual([0.5, 0.6, 0.75], Fitnesses).

get_mutation_history_test() ->
    State = setup(),
    IndId = <<"ind-mutations">>,

    Events = [
        #{event_type => offspring_born, individual_id => IndId, parent_ids => [<<"p1">>]},
        #{event_type => fitness_evaluated, individual_id => IndId, fitness => 0.3},
        #{event_type => neuron_added, individual_id => IndId, neuron_id => <<"n1">>},
        #{event_type => fitness_evaluated, individual_id => IndId, fitness => 0.4},
        #{event_type => connection_added, individual_id => IndId, from => <<"n1">>, to => <<"out">>},
        #{event_type => weight_perturbed, individual_id => IndId, delta => 0.1}
    ],
    mock_lineage_backend:persist_batch(Events, State),

    {ok, Mutations} = mock_lineage_backend:get_mutation_history(IndId, State),

    %% Should have 3 mutation events
    ?assertEqual(3, length(Mutations)),
    MutationTypes = [maps:get(event_type, M) || M <- Mutations],
    ?assertEqual([neuron_added, connection_added, weight_perturbed], MutationTypes).

get_breeding_tree_pioneer_test() ->
    State = setup(),
    IndId = <<"ind-pioneer-tree">>,

    %% Pioneer has no parents
    mock_lineage_backend:persist_event(
        #{event_type => pioneer_spawned, individual_id => IndId},
        State
    ),

    {ok, Tree} = mock_lineage_backend:get_breeding_tree(IndId, 3, State),

    ?assertEqual(IndId, maps:get(individual_id, Tree)),
    ?assertEqual([], maps:get(parents, Tree, undefined)).

get_breeding_tree_with_ancestry_test() ->
    State = setup(),

    %% Create a family tree: grandparent -> parent -> child
    mock_lineage_backend:persist_event(
        #{event_type => pioneer_spawned, individual_id => <<"grandparent">>},
        State
    ),
    mock_lineage_backend:persist_event(
        #{event_type => offspring_born, individual_id => <<"parent">>, parent_ids => [<<"grandparent">>]},
        State
    ),
    mock_lineage_backend:persist_event(
        #{event_type => offspring_born, individual_id => <<"child">>, parent_ids => [<<"parent">>]},
        State
    ),

    {ok, Tree} = mock_lineage_backend:get_breeding_tree(<<"child">>, 3, State),

    ?assertEqual(<<"child">>, maps:get(individual_id, Tree)),
    Parents = maps:get(parents, Tree),
    ?assertEqual(1, length(Parents)),
    [Parent] = Parents,
    ?assertEqual(<<"parent">>, maps:get(individual_id, Parent)).

get_knowledge_transfers_test() ->
    State = setup(),
    IndId = <<"ind-learner">>,

    Events = [
        #{event_type => offspring_born, individual_id => IndId, parent_ids => [<<"p1">>]},
        #{event_type => mentor_assigned, individual_id => IndId, mentor_id => <<"mentor1">>},
        #{event_type => fitness_evaluated, individual_id => IndId, fitness => 0.3},
        #{event_type => knowledge_transferred, individual_id => IndId, mentor_id => <<"mentor1">>},
        #{event_type => fitness_evaluated, individual_id => IndId, fitness => 0.5}
    ],
    mock_lineage_backend:persist_batch(Events, State),

    {ok, Transfers} = mock_lineage_backend:get_knowledge_transfers(IndId, State),

    ?assertEqual(2, length(Transfers)),
    TransferTypes = [maps:get(event_type, T) || T <- Transfers],
    ?assertEqual([mentor_assigned, knowledge_transferred], TransferTypes).

get_by_causation_test() ->
    State = setup(),
    CausationId = <<"batch-123">>,

    %% Events with same causation_id
    Events = [
        #{event_type => fitness_evaluated, individual_id => <<"ind-1">>, fitness => 0.5,
          metadata => #{causation_id => CausationId}},
        #{event_type => fitness_evaluated, individual_id => <<"ind-2">>, fitness => 0.6,
          metadata => #{causation_id => CausationId}},
        #{event_type => fitness_evaluated, individual_id => <<"ind-3">>, fitness => 0.7,
          metadata => #{causation_id => <<"other-batch">>}}
    ],
    mock_lineage_backend:persist_batch(Events, State),

    {ok, Related} = mock_lineage_backend:get_by_causation(CausationId, State),

    %% Should find 2 events with matching causation
    ?assertEqual(2, length(Related)).
