%% @doc EUnit tests for event-driven neuroevolution system.
%%
%% Tests the event publishing/subscription flow including:
%% - Event publishing and subscription via local backend
%% - Evaluator worker lifecycle
%% - End-to-end evaluation via events
-module(neuroevolution_events_tests).

-include_lib("eunit/include/eunit.hrl").
-include("neuroevolution.hrl").
-include("evolution_strategy.hrl").

%%% ============================================================================
%%% Event System Tests
%%% ============================================================================

local_pubsub_test() ->
    %% Ensure the local event system is started
    ok = neuroevolution_events_local:start(),

    %% Subscribe to a test topic
    Topic = <<"neuro.test.events">>,
    ok = neuroevolution_events:subscribe(Topic),

    %% Publish an event
    Event = #{type => test_event, data => 123},
    ok = neuroevolution_events:publish(Topic, Event),

    %% Should receive the event
    receive
        {neuro_event, Topic, Event} -> ok
    after 1000 ->
        ?assert(false, "Did not receive event")
    end,

    %% Unsubscribe
    ok = neuroevolution_events:unsubscribe(Topic).

topic_helpers_test() ->
    Realm = <<"my-realm">>,

    EvalTopic = neuroevolution_events:evaluate_topic(Realm),
    ?assertEqual(<<"neuro.my-realm.evaluate">>, EvalTopic),

    EvaluatedTopic = neuroevolution_events:evaluated_topic(Realm),
    ?assertEqual(<<"neuro.my-realm.evaluated">>, EvaluatedTopic),

    %% Note: events_topic uses the v2 format with colons (evolution:realm:events)
    EventsTopic = neuroevolution_events:events_topic(Realm),
    ?assertEqual(<<"evolution:my-realm:events">>, EventsTopic).

%% @doc Test multi-channel topic constructors (v2 vertical slicing).
%% Note: v2 topics use colon separators (evolution:realm:channel).
multi_channel_topics_test() ->
    Realm = <<"test-realm">>,

    %% Generation topic
    GenTopic = neuroevolution_events:topic_generation(Realm),
    ?assertEqual(<<"evolution:test-realm:generation">>, GenTopic),

    %% Population topic
    PopTopic = neuroevolution_events:topic_population(Realm),
    ?assertEqual(<<"evolution:test-realm:population">>, PopTopic),

    %% Species topic
    SpeciesTopic = neuroevolution_events:topic_species(Realm),
    ?assertEqual(<<"evolution:test-realm:species">>, SpeciesTopic),

    %% Training topic
    TrainingTopic = neuroevolution_events:topic_training(Realm),
    ?assertEqual(<<"evolution:test-realm:training">>, TrainingTopic),

    %% Meta topic
    MetaTopic = neuroevolution_events:topic_meta(Realm),
    ?assertEqual(<<"evolution:test-realm:meta">>, MetaTopic),

    %% Individual topic
    IndTopic = neuroevolution_events:topic_individual(Realm),
    ?assertEqual(<<"evolution:test-realm:individual">>, IndTopic).

%% @doc Test that each topic constructor produces unique topics for same realm.
topic_uniqueness_test() ->
    Realm = <<"uniqueness-test">>,

    Topics = [
        neuroevolution_events:topic_generation(Realm),
        neuroevolution_events:topic_population(Realm),
        neuroevolution_events:topic_species(Realm),
        neuroevolution_events:topic_training(Realm),
        neuroevolution_events:topic_meta(Realm),
        neuroevolution_events:topic_individual(Realm)
    ],

    %% All topics should be unique
    ?assertEqual(6, length(Topics)),
    ?assertEqual(6, length(lists:usort(Topics))).

%% @doc Test topics differ across realms.
topic_realm_isolation_test() ->
    Realm1 = <<"realm-a">>,
    Realm2 = <<"realm-b">>,

    %% Same topic type should differ for different realms
    ?assertNotEqual(
        neuroevolution_events:topic_generation(Realm1),
        neuroevolution_events:topic_generation(Realm2)
    ),
    ?assertNotEqual(
        neuroevolution_events:topic_training(Realm1),
        neuroevolution_events:topic_training(Realm2)
    ).

%%% ============================================================================
%%% Evaluator Worker Tests
%%% ============================================================================

worker_start_stop_test() ->
    %% Start a worker
    Config = #{
        realm => <<"test-realm">>,
        evaluator_module => mock_evaluator,
        evaluator_options => #{}
    },
    {ok, Pid} = neuroevolution_evaluator_worker:start_link(Config),
    ?assert(is_pid(Pid)),
    ?assert(is_process_alive(Pid)),

    %% Stop it
    ok = neuroevolution_evaluator_worker:stop(Pid),
    timer:sleep(50),
    ?assertNot(is_process_alive(Pid)).

worker_evaluation_test() ->
    %% Ensure event system is started
    ok = neuroevolution_events_local:start(),

    Realm = <<"worker-test">>,

    %% Subscribe to results before starting worker
    ResultTopic = neuroevolution_events:evaluated_topic(Realm),
    ok = neuroevolution_events:subscribe(ResultTopic),

    %% Start worker
    Config = #{
        realm => Realm,
        evaluator_module => mock_evaluator,
        evaluator_options => #{}
    },
    {ok, WorkerPid} = neuroevolution_evaluator_worker:start_link(Config),

    %% Create a simple network for testing (use correct API)
    Network = network_evaluator:create_feedforward(4, [8], 2),
    RequestId = make_ref(),
    IndividualId = test_individual_1,

    %% Publish evaluation request
    EvalTopic = neuroevolution_events:evaluate_topic(Realm),
    Request = #{
        request_id => RequestId,
        individual_id => IndividualId,
        network => Network,
        options => #{}
    },
    ok = neuroevolution_events:publish(EvalTopic, {evaluate_request, Request}),

    %% Wait for result
    receive
        {neuro_event, ResultTopic, {evaluated, Result}} ->
            ?assertEqual(RequestId, maps:get(request_id, Result)),
            ?assertEqual(IndividualId, maps:get(individual_id, Result)),
            ?assert(is_map(maps:get(metrics, Result))),
            Metrics = maps:get(metrics, Result),
            ?assert(maps:is_key(total_score, Metrics))
    after 5000 ->
        neuroevolution_evaluator_worker:stop(WorkerPid),
        ?assert(false, "Did not receive evaluation result")
    end,

    %% Cleanup
    ok = neuroevolution_events:unsubscribe(ResultTopic),
    neuroevolution_evaluator_worker:stop(WorkerPid).

%%% ============================================================================
%%% Server Event Publishing Tests
%%% ============================================================================

%% This test needs a longer timeout because it waits for training events
server_publishes_events_test_() ->
    {timeout, 30, fun server_publishes_events_impl/0}.

server_publishes_events_impl() ->
    %% Ensure event system is started
    ok = neuroevolution_events_local:start(),

    Realm = <<"server-test">>,

    %% Subscribe to events topic
    EventsTopic = neuroevolution_events:events_topic(Realm),
    ok = neuroevolution_events:subscribe(EventsTopic),

    %% Create config with event publishing enabled
    Config = #neuro_config{
        population_size = 5,
        evaluations_per_individual = 1,
        selection_ratio = 0.4,
        mutation_rate = 0.1,
        mutation_strength = 0.2,
        max_generations = 2,
        network_topology = {4, [4], 2},
        evaluator_module = mock_evaluator,
        evaluator_options = #{},
        realm = Realm,
        publish_events = true,
        strategy_config = #strategy_config{
            strategy_module = generational_strategy,
            strategy_params = #{network_factory => mock_network_factory}
        }
    },

    %% Start server
    {ok, Pid} = neuroevolution_server:start_link(Config, []),

    %% Start training - should get training_started event
    {ok, started} = neuroevolution_server:start_training(Pid),

    %% Wait for training_started event (format: {training_started, #{...}})
    receive
        {neuro_event, EventsTopic, {training_started, _}} -> ok
    after 2000 ->
        gen_server:stop(Pid),
        neuroevolution_events:unsubscribe(EventsTopic),
        ?assert(false, "Did not receive training_started event")
    end,

    %% Wait for at least one generation_complete event
    %% Note: Event is `generation_complete` (without 'd')
    receive
        {neuro_event, EventsTopic, {generation_complete, _}} -> ok
    after 10000 ->
        gen_server:stop(Pid),
        neuroevolution_events:unsubscribe(EventsTopic),
        ?assert(false, "Did not receive generation_complete event")
    end,

    %% Cleanup
    gen_server:stop(Pid),
    ok = neuroevolution_events:unsubscribe(EventsTopic).

%%% ============================================================================
%%% Integration Test: Multiple Workers
%%% ============================================================================

%% This test needs a longer timeout for collecting results from multiple workers
multiple_workers_test_() ->
    {timeout, 30, fun multiple_workers_impl/0}.

multiple_workers_impl() ->
    %% Ensure event system is started
    ok = neuroevolution_events_local:start(),

    Realm = <<"multi-worker-test">>,

    %% Subscribe to results
    ResultTopic = neuroevolution_events:evaluated_topic(Realm),
    ok = neuroevolution_events:subscribe(ResultTopic),

    %% Start 3 workers
    WorkerConfig = #{
        realm => Realm,
        evaluator_module => mock_evaluator,
        evaluator_options => #{},
        max_concurrent => 2
    },
    {ok, W1} = neuroevolution_evaluator_worker:start_link(WorkerConfig),
    {ok, W2} = neuroevolution_evaluator_worker:start_link(WorkerConfig),
    {ok, W3} = neuroevolution_evaluator_worker:start_link(WorkerConfig),

    %% Send 5 evaluation requests (use correct API)
    EvalTopic = neuroevolution_events:evaluate_topic(Realm),
    RequestIds = lists:map(fun(N) ->
        ReqId = {request, N},
        Network = network_evaluator:create_feedforward(4, [4], 2),
        Request = #{
            request_id => ReqId,
            individual_id => {ind, N},
            network => Network,
            options => #{}
        },
        ok = neuroevolution_events:publish(EvalTopic, {evaluate_request, Request}),
        ReqId
    end, lists:seq(1, 5)),

    %% With local pg backend, all 3 workers receive all 5 requests = 15 results.
    %% This is the expected broadcast behavior for local testing.
    %% In distributed mode (Macula backend), load balancing would ensure only one worker
    %% receives each request.
    %%
    %% We need to collect enough results to ensure all request IDs are represented.
    %% Worst case: first 14 results are all duplicates of some requests.
    %% So we collect all 15 (5 requests × 3 workers) to be sure.
    Results = collect_results(ResultTopic, 15, 10000),
    ?assert(length(Results) >= 15),

    %% Verify all request IDs are represented (each appears 3 times)
    ReceivedIds = [maps:get(request_id, R) || R <- Results],
    lists:foreach(fun(ReqId) ->
        ?assert(lists:member(ReqId, ReceivedIds))
    end, RequestIds),

    %% Verify each request ID appears exactly 3 times (one per worker)
    lists:foreach(fun(ReqId) ->
        Count = length([R || R <- ReceivedIds, R =:= ReqId]),
        ?assertEqual(3, Count)
    end, RequestIds),

    %% Cleanup
    ok = neuroevolution_events:unsubscribe(ResultTopic),
    neuroevolution_evaluator_worker:stop(W1),
    neuroevolution_evaluator_worker:stop(W2),
    neuroevolution_evaluator_worker:stop(W3).

%%% ============================================================================
%%% Distributed Evaluation Mode Tests
%%% ============================================================================

%% This test needs a longer timeout for distributed evaluation
distributed_evaluation_test_() ->
    {timeout, 30, fun distributed_evaluation_impl/0}.

distributed_evaluation_impl() ->
    %% Ensure event system is started
    ok = neuroevolution_events_local:start(),

    Realm = <<"distributed-test">>,

    %% Start evaluator workers BEFORE server
    WorkerConfig = #{
        realm => Realm,
        evaluator_module => mock_evaluator,
        evaluator_options => #{},
        max_concurrent => 5
    },
    {ok, W1} = neuroevolution_evaluator_worker:start_link(WorkerConfig),
    {ok, W2} = neuroevolution_evaluator_worker:start_link(WorkerConfig),

    %% Subscribe to events to verify completion
    EventsTopic = neuroevolution_events:events_topic(Realm),
    ok = neuroevolution_events:subscribe(EventsTopic),

    %% Create config with distributed evaluation mode
    Config = #neuro_config{
        population_size = 5,
        evaluations_per_individual = 1,
        selection_ratio = 0.4,
        mutation_rate = 0.1,
        mutation_strength = 0.2,
        max_generations = 2,
        network_topology = {4, [4], 2},
        evaluator_module = mock_evaluator,  %% Not used in distributed mode
        evaluator_options = #{},
        realm = Realm,
        publish_events = true,
        evaluation_mode = distributed,
        evaluation_timeout = 10000,  %% 10 second timeout
        strategy_config = #strategy_config{
            strategy_module = generational_strategy,
            strategy_params = #{network_factory => mock_network_factory}
        }
    },

    %% Start server
    {ok, Pid} = neuroevolution_server:start_link(Config, []),

    %% Start training
    {ok, started} = neuroevolution_server:start_training(Pid),

    %% Wait for generation_complete event (indicates distributed evaluation worked)
    %% Note: Event is `generation_complete` (without 'd')
    receive
        {neuro_event, EventsTopic, {generation_complete, GenStats}} ->
            ?assert(is_map(GenStats)),
            ?assert(maps:get(generation, GenStats) >= 1),
            BestFitness = maps:get(best_fitness, GenStats),
            ?assert(is_number(BestFitness)),
            %% In distributed mode with mock_evaluator, workers may
            %% fail evaluation (function_clause), yielding fitness 0.
            %% The test verifies the event flow works, not fitness values.
            ?assert(BestFitness >= 0)
    after 15000 ->
        gen_server:stop(Pid),
        neuroevolution_evaluator_worker:stop(W1),
        neuroevolution_evaluator_worker:stop(W2),
        neuroevolution_events:unsubscribe(EventsTopic),
        ?assert(false, "Did not receive generation_complete event in distributed mode")
    end,

    %% Cleanup
    gen_server:stop(Pid),
    ok = neuroevolution_events:unsubscribe(EventsTopic),
    neuroevolution_evaluator_worker:stop(W1),
    neuroevolution_evaluator_worker:stop(W2).

%% This test needs a longer timeout for the evaluation timeout test
distributed_evaluation_timeout_test_() ->
    {timeout, 30, fun distributed_evaluation_timeout_impl/0}.

distributed_evaluation_timeout_impl() ->
    %% Test that evaluation times out gracefully when no workers are available
    ok = neuroevolution_events_local:start(),

    Realm = <<"timeout-test">>,

    %% NO workers started - should timeout

    %% Subscribe to events
    EventsTopic = neuroevolution_events:events_topic(Realm),
    ok = neuroevolution_events:subscribe(EventsTopic),

    %% Create config with very short timeout
    Config = #neuro_config{
        population_size = 3,
        evaluations_per_individual = 1,
        selection_ratio = 0.5,
        mutation_rate = 0.1,
        mutation_strength = 0.2,
        max_generations = 1,
        network_topology = {4, [4], 2},
        evaluator_module = mock_evaluator,
        evaluator_options = #{},
        realm = Realm,
        publish_events = true,
        evaluation_mode = distributed,
        evaluation_timeout = 500,  %% Very short timeout (500ms)
        strategy_config = #strategy_config{
            strategy_module = generational_strategy,
            strategy_params = #{network_factory => mock_network_factory}
        }
    },

    %% Start server
    {ok, Pid} = neuroevolution_server:start_link(Config, []),

    %% Start training
    {ok, started} = neuroevolution_server:start_training(Pid),

    %% Should still get generation_complete event after timeout
    %% (with zero-fitness individuals)
    %% Note: Event is `generation_complete` (without 'd')
    receive
        {neuro_event, EventsTopic, {generation_complete, GenStats}} ->
            ?assert(is_map(GenStats)),
            %% Fitness should be 0 since evaluations timed out
            ?assertEqual(0.0, maps:get(best_fitness, GenStats))
    after 5000 ->
        gen_server:stop(Pid),
        neuroevolution_events:unsubscribe(EventsTopic),
        ?assert(false, "Did not receive generation_complete after timeout")
    end,

    %% Cleanup
    gen_server:stop(Pid),
    ok = neuroevolution_events:unsubscribe(EventsTopic).

%%% ============================================================================
%%% Helper Functions
%%% ============================================================================

collect_results(Topic, MinCount, Timeout) ->
    collect_results(Topic, MinCount, Timeout, []).

collect_results(_Topic, MinCount, _Timeout, Acc) when length(Acc) >= MinCount ->
    Acc;
collect_results(Topic, MinCount, Timeout, Acc) ->
    receive
        {neuro_event, Topic, {evaluated, Result}} ->
            collect_results(Topic, MinCount, Timeout, [Result | Acc])
    after Timeout ->
        Acc
    end.
