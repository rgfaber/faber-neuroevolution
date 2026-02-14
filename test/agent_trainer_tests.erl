%% @doc Tests for agent_trainer module.
-module(agent_trainer_tests).

-include_lib("eunit/include/eunit.hrl").

%%% ============================================================================
%%% Test: to_fitness_fn/2
%%% ============================================================================

to_fitness_fn_creates_function_test() ->
    {ok, Bridge} = create_test_bridge(),
    FitnessFn = agent_trainer:to_fitness_fn(Bridge, #{max_ticks => 5}),
    ?assert(is_function(FitnessFn, 1)).

to_fitness_fn_returns_fitness_test() ->
    {ok, Bridge} = create_test_bridge(),
    FitnessFn = agent_trainer:to_fitness_fn(Bridge, #{max_ticks => 5}),
    %% Use a simple network that always outputs 0.6 (> 0.5 -> up -> +1 score)
    Network = fun(_Inputs) -> [0.6] end,
    Fitness = FitnessFn(Network),
    ?assert(is_float(Fitness)),
    ?assert(Fitness > 0).

%%% ============================================================================
%%% Test: to_neuro_config/2,3
%%% ============================================================================

to_neuro_config_extracts_topology_test() ->
    {ok, Bridge} = create_test_bridge(),
    {ok, Config} = agent_trainer:to_neuro_config(Bridge, #{max_ticks => 5}),
    ConfigMap = neuro_config:to_map(Config),
    ?assertEqual({2, [4], 1}, maps:get(network_topology, ConfigMap)).

to_neuro_config_sets_population_size_test() ->
    {ok, Bridge} = create_test_bridge(),
    {ok, Config} = agent_trainer:to_neuro_config(Bridge, #{}, #{population_size => 50}),
    ConfigMap = neuro_config:to_map(Config),
    ?assertEqual(50, maps:get(population_size, ConfigMap)).

to_neuro_config_uses_bridge_evaluator_test() ->
    {ok, Bridge} = create_test_bridge(),
    {ok, Config} = agent_trainer:to_neuro_config(Bridge, #{}),
    ConfigMap = neuro_config:to_map(Config),
    ?assertEqual(bridge_evaluator, maps:get(evaluator_module, ConfigMap)).

to_neuro_config_requires_evaluator_test() ->
    %% Create bridge without evaluator
    {ok, Bridge} = agent_bridge:new(#{
        definition => test_trainer_definition,
        sensors => [test_trainer_sensor],
        actuators => [test_trainer_actuator],
        environment => test_trainer_environment
        %% No evaluator!
    }),
    Result = agent_trainer:to_neuro_config(Bridge, #{}),
    ?assertMatch({error, {missing_evaluator, _}}, Result).

%%% ============================================================================
%%% Test: evaluate/3
%%% ============================================================================

evaluate_returns_fitness_and_metrics_test() ->
    {ok, Bridge} = create_test_bridge(),
    Network = fun(_Inputs) -> [0.6] end,
    Result = agent_trainer:evaluate(Bridge, Network, #{max_ticks => 5}),
    ?assertMatch({ok, Fitness, Metrics} when is_float(Fitness) andalso is_map(Metrics), Result).

evaluate_requires_evaluator_test() ->
    {ok, Bridge} = agent_bridge:new(#{
        definition => test_trainer_definition,
        sensors => [test_trainer_sensor],
        actuators => [test_trainer_actuator],
        environment => test_trainer_environment
    }),
    Network = fun(_Inputs) -> [0.5] end,
    Result = agent_trainer:evaluate(Bridge, Network, #{}),
    ?assertMatch({error, {missing_evaluator, _}}, Result).

%%% ============================================================================
%%% Test: evaluate_many/4
%%% ============================================================================

evaluate_many_averages_fitness_test() ->
    {ok, Bridge} = create_test_bridge(),
    Network = fun(_Inputs) -> [0.6] end,
    Result = agent_trainer:evaluate_many(Bridge, Network, #{max_ticks => 5}, 3),
    ?assertMatch({ok, AvgFitness, MetricsList}
                 when is_float(AvgFitness) andalso is_list(MetricsList), Result),
    {ok, _Fitness, MetricsList} = Result,
    ?assertEqual(3, length(MetricsList)).

%%% ============================================================================
%%% Test: train/2,3 (requires evaluator)
%%% ============================================================================

train_requires_evaluator_test() ->
    {ok, Bridge} = agent_bridge:new(#{
        definition => test_trainer_definition,
        sensors => [test_trainer_sensor],
        actuators => [test_trainer_actuator],
        environment => test_trainer_environment
    }),
    Result = agent_trainer:train(Bridge, #{generations => 1}),
    ?assertMatch({error, {missing_evaluator, _}}, Result).

%%% ============================================================================
%%% Test: Bridge with evaluator integration
%%% ============================================================================

bridge_with_evaluator_returns_fitness_test() ->
    {ok, Bridge} = create_test_bridge(),
    Network = fun(_Inputs) -> [0.6] end,
    Result = agent_bridge:run_episode(Bridge, Network, #{max_ticks => 5}),
    %% Should return {ok, Fitness, Metrics} because evaluator is configured
    ?assertMatch({ok, Fitness, Metrics} when is_float(Fitness) andalso is_map(Metrics), Result).

bridge_without_evaluator_returns_metrics_only_test() ->
    {ok, Bridge} = agent_bridge:new(#{
        definition => test_trainer_definition,
        sensors => [test_trainer_sensor],
        actuators => [test_trainer_actuator],
        environment => test_trainer_environment
        %% No evaluator
    }),
    Network = fun(_Inputs) -> [0.5] end,
    Result = agent_bridge:run_episode(Bridge, Network, #{max_ticks => 3}),
    %% Should return {ok, Metrics} only (backward compatible)
    ?assertMatch({ok, Metrics} when is_map(Metrics), Result).

bridge_validates_evaluator_test() ->
    {ok, Bridge} = create_test_bridge(),
    %% Bridge should have evaluator in validated state
    ?assert(maps:is_key(evaluator, Bridge)),
    ?assertEqual(test_trainer_evaluator, maps:get(evaluator, Bridge)).

%%% ============================================================================
%%% Helper Functions
%%% ============================================================================

create_test_bridge() ->
    agent_bridge:new(#{
        definition => test_trainer_definition,
        sensors => [test_trainer_sensor],
        actuators => [test_trainer_actuator],
        environment => test_trainer_environment,
        evaluator => test_trainer_evaluator
    }).
