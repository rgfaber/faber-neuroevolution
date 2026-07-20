-module(xor_seed_tests).
-include_lib("eunit/include/eunit.hrl").

bridge() ->
    agent_bridge:new(#{definition => xor_definition, sensors => [xor_sensor],
        actuators => [xor_actuator], environment => xor_environment,
        evaluator => xor_evaluator}).

run(Seed) ->
    {ok, B} = bridge(),
    {ok, Net, _} = agent_trainer:train(B, #{}, #{
        generations => 15, population_size => 20, rng_seed => Seed}),
    {ok, F, M} = agent_trainer:evaluate(B, Net, #{}),
    {F, maps:get(sse, M)}.

same_seed_reproduces_test_() ->
    {timeout, 300, fun() ->
        A = run(42), B = run(42),
        ?assertEqual(A, B)
    end}.

different_seed_diverges_test_() ->
    {timeout, 300, fun() ->
        A = run(42), B = run(99),
        ?assertNotEqual(A, B)
    end}.
