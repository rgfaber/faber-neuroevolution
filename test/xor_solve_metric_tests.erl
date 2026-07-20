%% @doc evaluations-to-solve: the metric the literature reports.
-module(xor_solve_metric_tests).
-include_lib("eunit/include/eunit.hrl").

bridge() ->
    agent_bridge:new(#{definition => xor_definition, sensors => [xor_sensor],
        actuators => [xor_actuator], environment => xor_environment,
        evaluator => xor_evaluator}).

train(Opts) ->
    {ok, B} = bridge(),
    {ok, _Net, Stats} = agent_trainer:train(B, #{}, maps:merge(
        #{generations => 60, population_size => 50}, Opts)),
    Stats.

%% The evaluator's is_solved/1 is what makes the metric possible at all.
evaluator_declares_solved_test() ->
    {module, _} = code:ensure_loaded(xor_evaluator),
    ?assert(erlang:function_exported(xor_evaluator, is_solved, 1)),
    ?assertEqual(true,  xor_evaluator:is_solved(#{correct => 4, cases => 4})),
    ?assertEqual(false, xor_evaluator:is_solved(#{correct => 3, cases => 4})).

records_evaluations_to_solve_test_() ->
    {timeout, 300, fun() ->
        Stats = train(#{rng_seed => 7}),
        N = maps:get(evaluations_to_solve, Stats),
        ?assert(is_integer(N), "task was never solved, so no count was recorded"),
        ?assert(N > 0)
    end}.

%% Reproducibility of the metric itself, not merely of the champion.
same_seed_same_evaluations_to_solve_test_() ->
    {timeout, 600, fun() ->
        A = maps:get(evaluations_to_solve, train(#{rng_seed => 7})),
        B = maps:get(evaluations_to_solve, train(#{rng_seed => 7})),
        ?assertEqual(A, B)
    end}.

%% stop_on_solved must halt earlier than running to max_generations.
stop_on_solved_halts_early_test_() ->
    {timeout, 600, fun() ->
        Full  = train(#{rng_seed => 7}),
        Early = train(#{rng_seed => 7, stop_on_solved => true}),
        ?assertEqual(maps:get(evaluations_to_solve, Full),
                     maps:get(evaluations_to_solve, Early)),
        ?assert(maps:get(generation, Early) =< maps:get(generation, Full))
    end}.
