%% @doc Test evaluator for agent_trainer tests.
-module(test_trainer_evaluator).
-behaviour(agent_evaluator).

-export([name/0, calculate_fitness/1]).

name() -> <<"test_fitness">>.

calculate_fitness(#{score := Score, ticks := Ticks}) ->
    Score * 10.0 + Ticks * 1.0.
