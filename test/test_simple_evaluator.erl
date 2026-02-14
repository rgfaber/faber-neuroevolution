%% @doc Simple test evaluator without fitness_components.
-module(test_simple_evaluator).
-behaviour(agent_evaluator).

-export([name/0, calculate_fitness/1]).

name() -> <<"simple_fitness">>.

calculate_fitness(Metrics) ->
    %% Simple fitness: just survival time
    maps:get(ticks_survived, Metrics, 0) * 1.0.
