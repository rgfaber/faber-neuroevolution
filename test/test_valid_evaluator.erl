%% @doc Valid test evaluator for agent_evaluator tests.
%% Implements full evaluator with fitness_components.
-module(test_valid_evaluator).
-behaviour(agent_evaluator).

-export([name/0, calculate_fitness/1, fitness_components/1]).

name() -> <<"hex_arena_fitness">>.

calculate_fitness(Metrics) ->
    Survival = maps:get(ticks_survived, Metrics, 0) * 0.1,
    Food = maps:get(food_eaten, Metrics, 0) * 150.0,
    Kills = maps:get(kills, Metrics, 0) * 100.0,
    Survival + Food + Kills.

fitness_components(Metrics) ->
    #{
        survival => maps:get(ticks_survived, Metrics, 0) * 0.1,
        food => maps:get(food_eaten, Metrics, 0) * 150.0,
        kills => maps:get(kills, Metrics, 0) * 100.0
    }.
