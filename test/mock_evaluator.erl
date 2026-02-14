%% @doc Mock evaluator for testing.
-module(mock_evaluator).
-behaviour(neuroevolution_evaluator).

-include("neuroevolution.hrl").

-export([evaluate/2, calculate_fitness/1]).

%% @doc Simple evaluation: assigns random fitness.
evaluate(Individual, _Options) ->
    Score = rand:uniform() * 100,
    UpdatedIndividual = Individual#individual{
        metrics = #{total_score => Score}
    },
    {ok, UpdatedIndividual}.

%% @doc Extract fitness from metrics.
calculate_fitness(Metrics) ->
    maps:get(total_score, Metrics, 0.0).
