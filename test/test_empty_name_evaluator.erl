%% @doc Test evaluator with empty name for agent_evaluator tests.
-module(test_empty_name_evaluator).

-export([name/0, calculate_fitness/1]).

name() -> <<>>.  %% Invalid: empty binary
calculate_fitness(_Metrics) -> 0.0.
