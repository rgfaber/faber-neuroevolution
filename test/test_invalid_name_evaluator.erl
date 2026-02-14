%% @doc Test evaluator with invalid name for agent_evaluator tests.
-module(test_invalid_name_evaluator).

-export([name/0, calculate_fitness/1]).

name() -> not_a_binary.  %% Invalid: should be binary
calculate_fitness(_Metrics) -> 0.0.
