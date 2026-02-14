%% @doc Test evaluator with missing exports for agent_evaluator tests.
-module(test_missing_exports_evaluator).

%% Only exports name/0, missing calculate_fitness/1
-export([name/0]).

name() -> <<"incomplete_evaluator">>.
