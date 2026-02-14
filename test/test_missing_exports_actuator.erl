%% @doc Test actuator with missing exports for agent_actuator tests.
-module(test_missing_exports_actuator).

%% Only exports name/0, missing output_count/0 and act/3
-export([name/0]).

name() -> <<"incomplete_actuator">>.
