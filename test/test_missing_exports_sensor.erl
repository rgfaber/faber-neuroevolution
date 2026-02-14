%% @doc Test sensor with missing exports for agent_sensor tests.
-module(test_missing_exports_sensor).

%% Only exports name/0, missing input_count/0 and read/2
-export([name/0]).

name() -> <<"incomplete_sensor">>.
