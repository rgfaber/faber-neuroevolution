%% @doc Test sensor with empty name for agent_sensor tests.
-module(test_empty_name_sensor).

-export([name/0, input_count/0, read/2]).

name() -> <<>>.  %% Invalid: empty binary
input_count() -> 4.
read(_AgentState, _EnvState) -> [0.0, 0.0, 0.0, 0.0].
