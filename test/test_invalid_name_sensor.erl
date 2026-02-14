%% @doc Test sensor with invalid name for agent_sensor tests.
-module(test_invalid_name_sensor).

-export([name/0, input_count/0, read/2]).

name() -> not_a_binary.  %% Invalid: should be binary
input_count() -> 4.
read(_AgentState, _EnvState) -> [0.0, 0.0, 0.0, 0.0].
