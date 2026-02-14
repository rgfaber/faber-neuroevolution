%% @doc Test sensor with invalid input count for agent_sensor tests.
-module(test_invalid_input_count_sensor).

-export([name/0, input_count/0, read/2]).

name() -> <<"test_sensor">>.
input_count() -> "not_an_integer".  %% Invalid: should be integer
read(_AgentState, _EnvState) -> [0.0].
