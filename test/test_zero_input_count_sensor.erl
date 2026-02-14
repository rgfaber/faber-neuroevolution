%% @doc Test sensor with zero input count for agent_sensor tests.
-module(test_zero_input_count_sensor).

-export([name/0, input_count/0, read/2]).

name() -> <<"test_sensor">>.
input_count() -> 0.  %% Invalid: must be positive
read(_AgentState, _EnvState) -> [].
