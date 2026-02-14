%% @doc Test actuator with invalid name for agent_actuator tests.
-module(test_invalid_name_actuator).

-export([name/0, output_count/0, act/3]).

name() -> not_a_binary.  %% Invalid: should be binary
output_count() -> 2.
act(_Outputs, _AgentState, _EnvState) -> {ok, #{type => noop}}.
