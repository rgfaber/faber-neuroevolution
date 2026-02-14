%% @doc Test actuator with invalid output count for agent_actuator tests.
-module(test_invalid_output_count_actuator).

-export([name/0, output_count/0, act/3]).

name() -> <<"test_actuator">>.
output_count() -> "not_an_integer".  %% Invalid: should be integer
act(_Outputs, _AgentState, _EnvState) -> {ok, #{type => noop}}.
