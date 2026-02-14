%% @doc Test actuator with empty name for agent_actuator tests.
-module(test_empty_name_actuator).

-export([name/0, output_count/0, act/3]).

name() -> <<>>.  %% Invalid: empty binary
output_count() -> 2.
act(_Outputs, _AgentState, _EnvState) -> {ok, #{type => noop}}.
