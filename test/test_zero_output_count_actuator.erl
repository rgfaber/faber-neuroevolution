%% @doc Test actuator with zero output count for agent_actuator tests.
-module(test_zero_output_count_actuator).

-export([name/0, output_count/0, act/3]).

name() -> <<"test_actuator">>.
output_count() -> 0.  %% Invalid: must be positive
act(_Outputs, _AgentState, _EnvState) -> {ok, #{type => noop}}.
