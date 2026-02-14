%% @doc Test actuator for agent_trainer tests.
-module(test_trainer_actuator).
-behaviour(agent_actuator).

-export([name/0, output_count/0, act/3]).

name() -> <<"test_actuator">>.
output_count() -> 1.

act([Output], _AgentState, _EnvState) ->
    Action = if Output > 0.5 -> up; true -> down end,
    {ok, Action}.
