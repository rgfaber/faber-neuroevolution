%% @doc Test vision sensor for agent_bridge tests.
%% 6 rays × 3 channels = 18 inputs
-module(test_bridge_vision_sensor).
-behaviour(agent_sensor).

-export([name/0, input_count/0, read/2]).

name() -> <<"vision">>.
input_count() -> 18.

read(_AgentState, _EnvState) ->
    %% Return 18 zeros (no visible objects)
    lists:duplicate(18, 0.0).
