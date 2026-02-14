%% @doc Test hearing sensor for agent_bridge tests.
%% 4 directional hearing inputs
-module(test_bridge_hearing_sensor).
-behaviour(agent_sensor).

-export([name/0, input_count/0, read/2]).

name() -> <<"hearing">>.
input_count() -> 4.

read(_AgentState, _EnvState) ->
    %% Return 4 zeros (no sounds)
    lists:duplicate(4, 0.0).
