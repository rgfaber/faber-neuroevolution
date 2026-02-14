%% @doc Test signal actuator for agent_bridge tests.
%% 1 output: signal strength
-module(test_bridge_signal_actuator).
-behaviour(agent_actuator).

-export([name/0, output_count/0, act/3]).

name() -> <<"signal">>.
output_count() -> 1.

act([Strength], _AgentState, _EnvState) ->
    %% Emit signal with given strength
    {ok, #{type => signal, strength => Strength}}.
