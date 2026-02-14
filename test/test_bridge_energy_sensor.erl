%% @doc Test energy sensor for agent_bridge tests.
%% 1 input: current energy level (normalized)
-module(test_bridge_energy_sensor).
-behaviour(agent_sensor).

-export([name/0, input_count/0, read/2]).

name() -> <<"energy">>.
input_count() -> 1.

read(AgentState, _EnvState) ->
    %% Return normalized energy (0.0 to 1.0)
    Energy = maps:get(energy, AgentState, 100.0),
    MaxEnergy = 100.0,
    [Energy / MaxEnergy].
