%% @doc Test agent definition for agent_bridge tests.
%% Topology: 23 inputs, [16] hidden, 8 outputs
-module(test_bridge_agent).
-behaviour(agent_definition).

-export([name/0, version/0, network_topology/0]).

name() -> <<"test_bridge_agent">>.
version() -> <<"1.0.0">>.

%% 23 inputs (18 vision + 4 hearing + 1 energy)
%% 8 outputs (7 movement + 1 signal)
network_topology() -> {23, [16], 8}.
