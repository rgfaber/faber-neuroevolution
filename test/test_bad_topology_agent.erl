%% @doc Test agent with bad topology for agent_definition tests.
-module(test_bad_topology_agent).

-export([name/0, version/0, network_topology/0]).

name() -> <<"test_agent">>.
version() -> <<"1.0.0">>.
network_topology() -> {0, [8], 4}.  %% Invalid: 0 inputs
