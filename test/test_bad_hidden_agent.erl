%% @doc Test agent with bad hidden layers for agent_definition tests.
-module(test_bad_hidden_agent).

-export([name/0, version/0, network_topology/0]).

name() -> <<"test_agent">>.
version() -> <<"1.0.0">>.
network_topology() -> {10, [8, -1, 4], 4}.  %% Invalid: -1 in hidden layers
