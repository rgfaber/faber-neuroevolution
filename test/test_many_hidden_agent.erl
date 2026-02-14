%% @doc Test agent with many hidden layers for agent_definition tests.
-module(test_many_hidden_agent).
-behaviour(agent_definition).

-export([name/0, version/0, network_topology/0]).

name() -> <<"deep_agent">>.
version() -> <<"1.0.0">>.
network_topology() -> {100, [64, 64, 32, 32, 16, 8], 10}.  %% Valid: deep network
