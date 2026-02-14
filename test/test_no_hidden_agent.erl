%% @doc Test agent with no hidden layers for agent_definition tests.
-module(test_no_hidden_agent).
-behaviour(agent_definition).

-export([name/0, version/0, network_topology/0]).

name() -> <<"simple_agent">>.
version() -> <<"1.0.0">>.
network_topology() -> {4, [], 2}.  %% Valid: direct connections, no hidden layers
