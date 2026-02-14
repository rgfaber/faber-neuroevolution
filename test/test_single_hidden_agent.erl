%% @doc Test agent with single hidden layer for agent_definition tests.
-module(test_single_hidden_agent).
-behaviour(agent_definition).

-export([name/0, version/0, network_topology/0]).

name() -> <<"simple_agent">>.
version() -> <<"1.0.0">>.
network_topology() -> {4, [8], 2}.  %% Valid: single hidden layer
