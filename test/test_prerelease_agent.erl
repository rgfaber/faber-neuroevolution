%% @doc Test agent with prerelease version for agent_definition tests.
-module(test_prerelease_agent).
-behaviour(agent_definition).

-export([name/0, version/0, network_topology/0]).

name() -> <<"test_agent">>.
version() -> <<"1.0.0-alpha">>.  %% Valid: starts with X.Y.Z
network_topology() -> {4, [8], 2}.
