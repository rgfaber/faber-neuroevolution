%% @doc Valid test agent for agent_definition tests.
-module(test_valid_agent).
-behaviour(agent_definition).

-export([name/0, version/0, network_topology/0]).

name() -> <<"test_arena_agent">>.
version() -> <<"1.0.0">>.
network_topology() -> {29, [32, 16], 9}.
