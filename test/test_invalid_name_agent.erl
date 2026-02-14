%% @doc Test agent with invalid name for agent_definition tests.
-module(test_invalid_name_agent).

-export([name/0, version/0, network_topology/0]).

name() -> not_a_binary.  %% Invalid: should be binary
version() -> <<"1.0.0">>.
network_topology() -> {10, [8], 4}.
