%% @doc Test agent with empty name for agent_definition tests.
-module(test_empty_name_agent).

-export([name/0, version/0, network_topology/0]).

name() -> <<>>.  %% Invalid: empty binary
version() -> <<"1.0.0">>.
network_topology() -> {10, [8], 4}.
