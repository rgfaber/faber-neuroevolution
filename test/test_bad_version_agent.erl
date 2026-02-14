%% @doc Test agent with bad version for agent_definition tests.
-module(test_bad_version_agent).

-export([name/0, version/0, network_topology/0]).

name() -> <<"test_agent">>.
version() -> <<"not-semver">>.  %% Invalid: not X.Y.Z format
network_topology() -> {10, [8], 4}.
