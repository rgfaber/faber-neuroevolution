%% @doc Test agent with missing exports for agent_definition tests.
-module(test_missing_exports_agent).

%% Only exports name/0, missing version/0 and network_topology/0
-export([name/0]).

name() -> <<"test_agent">>.
