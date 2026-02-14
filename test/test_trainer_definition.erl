%% @doc Test definition for agent_trainer tests.
-module(test_trainer_definition).
-behaviour(agent_definition).

-export([name/0, version/0, network_topology/0]).

name() -> <<"test_trainer_agent">>.
version() -> <<"1.0.0">>.
network_topology() -> {2, [4], 1}.  %% 2 inputs, 4 hidden, 1 output
