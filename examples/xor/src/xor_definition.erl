%% @doc XOR agent definition.
%%
%% Topology is {2, [3], 1}: two inputs, one hidden layer of three, one output.
%%
%% XOR is not linearly separable, so at least one hidden unit is required and
%% two is the theoretical minimum. Three gives evolution a little slack
%% without making the search trivial.
%%
%% @copyright 2024-2026 R.G. Lefever
%% @license Apache-2.0
-module(xor_definition).
-behaviour(agent_definition).

-export([name/0, version/0, network_topology/0]).

name() -> <<"xor_solver">>.

version() -> <<"1.0.0">>.

network_topology() -> {2, [3], 1}.
