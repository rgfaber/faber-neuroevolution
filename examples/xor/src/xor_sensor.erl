%% @doc XOR sensor: reads the two inputs of the current case.
%%
%% Deliberately has no fall-through clause. agent_bridge:episode_loop/5 always
%% calls tick before sense, so `current' is set whenever read/2 runs. If that
%% invariant ever breaks, this crashes with a function_clause naming this
%% module, which is what we want. A defensive clause returning zeros would
%% instead feed the network silent garbage and quietly corrupt fitness, which
%% is exactly the failure mode faber-tweann's sensor.erl shipped with.
%%
%% @copyright 2024-2026 R.G. Lefever
%% @license Apache-2.0
-module(xor_sensor).
-behaviour(agent_sensor).

-export([name/0, input_count/0, read/2]).

name() -> <<"xor_inputs">>.

input_count() -> 2.

read(_AgentState, #{current := {Inputs, _Target}}) ->
    Inputs.
