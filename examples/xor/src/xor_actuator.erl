%% @doc XOR actuator: forwards the network's single output for scoring.
%%
%% XOR has no discrete action; the output value itself is what gets compared
%% to the target. Passing the raw float through, rather than thresholding it
%% here, keeps the error signal continuous so evolution has a gradient to
%% climb instead of a step function.
%%
%% @copyright 2024-2026 R.G. Lefever
%% @license Apache-2.0
-module(xor_actuator).
-behaviour(agent_actuator).

-export([name/0, output_count/0, act/3]).

name() -> <<"xor_output">>.

output_count() -> 1.

act([Output], _AgentState, _EnvState) when is_number(Output) ->
    {ok, #{output => Output}};
act(Outputs, _AgentState, _EnvState) ->
    {error, {expected_one_output, Outputs}}.
