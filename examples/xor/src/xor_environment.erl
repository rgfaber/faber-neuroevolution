%% @doc XOR environment: the "hello world" of neuroevolution.
%%
%% Presents each of the four XOR cases exactly once per episode, in fixed
%% order, and accumulates squared error against the expected output.
%%
%% Encoding follows Gene Sher's xor_sim (Handbook of Neuroevolution Through
%% Erlang, Ch 7): inputs and targets are -1.0 / 1.0 rather than 0.0 / 1.0,
%% which matches the tanh output range of the default network.
%%
%%   [-1, -1] -> -1
%%   [ 1, -1] ->  1
%%   [-1,  1] ->  1
%%   [ 1,  1] -> -1
%%
%% == Episode ordering ==
%%
%% agent_bridge:episode_loop/5 runs is_terminal -> tick -> sense -> think ->
%% act -> apply_action. tick therefore selects the case that sense will read
%% and apply_action will score, and the episode ends when no cases remain.
%% With four cases this yields exactly four scored presentations.
%%
%% Fixed order, not shuffled: an evolutionary benchmark must be reproducible
%% for a given seed.
%%
%% @copyright 2024-2026 R.G. Lefever
%% @license Apache-2.0
-module(xor_environment).
-behaviour(agent_environment).

-export([name/0, init/1, spawn_agent/2, tick/2, apply_action/3,
         is_terminal/2, extract_metrics/2]).

%% Exported so the sensor, the evaluator and tests share one definition of
%% the problem rather than each restating it.
-export([cases/0, tolerance/0, case_count/0]).

-define(CASES, [
    {[-1.0, -1.0], -1.0},
    {[ 1.0, -1.0],  1.0},
    {[-1.0,  1.0],  1.0},
    {[ 1.0,  1.0], -1.0}
]).

%% A case counts as correct when the output is within this distance of the
%% target, i.e. unambiguously on the right side of zero.
-define(TOLERANCE, 0.5).

%% @doc Environment name.
name() -> <<"xor">>.

%% @doc The four XOR cases as {Inputs, Target}.
cases() -> ?CASES.

%% @doc Distance from target within which an output counts as correct.
tolerance() -> ?TOLERANCE.

%% @doc Number of cases presented per episode.
case_count() -> length(?CASES).

%% @doc Initialise. Takes no configuration: XOR has no parameters.
init(_Config) ->
    {ok, #{remaining  => ?CASES,
           current    => undefined,
           sse        => 0.0,
           correct    => 0,
           presented  => 0}}.

%% @doc XOR is single-agent and stateless per agent.
spawn_agent(Id, EnvState) ->
    {ok, #{id => Id}, EnvState}.

%% @doc Select the next case. Runs before sense, so this is what sense reads.
tick(AgentState, #{remaining := [Case | Rest]} = EnvState) ->
    {ok, AgentState, EnvState#{current => Case, remaining => Rest}}.

%% @doc Score the network's output against the current case.
apply_action(#{output := Output}, AgentState,
             #{current := {_Inputs, Target}, sse := Sse,
               correct := Correct, presented := Presented} = EnvState) ->
    Error = Target - Output,
    Correct1 = case abs(Error) < ?TOLERANCE of
                   true  -> Correct + 1;
                   false -> Correct
               end,
    {ok, AgentState, EnvState#{sse       => Sse + Error * Error,
                               correct   => Correct1,
                               presented => Presented + 1}}.

%% @doc Terminal once every case has been presented.
is_terminal(_AgentState, #{remaining := Remaining}) ->
    Remaining =:= [].

%% @doc Metrics consumed by xor_evaluator.
%%
%% solved is the honest success criterion: every case on the correct side of
%% zero by a clear margin. Fitness alone is a poor stopping signal because it
%% is continuous and unbounded.
extract_metrics(_AgentState, #{sse := Sse, correct := Correct, presented := Presented}) ->
    #{sse       => Sse,
      correct   => Correct,
      presented => Presented,
      cases     => length(?CASES),
      solved    => Correct =:= length(?CASES)}.
