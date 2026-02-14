%% @doc Test environment for agent_trainer tests.
-module(test_trainer_environment).
-behaviour(agent_environment).

-export([name/0, init/1, spawn_agent/2, tick/2, apply_action/3, is_terminal/2, extract_metrics/2]).

name() -> <<"test_env">>.

init(Config) ->
    MaxTicks = maps:get(max_ticks, Config, 10),
    {ok, #{tick => 0, max_ticks => MaxTicks, x => 0.5, y => 0.5, score => 0}}.

spawn_agent(Id, EnvState) ->
    {ok, #{id => Id, actions => []}, EnvState}.

tick(AgentState, #{tick := T} = EnvState) ->
    {ok, AgentState, EnvState#{tick => T + 1}}.

apply_action(Action, #{actions := Actions} = AgentState, #{score := S} = EnvState) ->
    NewScore = case Action of up -> S + 1; down -> S end,
    {ok, AgentState#{actions => [Action | Actions]}, EnvState#{score => NewScore}}.

is_terminal(_AgentState, #{tick := T, max_ticks := Max}) ->
    T >= Max.

extract_metrics(#{actions := Actions}, #{score := Score, tick := Ticks}) ->
    #{score => Score, ticks => Ticks, actions => length(Actions)}.
