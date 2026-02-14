%% @doc Test environment with empty name for agent_environment tests.
-module(test_empty_name_environment).

-export([name/0, init/1, spawn_agent/2, tick/2, apply_action/3,
         is_terminal/2, extract_metrics/2]).

name() -> <<>>.  %% Invalid: empty binary

init(_Config) -> {ok, #{}}.
spawn_agent(_Id, Env) -> {ok, #{}, Env}.
tick(Agent, Env) -> {ok, Agent, Env}.
apply_action(_Action, Agent, Env) -> {ok, Agent, Env}.
is_terminal(_Agent, _Env) -> false.
extract_metrics(_Agent, _Env) -> #{}.
