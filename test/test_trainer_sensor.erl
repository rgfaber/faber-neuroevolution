%% @doc Test sensor for agent_trainer tests.
-module(test_trainer_sensor).
-behaviour(agent_sensor).

-export([name/0, input_count/0, read/2]).

name() -> <<"test_sensor">>.
input_count() -> 2.

read(_AgentState, EnvState) ->
    X = maps:get(x, EnvState, 0.5),
    Y = maps:get(y, EnvState, 0.5),
    [X, Y].
