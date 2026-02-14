%% @doc Valid test sensor for agent_sensor tests.
-module(test_valid_sensor).
-behaviour(agent_sensor).

-export([name/0, input_count/0, read/2]).

name() -> <<"vision">>.

%% 6 rays × 3 channels = 18 inputs
input_count() -> 18.

read(_AgentState, _EnvState) ->
    %% Return 18 normalized values (simulated vision)
    lists:duplicate(18, 0.0).
