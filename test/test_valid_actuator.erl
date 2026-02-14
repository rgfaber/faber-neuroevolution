%% @doc Valid test actuator for agent_actuator tests.
-module(test_valid_actuator).
-behaviour(agent_actuator).

-export([name/0, output_count/0, act/3]).

name() -> <<"movement">>.

%% 6 hex directions + 1 stay = 7 outputs
output_count() -> 7.

act(Outputs, _AgentState, _EnvState) ->
    %% Find direction with highest activation
    {BestIdx, _MaxVal} = find_max_with_index(Outputs, 0, -999.0, 0),
    case BestIdx of
        6 -> {ok, #{type => stay}};
        Dir -> {ok, #{type => move, direction => Dir}}
    end.

%% Helper: find index of maximum value
find_max_with_index([], _Idx, _Max, MaxIdx) ->
    {MaxIdx, _Max};
find_max_with_index([V | Rest], Idx, Max, _MaxIdx) when V > Max ->
    find_max_with_index(Rest, Idx + 1, V, Idx);
find_max_with_index([_V | Rest], Idx, Max, MaxIdx) ->
    find_max_with_index(Rest, Idx + 1, Max, MaxIdx).
