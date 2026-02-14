%% @doc Valid test environment for agent_environment tests.
-module(test_valid_environment).
-behaviour(agent_environment).

-export([name/0, init/1, spawn_agent/2, tick/2, apply_action/3,
         is_terminal/2, extract_metrics/2]).

name() -> <<"test_arena">>.

init(Config) ->
    {ok, #{
        tick => 0,
        max_ticks => maps:get(max_ticks, Config, 500),
        arena_radius => maps:get(arena_radius, Config, 10),
        walls => #{},
        food => #{}
    }}.

spawn_agent(AgentId, EnvState) ->
    Agent = #{
        id => AgentId,
        pos => {0, 0},
        energy => 100.0,
        age => 0,
        food_eaten => 0
    },
    {ok, Agent, EnvState}.

tick(AgentState, EnvState) ->
    %% Advance tick counter
    NewTick = maps:get(tick, EnvState) + 1,

    %% Age agent and decay energy
    NewAge = maps:get(age, AgentState, 0) + 1,
    Energy = maps:get(energy, AgentState, 100.0),
    NewEnergy = Energy - 0.5,  %% Energy decay per tick

    NewAgent = AgentState#{age => NewAge, energy => NewEnergy},
    NewEnv = EnvState#{tick => NewTick},

    {ok, NewAgent, NewEnv}.

apply_action(#{type := move, direction := Dir}, AgentState, EnvState) ->
    %% Simple movement: compute new position
    {X, Y} = maps:get(pos, AgentState),
    NewPos = case Dir of
        0 -> {X + 1, Y};      %% East
        1 -> {X + 1, Y - 1};  %% NE
        2 -> {X, Y - 1};      %% NW
        3 -> {X - 1, Y};      %% West
        4 -> {X - 1, Y + 1};  %% SW
        5 -> {X, Y + 1};      %% SE
        _ -> {X, Y}           %% Stay
    end,
    {ok, AgentState#{pos => NewPos}, EnvState};

apply_action(#{type := stay}, AgentState, EnvState) ->
    {ok, AgentState, EnvState};

apply_action(_Action, AgentState, EnvState) ->
    {ok, AgentState, EnvState}.

is_terminal(AgentState, EnvState) ->
    Energy = maps:get(energy, AgentState, 0),
    Tick = maps:get(tick, EnvState, 0),
    MaxTicks = maps:get(max_ticks, EnvState, 500),
    Energy =< 0 orelse Tick >= MaxTicks.

extract_metrics(AgentState, _EnvState) ->
    #{
        ticks_survived => maps:get(age, AgentState, 0),
        food_eaten => maps:get(food_eaten, AgentState, 0),
        final_energy => maps:get(energy, AgentState, 0)
    }.
