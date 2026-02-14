%% @doc Tests for agent_environment behaviour.
%%
%% Tests validation logic, callback contracts, and lifecycle
%% for the agent_environment behaviour module.
%%
%% @author R.G. Lefever
%% @copyright 2024-2026 R.G. Lefever
-module(agent_environment_tests).

-include_lib("eunit/include/eunit.hrl").

%%% ============================================================================
%%% Test Generators
%%% ============================================================================

agent_environment_test_() ->
    [
        {"validate/1 returns ok for valid module", fun test_validate_valid_module/0},
        {"validate/1 detects missing exports", fun test_validate_missing_exports/0},
        {"validate/1 detects invalid name", fun test_validate_invalid_name/0},
        {"validate/1 detects empty name", fun test_validate_empty_name/0},
        {"get_info/1 returns info for valid module", fun test_get_info_valid/0},
        {"get_info/1 returns error for invalid module", fun test_get_info_invalid/0}
    ].

lifecycle_test_() ->
    [
        {"init/1 creates environment state", fun test_init_creates_state/0},
        {"spawn_agent/2 creates agent state", fun test_spawn_agent_creates_state/0},
        {"tick/2 advances simulation", fun test_tick_advances/0},
        {"apply_action/3 updates state", fun test_apply_action_updates/0},
        {"is_terminal/2 detects end conditions", fun test_is_terminal/0},
        {"extract_metrics/2 returns metrics", fun test_extract_metrics/0}
    ].

episode_flow_test_() ->
    [
        {"complete episode runs successfully", fun test_complete_episode/0},
        {"episode terminates on max ticks", fun test_episode_max_ticks/0},
        {"episode terminates on agent death", fun test_episode_agent_death/0}
    ].

%%% ============================================================================
%%% Validation Test Cases
%%% ============================================================================

test_validate_valid_module() ->
    ?assertEqual(ok, agent_environment:validate(test_valid_environment)).

test_validate_missing_exports() ->
    Result = agent_environment:validate(test_missing_exports_environment),
    ?assertMatch({error, _}, Result),
    {error, Errors} = Result,
    ?assert(lists:any(
        fun({missing_exports, Missing}) ->
            lists:member({init, 1}, Missing) orelse
            lists:member({tick, 2}, Missing);
           (_) -> false
        end,
        Errors
    )).

test_validate_invalid_name() ->
    Result = agent_environment:validate(test_invalid_name_environment),
    ?assertMatch({error, _}, Result),
    {error, Errors} = Result,
    ?assert(lists:any(
        fun({invalid_name, _}) -> true;
           (_) -> false
        end,
        Errors
    )).

test_validate_empty_name() ->
    Result = agent_environment:validate(test_empty_name_environment),
    ?assertMatch({error, _}, Result),
    {error, Errors} = Result,
    ?assert(lists:any(
        fun({invalid_name, empty_binary}) -> true;
           (_) -> false
        end,
        Errors
    )).

test_get_info_valid() ->
    Result = agent_environment:get_info(test_valid_environment),
    ?assertMatch({ok, _}, Result),
    {ok, Info} = Result,
    ?assertEqual(<<"test_arena">>, maps:get(name, Info)).

test_get_info_invalid() ->
    Result = agent_environment:get_info(test_invalid_name_environment),
    ?assertMatch({error, _}, Result).

%%% ============================================================================
%%% Lifecycle Test Cases
%%% ============================================================================

test_init_creates_state() ->
    Config = #{max_ticks => 100, arena_radius => 5},
    {ok, EnvState} = test_valid_environment:init(Config),
    ?assertEqual(0, maps:get(tick, EnvState)),
    ?assertEqual(100, maps:get(max_ticks, EnvState)),
    ?assertEqual(5, maps:get(arena_radius, EnvState)).

test_spawn_agent_creates_state() ->
    {ok, EnvState} = test_valid_environment:init(#{}),
    {ok, AgentState, _NewEnvState} = test_valid_environment:spawn_agent(agent_1, EnvState),
    ?assertEqual(agent_1, maps:get(id, AgentState)),
    ?assertEqual({0, 0}, maps:get(pos, AgentState)),
    ?assertEqual(100.0, maps:get(energy, AgentState)).

test_tick_advances() ->
    {ok, EnvState} = test_valid_environment:init(#{}),
    {ok, AgentState, _} = test_valid_environment:spawn_agent(agent_1, EnvState),
    {ok, NewAgent, NewEnv} = test_valid_environment:tick(AgentState, EnvState),
    ?assertEqual(1, maps:get(tick, NewEnv)),
    ?assertEqual(1, maps:get(age, NewAgent)).

test_apply_action_updates() ->
    {ok, EnvState} = test_valid_environment:init(#{}),
    {ok, AgentState, _} = test_valid_environment:spawn_agent(agent_1, EnvState),
    Action = #{type => move, direction => 0},
    {ok, NewAgent, _NewEnv} = test_valid_environment:apply_action(Action, AgentState, EnvState),
    ?assertNotEqual(maps:get(pos, AgentState), maps:get(pos, NewAgent)).

test_is_terminal() ->
    {ok, EnvState} = test_valid_environment:init(#{max_ticks => 10}),
    {ok, AgentState, _} = test_valid_environment:spawn_agent(agent_1, EnvState),

    %% Not terminal at start
    ?assertEqual(false, test_valid_environment:is_terminal(AgentState, EnvState)),

    %% Terminal when max ticks reached
    EnvAtMax = EnvState#{tick => 10},
    ?assertEqual(true, test_valid_environment:is_terminal(AgentState, EnvAtMax)),

    %% Terminal when agent has no energy
    DeadAgent = AgentState#{energy => 0.0},
    ?assertEqual(true, test_valid_environment:is_terminal(DeadAgent, EnvState)).

test_extract_metrics() ->
    {ok, EnvState} = test_valid_environment:init(#{}),
    AgentState = #{
        id => agent_1,
        age => 50,
        food_eaten => 3,
        energy => 75.0
    },
    Metrics = test_valid_environment:extract_metrics(AgentState, EnvState),
    ?assertEqual(50, maps:get(ticks_survived, Metrics)),
    ?assertEqual(3, maps:get(food_eaten, Metrics)),
    ?assertEqual(75.0, maps:get(final_energy, Metrics)).

%%% ============================================================================
%%% Episode Flow Test Cases
%%% ============================================================================

test_complete_episode() ->
    %% Run a complete episode
    {ok, EnvState0} = test_valid_environment:init(#{max_ticks => 5}),
    {ok, Agent0, Env0} = test_valid_environment:spawn_agent(agent_1, EnvState0),

    %% Run 5 ticks
    {FinalAgent, FinalEnv} = run_episode(Agent0, Env0, test_valid_environment),

    %% Should have run to completion
    ?assertEqual(true, test_valid_environment:is_terminal(FinalAgent, FinalEnv)),

    %% Extract metrics
    Metrics = test_valid_environment:extract_metrics(FinalAgent, FinalEnv),
    ?assert(maps:get(ticks_survived, Metrics) >= 0).

test_episode_max_ticks() ->
    {ok, Env0} = test_valid_environment:init(#{max_ticks => 3}),
    {ok, Agent0, Env1} = test_valid_environment:spawn_agent(agent_1, Env0),

    {FinalAgent, FinalEnv} = run_episode(Agent0, Env1, test_valid_environment),

    ?assertEqual(true, test_valid_environment:is_terminal(FinalAgent, FinalEnv)),
    ?assertEqual(3, maps:get(tick, FinalEnv)).

test_episode_agent_death() ->
    {ok, Env0} = test_valid_environment:init(#{max_ticks => 100}),
    {ok, Agent0, Env1} = test_valid_environment:spawn_agent(agent_1, Env0),

    %% Set agent energy very low so it dies quickly
    DyingAgent = Agent0#{energy => 1.0},

    {FinalAgent, FinalEnv} = run_episode(DyingAgent, Env1, test_valid_environment),

    ?assertEqual(true, test_valid_environment:is_terminal(FinalAgent, FinalEnv)),
    %% Should terminate before max ticks
    ?assert(maps:get(tick, FinalEnv) < 100).

%%% ============================================================================
%%% Helper Functions
%%% ============================================================================

run_episode(Agent, Env, EnvModule) ->
    case EnvModule:is_terminal(Agent, Env) of
        true ->
            {Agent, Env};
        false ->
            %% Tick
            {ok, Agent1, Env1} = EnvModule:tick(Agent, Env),
            %% Apply a simple action
            Action = #{type => stay},
            {ok, Agent2, Env2} = EnvModule:apply_action(Action, Agent1, Env1),
            %% Continue
            run_episode(Agent2, Env2, EnvModule)
    end.
