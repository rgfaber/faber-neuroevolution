%% @doc Unit tests for temporal_silo module.
%%
%% Tests temporal management and episode timing functionality.
%%
%% @author R.G. Lefever
%% @copyright 2024-2026 R.G. Lefever
-module(temporal_silo_tests).

-include_lib("eunit/include/eunit.hrl").

%%% ============================================================================
%%% Setup/Teardown
%%% ============================================================================

setup() ->
    %% Ensure any existing temporal_silo is stopped
    case whereis(temporal_silo) of
        undefined -> ok;
        Pid ->
            catch gen_server:stop(Pid, normal, 1000),
            timer:sleep(50)
    end,
    ok.

cleanup(_) ->
    case whereis(temporal_silo) of
        undefined -> ok;
        Pid ->
            catch gen_server:stop(Pid, normal, 1000),
            timer:sleep(50)
    end,
    ok.

%%% ============================================================================
%%% Test Suite
%%% ============================================================================

temporal_silo_test_() ->
    {foreach,
     fun setup/0,
     fun cleanup/1,
     [
         {"start_link with default config", fun start_link_default_test/0},
         {"start_link with custom config", fun start_link_custom_config_test/0},
         {"get_params returns defaults", fun get_params_test/0},
         {"update_episode records episode", fun update_episode_test/0},
         {"record_reaction_time tracks times", fun record_reaction_time_test/0},
         {"record_timeout increments count", fun record_timeout_test/0},
         {"record_early_termination increments count", fun record_early_termination_test/0},
         {"reset clears state", fun reset_test/0},
         {"get_state returns full state", fun get_state_test/0}
     ]}.

%%% ============================================================================
%%% Individual Tests
%%% ============================================================================

start_link_default_test() ->
    {ok, Pid} = temporal_silo:start_link(),
    ?assert(is_pid(Pid)),
    ?assert(is_process_alive(Pid)),
    ok = gen_server:stop(Pid).

start_link_custom_config_test() ->
    Config = #{
        realm => <<"test_realm">>,
        enabled_levels => [l0]
    },
    {ok, Pid} = temporal_silo:start_link(Config),
    ?assert(is_pid(Pid)),
    ok = gen_server:stop(Pid).

get_params_test() ->
    {ok, Pid} = temporal_silo:start_link(),
    Params = temporal_silo:get_params(Pid),
    ?assert(is_map(Params)),
    ?assert(maps:is_key(episode_length_target, Params)),
    ?assert(maps:is_key(evaluation_timeout_ms, Params)),
    ?assert(maps:is_key(learning_rate_multiplier, Params)),
    ?assertEqual(1000, maps:get(episode_length_target, Params)),
    ok = gen_server:stop(Pid).

update_episode_test() ->
    {ok, Pid} = temporal_silo:start_link(),
    %% Record some episodes
    ok = temporal_silo:update_episode(Pid, 100, 0.5),
    ok = temporal_silo:update_episode(Pid, 150, 0.6),
    ok = temporal_silo:update_episode(Pid, 120, 0.55),
    timer:sleep(50),  %% Allow casts to process

    State = temporal_silo:get_state(Pid),
    ?assertEqual(3, maps:get(episode_count, State)),
    ok = gen_server:stop(Pid).

record_reaction_time_test() ->
    {ok, Pid} = temporal_silo:start_link(),
    ok = temporal_silo:record_reaction_time(Pid, 50),
    ok = temporal_silo:record_reaction_time(Pid, 75),
    ok = temporal_silo:record_reaction_time(Pid, 60),
    timer:sleep(50),

    State = temporal_silo:get_state(Pid),
    Sensors = maps:get(sensors, State),
    %% Reaction time mean should be recorded
    ?assert(maps:is_key(reaction_time_mean, Sensors)),
    ok = gen_server:stop(Pid).

record_timeout_test() ->
    {ok, Pid} = temporal_silo:start_link(),
    ok = temporal_silo:record_timeout(Pid),
    ok = temporal_silo:record_timeout(Pid),
    timer:sleep(50),

    State = temporal_silo:get_state(Pid),
    ?assertEqual(2, maps:get(timeout_count, State)),
    ok = gen_server:stop(Pid).

record_early_termination_test() ->
    {ok, Pid} = temporal_silo:start_link(),
    ok = temporal_silo:record_early_termination(Pid),
    ok = temporal_silo:record_early_termination(Pid),
    ok = temporal_silo:record_early_termination(Pid),
    timer:sleep(50),

    State = temporal_silo:get_state(Pid),
    ?assertEqual(3, maps:get(early_termination_count, State)),
    ok = gen_server:stop(Pid).

reset_test() ->
    {ok, Pid} = temporal_silo:start_link(),
    %% Add some data
    ok = temporal_silo:update_episode(Pid, 100, 0.5),
    ok = temporal_silo:record_timeout(Pid),
    timer:sleep(50),

    %% Reset
    ok = temporal_silo:reset(Pid),

    %% Verify reset
    State = temporal_silo:get_state(Pid),
    ?assertEqual(0, maps:get(episode_count, State)),
    ?assertEqual(0, maps:get(timeout_count, State)),
    ok = gen_server:stop(Pid).

get_state_test() ->
    {ok, Pid} = temporal_silo:start_link(#{realm => <<"test">>}),
    State = temporal_silo:get_state(Pid),
    ?assert(is_map(State)),
    ?assertEqual(<<"test">>, maps:get(realm, State)),
    ?assert(maps:is_key(current_params, State)),
    ?assert(maps:is_key(sensors, State)),
    ok = gen_server:stop(Pid).

%%% ============================================================================
%%% Behavior Callback Tests
%%% ============================================================================

behavior_callbacks_test_() ->
    [
        {"get_silo_type returns temporal",
         ?_assertEqual(temporal, temporal_silo:get_silo_type())},
        {"get_time_constant returns 10.0",
         ?_assertEqual(10.0, temporal_silo:get_time_constant())}
    ].
