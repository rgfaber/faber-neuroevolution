%% @doc Unit tests for social_silo module.
%%
%% Tests reputation, coalitions, and social network functionality.
%%
%% @author R.G. Lefever
%% @copyright 2024-2026 R.G. Lefever
-module(social_silo_tests).

-include_lib("eunit/include/eunit.hrl").

%%% ============================================================================
%%% Setup/Teardown
%%% ============================================================================

setup() ->
    case whereis(social_silo) of
        undefined -> ok;
        Pid ->
            catch gen_server:stop(Pid, normal, 1000),
            timer:sleep(50)
    end,
    ok.

cleanup(_) ->
    case whereis(social_silo) of
        undefined -> ok;
        Pid ->
            catch gen_server:stop(Pid, normal, 1000),
            timer:sleep(50)
    end,
    ok.

%%% ============================================================================
%%% Test Suite
%%% ============================================================================

social_silo_test_() ->
    {foreach,
     fun setup/0,
     fun cleanup/1,
     [
         {"start_link with default config", fun start_link_default_test/0},
         {"start_link with custom config", fun start_link_custom_config_test/0},
         {"get_params returns defaults", fun get_params_test/0},
         {"update_reputation modifies reputation", fun update_reputation_test/0},
         {"get_reputation returns value", fun get_reputation_test/0},
         {"record_interaction tracks interactions", fun record_interaction_test/0},
         {"form_coalition creates coalition", fun form_coalition_test/0},
         {"dissolve_coalition removes coalition", fun dissolve_coalition_test/0},
         {"get_social_metrics returns metrics", fun get_social_metrics_test/0},
         {"reset clears state", fun reset_test/0},
         {"get_state returns full state", fun get_state_test/0}
     ]}.

%%% ============================================================================
%%% Individual Tests
%%% ============================================================================

start_link_default_test() ->
    {ok, Pid} = social_silo:start_link(),
    ?assert(is_pid(Pid)),
    ?assert(is_process_alive(Pid)),
    ok = gen_server:stop(Pid).

start_link_custom_config_test() ->
    Config = #{
        realm => <<"test_realm">>,
        enabled_levels => [l0]
    },
    {ok, Pid} = social_silo:start_link(Config),
    ?assert(is_pid(Pid)),
    ok = gen_server:stop(Pid).

get_params_test() ->
    {ok, Pid} = social_silo:start_link(),
    Params = social_silo:get_params(Pid),
    ?assert(is_map(Params)),
    ?assert(maps:is_key(reputation_decay_rate, Params)),
    ?assert(maps:is_key(coalition_formation_threshold, Params)),
    ?assert(maps:is_key(mentoring_bonus, Params)),
    ?assert(maps:is_key(coalition_size_limit, Params)),
    ?assertEqual(10, maps:get(coalition_size_limit, Params)),
    ?assertEqual(0.05, maps:get(reputation_decay_rate, Params)),
    ok = gen_server:stop(Pid).

update_reputation_test() ->
    {ok, Pid} = social_silo:start_link(),

    %% Update reputation (creates if not exists)
    ok = social_silo:update_reputation(Pid, individual_1, 0.1),
    timer:sleep(50),

    %% Check reputation increased from default (0.5)
    {ok, Rep} = social_silo:get_reputation(Pid, individual_1),
    ?assertEqual(0.6, Rep),

    %% Update again
    ok = social_silo:update_reputation(Pid, individual_1, -0.2),
    timer:sleep(50),

    {ok, Rep2} = social_silo:get_reputation(Pid, individual_1),
    ?assert(Rep2 < Rep),
    ok = gen_server:stop(Pid).

get_reputation_test() ->
    {ok, Pid} = social_silo:start_link(),

    %% Non-existent individual
    ?assertEqual(not_found, social_silo:get_reputation(Pid, unknown_id)),

    %% After update
    ok = social_silo:update_reputation(Pid, ind_1, 0.0),
    timer:sleep(50),
    {ok, Rep} = social_silo:get_reputation(Pid, ind_1),
    ?assertEqual(0.5, Rep),  %% Default + 0.0
    ok = gen_server:stop(Pid).

record_interaction_test() ->
    {ok, Pid} = social_silo:start_link(),

    %% Record interactions
    ok = social_silo:record_interaction(Pid, ind_1, ind_2, cooperation),
    ok = social_silo:record_interaction(Pid, ind_2, ind_3, mentoring),
    ok = social_silo:record_interaction(Pid, ind_1, ind_3, cooperation),
    timer:sleep(50),

    %% Check metrics
    Metrics = social_silo:get_social_metrics(Pid),
    ?assertEqual(3, maps:get(interaction_count, Metrics)),
    ok = gen_server:stop(Pid).

form_coalition_test() ->
    {ok, Pid} = social_silo:start_link(),

    %% Form coalition
    {ok, coalition_1} = social_silo:form_coalition(Pid, coalition_1, [ind_1, ind_2, ind_3]),

    %% Verify coalition exists
    {ok, CoalitionData} = social_silo:get_coalition(Pid, coalition_1),
    ?assertEqual([ind_1, ind_2, ind_3], maps:get(members, CoalitionData)),

    %% Too small coalition
    ?assertEqual({error, need_at_least_two_members},
                 social_silo:form_coalition(Pid, coalition_2, [ind_4])),

    ok = gen_server:stop(Pid).

dissolve_coalition_test() ->
    {ok, Pid} = social_silo:start_link(),

    %% Form coalition
    {ok, coalition_1} = social_silo:form_coalition(Pid, coalition_1, [ind_1, ind_2]),

    %% Dissolve
    ok = social_silo:dissolve_coalition(Pid, coalition_1),

    %% Verify gone
    ?assertEqual(not_found, social_silo:get_coalition(Pid, coalition_1)),

    %% Dissolve non-existent
    ?assertEqual(not_found, social_silo:dissolve_coalition(Pid, non_existent)),
    ok = gen_server:stop(Pid).

get_social_metrics_test() ->
    {ok, Pid} = social_silo:start_link(),

    %% Empty metrics
    Metrics1 = social_silo:get_social_metrics(Pid),
    ?assertEqual(0, maps:get(reputation_count, Metrics1)),
    ?assertEqual(0, maps:get(coalition_count, Metrics1)),
    ?assertEqual(0, maps:get(interaction_count, Metrics1)),

    %% Add data
    ok = social_silo:update_reputation(Pid, ind_1, 0.1),
    ok = social_silo:update_reputation(Pid, ind_2, 0.2),
    {ok, _} = social_silo:form_coalition(Pid, coal_1, [ind_1, ind_2]),
    ok = social_silo:record_interaction(Pid, ind_1, ind_2, cooperation),
    timer:sleep(50),

    Metrics2 = social_silo:get_social_metrics(Pid),
    ?assertEqual(2, maps:get(reputation_count, Metrics2)),
    ?assertEqual(1, maps:get(coalition_count, Metrics2)),
    ?assertEqual(1, maps:get(interaction_count, Metrics2)),
    ok = gen_server:stop(Pid).

reset_test() ->
    {ok, Pid} = social_silo:start_link(),

    %% Add data
    ok = social_silo:update_reputation(Pid, ind_1, 0.1),
    {ok, _} = social_silo:form_coalition(Pid, coal_1, [ind_1, ind_2]),
    ok = social_silo:record_interaction(Pid, ind_1, ind_2, cooperation),
    timer:sleep(50),

    %% Verify data exists
    Metrics1 = social_silo:get_social_metrics(Pid),
    ?assert(maps:get(reputation_count, Metrics1) > 0),

    %% Reset
    ok = social_silo:reset(Pid),

    %% Verify reset
    Metrics2 = social_silo:get_social_metrics(Pid),
    ?assertEqual(0, maps:get(reputation_count, Metrics2)),
    ?assertEqual(0, maps:get(coalition_count, Metrics2)),
    ?assertEqual(0, maps:get(interaction_count, Metrics2)),
    ok = gen_server:stop(Pid).

get_state_test() ->
    {ok, Pid} = social_silo:start_link(#{realm => <<"test">>}),
    State = social_silo:get_state(Pid),
    ?assert(is_map(State)),
    ?assertEqual(<<"test">>, maps:get(realm, State)),
    ?assert(maps:is_key(current_params, State)),
    ?assert(maps:is_key(interaction_count, State)),
    ?assert(maps:is_key(sensors, State)),
    ok = gen_server:stop(Pid).

%%% ============================================================================
%%% Behavior Callback Tests
%%% ============================================================================

behavior_callbacks_test_() ->
    [
        {"get_silo_type returns social",
         ?_assertEqual(social, social_silo:get_silo_type())},
        {"get_time_constant returns 25.0",
         ?_assertEqual(25.0, social_silo:get_time_constant())}
    ].
