%% @doc Unit tests for ecological_silo module.
%%
%% Tests niches, resource pools, and ecological dynamics.
%%
%% @author R.G. Lefever
%% @copyright 2024-2026 R.G. Lefever
-module(ecological_silo_tests).

-include_lib("eunit/include/eunit.hrl").

%%% ============================================================================
%%% Setup/Teardown
%%% ============================================================================

setup() ->
    case whereis(ecological_silo) of
        undefined -> ok;
        Pid ->
            catch gen_server:stop(Pid, normal, 1000),
            timer:sleep(50)
    end,
    ok.

cleanup(_) ->
    case whereis(ecological_silo) of
        undefined -> ok;
        Pid ->
            catch gen_server:stop(Pid, normal, 1000),
            timer:sleep(50)
    end,
    ok.

%%% ============================================================================
%%% Test Suite
%%% ============================================================================

ecological_silo_test_() ->
    {foreach,
     fun setup/0,
     fun cleanup/1,
     [
         {"start_link with default config", fun start_link_default_test/0},
         {"start_link with custom config", fun start_link_custom_config_test/0},
         {"get_params returns defaults", fun get_params_test/0},
         {"register_niche creates niche", fun register_niche_test/0},
         {"get_niche retrieves niche", fun get_niche_test/0},
         {"add_to_niche adds individuals", fun add_to_niche_test/0},
         {"remove_from_niche removes individuals", fun remove_from_niche_test/0},
         {"update_resource_pool tracks resources", fun update_resource_pool_test/0},
         {"get_ecological_stats returns metrics", fun get_ecological_stats_test/0},
         {"reset clears state", fun reset_test/0},
         {"get_state returns full state", fun get_state_test/0}
     ]}.

%%% ============================================================================
%%% Individual Tests
%%% ============================================================================

start_link_default_test() ->
    {ok, Pid} = ecological_silo:start_link(),
    ?assert(is_pid(Pid)),
    ?assert(is_process_alive(Pid)),
    ok = gen_server:stop(Pid).

start_link_custom_config_test() ->
    Config = #{
        realm => <<"test_realm">>,
        enabled_levels => [l0]
    },
    {ok, Pid} = ecological_silo:start_link(Config),
    ?assert(is_pid(Pid)),
    ok = gen_server:stop(Pid).

get_params_test() ->
    {ok, Pid} = ecological_silo:start_link(),
    Params = ecological_silo:get_params(Pid),
    ?assert(is_map(Params)),
    ?assert(maps:is_key(carrying_capacity, Params)),
    ?assert(maps:is_key(niche_formation_threshold, Params)),
    ?assert(maps:is_key(resource_regeneration_rate, Params)),
    ?assertEqual(100, maps:get(carrying_capacity, Params)),
    ?assertEqual(0.5, maps:get(niche_formation_threshold, Params)),
    ok = gen_server:stop(Pid).

register_niche_test() ->
    {ok, Pid} = ecological_silo:start_link(),

    %% Register niches
    ok = ecological_silo:register_niche(Pid, niche_1, #{capacity => 20}),
    ok = ecological_silo:register_niche(Pid, niche_2, #{capacity => 15, resource_type => food}),
    timer:sleep(50),

    %% Check stats
    Stats = ecological_silo:get_ecological_stats(Pid),
    ?assertEqual(2, maps:get(niche_count, Stats)),
    ok = gen_server:stop(Pid).

get_niche_test() ->
    {ok, Pid} = ecological_silo:start_link(),

    %% Register a niche
    ok = ecological_silo:register_niche(Pid, niche_1, #{capacity => 20}),
    timer:sleep(50),

    %% Retrieve it
    {ok, Niche} = ecological_silo:get_niche(Pid, niche_1),
    ?assert(is_map(Niche)),
    ?assertEqual(20, maps:get(capacity, Niche)),

    %% Non-existent niche
    ?assertEqual(not_found, ecological_silo:get_niche(Pid, non_existent)),
    ok = gen_server:stop(Pid).

add_to_niche_test() ->
    {ok, Pid} = ecological_silo:start_link(),

    %% Register a niche
    ok = ecological_silo:register_niche(Pid, niche_1, #{capacity => 20}),
    timer:sleep(50),

    %% Add individuals
    ok = ecological_silo:add_to_niche(Pid, niche_1, ind_1),
    ok = ecological_silo:add_to_niche(Pid, niche_1, ind_2),
    ok = ecological_silo:add_to_niche(Pid, niche_1, ind_3),
    timer:sleep(50),

    %% Check niche occupancy
    {ok, Niche} = ecological_silo:get_niche(Pid, niche_1),
    Occupants = maps:get(occupants, Niche, []),
    ?assertEqual(3, length(Occupants)),
    ok = gen_server:stop(Pid).

remove_from_niche_test() ->
    {ok, Pid} = ecological_silo:start_link(),

    %% Setup: register niche and add individuals
    ok = ecological_silo:register_niche(Pid, niche_1, #{}),
    ok = ecological_silo:add_to_niche(Pid, niche_1, ind_1),
    ok = ecological_silo:add_to_niche(Pid, niche_1, ind_2),
    timer:sleep(50),

    %% Remove one
    ok = ecological_silo:remove_from_niche(Pid, niche_1, ind_1),
    timer:sleep(50),

    %% Check
    {ok, Niche} = ecological_silo:get_niche(Pid, niche_1),
    Occupants = maps:get(occupants, Niche, []),
    ?assertEqual(1, length(Occupants)),
    ?assert(lists:member(ind_2, Occupants)),
    ok = gen_server:stop(Pid).

update_resource_pool_test() ->
    {ok, Pid} = ecological_silo:start_link(),

    %% Update resource pools
    ok = ecological_silo:update_resource_pool(Pid, food, 100.0),
    ok = ecological_silo:update_resource_pool(Pid, water, 50.0),
    timer:sleep(50),

    %% Check stats
    Stats = ecological_silo:get_ecological_stats(Pid),
    ?assertEqual(2, maps:get(resource_pool_count, Stats)),
    ok = gen_server:stop(Pid).

get_ecological_stats_test() ->
    {ok, Pid} = ecological_silo:start_link(),

    %% Empty stats
    Stats1 = ecological_silo:get_ecological_stats(Pid),
    ?assertEqual(0, maps:get(niche_count, Stats1)),
    ?assertEqual(0, maps:get(resource_pool_count, Stats1)),
    ?assertEqual(0, maps:get(extinction_count, Stats1)),

    %% Add some data
    ok = ecological_silo:register_niche(Pid, niche_1, #{}),
    ok = ecological_silo:update_resource_pool(Pid, food, 100.0),
    timer:sleep(50),

    Stats2 = ecological_silo:get_ecological_stats(Pid),
    ?assertEqual(1, maps:get(niche_count, Stats2)),
    ?assertEqual(1, maps:get(resource_pool_count, Stats2)),
    ok = gen_server:stop(Pid).

reset_test() ->
    {ok, Pid} = ecological_silo:start_link(),

    %% Add data
    ok = ecological_silo:register_niche(Pid, niche_1, #{}),
    ok = ecological_silo:update_resource_pool(Pid, food, 100.0),
    timer:sleep(50),

    %% Verify data exists
    Stats1 = ecological_silo:get_ecological_stats(Pid),
    ?assert(maps:get(niche_count, Stats1) > 0),

    %% Reset
    ok = ecological_silo:reset(Pid),

    %% Verify reset
    Stats2 = ecological_silo:get_ecological_stats(Pid),
    ?assertEqual(0, maps:get(niche_count, Stats2)),
    ?assertEqual(0, maps:get(resource_pool_count, Stats2)),
    ok = gen_server:stop(Pid).

get_state_test() ->
    {ok, Pid} = ecological_silo:start_link(#{realm => <<"test">>}),
    State = ecological_silo:get_state(Pid),
    ?assert(is_map(State)),
    ?assertEqual(<<"test">>, maps:get(realm, State)),
    ?assert(maps:is_key(current_params, State)),
    ?assert(maps:is_key(extinction_count, State)),
    ?assert(maps:is_key(sensors, State)),
    ok = gen_server:stop(Pid).

%%% ============================================================================
%%% Behavior Callback Tests
%%% ============================================================================

behavior_callbacks_test_() ->
    [
        {"get_silo_type returns ecological",
         ?_assertEqual(ecological, ecological_silo:get_silo_type())},
        {"get_time_constant returns 50.0",
         ?_assertEqual(50.0, ecological_silo:get_time_constant())}
    ].
