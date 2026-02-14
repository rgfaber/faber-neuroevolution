%% @doc Unit tests for distribution_silo module.
%%
%% Tests island management, migration, and load balancing functionality.
%%
%% @author R.G. Lefever
%% @copyright 2024-2026 R.G. Lefever
-module(distribution_silo_tests).

-include_lib("eunit/include/eunit.hrl").

%%% ============================================================================
%%% Setup/Teardown
%%% ============================================================================

setup() ->
    case whereis(distribution_silo) of
        undefined -> ok;
        Pid ->
            catch gen_server:stop(Pid, normal, 1000),
            timer:sleep(50)
    end,
    ok.

cleanup(_) ->
    case whereis(distribution_silo) of
        undefined -> ok;
        Pid ->
            catch gen_server:stop(Pid, normal, 1000),
            timer:sleep(50)
    end,
    ok.

%%% ============================================================================
%%% Test Suite
%%% ============================================================================

distribution_silo_test_() ->
    {foreach,
     fun setup/0,
     fun cleanup/1,
     [
         {"start_link with default config", fun start_link_default_test/0},
         {"start_link with custom config", fun start_link_custom_config_test/0},
         {"get_params returns defaults", fun get_params_test/0},
         {"register_island creates island", fun register_island_test/0},
         {"update_island_stats updates stats", fun update_island_stats_test/0},
         {"get_island retrieves island", fun get_island_test/0},
         {"record_migration tracks migrations", fun record_migration_test/0},
         {"get_migrations returns history", fun get_migrations_test/0},
         {"get_distribution_stats returns metrics", fun get_distribution_stats_test/0},
         {"reset clears state", fun reset_test/0},
         {"get_state returns full state", fun get_state_test/0}
     ]}.

%%% ============================================================================
%%% Individual Tests
%%% ============================================================================

start_link_default_test() ->
    {ok, Pid} = distribution_silo:start_link(),
    ?assert(is_pid(Pid)),
    ?assert(is_process_alive(Pid)),
    ok = gen_server:stop(Pid).

start_link_custom_config_test() ->
    Config = #{
        realm => <<"test_realm">>,
        enabled_levels => [l0]
    },
    {ok, Pid} = distribution_silo:start_link(Config),
    ?assert(is_pid(Pid)),
    ok = gen_server:stop(Pid).

get_params_test() ->
    {ok, Pid} = distribution_silo:start_link(),
    Params = distribution_silo:get_params(Pid),
    ?assert(is_map(Params)),
    ?assert(maps:is_key(migration_probability, Params)),
    ?assert(maps:is_key(load_balance_threshold, Params)),
    ?assert(maps:is_key(offload_preference, Params)),
    ?assertEqual(0.05, maps:get(migration_probability, Params)),
    ?assertEqual(0.2, maps:get(load_balance_threshold, Params)),
    ok = gen_server:stop(Pid).

register_island_test() ->
    {ok, Pid} = distribution_silo:start_link(),

    %% Register islands
    ok = distribution_silo:register_island(Pid, island_1),
    ok = distribution_silo:register_island(Pid, island_2),
    ok = distribution_silo:register_island(Pid, island_3),
    timer:sleep(50),

    %% Check stats
    Stats = distribution_silo:get_distribution_stats(Pid),
    ?assertEqual(3, maps:get(island_count, Stats)),
    ok = gen_server:stop(Pid).

update_island_stats_test() ->
    {ok, Pid} = distribution_silo:start_link(),

    %% Register an island
    ok = distribution_silo:register_island(Pid, island_1),
    timer:sleep(50),

    %% Update its stats
    ok = distribution_silo:update_island_stats(Pid, island_1, #{
        fitness_mean => 0.75,
        population_size => 50,
        load => 0.6
    }),
    timer:sleep(50),

    %% Verify update
    {ok, Island} = distribution_silo:get_island(Pid, island_1),
    ?assertEqual(0.75, maps:get(fitness_mean, Island)),
    ?assertEqual(50, maps:get(population_size, Island)),
    ok = gen_server:stop(Pid).

get_island_test() ->
    {ok, Pid} = distribution_silo:start_link(),

    %% Register an island
    ok = distribution_silo:register_island(Pid, island_1),
    timer:sleep(50),

    %% Retrieve it
    {ok, Island} = distribution_silo:get_island(Pid, island_1),
    ?assert(is_map(Island)),
    ?assert(maps:is_key(fitness_mean, Island)),

    %% Non-existent island
    ?assertEqual(not_found, distribution_silo:get_island(Pid, non_existent)),
    ok = gen_server:stop(Pid).

record_migration_test() ->
    {ok, Pid} = distribution_silo:start_link(),

    %% Register islands
    ok = distribution_silo:register_island(Pid, island_1),
    ok = distribution_silo:register_island(Pid, island_2),
    timer:sleep(50),

    %% Record migrations
    ok = distribution_silo:record_migration(Pid, island_1, island_2, ind_1),
    ok = distribution_silo:record_migration(Pid, island_2, island_1, ind_2),
    ok = distribution_silo:record_migration(Pid, island_1, island_2, ind_3),
    timer:sleep(50),

    %% Check stats
    Stats = distribution_silo:get_distribution_stats(Pid),
    ?assertEqual(3, maps:get(total_migrations, Stats)),
    ?assertEqual(3, maps:get(successful_migrations, Stats)),
    ok = gen_server:stop(Pid).

get_migrations_test() ->
    {ok, Pid} = distribution_silo:start_link(),

    %% Empty migrations
    Migrations1 = distribution_silo:get_migrations(Pid),
    ?assertEqual([], Migrations1),

    %% Add some migrations
    ok = distribution_silo:record_migration(Pid, island_1, island_2, ind_1),
    ok = distribution_silo:record_migration(Pid, island_2, island_1, ind_2),
    timer:sleep(50),

    Migrations2 = distribution_silo:get_migrations(Pid),
    ?assertEqual(2, length(Migrations2)),
    ok = gen_server:stop(Pid).

get_distribution_stats_test() ->
    {ok, Pid} = distribution_silo:start_link(),

    %% Empty stats
    Stats1 = distribution_silo:get_distribution_stats(Pid),
    ?assertEqual(0, maps:get(island_count, Stats1)),
    ?assertEqual(0, maps:get(total_migrations, Stats1)),
    ?assertEqual(1.0, maps:get(current_load_balance, Stats1)),

    %% Add some data
    ok = distribution_silo:register_island(Pid, island_1),
    ok = distribution_silo:record_migration(Pid, island_1, island_2, ind_1),
    timer:sleep(50),

    Stats2 = distribution_silo:get_distribution_stats(Pid),
    ?assertEqual(1, maps:get(island_count, Stats2)),
    ?assertEqual(1, maps:get(total_migrations, Stats2)),
    ok = gen_server:stop(Pid).

reset_test() ->
    {ok, Pid} = distribution_silo:start_link(),

    %% Add data
    ok = distribution_silo:register_island(Pid, island_1),
    ok = distribution_silo:record_migration(Pid, island_1, island_2, ind_1),
    timer:sleep(50),

    %% Verify data exists
    Stats1 = distribution_silo:get_distribution_stats(Pid),
    ?assert(maps:get(island_count, Stats1) > 0),

    %% Reset
    ok = distribution_silo:reset(Pid),

    %% Verify reset
    Stats2 = distribution_silo:get_distribution_stats(Pid),
    ?assertEqual(0, maps:get(island_count, Stats2)),
    ?assertEqual(0, maps:get(total_migrations, Stats2)),
    ok = gen_server:stop(Pid).

get_state_test() ->
    {ok, Pid} = distribution_silo:start_link(#{realm => <<"test">>}),
    State = distribution_silo:get_state(Pid),
    ?assert(is_map(State)),
    ?assertEqual(<<"test">>, maps:get(realm, State)),
    ?assert(maps:is_key(current_params, State)),
    ?assert(maps:is_key(total_migrations, State)),
    ?assert(maps:is_key(sensors, State)),
    ok = gen_server:stop(Pid).

%%% ============================================================================
%%% Behavior Callback Tests
%%% ============================================================================

behavior_callbacks_test_() ->
    [
        {"get_silo_type returns distribution",
         ?_assertEqual(distribution, distribution_silo:get_silo_type())},
        {"get_time_constant returns 60.0",
         ?_assertEqual(60.0, distribution_silo:get_time_constant())}
    ].
