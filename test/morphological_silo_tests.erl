%% @doc Unit tests for morphological_silo module.
%%
%% Tests network structure and complexity management functionality.
%%
%% @author R.G. Lefever
%% @copyright 2024-2026 R.G. Lefever
-module(morphological_silo_tests).

-include_lib("eunit/include/eunit.hrl").

%%% ============================================================================
%%% Setup/Teardown
%%% ============================================================================

setup() ->
    case whereis(morphological_silo) of
        undefined -> ok;
        Pid ->
            catch gen_server:stop(Pid, normal, 1000),
            timer:sleep(50)
    end,
    ok.

cleanup(_) ->
    case whereis(morphological_silo) of
        undefined -> ok;
        Pid ->
            catch gen_server:stop(Pid, normal, 1000),
            timer:sleep(50)
    end,
    ok.

%%% ============================================================================
%%% Test Suite
%%% ============================================================================

morphological_silo_test_() ->
    {foreach,
     fun setup/0,
     fun cleanup/1,
     [
         {"start_link with default config", fun start_link_default_test/0},
         {"start_link with custom config", fun start_link_custom_config_test/0},
         {"get_params returns defaults", fun get_params_test/0},
         {"record_network_size tracks sizes", fun record_network_size_test/0},
         {"get_complexity_stats returns metrics", fun get_complexity_stats_test/0},
         {"reset clears state", fun reset_test/0},
         {"get_state returns full state", fun get_state_test/0}
     ]}.

%%% ============================================================================
%%% Individual Tests
%%% ============================================================================

start_link_default_test() ->
    {ok, Pid} = morphological_silo:start_link(),
    ?assert(is_pid(Pid)),
    ?assert(is_process_alive(Pid)),
    ok = gen_server:stop(Pid).

start_link_custom_config_test() ->
    Config = #{
        realm => <<"test_realm">>,
        enabled_levels => [l0]
    },
    {ok, Pid} = morphological_silo:start_link(Config),
    ?assert(is_pid(Pid)),
    ok = gen_server:stop(Pid).

get_params_test() ->
    {ok, Pid} = morphological_silo:start_link(),
    Params = morphological_silo:get_params(Pid),
    ?assert(is_map(Params)),
    ?assert(maps:is_key(max_neurons, Params)),
    ?assert(maps:is_key(max_connections, Params)),
    ?assert(maps:is_key(min_neurons, Params)),
    ?assert(maps:is_key(pruning_threshold, Params)),
    ?assert(maps:is_key(complexity_penalty, Params)),
    ?assertEqual(100, maps:get(max_neurons, Params)),
    ?assertEqual(500, maps:get(max_connections, Params)),
    ok = gen_server:stop(Pid).

record_network_size_test() ->
    {ok, Pid} = morphological_silo:start_link(),
    %% Record several network sizes
    ok = morphological_silo:record_network_size(Pid, ind_1, 20, 50),
    ok = morphological_silo:record_network_size(Pid, ind_2, 30, 75),
    ok = morphological_silo:record_network_size(Pid, ind_3, 25, 60),
    timer:sleep(50),  %% Allow casts to process

    %% Check complexity stats
    Stats = morphological_silo:get_complexity_stats(Pid),
    ?assert(is_map(Stats)),
    ?assertEqual(3, maps:get(sample_count, Stats)),
    NeuronMean = maps:get(neuron_mean, Stats),
    ?assert(NeuronMean > 20),
    ?assert(NeuronMean < 30),
    ok = gen_server:stop(Pid).

get_complexity_stats_test() ->
    {ok, Pid} = morphological_silo:start_link(),

    %% Empty stats
    EmptyStats = morphological_silo:get_complexity_stats(Pid),
    ?assertEqual(0.0, maps:get(neuron_mean, EmptyStats)),
    ?assertEqual(0, maps:get(sample_count, EmptyStats)),

    %% Add some data
    ok = morphological_silo:record_network_size(Pid, ind_1, 50, 200),
    ok = morphological_silo:record_network_size(Pid, ind_2, 60, 250),
    timer:sleep(50),

    Stats = morphological_silo:get_complexity_stats(Pid),
    ?assertEqual(2, maps:get(sample_count, Stats)),
    ?assertEqual(55.0, maps:get(neuron_mean, Stats)),
    ?assertEqual(225.0, maps:get(connection_mean, Stats)),
    ok = gen_server:stop(Pid).

reset_test() ->
    {ok, Pid} = morphological_silo:start_link(),
    %% Add some data
    ok = morphological_silo:record_network_size(Pid, ind_1, 30, 100),
    ok = morphological_silo:record_network_size(Pid, ind_2, 40, 150),
    timer:sleep(50),

    %% Verify data exists
    Stats1 = morphological_silo:get_complexity_stats(Pid),
    ?assertEqual(2, maps:get(sample_count, Stats1)),

    %% Reset
    ok = morphological_silo:reset(Pid),

    %% Verify reset
    Stats2 = morphological_silo:get_complexity_stats(Pid),
    ?assertEqual(0, maps:get(sample_count, Stats2)),
    ?assertEqual(0.0, maps:get(neuron_mean, Stats2)),
    ok = gen_server:stop(Pid).

get_state_test() ->
    {ok, Pid} = morphological_silo:start_link(#{realm => <<"test">>}),
    State = morphological_silo:get_state(Pid),
    ?assert(is_map(State)),
    ?assertEqual(<<"test">>, maps:get(realm, State)),
    ?assert(maps:is_key(current_params, State)),
    ?assert(maps:is_key(sensors, State)),
    ?assert(maps:is_key(neuron_history_size, State)),
    ok = gen_server:stop(Pid).

%%% ============================================================================
%%% Behavior Callback Tests
%%% ============================================================================

behavior_callbacks_test_() ->
    [
        {"get_silo_type returns morphological",
         ?_assertEqual(morphological, morphological_silo:get_silo_type())},
        {"get_time_constant returns 30.0",
         ?_assertEqual(30.0, morphological_silo:get_time_constant())}
    ].

