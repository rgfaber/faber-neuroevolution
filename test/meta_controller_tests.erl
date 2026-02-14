%% @doc Unit tests for meta_controller module.
-module(meta_controller_tests).

-include_lib("eunit/include/eunit.hrl").
-include("neuroevolution.hrl").
-include("meta_controller.hrl").

%%% ============================================================================
%%% Test Fixtures
%%% ============================================================================

default_config() ->
    %% Network topology: {InputSize, HiddenLayers, OutputSize}
    %% InputSize = 11 to match compute_input_features/3 which returns 11 features:
    %%   8 evolution metrics + 3 resource metrics (memory, cpu, process pressure)
    %% OutputSize = 5 to match output_mapping which has 5 parameters:
    %%   3 base (mutation_rate, mutation_strength, selection_ratio)
    %%   + 2 resource (evaluations_per_individual, max_concurrent_evaluations)
    #meta_config{
        network_topology = {11, [8], 5},
        neuron_type = cfc,
        time_constant = 10.0,
        state_bound = 1.0,
        learning_rate = 0.01
    }.

sample_generation_stats() ->
    #generation_stats{
        generation = 1,
        best_fitness = 0.8,
        avg_fitness = 0.5,
        worst_fitness = 0.2,
        best_individual_id = 1,
        survivors = [1, 2],
        eliminated = [3, 4],
        offspring = [5, 6]
    }.

sample_generation_stats_map() ->
    #{
        generation => 1,
        best_fitness => 0.8,
        avg_fitness => 0.5,
        worst_fitness => 0.2
    }.

%%% ============================================================================
%%% Lifecycle Tests
%%% ============================================================================

start_link_test() ->
    Config = default_config(),
    {ok, Pid} = meta_controller:start_link(Config),
    ?assert(is_pid(Pid)),
    ?assert(is_process_alive(Pid)),
    gen_server:stop(Pid).

start_link_with_name_test() ->
    Config = default_config(),
    {ok, Pid} = meta_controller:start_link(Config, [{name, {local, test_meta}}]),
    ?assert(is_pid(Pid)),
    ?assertEqual(Pid, whereis(test_meta)),
    gen_server:stop(Pid).

start_link_with_id_test() ->
    Config = default_config(),
    {ok, Pid} = meta_controller:start_link(Config, [{id, my_meta_id}]),
    ?assert(is_pid(Pid)),
    gen_server:stop(Pid).

%%% ============================================================================
%%% Training Control Tests
%%% ============================================================================

start_training_test() ->
    Config = default_config(),
    {ok, Pid} = meta_controller:start_link(Config),

    %% First start should succeed
    ?assertEqual({ok, started}, meta_controller:start_training(Pid)),

    %% Second start should return already_running
    ?assertEqual({ok, already_running}, meta_controller:start_training(Pid)),

    gen_server:stop(Pid).

stop_training_test() ->
    Config = default_config(),
    {ok, Pid} = meta_controller:start_link(Config),

    {ok, started} = meta_controller:start_training(Pid),
    ?assertEqual(ok, meta_controller:stop_training(Pid)),

    %% Can start again after stopping
    ?assertEqual({ok, started}, meta_controller:start_training(Pid)),

    gen_server:stop(Pid).

%%% ============================================================================
%%% Update Tests
%%% ============================================================================

update_with_record_test() ->
    Config = default_config(),
    {ok, Pid} = meta_controller:start_link(Config),
    _ = meta_controller:start_training(Pid),

    Stats = sample_generation_stats(),
    Params = meta_controller:update(Pid, Stats),

    ?assert(is_map(Params)),
    ?assert(maps:is_key(mutation_rate, Params)),
    ?assert(maps:is_key(mutation_strength, Params)),
    ?assert(maps:is_key(selection_ratio, Params)),

    %% Parameters should be in valid ranges
    MR = maps:get(mutation_rate, Params),
    MS = maps:get(mutation_strength, Params),
    SR = maps:get(selection_ratio, Params),

    ?assert(MR >= 0.01 andalso MR =< 0.5),
    ?assert(MS >= 0.05 andalso MS =< 1.0),
    ?assert(SR >= 0.10 andalso SR =< 0.50),

    gen_server:stop(Pid).

update_with_map_test() ->
    Config = default_config(),
    {ok, Pid} = meta_controller:start_link(Config),
    _ = meta_controller:start_training(Pid),

    Stats = sample_generation_stats_map(),
    Params = meta_controller:update(Pid, Stats),

    ?assert(is_map(Params)),
    ?assert(maps:is_key(mutation_rate, Params)),

    gen_server:stop(Pid).

multiple_updates_test() ->
    Config = default_config(),
    {ok, Pid} = meta_controller:start_link(Config),
    _ = meta_controller:start_training(Pid),

    %% Run multiple updates to test state persistence
    Stats1 = #{generation => 1, best_fitness => 0.5, avg_fitness => 0.3, worst_fitness => 0.1},
    Stats2 = #{generation => 2, best_fitness => 0.6, avg_fitness => 0.4, worst_fitness => 0.2},
    Stats3 = #{generation => 3, best_fitness => 0.7, avg_fitness => 0.5, worst_fitness => 0.3},

    Params1 = meta_controller:update(Pid, Stats1),
    Params2 = meta_controller:update(Pid, Stats2),
    Params3 = meta_controller:update(Pid, Stats3),

    %% All should return valid parameter maps
    ?assert(is_map(Params1)),
    ?assert(is_map(Params2)),
    ?assert(is_map(Params3)),

    gen_server:stop(Pid).

%%% ============================================================================
%%% State Inspection Tests
%%% ============================================================================

get_state_test() ->
    Config = default_config(),
    {ok, Pid} = meta_controller:start_link(Config),

    {ok, State} = meta_controller:get_state(Pid),

    ?assert(is_map(State)),
    ?assertEqual(0, maps:get(generation, State)),
    ?assertEqual(false, maps:get(running, State)),
    ?assertEqual(0.0, maps:get(cumulative_reward, State)),

    gen_server:stop(Pid).

get_state_after_updates_test() ->
    Config = default_config(),
    {ok, Pid} = meta_controller:start_link(Config),
    _ = meta_controller:start_training(Pid),

    %% Initial state
    {ok, State1} = meta_controller:get_state(Pid),
    ?assertEqual(0, maps:get(generation, State1)),

    %% Update
    _ = meta_controller:update(Pid, sample_generation_stats_map()),

    %% State after update
    {ok, State2} = meta_controller:get_state(Pid),
    ?assertEqual(1, maps:get(generation, State2)),
    ?assertEqual(true, maps:get(running, State2)),

    gen_server:stop(Pid).

get_params_test() ->
    Config = default_config(),
    {ok, Pid} = meta_controller:start_link(Config),

    Params = meta_controller:get_params(Pid),

    ?assert(is_map(Params)),
    %% Check default values
    ?assertEqual(0.10, maps:get(mutation_rate, Params)),
    ?assertEqual(0.30, maps:get(mutation_strength, Params)),
    ?assertEqual(0.20, maps:get(selection_ratio, Params)),

    gen_server:stop(Pid).

%%% ============================================================================
%%% Reset Tests
%%% ============================================================================

reset_test() ->
    Config = default_config(),
    {ok, Pid} = meta_controller:start_link(Config),
    _ = meta_controller:start_training(Pid),

    %% Run some updates
    _ = meta_controller:update(Pid, sample_generation_stats_map()),
    _ = meta_controller:update(Pid, sample_generation_stats_map()),

    {ok, StateBeforeReset} = meta_controller:get_state(Pid),
    ?assertEqual(2, maps:get(generation, StateBeforeReset)),

    %% Reset
    ?assertEqual(ok, meta_controller:reset(Pid)),

    {ok, StateAfterReset} = meta_controller:get_state(Pid),
    ?assertEqual(0, maps:get(generation, StateAfterReset)),
    ?assertEqual(0.0, maps:get(cumulative_reward, StateAfterReset)),

    gen_server:stop(Pid).

%%% ============================================================================
%%% Edge Cases
%%% ============================================================================

zero_fitness_test() ->
    Config = default_config(),
    {ok, Pid} = meta_controller:start_link(Config),
    _ = meta_controller:start_training(Pid),

    Stats = #{generation => 1, best_fitness => 0.0, avg_fitness => 0.0, worst_fitness => 0.0},
    Params = meta_controller:update(Pid, Stats),

    %% Should still return valid parameters
    ?assert(is_map(Params)),

    gen_server:stop(Pid).

negative_fitness_test() ->
    Config = default_config(),
    {ok, Pid} = meta_controller:start_link(Config),
    _ = meta_controller:start_training(Pid),

    Stats = #{generation => 1, best_fitness => -0.5, avg_fitness => -1.0, worst_fitness => -2.0},
    Params = meta_controller:update(Pid, Stats),

    ?assert(is_map(Params)),

    gen_server:stop(Pid).

large_fitness_test() ->
    Config = default_config(),
    {ok, Pid} = meta_controller:start_link(Config),
    _ = meta_controller:start_training(Pid),

    Stats = #{generation => 1, best_fitness => 1000000.0, avg_fitness => 500000.0, worst_fitness => 0.0},
    Params = meta_controller:update(Pid, Stats),

    ?assert(is_map(Params)),

    gen_server:stop(Pid).
