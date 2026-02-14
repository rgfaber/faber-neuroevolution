%% @doc EUnit tests for neuroevolution_server module.
-module(neuroevolution_server_tests).

-include_lib("eunit/include/eunit.hrl").
-include("neuroevolution.hrl").
-include("evolution_strategy.hrl").

%%% ============================================================================
%%% Test Fixtures
%%% ============================================================================

sample_config() ->
    #neuro_config{
        population_size = 10,
        evaluations_per_individual = 1,
        selection_ratio = 0.3,
        mutation_rate = 0.1,
        mutation_strength = 0.2,
        max_generations = 5,
        network_topology = {4, [8], 2},
        evaluator_module = mock_evaluator,
        evaluator_options = #{},
        strategy_config = #strategy_config{
            strategy_module = generational_strategy,
            strategy_params = #{network_factory => mock_network_factory}
        }
    }.

%%% ============================================================================
%%% Server Lifecycle Tests
%%% ============================================================================

start_stop_test() ->
    Config = sample_config(),
    {ok, Pid} = neuroevolution_server:start_link(Config, []),
    ?assert(is_pid(Pid)),
    ?assert(is_process_alive(Pid)),
    gen_server:stop(Pid).

%%% ============================================================================
%%% Population Query Tests
%%% ============================================================================

get_population_test() ->
    Config = sample_config(),
    {ok, Pid} = neuroevolution_server:start_link(Config, []),

    {ok, Population} = neuroevolution_server:get_population(Pid),
    ?assert(is_list(Population)),
    ?assertEqual(10, length(Population)),

    %% All should be individuals with networks
    lists:foreach(fun(I) ->
        ?assert(is_record(I, individual)),
        ?assertNotEqual(undefined, I#individual.network)
    end, Population),

    gen_server:stop(Pid).

%%% ============================================================================
%%% Statistics Tests
%%% ============================================================================

get_stats_test() ->
    Config = sample_config(),
    {ok, Pid} = neuroevolution_server:start_link(Config, []),

    {ok, Stats} = neuroevolution_server:get_stats(Pid),
    ?assert(is_map(Stats)),
    ?assert(maps:is_key(generation, Stats)),
    ?assert(maps:is_key(population_size, Stats)),
    ?assert(maps:is_key(running, Stats)),

    gen_server:stop(Pid).

%%% ============================================================================
%%% Training Control Tests
%%% ============================================================================

start_training_test() ->
    Config = sample_config(),
    {ok, Pid} = neuroevolution_server:start_link(Config, []),

    %% Start training
    {ok, started} = neuroevolution_server:start_training(Pid),

    %% Starting again should return already_running
    {ok, already_running} = neuroevolution_server:start_training(Pid),

    gen_server:stop(Pid).

stop_training_test() ->
    Config = sample_config(),
    {ok, Pid} = neuroevolution_server:start_link(Config, []),

    %% Start then stop
    {ok, started} = neuroevolution_server:start_training(Pid),
    ok = neuroevolution_server:stop_training(Pid),

    %% Stats should show not running
    {ok, Stats} = neuroevolution_server:get_stats(Pid),
    ?assertEqual(false, maps:get(running, Stats)),

    gen_server:stop(Pid).
