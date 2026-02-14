%% @doc EUnit tests for neuroevolution_stats module.
-module(neuroevolution_stats_tests).

-include_lib("eunit/include/eunit.hrl").
-include("neuroevolution.hrl").

%%% ============================================================================
%%% Test Fixtures
%%% ============================================================================

sample_population() ->
    [
        #individual{id = make_ref(), fitness = 100.0, is_survivor = true},
        #individual{id = make_ref(), fitness = 80.0, is_survivor = true},
        #individual{id = make_ref(), fitness = 60.0, is_offspring = true},
        #individual{id = make_ref(), fitness = 40.0, is_offspring = true},
        #individual{id = make_ref(), fitness = 20.0, is_offspring = true}
    ].

%%% ============================================================================
%%% Average Fitness Tests
%%% ============================================================================

avg_fitness_test() ->
    Population = sample_population(),
    Avg = neuroevolution_stats:avg_fitness(Population),
    %% (100 + 80 + 60 + 40 + 20) / 5 = 60
    ?assertEqual(60.0, Avg).

avg_fitness_single_test() ->
    Population = [#individual{id = make_ref(), fitness = 50.0}],
    Avg = neuroevolution_stats:avg_fitness(Population),
    ?assertEqual(50.0, Avg).

avg_fitness_empty_test() ->
    Avg = neuroevolution_stats:avg_fitness([]),
    ?assertEqual(0.0, Avg).

%%% ============================================================================
%%% Min Fitness Tests
%%% ============================================================================

min_fitness_test() ->
    Population = sample_population(),
    Min = neuroevolution_stats:min_fitness(Population),
    ?assertEqual(20.0, Min).

min_fitness_single_test() ->
    Population = [#individual{id = make_ref(), fitness = 50.0}],
    Min = neuroevolution_stats:min_fitness(Population),
    ?assertEqual(50.0, Min).

min_fitness_empty_test() ->
    Min = neuroevolution_stats:min_fitness([]),
    ?assertEqual(0.0, Min).

%%% ============================================================================
%%% Max Fitness Tests
%%% ============================================================================

max_fitness_test() ->
    Population = sample_population(),
    Max = neuroevolution_stats:max_fitness(Population),
    ?assertEqual(100.0, Max).

max_fitness_single_test() ->
    Population = [#individual{id = make_ref(), fitness = 50.0}],
    Max = neuroevolution_stats:max_fitness(Population),
    ?assertEqual(50.0, Max).

max_fitness_empty_test() ->
    Max = neuroevolution_stats:max_fitness([]),
    ?assertEqual(0.0, Max).

%%% ============================================================================
%%% Fitness Std Dev Tests
%%% ============================================================================

fitness_std_dev_test() ->
    Population = sample_population(),
    StdDev = neuroevolution_stats:fitness_std_dev(Population),
    %% Variance = ((100-60)^2 + (80-60)^2 + (60-60)^2 + (40-60)^2 + (20-60)^2) / 5
    %% = (1600 + 400 + 0 + 400 + 1600) / 5 = 800
    %% StdDev = sqrt(800) ≈ 28.28
    ?assert(abs(StdDev - 28.284271247461902) < 0.0001).

fitness_std_dev_uniform_test() ->
    %% All same fitness should have 0 std dev
    Population = [#individual{id = make_ref(), fitness = 50.0} || _ <- lists:seq(1, 5)],
    StdDev = neuroevolution_stats:fitness_std_dev(Population),
    ?assertEqual(0.0, StdDev).

fitness_std_dev_empty_test() ->
    StdDev = neuroevolution_stats:fitness_std_dev([]),
    ?assertEqual(0.0, StdDev).

fitness_std_dev_single_test() ->
    StdDev = neuroevolution_stats:fitness_std_dev([#individual{id = make_ref(), fitness = 50.0}]),
    ?assertEqual(0.0, StdDev).

%%% ============================================================================
%%% Population Summary Tests
%%% ============================================================================

population_summary_test() ->
    Population = sample_population(),
    Summary = neuroevolution_stats:population_summary(Population),

    ?assertEqual(5, maps:get(count, Summary)),
    ?assertEqual(60.0, maps:get(avg_fitness, Summary)),
    ?assertEqual(20.0, maps:get(min_fitness, Summary)),
    ?assertEqual(100.0, maps:get(max_fitness, Summary)),
    ?assertEqual(2, maps:get(survivors, Summary)),
    ?assertEqual(3, maps:get(offspring, Summary)),
    ?assert(maps:is_key(std_dev, Summary)).

population_summary_empty_test() ->
    Summary = neuroevolution_stats:population_summary([]),

    ?assertEqual(0, maps:get(count, Summary)),
    ?assertEqual(0.0, maps:get(avg_fitness, Summary)),
    ?assertEqual(0.0, maps:get(min_fitness, Summary)),
    ?assertEqual(0.0, maps:get(max_fitness, Summary)),
    ?assertEqual(0, maps:get(survivors, Summary)),
    ?assertEqual(0, maps:get(offspring, Summary)).
