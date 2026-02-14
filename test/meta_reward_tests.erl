%% @doc Unit tests for meta_reward module.
-module(meta_reward_tests).

-include_lib("eunit/include/eunit.hrl").
-include("meta_controller.hrl").

%%% ============================================================================
%%% Test Fixtures
%%% ============================================================================

default_config() ->
    #meta_config{
        reward_weights = #{
            convergence_speed => 0.25,
            final_fitness => 0.25,
            efficiency_ratio => 0.20,
            diversity_aware => 0.15,
            normative_structure => 0.15
        }
    }.

sample_metrics() ->
    #generation_metrics{
        generation = 1,
        best_fitness = 0.8,
        avg_fitness = 0.5,
        worst_fitness = 0.2,
        fitness_std_dev = 0.1,
        fitness_delta = 0.1,
        relative_improvement = 0.12,
        population_diversity = 0.2,
        strategy_entropy = 0.5,
        evaluations_used = 10,
        fitness_per_evaluation = 0.08,
        diversity_corridors = 0.5,
        adaptation_readiness = 0.5,
        params_used = #{mutation_rate => 0.1, mutation_strength => 0.3},
        timestamp = erlang:timestamp()
    }.

sample_history() ->
    [
        #generation_metrics{
            generation = 1,
            best_fitness = 0.7,
            avg_fitness = 0.4,
            worst_fitness = 0.1,
            fitness_std_dev = 0.1,
            fitness_delta = 0.05,
            relative_improvement = 0.08,
            population_diversity = 0.2,
            strategy_entropy = 0.5,
            evaluations_used = 10,
            fitness_per_evaluation = 0.07,
            diversity_corridors = 0.5,
            adaptation_readiness = 0.5,
            params_used = #{mutation_rate => 0.1, mutation_strength => 0.3},
            timestamp = erlang:timestamp()
        }
    ].

%%% ============================================================================
%%% Compute Tests
%%% ============================================================================

compute_basic_test() ->
    Config = default_config(),
    Metrics = sample_metrics(),
    History = [],

    Reward = meta_reward:compute(Metrics, History, Config),

    ?assert(is_record(Reward, meta_reward)),
    ?assert(is_float(Reward#meta_reward.total)),
    ?assert(is_float(Reward#meta_reward.convergence_speed)),
    ?assert(is_float(Reward#meta_reward.final_fitness)),
    ?assert(is_float(Reward#meta_reward.efficiency_ratio)),
    ?assert(is_float(Reward#meta_reward.diversity_aware)),
    ?assert(is_float(Reward#meta_reward.normative_structure)),
    ?assertEqual(1, Reward#meta_reward.generation).

compute_with_history_test() ->
    Config = default_config(),
    Metrics = sample_metrics(),
    History = sample_history(),

    Reward = meta_reward:compute(Metrics, History, Config),

    ?assert(is_record(Reward, meta_reward)),
    ?assert(is_float(Reward#meta_reward.total)).

compute_total_is_weighted_sum_test() ->
    Config = default_config(),
    Metrics = sample_metrics(),
    History = [],

    Reward = meta_reward:compute(Metrics, History, Config),
    Components = meta_reward:compute_components(Metrics, History, Config),
    Weights = Config#meta_config.reward_weights,

    %% Manually compute expected total
    ExpectedTotal = lists:sum([
        maps:get(Component, Components, 0.0) * maps:get(Component, Weights, 0.0)
        || Component <- maps:keys(Weights)
    ]),

    ?assert(abs(Reward#meta_reward.total - ExpectedTotal) < 0.0001).

%%% ============================================================================
%%% Component Tests
%%% ============================================================================

compute_components_test() ->
    Config = default_config(),
    Metrics = sample_metrics(),
    History = [],

    Components = meta_reward:compute_components(Metrics, History, Config),

    ?assert(is_map(Components)),
    ?assert(maps:is_key(convergence_speed, Components)),
    ?assert(maps:is_key(final_fitness, Components)),
    ?assert(maps:is_key(efficiency_ratio, Components)),
    ?assert(maps:is_key(diversity_aware, Components)),
    ?assert(maps:is_key(normative_structure, Components)).

components_are_bounded_test() ->
    Config = default_config(),
    Metrics = sample_metrics(),
    History = sample_history(),

    Components = meta_reward:compute_components(Metrics, History, Config),

    %% All components should be in [0, 1] range (approximately)
    lists:foreach(
        fun({_Name, Value}) ->
            ?assert(Value >= -0.1 andalso Value =< 1.1)
        end,
        maps:to_list(Components)
    ).

%%% ============================================================================
%%% Normalize Component Tests
%%% ============================================================================

normalize_component_no_history_test() ->
    Normalized = meta_reward:normalize_component(test, 0.5, []),
    ?assert(Normalized >= 0.0 andalso Normalized =< 1.0).

normalize_component_with_history_test() ->
    History = [0.1, 0.2, 0.3, 0.4, 0.5],
    Normalized = meta_reward:normalize_component(test, 0.3, History),
    ?assert(Normalized >= 0.0 andalso Normalized =< 1.0).

normalize_component_above_history_test() ->
    History = [0.1, 0.2, 0.3],
    Normalized = meta_reward:normalize_component(test, 1.0, History),
    %% Should be close to 1.0 (high relative to history)
    ?assert(Normalized >= 0.5).

normalize_component_below_history_test() ->
    History = [0.5, 0.6, 0.7],
    Normalized = meta_reward:normalize_component(test, 0.1, History),
    %% Should be close to 0.0 (low relative to history)
    ?assert(Normalized =< 0.5).

normalize_component_uniform_history_test() ->
    History = [0.5, 0.5, 0.5, 0.5],
    Normalized = meta_reward:normalize_component(test, 0.5, History),
    %% All same values should return 0.5 (neutral)
    ?assertEqual(0.5, Normalized).

%%% ============================================================================
%%% Edge Cases
%%% ============================================================================

zero_fitness_metrics_test() ->
    Config = default_config(),
    Metrics = #generation_metrics{
        generation = 1,
        best_fitness = 0.0,
        avg_fitness = 0.0,
        worst_fitness = 0.0,
        fitness_std_dev = 0.0,
        fitness_delta = 0.0,
        relative_improvement = 0.0,
        population_diversity = 0.0,
        strategy_entropy = 0.0,
        evaluations_used = 1,
        fitness_per_evaluation = 0.0,
        diversity_corridors = 0.0,
        adaptation_readiness = 0.0,
        params_used = #{mutation_rate => 0.1, mutation_strength => 0.3},
        timestamp = erlang:timestamp()
    },

    Reward = meta_reward:compute(Metrics, [], Config),
    ?assert(is_float(Reward#meta_reward.total)).

negative_improvement_test() ->
    Config = default_config(),
    Metrics = #generation_metrics{
        generation = 2,
        best_fitness = 0.3,
        avg_fitness = 0.2,
        worst_fitness = 0.1,
        fitness_std_dev = 0.05,
        fitness_delta = -0.2,  %% Negative delta = getting worse
        relative_improvement = -0.4,
        population_diversity = 0.1,
        strategy_entropy = 0.3,
        evaluations_used = 10,
        fitness_per_evaluation = 0.03,
        diversity_corridors = 0.3,
        adaptation_readiness = 0.3,
        params_used = #{mutation_rate => 0.1, mutation_strength => 0.3},
        timestamp = erlang:timestamp()
    },
    History = sample_history(),

    Reward = meta_reward:compute(Metrics, History, Config),
    ?assert(is_float(Reward#meta_reward.total)).

high_fitness_test() ->
    Config = default_config(),
    Metrics = #generation_metrics{
        generation = 100,
        best_fitness = 1000.0,
        avg_fitness = 900.0,
        worst_fitness = 800.0,
        fitness_std_dev = 50.0,
        fitness_delta = 10.0,
        relative_improvement = 0.01,
        population_diversity = 50.0,
        strategy_entropy = 0.9,
        evaluations_used = 100,
        fitness_per_evaluation = 10.0,
        diversity_corridors = 0.8,
        adaptation_readiness = 0.7,
        params_used = #{mutation_rate => 0.05, mutation_strength => 0.1},
        timestamp = erlang:timestamp()
    },

    Reward = meta_reward:compute(Metrics, [], Config),
    ?assert(is_float(Reward#meta_reward.total)).

empty_params_used_test() ->
    Config = default_config(),
    Metrics = #generation_metrics{
        generation = 1,
        best_fitness = 0.5,
        avg_fitness = 0.3,
        worst_fitness = 0.1,
        fitness_std_dev = 0.1,
        fitness_delta = 0.05,
        relative_improvement = 0.1,
        population_diversity = 0.2,
        strategy_entropy = 0.5,
        evaluations_used = 10,
        fitness_per_evaluation = 0.05,
        diversity_corridors = 0.5,
        adaptation_readiness = 0.5,
        params_used = #{},  %% Empty params
        timestamp = erlang:timestamp()
    },

    Reward = meta_reward:compute(Metrics, [], Config),
    ?assert(is_float(Reward#meta_reward.total)).
