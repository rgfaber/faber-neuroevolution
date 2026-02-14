%% @doc Unit tests for lc_reward module.
%%
%% Tests cooperative reward signal computation for all silos.
%%
%% @author R.G. Lefever
%% @copyright 2024-2026 R.G. Lefever
-module(lc_reward_tests).

-include_lib("eunit/include/eunit.hrl").

%%% ============================================================================
%%% Resource Silo Reward Tests
%%% ============================================================================

resource_reward_empty_metrics_test() ->
    Reward = lc_reward:compute_resource_reward(#{}),
    %% With defaults: throughput=0, gc=0, pauses=0, mem=0.5, cpu=0.5
    %% Should return some positive value due to stability
    ?assert(is_float(Reward)),
    ?assert(Reward >= -1.0),
    ?assert(Reward =< 1.0).

resource_reward_high_throughput_test() ->
    Metrics = #{
        throughput => 1.0,           % Max throughput
        gc_triggers => 0,
        pauses => 0,
        memory_utilization => 0.65,  % Optimal
        cpu_utilization => 0.65      % Optimal
    },
    Reward = lc_reward:compute_resource_reward(Metrics),
    %% Should be high with perfect metrics
    ?assert(Reward > 0.5).

resource_reward_penalizes_gc_triggers_test() ->
    LowGc = lc_reward:compute_resource_reward(#{gc_triggers => 0, pauses => 0}),
    HighGc = lc_reward:compute_resource_reward(#{gc_triggers => 5, pauses => 0}),
    ?assert(LowGc > HighGc).

resource_reward_penalizes_pauses_test() ->
    NoPauses = lc_reward:compute_resource_reward(#{pauses => 0}),
    WithPauses = lc_reward:compute_resource_reward(#{pauses => 2}),
    ?assert(NoPauses > WithPauses).

resource_reward_penalizes_task_blocking_test() ->
    NoBlocking = lc_reward:compute_resource_reward(#{task_silo_blocked => 0.0}),
    WithBlocking = lc_reward:compute_resource_reward(#{task_silo_blocked => 1.0}),
    ?assert(NoBlocking > WithBlocking).

resource_reward_penalizes_distribution_blocking_test() ->
    NoBlocking = lc_reward:compute_resource_reward(#{distribution_blocked => 0.0}),
    WithBlocking = lc_reward:compute_resource_reward(#{distribution_blocked => 1.0}),
    ?assert(NoBlocking > WithBlocking).

resource_reward_optimal_utilization_test() ->
    %% Optimal utilization is around 0.65
    Optimal = lc_reward:compute_resource_reward(#{
        memory_utilization => 0.65,
        cpu_utilization => 0.65
    }),
    TooLow = lc_reward:compute_resource_reward(#{
        memory_utilization => 0.2,
        cpu_utilization => 0.2
    }),
    TooHigh = lc_reward:compute_resource_reward(#{
        memory_utilization => 0.95,
        cpu_utilization => 0.95
    }),
    ?assert(Optimal > TooLow),
    ?assert(Optimal > TooHigh).

resource_reward_clamped_to_range_test() ->
    %% Even with extreme values, reward should be in [-1, 1]
    ExtremeHigh = lc_reward:compute_resource_reward(#{
        throughput => 10.0,
        gc_triggers => 0,
        pauses => 0
    }),
    ExtremeLow = lc_reward:compute_resource_reward(#{
        throughput => 0.0,
        gc_triggers => 100,
        pauses => 100,
        task_silo_blocked => 1.0,
        distribution_blocked => 1.0
    }),
    ?assert(ExtremeHigh =< 1.0),
    ?assert(ExtremeLow >= -1.0).

%%% ============================================================================
%%% Task Silo Reward Tests
%%% ============================================================================

task_reward_empty_metrics_test() ->
    Reward = lc_reward:compute_task_reward(#{}),
    ?assert(is_float(Reward)),
    ?assert(Reward >= -1.0),
    ?assert(Reward =< 1.0).

task_reward_high_improvement_test() ->
    Metrics = #{
        improvement_velocity => 1.0,  % Max improvement
        diversity_index => 0.55,      % Optimal diversity
        complexity_growth => 0.0,     % No bloat
        resource_pressure_caused => 0.0
    },
    Reward = lc_reward:compute_task_reward(Metrics),
    ?assert(Reward > 0.5).

task_reward_penalizes_negative_improvement_test() ->
    Improving = lc_reward:compute_task_reward(#{improvement_velocity => 0.5}),
    Declining = lc_reward:compute_task_reward(#{improvement_velocity => -0.5}),
    ?assert(Improving > Declining).

task_reward_efficiency_near_target_test() ->
    %% Near target fitness should be rewarded
    NearTarget = lc_reward:compute_task_reward(#{
        best_fitness => 0.95,
        fitness_target => 1.0,
        evaluations_used => 5000,
        evaluations_budget => 100000
    }),
    FarFromTarget = lc_reward:compute_task_reward(#{
        best_fitness => 0.3,
        fitness_target => 1.0,
        evaluations_used => 5000,
        evaluations_budget => 100000
    }),
    ?assert(NearTarget > FarFromTarget).

task_reward_penalizes_premature_convergence_test() ->
    %% Stagnation at LOW fitness should be punished
    %% Stagnation at HIGH fitness is OK
    StagnantLow = lc_reward:compute_task_reward(#{
        stagnation_severity => 0.8,
        best_fitness => 0.2,
        fitness_target => 1.0
    }),
    StagnantHigh = lc_reward:compute_task_reward(#{
        stagnation_severity => 0.8,
        best_fitness => 0.95,
        fitness_target => 1.0
    }),
    ?assert(StagnantHigh > StagnantLow).

task_reward_penalizes_resource_usage_test() ->
    %% High resource usage (memory/CPU) should be penalized
    LowResources = lc_reward:compute_task_reward(#{
        memory_pressure => 0.2,
        cpu_pressure => 0.2
    }),
    HighResources = lc_reward:compute_task_reward(#{
        memory_pressure => 0.9,
        cpu_pressure => 0.9
    }),
    ?assert(LowResources > HighResources).

task_reward_rewards_fast_improvement_test() ->
    %% Fast improvement velocity should get exponential reward
    FastImprover = lc_reward:compute_task_reward(#{improvement_velocity => 0.1}),
    SlowImprover = lc_reward:compute_task_reward(#{improvement_velocity => 0.01}),
    ?assert(FastImprover > SlowImprover).

%%% ============================================================================
%%% Distribution Silo Reward Tests
%%% ============================================================================

distribution_reward_empty_metrics_test() ->
    Reward = lc_reward:compute_distribution_reward(#{}),
    ?assert(is_float(Reward)),
    ?assert(Reward >= -1.0),
    ?assert(Reward =< 1.0).

distribution_reward_high_balance_test() ->
    Metrics = #{
        load_balance_score => 1.0,
        migration_effectiveness => 1.0,
        network_efficiency => 1.0,
        resource_pressure_caused => 0.0,
        task_diversity_contribution => 1.0
    },
    Reward = lc_reward:compute_distribution_reward(Metrics),
    ?assert(Reward > 0.7).

distribution_reward_penalizes_imbalance_test() ->
    Balanced = lc_reward:compute_distribution_reward(#{load_balance_score => 1.0}),
    Imbalanced = lc_reward:compute_distribution_reward(#{load_balance_score => 0.2}),
    ?assert(Balanced > Imbalanced).

distribution_reward_rewards_migration_test() ->
    Effective = lc_reward:compute_distribution_reward(#{migration_effectiveness => 1.0}),
    Ineffective = lc_reward:compute_distribution_reward(#{migration_effectiveness => 0.0}),
    ?assert(Effective > Ineffective).

distribution_reward_penalizes_resource_pressure_test() ->
    NoPressure = lc_reward:compute_distribution_reward(#{resource_pressure_caused => 0.0}),
    WithPressure = lc_reward:compute_distribution_reward(#{resource_pressure_caused => 1.0}),
    ?assert(NoPressure > WithPressure).

%%% ============================================================================
%%% Global Health Tests
%%% ============================================================================

global_health_empty_test() ->
    Health = lc_reward:compute_global_health(#{}),
    ?assertEqual(0.0, Health).

global_health_all_positive_test() ->
    SiloRewards = #{
        resource => 0.8,
        task => 0.7,
        distribution => 0.9
    },
    Health = lc_reward:compute_global_health(SiloRewards),
    ?assert(Health > 0.7).

global_health_mixed_test() ->
    SiloRewards = #{
        resource => 0.5,
        task => -0.5,
        distribution => 0.5
    },
    Health = lc_reward:compute_global_health(SiloRewards),
    %% (0.5 - 0.5 + 0.5) / 3 = 0.166...
    ?assert(abs(Health - 0.166666) < 0.01).

global_health_clamped_test() ->
    ExtremeRewards = #{
        resource => 5.0,  % Out of normal range
        task => 5.0,
        distribution => 5.0
    },
    Health = lc_reward:compute_global_health(ExtremeRewards),
    ?assert(Health =< 1.0).

%%% ============================================================================
%%% Global Bonus Tests
%%% ============================================================================

global_bonus_improvement_test() ->
    GlobalState = #{
        current_global_health => 0.8,
        previous_global_health => 0.5
    },
    %% Improvement = 0.3, bonus = 0.03
    BaseReward = 0.5,
    NewReward = lc_reward:add_global_bonus(BaseReward, GlobalState),
    ?assert(NewReward > BaseReward).

global_bonus_decline_test() ->
    GlobalState = #{
        current_global_health => 0.3,
        previous_global_health => 0.7
    },
    %% Decline = -0.4, penalty = -0.04
    BaseReward = 0.5,
    NewReward = lc_reward:add_global_bonus(BaseReward, GlobalState),
    ?assert(NewReward < BaseReward).

global_bonus_no_change_test() ->
    GlobalState = #{
        current_global_health => 0.5,
        previous_global_health => 0.5
    },
    BaseReward = 0.5,
    NewReward = lc_reward:add_global_bonus(BaseReward, GlobalState),
    ?assertEqual(0.5, NewReward).

global_bonus_clamped_test() ->
    %% Even with extreme bonus, should stay in [-1, 1]
    GlobalState = #{
        current_global_health => 1.0,
        previous_global_health => -1.0
    },
    %% Improvement = 2.0, bonus = 0.2
    NewReward = lc_reward:add_global_bonus(0.95, GlobalState),
    ?assert(NewReward =< 1.0).

%%% ============================================================================
%%% Reward Weight Verification Tests
%%% ============================================================================

resource_reward_weights_sum_test() ->
    %% Weights: 0.35 + 0.25 + 0.15 + 0.15 + 0.10 = 1.0
    %% Maximum possible reward should approach 0.75 (all positive terms)
    MaxPossible = lc_reward:compute_resource_reward(#{
        throughput => 1.0,
        gc_triggers => 0,
        pauses => 0,
        memory_utilization => 0.65,
        cpu_utilization => 0.65,
        task_silo_blocked => 0.0,
        distribution_blocked => 0.0
    }),
    ?assert(MaxPossible > 0.6),
    ?assert(MaxPossible =< 1.0).

task_reward_weights_sum_test() ->
    %% Weights: 0.40 + 0.20 + 0.15 + 0.15 + 0.10 = 1.0
    MaxPossible = lc_reward:compute_task_reward(#{
        improvement_velocity => 1.0,
        diversity_index => 0.55,
        complexity_growth => 0.0,
        resource_pressure_caused => 0.0,
        distribution_diversity_bonus => 1.0
    }),
    ?assert(MaxPossible > 0.6),
    ?assert(MaxPossible =< 1.0).

distribution_reward_weights_sum_test() ->
    %% Weights: 0.30 + 0.25 + 0.20 + 0.15 + 0.10 = 1.0
    MaxPossible = lc_reward:compute_distribution_reward(#{
        load_balance_score => 1.0,
        migration_effectiveness => 1.0,
        network_efficiency => 1.0,
        resource_pressure_caused => 0.0,
        task_diversity_contribution => 1.0
    }),
    ?assert(MaxPossible > 0.7),
    ?assert(MaxPossible =< 1.0).
