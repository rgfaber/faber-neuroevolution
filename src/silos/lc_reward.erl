%% @doc Reward Signal Computation for Liquid Conglomerate Silos.
%%
%% Part of the Liquid Conglomerate v2 architecture. This module computes
%% reward signals for each silo's L0 TWEANN, including cooperative
%% penalties/bonuses to encourage alignment between silos.
%%
%% == Reward Structure ==
%%
%% Each silo has local objectives plus cooperation terms:
%%
%% Resource Silo:
%%   reward = 0.35*throughput + 0.25*stability + 0.15*efficiency
%%          - 0.15*task_blocked - 0.10*distribution_blocked
%%
%% Task Silo:
%%   reward = 0.40*improvement + 0.20*diversity - 0.15*complexity
%%          - 0.15*resource_pressure + 0.10*distribution_diversity
%%
%% Distribution Silo:
%%   reward = 0.30*load_balance + 0.25*migration + 0.20*network_eff
%%          - 0.15*resource_pressure + 0.10*task_diversity
%%
%% == Global Health Bonus ==
%%
%% All silos receive: +0.1 * global_health_improvement
%% where global_health = weighted sum of all silo performances
%%
%% @author R.G. Lefever
%% @copyright 2024-2026 R.G. Lefever
-module(lc_reward).

%% API
-export([
    compute_resource_reward/1,
    compute_task_reward/1,
    compute_task_reward_breakdown/1,
    compute_trial_reward/1,
    compute_distribution_reward/1,
    compute_global_health/1,
    add_global_bonus/2
]).

%%% ============================================================================
%%% Resource Silo Reward
%%% ============================================================================

%% @doc Compute reward signal for Resource Silo L0.
%%
%% Input metrics map should contain:
%% - throughput: Evaluations per second (normalized 0-1)
%% - gc_triggers: Number of GC triggers in period (0 = best)
%% - pauses: Number of evolution pauses (0 = best)
%% - memory_utilization: Current memory usage (0-1)
%% - cpu_utilization: Current CPU usage (0-1)
%% - task_silo_blocked: Was Task Silo blocked? (0-1)
%% - distribution_blocked: Was Distribution blocked? (0-1)
-spec compute_resource_reward(map()) -> float().
compute_resource_reward(Metrics) ->
    %% Extract metrics with defaults
    Throughput = maps:get(throughput, Metrics, 0.0),
    GcTriggers = maps:get(gc_triggers, Metrics, 0),
    Pauses = maps:get(pauses, Metrics, 0),
    MemUtil = maps:get(memory_utilization, Metrics, 0.5),
    CpuUtil = maps:get(cpu_utilization, Metrics, 0.5),
    TaskBlocked = maps:get(task_silo_blocked, Metrics, 0.0),
    DistBlocked = maps:get(distribution_blocked, Metrics, 0.0),

    %% Compute component scores

    %% Throughput score: higher is better (already normalized)
    ThroughputScore = clamp(Throughput, 0.0, 1.0),

    %% Stability score: penalize GC triggers and pauses
    %% Each GC trigger costs 0.1, each pause costs 0.3
    StabilityScore = max(0.0, 1.0 - (GcTriggers * 0.1) - (Pauses * 0.3)),

    %% Efficiency score: balance utilization (not too high, not too low)
    %% Optimal around 0.6-0.7
    MemEfficiency = 1.0 - abs(MemUtil - 0.65) * 2.0,
    CpuEfficiency = 1.0 - abs(CpuUtil - 0.65) * 2.0,
    EfficiencyScore = clamp((MemEfficiency + CpuEfficiency) / 2.0, 0.0, 1.0),

    %% Cooperation penalties
    TaskBlockedPenalty = clamp(TaskBlocked, 0.0, 1.0),
    DistBlockedPenalty = clamp(DistBlocked, 0.0, 1.0),

    %% Weighted sum
    Reward = 0.35 * ThroughputScore
           + 0.25 * StabilityScore
           + 0.15 * EfficiencyScore
           - 0.15 * TaskBlockedPenalty
           - 0.10 * DistBlockedPenalty,

    clamp(Reward, -1.0, 1.0).

%%% ============================================================================
%%% Task Silo Reward (TWEANN Fitness Function)
%%% ============================================================================

%% @doc Compute reward signal for Task Silo L0 TWEANN.
%%
%% Designed to reward LC controllers that:
%% 1. MAXIMIZE fitness improvement velocity (faster learning)
%% 2. MINIMIZE training time to convergence (efficiency)
%% 3. PUNISH premature convergence (getting stuck at low fitness)
%% 4. MINIMIZE resource usage (memory/CPU efficiency)
%%
%% Input metrics map should contain:
%% - improvement_velocity: Fitness improvement rate per 1000 evals (-1 to 1)
%% - best_fitness: Current best fitness achieved (0-1)
%% - fitness_target: Target fitness to reach (0-1)
%% - evaluations_used: Total evaluations consumed
%% - evaluations_budget: Total evaluation budget
%% - stagnation_severity: How stuck evolution is (0=progressing, 1=stuck)
%% - diversity_index: Population diversity (0-1)
%% - memory_pressure: Current memory usage (0-1)
%% - cpu_pressure: Current CPU usage (0-1)
-spec compute_task_reward(map()) -> float().
compute_task_reward(Metrics) ->
    %% Extract metrics with defaults
    ImprovementVelocity = maps:get(improvement_velocity, Metrics, 0.0),
    BestFitness = maps:get(best_fitness, Metrics, 0.0),
    FitnessTarget = maps:get(fitness_target, Metrics, 1.0),
    EvalsUsed = maps:get(evaluations_used, Metrics, 0),
    EvalsBudget = maps:get(evaluations_budget, Metrics, 100000),
    StagnationSeverity = maps:get(stagnation_severity, Metrics, 0.0),
    DiversityIndex = maps:get(diversity_index, Metrics, 0.5),
    MemoryPressure = maps:get(memory_pressure, Metrics, 0.0),
    CpuPressure = maps:get(cpu_pressure, Metrics, 0.0),

    %% =========================================================================
    %% Component 1: Fitness Improvement Velocity (40% weight)
    %% Higher velocity = better. Strongly rewards rapid improvement.
    %% =========================================================================
    %% Map velocity from [-0.1, 0.1] to [0, 1] with scaling
    %% Positive velocity gets exponential boost
    VelocityScore = case ImprovementVelocity > 0 of
        true ->
            %% Positive velocity: exponential reward (accelerating improvement is great)
            min(1.0, math:pow(ImprovementVelocity * 10, 0.7));
        false ->
            %% Negative velocity: linear penalty (declining is bad)
            max(0.0, 0.5 + ImprovementVelocity * 5)
    end,

    %% =========================================================================
    %% Component 2: Training Efficiency (20% weight)
    %% Reward reaching target fitness with fewer evaluations.
    %% =========================================================================
    FitnessRatio = min(1.0, BestFitness / max(0.01, FitnessTarget)),
    EvalRatio = EvalsUsed / max(1, EvalsBudget),

    %% Efficiency = fitness achieved per evaluation used
    %% Early achievement of high fitness = high score
    EfficiencyScore = case FitnessRatio > 0.9 of
        true ->
            %% Near target: reward reaching it quickly
            1.0 - EvalRatio;
        false ->
            %% Not near target: score based on progress vs time
            FitnessRatio * (1.0 - EvalRatio * 0.5)
    end,

    %% =========================================================================
    %% Component 3: Stagnation Penalty (20% weight)
    %% PUNISH stagnation - LC TWEANN needs to know when to try different strategies.
    %% =========================================================================
    %% For LC control purposes, stagnation is ALWAYS bad because we want the
    %% controller to adapt hyperparameters when stuck, regardless of fitness level.
    %% The TWEANN should learn: high stagnation = low reward = need to change.
    %%
    %% Stagnation severity is already computed in velocity_state:
    %% - 0.0 = healthy (improving)
    %% - 1.0 = critical (completely stuck)
    StagnationPenalty = StagnationSeverity,

    %% Also penalize low diversity (premature collapse of exploration)
    DiversityCollapsePenalty = case DiversityIndex < 0.3 of
        true -> (0.3 - DiversityIndex) * 0.5;  %% Up to 0.15 penalty
        false -> 0.0
    end,

    ConvergenceScore = 1.0 - clamp(StagnationPenalty + DiversityCollapsePenalty, 0.0, 1.0),

    %% =========================================================================
    %% Component 4: Resource Efficiency (20% weight)
    %% Minimize memory and CPU usage while still making progress.
    %% =========================================================================
    %% Memory and CPU pressure are bad, but only if not improving
    %% If improving rapidly, some resource usage is acceptable
    VelocityBonus = max(0.0, ImprovementVelocity * 2.0),
    EffectiveMemPressure = max(0.0, MemoryPressure - VelocityBonus),
    EffectiveCpuPressure = max(0.0, CpuPressure - VelocityBonus),

    ResourceScore = 1.0 - (EffectiveMemPressure * 0.6 + EffectiveCpuPressure * 0.4),

    %% =========================================================================
    %% Composite Reward
    %% =========================================================================
    %% Weights: Velocity=40%, Efficiency=20%, Convergence=20%, Resources=20%
    Reward = 0.40 * VelocityScore
           + 0.20 * EfficiencyScore
           + 0.20 * ConvergenceScore
           + 0.20 * clamp(ResourceScore, 0.0, 1.0),

    clamp(Reward, -1.0, 1.0).

%% @doc Compute reward breakdown for Task Silo L0 - returns individual component scores.
%%
%% Returns a map containing:
%% - total_reward: The combined weighted reward (-1.0 to 1.0)
%% - velocity_score: Fitness improvement velocity score (40% weight)
%% - efficiency_score: Training time efficiency score (20% weight)
%% - convergence_score: Premature convergence penalty score (20% weight)
%% - resource_score: Memory/CPU efficiency score (20% weight)
%% - velocity_weight, efficiency_weight, etc.: The weights used
%%
%% This allows visualization of which factors are contributing to the reward.
-spec compute_task_reward_breakdown(map()) -> map().
compute_task_reward_breakdown(Metrics) ->
    %% Extract metrics with defaults
    ImprovementVelocity = maps:get(improvement_velocity, Metrics, 0.0),
    BestFitness = maps:get(best_fitness, Metrics, 0.0),
    FitnessTarget = maps:get(fitness_target, Metrics, 1.0),
    EvalsUsed = maps:get(evaluations_used, Metrics, 0),
    EvalsBudget = maps:get(evaluations_budget, Metrics, 100000),
    StagnationSeverity = maps:get(stagnation_severity, Metrics, 0.0),
    DiversityIndex = maps:get(diversity_index, Metrics, 0.5),
    MemoryPressure = maps:get(memory_pressure, Metrics, 0.0),
    CpuPressure = maps:get(cpu_pressure, Metrics, 0.0),

    %% Component 1: Velocity Score (40%)
    VelocityScore = case ImprovementVelocity > 0 of
        true ->
            min(1.0, math:pow(ImprovementVelocity * 10, 0.7));
        false ->
            max(0.0, 0.5 + ImprovementVelocity * 5)
    end,

    %% Component 2: Efficiency Score (20%)
    FitnessRatio = min(1.0, BestFitness / max(0.01, FitnessTarget)),
    EvalRatio = EvalsUsed / max(1, EvalsBudget),
    EfficiencyScore = case FitnessRatio > 0.9 of
        true ->
            1.0 - EvalRatio;
        false ->
            FitnessRatio * (1.0 - EvalRatio * 0.5)
    end,

    %% Component 3: Convergence Score (20%)
    LowFitnessMultiplier = max(0.0, 1.0 - FitnessRatio),
    PrematureConvergencePenalty = StagnationSeverity * LowFitnessMultiplier,
    DiversityCollapsePenalty = case DiversityIndex < 0.2 andalso FitnessRatio < 0.5 of
        true -> (0.2 - DiversityIndex) * (0.5 - FitnessRatio) * 2.0;
        false -> 0.0
    end,
    ConvergenceScore = 1.0 - clamp(PrematureConvergencePenalty + DiversityCollapsePenalty, 0.0, 1.0),

    %% Component 4: Resource Score (20%)
    VelocityBonus = max(0.0, ImprovementVelocity * 2.0),
    EffectiveMemPressure = max(0.0, MemoryPressure - VelocityBonus),
    EffectiveCpuPressure = max(0.0, CpuPressure - VelocityBonus),
    ResourceScore = 1.0 - (EffectiveMemPressure * 0.6 + EffectiveCpuPressure * 0.4),

    %% Composite Reward
    TotalReward = 0.40 * VelocityScore
               + 0.20 * EfficiencyScore
               + 0.20 * ConvergenceScore
               + 0.20 * clamp(ResourceScore, 0.0, 1.0),

    #{
        %% Total weighted reward
        total_reward => clamp(TotalReward, -1.0, 1.0),
        %% Individual component scores (0.0-1.0)
        velocity_score => VelocityScore,
        efficiency_score => EfficiencyScore,
        convergence_score => ConvergenceScore,
        resource_score => clamp(ResourceScore, 0.0, 1.0),
        %% Weights for visualization
        velocity_weight => 0.40,
        efficiency_weight => 0.20,
        convergence_weight => 0.20,
        resource_weight => 0.20,
        %% Input metrics (for debugging)
        improvement_velocity => ImprovementVelocity,
        stagnation_severity => StagnationSeverity,
        fitness_ratio => FitnessRatio,
        eval_ratio => EvalRatio
    }.

%% @doc Compute cumulative reward for an LC TWEANN trial.
%%
%% Called at the end of a trial period to evaluate the LC controller's performance.
%% Uses fitness improvement achieved during the trial as the primary metric.
%%
%% Input metrics map should contain:
%% - start_fitness: Fitness at trial start
%% - end_fitness: Fitness at trial end
%% - evaluations_used: Evaluations consumed during trial
%% - evaluation_budget: Evaluation budget for trial
%% - peak_memory: Peak memory usage during trial (0-1)
%% - avg_cpu: Average CPU usage during trial (0-1)
%% - stagnation_events: Number of stagnation events during trial
%% - convergence_reached: true if fitness target was reached
-spec compute_trial_reward(map()) -> float().
compute_trial_reward(Metrics) ->
    StartFitness = maps:get(start_fitness, Metrics, 0.0),
    EndFitness = maps:get(end_fitness, Metrics, 0.0),
    EvalsUsed = maps:get(evaluations_used, Metrics, 0),
    EvalsBudget = maps:get(evaluation_budget, Metrics, 10000),
    PeakMemory = maps:get(peak_memory, Metrics, 0.5),
    AvgCpu = maps:get(avg_cpu, Metrics, 0.5),
    StagnationEvents = maps:get(stagnation_events, Metrics, 0),
    ConvergenceReached = maps:get(convergence_reached, Metrics, false),

    %% Fitness improvement: main objective
    FitnessImprovement = EndFitness - StartFitness,

    %% Bonus for reaching convergence
    ConvergenceBonus = case ConvergenceReached of
        true -> 0.5 * (1.0 - EvalsUsed / max(1, EvalsBudget));  %% Faster = better
        false -> 0.0
    end,

    %% Penalty for stagnation events
    StagnationPenalty = min(0.3, StagnationEvents * 0.05),

    %% Resource efficiency
    ResourceEfficiency = 1.0 - (PeakMemory * 0.5 + AvgCpu * 0.5),

    %% Composite trial reward
    TrialReward = FitnessImprovement * 2.0      %% Main: fitness gained
               + ConvergenceBonus               %% Bonus: reached target quickly
               - StagnationPenalty              %% Penalty: got stuck
               + ResourceEfficiency * 0.2,      %% Bonus: used resources efficiently

    clamp(TrialReward, -1.0, 2.0).

%%% ============================================================================
%%% Distribution Silo Reward
%%% ============================================================================

%% @doc Compute reward signal for Distribution Silo L0.
%%
%% Input metrics map should contain:
%% - load_balance_score: How evenly distributed load is (0-1)
%% - migration_effectiveness: Do migrations improve diversity? (0-1)
%% - network_efficiency: Low overhead, successful transfers (0-1)
%% - resource_pressure_caused: Pressure on Resource Silo (0-1)
%% - task_diversity_contribution: Helping Task Silo diversity (0-1)
-spec compute_distribution_reward(map()) -> float().
compute_distribution_reward(Metrics) ->
    %% Extract metrics with defaults
    LoadBalanceScore = maps:get(load_balance_score, Metrics, 0.5),
    MigrationEffectiveness = maps:get(migration_effectiveness, Metrics, 0.5),
    NetworkEfficiency = maps:get(network_efficiency, Metrics, 0.8),
    ResourcePressure = maps:get(resource_pressure_caused, Metrics, 0.0),
    TaskDiversityContrib = maps:get(task_diversity_contribution, Metrics, 0.0),

    %% All scores are already in 0-1 range, just clamp
    LoadScore = clamp(LoadBalanceScore, 0.0, 1.0),
    MigrationScore = clamp(MigrationEffectiveness, 0.0, 1.0),
    NetworkScore = clamp(NetworkEfficiency, 0.0, 1.0),
    ResourcePenalty = clamp(ResourcePressure, 0.0, 1.0),
    TaskBonus = clamp(TaskDiversityContrib, 0.0, 1.0),

    %% Weighted sum
    Reward = 0.30 * LoadScore
           + 0.25 * MigrationScore
           + 0.20 * NetworkScore
           - 0.15 * ResourcePenalty
           + 0.10 * TaskBonus,

    clamp(Reward, -1.0, 1.0).

%%% ============================================================================
%%% Global Health and Bonus
%%% ============================================================================

%% @doc Compute global health metric from all silo rewards.
%%
%% Global health is a weighted average of silo rewards.
%% This provides a holistic view of system performance.
-spec compute_global_health(map()) -> float().
compute_global_health(SiloRewards) ->
    ResourceReward = maps:get(resource, SiloRewards, 0.0),
    TaskReward = maps:get(task, SiloRewards, 0.0),
    DistReward = maps:get(distribution, SiloRewards, 0.0),

    %% Equal weighting for global health
    GlobalHealth = (ResourceReward + TaskReward + DistReward) / 3.0,

    clamp(GlobalHealth, -1.0, 1.0).

%% @doc Add global bonus to a silo's reward based on global health improvement.
%%
%% Each silo gets +0.1 * (current_global - previous_global) as bonus.
%% This encourages silos to cooperate for overall system improvement.
-spec add_global_bonus(float(), map()) -> float().
add_global_bonus(SiloReward, GlobalState) ->
    CurrentGlobal = maps:get(current_global_health, GlobalState, 0.0),
    PreviousGlobal = maps:get(previous_global_health, GlobalState, 0.0),

    GlobalImprovement = CurrentGlobal - PreviousGlobal,

    %% Add 10% of global improvement as bonus
    Bonus = 0.1 * GlobalImprovement,

    clamp(SiloReward + Bonus, -1.0, 1.0).

%%% ============================================================================
%%% Internal Functions
%%% ============================================================================

%% @private Clamp value to range.
clamp(Value, Min, Max) ->
    max(Min, min(Max, Value)).
