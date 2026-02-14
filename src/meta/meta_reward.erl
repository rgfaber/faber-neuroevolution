%% @doc Composite reward computation for meta-controller training.
%%
%% This module computes a multi-objective reward signal for training the
%% meta-controller. The reward balances multiple objectives:
%%
%% 1. **Convergence Speed** - How quickly fitness improves
%% 2. **Final Fitness** - Absolute performance achieved
%% 3. **Efficiency Ratio** - Fitness gained per computation spent
%% 4. **Diversity Awareness** - Maintains exploration capacity
%% 5. **Normative Structure** - Preserves capacity for future adaptation
%%
%% == Reward Formula ==
%%
%% The total reward is a weighted sum:
%%
%% `R = w1*convergence + w2*fitness + w3*efficiency + w4*diversity + w5*normative'
%%
%% Where weights are configured in meta_config.reward_weights.
%%
%% == Normalization ==
%%
%% All reward components are normalized to [0, 1] or [-1, 1] range
%% for stable training. Historical data is used for normalization.
%%
%% @author R.G. Lefever
%% @copyright 2024-2026 R.G. Lefever
-module(meta_reward).

-include("meta_controller.hrl").

-export([
    compute/3,
    compute_components/3,
    normalize_component/3
]).

%%% ============================================================================
%%% API Functions
%%% ============================================================================

%% @doc Compute composite reward for a generation.
%%
%% @param Metrics Current generation metrics
%% @param History List of previous generation metrics
%% @param Config Meta-controller configuration
%% @returns meta_reward record with all components and total
-spec compute(generation_metrics(), [generation_metrics()], meta_config()) -> meta_reward().
compute(Metrics, History, Config) ->
    Components = compute_components(Metrics, History, Config),
    Weights = Config#meta_config.reward_weights,

    %% Compute weighted sum
    Total = maps:fold(
        fun(Component, Value, Acc) ->
            Weight = maps:get(Component, Weights, 0.0),
            Acc + Weight * Value
        end,
        0.0,
        Components
    ),

    #meta_reward{
        convergence_speed = maps:get(convergence_speed, Components),
        final_fitness = maps:get(final_fitness, Components),
        efficiency_ratio = maps:get(efficiency_ratio, Components),
        diversity_aware = maps:get(diversity_aware, Components),
        normative_structure = maps:get(normative_structure, Components),
        total = Total,
        generation = Metrics#generation_metrics.generation
    }.

%% @doc Compute individual reward components.
%%
%% @param Metrics Current generation metrics
%% @param History List of previous generation metrics
%% @param Config Meta-controller configuration
%% @returns Map of component name to normalized value
-spec compute_components(generation_metrics(), [generation_metrics()], meta_config()) ->
    #{atom() => float()}.
compute_components(Metrics, History, _Config) ->
    #{
        convergence_speed => compute_convergence_speed(Metrics, History),
        final_fitness => compute_final_fitness(Metrics, History),
        efficiency_ratio => compute_efficiency_ratio(Metrics, History),
        diversity_aware => compute_diversity_aware(Metrics, History),
        normative_structure => compute_normative_structure(Metrics, History)
    }.

%% @doc Normalize a reward component value.
%%
%% Uses tweann_nif:z_score/3 for NIF-accelerated Z-score computation.
%%
%% @param Component Component name
%% @param Value Raw value
%% @param History Historical values for normalization
%% @returns Normalized value in appropriate range
-spec normalize_component(atom(), float(), [float()]) -> float().
normalize_component(_Component, Value, []) ->
    %% No history, use default normalization
    clamp(Value, 0.0, 1.0);
normalize_component(_Component, Value, History) ->
    %% Use NIF-accelerated fitness stats to get mean and std_dev in one pass
    %% Returns {Min, Max, Mean, Variance, StdDev, Sum}
    {_Min, _Max, Mean, _Variance, StdDev, _Sum} = tweann_nif:fitness_stats(History),
    case StdDev < 0.0001 of
        true -> 0.5;  %% All values same (or nearly same)
        false ->
            %% Use NIF-accelerated Z-score computation
            ZScore = tweann_nif:z_score(Value, Mean, StdDev),
            %% Squash to [0, 1] with sigmoid
            sigmoid(ZScore)
    end.

%%% ============================================================================
%%% Internal Functions - Reward Components
%%% ============================================================================

%% @private Compute convergence speed reward.
%%
%% Measures how quickly fitness is improving. Higher improvement rate = higher reward.
compute_convergence_speed(Metrics, History) ->
    %% Current improvement rate
    CurrentImprovement = Metrics#generation_metrics.relative_improvement,

    %% Moving average of recent improvements
    RecentImprovements = [M#generation_metrics.relative_improvement || M <- lists:sublist(History, 5)],
    AvgImprovement = safe_average(RecentImprovements, 0.0),

    %% Reward is based on both current and trend
    %% Bonus for improving faster than recent average
    ImprovementSignal = case abs(AvgImprovement) < 0.0001 of
        true -> CurrentImprovement;
        false -> CurrentImprovement / max(0.001, abs(AvgImprovement))
    end,

    %% Normalize: positive improvement is good, zero is neutral, negative is bad
    normalize_to_reward(ImprovementSignal, -2.0, 2.0).

%% @private Compute final fitness reward.
%%
%% Measures absolute fitness achieved. Higher fitness = higher reward.
compute_final_fitness(Metrics, History) ->
    CurrentFitness = Metrics#generation_metrics.best_fitness,

    %% Normalize against historical best
    HistoricalBest = case History of
        [] -> CurrentFitness;
        _ -> lists:max([M#generation_metrics.best_fitness || M <- History])
    end,

    MaxFitness = max(1.0, max(CurrentFitness, HistoricalBest)),

    %% Reward is normalized fitness
    CurrentFitness / MaxFitness.

%% @private Compute efficiency ratio reward.
%%
%% Measures fitness gained per unit of computation. More efficient = higher reward.
compute_efficiency_ratio(Metrics, History) ->
    %% Fitness improvement
    FitnessGain = max(0.0, Metrics#generation_metrics.fitness_delta),

    %% Evaluations used (approximated from params)
    ParamsUsed = Metrics#generation_metrics.params_used,
    MutationRate = maps:get(mutation_rate, ParamsUsed, 0.1),
    MutationStrength = maps:get(mutation_strength, ParamsUsed, 0.3),

    %% Higher mutation = more "computational effort"
    ComputationalCost = 1.0 + MutationRate + MutationStrength,

    %% Efficiency = gain / cost
    RawEfficiency = FitnessGain / ComputationalCost,

    %% Normalize against historical efficiency
    HistoricalEfficiencies = [
        max(0.0, M#generation_metrics.fitness_delta) /
        (1.0 + maps:get(mutation_rate, M#generation_metrics.params_used, 0.1) +
               maps:get(mutation_strength, M#generation_metrics.params_used, 0.3))
        || M <- History
    ],

    normalize_against_history(RawEfficiency, HistoricalEfficiencies).

%% @private Compute diversity-aware reward.
%%
%% Rewards maintaining population diversity for exploration.
%% Too little diversity = low reward, too much = diminishing returns.
compute_diversity_aware(Metrics, History) ->
    CurrentDiversity = Metrics#generation_metrics.population_diversity,
    CurrentEntropy = Metrics#generation_metrics.strategy_entropy,

    %% Historical diversity
    HistoricalDiversity = [M#generation_metrics.population_diversity || M <- History],
    AvgHistDiversity = safe_average(HistoricalDiversity, CurrentDiversity),

    %% We want diversity to be maintained, not maximized
    %% Reward function: peaked around historical average
    DiversityRatio = case abs(AvgHistDiversity) < 0.0001 of
        true -> 1.0;
        false -> CurrentDiversity / AvgHistDiversity
    end,

    %% Peaked reward: 1.0 at ratio=1, decreasing for extremes
    DiversityReward = 1.0 - abs(DiversityRatio - 1.0) * 0.5,

    %% Entropy component (want non-zero entropy)
    EntropyReward = min(1.0, CurrentEntropy * 2.0),

    %% Combine
    (DiversityReward * 0.6 + EntropyReward * 0.4).

%% @private Compute normative structure reward.
%%
%% Rewards maintaining population structure that enables future adaptation.
%% This is the "capacity to improve" rather than current improvement.
compute_normative_structure(Metrics, History) ->
    %% Diversity corridors: distinct strategy clusters
    DiversityCorridors = Metrics#generation_metrics.diversity_corridors,

    %% Adaptation readiness: variance in fitness-adjacent traits
    AdaptationReadiness = Metrics#generation_metrics.adaptation_readiness,

    %% Breakthrough potential: based on improvement trajectory
    BreakthroughPotential = compute_breakthrough_potential(Metrics, History),

    %% Weighted combination
    StructureScore = DiversityCorridors * 0.3 +
                    AdaptationReadiness * 0.3 +
                    BreakthroughPotential * 0.4,

    clamp(StructureScore, 0.0, 1.0).

%% @private Estimate breakthrough potential from trajectory.
compute_breakthrough_potential(Metrics, History) ->
    %% Look for patterns that historically preceded breakthroughs
    %% Simplified: high diversity + moderate improvement rate suggests potential

    Diversity = Metrics#generation_metrics.population_diversity,
    Improvement = Metrics#generation_metrics.relative_improvement,

    %% Second derivative of fitness (acceleration)
    Acceleration = case History of
        [] -> 0.0;
        [Prev | _] ->
            PrevImprovement = Prev#generation_metrics.relative_improvement,
            Improvement - PrevImprovement
    end,

    %% Breakthrough potential is high when:
    %% - Diversity is maintained
    %% - Improvement is accelerating (or about to)
    %% - Not already at maximum performance

    HistoricalBest = case History of
        [] -> Metrics#generation_metrics.best_fitness;
        _ -> lists:max([M#generation_metrics.best_fitness || M <- History])
    end,

    HeadroomFactor = case abs(HistoricalBest) < 0.0001 of
        true -> 1.0;
        false -> 1.0 - min(1.0, Metrics#generation_metrics.best_fitness / HistoricalBest * 0.9)
    end,

    %% Combine factors
    DiversityFactor = min(1.0, Diversity * 2.0),
    AccelerationFactor = sigmoid(Acceleration * 10.0),

    (DiversityFactor * 0.4 + AccelerationFactor * 0.3 + HeadroomFactor * 0.3).

%%% ============================================================================
%%% Utility Functions
%%% ============================================================================

%% @private Normalize value to reward range [0, 1].
normalize_to_reward(Value, MinExpected, MaxExpected) ->
    Range = MaxExpected - MinExpected,
    case abs(Range) < 0.0001 of
        true -> 0.5;
        false -> clamp((Value - MinExpected) / Range, 0.0, 1.0)
    end.

%% @private Normalize value against historical distribution.
normalize_against_history(Value, []) ->
    sigmoid(Value);
normalize_against_history(Value, History) ->
    Max = lists:max([Value | History]),
    Min = lists:min([Value | History]),
    Range = Max - Min,
    case abs(Range) < 0.0001 of
        true -> 0.5;
        false -> clamp((Value - Min) / Range, 0.0, 1.0)
    end.

%% @private Safe average with default.
safe_average([], Default) -> Default;
safe_average(List, _Default) ->
    lists:sum(List) / length(List).

%% @private Sigmoid function.
sigmoid(X) ->
    V = clamp(X, -10.0, 10.0),
    1.0 / (1.0 + math:exp(-V)).

%% @private Clamp value to range.
clamp(Val, Min, _Max) when Val < Min -> Min;
clamp(Val, _Min, Max) when Val > Max -> Max;
clamp(Val, _Min, _Max) -> Val.
