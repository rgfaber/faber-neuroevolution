# Continual Learning: LC at Inference Time

This guide explores how the Liquid Conglomerate architecture can be useful beyond training - enabling **continual adaptation during deployment**.

## Overview

Traditional neuroevolution follows a simple pattern:

```
[Training] ──→ [Deploy Best Network] ──→ [Static Inference]
```

But what if the environment changes after deployment? What if the task distribution shifts? The deployed network becomes stale.

**Continual Learning** extends the LC to monitor and adapt networks during inference:

```
[Training] ──→ [Deploy with LC Monitoring] ──→ [Adaptive Inference]
                           ↓
                  [Detect Drift] ──→ [Adapt/Retrain]
```

![Continual Learning Architecture](assets/continual-learning.svg)

## Why Continual Learning?

### The Deployment Problem

Evolved networks face challenges in production:

| Challenge | Example | Impact |
|-----------|---------|--------|
| **Concept drift** | User behavior changes over time | Model accuracy degrades |
| **Distribution shift** | New input patterns appear | Model makes wrong predictions |
| **Non-stationarity** | Environment dynamics change | Control policy fails |
| **Adversarial adaptation** | Opponents learn your strategy | Game AI becomes exploitable |

### Traditional Solutions and Their Limits

1. **Periodic retraining**: Expensive, reactive (damage already done)
2. **Online learning**: Risk of catastrophic forgetting
3. **Ensemble methods**: High inference cost, doesn't solve drift detection
4. **Monitoring dashboards**: Manual intervention required

The LC offers **automated, continuous adaptation** with minimal overhead.

## LC Inference Modes

The LC can operate in several modes during inference:

### 1. Monitoring Only (Lowest Overhead)

Only the Resource Silo runs, tracking performance metrics:

```erlang
%% Start LC in monitoring mode
{ok, Pid} = lc_supervisor:start_link(#{
    mode => inference,
    silos => [resource],  %% Only Resource Silo
    thresholds => #{
        latency_p99 => 50,      %% ms
        error_rate => 0.01,     %% 1%
        throughput_min => 1000  %% requests/sec
    }
}).
```

**What it monitors:**
- Inference latency (p50, p95, p99)
- Error rates (if feedback available)
- Throughput metrics
- Resource utilization (memory, CPU)

**Actions:** Alerts only, no automatic adaptation.

### 2. Portfolio Switching (Low Overhead)

Task Silo selects from a portfolio of evolved networks:

```erlang
{ok, Pid} = lc_supervisor:start_link(#{
    mode => inference,
    silos => [resource, task],
    portfolio => #{
        networks => [Network1, Network2, Network3],
        selection_strategy => performance_based,
        switch_threshold => 0.1  %% 10% performance drop triggers switch
    }
}).
```

**How it works:**
1. Monitor current network's performance
2. Detect degradation (error rate increase, reward drop)
3. Switch to alternative network from portfolio
4. Track which network works best for current conditions

**Use cases:**
- Multi-task domains with periodic task changes
- Environments with distinct operating modes
- Fallback to simpler network under resource pressure

### 3. Micro-Evolution (Medium Overhead)

Task Silo performs small weight perturbations during inference:

```erlang
{ok, Pid} = lc_supervisor:start_link(#{
    mode => inference,
    silos => [resource, task],
    micro_evolution => #{
        enabled => true,
        mutation_strength => 0.01,  %% Very small (1% of training)
        evaluation_window => 100,   %% Evaluate over 100 inferences
        acceptance_threshold => 0.0 %% Accept if not worse
    }
}).
```

**How it works:**
1. Periodically perturb network weights (very small mutations)
2. Evaluate performance over a window of inferences
3. If better (or not worse), keep the mutation
4. If worse, revert to previous weights

This is [Evolution Strategies](https://arxiv.org/abs/1703.03864) at inference time.

**Trade-off:** Slight performance variance during exploration vs. gradual adaptation.

### 4. Full Continual Learning (Higher Overhead)

Multiple silos work together for comprehensive adaptation:

```erlang
{ok, Pid} = lc_supervisor:start_link(#{
    mode => inference,
    silos => [resource, task, temporal, ecological],
    continual_learning => #{
        concept_drift_detector => true,
        retraining_trigger => #{
            drift_threshold => 0.2,
            performance_drop => 0.15,
            time_since_training => 86400  %% 24 hours
        }
    }
}).
```

**Components:**
- **Resource Silo**: Performance monitoring
- **Task Silo**: Network switching and micro-evolution
- **Temporal Silo**: Tracks performance over time, detects trends
- **Ecological Silo**: Monitors input distribution for concept drift

## Key Components

### 1. Performance Monitor

Extends the Resource Silo for inference-time metrics:

```erlang
-record(inference_metrics, {
    latency_histogram,     %% Latency distribution
    error_count,           %% Errors in current window
    total_count,           %% Total inferences in window
    feedback_scores,       %% Reward/accuracy if available
    window_start,          %% Timestamp
    trend                  %% improving | stable | degrading
}).
```

**Key signals:**
- `inference_latency_exceeded` - P99 latency above threshold
- `error_rate_elevated` - Error rate above threshold
- `throughput_degraded` - Throughput below threshold
- `performance_trending_down` - Gradual degradation detected

### 2. Concept Drift Detector

Monitors input distribution for changes:

```erlang
-record(drift_detector, {
    reference_distribution,  %% Input stats from training
    current_window,          %% Recent input stats
    drift_score,             %% How much has distribution changed
    drift_type               %% gradual | sudden | recurring
}).
```

**Detection methods:**
- **Statistical tests**: Kolmogorov-Smirnov, Chi-squared on input features
- **Page-Hinkley**: Detects mean shift in performance metrics
- **ADWIN**: Adaptive windowing for gradual drift
- **Feature monitoring**: Track individual input feature distributions

### 3. Network Portfolio Manager

Manages multiple evolved networks:

```erlang
-record(portfolio, {
    networks,              %% List of {NetworkId, NetworkData}
    performance_history,   %% #{NetworkId => [scores]}
    active_network,        %% Currently deployed network
    switch_count,          %% How often we've switched
    context_mapping        %% #{Context => PreferredNetwork}
}).
```

**Portfolio strategies:**
- **Best performer**: Always use highest-scoring network
- **Context-aware**: Different networks for different input contexts
- **Exploration**: Occasionally try other networks to update scores
- **Aging**: Retire networks that haven't been good recently

### 4. Micro-Evolution Module

Applies ES-style updates during inference:

```erlang
micro_evolve(Network, Performance, Config) ->
    %% Generate small perturbation
    Delta = generate_perturbation(Network, Config#config.mutation_strength),
    PerturbedNetwork = apply_perturbation(Network, Delta),

    %% Evaluate over window
    NewPerformance = evaluate_window(PerturbedNetwork, Config#config.window_size),

    %% Accept or reject
    case NewPerformance >= Performance - Config#config.acceptance_threshold of
        true -> {ok, PerturbedNetwork, NewPerformance};
        false -> {reject, Network, Performance}
    end.
```

## Implementation Phases

We recommend implementing continual learning in phases:

| Phase | Feature | Complexity | Value |
|-------|---------|------------|-------|
| 1 | Performance monitoring (latency, error rate) | Low | High |
| 2 | Network switching (manual portfolio) | Low | Medium |
| 3 | Automatic concept drift detection | Medium | High |
| 4 | Micro-evolution during inference | Medium | Medium |
| 5 | Automatic network selection | High | High |
| 6 | Full retraining integration | High | Medium |

### Phase 1: Performance Monitoring

Start with basic metrics collection:

```erlang
%% In your inference path
Result = network:evaluate(Network, Input),
lc_resource_silo:record_inference(#{
    latency => timer:now_diff(End, Start) / 1000,
    success => is_valid(Result),
    input_features => extract_features(Input)
}).

%% Resource Silo publishes alerts
handle_info({alert, inference_latency_exceeded, #{p99 := Latency}}, State) ->
    %% Log, notify, or take action
    ...
```

### Phase 2: Network Switching

Add portfolio management:

```erlang
%% Initialize portfolio during deployment
Portfolio = portfolio_manager:init([
    {champion, ChampionNetwork},      %% Best from training
    {runner_up, RunnerUpNetwork},     %% Second best
    {generalist, GeneralistNetwork}   %% Robust to variation
]),

%% Switch on performance drop
handle_info({performance_drop, NetworkId, Drop}, State) when Drop > 0.1 ->
    {ok, NewActive} = portfolio_manager:select_alternative(Portfolio, NetworkId),
    {noreply, State#state{active_network = NewActive}}.
```

### Phase 3: Drift Detection

Implement statistical monitoring:

```erlang
%% During training, capture reference distribution
ReferenceStats = drift_detector:capture_training_distribution(TrainingData),

%% During inference, monitor for drift
handle_inference(Input, State) ->
    drift_detector:update(State#state.detector, Input),
    DriftScore = drift_detector:get_drift_score(State#state.detector),

    case DriftScore > 0.2 of
        true ->
            %% Significant drift detected
            trigger_adaptation(State);
        false ->
            State
    end.
```

## Research Questions

Continual learning opens several research directions:

### Stability-Plasticity Tradeoff

How much should the network adapt without forgetting what it learned during training?

- **Too much plasticity**: Catastrophic forgetting, overfit to recent data
- **Too little plasticity**: Cannot adapt to genuine changes

**Approaches:**
- Elastic Weight Consolidation (EWC): Penalize changes to important weights
- Progressive networks: Add new capacity rather than modifying old
- Rehearsal: Occasionally replay training examples

### When to Switch vs Adapt

Is it better to switch to a different network or adapt the current one?

- **Switch**: Fast, no risk to current network, limited expressiveness
- **Adapt**: Slower, risk of degradation, potentially better fit

**Factors to consider:**
- Speed of change (sudden vs gradual)
- Reversibility (will conditions return?)
- Performance gap between portfolio members

### Feedback Signal Design

What defines "good" inference performance when we don't have explicit rewards?

**Proxy signals:**
- User engagement (clicks, time on task)
- Downstream system behavior
- Anomaly detection (unusual outputs)
- Consistency with recent inferences

### Catastrophic Forgetting Prevention

How to prevent unlearning useful knowledge during adaptation?

**Strategies:**
- Keep a frozen copy of the original network for comparison
- Use very small mutation strengths (1% of training)
- Implement change budgets (max total change allowed)
- Require improvements on validation set, not just recent data

## Use Cases

### 1. Game AI

**Scenario:** Opponents learn and adapt their strategies.

```erlang
%% Competitive Silo tracks opponent strategies
%% Network portfolio includes counter-strategies
%% Micro-evolution allows fine-tuning against current opponent
```

**Approach:**
- Track opponent behavior patterns (Competitive Silo)
- Switch to network trained against similar patterns
- Micro-evolve to specialize against this specific opponent

### 2. Robotics Control

**Scenario:** Physical wear changes robot dynamics over time.

```erlang
%% Resource Silo monitors control errors
%% Temporal Silo detects gradual degradation trends
%% Micro-evolution adjusts motor commands
```

**Approach:**
- Monitor position/velocity errors
- Detect when errors trend upward
- Small weight adjustments to compensate for wear

### 3. Anomaly Detection

**Scenario:** Normal behavior patterns shift seasonally.

```erlang
%% Ecological Silo monitors input distribution
%% Concept drift detection triggers adaptation
%% Portfolio includes networks trained on different seasons
```

**Approach:**
- Track input feature distributions
- Detect seasonal shifts in patterns
- Switch to appropriate seasonal model

### 4. Trading Systems

**Scenario:** Market regime changes (volatility, trends).

```erlang
%% Ecological Silo detects regime changes
%% Portfolio includes regime-specific strategies
%% Risk management via Resource Silo
```

**Approach:**
- Monitor market indicators for regime detection
- Switch to appropriate strategy (momentum vs mean-reversion)
- Conservative mode under uncertainty

## Summary

| Mode | Overhead | Adaptation Speed | Risk | Best For |
|------|----------|------------------|------|----------|
| **Monitoring only** | Minimal | None (manual) | None | Alerting, debugging |
| **Portfolio switching** | Low | Fast | Low | Multi-regime environments |
| **Micro-evolution** | Medium | Gradual | Medium | Gradual drift |
| **Full continual** | Higher | Comprehensive | Medium | Complex, dynamic environments |

**Key insight:** Start simple with monitoring and portfolio switching. Add micro-evolution only when you've validated that gradual adaptation helps. Full continual learning is powerful but requires careful design to avoid catastrophic forgetting.

## References

### Continual Learning

- Parisi, G. I., et al. (2019). "Continual lifelong learning with neural networks: A review." *Neural Networks*, 113, 54-71.
- Kirkpatrick, J., et al. (2017). "Overcoming catastrophic forgetting in neural networks." *PNAS*, 114(13), 3521-3526.

### Concept Drift

- Gama, J., et al. (2014). "A survey on concept drift adaptation." *ACM computing surveys*, 46(4), 1-37.
- Lu, J., et al. (2018). "Learning under concept drift: A review." *IEEE Transactions on Knowledge and Data Engineering*, 31(12), 2346-2363.

### Evolution Strategies

- Salimans, T., et al. (2017). "Evolution strategies as a scalable alternative to reinforcement learning." *arXiv:1703.03864*.

### Multi-Armed Bandits (for Network Selection)

- Auer, P., et al. (2002). "Finite-time analysis of the multiarmed bandit problem." *Machine learning*, 47(2), 235-256.

## Related Guides

- [Liquid Conglomerate](liquid-conglomerate.md) - Full LC architecture
- [Training Strategies](training-strategies.md) - Training phase strategies
- [Inference Scenarios](inference-scenarios.md) - Deployment patterns
- [Custom Evaluator](custom-evaluator.md) - Domain-specific metrics
