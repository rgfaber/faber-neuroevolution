# Plan: Temporal Silo for Liquid Conglomerate

**Status:** Planning
**Created:** 2025-12-23
**Last Updated:** 2025-12-23
**Related:** PLAN_SOCIAL_SILO.md, PLAN_CULTURAL_SILO.md, PLAN_ECOLOGICAL_SILO.md, PLAN_MORPHOLOGICAL_SILO.md

---

## Overview

The Temporal Silo manages time-related dynamics in neuroevolution: how long to evaluate individuals, episode durations, reaction time budgets, and credit assignment horizons. Poor time management wastes compute on hopeless individuals or cuts off promising ones too early.

---

## 1. Motivation

### Problem Statement

Traditional neuroevolution uses fixed evaluation times regardless of:
- Individual quality (bad individuals evaluated as long as good ones)
- Task difficulty (simple tasks get same time as complex ones)
- Convergence state (near-converged populations don't need long evaluations)
- Resource constraints (no adaptive time budgeting)

This wastes 30-70% of evaluation compute on:
- Hopeless individuals that should be terminated early
- Already-converged populations that don't need full episodes
- Overly long episodes for simple tasks
- Insufficient episodes for complex tasks

### Business Value

| Benefit | Impact |
|---------|--------|
| Training efficiency | 2-5x speedup from early termination |
| Real-time guarantees | Evolve within latency SLAs |
| Cloud cost optimization | Pay only for necessary compute |
| Deployment flexibility | Same network, different time budgets |
| Adaptive difficulty | Episode length matches task complexity |

### Training Velocity Impact

| Metric | Without Temporal Silo | With Temporal Silo |
|--------|----------------------|-------------------|
| Wasted evaluation time | 40-60% | 10-20% |
| Time to convergence | Baseline | 0.4-0.6x (faster) |
| Compute efficiency | 1.0x | 2-4x |
| Real-time compliance | No guarantees | Enforced limits |

---

## 2. Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        TEMPORAL SILO                                 │
├─────────────────────────────────────────────────────────────────────┤
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │                      L0 SENSORS (12)                         │    │
│  │                                                              │    │
│  │  Learning         Episode        Reaction      Convergence  │    │
│  │  ┌─────────┐     ┌─────────┐    ┌─────────┐  ┌─────────┐   │    │
│  │  │learning_│     │episode_ │    │reaction_│  │convergence│  │    │
│  │  │rate     │     │length_  │    │time_    │  │_rate      │  │    │
│  │  │learning_│     │mean     │    │budget   │  │oscillation│  │    │
│  │  │rate_    │     │episode_ │    │reaction_│  │_frequency │  │    │
│  │  │trend    │     │variance │    │time_used│  └─────────┘   │    │
│  │  └─────────┘     └─────────┘    └─────────┘                 │    │
│  │                                                              │    │
│  │  Credit          Patience       Efficiency                  │    │
│  │  ┌─────────┐     ┌─────────┐   ┌─────────┐                 │    │
│  │  │discount_│     │patience_│   │eval_    │                 │    │
│  │  │factor   │     │remaining│   │efficiency│                │    │
│  │  │credit_  │     │early_   │   └─────────┘                 │    │
│  │  │horizon  │     │term_rate│                               │    │
│  │  └─────────┘     └─────────┘                               │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                                │                                     │
│                       ┌────────▼────────┐                           │
│                       │ TWEANN Controller│                          │
│                       │   (online ES)   │                           │
│                       └────────┬────────┘                           │
│                                │                                     │
│  ┌─────────────────────────────▼───────────────────────────────┐    │
│  │                      L0 ACTUATORS (10)                       │    │
│  │                                                              │    │
│  │  Learning         Episode        Timeout       Patience     │    │
│  │  ┌─────────┐     ┌─────────┐    ┌─────────┐  ┌─────────┐   │    │
│  │  │learning_│     │episode_ │    │eval_    │  │patience_ │   │    │
│  │  │rate_mult│     │length_  │    │timeout  │  │threshold │   │    │
│  │  │learning_│     │target   │    │reaction_│  │early_    │   │    │
│  │  │rate_    │     │episode_ │    │time_    │  │term_     │   │    │
│  │  │decay    │     │variance │    │limit    │  │threshold │   │    │
│  │  └─────────┘     │_allowed │    └─────────┘  └─────────┘   │    │
│  │                  └─────────┘                                │    │
│  └─────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 3. L0 Sensors

### 3.1 Sensor Specifications

| ID | Name | Range | Description |
|----|------|-------|-------------|
| 1 | `learning_rate_current` | [0.0, 1.0] | Current network learning rate (normalized) |
| 2 | `learning_rate_trend` | [-1.0, 1.0] | Direction of learning rate change over time |
| 3 | `episode_length_mean` | [0.0, 1.0] | Average episode duration / max duration |
| 4 | `episode_length_variance` | [0.0, 1.0] | Variance in episode lengths (normalized) |
| 5 | `reaction_time_budget` | [0.0, 1.0] | Allowed reaction time per step (normalized) |
| 6 | `reaction_time_used` | [0.0, 1.0] | Actual reaction time / budget |
| 7 | `discount_factor` | [0.0, 1.0] | Temporal discount for credit assignment |
| 8 | `credit_horizon` | [0.0, 1.0] | How far back credit propagates (normalized) |
| 9 | `convergence_rate` | [0.0, 1.0] | Speed of fitness convergence |
| 10 | `oscillation_frequency` | [0.0, 1.0] | Fitness oscillation (instability indicator) |
| 11 | `patience_remaining` | [0.0, 1.0] | Generations until early stopping |
| 12 | `evaluation_efficiency` | [0.0, 1.0] | Fitness gained per evaluation time |

### 3.2 Sensor Module

```erlang
%%%-------------------------------------------------------------------
%%% @doc Temporal Silo L0 Sensors
%%% Monitors time-related dynamics: episode lengths, reaction times,
%%% convergence patterns, and evaluation efficiency.
%%% @end
%%%-------------------------------------------------------------------
-module(temporal_silo_sensors).

-behaviour(l0_sensor_behaviour).

%% API
-export([sensor_specs/0,
         collect_sensors/1,
         sensor_count/0]).

%% Sensor collection
-export([collect_learning_rate_current/1,
         collect_learning_rate_trend/1,
         collect_episode_length_mean/1,
         collect_episode_length_variance/1,
         collect_reaction_time_budget/1,
         collect_reaction_time_used/1,
         collect_discount_factor/1,
         collect_credit_horizon/1,
         collect_convergence_rate/1,
         collect_oscillation_frequency/1,
         collect_patience_remaining/1,
         collect_evaluation_efficiency/1]).

-include("temporal_silo.hrl").

%%====================================================================
%% Behaviour Callbacks
%%====================================================================

-spec sensor_specs() -> [l0_sensor_spec()].
sensor_specs() ->
    [
        #{id => learning_rate_current,
          name => <<"Learning Rate Current">>,
          range => {0.0, 1.0},
          description => <<"Current network learning rate (normalized)">>},

        #{id => learning_rate_trend,
          name => <<"Learning Rate Trend">>,
          range => {-1.0, 1.0},
          description => <<"Direction of learning rate change">>},

        #{id => episode_length_mean,
          name => <<"Episode Length Mean">>,
          range => {0.0, 1.0},
          description => <<"Average episode duration / max duration">>},

        #{id => episode_length_variance,
          name => <<"Episode Length Variance">>,
          range => {0.0, 1.0},
          description => <<"Variance in episode lengths">>},

        #{id => reaction_time_budget,
          name => <<"Reaction Time Budget">>,
          range => {0.0, 1.0},
          description => <<"Allowed reaction time (normalized)">>},

        #{id => reaction_time_used,
          name => <<"Reaction Time Used">>,
          range => {0.0, 1.0},
          description => <<"Actual reaction time / budget">>},

        #{id => discount_factor,
          name => <<"Discount Factor">>,
          range => {0.0, 1.0},
          description => <<"Temporal discount for credit assignment">>},

        #{id => credit_horizon,
          name => <<"Credit Horizon">>,
          range => {0.0, 1.0},
          description => <<"How far back credit propagates">>},

        #{id => convergence_rate,
          name => <<"Convergence Rate">>,
          range => {0.0, 1.0},
          description => <<"Speed of fitness convergence">>},

        #{id => oscillation_frequency,
          name => <<"Oscillation Frequency">>,
          range => {0.0, 1.0},
          description => <<"Fitness oscillation (instability)">>},

        #{id => patience_remaining,
          name => <<"Patience Remaining">>,
          range => {0.0, 1.0},
          description => <<"Generations until early stopping">>},

        #{id => evaluation_efficiency,
          name => <<"Evaluation Efficiency">>,
          range => {0.0, 1.0},
          description => <<"Fitness gained per evaluation time">>}
    ].

-spec sensor_count() -> pos_integer().
sensor_count() -> 12.

-spec collect_sensors(temporal_context()) -> sensor_vector().
collect_sensors(Context) ->
    [
        collect_learning_rate_current(Context),
        collect_learning_rate_trend(Context),
        collect_episode_length_mean(Context),
        collect_episode_length_variance(Context),
        collect_reaction_time_budget(Context),
        collect_reaction_time_used(Context),
        collect_discount_factor(Context),
        collect_credit_horizon(Context),
        collect_convergence_rate(Context),
        collect_oscillation_frequency(Context),
        collect_patience_remaining(Context),
        collect_evaluation_efficiency(Context)
    ].

%%====================================================================
%% Individual Sensor Collection
%%====================================================================

%% @doc Current learning rate normalized to [0,1]
-spec collect_learning_rate_current(temporal_context()) -> float().
collect_learning_rate_current(#temporal_context{learning_rates = Rates}) ->
    case Rates of
        [] -> 0.5;
        [Current | _] ->
            %% Normalize assuming typical range 0.0001 to 0.1
            normalize_log_scale(Current, 0.0001, 0.1)
    end.

%% @doc Trend in learning rate change
-spec collect_learning_rate_trend(temporal_context()) -> float().
collect_learning_rate_trend(#temporal_context{learning_rates = Rates}) ->
    case length(Rates) >= 3 of
        false -> 0.0;
        true ->
            Recent = lists:sublist(Rates, 3),
            calculate_trend(Recent)
    end.

%% @doc Mean episode length relative to max
-spec collect_episode_length_mean(temporal_context()) -> float().
collect_episode_length_mean(#temporal_context{
    episode_lengths = Lengths,
    max_episode_length = MaxLen
}) ->
    case Lengths of
        [] -> 0.5;
        _ ->
            Mean = lists:sum(Lengths) / length(Lengths),
            clamp(Mean / MaxLen, 0.0, 1.0)
    end.

%% @doc Variance in episode lengths
-spec collect_episode_length_variance(temporal_context()) -> float().
collect_episode_length_variance(#temporal_context{
    episode_lengths = Lengths,
    max_episode_length = MaxLen
}) ->
    case length(Lengths) >= 2 of
        false -> 0.0;
        true ->
            Mean = lists:sum(Lengths) / length(Lengths),
            Variance = lists:sum([math:pow(L - Mean, 2) || L <- Lengths]) / length(Lengths),
            StdDev = math:sqrt(Variance),
            %% Normalize by max possible variance
            clamp(StdDev / (MaxLen / 2), 0.0, 1.0)
    end.

%% @doc Reaction time budget normalized
-spec collect_reaction_time_budget(temporal_context()) -> float().
collect_reaction_time_budget(#temporal_context{
    reaction_time_budget_ms = Budget,
    max_reaction_time_ms = MaxReaction
}) ->
    clamp(Budget / MaxReaction, 0.0, 1.0).

%% @doc Reaction time used relative to budget
-spec collect_reaction_time_used(temporal_context()) -> float().
collect_reaction_time_used(#temporal_context{
    reaction_times = Times,
    reaction_time_budget_ms = Budget
}) ->
    case Times of
        [] -> 0.5;
        _ ->
            MeanUsed = lists:sum(Times) / length(Times),
            clamp(MeanUsed / Budget, 0.0, 1.0)
    end.

%% @doc Current discount factor for credit assignment
-spec collect_discount_factor(temporal_context()) -> float().
collect_discount_factor(#temporal_context{discount_factor = Gamma}) ->
    clamp(Gamma, 0.0, 1.0).

%% @doc Credit assignment horizon normalized
-spec collect_credit_horizon(temporal_context()) -> float().
collect_credit_horizon(#temporal_context{
    credit_horizon = Horizon,
    max_credit_horizon = MaxHorizon
}) ->
    clamp(Horizon / MaxHorizon, 0.0, 1.0).

%% @doc Rate of fitness convergence
-spec collect_convergence_rate(temporal_context()) -> float().
collect_convergence_rate(#temporal_context{fitness_history = History}) ->
    case length(History) >= 5 of
        false -> 0.5;
        true ->
            Recent = lists:sublist(History, 10),
            %% Compute convergence as reduction in variance
            calculate_convergence_rate(Recent)
    end.

%% @doc Frequency of fitness oscillation
-spec collect_oscillation_frequency(temporal_context()) -> float().
collect_oscillation_frequency(#temporal_context{fitness_history = History}) ->
    case length(History) >= 5 of
        false -> 0.0;
        true ->
            Recent = lists:sublist(History, 20),
            calculate_oscillation_frequency(Recent)
    end.

%% @doc Patience remaining before early stopping
-spec collect_patience_remaining(temporal_context()) -> float().
collect_patience_remaining(#temporal_context{
    generations_without_improvement = NoImprovement,
    patience_threshold = Patience
}) ->
    Remaining = max(0, Patience - NoImprovement),
    clamp(Remaining / Patience, 0.0, 1.0).

%% @doc Evaluation efficiency (fitness per time)
-spec collect_evaluation_efficiency(temporal_context()) -> float().
collect_evaluation_efficiency(#temporal_context{
    fitness_history = History,
    evaluation_times = Times
}) ->
    case {History, Times} of
        {[], _} -> 0.5;
        {_, []} -> 0.5;
        {[F1, F2 | _], [T1 | _]} when T1 > 0 ->
            FitnessGain = abs(F1 - F2),
            %% Normalize efficiency
            Efficiency = FitnessGain / T1,
            sigmoid(Efficiency * 1000)  % Scale appropriately
    end.

%%====================================================================
%% Internal Functions
%%====================================================================

normalize_log_scale(Value, Min, Max) ->
    LogMin = math:log10(Min),
    LogMax = math:log10(Max),
    LogVal = math:log10(max(Min, min(Max, Value))),
    clamp((LogVal - LogMin) / (LogMax - LogMin), 0.0, 1.0).

calculate_trend(Values) ->
    N = length(Values),
    Indices = lists:seq(1, N),
    MeanX = (N + 1) / 2,
    MeanY = lists:sum(Values) / N,
    Numerator = lists:sum([((I - MeanX) * (V - MeanY)) || {I, V} <- lists:zip(Indices, Values)]),
    Denominator = lists:sum([math:pow(I - MeanX, 2) || I <- Indices]),
    case Denominator of
        0.0 -> 0.0;
        _ -> clamp(Numerator / Denominator, -1.0, 1.0)
    end.

calculate_convergence_rate(History) ->
    N = length(History),
    case N >= 3 of
        false -> 0.5;
        true ->
            FirstHalf = lists:sublist(History, N div 2),
            SecondHalf = lists:nthtail(N div 2, History),
            Var1 = variance(FirstHalf),
            Var2 = variance(SecondHalf),
            case Var1 of
                0.0 -> 1.0;
                _ -> clamp(1.0 - (Var2 / Var1), 0.0, 1.0)
            end
    end.

calculate_oscillation_frequency(History) ->
    %% Count sign changes in differences
    Diffs = [B - A || {A, B} <- lists:zip(lists:droplast(History), tl(History))],
    Signs = [sign(D) || D <- Diffs],
    SignChanges = count_sign_changes(Signs),
    MaxChanges = length(Signs) - 1,
    case MaxChanges of
        0 -> 0.0;
        _ -> clamp(SignChanges / MaxChanges, 0.0, 1.0)
    end.

count_sign_changes(Signs) ->
    count_sign_changes(Signs, 0).

count_sign_changes([_], Count) -> Count;
count_sign_changes([A, B | Rest], Count) when A =/= B, A =/= 0, B =/= 0 ->
    count_sign_changes([B | Rest], Count + 1);
count_sign_changes([_, B | Rest], Count) ->
    count_sign_changes([B | Rest], Count).

sign(X) when X > 0 -> 1;
sign(X) when X < 0 -> -1;
sign(_) -> 0.

variance([]) -> 0.0;
variance(Values) ->
    Mean = lists:sum(Values) / length(Values),
    lists:sum([math:pow(V - Mean, 2) || V <- Values]) / length(Values).

sigmoid(X) ->
    1.0 / (1.0 + math:exp(-X)).

clamp(Value, Min, Max) ->
    max(Min, min(Max, Value)).
```

---

## 4. L0 Actuators

### 4.1 Actuator Specifications

| ID | Name | Range | Default | Description |
|----|------|-------|---------|-------------|
| 1 | `learning_rate_multiplier` | [0.1, 10.0] | 1.0 | Scale factor for learning rates |
| 2 | `learning_rate_decay` | [0.9, 1.0] | 0.99 | Per-generation learning rate decay |
| 3 | `episode_length_target` | [10, 10000] | 1000 | Target episode steps |
| 4 | `episode_variance_allowed` | [0.0, 1.0] | 0.3 | Acceptable variance in episode length |
| 5 | `evaluation_timeout_ms` | [100, 60000] | 5000 | Maximum evaluation time per individual |
| 6 | `reaction_time_limit_ms` | [1, 1000] | 100 | Maximum per-step thinking time |
| 7 | `discount_factor` | [0.9, 0.999] | 0.99 | Gamma for credit assignment |
| 8 | `patience_threshold` | [5, 100] | 20 | Generations without improvement before action |
| 9 | `early_termination_fitness` | [0.0, 1.0] | 0.1 | Fitness below which to terminate early |
| 10 | `eligibility_trace_decay` | [0.0, 1.0] | 0.9 | Lambda for eligibility traces |

### 4.2 Actuator Module

```erlang
%%%-------------------------------------------------------------------
%%% @doc Temporal Silo L0 Actuators
%%% Controls time-related parameters: episode lengths, timeouts,
%%% learning rates, and early termination thresholds.
%%% @end
%%%-------------------------------------------------------------------
-module(temporal_silo_actuators).

-behaviour(l0_actuator_behaviour).

%% API
-export([actuator_specs/0,
         apply_actuators/2,
         actuator_count/0]).

%% Individual actuator application
-export([apply_learning_rate_multiplier/2,
         apply_learning_rate_decay/2,
         apply_episode_length_target/2,
         apply_episode_variance_allowed/2,
         apply_evaluation_timeout/2,
         apply_reaction_time_limit/2,
         apply_discount_factor/2,
         apply_patience_threshold/2,
         apply_early_termination_fitness/2,
         apply_eligibility_trace_decay/2]).

-include("temporal_silo.hrl").

%%====================================================================
%% Behaviour Callbacks
%%====================================================================

-spec actuator_specs() -> [l0_actuator_spec()].
actuator_specs() ->
    [
        #{id => learning_rate_multiplier,
          name => <<"Learning Rate Multiplier">>,
          range => {0.1, 10.0},
          default => 1.0,
          description => <<"Scale factor for learning rates">>},

        #{id => learning_rate_decay,
          name => <<"Learning Rate Decay">>,
          range => {0.9, 1.0},
          default => 0.99,
          description => <<"Per-generation learning rate decay">>},

        #{id => episode_length_target,
          name => <<"Episode Length Target">>,
          range => {10, 10000},
          default => 1000,
          description => <<"Target episode steps">>},

        #{id => episode_variance_allowed,
          name => <<"Episode Variance Allowed">>,
          range => {0.0, 1.0},
          default => 0.3,
          description => <<"Acceptable variance in episode length">>},

        #{id => evaluation_timeout_ms,
          name => <<"Evaluation Timeout">>,
          range => {100, 60000},
          default => 5000,
          description => <<"Maximum evaluation time (ms)">>},

        #{id => reaction_time_limit_ms,
          name => <<"Reaction Time Limit">>,
          range => {1, 1000},
          default => 100,
          description => <<"Max per-step thinking time (ms)">>},

        #{id => discount_factor,
          name => <<"Discount Factor">>,
          range => {0.9, 0.999},
          default => 0.99,
          description => <<"Gamma for credit assignment">>},

        #{id => patience_threshold,
          name => <<"Patience Threshold">>,
          range => {5, 100},
          default => 20,
          description => <<"Generations without improvement tolerance">>},

        #{id => early_termination_fitness,
          name => <<"Early Termination Fitness">>,
          range => {0.0, 1.0},
          default => 0.1,
          description => <<"Fitness below which to terminate">>},

        #{id => eligibility_trace_decay,
          name => <<"Eligibility Trace Decay">>,
          range => {0.0, 1.0},
          default => 0.9,
          description => <<"Lambda for eligibility traces">>}
    ].

-spec actuator_count() -> pos_integer().
actuator_count() -> 10.

-spec apply_actuators(actuator_vector(), temporal_state()) -> temporal_state().
apply_actuators(Outputs, State) when length(Outputs) =:= 10 ->
    [LRMult, LRDecay, EpLen, EpVar, Timeout, ReactLimit,
     Discount, Patience, EarlyTerm, EligDecay] = Outputs,

    State1 = apply_learning_rate_multiplier(LRMult, State),
    State2 = apply_learning_rate_decay(LRDecay, State1),
    State3 = apply_episode_length_target(EpLen, State2),
    State4 = apply_episode_variance_allowed(EpVar, State3),
    State5 = apply_evaluation_timeout(Timeout, State4),
    State6 = apply_reaction_time_limit(ReactLimit, State5),
    State7 = apply_discount_factor(Discount, State6),
    State8 = apply_patience_threshold(Patience, State7),
    State9 = apply_early_termination_fitness(EarlyTerm, State8),
    apply_eligibility_trace_decay(EligDecay, State9).

%%====================================================================
%% Individual Actuator Application
%%====================================================================

%% @doc Apply learning rate multiplier
-spec apply_learning_rate_multiplier(float(), temporal_state()) -> temporal_state().
apply_learning_rate_multiplier(Output, State) ->
    %% Output in [0,1] -> Multiplier in [0.1, 10.0]
    Multiplier = 0.1 + Output * 9.9,
    State#temporal_state{learning_rate_multiplier = Multiplier}.

%% @doc Apply learning rate decay
-spec apply_learning_rate_decay(float(), temporal_state()) -> temporal_state().
apply_learning_rate_decay(Output, State) ->
    %% Output in [0,1] -> Decay in [0.9, 1.0]
    Decay = 0.9 + Output * 0.1,
    State#temporal_state{learning_rate_decay = Decay}.

%% @doc Apply episode length target
-spec apply_episode_length_target(float(), temporal_state()) -> temporal_state().
apply_episode_length_target(Output, State) ->
    #temporal_state{config = Config} = State,
    MinLen = Config#temporal_config.min_episode_length,
    MaxLen = Config#temporal_config.max_episode_length,
    %% Output in [0,1] -> Length in [MinLen, MaxLen]
    Target = round(MinLen + Output * (MaxLen - MinLen)),
    State#temporal_state{episode_length_target = Target}.

%% @doc Apply episode variance allowed
-spec apply_episode_variance_allowed(float(), temporal_state()) -> temporal_state().
apply_episode_variance_allowed(Output, State) ->
    %% Output in [0,1] -> Variance in [0.0, 1.0]
    State#temporal_state{episode_variance_allowed = Output}.

%% @doc Apply evaluation timeout
-spec apply_evaluation_timeout(float(), temporal_state()) -> temporal_state().
apply_evaluation_timeout(Output, State) ->
    #temporal_state{config = Config} = State,
    MinTimeout = Config#temporal_config.min_timeout_ms,
    MaxTimeout = Config#temporal_config.max_timeout_ms,
    %% Output in [0,1] -> Timeout in [MinTimeout, MaxTimeout]
    Timeout = round(MinTimeout + Output * (MaxTimeout - MinTimeout)),
    State#temporal_state{evaluation_timeout_ms = Timeout}.

%% @doc Apply reaction time limit
-spec apply_reaction_time_limit(float(), temporal_state()) -> temporal_state().
apply_reaction_time_limit(Output, State) ->
    %% Output in [0,1] -> Limit in [1, 1000] ms
    Limit = round(1 + Output * 999),
    State#temporal_state{reaction_time_limit_ms = Limit}.

%% @doc Apply discount factor
-spec apply_discount_factor(float(), temporal_state()) -> temporal_state().
apply_discount_factor(Output, State) ->
    %% Output in [0,1] -> Gamma in [0.9, 0.999]
    Gamma = 0.9 + Output * 0.099,
    State#temporal_state{discount_factor = Gamma}.

%% @doc Apply patience threshold
-spec apply_patience_threshold(float(), temporal_state()) -> temporal_state().
apply_patience_threshold(Output, State) ->
    %% Output in [0,1] -> Patience in [5, 100]
    Patience = round(5 + Output * 95),
    State#temporal_state{patience_threshold = Patience}.

%% @doc Apply early termination fitness threshold
-spec apply_early_termination_fitness(float(), temporal_state()) -> temporal_state().
apply_early_termination_fitness(Output, State) ->
    %% Output in [0,1] -> Threshold in [0.0, 1.0]
    State#temporal_state{early_termination_fitness = Output}.

%% @doc Apply eligibility trace decay
-spec apply_eligibility_trace_decay(float(), temporal_state()) -> temporal_state().
apply_eligibility_trace_decay(Output, State) ->
    %% Output in [0,1] -> Lambda in [0.0, 1.0]
    State#temporal_state{eligibility_trace_decay = Output}.
```

---

## 5. Record Definitions

```erlang
%%%-------------------------------------------------------------------
%%% @doc Temporal Silo Header
%%% Record definitions for temporal dynamics management.
%%% @end
%%%-------------------------------------------------------------------

-ifndef(TEMPORAL_SILO_HRL).
-define(TEMPORAL_SILO_HRL, true).

%%====================================================================
%% Types
%%====================================================================

-type sensor_vector() :: [float()].
-type actuator_vector() :: [float()].
-type timestamp_ms() :: non_neg_integer().
-type episode_length() :: pos_integer().
-type generation() :: non_neg_integer().

%%====================================================================
%% Context Record (Input to Sensors)
%%====================================================================

-record(temporal_context, {
    %% Learning rate history (most recent first)
    learning_rates = [] :: [float()],

    %% Episode length history
    episode_lengths = [] :: [episode_length()],
    max_episode_length = 10000 :: episode_length(),

    %% Reaction time tracking
    reaction_times = [] :: [timestamp_ms()],
    reaction_time_budget_ms = 100 :: timestamp_ms(),
    max_reaction_time_ms = 1000 :: timestamp_ms(),

    %% Credit assignment
    discount_factor = 0.99 :: float(),
    credit_horizon = 50 :: pos_integer(),
    max_credit_horizon = 100 :: pos_integer(),

    %% Fitness history
    fitness_history = [] :: [float()],

    %% Patience tracking
    generations_without_improvement = 0 :: non_neg_integer(),
    patience_threshold = 20 :: pos_integer(),

    %% Evaluation timing
    evaluation_times = [] :: [timestamp_ms()]
}).

-type temporal_context() :: #temporal_context{}.

%%====================================================================
%% State Record (Silo Internal State)
%%====================================================================

-record(temporal_state, {
    %% Configuration
    config :: temporal_config(),

    %% Current actuator outputs
    learning_rate_multiplier = 1.0 :: float(),
    learning_rate_decay = 0.99 :: float(),
    episode_length_target = 1000 :: episode_length(),
    episode_variance_allowed = 0.3 :: float(),
    evaluation_timeout_ms = 5000 :: timestamp_ms(),
    reaction_time_limit_ms = 100 :: timestamp_ms(),
    discount_factor = 0.99 :: float(),
    patience_threshold = 20 :: pos_integer(),
    early_termination_fitness = 0.1 :: float(),
    eligibility_trace_decay = 0.9 :: float(),

    %% Tracking
    current_generation = 0 :: generation(),
    total_evaluation_time_ms = 0 :: timestamp_ms(),
    total_episodes_completed = 0 :: non_neg_integer(),
    early_terminations = 0 :: non_neg_integer(),
    timeout_count = 0 :: non_neg_integer(),

    %% L2 integration
    l2_enabled = false :: boolean(),
    l2_guidance = undefined :: l2_guidance() | undefined,

    %% History for analysis
    episode_history = [] :: [{generation(), episode_length()}],
    efficiency_history = [] :: [{generation(), float()}]
}).

-type temporal_state() :: #temporal_state{}.

%%====================================================================
%% Configuration Record
%%====================================================================

-record(temporal_config, {
    %% Enable/disable
    enabled = true :: boolean(),

    %% Episode length bounds
    min_episode_length = 10 :: episode_length(),
    max_episode_length = 10000 :: episode_length(),

    %% Timeout bounds
    min_timeout_ms = 100 :: timestamp_ms(),
    max_timeout_ms = 60000 :: timestamp_ms(),

    %% Early termination
    enable_early_termination = true :: boolean(),
    min_steps_before_termination = 10 :: pos_integer(),

    %% Real-time constraints
    enforce_reaction_time = false :: boolean(),

    %% History limits
    max_history_size = 100 :: pos_integer(),

    %% L2 integration
    l2_query_interval = 5 :: pos_integer(),

    %% Event emission
    emit_events = true :: boolean()
}).

-type temporal_config() :: #temporal_config{}.

%%====================================================================
%% Evaluation Session Record
%%====================================================================

-record(evaluation_session, {
    %% Identity
    individual_id :: binary(),
    generation :: generation(),

    %% Timing
    start_time_ms :: timestamp_ms(),
    end_time_ms :: timestamp_ms() | undefined,

    %% Episode
    target_episode_length :: episode_length(),
    actual_episode_length = 0 :: episode_length(),

    %% Reaction times per step
    step_reaction_times = [] :: [timestamp_ms()],

    %% Termination
    termination_reason :: normal | timeout | early_termination | reaction_time_exceeded,

    %% Results
    fitness :: float() | undefined,
    fitness_trajectory = [] :: [{pos_integer(), float()}]
}).

-type evaluation_session() :: #evaluation_session{}.

%%====================================================================
%% L2 Guidance Record
%%====================================================================

-record(l2_guidance, {
    %% Time pressure multiplier
    time_pressure = 1.0 :: float(),

    %% Episode length guidance
    episode_length_factor = 1.0 :: float(),

    %% Early termination aggressiveness
    termination_aggression = 0.5 :: float(),

    %% Patience guidance
    patience_factor = 1.0 :: float()
}).

-type l2_guidance() :: #l2_guidance{}.

%%====================================================================
%% Constants
%%====================================================================

-define(DEFAULT_EPISODE_LENGTH, 1000).
-define(DEFAULT_TIMEOUT_MS, 5000).
-define(DEFAULT_REACTION_TIME_MS, 100).
-define(HISTORY_WINDOW, 50).

-endif.
```

---

## 6. Core Silo Implementation

```erlang
%%%-------------------------------------------------------------------
%%% @doc Temporal Silo
%%% Manages time-related dynamics for neuroevolution: episode lengths,
%%% evaluation timeouts, reaction times, and early termination.
%%% @end
%%%-------------------------------------------------------------------
-module(temporal_silo).

-behaviour(gen_server).

%% API
-export([start_link/1,
         get_temporal_params/1,
         update_context/2,
         record_episode/3,
         record_reaction_time/3,
         should_terminate_early/2,
         check_timeout/2,
         get_state/1,
         enable/1,
         disable/1,
         is_enabled/1]).

%% Cross-silo signals
-export([signal_time_pressure/1,
         signal_convergence_status/1,
         signal_episode_efficiency/1,
         receive_computation_budget/2,
         receive_stagnation_severity/2,
         receive_population_diversity/2]).

%% gen_server callbacks
-export([init/1, handle_call/3, handle_cast/2, handle_info/2,
         terminate/2, code_change/3]).

-include("temporal_silo.hrl").

%%====================================================================
%% API
%%====================================================================

-spec start_link(temporal_config()) -> {ok, pid()} | {error, term()}.
start_link(Config) ->
    gen_server:start_link({local, ?MODULE}, ?MODULE, Config, []).

-spec get_temporal_params(pid()) -> map().
get_temporal_params(Pid) ->
    gen_server:call(Pid, get_temporal_params).

-spec update_context(pid(), temporal_context()) -> ok.
update_context(Pid, Context) ->
    gen_server:cast(Pid, {update_context, Context}).

-spec record_episode(pid(), binary(), episode_length()) -> ok.
record_episode(Pid, IndividualId, Length) ->
    gen_server:cast(Pid, {record_episode, IndividualId, Length}).

-spec record_reaction_time(pid(), binary(), timestamp_ms()) -> ok.
record_reaction_time(Pid, IndividualId, TimeMs) ->
    gen_server:cast(Pid, {record_reaction_time, IndividualId, TimeMs}).

-spec should_terminate_early(pid(), float()) -> boolean().
should_terminate_early(Pid, CurrentFitness) ->
    gen_server:call(Pid, {should_terminate_early, CurrentFitness}).

-spec check_timeout(pid(), timestamp_ms()) -> boolean().
check_timeout(Pid, ElapsedMs) ->
    gen_server:call(Pid, {check_timeout, ElapsedMs}).

-spec get_state(pid()) -> temporal_state().
get_state(Pid) ->
    gen_server:call(Pid, get_state).

-spec enable(pid()) -> ok.
enable(Pid) ->
    gen_server:call(Pid, enable).

-spec disable(pid()) -> ok.
disable(Pid) ->
    gen_server:call(Pid, disable).

-spec is_enabled(pid()) -> boolean().
is_enabled(Pid) ->
    gen_server:call(Pid, is_enabled).

%%====================================================================
%% Cross-Silo Signal API
%%====================================================================

%% @doc Get current time pressure signal for other silos
-spec signal_time_pressure(pid()) -> float().
signal_time_pressure(Pid) ->
    gen_server:call(Pid, signal_time_pressure).

%% @doc Get convergence status for resource silo
-spec signal_convergence_status(pid()) -> float().
signal_convergence_status(Pid) ->
    gen_server:call(Pid, signal_convergence_status).

%% @doc Get episode efficiency for economic silo
-spec signal_episode_efficiency(pid()) -> float().
signal_episode_efficiency(Pid) ->
    gen_server:call(Pid, signal_episode_efficiency).

%% @doc Receive computation budget signal from economic silo
-spec receive_computation_budget(pid(), float()) -> ok.
receive_computation_budget(Pid, Budget) ->
    gen_server:cast(Pid, {cross_silo, computation_budget, Budget}).

%% @doc Receive stagnation severity from task silo
-spec receive_stagnation_severity(pid(), float()) -> ok.
receive_stagnation_severity(Pid, Severity) ->
    gen_server:cast(Pid, {cross_silo, stagnation_severity, Severity}).

%% @doc Receive population diversity from distribution silo
-spec receive_population_diversity(pid(), float()) -> ok.
receive_population_diversity(Pid, Diversity) ->
    gen_server:cast(Pid, {cross_silo, population_diversity, Diversity}).

%%====================================================================
%% gen_server Callbacks
%%====================================================================

init(Config) ->
    State = #temporal_state{
        config = Config,
        episode_length_target = ?DEFAULT_EPISODE_LENGTH,
        evaluation_timeout_ms = ?DEFAULT_TIMEOUT_MS,
        reaction_time_limit_ms = ?DEFAULT_REACTION_TIME_MS
    },
    {ok, State}.

handle_call(get_temporal_params, _From, State) ->
    Params = #{
        episode_length_target => State#temporal_state.episode_length_target,
        episode_variance_allowed => State#temporal_state.episode_variance_allowed,
        evaluation_timeout_ms => State#temporal_state.evaluation_timeout_ms,
        reaction_time_limit_ms => State#temporal_state.reaction_time_limit_ms,
        learning_rate_multiplier => State#temporal_state.learning_rate_multiplier,
        learning_rate_decay => State#temporal_state.learning_rate_decay,
        discount_factor => State#temporal_state.discount_factor,
        early_termination_fitness => State#temporal_state.early_termination_fitness,
        eligibility_trace_decay => State#temporal_state.eligibility_trace_decay
    },
    {reply, Params, State};

handle_call({should_terminate_early, CurrentFitness}, _From, State) ->
    #temporal_state{
        config = Config,
        early_termination_fitness = Threshold
    } = State,
    ShouldTerminate = Config#temporal_config.enable_early_termination
        andalso CurrentFitness < Threshold,
    case ShouldTerminate of
        true ->
            NewState = State#temporal_state{
                early_terminations = State#temporal_state.early_terminations + 1
            },
            maybe_emit_event(early_termination, #{
                fitness => CurrentFitness,
                threshold => Threshold
            }, NewState),
            {reply, true, NewState};
        false ->
            {reply, false, State}
    end;

handle_call({check_timeout, ElapsedMs}, _From, State) ->
    Timeout = State#temporal_state.evaluation_timeout_ms,
    IsTimeout = ElapsedMs >= Timeout,
    case IsTimeout of
        true ->
            NewState = State#temporal_state{
                timeout_count = State#temporal_state.timeout_count + 1
            },
            maybe_emit_event(evaluation_timeout, #{
                elapsed_ms => ElapsedMs,
                timeout_ms => Timeout
            }, NewState),
            {reply, true, NewState};
        false ->
            {reply, false, State}
    end;

handle_call(get_state, _From, State) ->
    {reply, State, State};

handle_call(enable, _From, State) ->
    Config = State#temporal_state.config,
    NewConfig = Config#temporal_config{enabled = true},
    {reply, ok, State#temporal_state{config = NewConfig}};

handle_call(disable, _From, State) ->
    Config = State#temporal_state.config,
    NewConfig = Config#temporal_config{enabled = false},
    {reply, ok, State#temporal_state{config = NewConfig}};

handle_call(is_enabled, _From, State) ->
    {reply, State#temporal_state.config#temporal_config.enabled, State};

handle_call(signal_time_pressure, _From, State) ->
    %% Compute time pressure based on timeout rate and efficiency
    TimeoutRate = case State#temporal_state.total_episodes_completed of
        0 -> 0.0;
        Total -> State#temporal_state.timeout_count / Total
    end,
    %% High timeout rate = high time pressure
    Pressure = clamp(TimeoutRate * 2.0, 0.0, 1.0),
    {reply, Pressure, State};

handle_call(signal_convergence_status, _From, State) ->
    %% Use efficiency history to estimate convergence
    case State#temporal_state.efficiency_history of
        [] -> {reply, 0.5, State};
        History ->
            Recent = lists:sublist(History, 10),
            Efficiencies = [E || {_, E} <- Recent],
            %% Lower variance = more converged
            Variance = variance(Efficiencies),
            Convergence = clamp(1.0 - Variance * 10, 0.0, 1.0),
            {reply, Convergence, State}
    end;

handle_call(signal_episode_efficiency, _From, State) ->
    case State#temporal_state.efficiency_history of
        [] -> {reply, 0.5, State};
        [{_, Latest} | _] -> {reply, clamp(Latest, 0.0, 1.0), State}
    end;

handle_call(_Request, _From, State) ->
    {reply, {error, unknown_request}, State}.

handle_cast({update_context, Context}, State) ->
    #temporal_state{config = Config} = State,
    case Config#temporal_config.enabled of
        false ->
            {noreply, State};
        true ->
            %% Collect sensors
            SensorVector = temporal_silo_sensors:collect_sensors(Context),

            %% Process through TWEANN controller (if available)
            ActuatorVector = process_through_controller(SensorVector, State),

            %% Apply actuators
            NewState = temporal_silo_actuators:apply_actuators(ActuatorVector, State),

            %% Update generation
            FinalState = NewState#temporal_state{
                current_generation = State#temporal_state.current_generation + 1
            },

            {noreply, FinalState}
    end;

handle_cast({record_episode, IndividualId, Length}, State) ->
    Gen = State#temporal_state.current_generation,
    NewHistory = [{Gen, Length} |
                  lists:sublist(State#temporal_state.episode_history,
                               State#temporal_state.config#temporal_config.max_history_size - 1)],

    maybe_emit_event(episode_completed, #{
        individual_id => IndividualId,
        length => Length,
        target => State#temporal_state.episode_length_target,
        generation => Gen
    }, State),

    NewState = State#temporal_state{
        episode_history = NewHistory,
        total_episodes_completed = State#temporal_state.total_episodes_completed + 1
    },
    {noreply, NewState};

handle_cast({record_reaction_time, IndividualId, TimeMs}, State) ->
    #temporal_state{
        config = Config,
        reaction_time_limit_ms = Limit
    } = State,

    case Config#temporal_config.enforce_reaction_time andalso TimeMs > Limit of
        true ->
            maybe_emit_event(reaction_time_exceeded, #{
                individual_id => IndividualId,
                time_ms => TimeMs,
                limit_ms => Limit
            }, State);
        false ->
            ok
    end,
    {noreply, State};

handle_cast({cross_silo, computation_budget, Budget}, State) ->
    %% Adjust timeout based on budget
    #temporal_state{config = Config} = State,
    MaxTimeout = Config#temporal_config.max_timeout_ms,
    MinTimeout = Config#temporal_config.min_timeout_ms,

    %% Budget in [0,1] -> Timeout scales linearly
    NewTimeout = round(MinTimeout + Budget * (MaxTimeout - MinTimeout)),
    NewState = State#temporal_state{evaluation_timeout_ms = NewTimeout},
    {noreply, NewState};

handle_cast({cross_silo, stagnation_severity, Severity}, State) ->
    %% High stagnation -> try longer episodes
    CurrentTarget = State#temporal_state.episode_length_target,
    #temporal_state{config = Config} = State,
    MaxLen = Config#temporal_config.max_episode_length,

    %% Increase episode length with stagnation
    Adjustment = 1.0 + Severity * 0.5,
    NewTarget = min(MaxLen, round(CurrentTarget * Adjustment)),

    maybe_emit_event(episode_extended, #{
        old_length => CurrentTarget,
        new_length => NewTarget,
        reason => stagnation
    }, State),

    NewState = State#temporal_state{episode_length_target = NewTarget},
    {noreply, NewState};

handle_cast({cross_silo, population_diversity, Diversity}, State) ->
    %% Low diversity -> can terminate similar individuals earlier
    CurrentThreshold = State#temporal_state.early_termination_fitness,

    %% Low diversity raises termination threshold (more aggressive)
    Adjustment = 0.1 * (1.0 - Diversity),
    NewThreshold = clamp(CurrentThreshold + Adjustment, 0.0, 0.5),

    NewState = State#temporal_state{early_termination_fitness = NewThreshold},
    {noreply, NewState};

handle_cast(_Msg, State) ->
    {noreply, State}.

handle_info(_Info, State) ->
    {noreply, State}.

terminate(_Reason, _State) ->
    ok.

code_change(_OldVsn, State, _Extra) ->
    {ok, State}.

%%====================================================================
%% Internal Functions
%%====================================================================

process_through_controller(SensorVector, State) ->
    %% TODO: Integrate with actual TWEANN controller
    %% For now, return default actuator values
    case State#temporal_state.l2_enabled of
        true ->
            apply_l2_guidance(SensorVector, State);
        false ->
            %% Default actuator values (all at 0.5)
            lists:duplicate(temporal_silo_actuators:actuator_count(), 0.5)
    end.

apply_l2_guidance(SensorVector, State) ->
    #temporal_state{l2_guidance = Guidance} = State,
    case Guidance of
        undefined ->
            lists:duplicate(temporal_silo_actuators:actuator_count(), 0.5);
        #l2_guidance{
            time_pressure = TimePressure,
            episode_length_factor = EpFactor,
            termination_aggression = TermAgg,
            patience_factor = PatFactor
        } ->
            %% Apply L2 guidance to actuator outputs
            [
                0.5,                           % learning_rate_multiplier
                0.5,                           % learning_rate_decay
                EpFactor,                      % episode_length_target
                0.3,                           % episode_variance_allowed
                1.0 - TimePressure,            % evaluation_timeout (inverse of pressure)
                1.0 - TimePressure,            % reaction_time_limit
                0.5,                           % discount_factor
                PatFactor,                     % patience_threshold
                TermAgg,                       % early_termination_fitness
                0.5                            % eligibility_trace_decay
            ]
    end.

maybe_emit_event(EventType, Payload, State) ->
    case State#temporal_state.config#temporal_config.emit_events of
        true ->
            Event = #{
                type => EventType,
                silo => temporal,
                timestamp => erlang:system_time(millisecond),
                generation => State#temporal_state.current_generation,
                payload => Payload
            },
            %% TODO: Send to event store
            event_bus:publish(temporal_silo_events, Event);
        false ->
            ok
    end.

variance([]) -> 0.0;
variance(Values) ->
    Mean = lists:sum(Values) / length(Values),
    lists:sum([math:pow(V - Mean, 2) || V <- Values]) / length(Values).

clamp(Value, Min, Max) ->
    max(Min, min(Max, Value)).
```

---

## 7. Cross-Silo Signal Matrix

### 7.1 Outgoing Signals

| Signal | To Silo | Type | Description |
|--------|---------|------|-------------|
| `time_pressure` | Task | float() | High pressure suggests simpler solutions needed |
| `convergence_status` | Resource | float() | Near convergence allows compute reduction |
| `episode_efficiency` | Economic | float() | Compute spent per fitness gained |
| `eval_time_constraint` | Competitive | float() | Available time for evaluation matches |
| `critical_timing` | Developmental | float() | Whether critical period timing matters |

### 7.2 Incoming Signals

| Signal | From Silo | Effect |
|--------|-----------|--------|
| `computation_budget` | Economic | Constrains evaluation timeout |
| `stagnation_severity` | Task | Stagnation may warrant longer episodes |
| `population_diversity` | Distribution | Low diversity allows earlier termination of similar individuals |
| `arms_race_active` | Competitive | Arms race may need more evaluation time |
| `resource_pressure` | Resource | Pressure reduces available time |
| `developmental_stage` | Developmental | Stage affects time budgets |

### 7.3 Complete 13-Silo Matrix (Temporal Row)

| To → | Task | Resource | Distrib | Social | Cultural | Ecological | Morpho | Temporal | Competitive | Develop | Regulatory | Economic | Comm |
|------|------|----------|---------|--------|----------|------------|--------|----------|-------------|---------|------------|----------|------|
| **From Temporal** | time_pres | convergence | - | - | - | cycles | - | - | eval_time | critical | - | budget | - |

---

## 8. Events Emitted

### 8.1 Event Specifications

| Event | Trigger | Payload |
|-------|---------|---------|
| `evaluation_timeout` | Hit time limit | `{individual_id, elapsed_ms, timeout_ms, fitness_at_timeout}` |
| `early_termination` | Terminated early | `{individual_id, reason, fitness, steps_completed}` |
| `episode_extended` | Episode length increased | `{old_length, new_length, reason}` |
| `episode_shortened` | Episode length decreased | `{old_length, new_length, reason}` |
| `reaction_time_exceeded` | Over time budget | `{individual_id, budget_ms, actual_ms}` |
| `convergence_detected` | Fitness stabilized | `{population_id, convergence_rate, final_fitness}` |
| `patience_exhausted` | No improvement | `{population_id, generations_waited, action_taken}` |
| `learning_rate_adjusted` | Rate changed | `{old_rate, new_rate, reason}` |
| `episode_completed` | Episode finished | `{individual_id, length, target, generation}` |
| `temporal_params_updated` | Parameters changed | `{old_params, new_params, trigger}` |

### 8.2 Event Payload Types

```erlang
%% Event type specifications
-type temporal_event() ::
    evaluation_timeout_event() |
    early_termination_event() |
    episode_extended_event() |
    episode_shortened_event() |
    reaction_time_exceeded_event() |
    convergence_detected_event() |
    patience_exhausted_event() |
    learning_rate_adjusted_event() |
    episode_completed_event() |
    temporal_params_updated_event().

-type evaluation_timeout_event() :: #{
    type := evaluation_timeout,
    silo := temporal,
    timestamp := timestamp_ms(),
    generation := generation(),
    payload := #{
        individual_id := binary(),
        elapsed_ms := timestamp_ms(),
        timeout_ms := timestamp_ms(),
        fitness_at_timeout := float() | undefined
    }
}.

-type early_termination_event() :: #{
    type := early_termination,
    silo := temporal,
    timestamp := timestamp_ms(),
    generation := generation(),
    payload := #{
        individual_id := binary(),
        reason := low_fitness | no_progress | resource_limit,
        fitness := float(),
        steps_completed := non_neg_integer()
    }
}.

-type episode_extended_event() :: #{
    type := episode_extended,
    silo := temporal,
    timestamp := timestamp_ms(),
    generation := generation(),
    payload := #{
        old_length := episode_length(),
        new_length := episode_length(),
        reason := stagnation | complexity | l2_guidance
    }
}.

-type convergence_detected_event() :: #{
    type := convergence_detected,
    silo := temporal,
    timestamp := timestamp_ms(),
    generation := generation(),
    payload := #{
        population_id := binary(),
        convergence_rate := float(),
        final_fitness := float(),
        generations_to_converge := non_neg_integer()
    }
}.

-type patience_exhausted_event() :: #{
    type := patience_exhausted,
    silo := temporal,
    timestamp := timestamp_ms(),
    generation := generation(),
    payload := #{
        population_id := binary(),
        generations_waited := non_neg_integer(),
        action_taken := increase_exploration | reset_population | terminate_run
    }
}.
```

---

## 9. Value of Event Storage

### 9.1 Analysis Capabilities

| Stored Events | Analysis Enabled |
|---------------|------------------|
| `evaluation_timeout` | Identify networks that need more time vs hopeless |
| `early_termination` | Learn early signals that predict failure |
| `episode_extended/shortened` | Find optimal episode lengths for task types |
| `convergence_detected` | Build models predicting when to stop training |
| `patience_exhausted` | Tune patience thresholds per task complexity |
| `reaction_time_exceeded` | Profile reaction time distributions for deployment |

### 9.2 Business Intelligence

- **Evaluation Optimization**: Find minimum effective episode length per task type
- **Early Termination Patterns**: ML model training on "which early signals predict final failure?"
- **Timeout Analysis**: Distinguish "needs more time" from "hopeless" individuals
- **Convergence Prediction**: Predict training completion time for resource planning
- **Real-time Profiling**: Understand reaction time distributions for SLA guarantees
- **Cost Attribution**: Track compute time per fitness improvement

### 9.3 Replay Scenarios

| Scenario | Value |
|----------|-------|
| "Find runs that converged in < 50 generations" | Identify efficient configurations |
| "Show all early terminations with fitness > 0.3" | Find false terminations (threshold too aggressive) |
| "Episodes that exceeded target by > 50%" | Identify task complexity mismatches |
| "Networks that timed out but had improving fitness" | Candidates for longer evaluation |

---

## 10. Multi-Layer Hierarchy

### 10.1 Layer Responsibilities

| Level | Role | Controls |
|-------|------|----------|
| **L0** | Hard limits | Absolute timeouts, minimum episode lengths, safety bounds |
| **L1** | Tactical | Adjust episode length based on recent performance, react to timeouts |
| **L2** | Strategic | Learn task-appropriate temporal profiles, optimize across generations |

### 10.2 L2 Integration

```erlang
%% L2 guidance for temporal silo
-record(temporal_l2_guidance, {
    %% How aggressive time pressure should be
    time_pressure_factor = 1.0 :: float(),  % [0.5, 2.0]

    %% Episode length scaling
    episode_length_factor = 1.0 :: float(),  % [0.5, 2.0]

    %% Early termination aggressiveness
    termination_aggression = 0.5 :: float(),  % [0.0, 1.0]

    %% Patience multiplier
    patience_factor = 1.0 :: float()  % [0.5, 2.0]
}).

%% L2 queries temporal silo state and provides guidance
-spec get_l2_guidance(temporal_state(), l2_context()) -> temporal_l2_guidance().
get_l2_guidance(State, L2Context) ->
    %% L2 network processes state and context
    %% Returns guidance that L1 uses to scale its adjustments
    #temporal_l2_guidance{
        time_pressure_factor = compute_time_pressure_factor(State, L2Context),
        episode_length_factor = compute_episode_factor(State, L2Context),
        termination_aggression = compute_termination_aggression(State, L2Context),
        patience_factor = compute_patience_factor(State, L2Context)
    }.
```

---

## 11. Enable/Disable Effects

### 11.1 When Disabled

| Aspect | Behavior |
|--------|----------|
| Episode length | Fixed at configured default (no adaptation) |
| Early termination | Disabled (all individuals evaluated fully) |
| Reaction time | No enforcement (networks can think indefinitely) |
| Convergence detection | No detection (no automatic stopping) |
| Patience | No patience mechanism (runs until max generations) |
| Timeouts | Fixed timeout (no adaptive adjustment) |

### 11.2 When Enabled

| Aspect | Behavior |
|--------|----------|
| Episode length | Adapts to task difficulty and stagnation |
| Early termination | Saves compute on hopeless individuals |
| Reaction time | Enforced for real-time applications |
| Convergence detection | Detects and signals when converged |
| Patience | Adaptive patience based on improvement rate |
| Timeouts | Adjust based on budget and performance |

### 11.3 Switching Effects

**Enabling mid-run:**
- Immediate effect on new evaluations
- May terminate currently-evaluating individuals if over new limits
- Episode lengths may change abruptly
- Historical data starts accumulating

**Disabling mid-run:**
- All time constraints removed
- Evaluations run to completion regardless of fitness
- No more temporal events emitted
- Accumulated history preserved but not updated

---

## 12. Implementation Phases

### Phase 1: Core Infrastructure
- [ ] Create `temporal_silo.hrl` with record definitions
- [ ] Implement `temporal_silo_sensors.erl` with all 12 sensors
- [ ] Implement `temporal_silo_actuators.erl` with all 10 actuators
- [ ] Basic `temporal_silo.erl` gen_server

### Phase 2: Time Management
- [ ] Episode length management system
- [ ] Evaluation timeout enforcement
- [ ] Reaction time tracking and enforcement
- [ ] Early termination logic

### Phase 3: Learning Integration
- [ ] TWEANN controller integration
- [ ] L2 guidance support
- [ ] Convergence detection algorithms
- [ ] Patience mechanism

### Phase 4: Cross-Silo Integration
- [ ] Outgoing signal implementations
- [ ] Incoming signal handlers
- [ ] Event emission to event store
- [ ] Integration with task_silo, economic_silo

### Phase 5: Testing & Tuning
- [ ] Unit tests for all sensors
- [ ] Unit tests for all actuators
- [ ] Integration tests with neuroevolution engine
- [ ] Performance benchmarks
- [ ] Tune default parameters

---

## 13. Success Criteria

1. **Training Efficiency**: 2-5x reduction in wasted evaluation time
2. **Real-time Compliance**: 100% enforcement of reaction time budgets when enabled
3. **Early Termination Accuracy**: < 5% false terminations (individuals that would have improved)
4. **Convergence Detection**: Detect convergence within 5 generations of stabilization
5. **Adaptive Episodes**: Episode length correlates with task complexity
6. **Cross-Silo Integration**: All signals documented and functional
7. **Event Coverage**: All 10 event types emitted with proper payloads

---

## Appendix A: Algorithm Details

### A.1 Convergence Detection

```erlang
%% Convergence detected when fitness variance drops below threshold
detect_convergence(FitnessHistory, Window, Threshold) ->
    case length(FitnessHistory) >= Window of
        false -> {not_converged, undefined};
        true ->
            Recent = lists:sublist(FitnessHistory, Window),
            Variance = variance(Recent),
            case Variance < Threshold of
                true ->
                    Mean = lists:sum(Recent) / Window,
                    {converged, Mean};
                false ->
                    {not_converged, Variance}
            end
    end.
```

### A.2 Early Termination Decision

```erlang
%% Decide whether to terminate based on fitness trajectory
should_terminate(FitnessTrajectory, MinSteps, FitnessThreshold, TrendThreshold) ->
    case length(FitnessTrajectory) < MinSteps of
        true -> {continue, insufficient_data};
        false ->
            [Current | _] = FitnessTrajectory,
            Trend = calculate_trend(FitnessTrajectory),
            case Current < FitnessThreshold andalso Trend < TrendThreshold of
                true -> {terminate, {low_fitness, Current, Trend}};
                false -> {continue, {fitness, Current, Trend}}
            end
    end.
```

### A.3 Episode Length Adaptation

```erlang
%% Adapt episode length based on fitness gain and time spent
adapt_episode_length(CurrentLength, FitnessGain, TimeSpent, Config) ->
    #temporal_config{
        min_episode_length = MinLen,
        max_episode_length = MaxLen
    } = Config,

    Efficiency = FitnessGain / max(1, TimeSpent),

    %% If efficiency is high, try shorter episodes
    %% If efficiency is low, try longer episodes
    Adjustment = case Efficiency of
        E when E > 0.01 -> -0.1;  % Good efficiency, try shorter
        E when E < 0.001 -> 0.2;  % Poor efficiency, try longer
        _ -> 0.0                   % Acceptable, no change
    end,

    NewLength = round(CurrentLength * (1.0 + Adjustment)),
    clamp(NewLength, MinLen, MaxLen).
```

---

## Appendix B: Configuration Examples

### B.1 Real-time Application Config

```erlang
#temporal_config{
    enabled = true,
    min_episode_length = 100,
    max_episode_length = 1000,
    min_timeout_ms = 500,
    max_timeout_ms = 5000,
    enable_early_termination = true,
    min_steps_before_termination = 50,
    enforce_reaction_time = true,  % Critical for real-time
    max_history_size = 50,
    emit_events = true
}.
```

### B.2 Exploratory Research Config

```erlang
#temporal_config{
    enabled = true,
    min_episode_length = 1000,
    max_episode_length = 100000,
    min_timeout_ms = 10000,
    max_timeout_ms = 600000,  % 10 minutes
    enable_early_termination = false,  % Let everything run
    enforce_reaction_time = false,
    max_history_size = 1000,
    emit_events = true
}.
```

### B.3 Cost-Optimized Config

```erlang
#temporal_config{
    enabled = true,
    min_episode_length = 50,
    max_episode_length = 500,
    min_timeout_ms = 100,
    max_timeout_ms = 2000,
    enable_early_termination = true,
    min_steps_before_termination = 10,  % Aggressive early termination
    enforce_reaction_time = true,
    max_history_size = 100,
    emit_events = true,
    l2_query_interval = 3  % Frequent L2 guidance
}.
```
