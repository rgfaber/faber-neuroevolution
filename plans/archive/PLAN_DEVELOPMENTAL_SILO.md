# Plan: Developmental Silo for Liquid Conglomerate

**Status:** Planning
**Created:** 2025-12-23
**Last Updated:** 2025-12-23
**Related:** PLAN_SOCIAL_SILO.md, PLAN_CULTURAL_SILO.md, PLAN_ECOLOGICAL_SILO.md, PLAN_MORPHOLOGICAL_SILO.md, PLAN_TEMPORAL_SILO.md, PLAN_COMPETITIVE_SILO.md

---

## Overview

The Developmental Silo manages ontogeny - how individuals develop over their lifetime. Instead of being born fully formed, networks can grow, have critical periods, and mature through stages. This mirrors biological development where early plasticity enables learning and later stability enables efficiency.

---

## 1. Motivation

### Problem Statement

Traditional neuroevolution treats networks as static entities:
- **No lifetime learning**: Networks are fixed after evolution
- **No critical periods**: All learning is equivalent regardless of "age"
- **No maturation**: No transition from plastic learner to stable expert
- **No metamorphosis**: No capability for radical structural change during lifetime

This misses biological insights about development:
- Young organisms are more plastic, learning rapidly
- Critical periods enable specific learning at optimal times
- Maturation stabilizes learned behaviors
- Metamorphosis enables adaptation to changing life stages

### Business Value

| Benefit | Impact |
|---------|--------|
| Deployment adaptation | Networks fine-tune to specific environments |
| Reduced training | Some learning happens "in the field" |
| Developmental testing | Verify robustness across development |
| Age-appropriate behavior | Different behaviors at different life stages |
| Transfer learning | Critical periods enable domain adaptation |

### Training Velocity Impact

| Metric | Without Developmental Silo | With Developmental Silo |
|--------|---------------------------|------------------------|
| Deployment adaptation | None | Continuous |
| Learning efficiency | Fixed | Stage-appropriate |
| Robustness testing | Manual | Automated across stages |
| Behavioral diversity | Low | High (stage-dependent) |

---

## 2. Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                      DEVELOPMENTAL SILO                              │
├─────────────────────────────────────────────────────────────────────┤
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │                      L0 SENSORS (10)                         │    │
│  │                                                              │    │
│  │  Stage            Plasticity    Critical      Canalization  │    │
│  │  ┌─────────┐     ┌─────────┐   ┌─────────┐  ┌─────────┐    │    │
│  │  │develop_ │     │plasticity│  │critical_│  │canalization│  │    │
│  │  │stage    │     │_level    │  │period_  │  │_strength   │  │    │
│  │  │maturation│    │plasticity│  │active   │  │heterochrony│  │    │
│  │  │_level   │     │_trend    │  └─────────┘  │_index      │  │    │
│  │  └─────────┘     └─────────┘               └─────────┘    │    │
│  │                                                              │    │
│  │  Metamorphosis    Noise         Fitness                     │    │
│  │  ┌─────────┐     ┌─────────┐   ┌─────────┐                 │    │
│  │  │metamorph│     │develop_ │   │develop_ │                 │    │
│  │  │_proximity│    │noise    │   │fitness  │                 │    │
│  │  └─────────┘     └─────────┘   └─────────┘                 │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                                │                                     │
│                       ┌────────▼────────┐                           │
│                       │ TWEANN Controller│                          │
│                       │   (online ES)   │                           │
│                       └────────┬────────┘                           │
│                                │                                     │
│  ┌─────────────────────────────▼───────────────────────────────┐    │
│  │                      L0 ACTUATORS (8)                        │    │
│  │                                                              │    │
│  │  Growth           Critical       Plasticity   Metamorphosis │    │
│  │  ┌─────────┐     ┌─────────┐    ┌─────────┐  ┌─────────┐   │    │
│  │  │growth_  │     │critical_│    │plasticity│ │metamorph_│   │    │
│  │  │rate     │     │period_  │    │_decay    │ │trigger   │   │    │
│  │  │maturation│    │duration │    │initial_  │ │metamorph_│   │    │
│  │  │_speed   │     └─────────┘    │plasticity│ │severity  │   │    │
│  │  └─────────┘                    └─────────┘  └─────────┘   │    │
│  └─────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 3. L0 Sensors

### 3.1 Sensor Specifications

| ID | Name | Range | Description |
|----|------|-------|-------------|
| 1 | `developmental_stage` | [0.0, 1.0] | Progress through development (0=birth, 1=mature) |
| 2 | `maturation_level` | [0.0, 1.0] | Physiological maturity |
| 3 | `plasticity_level` | [0.0, 1.0] | Current learning capacity |
| 4 | `plasticity_trend` | [-1.0, 1.0] | Direction of plasticity change |
| 5 | `critical_period_active` | [0.0, 1.0] | Is a critical period open? |
| 6 | `canalization_strength` | [0.0, 1.0] | Resistance to perturbation |
| 7 | `heterochrony_index` | [0.0, 1.0] | Timing variation from population mean |
| 8 | `metamorphosis_proximity` | [0.0, 1.0] | Distance to metamorphosis threshold |
| 9 | `developmental_noise` | [0.0, 1.0] | Random variation in development |
| 10 | `developmental_fitness` | [0.0, 1.0] | Fitness relative to developmental stage |

### 3.2 Sensor Module

```erlang
%%%-------------------------------------------------------------------
%%% @doc Developmental Silo L0 Sensors
%%% Monitors ontogenic dynamics: developmental stages, plasticity,
%%% critical periods, and maturation.
%%% @end
%%%-------------------------------------------------------------------
-module(developmental_silo_sensors).

-behaviour(l0_sensor_behaviour).

%% API
-export([sensor_specs/0,
         collect_sensors/1,
         sensor_count/0]).

%% Sensor collection
-export([collect_developmental_stage/1,
         collect_maturation_level/1,
         collect_plasticity_level/1,
         collect_plasticity_trend/1,
         collect_critical_period_active/1,
         collect_canalization_strength/1,
         collect_heterochrony_index/1,
         collect_metamorphosis_proximity/1,
         collect_developmental_noise/1,
         collect_developmental_fitness/1]).

-include("developmental_silo.hrl").

%%====================================================================
%% Behaviour Callbacks
%%====================================================================

-spec sensor_specs() -> [l0_sensor_spec()].
sensor_specs() ->
    [
        #{id => developmental_stage,
          name => <<"Developmental Stage">>,
          range => {0.0, 1.0},
          description => <<"Progress through development (0=birth, 1=mature)">>},

        #{id => maturation_level,
          name => <<"Maturation Level">>,
          range => {0.0, 1.0},
          description => <<"Physiological maturity">>},

        #{id => plasticity_level,
          name => <<"Plasticity Level">>,
          range => {0.0, 1.0},
          description => <<"Current learning capacity">>},

        #{id => plasticity_trend,
          name => <<"Plasticity Trend">>,
          range => {-1.0, 1.0},
          description => <<"Direction of plasticity change">>},

        #{id => critical_period_active,
          name => <<"Critical Period Active">>,
          range => {0.0, 1.0},
          description => <<"Is a critical period open?">>},

        #{id => canalization_strength,
          name => <<"Canalization Strength">>,
          range => {0.0, 1.0},
          description => <<"Resistance to perturbation">>},

        #{id => heterochrony_index,
          name => <<"Heterochrony Index">>,
          range => {0.0, 1.0},
          description => <<"Timing variation from population mean">>},

        #{id => metamorphosis_proximity,
          name => <<"Metamorphosis Proximity">>,
          range => {0.0, 1.0},
          description => <<"Distance to metamorphosis threshold">>},

        #{id => developmental_noise,
          name => <<"Developmental Noise">>,
          range => {0.0, 1.0},
          description => <<"Random variation in development">>},

        #{id => developmental_fitness,
          name => <<"Developmental Fitness">>,
          range => {0.0, 1.0},
          description => <<"Fitness relative to developmental stage">>}
    ].

-spec sensor_count() -> pos_integer().
sensor_count() -> 10.

-spec collect_sensors(developmental_context()) -> sensor_vector().
collect_sensors(Context) ->
    [
        collect_developmental_stage(Context),
        collect_maturation_level(Context),
        collect_plasticity_level(Context),
        collect_plasticity_trend(Context),
        collect_critical_period_active(Context),
        collect_canalization_strength(Context),
        collect_heterochrony_index(Context),
        collect_metamorphosis_proximity(Context),
        collect_developmental_noise(Context),
        collect_developmental_fitness(Context)
    ].

%%====================================================================
%% Individual Sensor Collection
%%====================================================================

%% @doc Progress through developmental stages
-spec collect_developmental_stage(developmental_context()) -> float().
collect_developmental_stage(#developmental_context{
    current_age = Age,
    max_developmental_age = MaxAge
}) ->
    clamp(Age / MaxAge, 0.0, 1.0).

%% @doc Physiological maturity level
-spec collect_maturation_level(developmental_context()) -> float().
collect_maturation_level(#developmental_context{
    maturation_progress = Progress
}) ->
    clamp(Progress, 0.0, 1.0).

%% @doc Current learning capacity
-spec collect_plasticity_level(developmental_context()) -> float().
collect_plasticity_level(#developmental_context{
    current_plasticity = Plasticity
}) ->
    clamp(Plasticity, 0.0, 1.0).

%% @doc Direction of plasticity change
-spec collect_plasticity_trend(developmental_context()) -> float().
collect_plasticity_trend(#developmental_context{
    plasticity_history = History
}) ->
    case length(History) >= 3 of
        false -> 0.0;
        true ->
            Recent = lists:sublist(History, 5),
            calculate_trend(Recent)
    end.

%% @doc Whether a critical period is currently active
-spec collect_critical_period_active(developmental_context()) -> float().
collect_critical_period_active(#developmental_context{
    active_critical_periods = Periods
}) ->
    case Periods of
        [] -> 0.0;
        _ ->
            %% Return intensity of most active period
            MaxIntensity = lists:max([P#critical_period.intensity || P <- Periods]),
            clamp(MaxIntensity, 0.0, 1.0)
    end.

%% @doc Resistance to perturbation (canalization)
-spec collect_canalization_strength(developmental_context()) -> float().
collect_canalization_strength(#developmental_context{
    canalization_level = Level
}) ->
    clamp(Level, 0.0, 1.0).

%% @doc Timing variation from population mean
-spec collect_heterochrony_index(developmental_context()) -> float().
collect_heterochrony_index(#developmental_context{
    current_age = Age,
    population_mean_age = MeanAge,
    population_std_age = StdAge
}) ->
    case StdAge of
        0.0 -> 0.5;
        _ ->
            ZScore = abs(Age - MeanAge) / StdAge,
            %% Convert to 0-1 where 0.5 = average, 1.0 = 2+ std devs
            clamp(ZScore / 4.0 + 0.5, 0.0, 1.0)
    end.

%% @doc Distance to metamorphosis threshold
-spec collect_metamorphosis_proximity(developmental_context()) -> float().
collect_metamorphosis_proximity(#developmental_context{
    metamorphosis_threshold = Threshold,
    maturation_progress = Progress
}) ->
    case Threshold of
        undefined -> 0.0;
        _ ->
            %% How close to triggering metamorphosis
            Proximity = Progress / Threshold,
            clamp(Proximity, 0.0, 1.0)
    end.

%% @doc Stochastic variation in development
-spec collect_developmental_noise(developmental_context()) -> float().
collect_developmental_noise(#developmental_context{
    developmental_variance = Variance
}) ->
    %% Normalize variance to 0-1
    clamp(Variance * 10.0, 0.0, 1.0).

%% @doc Fitness relative to developmental stage
-spec collect_developmental_fitness(developmental_context()) -> float().
collect_developmental_fitness(#developmental_context{
    current_fitness = Fitness,
    expected_fitness_for_stage = ExpectedFitness
}) ->
    case ExpectedFitness of
        0.0 -> 0.5;
        _ ->
            Ratio = Fitness / ExpectedFitness,
            %% 0.5 = as expected, >0.5 = ahead, <0.5 = behind
            clamp(Ratio / 2.0, 0.0, 1.0)
    end.

%%====================================================================
%% Internal Functions
%%====================================================================

calculate_trend(Values) ->
    N = length(Values),
    case N >= 2 of
        false -> 0.0;
        true ->
            Indices = lists:seq(1, N),
            MeanX = (N + 1) / 2,
            MeanY = lists:sum(Values) / N,
            Numerator = lists:sum([((I - MeanX) * (V - MeanY))
                                   || {I, V} <- lists:zip(Indices, Values)]),
            Denominator = lists:sum([math:pow(I - MeanX, 2) || I <- Indices]),
            case Denominator of
                0.0 -> 0.0;
                _ -> clamp(Numerator / Denominator, -1.0, 1.0)
            end
    end.

clamp(Value, Min, Max) ->
    max(Min, min(Max, Value)).
```

---

## 4. L0 Actuators

### 4.1 Actuator Specifications

| ID | Name | Range | Default | Description |
|----|------|-------|---------|-------------|
| 1 | `growth_rate` | [0.0, 0.2] | 0.05 | Speed of structural growth |
| 2 | `maturation_speed` | [0.0, 0.1] | 0.02 | Speed of physiological maturation |
| 3 | `critical_period_duration` | [1, 20] | 5 | Generations critical period lasts |
| 4 | `plasticity_decay_rate` | [0.0, 0.2] | 0.05 | How fast plasticity decreases |
| 5 | `initial_plasticity` | [0.5, 1.0] | 0.9 | Plasticity at birth |
| 6 | `developmental_noise_level` | [0.0, 0.3] | 0.1 | Stochasticity in development |
| 7 | `metamorphosis_trigger` | [0.5, 0.95] | 0.8 | Threshold for metamorphosis |
| 8 | `metamorphosis_severity` | [0.0, 1.0] | 0.5 | How drastic metamorphosis is |

### 4.2 Actuator Module

```erlang
%%%-------------------------------------------------------------------
%%% @doc Developmental Silo L0 Actuators
%%% Controls ontogenic parameters: growth rates, plasticity decay,
%%% critical periods, and metamorphosis triggers.
%%% @end
%%%-------------------------------------------------------------------
-module(developmental_silo_actuators).

-behaviour(l0_actuator_behaviour).

%% API
-export([actuator_specs/0,
         apply_actuators/2,
         actuator_count/0]).

%% Individual actuator application
-export([apply_growth_rate/2,
         apply_maturation_speed/2,
         apply_critical_period_duration/2,
         apply_plasticity_decay_rate/2,
         apply_initial_plasticity/2,
         apply_developmental_noise_level/2,
         apply_metamorphosis_trigger/2,
         apply_metamorphosis_severity/2]).

-include("developmental_silo.hrl").

%%====================================================================
%% Behaviour Callbacks
%%====================================================================

-spec actuator_specs() -> [l0_actuator_spec()].
actuator_specs() ->
    [
        #{id => growth_rate,
          name => <<"Growth Rate">>,
          range => {0.0, 0.2},
          default => 0.05,
          description => <<"Speed of structural growth">>},

        #{id => maturation_speed,
          name => <<"Maturation Speed">>,
          range => {0.0, 0.1},
          default => 0.02,
          description => <<"Speed of physiological maturation">>},

        #{id => critical_period_duration,
          name => <<"Critical Period Duration">>,
          range => {1, 20},
          default => 5,
          description => <<"Generations critical period lasts">>},

        #{id => plasticity_decay_rate,
          name => <<"Plasticity Decay Rate">>,
          range => {0.0, 0.2},
          default => 0.05,
          description => <<"How fast plasticity decreases">>},

        #{id => initial_plasticity,
          name => <<"Initial Plasticity">>,
          range => {0.5, 1.0},
          default => 0.9,
          description => <<"Plasticity at birth">>},

        #{id => developmental_noise_level,
          name => <<"Developmental Noise">>,
          range => {0.0, 0.3},
          default => 0.1,
          description => <<"Stochasticity in development">>},

        #{id => metamorphosis_trigger,
          name => <<"Metamorphosis Trigger">>,
          range => {0.5, 0.95},
          default => 0.8,
          description => <<"Threshold for metamorphosis">>},

        #{id => metamorphosis_severity,
          name => <<"Metamorphosis Severity">>,
          range => {0.0, 1.0},
          default => 0.5,
          description => <<"How drastic metamorphosis is">>}
    ].

-spec actuator_count() -> pos_integer().
actuator_count() -> 8.

-spec apply_actuators(actuator_vector(), developmental_state()) -> developmental_state().
apply_actuators(Outputs, State) when length(Outputs) =:= 8 ->
    [GrowthRate, MatSpeed, CritDur, PlastDecay,
     InitPlast, DevNoise, MetaTrig, MetaSev] = Outputs,

    State1 = apply_growth_rate(GrowthRate, State),
    State2 = apply_maturation_speed(MatSpeed, State1),
    State3 = apply_critical_period_duration(CritDur, State2),
    State4 = apply_plasticity_decay_rate(PlastDecay, State3),
    State5 = apply_initial_plasticity(InitPlast, State4),
    State6 = apply_developmental_noise_level(DevNoise, State5),
    State7 = apply_metamorphosis_trigger(MetaTrig, State6),
    apply_metamorphosis_severity(MetaSev, State7).

%%====================================================================
%% Individual Actuator Application
%%====================================================================

%% @doc Apply structural growth rate
-spec apply_growth_rate(float(), developmental_state()) -> developmental_state().
apply_growth_rate(Output, State) ->
    %% Output in [0,1] -> Rate in [0.0, 0.2]
    Rate = Output * 0.2,
    State#developmental_state{growth_rate = Rate}.

%% @doc Apply maturation speed
-spec apply_maturation_speed(float(), developmental_state()) -> developmental_state().
apply_maturation_speed(Output, State) ->
    %% Output in [0,1] -> Speed in [0.0, 0.1]
    Speed = Output * 0.1,
    State#developmental_state{maturation_speed = Speed}.

%% @doc Apply critical period duration
-spec apply_critical_period_duration(float(), developmental_state()) -> developmental_state().
apply_critical_period_duration(Output, State) ->
    %% Output in [0,1] -> Duration in [1, 20]
    Duration = round(1 + Output * 19),
    State#developmental_state{critical_period_duration = Duration}.

%% @doc Apply plasticity decay rate
-spec apply_plasticity_decay_rate(float(), developmental_state()) -> developmental_state().
apply_plasticity_decay_rate(Output, State) ->
    %% Output in [0,1] -> Decay in [0.0, 0.2]
    Decay = Output * 0.2,
    State#developmental_state{plasticity_decay_rate = Decay}.

%% @doc Apply initial plasticity
-spec apply_initial_plasticity(float(), developmental_state()) -> developmental_state().
apply_initial_plasticity(Output, State) ->
    %% Output in [0,1] -> Plasticity in [0.5, 1.0]
    Plasticity = 0.5 + Output * 0.5,
    State#developmental_state{initial_plasticity = Plasticity}.

%% @doc Apply developmental noise level
-spec apply_developmental_noise_level(float(), developmental_state()) -> developmental_state().
apply_developmental_noise_level(Output, State) ->
    %% Output in [0,1] -> Noise in [0.0, 0.3]
    Noise = Output * 0.3,
    State#developmental_state{developmental_noise_level = Noise}.

%% @doc Apply metamorphosis trigger threshold
-spec apply_metamorphosis_trigger(float(), developmental_state()) -> developmental_state().
apply_metamorphosis_trigger(Output, State) ->
    %% Output in [0,1] -> Trigger in [0.5, 0.95]
    Trigger = 0.5 + Output * 0.45,
    State#developmental_state{metamorphosis_trigger = Trigger}.

%% @doc Apply metamorphosis severity
-spec apply_metamorphosis_severity(float(), developmental_state()) -> developmental_state().
apply_metamorphosis_severity(Output, State) ->
    %% Output in [0,1] -> Severity in [0.0, 1.0]
    State#developmental_state{metamorphosis_severity = Output}.
```

---

## 5. Record Definitions

```erlang
%%%-------------------------------------------------------------------
%%% @doc Developmental Silo Header
%%% Record definitions for ontogenic dynamics management.
%%% @end
%%%-------------------------------------------------------------------

-ifndef(DEVELOPMENTAL_SILO_HRL).
-define(DEVELOPMENTAL_SILO_HRL, true).

%%====================================================================
%% Types
%%====================================================================

-type sensor_vector() :: [float()].
-type actuator_vector() :: [float()].
-type generation() :: non_neg_integer().
-type age() :: non_neg_integer().
-type plasticity() :: float().

%%====================================================================
%% Critical Period Record
%%====================================================================

-record(critical_period, {
    %% Identity
    period_type :: atom(),  % sensory, motor, language, etc.
    target_capability :: atom(),

    %% Timing
    start_generation :: generation(),
    end_generation :: generation() | undefined,
    duration :: pos_integer(),

    %% Intensity
    intensity :: float(),  % [0.0, 1.0]
    decay_rate :: float(),

    %% Learning during period
    learning_acquired = [] :: [atom()],
    learning_rate_multiplier :: float()
}).

-type critical_period() :: #critical_period{}.

%%====================================================================
%% Developmental Stage Record
%%====================================================================

-record(developmental_stage, {
    %% Identity
    stage_name :: atom(),  % embryonic, juvenile, adolescent, adult, senescent
    stage_index :: non_neg_integer(),

    %% Characteristics
    plasticity_range :: {float(), float()},
    growth_rate_range :: {float(), float()},
    typical_duration :: pos_integer(),

    %% Transitions
    entry_condition :: fun((developmental_context()) -> boolean()),
    exit_condition :: fun((developmental_context()) -> boolean()),

    %% Stage-specific behaviors
    available_critical_periods :: [atom()],
    mutation_rate_modifier :: float()
}).

-type developmental_stage() :: #developmental_stage{}.

%%====================================================================
%% Context Record (Input to Sensors)
%%====================================================================

-record(developmental_context, {
    %% Age and timing
    current_age = 0 :: age(),
    max_developmental_age = 100 :: age(),

    %% Maturation
    maturation_progress = 0.0 :: float(),

    %% Plasticity
    current_plasticity = 1.0 :: plasticity(),
    plasticity_history = [] :: [plasticity()],

    %% Critical periods
    active_critical_periods = [] :: [critical_period()],
    completed_critical_periods = [] :: [critical_period()],

    %% Canalization
    canalization_level = 0.0 :: float(),

    %% Population comparison
    population_mean_age = 50 :: float(),
    population_std_age = 20 :: float(),

    %% Metamorphosis
    metamorphosis_threshold :: float() | undefined,
    metamorphosis_history = [] :: [{generation(), atom()}],

    %% Variance
    developmental_variance = 0.0 :: float(),

    %% Fitness
    current_fitness = 0.0 :: float(),
    expected_fitness_for_stage = 0.5 :: float()
}).

-type developmental_context() :: #developmental_context{}.

%%====================================================================
%% State Record (Silo Internal State)
%%====================================================================

-record(developmental_state, {
    %% Configuration
    config :: developmental_config(),

    %% Current actuator outputs
    growth_rate = 0.05 :: float(),
    maturation_speed = 0.02 :: float(),
    critical_period_duration = 5 :: pos_integer(),
    plasticity_decay_rate = 0.05 :: float(),
    initial_plasticity = 0.9 :: float(),
    developmental_noise_level = 0.1 :: float(),
    metamorphosis_trigger = 0.8 :: float(),
    metamorphosis_severity = 0.5 :: float(),

    %% Stage definitions
    stages = [] :: [developmental_stage()],
    current_stage_index = 0 :: non_neg_integer(),

    %% Population developmental state
    individual_states = #{} :: #{binary() => individual_dev_state()},

    %% Tracking
    current_generation = 0 :: generation(),
    total_metamorphoses = 0 :: non_neg_integer(),
    critical_periods_opened = 0 :: non_neg_integer(),
    critical_periods_closed = 0 :: non_neg_integer(),

    %% L2 integration
    l2_enabled = false :: boolean(),
    l2_guidance = undefined :: l2_guidance() | undefined
}).

-type developmental_state() :: #developmental_state{}.

%%====================================================================
%% Individual Developmental State
%%====================================================================

-record(individual_dev_state, {
    %% Identity
    individual_id :: binary(),

    %% Age
    birth_generation :: generation(),
    current_age = 0 :: age(),

    %% Stage
    current_stage :: atom(),
    stage_entry_generation :: generation(),

    %% Plasticity
    plasticity = 1.0 :: plasticity(),
    plasticity_history = [] :: [plasticity()],

    %% Critical periods
    active_periods = [] :: [critical_period()],
    period_history = [] :: [{atom(), generation(), generation()}],

    %% Metamorphosis
    metamorphosis_count = 0 :: non_neg_integer(),
    last_metamorphosis :: generation() | undefined,

    %% Canalization
    canalization = 0.0 :: float(),
    stable_traits = [] :: [atom()],

    %% Milestones
    milestones_achieved = [] :: [{atom(), generation()}]
}).

-type individual_dev_state() :: #individual_dev_state{}.

%%====================================================================
%% Configuration Record
%%====================================================================

-record(developmental_config, {
    %% Enable/disable
    enabled = true :: boolean(),

    %% Stage configuration
    num_stages = 5 :: pos_integer(),
    stage_definitions :: [developmental_stage()] | undefined,

    %% Plasticity
    min_plasticity = 0.1 :: float(),
    max_plasticity = 1.0 :: float(),

    %% Critical periods
    enable_critical_periods = true :: boolean(),
    available_period_types = [sensory, motor, cognitive] :: [atom()],

    %% Metamorphosis
    enable_metamorphosis = true :: boolean(),
    max_metamorphoses = 3 :: pos_integer(),

    %% Event emission
    emit_events = true :: boolean()
}).

-type developmental_config() :: #developmental_config{}.

%%====================================================================
%% L2 Guidance Record
%%====================================================================

-record(l2_guidance, {
    %% Development speed factor
    development_speed = 1.0 :: float(),

    %% Plasticity guidance
    plasticity_pressure = 0.5 :: float(),

    %% Critical period guidance
    critical_period_incentive = 0.5 :: float(),

    %% Metamorphosis guidance
    metamorphosis_incentive = 0.5 :: float()
}).

-type l2_guidance() :: #l2_guidance{}.

%%====================================================================
%% Constants
%%====================================================================

-define(DEFAULT_STAGES, [embryonic, juvenile, adolescent, adult, senescent]).
-define(DEFAULT_PLASTICITY, 1.0).
-define(MIN_PLASTICITY, 0.1).
-define(MAX_AGE, 100).

-endif.
```

---

## 6. Core Silo Implementation

```erlang
%%%-------------------------------------------------------------------
%%% @doc Developmental Silo
%%% Manages ontogenic dynamics for neuroevolution: developmental stages,
%%% plasticity, critical periods, and metamorphosis.
%%% @end
%%%-------------------------------------------------------------------
-module(developmental_silo).

-behaviour(gen_server).

%% API
-export([start_link/1,
         get_developmental_params/1,
         update_context/2,
         register_individual/2,
         advance_age/2,
         get_plasticity/2,
         open_critical_period/3,
         close_critical_period/3,
         trigger_metamorphosis/2,
         get_individual_state/2,
         get_state/1,
         enable/1,
         disable/1,
         is_enabled/1]).

%% Cross-silo signals
-export([signal_maturity_distribution/1,
         signal_plasticity_available/1,
         signal_metamorphosis_rate/1,
         receive_stress_level/2,
         receive_learning_opportunity/2,
         receive_resource_scarcity/2]).

%% gen_server callbacks
-export([init/1, handle_call/3, handle_cast/2, handle_info/2,
         terminate/2, code_change/3]).

-include("developmental_silo.hrl").

%%====================================================================
%% API
%%====================================================================

-spec start_link(developmental_config()) -> {ok, pid()} | {error, term()}.
start_link(Config) ->
    gen_server:start_link({local, ?MODULE}, ?MODULE, Config, []).

-spec get_developmental_params(pid()) -> map().
get_developmental_params(Pid) ->
    gen_server:call(Pid, get_developmental_params).

-spec update_context(pid(), developmental_context()) -> ok.
update_context(Pid, Context) ->
    gen_server:cast(Pid, {update_context, Context}).

-spec register_individual(pid(), binary()) -> ok.
register_individual(Pid, IndividualId) ->
    gen_server:cast(Pid, {register_individual, IndividualId}).

-spec advance_age(pid(), binary()) -> {ok, age()} | {error, term()}.
advance_age(Pid, IndividualId) ->
    gen_server:call(Pid, {advance_age, IndividualId}).

-spec get_plasticity(pid(), binary()) -> plasticity().
get_plasticity(Pid, IndividualId) ->
    gen_server:call(Pid, {get_plasticity, IndividualId}).

-spec open_critical_period(pid(), binary(), atom()) -> ok | {error, term()}.
open_critical_period(Pid, IndividualId, PeriodType) ->
    gen_server:call(Pid, {open_critical_period, IndividualId, PeriodType}).

-spec close_critical_period(pid(), binary(), atom()) -> ok | {error, term()}.
close_critical_period(Pid, IndividualId, PeriodType) ->
    gen_server:call(Pid, {close_critical_period, IndividualId, PeriodType}).

-spec trigger_metamorphosis(pid(), binary()) -> ok | {error, term()}.
trigger_metamorphosis(Pid, IndividualId) ->
    gen_server:call(Pid, {trigger_metamorphosis, IndividualId}).

-spec get_individual_state(pid(), binary()) -> individual_dev_state() | undefined.
get_individual_state(Pid, IndividualId) ->
    gen_server:call(Pid, {get_individual_state, IndividualId}).

-spec get_state(pid()) -> developmental_state().
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

%% @doc Get maturity distribution for other silos
-spec signal_maturity_distribution(pid()) -> float().
signal_maturity_distribution(Pid) ->
    gen_server:call(Pid, signal_maturity_distribution).

%% @doc Get available plasticity for cultural silo
-spec signal_plasticity_available(pid()) -> float().
signal_plasticity_available(Pid) ->
    gen_server:call(Pid, signal_plasticity_available).

%% @doc Get metamorphosis rate for ecological silo
-spec signal_metamorphosis_rate(pid()) -> float().
signal_metamorphosis_rate(Pid) ->
    gen_server:call(Pid, signal_metamorphosis_rate).

%% @doc Receive stress level from ecological silo
-spec receive_stress_level(pid(), float()) -> ok.
receive_stress_level(Pid, Stress) ->
    gen_server:cast(Pid, {cross_silo, stress_level, Stress}).

%% @doc Receive learning opportunity from cultural silo
-spec receive_learning_opportunity(pid(), float()) -> ok.
receive_learning_opportunity(Pid, Opportunity) ->
    gen_server:cast(Pid, {cross_silo, learning_opportunity, Opportunity}).

%% @doc Receive resource scarcity from resource silo
-spec receive_resource_scarcity(pid(), float()) -> ok.
receive_resource_scarcity(Pid, Scarcity) ->
    gen_server:cast(Pid, {cross_silo, resource_scarcity, Scarcity}).

%%====================================================================
%% gen_server Callbacks
%%====================================================================

init(Config) ->
    Stages = initialize_stages(Config),
    State = #developmental_state{
        config = Config,
        stages = Stages
    },
    {ok, State}.

handle_call(get_developmental_params, _From, State) ->
    Params = #{
        growth_rate => State#developmental_state.growth_rate,
        maturation_speed => State#developmental_state.maturation_speed,
        critical_period_duration => State#developmental_state.critical_period_duration,
        plasticity_decay_rate => State#developmental_state.plasticity_decay_rate,
        initial_plasticity => State#developmental_state.initial_plasticity,
        developmental_noise_level => State#developmental_state.developmental_noise_level,
        metamorphosis_trigger => State#developmental_state.metamorphosis_trigger,
        metamorphosis_severity => State#developmental_state.metamorphosis_severity
    },
    {reply, Params, State};

handle_call({advance_age, IndividualId}, _From, State) ->
    case maps:get(IndividualId, State#developmental_state.individual_states, undefined) of
        undefined ->
            {reply, {error, not_found}, State};
        IndState ->
            %% Advance age
            NewAge = IndState#individual_dev_state.current_age + 1,

            %% Decay plasticity
            DecayRate = State#developmental_state.plasticity_decay_rate,
            MinPlasticity = State#developmental_state.config#developmental_config.min_plasticity,
            NewPlasticity = max(MinPlasticity,
                               IndState#individual_dev_state.plasticity * (1.0 - DecayRate)),

            %% Check stage transition
            NewStage = check_stage_transition(IndState, NewAge, State),

            %% Update individual state
            NewIndState = IndState#individual_dev_state{
                current_age = NewAge,
                plasticity = NewPlasticity,
                plasticity_history = [NewPlasticity |
                    lists:sublist(IndState#individual_dev_state.plasticity_history, 49)],
                current_stage = NewStage
            },

            %% Check for automatic critical periods
            NewIndState2 = check_automatic_critical_periods(NewIndState, State),

            %% Check metamorphosis
            NewIndState3 = check_metamorphosis(NewIndState2, State),

            %% Update state
            NewStates = maps:put(IndividualId, NewIndState3,
                                State#developmental_state.individual_states),
            NewState = State#developmental_state{individual_states = NewStates},

            %% Emit events if stage changed
            case NewStage =/= IndState#individual_dev_state.current_stage of
                true ->
                    maybe_emit_event(development_stage_changed, #{
                        individual_id => IndividualId,
                        old_stage => IndState#individual_dev_state.current_stage,
                        new_stage => NewStage
                    }, State);
                false ->
                    ok
            end,

            {reply, {ok, NewAge}, NewState}
    end;

handle_call({get_plasticity, IndividualId}, _From, State) ->
    case maps:get(IndividualId, State#developmental_state.individual_states, undefined) of
        undefined ->
            {reply, State#developmental_state.initial_plasticity, State};
        IndState ->
            {reply, IndState#individual_dev_state.plasticity, State}
    end;

handle_call({open_critical_period, IndividualId, PeriodType}, _From, State) ->
    case maps:get(IndividualId, State#developmental_state.individual_states, undefined) of
        undefined ->
            {reply, {error, not_found}, State};
        IndState ->
            %% Create new critical period
            Period = #critical_period{
                period_type = PeriodType,
                target_capability = PeriodType,
                start_generation = State#developmental_state.current_generation,
                duration = State#developmental_state.critical_period_duration,
                intensity = 1.0,
                decay_rate = 0.1,
                learning_rate_multiplier = 2.0
            },

            %% Add to active periods
            NewPeriods = [Period | IndState#individual_dev_state.active_periods],
            NewIndState = IndState#individual_dev_state{active_periods = NewPeriods},

            NewStates = maps:put(IndividualId, NewIndState,
                                State#developmental_state.individual_states),
            NewState = State#developmental_state{
                individual_states = NewStates,
                critical_periods_opened = State#developmental_state.critical_periods_opened + 1
            },

            maybe_emit_event(critical_period_opened, #{
                individual_id => IndividualId,
                period_type => PeriodType,
                duration => State#developmental_state.critical_period_duration
            }, State),

            {reply, ok, NewState}
    end;

handle_call({close_critical_period, IndividualId, PeriodType}, _From, State) ->
    case maps:get(IndividualId, State#developmental_state.individual_states, undefined) of
        undefined ->
            {reply, {error, not_found}, State};
        IndState ->
            ActivePeriods = IndState#individual_dev_state.active_periods,
            {ClosedPeriods, RemainingPeriods} = lists:partition(
                fun(P) -> P#critical_period.period_type =:= PeriodType end,
                ActivePeriods
            ),

            case ClosedPeriods of
                [] ->
                    {reply, {error, period_not_active}, State};
                [Closed | _] ->
                    %% Record in history
                    Gen = State#developmental_state.current_generation,
                    HistoryEntry = {PeriodType, Closed#critical_period.start_generation, Gen},
                    NewHistory = [HistoryEntry | IndState#individual_dev_state.period_history],

                    NewIndState = IndState#individual_dev_state{
                        active_periods = RemainingPeriods,
                        period_history = NewHistory
                    },

                    NewStates = maps:put(IndividualId, NewIndState,
                                        State#developmental_state.individual_states),
                    NewState = State#developmental_state{
                        individual_states = NewStates,
                        critical_periods_closed = State#developmental_state.critical_periods_closed + 1
                    },

                    maybe_emit_event(critical_period_closed, #{
                        individual_id => IndividualId,
                        period_type => PeriodType,
                        learning_acquired => Closed#critical_period.learning_acquired
                    }, State),

                    {reply, ok, NewState}
            end
    end;

handle_call({trigger_metamorphosis, IndividualId}, _From, State) ->
    case maps:get(IndividualId, State#developmental_state.individual_states, undefined) of
        undefined ->
            {reply, {error, not_found}, State};
        IndState ->
            #developmental_config{max_metamorphoses = MaxMeta} = State#developmental_state.config,
            case IndState#individual_dev_state.metamorphosis_count >= MaxMeta of
                true ->
                    {reply, {error, max_metamorphoses_reached}, State};
                false ->
                    %% Perform metamorphosis
                    Severity = State#developmental_state.metamorphosis_severity,
                    Gen = State#developmental_state.current_generation,

                    %% Reset plasticity based on severity
                    NewPlasticity = State#developmental_state.initial_plasticity * Severity +
                                   IndState#individual_dev_state.plasticity * (1.0 - Severity),

                    %% Reduce canalization
                    NewCanalization = IndState#individual_dev_state.canalization * (1.0 - Severity),

                    NewIndState = IndState#individual_dev_state{
                        plasticity = NewPlasticity,
                        canalization = NewCanalization,
                        metamorphosis_count = IndState#individual_dev_state.metamorphosis_count + 1,
                        last_metamorphosis = Gen
                    },

                    NewStates = maps:put(IndividualId, NewIndState,
                                        State#developmental_state.individual_states),
                    NewState = State#developmental_state{
                        individual_states = NewStates,
                        total_metamorphoses = State#developmental_state.total_metamorphoses + 1
                    },

                    maybe_emit_event(metamorphosis_triggered, #{
                        individual_id => IndividualId,
                        severity => Severity,
                        new_plasticity => NewPlasticity
                    }, State),

                    {reply, ok, NewState}
            end
    end;

handle_call({get_individual_state, IndividualId}, _From, State) ->
    IndState = maps:get(IndividualId, State#developmental_state.individual_states, undefined),
    {reply, IndState, State};

handle_call(get_state, _From, State) ->
    {reply, State, State};

handle_call(enable, _From, State) ->
    Config = State#developmental_state.config,
    NewConfig = Config#developmental_config{enabled = true},
    {reply, ok, State#developmental_state{config = NewConfig}};

handle_call(disable, _From, State) ->
    Config = State#developmental_state.config,
    NewConfig = Config#developmental_config{enabled = false},
    {reply, ok, State#developmental_state{config = NewConfig}};

handle_call(is_enabled, _From, State) ->
    {reply, State#developmental_state.config#developmental_config.enabled, State};

handle_call(signal_maturity_distribution, _From, State) ->
    %% Calculate average maturation across population
    IndStates = maps:values(State#developmental_state.individual_states),
    case IndStates of
        [] -> {reply, 0.5, State};
        _ ->
            Ages = [I#individual_dev_state.current_age || I <- IndStates],
            MaxAge = ?MAX_AGE,
            AvgMaturity = (lists:sum(Ages) / length(Ages)) / MaxAge,
            {reply, clamp(AvgMaturity, 0.0, 1.0), State}
    end;

handle_call(signal_plasticity_available, _From, State) ->
    IndStates = maps:values(State#developmental_state.individual_states),
    case IndStates of
        [] -> {reply, State#developmental_state.initial_plasticity, State};
        _ ->
            Plasticities = [I#individual_dev_state.plasticity || I <- IndStates],
            AvgPlasticity = lists:sum(Plasticities) / length(Plasticities),
            {reply, clamp(AvgPlasticity, 0.0, 1.0), State}
    end;

handle_call(signal_metamorphosis_rate, _From, State) ->
    %% Metamorphoses per generation
    Gen = max(1, State#developmental_state.current_generation),
    Rate = State#developmental_state.total_metamorphoses / Gen,
    {reply, clamp(Rate, 0.0, 1.0), State};

handle_call(_Request, _From, State) ->
    {reply, {error, unknown_request}, State}.

handle_cast({update_context, Context}, State) ->
    #developmental_state{config = Config} = State,
    case Config#developmental_config.enabled of
        false ->
            {noreply, State};
        true ->
            %% Collect sensors
            SensorVector = developmental_silo_sensors:collect_sensors(Context),

            %% Process through TWEANN controller
            ActuatorVector = process_through_controller(SensorVector, State),

            %% Apply actuators
            NewState = developmental_silo_actuators:apply_actuators(ActuatorVector, State),

            %% Update generation
            FinalState = NewState#developmental_state{
                current_generation = State#developmental_state.current_generation + 1
            },

            {noreply, FinalState}
    end;

handle_cast({register_individual, IndividualId}, State) ->
    Gen = State#developmental_state.current_generation,
    IndState = #individual_dev_state{
        individual_id = IndividualId,
        birth_generation = Gen,
        current_age = 0,
        current_stage = embryonic,
        stage_entry_generation = Gen,
        plasticity = State#developmental_state.initial_plasticity
    },
    NewStates = maps:put(IndividualId, IndState,
                        State#developmental_state.individual_states),
    {noreply, State#developmental_state{individual_states = NewStates}};

handle_cast({cross_silo, stress_level, Stress}, State) ->
    %% High stress may trigger early maturation
    case Stress > 0.7 of
        true ->
            %% Accelerate development
            NewSpeed = State#developmental_state.maturation_speed * 1.5,
            {noreply, State#developmental_state{maturation_speed = min(0.1, NewSpeed)}};
        false ->
            {noreply, State}
    end;

handle_cast({cross_silo, learning_opportunity, Opportunity}, State) ->
    %% Learning opportunity extends critical periods
    case Opportunity > 0.5 of
        true ->
            NewDuration = round(State#developmental_state.critical_period_duration * 1.2),
            {noreply, State#developmental_state{critical_period_duration = min(20, NewDuration)}};
        false ->
            {noreply, State}
    end;

handle_cast({cross_silo, resource_scarcity, Scarcity}, State) ->
    %% Scarcity may accelerate development (grow up faster in harsh environments)
    case Scarcity > 0.6 of
        true ->
            NewRate = State#developmental_state.growth_rate * 1.3,
            {noreply, State#developmental_state{growth_rate = min(0.2, NewRate)}};
        false ->
            {noreply, State}
    end;

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

initialize_stages(_Config) ->
    %% Default developmental stages
    [
        #developmental_stage{
            stage_name = embryonic,
            stage_index = 0,
            plasticity_range = {0.9, 1.0},
            growth_rate_range = {0.1, 0.2},
            typical_duration = 5,
            available_critical_periods = [sensory],
            mutation_rate_modifier = 1.5
        },
        #developmental_stage{
            stage_name = juvenile,
            stage_index = 1,
            plasticity_range = {0.7, 0.9},
            growth_rate_range = {0.05, 0.1},
            typical_duration = 15,
            available_critical_periods = [sensory, motor],
            mutation_rate_modifier = 1.2
        },
        #developmental_stage{
            stage_name = adolescent,
            stage_index = 2,
            plasticity_range = {0.4, 0.7},
            growth_rate_range = {0.02, 0.05},
            typical_duration = 20,
            available_critical_periods = [cognitive, social],
            mutation_rate_modifier = 1.0
        },
        #developmental_stage{
            stage_name = adult,
            stage_index = 3,
            plasticity_range = {0.2, 0.4},
            growth_rate_range = {0.0, 0.02},
            typical_duration = 40,
            available_critical_periods = [],
            mutation_rate_modifier = 0.8
        },
        #developmental_stage{
            stage_name = senescent,
            stage_index = 4,
            plasticity_range = {0.1, 0.2},
            growth_rate_range = {0.0, 0.0},
            typical_duration = 20,
            available_critical_periods = [],
            mutation_rate_modifier = 0.5
        }
    ].

check_stage_transition(IndState, Age, State) ->
    Stages = State#developmental_state.stages,
    CurrentStage = IndState#individual_dev_state.current_stage,
    CurrentIndex = stage_index(CurrentStage, Stages),

    %% Check if should advance
    case CurrentIndex < length(Stages) - 1 of
        false -> CurrentStage;
        true ->
            NextStage = lists:nth(CurrentIndex + 2, Stages),
            CumulativeDuration = cumulative_duration(Stages, CurrentIndex),
            case Age >= CumulativeDuration of
                true -> NextStage#developmental_stage.stage_name;
                false -> CurrentStage
            end
    end.

stage_index(StageName, Stages) ->
    case lists:search(fun(S) -> S#developmental_stage.stage_name =:= StageName end, Stages) of
        {value, Stage} -> Stage#developmental_stage.stage_index;
        false -> 0
    end.

cumulative_duration(Stages, UpToIndex) ->
    RelevantStages = lists:sublist(Stages, UpToIndex + 1),
    lists:sum([S#developmental_stage.typical_duration || S <- RelevantStages]).

check_automatic_critical_periods(IndState, State) ->
    #developmental_state{config = Config} = State,
    case Config#developmental_config.enable_critical_periods of
        false -> IndState;
        true ->
            %% Check if current stage has critical periods to open
            Stages = State#developmental_state.stages,
            CurrentStage = IndState#individual_dev_state.current_stage,
            StageInfo = lists:keyfind(CurrentStage, #developmental_stage.stage_name, Stages),
            case StageInfo of
                false -> IndState;
                _ ->
                    AvailablePeriods = StageInfo#developmental_stage.available_critical_periods,
                    ActiveTypes = [P#critical_period.period_type
                                  || P <- IndState#individual_dev_state.active_periods],
                    HistoryTypes = [Type || {Type, _, _} <- IndState#individual_dev_state.period_history],

                    %% Open periods not yet experienced
                    ToOpen = AvailablePeriods -- (ActiveTypes ++ HistoryTypes),

                    %% Randomly open one if available
                    case ToOpen of
                        [] -> IndState;
                        _ ->
                            %% 10% chance per generation to open a period
                            case rand:uniform() < 0.1 of
                                true ->
                                    PeriodType = lists:nth(rand:uniform(length(ToOpen)), ToOpen),
                                    Period = #critical_period{
                                        period_type = PeriodType,
                                        target_capability = PeriodType,
                                        start_generation = State#developmental_state.current_generation,
                                        duration = State#developmental_state.critical_period_duration,
                                        intensity = 1.0,
                                        decay_rate = 0.1,
                                        learning_rate_multiplier = 2.0
                                    },
                                    IndState#individual_dev_state{
                                        active_periods = [Period | IndState#individual_dev_state.active_periods]
                                    };
                                false ->
                                    IndState
                            end
                    end
            end
    end.

check_metamorphosis(IndState, State) ->
    #developmental_state{
        config = Config,
        metamorphosis_trigger = Trigger,
        metamorphosis_severity = Severity
    } = State,

    case Config#developmental_config.enable_metamorphosis of
        false -> IndState;
        true ->
            %% Check if maturity threshold reached
            Maturity = IndState#individual_dev_state.current_age / ?MAX_AGE,
            MaxMeta = Config#developmental_config.max_metamorphoses,

            case Maturity >= Trigger andalso
                 IndState#individual_dev_state.metamorphosis_count < MaxMeta of
                false -> IndState;
                true ->
                    %% 20% chance when conditions met
                    case rand:uniform() < 0.2 of
                        true ->
                            Gen = State#developmental_state.current_generation,
                            NewPlasticity = State#developmental_state.initial_plasticity * Severity +
                                           IndState#individual_dev_state.plasticity * (1.0 - Severity),

                            maybe_emit_event(metamorphosis_triggered, #{
                                individual_id => IndState#individual_dev_state.individual_id,
                                severity => Severity,
                                new_plasticity => NewPlasticity
                            }, State),

                            IndState#individual_dev_state{
                                plasticity = NewPlasticity,
                                canalization = IndState#individual_dev_state.canalization * (1.0 - Severity),
                                metamorphosis_count = IndState#individual_dev_state.metamorphosis_count + 1,
                                last_metamorphosis = Gen
                            };
                        false ->
                            IndState
                    end
            end
    end.

process_through_controller(SensorVector, State) ->
    case State#developmental_state.l2_enabled of
        true ->
            apply_l2_guidance(SensorVector, State);
        false ->
            lists:duplicate(developmental_silo_actuators:actuator_count(), 0.5)
    end.

apply_l2_guidance(_SensorVector, State) ->
    case State#developmental_state.l2_guidance of
        undefined ->
            lists:duplicate(developmental_silo_actuators:actuator_count(), 0.5);
        #l2_guidance{
            development_speed = Speed,
            plasticity_pressure = PlastPress,
            critical_period_incentive = CritInc,
            metamorphosis_incentive = MetaInc
        } ->
            [
                Speed,              % growth_rate
                Speed,              % maturation_speed
                CritInc,            % critical_period_duration
                1.0 - PlastPress,   % plasticity_decay_rate (inverse)
                PlastPress,         % initial_plasticity
                0.1,                % developmental_noise_level
                MetaInc,            % metamorphosis_trigger
                MetaInc             % metamorphosis_severity
            ]
    end.

maybe_emit_event(EventType, Payload, State) ->
    case State#developmental_state.config#developmental_config.emit_events of
        true ->
            Event = #{
                type => EventType,
                silo => developmental,
                timestamp => erlang:system_time(millisecond),
                generation => State#developmental_state.current_generation,
                payload => Payload
            },
            event_bus:publish(developmental_silo_events, Event);
        false ->
            ok
    end.

clamp(Value, Min, Max) ->
    max(Min, min(Max, Value)).
```

---

## 7. Cross-Silo Signal Matrix

### 7.1 Outgoing Signals

| Signal | To Silo | Type | Description |
|--------|---------|------|-------------|
| `maturity_distribution` | Task | float() | Population age structure affects mutation rates |
| `plasticity_available` | Cultural | float() | High plasticity enables faster cultural learning |
| `metamorphosis_rate` | Ecological | float() | Radical changes affect population stability |
| `growth_stage` | Morphological | float() | Developmental stage affects structure growth |
| `critical_timing` | Temporal | float() | Critical period timing affects time budgets |

### 7.2 Incoming Signals

| Signal | From Silo | Effect |
|--------|-----------|--------|
| `stress_level` | Ecological | Stress may trigger early maturation |
| `learning_opportunity` | Cultural | Opportunity extends critical periods |
| `resource_scarcity` | Resource | Scarcity may accelerate development |
| `environmental_pressure` | Ecological | Pressure affects developmental speed |
| `social_mentoring` | Social | Mentoring affects learning during critical periods |

### 7.3 Complete 13-Silo Matrix (Developmental Row)

| To → | Task | Resource | Distrib | Social | Cultural | Ecological | Morpho | Temporal | Competitive | Develop | Regulatory | Economic | Comm |
|------|------|----------|---------|--------|----------|------------|--------|----------|-------------|---------|------------|----------|------|
| **From Developmental** | maturity | - | age_dist | mentoring | plasticity | stress_resp | growth | critical | - | - | expression | invest | learning |

---

## 8. Events Emitted

### 8.1 Event Specifications

| Event | Trigger | Payload |
|-------|---------|---------|
| `development_stage_changed` | Stage transition | `{individual_id, old_stage, new_stage}` |
| `critical_period_opened` | Period began | `{individual_id, period_type, duration}` |
| `critical_period_closed` | Period ended | `{individual_id, period_type, learning_acquired}` |
| `plasticity_decreased` | Plasticity dropped | `{individual_id, old_level, new_level}` |
| `maturation_completed` | Fully mature | `{individual_id, final_structure, age}` |
| `metamorphosis_triggered` | Radical change | `{individual_id, trigger_reason, changes}` |
| `developmental_milestone` | Key achievement | `{individual_id, milestone, age}` |
| `canalization_established` | Trait fixed | `{individual_id, trait, stability}` |

### 8.2 Event Payload Types

```erlang
%% Event type specifications
-type developmental_event() ::
    stage_changed_event() |
    critical_period_opened_event() |
    critical_period_closed_event() |
    metamorphosis_triggered_event() |
    milestone_event().

-type stage_changed_event() :: #{
    type := development_stage_changed,
    silo := developmental,
    timestamp := non_neg_integer(),
    generation := generation(),
    payload := #{
        individual_id := binary(),
        old_stage := atom(),
        new_stage := atom(),
        age_at_transition := age()
    }
}.

-type critical_period_opened_event() :: #{
    type := critical_period_opened,
    silo := developmental,
    timestamp := non_neg_integer(),
    generation := generation(),
    payload := #{
        individual_id := binary(),
        period_type := atom(),
        duration := pos_integer(),
        intensity := float()
    }
}.

-type metamorphosis_triggered_event() :: #{
    type := metamorphosis_triggered,
    silo := developmental,
    timestamp := non_neg_integer(),
    generation := generation(),
    payload := #{
        individual_id := binary(),
        severity := float(),
        old_plasticity := float(),
        new_plasticity := float(),
        metamorphosis_count := pos_integer()
    }
}.
```

---

## 9. Value of Event Storage

### 9.1 Analysis Capabilities

| Stored Events | Analysis Enabled |
|---------------|------------------|
| `stage_changed` | Understand developmental trajectories |
| `critical_period_*` | Identify optimal critical period timing |
| `metamorphosis_triggered` | Track which metamorphoses improve fitness |
| `plasticity_decreased` | Understand plasticity/stability tradeoffs |
| `canalization_established` | Identify which traits should be fixed |

### 9.2 Business Intelligence

- **Development Patterns**: Identify optimal developmental trajectories
- **Critical Period Timing**: Learn when to open/close learning windows
- **Metamorphosis Success**: Track which metamorphoses improve fitness
- **Plasticity Management**: Understand plasticity/stability tradeoffs
- **Lifetime Profiles**: Build age-appropriate deployment models
- **Transfer Learning**: Identify best critical periods for domain adaptation

---

## 10. Multi-Layer Hierarchy

### 10.1 Layer Responsibilities

| Level | Role | Controls |
|-------|------|----------|
| **L0** | Hard limits | Minimum maturation time, maximum plasticity, stage boundaries |
| **L1** | Tactical | Adjust development speed based on environment |
| **L2** | Strategic | Learn optimal developmental programs for task classes |

---

## 11. Enable/Disable Effects

### 11.1 When Disabled

| Aspect | Behavior |
|--------|----------|
| Developmental stages | Networks born fully formed (no stages) |
| Critical periods | No critical periods (learning is uniform) |
| Plasticity decay | No decay (always equally learnable) |
| Metamorphosis | No metamorphosis (structure fixed at birth) |
| Age tracking | No age concept |

### 11.2 When Enabled

| Aspect | Behavior |
|--------|----------|
| Developmental stages | Networks progress through life stages |
| Critical periods | Time-limited learning windows |
| Plasticity decay | Plasticity decreases with age |
| Metamorphosis | Radical adaptation possible |
| Age tracking | Full age-aware development |

---

## 12. Implementation Phases

### Phase 1: Core Infrastructure
- [ ] Create `developmental_silo.hrl` with record definitions
- [ ] Implement `developmental_silo_sensors.erl` with all 10 sensors
- [ ] Implement `developmental_silo_actuators.erl` with all 8 actuators
- [ ] Basic `developmental_silo.erl` gen_server

### Phase 2: Developmental Stages
- [ ] Stage definition and configuration
- [ ] Stage transition logic
- [ ] Age advancement system
- [ ] Stage-specific parameters

### Phase 3: Critical Periods
- [ ] Critical period opening/closing
- [ ] Period intensity decay
- [ ] Learning rate modulation during periods
- [ ] Automatic period management

### Phase 4: Plasticity & Metamorphosis
- [ ] Plasticity decay system
- [ ] Canalization tracking
- [ ] Metamorphosis triggering
- [ ] Post-metamorphosis reset

### Phase 5: Cross-Silo Integration
- [ ] Outgoing signal implementations
- [ ] Incoming signal handlers
- [ ] Event emission to event store
- [ ] Integration with cultural and ecological silos

### Phase 6: Testing & Tuning
- [ ] Unit tests for all stages
- [ ] Integration tests with lifetime learning
- [ ] Tune developmental parameters
- [ ] Performance benchmarks

---

## 13. Success Criteria

1. **Stage Progression**: All individuals progress through defined stages
2. **Critical Period Function**: Learning during critical periods > 2x faster
3. **Plasticity Decay**: Plasticity decreases predictably with age
4. **Metamorphosis Success**: > 50% of metamorphoses improve fitness
5. **Cross-Silo Integration**: All signals documented and functional
6. **Event Coverage**: All event types emitted with proper payloads
7. **Deployment Adaptation**: Networks can fine-tune in deployment
