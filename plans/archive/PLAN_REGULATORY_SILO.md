# Plan: Regulatory Silo for Liquid Conglomerate

**Status:** Planning
**Created:** 2025-12-23
**Last Updated:** 2025-12-23
**Related:** PLAN_SOCIAL_SILO.md, PLAN_CULTURAL_SILO.md, PLAN_ECOLOGICAL_SILO.md, PLAN_MORPHOLOGICAL_SILO.md, PLAN_TEMPORAL_SILO.md, PLAN_COMPETITIVE_SILO.md, PLAN_DEVELOPMENTAL_SILO.md

---

## Overview

The Regulatory Silo manages gene expression and network activation - which parts of the genome are expressed in which contexts. Like biological gene regulation, this enables the same genotype to produce different phenotypes based on environmental conditions, supporting multi-modal behavior and graceful degradation.

---

## 1. Motivation

### Problem Statement

Traditional neuroevolution treats networks as static expression of genotype:
- **No context-dependent behavior**: Same network behavior in all contexts
- **No dormant capabilities**: All genes always expressed (wasteful)
- **No mode switching**: Can't change "modes" based on environment
- **No graceful degradation**: Can't disable modules when resources constrained

Biological systems show sophisticated gene regulation:
- Context-dependent gene expression
- Dormant genes activated under stress
- Epigenetic inheritance of expression patterns
- Multi-modal behavior from single genome

### Business Value

| Benefit | Impact |
|---------|--------|
| Multi-task agents | Same network, different operational modes |
| Graceful degradation | Disable modules if resources constrained |
| Future-proofing | Dormant capabilities for future scenarios |
| Energy efficiency | Only run what's needed |
| Context awareness | Automatic adaptation to environment |

### Training Velocity Impact

| Metric | Without Regulatory Silo | With Regulatory Silo |
|--------|------------------------|---------------------|
| Multi-context fitness | Average | Context-optimal |
| Resource efficiency | Low (always full) | High (on-demand) |
| Behavioral diversity | Single mode | Multiple modes |
| Adaptation speed | Slow (re-evolve) | Fast (switch modes) |

---

## 2. Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                       REGULATORY SILO                                │
├─────────────────────────────────────────────────────────────────────┤
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │                      L0 SENSORS (10)                         │    │
│  │                                                              │    │
│  │  Expression       Module         Regulatory    Epigenetic   │    │
│  │  ┌─────────┐     ┌─────────┐    ┌─────────┐  ┌─────────┐   │    │
│  │  │active_  │     │module_  │    │regulatory│ │epigenetic│   │    │
│  │  │gene_    │     │activation│   │_network_ │ │_mark_    │   │    │
│  │  │ratio    │     │_pattern │    │complexity│ │density   │   │    │
│  │  │dormant_ │     │context_ │    │expression│ └─────────┘   │    │
│  │  │capability│    │switch_  │    │_noise    │               │    │
│  │  │_count   │     │frequency│    └─────────┘               │    │
│  │  └─────────┘     └─────────┘                               │    │
│  │                                                              │    │
│  │  Transcription    Conditional                               │    │
│  │  ┌─────────┐     ┌─────────┐                                │    │
│  │  │transcrip│     │conditional│                               │    │
│  │  │_factor_ │     │_expression│                               │    │
│  │  │diversity│     │_ratio    │                               │    │
│  │  └─────────┘     └─────────┘                               │    │
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
│  │  Expression       Switching       Dormancy     Epigenetic   │    │
│  │  ┌─────────┐     ┌─────────┐     ┌─────────┐  ┌─────────┐  │    │
│  │  │expression│    │context_ │     │dormancy_ │ │epigenetic│  │    │
│  │  │_threshold│    │sensitivity│   │maintenance││_inherit_ │  │    │
│  │  │regulatory│    │module_  │     │_cost     │ │strength  │  │    │
│  │  │_mutation_│    │switching│     └─────────┘  └─────────┘  │    │
│  │  │rate      │    │_cost    │                               │    │
│  │  └─────────┘     └─────────┘                               │    │
│  └─────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 3. L0 Sensors

### 3.1 Sensor Specifications

| ID | Name | Range | Description |
|----|------|-------|-------------|
| 1 | `active_gene_ratio` | [0.0, 1.0] | Expressed genes / total genes |
| 2 | `dormant_capability_count` | [0.0, 1.0] | Unexpressed but available capabilities |
| 3 | `module_activation_pattern` | [0.0, 1.0] | Entropy of module activation |
| 4 | `context_switch_frequency` | [0.0, 1.0] | How often context changes |
| 5 | `regulatory_network_complexity` | [0.0, 1.0] | Complexity of regulation logic |
| 6 | `expression_noise` | [0.0, 1.0] | Stochasticity in expression |
| 7 | `epigenetic_mark_density` | [0.0, 1.0] | Epigenetic marks per gene |
| 8 | `transcription_factor_diversity` | [0.0, 1.0] | Variety of regulatory signals |
| 9 | `conditional_expression_ratio` | [0.0, 1.0] | Context-dependent vs constitutive |
| 10 | `regulatory_fitness_contribution` | [0.0, 1.0] | How much regulation helps fitness |

### 3.2 Sensor Module

```erlang
%%%-------------------------------------------------------------------
%%% @doc Regulatory Silo L0 Sensors
%%% Monitors gene expression dynamics: activation patterns, context
%%% switching, epigenetic marks, and regulatory complexity.
%%% @end
%%%-------------------------------------------------------------------
-module(regulatory_silo_sensors).

-behaviour(l0_sensor_behaviour).

%% API
-export([sensor_specs/0,
         collect_sensors/1,
         sensor_count/0]).

%% Sensor collection
-export([collect_active_gene_ratio/1,
         collect_dormant_capability_count/1,
         collect_module_activation_pattern/1,
         collect_context_switch_frequency/1,
         collect_regulatory_network_complexity/1,
         collect_expression_noise/1,
         collect_epigenetic_mark_density/1,
         collect_transcription_factor_diversity/1,
         collect_conditional_expression_ratio/1,
         collect_regulatory_fitness_contribution/1]).

-include("regulatory_silo.hrl").

%%====================================================================
%% Behaviour Callbacks
%%====================================================================

-spec sensor_specs() -> [l0_sensor_spec()].
sensor_specs() ->
    [
        #{id => active_gene_ratio,
          name => <<"Active Gene Ratio">>,
          range => {0.0, 1.0},
          description => <<"Expressed genes / total genes">>},

        #{id => dormant_capability_count,
          name => <<"Dormant Capabilities">>,
          range => {0.0, 1.0},
          description => <<"Unexpressed but available capabilities">>},

        #{id => module_activation_pattern,
          name => <<"Module Activation Pattern">>,
          range => {0.0, 1.0},
          description => <<"Entropy of module activation">>},

        #{id => context_switch_frequency,
          name => <<"Context Switch Frequency">>,
          range => {0.0, 1.0},
          description => <<"How often context changes">>},

        #{id => regulatory_network_complexity,
          name => <<"Regulatory Complexity">>,
          range => {0.0, 1.0},
          description => <<"Complexity of regulation logic">>},

        #{id => expression_noise,
          name => <<"Expression Noise">>,
          range => {0.0, 1.0},
          description => <<"Stochasticity in expression">>},

        #{id => epigenetic_mark_density,
          name => <<"Epigenetic Mark Density">>,
          range => {0.0, 1.0},
          description => <<"Epigenetic marks per gene">>},

        #{id => transcription_factor_diversity,
          name => <<"Transcription Factor Diversity">>,
          range => {0.0, 1.0},
          description => <<"Variety of regulatory signals">>},

        #{id => conditional_expression_ratio,
          name => <<"Conditional Expression Ratio">>,
          range => {0.0, 1.0},
          description => <<"Context-dependent vs constitutive">>},

        #{id => regulatory_fitness_contribution,
          name => <<"Regulatory Fitness Contribution">>,
          range => {0.0, 1.0},
          description => <<"How much regulation helps fitness">>}
    ].

-spec sensor_count() -> pos_integer().
sensor_count() -> 10.

-spec collect_sensors(regulatory_context()) -> sensor_vector().
collect_sensors(Context) ->
    [
        collect_active_gene_ratio(Context),
        collect_dormant_capability_count(Context),
        collect_module_activation_pattern(Context),
        collect_context_switch_frequency(Context),
        collect_regulatory_network_complexity(Context),
        collect_expression_noise(Context),
        collect_epigenetic_mark_density(Context),
        collect_transcription_factor_diversity(Context),
        collect_conditional_expression_ratio(Context),
        collect_regulatory_fitness_contribution(Context)
    ].

%%====================================================================
%% Individual Sensor Collection
%%====================================================================

%% @doc Ratio of actively expressed genes
-spec collect_active_gene_ratio(regulatory_context()) -> float().
collect_active_gene_ratio(#regulatory_context{
    active_genes = Active,
    total_genes = Total
}) ->
    case Total of
        0 -> 0.5;
        _ -> clamp(Active / Total, 0.0, 1.0)
    end.

%% @doc Count of dormant but available capabilities
-spec collect_dormant_capability_count(regulatory_context()) -> float().
collect_dormant_capability_count(#regulatory_context{
    dormant_genes = Dormant,
    total_genes = Total
}) ->
    case Total of
        0 -> 0.0;
        _ -> clamp(Dormant / Total, 0.0, 1.0)
    end.

%% @doc Entropy of module activation pattern
-spec collect_module_activation_pattern(regulatory_context()) -> float().
collect_module_activation_pattern(#regulatory_context{
    module_activations = Activations
}) ->
    case Activations of
        [] -> 0.5;
        _ ->
            %% Calculate entropy of activation distribution
            Total = lists:sum(Activations),
            case Total of
                0 -> 0.5;
                _ ->
                    Probs = [A / Total || A <- Activations, A > 0],
                    Entropy = -lists:sum([P * math:log2(P) || P <- Probs]),
                    MaxEntropy = math:log2(max(1, length(Activations))),
                    case MaxEntropy of
                        0.0 -> 0.5;
                        _ -> clamp(Entropy / MaxEntropy, 0.0, 1.0)
                    end
            end
    end.

%% @doc Frequency of context switches
-spec collect_context_switch_frequency(regulatory_context()) -> float().
collect_context_switch_frequency(#regulatory_context{
    context_switches = Switches,
    observation_window = Window
}) ->
    case Window of
        0 -> 0.0;
        _ -> clamp(Switches / Window, 0.0, 1.0)
    end.

%% @doc Complexity of regulatory network
-spec collect_regulatory_network_complexity(regulatory_context()) -> float().
collect_regulatory_network_complexity(#regulatory_context{
    regulatory_connections = Connections,
    total_genes = TotalGenes
}) ->
    %% Complexity = connections / (genes * genes)
    MaxConnections = TotalGenes * TotalGenes,
    case MaxConnections of
        0 -> 0.0;
        _ -> clamp(Connections / MaxConnections, 0.0, 1.0)
    end.

%% @doc Noise in expression levels
-spec collect_expression_noise(regulatory_context()) -> float().
collect_expression_noise(#regulatory_context{
    expression_variance = Variance
}) ->
    %% Normalize variance
    clamp(Variance * 4.0, 0.0, 1.0).

%% @doc Density of epigenetic marks
-spec collect_epigenetic_mark_density(regulatory_context()) -> float().
collect_epigenetic_mark_density(#regulatory_context{
    epigenetic_marks = Marks,
    total_genes = Total
}) ->
    case Total of
        0 -> 0.0;
        _ -> clamp(Marks / Total, 0.0, 1.0)
    end.

%% @doc Diversity of transcription factors
-spec collect_transcription_factor_diversity(regulatory_context()) -> float().
collect_transcription_factor_diversity(#regulatory_context{
    unique_transcription_factors = Unique,
    max_transcription_factors = Max
}) ->
    case Max of
        0 -> 0.5;
        _ -> clamp(Unique / Max, 0.0, 1.0)
    end.

%% @doc Ratio of conditional vs constitutive expression
-spec collect_conditional_expression_ratio(regulatory_context()) -> float().
collect_conditional_expression_ratio(#regulatory_context{
    conditional_genes = Conditional,
    constitutive_genes = Constitutive
}) ->
    Total = Conditional + Constitutive,
    case Total of
        0 -> 0.5;
        _ -> clamp(Conditional / Total, 0.0, 1.0)
    end.

%% @doc How much regulation contributes to fitness
-spec collect_regulatory_fitness_contribution(regulatory_context()) -> float().
collect_regulatory_fitness_contribution(#regulatory_context{
    fitness_with_regulation = WithReg,
    fitness_without_regulation = WithoutReg
}) ->
    case WithoutReg of
        0.0 -> 0.5;
        _ ->
            Improvement = (WithReg - WithoutReg) / WithoutReg,
            %% Map [-0.5, 0.5] to [0, 1]
            clamp(Improvement + 0.5, 0.0, 1.0)
    end.

%%====================================================================
%% Internal Functions
%%====================================================================

clamp(Value, Min, Max) ->
    max(Min, min(Max, Value)).
```

---

## 4. L0 Actuators

### 4.1 Actuator Specifications

| ID | Name | Range | Default | Description |
|----|------|-------|---------|-------------|
| 1 | `expression_threshold` | [0.0, 1.0] | 0.5 | Signal needed for expression |
| 2 | `regulatory_mutation_rate` | [0.0, 0.1] | 0.02 | Rate of regulation changes |
| 3 | `context_sensitivity` | [0.0, 1.0] | 0.5 | How responsive to context |
| 4 | `module_switching_cost` | [0.0, 0.5] | 0.1 | Fitness cost of switching |
| 5 | `dormancy_maintenance_cost` | [0.0, 0.1] | 0.01 | Cost per dormant gene |
| 6 | `epigenetic_inheritance_strength` | [0.0, 1.0] | 0.5 | How much marks inherit |
| 7 | `constitutive_expression_bonus` | [0.0, 0.2] | 0.05 | Bonus for always-on genes |
| 8 | `regulatory_complexity_penalty` | [0.0, 0.1] | 0.02 | Cost of complex regulation |

### 4.2 Actuator Module

```erlang
%%%-------------------------------------------------------------------
%%% @doc Regulatory Silo L0 Actuators
%%% Controls gene expression parameters: thresholds, mutation rates,
%%% context sensitivity, and epigenetic inheritance.
%%% @end
%%%-------------------------------------------------------------------
-module(regulatory_silo_actuators).

-behaviour(l0_actuator_behaviour).

%% API
-export([actuator_specs/0,
         apply_actuators/2,
         actuator_count/0]).

%% Individual actuator application
-export([apply_expression_threshold/2,
         apply_regulatory_mutation_rate/2,
         apply_context_sensitivity/2,
         apply_module_switching_cost/2,
         apply_dormancy_maintenance_cost/2,
         apply_epigenetic_inheritance_strength/2,
         apply_constitutive_expression_bonus/2,
         apply_regulatory_complexity_penalty/2]).

-include("regulatory_silo.hrl").

%%====================================================================
%% Behaviour Callbacks
%%====================================================================

-spec actuator_specs() -> [l0_actuator_spec()].
actuator_specs() ->
    [
        #{id => expression_threshold,
          name => <<"Expression Threshold">>,
          range => {0.0, 1.0},
          default => 0.5,
          description => <<"Signal needed for expression">>},

        #{id => regulatory_mutation_rate,
          name => <<"Regulatory Mutation Rate">>,
          range => {0.0, 0.1},
          default => 0.02,
          description => <<"Rate of regulation changes">>},

        #{id => context_sensitivity,
          name => <<"Context Sensitivity">>,
          range => {0.0, 1.0},
          default => 0.5,
          description => <<"How responsive to context">>},

        #{id => module_switching_cost,
          name => <<"Module Switching Cost">>,
          range => {0.0, 0.5},
          default => 0.1,
          description => <<"Fitness cost of switching">>},

        #{id => dormancy_maintenance_cost,
          name => <<"Dormancy Maintenance Cost">>,
          range => {0.0, 0.1},
          default => 0.01,
          description => <<"Cost per dormant gene">>},

        #{id => epigenetic_inheritance_strength,
          name => <<"Epigenetic Inheritance">>,
          range => {0.0, 1.0},
          default => 0.5,
          description => <<"How much marks inherit">>},

        #{id => constitutive_expression_bonus,
          name => <<"Constitutive Expression Bonus">>,
          range => {0.0, 0.2},
          default => 0.05,
          description => <<"Bonus for always-on genes">>},

        #{id => regulatory_complexity_penalty,
          name => <<"Regulatory Complexity Penalty">>,
          range => {0.0, 0.1},
          default => 0.02,
          description => <<"Cost of complex regulation">>}
    ].

-spec actuator_count() -> pos_integer().
actuator_count() -> 8.

-spec apply_actuators(actuator_vector(), regulatory_state()) -> regulatory_state().
apply_actuators(Outputs, State) when length(Outputs) =:= 8 ->
    [ExprThresh, RegMut, CtxSens, SwitchCost,
     DormCost, EpiInherit, ConstBonus, ComplexPen] = Outputs,

    State1 = apply_expression_threshold(ExprThresh, State),
    State2 = apply_regulatory_mutation_rate(RegMut, State1),
    State3 = apply_context_sensitivity(CtxSens, State2),
    State4 = apply_module_switching_cost(SwitchCost, State3),
    State5 = apply_dormancy_maintenance_cost(DormCost, State4),
    State6 = apply_epigenetic_inheritance_strength(EpiInherit, State5),
    State7 = apply_constitutive_expression_bonus(ConstBonus, State6),
    apply_regulatory_complexity_penalty(ComplexPen, State7).

%%====================================================================
%% Individual Actuator Application
%%====================================================================

%% @doc Apply expression threshold
-spec apply_expression_threshold(float(), regulatory_state()) -> regulatory_state().
apply_expression_threshold(Output, State) ->
    %% Output in [0,1] -> Threshold in [0.0, 1.0]
    State#regulatory_state{expression_threshold = Output}.

%% @doc Apply regulatory mutation rate
-spec apply_regulatory_mutation_rate(float(), regulatory_state()) -> regulatory_state().
apply_regulatory_mutation_rate(Output, State) ->
    %% Output in [0,1] -> Rate in [0.0, 0.1]
    Rate = Output * 0.1,
    State#regulatory_state{regulatory_mutation_rate = Rate}.

%% @doc Apply context sensitivity
-spec apply_context_sensitivity(float(), regulatory_state()) -> regulatory_state().
apply_context_sensitivity(Output, State) ->
    %% Output in [0,1] -> Sensitivity in [0.0, 1.0]
    State#regulatory_state{context_sensitivity = Output}.

%% @doc Apply module switching cost
-spec apply_module_switching_cost(float(), regulatory_state()) -> regulatory_state().
apply_module_switching_cost(Output, State) ->
    %% Output in [0,1] -> Cost in [0.0, 0.5]
    Cost = Output * 0.5,
    State#regulatory_state{module_switching_cost = Cost}.

%% @doc Apply dormancy maintenance cost
-spec apply_dormancy_maintenance_cost(float(), regulatory_state()) -> regulatory_state().
apply_dormancy_maintenance_cost(Output, State) ->
    %% Output in [0,1] -> Cost in [0.0, 0.1]
    Cost = Output * 0.1,
    State#regulatory_state{dormancy_maintenance_cost = Cost}.

%% @doc Apply epigenetic inheritance strength
-spec apply_epigenetic_inheritance_strength(float(), regulatory_state()) -> regulatory_state().
apply_epigenetic_inheritance_strength(Output, State) ->
    %% Output in [0,1] -> Strength in [0.0, 1.0]
    State#regulatory_state{epigenetic_inheritance_strength = Output}.

%% @doc Apply constitutive expression bonus
-spec apply_constitutive_expression_bonus(float(), regulatory_state()) -> regulatory_state().
apply_constitutive_expression_bonus(Output, State) ->
    %% Output in [0,1] -> Bonus in [0.0, 0.2]
    Bonus = Output * 0.2,
    State#regulatory_state{constitutive_expression_bonus = Bonus}.

%% @doc Apply regulatory complexity penalty
-spec apply_regulatory_complexity_penalty(float(), regulatory_state()) -> regulatory_state().
apply_regulatory_complexity_penalty(Output, State) ->
    %% Output in [0,1] -> Penalty in [0.0, 0.1]
    Penalty = Output * 0.1,
    State#regulatory_state{regulatory_complexity_penalty = Penalty}.
```

---

## 5. Record Definitions

```erlang
%%%-------------------------------------------------------------------
%%% @doc Regulatory Silo Header
%%% Record definitions for gene expression and regulatory dynamics.
%%% @end
%%%-------------------------------------------------------------------

-ifndef(REGULATORY_SILO_HRL).
-define(REGULATORY_SILO_HRL, true).

%%====================================================================
%% Types
%%====================================================================

-type sensor_vector() :: [float()].
-type actuator_vector() :: [float()].
-type gene_id() :: binary().
-type module_id() :: binary().
-type context_id() :: atom().
-type generation() :: non_neg_integer().
-type expression_level() :: float().

%%====================================================================
%% Gene Record
%%====================================================================

-record(gene, {
    gene_id :: gene_id(),
    module :: module_id(),

    %% Expression state
    expressed = false :: boolean(),
    expression_level = 0.0 :: expression_level(),

    %% Regulation
    promoters = [] :: [transcription_factor()],
    repressors = [] :: [transcription_factor()],
    expression_condition :: fun((context()) -> boolean()) | undefined,

    %% Type
    is_constitutive = false :: boolean(),
    is_dormant = false :: boolean(),

    %% Epigenetics
    epigenetic_marks = [] :: [epigenetic_mark()]
}).

-type gene() :: #gene{}.

%%====================================================================
%% Transcription Factor Record
%%====================================================================

-record(transcription_factor, {
    factor_id :: binary(),
    factor_type :: activator | repressor,
    binding_strength :: float(),
    context_conditions :: [context_id()]
}).

-type transcription_factor() :: #transcription_factor{}.

%%====================================================================
%% Epigenetic Mark Record
%%====================================================================

-record(epigenetic_mark, {
    mark_type :: methylation | acetylation | phosphorylation,
    position :: pos_integer(),
    strength :: float(),
    inherited_from :: generation() | undefined
}).

-type epigenetic_mark() :: #epigenetic_mark{}.

%%====================================================================
%% Module Record
%%====================================================================

-record(module, {
    module_id :: module_id(),
    name :: binary(),

    %% Genes in module
    gene_ids :: [gene_id()],

    %% Activation
    is_active = false :: boolean(),
    activation_threshold :: float(),
    activation_history = [] :: [{generation(), boolean()}],

    %% Context
    preferred_contexts = [] :: [context_id()],
    excluded_contexts = [] :: [context_id()]
}).

-type module() :: #module{}.

%%====================================================================
%% Context Record
%%====================================================================

-record(context, {
    context_id :: context_id(),
    context_type :: environmental | behavioral | metabolic,

    %% Signals
    active_signals = [] :: [atom()],
    signal_strengths = #{} :: #{atom() => float()},

    %% History
    duration = 0 :: non_neg_integer(),
    entry_generation :: generation()
}).

-type context() :: #context{}.

%%====================================================================
%% Context (Input to Sensors)
%%====================================================================

-record(regulatory_context, {
    %% Gene counts
    active_genes = 0 :: non_neg_integer(),
    dormant_genes = 0 :: non_neg_integer(),
    total_genes = 100 :: non_neg_integer(),

    %% Module activations
    module_activations = [] :: [non_neg_integer()],

    %% Switching
    context_switches = 0 :: non_neg_integer(),
    observation_window = 100 :: non_neg_integer(),

    %% Complexity
    regulatory_connections = 0 :: non_neg_integer(),

    %% Noise
    expression_variance = 0.0 :: float(),

    %% Epigenetics
    epigenetic_marks = 0 :: non_neg_integer(),

    %% Transcription factors
    unique_transcription_factors = 10 :: non_neg_integer(),
    max_transcription_factors = 50 :: non_neg_integer(),

    %% Expression types
    conditional_genes = 0 :: non_neg_integer(),
    constitutive_genes = 0 :: non_neg_integer(),

    %% Fitness
    fitness_with_regulation = 0.0 :: float(),
    fitness_without_regulation = 0.0 :: float()
}).

-type regulatory_context() :: #regulatory_context{}.

%%====================================================================
%% State Record (Silo Internal State)
%%====================================================================

-record(regulatory_state, {
    %% Configuration
    config :: regulatory_config(),

    %% Current actuator outputs
    expression_threshold = 0.5 :: float(),
    regulatory_mutation_rate = 0.02 :: float(),
    context_sensitivity = 0.5 :: float(),
    module_switching_cost = 0.1 :: float(),
    dormancy_maintenance_cost = 0.01 :: float(),
    epigenetic_inheritance_strength = 0.5 :: float(),
    constitutive_expression_bonus = 0.05 :: float(),
    regulatory_complexity_penalty = 0.02 :: float(),

    %% Gene registry
    genes = #{} :: #{gene_id() => gene()},
    modules = #{} :: #{module_id() => module()},

    %% Current context
    current_context :: context() | undefined,
    context_history = [] :: [context()],

    %% Tracking
    current_generation = 0 :: generation(),
    total_context_switches = 0 :: non_neg_integer(),
    genes_expressed_history = [] :: [{generation(), non_neg_integer()}],

    %% L2 integration
    l2_enabled = false :: boolean(),
    l2_guidance = undefined :: l2_guidance() | undefined
}).

-type regulatory_state() :: #regulatory_state{}.

%%====================================================================
%% Configuration Record
%%====================================================================

-record(regulatory_config, {
    %% Enable/disable
    enabled = true :: boolean(),

    %% Gene configuration
    max_genes = 1000 :: pos_integer(),
    max_modules = 50 :: pos_integer(),

    %% Epigenetics
    enable_epigenetics = true :: boolean(),
    max_marks_per_gene = 10 :: pos_integer(),

    %% Context
    available_contexts = [default] :: [context_id()],

    %% Event emission
    emit_events = true :: boolean()
}).

-type regulatory_config() :: #regulatory_config{}.

%%====================================================================
%% L2 Guidance Record
%%====================================================================

-record(l2_guidance, {
    %% Expression pressure
    expression_pressure = 0.5 :: float(),

    %% Context adaptation
    context_adaptation = 0.5 :: float(),

    %% Epigenetic emphasis
    epigenetic_emphasis = 0.5 :: float(),

    %% Dormancy pressure
    dormancy_pressure = 0.5 :: float()
}).

-type l2_guidance() :: #l2_guidance{}.

%%====================================================================
%% Constants
%%====================================================================

-define(DEFAULT_EXPRESSION_THRESHOLD, 0.5).
-define(DEFAULT_CONTEXT, default).
-define(MAX_CONTEXT_HISTORY, 100).

-endif.
```

---

## 6. Core Silo Implementation

```erlang
%%%-------------------------------------------------------------------
%%% @doc Regulatory Silo
%%% Manages gene expression and network activation for context-dependent
%%% behavior in neuroevolution.
%%% @end
%%%-------------------------------------------------------------------
-module(regulatory_silo).

-behaviour(gen_server).

%% API
-export([start_link/1,
         get_regulatory_params/1,
         update_context/2,
         set_context/2,
         express_gene/2,
         silence_gene/2,
         activate_module/2,
         deactivate_module/2,
         get_expressed_genes/1,
         get_active_modules/1,
         add_epigenetic_mark/3,
         inherit_marks/3,
         get_state/1,
         enable/1,
         disable/1,
         is_enabled/1]).

%% Cross-silo signals
-export([signal_expression_flexibility/1,
         signal_dormant_potential/1,
         signal_context_awareness/1,
         receive_environmental_context/2,
         receive_stress_level/2,
         receive_task_complexity/2]).

%% gen_server callbacks
-export([init/1, handle_call/3, handle_cast/2, handle_info/2,
         terminate/2, code_change/3]).

-include("regulatory_silo.hrl").

%%====================================================================
%% API
%%====================================================================

-spec start_link(regulatory_config()) -> {ok, pid()} | {error, term()}.
start_link(Config) ->
    gen_server:start_link({local, ?MODULE}, ?MODULE, Config, []).

-spec get_regulatory_params(pid()) -> map().
get_regulatory_params(Pid) ->
    gen_server:call(Pid, get_regulatory_params).

-spec update_context(pid(), regulatory_context()) -> ok.
update_context(Pid, Context) ->
    gen_server:cast(Pid, {update_context, Context}).

-spec set_context(pid(), context_id()) -> ok.
set_context(Pid, ContextId) ->
    gen_server:cast(Pid, {set_context, ContextId}).

-spec express_gene(pid(), gene_id()) -> ok | {error, term()}.
express_gene(Pid, GeneId) ->
    gen_server:call(Pid, {express_gene, GeneId}).

-spec silence_gene(pid(), gene_id()) -> ok | {error, term()}.
silence_gene(Pid, GeneId) ->
    gen_server:call(Pid, {silence_gene, GeneId}).

-spec activate_module(pid(), module_id()) -> ok | {error, term()}.
activate_module(Pid, ModuleId) ->
    gen_server:call(Pid, {activate_module, ModuleId}).

-spec deactivate_module(pid(), module_id()) -> ok | {error, term()}.
deactivate_module(Pid, ModuleId) ->
    gen_server:call(Pid, {deactivate_module, ModuleId}).

-spec get_expressed_genes(pid()) -> [gene_id()].
get_expressed_genes(Pid) ->
    gen_server:call(Pid, get_expressed_genes).

-spec get_active_modules(pid()) -> [module_id()].
get_active_modules(Pid) ->
    gen_server:call(Pid, get_active_modules).

-spec add_epigenetic_mark(pid(), gene_id(), epigenetic_mark()) -> ok | {error, term()}.
add_epigenetic_mark(Pid, GeneId, Mark) ->
    gen_server:call(Pid, {add_epigenetic_mark, GeneId, Mark}).

-spec inherit_marks(pid(), gene_id(), gene_id()) -> ok.
inherit_marks(Pid, ParentGeneId, ChildGeneId) ->
    gen_server:cast(Pid, {inherit_marks, ParentGeneId, ChildGeneId}).

-spec get_state(pid()) -> regulatory_state().
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

%% @doc Get expression flexibility for cultural silo
-spec signal_expression_flexibility(pid()) -> float().
signal_expression_flexibility(Pid) ->
    gen_server:call(Pid, signal_expression_flexibility).

%% @doc Get dormant potential for competitive silo
-spec signal_dormant_potential(pid()) -> float().
signal_dormant_potential(Pid) ->
    gen_server:call(Pid, signal_dormant_potential).

%% @doc Get context awareness for task silo
-spec signal_context_awareness(pid()) -> float().
signal_context_awareness(Pid) ->
    gen_server:call(Pid, signal_context_awareness).

%% @doc Receive environmental context from ecological silo
-spec receive_environmental_context(pid(), context_id()) -> ok.
receive_environmental_context(Pid, Context) ->
    gen_server:cast(Pid, {cross_silo, environmental_context, Context}).

%% @doc Receive stress level from ecological silo
-spec receive_stress_level(pid(), float()) -> ok.
receive_stress_level(Pid, Stress) ->
    gen_server:cast(Pid, {cross_silo, stress_level, Stress}).

%% @doc Receive task complexity from task silo
-spec receive_task_complexity(pid(), float()) -> ok.
receive_task_complexity(Pid, Complexity) ->
    gen_server:cast(Pid, {cross_silo, task_complexity, Complexity}).

%%====================================================================
%% gen_server Callbacks
%%====================================================================

init(Config) ->
    State = #regulatory_state{
        config = Config,
        current_context = #context{context_id = ?DEFAULT_CONTEXT}
    },
    {ok, State}.

handle_call(get_regulatory_params, _From, State) ->
    Params = #{
        expression_threshold => State#regulatory_state.expression_threshold,
        regulatory_mutation_rate => State#regulatory_state.regulatory_mutation_rate,
        context_sensitivity => State#regulatory_state.context_sensitivity,
        module_switching_cost => State#regulatory_state.module_switching_cost,
        dormancy_maintenance_cost => State#regulatory_state.dormancy_maintenance_cost,
        epigenetic_inheritance_strength => State#regulatory_state.epigenetic_inheritance_strength,
        constitutive_expression_bonus => State#regulatory_state.constitutive_expression_bonus,
        regulatory_complexity_penalty => State#regulatory_state.regulatory_complexity_penalty
    },
    {reply, Params, State};

handle_call({express_gene, GeneId}, _From, State) ->
    case maps:get(GeneId, State#regulatory_state.genes, undefined) of
        undefined ->
            {reply, {error, not_found}, State};
        Gene ->
            NewGene = Gene#gene{expressed = true, expression_level = 1.0},
            NewGenes = maps:put(GeneId, NewGene, State#regulatory_state.genes),

            maybe_emit_event(gene_expressed, #{
                gene_id => GeneId,
                trigger => manual
            }, State),

            {reply, ok, State#regulatory_state{genes = NewGenes}}
    end;

handle_call({silence_gene, GeneId}, _From, State) ->
    case maps:get(GeneId, State#regulatory_state.genes, undefined) of
        undefined ->
            {reply, {error, not_found}, State};
        Gene ->
            NewGene = Gene#gene{expressed = false, expression_level = 0.0},
            NewGenes = maps:put(GeneId, NewGene, State#regulatory_state.genes),

            maybe_emit_event(gene_silenced, #{
                gene_id => GeneId,
                reason => manual
            }, State),

            {reply, ok, State#regulatory_state{genes = NewGenes}}
    end;

handle_call({activate_module, ModuleId}, _From, State) ->
    case maps:get(ModuleId, State#regulatory_state.modules, undefined) of
        undefined ->
            {reply, {error, not_found}, State};
        Module ->
            %% Activate module and express its genes
            NewModule = Module#module{is_active = true},
            NewModules = maps:put(ModuleId, NewModule, State#regulatory_state.modules),

            %% Express genes in module
            NewGenes = lists:foldl(fun(GeneId, Acc) ->
                case maps:get(GeneId, Acc, undefined) of
                    undefined -> Acc;
                    Gene ->
                        maps:put(GeneId, Gene#gene{expressed = true}, Acc)
                end
            end, State#regulatory_state.genes, Module#module.gene_ids),

            maybe_emit_event(module_activated, #{
                module_id => ModuleId,
                context => State#regulatory_state.current_context#context.context_id
            }, State),

            {reply, ok, State#regulatory_state{
                modules = NewModules,
                genes = NewGenes
            }}
    end;

handle_call({deactivate_module, ModuleId}, _From, State) ->
    case maps:get(ModuleId, State#regulatory_state.modules, undefined) of
        undefined ->
            {reply, {error, not_found}, State};
        Module ->
            NewModule = Module#module{is_active = false},
            NewModules = maps:put(ModuleId, NewModule, State#regulatory_state.modules),

            %% Silence genes in module (unless constitutive)
            NewGenes = lists:foldl(fun(GeneId, Acc) ->
                case maps:get(GeneId, Acc, undefined) of
                    undefined -> Acc;
                    Gene when Gene#gene.is_constitutive -> Acc;
                    Gene ->
                        maps:put(GeneId, Gene#gene{expressed = false}, Acc)
                end
            end, State#regulatory_state.genes, Module#module.gene_ids),

            maybe_emit_event(module_deactivated, #{
                module_id => ModuleId
            }, State),

            {reply, ok, State#regulatory_state{
                modules = NewModules,
                genes = NewGenes
            }}
    end;

handle_call(get_expressed_genes, _From, State) ->
    Expressed = [GeneId || {GeneId, Gene} <- maps:to_list(State#regulatory_state.genes),
                          Gene#gene.expressed],
    {reply, Expressed, State};

handle_call(get_active_modules, _From, State) ->
    Active = [ModuleId || {ModuleId, Module} <- maps:to_list(State#regulatory_state.modules),
                         Module#module.is_active],
    {reply, Active, State};

handle_call({add_epigenetic_mark, GeneId, Mark}, _From, State) ->
    case maps:get(GeneId, State#regulatory_state.genes, undefined) of
        undefined ->
            {reply, {error, not_found}, State};
        Gene ->
            #regulatory_config{max_marks_per_gene = MaxMarks} = State#regulatory_state.config,
            CurrentMarks = Gene#gene.epigenetic_marks,
            case length(CurrentMarks) >= MaxMarks of
                true ->
                    {reply, {error, max_marks_reached}, State};
                false ->
                    NewGene = Gene#gene{epigenetic_marks = [Mark | CurrentMarks]},
                    NewGenes = maps:put(GeneId, NewGene, State#regulatory_state.genes),

                    maybe_emit_event(epigenetic_mark_acquired, #{
                        gene_id => GeneId,
                        mark_type => Mark#epigenetic_mark.mark_type
                    }, State),

                    {reply, ok, State#regulatory_state{genes = NewGenes}}
            end
    end;

handle_call(get_state, _From, State) ->
    {reply, State, State};

handle_call(enable, _From, State) ->
    Config = State#regulatory_state.config,
    NewConfig = Config#regulatory_config{enabled = true},
    {reply, ok, State#regulatory_state{config = NewConfig}};

handle_call(disable, _From, State) ->
    Config = State#regulatory_state.config,
    NewConfig = Config#regulatory_config{enabled = false},
    {reply, ok, State#regulatory_state{config = NewConfig}};

handle_call(is_enabled, _From, State) ->
    {reply, State#regulatory_state.config#regulatory_config.enabled, State};

handle_call(signal_expression_flexibility, _From, State) ->
    %% Flexibility based on conditional vs constitutive ratio
    Genes = maps:values(State#regulatory_state.genes),
    case Genes of
        [] -> {reply, 0.5, State};
        _ ->
            Conditional = length([G || G <- Genes, not G#gene.is_constitutive]),
            Flexibility = Conditional / length(Genes),
            {reply, clamp(Flexibility, 0.0, 1.0), State}
    end;

handle_call(signal_dormant_potential, _From, State) ->
    %% Dormant genes as potential
    Genes = maps:values(State#regulatory_state.genes),
    case Genes of
        [] -> {reply, 0.0, State};
        _ ->
            Dormant = length([G || G <- Genes, G#gene.is_dormant]),
            Potential = Dormant / length(Genes),
            {reply, clamp(Potential, 0.0, 1.0), State}
    end;

handle_call(signal_context_awareness, _From, State) ->
    %% Context awareness based on sensitivity and switch frequency
    Sensitivity = State#regulatory_state.context_sensitivity,
    SwitchRate = case State#regulatory_state.current_generation of
        0 -> 0.0;
        Gen -> State#regulatory_state.total_context_switches / Gen
    end,
    Awareness = (Sensitivity + clamp(SwitchRate * 10, 0.0, 1.0)) / 2.0,
    {reply, Awareness, State};

handle_call(_Request, _From, State) ->
    {reply, {error, unknown_request}, State}.

handle_cast({update_context, Context}, State) ->
    #regulatory_state{config = Config} = State,
    case Config#regulatory_config.enabled of
        false ->
            {noreply, State};
        true ->
            %% Collect sensors
            SensorVector = regulatory_silo_sensors:collect_sensors(Context),

            %% Process through TWEANN controller
            ActuatorVector = process_through_controller(SensorVector, State),

            %% Apply actuators
            NewState = regulatory_silo_actuators:apply_actuators(ActuatorVector, State),

            %% Update generation
            FinalState = NewState#regulatory_state{
                current_generation = State#regulatory_state.current_generation + 1
            },

            {noreply, FinalState}
    end;

handle_cast({set_context, ContextId}, State) ->
    OldContext = State#regulatory_state.current_context,
    OldContextId = case OldContext of
        undefined -> undefined;
        _ -> OldContext#context.context_id
    end,

    case ContextId =/= OldContextId of
        false ->
            {noreply, State};
        true ->
            %% Context switch occurred
            NewContext = #context{
                context_id = ContextId,
                context_type = environmental,
                entry_generation = State#regulatory_state.current_generation
            },

            %% Update modules based on context
            NewModules = update_modules_for_context(ContextId, State),

            %% Update history
            NewHistory = [OldContext | lists:sublist(State#regulatory_state.context_history,
                                                    ?MAX_CONTEXT_HISTORY - 1)],

            maybe_emit_event(context_switch_occurred, #{
                old_context => OldContextId,
                new_context => ContextId
            }, State),

            {noreply, State#regulatory_state{
                current_context = NewContext,
                context_history = NewHistory,
                modules = NewModules,
                total_context_switches = State#regulatory_state.total_context_switches + 1
            }}
    end;

handle_cast({inherit_marks, ParentGeneId, ChildGeneId}, State) ->
    InheritStrength = State#regulatory_state.epigenetic_inheritance_strength,

    case {maps:get(ParentGeneId, State#regulatory_state.genes, undefined),
          maps:get(ChildGeneId, State#regulatory_state.genes, undefined)} of
        {undefined, _} -> {noreply, State};
        {_, undefined} -> {noreply, State};
        {ParentGene, ChildGene} ->
            ParentMarks = ParentGene#gene.epigenetic_marks,

            %% Inherit marks probabilistically
            InheritedMarks = lists:filter(fun(_) ->
                rand:uniform() < InheritStrength
            end, ParentMarks),

            %% Update mark generation
            UpdatedMarks = [M#epigenetic_mark{
                inherited_from = State#regulatory_state.current_generation
            } || M <- InheritedMarks],

            NewChildGene = ChildGene#gene{
                epigenetic_marks = ChildGene#gene.epigenetic_marks ++ UpdatedMarks
            },

            NewGenes = maps:put(ChildGeneId, NewChildGene, State#regulatory_state.genes),
            {noreply, State#regulatory_state{genes = NewGenes}}
    end;

handle_cast({cross_silo, environmental_context, ContextId}, State) ->
    %% Environmental context change
    gen_server:cast(self(), {set_context, ContextId}),
    {noreply, State};

handle_cast({cross_silo, stress_level, Stress}, State) ->
    %% High stress may activate dormant genes
    case Stress > 0.7 of
        true ->
            %% Activate some dormant genes
            NewGenes = activate_dormant_genes(State#regulatory_state.genes, Stress),
            {noreply, State#regulatory_state{genes = NewGenes}};
        false ->
            {noreply, State}
    end;

handle_cast({cross_silo, task_complexity, Complexity}, State) ->
    %% Complex tasks need more modules active
    case Complexity > 0.7 of
        true ->
            %% Lower expression threshold to activate more
            NewThreshold = max(0.2, State#regulatory_state.expression_threshold * 0.8),
            {noreply, State#regulatory_state{expression_threshold = NewThreshold}};
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

update_modules_for_context(ContextId, State) ->
    maps:map(fun(_ModuleId, Module) ->
        Preferred = lists:member(ContextId, Module#module.preferred_contexts),
        Excluded = lists:member(ContextId, Module#module.excluded_contexts),

        case {Preferred, Excluded} of
            {true, false} -> Module#module{is_active = true};
            {false, true} -> Module#module{is_active = false};
            _ -> Module  % Keep current state
        end
    end, State#regulatory_state.modules).

activate_dormant_genes(Genes, Stress) ->
    maps:map(fun(_GeneId, Gene) ->
        case Gene#gene.is_dormant andalso rand:uniform() < Stress * 0.3 of
            true ->
                Gene#gene{expressed = true, is_dormant = false};
            false ->
                Gene
        end
    end, Genes).

process_through_controller(SensorVector, State) ->
    case State#regulatory_state.l2_enabled of
        true ->
            apply_l2_guidance(SensorVector, State);
        false ->
            lists:duplicate(regulatory_silo_actuators:actuator_count(), 0.5)
    end.

apply_l2_guidance(_SensorVector, State) ->
    case State#regulatory_state.l2_guidance of
        undefined ->
            lists:duplicate(regulatory_silo_actuators:actuator_count(), 0.5);
        #l2_guidance{
            expression_pressure = ExprPress,
            context_adaptation = CtxAdapt,
            epigenetic_emphasis = EpiEmph,
            dormancy_pressure = DormPress
        } ->
            [
                1.0 - ExprPress,    % expression_threshold (inverse)
                0.02,               % regulatory_mutation_rate
                CtxAdapt,           % context_sensitivity
                0.1,                % module_switching_cost
                DormPress * 0.1,    % dormancy_maintenance_cost
                EpiEmph,            % epigenetic_inheritance_strength
                0.05,               % constitutive_expression_bonus
                0.02                % regulatory_complexity_penalty
            ]
    end.

maybe_emit_event(EventType, Payload, State) ->
    case State#regulatory_state.config#regulatory_config.emit_events of
        true ->
            Event = #{
                type => EventType,
                silo => regulatory,
                timestamp => erlang:system_time(millisecond),
                generation => State#regulatory_state.current_generation,
                payload => Payload
            },
            event_bus:publish(regulatory_silo_events, Event);
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
| `expression_flexibility` | Cultural | float() | Flexible expression enables faster imitation |
| `dormant_potential` | Competitive | float() | Hidden capabilities for counter-strategies |
| `context_awareness` | Task | float() | Context switching affects learning |
| `expression_cost` | Morphological | float() | Gene expression affects network structure |
| `energy_requirement` | Economic | float() | Expression requires energy |

### 7.2 Incoming Signals

| Signal | From Silo | Effect |
|--------|-----------|--------|
| `environmental_context` | Ecological | Context determines expression |
| `stress_level` | Ecological | Stress triggers dormant gene activation |
| `task_complexity` | Task | Complex tasks need more modules |
| `developmental_stage` | Developmental | Stage affects expression patterns |
| `energy_available` | Economic | Energy limits expression |

### 7.3 Complete 13-Silo Matrix (Regulatory Row)

| To → | Task | Resource | Distrib | Social | Cultural | Ecological | Morpho | Temporal | Competitive | Develop | Regulatory | Economic | Comm |
|------|------|----------|---------|--------|----------|------------|--------|----------|-------------|---------|------------|----------|------|
| **From Regulatory** | context | - | - | - | flexibility | env_resp | expression | - | hidden | expression | - | energy | - |

---

## 8. Events Emitted

### 8.1 Event Specifications

| Event | Trigger | Payload |
|-------|---------|---------|
| `gene_expressed` | Gene turned on | `{gene_id, trigger}` |
| `gene_silenced` | Gene turned off | `{gene_id, reason}` |
| `module_activated` | Module switched on | `{module_id, context}` |
| `module_deactivated` | Module switched off | `{module_id}` |
| `context_switch_occurred` | Context changed | `{old_context, new_context}` |
| `dormant_capability_awakened` | Hidden gene expressed | `{gene_id, trigger}` |
| `epigenetic_mark_acquired` | Mark added | `{gene_id, mark_type}` |
| `regulatory_mutation` | Regulation changed | `{gene_id, old_regulation, new_regulation}` |

---

## 9. Value of Event Storage

- **Expression Patterns**: Learn which genes are useful in which contexts
- **Module Discovery**: Identify useful module combinations
- **Context Mapping**: Build context → expression mappings
- **Dormancy Value**: Understand value of maintaining dormant capabilities
- **Epigenetic Effects**: Track cross-generational regulatory inheritance

---

## 10. Enable/Disable Effects

### When Disabled
- All genes always expressed (wasteful)
- No context-dependent behavior
- No dormant capabilities
- No epigenetic inheritance

### When Enabled
- Genes expressed as needed
- Context-appropriate behavior
- Hidden capabilities emerge when needed
- Epigenetic marks inherited

---

## 11. Implementation Phases

### Phase 1: Core Infrastructure
- [ ] Create `regulatory_silo.hrl` with record definitions
- [ ] Implement `regulatory_silo_sensors.erl`
- [ ] Implement `regulatory_silo_actuators.erl`
- [ ] Basic `regulatory_silo.erl` gen_server

### Phase 2: Gene Expression
- [ ] Gene registry and management
- [ ] Expression threshold logic
- [ ] Context-dependent expression
- [ ] Module activation system

### Phase 3: Epigenetics
- [ ] Epigenetic mark system
- [ ] Mark inheritance
- [ ] Mark effects on expression
- [ ] Cross-generation tracking

### Phase 4: Cross-Silo Integration
- [ ] Outgoing signal implementations
- [ ] Incoming signal handlers
- [ ] Event emission
- [ ] Integration tests

---

## 12. Success Criteria

1. **Context Switching**: Modules switch correctly on context change
2. **Dormant Activation**: Dormant genes activate under stress
3. **Epigenetic Inheritance**: Marks inherit with configured strength
4. **Energy Efficiency**: Active genes < 50% in specialized contexts
5. **Multi-modal Behavior**: Same network shows different behaviors in different contexts
