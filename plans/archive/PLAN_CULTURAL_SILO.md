# PLAN_CULTURAL_SILO.md

**Status:** Planning
**Created:** 2025-12-23
**Last Updated:** 2025-12-23
**Related:** PLAN_BEHAVIORAL_EVENTS.md, PLAN_KNOWLEDGE_TRANSFER.md, PLAN_SOCIAL_SILO.md

---

## Overview

This document specifies the **Cultural Silo** for the Liquid Conglomerate (LC) meta-controller. The Cultural Silo manages the evolution of learned behaviors, traditions, innovations, and knowledge transmission within populations - distinct from genetic evolution.

### Purpose

Cultural evolution operates on a faster timescale than genetic evolution:

1. **Innovation Discovery**: Novel beneficial behaviors emerge
2. **Tradition Maintenance**: Successful patterns persist across generations
3. **Imitation & Learning**: Behaviors spread through observation
4. **Cultural Drift**: Gradual change in population-level behaviors
5. **Cumulative Culture**: Innovations build on previous innovations

### Distinction from Genetic Evolution

| Aspect | Genetic | Cultural |
|--------|---------|----------|
| **Inheritance** | Parent → Offspring | Any → Any (horizontal) |
| **Timescale** | Generations | Within generations |
| **Mechanism** | DNA replication | Observation/teaching |
| **Fidelity** | High (mutations rare) | Variable (imperfect copying) |
| **Accumulation** | Slow | Rapid (ratchet effect) |

### Training Velocity & Inference Impact

| Metric | Without Cultural Silo | With Cultural Silo |
|--------|----------------------|-------------------|
| Knowledge transfer | Genetic only (slow) | Horizontal + vertical (fast) |
| Training velocity | Baseline (1.0x) | Improved (1.2-1.5x) |
| Inference latency | No imitation overhead | +1-3ms for tradition lookup |
| Convergence speed | Slow (genetic only) | Fast (cultural ratchet) |
| Innovation preservation | Lost on death | Persists in traditions |

**Note:** Training velocity is **improved** because cultural learning allows rapid propagation of successful behaviors without waiting for genetic selection. The "cultural ratchet" effect means innovations accumulate faster than genetic mutations could achieve alone.

---

## Silo Architecture

### Cultural Silo Structure

```
┌─────────────────────────────────────────────────────────────────────┐
│                        CULTURAL SILO                                 │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │                      L0 SENSORS (10)                         │    │
│  │                                                              │    │
│  │  Innovation       Tradition    Learning     Diversity        │    │
│  │  ┌─────────┐     ┌─────────┐  ┌─────────┐  ┌─────────┐      │    │
│  │  │innov_   │     │tradition│  │imitation│  │cultural_│      │    │
│  │  │rate     │     │_count   │  │_rate    │  │diversity│      │    │
│  │  │innov_   │     │trad_    │  │teaching_│  │behavior_│      │    │
│  │  │spread   │     │strength │  │success  │  │clusters │      │    │
│  │  └─────────┘     └─────────┘  └─────────┘  └─────────┘      │    │
│  │                                                              │    │
│  │  Cumulative                   Fads                          │    │
│  │  ┌─────────┐     ┌─────────┐                                │    │
│  │  │chain_   │     │fad_count│                                │    │
│  │  │depth    │     │fad_     │                                │    │
│  │  └─────────┘     │velocity │                                │    │
│  │                  └─────────┘                                │    │
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
│  │  Innovation       Tradition    Learning     Conformity      │    │
│  │  ┌─────────┐     ┌─────────┐  ┌─────────┐  ┌─────────┐      │    │
│  │  │innov_   │     │trad_    │  │imitation│  │conformity│     │    │
│  │  │threshold│     │decay    │  │fidelity │  │pressure  │     │    │
│  │  │innov_   │     │trad_    │  │learning_│  │deviance_ │     │    │
│  │  │bonus    │     │boost    │  │rate     │  │tolerance │     │    │
│  │  └─────────┘     └─────────┘  └─────────┘  └─────────┘      │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Core Concepts

### Cultural Trait

```erlang
-record(cultural_trait, {
    id :: binary(),
    name :: binary(),
    description :: binary(),

    %% What this trait represents
    trait_type :: behavioral | strategic | technical | social,
    domain :: navigation | combat | foraging | communication | other,

    %% How it manifests in the network
    network_signature :: #{
        activation_pattern => [float()],    % Characteristic output pattern
        weight_cluster => [binary()],       % Associated connection IDs
        structural_motif => term()          % Network structure pattern
    },

    %% Tracking
    discovered_by :: binary(),             % Original innovator
    discovered_at :: integer(),
    current_prevalence :: float(),         % 0.0-1.0 (% of population)
    peak_prevalence :: float(),
    status :: emerging | spreading | established | declining | extinct,

    %% Lineage
    parent_traits :: [binary()],           % Traits this built upon
    child_traits :: [binary()],            % Traits derived from this
    innovation_depth :: non_neg_integer()  % Distance from base behaviors
}).
```

### Tradition

```erlang
-record(tradition, {
    id :: binary(),
    trait_id :: binary(),                  % The underlying cultural trait

    %% Tradition properties
    establishment_threshold :: float(),    % Prevalence needed to become tradition
    strength :: float(),                   % How strongly enforced
    age_generations :: non_neg_integer(),

    %% Practitioners
    practitioners :: [binary()],
    practitioner_count :: non_neg_integer(),

    %% Enforcement
    norm_associated :: binary() | none,    % Associated social norm (if any)
    violation_penalty :: float(),

    %% Status
    status :: emerging | established | declining | abandoned
}).
```

### Innovation

```erlang
-record(innovation, {
    id :: binary(),
    innovator_id :: binary(),
    trait_id :: binary(),                  % Created trait

    %% Innovation properties
    novelty_score :: float(),              % How different from existing
    fitness_advantage :: float(),          % Performance improvement
    complexity :: float(),                 % Network complexity delta

    %% Spread tracking
    adopters :: [binary()],
    adoption_rate :: float(),              % Adoptions per generation
    spread_pattern :: viral | slow | niche,

    %% Timing
    emerged_at :: integer(),
    spread_started_at :: integer() | not_yet
}).
```

---

## L0 Sensors

### Sensor Specifications

```erlang
-module(cultural_silo_sensors).
-behaviour(l0_sensor_behaviour).

-export([sensor_specs/0, read_sensors/1]).

sensor_specs() ->
    [
        %% Innovation sensors
        #{
            id => innovation_rate,
            range => {0.0, 1.0},
            description => "Rate of new innovations per generation"
        },
        #{
            id => innovation_spread_rate,
            range => {0.0, 1.0},
            description => "How fast innovations spread"
        },

        %% Tradition sensors
        #{
            id => tradition_count,
            range => {0.0, 1.0},
            description => "Number of active traditions (normalized)"
        },
        #{
            id => tradition_strength,
            range => {0.0, 1.0},
            description => "Average strength of traditions"
        },

        %% Learning sensors
        #{
            id => imitation_rate,
            range => {0.0, 1.0},
            description => "Rate of successful imitation events"
        },
        #{
            id => teaching_success_rate,
            range => {0.0, 1.0},
            description => "Success rate of teaching attempts"
        },

        %% Diversity sensors
        #{
            id => cultural_diversity,
            range => {0.0, 1.0},
            description => "Diversity of behaviors in population"
        },
        #{
            id => behavior_cluster_count,
            range => {0.0, 1.0},
            description => "Number of distinct behavioral clusters"
        },

        %% Cumulative culture
        #{
            id => innovation_chain_depth,
            range => {0.0, 1.0},
            description => "Max depth of innovation dependency chains"
        },

        %% Fads
        #{
            id => fad_velocity,
            range => {0.0, 1.0},
            description => "Speed of fad rise and fall"
        }
    ].

read_sensors(#cultural_state{} = State) ->
    [
        State#cultural_state.innovation_rate,
        State#cultural_state.innovation_spread_rate,
        normalize_count(State#cultural_state.tradition_count, 20),
        State#cultural_state.mean_tradition_strength,
        State#cultural_state.imitation_rate,
        State#cultural_state.teaching_success_rate,
        State#cultural_state.cultural_diversity,
        normalize_count(State#cultural_state.behavior_clusters, 10),
        normalize_count(State#cultural_state.max_chain_depth, 10),
        State#cultural_state.fad_velocity
    ].
```

---

## L0 Actuators

### Actuator Specifications

```erlang
-module(cultural_silo_actuators).
-behaviour(l0_actuator_behaviour).

-export([actuator_specs/0, apply_actuators/2]).

actuator_specs() ->
    [
        %% Innovation actuators
        #{
            id => innovation_threshold,
            range => {0.1, 0.9},
            default => 0.5,
            description => "Novelty threshold to count as innovation"
        },
        #{
            id => innovation_fitness_bonus,
            range => {0.0, 0.5},
            default => 0.2,
            description => "Fitness bonus for innovators"
        },

        %% Tradition actuators
        #{
            id => tradition_decay_rate,
            range => {0.0, 0.3},
            default => 0.05,
            description => "Rate at which traditions weaken"
        },
        #{
            id => tradition_establishment_boost,
            range => {1.0, 2.0},
            default => 1.2,
            description => "Fitness multiplier for tradition practitioners"
        },

        %% Learning actuators
        #{
            id => imitation_fidelity,
            range => {0.5, 1.0},
            default => 0.8,
            description => "Accuracy of behavior copying"
        },
        #{
            id => learning_rate_multiplier,
            range => {0.5, 2.0},
            default => 1.0,
            description => "Speed of cultural learning"
        },

        %% Conformity actuators
        #{
            id => conformity_pressure,
            range => {0.0, 1.0},
            default => 0.3,
            description => "Pressure to adopt common behaviors"
        },
        #{
            id => deviance_tolerance,
            range => {0.0, 1.0},
            default => 0.5,
            description => "Tolerance for non-conforming behaviors"
        }
    ].

apply_actuators(Outputs, #cultural_config{} = Config) ->
    [InnovThresh, InnovBonus, TradDecay, TradBoost,
     ImitFidelity, LearnRate, ConformPress, DevTol] = Outputs,

    Config#cultural_config{
        innovation_threshold = denormalize(innovation_threshold, InnovThresh),
        innovation_fitness_bonus = denormalize(innovation_fitness_bonus, InnovBonus),
        tradition_decay_rate = denormalize(tradition_decay_rate, TradDecay),
        tradition_establishment_boost = denormalize(tradition_establishment_boost, TradBoost),
        imitation_fidelity = denormalize(imitation_fidelity, ImitFidelity),
        learning_rate_multiplier = denormalize(learning_rate_multiplier, LearnRate),
        conformity_pressure = denormalize(conformity_pressure, ConformPress),
        deviance_tolerance = denormalize(deviance_tolerance, DevTol)
    }.
```

---

## Innovation Detection

### Detecting Novel Behaviors

```erlang
-module(cultural_innovation).

%% Detect if a behavior represents an innovation
-spec detect_innovation(Individual, Behavior, PopulationBehaviors, Config) ->
    {innovation, #innovation{}} | not_innovative
    when Individual :: #individual{},
         Behavior :: #behavioral_signature{},
         PopulationBehaviors :: [#behavioral_signature{}],
         Config :: #cultural_config{}.

detect_innovation(Individual, Behavior, PopulationBehaviors, Config) ->
    %% Calculate novelty score
    NoveltyScore = calculate_behavioral_novelty(Behavior, PopulationBehaviors),

    %% Check fitness advantage
    FitnessAdvantage = Individual#individual.fitness -
                       get_population_mean_fitness(),

    %% Check complexity
    Complexity = calculate_behavior_complexity(Behavior),

    %% Is this an innovation?
    case NoveltyScore >= Config#cultural_config.innovation_threshold of
        true when FitnessAdvantage > 0 ->
            %% Create innovation record
            TraitId = generate_trait_id(),
            Trait = create_cultural_trait(Individual, Behavior, NoveltyScore),
            Innovation = #innovation{
                id = generate_innovation_id(),
                innovator_id = Individual#individual.id,
                trait_id = TraitId,
                novelty_score = NoveltyScore,
                fitness_advantage = FitnessAdvantage,
                complexity = Complexity,
                adopters = [Individual#individual.id],
                adoption_rate = 0.0,
                spread_pattern = determine_spread_pattern(NoveltyScore, FitnessAdvantage),
                emerged_at = erlang:system_time(microsecond),
                spread_started_at = not_yet
            },

            %% Store trait and innovation
            store_cultural_trait(Trait),
            store_innovation(Innovation),

            %% Emit event
            neuroevolution_events:emit(innovation_emerged, #{
                event_type => innovation_emerged,
                innovator_id => Individual#individual.id,
                innovation_id => Innovation#innovation.id,
                population_id => Individual#individual.population_id,
                innovation_type => Trait#cultural_trait.trait_type,
                novelty_score => NoveltyScore,
                fitness_advantage => FitnessAdvantage,
                emerged_at => Innovation#innovation.emerged_at
            }),

            {innovation, Innovation};

        _ ->
            not_innovative
    end.

calculate_behavioral_novelty(Behavior, PopulationBehaviors) ->
    %% Calculate minimum distance to any existing behavior
    Distances = lists:map(fun(OtherBehavior) ->
        behavioral_distance(Behavior, OtherBehavior)
    end, PopulationBehaviors),

    case Distances of
        [] -> 1.0;  % First behavior = maximally novel
        _ -> lists:min(Distances)
    end.

behavioral_distance(BehaviorA, BehaviorB) ->
    %% Compare activation patterns
    PatternA = BehaviorA#behavioral_signature.activation_pattern,
    PatternB = BehaviorB#behavioral_signature.activation_pattern,

    %% Euclidean distance normalized to 0-1
    SumSquares = lists:sum([
        math:pow(A - B, 2) || {A, B} <- lists:zip(PatternA, PatternB)
    ]),
    math:sqrt(SumSquares) / math:sqrt(length(PatternA)).
```

---

## Tradition Management

### Tradition Formation

```erlang
-module(cultural_traditions).

%% Check if a trait should become a tradition
-spec check_tradition_formation(Trait, Config) -> {tradition, #tradition{}} | not_yet
    when Trait :: #cultural_trait{},
         Config :: #cultural_config{}.

check_tradition_formation(#cultural_trait{} = Trait, Config) ->
    EstablishmentThreshold = Config#cultural_config.tradition_establishment_threshold,

    case Trait#cultural_trait.current_prevalence >= EstablishmentThreshold of
        true when Trait#cultural_trait.status == spreading ->
            %% Trait becomes a tradition
            Tradition = #tradition{
                id = generate_tradition_id(),
                trait_id = Trait#cultural_trait.id,
                establishment_threshold = EstablishmentThreshold,
                strength = Trait#cultural_trait.current_prevalence,
                age_generations = 0,
                practitioners = get_trait_practitioners(Trait#cultural_trait.id),
                practitioner_count = count_practitioners(Trait#cultural_trait.id),
                norm_associated = none,
                violation_penalty = 0.0,
                status = emerging
            },

            store_tradition(Tradition),

            %% Update trait status
            UpdatedTrait = Trait#cultural_trait{status = established},
            store_cultural_trait(UpdatedTrait),

            %% Emit event
            neuroevolution_events:emit(tradition_established, #{
                event_type => tradition_established,
                tradition_id => Tradition#tradition.id,
                trait_id => Trait#cultural_trait.id,
                prevalence => Trait#cultural_trait.current_prevalence,
                practitioner_count => Tradition#tradition.practitioner_count,
                established_at => erlang:system_time(microsecond)
            }),

            {tradition, Tradition};

        _ ->
            not_yet
    end.

%% Decay traditions over time
decay_traditions(Traditions, Config) ->
    DecayRate = Config#cultural_config.tradition_decay_rate,

    lists:map(fun(#tradition{} = Trad) ->
        %% Apply decay
        NewStrength = Trad#tradition.strength * (1.0 - DecayRate),

        %% Check for abandonment
        case NewStrength < 0.1 of
            true ->
                abandon_tradition(Trad);
            false ->
                Trad#tradition{
                    strength = NewStrength,
                    age_generations = Trad#tradition.age_generations + 1
                }
        end
    end, Traditions).

abandon_tradition(#tradition{} = Trad) ->
    neuroevolution_events:emit(tradition_abandoned, #{
        event_type => tradition_abandoned,
        tradition_id => Trad#tradition.id,
        final_strength => Trad#tradition.strength,
        lifespan_generations => Trad#tradition.age_generations,
        abandoned_at => erlang:system_time(microsecond)
    }),

    Trad#tradition{status = abandoned}.
```

---

## Imitation and Learning

### Imitation Process

```erlang
-module(cultural_imitation).

%% Attempt to imitate a behavior from another individual
-spec imitate(Observer, Target, Config) -> Result
    when Observer :: #individual{},
         Target :: #individual{},
         Config :: #cultural_config{},
         Result :: {success, #cultural_trait{}} | {failure, reason()}.

imitate(Observer, Target, Config) ->
    %% Get target's behavioral signature
    TargetBehavior = extract_behavioral_signature(Target),

    %% Calculate imitation success probability
    %% Factors: fidelity setting, observer learning capacity, target clarity
    BaseFidelity = Config#cultural_config.imitation_fidelity,
    ObserverCapacity = get_learning_capacity(Observer),
    TargetClarity = calculate_behavior_clarity(TargetBehavior),

    SuccessProbability = BaseFidelity * ObserverCapacity * TargetClarity,

    case rand:uniform() < SuccessProbability of
        true ->
            %% Successful imitation
            %% Copy behavior with some noise
            NoiseFactor = 1.0 - BaseFidelity,
            CopiedBehavior = copy_behavior_with_noise(TargetBehavior, NoiseFactor),

            %% Apply to observer's network
            apply_behavioral_modification(Observer, CopiedBehavior),

            %% Find or create trait
            Trait = get_or_create_trait_for_behavior(CopiedBehavior),

            %% Update trait prevalence
            add_practitioner(Trait#cultural_trait.id, Observer#individual.id),

            neuroevolution_events:emit(skill_imitated, #{
                event_type => skill_imitated,
                observer_id => Observer#individual.id,
                target_id => Target#individual.id,
                trait_id => Trait#cultural_trait.id,
                fidelity => SuccessProbability,
                imitated_at => erlang:system_time(microsecond)
            }),

            {success, Trait};

        false ->
            %% Failed imitation
            neuroevolution_events:emit(learning_failed, #{
                event_type => learning_failed,
                observer_id => Observer#individual.id,
                target_id => Target#individual.id,
                failure_reason => imitation_failure,
                failed_at => erlang:system_time(microsecond)
            }),

            {failure, imitation_failure}
    end.

copy_behavior_with_noise(#behavioral_signature{} = Behavior, NoiseFactor) ->
    %% Add Gaussian noise to activation pattern
    NoisyPattern = lists:map(fun(Value) ->
        Noise = rand:normal() * NoiseFactor * 0.1,
        clamp(Value + Noise, -1.0, 1.0)
    end, Behavior#behavioral_signature.activation_pattern),

    Behavior#behavioral_signature{activation_pattern = NoisyPattern}.
```

### Cumulative Culture

```erlang
%% Track innovation chains (innovations building on innovations)
-spec register_innovation_dependency(NewInnovation, ParentInnovations) -> ok
    when NewInnovation :: #innovation{},
         ParentInnovations :: [#innovation{}].

register_innovation_dependency(NewInnovation, ParentInnovations) ->
    NewTrait = get_cultural_trait(NewInnovation#innovation.trait_id),

    %% Link to parent traits
    ParentTraitIds = [PI#innovation.trait_id || PI <- ParentInnovations],
    UpdatedTrait = NewTrait#cultural_trait{
        parent_traits = ParentTraitIds,
        innovation_depth = max_innovation_depth(ParentInnovations) + 1
    },

    %% Update parent traits with child reference
    lists:foreach(fun(ParentId) ->
        ParentTrait = get_cultural_trait(ParentId),
        UpdatedParent = ParentTrait#cultural_trait{
            child_traits = [NewTrait#cultural_trait.id |
                           ParentTrait#cultural_trait.child_traits]
        },
        store_cultural_trait(UpdatedParent)
    end, ParentTraitIds),

    store_cultural_trait(UpdatedTrait),

    %% Emit cumulative culture event if depth > 1
    case UpdatedTrait#cultural_trait.innovation_depth > 1 of
        true ->
            neuroevolution_events:emit(cumulative_innovation, #{
                event_type => cumulative_innovation,
                innovation_id => NewInnovation#innovation.id,
                parent_innovations => [PI#innovation.id || PI <- ParentInnovations],
                chain_depth => UpdatedTrait#cultural_trait.innovation_depth,
                recorded_at => erlang:system_time(microsecond)
            });
        false ->
            ok
    end.

max_innovation_depth(Innovations) ->
    case Innovations of
        [] -> 0;
        _ ->
            Depths = [get_cultural_trait(I#innovation.trait_id)#cultural_trait.innovation_depth
                     || I <- Innovations],
            lists:max(Depths)
    end.
```

---

## Cultural Drift

### Population-Level Behavioral Change

```erlang
-module(cultural_drift).

%% Process gradual cultural change
-spec process_cultural_drift(Population, Config) -> DriftReport
    when Population :: #population{},
         Config :: #cultural_config{},
         DriftReport :: #{atom() => term()}.

process_cultural_drift(Population, Config) ->
    %% Get current behavioral distribution
    CurrentDistribution = calculate_behavioral_distribution(Population),
    PreviousDistribution = get_previous_distribution(Population#population.id),

    %% Calculate drift metrics
    DriftMagnitude = calculate_distribution_shift(
        CurrentDistribution, PreviousDistribution),

    DriftDirection = calculate_drift_direction(
        CurrentDistribution, PreviousDistribution),

    %% Store new distribution
    store_distribution(Population#population.id, CurrentDistribution),

    %% Check for significant drift
    case DriftMagnitude > Config#cultural_config.drift_significance_threshold of
        true ->
            neuroevolution_events:emit(cultural_drift, #{
                event_type => cultural_drift,
                population_id => Population#population.id,
                drift_magnitude => DriftMagnitude,
                drift_direction => DriftDirection,
                generation => Population#population.generation,
                drifted_at => erlang:system_time(microsecond)
            });
        false ->
            ok
    end,

    #{
        drift_magnitude => DriftMagnitude,
        drift_direction => DriftDirection,
        distribution => CurrentDistribution
    }.
```

---

## Cultural Silo Server

### Module: `cultural_silo.erl`

```erlang
-module(cultural_silo).
-behaviour(gen_server).
-behaviour(lc_silo_behaviour).

%% API
-export([
    start_link/1,
    read_sensors/1,
    apply_actuators/2,
    detect_innovations/2,
    process_imitations/2,
    get_traditions/1,
    get_state/1
]).

-record(state, {
    population_id :: binary(),
    cultural_state :: #cultural_state{},
    config :: #cultural_config{},
    controller :: pid(),

    %% Cultural inventory
    traits :: #{binary() => #cultural_trait{}},
    traditions :: #{binary() => #tradition{}},
    innovations :: #{binary() => #innovation{}},

    %% Tracking
    behavioral_archive :: [#behavioral_signature{}],
    imitation_history :: [map()],
    drift_history :: [map()]
}).

%%====================================================================
%% Silo Behaviour Implementation
%%====================================================================

init_silo(Config) ->
    {ok, #state{
        population_id = maps:get(population_id, Config),
        cultural_state = initial_cultural_state(),
        config = default_cultural_config(),
        traits = #{},
        traditions = #{},
        innovations = #{},
        behavioral_archive = [],
        imitation_history = [],
        drift_history = []
    }}.

step(#state{} = State, Generation) ->
    %% 1. Read current cultural state
    SensorInputs = cultural_silo_sensors:read_sensors(State#state.cultural_state),

    %% 2. Get controller outputs
    ActuatorOutputs = lc_controller:evaluate(State#state.controller, SensorInputs),

    %% 3. Apply actuator changes
    NewConfig = cultural_silo_actuators:apply_actuators(
        ActuatorOutputs, State#state.config),

    %% 4. Detect innovations in population
    Population = get_population(State#state.population_id),
    NewInnovations = detect_population_innovations(
        Population, State#state.behavioral_archive, NewConfig),

    %% 5. Process imitation events
    ImitationResults = process_population_imitations(Population, NewConfig),

    %% 6. Update tradition strength and decay
    UpdatedTraditions = cultural_traditions:decay_traditions(
        maps:values(State#state.traditions), NewConfig),

    %% 7. Check for new tradition formation
    NewTraditions = check_all_tradition_formation(
        maps:values(State#state.traits), NewConfig),

    %% 8. Process cultural drift
    DriftReport = cultural_drift:process_cultural_drift(Population, NewConfig),

    %% 9. Update cultural state
    NewCulturalState = calculate_cultural_state(
        State#state.traits,
        UpdatedTraditions,
        State#state.innovations,
        ImitationResults,
        DriftReport
    ),

    {ok, State#state{
        cultural_state = NewCulturalState,
        config = NewConfig,
        innovations = merge_innovations(State#state.innovations, NewInnovations),
        traditions = maps:from_list([{T#tradition.id, T} || T <- UpdatedTraditions ++ NewTraditions]),
        drift_history = [DriftReport | State#state.drift_history]
    }}.

get_sensor_count() -> 10.
get_actuator_count() -> 8.
```

---

## Cross-Silo Signals

### Signals to Other Silos

| Signal | To Silo | Meaning |
|--------|---------|---------|
| `innovation_rate` | Task | High innovation = try new strategies |
| `tradition_strength` | Social | Strong traditions = social stability |
| `cultural_diversity` | Ecological | Diverse cultures = adaptive capacity |

### Signals from Other Silos

| Signal | From Silo | Effect |
|--------|-----------|--------|
| `social_network_density` | Social | Dense networks = faster cultural spread |
| `resource_abundance` | Ecological | Abundance = more innovation time |
| `stagnation_severity` | Task | Stagnation = increase innovation bonus |

---

## Events Emitted

| Event | Trigger |
|-------|---------|
| `innovation_emerged` | Novel beneficial behavior discovered |
| `innovation_spread` | Innovation adopted by others |
| `tradition_established` | Trait became tradition |
| `tradition_abandoned` | Tradition died out |
| `skill_imitated` | Successful imitation |
| `behavior_cloned` | Full behavioral cloning |
| `learning_succeeded` | Knowledge transfer worked |
| `learning_failed` | Knowledge transfer failed |
| `cultural_drift` | Significant population behavior shift |
| `fad_started` | Rapidly spreading temporary behavior |
| `fad_ended` | Fad behavior abandoned |

---

## Implementation Phases

- [ ] **Phase 1:** Cultural trait structure and storage
- [ ] **Phase 2:** Innovation detection algorithm
- [ ] **Phase 3:** L0 Sensors for cultural state
- [ ] **Phase 4:** L0 Actuators for cultural parameters
- [ ] **Phase 5:** Imitation and learning mechanics
- [ ] **Phase 6:** Tradition formation and decay
- [ ] **Phase 7:** Cumulative culture tracking
- [ ] **Phase 8:** Cultural silo server with TWEANN controller
- [ ] **Phase 9:** Cross-silo signal integration

---

## Success Criteria

- [ ] Innovations detected and tracked correctly
- [ ] Traditions form from widespread traits
- [ ] Imitation spreads behaviors through population
- [ ] Cumulative culture chains develop
- [ ] Cultural drift measured and reported
- [ ] TWEANN controller adapts cultural parameters
- [ ] All cultural events emitted correctly
- [ ] Cross-silo signals flow correctly

---

## References

- PLAN_BEHAVIORAL_EVENTS.md - Event definitions
- PLAN_KNOWLEDGE_TRANSFER.md - Learning mechanisms
- PLAN_SOCIAL_SILO.md - Social integration
- "The Secret of Our Success" - Henrich
- "Cultural Evolution" - Mesoudi
- "The Evolution of Culture" - Boyd & Richerson
