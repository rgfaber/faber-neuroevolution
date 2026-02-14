# PLAN_ECOLOGICAL_SILO.md

**Status:** Planning
**Created:** 2025-12-23
**Last Updated:** 2025-12-23
**Related:** PLAN_BEHAVIORAL_EVENTS.md, PLAN_EPIGENETICS.md, PLAN_SOCIAL_SILO.md

---

## Overview

This document specifies the **Ecological Silo** for the Liquid Conglomerate (LC) meta-controller. The Ecological Silo manages environmental dynamics: carrying capacity, resources, disease, environmental cycles, catastrophes, and inter-species relationships.

### Purpose

The Ecological Silo simulates environmental pressures that shape evolution:

1. **Carrying Capacity**: Population limits based on resources
2. **Resource Dynamics**: Availability, depletion, and regeneration
3. **Disease & Immunity**: Pathogen spread and resistance evolution
4. **Environmental Cycles**: Seasonal and periodic changes
5. **Catastrophes**: Mass extinction and recovery events
6. **Symbiosis**: Inter-individual and inter-species relationships

### Ecological Pressures on Evolution

| Pressure | Effect on Evolution |
|----------|---------------------|
| Resource scarcity | Competition, efficiency |
| Disease | Immune diversity, avoidance |
| Environmental change | Adaptability, generalism |
| Catastrophes | Recovery speed, robustness |
| Competition | Specialization, niche differentiation |

### Training Velocity & Inference Impact

| Metric | Without Ecological Silo | With Ecological Silo |
|--------|------------------------|---------------------|
| Environmental realism | Static environment | Dynamic pressures |
| Training velocity | Baseline (1.0x) | Variable (0.7-1.3x) |
| Inference latency | No environment checks | +3-8ms for resource/disease checks |
| Robustness | Brittle (overfits environment) | Robust (stress-tested) |
| Generalization | Poor (single niche) | Good (multi-niche) |

**Note:** Training velocity is variable - during resource abundance phases, evolution proceeds faster; during scarcity or disease outbreaks, it slows. Overall, ecological pressures produce more robust solutions that generalize better to real-world deployment. The variable pace also prevents premature convergence.

---

## Silo Architecture

### Ecological Silo Structure

```
┌─────────────────────────────────────────────────────────────────────┐
│                       ECOLOGICAL SILO                                │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │                      L0 SENSORS (14)                         │    │
│  │                                                              │    │
│  │  Resources        Population   Disease      Environment     │    │
│  │  ┌─────────┐     ┌─────────┐  ┌─────────┐  ┌─────────┐      │    │
│  │  │resource_│     │pop_     │  │disease_ │  │env_     │      │    │
│  │  │level    │     │density  │  │prevalence│ │stability│      │    │
│  │  │resource_│     │carrying_│  │immunity_│  │cycle_   │      │    │
│  │  │trend    │     │pressure │  │level    │  │phase    │      │    │
│  │  │resource_│     │growth_  │  │outbreak_│  │stress_  │      │    │
│  │  │variance │     │rate     │  │risk     │  │level    │      │    │
│  │  └─────────┘     └─────────┘  └─────────┘  └─────────┘      │    │
│  │                                                              │    │
│  │  Competition      Catastrophe                               │    │
│  │  ┌─────────┐     ┌─────────┐                                │    │
│  │  │niche_   │     │catast_  │                                │    │
│  │  │overlap  │     │risk     │                                │    │
│  │  └─────────┘     └─────────┘                                │    │
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
│  │  Resources        Carrying     Disease      Catastrophe     │    │
│  │  ┌─────────┐     ┌─────────┐  ┌─────────┐  ┌─────────┐      │    │
│  │  │resource_│     │capacity_│  │disease_ │  │catast_  │      │    │
│  │  │regen    │     │softness │  │virulence│  │frequency│      │    │
│  │  │resource_│     │overflow_│  │transmit_│  │severity_│      │    │
│  │  │decay    │     │penalty  │  │rate     │  │cap      │      │    │
│  │  └─────────┘     └─────────┘  └─────────┘  └─────────┘      │    │
│  │                                                              │    │
│  │  Environment                                                │    │
│  │  ┌─────────┐     ┌─────────┐                                │    │
│  │  │cycle_   │     │stress_  │                                │    │
│  │  │amplitude│     │threshold│                                │    │
│  │  └─────────┘     └─────────┘                                │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Core Concepts

### Environment State

```erlang
-record(environment, {
    id :: binary(),
    population_id :: binary(),

    %% Resources
    resource_level :: float(),           % 0.0-1.0 (current availability)
    resource_capacity :: float(),        % Maximum possible
    resource_regeneration_rate :: float(),
    resource_consumption_rate :: float(),

    %% Population dynamics
    carrying_capacity :: non_neg_integer(),
    current_population :: non_neg_integer(),
    population_density :: float(),       % current / carrying_capacity
    growth_rate :: float(),

    %% Disease state
    active_diseases :: [#disease{}],
    population_immunity :: #{binary() => float()},  % disease_id => immunity level
    epidemic_active :: boolean(),

    %% Environmental conditions
    stability :: float(),                % 0.0-1.0 (how stable conditions are)
    stress_level :: float(),             % 0.0-1.0 (environmental stress)
    cycle_phase :: float(),              % 0.0-1.0 (position in environmental cycle)
    cycle_period :: non_neg_integer(),   % Generations per full cycle

    %% Catastrophe risk
    catastrophe_risk :: float(),         % 0.0-1.0
    generations_since_catastrophe :: non_neg_integer(),

    %% Competition
    niche_count :: non_neg_integer(),
    niche_occupancy :: #{binary() => [binary()]}  % niche_id => [species_ids]
}).
```

### Disease

```erlang
-record(disease, {
    id :: binary(),
    name :: binary(),

    %% Characteristics
    virulence :: float(),                % 0.0-1.0 (how deadly)
    transmissibility :: float(),         % 0.0-1.0 (how contagious)
    mutation_rate :: float(),            % How fast disease evolves

    %% Spread
    infected_individuals :: [binary()],
    infection_count :: non_neg_integer(),
    peak_infection :: non_neg_integer(),

    %% Status
    status :: emerging | epidemic | endemic | declining | eradicated,
    emerged_at :: integer(),
    eradicated_at :: integer() | active
}).
```

### Catastrophe

```erlang
-record(catastrophe, {
    id :: binary(),

    %% Type
    catastrophe_type :: mass_extinction | environmental_shift |
                        resource_crash | epidemic | cosmic_event,

    %% Severity
    severity :: float(),                 % 0.0-1.0
    mortality_rate :: float(),           % Proportion killed
    resource_impact :: float(),          % Resource level reduction

    %% Affected
    affected_populations :: [binary()],
    survivors_count :: non_neg_integer(),

    %% Recovery
    recovery_estimate_generations :: non_neg_integer(),
    actual_recovery_generations :: non_neg_integer() | ongoing,

    %% Timing
    occurred_at :: integer(),
    recovery_began_at :: integer() | not_yet,
    recovery_completed_at :: integer() | ongoing
}).
```

---

## L0 Sensors

### Sensor Specifications

```erlang
-module(ecological_silo_sensors).
-behaviour(l0_sensor_behaviour).

-export([sensor_specs/0, read_sensors/1]).

sensor_specs() ->
    [
        %% Resource sensors
        #{
            id => resource_level,
            range => {0.0, 1.0},
            description => "Current resource availability"
        },
        #{
            id => resource_trend,
            range => {-1.0, 1.0},
            description => "Resource change direction"
        },
        #{
            id => resource_variance,
            range => {0.0, 1.0},
            description => "Resource distribution variance"
        },

        %% Population sensors
        #{
            id => population_density,
            range => {0.0, 2.0},  % Can exceed 1.0 (overcrowding)
            description => "Population relative to carrying capacity"
        },
        #{
            id => carrying_pressure,
            range => {0.0, 1.0},
            description => "How close to carrying capacity limit"
        },
        #{
            id => growth_rate,
            range => {-1.0, 1.0},
            description => "Population growth rate"
        },

        %% Disease sensors
        #{
            id => disease_prevalence,
            range => {0.0, 1.0},
            description => "Proportion of population infected"
        },
        #{
            id => immunity_level,
            range => {0.0, 1.0},
            description => "Average population immunity"
        },
        #{
            id => outbreak_risk,
            range => {0.0, 1.0},
            description => "Risk of new disease outbreak"
        },

        %% Environment sensors
        #{
            id => environmental_stability,
            range => {0.0, 1.0},
            description => "How stable current conditions are"
        },
        #{
            id => cycle_phase,
            range => {0.0, 1.0},
            description => "Position in environmental cycle"
        },
        #{
            id => stress_level,
            range => {0.0, 1.0},
            description => "Environmental stress on population"
        },

        %% Competition
        #{
            id => niche_overlap,
            range => {0.0, 1.0},
            description => "Degree of niche competition"
        },

        %% Catastrophe
        #{
            id => catastrophe_risk,
            range => {0.0, 1.0},
            description => "Risk of catastrophic event"
        }
    ].

read_sensors(#environment{} = Env) ->
    [
        Env#environment.resource_level,
        calculate_resource_trend(Env),
        calculate_resource_variance(Env),
        min(2.0, Env#environment.population_density),
        calculate_carrying_pressure(Env),
        clamp(Env#environment.growth_rate, -1.0, 1.0),
        calculate_disease_prevalence(Env),
        calculate_mean_immunity(Env),
        calculate_outbreak_risk(Env),
        Env#environment.stability,
        Env#environment.cycle_phase,
        Env#environment.stress_level,
        calculate_niche_overlap(Env),
        Env#environment.catastrophe_risk
    ].
```

---

## L0 Actuators

### Actuator Specifications

```erlang
-module(ecological_silo_actuators).
-behaviour(l0_actuator_behaviour).

-export([actuator_specs/0, apply_actuators/2]).

actuator_specs() ->
    [
        %% Resource actuators
        #{
            id => resource_regeneration_multiplier,
            range => {0.5, 2.0},
            default => 1.0,
            description => "Resource regeneration speed multiplier"
        },
        #{
            id => resource_decay_rate,
            range => {0.0, 0.2},
            default => 0.05,
            description => "Rate of resource decay/consumption"
        },

        %% Carrying capacity actuators
        #{
            id => capacity_softness,
            range => {0.0, 1.0},
            default => 0.3,
            description => "How soft the carrying capacity limit is"
        },
        #{
            id => overflow_penalty,
            range => {0.0, 0.5},
            default => 0.1,
            description => "Fitness penalty when over carrying capacity"
        },

        %% Disease actuators
        #{
            id => disease_virulence_cap,
            range => {0.0, 1.0},
            default => 0.8,
            description => "Maximum disease virulence"
        },
        #{
            id => transmission_rate_modifier,
            range => {0.5, 2.0},
            default => 1.0,
            description => "Disease transmission rate multiplier"
        },

        %% Catastrophe actuators
        #{
            id => catastrophe_frequency,
            range => {0.0, 0.1},
            default => 0.01,
            description => "Base probability of catastrophe per generation"
        },
        #{
            id => catastrophe_severity_cap,
            range => {0.3, 1.0},
            default => 0.7,
            description => "Maximum catastrophe severity"
        },

        %% Environment actuators
        #{
            id => cycle_amplitude,
            range => {0.0, 1.0},
            default => 0.3,
            description => "Amplitude of environmental cycles"
        },
        #{
            id => stress_threshold,
            range => {0.3, 0.9},
            default => 0.6,
            description => "Stress level that triggers epigenetic marks"
        }
    ].

apply_actuators(Outputs, #ecological_config{} = Config) ->
    [ResRegen, ResDecay, CapSoft, OverPen,
     DisVir, TransMod, CatFreq, CatSev, CycAmp, StressThresh] = Outputs,

    Config#ecological_config{
        resource_regeneration_multiplier = denormalize(resource_regeneration_multiplier, ResRegen),
        resource_decay_rate = denormalize(resource_decay_rate, ResDecay),
        capacity_softness = denormalize(capacity_softness, CapSoft),
        overflow_penalty = denormalize(overflow_penalty, OverPen),
        disease_virulence_cap = denormalize(disease_virulence_cap, DisVir),
        transmission_rate_modifier = denormalize(transmission_rate_modifier, TransMod),
        catastrophe_frequency = denormalize(catastrophe_frequency, CatFreq),
        catastrophe_severity_cap = denormalize(catastrophe_severity_cap, CatSev),
        cycle_amplitude = denormalize(cycle_amplitude, CycAmp),
        stress_threshold = denormalize(stress_threshold, StressThresh)
    }.
```

---

## Carrying Capacity System

### Population Regulation

```erlang
-module(ecological_carrying).

%% Calculate fitness penalty for overcrowding
-spec calculate_density_penalty(Individual, Environment, Config) -> float()
    when Individual :: #individual{},
         Environment :: #environment{},
         Config :: #ecological_config{}.

calculate_density_penalty(Individual, Env, Config) ->
    Density = Env#environment.population_density,
    Softness = Config#ecological_config.capacity_softness,
    OverflowPenalty = Config#ecological_config.overflow_penalty,

    case Density > 1.0 of
        true ->
            %% Over carrying capacity
            Overflow = Density - 1.0,
            %% Soft cap: penalty increases gradually
            BasePenalty = OverflowPenalty * (1 - math:exp(-Overflow / Softness)),
            %% Fitness multiplier (1.0 - penalty)
            max(0.1, 1.0 - BasePenalty);
        false ->
            %% Under carrying capacity - no penalty
            1.0
    end.

%% Process population against carrying capacity
-spec enforce_carrying_capacity(Population, Environment, Config) -> {Deaths, UpdatedEnv}
    when Population :: #population{},
         Environment :: #environment{},
         Config :: #ecological_config{},
         Deaths :: [#individual{}],
         UpdatedEnv :: #environment{}.

enforce_carrying_capacity(Population, Env, Config) ->
    CurrentPop = length(Population#population.individuals),
    Capacity = Env#environment.carrying_capacity,

    case CurrentPop > Capacity of
        true ->
            %% Overcrowding - some must die
            Excess = CurrentPop - Capacity,
            Softness = Config#ecological_config.capacity_softness,

            %% Probabilistic death based on excess and softness
            DeathProbability = (1 - Softness) * (Excess / CurrentPop),

            Deaths = lists:filter(fun(Ind) ->
                %% Lower fitness = higher death probability
                FitnessModifier = 1 - (Ind#individual.fitness /
                                      Population#population.max_fitness),
                rand:uniform() < (DeathProbability * (1 + FitnessModifier))
            end, Population#population.individuals),

            %% Emit event
            case length(Deaths) > 0 of
                true ->
                    neuroevolution_events:emit(carrying_capacity_exceeded, #{
                        event_type => carrying_capacity_exceeded,
                        population_id => Population#population.id,
                        capacity => Capacity,
                        population_size => CurrentPop,
                        deaths_count => length(Deaths),
                        occurred_at => erlang:system_time(microsecond)
                    });
                false ->
                    ok
            end,

            UpdatedEnv = Env#environment{
                current_population = CurrentPop - length(Deaths),
                population_density = (CurrentPop - length(Deaths)) / Capacity
            },

            {Deaths, UpdatedEnv};

        false ->
            {[], Env}
    end.
```

---

## Disease System

### Disease Spread

```erlang
-module(ecological_disease).

%% Simulate disease spread for one generation
-spec spread_diseases(Population, Environment, Config) -> {UpdatedPop, UpdatedEnv}
    when Population :: #population{},
         Environment :: #environment{},
         Config :: #ecological_config{},
         UpdatedPop :: #population{},
         UpdatedEnv :: #environment{}.

spread_diseases(Population, Env, Config) ->
    %% Process each active disease
    {FinalPop, FinalEnv} = lists:foldl(fun(Disease, {AccPop, AccEnv}) ->
        spread_single_disease(Disease, AccPop, AccEnv, Config)
    end, {Population, Env}, Env#environment.active_diseases),

    %% Check for new disease emergence
    {FinalPop2, FinalEnv2} = check_disease_emergence(FinalPop, FinalEnv, Config),

    {FinalPop2, FinalEnv2}.

spread_single_disease(#disease{} = Disease, Population, Env, Config) ->
    Infected = Disease#disease.infected_individuals,
    Susceptible = get_susceptible_individuals(Population, Disease, Env),

    TransmissionMod = Config#ecological_config.transmission_rate_modifier,
    BaseTransmission = Disease#disease.transmissibility * TransmissionMod,

    %% Calculate new infections
    NewInfections = lists:filter(fun(Ind) ->
        %% Contact probability based on population density
        ContactProb = min(1.0, Env#environment.population_density * 0.5),

        %% Transmission probability given contact
        TransmitProb = BaseTransmission * (1 - get_immunity(Ind, Disease, Env)),

        rand:uniform() < (ContactProb * TransmitProb)
    end, Susceptible),

    %% Calculate recoveries and deaths
    {Recovered, Died} = process_infected(Infected, Disease, Population, Config),

    %% Update disease state
    NewInfected = (Disease#disease.infected_individuals -- Recovered -- Died) ++
                  [I#individual.id || I <- NewInfections],

    UpdatedDisease = Disease#disease{
        infected_individuals = NewInfected,
        infection_count = length(NewInfected),
        peak_infection = max(Disease#disease.peak_infection, length(NewInfected)),
        status = determine_disease_status(length(NewInfected), length(Population#population.individuals))
    },

    %% Update immunity for recovered
    UpdatedImmunity = lists:foldl(fun(RecoveredId, Acc) ->
        CurrentImmunity = maps:get(Disease#disease.id, Acc, 0.0),
        maps:put(Disease#disease.id, min(1.0, CurrentImmunity + 0.3), Acc)
    end, Env#environment.population_immunity, Recovered),

    %% Emit events
    emit_disease_events(Disease, NewInfections, Recovered, Died),

    %% Apply fitness penalty to infected
    PenalizedPop = apply_disease_penalties(Population, NewInfected, Disease),

    UpdatedEnv = Env#environment{
        active_diseases = update_disease_list(Env#environment.active_diseases, UpdatedDisease),
        population_immunity = UpdatedImmunity
    },

    {PenalizedPop, UpdatedEnv}.

check_disease_emergence(Population, Env, Config) ->
    %% Disease emergence probability increases with density and stress
    BaseProb = 0.001,
    DensityFactor = Env#environment.population_density,
    StressFactor = Env#environment.stress_level,

    EmergenceProb = BaseProb * (1 + DensityFactor) * (1 + StressFactor),

    case rand:uniform() < EmergenceProb of
        true ->
            NewDisease = generate_random_disease(Config),

            %% Select patient zero
            PatientZero = lists:nth(
                rand:uniform(length(Population#population.individuals)),
                Population#population.individuals
            ),

            InfectedDisease = NewDisease#disease{
                infected_individuals = [PatientZero#individual.id],
                infection_count = 1,
                status = emerging,
                emerged_at = erlang:system_time(microsecond)
            },

            neuroevolution_events:emit(disease_emerged, #{
                event_type => disease_emerged,
                disease_id => NewDisease#disease.id,
                population_id => Population#population.id,
                patient_zero => PatientZero#individual.id,
                virulence => NewDisease#disease.virulence,
                transmissibility => NewDisease#disease.transmissibility,
                emerged_at => InfectedDisease#disease.emerged_at
            }),

            UpdatedEnv = Env#environment{
                active_diseases = [InfectedDisease | Env#environment.active_diseases]
            },

            {Population, UpdatedEnv};

        false ->
            {Population, Env}
    end.
```

---

## Environmental Cycles

### Cycle Processing

```erlang
-module(ecological_cycles).

%% Advance environmental cycle by one generation
-spec advance_cycle(Environment, Config) -> UpdatedEnvironment
    when Environment :: #environment{},
         Config :: #ecological_config{},
         UpdatedEnvironment :: #environment{}.

advance_cycle(#environment{} = Env, Config) ->
    %% Advance phase
    Period = Env#environment.cycle_period,
    CurrentPhase = Env#environment.cycle_phase,
    NewPhase = case CurrentPhase + (1.0 / Period) of
        P when P >= 1.0 -> P - 1.0;
        P -> P
    end,

    %% Calculate cycle effects
    Amplitude = Config#ecological_config.cycle_amplitude,

    %% Sinusoidal resource variation
    ResourceModifier = 1.0 + (Amplitude * math:sin(2 * math:pi() * NewPhase)),
    NewResourceLevel = clamp(
        Env#environment.resource_level * ResourceModifier,
        0.0, Env#environment.resource_capacity
    ),

    %% Stress varies inversely with resources
    StressFromResources = 1.0 - (NewResourceLevel / Env#environment.resource_capacity),
    NewStressLevel = clamp(StressFromResources, 0.0, 1.0),

    %% Check for phase transitions (emit events at key points)
    emit_cycle_events(CurrentPhase, NewPhase),

    Env#environment{
        cycle_phase = NewPhase,
        resource_level = NewResourceLevel,
        stress_level = NewStressLevel
    }.

emit_cycle_events(OldPhase, NewPhase) ->
    %% Emit event at cycle quarters
    OldQuarter = trunc(OldPhase * 4),
    NewQuarter = trunc(NewPhase * 4),

    case NewQuarter =/= OldQuarter of
        true ->
            PhaseName = case NewQuarter of
                0 -> spring;
                1 -> summer;
                2 -> autumn;
                3 -> winter
            end,
            neuroevolution_events:emit(environmental_cycle_shifted, #{
                event_type => environmental_cycle_shifted,
                new_phase => PhaseName,
                phase_value => NewPhase,
                shifted_at => erlang:system_time(microsecond)
            });
        false ->
            ok
    end.
```

---

## Catastrophe System

### Catastrophe Generation

```erlang
-module(ecological_catastrophe).

%% Check for and process catastrophes
-spec check_catastrophe(Environment, Config) -> {Catastrophe | none, UpdatedEnv}
    when Environment :: #environment{},
         Config :: #ecological_config{},
         Catastrophe :: #catastrophe{},
         UpdatedEnv :: #environment{}.

check_catastrophe(Env, Config) ->
    BaseFreq = Config#ecological_config.catastrophe_frequency,

    %% Catastrophe risk increases with time since last one
    TimeFactor = 1 + (Env#environment.generations_since_catastrophe / 100),

    %% Stress also increases risk
    StressFactor = 1 + Env#environment.stress_level,

    ActualRisk = min(0.5, BaseFreq * TimeFactor * StressFactor),

    UpdatedEnv = Env#environment{
        catastrophe_risk = ActualRisk,
        generations_since_catastrophe = Env#environment.generations_since_catastrophe + 1
    },

    case rand:uniform() < ActualRisk of
        true ->
            %% Catastrophe occurs!
            Catastrophe = generate_catastrophe(UpdatedEnv, Config),
            {Catastrophe, UpdatedEnv#environment{
                generations_since_catastrophe = 0
            }};
        false ->
            {none, UpdatedEnv}
    end.

generate_catastrophe(Env, Config) ->
    %% Determine type based on conditions
    Type = select_catastrophe_type(Env),

    %% Calculate severity (capped by config)
    BaseSeverity = 0.3 + (rand:uniform() * 0.7),
    Severity = min(BaseSeverity, Config#ecological_config.catastrophe_severity_cap),

    %% Calculate effects
    MortalityRate = Severity * 0.8,  % Up to 80% mortality
    ResourceImpact = Severity * 0.6,  % Up to 60% resource loss

    #catastrophe{
        id = generate_catastrophe_id(),
        catastrophe_type = Type,
        severity = Severity,
        mortality_rate = MortalityRate,
        resource_impact = ResourceImpact,
        affected_populations = [Env#environment.population_id],
        recovery_estimate_generations = trunc(Severity * 50),
        occurred_at = erlang:system_time(microsecond)
    }.

select_catastrophe_type(Env) ->
    %% Weight by environmental conditions
    Weights = [
        {mass_extinction, 0.1},
        {environmental_shift, 0.3},
        {resource_crash, 0.3 * (1 - Env#environment.resource_level)},
        {epidemic, 0.2 * Env#environment.population_density},
        {cosmic_event, 0.1}
    ],

    TotalWeight = lists:sum([W || {_, W} <- Weights]),
    Normalized = [{T, W / TotalWeight} || {T, W} <- Weights],

    %% Weighted random selection
    Roll = rand:uniform(),
    select_by_weight(Normalized, Roll, 0.0).

%% Execute catastrophe
-spec execute_catastrophe(Catastrophe, Population, Environment) ->
    {Survivors, UpdatedEnv}
    when Catastrophe :: #catastrophe{},
         Population :: #population{},
         Environment :: #environment{},
         Survivors :: [#individual{}],
         UpdatedEnv :: #environment{}.

execute_catastrophe(#catastrophe{} = Cat, Population, Env) ->
    %% Determine survivors
    Survivors = lists:filter(fun(Ind) ->
        %% Survival chance based on fitness and random factors
        SurvivalChance = (1 - Cat#catastrophe.mortality_rate) *
                        (0.5 + (Ind#individual.fitness / Population#population.max_fitness * 0.5)),
        rand:uniform() < SurvivalChance
    end, Population#population.individuals),

    DeadCount = length(Population#population.individuals) - length(Survivors),

    %% Apply resource impact
    NewResourceLevel = Env#environment.resource_level * (1 - Cat#catastrophe.resource_impact),

    %% Emit event
    neuroevolution_events:emit(catastrophe_occurred, #{
        event_type => catastrophe_occurred,
        catastrophe_id => Cat#catastrophe.id,
        catastrophe_type => Cat#catastrophe.catastrophe_type,
        affected_populations => Cat#catastrophe.affected_populations,
        severity => Cat#catastrophe.severity,
        mortality_rate => Cat#catastrophe.mortality_rate,
        survivors_count => length(Survivors),
        deaths_count => DeadCount,
        recovery_estimate_generations => Cat#catastrophe.recovery_estimate_generations,
        occurred_at => Cat#catastrophe.occurred_at
    }),

    UpdatedEnv = Env#environment{
        resource_level = NewResourceLevel,
        current_population = length(Survivors),
        population_density = length(Survivors) / Env#environment.carrying_capacity,
        stability = max(0.1, Env#environment.stability - Cat#catastrophe.severity),
        stress_level = min(1.0, Env#environment.stress_level + Cat#catastrophe.severity)
    },

    {Survivors, UpdatedEnv}.
```

---

## Symbiosis

### Inter-Individual Relationships

```erlang
-module(ecological_symbiosis).

-record(symbiotic_relationship, {
    id :: binary(),
    partner_a :: binary(),
    partner_b :: binary(),
    relationship_type :: mutualism | commensalism | parasitism,
    benefit_a :: float(),              % -1.0 to 1.0
    benefit_b :: float(),              % -1.0 to 1.0
    strength :: float(),               % 0.0 to 1.0
    formed_at :: integer(),
    duration_generations :: non_neg_integer()
}).

%% Check for symbiosis formation between individuals
-spec check_symbiosis_formation(IndA, IndB, Config) ->
    {symbiosis, #symbiotic_relationship{}} | no_symbiosis
    when IndA :: #individual{},
         IndB :: #individual{},
         Config :: #ecological_config{}.

check_symbiosis_formation(IndA, IndB, Config) ->
    %% Calculate behavioral compatibility
    Compatibility = calculate_behavioral_compatibility(IndA, IndB),

    %% Formation probability based on compatibility
    FormationProb = Compatibility * Config#ecological_config.symbiosis_formation_rate,

    case rand:uniform() < FormationProb of
        true ->
            %% Determine relationship type and benefits
            {Type, BenefitA, BenefitB} = determine_relationship_dynamics(IndA, IndB),

            Relationship = #symbiotic_relationship{
                id = generate_symbiosis_id(),
                partner_a = IndA#individual.id,
                partner_b = IndB#individual.id,
                relationship_type = Type,
                benefit_a = BenefitA,
                benefit_b = BenefitB,
                strength = Compatibility,
                formed_at = erlang:system_time(microsecond),
                duration_generations = 0
            },

            neuroevolution_events:emit(symbiosis_formed, #{
                event_type => symbiosis_formed,
                symbiosis_id => Relationship#symbiotic_relationship.id,
                partner_a_id => IndA#individual.id,
                partner_b_id => IndB#individual.id,
                symbiosis_type => Type,
                benefit_a => BenefitA,
                benefit_b => BenefitB,
                formed_at => Relationship#symbiotic_relationship.formed_at
            }),

            {symbiosis, Relationship};

        false ->
            no_symbiosis
    end.

determine_relationship_dynamics(IndA, IndB) ->
    %% Based on fitness differential and behavioral patterns
    FitDiff = IndA#individual.fitness - IndB#individual.fitness,

    case abs(FitDiff) < 0.2 of
        true ->
            %% Similar fitness: likely mutualism
            Benefit = 0.1 + rand:uniform() * 0.2,
            {mutualism, Benefit, Benefit};
        false when FitDiff > 0 ->
            %% A much fitter: could be commensalism or parasitism
            case rand:uniform() < 0.7 of
                true ->
                    %% Commensalism: A helps B, A unaffected
                    {commensalism, 0.0, 0.1 + rand:uniform() * 0.2};
                false ->
                    %% Parasitism: A exploits B
                    {parasitism, 0.15, -0.1}
            end;
        false ->
            %% B much fitter
            case rand:uniform() < 0.7 of
                true ->
                    {commensalism, 0.1 + rand:uniform() * 0.2, 0.0};
                false ->
                    {parasitism, -0.1, 0.15}
            end
    end.
```

---

## Ecological Silo Server

### Module: `ecological_silo.erl`

```erlang
-module(ecological_silo).
-behaviour(gen_server).
-behaviour(lc_silo_behaviour).

%% API
-export([
    start_link/1,
    read_sensors/1,
    apply_actuators/2,
    get_environment/1,
    trigger_catastrophe/2,
    introduce_disease/2,
    get_state/1
]).

-record(state, {
    population_id :: binary(),
    environment :: #environment{},
    config :: #ecological_config{},
    controller :: pid(),

    %% Tracking
    catastrophe_history :: [#catastrophe{}],
    disease_history :: [#disease{}],
    symbiotic_relationships :: #{binary() => #symbiotic_relationship{}}
}).

%%====================================================================
%% Silo Behaviour Implementation
%%====================================================================

init_silo(Config) ->
    {ok, #state{
        population_id = maps:get(population_id, Config),
        environment = initial_environment(Config),
        config = default_ecological_config(),
        catastrophe_history = [],
        disease_history = [],
        symbiotic_relationships = #{}
    }}.

step(#state{} = State, Generation) ->
    %% 1. Read current ecological state
    SensorInputs = ecological_silo_sensors:read_sensors(State#state.environment),

    %% 2. Get controller outputs
    ActuatorOutputs = lc_controller:evaluate(State#state.controller, SensorInputs),

    %% 3. Apply actuator changes
    NewConfig = ecological_silo_actuators:apply_actuators(
        ActuatorOutputs, State#state.config),

    %% 4. Advance environmental cycle
    Env1 = ecological_cycles:advance_cycle(State#state.environment, NewConfig),

    %% 5. Process resource dynamics
    Env2 = process_resources(Env1, NewConfig),

    %% 6. Check for catastrophe
    {MaybeCatastrophe, Env3} = ecological_catastrophe:check_catastrophe(Env2, NewConfig),

    %% 7. Execute catastrophe if occurred
    Env4 = case MaybeCatastrophe of
        none -> Env3;
        Cat ->
            Population = get_population(State#state.population_id),
            {Survivors, UpdatedEnv} = ecological_catastrophe:execute_catastrophe(
                Cat, Population, Env3),
            update_population_individuals(State#state.population_id, Survivors),
            UpdatedEnv
    end,

    %% 8. Spread diseases
    Population2 = get_population(State#state.population_id),
    {_, Env5} = ecological_disease:spread_diseases(Population2, Env4, NewConfig),

    %% 9. Enforce carrying capacity
    Population3 = get_population(State#state.population_id),
    {Deaths, Env6} = ecological_carrying:enforce_carrying_capacity(
        Population3, Env5, NewConfig),
    remove_dead_individuals(State#state.population_id, Deaths),

    %% 10. Trigger epigenetic marks for stress
    trigger_stress_marks(Env6, NewConfig),

    {ok, State#state{
        environment = Env6,
        config = NewConfig,
        catastrophe_history = maybe_add_catastrophe(MaybeCatastrophe, State#state.catastrophe_history)
    }}.

get_sensor_count() -> 14.
get_actuator_count() -> 10.
```

---

## Cross-Silo Signals

### Signals to Other Silos

| Signal | To Silo | Meaning |
|--------|---------|---------|
| `resource_scarcity` | Social | Scarcity increases competition |
| `stress_level` | All | Triggers epigenetic marking |
| `catastrophe_recovery` | Task | Recovery = opportunity for innovation |

### Signals from Other Silos

| Signal | From Silo | Effect |
|--------|-----------|--------|
| `population_growth` | Task | Affects carrying capacity pressure |
| `conflict_level` | Social | Conflict depletes resources |
| `innovation_rate` | Cultural | Innovation improves resource efficiency |

---

## Events Emitted

| Event | Trigger |
|-------|---------|
| `carrying_capacity_reached` | Population at capacity |
| `carrying_capacity_exceeded` | Population over capacity |
| `resource_depleted` | Resources exhausted |
| `resource_replenished` | Resources restored |
| `disease_emerged` | New disease appeared |
| `disease_spread` | Infection transmitted |
| `immunity_developed` | Resistance evolved |
| `epidemic_started` | Widespread outbreak |
| `epidemic_ended` | Outbreak concluded |
| `environmental_cycle_shifted` | Seasonal change |
| `catastrophe_occurred` | Major event |
| `recovery_began` | Post-catastrophe recovery |
| `symbiosis_formed` | Mutualistic relationship |
| `parasitism_detected` | Exploitative relationship |

---

## Implementation Phases

- [ ] **Phase 1:** Environment state and resource tracking
- [ ] **Phase 2:** L0 Sensors for ecological state
- [ ] **Phase 3:** L0 Actuators for ecological parameters
- [ ] **Phase 4:** Carrying capacity system
- [ ] **Phase 5:** Disease spread and immunity
- [ ] **Phase 6:** Environmental cycles
- [ ] **Phase 7:** Catastrophe system
- [ ] **Phase 8:** Symbiosis relationships
- [ ] **Phase 9:** Ecological silo server with TWEANN controller
- [ ] **Phase 10:** Cross-silo signal integration

---

## Success Criteria

- [ ] Resource dynamics affect population fitness
- [ ] Carrying capacity limits population size
- [ ] Diseases spread and drive immunity evolution
- [ ] Environmental cycles create periodic pressure
- [ ] Catastrophes cause mass deaths and recovery
- [ ] Symbiotic relationships form and affect fitness
- [ ] TWEANN controller adapts ecological parameters
- [ ] All ecological events emitted correctly
- [ ] Cross-silo signals flow correctly

---

## References

- PLAN_BEHAVIORAL_EVENTS.md - Event definitions
- PLAN_EPIGENETICS.md - Stress-triggered marks
- PLAN_SOCIAL_SILO.md - Competition dynamics
- "Ecology: Concepts and Applications" - Molles
- "The Theory of Island Biogeography" - MacArthur & Wilson
- "Epidemiology" - Rothman
