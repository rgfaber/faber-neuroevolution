# PLAN_AGE_LIFECYCLE.md

**Status:** Planning
**Created:** 2025-12-23
**Last Updated:** 2025-12-23
**Related:** PLAN_BEHAVIORAL_EVENTS.md, PLAN_EPIGENETICS.md

---

## Overview

This document specifies an age-based lifecycle system for neuroevolution, introducing natural aging, lifecycle stages, breeding windows, and death from senescence. This adds biological realism and creates pressure for efficient knowledge transfer between generations.

### Motivation

Traditional neuroevolution treats individuals as ageless - they exist until culled by selection. Real biological systems have:

1. **Development Periods**: Juveniles cannot reproduce
2. **Fertility Windows**: Optimal breeding age exists
3. **Senescence**: Performance declines with age
4. **Natural Death**: Even fit individuals eventually die

These constraints create evolutionary pressure for:
- Rapid learning during development
- Knowledge transfer to offspring
- Lineage continuity over individual immortality
- Diverse strategies (fast reproduction vs. long learning)

---

## Lifecycle Stages

```
┌─────────────────────────────────────────────────────────────────┐
│                       INDIVIDUAL LIFECYCLE                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   BIRTH ──► JUVENILE ──► FERTILE ──► SENESCENT ──► DEATH        │
│              │            │            │                         │
│              │            │            │                         │
│              ▼            ▼            ▼                         │
│           Learning    Breeding    Mentoring                      │
│           Protected   Eligible    Declining                      │
│           No mating   Peak vigor  Wisdom keeper                  │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Stage Definitions

| Stage | Duration | Breeding | Selection | Role |
|-------|----------|----------|-----------|------|
| **Juvenile** | 0 to maturity_age | NO | Protected | Learning, development |
| **Fertile** | maturity_age to senescence_age | YES | Normal | Breeding, peak performance |
| **Senescent** | senescence_age to max_age | NO | Reduced pressure | Mentoring, knowledge transfer |

---

## Configuration

### Lifecycle Parameters

```erlang
-record(lifecycle_config, {
    %% Age is measured in generations (evaluation cycles)

    %% Juvenile stage
    maturity_age = 5 :: non_neg_integer(),      % Generations until fertile
    juvenile_protection = true :: boolean(),     % Protect from culling
    juvenile_learning_bonus = 1.5 :: float(),   % Learning rate multiplier

    %% Fertile stage
    senescence_age = 50 :: non_neg_integer(),   % Generations until senescent
    peak_fertility_age = 15 :: non_neg_integer(), % Optimal breeding age
    fertility_curve = gaussian :: gaussian | flat | declining,

    %% Senescent stage
    max_age = 100 :: non_neg_integer(),         % Maximum lifespan
    senescent_decay_rate = 0.02 :: float(),     % Fitness decline per gen
    senescent_wisdom_bonus = 1.3 :: float(),    % Knowledge transfer bonus

    %% Breeding constraints
    min_breeding_age = 5 :: non_neg_integer(),  % Same as maturity_age
    max_breeding_age = 50 :: non_neg_integer(), % Same as senescence_age
    breeding_cooldown = 1 :: non_neg_integer(), % Generations between breeding
    max_lifetime_offspring = unlimited :: unlimited | non_neg_integer(),

    %% Death probabilities (per generation)
    juvenile_death_prob = 0.0 :: float(),       % Usually protected
    fertile_death_prob = 0.01 :: float(),       % Small random death
    senescent_death_prob_base = 0.05 :: float(),% Increases with age
    senescent_death_prob_growth = 0.02 :: float() % Per gen increase
}).
```

### Preset Configurations

```erlang
%% Fast lifecycle - many generations, quick turnover
fast_lifecycle() ->
    #lifecycle_config{
        maturity_age = 2,
        senescence_age = 20,
        max_age = 30,
        peak_fertility_age = 8
    }.

%% Slow lifecycle - long-lived, emphasis on learning
slow_lifecycle() ->
    #lifecycle_config{
        maturity_age = 10,
        senescence_age = 100,
        max_age = 200,
        peak_fertility_age = 40,
        juvenile_learning_bonus = 2.0,
        senescent_wisdom_bonus = 1.5
    }.

%% No aging - traditional neuroevolution (opt-out)
immortal_lifecycle() ->
    #lifecycle_config{
        maturity_age = 0,
        senescence_age = infinity,
        max_age = infinity,
        juvenile_protection = false
    }.
```

---

## Individual Record Extension

```erlang
-record(individual, {
    %% Existing fields
    id :: binary(),
    genome :: term(),
    fitness :: float(),
    species_id :: binary(),

    %% NEW: Lifecycle fields
    birth_generation :: non_neg_integer(),
    current_age :: non_neg_integer(),           % Generations since birth
    lifecycle_stage :: juvenile | fertile | senescent,

    %% Breeding tracking
    last_breeding_generation :: non_neg_integer() | never,
    offspring_count :: non_neg_integer(),
    breeding_eligibility :: eligible | cooldown | ineligible,

    %% Performance tracking
    peak_fitness :: float(),                    % Historical maximum
    peak_fitness_age :: non_neg_integer(),      % Age when peak achieved
    fitness_trajectory :: improving | stable | declining,

    %% Death prediction
    expected_remaining_generations :: non_neg_integer() | unknown,
    death_probability_current :: float()
}).
```

---

## Lifecycle Server

### Module: `lifecycle_server.erl`

```erlang
-module(lifecycle_server).
-behaviour(gen_server).

%% API
-export([
    start_link/1,
    tick_generation/1,
    get_stage/2,
    can_breed/2,
    apply_aging_effects/2,
    get_death_probability/2,
    process_natural_deaths/1
]).

%% Callbacks
-export([init/1, handle_call/3, handle_cast/2, handle_info/2]).

-record(state, {
    population_id :: binary(),
    config :: #lifecycle_config{},
    individuals :: #{binary() => #individual{}},
    generation :: non_neg_integer()
}).

%%====================================================================
%% API Functions
%%====================================================================

%% Called at end of each generation
tick_generation(PopulationId) ->
    gen_server:call(?SERVER(PopulationId), tick_generation).

%% Get current lifecycle stage for an individual
get_stage(PopulationId, IndividualId) ->
    gen_server:call(?SERVER(PopulationId), {get_stage, IndividualId}).

%% Check if individual can breed this generation
can_breed(PopulationId, IndividualId) ->
    gen_server:call(?SERVER(PopulationId), {can_breed, IndividualId}).

%% Apply aging effects (fitness decay, etc.)
apply_aging_effects(PopulationId, IndividualId) ->
    gen_server:call(?SERVER(PopulationId), {apply_aging_effects, IndividualId}).

%% Get death probability for this generation
get_death_probability(PopulationId, IndividualId) ->
    gen_server:call(?SERVER(PopulationId), {get_death_probability, IndividualId}).

%% Process all natural deaths for the generation
process_natural_deaths(PopulationId) ->
    gen_server:call(?SERVER(PopulationId), process_natural_deaths).
```

### Core Logic

```erlang
%%====================================================================
%% Internal Functions
%%====================================================================

%% Determine lifecycle stage based on age
calculate_stage(Age, #lifecycle_config{} = Config) ->
    #lifecycle_config{
        maturity_age = MaturityAge,
        senescence_age = SenescenceAge
    } = Config,

    if
        Age < MaturityAge -> juvenile;
        Age < SenescenceAge -> fertile;
        true -> senescent
    end.

%% Check breeding eligibility
check_breeding_eligibility(#individual{} = Ind, #lifecycle_config{} = Config) ->
    #individual{
        current_age = Age,
        lifecycle_stage = Stage,
        last_breeding_generation = LastBreed,
        offspring_count = OffspringCount
    } = Ind,

    #lifecycle_config{
        min_breeding_age = MinAge,
        max_breeding_age = MaxAge,
        breeding_cooldown = Cooldown,
        max_lifetime_offspring = MaxOffspring
    } = Config,

    %% Check all conditions
    AgeOk = (Age >= MinAge) andalso (Age =< MaxAge),
    StageOk = (Stage == fertile),
    CooldownOk = case LastBreed of
        never -> true;
        Gen -> (Ind#individual.current_age - Gen) >= Cooldown
    end,
    OffspringOk = case MaxOffspring of
        unlimited -> true;
        Max -> OffspringCount < Max
    end,

    case {AgeOk, StageOk, CooldownOk, OffspringOk} of
        {true, true, true, true} -> eligible;
        {_, false, _, _} -> ineligible;  % Wrong stage
        {false, _, _, _} -> ineligible;  % Wrong age
        {_, _, false, _} -> cooldown;     % Still in cooldown
        {_, _, _, false} -> ineligible    % Max offspring reached
    end.

%% Calculate fertility score (affects mate selection)
calculate_fertility_score(#individual{} = Ind, #lifecycle_config{} = Config) ->
    #individual{current_age = Age} = Ind,
    #lifecycle_config{
        maturity_age = MaturityAge,
        senescence_age = SenescenceAge,
        peak_fertility_age = PeakAge,
        fertility_curve = Curve
    } = Config,

    case Curve of
        flat ->
            %% Constant fertility during fertile period
            1.0;

        gaussian ->
            %% Bell curve centered on peak age
            Sigma = (SenescenceAge - MaturityAge) / 4,
            math:exp(-math:pow(Age - PeakAge, 2) / (2 * Sigma * Sigma));

        declining ->
            %% Linear decline from maturity
            FertileSpan = SenescenceAge - MaturityAge,
            AgeInFertile = Age - MaturityAge,
            max(0.0, 1.0 - (AgeInFertile / FertileSpan))
    end.

%% Calculate death probability for current generation
calculate_death_probability(#individual{} = Ind, #lifecycle_config{} = Config) ->
    #individual{
        lifecycle_stage = Stage,
        current_age = Age
    } = Ind,

    #lifecycle_config{
        senescence_age = SenescenceAge,
        max_age = MaxAge,
        juvenile_death_prob = JuvDeathProb,
        fertile_death_prob = FertileDeathProb,
        senescent_death_prob_base = SenDeathBase,
        senescent_death_prob_growth = SenDeathGrowth
    } = Config,

    case Stage of
        juvenile ->
            JuvDeathProb;

        fertile ->
            FertileDeathProb;

        senescent ->
            %% Increasing probability with age
            AgeInSenescence = Age - SenescenceAge,
            BaseProb = SenDeathBase + (AgeInSenescence * SenDeathGrowth),

            %% Guaranteed death at max age
            case Age >= MaxAge of
                true -> 1.0;
                false -> min(1.0, BaseProb)
            end
    end.

%% Apply fitness decay for senescent individuals
apply_senescent_decay(#individual{} = Ind, #lifecycle_config{} = Config) ->
    case Ind#individual.lifecycle_stage of
        senescent ->
            #lifecycle_config{senescent_decay_rate = DecayRate} = Config,
            CurrentFitness = Ind#individual.fitness,
            DecayedFitness = CurrentFitness * (1.0 - DecayRate),
            Ind#individual{fitness = DecayedFitness};
        _ ->
            Ind
    end.

%% Process stage transitions
process_stage_transition(#individual{} = Ind, OldStage, NewStage, Generation) ->
    case {OldStage, NewStage} of
        {juvenile, fertile} ->
            %% Emit maturation event
            Event = #{
                event_type => individual_matured,
                individual_id => Ind#individual.id,
                previous_stage => juvenile,
                new_stage => fertile,
                age_generations => Ind#individual.current_age,
                fitness_at_maturity => Ind#individual.fitness,
                matured_at => erlang:system_time(microsecond)
            },
            neuroevolution_events:emit(individual_matured, Event);

        {fertile, senescent} ->
            %% Emit fertility waned event
            Event = #{
                event_type => fertility_waned,
                individual_id => Ind#individual.id,
                previous_stage => fertile,
                new_stage => senescent,
                age_generations => Ind#individual.current_age,
                peak_fitness => Ind#individual.peak_fitness,
                offspring_count => Ind#individual.offspring_count,
                waned_at => erlang:system_time(microsecond)
            },
            neuroevolution_events:emit(fertility_waned, Event);

        _ ->
            ok
    end.

%% Process natural death
process_natural_death(#individual{} = Ind, Cause) ->
    Event = #{
        event_type => lifespan_expired,
        individual_id => Ind#individual.id,
        population_id => Ind#individual.population_id,
        age_generations => Ind#individual.current_age,
        lifecycle_stage => Ind#individual.lifecycle_stage,
        peak_fitness => Ind#individual.peak_fitness,
        offspring_count => Ind#individual.offspring_count,
        death_cause => Cause,  % old_age | random | senescent_failure
        expired_at => erlang:system_time(microsecond)
    },
    neuroevolution_events:emit(lifespan_expired, Event).
```

---

## Integration with Evolution Loop

### Generation Cycle with Lifecycle

```erlang
%% In population_server.erl or similar
run_generation(PopulationId) ->
    %% 1. Increment age for all individuals
    lifecycle_server:tick_generation(PopulationId),

    %% 2. Process natural deaths BEFORE evaluation
    {ok, Deaths} = lifecycle_server:process_natural_deaths(PopulationId),
    lists:foreach(fun(Ind) ->
        remove_individual(PopulationId, Ind#individual.id)
    end, Deaths),

    %% 3. Evaluate survivors
    evaluate_population(PopulationId),

    %% 4. Apply aging effects (fitness decay for senescent)
    apply_all_aging_effects(PopulationId),

    %% 5. Select parents (only from fertile individuals)
    FertileParents = get_fertile_individuals(PopulationId),
    SelectedParents = selection:select_parents(FertileParents),

    %% 6. Breed with breeding eligibility check
    EligibleParents = lists:filter(fun(Ind) ->
        lifecycle_server:can_breed(PopulationId, Ind#individual.id) == eligible
    end, SelectedParents),
    Offspring = breed(EligibleParents),

    %% 7. Selection pressure (protect juveniles if configured)
    apply_selection_with_protection(PopulationId),

    %% 8. Add offspring as juveniles
    lists:foreach(fun(Child) ->
        add_individual(PopulationId, Child#individual{
            lifecycle_stage = juvenile,
            current_age = 0,
            birth_generation = get_generation(PopulationId)
        })
    end, Offspring).
```

### Selection with Juvenile Protection

```erlang
apply_selection_with_protection(PopulationId) ->
    Config = get_lifecycle_config(PopulationId),

    case Config#lifecycle_config.juvenile_protection of
        true ->
            %% Only cull from non-juveniles
            NonJuveniles = get_non_juvenile_individuals(PopulationId),
            apply_selection(NonJuveniles);
        false ->
            %% Traditional selection on all
            apply_selection(get_all_individuals(PopulationId))
    end.
```

---

## Lifecycle Strategies

Different lifecycle configurations create different evolutionary pressures:

### r-Strategy (Fast Reproduction)

```erlang
%% Many offspring, short lives, low parental investment
r_strategy() ->
    #lifecycle_config{
        maturity_age = 1,
        senescence_age = 10,
        max_age = 15,
        juvenile_protection = false,
        max_lifetime_offspring = unlimited,
        breeding_cooldown = 0,
        juvenile_learning_bonus = 1.0
    }.
```

### K-Strategy (Quality over Quantity)

```erlang
%% Few offspring, long lives, high parental investment
k_strategy() ->
    #lifecycle_config{
        maturity_age = 10,
        senescence_age = 80,
        max_age = 150,
        juvenile_protection = true,
        juvenile_learning_bonus = 2.0,
        max_lifetime_offspring = 10,
        breeding_cooldown = 5,
        senescent_wisdom_bonus = 1.8
    }.
```

### Mixed Strategy

```erlang
%% Adaptive based on population density
adaptive_strategy(PopulationDensity) when PopulationDensity < 0.3 ->
    %% Low density: r-strategy (expand quickly)
    r_strategy();
adaptive_strategy(PopulationDensity) when PopulationDensity > 0.7 ->
    %% High density: K-strategy (compete for quality)
    k_strategy();
adaptive_strategy(_) ->
    %% Medium density: balanced
    balanced_lifecycle().
```

---

## Events Emitted

| Event | Trigger | Stage Transition |
|-------|---------|------------------|
| `individual_matured` | Age reaches maturity_age | juvenile → fertile |
| `fertility_waned` | Age reaches senescence_age | fertile → senescent |
| `lifespan_expired` | Death probability triggered OR max_age reached | any → death |
| `vigor_declined` | Fitness decay applied | senescent only |
| `development_milestone` | Specific age thresholds | configurable |

---

## Implementation Phases

- [ ] **Phase 1:** Basic lifecycle stages (juvenile, fertile, senescent)
- [ ] **Phase 2:** Natural death mechanics (probability-based, max age)
- [ ] **Phase 3:** Breeding eligibility integration
- [ ] **Phase 4:** Fitness decay for senescent individuals
- [ ] **Phase 5:** Juvenile protection in selection
- [ ] **Phase 6:** Fertility curves (gaussian, declining)
- [ ] **Phase 7:** Lifecycle presets (r-strategy, K-strategy)
- [ ] **Phase 8:** Adaptive lifecycle strategies

---

## Success Criteria

- [ ] Individuals progress through lifecycle stages correctly
- [ ] Breeding restricted to fertile individuals within age range
- [ ] Natural death occurs based on configurable probabilities
- [ ] Juveniles protected from selection pressure (when configured)
- [ ] Senescent fitness decay applied correctly
- [ ] All lifecycle events emitted with complete payloads
- [ ] Lifecycle configuration is fully customizable
- [ ] Preset strategies available for common patterns

---

## References

- PLAN_BEHAVIORAL_EVENTS.md - Event definitions
- PLAN_KNOWLEDGE_TRANSFER.md - Mentorship during senescence
- PLAN_EPIGENETICS.md - Age-related epigenetic effects
- "Life history evolution" - Wikipedia
- r/K selection theory
