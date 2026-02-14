# PLAN_EPIGENETICS.md

**Status:** Planning
**Created:** 2025-12-23
**Last Updated:** 2025-12-23
**Related:** PLAN_BEHAVIORAL_EVENTS.md, PLAN_AGE_LIFECYCLE.md, PLAN_ECOLOGICAL_SILO.md

---

## Overview

This document specifies an epigenetic system for faber-tweann and faber-neuroevolution. Epigenetics allows environmental experiences to modify gene expression without changing the underlying genetic sequence - and these modifications can be inherited across generations with gradual decay.

### Biological Background

In real biology, epigenetic mechanisms (DNA methylation, histone modification) allow:

1. **Environmental Responsiveness**: Organisms adapt to conditions during their lifetime
2. **Transgenerational Inheritance**: Some adaptations pass to offspring
3. **Gradual Decay**: Marks weaken over generations if not reinforced
4. **Gene Silencing/Activation**: Same genome, different expression patterns

### Application to Neural Networks

In TWEANN, epigenetic marks modify how the genome is expressed:

- **Connection Strength Modifiers**: Scale weights without changing base values
- **Neuron Activation Modifiers**: Adjust activation function parameters
- **Structural Expression**: Enable/disable connections or neurons
- **Learning Rate Modifiers**: Affect plasticity of specific components

---

## Core Concepts

### Epigenetic Mark Structure

```erlang
-record(epigenetic_mark, {
    id :: binary(),
    mark_type :: stress | nutrition | social | environmental | learned,

    %% What this mark affects
    target_type :: connection | neuron | module | global,
    target_ids :: [binary()] | all,         % Specific components or all

    %% How the mark modifies expression
    expression_modifier :: float(),          % -1.0 to +1.0 (suppress to enhance)
    modifier_type :: multiplicative | additive | threshold,

    %% Inheritance properties
    persistence :: transient | stable | heritable,
    inheritance_probability :: float(),      % 0.0 to 1.0 (if heritable)
    decay_rate :: float(),                   % Per-generation decay

    %% Tracking
    acquired_at :: integer(),
    generation_acquired :: non_neg_integer(),
    generations_inherited :: non_neg_integer(), % 0 if original acquisition
    trigger_condition :: term(),             % What caused this mark
    strength :: float()                      % Current mark strength (0.0 to 1.0)
}).
```

### Mark Types

| Type | Trigger | Effect | Typical Duration |
|------|---------|--------|------------------|
| **stress** | Low fitness, threats, competition | Heightened sensitivity, faster reactions | 2-5 generations |
| **nutrition** | Resource abundance/scarcity | Metabolism adjustment, growth modifiers | 3-7 generations |
| **social** | Social status, group dynamics | Aggression/cooperation tendencies | 1-3 generations |
| **environmental** | Climate, habitat conditions | General adaptation markers | 5-10 generations |
| **learned** | Successful behaviors | Skill enhancement, knowledge retention | 1-2 generations |

---

## Architecture

### Epigenome Structure

```erlang
-record(epigenome, {
    individual_id :: binary(),
    marks :: [#epigenetic_mark{}],
    mark_count :: non_neg_integer(),

    %% Aggregate effects (cached for performance)
    global_modifiers :: #{
        fitness_modifier => float(),
        learning_rate_modifier => float(),
        mutation_rate_modifier => float(),
        stress_response => float()
    },

    %% History
    marks_acquired_lifetime :: non_neg_integer(),
    marks_inherited :: non_neg_integer(),
    marks_decayed :: non_neg_integer()
}).
```

### Extended Individual Record

```erlang
-record(individual, {
    %% Existing fields
    id :: binary(),
    genome :: #genome{},
    fitness :: float(),

    %% NEW: Epigenome
    epigenome :: #epigenome{},

    %% Effective genome (genome + epigenetic expression)
    expressed_genome :: #genome{} | undefined  % Computed when needed
}).
```

---

## Mark Acquisition

### Environmental Triggers

```erlang
-module(epigenetics_triggers).

%% Stress marks
-spec check_stress_trigger(Individual, Environment) -> [#epigenetic_mark{}].
check_stress_trigger(#individual{} = Ind, Environment) ->
    Marks = [],

    %% Low fitness stress
    Marks1 = case Ind#individual.fitness < Environment#env.mean_fitness * 0.5 of
        true ->
            [create_stress_mark(low_fitness, Ind) | Marks];
        false ->
            Marks
    end,

    %% Competition stress
    Marks2 = case Environment#env.population_density > 0.9 of
        true ->
            [create_stress_mark(overcrowding, Ind) | Marks1];
        false ->
            Marks1
    end,

    %% Predation stress
    Marks3 = case Ind#individual.recent_predation_events > 2 of
        true ->
            [create_stress_mark(predation_pressure, Ind) | Marks2];
        false ->
            Marks2
    end,

    Marks3.

%% Nutrition marks
check_nutrition_trigger(#individual{} = Ind, Environment) ->
    case Environment#env.resource_level of
        abundant when Environment#env.resource_level > 0.8 ->
            [create_nutrition_mark(feast, Ind)];
        scarce when Environment#env.resource_level < 0.2 ->
            [create_nutrition_mark(famine, Ind)];
        _ ->
            []
    end.

%% Social marks
check_social_trigger(#individual{} = Ind, SocialContext) ->
    Marks = [],

    %% Dominance status
    Marks1 = case SocialContext#social.dominance_rank of
        Rank when Rank =< 3 ->
            [create_social_mark(dominant, Ind) | Marks];
        Rank when Rank >= SocialContext#social.population_size - 3 ->
            [create_social_mark(subordinate, Ind) | Marks];
        _ ->
            Marks
    end,

    %% Isolation
    Marks2 = case SocialContext#social.interaction_count < 2 of
        true ->
            [create_social_mark(isolated, Ind) | Marks1];
        false ->
            Marks1
    end,

    Marks2.

%% Learned behavior marks
check_learning_trigger(#individual{} = Ind, LearningEvent) ->
    case LearningEvent of
        {skill_mastered, Skill, Proficiency} when Proficiency > 0.9 ->
            [create_learned_mark(Skill, Ind)];
        {breakthrough, Domain} ->
            [create_learned_mark({breakthrough, Domain}, Ind)];
        _ ->
            []
    end.
```

### Mark Creation

```erlang
create_stress_mark(TriggerType, #individual{} = Ind) ->
    #epigenetic_mark{
        id = generate_mark_id(),
        mark_type = stress,
        target_type = global,
        target_ids = all,
        expression_modifier = calculate_stress_modifier(TriggerType),
        modifier_type = multiplicative,
        persistence = heritable,
        inheritance_probability = 0.7,
        decay_rate = 0.2,
        acquired_at = erlang:system_time(microsecond),
        generation_acquired = Ind#individual.birth_generation + Ind#individual.current_age,
        generations_inherited = 0,
        trigger_condition = TriggerType,
        strength = 1.0
    }.

calculate_stress_modifier(low_fitness) ->
    %% Increases sensitivity/reactivity
    0.3;  % +30% to relevant weights
calculate_stress_modifier(overcrowding) ->
    %% Increases aggression
    0.2;
calculate_stress_modifier(predation_pressure) ->
    %% Increases vigilance/avoidance
    0.4.
```

---

## Expression Modification

### Applying Epigenetic Effects

```erlang
-module(epigenetics_expression).

%% Compute expressed genome from base genome + epigenome
-spec express_genome(Genome, Epigenome) -> ExpressedGenome
    when Genome :: #genome{},
         Epigenome :: #epigenome{},
         ExpressedGenome :: #genome{}.

express_genome(#genome{} = Genome, #epigenome{} = Epigenome) ->
    %% Separate marks by target type
    GlobalMarks = [M || M <- Epigenome#epigenome.marks,
                        M#epigenetic_mark.target_type == global],
    NeuronMarks = [M || M <- Epigenome#epigenome.marks,
                        M#epigenetic_mark.target_type == neuron],
    ConnectionMarks = [M || M <- Epigenome#epigenome.marks,
                            M#epigenetic_mark.target_type == connection],

    %% Apply global modifiers
    GlobalModifier = calculate_aggregate_modifier(GlobalMarks),

    %% Modify neurons
    ModifiedNeurons = lists:map(fun(Neuron) ->
        NeuronMods = get_marks_for_target(Neuron#neuron.id, NeuronMarks),
        apply_neuron_modifications(Neuron, NeuronMods, GlobalModifier)
    end, Genome#genome.neurons),

    %% Modify connections
    ModifiedConnections = lists:map(fun(Conn) ->
        ConnMods = get_marks_for_target(Conn#connection.id, ConnectionMarks),
        apply_connection_modifications(Conn, ConnMods, GlobalModifier)
    end, Genome#genome.connections),

    Genome#genome{
        neurons = ModifiedNeurons,
        connections = ModifiedConnections
    }.

%% Apply modifications to a connection
apply_connection_modifications(#connection{} = Conn, Marks, GlobalMod) ->
    %% Combine all applicable modifiers
    TotalModifier = lists:foldl(fun(Mark, Acc) ->
        case Mark#epigenetic_mark.modifier_type of
            multiplicative ->
                Acc * (1.0 + Mark#epigenetic_mark.expression_modifier *
                       Mark#epigenetic_mark.strength);
            additive ->
                Acc + (Mark#epigenetic_mark.expression_modifier *
                       Mark#epigenetic_mark.strength);
            threshold ->
                %% Binary: suppress if below threshold
                case Mark#epigenetic_mark.strength > 0.5 of
                    true -> Acc * Mark#epigenetic_mark.expression_modifier;
                    false -> Acc
                end
        end
    end, GlobalMod, Marks),

    %% Apply to weight
    Conn#connection{
        weight = Conn#connection.weight * TotalModifier
    }.

%% Apply modifications to a neuron
apply_neuron_modifications(#neuron{} = Neuron, Marks, GlobalMod) ->
    %% Modify bias
    BiasModifier = calculate_aggregate_modifier(Marks) * GlobalMod,
    ModifiedBias = Neuron#neuron.bias * BiasModifier,

    %% Potentially modify activation function parameters
    ModifiedParams = modify_activation_params(
        Neuron#neuron.activation_params,
        Marks
    ),

    Neuron#neuron{
        bias = ModifiedBias,
        activation_params = ModifiedParams
    }.
```

---

## Inheritance

### Mark Inheritance During Reproduction

```erlang
-module(epigenetics_inheritance).

%% Inherit marks from parents to offspring
-spec inherit_marks(ParentA, ParentB, Offspring, Opts) -> InheritedEpigenome
    when ParentA :: #individual{},
         ParentB :: #individual{},
         Offspring :: #individual{},
         Opts :: #{
             inheritance_mode => biparental | maternal | paternal,
             mark_combination => union | intersection | dominant
         },
         InheritedEpigenome :: #epigenome{}.

inherit_marks(ParentA, ParentB, Offspring, Opts) ->
    Mode = maps:get(inheritance_mode, Opts, biparental),
    Combination = maps:get(mark_combination, Opts, union),

    %% Get heritable marks from parents
    HeritableA = get_heritable_marks(ParentA#individual.epigenome),
    HeritableB = get_heritable_marks(ParentB#individual.epigenome),

    %% Combine based on mode
    CandidateMarks = case Mode of
        biparental ->
            combine_marks(HeritableA, HeritableB, Combination);
        maternal ->
            HeritableA;
        paternal ->
            HeritableB
    end,

    %% Probabilistic inheritance for each mark
    InheritedMarks = lists:filtermap(fun(Mark) ->
        case rand:uniform() < Mark#epigenetic_mark.inheritance_probability of
            true ->
                %% Apply decay and increment generation counter
                DecayedMark = apply_inheritance_decay(Mark),
                {true, DecayedMark#epigenetic_mark{
                    generations_inherited = Mark#epigenetic_mark.generations_inherited + 1
                }};
            false ->
                false
        end
    end, CandidateMarks),

    %% Create new epigenome for offspring
    #epigenome{
        individual_id = Offspring#individual.id,
        marks = InheritedMarks,
        mark_count = length(InheritedMarks),
        global_modifiers = calculate_global_modifiers(InheritedMarks),
        marks_acquired_lifetime = 0,
        marks_inherited = length(InheritedMarks),
        marks_decayed = 0
    }.

get_heritable_marks(#epigenome{marks = Marks}) ->
    [M || M <- Marks, M#epigenetic_mark.persistence == heritable].

combine_marks(MarksA, MarksB, union) ->
    %% All marks from both parents (may have duplicates)
    MarksA ++ MarksB;
combine_marks(MarksA, MarksB, intersection) ->
    %% Only marks present in both parents
    TypesA = sets:from_list([{M#epigenetic_mark.mark_type,
                               M#epigenetic_mark.trigger_condition} || M <- MarksA]),
    [M || M <- MarksB,
          sets:is_element({M#epigenetic_mark.mark_type,
                           M#epigenetic_mark.trigger_condition}, TypesA)];
combine_marks(MarksA, MarksB, dominant) ->
    %% Stronger parent's marks dominate
    StrengthA = lists:sum([M#epigenetic_mark.strength || M <- MarksA]),
    StrengthB = lists:sum([M#epigenetic_mark.strength || M <- MarksB]),
    case StrengthA >= StrengthB of
        true -> MarksA;
        false -> MarksB
    end.

apply_inheritance_decay(#epigenetic_mark{} = Mark) ->
    NewStrength = Mark#epigenetic_mark.strength *
                  (1.0 - Mark#epigenetic_mark.decay_rate),

    %% Check if mark should be erased
    case NewStrength < 0.1 of
        true ->
            %% Mark too weak, will be filtered out
            Mark#epigenetic_mark{strength = 0.0};
        false ->
            Mark#epigenetic_mark{strength = NewStrength}
    end.
```

---

## Decay and Maintenance

### Per-Generation Decay

```erlang
-module(epigenetics_decay).

%% Process decay for all marks each generation
-spec process_generation_decay(Epigenome, Environment) -> UpdatedEpigenome
    when Epigenome :: #epigenome{},
         Environment :: term(),
         UpdatedEpigenome :: #epigenome{}.

process_generation_decay(#epigenome{} = Epi, Environment) ->
    {ActiveMarks, DecayedMarks, ErasedMarks} = lists:foldl(
        fun(Mark, {Active, Decayed, Erased}) ->
            case should_reinforce(Mark, Environment) of
                true ->
                    %% Environmental reinforcement - no decay this generation
                    {[Mark | Active], Decayed, Erased};
                false ->
                    %% Apply decay
                    DecayedMark = apply_decay(Mark),
                    case DecayedMark#epigenetic_mark.strength < 0.1 of
                        true ->
                            %% Mark erased
                            emit_mark_erased_event(Mark),
                            {Active, Decayed, [Mark | Erased]};
                        false ->
                            emit_mark_decayed_event(Mark, DecayedMark),
                            {[DecayedMark | Active], [DecayedMark | Decayed], Erased}
                    end
            end
        end,
        {[], [], []},
        Epi#epigenome.marks
    ),

    Epi#epigenome{
        marks = ActiveMarks,
        mark_count = length(ActiveMarks),
        global_modifiers = calculate_global_modifiers(ActiveMarks),
        marks_decayed = Epi#epigenome.marks_decayed + length(ErasedMarks)
    }.

should_reinforce(#epigenetic_mark{} = Mark, Environment) ->
    %% Check if current environment matches the trigger condition
    case Mark#epigenetic_mark.trigger_condition of
        low_fitness ->
            Environment#env.individual_fitness < Environment#env.mean_fitness * 0.5;
        overcrowding ->
            Environment#env.population_density > 0.9;
        famine ->
            Environment#env.resource_level < 0.2;
        feast ->
            Environment#env.resource_level > 0.8;
        _ ->
            false
    end.

apply_decay(#epigenetic_mark{} = Mark) ->
    DecayRate = Mark#epigenetic_mark.decay_rate,
    CurrentStrength = Mark#epigenetic_mark.strength,
    NewStrength = CurrentStrength * (1.0 - DecayRate),
    Mark#epigenetic_mark{strength = max(0.0, NewStrength)}.
```

---

## Epigenetics Server

### Module: `epigenetics_server.erl`

```erlang
-module(epigenetics_server).
-behaviour(gen_server).

%% API
-export([
    start_link/1,
    acquire_mark/3,
    process_environment/2,
    inherit_to_offspring/4,
    get_expressed_genome/2,
    get_epigenome/2,
    process_generation/1
]).

-record(state, {
    population_id :: binary(),
    epigenomes :: #{binary() => #epigenome{}},
    environment :: term(),
    config :: #{
        enable_stress_marks => boolean(),
        enable_nutrition_marks => boolean(),
        enable_social_marks => boolean(),
        enable_learned_marks => boolean(),
        default_decay_rate => float(),
        default_inheritance_prob => float()
    }
}).

%%====================================================================
%% API Functions
%%====================================================================

%% Manually acquire a mark (for external triggers)
acquire_mark(PopulationId, IndividualId, Mark) ->
    gen_server:call(?SERVER(PopulationId), {acquire_mark, IndividualId, Mark}).

%% Process environmental effects on all individuals
process_environment(PopulationId, Environment) ->
    gen_server:call(?SERVER(PopulationId), {process_environment, Environment}).

%% Inherit marks during reproduction
inherit_to_offspring(PopulationId, ParentAId, ParentBId, OffspringId) ->
    gen_server:call(?SERVER(PopulationId),
                    {inherit, ParentAId, ParentBId, OffspringId}).

%% Get genome with epigenetic modifications applied
get_expressed_genome(PopulationId, IndividualId) ->
    gen_server:call(?SERVER(PopulationId), {get_expressed, IndividualId}).

%% Get raw epigenome
get_epigenome(PopulationId, IndividualId) ->
    gen_server:call(?SERVER(PopulationId), {get_epigenome, IndividualId}).

%% Process all epigenetic effects for a generation
process_generation(PopulationId) ->
    gen_server:call(?SERVER(PopulationId), process_generation).
```

---

## Module: `tweann_epigenetics.erl`

New module for faber-tweann implementing epigenetic primitives.

```erlang
-module(tweann_epigenetics).

%% Core types
-export_type([
    epigenetic_mark/0,
    epigenome/0,
    mark_type/0,
    persistence/0
]).

%% Mark management
-export([
    create_mark/3,
    add_mark/2,
    remove_mark/2,
    get_marks_by_type/2,
    get_marks_for_target/2
]).

%% Expression
-export([
    express_genome/2,
    calculate_modifier/1,
    apply_to_connection/2,
    apply_to_neuron/2
]).

%% Inheritance
-export([
    inherit/3,
    combine_epigenomes/3,
    calculate_inheritance_probability/2
]).

%% Decay
-export([
    apply_decay/1,
    process_generation_decay/2,
    should_reinforce/2
]).

%% Analysis
-export([
    summarize_epigenome/1,
    calculate_heritability/1,
    trace_mark_lineage/2
]).
```

---

## Integration with Evolution Loop

### Generation Cycle with Epigenetics

```erlang
run_generation_with_epigenetics(PopulationId) ->
    %% 1. Process environmental triggers (acquire new marks)
    Environment = get_current_environment(PopulationId),
    epigenetics_server:process_environment(PopulationId, Environment),

    %% 2. Apply epigenetic expression to all individuals
    Individuals = get_all_individuals(PopulationId),
    lists:foreach(fun(Ind) ->
        ExpressedGenome = epigenetics_server:get_expressed_genome(
            PopulationId, Ind#individual.id),
        update_expressed_genome(Ind#individual.id, ExpressedGenome)
    end, Individuals),

    %% 3. Evaluate with expressed genomes
    evaluate_population(PopulationId),

    %% 4. Selection and breeding
    {Parents, Offspring} = select_and_breed(PopulationId),

    %% 5. Inherit epigenetic marks to offspring
    lists:foreach(fun({ParentA, ParentB, Child}) ->
        epigenetics_server:inherit_to_offspring(
            PopulationId,
            ParentA#individual.id,
            ParentB#individual.id,
            Child#individual.id
        )
    end, lists:zip3(Parents, shift(Parents), Offspring)),

    %% 6. Process mark decay
    epigenetics_server:process_generation(PopulationId).
```

---

## Events Emitted

| Event | Trigger |
|-------|---------|
| `mark_acquired` | New epigenetic mark created |
| `mark_inherited` | Mark passed to offspring |
| `mark_decayed` | Mark strength reduced |
| `mark_erased` | Mark strength reached zero |
| `expression_modified` | Genome expression changed |
| `stress_response_triggered` | Stress marks activated |
| `adaptation_acquired` | Beneficial mark established |

---

## Configuration

```erlang
-record(epigenetics_config, {
    %% Enable/disable mark types
    enable_stress_marks = true :: boolean(),
    enable_nutrition_marks = true :: boolean(),
    enable_social_marks = true :: boolean(),
    enable_environmental_marks = true :: boolean(),
    enable_learned_marks = true :: boolean(),

    %% Default parameters
    default_decay_rate = 0.15 :: float(),
    default_inheritance_probability = 0.6 :: float(),
    max_marks_per_individual = 20 :: non_neg_integer(),

    %% Inheritance mode
    inheritance_mode = biparental :: biparental | maternal | paternal,
    mark_combination = union :: union | intersection | dominant,

    %% Expression limits
    max_expression_modifier = 2.0 :: float(),
    min_expression_modifier = 0.1 :: float()
}).
```

---

## Implementation Phases

- [ ] **Phase 1:** Core mark structure and storage
- [ ] **Phase 2:** Mark acquisition from environmental triggers
- [ ] **Phase 3:** Expression modification (apply marks to genome)
- [ ] **Phase 4:** Inheritance during reproduction
- [ ] **Phase 5:** Decay and reinforcement mechanics
- [ ] **Phase 6:** Epigenetics server integration
- [ ] **Phase 7:** Event emission for all operations
- [ ] **Phase 8:** tweann_epigenetics.erl module

---

## Success Criteria

- [ ] Marks acquired in response to environmental conditions
- [ ] Marks modify genome expression correctly
- [ ] Marks inherited with configurable probability
- [ ] Marks decay over generations without reinforcement
- [ ] Marks can be reinforced by matching conditions
- [ ] All epigenetic events emitted
- [ ] Integration with evolution loop complete
- [ ] Performance acceptable (< 5% overhead)

---

## References

- PLAN_BEHAVIORAL_EVENTS.md - Event definitions
- PLAN_ECOLOGICAL_SILO.md - Environmental triggers
- PLAN_AGE_LIFECYCLE.md - Age-related epigenetic effects
- "Transgenerational Epigenetic Inheritance" - Heard & Martienssen
- "Epigenetics and Evolution" - Jablonka & Lamb
