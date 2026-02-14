# PLAN_KNOWLEDGE_TRANSFER.md

**Status:** Planning
**Created:** 2025-12-23
**Last Updated:** 2025-12-23
**Related:** PLAN_BEHAVIORAL_EVENTS.md, PLAN_AGE_LIFECYCLE.md, PLAN_CULTURAL_SILO.md

---

## Overview

This document specifies non-genetic knowledge transfer mechanisms for faber-tweann and faber-neuroevolution. Unlike genetic inheritance (crossover/mutation), knowledge transfer allows information to flow between individuals during their lifetime - enabling cultural evolution, mentorship, and accelerated learning.

### Motivation

Genetic evolution is slow - beneficial traits take many generations to spread. Real biological and social systems accelerate adaptation through:

1. **Imitation**: Copying observed successful behaviors
2. **Teaching**: Active knowledge transfer from expert to novice
3. **Cultural Inheritance**: Passing learned behaviors to offspring
4. **Social Learning**: Learning from group behavior patterns

In neural networks, this translates to transferring network weights, structures, or behaviors between individuals without genetic reproduction.

---

## Transfer Methods

### Overview Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                    KNOWLEDGE TRANSFER METHODS                        │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐   │
│  │   BEHAVIORAL     │  │     WEIGHT       │  │   STRUCTURAL     │   │
│  │    CLONING       │  │    GRAFTING      │  │    SEEDING       │   │
│  │                  │  │                  │  │                  │   │
│  │  Copy input→     │  │  Copy specific   │  │  Copy network    │   │
│  │  output mapping  │  │  weight values   │  │  topology only   │   │
│  │                  │  │                  │  │                  │   │
│  │  "Do what I do"  │  │  "Think like me" │  │  "Wire like me"  │   │
│  └──────────────────┘  └──────────────────┘  └──────────────────┘   │
│           │                    │                    │               │
│           ▼                    ▼                    ▼               │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │                    TRANSFER FIDELITY                          │   │
│  │                                                               │   │
│  │   0.0 ─────────────────────────────────────────────────► 1.0  │   │
│  │   Noisy copy                                     Perfect copy │   │
│  │   (exploration)                                 (exploitation)│   │
│  └──────────────────────────────────────────────────────────────┘   │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Method 1: Behavioral Cloning

### Concept

Behavioral cloning trains the student network to reproduce the mentor's input-output mapping without direct access to the mentor's weights. The student learns by observing mentor behavior.

### Algorithm

```erlang
%% In tweann_transfer.erl

-spec behavioral_cloning(MentorPid, StudentPid, TrainingData, Opts) -> Result
    when MentorPid :: pid(),
         StudentPid :: pid(),
         TrainingData :: [{Input :: [float()], Context :: term()}],
         Opts :: #{
             epochs => non_neg_integer(),
             learning_rate => float(),
             noise_std => float(),      % Add noise to mentor outputs
             batch_size => non_neg_integer()
         },
         Result :: {ok, #{fitness_improvement => float()}} | {error, term()}.

behavioral_cloning(MentorPid, StudentPid, TrainingData, Opts) ->
    Epochs = maps:get(epochs, Opts, 100),
    LearningRate = maps:get(learning_rate, Opts, 0.01),
    NoiseStd = maps:get(noise_std, Opts, 0.0),
    BatchSize = maps:get(batch_size, Opts, 32),

    %% Generate mentor demonstrations
    Demonstrations = lists:map(fun({Input, _Context}) ->
        MentorOutput = brain:evaluate(MentorPid, Input),
        NoisyOutput = add_gaussian_noise(MentorOutput, NoiseStd),
        {Input, NoisyOutput}
    end, TrainingData),

    %% Train student to mimic mentor
    StudentFitnessBefore = get_fitness(StudentPid),

    lists:foldl(fun(_Epoch, Acc) ->
        Batches = create_batches(Demonstrations, BatchSize),
        lists:foldl(fun(Batch, InnerAcc) ->
            train_on_batch(StudentPid, Batch, LearningRate),
            InnerAcc
        end, Acc, Batches)
    end, ok, lists:seq(1, Epochs)),

    StudentFitnessAfter = get_fitness(StudentPid),

    %% Emit event
    emit_transfer_event(behavioral_cloning, MentorPid, StudentPid,
                        StudentFitnessBefore, StudentFitnessAfter),

    {ok, #{
        fitness_before => StudentFitnessBefore,
        fitness_after => StudentFitnessAfter,
        fitness_improvement => StudentFitnessAfter - StudentFitnessBefore
    }}.
```

### Use Cases

- **Juvenile Learning**: Young individuals learn from experienced adults
- **Skill Transfer**: Specific domain expertise (navigation, combat, foraging)
- **Recovery**: Helping struggling individuals catch up

---

## Method 2: Weight Grafting

### Concept

Weight grafting directly copies weight values from mentor to student for aligned network components. Requires compatible network structures.

### Algorithm

```erlang
-spec weight_grafting(MentorGenome, StudentGenome, Opts) -> Result
    when MentorGenome :: #genome{},
         StudentGenome :: #genome{},
         Opts :: #{
             graft_ratio => float(),      % 0.0-1.0, how much to copy
             target_layers => all | [LayerIndex :: integer()],
             blend_mode => replace | average | weighted,
             compatibility_threshold => float()
         },
         Result :: {ok, #genome{}} | {error, incompatible}.

weight_grafting(MentorGenome, StudentGenome, Opts) ->
    GraftRatio = maps:get(graft_ratio, Opts, 0.5),
    TargetLayers = maps:get(target_layers, Opts, all),
    BlendMode = maps:get(blend_mode, Opts, weighted),
    CompatThreshold = maps:get(compatibility_threshold, Opts, 0.7),

    %% Check structural compatibility
    Compatibility = calculate_structural_compatibility(MentorGenome, StudentGenome),
    case Compatibility < CompatThreshold of
        true ->
            {error, incompatible};
        false ->
            %% Find aligned neurons/connections
            AlignedComponents = find_aligned_components(MentorGenome, StudentGenome),

            %% Graft weights based on options
            GraftedGenome = lists:foldl(fun({MentorComp, StudentComp}, AccGenome) ->
                case should_graft(MentorComp, TargetLayers) of
                    true ->
                        NewWeight = blend_weights(
                            get_weight(MentorComp),
                            get_weight(StudentComp),
                            GraftRatio,
                            BlendMode
                        ),
                        set_weight(AccGenome, StudentComp, NewWeight);
                    false ->
                        AccGenome
                end
            end, StudentGenome, AlignedComponents),

            {ok, GraftedGenome}
    end.

%% Blending strategies
blend_weights(MentorW, StudentW, Ratio, replace) ->
    %% Full replacement with probability = Ratio
    case rand:uniform() < Ratio of
        true -> MentorW;
        false -> StudentW
    end;

blend_weights(MentorW, StudentW, Ratio, average) ->
    %% Weighted average
    (MentorW * Ratio) + (StudentW * (1.0 - Ratio));

blend_weights(MentorW, StudentW, Ratio, weighted) ->
    %% Average with noise for exploration
    Base = (MentorW * Ratio) + (StudentW * (1.0 - Ratio)),
    Noise = rand:normal() * 0.1 * abs(MentorW - StudentW),
    Base + Noise.
```

### Compatibility Calculation

```erlang
calculate_structural_compatibility(GenomeA, GenomeB) ->
    %% Count matching vs disjoint/excess genes
    InnovationsA = get_innovation_numbers(GenomeA),
    InnovationsB = get_innovation_numbers(GenomeB),

    Matching = sets:intersection(
        sets:from_list(InnovationsA),
        sets:from_list(InnovationsB)
    ),
    MatchingCount = sets:size(Matching),

    TotalA = length(InnovationsA),
    TotalB = length(InnovationsB),
    MaxGenes = max(TotalA, TotalB),

    %% Compatibility = matching / max_genes
    case MaxGenes of
        0 -> 1.0;  % Both empty = compatible
        _ -> MatchingCount / MaxGenes
    end.
```

### Use Cases

- **Preserving Expertise**: Copy proven weight patterns
- **Module Transplant**: Transfer specific functional modules
- **Hybridization**: Combine strengths of multiple individuals

---

## Method 3: Structural Seeding

### Concept

Structural seeding copies the network topology (neurons, connections) without copying weights. The student inherits the "architecture" but must learn its own weights.

### Algorithm

```erlang
-spec structural_seeding(MentorGenome, Opts) -> Result
    when MentorGenome :: #genome{},
         Opts :: #{
             include_layers => all | [LayerIndex :: integer()],
             preserve_innovations => boolean(),
             weight_init => zero | random | small_random,
             topology_noise => float()   % Probability of mutations
         },
         Result :: {ok, #genome{}}.

structural_seeding(MentorGenome, Opts) ->
    IncludeLayers = maps:get(include_layers, Opts, all),
    PreserveInnovations = maps:get(preserve_innovations, Opts, true),
    WeightInit = maps:get(weight_init, Opts, small_random),
    TopoNoise = maps:get(topology_noise, Opts, 0.0),

    %% Copy structure
    Neurons = filter_by_layers(MentorGenome#genome.neurons, IncludeLayers),
    Connections = filter_connections_by_neurons(
        MentorGenome#genome.connections,
        Neurons
    ),

    %% Reinitialize weights
    NewConnections = lists:map(fun(Conn) ->
        NewWeight = init_weight(WeightInit),
        Conn#connection{weight = NewWeight}
    end, Connections),

    NewNeurons = lists:map(fun(Neuron) ->
        NewBias = init_weight(WeightInit),
        Neuron#neuron{bias = NewBias}
    end, Neurons),

    %% Optionally add topology noise (mutations)
    SeededGenome = #genome{
        neurons = NewNeurons,
        connections = NewConnections
    },

    NoisyGenome = apply_topology_noise(SeededGenome, TopoNoise),

    %% Handle innovation numbers
    FinalGenome = case PreserveInnovations of
        true -> NoisyGenome;
        false -> reassign_innovations(NoisyGenome)
    end,

    {ok, FinalGenome}.

init_weight(zero) -> 0.0;
init_weight(random) -> rand:uniform() * 2.0 - 1.0;
init_weight(small_random) -> rand:normal() * 0.1.
```

### Use Cases

- **Architecture Discovery**: Successful topologies spread faster than weights
- **Speciation Control**: Maintain structural diversity while allowing weight variation
- **Hybrid Offspring**: Combine structures from multiple successful parents

---

## Mentorship System

### Mentor-Student Relationships

```erlang
-record(mentorship, {
    id :: binary(),
    mentor_id :: binary(),
    student_id :: binary(),
    population_id :: binary(),
    started_at :: integer(),
    ended_at :: integer() | active,
    transfer_method :: behavioral_cloning | weight_grafting | structural_seeding,
    sessions_count :: non_neg_integer(),
    total_knowledge_transferred :: float(),
    student_fitness_start :: float(),
    student_fitness_current :: float()
}).
```

### Mentor Selection Criteria

```erlang
-spec select_mentor(StudentId, Population, Opts) -> {ok, MentorId} | no_suitable_mentor
    when StudentId :: binary(),
         Population :: #population{},
         Opts :: #{
             min_fitness_ratio => float(),    % Mentor must be this much better
             max_age_difference => non_neg_integer(),
             prefer_kin => boolean(),
             prefer_same_species => boolean(),
             mentor_capacity => non_neg_integer()  % Max students per mentor
         }.

select_mentor(StudentId, Population, Opts) ->
    Student = get_individual(Population, StudentId),

    %% Filter eligible mentors
    Candidates = lists:filter(fun(Ind) ->
        is_eligible_mentor(Ind, Student, Opts)
    end, Population#population.individuals),

    case Candidates of
        [] ->
            no_suitable_mentor;
        _ ->
            %% Score and select best mentor
            Scored = lists:map(fun(Mentor) ->
                Score = calculate_mentor_score(Mentor, Student, Opts),
                {Score, Mentor}
            end, Candidates),
            {_, BestMentor} = lists:max(Scored),
            {ok, BestMentor#individual.id}
    end.

is_eligible_mentor(Mentor, Student, Opts) ->
    MinFitnessRatio = maps:get(min_fitness_ratio, Opts, 1.2),
    MaxAgeDiff = maps:get(max_age_difference, Opts, infinity),
    MentorCapacity = maps:get(mentor_capacity, Opts, 3),

    %% Must be significantly fitter
    FitnessOk = Mentor#individual.fitness >
                Student#individual.fitness * MinFitnessRatio,

    %% Age difference check
    AgeDiff = abs(Mentor#individual.current_age - Student#individual.current_age),
    AgeOk = AgeDiff =< MaxAgeDiff,

    %% Not already at capacity
    CurrentStudents = count_current_students(Mentor#individual.id),
    CapacityOk = CurrentStudents < MentorCapacity,

    %% Not self
    NotSelf = Mentor#individual.id =/= Student#individual.id,

    FitnessOk andalso AgeOk andalso CapacityOk andalso NotSelf.

calculate_mentor_score(Mentor, Student, Opts) ->
    PreferKin = maps:get(prefer_kin, Opts, true),
    PreferSameSpecies = maps:get(prefer_same_species, Opts, true),

    %% Base score from fitness difference
    FitnessScore = Mentor#individual.fitness - Student#individual.fitness,

    %% Kin bonus
    KinBonus = case PreferKin of
        true ->
            Relatedness = calculate_relatedness(Mentor, Student),
            Relatedness * 0.5;  % Up to 0.5 bonus for close relatives
        false ->
            0.0
    end,

    %% Species bonus
    SpeciesBonus = case PreferSameSpecies of
        true when Mentor#individual.species_id == Student#individual.species_id ->
            0.3;
        _ ->
            0.0
    end,

    %% Senescent wisdom bonus (older mentors may be better teachers)
    WisdomBonus = case Mentor#individual.lifecycle_stage of
        senescent -> 0.2;
        _ -> 0.0
    end,

    FitnessScore + KinBonus + SpeciesBonus + WisdomBonus.
```

### Mentorship Sessions

```erlang
-spec conduct_mentorship_session(Mentorship, Opts) -> Result
    when Mentorship :: #mentorship{},
         Opts :: #{
             session_duration => non_neg_integer(),  % Training iterations
             transfer_intensity => float()           % How much to transfer
         },
         Result :: {ok, UpdatedMentorship} | {error, term()}.

conduct_mentorship_session(#mentorship{} = M, Opts) ->
    Mentor = get_individual(M#mentorship.mentor_id),
    Student = get_individual(M#mentorship.student_id),

    %% Conduct transfer based on method
    TransferResult = case M#mentorship.transfer_method of
        behavioral_cloning ->
            TrainingData = generate_training_scenarios(100),
            behavioral_cloning(Mentor, Student, TrainingData, Opts);

        weight_grafting ->
            GraftOpts = Opts#{graft_ratio => maps:get(transfer_intensity, Opts, 0.3)},
            weight_grafting(Mentor#individual.genome, Student#individual.genome, GraftOpts);

        structural_seeding ->
            %% Usually done once at relationship start
            {ok, already_seeded}
    end,

    case TransferResult of
        {ok, Result} ->
            %% Update mentorship record
            UpdatedM = M#mentorship{
                sessions_count = M#mentorship.sessions_count + 1,
                total_knowledge_transferred =
                    M#mentorship.total_knowledge_transferred +
                    maps:get(fitness_improvement, Result, 0.0),
                student_fitness_current = get_fitness(Student)
            },

            %% Emit event
            emit_teaching_event(M, Result),

            {ok, UpdatedM};

        {error, Reason} ->
            {error, Reason}
    end.
```

---

## Transfer Server

### Module: `transfer_server.erl`

```erlang
-module(transfer_server).
-behaviour(gen_server).

%% API
-export([
    start_link/1,
    request_mentorship/3,
    end_mentorship/2,
    imitate/3,
    clone_behavior/4,
    graft_weights/4,
    seed_structure/3,
    get_mentorships/1,
    get_transfer_stats/1
]).

-record(state, {
    population_id :: binary(),
    active_mentorships :: #{binary() => #mentorship{}},
    transfer_history :: [map()],
    config :: map()
}).

%%====================================================================
%% Public API
%%====================================================================

%% Request a mentor for a student
request_mentorship(PopulationId, StudentId, Opts) ->
    gen_server:call(?SERVER(PopulationId), {request_mentorship, StudentId, Opts}).

%% End a mentorship relationship
end_mentorship(PopulationId, MentorshipId) ->
    gen_server:call(?SERVER(PopulationId), {end_mentorship, MentorshipId}).

%% Quick imitation (one-shot learning from observation)
imitate(PopulationId, ObserverId, TargetId) ->
    gen_server:call(?SERVER(PopulationId), {imitate, ObserverId, TargetId}).

%% Full behavioral cloning session
clone_behavior(PopulationId, MentorId, StudentId, Opts) ->
    gen_server:call(?SERVER(PopulationId), {clone_behavior, MentorId, StudentId, Opts}).

%% Weight grafting between individuals
graft_weights(PopulationId, DonorId, RecipientId, Opts) ->
    gen_server:call(?SERVER(PopulationId), {graft_weights, DonorId, RecipientId, Opts}).

%% Structural seeding for new individual
seed_structure(PopulationId, TemplateId, Opts) ->
    gen_server:call(?SERVER(PopulationId), {seed_structure, TemplateId, Opts}).
```

---

## Integration with Lifecycle

### Juvenile Learning Phase

```erlang
%% During juvenile stage, prioritize learning
process_juvenile(#individual{} = Juvenile, Population, Config) ->
    case Config#lifecycle_config.juvenile_learning_bonus > 1.0 of
        true ->
            %% Find mentor for this juvenile
            case transfer_server:request_mentorship(
                    Population#population.id,
                    Juvenile#individual.id,
                    #{prefer_kin => true}
                 ) of
                {ok, MentorshipId} ->
                    %% Conduct learning sessions
                    conduct_juvenile_learning(MentorshipId, Config);
                no_suitable_mentor ->
                    %% Learn through exploration alone
                    ok
            end;
        false ->
            ok
    end.
```

### Senescent Mentoring Phase

```erlang
%% During senescent stage, prioritize teaching
process_senescent(#individual{} = Elder, Population, Config) ->
    case Config#lifecycle_config.senescent_wisdom_bonus > 1.0 of
        true ->
            %% Find students who could benefit
            PotentialStudents = find_struggling_juveniles(Population),

            %% Offer mentorship
            lists:foreach(fun(StudentId) ->
                transfer_server:request_mentorship(
                    Population#population.id,
                    StudentId,
                    #{preferred_mentor => Elder#individual.id}
                )
            end, PotentialStudents);
        false ->
            ok
    end.
```

---

## Events Emitted

| Event | Trigger |
|-------|---------|
| `mentor_assigned` | Mentorship relationship established |
| `mentorship_concluded` | Mentorship ended (natural or forced) |
| `teaching_occurred` | Mentorship session conducted |
| `knowledge_transferred` | Successful transfer of any type |
| `skill_imitated` | Quick imitation completed |
| `behavior_cloned` | Full behavioral cloning session |
| `weights_grafted` | Weight grafting completed |
| `structure_seeded` | Structural seeding completed |
| `learning_succeeded` | Transfer improved student fitness |
| `learning_failed` | Transfer did not improve fitness |

---

## Module: `tweann_transfer.erl`

New module for faber-tweann implementing transfer primitives.

```erlang
-module(tweann_transfer).

%% Core transfer functions
-export([
    behavioral_cloning/4,
    weight_grafting/3,
    structural_seeding/2
]).

%% Utility functions
-export([
    calculate_compatibility/2,
    find_aligned_components/2,
    generate_demonstrations/3,
    blend_weights/4
]).

%% Analysis functions
-export([
    measure_transfer_fidelity/3,
    estimate_transfer_benefit/3,
    find_transferable_modules/2
]).
```

---

## Implementation Phases

- [ ] **Phase 1:** Basic behavioral cloning
- [ ] **Phase 2:** Weight grafting with compatibility check
- [ ] **Phase 3:** Structural seeding
- [ ] **Phase 4:** Mentorship system (mentor selection, sessions)
- [ ] **Phase 5:** Integration with lifecycle (juvenile learning, senescent mentoring)
- [ ] **Phase 6:** Transfer server with async operations
- [ ] **Phase 7:** Transfer fidelity metrics and analysis
- [ ] **Phase 8:** Advanced blending strategies

---

## Success Criteria

- [ ] All three transfer methods implemented and tested
- [ ] Mentorship system with mentor selection and capacity limits
- [ ] Integration with lifecycle stages
- [ ] Events emitted for all transfer activities
- [ ] Transfer improves student fitness measurably
- [ ] Compatible with existing TWEANN genome structure
- [ ] Transfer_server handles concurrent mentorships

---

## References

- PLAN_BEHAVIORAL_EVENTS.md - Transfer event definitions
- PLAN_AGE_LIFECYCLE.md - Lifecycle integration
- PLAN_CULTURAL_SILO.md - Cultural transmission
- "Behavioral Cloning from Observation" - Torabi et al.
- "Policy Distillation" - Rusu et al.
- "Progressive Neural Networks" - Rusu et al.
