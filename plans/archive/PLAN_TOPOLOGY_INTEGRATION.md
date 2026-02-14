# Topology Evolution Integration Plan - faber-neuroevolution

**Date:** 2025-12-07
**Priority:** 1
**Status:** COMPLETE ✅
**Completed:** 2025-12-07
**Related:** faber-tweann/plans/PLAN_TOPOLOGY_EVOLUTION.md
**Prerequisite:** faber-tweann topology evolution phases T1-T3 ✅

---

## Overview

Integrate full TWEANN (Topology and Weight Evolving Artificial Neural Networks) capabilities from faber-tweann into faber-neuroevolution. This transforms the evolution engine from weight-only evolution to full structural evolution.

---

## Implementation Summary

All NEAT topology evolution features have been integrated successfully.

### What's Implemented ✅

| Component | Status |
|-----------|--------|
| Weight mutation | Complete (gaussian perturbation) |
| Weight crossover | Complete (uniform crossover) |
| Fixed topology networks | Complete |
| Evolution strategies | Complete (5 strategies with genome support) |
| Meta-controller | Complete (LTC-based with topology awareness) |
| Speciation | Complete (NEAT compatibility distance) |
| **Topological mutations** | ✅ Complete (add_node, add_connection, toggle) |
| **Variable topology crossover** | ✅ Complete (NEAT-style gene alignment) |
| **Innovation tracking** | ✅ Complete (via faber_tweann innovation.erl) |
| **Topology-aware speciation** | ✅ Complete (NEAT distance formula) |
| **Meta-controller topology control** | ✅ Complete (topology inputs/outputs) |

### Key Files Created

- `src/genome_factory.erl` - Genome creation, mutation, network conversion
- `test/unit/genome_factory_tests.erl` - Unit tests for genome operations

---

## Phase N1: Network Representation Upgrade

**Goal:** Switch from simple weight arrays to gene-based genome representation

### Current Representation

```erlang
%% network_factory.erl - Current approach
Network = network_evaluator:create_feedforward(InputSize, HiddenLayers, OutputSize),
Weights = network_evaluator:get_weights(Network),  % Simple float list
```

### Target Representation

```erlang
%% New approach - gene-based genome
-record(genome, {
    id :: genome_id(),
    node_genes :: [#node_gene{}],
    connection_genes :: [#connection_gene{}],
    innovation_history :: [pos_integer()]
}).

-record(node_gene, {
    id :: node_id(),
    innovation :: pos_integer(),
    node_type :: input | hidden | output | bias,
    activation_function :: atom(),
    %% LTC fields (optional)
    neuron_type = standard :: standard | ltc | cfc,
    time_constant = 1.0 :: float(),
    state_bound = 1.0 :: float()
}).

-record(connection_gene, {
    innovation :: pos_integer(),
    from_node :: node_id(),
    to_node :: node_id(),
    weight :: float(),
    enabled = true :: boolean()
}).
```

### Implementation

#### N1.1: Genome Record

**File:** `include/neuroevolution.hrl`

Add genome records:

```erlang
%%% ============================================================================
%%% Genome Records (TWEANN support)
%%% ============================================================================

-type node_id() :: pos_integer().
-type innovation_number() :: pos_integer().

-record(node_gene, {
    id :: node_id(),
    innovation :: innovation_number(),
    node_type :: input | hidden | output | bias,
    layer = 0 :: non_neg_integer(),  % For feedforward ordering
    activation_function = tanh :: atom(),
    %% LTC neuron fields
    neuron_type = standard :: standard | ltc | cfc,
    time_constant = 1.0 :: float(),
    state_bound = 1.0 :: float()
}).

-record(connection_gene, {
    innovation :: innovation_number(),
    from_node :: node_id(),
    to_node :: node_id(),
    weight :: float(),
    enabled = true :: boolean()
}).

-record(genome, {
    id :: term(),
    node_genes = [] :: [#node_gene{}],
    connection_genes = [] :: [#connection_gene{}],
    fitness = 0.0 :: float(),
    adjusted_fitness = 0.0 :: float(),
    species_id :: species_id() | undefined
}).

-type genome() :: #genome{}.
```

#### N1.2: Genome Factory

**File:** `src/genome_factory.erl` (NEW)

```erlang
-module(genome_factory).

-export([
    create_minimal/1,
    create_from_topology/1,
    to_network/1,
    from_network/2,
    mutate/2,
    crossover/3
]).

%% @doc Create minimal genome (inputs directly connected to outputs).
%% This is the NEAT starting point - networks grow from here.
-spec create_minimal(Config) -> genome() when
    Config :: neuro_config().
create_minimal(Config) ->
    {InputSize, _Hidden, OutputSize} = Config#neuro_config.network_topology,

    %% Create input nodes (innovations 1..InputSize)
    InputNodes = [#node_gene{
        id = I,
        innovation = I,
        node_type = input,
        layer = 0
    } || I <- lists:seq(1, InputSize)],

    %% Create output nodes (innovations InputSize+1..InputSize+OutputSize)
    OutputNodes = [#node_gene{
        id = InputSize + I,
        innovation = InputSize + I,
        node_type = output,
        layer = 1
    } || I <- lists:seq(1, OutputSize)],

    %% Create connections (all inputs to all outputs)
    Connections = [#connection_gene{
        innovation = (In - 1) * OutputSize + Out,
        from_node = In,
        to_node = InputSize + Out,
        weight = random_weight()
    } || In <- lists:seq(1, InputSize),
         Out <- lists:seq(1, OutputSize)],

    #genome{
        id = make_ref(),
        node_genes = InputNodes ++ OutputNodes,
        connection_genes = Connections
    }.

%% @doc Convert genome to network_evaluator format for evaluation.
-spec to_network(Genome) -> network() when
    Genome :: genome().
to_network(Genome) ->
    %% Build adjacency list and weight matrix from connection genes
    %% Then construct network_evaluator-compatible structure
    genome_to_network:convert(Genome).

%% @doc Mutate genome (structural + weight mutations).
-spec mutate(Genome, Config) -> MutatedGenome when
    Genome :: genome(),
    Config :: mutation_config(),
    MutatedGenome :: genome().
mutate(Genome, Config) ->
    G1 = maybe_add_node(Genome, Config),
    G2 = maybe_add_connection(G1, Config),
    G3 = mutate_weights(G2, Config),
    G4 = maybe_toggle_connection(G3, Config),
    G4.
```

#### N1.3: Genome-Network Conversion

**File:** `src/genome_to_network.erl` (NEW)

```erlang
-module(genome_to_network).

-export([convert/1, convert_back/2]).

%% @doc Convert genome to network for evaluation.
%% Builds the network structure from node and connection genes.
-spec convert(Genome) -> Network when
    Genome :: genome(),
    Network :: network_evaluator:network().
convert(Genome) ->
    %% 1. Sort nodes by layer
    SortedNodes = sort_by_layer(Genome#genome.node_genes),

    %% 2. Extract topology from node structure
    {InputSize, HiddenLayers, OutputSize} = extract_topology(SortedNodes),

    %% 3. Create base network
    Network = network_evaluator:create_feedforward(InputSize, HiddenLayers, OutputSize),

    %% 4. Set weights from connection genes
    Weights = extract_weights(Genome, SortedNodes),
    network_evaluator:set_weights(Network, Weights).
```

### Deliverables

- [x] Genome records in `neuroevolution.hrl`
- [x] `genome_factory.erl` module
- [x] Network conversion in `genome_factory:to_network/1`
- [x] Unit tests for genome operations

---

## Phase N2: Topological Mutation Integration

**Goal:** Connect faber-tweann's mutation operators to the evolution engine

### Implementation

#### N2.1: Mutation Dispatcher

**File:** `src/genome_mutations.erl` (NEW)

```erlang
-module(genome_mutations).

-export([
    apply_mutations/2,
    add_node/2,
    add_connection/2,
    mutate_weights/2,
    toggle_connection/2,
    add_sensor/2,
    add_actuator/2
]).

%% @doc Apply mutations based on configuration probabilities.
-spec apply_mutations(Genome, Config) -> MutatedGenome when
    Genome :: genome(),
    Config :: mutation_config(),
    MutatedGenome :: genome().
apply_mutations(Genome, Config) ->
    %% Structural mutations (less frequent)
    G1 = case rand:uniform() < Config#mutation_config.add_node_rate of
        true -> add_node(Genome, Config);
        false -> Genome
    end,

    G2 = case rand:uniform() < Config#mutation_config.add_connection_rate of
        true -> add_connection(G1, Config);
        false -> G1
    end,

    %% Sensor/actuator mutations (rare)
    G3 = case rand:uniform() < Config#mutation_config.add_sensor_rate of
        true -> add_sensor(G2, Config);
        false -> G2
    end,

    G4 = case rand:uniform() < Config#mutation_config.add_actuator_rate of
        true -> add_actuator(G3, Config);
        false -> G3
    end,

    %% Weight mutations (frequent)
    G5 = mutate_weights(G4, Config),

    %% Connection toggle (medium)
    G6 = case rand:uniform() < Config#mutation_config.toggle_rate of
        true -> toggle_connection(G5, Config);
        false -> G5
    end,

    G6.

%% @doc Add a node by splitting an existing connection.
add_node(Genome, _Config) ->
    EnabledConnections = [C || C <- Genome#genome.connection_genes,
                               C#connection_gene.enabled],
    case EnabledConnections of
        [] -> Genome;
        Connections ->
            %% Select random connection to split
            Conn = lists:nth(rand:uniform(length(Connections)), Connections),

            %% Get innovation numbers for new structure
            {NodeInn, InInn, OutInn} =
                innovation:get_or_create_node_innovation(Conn#connection_gene.innovation),

            %% Create new node
            NewNode = #node_gene{
                id = NodeInn,
                innovation = NodeInn,
                node_type = hidden,
                layer = calculate_layer(Conn, Genome)
            },

            %% Disable old connection
            UpdatedConns = disable_connection(Conn, Genome#genome.connection_genes),

            %% Create two new connections
            InConn = #connection_gene{
                innovation = InInn,
                from_node = Conn#connection_gene.from_node,
                to_node = NodeInn,
                weight = 1.0  % Pass through initially
            },
            OutConn = #connection_gene{
                innovation = OutInn,
                from_node = NodeInn,
                to_node = Conn#connection_gene.to_node,
                weight = Conn#connection_gene.weight
            },

            Genome#genome{
                node_genes = [NewNode | Genome#genome.node_genes],
                connection_genes = [InConn, OutConn | UpdatedConns]
            }
    end.
```

#### N2.2: Mutation Configuration

**File:** `include/neuroevolution.hrl`

```erlang
%% @doc Configuration for genome mutations.
-record(mutation_config, {
    %% Weight mutation
    weight_mutation_rate = 0.80 :: float(),    % 80% of offspring mutated
    weight_perturb_rate = 0.90 :: float(),     % 90% perturbed, 10% replaced
    weight_perturb_strength = 0.3 :: float(),

    %% Structural mutation
    add_node_rate = 0.03 :: float(),           % 3% chance per reproduction
    add_connection_rate = 0.05 :: float(),     % 5% chance per reproduction
    toggle_rate = 0.01 :: float(),             % 1% chance to toggle enable

    %% Sensor/actuator mutation (rare - morphological change)
    add_sensor_rate = 0.001 :: float(),        % 0.1% chance
    add_actuator_rate = 0.001 :: float(),      % 0.1% chance

    %% LTC mutation (when LTC neurons present)
    mutate_time_constant_rate = 0.05 :: float(),
    mutate_neuron_type_rate = 0.01 :: float()
}).

-type mutation_config() :: #mutation_config{}.
```

### Deliverables

- [x] Mutations in `genome_factory.erl`
- [x] `mutation_config` record in `neuroevolution.hrl`
- [x] Integration with evolution strategies
- [x] Unit tests for each mutation type

---

## Phase N3: NEAT Crossover Integration

**Goal:** Enable crossover between genomes with different topologies

### Implementation

#### N3.1: Genome Crossover

**File:** `src/genome_crossover.erl` (NEW)

```erlang
-module(genome_crossover).

-export([
    crossover/3,
    align_by_innovation/2
]).

%% @doc Perform NEAT-style crossover.
%% Matching genes: randomly select from either parent
%% Disjoint/Excess: inherit from fitter parent
-spec crossover(Parent1, Parent2, FitterParent) -> Child when
    Parent1 :: genome(),
    Parent2 :: genome(),
    FitterParent :: 1 | 2 | equal,
    Child :: genome().
crossover(Parent1, Parent2, FitterParent) ->
    %% Align connection genes by innovation number
    {Matching, Disjoint1, Excess1, Disjoint2, Excess2} =
        align_by_innovation(
            Parent1#genome.connection_genes,
            Parent2#genome.connection_genes
        ),

    %% Build child connections
    ChildConnections = build_child_connections(
        Matching, Disjoint1, Excess1, Disjoint2, Excess2, FitterParent
    ),

    %% Build child nodes (union of nodes referenced by connections)
    ChildNodes = build_child_nodes(
        ChildConnections,
        Parent1#genome.node_genes,
        Parent2#genome.node_genes
    ),

    #genome{
        id = make_ref(),
        node_genes = ChildNodes,
        connection_genes = ChildConnections
    }.

%% @doc Align genes by innovation number.
%% Returns {Matching, Disjoint1, Excess1, Disjoint2, Excess2}
align_by_innovation(Genes1, Genes2) ->
    %% Sort by innovation
    Sorted1 = lists:keysort(#connection_gene.innovation, Genes1),
    Sorted2 = lists:keysort(#connection_gene.innovation, Genes2),

    %% Find max innovation in each
    Max1 = case Sorted1 of
        [] -> 0;
        _ -> (lists:last(Sorted1))#connection_gene.innovation
    end,
    Max2 = case Sorted2 of
        [] -> 0;
        _ -> (lists:last(Sorted2))#connection_gene.innovation
    end,
    MaxCommon = min(Max1, Max2),

    %% Separate into matching, disjoint, excess
    do_align(Sorted1, Sorted2, MaxCommon, [], [], [], [], []).
```

#### N3.2: Update Genetic Operators

**File:** `src/neuroevolution_genetic.erl`

Add genome-aware crossover:

```erlang
%% @doc Create offspring from two parent genomes.
%% Uses NEAT-style crossover for variable topologies.
-spec create_offspring_genome(Parent1, Parent2, Config, Generation) -> Offspring when
    Parent1 :: individual(),
    Parent2 :: individual(),
    Config :: neuro_config(),
    Generation :: generation(),
    Offspring :: individual().
create_offspring_genome(Parent1, Parent2, Config, Generation) ->
    %% Determine fitter parent
    FitterParent = case Parent1#individual.fitness > Parent2#individual.fitness of
        true -> 1;
        false ->
            case Parent1#individual.fitness < Parent2#individual.fitness of
                true -> 2;
                false -> equal
            end
    end,

    %% NEAT crossover
    ChildGenome = genome_crossover:crossover(
        Parent1#individual.genome,
        Parent2#individual.genome,
        FitterParent
    ),

    %% Apply mutations
    MutationConfig = Config#neuro_config.mutation_config,
    MutatedGenome = genome_mutations:apply_mutations(ChildGenome, MutationConfig),

    %% Convert to network for evaluation
    Network = genome_to_network:convert(MutatedGenome),

    #individual{
        id = make_ref(),
        genome = MutatedGenome,
        network = Network,
        parent1_id = Parent1#individual.id,
        parent2_id = Parent2#individual.id,
        generation_born = Generation,
        is_offspring = true
    }.
```

### Deliverables

- [x] Crossover in `genome_factory.erl` (delegates to faber_tweann)
- [x] Updated `neuroevolution_genetic.erl`
- [x] Individual record with genome field
- [x] Unit tests for crossover

---

## Phase N4: Topology-Aware Speciation

**Goal:** Species based on structural similarity, not just weights

### Implementation

#### N4.1: Compatibility Distance

**File:** `src/neuroevolution_speciation.erl`

Update to use genome-based distance:

```erlang
%% @doc Calculate compatibility distance between two genomes.
%% Considers both structure (innovation alignment) and weights.
-spec compatibility_distance(Genome1, Genome2, Config) -> float() when
    Genome1 :: genome(),
    Genome2 :: genome(),
    Config :: speciation_config().
compatibility_distance(Genome1, Genome2, Config) ->
    {Matching, Disjoint1, Excess1, Disjoint2, Excess2} =
        genome_crossover:align_by_innovation(
            Genome1#genome.connection_genes,
            Genome2#genome.connection_genes
        ),

    N = max(
        length(Genome1#genome.connection_genes),
        length(Genome2#genome.connection_genes)
    ),
    N_norm = max(N, 1),

    ExcessCount = length(Excess1) + length(Excess2),
    DisjointCount = length(Disjoint1) + length(Disjoint2),

    WeightDiff = case Matching of
        [] -> 0.0;
        _ ->
            Diffs = [abs(G1#connection_gene.weight - G2#connection_gene.weight)
                     || {G1, G2} <- Matching],
            lists:sum(Diffs) / length(Matching)
    end,

    %% NEAT distance formula
    Config#speciation_config.c1_excess * ExcessCount / N_norm +
    Config#speciation_config.c2_disjoint * DisjointCount / N_norm +
    Config#speciation_config.c3_weight_diff * WeightDiff.
```

#### N4.2: Speciation Config Update

```erlang
-record(speciation_config, {
    enabled = false :: boolean(),

    %% NEAT compatibility coefficients
    c1_excess = 1.0 :: float(),       % Weight for excess genes
    c2_disjoint = 1.0 :: float(),     % Weight for disjoint genes
    c3_weight_diff = 0.4 :: float(),  % Weight for weight differences

    compatibility_threshold = 3.0 :: float(),
    target_species = 5 :: pos_integer(),
    %% ... rest unchanged ...
}).
```

### Deliverables

- [x] Updated compatibility distance (uses faber_tweann genome_crossover)
- [x] Speciation config with NEAT coefficients
- [x] Unit tests for speciation

---

## Phase N5: Meta-Controller Topology Control

**Goal:** Extend LTC meta-controller to adaptively control topology evolution

### Implementation

#### N5.1: Extended Meta-Controller Inputs

**File:** `src/meta_controller.erl`

Add topology metrics:

```erlang
%% Additional inputs for topology-aware meta-control
compute_topology_inputs(Population) ->
    Genomes = [I#individual.genome || I <- Population],

    %% Average complexity (nodes + connections)
    AvgNodes = lists:sum([length(G#genome.node_genes) || G <- Genomes]) / length(Genomes),
    AvgConns = lists:sum([length(G#genome.connection_genes) || G <- Genomes]) / length(Genomes),

    %% Complexity variance
    NodeVariance = variance([length(G#genome.node_genes) || G <- Genomes]),
    ConnVariance = variance([length(G#genome.connection_genes) || G <- Genomes]),

    %% Topology diversity (unique structures)
    UniqueTopologies = length(lists:usort([topology_signature(G) || G <- Genomes])),
    TopologyDiversity = UniqueTopologies / length(Genomes),

    %% Normalize to 0-1 range
    [
        normalize(AvgNodes, 2, 100),       % Avg node count
        normalize(AvgConns, 1, 500),       % Avg connection count
        normalize(NodeVariance, 0, 50),    % Node variance
        normalize(ConnVariance, 0, 200),   % Connection variance
        TopologyDiversity                   % Already 0-1
    ].
```

#### N5.2: Extended Meta-Controller Outputs

```erlang
%% Additional outputs for topology control
-record(meta_params, {
    %% Weight evolution (existing)
    mutation_rate :: float(),
    mutation_strength :: float(),
    selection_ratio :: float(),

    %% Topology evolution (new)
    add_node_rate :: float(),
    add_connection_rate :: float(),
    complexity_penalty :: float()
}).

apply_meta_params(Params, Config) ->
    MutConfig = Config#neuro_config.mutation_config,
    UpdatedMutConfig = MutConfig#mutation_config{
        weight_mutation_rate = Params#meta_params.mutation_rate,
        weight_perturb_strength = Params#meta_params.mutation_strength,
        add_node_rate = Params#meta_params.add_node_rate,
        add_connection_rate = Params#meta_params.add_connection_rate
    },
    Config#neuro_config{
        selection_ratio = Params#meta_params.selection_ratio,
        mutation_config = UpdatedMutConfig
    }.
```

#### N5.3: Topology-Aware Reward Signal

**File:** `src/meta_reward.erl`

```erlang
%% Extended reward for topology-aware meta-learning
compute_topology_reward(CurrentStats, PrevStats, Config) ->
    BaseReward = compute_standard_reward(CurrentStats, PrevStats, Config),

    %% Complexity efficiency bonus
    %% Reward fitness improvement relative to complexity growth
    FitnessGain = CurrentStats#gen_stats.best_fitness - PrevStats#gen_stats.best_fitness,
    ComplexityGrowth = CurrentStats#gen_stats.avg_complexity - PrevStats#gen_stats.avg_complexity,

    EfficiencyBonus = case ComplexityGrowth > 0 of
        true -> FitnessGain / (ComplexityGrowth + 0.1);  % Avoid division by zero
        false -> FitnessGain * 1.5  % Bonus for improving without growing
    end,

    %% Parsimony pressure (prefer smaller networks)
    ParsimonyPenalty = Config#meta_config.parsimony_coefficient *
        CurrentStats#gen_stats.avg_complexity,

    BaseReward + EfficiencyBonus * 0.1 - ParsimonyPenalty.
```

### Deliverables

- [x] Extended meta-controller inputs (topology metrics)
- [x] Extended meta-controller outputs (topology rates)
- [x] Topology-aware reward signal
- [x] Updated meta_config record in `meta_controller.hrl`
- [x] Unit tests

---

## Phase N6: Strategy Integration

**Goal:** Update all evolution strategies to use genome-based evolution

### Files to Update

| Strategy | Changes Needed |
|----------|---------------|
| `generational_strategy.erl` | Use genome crossover/mutation |
| `steady_state_strategy.erl` | Use genome crossover/mutation |
| `island_strategy.erl` | Genome-based migration |
| `novelty_strategy.erl` | Behavior descriptors from genome structure |
| `map_elites_strategy.erl` | Genome-based cells |

### Implementation Pattern

Each strategy needs:
1. Replace `network_factory:mutate/2` with `genome_mutations:apply_mutations/2`
2. Replace `network_factory:crossover/2` with `genome_crossover:crossover/3`
3. Store genome in individual record
4. Convert genome to network for evaluation

### Deliverables

- [x] Updated all 5 strategies (generational, steady_state, island, novelty, map_elites)
- [x] Backward compatibility (strategies work with both genome and weight-only)
- [x] Strategy-level tests

---

## Configuration Updates

### neuro_config Extensions

```erlang
-record(neuro_config, {
    %% ... existing fields ...

    %% Topology evolution mode
    topology_evolution = false :: boolean(),  % Enable/disable topology evolution

    %% Starting topology
    %% When topology_evolution=true, this is the minimal starting point
    %% When topology_evolution=false, this is the fixed topology
    network_topology :: {pos_integer(), [pos_integer()], pos_integer()},

    %% Mutation configuration (for topology evolution)
    mutation_config :: mutation_config() | undefined,

    %% Innovation tracker (shared across population)
    innovation_tracker :: pid() | undefined
}).
```

---

## Migration Path

### Backward Compatibility

1. **Default:** `topology_evolution = false` - behaves exactly as before
2. **Opt-in:** `topology_evolution = true` - enables full TWEANN

### Migration Steps

1. Existing code continues to work unchanged
2. New code can enable topology evolution via config
3. Gradual migration of examples/demos

---

## Success Criteria

1. [x] Genome-based representation works correctly
2. [x] All mutation operators functional
3. [x] NEAT crossover produces valid offspring
4. [x] Speciation correctly groups similar topologies
5. [x] Meta-controller adapts topology rates
6. [x] All 5 evolution strategies support topology evolution
7. [x] Backward compatibility maintained
8. [x] All existing tests pass (218 tests)
9. [x] New comprehensive test suite (genome_factory_tests.erl)

---

## Dependencies

| Dependency | Version | Purpose |
|------------|---------|---------|
| faber_tweann | ~> 0.12.0 | Innovation tracking, genome crossover, NEAT operations |

---

## Files to Create

| File | Purpose |
|------|---------|
| `src/genome_factory.erl` | Create and manage genomes |
| `src/genome_mutations.erl` | Apply mutations to genomes |
| `src/genome_crossover.erl` | NEAT-style crossover |
| `src/genome_to_network.erl` | Convert genome to network |
| `src/innovation_client.erl` | Client for innovation tracker |
| `test/unit/genome_*_tests.erl` | Unit tests for genome operations |

## Files to Modify

| File | Changes |
|------|---------|
| `include/neuroevolution.hrl` | Add genome records |
| `src/neuroevolution_server.erl` | Initialize innovation tracker |
| `src/neuroevolution_genetic.erl` | Use genome operations |
| `src/neuroevolution_speciation.erl` | Genome-based distance |
| `src/meta_controller.erl` | Topology inputs/outputs |
| `src/meta_reward.erl` | Topology-aware reward |
| All strategy modules | Genome-based evolution |

---

## Estimated Effort

| Phase | Complexity | Depends On |
|-------|------------|------------|
| N1: Representation | Medium | None |
| N2: Mutations | Medium | N1 |
| N3: Crossover | High | N1, faber-tweann T2 |
| N4: Speciation | Medium | N3 |
| N5: Meta-Controller | Medium | N1-N4 |
| N6: Strategies | High | N1-N3 |

---

## References

- Stanley, K.O. & Miikkulainen, R. (2002). "Evolving Neural Networks through Augmenting Topologies." *Evolutionary Computation*, 10(2), 99-127.
- Existing: `guides/topology-evolution.md` (conceptual roadmap)
- Existing: `guides/liquid-conglomerate.md` (meta-learning architecture)
