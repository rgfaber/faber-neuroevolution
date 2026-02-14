# Plan: Chained LTC Controller Architecture

**Status:** In Progress (Phases 1-4, 6 Complete; Tests Added)
**Created:** 2025-12-09
**Last Updated:** 2025-12-09

---

## Overview

Refactor the Liquid Conglomerate architecture to use **chained LTC TWEANN networks** where each level is an actual neural network, and outputs chain to inputs of the next level down.

This replaces the current hybrid approach (L2=LTC, L1=hardcoded logic, L0=static bounds) with a pure neural hierarchy.

---

## 1. Current vs. Proposed Architecture

### Current (Hybrid)

```
L2 (meta_controller) = LTC TWEANN
        │
        │ "guidance" (aggression_factor, etc.)
        ▼
L1 (task_silo) = Hardcoded Erlang logic
        │
        │ applies L0 bounds
        ▼
L0 (task_l0_defaults) = Static min/max values
```

**Problems:**
- L1 has hardcoded adjustment logic
- L0 is not adaptive
- No true hierarchical learning

### Proposed (Chained LTC)

```
    Evolution Metrics ─────────────────────────────────────┐
    (fitness, diversity, stagnation, etc.)                 │
                │                                          │
                ▼                                          │
    ┌───────────────────────────┐                          │
    │     L2 (τ = 100)          │                          │
    │     LTC TWEANN            │                          │
    │     Strategic Layer       │                          │
    │                           │                          │
    │  Inputs: evolution_metrics│                          │
    │  Outputs: strategic_signals (N outputs)              │
    └─────────────┬─────────────┘                          │
                  │ L2 outputs                             │
                  ▼                                        │
    ┌───────────────────────────┐                          │
    │     L1 (τ = 50)           │◄── L2 outputs as inputs  │
    │     LTC TWEANN            │                          │
    │     Tactical Layer        │                          │
    │                           │                          │
    │  Inputs: L2_outputs       │                          │
    │  Outputs: tactical_signals (M outputs)               │
    └─────────────┬─────────────┘                          │
                  │ L1 outputs                             │
                  ▼                                        │
    ┌───────────────────────────┐                          │
    │     L0 (τ = 10)           │◄── L1 outputs as inputs  │
    │     LTC TWEANN            │◄── Emergent metrics ─────┘
    │     Reactive Layer        │    (convergence_rate,
    │                           │     mutation_rate,
    │  Inputs: L1_outputs +     │     survival_rate, ...)
    │          emergent_metrics │
    │  Outputs: hyperparameters │
    └─────────────┬─────────────┘
                  │
                  ▼
    Final Hyperparameters for Model Under Training
    (mutation_rate, selection_ratio, add_node_rate, etc.)
```

---

## 2. Key Design Principles

### 2.1 Each Level is a TWEANN

Each level (L0, L1, L2) is a full LTC TWEANN that:
- Has its own genotype stored in Mnesia
- Can evolve its topology (add neurons, connections, sensors)
- Maintains internal LTC state (continuous dynamics)
- Learns through the reward signal

### 2.2 Output-to-Input Chaining

```
L2.outputs[i] ──► L1.inputs[i]  (for i in 0..N-1)
L1.outputs[j] ──► L0.inputs[j]  (for j in 0..M-1)
```

The chaining is deterministic - L2's output neurons connect to L1's input sensors by index.

### 2.3 L0's Additional Emergent Inputs

L0 receives TWO types of inputs:
1. **Fixed inputs**: L1's outputs (always connected)
2. **Emergent sensors**: Metrics from the model under training

The emergent sensors are **available** in L0's morphology but **not initially connected**. Topology evolution can add them via `add_sensor/1`.

### 2.4 Time Constants (τ)

Different τ values create temporal abstraction:

| Level | τ (time constant) | Adaptation Speed | Role |
|-------|-------------------|------------------|------|
| L2 | 100 | Very slow | Strategic decisions, long-term trends |
| L1 | 50 | Medium | Tactical adjustments, medium-term patterns |
| L0 | 10 | Fast | Reactive control, immediate response |

Lower τ = faster adaptation to inputs.

---

## 3. Morphology Definitions

### 3.1 L2 Morphology (Strategic Layer)

```erlang
-module(lc_l2_morphology).

%% Sensors: Evolution metrics (from neuroevolution_server)
get_sensors() -> [
    #sensor{name = best_fitness, vector_length = 1},
    #sensor{name = avg_fitness, vector_length = 1},
    #sensor{name = fitness_improvement, vector_length = 1},
    #sensor{name = fitness_variance, vector_length = 1},
    #sensor{name = stagnation_counter, vector_length = 1},
    #sensor{name = generation_progress, vector_length = 1},  % gen/max_gen
    #sensor{name = population_diversity, vector_length = 1},
    #sensor{name = species_count, vector_length = 1}
].

%% Actuators: Strategic signals (fed to L1)
get_actuators() -> [
    #actuator{name = strategic_signal_1, vector_length = 1},
    #actuator{name = strategic_signal_2, vector_length = 1},
    #actuator{name = strategic_signal_3, vector_length = 1},
    #actuator{name = strategic_signal_4, vector_length = 1}
].
```

### 3.2 L1 Morphology (Tactical Layer)

```erlang
-module(lc_l1_morphology).

%% Sensors: L2 outputs (fixed, always connected)
get_sensors() -> [
    #sensor{name = l2_strategic_1, vector_length = 1},
    #sensor{name = l2_strategic_2, vector_length = 1},
    #sensor{name = l2_strategic_3, vector_length = 1},
    #sensor{name = l2_strategic_4, vector_length = 1}
].

%% Actuators: Tactical signals (fed to L0)
get_actuators() -> [
    #actuator{name = tactical_signal_1, vector_length = 1},
    #actuator{name = tactical_signal_2, vector_length = 1},
    #actuator{name = tactical_signal_3, vector_length = 1},
    #actuator{name = tactical_signal_4, vector_length = 1},
    #actuator{name = tactical_signal_5, vector_length = 1}
].
```

### 3.3 L0 Morphology (Reactive Layer)

```erlang
-module(lc_l0_morphology).

%% Sensors: L1 outputs (fixed) + Emergent metrics (available for topology evolution)
get_sensors() -> [
    %% Fixed inputs from L1 (initially connected)
    #sensor{name = l1_tactical_1, vector_length = 1, fixed = true},
    #sensor{name = l1_tactical_2, vector_length = 1, fixed = true},
    #sensor{name = l1_tactical_3, vector_length = 1, fixed = true},
    #sensor{name = l1_tactical_4, vector_length = 1, fixed = true},
    #sensor{name = l1_tactical_5, vector_length = 1, fixed = true},

    %% Emergent metrics (available, not initially connected)
    #sensor{name = convergence_rate, vector_length = 1, fixed = false},
    #sensor{name = current_mutation_rate, vector_length = 1, fixed = false},
    #sensor{name = survival_rate, vector_length = 1, fixed = false},
    #sensor{name = offspring_rate, vector_length = 1, fixed = false},
    #sensor{name = complexity_trend, vector_length = 1, fixed = false},
    #sensor{name = fitness_plateau_duration, vector_length = 1, fixed = false},
    #sensor{name = species_extinction_rate, vector_length = 1, fixed = false},
    #sensor{name = innovation_rate, vector_length = 1, fixed = false}
].

%% Actuators: Final hyperparameters
get_actuators() -> [
    #actuator{name = mutation_rate, vector_length = 1, range = {0.01, 0.5}},
    #actuator{name = mutation_strength, vector_length = 1, range = {0.05, 1.0}},
    #actuator{name = selection_ratio, vector_length = 1, range = {0.1, 0.5}},
    #actuator{name = add_node_rate, vector_length = 1, range = {0.0, 0.1}},
    #actuator{name = add_connection_rate, vector_length = 1, range = {0.0, 0.2}}
].
```

---

## 4. Execution Flow

### 4.1 Per-Generation Update

```erlang
%% After each generation of the model under training:

update_lc_chain(GenStats, EmergentMetrics) ->
    %% 1. Prepare L2 inputs from evolution metrics
    L2Inputs = extract_l2_inputs(GenStats),

    %% 2. Forward pass through L2
    L2Outputs = ltc_forward(L2Agent, L2Inputs, DeltaT),

    %% 3. Forward pass through L1 (L2 outputs become L1 inputs)
    L1Inputs = L2Outputs,
    L1Outputs = ltc_forward(L1Agent, L1Inputs, DeltaT),

    %% 4. Forward pass through L0 (L1 outputs + emergent metrics)
    L0Inputs = L1Outputs ++ get_connected_emergent_metrics(L0Agent, EmergentMetrics),
    L0Outputs = ltc_forward(L0Agent, L0Inputs, DeltaT),

    %% 5. Scale L0 outputs to hyperparameter ranges
    Hyperparams = scale_to_ranges(L0Outputs, L0Actuators),

    Hyperparams.
```

### 4.2 Learning (Backpropagation Through Chain)

The reward signal propagates through all three levels:

```erlang
%% Reward based on model performance
Reward = compute_reward(GenStats),

%% Backprop through chain (reverse order)
L0Gradients = compute_ltc_gradients(L0Agent, Reward),
L1Gradients = compute_ltc_gradients(L1Agent, L0Gradients),
L2Gradients = compute_ltc_gradients(L2Agent, L1Gradients),

%% Update weights
update_weights(L0Agent, L0Gradients, LearningRate),
update_weights(L1Agent, L1Gradients, LearningRate),
update_weights(L2Agent, L2Gradients, LearningRate).
```

---

## 5. Implementation Phases

### Phase 1: Define LC Morphologies ✅ COMPLETE
- [x] Create `lc_l2_morphology.erl` with 8 evolution metric sensors, 4 strategic outputs
- [x] Create `lc_l1_morphology.erl` with 4 L2 inputs, 5 tactical outputs
- [x] Create `lc_l0_morphology.erl` with 5 L1 inputs + 13 emergent sensors, 5 hyperparam outputs
- [x] Create `lc_morphologies.erl` registration module

### Phase 2: Implement LC Chain Controller ✅ COMPLETE
- [x] Create `lc_chain.hrl` - records for chain config, state, metrics
- [x] Create `lc_chain.erl` - gen_server managing the L2→L1→L0 chain
- [x] Implement `init/1` - creates all three LTC agents via genotype:construct_agent
- [x] Implement `forward/3` - cascades inputs through L2→L1→L0
- [x] Implement `get_hyperparams/1` - extracts final parameters from L0 outputs
- [x] Implement LTC dynamics (dx/dt = -x/τ + f(x,I)) with Euler integration
- [x] Implement input normalization and output scaling

### Phase 3: LTC Forward Pass for Chain ✅ COMPLETE
- [x] Implement `ltc_forward/2` - LTC dynamics with state update
- [x] Handle inter-level output→input mapping (L2→L1→L0)
- [x] Support variable-length sensor connections (for emergent metrics)
- [x] Full implementation reading weights from genotype

### Phase 4: Emergent Metric Integration ✅ COMPLETE
- [x] Define `#emergent_metrics{}` record in `lc_chain.hrl`
- [x] Define `#evolution_metrics{}` record in `lc_chain.hrl`
- [x] Implement `get_emergent_inputs/2` for connected emergent sensors
- [x] Implement metric collection from neuroevolution_server (wiring)
- [ ] Test `add_sensor` for emergent metrics via topology evolution

### Phase 5: Learning/Training ✅ COMPLETE
- [x] Implement reward computation for LC chain (`do_train/2`)
- [x] Implement weight propagation through chain (`propagate_ltc/4`)
- [x] Add weight update with tau-scaled learning rate (`update_level_weights/3`)
- [x] Support topology evolution for each level (via genotype API)

### Phase 6: Integration with neuroevolution_server ✅ COMPLETE
- [x] Replace current meta_controller usage (when lc_chain_config is set)
- [x] Wire LC chain to generation_complete events
- [x] Provide hyperparameters to evolution loop
- [x] Add `lc_chain_config` to `#neuro_config{}`
- [x] Add `lc_chain` pid to `#neuro_state{}`
- [x] Implement `maybe_start_lc_chain/1`
- [x] Implement `maybe_update_lc_chain/3`
- [x] Implement `build_evolution_metrics/2` and `build_emergent_metrics/2`

---

## 6. Files to Create/Modify

| File | Purpose | Status |
|------|---------|--------|
| `src/lc_morphologies/lc_l2_morphology.erl` | L2 sensor/actuator definitions | ✅ DONE |
| `src/lc_morphologies/lc_l1_morphology.erl` | L1 sensor/actuator definitions | ✅ DONE |
| `src/lc_morphologies/lc_l0_morphology.erl` | L0 sensor/actuator definitions + emergent | ✅ DONE |
| `src/lc_morphologies/lc_morphologies.erl` | Registration module | ✅ DONE |
| `src/liquid_conglomerate/lc_chain.erl` | Chained LTC controller gen_server | ✅ DONE |
| `include/lc_chain.hrl` | Records for LC chain config/state/metrics | ✅ DONE |
| `src/liquid_conglomerate/lc_agent.erl` | Individual LC level wrapper | PENDING |
| `src/neuroevolution_server.erl` | Wire LC chain integration | ✅ DONE |
| `include/neuroevolution.hrl` | Added lc_chain_config + lc_chain fields | ✅ DONE |
| `test/lc_chain_tests.erl` | Unit tests for LC chain (15 tests) | ✅ DONE |

---

## 7. Success Criteria

1. **Chained Execution**: L2→L1→L0 forward pass works correctly
2. **Output-Input Mapping**: L(N) outputs correctly feed L(N-1) inputs
3. **Emergent Sensors**: L0 can dynamically add sensors via topology evolution
4. **Learning**: All three levels improve hyperparameter selection over time
5. **Temporal Abstraction**: Different τ values create meaningful specialization
6. **Integration**: neuroevolution_server uses LC chain for adaptive hyperparameters

---

## 8. Mathematical Foundation

### LTC Dynamics (per neuron)

```
dx/dt = -x/τ + f(x, I, t)

where:
  x = internal state
  τ = time constant
  I = weighted inputs
  f = nonlinear activation (CfC or ODE-based)
```

### Chained Hierarchy

```
L2: x₂(t+Δt) = LTC_forward(x₂(t), inputs_evolution, τ₂)
    y₂ = output(x₂)

L1: x₁(t+Δt) = LTC_forward(x₁(t), y₂, τ₁)
    y₁ = output(x₁)

L0: x₀(t+Δt) = LTC_forward(x₀(t), [y₁, emergent], τ₀)
    y₀ = output(x₀)  ← Final hyperparameters
```

### Reward Backpropagation

```
R = fitness_improvement * efficiency_factor

∂R/∂W₀ = ∂R/∂y₀ · ∂y₀/∂W₀
∂R/∂W₁ = ∂R/∂y₀ · ∂y₀/∂y₁ · ∂y₁/∂W₁
∂R/∂W₂ = ∂R/∂y₀ · ∂y₀/∂y₁ · ∂y₁/∂y₂ · ∂y₂/∂W₂
```

---

## 9. Relation to Previous Work

This plan **supersedes** `PLAN_L2_L1_HIERARCHICAL_INTERFACE.md` which implemented:
- L2 as the only LTC network
- L1 as hardcoded Erlang logic
- L2 "guidance" parameters controlling L1 behavior

The new chained architecture is more elegant and allows true hierarchical learning.

---

## Appendix A: Emergent Metrics for L0

These metrics can be dynamically sensed by L0 (via topology evolution):

| Metric | Description | Source |
|--------|-------------|--------|
| convergence_rate | Rate of fitness improvement | GenStats |
| current_mutation_rate | Active mutation rate | Hyperparams |
| survival_rate | Survivors / Population | Population |
| offspring_rate | New individuals / Population | Population |
| complexity_trend | Network size change rate | Individuals |
| fitness_plateau_duration | Generations at current best | GenStats |
| species_extinction_rate | Species dying per gen | Species |
| innovation_rate | New topologies per gen | Innovation |
| elite_age | Generations champion unchanged | GenStats |
| diversity_index | Genotype diversity measure | Population |
