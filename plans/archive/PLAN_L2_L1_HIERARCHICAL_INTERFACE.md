# Plan: L2→L1 Hierarchical Interface for Liquid Conglomerate

**Status:** ✅ COMPLETE
**Created:** 2025-12-09
**Last Updated:** 2025-12-24

---

## Overview

Refactor the Liquid Conglomerate architecture so that L2 (meta_controller) controls L1 (task_silo) behavior, rather than directly outputting hyperparameters. This creates a proper hierarchical control system.

---

## 1. Problem Statement

### Current Architecture (Broken)

```
L2 (meta_controller):
  Outputs: mutation_rate, mutation_strength, selection_ratio  ← DIRECT PARAMS (wrong!)
  Status: Orphaned, not connected to anything

L1 (task_silo):
  Inputs: fitness_history, stagnation_counter
  Outputs: mutation_rate, mutation_strength, selection_ratio
  Hardcoded factors:
    - NetFactor * 0.5 for mutation_rate               ← HARDCODED
    - NetFactor * 0.3 for mutation_strength           ← HARDCODED
    - ExplorationBoost step = 0.1/gen                 ← HARDCODED
    - Stagnation threshold = 5 gens                   ← HARDCODED
    - Topology boost = 1.5× when boost > 0.5          ← HARDCODED
    - Improvement threshold = 0.001                   ← HARDCODED

L0 (task_l0_defaults):
  Provides: Default values and hard bounds
  Status: Working correctly
```

### Problems

1. **L2 bypasses L1**: meta_controller outputs final params directly, making L1 irrelevant
2. **L1 is rigid**: Hardcoded adjustment factors cannot adapt to different scenarios
3. **No hierarchy**: L2 should control L1's behavior, not replace it
4. **Conservative interventions**: 25% max mutation boost is too gentle

---

## 2. Proposed Architecture

### Hierarchical Control Flow

```
                    ┌─────────────────────────────────────┐
                    │           L2 (Strategic)            │
                    │         meta_controller.erl         │
                    │                                     │
                    │  Inputs:                            │
                    │    - Fitness trends (multi-gen)     │
                    │    - Stagnation patterns            │
                    │    - Resource pressure              │
                    │    - Exploration/exploitation       │
                    │                                     │
                    │  Outputs (L1 GUIDANCE):             │
                    │    - aggression_factor     [0, 2]   │
                    │    - exploration_step      [0.05, 0.5] │
                    │    - stagnation_sensitivity [0.0001, 0.01] │
                    │    - topology_aggression   [1, 3]   │
                    │    - exploitation_weight   [0.2, 0.8] │
                    └─────────────────────────────────────┘
                                      │
                                      ▼ L2 Guidance
                    ┌─────────────────────────────────────┐
                    │           L1 (Tactical)             │
                    │           task_silo.erl             │
                    │                                     │
                    │  Inputs:                            │
                    │    - L2 guidance (above)            │
                    │    - Generation stats               │
                    │    - Stagnation counter             │
                    │                                     │
                    │  Applies:                           │
                    │    - mutation_rate × (1 + NetFactor × L2.aggression) │
                    │    - exploration_boost += L2.exploration_step │
                    │    - if improvement < L2.stagnation_sensitivity → stagnate │
                    │                                     │
                    │  Outputs (HYPERPARAMETERS):         │
                    │    - mutation_rate                  │
                    │    - mutation_strength              │
                    │    - selection_ratio                │
                    │    - add_node_rate                  │
                    └─────────────────────────────────────┘
                                      │
                                      ▼ Hyperparameters
                    ┌─────────────────────────────────────┐
                    │           L0 (Defaults)             │
                    │       task_l0_defaults.erl          │
                    │                                     │
                    │  Enforces: Hard bounds on all params│
                    │    - mutation_rate ∈ [0.01, 0.50]   │
                    │    - mutation_strength ∈ [0.05, 1.0]│
                    │    - selection_ratio ∈ [0.05, 0.50] │
                    └─────────────────────────────────────┘
```

---

## 3. L2 Output Specification (L1 Guidance)

### 3.1 Meta-Parameters

| Parameter | Range | Default | Description |
|-----------|-------|---------|-------------|
| `aggression_factor` | [0.0, 2.0] | 0.5 | Multiplier for L1 adjustment strength |
| `exploration_step` | [0.05, 0.5] | 0.1 | Per-generation exploration_boost increment |
| `stagnation_sensitivity` | [0.0001, 0.01] | 0.001 | Improvement threshold for stagnation |
| `topology_aggression` | [1.0, 3.0] | 1.5 | Multiplier for add_node_rate when stagnating |
| `exploitation_weight` | [0.2, 0.8] | 0.5 | How much to weight exploitation vs exploration |

### 3.2 L2 Guidance Record

```erlang
%% In meta_controller.hrl
-record(l2_guidance, {
    %% How aggressive L1 adjustments should be
    aggression_factor = 0.5 :: float(),       % [0.0, 2.0]

    %% How fast exploration boost increases per stagnating generation
    exploration_step = 0.1 :: float(),        % [0.05, 0.5]

    %% Threshold for what counts as "improvement" (below = stagnating)
    stagnation_sensitivity = 0.001 :: float(), % [0.0001, 0.01]

    %% How much to boost topology mutations when heavily stagnating
    topology_aggression = 1.5 :: float(),      % [1.0, 3.0]

    %% Weight for exploitation vs exploration (higher = more exploitation)
    exploitation_weight = 0.5 :: float()       % [0.2, 0.8]
}).

-type l2_guidance() :: #l2_guidance{}.
```

---

## 4. L1 Integration Changes

### 4.1 State Record Changes

```erlang
%% In task_silo.erl
-record(state, {
    %% ... existing fields ...

    %% NEW: L2 guidance (updated each generation when L2 enabled)
    l2_guidance :: l2_guidance() | undefined,

    %% NEW: Reference to meta_controller (if L2 enabled)
    meta_controller_ref :: pid() | undefined
}).
```

### 4.2 Updated L1 Adjustment Logic

```erlang
%% Current (hardcoded):
apply_l1_adjustments(Params, State) ->
    NetFactor = ExplorationBoost - (ExploitationBoost * 0.5),
    AdjustedMR = BaseMR * (1.0 + NetFactor * 0.5),  % HARDCODED
    ...

%% Proposed (L2-guided):
apply_l1_adjustments(Params, State) ->
    #state{l2_guidance = Guidance} = State,

    %% Get L2 guidance factors (or defaults if L2 disabled)
    Aggression = get_aggression(Guidance),           % Default: 0.5
    ExploitWeight = get_exploitation_weight(Guidance), % Default: 0.5

    %% Net adjustment factor using L2's exploitation_weight
    NetFactor = ExplorationBoost - (ExploitationBoost * ExploitWeight),

    %% Apply L2's aggression_factor to mutation adjustments
    AdjustedMR = BaseMR * (1.0 + NetFactor * Aggression),
    AdjustedMS = BaseMS * (1.0 + NetFactor * Aggression * 0.6),

    %% Use L2's topology_aggression for add_node_rate
    TopologyAggression = get_topology_aggression(Guidance),
    AdjustedANR = case ExplorationBoost > 0.5 of
        true -> BaseANR * TopologyAggression;  % L2-controlled
        false -> BaseANR
    end,
    ...
```

### 4.3 Stagnation Detection with L2 Sensitivity

```erlang
%% Current (hardcoded):
StagnationCounter = case Improvement > 0.001 of  % HARDCODED
    true -> 0;
    false -> State#state.stagnation_counter + 1
end,

%% Proposed (L2-guided):
Sensitivity = get_stagnation_sensitivity(State#state.l2_guidance),
StagnationCounter = case Improvement > Sensitivity of
    true -> 0;
    false -> State#state.stagnation_counter + 1
end,
```

### 4.4 Exploration Boost Step

```erlang
%% Current (hardcoded):
ExplorationBoost = min(1.0, (StagnationCounter - Threshold + 1) * 0.1),  % 0.1 HARDCODED

%% Proposed (L2-guided):
ExplorationStep = get_exploration_step(State#state.l2_guidance),
ExplorationBoost = min(1.0, (StagnationCounter - Threshold + 1) * ExplorationStep),
```

---

## 5. meta_controller Changes

### 5.1 New Output Mapping

Change from outputting hyperparameters to outputting L1 guidance:

```erlang
%% Current output mapping:
create_output_mapping(Config) ->
    BaseParams = [mutation_rate, mutation_strength, selection_ratio],  % WRONG
    ...

%% Proposed output mapping:
create_output_mapping(Config) ->
    %% L2 outputs meta-parameters that control L1
    L1GuidanceParams = [
        aggression_factor,        % How aggressive L1 should be
        exploration_step,         % How fast L1 ramps up
        stagnation_sensitivity,   % When L1 detects stagnation
        topology_aggression,      % How much L1 boosts topology
        exploitation_weight       % L1's explore/exploit balance
    ],
    ...
```

### 5.2 New Parameter Bounds

```erlang
%% In meta_config
param_bounds = #{
    %% L1 guidance parameters (meta-parameters)
    aggression_factor => {0.0, 2.0},
    exploration_step => {0.05, 0.5},
    stagnation_sensitivity => {0.0001, 0.01},
    topology_aggression => {1.0, 3.0},
    exploitation_weight => {0.2, 0.8}
}
```

### 5.3 New API

```erlang
%% Get L1 guidance based on current training state
-spec get_l1_guidance(pid(), generation_stats()) -> l2_guidance().
get_l1_guidance(MetaRef, GenStats) ->
    gen_server:call(MetaRef, {get_l1_guidance, GenStats}).
```

---

## 6. Integration Flow

### 6.1 Per-Generation Flow

```
1. neuroevolution_server completes generation
       │
       ▼
2. task_silo:get_recommendations(Stats)
       │
       ├─ If L2 enabled:
       │     │
       │     ▼
       │  meta_controller:get_l1_guidance(Stats)
       │     │
       │     ▼
       │  Update l2_guidance in state
       │
       ▼
3. apply_l1_adjustments(L0Params, State)
   Uses l2_guidance for adjustment factors
       │
       ▼
4. task_l0_defaults:apply_bounds(L1Params)
   Enforces hard safety limits
       │
       ▼
5. Return final hyperparameters to neuroevolution_server
```

### 6.2 L2 Learning Reward

L2 should learn from L1's effectiveness:

```erlang
%% Reward signal for L2
compute_l2_reward(GenStats, L1State) ->
    %% Reward components:
    %% 1. Fitness improvement (did L1's adjustments help?)
    FitnessReward = compute_fitness_improvement(GenStats),

    %% 2. Efficiency (did L1 avoid unnecessary interventions?)
    EfficiencyReward = case L1State#state.exploration_boost of
        0.0 -> 0.1;  % Small bonus for not intervening
        _ -> 0.0
    end,

    %% 3. Recovery speed (if stagnating, did we escape quickly?)
    RecoveryReward = compute_recovery_reward(L1State),

    FitnessReward * 0.6 + EfficiencyReward * 0.2 + RecoveryReward * 0.2.
```

---

## 7. Implementation Phases

### Phase 1: Define L2 Guidance Types ✅ COMPLETE
- [x] Add `l2_guidance` record to `meta_controller.hrl`
- [x] Define parameter bounds for L1 guidance outputs (`?L2_GUIDANCE_BOUNDS`)
- [x] Add type exports and defaults (`?L2_GUIDANCE_DEFAULTS`)

### Phase 2: Refactor task_silo ✅ COMPLETE
- [x] Add `l2_guidance` field to state record
- [x] Add `l2_enabled` field for dynamic L2 integration
- [x] Extract hardcoded factors to use L2 guidance values
- [x] Update `apply_l1_adjustments/2` to use L2 `aggression_factor`, `exploitation_weight`, `topology_aggression`
- [x] Update stagnation detection to use L2 `stagnation_sensitivity`
- [x] Update exploration_boost calculation to use L2 `exploration_step`
- [x] Add `set_l2_guidance/2` API for manual updates
- [x] Add `maybe_query_l2_guidance/2` for automatic L2 queries

### Phase 3: Refactor meta_controller ✅ COMPLETE
- [x] Add `get_l1_guidance/2` API - queries LTC network and returns l2_guidance
- [x] Add `get_current_guidance/1` API - returns current guidance without update
- [x] Add `process_for_l1_guidance/2` - processes gen stats through LTC network
- [x] Add `outputs_to_l1_guidance/3` - converts network outputs to l2_guidance record
- [x] Network topology configured for 5 outputs (L1 guidance params)

### Phase 4: Integration ✅ COMPLETE
- [x] Update `lc_supervisor` to optionally start meta_controller
- [x] Add `enable_meta_controller` config option
- [x] Add `l2_enabled` config passthrough to task_silo
- [x] task_silo queries meta_controller via `whereis(meta_controller)`
- [x] meta_controller registered as `{local, meta_controller}`

### Phase 5: Testing & Tuning ✅ COMPLETE
- [x] Unit tests for L2 guidance application (16 tests in `lc_l2_controller_tests.erl`)
- [x] Integration tests for hierarchical control (6 tests in `lc_hierarchical_interface_tests.erl`)
- [x] L2 learning rate and reward weights configured via `?L2_GUIDANCE_DEFAULTS`
- [x] Aggressive defaults tuned for faster stagnation escape (aggression=1.5, exploration_step=0.5)

---

## 8. Expected Impact

### Before (L1 Only, Hardcoded)

| Parameter | Max Adjustment |
|-----------|----------------|
| mutation_rate | +25% (×1.25) |
| mutation_strength | +15% (×1.15) |
| add_node_rate | +50% (×1.5) |

### After (L2→L1 Hierarchical)

| Parameter | Range (L2-controlled) |
|-----------|----------------------|
| mutation_rate | +0% to +200% (×1.0 to ×3.0) |
| mutation_strength | +0% to +120% (×1.0 to ×2.2) |
| add_node_rate | +0% to +300% (×1.0 to ×4.0) |

L2 learns the optimal aggressiveness for each scenario, adapting over time.

---

## 9. Files to Modify

| File | Changes |
|------|---------|
| `include/meta_controller.hrl` | Add `l2_guidance` record, new param bounds |
| `src/meta_controller.erl` | Change outputs to L1 guidance, add `get_l1_guidance/2` |
| `src/liquid_conglomerate/task_silo/task_silo.erl` | Accept L2 guidance, use instead of hardcoded |
| `src/liquid_conglomerate/lc_supervisor.erl` | Wire up L2→L1 connection |

---

## 10. Success Criteria

1. **Hierarchical Control**: L2 outputs control L1 behavior, not hyperparameters directly
2. **Adaptive Aggressiveness**: L2 learns scenario-appropriate aggression levels
3. **Backward Compatible**: L1-only mode still works with defaults
4. **Observable**: L2 guidance values visible in dashboard
5. **Effective**: Faster escape from local optima in stagnating populations

---

## Appendix A: Mathematical Justification

The hierarchical approach mirrors calculus concepts:

| Level | Analog | Controls |
|-------|--------|----------|
| L0 | Position f(t) | Parameter values (hard bounds) |
| L1 | Velocity f'(t) | Rate of change (tactical adjustments) |
| L2 | Acceleration f''(t) | Rate of change of velocity (strategic learning) |

L2 doesn't set `mutation_rate = 0.15` directly.
L2 sets `aggression_factor = 1.5` which makes L1 adjust mutation_rate more aggressively.

This is analogous to:
- L0: "The car is at position 100m"
- L1: "The car is moving at 10 m/s"
- L2: "The car is accelerating at 2 m/s²"

L2 controls the acceleration, which indirectly affects position through velocity.
