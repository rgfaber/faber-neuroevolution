# Macula-Neuroevolution Memory & Performance Analysis

**Date:** 2025-12-07
**Version:** 0.12.1
**Status:** Analysis Complete

---

## Executive Summary

Macula-neuroevolution has a **significantly simpler architecture** than faber-tweann, with fewer memory concerns. The library uses a single gen_server model with in-memory population storage (no Mnesia), making it more predictable and easier to optimize.

**Key Findings:**
- No critical memory leaks identified
- Architecture is clean and well-bounded
- Main optimization opportunities are in hot paths and data structure efficiency
- Integration with faber-tweann 0.13.0 NIF acceleration provides indirect benefits

**Estimated Memory Footprint:**
- 100 agents with 200-neuron networks: ~50-100 MB
- 1000 generations: Same (population replaced each generation)

---

## Architecture Overview

### Process Model

Unlike faber-tweann's process-per-neuron model, faber-neuroevolution uses:

```
neuroevolution_server (gen_server)
    |
    +-- [spawn_link] evaluation workers (temporary)
    |
    +-- meta_controller (optional gen_server)
```

**Benefits:**
- Single point of state management
- No zombie process accumulation
- Clean shutdown via spawn_link

### Data Storage

| Component | Storage | Bounds | Growth Pattern |
|-----------|---------|--------|----------------|
| Population | gen_server state | Fixed (population_size) | Replaced each gen |
| Generation history | List in state | 50 entries max | Bounded |
| Novelty archive | Strategy state | archive_size (default 1000) | Bounded |
| Ages map | Strategy state | population_size | Bounded |
| Metrics history | meta_state | history_window (default 10) | Bounded |
| Process dictionary | output_mapping | 1 entry | Static |

---

## Memory Analysis

### Good Patterns (Already Implemented)

1. **Bounded Generation History** (`neuroevolution_server.erl:790-791`):
   ```erlang
   generation_history = [{Generation, BestFitness, AvgFitness}
                         | lists:sublist(State#neuro_state.generation_history, 49)]
   ```

2. **Bounded Novelty Archive** (`novelty_strategy.erl:447`):
   ```erlang
   %% Trim archive if too large (remove oldest)
   ```

3. **Proper Process Dictionary Cleanup** (`neuroevolution_server.erl:702`):
   ```erlang
   erase(distributed_eval_results)
   ```

4. **Population Replacement**: Each generation completely replaces the population list, allowing previous generation to be GC'd.

### Potential Issues (Minor)

#### 1. Metrics History Bounds
**File:** `meta_controller.erl:342`

```erlang
NewHistory = update_history(Metrics, State#meta_state.metrics_history, Config)
```

**Analysis:** The `update_history` function uses `history_window` config (default 10) to bound the list. **Already bounded.**

#### 2. Ages Map Cleanup in Steady-State (VERIFIED - NO ISSUE)
**File:** `steady_state_strategy.erl:394-403`

```erlang
%% Update ages: remove victims, add offspring at age 0
NewAges = lists:foldl(
    fun(Ind, Acc) -> maps:remove(Ind#individual.id, Acc) end,
    Ages,
    Victims
),
FinalAges = lists:foldl(
    fun(Ind, Acc) -> maps:put(Ind#individual.id, 0, Acc) end,
    NewAges,
    Offspring
),
```

**Analysis:** Ages map is properly maintained. Victims are removed, offspring added at age 0. **No leak.**

#### 3. Individual Record Size
**File:** `neuroevolution.hrl` (individual record)

Each individual stores:
- `network` - Full network structure (~1-5 KB for 50-200 neurons)
- `genome` - NEAT genome if enabled (~0.5-2 KB)
- `metrics` - Evaluation metrics map (~100-500 bytes)

**For 100 individuals:** ~200 KB - 750 KB per generation

---

## Hot Path Analysis

### O(N) Operations Per Evaluation

| Operation | Location | Frequency | Impact |
|-----------|----------|-----------|--------|
| `update_individual_fitness/4` | generational_strategy.erl:423-432 | Per individual | Linear search through list |
| `calculate_fitness_stats/1` | generational_strategy.erl:583-590 | Per generation | Iterates full population |
| `lists:sort` by fitness | generational_strategy.erl:296-299 | Per generation | O(N log N) |

### O(N * K) Operations (Novelty Search)

| Operation | Location | Complexity | Impact |
|-----------|----------|------------|--------|
| `compute_novelty/4` | novelty_strategy.erl:366-415 | O(N * K) | K-nearest neighbor search |
| Distance calculations | novelty_strategy.erl:408-410 | O(N * Archive) | Archive can be 1000 entries |

**Recommendation:** For large populations (>500), consider k-d tree or approximate nearest neighbor.

### List Operations

No inefficient `++` or `lists:append` found in hot paths. List comprehensions are used appropriately.

---

## Optimization Recommendations

### Phase 1: Quick Wins (Low Effort)

#### 1.1 Verify Ages Map Cleanup
**Priority:** Medium
**Effort:** 1 hour

Check that ages map is cleaned up when individuals die in steady_state_strategy.

#### 1.2 Use Maps for Population Lookup
**Priority:** Medium
**Effort:** 2-4 hours

Current `update_individual_fitness/4` does O(N) linear search. Convert to map-based lookup:

```erlang
%% Current: O(N) search
update_individual_fitness(Id, Fitness, Metrics, [Ind | Rest], Acc) when Ind#individual.id =:= Id -> ...

%% Proposed: O(1) lookup
Population = #{Id => Individual, ...}
UpdatedPop = Population#{Id := Individual#individual{fitness = Fitness}}
```

**Impact:** 10-50x faster individual updates for populations > 50

### Phase 2: Data Structure Optimization (Medium Effort)

#### 2.1 Lazy Network Compilation
**Priority:** Medium
**Effort:** 1 week

Currently each individual stores full network. For NEAT mode, store only genome and compile to network on-demand for evaluation.

```erlang
%% Current
#individual{network = Network, genome = Genome}

%% Proposed
#individual{genome = Genome}  % Compile network only when needed
```

**Impact:** 30-50% memory reduction when using NEAT mode

#### 2.2 Shared Network Factory Cache
**Priority:** Low
**Effort:** 2-3 days

Cache compiled networks by topology hash to avoid recompilation.

### Phase 3: Leverage faber-tweann 0.13.0 (Already Done)

The dependency update to faber-tweann 0.13.0 provides:
- NIF-accelerated network evaluation
- Pre-compiled weight matrices
- Memory-optimized genotype storage

**No additional work needed** - benefits flow through automatically.

### Phase 4: Algorithm Optimization (High Effort)

#### 4.1 Approximate Nearest Neighbor for Novelty
**Priority:** Low (only if novelty search is slow)
**Effort:** 1-2 weeks

Replace brute-force k-NN with:
- k-d tree (deterministic)
- LSH - Locality Sensitive Hashing (approximate)
- Ball tree

**Impact:** O(N * K) -> O(K * log N) for novelty calculation

---

## Comparison: faber-neuroevolution vs faber-tweann

| Aspect | faber-tweann | faber-neuroevolution |
|--------|---------------|----------------------|
| Process model | Per-neuron processes | Single gen_server |
| Storage | Mnesia RAM tables | In-memory state |
| Unbounded growth | evo_hist, dead_pool, innovation | None identified |
| Process lifecycle | Zombie neuron risk | Clean spawn_link |
| Memory per 100 agents | 2-4 GB (pre-0.13.0) | 50-100 MB |
| Optimization priority | CRITICAL | LOW |

---

## Recommended Implementation Order

1. ~~**Verify ages map cleanup**~~ - DONE, no leak found
2. ~~**Map-based population lookup**~~ - DONE, implemented in all strategies
3. **Benchmark current performance** (2 hours) - Establish baseline
4. **Lazy network compilation** (if needed based on benchmarks)

---

## Benchmark Plan

Create `test/benchmark/` directory with:

```
bench_common.erl       - Shared utilities (timing, memory)
bench_generation.erl   - Full generation cycle timing
bench_strategies.erl   - Compare strategy performance
bench_novelty.erl      - Novelty calculation scaling
```

Key metrics to measure:
- Time per generation (population 50, 100, 200)
- Memory per generation
- Novelty calculation time vs archive size
- Individual update time

---

## Conclusion

Macula-neuroevolution is **well-designed with minimal memory concerns**. The architecture is significantly simpler than faber-tweann, with proper bounds on all accumulating data structures.

**Priority:** LOW - Focus optimization efforts on faber-tweann first.

**Quick wins completed:**
1. ~~Map-based population lookup~~ - DONE (all strategies updated)
2. ~~Ages map verification~~ - DONE (no leak found)

**Deferred:**
- Lazy network compilation (wait for usage patterns)
- Approximate k-NN (wait for scale requirements)
