# Mesh-Native Distribution for Neuroevolution

**Status:** Phase 1 Complete, Phase 2 Next
**Created:** 2025-12-24
**Last Updated:** 2025-12-24
**Related:** PLAN_MESH_NATIVE_OPPONENT_ARCHIVE.md, macula/architecture/ROADMAP.md

---

## Overview

Integrate faber-neuroevolution with the Macula HTTP/3 mesh platform to enable distributed evolutionary computation across multiple nodes. This plan covers:

1. **Distributed Fitness Evaluation** - Run evaluations across mesh nodes
2. **Archive Synchronization** - Sync opponent archives with CRDTs
3. **Island Migration** - Migrate individuals between nodes
4. **Signal Broadcasting** - Coordinate silos across mesh
5. **Global Metrics** - Aggregate diversity and coordination

### Why Macula Mesh?

| Feature | Benefit for Neuroevolution |
|---------|---------------------------|
| NAT traversal | Edge devices can participate |
| Async RPC | Perfect for stateless evaluation |
| Pub/Sub | Signal broadcasting across silos |
| DHT | Service discovery for evaluators |
| CRDTs | Conflict-free archive merging |
| HTTP/3 (QUIC) | Single port, firewall-friendly |

---

## Phase 1: Distributed Fitness Evaluation ✅

**Goal:** Enable fitness evaluations to run on remote mesh nodes
**Status:** Complete (2025-12-24)
**Estimated:** 2-3 weeks

### Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│  Node A (Coordinator)                                           │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ neuroevolution_server                                    │  │
│  │  ┌──────────────────┐    ┌─────────────────────────┐    │  │
│  │  │ Population       │───►│ distributed_evaluator   │    │  │
│  │  │ [Ind1..IndN]     │    │ (RPC dispatcher)        │    │  │
│  │  └──────────────────┘    └───────────┬─────────────┘    │  │
│  └──────────────────────────────────────┼──────────────────┘  │
│                                          │                     │
│                          ┌───────────────┼───────────────┐    │
│                          │               │               │    │
│                    macula_peer:request   │               │    │
│                          │               │               │    │
└──────────────────────────┼───────────────┼───────────────┼────┘
                           │               │               │
              ┌────────────▼───┐   ┌───────▼────┐   ┌──────▼─────┐
              │  Node B        │   │  Node C    │   │  Node D    │
              │  Evaluator     │   │  Evaluator │   │  Evaluator │
              │  Pool (4 CPU)  │   │  Pool (8)  │   │  Pool (2)  │
              └────────────────┘   └────────────┘   └────────────┘
```

### Components

| File | Purpose | LOC Est. |
|------|---------|----------|
| `src/mesh/macula_mesh.erl` | Macula integration facade | 150 |
| `src/mesh/distributed_evaluator.erl` | RPC-based evaluator dispatch | 250 |
| `src/mesh/evaluator_pool_registry.erl` | Track remote evaluator capacity | 200 |
| `src/mesh/mesh_sup.erl` | Supervisor for mesh components | 80 |

### API Design

```erlang
%% Start mesh integration (optional - backward compatible)
macula_mesh:start_link(#{
    realm => <<"neuroevolution.myapp">>,
    mode => production,  % or development
    evaluator_module => my_evaluator,
    evaluator_capacity => 4  % parallel evaluations
}).

%% Evaluate with automatic load balancing
distributed_evaluator:evaluate(Individual, #{
    prefer_local => 0.2,     % 20% chance use local
    timeout_ms => 30000,     % 30s timeout
    retry_count => 2         % retry on failure
}).

%% Register as evaluator node (worker nodes)
distributed_evaluator:register_evaluator(#{
    evaluator_module => my_evaluator,
    capacity => 4
}).
```

### Implementation Steps

- [x] Add macula as optional dependency in rebar.config
- [x] Create `src/mesh/` directory structure
- [x] Implement `macula_mesh.erl` - facade for macula_peer
- [x] Implement `evaluator_pool_registry.erl` - track capacity
- [x] Implement `distributed_evaluator.erl` - RPC dispatch
- [x] Modify `neuroevolution_server.erl` - mesh mode option
- [x] Create unit tests for mesh components (19 tests)
- [ ] Create Docker compose for multi-node testing
- [ ] Integration tests with 3-node cluster

### Success Metrics

- [ ] Evaluations run on 3+ mesh nodes
- [ ] Fair load balancing based on node capacity
- [ ] Graceful fallback when evaluator unavailable
- [ ] <100ms RPC latency for typical evaluation
- [ ] 3-5x speedup with 4 worker nodes

---

## Phase 2: CRDT-Based Archive Synchronization

**Goal:** Synchronize opponent archives across mesh with eventual consistency
**Status:** Planned
**Estimated:** 3-4 weeks
**Depends On:** Phase 1

### Architecture

```
┌─────────────────────────────────────────┐
│  Node A (red_team_archive)              │
│  ┌────────────┐  ┌──────────────┐      │
│  │ ETS (top N)│  │ CRDT state   │      │
│  └────────┬───┘  └──────┬───────┘      │
│           └──────┬──────┘              │
│                  │                     │
│    archive_sync_protocol:broadcast     │
│                  │                     │
└──────────────────┼─────────────────────┘
                   │
        macula_pubsub: archive.{realm}
                   │
┌──────────────────┼─────────────────────┐
│                  │                     │
│  Node B (red_team_archive)             │
│  ┌────────────┐  ┌──────────────┐      │
│  │ ETS (top N)│◄─┤ CRDT merge   │      │
│  └────────────┘  └──────────────┘      │
└────────────────────────────────────────┘
```

### Components

| File | Purpose | LOC Est. |
|------|---------|----------|
| `src/mesh/archive_sync_protocol.erl` | Sync orchestrator | 300 |
| `src/mesh/archive_merkle.erl` | Anti-entropy trees | 200 |

### Implementation Steps

- [ ] Fix archive_crdt memory leak (genome-only storage)
- [ ] Re-enable CRDT tracking in red_team_archive
- [ ] Implement archive_sync_protocol with pub/sub
- [ ] Add Merkle tree for efficient anti-entropy
- [ ] Integration tests for archive convergence

### Success Metrics

- [ ] Archives converge within 5 seconds
- [ ] Duplicate champions detected and merged
- [ ] Archive size bounded even with N nodes
- [ ] 0% data loss due to network partition

---

## Phase 3: Distributed Island Migration

**Goal:** Enable individuals to migrate between islands on different nodes
**Status:** Planned
**Estimated:** 3-4 weeks
**Depends On:** Phase 1, Phase 2

### Architecture

```
┌─────────────────────┐              ┌─────────────────────┐
│ Node A - Island 1   │              │ Node B - Island 2   │
│ ┌────────────────┐  │              │ ┌────────────────┐  │
│ │ 50 individuals │  │   DHT sync   │ │ 50 individuals │  │
│ │ avg_fit: 0.65  │  │ ◄──────────► │ │ avg_fit: 0.58  │  │
│ └────────────────┘  │    RPC       │ └────────────────┘  │
│        │            │   migrate(5) │        │            │
│        ▼            │ ───────────► │        ▼            │
│  Select top 5       │   (binary)   │  Receive & insert   │
└─────────────────────┘              └─────────────────────┘
```

### Components

| File | Purpose | LOC Est. |
|------|---------|----------|
| `src/mesh/distributed_island_migration.erl` | Cross-node migration | 350 |
| `src/mesh/individual_serialization.erl` | Binary format | 150 |
| `src/mesh/island_endpoint_registry.erl` | Discover islands | 200 |

### Implementation Steps

- [ ] Define individual serialization format
- [ ] Implement island_endpoint_registry with DHT
- [ ] Implement distributed_island_migration
- [ ] Modify island_strategy for mesh topology
- [ ] Integration tests for cross-node migration

---

## Phase 4: Cross-Silo Signal Broadcasting

**Goal:** Broadcast Liquid Conglomerate signals across mesh nodes
**Status:** Planned
**Estimated:** 2 weeks
**Depends On:** Phase 1

### Components

| File | Purpose | LOC Est. |
|------|---------|----------|
| `src/mesh/silo_signal_broadcaster.erl` | Pub/sub integration | 200 |
| `src/mesh/silo_signal_aggregator.erl` | Collect signals | 150 |

---

## Phase 5: Distributed Metrics & Coordination

**Goal:** Global diversity tracking and phase coordination
**Status:** Planned
**Estimated:** 2-3 weeks
**Depends On:** Phase 1-4

### Components

| File | Purpose | LOC Est. |
|------|---------|----------|
| `src/mesh/distributed_diversity_tracker.erl` | Aggregate diversity | 250 |
| `src/mesh/mesh_coordination_server.erl` | Phase transitions | 200 |

---

## Dependencies

### Required
- macula >= 0.14.2 (optional dependency)
- faber_tweann >= 0.16.0 (existing)

### Macula Features Used
- `macula_peer:advertise/4` - Register evaluator services
- `macula_peer:request/4` - Async RPC for evaluation
- `macula_peer:publish/3` - Signal broadcasting
- `macula_peer:subscribe/3` - Receive signals
- DHT service discovery - Find evaluators/islands
- NAT traversal - Automatic hole-punching

---

## Backward Compatibility

All phases maintain backward compatibility:

```erlang
%% Traditional single-node mode (unchanged)
neuroevolution_server:start_link(Config).

%% New mesh-enabled mode
neuroevolution_server:start_link(Config#{
    mesh_enabled => true,
    mesh_config => #{
        realm => <<"myapp">>,
        seed_nodes => [<<"quic://seed1:4433">>]
    }
}).
```

---

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| Macula API changes | Pin to v0.14.x, minimal API surface |
| Network latency | Prefer local evaluation, async dispatch |
| Node failures | Timeout + retry, fallback to local |
| CRDT memory | Genome-only storage, bounded archives |
| Complexity | Each phase independent, clear boundaries |

---

## Timeline Summary

| Phase | Duration | Cumulative |
|-------|----------|------------|
| Phase 1: Distributed Evaluation | 2-3 weeks | 2-3 weeks |
| Phase 2: Archive Sync | 3-4 weeks | 5-7 weeks |
| Phase 3: Island Migration | 3-4 weeks | 8-11 weeks |
| Phase 4: Signal Broadcasting | 2 weeks | 10-13 weeks |
| Phase 5: Metrics & Coordination | 2-3 weeks | 12-16 weeks |

**MVP (Phases 1-2):** 5-7 weeks for distributed evaluation + archive sync
