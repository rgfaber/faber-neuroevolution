# Mesh-Native Opponent Archive

**Status:** In Progress (Phase 1 Complete, Phase 2 Partial - CRDT Disabled, Phase 3 Blocked)
**Created:** 2025-12-12
**Last Updated:** 2025-12-24

> **NOTE:** The actual implementation uses `red_team_archive.erl` in `src/competitive_coevolution/`
> instead of `opponent_archive.erl` in `src/self_play/`. This plan has been updated to reflect
> the actual codebase state.

## Overview

Move the opponent archive from Elixir (macula-neurolab) to Erlang (faber-neuroevolution) and design it for mesh-native distributed scenarios. The current architecture has a fundamental disconnect: archive is in Elixir but evaluation happens in Erlang, making it impossible for the evaluator to use archived opponents.

## Problem Statement

### Current Architecture (Broken)

```
┌─────────────────────────────────────────────────────────────────┐
│  macula-neurolab (Elixir)                                       │
│  ┌──────────────────┐     ┌─────────────────────┐              │
│  │ OpponentArchive  │◄────│ SelfPlayMode        │              │
│  │ (ETS-based)      │     │ (receives candidates)│              │
│  └──────────────────┘     └─────────────────────┘              │
│           ▲                         ▲                           │
│           │                         │ archive_candidates event  │
└───────────┼─────────────────────────┼───────────────────────────┘
            │ ??? cannot access       │
┌───────────┼─────────────────────────┼───────────────────────────┐
│           │                         │                           │
│  ┌────────┴─────────┐     ┌─────────┴─────────┐                │
│  │ Evaluator        │     │ neuroevolution_   │                │
│  │ (needs opponents)│     │ server            │                │
│  └──────────────────┘     └───────────────────┘                │
│  faber-neuroevolution (Erlang)                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Target Architecture (Mesh-Native)

```
┌─────────────────────────────────────────────────────────────────┐
│  Node A (Local Training)                                        │
│  ┌──────────────────────────────────────────────────┐          │
│  │ neuroevolution_server                            │          │
│  │  ┌────────────────┐    ┌─────────────────────┐  │          │
│  │  │ opponent_      │◄───│ self_play_strategy  │  │          │
│  │  │ archive (ETS)  │    │                     │  │          │
│  │  │ + CRDT state   │    │ evaluate(NuT, Opp)  │  │          │
│  │  └────────────────┘    └─────────────────────┘  │          │
│  └──────────────────────────────────────────────────┘          │
│                    │ mesh sync (eventual)                       │
│                    ▼                                            │
│  ┌──────────────────────────────────────────────────┐          │
│  │ macula_pubsub (Macula Mesh)                      │          │
│  │ Topic: neuroevolution.archive.{realm}            │          │
│  └──────────────────────────────────────────────────┘          │
└─────────────────────────────────────────────────────────────────┘
                     │
        HTTP/3 (QUIC) │ NAT-friendly
                     ▼
┌─────────────────────────────────────────────────────────────────┐
│  Node B (Edge/Remote)                                           │
│  ┌──────────────────────────────────────────────────┐          │
│  │ opponent_archive (ETS) + CRDT state              │          │
│  │ (eventually consistent with Node A)              │          │
│  └──────────────────────────────────────────────────┘          │
└─────────────────────────────────────────────────────────────────┘
```

## Design Decisions

### Why NOT PlumDB?

PlumDB would be perfect for traditional BEAM clusters but won't work for Macula mesh:

| Feature | PlumDB | Macula Mesh Requirement |
|---------|--------|-------------------------|
| Transport | Partisan (Erlang distribution) | HTTP/3 (QUIC) |
| Network | LAN/datacenter assumed | NAT traversal, edge nodes |
| Topology | Static cluster membership | Dynamic peer discovery |
| Gossip | Partisan-based | Macula DHT/PubSub |

### Why ETS + CRDTs + Macula PubSub?

1. **ETS** - Local fast reads, O(1) lookup for opponent selection
2. **CRDTs** - Automatic conflict resolution when archives merge
3. **Macula PubSub** - Native mesh transport, NAT-friendly, HTTP/3

### CRDT Choice: OR-Set (Observed-Remove Set)

The archive is essentially a **set of champions with metadata**. Using an OR-Set:
- Add champion = add element with unique tag
- Remove champion (pruning) = remove specific tagged element
- Merge = union of all unique tagged elements
- No conflicts: concurrent adds/removes resolve automatically

Library: `riak_dt` (battle-tested, already available in BEAM ecosystem)

## Implementation Phases

### Phase 1: Local Erlang Archive (Single-Node)

Move archive to Erlang with full competitive coevolution integration.

**Files Created:**

| File | Purpose | Status |
|------|---------|--------|
| `src/competitive_coevolution/red_team_archive.erl` | ETS-based archive with weighted sampling | DONE |
| `src/competitive_coevolution/coevolution_manager.erl` | Coevolution mode manager | DONE |
| `src/competitive_coevolution/coevolution_sup.erl` | Supervisor for coevolution components | DONE |

**API:**

```erlang
%% red_team_archive.erl - ETS-based champion (Red Team) storage
-export([
    start_link/1,           % Start archive with ID
    start_link/2,           % Start archive with ID and config
    add/2,                  % Add champion to Red Team
    add/3,                  % Add with fitness threshold check
    sample/1,               % Weighted sample (fitness + recency)
    sample/2,               % Sample with options
    size/1,                 % Current archive size
    stats/1,                % Archive statistics
    prune/2,                % Keep top N champions
    clear/1,                % Clear all members
    stop/1,                 % Stop archive
    %% Red Team specific
    update_fitness/3,       % Update fitness after competition
    get_top/2,              % Get top N members
    get_member/2,           % Get specific member
    %% For mesh sync (Phase 2)
    export_crdt/1,          % Export as binary for sync
    merge_crdt/2            % Merge remote state
]).
```

**Integration Points:**

1. `neuroevolution_server.erl` - Uses coevolution_manager for Red Team vs Blue Team
2. Archive candidates flow directly to `red_team_archive` (no Elixir hop)
3. Evaluator gets opponent from archive (same process, fast)

### Phase 2: CRDT-Based Conflict Resolution - PARTIAL (Integration Disabled)

Add CRDT layer for eventual consistency.

**Implementation Note:** Instead of using `riak_dt` (last updated 2016), we implemented a lightweight custom OR-Set CRDT in `archive_crdt.erl`. This avoids the dependency on a potentially outdated library while providing exactly the functionality we need.

**⚠️ IMPORTANT:** CRDT tracking in `red_team_archive.erl` is currently **DISABLED** to prevent memory leaks. Full network data was being accumulated in the CRDT without pruning. When Phase 3 mesh sync is implemented, the CRDT should store only genome/weights, not full network data.

See `red_team_archive.erl` lines 344-346 for the disable note.

**Files Created:**

| File | Purpose | Status |
|------|---------|--------|
| `src/competitive_coevolution/archive_crdt.erl` | Custom OR-Set CRDT implementation | DONE |
| `test/archive_crdt_tests.erl` | CRDT unit tests (14) + 1 archive export test | DONE (15 tests) |

**Files Modified:**

| File | Changes | Status |
|------|---------|--------|
| `src/competitive_coevolution/red_team_archive.erl` | Added CRDT field to state, CRDT ops prepared but disabled | PARTIAL |

**CRDT Structure:**

```erlang
%% OR-Set entry with unique tag
-record(entry, {
    tag :: binary(),          % Unique identifier (actor_id + counter)
    value :: term(),          % The actual champion data
    tombstone = false :: boolean()  % Removed entries become tombstones
}).

%% OR-Set state
-record(orset, {
    actor_id :: binary(),     % This node's unique ID
    counter = 0 :: non_neg_integer(),  % Monotonic counter for tags
    entries = #{} :: #{binary() => #entry{}}  % Tag -> Entry map
}).
```

**Binary Format (for network serialization):**
- Version byte (1)
- Actor ID length (2 bytes) + Actor ID
- Counter (8 bytes, big-endian)
- Entry count (4 bytes, big-endian)
- For each entry: Tag length (2) + Tag + Value length (4) + Value (term_to_binary)

**Operations:**

```erlang
%% Add champion (with unique actor tag)
archive_crdt:add(Value, CRDT, ActorId) -> orset().

%% Remove by value (marks matching entries as tombstones)
archive_crdt:remove(Value, CRDT) -> orset().

%% Merge two archives (tombstones win)
archive_crdt:merge(CRDT1, CRDT2) -> orset().

%% Get all live entries
archive_crdt:value(CRDT) -> [term()].

%% Export/import for network transmission
archive_crdt:export_binary(CRDT) -> binary().
archive_crdt:import_binary(Binary) -> {ok, orset()} | {error, term()}.
```

### Phase 3: Mesh Sync via Macula PubSub

Integrate with Macula mesh for distributed training.

**Topics:**

| Topic | Purpose |
|-------|---------|
| `neuroevolution.archive.{realm}.update` | Broadcast new champions |
| `neuroevolution.archive.{realm}.sync` | Full archive sync request/response |
| `neuroevolution.archive.{realm}.anti_entropy` | Merkle tree comparison |

**Sync Protocol:**

```
1. Node adds champion locally
2. Broadcast delta to `archive.{realm}.update`
3. Other nodes merge delta into local CRDT
4. Periodic anti-entropy (every 30s):
   a. Compute Merkle tree root of archive
   b. Exchange roots with peers
   c. If different, request missing entries
   d. Merge received entries
```

**Files to Create:**

| File | Purpose |
|------|---------|
| `src/self_play/archive_mesh_sync.erl` | Mesh synchronization logic |
| `src/self_play/archive_merkle.erl` | Merkle tree for anti-entropy |

### Phase 4: Cleanup Elixir Side

Remove/deprecate Elixir archive components.

**Files to Remove/Deprecate:**

| File | Action |
|------|--------|
| `opponent_archive.ex` | Remove (functionality moved to Erlang) |
| `self_play_mode.ex` | Simplify - delegate to Erlang |

## Success Criteria

### Phase 1 (Local) - COMPLETED
- [x] `red_team_archive.erl` stores champions in ETS
- [x] `coevolution_manager.erl` provides Red Team vs Blue Team management
- [x] `coevolution_sup.erl` supervisor for coevolution components
- [x] Archive pruning maintains top N champions
- [x] Weighted sampling based on fitness and recency

### Phase 2 (CRDT) - PARTIAL
- [x] Custom OR-Set CRDT implemented (`archive_crdt.erl`)
- [x] CRDT operations: add, remove, merge, export_binary, import_binary
- [x] Concurrent adds/removes resolve without conflict (commutative, associative, idempotent)
- [x] Binary serialization format working (version-prefixed)
- [x] 15 tests passing (14 CRDT unit + 1 archive export)
- [ ] **BLOCKED:** CRDT integration in archive disabled due to memory leak
- [ ] **TODO:** Re-enable with genome-only storage when Phase 3 begins

### Phase 3 (Mesh Sync)
- [ ] Champions broadcast via Macula PubSub
- [ ] Nodes discover and sync archives
- [ ] Anti-entropy keeps archives consistent
- [ ] Works across NAT (via Macula mesh relay)

### Phase 4 (Cleanup)
- [ ] Elixir archive removed
- [ ] No performance regression
- [ ] Documentation updated

## Architecture Diagrams

### Phase 1: Local Architecture (Pure Self-Play)

```
┌─────────────────────────────────────────────────────────────────┐
│  neuroevolution_server                                          │
│                                                                 │
│  ┌────────────────────────────────────────────────────────┐    │
│  │ Pure Self-Play Evaluation                               │    │
│  │                                                         │    │
│  │  1. Extract batch_networks from population              │    │
│  │       │                                                 │    │
│  │       ▼                                                 │    │
│  │  2. get_opponent(ManagerPid, BatchNetworks)             │    │
│  │       │                                                 │    │
│  │       ├──► Archive has opponents? ──► sample(Archive)   │    │
│  │       │           │                        │            │    │
│  │       │           NO                       YES          │    │
│  │       │           ▼                        │            │    │
│  │       └──► sample_from_batch(BatchNetworks)│            │    │
│  │                   │                        │            │    │
│  │                   └────────┬───────────────┘            │    │
│  │                            ▼                            │    │
│  │  3. evaluate(NuT, Opponent)                             │    │
│  │       │                                                 │    │
│  │       ▼                                                 │    │
│  │  4. if top_performer: add_to_archive(NuT)               │    │
│  └────────────────────────────────────────────────────────┘    │
│                                                                 │
│  Key: Population IS the opponent pool (AlphaZero-style)         │
└─────────────────────────────────────────────────────────────────┘
```

### Phase 3: Distributed Architecture

```
┌─────────────────┐         ┌─────────────────┐
│  Node A         │         │  Node B         │
│  ┌───────────┐  │  QUIC   │  ┌───────────┐  │
│  │ Archive   │◄─┼─────────┼─►│ Archive   │  │
│  │ (ETS+CRDT)│  │  sync   │  │ (ETS+CRDT)│  │
│  └───────────┘  │         │  └───────────┘  │
│       │         │         │       │         │
│       ▼         │         │       ▼         │
│  ┌───────────┐  │         │  ┌───────────┐  │
│  │ Evaluator │  │         │  │ Evaluator │  │
│  │ (local)   │  │         │  │ (local)   │  │
│  └───────────┘  │         │  └───────────┘  │
└─────────────────┘         └─────────────────┘
        │                           │
        └───────────┬───────────────┘
                    │
             ┌──────┴──────┐
             │ Macula Mesh │
             │ (DHT/PubSub)│
             └─────────────┘
```

## Risk Assessment

| Risk | Mitigation | Status |
|------|------------|--------|
| riak_dt dependency issues | Implemented custom OR-Set CRDT instead | MITIGATED |
| Mesh sync latency | Archive works locally even without sync | Pending Phase 3 |
| Network serialization overhead | Use binary format, compress | MITIGATED (version-prefixed binary format) |
| Archive size explosion | Aggressive pruning, max size limits | MITIGATED |
| CRDT memory leak | CRDT tracking disabled; needs genome-only storage | ACTIVE - Phase 2 blocked |

## Dependencies

### Required
- None! Custom OR-Set CRDT implemented in `archive_crdt.erl` (avoided `riak_dt` due to age)

### Future (Phase 3)
- `macula_pubsub` (when ready)
- `macula_sdk` (for mesh integration)

## Notes

### Naming Convention: Red Team / Blue Team

The implementation uses a Red Team / Blue Team naming convention:
- **Red Team** = Champions / Hall of Fame / Elite Archive (opponents)
- **Blue Team** = Evolving challengers (current population)

This follows the "Red Queen" hypothesis from evolutionary biology, where the Red Queen (champion) sets the pace that challengers must match to survive.

### Competitive Coevolution Design

This implementation uses **competitive coevolution** with Red Team vs Blue Team dynamics:
- **Red Team Archive**: Elite networks that have proven themselves
- **Blue Team**: Evolving population that must beat Red Team to improve
- **Promotion**: High-performing Blue Team members promoted to Red Team
- **Weighted sampling**: Fitness and recency affect Red Team selection

### Design Benefits
- Fitness-weighted sampling ensures stronger opponents are selected more often
- No artificial ceiling from heuristic opponent quality
- Arms race dynamics emerge naturally
- Age decay prevents stale champions from dominating
- Immigration allows top Red Team members to re-enter Blue Team

## References

- [CRDT Primer](https://crdt.tech/papers.html)
- [OR-Set Paper](https://hal.inria.fr/inria-00555588/document)
- [riak_dt Documentation](https://github.com/basho/riak_dt)
- [Macula Architecture](../../../macula/architecture/)
