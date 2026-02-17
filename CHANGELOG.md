# Changelog

All notable changes to faber-neuroevolution will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.2.2] - 2026-02-17

### Fixed

- **`neuro_config:from_map/1` missing fields** — `lc_chain_config`, `checkpoint_interval`,
  and `checkpoint_config` fields exist in `#neuro_config{}` record but were never set by
  `from_map/1`, always remaining `undefined` regardless of input. Now properly extracted
  from the input map with support for both record and map forms of `lc_chain_config`.

### Added

- `lc_chain_config`, `checkpoint_interval`, `checkpoint_config` in `neuro_config:to_map/1`
- `lc_chain_config_from_map/1` and `lc_chain_config_to_map/1` internal helpers
- Unit tests for `neuro_config:from_map/1` with LC chain config

---

## [1.2.1] - 2026-02-17

### Fixed

- **Multi-champion extraction bug** — `get_population/1` returns the post-breeding
  population where non-elite individuals have fitness reset to 0. Added
  `get_last_evaluated_population/1` which returns the fully-evaluated population
  (sorted by fitness descending) captured before strategy replacement. This
  enables correct top-N champion extraction after training completes.

### Added

- `neuroevolution_server:get_last_evaluated_population/1` API function
- `last_evaluated_population` field in `#neuro_state{}` record

---

## [1.2.0] - 2026-02-17

### Added

- **`mutate_cfc/2`** in `network_factory` — CfC-aware mutation that evolves:
  - Tau (time constant): Gaussian perturbation, clamped to [0.01, 10.0]
  - State bound: Gaussian perturbation, clamped to [0.1, 5.0]
  - Neuron type toggle: ~5% chance to flip standard <-> cfc
- Falls back to standard weight mutation for non-CfC networks

---

## [1.1.0] - 2026-02-17

### Fixed

- Elitism implementation, honor crossover_rate/tournament_size, add stagnation response

---

## [1.0.0] - 2026-02-16

### Stable Release

First stable release. All APIs are considered stable.

### Features (stable)

- **5 Evolution Strategies**: Generational, steady-state, novelty search, MAP-Elites, island model
- **13-Silo Liquid Conglomerate**: Self-tuning subsystems with cross-silo signaling
- **LTC Meta-Controller**: Hierarchical L0/L1/L2 hyperparameter adaptation
- **Domain SDK**: Behaviour-based SDK for building neuroevolution domains
- **Seeded Population**: Continue training from champion networks
- **Distributed Evaluation**: Multi-node evaluation via mesh or Erlang distribution
- **Lineage Tracking**: CQRS-based genealogy with event store backend
- **93 Behavioral Events**: Per-silo event system with pub/sub
- **Network Checkpointing**: Save/load evolved networks at milestones
- **Layer-Specific Mutation**: Different rates for reservoir vs readout layers
- **944 tests passing**

### Removed

- **Self-play manager** — Incomplete feature, never implemented. Self-play evaluation will be handled at the application level (e.g., hecate-daemon's Snake Gladiators).
- **Coevolution strategy** — `coevolution_sup.erl`, `coevolution_manager.erl`, `coevolution_trainer.erl`, `red_team_archive.erl`, `archive_crdt.erl` removed. Red Team vs Blue Team coevolution will be developed as a standalone strategy when needed.
- **Self-play guides and assets** — Removed `guides/self-play.md` and related SVG assets
- **Self-play test stubs** — Removed `.pending` test files for unimplemented modules

### Changed

- `neuroevolution_server` no longer initializes self-play manager
- `neuroevolution_evaluator:evaluate_batch_distributed/4` simplified to `/3` (no SelfPlayManager param)
- `faber_neuroevolution_sup` no longer starts coevolution supervisor
- Morphology specs use generic `archive` source/target instead of `opponent_archive`
- Dependency: `faber_tweann ~> 1.0` (was `~> 0.1.0`)

---

## [0.2.0] - 2026-02-16

### Added
- **Seeded population creation** in `generational_strategy.erl`

---

## [0.1.0] - 2026-02-14

### Changed
- Renamed from `macula-neuroevolution` to `faber-neuroevolution` under `rgfaber` organization
- Renamed dependency from `macula_tweann` to `faber_tweann`
- Renamed NIF dependency from `macula_nn_nifs` to `faber_nn_nifs`
- Updated all module prefixes from `macula_neuroevolution` to `faber_neuroevolution`
- Updated copyright to `R.G. Lefever`
- Reset version to 0.1.0 for fresh start under new name

### Planned
- Integration tests for cross-silo communication
- Docker compose for multi-node mesh testing
- Integration tests with 3-node mesh cluster

---

*History from macula-neuroevolution below*

## [0.29.0] - 2025-12-29

### Changed

- **Renamed Agent SDK to Domain SDK** - Better reflects purpose: building neuroevolution domains
  - `src/agent_sdk/` → `src/domain_sdk/`
  - `guides/agent-sdk.md` → `guides/domain-sdk.md`
  - Module names unchanged (`agent_definition`, `agent_sensor`, etc.) as they define agents within domains

---

## [0.28.0] - 2025-12-29

### Summary
**Domain SDK** - Complete behaviour-based SDK enabling domain applications to build neuroevolution domains through Erlang behaviours. A domain defines agents (sensors, actuators), environments (where agents are evaluated), and evaluators (how performance becomes fitness).

### Added

#### Domain SDK Behaviours (`src/domain_sdk/`)

- **agent_definition.erl** - Agent identity and network topology behaviour
  - Callbacks: `name/0`, `version/0`, `network_topology/0`
  - Type: `network_topology() :: {InputCount, HiddenLayers, OutputCount}`
  - API: `validate/1`, `get_info/1`

- **agent_sensor.erl** - Sensory input behaviour
  - Callbacks: `name/0`, `input_count/0`, `read/2`
  - `input_count/0` = number of NN input nodes the sensor produces
  - API: `validate/1`, `get_info/1`, `validate_values/2`

- **agent_actuator.erl** - Motor output behaviour
  - Callbacks: `name/0`, `output_count/0`, `act/3`
  - `output_count/0` = number of NN output nodes the actuator consumes
  - API: `validate/1`, `get_info/1`, `validate_outputs/2`

- **agent_environment.erl** - Episode lifecycle behaviour
  - Callbacks: `name/0`, `init/1`, `spawn_agent/2`, `tick/2`, `apply_action/3`, `is_terminal/2`, `extract_metrics/2`
  - API: `validate/1`, `get_info/1`

- **agent_bridge.erl** - Orchestration module (not a behaviour)
  - Orchestrates sense→think→act cycle
  - Validates topology matches sensor inputs and actuator outputs
  - API: `new/1`, `validate/1`, `sense/3`, `act/4`, `sense_think_act/4`, `run_episode/3`

- **agent_evaluator.erl** - Fitness calculation behaviour
  - Callbacks: `name/0`, `calculate_fitness/1`, `fitness_components/1` (optional)
  - API: `validate/1`, `get_info/1`, `evaluate/2`, `evaluate_with_breakdown/2`

#### Tests (88 new tests)

- **agent_definition_tests.erl** - 13 tests for definition validation
- **agent_sensor_tests.erl** - 15 tests for sensor validation
- **agent_actuator_tests.erl** - 17 tests for actuator validation
- **agent_environment_tests.erl** - 15 tests for environment validation
- **agent_bridge_tests.erl** - 14 tests for bridge orchestration
- **agent_evaluator_tests.erl** - 14 tests for evaluator validation
- 44 test fixture modules for comprehensive edge case testing

### Fixed

- **signal_router.erl** - Added `domain_` prefix for cross-silo signal routing
  - Domain signals now properly routed via `domain_signal_name/1` helper

### Architecture Notes

The Domain SDK provides a clean separation between:
- **Neural Network Perspective**: Sensors produce inputs TO the network, actuators consume outputs FROM the network
- **Flat Interface**: All sensor/actuator I/O uses flat lists (internal can use tensors)
- **Topology Validation**: Bridge validates that sensor input counts match network inputs, actuator output counts match network outputs
- **Domain Independence**: Behaviours are domain-agnostic; domain applications implement callbacks

---

## [0.27.0] - 2025-12-27

### Summary
**Domain Signals for Silo Communication** - External applications can now inform silo decision-making by emitting categorized signals that route to appropriate silos.

### Added

- **Domain Signals Behaviour** (`src/domain/domain_signals.erl`):
  - 13 signal categories mapping to silos (ecological, competitive, cultural, etc.)
  - Signal levels (l0, l1, l2) for hierarchy targeting
  - `signal_spec/0` callback for signal definitions
  - `emit_signals/2` callback for signal emission

- **Signal Router** (`src/silos/signal_router.erl`):
  - Routes domain signals to silos by category
  - `register_domain_module/1` for application configuration
  - `emit_from_domain/2` convenience function
  - Integrates with `lc_cross_silo` for signal delivery

- **Domain Signal Validation** in `lc_cross_silo.erl`:
  - Accepts signals from `domain` source
  - Validates `domain_*` prefixed signals
  - Routes to any valid silo category

- **Tests** (`test/signal_router_tests.erl`):
  - 8 test cases covering routing, clamping, registration

- **Documentation**:
  - `guides/domain-signals.md` - Comprehensive usage guide
  - `assets/domain-signals.svg` - Architecture diagram

---

## [0.26.0] - 2025-12-27

### Summary
**Screaming Architecture Refactoring** - Major reorganization following vertical slicing and screaming architecture principles. Folder names now express intent, not technical layers.

### Changed

- **Directory Structure Reorganization**:
  - `src/liquid_conglomerate/` → `src/silos/` (self-tuning subsystems)
  - `src/mesh/` → `src/distribute/` (distributed evaluation)
  - `src/lc_morphologies/` → `src/meta/` (L1/L2 adaptation)
  - `src/competitive_coevolution/` → `src/strategies/coevolution/`
  - Root-level files organized into purpose-driven folders

- **New Folder Structure**:
  - `evolve/` - Core evolution loop (neuroevolution_server, genetic, selection, speciation)
  - `strategies/` - Evolution strategies (generational, steady_state, novelty, etc.)
  - `evaluate/` - Fitness computation (evaluator, worker, nif_network)
  - `silos/` - Self-tuning subsystems (13 silo folders, each vertical)
  - `meta/` - L1/L2 adaptation (morphologies, controllers, trainers)
  - `distribute/` - Mesh evaluation
  - `persist/` - Checkpoints
  - `stats/` - Monitoring
  - `config/` - Configuration

### Added

- **Domain Bridge Behaviours** (`src/domain/`):
  - `domain_sensors.erl` - Behaviour for domain sensor providers
  - `domain_actuators.erl` - Behaviour for domain actuator consumers
  - `domain_rewards.erl` - Behaviour for domain reward providers
  - Enables domain-agnostic neuroevolution with pluggable domains

### Documentation

- Updated all guides to reference new paths (`silos/` instead of `liquid_conglomerate/`)

---

## [0.25.0] - 2025-12-26

### Summary
**Network Checkpointing, Inference Mode & Real Population Metrics** - Add checkpoint_manager for saving evolved networks at milestones, inference mode for Resource Silo, and wire up real population metrics to LC sensors.

### Added

- **Network Checkpoint Manager** (`checkpoint_manager.erl`):
  - Save networks at key milestones: fitness records, generation intervals, training completion
  - Checkpoint configuration via `checkpoint_config` in neuro_config
  - `save_checkpoint/2,3` - Save individual with metadata
  - `load_latest/0,1` - Load most recent checkpoint
  - `load_best_fitness/0,1` - Load checkpoint with highest fitness
  - `list_checkpoints/0,1` - List all checkpoints
  - `prune_checkpoints/1` - Remove old checkpoints
  - Automatic pruning of old checkpoints (configurable max per reason)
  - Checkpoints include: network, fitness, generation, evaluations, timestamp

- **Checkpoint Configuration** in `neuro_config`:
  - `checkpoint_dir` - Directory for checkpoint files (default: "_checkpoints")
  - `save_on_fitness_record` - Save when new best fitness (default: true)
  - `generation_interval` - Save every N generations (0 = disabled)
  - `max_checkpoints_per_reason` - Max checkpoints to keep (default: 20)

- **Inference Mode for Resource Silo**:
  - `mode` config option: `training` (default) or `inference`
  - `set_mode/1` - Switch mode at runtime
  - `get_mode/0` - Get current mode
  - Inference mode behavior:
    - 5 second sample interval (vs 1s in training)
    - `should_pause()` always returns false
    - `force_gc` has no effect (prioritizes latency)
    - Still publishes monitoring events for observability

### Changed
- `neuroevolution_server.erl`:
  - Integrated checkpoint saving on fitness records, generation intervals, training completion
  - Added `maybe_init_checkpoint_manager/1` and `maybe_save_checkpoint/3` helpers
  - **Wire up real population metrics** for task_silo sensors:
    - `diversity_index` - computed from population fitness variance
    - `species_count_ratio` - actual species / expected species
    - `avg_network_complexity` - from top individuals' complexity
    - `resource_pressure_signal` - from cached resource_silo state
  - Added `compute_diversity_index/2` and `compute_avg_complexity/2` helpers
  - Added `calculate_avg_network_size/1` and `estimate_network_size/1` for emergent_metrics

- `task_silo.erl`:
  - Store and use real population metrics instead of hardcoded 0.5 placeholders
  - Added state fields: `diversity_index`, `species_count_ratio`, `avg_network_complexity`, `prev_complexity`, `resource_pressure_signal`
  - Compute `complexity_velocity` from previous vs current complexity

- `meta_controller.erl`:
  - Compute `evaluations_used` from population_size (was hardcoded 1)
  - Compute `fitness_per_evaluation` properly

- `lc_population.erl`:
  - Clarified `active_exoself = self()` design (intentional synchronous operation)

- `resource_silo.erl`:
  - Updated documentation with operating modes section
  - Added mode to state map from `get_state/0`

### Tests
- 9 new tests for checkpoint_manager operations
- All 446 tests passing

---

## [0.24.1] - 2025-12-26

### Summary
**Event-Driven L0 Actuator Architecture** - Refactored L0 actuators from imperative (direct calls) to reactive (event-driven) pattern. Actuators now publish events, and the neuroevolution_server subscribes and reacts.

### Changed (BREAKING ARCHITECTURE)

**Event-Driven Refactoring:**
- **`task_l0_actuators.erl`**: Now publishes events instead of calling `neuroevolution_server:update_config/2`
  - Publishes `<<"l0.evolution_params">>` with mutation rates, selection ratio, topology rates
  - Publishes `<<"l0.archive_params">>` for future self-play integration
  - Decoupled from neuroevolution_server - only knows about events

- **`resource_l0_actuators.erl`**: Now publishes events instead of direct calls
  - Publishes `<<"l0.resource_params">>` with evaluation timeout, concurrency settings
  - Publishes `<<"silo.resource.pressure">>` for cross-silo signaling
  - Publishes `<<"l0.archive_gc_pressure">>` for archive management

- **`neuroevolution_server.erl`**: Now subscribes to L0 events and reacts
  - Subscribes to `<<"l0.evolution_params">>` and `<<"l0.resource_params">>` in init
  - New `handle_info/2` clauses for `{neuro_event, Topic, Event}` messages
  - New `apply_l0_evolution_params/2` and `apply_l0_resource_params/2` reactive handlers

### Added
- **Event-Driven Refactoring Plan**: `plans/PLAN_EVENT_DRIVEN_REFACTORING.md`
  - Documents 57 imperative violations across faber-tweann and faber-neuroevolution
  - Provides remediation plan in 5 phases
  - Defines event naming conventions and topic hierarchy

### Architecture Principle
```
BEFORE (Imperative):
  L0 Actuators → CALLS → neuroevolution_server:update_config(Params)

AFTER (Event-Driven):
  L0 Actuators → PUBLISHES → <<"l0.evolution_params">> event
                                   ↓
                     neuroevolution_events pub/sub
                                   ↓
  neuroevolution_server → REACTS → updates config
```

### Notes
- This is the first step in a larger event-driven architecture migration
- See `plans/PLAN_EVENT_DRIVEN_REFACTORING.md` for full remediation plan

---

## [0.24.0] - 2025-12-26

### Summary
**Layer-Specific Mutation Rates with L0 Dynamic Control** - Implement configurable per-layer mutation rates, allowing different mutation strategies for reservoir (hidden) vs readout (output) layers. The L0 Task Silo can dynamically adjust these rates based on training progress.

### Added
- **Layer-specific mutation configuration** in `neuro_config.erl`:
  - `reservoir_mutation_rate` - Mutation rate for hidden layers (default: 0.05)
  - `reservoir_mutation_strength` - Mutation strength for hidden layers (default: 0.2)
  - `readout_mutation_rate` - Mutation rate for output layer (default: 0.2)
  - `readout_mutation_strength` - Mutation strength for output layer (default: 0.5)
  - `layer_mutation_mode` - Mode: `uniform` (legacy) or `layer_specific` (new)

- **`neuroevolution_genetic:mutate_layer_specific/2`** - New mutation function that applies different rates to reservoir vs readout weights

- **L0 dynamic mutation control** in Task Silo:
  - `task_l0_actuators:set_mutation_rates/1` - L0 can dynamically adjust mutation rates
  - `task_l0_morphology:mutation_rate_inputs/0` - Provides training state as L0 inputs

- **New SVG diagram**: `assets/layer-specific-mutation.svg` - Visualization of layer-specific mutation architecture

- **Comprehensive test suite** in `neuroevolution_genetic_tests.erl`:
  - Tests for layer-specific mutation
  - Tests for L0 dynamic control integration
  - Tests for backward compatibility with uniform mode

### Changed
- **Updated dependency**: `faber_tweann` ~> 0.17.0 (was ~> 0.16.0)
- **Evolution strategies** now respect layer-specific mutation configuration:
  - `generational_strategy.erl`
  - `novelty_strategy.erl`
  - `steady_state_strategy.erl`

### Migration
Existing configurations continue to work unchanged. To enable layer-specific mutation:

```erlang
Config = neuro_config:new(#{
    layer_mutation_mode => layer_specific,
    reservoir_mutation_rate => 0.05,
    readout_mutation_rate => 0.20
}).
```

---

## [0.23.3] - 2025-12-26

### Added
- **lc_sensor_publisher.erl** - Unified sensor publisher for extension silos
  - Polls all enabled extension silos for sensor data
  - Publishes events to `silo_sensors` topic for UI updates
  - Supports runtime enable/disable via API
  - Change detection with throttling (10Hz max)
  - Handles temporal, competitive, economic, social, morphological,
    communication, developmental, cultural, regulatory, ecological silos

### Changed
- **lc_supervisor.erl** - Added lc_sensor_publisher to supervision tree
  - Notifies sensor publisher on silo enable/disable
  - Publishes `silo_status_changed` events for UI updates

---

## [0.23.2] - 2025-12-24

### Fixed
- Fix broken documentation link in behavioral-events.md

---

## [0.23.1] - 2025-12-24

### Added
- **guides/behavioral-events.md** - Comprehensive user guide for 93 behavioral events
  - Event emission and subscription examples
  - Complete event catalog by silo
  - Event metadata structure
  - Best practices for event handling

---

## [0.23.0] - 2025-12-24

### Summary
**Macula Mesh Distribution & Behavioral Events** - Add distributed fitness evaluation across macula mesh nodes and comprehensive per-silo behavioral event infrastructure.

### Added

#### Mesh Distribution (Phase 1)
- **mesh_sup.erl** - Supervisor for mesh distribution components
- **evaluator_pool_registry.erl** - Track remote evaluator capacity with load balancing
- **macula_mesh.erl** - Macula integration facade (conditional compilation)
- **distributed_evaluator.erl** - RPC-based evaluation dispatch with retry
- **mesh_tests.erl** - 19 unit tests for mesh components
- `evaluation_mode = mesh` option in neuro_config
- `mesh_config` field for realm and preferences
- `rebar3 as mesh compile` for macula-enabled builds

#### Behavioral Events (93 events across 11 silos)
- **lc_events_common.hrl** - Shared macros and record definitions
- **neuroevolution_behavioral_events.erl** - Event emission/subscription API
- Per-silo event modules: temporal (9), economic (10), morphological (9), competitive (10), social (9), cultural (9), developmental (9), regulatory (9), ecological (9), communication (6), distribution (4)
- Event emission with `emit_*` functions and `to_map/1` conversion

#### Mesh Features
- Load balancing based on capacity, latency, and error rate
- Automatic retry on evaluation failure
- Local preference (configurable, default 30%)
- Graceful fallback when macula not compiled in
- EMA-based latency tracking for evaluator selection

### Tests
- 437 tests total (19 new mesh tests, behavioral event tests)

---

## [0.19.0] - 2025-12-23

### Summary
**Complete 13-Silo Liquid Conglomerate Architecture** - Full implementation of all 13 specialized silos with shared infrastructure, cross-silo signaling, and comprehensive unit tests.

### Added

#### Infrastructure Modules
- **lc_silo_behavior.erl** - Common behaviour with `get_silo_type/0` and `get_time_constant/0` callbacks
- **lc_ets_utils.erl** - Shared ETS utilities (CRUD, time-based operations, aggregations)
- **include/lc_silos.hrl** - Common silo records and type definitions
- **include/lc_signals.hrl** - Signal type definitions for cross-silo communication

#### Phase 2: High-Value Silos
- **temporal_silo.erl** - Episode timing, learning rates, convergence tracking (τ=10)
- **economic_silo.erl** - Compute budgets, energy economics, Gini coefficient (τ=20)
- **morphological_silo.erl** - Network complexity, pruning, efficiency metrics (τ=30)

#### Phase 3: Competition Silos
- **competitive_silo.erl** - Elo ratings, opponent archive, matchmaking (τ=15)
- **social_silo.erl** - Reputation, coalitions, social networks (τ=25)

#### Phase 4: Learning Silos
- **cultural_silo.erl** - Innovations, traditions, meme spread (τ=35)
- **developmental_silo.erl** - Ontogeny, plasticity, critical periods (τ=40)
- **regulatory_silo.erl** - Gene expression, module activation, epigenetics (τ=45)

#### Phase 5: Environment Silos
- **ecological_silo.erl** - Niches, resource pools, environmental stress (τ=50)
- **communication_silo.erl** - Vocabulary, messaging, coordination (τ=55)

#### Phase 6: Distribution Silo
- **distribution_silo.erl** - Islands, migration, load balancing (τ=60)

#### Cross-Silo Signal Routing
- Updated **lc_cross_silo.erl** with ~60+ signal types for all 13 silos
- Signal validation, decay, and batch emission support
- Updated **lc_supervisor.erl** with extension silo support

#### Unit Tests (340 silo tests)
- **lc_silo_behavior_tests.erl** - Behaviour contract tests
- **lc_ets_utils_tests.erl** - ETS utility tests
- **lc_cross_silo_tests.erl** - Signal routing tests
- **temporal_silo_tests.erl** - 24 tests
- **economic_silo_tests.erl** - 20 tests
- **morphological_silo_tests.erl** - 14 tests
- **competitive_silo_tests.erl** - 26 tests
- **social_silo_tests.erl** - 22 tests
- **cultural_silo_tests.erl** - 26 tests
- **developmental_silo_tests.erl** - 22 tests
- **regulatory_silo_tests.erl** - 20 tests
- **ecological_silo_tests.erl** - 24 tests
- **communication_silo_tests.erl** - 22 tests
- **distribution_silo_tests.erl** - 24 tests

#### Documentation
- 14 new silo guides in `guides/silos/`:
  - lc-overview.md, task-silo.md, resource-silo.md, distribution-silo.md
  - temporal-silo.md, economic-silo.md, morphological-silo.md
  - competitive-silo.md, social-silo.md, cultural-silo.md
  - ecological-silo.md, developmental-silo.md, regulatory-silo.md
  - communication-silo.md
- SVG diagrams for silo architecture

### Architecture

Each silo implements the `lc_silo_behavior` with:
- **gen_server** for state management
- **ETS tables** for persistent collections via `lc_ets_utils`
- **L0 Sensors** (normalized 0.0-1.0) for state observation
- **L0 Actuators** with configurable bounds
- **Cross-silo signals** via `lc_cross_silo` router
- **Time constants** (τ) for multi-timescale adaptation

### Test Results
- 726 tests passing (340 silo tests + 386 other tests)
- 4 tests skipped (pre-existing self_play_tests for unimplemented modules)
- Dialyzer: 192 pre-existing warnings (unrelated to silos)

---

## [0.18.4] - 2025-12-23

### Fixed
- Minor bug fixes

---

## [0.18.2] - 2025-12-23

### Summary
**Complete SVG Documentation** - All ASCII diagrams replaced with SVG files for hex.pm rendering.

### Added
- 26 new SVG diagrams covering all guides:
  - `evolution-lifecycle.svg`, `selection-breeding.svg`, `steady-state-flow.svg`, `island-topology.svg`, `novelty-search.svg`, `map-elites-grid.svg` (evolution-strategies.md)
  - `liquid-conglomerate-full.svg`, `hierarchy-levels.svg`, `adaptive-hyperparameters.svg`, `training-dashboard.svg`, `distributed-mesh.svg` (liquid-conglomerate.md)
  - `game-ai-architecture.svg`, `robot-architecture.svg`, `trading-architecture.svg`, `edge-ai-mesh.svg` (inference-scenarios.md)
  - `swarm-architecture.svg`, `swarm-mesh.svg`, `distributed-training.svg` (swarm-robotics.md)
  - `species-population.svg`, `species-hierarchy.svg` (topology-evolution.md)
  - `lineage_cqrs_architecture.svg` (lineage-tracking.md)
  - `self_play_first_gen.svg`, `self_play_archive_selection.svg` (self-play.md)
  - `neat_gene_alignment.svg` (topology-evolution.md)
  - `architecture-overview.svg` (overview.md)
  - `individual-structure.svg` (getting-started.md)
  - `liquid-conglomerate-v2.svg` (meta-controller.md)

### Changed
- Updated all guide files to reference SVG diagrams instead of ASCII art
- Total assets: 37 SVG files

---

## [0.18.1] - 2025-12-23

### Summary
**Documentation Fix** - Replace ASCII diagrams with SVG, fix EDoc generation errors.

### Fixed
- EDoc errors from `@doc` tags on record definitions and `-callback` declarations
- ASCII diagrams causing backtick parse errors in hex.pm docs

### Added
- `assets/lc_silo_chain.svg` - Hierarchical L0/L1/L2 LTC TWEANN architecture
- `assets/distributed_evaluation.svg` - Distributed batch evaluation diagram

---

## [0.18.0] - 2025-12-23

### Summary
**Lineage Tracking Release** - Complete genealogy tracking infrastructure with CQRS-compliant behaviour, event records, and comprehensive documentation.

### Added

#### Lineage Event System (CQRS Architecture)
- **neuroevolution_lineage_events.erl** - Behaviour with required and optional callbacks

**Required Callbacks (Event Store)**:
- `init/1` - Initialize event store backend
- `persist_event/2` - Persist single event to appropriate stream
- `persist_batch/2` - Batch persist for efficiency
- `read_stream/3` - Read raw events from a stream
- `subscribe/3` - Subscribe to stream events (for projections)
- `unsubscribe/3` - Unsubscribe from stream

**Optional Callbacks (Queries via Projections)**:
- `get_breeding_tree/3` - Build ancestry tree up to N generations
- `get_fitness_trajectory/2` - Get fitness over time
- `get_mutation_history/2` - Get all mutations for an individual
- `get_knowledge_transfers/2` - Get mentor/student events
- `get_by_causation/2` - Find events by causation ID

**CQRS Design**: Required callbacks handle event store operations. Optional callbacks provide query capabilities and should be implemented using internal projections (read models), not by scanning events directly.

#### Event Record Definitions (lineage_events.hrl)
- **Birth Events**: offspring_born, pioneer_spawned, clone_produced, immigrant_arrived
- **Death Events**: individual_culled, lifespan_expired, individual_perished
- **Lifecycle Events**: individual_matured, fertility_waned
- **Fitness Events**: fitness_evaluated, fitness_improved, fitness_declined, champion_crowned
- **Mutation Events**: mutation_applied, neuron_added, neuron_removed, connection_added, connection_removed, weight_perturbed
- **Species Events**: lineage_diverged, species_emerged, lineage_ended, lineage_merged
- **Knowledge Transfer Events**: knowledge_transferred, skill_imitated, behavior_cloned, weights_grafted, structure_seeded, mentor_assigned, mentorship_concluded
- **Epigenetic Events**: mark_acquired, mark_inherited, mark_decayed
- **Coalition Events**: coalition_formed, coalition_dissolved, coalition_joined
- **Population Events**: generation_completed, population_initialized, population_terminated, stagnation_detected, breakthrough_achieved, carrying_capacity_reached, catastrophe_occurred

#### Unit Tests
- **test/neuroevolution_lineage_events_tests.erl** - 23 tests for behaviour contract
  - 17 tests for required callbacks (event store operations)
  - 6 tests for optional callbacks (query operations)
- **test/mock_lineage_backend.erl** - In-memory mock implementation (implements all callbacks)

#### Documentation
- **guides/lineage-tracking.md** - CQRS architecture guide with usage examples
- **assets/lineage_architecture.svg** - Architecture overview diagram

### Stream Design
Events are routed to streams by entity type:
- `individual-{id}` - Birth, death, fitness, mutations, knowledge transfer
- `species-{id}` - Speciation, lineage divergence/merge
- `population-{id}` - Generation, capacity, catastrophe
- `coalition-{id}` - Coalition lifecycle

### Architecture Notes
- **Required Callbacks**: Event store operations (persist, read, subscribe)
- **Optional Callbacks**: Query operations (implemented via projections internally)
- **Mock Backend**: Implements all callbacks (scans events directly for queries)
- **Production Backend**: Should use projections for optional callbacks

### Test Results
- 419 tests passing (23 new lineage tests)
- 4 tests skipped (self_play_tests - pre-existing missing opponent_archive module)
- Dialyzer clean

---

## [0.13.0] - 2025-12-11

### Summary
**Liquid Conglomerate v2 Release** - Complete hierarchical meta-learning architecture with 3 specialized silos (Resource, Task, Distribution), L1/L2 controllers, and cross-silo communication.

### Added

#### LC v2 Architecture
- **lc_supervisor.erl** - Supervises all LC v2 child processes
- **lc_cross_silo.erl** - Signal routing between silos with validation and decay
- **lc_reward.erl** - Cooperative reward computation for all silos
- **lc_l1_controller.erl** - Generic L1 hyperparameter tuning
- **lc_l2_controller.erl** - Strategic meta-tuning with slow exploration

#### Silo Morphologies (TWEANN Sensor/Actuator Definitions)
- **resource_l0_morphology.erl** - Resource silo (13 sensors, 8 actuators)
- **task_l0_morphology.erl** - Task silo (16 sensors, 12 actuators)
- **distribution_l0_morphology.erl** - Distribution silo (14 sensors, 10 actuators)

#### Sensor & Actuator Implementations
- **resource_l0_sensors.erl** - System metrics collection
- **resource_l0_actuators.erl** - Resource control application
- **task_l0_sensors.erl** - Evolution statistics collection
- **task_l0_actuators.erl** - Evolution parameter application
- **distribution_l0_sensors.erl** - Network metrics collection
- **distribution_l0_actuators.erl** - Distribution control application

#### Unit Tests (108 new tests)
- **test/lc_morphology_tests.erl** - 43 tests for morphology modules
- **test/lc_reward_tests.erl** - 34 tests for reward computation
- **test/lc_l1_controller_tests.erl** - 15 tests for L1 controller
- **test/lc_l2_controller_tests.erl** - 16 tests for L2 controller

#### Documentation
- **guides/cooperative-silos.md** - Cross-silo communication guide
- **guides/assets/lc_architecture.svg** - Architecture overview diagram
- **guides/assets/lc_hierarchical_learning.svg** - Hierarchical learning diagram
- **guides/assets/lc_cross_silo_signals.svg** - Signal routing diagram
- **guides/assets/lc_silo_tweann_io.svg** - L0 TWEANN sensors/actuators diagram
- **guides/assets/lc_silo_interactions.svg** - Detailed cross-silo interaction flow diagram
- **guides/assets/lc_feedback_loop.svg** - Feedback loop between Resource and Task silos
- Updated **guides/meta-controller.md** with implementation roadmap

### Changed
- Updated implementation roadmap (Phases 3-6 complete)
- Enhanced LC v2 documentation with cooperative reward signals

### Theoretical Foundation
- Multi-timescale separation: Distribution (τ=1), Resource (τ=5), Task (τ=50)
- Hierarchical control: L0 (safety), L1 (feedback), L2 (feedforward)
- Graceful degradation: L2 failure → L1 fallback → L0 safety

### Test Results
- 326 tests passing (108 new LC v2 tests)
- Dialyzer clean
- All documentation links validated

---

## [0.12.1] - 2025-12-07

### Summary
**Safety & Dependency Update** - MAP-Elites grid validation and faber-tweann 0.13.0 compatibility.

### Changed
- **rebar.config**: Updated faber_tweann dependency from ~> 0.12.0 to ~> 0.13.0

### Fixed
- **map_elites_strategy.erl**: Added grid size validation to prevent unbounded memory growth
  - Maximum 10,000 cells enforced (e.g., 10 bins × 4 dimensions, or 100 bins × 2 dimensions)
  - Descriptive error with hint when configuration exceeds limit
  - Prevents potential OOM from high-dimensional behavior spaces

### Test Results
- 218 tests passing
- Dialyzer: 35 pre-existing warnings (unrelated to this release)

---

## [0.12.0] - 2025-12-01

### Summary
**Evolution Strategies & Meta-Learning Release** - Multiple evolution strategies with LTC meta-controller.

### Added

#### Evolution Strategies
- **generational_strategy.erl** - Classic generational GA
- **steady_state_strategy.erl** - Continuous replacement with age tracking
- **novelty_strategy.erl** - Novelty search with behavior archive
- **map_elites_strategy.erl** - Quality-Diversity algorithm with behavior grid
- **island_strategy.erl** - Island model with migration topology

#### Meta-Learning
- **meta_controller.erl** - LTC-based hyperparameter controller
- **meta_trainer.erl** - Advantage estimation and gradient computation
- **meta_reward.erl** - Multi-component reward calculation

#### Core Infrastructure
- **neuroevolution_server.erl** - Main gen_server for training orchestration
- **neuroevolution_events.erl** - Event system for training notifications
- **neuroevolution_selection.erl** - Selection operators (tournament, roulette, top-N)
- **neuroevolution_genetic.erl** - Crossover and mutation operators
- **neuroevolution_speciation.erl** - NEAT-style speciation support
- **neuroevolution_stats.erl** - Population statistics

#### Documentation
- Comprehensive guides for all features
- Architecture overview with diagrams
- LTC meta-controller explanation
- Liquid Conglomerate vision document

---

## [0.11.0] - 2025-11-15

### Summary
Initial public release with core neuroevolution functionality.

### Added
- Basic population management
- Parallel fitness evaluation
- Weight mutation and crossover
- Event callback system
- Integration with faber_tweann

---

[Unreleased]: https://github.com/rgfaber/faber-neuroevolution/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/rgfaber/faber-neuroevolution/compare/v0.1.0...v0.2.0
[0.28.0]: https://github.com/rgfaber/faber-neuroevolution/compare/v0.27.0...v0.28.0
[0.27.0]: https://github.com/rgfaber/faber-neuroevolution/compare/v0.26.0...v0.27.0
[0.26.0]: https://github.com/rgfaber/faber-neuroevolution/compare/v0.25.0...v0.26.0
[0.25.0]: https://github.com/rgfaber/faber-neuroevolution/compare/v0.24.1...v0.25.0
[0.24.1]: https://github.com/rgfaber/faber-neuroevolution/compare/v0.24.0...v0.24.1
[0.24.0]: https://github.com/rgfaber/faber-neuroevolution/compare/v0.23.3...v0.24.0
[0.23.3]: https://github.com/rgfaber/faber-neuroevolution/compare/v0.23.2...v0.23.3
[0.23.2]: https://github.com/rgfaber/faber-neuroevolution/compare/v0.23.1...v0.23.2
[0.23.1]: https://github.com/rgfaber/faber-neuroevolution/compare/v0.23.0...v0.23.1
[0.23.0]: https://github.com/rgfaber/faber-neuroevolution/compare/v0.19.0...v0.23.0
[0.19.0]: https://github.com/rgfaber/faber-neuroevolution/compare/v0.18.4...v0.19.0
[0.18.4]: https://github.com/rgfaber/faber-neuroevolution/compare/v0.18.2...v0.18.4
[0.18.2]: https://github.com/rgfaber/faber-neuroevolution/compare/v0.18.1...v0.18.2
[0.18.1]: https://github.com/rgfaber/faber-neuroevolution/compare/v0.18.0...v0.18.1
[0.18.0]: https://github.com/rgfaber/faber-neuroevolution/compare/v0.17.0...v0.18.0
[0.13.0]: https://github.com/rgfaber/faber-neuroevolution/compare/v0.12.1...v0.13.0
[0.12.1]: https://github.com/rgfaber/faber-neuroevolution/compare/v0.12.0...v0.12.1
[0.12.0]: https://github.com/rgfaber/faber-neuroevolution/compare/v0.11.0...v0.12.0
[0.11.0]: https://github.com/rgfaber/faber-neuroevolution/releases/tag/v0.11.0
