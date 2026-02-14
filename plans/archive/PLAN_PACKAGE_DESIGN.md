# faber-neuroevolution Package Plan

**Date:** 2025-12-04
**Priority:** 0
**Status:** COMPLETE ✅
**Completed:** 2025-12-07

---

## Implementation Notes

**Package published to hex.pm:** `faber_neuroevolution`

The package has been fully implemented with all planned features plus additional capabilities:
- NEAT topology evolution (integrated 2025-12-07)
- LTC meta-controller for hyperparameter optimization
- 5 pluggable evolution strategies
- Speciation with NEAT compatibility distance
- 218 passing tests

---

## Overview

Create a new Erlang/Rebar3 hex package that extracts and generalizes the evolutionary training patterns from snake-duel's PopulationServer. This separates training orchestration from faber-tweann (which handles neural network operations).

## Package Details

- **Name:** `faber_neuroevolution`
- **Language:** Erlang/OTP
- **Build:** Rebar3
- **Dependency:** faber_tweann ~> 0.10.0
- **Repo:** `/home/rl/work/github.com/rgfaber/faber-neuroevolution/`

## Module Structure

```
faber-neuroevolution/
├── rebar.config
├── src/
│   ├── faber_neuroevolution_app.erl      # OTP application
│   ├── faber_neuroevolution_sup.erl      # Top supervisor
│   ├── neuroevolution_server.erl          # Main gen_server (population loop)
│   ├── neuroevolution_evaluator.erl       # Behaviour for domain evaluators
│   ├── neuroevolution_genetic.erl         # Crossover & mutation operators
│   ├── neuroevolution_selection.erl       # Selection strategies
│   └── neuroevolution_stats.erl           # Statistics calculation
├── include/
│   └── neuroevolution.hrl                 # Record definitions
├── test/
│   ├── neuroevolution_server_tests.erl
│   ├── neuroevolution_genetic_tests.erl
│   └── neuroevolution_selection_tests.erl
└── guides/
    ├── getting-started.md
    └── custom-evaluator.md
```

## Key Records (neuroevolution.hrl)

```erlang
-record(individual, {
    id :: term(),
    network :: network_evaluator:network(),
    parent1_id :: term() | undefined,
    parent2_id :: term() | undefined,
    fitness = 0.0 :: float(),
    metrics = #{} :: map(),
    generation_born = 1 :: pos_integer(),
    is_survivor = false :: boolean(),
    is_offspring = false :: boolean()
}).

-record(neuro_config, {
    population_size = 50 :: pos_integer(),
    evaluations_per_individual = 10 :: pos_integer(),
    selection_ratio = 0.20 :: float(),
    mutation_rate = 0.10 :: float(),
    mutation_strength = 0.3 :: float(),
    max_generations = infinity :: pos_integer() | infinity,
    network_topology :: {pos_integer(), [pos_integer()], pos_integer()},
    evaluator_module :: module(),
    evaluator_options = #{} :: map(),
    event_handler :: {module(), term()} | undefined
}).

-record(generation_stats, {
    generation :: pos_integer(),
    best_fitness :: float(),
    avg_fitness :: float(),
    worst_fitness :: float(),
    best_individual_id :: term(),
    survivors :: [term()],
    eliminated :: [term()],
    offspring :: [term()]
}).

-record(breeding_event, {
    parent1_id :: term(),
    parent2_id :: term(),
    child_id :: term(),
    generation :: pos_integer()
}).
```

## Evaluator Behaviour

```erlang
-callback evaluate(Individual :: #individual{}, Options :: map()) ->
    {ok, EvaluatedIndividual :: #individual{}} | {error, term()}.

-callback calculate_fitness(Metrics :: map()) -> float().
-optional_callbacks([calculate_fitness/1]).
```

## Key Features

1. **Parallel Evaluation** - spawn_link workers for concurrent evaluation
2. **Sexual Reproduction** - Uniform crossover of parent weights
3. **Mutation** - Weight perturbation with configurable rate/strength
4. **Selection** - Top-N%, tournament, roulette wheel
5. **Lineage Tracking** - parent1_id, parent2_id, generation_born
6. **Event Callbacks** - generation_started, evaluation_progress, generation_complete
7. **Stats Tracking** - best/avg/worst fitness per generation

---

## Implementation Phases

### Phase 1: Project Setup
1. Create rebar.config with faber_tweann dependency
2. Create include/neuroevolution.hrl with all records
3. Create src/faber_neuroevolution.app.src

### Phase 2: Genetic Operators
1. neuroevolution_genetic.erl - crossover_uniform/2, mutate_weights/3
2. neuroevolution_selection.erl - top_n/2, tournament/3, random_select/1, roulette_wheel/1
3. Unit tests for both modules

### Phase 3: Core Server
1. neuroevolution_evaluator.erl - behaviour definition
2. neuroevolution_server.erl - gen_server with:
   - Population creation (using network_evaluator:create_feedforward)
   - Parallel evaluation (spawn_link workers)
   - Generation loop (evaluate → select → breed → mutate)
   - Event callbacks
3. neuroevolution_stats.erl - statistics calculation

### Phase 4: OTP Application
1. faber_neuroevolution_app.erl
2. faber_neuroevolution_sup.erl
3. EDoc documentation
4. Integration tests

### Phase 5: Snake-Duel Integration
1. Create snake_duel_evaluator implementing neuroevolution_evaluator
2. Update snake-duel to use faber_neuroevolution instead of inline PopulationServer
3. Remove duplicated code from snake-duel

---

## Source Files Reference

**Port from (Elixir → Erlang):**
- `/home/rl/work/github.com/rgfaber/macula-snake-duel/lib/snake_duel/training/population_server.ex`
- `/home/rl/work/github.com/rgfaber/macula-snake-duel/lib/snake_duel/training/game_evaluator.ex`

**Depend on:**
- `/home/rl/work/github.com/rgfaber/faber-tweann/src/network_evaluator.erl`

**Pattern reference:**
- `/home/rl/work/github.com/rgfaber/faber-tweann/src/selection_algorithm.erl`
- `/home/rl/work/github.com/rgfaber/faber-tweann/rebar.config`

## Key Differences from Snake-Duel

| Aspect | Snake-Duel (Elixir) | faber_neuroevolution (Erlang) |
|--------|---------------------|-------------------------------|
| Language | Elixir | Pure Erlang |
| Concurrency | Task.async/await | spawn_link/message passing |
| Events | Phoenix.PubSub | Callback behaviour |
| Domain | Snake game specific | Domain-agnostic via behaviour |
| State | Elixir structs | Erlang records |
