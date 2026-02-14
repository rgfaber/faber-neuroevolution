# Event-Carried State Transfer Refactoring

**Date:** 2025-12-08
**Status:** Implementation Required

## Problem

Current pattern floods state actors:
```
Event → Subscriber → get_state() → Actor overloaded
```

The neuroevolution_server receives events and then subscribers call `get_state/1`
to fetch the full state, causing unnecessary load on the gen_server.

## Solution

Events carry rich payloads - no follow-up queries needed.

## Multi-Topic Design

### Topic Constructors (neuroevolution_events.erl)

Add topic constructor functions to separate event types:

```erlang
-spec topic_generation(realm()) -> binary().
topic_generation(Realm) -> <<"neuro.", Realm/binary, ".generation">>.

-spec topic_population(realm()) -> binary().
topic_population(Realm) -> <<"neuro.", Realm/binary, ".population">>.

-spec topic_species(realm()) -> binary().
topic_species(Realm) -> <<"neuro.", Realm/binary, ".species">>.

-spec topic_training(realm()) -> binary().
topic_training(Realm) -> <<"neuro.", Realm/binary, ".training">>.

-spec topic_meta(realm()) -> binary().
topic_meta(Realm) -> <<"neuro.", Realm/binary, ".meta">>.
```

### Event Payloads (neuroevolution_server.erl)

#### generation_complete

```erlang
publish_generation_complete(State) ->
    Payload = #{
        realm => State#state.realm,
        generation => State#state.generation,
        generation_stats => #{
            best_fitness => get_best_fitness(State),
            avg_fitness => get_avg_fitness(State),
            worst_fitness => get_worst_fitness(State),
            population_size => length(State#state.population)
        },
        history_point => {State#state.generation,
                         get_best_fitness(State),
                         get_avg_fitness(State)},
        meta_params => get_current_meta_params(State),
        silo_state => get_silo_state(State),
        convergence_status => calculate_convergence(State),
        time_per_generation_ms => State#state.last_gen_time,
        timestamp => erlang:system_time(millisecond)
    },
    neuroevolution_events:publish(topic_generation(State#state.realm),
                                   generation_complete, Payload).
```

#### population_evaluated

```erlang
publish_population_evaluated(State) ->
    PopulationSummary = [
        #{
            id => I#individual.id,
            fitness => I#individual.fitness,
            is_survivor => I#individual.is_survivor,
            is_offspring => I#individual.is_offspring,
            complexity => calculate_complexity(I)
        } || I <- State#state.population
    ],
    Payload = #{
        realm => State#state.realm,
        generation => State#state.generation,
        population => PopulationSummary,
        top_individuals => lists:sublist(PopulationSummary, 5),
        best_fitness => get_best_fitness(State),
        avg_fitness => get_avg_fitness(State),
        timestamp => erlang:system_time(millisecond)
    },
    neuroevolution_events:publish(topic_population(State#state.realm),
                                   population_evaluated, Payload).
```

## State Classification

### neuroevolution_server State Fields

| Field | Category | JIT Opportunity |
|-------|----------|-----------------|
| `id` | ESSENTIAL | - |
| `config` | ESSENTIAL | - |
| `population` | ESSENTIAL | Core to evolution loop |
| `generation` | ESSENTIAL | Counter |
| `running` | ESSENTIAL | Status flag |
| `evaluating` | ESSENTIAL | Status flag |
| `games_completed` | DERIVED | Count from events |
| `total_games` | DERIVED | Compute from config |
| `best_fitness_ever` | DERIVED | Max from generation events |
| `last_gen_best` | DERIVED | From last generation event |
| `last_gen_avg` | DERIVED | From last generation event |
| `generation_history` | VISUALIZATION | Keep in Elixir projection |
| `breeding_events` | VISUALIZATION | Keep in Elixir projection |
| `species` | VISUALIZATION | For speciation UI |
| `pending_evaluations` | ESSENTIAL | Distributed eval tracking |

### task_silo State Fields

| Field | Category | JIT Opportunity |
|-------|----------|-----------------|
| `enabled_levels` | ESSENTIAL | Configuration |
| `stagnation_threshold` | ESSENTIAL | Configuration |
| `current_params` | DERIVED | Recompute on each call |
| `fitness_history` | ESSENTIAL | Needed for calculations |
| `improvement_history` | ESSENTIAL | Needed for calculations |
| `stagnation_counter` | DERIVED | Compute from history |
| `exploration_boost` | DERIVED | Compute on-demand |
| `exploitation_boost` | DERIVED | Compute on-demand |

### resource_silo State Fields

| Field | Category | JIT Opportunity |
|-------|----------|-----------------|
| `enabled_levels` | ESSENTIAL | Configuration |
| `current_metrics` | ESSENTIAL | Current system state |
| `pressure_history` | ESSENTIAL | Needed for trend detection |
| `current_concurrency` | DERIVED | Compute from metrics |
| `gc_triggered_count` | VISUALIZATION | Move to external metrics |
| `pause_count` | VISUALIZATION | Move to external metrics |

## Files to Modify

1. `src/neuroevolution_events.erl`
   - Add topic constructor functions
   - Add `subscribe_to/2` for specific topics
   - Update publish to use topic constructors

2. `src/neuroevolution_server.erl`
   - Enrich publish_* functions with full payloads
   - Use topic constructors instead of single topic
   - Add helper functions: `get_best_fitness/1`, `get_avg_fitness/1`, etc.

3. `src/neuroevolution_events_local.erl`
   - Support multiple pg groups per realm
   - Update join/leave to handle topic-specific groups

## Migration Steps

1. Add topic constructors (backward compatible)
2. Add rich payload builders as helper functions
3. Update publish calls to include rich payloads
4. Update publish calls to use topic constructors
5. Remove visualization-only fields from state (Phase 6)

## Expected Impact

| Metric | Before | After |
|--------|--------|-------|
| get_state() calls per generation | 5+ | 0 |
| Event payload size | ~100 bytes | ~2KB |
| gen_server call overhead | High | None |
| Visualization data availability | On-demand | Immediate |
