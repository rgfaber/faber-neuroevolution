# Behavioral Events Guide

This guide explains how to use the behavioral event system in faber-neuroevolution. Events capture domain-specific actions using evolutionary biology terminology, enabling event sourcing, auditing, and inter-component communication.

## Design Principles

### Behavioral, Not CRUD

Events describe what happened in domain language:

| Good (Behavioral) | Bad (CRUD) |
|-------------------|------------|
| `offspring_born` | `individual_created` |
| `individual_culled` | `individual_deleted` |
| `lineage_diverged` | `species_created` |
| `knowledge_transferred` | `knowledge_updated` |

### Past Tense

Events are immutable facts that already happened:

| Good | Bad |
|------|-----|
| `mutation_applied` | `apply_mutation` |
| `episode_completed` | `complete_episode` |

## Quick Start

### Emitting Events

```erlang
%% Include the event definitions
-include_lib("faber_neuroevolution/include/lc_temporal_events.hrl").

%% Emit an episode completion event
lc_temporal_events:emit_episode_completed(
    IndividualId,
    EpisodeNumber,
    DurationMs,
    success,      % outcome
    0.15,         % fitness_delta
    1000          % steps_taken
).
```

### Subscribing to Events

```erlang
%% Subscribe to temporal events
neuroevolution_behavioral_events:subscribe(temporal, self()),

%% Handle events in your process
handle_info({lc_event, temporal, #episode_completed{} = Event}, State) ->
    Duration = Event#episode_completed.duration_ms,
    %% Process the event...
    {noreply, State}.
```

### Querying Event History

```erlang
%% Get recent events for a silo
Events = neuroevolution_behavioral_events:get_events(temporal, #{
    since => erlang:system_time(millisecond) - 60000,  % Last minute
    limit => 100
}).
```

## Event Categories by Silo

### Temporal Silo (9 events)

Episode and timing management.

| Event | Description |
|-------|-------------|
| `episode_started` | New evaluation episode begins |
| `episode_completed` | Episode finished with outcome |
| `timing_adjusted` | Evaluation timeout changed |
| `learning_rate_adapted` | Learning rate modified |
| `patience_exhausted` | Stagnation limit reached |
| `early_termination_triggered` | Episode stopped early |
| `convergence_detected` | Fitness improvement slowed |
| `checkpoint_reached` | Training milestone achieved |
| `cohort_completed` | Population cohort finished |

```erlang
%% Example: Episode completion
#episode_completed{
    meta = #lc_event_meta{...},
    individual_id = <<"ind_123">>,
    episode_number = 42,
    duration_ms = 1500,
    outcome = success,           % success | failure | timeout | early_termination
    fitness_delta = 0.05,
    steps_taken = 1000
}
```

### Economic Silo (10 events)

Resource allocation and budgeting.

| Event | Description |
|-------|-------------|
| `budget_allocated` | Compute budget assigned |
| `budget_exhausted` | Budget fully consumed |
| `energy_consumed` | Energy used for operation |
| `energy_replenished` | Energy restored |
| `trade_executed` | Resource exchange completed |
| `wealth_redistributed` | Resources rebalanced |
| `bankruptcy_declared` | Individual out of resources |
| `investment_made` | Resources allocated for future |
| `efficiency_rewarded` | Bonus for efficient use |
| `waste_penalized` | Penalty for inefficiency |

### Morphological Silo (9 events)

Network structure changes.

| Event | Description |
|-------|-------------|
| `network_grown` | Neurons/connections added |
| `network_pruned` | Unused structures removed |
| `complexity_increased` | Network became more complex |
| `complexity_reduced` | Network simplified |
| `efficiency_improved` | Better performance/size ratio |
| `modularity_detected` | Functional modules found |
| `symmetry_broken` | Asymmetric structure emerged |
| `bottleneck_identified` | Structural limitation found |
| `capacity_reached` | Maximum size hit |

### Competitive Silo (10 events)

Adversarial dynamics.

| Event | Description |
|-------|-------------|
| `match_played` | Competition completed |
| `elo_updated` | Rating changed |
| `champion_crowned` | New best individual |
| `champion_dethroned` | Previous champion beaten |
| `archive_updated` | Opponent archive changed |
| `strategy_emerged` | New behavior pattern |
| `counter_strategy_found` | Exploit discovered |
| `arms_race_detected` | Escalating competition |
| `diversity_collapsed` | Strategies converging |
| `equilibrium_reached` | Stable competition state |

### Social Silo (9 events)

Cooperation and reputation.

| Event | Description |
|-------|-------------|
| `reputation_gained` | Standing improved |
| `reputation_lost` | Standing decreased |
| `coalition_formed` | Group created |
| `coalition_dissolved` | Group disbanded |
| `cooperation_offered` | Help extended |
| `cooperation_accepted` | Help received |
| `defection_detected` | Trust broken |
| `mentoring_provided` | Knowledge shared |
| `social_rank_changed` | Hierarchy position moved |

### Cultural Silo (9 events)

Innovation and tradition.

| Event | Description |
|-------|-------------|
| `innovation_discovered` | New behavior found |
| `innovation_spread` | Behavior adopted by others |
| `tradition_established` | Pattern became standard |
| `tradition_abandoned` | Pattern fell out of use |
| `imitation_attempted` | Copy of behavior tried |
| `imitation_succeeded` | Copy was successful |
| `meme_created` | Transmissible unit formed |
| `cultural_drift_detected` | Gradual change observed |
| `conformity_pressure_applied` | Norm enforcement |

### Developmental Silo (9 events)

Ontogeny and plasticity.

| Event | Description |
|-------|-------------|
| `development_started` | Growth phase begun |
| `milestone_reached` | Development checkpoint |
| `critical_period_opened` | Learning window active |
| `critical_period_closed` | Learning window ended |
| `plasticity_changed` | Adaptability modified |
| `maturation_advanced` | Development progressed |
| `metamorphosis_triggered` | Major restructuring |
| `canalization_detected` | Development constrained |
| `heterochrony_observed` | Timing variation |

### Regulatory Silo (9 events)

Gene expression control.

| Event | Description |
|-------|-------------|
| `gene_activated` | Gene turned on |
| `gene_silenced` | Gene turned off |
| `module_enabled` | Functional unit active |
| `module_disabled` | Functional unit inactive |
| `context_switched` | Environmental response |
| `expression_pattern_changed` | Regulation modified |
| `epigenetic_mark_set` | Heritable marker added |
| `regulatory_cascade_triggered` | Chain reaction started |
| `dormant_capability_awakened` | Hidden feature activated |

### Ecological Silo (9 events)

Environmental dynamics.

| Event | Description |
|-------|-------------|
| `niche_occupied` | Ecological role filled |
| `niche_vacated` | Role abandoned |
| `resource_discovered` | New resource found |
| `resource_depleted` | Resource exhausted |
| `stress_applied` | Environmental pressure |
| `adaptation_successful` | Stress response worked |
| `carrying_capacity_reached` | Population limit hit |
| `extinction_risk_elevated` | Danger level increased |
| `ecosystem_stabilized` | Balance achieved |

### Communication Silo (6 events)

Signaling and coordination.

| Event | Description |
|-------|-------------|
| `signal_sent` | Message transmitted |
| `signal_received` | Message received |
| `vocabulary_expanded` | New signal learned |
| `coordination_achieved` | Group synchronization |
| `deception_attempted` | False signal sent |
| `deception_detected` | False signal identified |

### Distribution Silo (4 events)

Population structure.

| Event | Description |
|-------|-------------|
| `migration_completed` | Individual moved islands |
| `island_formed` | New subpopulation created |
| `island_merged` | Subpopulations combined |
| `load_rebalanced` | Work redistributed |

## Event Metadata

All events include standard metadata:

```erlang
-record(lc_event_meta, {
    event_id :: binary(),           % Unique event identifier
    timestamp :: integer(),         % Unix milliseconds
    realm :: binary(),              % Multi-tenancy realm
    population_id :: binary(),      % Population context
    generation :: non_neg_integer(),% Generation number
    source_silo :: atom()           % Emitting silo type
}).
```

## Event Storage

Events can be stored for replay and analysis:

```erlang
%% Configure event storage
neuroevolution_behavioral_events:configure(#{
    storage => ets,              % ets | dets | custom
    max_events => 100000,        % Ring buffer size
    persistence => false         % Write to disk
}).
```

## Integration with Silos

Each silo's event module provides typed emission functions:

```erlang
%% Temporal silo
lc_temporal_events:emit_episode_started(IndId, EpisodeNum, ExpectedDuration, Gen).
lc_temporal_events:emit_learning_rate_adapted(PopId, OldRate, NewRate, Reason, Gen).

%% Economic silo
lc_economic_events:emit_budget_allocated(IndId, Amount, Source, Gen).
lc_economic_events:emit_trade_executed(FromId, ToId, ResourceType, Amount, Gen).

%% Competitive silo
lc_competitive_events:emit_match_played(Player1, Player2, Result, EloDelta, Gen).
lc_competitive_events:emit_champion_crowned(IndId, Fitness, PreviousChampion, Gen).
```

## Best Practices

1. **Subscribe Early**: Set up subscriptions before evolution starts
2. **Handle Async**: Events may arrive out of order
3. **Batch Processing**: Process events in batches for efficiency
4. **Selective Subscription**: Only subscribe to events you need
5. **Idempotent Handlers**: Design handlers to handle duplicates

## See Also

- [Liquid Conglomerate Overview](liquid-conglomerate.md)
- [Cross-Silo Communication](silos/lc-overview.md)
