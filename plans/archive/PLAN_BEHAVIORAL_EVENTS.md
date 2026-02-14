# PLAN_BEHAVIORAL_EVENTS.md

**Status:** In Progress (Phase 1 Complete)
**Created:** 2025-12-23
**Last Updated:** 2025-12-24
**Related:** PLAN_AGE_LIFECYCLE.md, PLAN_KNOWLEDGE_TRANSFER.md, PLAN_LINEAGE_BRIDGE.md

---

## Overview

This document defines the complete behavioral event catalog for faber-neuroevolution and faber-tweann. Events follow **behavioral naming conventions** - they describe what happened in domain language, not CRUD operations on entities.

### Design Principles

1. **Behavioral, Not CRUD**: Events capture domain actions, not database operations
   - `offspring_born` NOT `individual_created`
   - `lineage_diverged` NOT `species_created`
   - `individual_culled` NOT `individual_deleted`

2. **Past Tense**: Events are facts that already happened
   - `mutation_applied` NOT `apply_mutation`
   - `knowledge_transferred` NOT `transfer_knowledge`

3. **Domain Language**: Events use evolutionary biology terminology
   - Follows ubiquitous language of neuroevolution domain
   - Immediately understandable to domain experts

4. **Producer Ownership**: The component emitting the event owns its schema
   - Consumers must accept the event structure
   - No negotiation on event content

---

## Event Categories

### 1. Birth Events

Events related to the creation of new individuals.

| Event | Description | Emitted By |
|-------|-------------|------------|
| `offspring_born` | New individual created via sexual reproduction (crossover) | population_server |
| `pioneer_spawned` | First individual in a new population/species | population_server |
| `clone_produced` | Asexual reproduction (copy with potential mutation) | population_server |
| `immigrant_arrived` | Individual transferred from another population/island | migration_server |
| `synthetic_injected` | Externally designed individual added to population | population_server |

**Event Payload: `offspring_born`**
```erlang
#{
    event_type => offspring_born,
    individual_id => binary(),
    parent_ids => [binary()],           % Usually 2 for sexual reproduction
    generation => non_neg_integer(),
    species_id => binary(),
    population_id => binary(),
    birth_timestamp => integer(),
    genome_hash => binary(),            % For deduplication
    initial_fitness => undefined | float()
}
```

---

### 2. Death Events

Events related to the removal of individuals from the population.

| Event | Description | Emitted By |
|-------|-------------|------------|
| `individual_culled` | Removed due to low fitness (selection pressure) | selection_server |
| `lifespan_expired` | Natural death from old age | lifecycle_server |
| `individual_perished` | Death during evaluation (e.g., simulation death) | evaluation_server |
| `individual_emigrated` | Left population via migration (alive elsewhere) | migration_server |
| `sacrifice_offered` | Voluntarily removed to benefit kin/group | social_server |

**Event Payload: `individual_culled`**
```erlang
#{
    event_type => individual_culled,
    individual_id => binary(),
    population_id => binary(),
    species_id => binary(),
    final_fitness => float(),
    age_generations => non_neg_integer(),
    cause => selection | stagnation | overcrowding,
    culled_at => integer(),
    lineage_depth => non_neg_integer()
}
```

**Event Payload: `lifespan_expired`**
```erlang
#{
    event_type => lifespan_expired,
    individual_id => binary(),
    population_id => binary(),
    age_generations => non_neg_integer(),
    lifecycle_stage => senescent,
    peak_fitness => float(),
    offspring_count => non_neg_integer(),
    expired_at => integer()
}
```

---

### 3. Aging & Lifecycle Events

Events related to individual development and aging.

| Event | Description | Emitted By |
|-------|-------------|------------|
| `individual_matured` | Transitioned from juvenile to fertile | lifecycle_server |
| `fertility_waned` | Transitioned from fertile to senescent | lifecycle_server |
| `development_milestone` | Reached significant age/capability threshold | lifecycle_server |
| `vigor_declined` | Performance degradation due to age | lifecycle_server |

**Event Payload: `individual_matured`**
```erlang
#{
    event_type => individual_matured,
    individual_id => binary(),
    population_id => binary(),
    previous_stage => juvenile,
    new_stage => fertile,
    age_generations => non_neg_integer(),
    fitness_at_maturity => float(),
    matured_at => integer()
}
```

---

### 4. Species & Lineage Events

Events related to speciation and lineage tracking.

| Event | Description | Emitted By |
|-------|-------------|------------|
| `lineage_diverged` | New species emerged from existing one | speciation_server |
| `species_emerged` | First individual of genuinely new species | speciation_server |
| `lineage_ended` | Species has no living members | speciation_server |
| `species_collapsed` | Rapid extinction of a species | speciation_server |
| `lineage_merged` | Two species merged (rare, convergent evolution) | speciation_server |
| `founder_effect_detected` | Population bottleneck created genetic drift | speciation_server |

**Event Payload: `lineage_diverged`**
```erlang
#{
    event_type => lineage_diverged,
    new_species_id => binary(),
    parent_species_id => binary(),
    founder_individual_id => binary(),
    divergence_cause => structural | behavioral | geographic,
    compatibility_distance => float(),
    generation => non_neg_integer(),
    diverged_at => integer()
}
```

---

### 5. Mutation Events

Events related to genetic modifications.

| Event | Description | Emitted By |
|-------|-------------|------------|
| `mutation_applied` | Any mutation occurred | genome_mutator |
| `neuron_added` | New neuron inserted into network | topological_mutations |
| `neuron_removed` | Neuron pruned from network | topological_mutations |
| `connection_added` | New synaptic connection created | topological_mutations |
| `connection_removed` | Synaptic connection pruned | topological_mutations |
| `weight_perturbed` | Synaptic weight modified | weight_mutations |
| `bias_perturbed` | Neuron bias modified | weight_mutations |
| `activation_changed` | Neuron activation function changed | topological_mutations |
| `plasticity_rule_changed` | Learning rule modified | plasticity_mutations |
| `modular_duplication` | Network module duplicated | topological_mutations |
| `modular_deletion` | Network module removed | topological_mutations |

**Event Payload: `neuron_added`**
```erlang
#{
    event_type => neuron_added,
    individual_id => binary(),
    neuron_id => binary(),
    layer => input | hidden | output,
    layer_index => float(),             % For ordering within layer
    activation_function => atom(),
    bias => float(),
    added_at => integer(),
    mutation_id => binary()             % Groups related mutations
}
```

---

### 6. Breeding & Reproduction Events

Events related to mating and genetic exchange.

| Event | Description | Emitted By |
|-------|-------------|------------|
| `mating_occurred` | Two individuals bred | breeding_server |
| `crossover_performed` | Genetic material exchanged | crossover_server |
| `mate_selected` | Partner chosen for reproduction | selection_server |
| `mate_rejected` | Potential partner rejected | selection_server |
| `courtship_succeeded` | Sexual selection criteria met | social_server |
| `courtship_failed` | Sexual selection criteria not met | social_server |
| `inbreeding_detected` | Mating between close relatives | breeding_server |
| `outbreeding_detected` | Mating between distant relatives | breeding_server |
| `stud_service_requested` | Champion DNA requested for breeding | marketplace_server |
| `stud_service_completed` | Champion DNA used in breeding | marketplace_server |

**Event Payload: `mating_occurred`**
```erlang
#{
    event_type => mating_occurred,
    parent_a_id => binary(),
    parent_b_id => binary(),
    offspring_ids => [binary()],
    crossover_method => atom(),
    compatibility_distance => float(),
    relatedness_coefficient => float(),  % 0.0 = unrelated, 0.5 = siblings
    mated_at => integer()
}
```

---

### 7. Fitness & Evaluation Events

Events related to performance assessment.

| Event | Description | Emitted By |
|-------|-------------|------------|
| `fitness_evaluated` | Individual's fitness calculated | evaluation_server |
| `fitness_improved` | Fitness increased from previous | evaluation_server |
| `fitness_declined` | Fitness decreased from previous | evaluation_server |
| `champion_crowned` | New best individual in population | population_server |
| `champion_dethroned` | Previous champion surpassed | population_server |
| `evaluation_timeout` | Evaluation exceeded time limit | evaluation_server |
| `evaluation_crashed` | Evaluation failed unexpectedly | evaluation_server |
| `novelty_discovered` | Behavior significantly different from archive | novelty_server |
| `stagnation_detected` | No fitness improvement for N generations | population_server |
| `breakthrough_achieved` | Significant fitness jump | population_server |

**Event Payload: `fitness_evaluated`**
```erlang
#{
    event_type => fitness_evaluated,
    individual_id => binary(),
    population_id => binary(),
    fitness => float(),
    fitness_components => #{
        primary => float(),
        novelty => float(),
        complexity_penalty => float()
    },
    evaluation_duration_ms => non_neg_integer(),
    generation => non_neg_integer(),
    evaluated_at => integer()
}
```

---

### 8. Migration Events

Events related to movement between populations/islands.

| Event | Description | Emitted By |
|-------|-------------|------------|
| `migration_initiated` | Individual began migration | migration_server |
| `migration_completed` | Individual arrived at destination | migration_server |
| `migration_failed` | Migration could not complete | migration_server |
| `island_connected` | Two populations established migration route | topology_server |
| `island_isolated` | Population cut off from migration | topology_server |

**Event Payload: `migration_completed`**
```erlang
#{
    event_type => migration_completed,
    individual_id => binary(),
    source_population_id => binary(),
    target_population_id => binary(),
    source_species_id => binary(),
    target_species_id => binary(),      % May differ if respeciated
    migration_reason => fitness | diversity | random,
    migrated_at => integer()
}
```

---

### 9. Knowledge Transfer Events

Events related to non-genetic information transfer.

| Event | Description | Emitted By |
|-------|-------------|------------|
| `knowledge_transferred` | Information passed between individuals | transfer_server |
| `skill_imitated` | Behavior copied from observation | imitation_server |
| `behavior_cloned` | Direct behavioral copying | transfer_server |
| `weights_grafted` | Network weights partially copied | transfer_server |
| `structure_seeded` | Network topology influenced by mentor | transfer_server |
| `mentor_assigned` | Teaching relationship established | social_server |
| `mentorship_concluded` | Teaching relationship ended | social_server |
| `teaching_occurred` | Active knowledge transfer attempt | transfer_server |
| `learning_succeeded` | Knowledge successfully acquired | transfer_server |
| `learning_failed` | Knowledge transfer unsuccessful | transfer_server |

**Event Payload: `knowledge_transferred`**
```erlang
#{
    event_type => knowledge_transferred,
    mentor_id => binary(),
    student_id => binary(),
    transfer_method => behavioral_cloning | weight_grafting | structural_seeding,
    knowledge_domain => atom(),         % e.g., navigation, combat, foraging
    transfer_fidelity => float(),       % 0.0 - 1.0
    student_fitness_before => float(),
    student_fitness_after => float(),
    transferred_at => integer()
}
```

---

### 10. Social Dynamics Events

Events related to social interactions and structures.

| Event | Description | Emitted By |
|-------|-------------|------------|
| `reputation_updated` | Individual's social standing changed | reputation_server |
| `cooperation_occurred` | Individuals cooperated on task | social_server |
| `defection_occurred` | Individual defected from cooperation | social_server |
| `coalition_formed` | Group alliance established | coalition_server |
| `coalition_dissolved` | Group alliance ended | coalition_server |
| `coalition_joined` | Individual joined existing coalition | coalition_server |
| `coalition_expelled` | Individual removed from coalition | coalition_server |
| `dominance_established` | Hierarchy position determined | hierarchy_server |
| `dominance_challenged` | Hierarchy position contested | hierarchy_server |
| `altruism_performed` | Self-sacrificing behavior for others | social_server |
| `reciprocity_tracked` | Tit-for-tat interaction recorded | social_server |
| `kin_recognized` | Genetic relatedness detected | kin_server |
| `kin_favored` | Preferential treatment to relative | kin_server |

**Event Payload: `coalition_formed`**
```erlang
#{
    event_type => coalition_formed,
    coalition_id => binary(),
    founder_ids => [binary()],
    population_id => binary(),
    formation_reason => defense | resource | breeding,
    initial_strength => float(),
    formed_at => integer()
}
```

**Event Payload: `reputation_updated`**
```erlang
#{
    event_type => reputation_updated,
    individual_id => binary(),
    population_id => binary(),
    previous_reputation => float(),
    new_reputation => float(),
    update_cause => cooperation | defection | achievement | norm_violation,
    observer_ids => [binary()],         % Who witnessed the change
    updated_at => integer()
}
```

---

### 11. Norm & Cultural Events

Events related to behavioral norms and cultural evolution.

| Event | Description | Emitted By |
|-------|-------------|------------|
| `norm_established` | New behavioral norm emerged | norm_server |
| `norm_violated` | Individual broke established norm | norm_server |
| `norm_enforced` | Punishment for norm violation | norm_server |
| `norm_decayed` | Norm weakened from lack of enforcement | norm_server |
| `tradition_established` | Long-lasting behavioral pattern emerged | culture_server |
| `tradition_abandoned` | Behavioral pattern discontinued | culture_server |
| `innovation_emerged` | Novel successful behavior discovered | innovation_server |
| `innovation_spread` | Successful behavior copied by others | innovation_server |
| `cultural_drift` | Gradual change in population behavior | culture_server |
| `fad_started` | Temporarily popular behavior | culture_server |
| `fad_ended` | Temporary behavior abandoned | culture_server |

**Event Payload: `norm_violated`**
```erlang
#{
    event_type => norm_violated,
    violator_id => binary(),
    norm_id => binary(),
    norm_type => cooperation | territory | mating | resource,
    violation_severity => float(),      % 0.0 - 1.0
    witnesses => [binary()],
    punishment_applied => none | reputation_loss | exclusion | retaliation,
    violated_at => integer()
}
```

**Event Payload: `innovation_emerged`**
```erlang
#{
    event_type => innovation_emerged,
    innovator_id => binary(),
    innovation_id => binary(),
    population_id => binary(),
    innovation_type => behavioral | structural | strategic,
    novelty_score => float(),
    fitness_advantage => float(),
    emerged_at => integer()
}
```

---

### 12. Epigenetic Events

Events related to environmental influence on gene expression.

| Event | Description | Emitted By |
|-------|-------------|------------|
| `mark_acquired` | Environmental influence recorded | epigenetics_server |
| `mark_inherited` | Epigenetic mark passed to offspring | epigenetics_server |
| `mark_decayed` | Epigenetic mark weakened over time | epigenetics_server |
| `mark_erased` | Epigenetic mark fully removed | epigenetics_server |
| `expression_modified` | Gene expression altered by marks | epigenetics_server |
| `stress_response_triggered` | Environmental stress caused marking | epigenetics_server |
| `adaptation_acquired` | Beneficial epigenetic adaptation | epigenetics_server |

**Event Payload: `mark_acquired`**
```erlang
#{
    event_type => mark_acquired,
    individual_id => binary(),
    mark_id => binary(),
    mark_type => stress | nutrition | social | environmental,
    target_genes => [binary()],         % Affected network components
    expression_modifier => float(),     % -1.0 to +1.0
    trigger_condition => term(),
    persistence => transient | stable | heritable,
    acquired_at => integer()
}
```

---

### 13. Ecological Events

Events related to environmental and population dynamics.

| Event | Description | Emitted By |
|-------|-------------|------------|
| `carrying_capacity_reached` | Population hit resource limit | ecology_server |
| `carrying_capacity_exceeded` | Population over resource limit | ecology_server |
| `resource_depleted` | Environmental resource exhausted | ecology_server |
| `resource_replenished` | Environmental resource restored | ecology_server |
| `disease_emerged` | New pathogen appeared | disease_server |
| `disease_spread` | Pathogen transmitted between individuals | disease_server |
| `immunity_developed` | Resistance to pathogen evolved | disease_server |
| `epidemic_started` | Widespread disease outbreak | disease_server |
| `epidemic_ended` | Disease outbreak concluded | disease_server |
| `environmental_cycle_shifted` | Seasonal/periodic change occurred | ecology_server |
| `catastrophe_occurred` | Major disruptive event | ecology_server |
| `recovery_began` | Population recovering from catastrophe | ecology_server |
| `niche_occupied` | Population filled ecological niche | ecology_server |
| `niche_vacated` | Ecological niche became available | ecology_server |
| `competition_intensified` | Resource competition increased | ecology_server |
| `predation_event` | Predator-prey interaction | ecology_server |
| `symbiosis_formed` | Mutually beneficial relationship | ecology_server |
| `parasitism_detected` | Exploitative relationship | ecology_server |

**Event Payload: `catastrophe_occurred`**
```erlang
#{
    event_type => catastrophe_occurred,
    catastrophe_id => binary(),
    catastrophe_type => mass_extinction | environmental_shift | resource_crash | epidemic,
    affected_populations => [binary()],
    severity => float(),                % 0.0 - 1.0
    mortality_rate => float(),          % Proportion killed
    survivors_count => non_neg_integer(),
    recovery_estimate_generations => non_neg_integer(),
    occurred_at => integer()
}
```

**Event Payload: `symbiosis_formed`**
```erlang
#{
    event_type => symbiosis_formed,
    symbiosis_id => binary(),
    partner_a_id => binary(),
    partner_b_id => binary(),
    symbiosis_type => mutualism | commensalism,
    benefit_a => float(),
    benefit_b => float(),
    formed_at => integer()
}
```

---

### 14. Population-Level Events

Events related to population dynamics and statistics.

| Event | Description | Emitted By |
|-------|-------------|------------|
| `generation_completed` | All evaluations for generation done | population_server |
| `population_initialized` | New population created | population_server |
| `population_terminated` | Population ended | population_server |
| `diversity_measured` | Population diversity calculated | diversity_server |
| `diversity_crisis` | Diversity dropped below threshold | diversity_server |
| `diversity_restored` | Diversity recovered | diversity_server |
| `speciation_rate_changed` | Rate of new species formation changed | speciation_server |
| `extinction_rate_changed` | Rate of species extinction changed | speciation_server |
| `population_statistics_updated` | Periodic stats snapshot | statistics_server |

**Event Payload: `generation_completed`**
```erlang
#{
    event_type => generation_completed,
    population_id => binary(),
    generation => non_neg_integer(),
    statistics => #{
        population_size => non_neg_integer(),
        species_count => non_neg_integer(),
        mean_fitness => float(),
        max_fitness => float(),
        min_fitness => float(),
        std_fitness => float(),
        mean_complexity => float(),
        diversity_index => float()
    },
    champion_id => binary(),
    duration_ms => non_neg_integer(),
    completed_at => integer()
}
```

---

### 15. Meta-Controller Events

Events related to the Liquid Conglomerate meta-controller.

| Event | Description | Emitted By |
|-------|-------------|------------|
| `controller_adjusted` | Meta-controller changed parameters | lc_controller |
| `silo_activated` | LC silo became active | lc_supervisor |
| `silo_deactivated` | LC silo became inactive | lc_supervisor |
| `parameter_tuned` | Evolution parameter adjusted | lc_controller |
| `strategy_switched` | Evolution strategy changed | lc_controller |
| `exploration_exploitation_balanced` | E/E tradeoff adjusted | lc_controller |

**Event Payload: `controller_adjusted`**
```erlang
#{
    event_type => controller_adjusted,
    controller_id => binary(),
    population_id => binary(),
    adjustments => #{
        mutation_rate => float(),
        crossover_rate => float(),
        selection_pressure => float(),
        migration_rate => float()
    },
    trigger => stagnation | diversity_crisis | breakthrough | scheduled,
    adjusted_at => integer()
}
```

---

## Silo-Specific Event Categories

The following sections define behavioral events for each of the 13 Liquid Conglomerate silos.

---

### 16. Temporal Silo Events (τ=10)

Events related to timing, episodes, and learning rate adaptation.

| Event | Description | Emitted By |
|-------|-------------|------------|
| `episode_started` | Evaluation episode began | temporal_silo |
| `episode_completed` | Evaluation episode finished | temporal_silo |
| `timing_adjusted` | Episode timing parameters changed | temporal_silo |
| `learning_rate_adapted` | Learning rate multiplier changed | temporal_silo |
| `patience_exhausted` | Waited too long without improvement | temporal_silo |
| `convergence_detected` | Fitness converging, can terminate early | temporal_silo |
| `timeout_threshold_reached` | Evaluation approaching time limit | temporal_silo |

**Event Payload: `episode_completed`**
```erlang
#{
    event_type => episode_completed,
    individual_id => binary(),
    episode_number => non_neg_integer(),
    duration_ms => non_neg_integer(),
    outcome => success | failure | timeout | early_termination,
    fitness_delta => float(),
    steps_taken => non_neg_integer(),
    completed_at => integer()
}
```

**Event Payload: `learning_rate_adapted`**
```erlang
#{
    event_type => learning_rate_adapted,
    population_id => binary(),
    previous_rate => float(),
    new_rate => float(),
    adaptation_reason => stagnation | breakthrough | scheduled | convergence,
    generation => non_neg_integer(),
    adapted_at => integer()
}
```

---

### 17. Economic Silo Events (τ=20)

Events related to compute budgets, resource economics, and wealth distribution.

| Event | Description | Emitted By |
|-------|-------------|------------|
| `budget_allocated` | Compute budget assigned to individual/group | economic_silo |
| `budget_exhausted` | Individual ran out of compute budget | economic_silo |
| `compute_traded` | Resources exchanged between individuals | economic_silo |
| `wealth_redistributed` | Population-wide wealth rebalancing | economic_silo |
| `bankruptcy_declared` | Individual has zero or negative budget | economic_silo |
| `investment_made` | Resources committed for future return | economic_silo |
| `dividend_distributed` | Returns from successful investment | economic_silo |
| `inflation_adjusted` | Economic parameters recalibrated | economic_silo |

**Event Payload: `budget_exhausted`**
```erlang
#{
    event_type => budget_exhausted,
    individual_id => binary(),
    population_id => binary(),
    initial_budget => float(),
    expenditure_history => [float()],
    exhaustion_cause => evaluation | mutation | reproduction,
    exhausted_at => integer()
}
```

**Event Payload: `wealth_redistributed`**
```erlang
#{
    event_type => wealth_redistributed,
    population_id => binary(),
    gini_before => float(),
    gini_after => float(),
    redistribution_amount => float(),
    beneficiaries_count => non_neg_integer(),
    redistribution_method => progressive_tax | universal_basic | merit_bonus,
    redistributed_at => integer()
}
```

---

### 18. Morphological Silo Events (τ=30)

Events related to network complexity, structure, and efficiency.

| Event | Description | Emitted By |
|-------|-------------|------------|
| `complexity_measured` | Network complexity metrics calculated | morphological_silo |
| `pruning_triggered` | Unused connections/neurons removed | morphological_silo |
| `growth_occurred` | Network expanded (neurons/connections added) | morphological_silo |
| `modularity_detected` | Modular structure identified in network | morphological_silo |
| `efficiency_improved` | Better fitness-to-complexity ratio | morphological_silo |
| `bloat_detected` | Excessive complexity without fitness gain | morphological_silo |
| `symmetry_broken` | Network symmetry disrupted | morphological_silo |
| `topology_milestone` | Significant structural change | morphological_silo |

**Event Payload: `pruning_triggered`**
```erlang
#{
    event_type => pruning_triggered,
    individual_id => binary(),
    neurons_removed => non_neg_integer(),
    connections_removed => non_neg_integer(),
    complexity_before => float(),
    complexity_after => float(),
    fitness_impact => float(),
    pruning_criterion => unused | weak | redundant,
    pruned_at => integer()
}
```

**Event Payload: `modularity_detected`**
```erlang
#{
    event_type => modularity_detected,
    individual_id => binary(),
    module_count => non_neg_integer(),
    modularity_score => float(),
    module_sizes => [non_neg_integer()],
    inter_module_connectivity => float(),
    detected_at => integer()
}
```

---

### 19. Competitive Silo Events (τ=15)

Events related to Elo ratings, opponent archives, and matchmaking.

| Event | Description | Emitted By |
|-------|-------------|------------|
| `opponent_archived` | Champion added to opponent archive | competitive_silo |
| `opponent_retired` | Champion removed from archive | competitive_silo |
| `match_completed` | Competition between individuals finished | competitive_silo |
| `elo_updated` | Elo rating changed after match | competitive_silo |
| `strategy_countered` | Counter-strategy emerged | competitive_silo |
| `arms_race_detected` | Escalating competitive dynamics | competitive_silo |
| `matchmaking_adjusted` | Opponent selection criteria changed | competitive_silo |
| `dominance_matrix_updated` | Win/loss statistics recalculated | competitive_silo |

**Event Payload: `match_completed`**
```erlang
#{
    event_type => match_completed,
    match_id => binary(),
    player_a_id => binary(),
    player_b_id => binary(),
    winner_id => binary() | draw,
    player_a_elo_before => float(),
    player_b_elo_before => float(),
    player_a_elo_after => float(),
    player_b_elo_after => float(),
    match_duration_ms => non_neg_integer(),
    completed_at => integer()
}
```

**Event Payload: `opponent_archived`**
```erlang
#{
    event_type => opponent_archived,
    opponent_id => binary(),
    archive_id => binary(),
    fitness_at_archive => float(),
    elo_at_archive => float(),
    generation => non_neg_integer(),
    archive_size_after => non_neg_integer(),
    archived_at => integer()
}
```

---

### 20. Social Silo Events (τ=25)

Events related to reputation, coalitions, and social networks.

| Event | Description | Emitted By |
|-------|-------------|------------|
| `reputation_changed` | Individual's reputation score updated | social_silo |
| `coalition_formed` | Group alliance established | social_silo |
| `coalition_dissolved` | Group alliance ended | social_silo |
| `coalition_member_joined` | Individual joined coalition | social_silo |
| `coalition_member_expelled` | Individual removed from coalition | social_silo |
| `mentoring_started` | Teaching relationship began | social_silo |
| `mentoring_ended` | Teaching relationship concluded | social_silo |
| `social_network_updated` | Social graph connections changed | social_silo |
| `trust_established` | Mutual trust formed between individuals | social_silo |
| `betrayal_detected` | Trust violation occurred | social_silo |

**Event Payload: `reputation_changed`**
```erlang
#{
    event_type => reputation_changed,
    individual_id => binary(),
    population_id => binary(),
    reputation_before => float(),
    reputation_after => float(),
    change_cause => cooperation | defection | achievement | punishment,
    witnesses => [binary()],
    changed_at => integer()
}
```

---

### 21. Cultural Silo Events (τ=35)

Events related to behavioral innovations, traditions, and meme spread.

| Event | Description | Emitted By |
|-------|-------------|------------|
| `innovation_discovered` | Novel successful behavior found | cultural_silo |
| `innovation_spread` | Behavior adopted by others | cultural_silo |
| `tradition_established` | Long-lasting behavior pattern emerged | cultural_silo |
| `tradition_abandoned` | Behavioral pattern discontinued | cultural_silo |
| `meme_created` | Transmissible behavioral unit emerged | cultural_silo |
| `meme_spread` | Behavioral unit copied between individuals | cultural_silo |
| `meme_mutated` | Behavioral unit changed during transmission | cultural_silo |
| `cultural_convergence` | Population behaviors becoming similar | cultural_silo |
| `cultural_divergence` | Population behaviors becoming different | cultural_silo |

**Event Payload: `innovation_discovered`**
```erlang
#{
    event_type => innovation_discovered,
    innovator_id => binary(),
    innovation_id => binary(),
    population_id => binary(),
    novelty_score => float(),
    fitness_advantage => float(),
    innovation_type => behavioral | structural | strategic,
    discovered_at => integer()
}
```

**Event Payload: `meme_spread`**
```erlang
#{
    event_type => meme_spread,
    meme_id => binary(),
    source_id => binary(),
    target_id => binary(),
    spread_fidelity => float(),
    fitness_correlation => float(),
    total_adopters => non_neg_integer(),
    spread_at => integer()
}
```

---

### 22. Developmental Silo Events (τ=40)

Events related to ontogeny, critical periods, and plasticity.

| Event | Description | Emitted By |
|-------|-------------|------------|
| `critical_period_opened` | Sensitive learning window began | developmental_silo |
| `critical_period_closed` | Sensitive learning window ended | developmental_silo |
| `plasticity_changed` | Network plasticity level adjusted | developmental_silo |
| `developmental_stage_reached` | Individual reached new life stage | developmental_silo |
| `metamorphosis_triggered` | Major developmental transformation | developmental_silo |
| `canalization_increased` | Development became more constrained | developmental_silo |
| `heterochrony_detected` | Developmental timing shift | developmental_silo |
| `developmental_milestone` | Significant capability acquired | developmental_silo |

**Event Payload: `critical_period_opened`**
```erlang
#{
    event_type => critical_period_opened,
    individual_id => binary(),
    period_type => sensory | motor | cognitive | social,
    plasticity_boost => float(),
    expected_duration_generations => non_neg_integer(),
    opened_at => integer()
}
```

**Event Payload: `metamorphosis_triggered`**
```erlang
#{
    event_type => metamorphosis_triggered,
    individual_id => binary(),
    stage_before => atom(),
    stage_after => atom(),
    structural_changes => #{
        neurons_added => non_neg_integer(),
        connections_rewired => non_neg_integer()
    },
    fitness_before => float(),
    triggered_at => integer()
}
```

---

### 23. Regulatory Silo Events (τ=45)

Events related to gene expression and module activation.

| Event | Description | Emitted By |
|-------|-------------|------------|
| `gene_expressed` | Gene/module activated | regulatory_silo |
| `gene_silenced` | Gene/module deactivated | regulatory_silo |
| `module_activated` | Functional module turned on | regulatory_silo |
| `module_deactivated` | Functional module turned off | regulatory_silo |
| `context_switched` | Environmental context changed expression | regulatory_silo |
| `regulatory_network_updated` | Gene regulation relationships changed | regulatory_silo |
| `dormancy_entered` | Capabilities went dormant | regulatory_silo |
| `dormancy_exited` | Dormant capabilities reactivated | regulatory_silo |

**Event Payload: `gene_expressed`**
```erlang
#{
    event_type => gene_expressed,
    individual_id => binary(),
    gene_id => binary(),
    expression_level => float(),
    trigger => environmental | developmental | conditional,
    context => term(),
    expressed_at => integer()
}
```

**Event Payload: `context_switched`**
```erlang
#{
    event_type => context_switched,
    individual_id => binary(),
    context_before => term(),
    context_after => term(),
    genes_affected => [binary()],
    modules_toggled => non_neg_integer(),
    switched_at => integer()
}
```

---

### 24. Ecological Silo Events (τ=50)

Events related to niches, environmental stress, and carrying capacity.

| Event | Description | Emitted By |
|-------|-------------|------------|
| `niche_occupied` | Individual/species filled ecological niche | ecological_silo |
| `niche_vacated` | Ecological niche became available | ecological_silo |
| `stress_applied` | Environmental stress increased | ecological_silo |
| `stress_relieved` | Environmental stress decreased | ecological_silo |
| `carrying_capacity_changed` | Population limit adjusted | ecological_silo |
| `resource_scarcity_detected` | Resources below threshold | ecological_silo |
| `resource_abundance_detected` | Resources above threshold | ecological_silo |
| `extinction_risk_elevated` | Species at risk of extinction | ecological_silo |
| `ecosystem_disrupted` | Major environmental change | ecological_silo |

**Event Payload: `niche_occupied`**
```erlang
#{
    event_type => niche_occupied,
    niche_id => binary(),
    occupant_id => binary(),
    population_id => binary(),
    niche_fitness_range => {float(), float()},
    competition_level => float(),
    occupied_at => integer()
}
```

**Event Payload: `stress_applied`**
```erlang
#{
    event_type => stress_applied,
    population_id => binary(),
    stress_type => resource | environmental | competitive | random,
    stress_intensity => float(),
    affected_individuals => non_neg_integer(),
    expected_duration => non_neg_integer() | indefinite,
    applied_at => integer()
}
```

---

### 25. Communication Silo Events (τ=55)

Events related to signaling, vocabulary, and coordination.

| Event | Description | Emitted By |
|-------|-------------|------------|
| `signal_emitted` | Individual sent a signal | communication_silo |
| `signal_received` | Individual received a signal | communication_silo |
| `signal_interpreted` | Signal meaning was decoded | communication_silo |
| `vocabulary_expanded` | New signal added to repertoire | communication_silo |
| `vocabulary_contracted` | Signal removed from repertoire | communication_silo |
| `dialect_formed` | Subgroup developed distinct signals | communication_silo |
| `dialect_merged` | Dialects converged | communication_silo |
| `coordination_achieved` | Successful multi-agent coordination | communication_silo |
| `deception_detected` | Dishonest signal identified | communication_silo |

**Event Payload: `signal_emitted`**
```erlang
#{
    event_type => signal_emitted,
    sender_id => binary(),
    signal_id => binary(),
    signal_content => term(),
    intended_receivers => [binary()] | broadcast,
    honesty => honest | deceptive,
    emitted_at => integer()
}
```

**Event Payload: `vocabulary_expanded`**
```erlang
#{
    event_type => vocabulary_expanded,
    population_id => binary(),
    new_signal_id => binary(),
    signal_meaning => term(),
    inventor_id => binary(),
    vocabulary_size_after => non_neg_integer(),
    expanded_at => integer()
}
```

---

### 26. Distribution Silo Events (τ=60)

Events related to distributed computing, islands, and load balancing.

| Event | Description | Emitted By |
|-------|-------------|------------|
| `island_created` | New distributed population node | distribution_silo |
| `island_destroyed` | Population node removed | distribution_silo |
| `migration_route_established` | Connection between islands created | distribution_silo |
| `migration_route_severed` | Connection between islands removed | distribution_silo |
| `load_rebalanced` | Compute load redistributed | distribution_silo |
| `topology_updated` | Island network structure changed | distribution_silo |
| `node_joined` | New compute node entered cluster | distribution_silo |
| `node_departed` | Compute node left cluster | distribution_silo |
| `synchronization_completed` | Islands synchronized state | distribution_silo |

**Event Payload: `load_rebalanced`**
```erlang
#{
    event_type => load_rebalanced,
    cluster_id => binary(),
    load_variance_before => float(),
    load_variance_after => float(),
    individuals_migrated => non_neg_integer(),
    rebalance_strategy => random | fitness_based | load_based,
    rebalanced_at => integer()
}
```

**Event Payload: `island_created`**
```erlang
#{
    event_type => island_created,
    island_id => binary(),
    cluster_id => binary(),
    initial_population_size => non_neg_integer(),
    compute_capacity => float(),
    connected_islands => [binary()],
    created_at => integer()
}
```

---

## Behaviour Definition

### neuroevolution_lineage_events Behaviour

Bridge libraries implement this behaviour to persist events to erl-esdb/erl-evoq.

```erlang
-module(neuroevolution_lineage_events).

%% Behaviour callbacks
-callback init(Config :: map()) -> {ok, State :: term()} | {error, Reason :: term()}.

-callback persist_event(Event :: map(), State :: term()) ->
    {ok, State :: term()} | {error, Reason :: term()}.

-callback persist_batch(Events :: [map()], State :: term()) ->
    {ok, State :: term()} | {error, Reason :: term()}.

-callback get_lineage(IndividualId :: binary(), State :: term()) ->
    {ok, [map()]} | {error, Reason :: term()}.

-callback get_species_history(SpeciesId :: binary(), State :: term()) ->
    {ok, [map()]} | {error, Reason :: term()}.

-callback get_population_timeline(PopulationId :: binary(), Opts :: map(), State :: term()) ->
    {ok, [map()]} | {error, Reason :: term()}.

-callback replay_from(Position :: non_neg_integer(), State :: term()) ->
    {ok, [map()]} | {error, Reason :: term()}.

%% Optional callbacks for advanced queries
-optional_callbacks([
    get_breeding_tree/2,
    get_mutation_history/2,
    get_fitness_trajectory/2,
    get_knowledge_transfers/2
]).

-callback get_breeding_tree(IndividualId :: binary(), State :: term()) ->
    {ok, map()} | {error, Reason :: term()}.

-callback get_mutation_history(IndividualId :: binary(), State :: term()) ->
    {ok, [map()]} | {error, Reason :: term()}.

-callback get_fitness_trajectory(IndividualId :: binary(), State :: term()) ->
    {ok, [map()]} | {error, Reason :: term()}.

-callback get_knowledge_transfers(IndividualId :: binary(), State :: term()) ->
    {ok, [map()]} | {error, Reason :: term()}.
```

---

## Event Stream Design

### Stream Naming Convention

Following erl-esdb conventions, streams are named by aggregate type and ID:

| Stream Pattern | Example | Purpose |
|----------------|---------|---------|
| `individual-{id}` | `individual-abc123` | Complete history of one individual |
| `species-{id}` | `species-def456` | Species lifecycle events |
| `population-{id}` | `population-ghi789` | Population-level events |
| `lineage-{id}` | `lineage-jkl012` | Multi-generational lineage |
| `coalition-{id}` | `coalition-mno345` | Coalition lifecycle |
| `$ce-neuroevolution` | `$ce-neuroevolution` | Category stream (all events) |

### Event Metadata

All events include standard metadata:

```erlang
#{
    event_id => binary(),               % Unique event identifier
    event_type => atom(),               % Event type (from catalog above)
    stream_id => binary(),              % Target stream
    correlation_id => binary(),         % Links related events
    causation_id => binary(),           % Event that caused this one
    timestamp => integer(),             % Microseconds since epoch
    version => non_neg_integer(),       % Schema version
    emitter => atom()                   % Module that emitted event
}
```

---

## Implementation Phases

### Completed
- [x] **Phase 1:** Core events (birth, death, mutation, fitness, generation) - 24 tests
- [x] **Phase 2:** Lineage events (species, migration, breeding) - included in Phase 1
- [x] **Phase 3:** Lifecycle events (aging, maturity, lifecycle stages) - included in Phase 1
- [x] **Phase 8:** Meta-controller events (LC silo adjustments) - included in Phase 1

### Remaining - Silo-Specific Events (NEW)

These phases add behavioral events for all 13 implemented Liquid Conglomerate silos:

- [ ] **Phase 4:** Social Silo events (reputation, coalitions, mentoring)
- [ ] **Phase 5:** Cultural Silo events (innovations, traditions, memes)
- [ ] **Phase 6:** Ecological Silo events (niches, stress, carrying capacity)
- [ ] **Phase 7:** Epigenetic/Regulatory Silo events (gene expression, marks)
- [ ] **Phase 9:** Temporal Silo events (episodes, timing, learning rates)
- [ ] **Phase 10:** Economic Silo events (budgets, trade, wealth)
- [ ] **Phase 11:** Competitive Silo events (Elo, matches, opponents)
- [ ] **Phase 12:** Morphological Silo events (complexity, pruning, growth)
- [ ] **Phase 13:** Developmental Silo events (critical periods, plasticity)
- [ ] **Phase 14:** Communication Silo events (signals, vocabulary, dialects)
- [ ] **Phase 15:** Distribution Silo events (islands, migration routes, load balancing)

---

## Success Criteria

### Event Catalog Completeness
- [x] Core events defined (birth, death, mutation, fitness, generation) - 24 events
- [ ] All 13 silo-specific event categories defined - ~85 additional events
- [ ] Total: ~110 events defined with complete payloads

### Implementation Completeness
- [x] Phase 1-3: Core event constructors implemented (24)
- [ ] Phase 4-15: Silo event constructors implemented (~85)
- [ ] All events have corresponding records in header file
- [ ] All events have unit tests

### Quality Requirements
- [x] Event naming follows behavioral conventions (no CRUD)
- [x] Behaviour module defined (`neuroevolution_lineage_events`)
- [ ] Stream design supports lineage queries
- [ ] Metadata supports causation tracking
- [ ] Schema versioning in place for evolution

### Event Count by Category

| Category | Event Count | Status |
|----------|-------------|--------|
| Birth | 5 | ✅ Done |
| Death | 5 | ✅ Done |
| Aging/Lifecycle | 4 | ✅ Done |
| Species/Lineage | 6 | ✅ Done |
| Mutation | 11 | ✅ Done |
| Breeding | 10 | ✅ Done |
| Fitness/Evaluation | 10 | ✅ Done |
| Migration | 5 | ✅ Done |
| Knowledge Transfer | 10 | Pending |
| Social (Silo 20) | 10 | Pending |
| Cultural (Silo 21) | 9 | Pending |
| Ecological (Silo 24) | 9 | Pending |
| Epigenetic | 7 | Pending |
| Population-Level | 9 | Pending |
| Meta-Controller | 6 | ✅ Done |
| **Silo-Specific (16-26)** | | |
| Temporal (Silo 16) | 7 | Pending |
| Economic (Silo 17) | 8 | Pending |
| Morphological (Silo 18) | 8 | Pending |
| Competitive (Silo 19) | 8 | Pending |
| Developmental (Silo 22) | 8 | Pending |
| Regulatory (Silo 23) | 8 | Pending |
| Communication (Silo 25) | 9 | Pending |
| Distribution (Silo 26) | 9 | Pending |
| **TOTAL** | ~150 | ~30% Done |

---

## References

- `neuroevolution_events.erl` - Existing pubsub event system
- `EVENT_DRIVEN_ARCHITECTURE.md` - Current event architecture
- PLAN_LINEAGE_BRIDGE.md - Bridge library implementation
- erl-esdb documentation - Event store patterns
