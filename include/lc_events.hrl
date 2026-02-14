%% @doc LC Silo Event Type Definitions.
%%
%% Defines all event types emitted by Liquid Conglomerate silos.
%% Events are persisted to erl-esdb when faber_neuroevolution_esdb is available.
%%
%% == Stream Routing ==
%%
%% Events are routed to streams based on silo type:
%%   lc-{realm}.task       - Task silo events
%%   lc-{realm}.resource   - Resource silo events
%%   lc-{realm}.temporal   - Temporal silo events
%%   ... etc
%%
%% == Event Structure ==
%%
%% All events have the following base structure:
%%   #{
%%       event_type => atom(),           % Event type (see below)
%%       silo => atom(),                 % Silo type
%%       realm => binary(),              % Realm identifier
%%       timestamp => integer(),         % Milliseconds since epoch
%%       payload => map()                % Event-specific data
%%   }
%%
%% @author R.G. Lefever
%% @copyright 2024-2026 R.G. Lefever

-ifndef(LC_EVENTS_HRL).
-define(LC_EVENTS_HRL, true).

%%% ============================================================================
%%% Task Silo Events
%%% ============================================================================

%% Hyperparameters were adjusted by the L0 TWEANN
-define(EVENT_HYPERPARAMS_ADJUSTED, hyperparams_adjusted).

%% Evolution stagnation was detected
-define(EVENT_STAGNATION_DETECTED, stagnation_detected).

%% L1 intervention was triggered due to severity threshold
-define(EVENT_INTERVENTION_TRIGGERED, intervention_triggered).

%% Exploration boost was applied
-define(EVENT_EXPLORATION_BOOSTED, exploration_boosted).

%% Exploitation mode was entered
-define(EVENT_EXPLOITATION_ENTERED, exploitation_entered).

%%% ============================================================================
%%% Resource Silo Events
%%% ============================================================================

%% Garbage collection was triggered
-define(EVENT_GC_TRIGGERED, gc_triggered).

%% Evolution was throttled due to resource pressure
-define(EVENT_THROTTLE_APPLIED, throttle_applied).

%% Resource pressure alert (warning/intervention/critical)
-define(EVENT_PRESSURE_ALERT, pressure_alert).

%% Concurrency level was adjusted
-define(EVENT_CONCURRENCY_ADJUSTED, concurrency_adjusted).

%%% ============================================================================
%%% Temporal Silo Events
%%% ============================================================================

%% Episode timed out
-define(EVENT_EPISODE_TIMEOUT, episode_timeout).

%% Episode was terminated early
-define(EVENT_EARLY_TERMINATION, early_termination).

%% Patience counter was exhausted
-define(EVENT_PATIENCE_EXHAUSTED, patience_exhausted).

%% Learning rate was adjusted
-define(EVENT_LEARNING_RATE_ADJUSTED, learning_rate_adjusted).

%% Episode length target was changed
-define(EVENT_EPISODE_LENGTH_CHANGED, episode_length_changed).

%%% ============================================================================
%%% Competitive Silo Events
%%% ============================================================================

%% Opponent was added to archive
-define(EVENT_OPPONENT_ARCHIVED, opponent_archived).

%% Opponent was removed from archive
-define(EVENT_OPPONENT_REMOVED, opponent_removed).

%% Elo rating was updated
-define(EVENT_ELO_UPDATED, elo_updated).

%% Arms race was detected
-define(EVENT_ARMS_RACE_DETECTED, arms_race_detected).

%% Match was completed
-define(EVENT_MATCH_COMPLETED, match_completed).

%%% ============================================================================
%%% Economic Silo Events
%%% ============================================================================

%% Budget was allocated to individual
-define(EVENT_BUDGET_ALLOCATED, budget_allocated).

%% Bankruptcy occurred
-define(EVENT_BANKRUPTCY, bankruptcy).

%% Wealth redistribution occurred
-define(EVENT_WEALTH_REDISTRIBUTED, wealth_redistributed).

%% Gini coefficient changed significantly
-define(EVENT_GINI_CHANGED, gini_changed).

%% Trade completed
-define(EVENT_TRADE_COMPLETED, trade_completed).

%%% ============================================================================
%%% Social Silo Events
%%% ============================================================================

%% Coalition was formed
-define(EVENT_COALITION_FORMED, coalition_formed).

%% Coalition was dissolved
-define(EVENT_COALITION_DISSOLVED, coalition_dissolved).

%% Reputation changed significantly
-define(EVENT_REPUTATION_CHANGED, reputation_changed).

%% Defection was detected
-define(EVENT_DEFECTION_DETECTED, defection_detected).

%% Mentorship was established
-define(EVENT_MENTORSHIP_ESTABLISHED, mentorship_established).

%%% ============================================================================
%%% Cultural Silo Events
%%% ============================================================================

%% Innovation was discovered
-define(EVENT_INNOVATION_DISCOVERED, innovation_discovered).

%% Tradition was established
-define(EVENT_TRADITION_ESTABLISHED, tradition_established).

%% Meme spread occurred
-define(EVENT_MEME_SPREAD, meme_spread).

%% Imitation occurred
-define(EVENT_IMITATION_OCCURRED, imitation_occurred).

%% Tradition decayed
-define(EVENT_TRADITION_DECAYED, tradition_decayed).

%%% ============================================================================
%%% Ecological Silo Events
%%% ============================================================================

%% Niche was formed
-define(EVENT_NICHE_FORMED, niche_formed).

%% Extinction occurred
-define(EVENT_EXTINCTION, extinction).

%% Environmental stress was applied
-define(EVENT_STRESS_APPLIED, stress_applied).

%% Carrying capacity was reached
-define(EVENT_CAPACITY_REACHED, capacity_reached).

%% Resource pool regenerated
-define(EVENT_RESOURCE_REGENERATED, resource_regenerated).

%%% ============================================================================
%%% Developmental Silo Events
%%% ============================================================================

%% Critical period was opened
-define(EVENT_CRITICAL_PERIOD_OPENED, critical_period_opened).

%% Critical period was closed
-define(EVENT_CRITICAL_PERIOD_CLOSED, critical_period_closed).

%% Metamorphosis occurred
-define(EVENT_METAMORPHOSIS, metamorphosis).

%% Maturity was reached
-define(EVENT_MATURITY_REACHED, maturity_reached).

%% Plasticity decayed
-define(EVENT_PLASTICITY_DECAYED, plasticity_decayed).

%%% ============================================================================
%%% Regulatory Silo Events
%%% ============================================================================

%% Gene was expressed
-define(EVENT_GENE_EXPRESSED, gene_expressed).

%% Gene was silenced
-define(EVENT_GENE_SILENCED, gene_silenced).

%% Module was activated
-define(EVENT_MODULE_ACTIVATED, module_activated).

%% Context switch occurred
-define(EVENT_CONTEXT_SWITCHED, context_switched).

%% Epigenetic mark was acquired
-define(EVENT_EPIGENETIC_MARK_ACQUIRED, epigenetic_mark_acquired).

%%% ============================================================================
%%% Morphological Silo Events
%%% ============================================================================

%% Network was pruned
-define(EVENT_NETWORK_PRUNED, network_pruned).

%% Complexity threshold was exceeded
-define(EVENT_COMPLEXITY_EXCEEDED, complexity_exceeded).

%% Efficiency improved
-define(EVENT_EFFICIENCY_IMPROVED, efficiency_improved).

%% Neuron was added
-define(EVENT_NEURON_ADDED, neuron_added).

%% Connection was added
-define(EVENT_CONNECTION_ADDED, connection_added).

%%% ============================================================================
%%% Communication Silo Events
%%% ============================================================================

%% Signal was created
-define(EVENT_SIGNAL_CREATED, signal_created).

%% Deception was detected
-define(EVENT_DECEPTION_DETECTED, deception_detected).

%% Coordination succeeded
-define(EVENT_COORDINATION_SUCCESS, coordination_success).

%% Vocabulary grew
-define(EVENT_VOCABULARY_GREW, vocabulary_grew).

%% Dialect formed
-define(EVENT_DIALECT_FORMED, dialect_formed).

%%% ============================================================================
%%% Distribution Silo Events
%%% ============================================================================

%% Migration occurred
-define(EVENT_MIGRATION_OCCURRED, migration_occurred).

%% Island was formed
-define(EVENT_ISLAND_FORMED, island_formed).

%% Load was balanced
-define(EVENT_LOAD_BALANCED, load_balanced).

%% Remote capacity changed
-define(EVENT_REMOTE_CAPACITY_CHANGED, remote_capacity_changed).

%% Network topology changed
-define(EVENT_TOPOLOGY_CHANGED, topology_changed).

%%% ============================================================================
%%% Cross-Silo Events
%%% ============================================================================

%% Signal was emitted to another silo
-define(EVENT_CROSS_SILO_SIGNAL, cross_silo_signal).

%% Global health changed
-define(EVENT_GLOBAL_HEALTH_CHANGED, global_health_changed).

%% Cooperative reward was computed
-define(EVENT_COOPERATIVE_REWARD, cooperative_reward).

-endif. %% LC_EVENTS_HRL
