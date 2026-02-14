%%%-------------------------------------------------------------------
%%% @doc Behavioral event definitions for Distribution Silo (τ=60).
%%%
%%% Events related to distributed computing, islands, and load balancing.
%%%
%%% @end
%%%-------------------------------------------------------------------
-ifndef(LC_DISTRIBUTION_EVENTS_HRL).
-define(LC_DISTRIBUTION_EVENTS_HRL, true).

-include("lc_events_common.hrl").

%%% ============================================================================
%%% Island Events
%%% ============================================================================

-record(island_created, {
    meta :: #lc_event_meta{},
    island_id :: binary(),
    cluster_id :: binary(),
    initial_population_size :: non_neg_integer(),
    compute_capacity :: float(),
    connected_islands :: [binary()]
}).

-record(island_destroyed, {
    meta :: #lc_event_meta{},
    island_id :: binary(),
    cluster_id :: binary(),
    destruction_reason :: consolidation | failure | migration,
    final_population :: non_neg_integer(),
    individuals_migrated :: non_neg_integer()
}).

%%% ============================================================================
%%% Migration Route Events
%%% ============================================================================

-record(migration_route_established, {
    meta :: #lc_event_meta{},
    route_id :: binary(),
    source_island_id :: binary(),
    target_island_id :: binary(),
    bandwidth :: float(),
    latency_ms :: non_neg_integer()
}).

-record(migration_route_severed, {
    meta :: #lc_event_meta{},
    route_id :: binary(),
    source_island_id :: binary(),
    target_island_id :: binary(),
    severance_reason :: failure | policy | consolidation
}).

%%% ============================================================================
%%% Load Balancing Events
%%% ============================================================================

-record(load_rebalanced, {
    meta :: #lc_event_meta{},
    cluster_id :: binary(),
    load_variance_before :: float(),
    load_variance_after :: float(),
    individuals_migrated :: non_neg_integer(),
    rebalance_strategy :: random | fitness_based | load_based
}).

-record(topology_updated, {
    meta :: #lc_event_meta{},
    cluster_id :: binary(),
    topology_type :: ring | mesh | star | hierarchical,
    island_count :: non_neg_integer(),
    route_count :: non_neg_integer(),
    diameter :: non_neg_integer()
}).

%%% ============================================================================
%%% Node Events
%%% ============================================================================

-record(node_joined, {
    meta :: #lc_event_meta{},
    node_id :: binary(),
    cluster_id :: binary(),
    compute_capacity :: float(),
    assigned_islands :: [binary()]
}).

-record(node_departed, {
    meta :: #lc_event_meta{},
    node_id :: binary(),
    cluster_id :: binary(),
    departure_reason :: graceful | failure | eviction,
    affected_islands :: [binary()],
    individuals_orphaned :: non_neg_integer()
}).

-record(synchronization_completed, {
    meta :: #lc_event_meta{},
    cluster_id :: binary(),
    islands_synchronized :: [binary()],
    sync_duration_ms :: non_neg_integer(),
    conflicts_resolved :: non_neg_integer()
}).

%%% ============================================================================
%%% Type Exports
%%% ============================================================================

-type distribution_event() :: #island_created{} | #island_destroyed{} |
                              #migration_route_established{} | #migration_route_severed{} |
                              #load_rebalanced{} | #topology_updated{} |
                              #node_joined{} | #node_departed{} |
                              #synchronization_completed{}.

-endif. %% LC_DISTRIBUTION_EVENTS_HRL
