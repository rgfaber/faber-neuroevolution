%%%-------------------------------------------------------------------
%%% @doc Behavioral event constructors for Distribution Silo.
%%%
%%% Events related to distributed computing, islands, and load balancing.
%%% Distribution Silo operates at timescale τ=60.
%%%
%%% @end
%%%-------------------------------------------------------------------
-module(lc_distribution_events).

-include("lc_distribution_events.hrl").

%% Event constructors
-export([
    island_created/1,
    island_destroyed/1,
    migration_route_established/1,
    migration_route_severed/1,
    load_rebalanced/1,
    topology_updated/1,
    node_joined/1,
    node_departed/1,
    synchronization_completed/1
]).

%% Utility exports
-export([
    make_meta/1,
    make_meta/2,
    event_to_map/1
]).

%%% ============================================================================
%%% Metadata Construction
%%% ============================================================================

-spec make_meta(Emitter :: atom()) -> #lc_event_meta{}.
make_meta(Emitter) ->
    make_meta(Emitter, #{}).

-spec make_meta(Emitter :: atom(), Opts :: map()) -> #lc_event_meta{}.
make_meta(Emitter, Opts) ->
    #lc_event_meta{
        event_id = maps:get(event_id, Opts, generate_id()),
        correlation_id = maps:get(correlation_id, Opts, undefined),
        causation_id = maps:get(causation_id, Opts, undefined),
        timestamp = maps:get(timestamp, Opts, erlang:system_time(microsecond)),
        version = maps:get(version, Opts, ?LC_EVENT_VERSION),
        emitter = Emitter
    }.

%%% ============================================================================
%%% Island Event Constructors
%%% ============================================================================

-spec island_created(map()) -> #island_created{}.
island_created(Data) ->
    #island_created{
        meta = make_meta(maps:get(emitter, Data, distribution_silo), Data),
        island_id = maps:get(island_id, Data, generate_id()),
        cluster_id = maps:get(cluster_id, Data),
        initial_population_size = maps:get(initial_population_size, Data),
        compute_capacity = maps:get(compute_capacity, Data),
        connected_islands = maps:get(connected_islands, Data, [])
    }.

-spec island_destroyed(map()) -> #island_destroyed{}.
island_destroyed(Data) ->
    #island_destroyed{
        meta = make_meta(maps:get(emitter, Data, distribution_silo), Data),
        island_id = maps:get(island_id, Data),
        cluster_id = maps:get(cluster_id, Data),
        destruction_reason = maps:get(destruction_reason, Data, consolidation),
        final_population = maps:get(final_population, Data, 0),
        individuals_migrated = maps:get(individuals_migrated, Data, 0)
    }.

%%% ============================================================================
%%% Migration Route Event Constructors
%%% ============================================================================

-spec migration_route_established(map()) -> #migration_route_established{}.
migration_route_established(Data) ->
    #migration_route_established{
        meta = make_meta(maps:get(emitter, Data, distribution_silo), Data),
        route_id = maps:get(route_id, Data, generate_id()),
        source_island_id = maps:get(source_island_id, Data),
        target_island_id = maps:get(target_island_id, Data),
        bandwidth = maps:get(bandwidth, Data),
        latency_ms = maps:get(latency_ms, Data, 0)
    }.

-spec migration_route_severed(map()) -> #migration_route_severed{}.
migration_route_severed(Data) ->
    #migration_route_severed{
        meta = make_meta(maps:get(emitter, Data, distribution_silo), Data),
        route_id = maps:get(route_id, Data),
        source_island_id = maps:get(source_island_id, Data),
        target_island_id = maps:get(target_island_id, Data),
        severance_reason = maps:get(severance_reason, Data, failure)
    }.

%%% ============================================================================
%%% Load Balancing Event Constructors
%%% ============================================================================

-spec load_rebalanced(map()) -> #load_rebalanced{}.
load_rebalanced(Data) ->
    #load_rebalanced{
        meta = make_meta(maps:get(emitter, Data, distribution_silo), Data),
        cluster_id = maps:get(cluster_id, Data),
        load_variance_before = maps:get(load_variance_before, Data),
        load_variance_after = maps:get(load_variance_after, Data),
        individuals_migrated = maps:get(individuals_migrated, Data),
        rebalance_strategy = maps:get(rebalance_strategy, Data, random)
    }.

-spec topology_updated(map()) -> #topology_updated{}.
topology_updated(Data) ->
    #topology_updated{
        meta = make_meta(maps:get(emitter, Data, distribution_silo), Data),
        cluster_id = maps:get(cluster_id, Data),
        topology_type = maps:get(topology_type, Data, mesh),
        island_count = maps:get(island_count, Data),
        route_count = maps:get(route_count, Data),
        diameter = maps:get(diameter, Data, 0)
    }.

%%% ============================================================================
%%% Node Event Constructors
%%% ============================================================================

-spec node_joined(map()) -> #node_joined{}.
node_joined(Data) ->
    #node_joined{
        meta = make_meta(maps:get(emitter, Data, distribution_silo), Data),
        node_id = maps:get(node_id, Data),
        cluster_id = maps:get(cluster_id, Data),
        compute_capacity = maps:get(compute_capacity, Data),
        assigned_islands = maps:get(assigned_islands, Data, [])
    }.

-spec node_departed(map()) -> #node_departed{}.
node_departed(Data) ->
    #node_departed{
        meta = make_meta(maps:get(emitter, Data, distribution_silo), Data),
        node_id = maps:get(node_id, Data),
        cluster_id = maps:get(cluster_id, Data),
        departure_reason = maps:get(departure_reason, Data, graceful),
        affected_islands = maps:get(affected_islands, Data, []),
        individuals_orphaned = maps:get(individuals_orphaned, Data, 0)
    }.

-spec synchronization_completed(map()) -> #synchronization_completed{}.
synchronization_completed(Data) ->
    #synchronization_completed{
        meta = make_meta(maps:get(emitter, Data, distribution_silo), Data),
        cluster_id = maps:get(cluster_id, Data),
        islands_synchronized = maps:get(islands_synchronized, Data, []),
        sync_duration_ms = maps:get(sync_duration_ms, Data),
        conflicts_resolved = maps:get(conflicts_resolved, Data, 0)
    }.

%%% ============================================================================
%%% Utility Functions
%%% ============================================================================

-spec event_to_map(distribution_event()) -> map().
event_to_map(Event) when is_tuple(Event) ->
    [RecordName | Fields] = tuple_to_list(Event),
    FieldNames = record_fields(RecordName),
    MetaMap = meta_to_map(hd(Fields)),
    DataMap = maps:from_list(lists:zip(tl(FieldNames), tl(Fields))),
    maps:merge(#{event_type => RecordName, meta => MetaMap}, DataMap).

%%% ============================================================================
%%% Internal Functions
%%% ============================================================================

generate_id() ->
    Bytes = crypto:strong_rand_bytes(16),
    binary:encode_hex(Bytes).

meta_to_map(#lc_event_meta{} = Meta) ->
    #{
        event_id => Meta#lc_event_meta.event_id,
        correlation_id => Meta#lc_event_meta.correlation_id,
        causation_id => Meta#lc_event_meta.causation_id,
        timestamp => Meta#lc_event_meta.timestamp,
        version => Meta#lc_event_meta.version,
        emitter => Meta#lc_event_meta.emitter
    }.

record_fields(island_created) -> [meta, island_id, cluster_id, initial_population_size, compute_capacity, connected_islands];
record_fields(island_destroyed) -> [meta, island_id, cluster_id, destruction_reason, final_population, individuals_migrated];
record_fields(migration_route_established) -> [meta, route_id, source_island_id, target_island_id, bandwidth, latency_ms];
record_fields(migration_route_severed) -> [meta, route_id, source_island_id, target_island_id, severance_reason];
record_fields(load_rebalanced) -> [meta, cluster_id, load_variance_before, load_variance_after, individuals_migrated, rebalance_strategy];
record_fields(topology_updated) -> [meta, cluster_id, topology_type, island_count, route_count, diameter];
record_fields(node_joined) -> [meta, node_id, cluster_id, compute_capacity, assigned_islands];
record_fields(node_departed) -> [meta, node_id, cluster_id, departure_reason, affected_islands, individuals_orphaned];
record_fields(synchronization_completed) -> [meta, cluster_id, islands_synchronized, sync_duration_ms, conflicts_resolved].
