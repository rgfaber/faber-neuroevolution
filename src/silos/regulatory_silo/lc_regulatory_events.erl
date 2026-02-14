%%%-------------------------------------------------------------------
%%% @doc Behavioral event constructors for Regulatory Silo.
%%%
%%% Events related to gene expression and module activation.
%%% Regulatory Silo operates at timescale τ=45.
%%%
%%% @end
%%%-------------------------------------------------------------------
-module(lc_regulatory_events).

-include("lc_regulatory_events.hrl").

%% Event constructors
-export([
    gene_expressed/1,
    gene_silenced/1,
    module_activated/1,
    module_deactivated/1,
    context_switched/1,
    regulatory_network_updated/1,
    dormancy_entered/1,
    dormancy_exited/1
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
%%% Expression Event Constructors
%%% ============================================================================

-spec gene_expressed(map()) -> #gene_expressed{}.
gene_expressed(Data) ->
    #gene_expressed{
        meta = make_meta(maps:get(emitter, Data, regulatory_silo), Data),
        individual_id = maps:get(individual_id, Data),
        gene_id = maps:get(gene_id, Data),
        expression_level = maps:get(expression_level, Data),
        trigger = maps:get(trigger, Data, environmental),
        context = maps:get(context, Data, undefined)
    }.

-spec gene_silenced(map()) -> #gene_silenced{}.
gene_silenced(Data) ->
    #gene_silenced{
        meta = make_meta(maps:get(emitter, Data, regulatory_silo), Data),
        individual_id = maps:get(individual_id, Data),
        gene_id = maps:get(gene_id, Data),
        silencing_cause = maps:get(silencing_cause, Data, regulatory),
        duration = maps:get(duration, Data, temporary)
    }.

%%% ============================================================================
%%% Module Event Constructors
%%% ============================================================================

-spec module_activated(map()) -> #module_activated{}.
module_activated(Data) ->
    #module_activated{
        meta = make_meta(maps:get(emitter, Data, regulatory_silo), Data),
        individual_id = maps:get(individual_id, Data),
        module_id = maps:get(module_id, Data),
        activation_context = maps:get(activation_context, Data),
        neurons_involved = maps:get(neurons_involved, Data),
        switching_cost = maps:get(switching_cost, Data, 0.0)
    }.

-spec module_deactivated(map()) -> #module_deactivated{}.
module_deactivated(Data) ->
    #module_deactivated{
        meta = make_meta(maps:get(emitter, Data, regulatory_silo), Data),
        individual_id = maps:get(individual_id, Data),
        module_id = maps:get(module_id, Data),
        deactivation_reason = maps:get(deactivation_reason, Data, context_change),
        active_duration_ms = maps:get(active_duration_ms, Data, 0)
    }.

%%% ============================================================================
%%% Context Event Constructors
%%% ============================================================================

-spec context_switched(map()) -> #context_switched{}.
context_switched(Data) ->
    #context_switched{
        meta = make_meta(maps:get(emitter, Data, regulatory_silo), Data),
        individual_id = maps:get(individual_id, Data),
        context_before = maps:get(context_before, Data),
        context_after = maps:get(context_after, Data),
        genes_affected = maps:get(genes_affected, Data, []),
        modules_toggled = maps:get(modules_toggled, Data, 0)
    }.

-spec regulatory_network_updated(map()) -> #regulatory_network_updated{}.
regulatory_network_updated(Data) ->
    #regulatory_network_updated{
        meta = make_meta(maps:get(emitter, Data, regulatory_silo), Data),
        individual_id = maps:get(individual_id, Data),
        regulators_added = maps:get(regulators_added, Data, 0),
        regulators_removed = maps:get(regulators_removed, Data, 0),
        network_complexity = maps:get(network_complexity, Data)
    }.

%%% ============================================================================
%%% Dormancy Event Constructors
%%% ============================================================================

-spec dormancy_entered(map()) -> #dormancy_entered{}.
dormancy_entered(Data) ->
    #dormancy_entered{
        meta = make_meta(maps:get(emitter, Data, regulatory_silo), Data),
        individual_id = maps:get(individual_id, Data),
        dormant_capabilities = maps:get(dormant_capabilities, Data, []),
        dormancy_trigger = maps:get(dormancy_trigger, Data, resource_constraint),
        maintenance_cost = maps:get(maintenance_cost, Data, 0.0)
    }.

-spec dormancy_exited(map()) -> #dormancy_exited{}.
dormancy_exited(Data) ->
    #dormancy_exited{
        meta = make_meta(maps:get(emitter, Data, regulatory_silo), Data),
        individual_id = maps:get(individual_id, Data),
        reactivated_capabilities = maps:get(reactivated_capabilities, Data, []),
        reactivation_trigger = maps:get(reactivation_trigger, Data, opportunity),
        reactivation_delay_ms = maps:get(reactivation_delay_ms, Data, 0)
    }.

%%% ============================================================================
%%% Utility Functions
%%% ============================================================================

-spec event_to_map(regulatory_event()) -> map().
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

record_fields(gene_expressed) -> [meta, individual_id, gene_id, expression_level, trigger, context];
record_fields(gene_silenced) -> [meta, individual_id, gene_id, silencing_cause, duration];
record_fields(module_activated) -> [meta, individual_id, module_id, activation_context, neurons_involved, switching_cost];
record_fields(module_deactivated) -> [meta, individual_id, module_id, deactivation_reason, active_duration_ms];
record_fields(context_switched) -> [meta, individual_id, context_before, context_after, genes_affected, modules_toggled];
record_fields(regulatory_network_updated) -> [meta, individual_id, regulators_added, regulators_removed, network_complexity];
record_fields(dormancy_entered) -> [meta, individual_id, dormant_capabilities, dormancy_trigger, maintenance_cost];
record_fields(dormancy_exited) -> [meta, individual_id, reactivated_capabilities, reactivation_trigger, reactivation_delay_ms].
