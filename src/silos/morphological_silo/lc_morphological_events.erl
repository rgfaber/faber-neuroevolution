%%%-------------------------------------------------------------------
%%% @doc Behavioral event constructors for Morphological Silo.
%%%
%%% Events related to network complexity, structure, and efficiency.
%%% Morphological Silo operates at timescale τ=30.
%%%
%%% @end
%%%-------------------------------------------------------------------
-module(lc_morphological_events).

-include("lc_morphological_events.hrl").

%% Event constructors
-export([
    complexity_measured/1,
    pruning_triggered/1,
    growth_occurred/1,
    modularity_detected/1,
    efficiency_improved/1,
    bloat_detected/1,
    symmetry_broken/1,
    topology_milestone/1
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
%%% Complexity Event Constructors
%%% ============================================================================

-spec complexity_measured(map()) -> #complexity_measured{}.
complexity_measured(Data) ->
    #complexity_measured{
        meta = make_meta(maps:get(emitter, Data, morphological_silo), Data),
        individual_id = maps:get(individual_id, Data),
        neuron_count = maps:get(neuron_count, Data),
        connection_count = maps:get(connection_count, Data),
        parameter_count = maps:get(parameter_count, Data),
        complexity_score = maps:get(complexity_score, Data)
    }.

-spec pruning_triggered(map()) -> #pruning_triggered{}.
pruning_triggered(Data) ->
    #pruning_triggered{
        meta = make_meta(maps:get(emitter, Data, morphological_silo), Data),
        individual_id = maps:get(individual_id, Data),
        neurons_removed = maps:get(neurons_removed, Data, 0),
        connections_removed = maps:get(connections_removed, Data, 0),
        complexity_before = maps:get(complexity_before, Data),
        complexity_after = maps:get(complexity_after, Data),
        fitness_impact = maps:get(fitness_impact, Data, 0.0),
        pruning_criterion = maps:get(pruning_criterion, Data, unused)
    }.

%%% ============================================================================
%%% Growth Event Constructors
%%% ============================================================================

-spec growth_occurred(map()) -> #growth_occurred{}.
growth_occurred(Data) ->
    #growth_occurred{
        meta = make_meta(maps:get(emitter, Data, morphological_silo), Data),
        individual_id = maps:get(individual_id, Data),
        neurons_added = maps:get(neurons_added, Data, 0),
        connections_added = maps:get(connections_added, Data, 0),
        growth_trigger = maps:get(growth_trigger, Data, mutation),
        complexity_before = maps:get(complexity_before, Data),
        complexity_after = maps:get(complexity_after, Data)
    }.

-spec modularity_detected(map()) -> #modularity_detected{}.
modularity_detected(Data) ->
    #modularity_detected{
        meta = make_meta(maps:get(emitter, Data, morphological_silo), Data),
        individual_id = maps:get(individual_id, Data),
        module_count = maps:get(module_count, Data),
        modularity_score = maps:get(modularity_score, Data),
        module_sizes = maps:get(module_sizes, Data, []),
        inter_module_connectivity = maps:get(inter_module_connectivity, Data)
    }.

%%% ============================================================================
%%% Efficiency Event Constructors
%%% ============================================================================

-spec efficiency_improved(map()) -> #efficiency_improved{}.
efficiency_improved(Data) ->
    #efficiency_improved{
        meta = make_meta(maps:get(emitter, Data, morphological_silo), Data),
        individual_id = maps:get(individual_id, Data),
        fitness_before = maps:get(fitness_before, Data),
        fitness_after = maps:get(fitness_after, Data),
        complexity_before = maps:get(complexity_before, Data),
        complexity_after = maps:get(complexity_after, Data),
        efficiency_ratio_before = maps:get(efficiency_ratio_before, Data),
        efficiency_ratio_after = maps:get(efficiency_ratio_after, Data)
    }.

-spec bloat_detected(map()) -> #bloat_detected{}.
bloat_detected(Data) ->
    #bloat_detected{
        meta = make_meta(maps:get(emitter, Data, morphological_silo), Data),
        individual_id = maps:get(individual_id, Data),
        complexity = maps:get(complexity, Data),
        fitness = maps:get(fitness, Data),
        expected_complexity = maps:get(expected_complexity, Data),
        bloat_ratio = maps:get(bloat_ratio, Data)
    }.

%%% ============================================================================
%%% Structural Event Constructors
%%% ============================================================================

-spec symmetry_broken(map()) -> #symmetry_broken{}.
symmetry_broken(Data) ->
    #symmetry_broken{
        meta = make_meta(maps:get(emitter, Data, morphological_silo), Data),
        individual_id = maps:get(individual_id, Data),
        symmetry_before = maps:get(symmetry_before, Data),
        symmetry_after = maps:get(symmetry_after, Data),
        breaking_cause = maps:get(breaking_cause, Data, mutation)
    }.

-spec topology_milestone(map()) -> #topology_milestone{}.
topology_milestone(Data) ->
    #topology_milestone{
        meta = make_meta(maps:get(emitter, Data, morphological_silo), Data),
        individual_id = maps:get(individual_id, Data),
        milestone_type = maps:get(milestone_type, Data),
        neuron_count = maps:get(neuron_count, Data),
        connection_count = maps:get(connection_count, Data),
        generation = maps:get(generation, Data)
    }.

%%% ============================================================================
%%% Utility Functions
%%% ============================================================================

-spec event_to_map(morphological_event()) -> map().
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

record_fields(complexity_measured) -> [meta, individual_id, neuron_count, connection_count, parameter_count, complexity_score];
record_fields(pruning_triggered) -> [meta, individual_id, neurons_removed, connections_removed, complexity_before, complexity_after, fitness_impact, pruning_criterion];
record_fields(growth_occurred) -> [meta, individual_id, neurons_added, connections_added, growth_trigger, complexity_before, complexity_after];
record_fields(modularity_detected) -> [meta, individual_id, module_count, modularity_score, module_sizes, inter_module_connectivity];
record_fields(efficiency_improved) -> [meta, individual_id, fitness_before, fitness_after, complexity_before, complexity_after, efficiency_ratio_before, efficiency_ratio_after];
record_fields(bloat_detected) -> [meta, individual_id, complexity, fitness, expected_complexity, bloat_ratio];
record_fields(symmetry_broken) -> [meta, individual_id, symmetry_before, symmetry_after, breaking_cause];
record_fields(topology_milestone) -> [meta, individual_id, milestone_type, neuron_count, connection_count, generation].
