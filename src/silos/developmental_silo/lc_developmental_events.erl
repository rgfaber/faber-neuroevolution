%%%-------------------------------------------------------------------
%%% @doc Behavioral event constructors for Developmental Silo.
%%%
%%% Events related to ontogeny, critical periods, and plasticity.
%%% Developmental Silo operates at timescale τ=40.
%%%
%%% @end
%%%-------------------------------------------------------------------
-module(lc_developmental_events).

-include("lc_developmental_events.hrl").

%% Event constructors
-export([
    critical_period_opened/1,
    critical_period_closed/1,
    plasticity_changed/1,
    developmental_stage_reached/1,
    metamorphosis_triggered/1,
    canalization_increased/1,
    heterochrony_detected/1,
    developmental_milestone/1
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
%%% Critical Period Event Constructors
%%% ============================================================================

-spec critical_period_opened(map()) -> #critical_period_opened{}.
critical_period_opened(Data) ->
    #critical_period_opened{
        meta = make_meta(maps:get(emitter, Data, developmental_silo), Data),
        individual_id = maps:get(individual_id, Data),
        period_type = maps:get(period_type, Data, cognitive),
        plasticity_boost = maps:get(plasticity_boost, Data),
        expected_duration_generations = maps:get(expected_duration_generations, Data)
    }.

-spec critical_period_closed(map()) -> #critical_period_closed{}.
critical_period_closed(Data) ->
    #critical_period_closed{
        meta = make_meta(maps:get(emitter, Data, developmental_silo), Data),
        individual_id = maps:get(individual_id, Data),
        period_type = maps:get(period_type, Data, cognitive),
        actual_duration_generations = maps:get(actual_duration_generations, Data),
        learning_achieved = maps:get(learning_achieved, Data, 0.0)
    }.

%%% ============================================================================
%%% Plasticity Event Constructors
%%% ============================================================================

-spec plasticity_changed(map()) -> #plasticity_changed{}.
plasticity_changed(Data) ->
    #plasticity_changed{
        meta = make_meta(maps:get(emitter, Data, developmental_silo), Data),
        individual_id = maps:get(individual_id, Data),
        plasticity_before = maps:get(plasticity_before, Data),
        plasticity_after = maps:get(plasticity_after, Data),
        change_cause = maps:get(change_cause, Data, age)
    }.

%%% ============================================================================
%%% Developmental Stage Event Constructors
%%% ============================================================================

-spec developmental_stage_reached(map()) -> #developmental_stage_reached{}.
developmental_stage_reached(Data) ->
    #developmental_stage_reached{
        meta = make_meta(maps:get(emitter, Data, developmental_silo), Data),
        individual_id = maps:get(individual_id, Data),
        stage_before = maps:get(stage_before, Data),
        stage_after = maps:get(stage_after, Data),
        age_generations = maps:get(age_generations, Data),
        fitness_at_transition = maps:get(fitness_at_transition, Data)
    }.

-spec metamorphosis_triggered(map()) -> #metamorphosis_triggered{}.
metamorphosis_triggered(Data) ->
    #metamorphosis_triggered{
        meta = make_meta(maps:get(emitter, Data, developmental_silo), Data),
        individual_id = maps:get(individual_id, Data),
        stage_before = maps:get(stage_before, Data),
        stage_after = maps:get(stage_after, Data),
        structural_changes = maps:get(structural_changes, Data, #{neurons_added => 0, connections_rewired => 0}),
        fitness_before = maps:get(fitness_before, Data)
    }.

%%% ============================================================================
%%% Canalization Event Constructors
%%% ============================================================================

-spec canalization_increased(map()) -> #canalization_increased{}.
canalization_increased(Data) ->
    #canalization_increased{
        meta = make_meta(maps:get(emitter, Data, developmental_silo), Data),
        individual_id = maps:get(individual_id, Data),
        canalization_before = maps:get(canalization_before, Data),
        canalization_after = maps:get(canalization_after, Data),
        perturbation_resistance = maps:get(perturbation_resistance, Data)
    }.

-spec heterochrony_detected(map()) -> #heterochrony_detected{}.
heterochrony_detected(Data) ->
    #heterochrony_detected{
        meta = make_meta(maps:get(emitter, Data, developmental_silo), Data),
        individual_id = maps:get(individual_id, Data),
        timing_deviation = maps:get(timing_deviation, Data),
        affected_traits = maps:get(affected_traits, Data, []),
        fitness_impact = maps:get(fitness_impact, Data, 0.0)
    }.

-spec developmental_milestone(map()) -> #developmental_milestone{}.
developmental_milestone(Data) ->
    #developmental_milestone{
        meta = make_meta(maps:get(emitter, Data, developmental_silo), Data),
        individual_id = maps:get(individual_id, Data),
        milestone_type = maps:get(milestone_type, Data),
        age_generations = maps:get(age_generations, Data),
        performance_level = maps:get(performance_level, Data)
    }.

%%% ============================================================================
%%% Utility Functions
%%% ============================================================================

-spec event_to_map(developmental_event()) -> map().
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

record_fields(critical_period_opened) -> [meta, individual_id, period_type, plasticity_boost, expected_duration_generations];
record_fields(critical_period_closed) -> [meta, individual_id, period_type, actual_duration_generations, learning_achieved];
record_fields(plasticity_changed) -> [meta, individual_id, plasticity_before, plasticity_after, change_cause];
record_fields(developmental_stage_reached) -> [meta, individual_id, stage_before, stage_after, age_generations, fitness_at_transition];
record_fields(metamorphosis_triggered) -> [meta, individual_id, stage_before, stage_after, structural_changes, fitness_before];
record_fields(canalization_increased) -> [meta, individual_id, canalization_before, canalization_after, perturbation_resistance];
record_fields(heterochrony_detected) -> [meta, individual_id, timing_deviation, affected_traits, fitness_impact];
record_fields(developmental_milestone) -> [meta, individual_id, milestone_type, age_generations, performance_level].
