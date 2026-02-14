%%%-------------------------------------------------------------------
%%% @doc Behavioral event constructors for Ecological Silo.
%%%
%%% Events related to niches, environmental stress, and carrying capacity.
%%% Ecological Silo operates at timescale τ=50.
%%%
%%% @end
%%%-------------------------------------------------------------------
-module(lc_ecological_events).

-include("lc_ecological_events.hrl").

%% Event constructors
-export([
    niche_occupied/1,
    niche_vacated/1,
    stress_applied/1,
    stress_relieved/1,
    carrying_capacity_changed/1,
    resource_scarcity_detected/1,
    resource_abundance_detected/1,
    extinction_risk_elevated/1,
    ecosystem_disrupted/1
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
%%% Niche Event Constructors
%%% ============================================================================

-spec niche_occupied(map()) -> #niche_occupied{}.
niche_occupied(Data) ->
    #niche_occupied{
        meta = make_meta(maps:get(emitter, Data, ecological_silo), Data),
        niche_id = maps:get(niche_id, Data),
        occupant_id = maps:get(occupant_id, Data),
        population_id = maps:get(population_id, Data),
        niche_fitness_range = maps:get(niche_fitness_range, Data),
        competition_level = maps:get(competition_level, Data, 0.0)
    }.

-spec niche_vacated(map()) -> #niche_vacated{}.
niche_vacated(Data) ->
    #niche_vacated{
        meta = make_meta(maps:get(emitter, Data, ecological_silo), Data),
        niche_id = maps:get(niche_id, Data),
        vacating_id = maps:get(vacating_id, Data, undefined),
        population_id = maps:get(population_id, Data),
        vacation_reason = maps:get(vacation_reason, Data, extinction)
    }.

%%% ============================================================================
%%% Stress Event Constructors
%%% ============================================================================

-spec stress_applied(map()) -> #stress_applied{}.
stress_applied(Data) ->
    #stress_applied{
        meta = make_meta(maps:get(emitter, Data, ecological_silo), Data),
        population_id = maps:get(population_id, Data),
        stress_type = maps:get(stress_type, Data, environmental),
        stress_intensity = maps:get(stress_intensity, Data),
        affected_individuals = maps:get(affected_individuals, Data),
        expected_duration = maps:get(expected_duration, Data, indefinite)
    }.

-spec stress_relieved(map()) -> #stress_relieved{}.
stress_relieved(Data) ->
    #stress_relieved{
        meta = make_meta(maps:get(emitter, Data, ecological_silo), Data),
        population_id = maps:get(population_id, Data),
        stress_type = maps:get(stress_type, Data, environmental),
        previous_intensity = maps:get(previous_intensity, Data),
        relief_cause = maps:get(relief_cause, Data, adaptation)
    }.

%%% ============================================================================
%%% Capacity Event Constructors
%%% ============================================================================

-spec carrying_capacity_changed(map()) -> #carrying_capacity_changed{}.
carrying_capacity_changed(Data) ->
    #carrying_capacity_changed{
        meta = make_meta(maps:get(emitter, Data, ecological_silo), Data),
        population_id = maps:get(population_id, Data),
        capacity_before = maps:get(capacity_before, Data),
        capacity_after = maps:get(capacity_after, Data),
        change_cause = maps:get(change_cause, Data, resource_availability)
    }.

%%% ============================================================================
%%% Resource Event Constructors
%%% ============================================================================

-spec resource_scarcity_detected(map()) -> #resource_scarcity_detected{}.
resource_scarcity_detected(Data) ->
    #resource_scarcity_detected{
        meta = make_meta(maps:get(emitter, Data, ecological_silo), Data),
        population_id = maps:get(population_id, Data),
        resource_type = maps:get(resource_type, Data, compute),
        availability = maps:get(availability, Data),
        threshold = maps:get(threshold, Data),
        affected_ratio = maps:get(affected_ratio, Data)
    }.

-spec resource_abundance_detected(map()) -> #resource_abundance_detected{}.
resource_abundance_detected(Data) ->
    #resource_abundance_detected{
        meta = make_meta(maps:get(emitter, Data, ecological_silo), Data),
        population_id = maps:get(population_id, Data),
        resource_type = maps:get(resource_type, Data, compute),
        availability = maps:get(availability, Data),
        surplus_ratio = maps:get(surplus_ratio, Data)
    }.

%%% ============================================================================
%%% Risk Event Constructors
%%% ============================================================================

-spec extinction_risk_elevated(map()) -> #extinction_risk_elevated{}.
extinction_risk_elevated(Data) ->
    #extinction_risk_elevated{
        meta = make_meta(maps:get(emitter, Data, ecological_silo), Data),
        species_id = maps:get(species_id, Data),
        population_id = maps:get(population_id, Data),
        risk_level = maps:get(risk_level, Data),
        risk_factors = maps:get(risk_factors, Data, []),
        population_size = maps:get(population_size, Data)
    }.

-spec ecosystem_disrupted(map()) -> #ecosystem_disrupted{}.
ecosystem_disrupted(Data) ->
    #ecosystem_disrupted{
        meta = make_meta(maps:get(emitter, Data, ecological_silo), Data),
        population_id = maps:get(population_id, Data),
        disruption_type = maps:get(disruption_type, Data, catastrophe),
        disruption_severity = maps:get(disruption_severity, Data),
        species_affected = maps:get(species_affected, Data),
        recovery_estimate_generations = maps:get(recovery_estimate_generations, Data)
    }.

%%% ============================================================================
%%% Utility Functions
%%% ============================================================================

-spec event_to_map(ecological_event()) -> map().
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

record_fields(niche_occupied) -> [meta, niche_id, occupant_id, population_id, niche_fitness_range, competition_level];
record_fields(niche_vacated) -> [meta, niche_id, vacating_id, population_id, vacation_reason];
record_fields(stress_applied) -> [meta, population_id, stress_type, stress_intensity, affected_individuals, expected_duration];
record_fields(stress_relieved) -> [meta, population_id, stress_type, previous_intensity, relief_cause];
record_fields(carrying_capacity_changed) -> [meta, population_id, capacity_before, capacity_after, change_cause];
record_fields(resource_scarcity_detected) -> [meta, population_id, resource_type, availability, threshold, affected_ratio];
record_fields(resource_abundance_detected) -> [meta, population_id, resource_type, availability, surplus_ratio];
record_fields(extinction_risk_elevated) -> [meta, species_id, population_id, risk_level, risk_factors, population_size];
record_fields(ecosystem_disrupted) -> [meta, population_id, disruption_type, disruption_severity, species_affected, recovery_estimate_generations].
