%%%-------------------------------------------------------------------
%%% @doc Behavioral event constructors for Social Silo.
%%%
%%% Events related to reputation, coalitions, and social networks.
%%% Social Silo operates at timescale τ=25.
%%%
%%% @end
%%%-------------------------------------------------------------------
-module(lc_social_events).

-include("lc_social_events.hrl").

%% Event constructors
-export([
    reputation_changed/1,
    coalition_formed/1,
    coalition_dissolved/1,
    coalition_member_joined/1,
    coalition_member_expelled/1,
    mentoring_started/1,
    mentoring_ended/1,
    social_network_updated/1,
    trust_established/1,
    betrayal_detected/1
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
%%% Reputation Event Constructors
%%% ============================================================================

-spec reputation_changed(map()) -> #reputation_changed{}.
reputation_changed(Data) ->
    #reputation_changed{
        meta = make_meta(maps:get(emitter, Data, social_silo), Data),
        individual_id = maps:get(individual_id, Data),
        population_id = maps:get(population_id, Data),
        reputation_before = maps:get(reputation_before, Data),
        reputation_after = maps:get(reputation_after, Data),
        change_cause = maps:get(change_cause, Data, cooperation),
        witnesses = maps:get(witnesses, Data, [])
    }.

%%% ============================================================================
%%% Coalition Event Constructors
%%% ============================================================================

-spec coalition_formed(map()) -> #coalition_formed{}.
coalition_formed(Data) ->
    #coalition_formed{
        meta = make_meta(maps:get(emitter, Data, social_silo), Data),
        coalition_id = maps:get(coalition_id, Data, generate_id()),
        founder_ids = maps:get(founder_ids, Data),
        population_id = maps:get(population_id, Data),
        formation_reason = maps:get(formation_reason, Data, defense),
        initial_strength = maps:get(initial_strength, Data, 1.0)
    }.

-spec coalition_dissolved(map()) -> #coalition_dissolved{}.
coalition_dissolved(Data) ->
    #coalition_dissolved{
        meta = make_meta(maps:get(emitter, Data, social_silo), Data),
        coalition_id = maps:get(coalition_id, Data),
        population_id = maps:get(population_id, Data),
        dissolution_reason = maps:get(dissolution_reason, Data, success),
        duration_generations = maps:get(duration_generations, Data, 0),
        final_members = maps:get(final_members, Data, [])
    }.

-spec coalition_member_joined(map()) -> #coalition_member_joined{}.
coalition_member_joined(Data) ->
    #coalition_member_joined{
        meta = make_meta(maps:get(emitter, Data, social_silo), Data),
        coalition_id = maps:get(coalition_id, Data),
        individual_id = maps:get(individual_id, Data),
        membership_type = maps:get(membership_type, Data, full),
        coalition_size_after = maps:get(coalition_size_after, Data)
    }.

-spec coalition_member_expelled(map()) -> #coalition_member_expelled{}.
coalition_member_expelled(Data) ->
    #coalition_member_expelled{
        meta = make_meta(maps:get(emitter, Data, social_silo), Data),
        coalition_id = maps:get(coalition_id, Data),
        individual_id = maps:get(individual_id, Data),
        expulsion_reason = maps:get(expulsion_reason, Data, defection),
        coalition_size_after = maps:get(coalition_size_after, Data)
    }.

%%% ============================================================================
%%% Mentoring Event Constructors
%%% ============================================================================

-spec mentoring_started(map()) -> #mentoring_started{}.
mentoring_started(Data) ->
    #mentoring_started{
        meta = make_meta(maps:get(emitter, Data, social_silo), Data),
        mentor_id = maps:get(mentor_id, Data),
        mentee_id = maps:get(mentee_id, Data),
        mentoring_type = maps:get(mentoring_type, Data, general),
        expected_duration = maps:get(expected_duration, Data, indefinite)
    }.

-spec mentoring_ended(map()) -> #mentoring_ended{}.
mentoring_ended(Data) ->
    #mentoring_ended{
        meta = make_meta(maps:get(emitter, Data, social_silo), Data),
        mentor_id = maps:get(mentor_id, Data),
        mentee_id = maps:get(mentee_id, Data),
        outcome = maps:get(outcome, Data, success),
        duration_generations = maps:get(duration_generations, Data, 0),
        mentee_improvement = maps:get(mentee_improvement, Data, 0.0)
    }.

%%% ============================================================================
%%% Network Event Constructors
%%% ============================================================================

-spec social_network_updated(map()) -> #social_network_updated{}.
social_network_updated(Data) ->
    #social_network_updated{
        meta = make_meta(maps:get(emitter, Data, social_silo), Data),
        population_id = maps:get(population_id, Data),
        connections_added = maps:get(connections_added, Data, 0),
        connections_removed = maps:get(connections_removed, Data, 0),
        network_density = maps:get(network_density, Data),
        clustering_coefficient = maps:get(clustering_coefficient, Data)
    }.

-spec trust_established(map()) -> #trust_established{}.
trust_established(Data) ->
    #trust_established{
        meta = make_meta(maps:get(emitter, Data, social_silo), Data),
        individual_a_id = maps:get(individual_a_id, Data),
        individual_b_id = maps:get(individual_b_id, Data),
        trust_level = maps:get(trust_level, Data),
        trust_basis = maps:get(trust_basis, Data, cooperation)
    }.

-spec betrayal_detected(map()) -> #betrayal_detected{}.
betrayal_detected(Data) ->
    #betrayal_detected{
        meta = make_meta(maps:get(emitter, Data, social_silo), Data),
        betrayer_id = maps:get(betrayer_id, Data),
        victim_id = maps:get(victim_id, Data),
        betrayal_type = maps:get(betrayal_type, Data, defection),
        trust_before = maps:get(trust_before, Data),
        trust_after = maps:get(trust_after, Data)
    }.

%%% ============================================================================
%%% Utility Functions
%%% ============================================================================

-spec event_to_map(social_event()) -> map().
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

record_fields(reputation_changed) -> [meta, individual_id, population_id, reputation_before, reputation_after, change_cause, witnesses];
record_fields(coalition_formed) -> [meta, coalition_id, founder_ids, population_id, formation_reason, initial_strength];
record_fields(coalition_dissolved) -> [meta, coalition_id, population_id, dissolution_reason, duration_generations, final_members];
record_fields(coalition_member_joined) -> [meta, coalition_id, individual_id, membership_type, coalition_size_after];
record_fields(coalition_member_expelled) -> [meta, coalition_id, individual_id, expulsion_reason, coalition_size_after];
record_fields(mentoring_started) -> [meta, mentor_id, mentee_id, mentoring_type, expected_duration];
record_fields(mentoring_ended) -> [meta, mentor_id, mentee_id, outcome, duration_generations, mentee_improvement];
record_fields(social_network_updated) -> [meta, population_id, connections_added, connections_removed, network_density, clustering_coefficient];
record_fields(trust_established) -> [meta, individual_a_id, individual_b_id, trust_level, trust_basis];
record_fields(betrayal_detected) -> [meta, betrayer_id, victim_id, betrayal_type, trust_before, trust_after].
