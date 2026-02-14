%%%-------------------------------------------------------------------
%%% @doc Behavioral event constructors for Communication Silo.
%%%
%%% Events related to signaling, vocabulary, and coordination.
%%% Communication Silo operates at timescale τ=55.
%%%
%%% @end
%%%-------------------------------------------------------------------
-module(lc_communication_events).

-include("lc_communication_events.hrl").

%% Event constructors
-export([
    signal_emitted/1,
    signal_received/1,
    signal_interpreted/1,
    vocabulary_expanded/1,
    vocabulary_contracted/1,
    dialect_formed/1,
    dialect_merged/1,
    coordination_achieved/1,
    deception_detected/1
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
%%% Signal Event Constructors
%%% ============================================================================

-spec signal_emitted(map()) -> #signal_emitted{}.
signal_emitted(Data) ->
    #signal_emitted{
        meta = make_meta(maps:get(emitter, Data, communication_silo), Data),
        sender_id = maps:get(sender_id, Data),
        signal_id = maps:get(signal_id, Data, generate_id()),
        signal_content = maps:get(signal_content, Data),
        intended_receivers = maps:get(intended_receivers, Data, broadcast),
        honesty = maps:get(honesty, Data, honest)
    }.

-spec signal_received(map()) -> #signal_received{}.
signal_received(Data) ->
    #signal_received{
        meta = make_meta(maps:get(emitter, Data, communication_silo), Data),
        receiver_id = maps:get(receiver_id, Data),
        sender_id = maps:get(sender_id, Data),
        signal_id = maps:get(signal_id, Data),
        interpretation = maps:get(interpretation, Data, correct)
    }.

-spec signal_interpreted(map()) -> #signal_interpreted{}.
signal_interpreted(Data) ->
    #signal_interpreted{
        meta = make_meta(maps:get(emitter, Data, communication_silo), Data),
        interpreter_id = maps:get(interpreter_id, Data),
        signal_id = maps:get(signal_id, Data),
        intended_meaning = maps:get(intended_meaning, Data),
        interpreted_meaning = maps:get(interpreted_meaning, Data),
        accuracy = maps:get(accuracy, Data, 1.0)
    }.

%%% ============================================================================
%%% Vocabulary Event Constructors
%%% ============================================================================

-spec vocabulary_expanded(map()) -> #vocabulary_expanded{}.
vocabulary_expanded(Data) ->
    #vocabulary_expanded{
        meta = make_meta(maps:get(emitter, Data, communication_silo), Data),
        population_id = maps:get(population_id, Data),
        new_signal_id = maps:get(new_signal_id, Data),
        signal_meaning = maps:get(signal_meaning, Data),
        inventor_id = maps:get(inventor_id, Data),
        vocabulary_size_after = maps:get(vocabulary_size_after, Data)
    }.

-spec vocabulary_contracted(map()) -> #vocabulary_contracted{}.
vocabulary_contracted(Data) ->
    #vocabulary_contracted{
        meta = make_meta(maps:get(emitter, Data, communication_silo), Data),
        population_id = maps:get(population_id, Data),
        removed_signal_id = maps:get(removed_signal_id, Data),
        removal_reason = maps:get(removal_reason, Data, obsolescence),
        vocabulary_size_after = maps:get(vocabulary_size_after, Data)
    }.

%%% ============================================================================
%%% Dialect Event Constructors
%%% ============================================================================

-spec dialect_formed(map()) -> #dialect_formed{}.
dialect_formed(Data) ->
    #dialect_formed{
        meta = make_meta(maps:get(emitter, Data, communication_silo), Data),
        dialect_id = maps:get(dialect_id, Data, generate_id()),
        population_id = maps:get(population_id, Data),
        speaker_ids = maps:get(speaker_ids, Data, []),
        unique_signals = maps:get(unique_signals, Data),
        formation_cause = maps:get(formation_cause, Data, isolation)
    }.

-spec dialect_merged(map()) -> #dialect_merged{}.
dialect_merged(Data) ->
    #dialect_merged{
        meta = make_meta(maps:get(emitter, Data, communication_silo), Data),
        dialect_a_id = maps:get(dialect_a_id, Data),
        dialect_b_id = maps:get(dialect_b_id, Data),
        merged_dialect_id = maps:get(merged_dialect_id, Data, generate_id()),
        merge_cause = maps:get(merge_cause, Data, contact)
    }.

%%% ============================================================================
%%% Coordination Event Constructors
%%% ============================================================================

-spec coordination_achieved(map()) -> #coordination_achieved{}.
coordination_achieved(Data) ->
    #coordination_achieved{
        meta = make_meta(maps:get(emitter, Data, communication_silo), Data),
        coordination_id = maps:get(coordination_id, Data, generate_id()),
        participant_ids = maps:get(participant_ids, Data, []),
        coordination_type = maps:get(coordination_type, Data, synchronization),
        success_level = maps:get(success_level, Data),
        communication_rounds = maps:get(communication_rounds, Data, 0)
    }.

-spec deception_detected(map()) -> #deception_detected{}.
deception_detected(Data) ->
    #deception_detected{
        meta = make_meta(maps:get(emitter, Data, communication_silo), Data),
        deceiver_id = maps:get(deceiver_id, Data),
        detector_id = maps:get(detector_id, Data),
        signal_id = maps:get(signal_id, Data),
        deception_type = maps:get(deception_type, Data, false_alarm),
        detection_confidence = maps:get(detection_confidence, Data)
    }.

%%% ============================================================================
%%% Utility Functions
%%% ============================================================================

-spec event_to_map(communication_event()) -> map().
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

record_fields(signal_emitted) -> [meta, sender_id, signal_id, signal_content, intended_receivers, honesty];
record_fields(signal_received) -> [meta, receiver_id, sender_id, signal_id, interpretation];
record_fields(signal_interpreted) -> [meta, interpreter_id, signal_id, intended_meaning, interpreted_meaning, accuracy];
record_fields(vocabulary_expanded) -> [meta, population_id, new_signal_id, signal_meaning, inventor_id, vocabulary_size_after];
record_fields(vocabulary_contracted) -> [meta, population_id, removed_signal_id, removal_reason, vocabulary_size_after];
record_fields(dialect_formed) -> [meta, dialect_id, population_id, speaker_ids, unique_signals, formation_cause];
record_fields(dialect_merged) -> [meta, dialect_a_id, dialect_b_id, merged_dialect_id, merge_cause];
record_fields(coordination_achieved) -> [meta, coordination_id, participant_ids, coordination_type, success_level, communication_rounds];
record_fields(deception_detected) -> [meta, deceiver_id, detector_id, signal_id, deception_type, detection_confidence].
