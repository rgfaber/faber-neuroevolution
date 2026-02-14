%%%-------------------------------------------------------------------
%%% @doc Behavioral event definitions for Communication Silo (τ=55).
%%%
%%% Events related to signaling, vocabulary, and coordination.
%%%
%%% @end
%%%-------------------------------------------------------------------
-ifndef(LC_COMMUNICATION_EVENTS_HRL).
-define(LC_COMMUNICATION_EVENTS_HRL, true).

-include("lc_events_common.hrl").

%%% ============================================================================
%%% Signal Events
%%% ============================================================================

-record(signal_emitted, {
    meta :: #lc_event_meta{},
    sender_id :: individual_id(),
    signal_id :: binary(),
    signal_content :: term(),
    intended_receivers :: [individual_id()] | broadcast,
    honesty :: honest | deceptive
}).

-record(signal_received, {
    meta :: #lc_event_meta{},
    receiver_id :: individual_id(),
    sender_id :: individual_id(),
    signal_id :: binary(),
    interpretation :: correct | incorrect | ambiguous
}).

-record(signal_interpreted, {
    meta :: #lc_event_meta{},
    interpreter_id :: individual_id(),
    signal_id :: binary(),
    intended_meaning :: term(),
    interpreted_meaning :: term(),
    accuracy :: float()
}).

%%% ============================================================================
%%% Vocabulary Events
%%% ============================================================================

-record(vocabulary_expanded, {
    meta :: #lc_event_meta{},
    population_id :: population_id(),
    new_signal_id :: binary(),
    signal_meaning :: term(),
    inventor_id :: individual_id(),
    vocabulary_size_after :: non_neg_integer()
}).

-record(vocabulary_contracted, {
    meta :: #lc_event_meta{},
    population_id :: population_id(),
    removed_signal_id :: binary(),
    removal_reason :: obsolescence | ambiguity | replacement,
    vocabulary_size_after :: non_neg_integer()
}).

%%% ============================================================================
%%% Dialect Events
%%% ============================================================================

-record(dialect_formed, {
    meta :: #lc_event_meta{},
    dialect_id :: binary(),
    population_id :: population_id(),
    speaker_ids :: [individual_id()],
    unique_signals :: non_neg_integer(),
    formation_cause :: isolation | drift | innovation
}).

-record(dialect_merged, {
    meta :: #lc_event_meta{},
    dialect_a_id :: binary(),
    dialect_b_id :: binary(),
    merged_dialect_id :: binary(),
    merge_cause :: contact | migration | normalization
}).

%%% ============================================================================
%%% Coordination Events
%%% ============================================================================

-record(coordination_achieved, {
    meta :: #lc_event_meta{},
    coordination_id :: binary(),
    participant_ids :: [individual_id()],
    coordination_type :: synchronization | division_of_labor | joint_action,
    success_level :: float(),
    communication_rounds :: non_neg_integer()
}).

-record(deception_detected, {
    meta :: #lc_event_meta{},
    deceiver_id :: individual_id(),
    detector_id :: individual_id(),
    signal_id :: binary(),
    deception_type :: false_alarm | concealment | manipulation,
    detection_confidence :: float()
}).

%%% ============================================================================
%%% Type Exports
%%% ============================================================================

-type communication_event() :: #signal_emitted{} | #signal_received{} | #signal_interpreted{} |
                               #vocabulary_expanded{} | #vocabulary_contracted{} |
                               #dialect_formed{} | #dialect_merged{} |
                               #coordination_achieved{} | #deception_detected{}.

-endif. %% LC_COMMUNICATION_EVENTS_HRL
