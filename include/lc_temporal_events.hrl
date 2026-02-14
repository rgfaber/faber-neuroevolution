%%%-------------------------------------------------------------------
%%% @doc Behavioral event definitions for Temporal Silo (τ=10).
%%%
%%% Events related to timing, episodes, and learning rate adaptation.
%%%
%%% @end
%%%-------------------------------------------------------------------
-ifndef(LC_TEMPORAL_EVENTS_HRL).
-define(LC_TEMPORAL_EVENTS_HRL, true).

-include("lc_events_common.hrl").

%%% ============================================================================
%%% Episode Events
%%% ============================================================================

-record(episode_started, {
    meta :: #lc_event_meta{},
    individual_id :: individual_id(),
    episode_number :: non_neg_integer(),
    expected_duration_ms :: non_neg_integer() | undefined,
    generation :: generation()
}).

-record(episode_completed, {
    meta :: #lc_event_meta{},
    individual_id :: individual_id(),
    episode_number :: non_neg_integer(),
    duration_ms :: non_neg_integer(),
    outcome :: success | failure | timeout | early_termination,
    fitness_delta :: float(),
    steps_taken :: non_neg_integer()
}).

%%% ============================================================================
%%% Timing Events
%%% ============================================================================

-record(timing_adjusted, {
    meta :: #lc_event_meta{},
    population_id :: population_id(),
    previous_timeout_ms :: non_neg_integer(),
    new_timeout_ms :: non_neg_integer(),
    adjustment_reason :: convergence | stagnation | resource_pressure
}).

-record(learning_rate_adapted, {
    meta :: #lc_event_meta{},
    population_id :: population_id(),
    previous_rate :: float(),
    new_rate :: float(),
    adaptation_reason :: stagnation | breakthrough | scheduled | convergence,
    generation :: generation()
}).

%%% ============================================================================
%%% Termination Events
%%% ============================================================================

-record(patience_exhausted, {
    meta :: #lc_event_meta{},
    population_id :: population_id(),
    generations_waited :: non_neg_integer(),
    best_fitness :: float(),
    action_taken :: terminate | reset | diversify
}).

-record(convergence_detected, {
    meta :: #lc_event_meta{},
    population_id :: population_id(),
    fitness_variance :: float(),
    generations_stable :: non_neg_integer(),
    generation :: generation()
}).

-record(timeout_threshold_reached, {
    meta :: #lc_event_meta{},
    individual_id :: individual_id(),
    elapsed_ms :: non_neg_integer(),
    threshold_ms :: non_neg_integer(),
    action :: terminate | extend | warn
}).

%%% ============================================================================
%%% Type Exports
%%% ============================================================================

-type temporal_event() :: #episode_started{} | #episode_completed{} |
                          #timing_adjusted{} | #learning_rate_adapted{} |
                          #patience_exhausted{} | #convergence_detected{} |
                          #timeout_threshold_reached{}.

-endif. %% LC_TEMPORAL_EVENTS_HRL
