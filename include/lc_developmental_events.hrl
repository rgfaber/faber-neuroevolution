%%%-------------------------------------------------------------------
%%% @doc Behavioral event definitions for Developmental Silo (τ=40).
%%%
%%% Events related to ontogeny, critical periods, and plasticity.
%%%
%%% @end
%%%-------------------------------------------------------------------
-ifndef(LC_DEVELOPMENTAL_EVENTS_HRL).
-define(LC_DEVELOPMENTAL_EVENTS_HRL, true).

-include("lc_events_common.hrl").

%%% ============================================================================
%%% Critical Period Events
%%% ============================================================================

-record(critical_period_opened, {
    meta :: #lc_event_meta{},
    individual_id :: individual_id(),
    period_type :: sensory | motor | cognitive | social,
    plasticity_boost :: float(),
    expected_duration_generations :: non_neg_integer()
}).

-record(critical_period_closed, {
    meta :: #lc_event_meta{},
    individual_id :: individual_id(),
    period_type :: sensory | motor | cognitive | social,
    actual_duration_generations :: non_neg_integer(),
    learning_achieved :: float()
}).

%%% ============================================================================
%%% Plasticity Events
%%% ============================================================================

-record(plasticity_changed, {
    meta :: #lc_event_meta{},
    individual_id :: individual_id(),
    plasticity_before :: float(),
    plasticity_after :: float(),
    change_cause :: age | experience | stress | development
}).

%%% ============================================================================
%%% Developmental Stage Events
%%% ============================================================================

-record(developmental_stage_reached, {
    meta :: #lc_event_meta{},
    individual_id :: individual_id(),
    stage_before :: atom(),
    stage_after :: atom(),
    age_generations :: non_neg_integer(),
    fitness_at_transition :: float()
}).

-record(metamorphosis_triggered, {
    meta :: #lc_event_meta{},
    individual_id :: individual_id(),
    stage_before :: atom(),
    stage_after :: atom(),
    structural_changes :: #{
        neurons_added => non_neg_integer(),
        connections_rewired => non_neg_integer()
    },
    fitness_before :: float()
}).

%%% ============================================================================
%%% Canalization Events
%%% ============================================================================

-record(canalization_increased, {
    meta :: #lc_event_meta{},
    individual_id :: individual_id(),
    canalization_before :: float(),
    canalization_after :: float(),
    perturbation_resistance :: float()
}).

-record(heterochrony_detected, {
    meta :: #lc_event_meta{},
    individual_id :: individual_id(),
    timing_deviation :: float(),
    affected_traits :: [atom()],
    fitness_impact :: float()
}).

-record(developmental_milestone, {
    meta :: #lc_event_meta{},
    individual_id :: individual_id(),
    milestone_type :: first_signal | first_action | first_success | maturity,
    age_generations :: non_neg_integer(),
    performance_level :: float()
}).

%%% ============================================================================
%%% Type Exports
%%% ============================================================================

-type developmental_event() :: #critical_period_opened{} | #critical_period_closed{} |
                               #plasticity_changed{} |
                               #developmental_stage_reached{} | #metamorphosis_triggered{} |
                               #canalization_increased{} | #heterochrony_detected{} |
                               #developmental_milestone{}.

-endif. %% LC_DEVELOPMENTAL_EVENTS_HRL
