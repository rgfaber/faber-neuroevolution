%%%-------------------------------------------------------------------
%%% @doc Behavioral event definitions for Social Silo (τ=25).
%%%
%%% Events related to reputation, coalitions, and social networks.
%%%
%%% @end
%%%-------------------------------------------------------------------
-ifndef(LC_SOCIAL_EVENTS_HRL).
-define(LC_SOCIAL_EVENTS_HRL, true).

-include("lc_events_common.hrl").

%%% ============================================================================
%%% Reputation Events
%%% ============================================================================

-record(reputation_changed, {
    meta :: #lc_event_meta{},
    individual_id :: individual_id(),
    population_id :: population_id(),
    reputation_before :: float(),
    reputation_after :: float(),
    change_cause :: cooperation | defection | achievement | punishment,
    witnesses :: [individual_id()]
}).

%%% ============================================================================
%%% Coalition Events
%%% ============================================================================

-record(coalition_formed, {
    meta :: #lc_event_meta{},
    coalition_id :: binary(),
    founder_ids :: [individual_id()],
    population_id :: population_id(),
    formation_reason :: defense | resource | breeding,
    initial_strength :: float()
}).

-record(coalition_dissolved, {
    meta :: #lc_event_meta{},
    coalition_id :: binary(),
    population_id :: population_id(),
    dissolution_reason :: success | failure | conflict | attrition,
    duration_generations :: non_neg_integer(),
    final_members :: [individual_id()]
}).

-record(coalition_member_joined, {
    meta :: #lc_event_meta{},
    coalition_id :: binary(),
    individual_id :: individual_id(),
    membership_type :: full | associate | candidate,
    coalition_size_after :: non_neg_integer()
}).

-record(coalition_member_expelled, {
    meta :: #lc_event_meta{},
    coalition_id :: binary(),
    individual_id :: individual_id(),
    expulsion_reason :: defection | underperformance | conflict,
    coalition_size_after :: non_neg_integer()
}).

%%% ============================================================================
%%% Mentoring Events
%%% ============================================================================

-record(mentoring_started, {
    meta :: #lc_event_meta{},
    mentor_id :: individual_id(),
    mentee_id :: individual_id(),
    mentoring_type :: skill | strategy | general,
    expected_duration :: non_neg_integer() | indefinite
}).

-record(mentoring_ended, {
    meta :: #lc_event_meta{},
    mentor_id :: individual_id(),
    mentee_id :: individual_id(),
    outcome :: success | failure | interrupted,
    duration_generations :: non_neg_integer(),
    mentee_improvement :: float()
}).

%%% ============================================================================
%%% Network Events
%%% ============================================================================

-record(social_network_updated, {
    meta :: #lc_event_meta{},
    population_id :: population_id(),
    connections_added :: non_neg_integer(),
    connections_removed :: non_neg_integer(),
    network_density :: float(),
    clustering_coefficient :: float()
}).

-record(trust_established, {
    meta :: #lc_event_meta{},
    individual_a_id :: individual_id(),
    individual_b_id :: individual_id(),
    trust_level :: float(),
    trust_basis :: cooperation | kinship | reputation
}).

-record(betrayal_detected, {
    meta :: #lc_event_meta{},
    betrayer_id :: individual_id(),
    victim_id :: individual_id(),
    betrayal_type :: defection | deception | abandonment,
    trust_before :: float(),
    trust_after :: float()
}).

%%% ============================================================================
%%% Type Exports
%%% ============================================================================

-type social_event() :: #reputation_changed{} |
                        #coalition_formed{} | #coalition_dissolved{} |
                        #coalition_member_joined{} | #coalition_member_expelled{} |
                        #mentoring_started{} | #mentoring_ended{} |
                        #social_network_updated{} | #trust_established{} |
                        #betrayal_detected{}.

-endif. %% LC_SOCIAL_EVENTS_HRL
