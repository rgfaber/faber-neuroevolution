%%%-------------------------------------------------------------------
%%% @doc Behavioral event definitions for Competitive Silo (τ=15).
%%%
%%% Events related to Elo ratings, opponent archives, and matchmaking.
%%%
%%% @end
%%%-------------------------------------------------------------------
-ifndef(LC_COMPETITIVE_EVENTS_HRL).
-define(LC_COMPETITIVE_EVENTS_HRL, true).

-include("lc_events_common.hrl").

%%% ============================================================================
%%% Archive Events
%%% ============================================================================

-record(opponent_archived, {
    meta :: #lc_event_meta{},
    opponent_id :: individual_id(),
    archive_id :: binary(),
    fitness_at_archive :: float(),
    elo_at_archive :: float(),
    generation :: generation(),
    archive_size_after :: non_neg_integer()
}).

-record(opponent_retired, {
    meta :: #lc_event_meta{},
    opponent_id :: individual_id(),
    archive_id :: binary(),
    retirement_reason :: age | performance | diversity,
    games_played :: non_neg_integer(),
    final_elo :: float()
}).

%%% ============================================================================
%%% Match Events
%%% ============================================================================

-record(match_completed, {
    meta :: #lc_event_meta{},
    match_id :: binary(),
    player_a_id :: individual_id(),
    player_b_id :: individual_id(),
    winner_id :: individual_id() | draw,
    player_a_elo_before :: float(),
    player_b_elo_before :: float(),
    player_a_elo_after :: float(),
    player_b_elo_after :: float(),
    match_duration_ms :: non_neg_integer()
}).

-record(elo_updated, {
    meta :: #lc_event_meta{},
    individual_id :: individual_id(),
    elo_before :: float(),
    elo_after :: float(),
    k_factor :: float(),
    match_result :: win | loss | draw
}).

%%% ============================================================================
%%% Strategy Events
%%% ============================================================================

-record(strategy_countered, {
    meta :: #lc_event_meta{},
    counter_id :: individual_id(),
    countered_id :: individual_id(),
    win_rate_against :: float(),
    strategy_signature :: binary(),
    generation :: generation()
}).

-record(arms_race_detected, {
    meta :: #lc_event_meta{},
    population_id :: population_id(),
    cycle_length :: non_neg_integer(),
    participants :: [individual_id()],
    intensity :: float(),
    generation :: generation()
}).

%%% ============================================================================
%%% Matchmaking Events
%%% ============================================================================

-record(matchmaking_adjusted, {
    meta :: #lc_event_meta{},
    population_id :: population_id(),
    elo_range_before :: float(),
    elo_range_after :: float(),
    self_play_ratio_before :: float(),
    self_play_ratio_after :: float(),
    adjustment_reason :: diversity | exploitation | exploration
}).

-record(dominance_matrix_updated, {
    meta :: #lc_event_meta{},
    population_id :: population_id(),
    matrix_size :: non_neg_integer(),
    strongest_counter :: {individual_id(), individual_id(), float()},
    cycle_detected :: boolean(),
    update_generation :: generation()
}).

%%% ============================================================================
%%% Type Exports
%%% ============================================================================

-type competitive_event() :: #opponent_archived{} | #opponent_retired{} |
                             #match_completed{} | #elo_updated{} |
                             #strategy_countered{} | #arms_race_detected{} |
                             #matchmaking_adjusted{} | #dominance_matrix_updated{}.

-endif. %% LC_COMPETITIVE_EVENTS_HRL
