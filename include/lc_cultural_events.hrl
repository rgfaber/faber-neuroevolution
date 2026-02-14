%%%-------------------------------------------------------------------
%%% @doc Behavioral event definitions for Cultural Silo (τ=35).
%%%
%%% Events related to behavioral innovations, traditions, and memes.
%%%
%%% @end
%%%-------------------------------------------------------------------
-ifndef(LC_CULTURAL_EVENTS_HRL).
-define(LC_CULTURAL_EVENTS_HRL, true).

-include("lc_events_common.hrl").

%%% ============================================================================
%%% Innovation Events
%%% ============================================================================

-record(innovation_discovered, {
    meta :: #lc_event_meta{},
    innovator_id :: individual_id(),
    innovation_id :: binary(),
    population_id :: population_id(),
    novelty_score :: float(),
    fitness_advantage :: float(),
    innovation_type :: behavioral | structural | strategic
}).

-record(innovation_spread, {
    meta :: #lc_event_meta{},
    innovation_id :: binary(),
    source_id :: individual_id(),
    adopter_id :: individual_id(),
    spread_fidelity :: float(),
    total_adopters :: non_neg_integer()
}).

%%% ============================================================================
%%% Tradition Events
%%% ============================================================================

-record(tradition_established, {
    meta :: #lc_event_meta{},
    tradition_id :: binary(),
    population_id :: population_id(),
    behavior_signature :: binary(),
    initial_practitioners :: non_neg_integer(),
    establishment_threshold :: non_neg_integer()
}).

-record(tradition_abandoned, {
    meta :: #lc_event_meta{},
    tradition_id :: binary(),
    population_id :: population_id(),
    abandonment_reason :: obsolescence | competition | drift,
    final_practitioners :: non_neg_integer(),
    duration_generations :: non_neg_integer()
}).

%%% ============================================================================
%%% Meme Events
%%% ============================================================================

-record(meme_created, {
    meta :: #lc_event_meta{},
    meme_id :: binary(),
    creator_id :: individual_id(),
    population_id :: population_id(),
    meme_encoding :: term(),
    initial_fitness_correlation :: float()
}).

-record(meme_spread, {
    meta :: #lc_event_meta{},
    meme_id :: binary(),
    source_id :: individual_id(),
    target_id :: individual_id(),
    spread_fidelity :: float(),
    fitness_correlation :: float(),
    total_adopters :: non_neg_integer()
}).

-record(meme_mutated, {
    meta :: #lc_event_meta{},
    original_meme_id :: binary(),
    mutated_meme_id :: binary(),
    mutator_id :: individual_id(),
    mutation_magnitude :: float(),
    fitness_change :: float()
}).

%%% ============================================================================
%%% Cultural Dynamics Events
%%% ============================================================================

-record(cultural_convergence, {
    meta :: #lc_event_meta{},
    population_id :: population_id(),
    behavioral_variance_before :: float(),
    behavioral_variance_after :: float(),
    convergence_driver :: selection | imitation | drift
}).

-record(cultural_divergence, {
    meta :: #lc_event_meta{},
    population_id :: population_id(),
    behavioral_variance_before :: float(),
    behavioral_variance_after :: float(),
    divergence_driver :: exploration | speciation | isolation
}).

%%% ============================================================================
%%% Type Exports
%%% ============================================================================

-type cultural_event() :: #innovation_discovered{} | #innovation_spread{} |
                          #tradition_established{} | #tradition_abandoned{} |
                          #meme_created{} | #meme_spread{} | #meme_mutated{} |
                          #cultural_convergence{} | #cultural_divergence{}.

-endif. %% LC_CULTURAL_EVENTS_HRL
