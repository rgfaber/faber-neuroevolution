%%%-------------------------------------------------------------------
%%% @doc Behavioral event definitions for Ecological Silo (τ=50).
%%%
%%% Events related to niches, environmental stress, and carrying capacity.
%%%
%%% @end
%%%-------------------------------------------------------------------
-ifndef(LC_ECOLOGICAL_EVENTS_HRL).
-define(LC_ECOLOGICAL_EVENTS_HRL, true).

-include("lc_events_common.hrl").

%%% ============================================================================
%%% Niche Events
%%% ============================================================================

-record(niche_occupied, {
    meta :: #lc_event_meta{},
    niche_id :: binary(),
    occupant_id :: individual_id(),
    population_id :: population_id(),
    niche_fitness_range :: {float(), float()},
    competition_level :: float()
}).

-record(niche_vacated, {
    meta :: #lc_event_meta{},
    niche_id :: binary(),
    vacating_id :: individual_id() | undefined,
    population_id :: population_id(),
    vacation_reason :: extinction | migration | outcompeted
}).

%%% ============================================================================
%%% Stress Events
%%% ============================================================================

-record(stress_applied, {
    meta :: #lc_event_meta{},
    population_id :: population_id(),
    stress_type :: resource | environmental | competitive | random,
    stress_intensity :: float(),
    affected_individuals :: non_neg_integer(),
    expected_duration :: non_neg_integer() | indefinite
}).

-record(stress_relieved, {
    meta :: #lc_event_meta{},
    population_id :: population_id(),
    stress_type :: resource | environmental | competitive | random,
    previous_intensity :: float(),
    relief_cause :: adaptation | environment_change | intervention
}).

%%% ============================================================================
%%% Capacity Events
%%% ============================================================================

-record(carrying_capacity_changed, {
    meta :: #lc_event_meta{},
    population_id :: population_id(),
    capacity_before :: non_neg_integer(),
    capacity_after :: non_neg_integer(),
    change_cause :: resource_availability | environmental | policy
}).

%%% ============================================================================
%%% Resource Events
%%% ============================================================================

-record(resource_scarcity_detected, {
    meta :: #lc_event_meta{},
    population_id :: population_id(),
    resource_type :: compute | memory | evaluation_time,
    availability :: float(),
    threshold :: float(),
    affected_ratio :: float()
}).

-record(resource_abundance_detected, {
    meta :: #lc_event_meta{},
    population_id :: population_id(),
    resource_type :: compute | memory | evaluation_time,
    availability :: float(),
    surplus_ratio :: float()
}).

%%% ============================================================================
%%% Risk Events
%%% ============================================================================

-record(extinction_risk_elevated, {
    meta :: #lc_event_meta{},
    species_id :: species_id(),
    population_id :: population_id(),
    risk_level :: float(),
    risk_factors :: [atom()],
    population_size :: non_neg_integer()
}).

-record(ecosystem_disrupted, {
    meta :: #lc_event_meta{},
    population_id :: population_id(),
    disruption_type :: catastrophe | invasion | collapse,
    disruption_severity :: float(),
    species_affected :: non_neg_integer(),
    recovery_estimate_generations :: non_neg_integer()
}).

%%% ============================================================================
%%% Type Exports
%%% ============================================================================

-type ecological_event() :: #niche_occupied{} | #niche_vacated{} |
                            #stress_applied{} | #stress_relieved{} |
                            #carrying_capacity_changed{} |
                            #resource_scarcity_detected{} | #resource_abundance_detected{} |
                            #extinction_risk_elevated{} | #ecosystem_disrupted{}.

-endif. %% LC_ECOLOGICAL_EVENTS_HRL
