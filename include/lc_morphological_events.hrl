%%%-------------------------------------------------------------------
%%% @doc Behavioral event definitions for Morphological Silo (τ=30).
%%%
%%% Events related to network complexity, structure, and efficiency.
%%%
%%% @end
%%%-------------------------------------------------------------------
-ifndef(LC_MORPHOLOGICAL_EVENTS_HRL).
-define(LC_MORPHOLOGICAL_EVENTS_HRL, true).

-include("lc_events_common.hrl").

%%% ============================================================================
%%% Complexity Events
%%% ============================================================================

-record(complexity_measured, {
    meta :: #lc_event_meta{},
    individual_id :: individual_id(),
    neuron_count :: non_neg_integer(),
    connection_count :: non_neg_integer(),
    parameter_count :: non_neg_integer(),
    complexity_score :: float()
}).

-record(pruning_triggered, {
    meta :: #lc_event_meta{},
    individual_id :: individual_id(),
    neurons_removed :: non_neg_integer(),
    connections_removed :: non_neg_integer(),
    complexity_before :: float(),
    complexity_after :: float(),
    fitness_impact :: float(),
    pruning_criterion :: unused | weak | redundant
}).

%%% ============================================================================
%%% Growth Events
%%% ============================================================================

-record(growth_occurred, {
    meta :: #lc_event_meta{},
    individual_id :: individual_id(),
    neurons_added :: non_neg_integer(),
    connections_added :: non_neg_integer(),
    growth_trigger :: mutation | development | learning,
    complexity_before :: float(),
    complexity_after :: float()
}).

-record(modularity_detected, {
    meta :: #lc_event_meta{},
    individual_id :: individual_id(),
    module_count :: non_neg_integer(),
    modularity_score :: float(),
    module_sizes :: [non_neg_integer()],
    inter_module_connectivity :: float()
}).

%%% ============================================================================
%%% Efficiency Events
%%% ============================================================================

-record(efficiency_improved, {
    meta :: #lc_event_meta{},
    individual_id :: individual_id(),
    fitness_before :: float(),
    fitness_after :: float(),
    complexity_before :: float(),
    complexity_after :: float(),
    efficiency_ratio_before :: float(),
    efficiency_ratio_after :: float()
}).

-record(bloat_detected, {
    meta :: #lc_event_meta{},
    individual_id :: individual_id(),
    complexity :: float(),
    fitness :: float(),
    expected_complexity :: float(),
    bloat_ratio :: float()
}).

%%% ============================================================================
%%% Structural Events
%%% ============================================================================

-record(symmetry_broken, {
    meta :: #lc_event_meta{},
    individual_id :: individual_id(),
    symmetry_before :: float(),
    symmetry_after :: float(),
    breaking_cause :: mutation | pruning | growth
}).

-record(topology_milestone, {
    meta :: #lc_event_meta{},
    individual_id :: individual_id(),
    milestone_type :: first_hidden | multi_layer | recurrent | modular,
    neuron_count :: non_neg_integer(),
    connection_count :: non_neg_integer(),
    generation :: generation()
}).

%%% ============================================================================
%%% Type Exports
%%% ============================================================================

-type morphological_event() :: #complexity_measured{} | #pruning_triggered{} |
                               #growth_occurred{} | #modularity_detected{} |
                               #efficiency_improved{} | #bloat_detected{} |
                               #symmetry_broken{} | #topology_milestone{}.

-endif. %% LC_MORPHOLOGICAL_EVENTS_HRL
