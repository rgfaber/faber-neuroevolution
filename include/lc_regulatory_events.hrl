%%%-------------------------------------------------------------------
%%% @doc Behavioral event definitions for Regulatory Silo (τ=45).
%%%
%%% Events related to gene expression and module activation.
%%%
%%% @end
%%%-------------------------------------------------------------------
-ifndef(LC_REGULATORY_EVENTS_HRL).
-define(LC_REGULATORY_EVENTS_HRL, true).

-include("lc_events_common.hrl").

%%% ============================================================================
%%% Expression Events
%%% ============================================================================

-record(gene_expressed, {
    meta :: #lc_event_meta{},
    individual_id :: individual_id(),
    gene_id :: binary(),
    expression_level :: float(),
    trigger :: environmental | developmental | conditional,
    context :: term()
}).

-record(gene_silenced, {
    meta :: #lc_event_meta{},
    individual_id :: individual_id(),
    gene_id :: binary(),
    silencing_cause :: regulatory | epigenetic | conditional,
    duration :: temporary | permanent
}).

%%% ============================================================================
%%% Module Events
%%% ============================================================================

-record(module_activated, {
    meta :: #lc_event_meta{},
    individual_id :: individual_id(),
    module_id :: binary(),
    activation_context :: term(),
    neurons_involved :: non_neg_integer(),
    switching_cost :: float()
}).

-record(module_deactivated, {
    meta :: #lc_event_meta{},
    individual_id :: individual_id(),
    module_id :: binary(),
    deactivation_reason :: context_change | resource_constraint | completion,
    active_duration_ms :: non_neg_integer()
}).

%%% ============================================================================
%%% Context Events
%%% ============================================================================

-record(context_switched, {
    meta :: #lc_event_meta{},
    individual_id :: individual_id(),
    context_before :: term(),
    context_after :: term(),
    genes_affected :: [binary()],
    modules_toggled :: non_neg_integer()
}).

-record(regulatory_network_updated, {
    meta :: #lc_event_meta{},
    individual_id :: individual_id(),
    regulators_added :: non_neg_integer(),
    regulators_removed :: non_neg_integer(),
    network_complexity :: float()
}).

%%% ============================================================================
%%% Dormancy Events
%%% ============================================================================

-record(dormancy_entered, {
    meta :: #lc_event_meta{},
    individual_id :: individual_id(),
    dormant_capabilities :: [binary()],
    dormancy_trigger :: resource_constraint | environmental | developmental,
    maintenance_cost :: float()
}).

-record(dormancy_exited, {
    meta :: #lc_event_meta{},
    individual_id :: individual_id(),
    reactivated_capabilities :: [binary()],
    reactivation_trigger :: opportunity | stress | development,
    reactivation_delay_ms :: non_neg_integer()
}).

%%% ============================================================================
%%% Type Exports
%%% ============================================================================

-type regulatory_event() :: #gene_expressed{} | #gene_silenced{} |
                            #module_activated{} | #module_deactivated{} |
                            #context_switched{} | #regulatory_network_updated{} |
                            #dormancy_entered{} | #dormancy_exited{}.

-endif. %% LC_REGULATORY_EVENTS_HRL
