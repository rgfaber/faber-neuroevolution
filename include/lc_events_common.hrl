%%%-------------------------------------------------------------------
%%% @doc Common event types and metadata for Liquid Conglomerate silos.
%%%
%%% This header provides shared definitions used by all per-silo event
%%% modules. It enables future package separation by keeping common
%%% types in one place.
%%%
%%% @end
%%%-------------------------------------------------------------------
-ifndef(LC_EVENTS_COMMON_HRL).
-define(LC_EVENTS_COMMON_HRL, true).

%%% ============================================================================
%%% Common Types
%%% ============================================================================

-type individual_id() :: binary().
-type species_id() :: binary().
-type population_id() :: binary().
-type generation() :: non_neg_integer().
-type timestamp() :: integer().
-type fitness() :: float().

%%% ============================================================================
%%% Event Metadata (included in all events)
%%% ============================================================================

-record(lc_event_meta, {
    event_id :: binary(),
    correlation_id :: binary() | undefined,
    causation_id :: binary() | undefined,
    timestamp :: timestamp(),
    version = 1 :: pos_integer(),
    emitter :: atom()
}).

-type lc_event_meta() :: #lc_event_meta{}.

%%% ============================================================================
%%% Event Category Type (for routing and filtering)
%%% ============================================================================

-type silo_type() :: temporal | economic | morphological | competitive |
                     social | cultural | developmental | regulatory |
                     ecological | communication | distribution.

-type event_severity() :: info | warning | critical.

%%% ============================================================================
%%% Macros for Event Construction
%%% ============================================================================

-define(LC_EVENT_VERSION, 1).

-endif. %% LC_EVENTS_COMMON_HRL
