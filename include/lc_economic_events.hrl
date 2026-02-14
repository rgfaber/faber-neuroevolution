%%%-------------------------------------------------------------------
%%% @doc Behavioral event definitions for Economic Silo (τ=20).
%%%
%%% Events related to compute budgets, resource economics, and wealth.
%%%
%%% @end
%%%-------------------------------------------------------------------
-ifndef(LC_ECONOMIC_EVENTS_HRL).
-define(LC_ECONOMIC_EVENTS_HRL, true).

-include("lc_events_common.hrl").

%%% ============================================================================
%%% Budget Events
%%% ============================================================================

-record(budget_allocated, {
    meta :: #lc_event_meta{},
    individual_id :: individual_id(),
    population_id :: population_id(),
    amount :: float(),
    allocation_type :: initial | bonus | redistribution
}).

-record(budget_exhausted, {
    meta :: #lc_event_meta{},
    individual_id :: individual_id(),
    population_id :: population_id(),
    initial_budget :: float(),
    expenditure_history :: [float()],
    exhaustion_cause :: evaluation | mutation | reproduction
}).

%%% ============================================================================
%%% Trade Events
%%% ============================================================================

-record(compute_traded, {
    meta :: #lc_event_meta{},
    seller_id :: individual_id(),
    buyer_id :: individual_id(),
    amount :: float(),
    price :: float(),
    trade_reason :: scarcity | opportunity | altruism
}).

-record(wealth_redistributed, {
    meta :: #lc_event_meta{},
    population_id :: population_id(),
    gini_before :: float(),
    gini_after :: float(),
    redistribution_amount :: float(),
    beneficiaries_count :: non_neg_integer(),
    redistribution_method :: progressive_tax | universal_basic | merit_bonus
}).

%%% ============================================================================
%%% Financial Events
%%% ============================================================================

-record(bankruptcy_declared, {
    meta :: #lc_event_meta{},
    individual_id :: individual_id(),
    population_id :: population_id(),
    final_balance :: float(),
    debt_amount :: float(),
    consequence :: removal | bailout | restructure
}).

-record(investment_made, {
    meta :: #lc_event_meta{},
    investor_id :: individual_id(),
    target_id :: individual_id() | population_id(),
    amount :: float(),
    expected_return :: float(),
    investment_type :: offspring | coalition | exploration
}).

-record(dividend_distributed, {
    meta :: #lc_event_meta{},
    investment_id :: binary(),
    investor_id :: individual_id(),
    return_amount :: float(),
    roi_percentage :: float()
}).

-record(inflation_adjusted, {
    meta :: #lc_event_meta{},
    population_id :: population_id(),
    inflation_rate :: float(),
    price_level_before :: float(),
    price_level_after :: float(),
    adjustment_trigger :: scheduled | market_pressure | policy
}).

%%% ============================================================================
%%% Type Exports
%%% ============================================================================

-type economic_event() :: #budget_allocated{} | #budget_exhausted{} |
                          #compute_traded{} | #wealth_redistributed{} |
                          #bankruptcy_declared{} | #investment_made{} |
                          #dividend_distributed{} | #inflation_adjusted{}.

-endif. %% LC_ECONOMIC_EVENTS_HRL
