%%%-------------------------------------------------------------------
%%% @doc Behavioral event constructors for Economic Silo.
%%%
%%% Events related to compute budgets, resource economics, and wealth.
%%% Economic Silo operates at timescale τ=20.
%%%
%%% @end
%%%-------------------------------------------------------------------
-module(lc_economic_events).

-include("lc_economic_events.hrl").

%% Event constructors
-export([
    budget_allocated/1,
    budget_exhausted/1,
    compute_traded/1,
    wealth_redistributed/1,
    bankruptcy_declared/1,
    investment_made/1,
    dividend_distributed/1,
    inflation_adjusted/1
]).

%% Utility exports
-export([
    make_meta/1,
    make_meta/2,
    event_to_map/1
]).

%%% ============================================================================
%%% Metadata Construction
%%% ============================================================================

-spec make_meta(Emitter :: atom()) -> #lc_event_meta{}.
make_meta(Emitter) ->
    make_meta(Emitter, #{}).

-spec make_meta(Emitter :: atom(), Opts :: map()) -> #lc_event_meta{}.
make_meta(Emitter, Opts) ->
    #lc_event_meta{
        event_id = maps:get(event_id, Opts, generate_id()),
        correlation_id = maps:get(correlation_id, Opts, undefined),
        causation_id = maps:get(causation_id, Opts, undefined),
        timestamp = maps:get(timestamp, Opts, erlang:system_time(microsecond)),
        version = maps:get(version, Opts, ?LC_EVENT_VERSION),
        emitter = Emitter
    }.

%%% ============================================================================
%%% Budget Event Constructors
%%% ============================================================================

-spec budget_allocated(map()) -> #budget_allocated{}.
budget_allocated(Data) ->
    #budget_allocated{
        meta = make_meta(maps:get(emitter, Data, economic_silo), Data),
        individual_id = maps:get(individual_id, Data),
        population_id = maps:get(population_id, Data),
        amount = maps:get(amount, Data),
        allocation_type = maps:get(allocation_type, Data, initial)
    }.

-spec budget_exhausted(map()) -> #budget_exhausted{}.
budget_exhausted(Data) ->
    #budget_exhausted{
        meta = make_meta(maps:get(emitter, Data, economic_silo), Data),
        individual_id = maps:get(individual_id, Data),
        population_id = maps:get(population_id, Data),
        initial_budget = maps:get(initial_budget, Data),
        expenditure_history = maps:get(expenditure_history, Data, []),
        exhaustion_cause = maps:get(exhaustion_cause, Data, evaluation)
    }.

%%% ============================================================================
%%% Trade Event Constructors
%%% ============================================================================

-spec compute_traded(map()) -> #compute_traded{}.
compute_traded(Data) ->
    #compute_traded{
        meta = make_meta(maps:get(emitter, Data, economic_silo), Data),
        seller_id = maps:get(seller_id, Data),
        buyer_id = maps:get(buyer_id, Data),
        amount = maps:get(amount, Data),
        price = maps:get(price, Data),
        trade_reason = maps:get(trade_reason, Data, scarcity)
    }.

-spec wealth_redistributed(map()) -> #wealth_redistributed{}.
wealth_redistributed(Data) ->
    #wealth_redistributed{
        meta = make_meta(maps:get(emitter, Data, economic_silo), Data),
        population_id = maps:get(population_id, Data),
        gini_before = maps:get(gini_before, Data),
        gini_after = maps:get(gini_after, Data),
        redistribution_amount = maps:get(redistribution_amount, Data),
        beneficiaries_count = maps:get(beneficiaries_count, Data),
        redistribution_method = maps:get(redistribution_method, Data, progressive_tax)
    }.

%%% ============================================================================
%%% Financial Event Constructors
%%% ============================================================================

-spec bankruptcy_declared(map()) -> #bankruptcy_declared{}.
bankruptcy_declared(Data) ->
    #bankruptcy_declared{
        meta = make_meta(maps:get(emitter, Data, economic_silo), Data),
        individual_id = maps:get(individual_id, Data),
        population_id = maps:get(population_id, Data),
        final_balance = maps:get(final_balance, Data),
        debt_amount = maps:get(debt_amount, Data, 0.0),
        consequence = maps:get(consequence, Data, removal)
    }.

-spec investment_made(map()) -> #investment_made{}.
investment_made(Data) ->
    #investment_made{
        meta = make_meta(maps:get(emitter, Data, economic_silo), Data),
        investor_id = maps:get(investor_id, Data),
        target_id = maps:get(target_id, Data),
        amount = maps:get(amount, Data),
        expected_return = maps:get(expected_return, Data),
        investment_type = maps:get(investment_type, Data, offspring)
    }.

-spec dividend_distributed(map()) -> #dividend_distributed{}.
dividend_distributed(Data) ->
    #dividend_distributed{
        meta = make_meta(maps:get(emitter, Data, economic_silo), Data),
        investment_id = maps:get(investment_id, Data),
        investor_id = maps:get(investor_id, Data),
        return_amount = maps:get(return_amount, Data),
        roi_percentage = maps:get(roi_percentage, Data)
    }.

-spec inflation_adjusted(map()) -> #inflation_adjusted{}.
inflation_adjusted(Data) ->
    #inflation_adjusted{
        meta = make_meta(maps:get(emitter, Data, economic_silo), Data),
        population_id = maps:get(population_id, Data),
        inflation_rate = maps:get(inflation_rate, Data),
        price_level_before = maps:get(price_level_before, Data),
        price_level_after = maps:get(price_level_after, Data),
        adjustment_trigger = maps:get(adjustment_trigger, Data, scheduled)
    }.

%%% ============================================================================
%%% Utility Functions
%%% ============================================================================

-spec event_to_map(economic_event()) -> map().
event_to_map(Event) when is_tuple(Event) ->
    [RecordName | Fields] = tuple_to_list(Event),
    FieldNames = record_fields(RecordName),
    MetaMap = meta_to_map(hd(Fields)),
    DataMap = maps:from_list(lists:zip(tl(FieldNames), tl(Fields))),
    maps:merge(#{event_type => RecordName, meta => MetaMap}, DataMap).

%%% ============================================================================
%%% Internal Functions
%%% ============================================================================

generate_id() ->
    Bytes = crypto:strong_rand_bytes(16),
    binary:encode_hex(Bytes).

meta_to_map(#lc_event_meta{} = Meta) ->
    #{
        event_id => Meta#lc_event_meta.event_id,
        correlation_id => Meta#lc_event_meta.correlation_id,
        causation_id => Meta#lc_event_meta.causation_id,
        timestamp => Meta#lc_event_meta.timestamp,
        version => Meta#lc_event_meta.version,
        emitter => Meta#lc_event_meta.emitter
    }.

record_fields(budget_allocated) -> [meta, individual_id, population_id, amount, allocation_type];
record_fields(budget_exhausted) -> [meta, individual_id, population_id, initial_budget, expenditure_history, exhaustion_cause];
record_fields(compute_traded) -> [meta, seller_id, buyer_id, amount, price, trade_reason];
record_fields(wealth_redistributed) -> [meta, population_id, gini_before, gini_after, redistribution_amount, beneficiaries_count, redistribution_method];
record_fields(bankruptcy_declared) -> [meta, individual_id, population_id, final_balance, debt_amount, consequence];
record_fields(investment_made) -> [meta, investor_id, target_id, amount, expected_return, investment_type];
record_fields(dividend_distributed) -> [meta, investment_id, investor_id, return_amount, roi_percentage];
record_fields(inflation_adjusted) -> [meta, population_id, inflation_rate, price_level_before, price_level_after, adjustment_trigger].
