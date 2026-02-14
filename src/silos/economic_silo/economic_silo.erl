%% @doc Economic Silo - Compute budget and resource allocation for neuroevolution.
%%
%% Part of the Liquid Conglomerate v2 architecture. The Economic Silo manages:
%%   Per-individual compute budgets
%%   Energy accounting (income/expenditure)
%%   Wealth distribution and Gini coefficient tracking
%%   Trade between individuals/species
%%   Bankruptcy and debt management
%%
%% == Time Constant ==
%%
%% τ = 20 (medium-fast adaptation for responsive budget management)
%%
%% == Cross-Silo Signals ==
%%
%% Outgoing:
%%   economic_pressure to task: Budget constraint severity
%%   budget_available to temporal: Available computation budget
%%   efficiency_requirement to morphological: Efficiency targets
%%   trade_opportunity to social: Trading possibilities
%%
%% Incoming:
%%   episode_efficiency from temporal: Episode cost efficiency
%%   complexity_signal from morphological: Network complexity
%%   trust_network from social: Trust for trades
%%   budget_signal from resource: Available system resources
%%
%% @author R.G. Lefever
%% @copyright 2024-2026 R.G. Lefever
-module(economic_silo).
-behaviour(gen_server).
-behaviour(lc_silo_behavior).

-include("lc_silos.hrl").
-include("lc_signals.hrl").

%% API
-export([
    start_link/0,
    start_link/1,
    get_params/1,
    allocate_budget/2,
    record_expenditure/3,
    record_income/3,
    get_balance/2,
    get_wealth_distribution/1,
    get_state/1,
    reset/1
]).

%% gen_server callbacks
-export([
    init/1,
    handle_call/3,
    handle_cast/2,
    handle_info/2,
    terminate/2
]).

%% lc_silo_behavior callbacks
-export([
    init_silo/1,
    collect_sensors/1,
    apply_actuators/2,
    compute_reward/1,
    get_silo_type/0,
    get_time_constant/0,
    handle_cross_silo_signals/2,
    emit_cross_silo_signals/1
]).

-define(SERVER, ?MODULE).
-define(TIME_CONSTANT, 20.0).
-define(HISTORY_SIZE, 100).

%% Default actuator values
-define(DEFAULT_PARAMS, #{
    compute_allocation_strategy => 0.5,  %% 0 = equal, 1 = fitness-proportional
    budget_per_individual => 1.0,
    energy_tax_rate => 0.1,
    wealth_redistribution_rate => 0.1,
    trade_incentive => 0.2,
    bankruptcy_threshold => 0.05,
    investment_horizon => 5,
    resource_discovery_bonus => 0.1,
    inflation_control => 0.02,
    debt_penalty => 0.05
}).

%% Actuator bounds
-define(ACTUATOR_BOUNDS, #{
    compute_allocation_strategy => {0.0, 1.0},
    budget_per_individual => {0.1, 10.0},
    energy_tax_rate => {0.0, 0.3},
    wealth_redistribution_rate => {0.0, 0.5},
    trade_incentive => {0.0, 0.5},
    bankruptcy_threshold => {0.0, 0.3},
    investment_horizon => {1, 20},
    resource_discovery_bonus => {0.0, 0.5},
    inflation_control => {0.0, 0.1},
    debt_penalty => {0.0, 0.2}
}).

-record(state, {
    %% Core silo state
    realm :: binary(),
    enabled_levels :: [l0 | l1 | l2],
    l0_tweann_enabled :: boolean(),
    l2_enabled :: boolean(),

    %% Current parameters (actuator outputs)
    current_params :: map(),

    %% ETS tables
    ets_tables :: #{atom() => ets:tid()},

    %% Aggregate tracking
    total_budget :: float(),
    total_spent :: float(),
    total_income :: float(),
    transaction_count :: non_neg_integer(),

    %% Market state
    market_price_history :: [float()],
    trade_volume_history :: [float()],

    %% Cross-silo signal cache
    incoming_signals :: map(),

    %% Previous values for smoothing
    prev_economic_pressure :: float(),
    prev_budget_available :: float()
}).

%%% ============================================================================
%%% API Functions
%%% ============================================================================

-spec start_link() -> {ok, pid()} | ignore | {error, term()}.
start_link() ->
    start_link(#{}).

-spec start_link(map()) -> {ok, pid()} | ignore | {error, term()}.
start_link(Config) ->
    gen_server:start_link({local, ?SERVER}, ?MODULE, Config, []).

-spec get_params(pid()) -> map().
get_params(Pid) ->
    gen_server:call(Pid, get_params).

-spec allocate_budget(pid(), term()) -> {ok, float()} | {error, term()}.
allocate_budget(Pid, IndividualId) ->
    gen_server:call(Pid, {allocate_budget, IndividualId}).

-spec record_expenditure(pid(), term(), float()) -> ok.
record_expenditure(Pid, IndividualId, Amount) ->
    gen_server:cast(Pid, {record_expenditure, IndividualId, Amount}).

-spec record_income(pid(), term(), float()) -> ok.
record_income(Pid, IndividualId, Amount) ->
    gen_server:cast(Pid, {record_income, IndividualId, Amount}).

-spec get_balance(pid(), term()) -> {ok, float()} | not_found.
get_balance(Pid, IndividualId) ->
    gen_server:call(Pid, {get_balance, IndividualId}).

-spec get_wealth_distribution(pid()) -> {float(), float()}.
get_wealth_distribution(Pid) ->
    gen_server:call(Pid, get_wealth_distribution).

-spec get_state(pid()) -> map().
get_state(Pid) ->
    gen_server:call(Pid, get_state).

-spec reset(pid()) -> ok.
reset(Pid) ->
    gen_server:call(Pid, reset).

%%% ============================================================================
%%% lc_silo_behavior Callbacks
%%% ============================================================================

get_silo_type() -> economic.

get_time_constant() -> ?TIME_CONSTANT.

init_silo(Config) ->
    Realm = maps:get(realm, Config, <<"default">>),
    EtsTables = lc_ets_utils:create_tables(economic, Realm, [
        {accounts, [{keypos, 1}]},
        {transactions, [{keypos, 1}]},
        {market_history, [{keypos, 1}]}
    ]),
    {ok, #{
        ets_tables => EtsTables,
        realm => Realm
    }}.

collect_sensors(State) ->
    #state{
        total_budget = TotalBudget,
        total_spent = TotalSpent,
        total_income = TotalIncome,
        ets_tables = EtsTables,
        market_price_history = PriceHistory,
        trade_volume_history = VolumeHistory,
        incoming_signals = InSignals
    } = State,

    %% Budget metrics
    BudgetRemaining = safe_ratio(TotalBudget - TotalSpent, TotalBudget),
    BudgetTrend = compute_budget_trend(State),

    %% Energy metrics from accounts
    AccountsTable = maps:get(accounts, EtsTables),
    {EnergyMean, Gini} = compute_account_stats(AccountsTable),
    IncomeRate = safe_ratio(TotalIncome, TotalBudget),
    ExpenditureRate = safe_ratio(TotalSpent, TotalBudget),

    %% Trade metrics
    TradeVolume = compute_recent_volume(VolumeHistory),
    MarketPrice = compute_market_price(PriceHistory),
    TradeBalance = compute_trade_balance(State),

    %% Debt tracking
    DebtLevel = compute_debt_level(AccountsTable),

    %% Efficiency metrics
    FitnessPerCost = compute_fitness_per_cost(State),
    ScarcityIndex = compute_scarcity(State),

    %% Cross-silo signals as sensors
    EpisodeEfficiency = maps:get(episode_efficiency, InSignals, 0.5),
    ComplexitySignal = maps:get(complexity_signal, InSignals, 0.5),

    #{
        budget_remaining => BudgetRemaining,
        budget_trend => BudgetTrend,
        energy_level_mean => EnergyMean,
        energy_income_rate => IncomeRate,
        energy_expenditure_rate => ExpenditureRate,
        trade_volume => TradeVolume,
        market_price_fitness => MarketPrice,
        trade_balance => TradeBalance,
        wealth_gini => Gini,
        debt_level => DebtLevel,
        fitness_per_cost => FitnessPerCost,
        scarcity_index => ScarcityIndex,
        %% External signals
        episode_efficiency => EpisodeEfficiency,
        complexity_signal => ComplexitySignal
    }.

apply_actuators(Actuators, State) ->
    BoundedParams = apply_bounds(Actuators, ?ACTUATOR_BOUNDS),
    NewState = State#state{current_params = BoundedParams},
    emit_cross_silo_signals(NewState),
    {ok, NewState}.

compute_reward(State) ->
    Sensors = collect_sensors(State),

    %% Reward components:
    %% 1. Healthy budget remaining (not too low, not wasteful)
    BudgetHealth = 1.0 - abs(maps:get(budget_remaining, Sensors, 0.5) - 0.5) * 2,

    %% 2. Low Gini coefficient (more equal distribution)
    EqualityBonus = 1.0 - maps:get(wealth_gini, Sensors, 0.5),

    %% 3. Low debt levels
    DebtPenalty = maps:get(debt_level, Sensors, 0.0),

    %% 4. Good fitness per cost efficiency
    Efficiency = maps:get(fitness_per_cost, Sensors, 0.5),

    %% 5. Active trade (some trading is healthy)
    TradeActivity = lc_silo_behavior:normalize(
        maps:get(trade_volume, Sensors, 0.0), 0.0, 0.5
    ),

    %% Combined reward
    Reward = 0.25 * BudgetHealth +
             0.2 * EqualityBonus +
             0.2 * (1.0 - DebtPenalty) +
             0.25 * Efficiency +
             0.1 * TradeActivity,

    lc_silo_behavior:clamp(Reward, 0.0, 1.0).

handle_cross_silo_signals(Signals, State) ->
    CurrentSignals = State#state.incoming_signals,
    UpdatedSignals = maps:merge(CurrentSignals, Signals),
    {ok, State#state{incoming_signals = UpdatedSignals}}.

emit_cross_silo_signals(State) ->
    Sensors = collect_sensors(State),

    %% Economic pressure: high when budget is low or debt is high
    EconomicPressure = compute_economic_pressure(Sensors),

    %% Budget available: normalized remaining budget
    BudgetAvailable = maps:get(budget_remaining, Sensors, 0.5),

    %% Efficiency requirement: based on scarcity
    EfficiencyRequirement = maps:get(scarcity_index, Sensors, 0.5),

    %% Trade opportunity: when trade volume is low but resources exist
    TradeOpportunity = compute_trade_opportunity(Sensors),

    %% Emit signals
    emit_signal(task, economic_pressure, EconomicPressure),
    emit_signal(temporal, budget_available, BudgetAvailable),
    emit_signal(morphological, efficiency_requirement, EfficiencyRequirement),
    emit_signal(social, trade_opportunity, TradeOpportunity),
    ok.

%%% ============================================================================
%%% gen_server Callbacks
%%% ============================================================================

init(Config) ->
    Realm = maps:get(realm, Config, <<"default">>),
    EnabledLevels = maps:get(enabled_levels, Config, [l0, l1]),
    L0TweannEnabled = maps:get(l0_tweann_enabled, Config, false),
    L2Enabled = maps:get(l2_enabled, Config, false),
    InitialBudget = maps:get(initial_budget, Config, 1000.0),

    %% Create ETS tables
    EtsTables = lc_ets_utils:create_tables(economic, Realm, [
        {accounts, [{keypos, 1}]},
        {transactions, [{keypos, 1}]},
        {market_history, [{keypos, 1}]}
    ]),

    State = #state{
        realm = Realm,
        enabled_levels = EnabledLevels,
        l0_tweann_enabled = L0TweannEnabled,
        l2_enabled = L2Enabled,
        current_params = ?DEFAULT_PARAMS,
        ets_tables = EtsTables,
        total_budget = InitialBudget,
        total_spent = 0.0,
        total_income = 0.0,
        transaction_count = 0,
        market_price_history = [],
        trade_volume_history = [],
        incoming_signals = #{},
        prev_economic_pressure = 0.0,
        prev_budget_available = 1.0
    },

    %% Schedule periodic cross-silo signal update
    erlang:send_after(1000, self(), update_signals),

    {ok, State}.

handle_call(get_params, _From, State) ->
    {reply, State#state.current_params, State};

handle_call({allocate_budget, IndividualId}, _From, State) ->
    Params = State#state.current_params,
    BudgetPerIndividual = maps:get(budget_per_individual, Params, 1.0),

    %% Create or update account
    AccountsTable = maps:get(accounts, State#state.ets_tables),
    Account = get_or_create_account(AccountsTable, IndividualId, BudgetPerIndividual),

    %% Update total budget tracking
    NewTotalBudget = State#state.total_budget + BudgetPerIndividual,

    NewState = State#state{total_budget = NewTotalBudget},
    {reply, {ok, Account}, NewState};

handle_call({get_balance, IndividualId}, _From, State) ->
    AccountsTable = maps:get(accounts, State#state.ets_tables),
    Result = lc_ets_utils:lookup(AccountsTable, IndividualId),
    Reply = extract_balance(Result),
    {reply, Reply, State};

handle_call(get_wealth_distribution, _From, State) ->
    AccountsTable = maps:get(accounts, State#state.ets_tables),
    {Mean, Gini} = compute_account_stats(AccountsTable),
    {reply, {Mean, Gini}, State};

handle_call(get_state, _From, State) ->
    StateMap = #{
        realm => State#state.realm,
        enabled_levels => State#state.enabled_levels,
        current_params => State#state.current_params,
        total_budget => State#state.total_budget,
        total_spent => State#state.total_spent,
        total_income => State#state.total_income,
        transaction_count => State#state.transaction_count,
        sensors => collect_sensors(State)
    },
    {reply, StateMap, State};

handle_call(reset, _From, State) ->
    %% Clear ETS tables
    AccountsTable = maps:get(accounts, State#state.ets_tables),
    TransactionsTable = maps:get(transactions, State#state.ets_tables),
    MarketTable = maps:get(market_history, State#state.ets_tables),
    ets:delete_all_objects(AccountsTable),
    ets:delete_all_objects(TransactionsTable),
    ets:delete_all_objects(MarketTable),

    NewState = State#state{
        current_params = ?DEFAULT_PARAMS,
        total_budget = 1000.0,
        total_spent = 0.0,
        total_income = 0.0,
        transaction_count = 0,
        market_price_history = [],
        trade_volume_history = [],
        incoming_signals = #{},
        prev_economic_pressure = 0.0,
        prev_budget_available = 1.0
    },
    {reply, ok, NewState};

handle_call(_Request, _From, State) ->
    {reply, {error, unknown_request}, State}.

handle_cast({record_expenditure, IndividualId, Amount}, State) ->
    AccountsTable = maps:get(accounts, State#state.ets_tables),
    TransactionsTable = maps:get(transactions, State#state.ets_tables),

    %% Update account balance
    update_account_balance(AccountsTable, IndividualId, -Amount),

    %% Record transaction
    TxId = State#state.transaction_count + 1,
    lc_ets_utils:insert(TransactionsTable, TxId, #{
        from => IndividualId,
        to => system,
        amount => Amount,
        type => expenditure
    }),

    NewState = State#state{
        total_spent = State#state.total_spent + Amount,
        transaction_count = TxId
    },
    {noreply, NewState};

handle_cast({record_income, IndividualId, Amount}, State) ->
    AccountsTable = maps:get(accounts, State#state.ets_tables),
    TransactionsTable = maps:get(transactions, State#state.ets_tables),

    %% Update account balance
    update_account_balance(AccountsTable, IndividualId, Amount),

    %% Record transaction
    TxId = State#state.transaction_count + 1,
    lc_ets_utils:insert(TransactionsTable, TxId, #{
        from => system,
        to => IndividualId,
        amount => Amount,
        type => income
    }),

    NewState = State#state{
        total_income = State#state.total_income + Amount,
        transaction_count = TxId
    },
    {noreply, NewState};

handle_cast(_Msg, State) ->
    {noreply, State}.

handle_info(update_signals, State) ->
    %% Fetch incoming signals from cross-silo coordinator
    NewSignals = fetch_incoming_signals(),
    UpdatedState = State#state{
        incoming_signals = maps:merge(State#state.incoming_signals, NewSignals)
    },

    %% Emit outgoing signals
    emit_cross_silo_signals(UpdatedState),

    %% Reschedule
    erlang:send_after(1000, self(), update_signals),
    {noreply, UpdatedState};

handle_info(_Info, State) ->
    {noreply, State}.

terminate(_Reason, State) ->
    lc_ets_utils:delete_tables(State#state.ets_tables),
    ok.

%%% ============================================================================
%%% Internal Functions - Account Management
%%% ============================================================================

get_or_create_account(Table, IndividualId, InitialBalance) ->
    case lc_ets_utils:lookup(Table, IndividualId) of
        {ok, Account} ->
            maps:get(balance, Account, InitialBalance);
        not_found ->
            lc_ets_utils:insert(Table, IndividualId, #{
                balance => InitialBalance,
                income => 0.0,
                expenditure => 0.0
            }),
            InitialBalance
    end.

update_account_balance(Table, IndividualId, Delta) ->
    lc_ets_utils:update(Table, IndividualId, fun(undefined) ->
        #{balance => max(0.0, Delta), income => 0.0, expenditure => 0.0};
    (Account) ->
        OldBalance = maps:get(balance, Account, 0.0),
        NewBalance = max(0.0, OldBalance + Delta),
        update_income_expenditure(Account, Delta, NewBalance)
    end).

update_income_expenditure(Account, Delta, NewBalance) when Delta > 0 ->
    OldIncome = maps:get(income, Account, 0.0),
    Account#{balance => NewBalance, income => OldIncome + Delta};
update_income_expenditure(Account, Delta, NewBalance) ->
    OldExpenditure = maps:get(expenditure, Account, 0.0),
    Account#{balance => NewBalance, expenditure => OldExpenditure - Delta}.

extract_balance({ok, Account}) -> {ok, maps:get(balance, Account, 0.0)};
extract_balance(not_found) -> not_found.

%%% ============================================================================
%%% Internal Functions - Statistics
%%% ============================================================================

compute_account_stats(Table) ->
    Balances = get_all_balances(Table),
    compute_mean_and_gini(Balances).

get_all_balances(Table) ->
    lc_ets_utils:fold(
        fun({_Id, Account, _Ts}, Acc) ->
            [maps:get(balance, Account, 0.0) | Acc]
        end,
        [],
        Table
    ).

compute_mean_and_gini([]) ->
    {0.5, 0.0};
compute_mean_and_gini(Balances) ->
    Mean = lists:sum(Balances) / length(Balances),
    NormMean = lc_silo_behavior:normalize(Mean, 0.0, 10.0),
    Gini = compute_gini(Balances),
    {NormMean, Gini}.

compute_gini([]) -> 0.0;
compute_gini([_]) -> 0.0;
compute_gini(Values) ->
    Sorted = lists:sort(Values),
    N = length(Sorted),
    Total = lists:sum(Sorted),
    compute_gini_sum(Sorted, N, Total).

compute_gini_sum(_Sorted, _N, Total) when Total == 0.0 -> 0.0;
compute_gini_sum(Sorted, N, Total) ->
    %% Gini = (2 * sum(i * x_i)) / (n * sum(x_i)) - (n + 1) / n
    IndexedSum = lists:foldl(
        fun({I, X}, Acc) -> Acc + I * X end,
        0.0,
        lists:zip(lists:seq(1, N), Sorted)
    ),
    Gini = (2 * IndexedSum) / (N * Total) - (N + 1) / N,
    lc_silo_behavior:clamp(Gini, 0.0, 1.0).

compute_budget_trend(State) ->
    %% Simple trend: (income - expenditure) / budget
    Income = State#state.total_income,
    Spent = State#state.total_spent,
    Budget = State#state.total_budget,
    Trend = safe_ratio(Income - Spent, Budget),
    lc_silo_behavior:normalize(Trend, -1.0, 1.0).

compute_recent_volume([]) -> 0.0;
compute_recent_volume(History) ->
    Recent = lists:sublist(History, 10),
    Mean = lists:sum(Recent) / length(Recent),
    lc_silo_behavior:normalize(Mean, 0.0, 100.0).

compute_market_price([]) -> 0.5;
compute_market_price([Latest | _]) ->
    lc_silo_behavior:normalize(Latest, 0.0, 10.0).

compute_trade_balance(State) ->
    %% Balance of imports vs exports (simplified)
    Income = State#state.total_income,
    Spent = State#state.total_spent,
    safe_ratio(Income - Spent, Income + Spent + 0.001).

compute_debt_level(Table) ->
    Balances = get_all_balances(Table),
    compute_debt_ratio(Balances).

compute_debt_ratio([]) -> 0.0;
compute_debt_ratio(Balances) ->
    NegativeBalances = [B || B <- Balances, B < 0],
    TotalDebt = abs(lists:sum(NegativeBalances)),
    TotalWealth = lists:sum([max(0, B) || B <- Balances]),
    safe_ratio(TotalDebt, TotalWealth + TotalDebt + 0.001).

compute_fitness_per_cost(State) ->
    %% Simplified: inverse of expenditure rate
    ExpenditureRate = safe_ratio(State#state.total_spent, State#state.total_budget),
    1.0 - lc_silo_behavior:clamp(ExpenditureRate, 0.0, 1.0).

compute_scarcity(State) ->
    %% Scarcity = 1 - budget_remaining
    BudgetRemaining = safe_ratio(
        State#state.total_budget - State#state.total_spent,
        State#state.total_budget
    ),
    1.0 - BudgetRemaining.

safe_ratio(_Num, Denom) when Denom == 0.0; Denom == 0 -> 0.0;
safe_ratio(Num, Denom) -> Num / Denom.

%%% ============================================================================
%%% Internal Functions - Cross-Silo
%%% ============================================================================

compute_economic_pressure(Sensors) ->
    BudgetRemaining = maps:get(budget_remaining, Sensors, 0.5),
    DebtLevel = maps:get(debt_level, Sensors, 0.0),
    Scarcity = maps:get(scarcity_index, Sensors, 0.5),

    %% High pressure when low budget, high debt, or high scarcity
    0.4 * (1.0 - BudgetRemaining) + 0.3 * DebtLevel + 0.3 * Scarcity.

compute_trade_opportunity(Sensors) ->
    TradeVolume = maps:get(trade_volume, Sensors, 0.0),
    BudgetRemaining = maps:get(budget_remaining, Sensors, 0.5),

    %% Opportunity when volume is low but budget exists
    (1.0 - TradeVolume) * BudgetRemaining.

emit_signal(_ToSilo, SignalName, Value) ->
    %% Event-driven: publish signal, lc_cross_silo routes to valid destinations
    silo_events:publish_signal(economic, SignalName, Value).

fetch_incoming_signals() ->
    case whereis(lc_cross_silo) of
        undefined -> #{};
        _Pid -> lc_cross_silo:get_signals_for(economic)
    end.

%%% ============================================================================
%%% Internal Functions - Bounds
%%% ============================================================================

apply_bounds(Params, Bounds) ->
    maps:fold(
        fun(Key, Value, Acc) ->
            BoundedValue = apply_single_bound(Key, Value, Bounds),
            maps:put(Key, BoundedValue, Acc)
        end,
        #{},
        Params
    ).

apply_single_bound(Key, Value, Bounds) ->
    case maps:get(Key, Bounds, undefined) of
        undefined -> Value;
        {Min, Max} -> lc_silo_behavior:clamp(Value, Min, Max)
    end.
