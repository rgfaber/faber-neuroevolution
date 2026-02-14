# Plan: Economic Silo for Liquid Conglomerate

**Status:** Planning
**Created:** 2025-12-23
**Last Updated:** 2025-12-23
**Related:** PLAN_SOCIAL_SILO.md, PLAN_CULTURAL_SILO.md, PLAN_ECOLOGICAL_SILO.md, PLAN_MORPHOLOGICAL_SILO.md, PLAN_TEMPORAL_SILO.md, PLAN_COMPETITIVE_SILO.md, PLAN_DEVELOPMENTAL_SILO.md, PLAN_REGULATORY_SILO.md

---

## Overview

The Economic Silo manages resource allocation and trade: computation budgets, energy economics, and inter-individual exchange. This creates efficiency pressure - networks must be cost-effective, not just fit.

---

## 1. Motivation

### Problem Statement

Traditional neuroevolution ignores economic constraints:
- **Unlimited compute**: All individuals get equal evaluation regardless of promise
- **No cost pressure**: Fitness alone drives selection, ignoring efficiency
- **No specialization**: No trade between individuals enables specialization
- **No budget awareness**: Training doesn't respect real-world cost limits

Real-world deployment requires economic thinking:
- Cloud compute costs money
- Energy consumption matters for edge/mobile
- Different networks have different inference costs
- ROI of training must be considered

### Business Value

| Benefit | Impact |
|---------|--------|
| Cloud cost optimization | Evolve within compute budgets |
| Edge deployment | Resource-aware agents |
| Marketplace agents | Agents that can trade effectively |
| Cost modeling | Understand training economics |
| ROI tracking | Investment vs returns |

### Training Velocity Impact

| Metric | Without Economic Silo | With Economic Silo |
|--------|----------------------|-------------------|
| Compute efficiency | 1.0x | 2-4x |
| Cost per fitness | Untracked | Optimized |
| Resource waste | High | Low |
| Deployment readiness | Variable | Cost-aware |

---

## 2. Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        ECONOMIC SILO                                 │
├─────────────────────────────────────────────────────────────────────┤
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │                      L0 SENSORS (12)                         │    │
│  │                                                              │    │
│  │  Budget           Energy         Market       Distribution  │    │
│  │  ┌─────────┐     ┌─────────┐    ┌─────────┐  ┌─────────┐   │    │
│  │  │compute_ │     │energy_  │    │trade_   │  │wealth_  │   │    │
│  │  │budget   │     │level    │    │volume   │  │gini     │   │    │
│  │  │budget_  │     │energy_  │    │market_  │  │debt_    │   │    │
│  │  │trend    │     │income   │    │price    │  │level    │   │    │
│  │  └─────────┘     │energy_  │    │trade_   │  └─────────┘   │    │
│  │                  │expend   │    │balance  │                 │    │
│  │                  └─────────┘    └─────────┘                 │    │
│  │                                                              │    │
│  │  Efficiency       Scarcity                                  │    │
│  │  ┌─────────┐     ┌─────────┐                                │    │
│  │  │fitness_ │     │scarcity_│                                │    │
│  │  │per_cost │     │index    │                                │    │
│  │  └─────────┘     └─────────┘                                │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                                │                                     │
│                       ┌────────▼────────┐                           │
│                       │ TWEANN Controller│                          │
│                       │   (online ES)   │                           │
│                       └────────┬────────┘                           │
│                                │                                     │
│  ┌─────────────────────────────▼───────────────────────────────┐    │
│  │                      L0 ACTUATORS (10)                       │    │
│  │                                                              │    │
│  │  Budget           Tax            Trade        Investment    │    │
│  │  ┌─────────┐     ┌─────────┐    ┌─────────┐  ┌─────────┐   │    │
│  │  │compute_ │     │energy_  │    │trade_   │  │investment│   │    │
│  │  │allocation│    │tax_rate │    │incentive│  │_horizon  │   │    │
│  │  │budget_  │     │wealth_  │    │bankruptcy│ │resource_ │   │    │
│  │  │per_ind  │     │redistrib│    │_threshold│ │discovery_│   │    │
│  │  └─────────┘     └─────────┘    └─────────┘  │bonus     │   │    │
│  │                                              └─────────┘   │    │
│  └─────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 3. L0 Sensors

### 3.1 Sensor Specifications

| ID | Name | Range | Description |
|----|------|-------|-------------|
| 1 | `computation_budget_remaining` | [0.0, 1.0] | Available compute / total budget |
| 2 | `budget_trend` | [-1.0, 1.0] | Direction of budget change |
| 3 | `energy_level` | [0.0, 1.0] | Current energy reserves |
| 4 | `energy_income` | [0.0, 1.0] | Energy gain rate |
| 5 | `energy_expenditure` | [0.0, 1.0] | Energy spend rate |
| 6 | `trade_volume` | [0.0, 1.0] | Amount of trading activity |
| 7 | `market_price_fitness` | [0.0, 1.0] | Cost of fitness improvement |
| 8 | `trade_balance` | [-1.0, 1.0] | Net trade position |
| 9 | `wealth_gini_coefficient` | [0.0, 1.0] | Wealth inequality |
| 10 | `debt_level` | [0.0, 1.0] | Population debt |
| 11 | `fitness_per_cost` | [0.0, 1.0] | Efficiency ratio |
| 12 | `scarcity_index` | [0.0, 1.0] | Resource scarcity level |

### 3.2 Sensor Module

```erlang
%%%-------------------------------------------------------------------
%%% @doc Economic Silo L0 Sensors
%%% Monitors economic dynamics: budgets, energy, trade, and efficiency.
%%% @end
%%%-------------------------------------------------------------------
-module(economic_silo_sensors).

-behaviour(l0_sensor_behaviour).

%% API
-export([sensor_specs/0,
         collect_sensors/1,
         sensor_count/0]).

%% Sensor collection
-export([collect_computation_budget_remaining/1,
         collect_budget_trend/1,
         collect_energy_level/1,
         collect_energy_income/1,
         collect_energy_expenditure/1,
         collect_trade_volume/1,
         collect_market_price_fitness/1,
         collect_trade_balance/1,
         collect_wealth_gini_coefficient/1,
         collect_debt_level/1,
         collect_fitness_per_cost/1,
         collect_scarcity_index/1]).

-include("economic_silo.hrl").

%%====================================================================
%% Behaviour Callbacks
%%====================================================================

-spec sensor_specs() -> [l0_sensor_spec()].
sensor_specs() ->
    [
        #{id => computation_budget_remaining,
          name => <<"Budget Remaining">>,
          range => {0.0, 1.0},
          description => <<"Available compute / total budget">>},

        #{id => budget_trend,
          name => <<"Budget Trend">>,
          range => {-1.0, 1.0},
          description => <<"Direction of budget change">>},

        #{id => energy_level,
          name => <<"Energy Level">>,
          range => {0.0, 1.0},
          description => <<"Current energy reserves">>},

        #{id => energy_income,
          name => <<"Energy Income">>,
          range => {0.0, 1.0},
          description => <<"Energy gain rate">>},

        #{id => energy_expenditure,
          name => <<"Energy Expenditure">>,
          range => {0.0, 1.0},
          description => <<"Energy spend rate">>},

        #{id => trade_volume,
          name => <<"Trade Volume">>,
          range => {0.0, 1.0},
          description => <<"Amount of trading activity">>},

        #{id => market_price_fitness,
          name => <<"Market Price (Fitness)">>,
          range => {0.0, 1.0},
          description => <<"Cost of fitness improvement">>},

        #{id => trade_balance,
          name => <<"Trade Balance">>,
          range => {-1.0, 1.0},
          description => <<"Net trade position">>},

        #{id => wealth_gini_coefficient,
          name => <<"Wealth Gini">>,
          range => {0.0, 1.0},
          description => <<"Wealth inequality">>},

        #{id => debt_level,
          name => <<"Debt Level">>,
          range => {0.0, 1.0},
          description => <<"Population debt">>},

        #{id => fitness_per_cost,
          name => <<"Fitness per Cost">>,
          range => {0.0, 1.0},
          description => <<"Efficiency ratio">>},

        #{id => scarcity_index,
          name => <<"Scarcity Index">>,
          range => {0.0, 1.0},
          description => <<"Resource scarcity level">>}
    ].

-spec sensor_count() -> pos_integer().
sensor_count() -> 12.

-spec collect_sensors(economic_context()) -> sensor_vector().
collect_sensors(Context) ->
    [
        collect_computation_budget_remaining(Context),
        collect_budget_trend(Context),
        collect_energy_level(Context),
        collect_energy_income(Context),
        collect_energy_expenditure(Context),
        collect_trade_volume(Context),
        collect_market_price_fitness(Context),
        collect_trade_balance(Context),
        collect_wealth_gini_coefficient(Context),
        collect_debt_level(Context),
        collect_fitness_per_cost(Context),
        collect_scarcity_index(Context)
    ].

%%====================================================================
%% Individual Sensor Collection
%%====================================================================

%% @doc Remaining computation budget
-spec collect_computation_budget_remaining(economic_context()) -> float().
collect_computation_budget_remaining(#economic_context{
    budget_remaining = Remaining,
    total_budget = Total
}) ->
    case Total of
        0 -> 0.0;
        _ -> clamp(Remaining / Total, 0.0, 1.0)
    end.

%% @doc Trend in budget consumption
-spec collect_budget_trend(economic_context()) -> float().
collect_budget_trend(#economic_context{budget_history = History}) ->
    case length(History) >= 3 of
        false -> 0.0;
        true ->
            Recent = lists:sublist(History, 5),
            calculate_trend(Recent)
    end.

%% @doc Current energy level
-spec collect_energy_level(economic_context()) -> float().
collect_energy_level(#economic_context{
    total_energy = Energy,
    max_energy = MaxEnergy
}) ->
    case MaxEnergy of
        0 -> 0.5;
        _ -> clamp(Energy / MaxEnergy, 0.0, 1.0)
    end.

%% @doc Energy income rate
-spec collect_energy_income(economic_context()) -> float().
collect_energy_income(#economic_context{
    energy_income_rate = Income,
    max_income_rate = MaxIncome
}) ->
    case MaxIncome of
        0 -> 0.0;
        _ -> clamp(Income / MaxIncome, 0.0, 1.0)
    end.

%% @doc Energy expenditure rate
-spec collect_energy_expenditure(economic_context()) -> float().
collect_energy_expenditure(#economic_context{
    energy_expenditure_rate = Expenditure,
    max_expenditure_rate = MaxExp
}) ->
    case MaxExp of
        0 -> 0.0;
        _ -> clamp(Expenditure / MaxExp, 0.0, 1.0)
    end.

%% @doc Trade activity volume
-spec collect_trade_volume(economic_context()) -> float().
collect_trade_volume(#economic_context{
    trade_volume = Volume,
    max_trade_volume = MaxVolume
}) ->
    case MaxVolume of
        0 -> 0.0;
        _ -> clamp(Volume / MaxVolume, 0.0, 1.0)
    end.

%% @doc Market price for fitness improvement
-spec collect_market_price_fitness(economic_context()) -> float().
collect_market_price_fitness(#economic_context{
    cost_per_fitness_point = Cost,
    baseline_cost = Baseline
}) ->
    case Baseline of
        0 -> 0.5;
        _ -> clamp(Cost / (Baseline * 2), 0.0, 1.0)
    end.

%% @doc Trade balance (positive = net exporter)
-spec collect_trade_balance(economic_context()) -> float().
collect_trade_balance(#economic_context{
    exports = Exports,
    imports = Imports
}) ->
    Total = Exports + Imports,
    case Total of
        0 -> 0.0;
        _ ->
            Balance = (Exports - Imports) / Total,
            clamp(Balance, -1.0, 1.0)
    end.

%% @doc Wealth inequality (Gini coefficient)
-spec collect_wealth_gini_coefficient(economic_context()) -> float().
collect_wealth_gini_coefficient(#economic_context{
    individual_wealth = Wealth
}) ->
    case length(Wealth) >= 2 of
        false -> 0.0;
        true -> calculate_gini(Wealth)
    end.

%% @doc Population debt level
-spec collect_debt_level(economic_context()) -> float().
collect_debt_level(#economic_context{
    total_debt = Debt,
    total_assets = Assets
}) ->
    case Assets of
        0 when Debt > 0 -> 1.0;
        0 -> 0.0;
        _ -> clamp(Debt / Assets, 0.0, 1.0)
    end.

%% @doc Fitness gained per unit cost
-spec collect_fitness_per_cost(economic_context()) -> float().
collect_fitness_per_cost(#economic_context{
    fitness_gained = Fitness,
    cost_incurred = Cost
}) ->
    case Cost of
        0 -> 0.5;
        _ ->
            Efficiency = Fitness / Cost,
            %% Normalize to [0,1] assuming typical range
            sigmoid(Efficiency * 10 - 5)
    end.

%% @doc Resource scarcity level
-spec collect_scarcity_index(economic_context()) -> float().
collect_scarcity_index(#economic_context{
    available_resources = Available,
    demanded_resources = Demanded
}) ->
    case Demanded of
        0 -> 0.0;
        _ ->
            Scarcity = 1.0 - (Available / Demanded),
            clamp(Scarcity, 0.0, 1.0)
    end.

%%====================================================================
%% Internal Functions
%%====================================================================

calculate_trend(Values) ->
    N = length(Values),
    case N >= 2 of
        false -> 0.0;
        true ->
            Indices = lists:seq(1, N),
            MeanX = (N + 1) / 2,
            MeanY = lists:sum(Values) / N,
            Numerator = lists:sum([((I - MeanX) * (V - MeanY))
                                   || {I, V} <- lists:zip(Indices, Values)]),
            Denominator = lists:sum([math:pow(I - MeanX, 2) || I <- Indices]),
            case Denominator of
                0.0 -> 0.0;
                _ -> clamp(Numerator / Denominator, -1.0, 1.0)
            end
    end.

calculate_gini(Wealth) ->
    Sorted = lists:sort(Wealth),
    N = length(Sorted),
    Total = lists:sum(Sorted),
    case {N, Total} of
        {0, _} -> 0.0;
        {_, 0} -> 0.0;
        _ ->
            IndexedSum = lists:sum([
                (2 * I - N - 1) * W
                || {I, W} <- lists:zip(lists:seq(1, N), Sorted)
            ]),
            IndexedSum / (N * Total)
    end.

sigmoid(X) ->
    1.0 / (1.0 + math:exp(-X)).

clamp(Value, Min, Max) ->
    max(Min, min(Max, Value)).
```

---

## 4. L0 Actuators

### 4.1 Actuator Specifications

| ID | Name | Range | Default | Description |
|----|------|-------|---------|-------------|
| 1 | `computation_allocation_strategy` | [0.0, 1.0] | 0.5 | Equal vs fitness-proportional |
| 2 | `budget_per_individual` | [0.1, 10.0] | 1.0 | Compute units per individual |
| 3 | `energy_tax_rate` | [0.0, 0.3] | 0.1 | Tax on energy income |
| 4 | `wealth_redistribution_rate` | [0.0, 0.5] | 0.1 | Redistribution from rich to poor |
| 5 | `trade_incentive` | [0.0, 0.5] | 0.2 | Bonus for trading |
| 6 | `bankruptcy_threshold` | [0.0, 0.3] | 0.05 | When to declare bankruptcy |
| 7 | `investment_horizon` | [1, 20] | 5 | Generations for ROI calculation |
| 8 | `resource_discovery_bonus` | [0.0, 0.5] | 0.1 | Bonus for finding resources |
| 9 | `inflation_control` | [0.0, 0.1] | 0.02 | Control fitness inflation |
| 10 | `debt_penalty` | [0.0, 0.2] | 0.05 | Fitness penalty for debt |

### 4.2 Actuator Module

```erlang
%%%-------------------------------------------------------------------
%%% @doc Economic Silo L0 Actuators
%%% Controls economic parameters: budgets, taxes, trade, investment.
%%% @end
%%%-------------------------------------------------------------------
-module(economic_silo_actuators).

-behaviour(l0_actuator_behaviour).

%% API
-export([actuator_specs/0,
         apply_actuators/2,
         actuator_count/0]).

%% Individual actuator application
-export([apply_computation_allocation_strategy/2,
         apply_budget_per_individual/2,
         apply_energy_tax_rate/2,
         apply_wealth_redistribution_rate/2,
         apply_trade_incentive/2,
         apply_bankruptcy_threshold/2,
         apply_investment_horizon/2,
         apply_resource_discovery_bonus/2,
         apply_inflation_control/2,
         apply_debt_penalty/2]).

-include("economic_silo.hrl").

%%====================================================================
%% Behaviour Callbacks
%%====================================================================

-spec actuator_specs() -> [l0_actuator_spec()].
actuator_specs() ->
    [
        #{id => computation_allocation_strategy,
          name => <<"Compute Allocation">>,
          range => {0.0, 1.0},
          default => 0.5,
          description => <<"Equal vs fitness-proportional allocation">>},

        #{id => budget_per_individual,
          name => <<"Budget per Individual">>,
          range => {0.1, 10.0},
          default => 1.0,
          description => <<"Compute units per individual">>},

        #{id => energy_tax_rate,
          name => <<"Energy Tax Rate">>,
          range => {0.0, 0.3},
          default => 0.1,
          description => <<"Tax on energy income">>},

        #{id => wealth_redistribution_rate,
          name => <<"Wealth Redistribution">>,
          range => {0.0, 0.5},
          default => 0.1,
          description => <<"Redistribution rate">>},

        #{id => trade_incentive,
          name => <<"Trade Incentive">>,
          range => {0.0, 0.5},
          default => 0.2,
          description => <<"Bonus for trading">>},

        #{id => bankruptcy_threshold,
          name => <<"Bankruptcy Threshold">>,
          range => {0.0, 0.3},
          default => 0.05,
          description => <<"When to declare bankruptcy">>},

        #{id => investment_horizon,
          name => <<"Investment Horizon">>,
          range => {1, 20},
          default => 5,
          description => <<"Generations for ROI">>},

        #{id => resource_discovery_bonus,
          name => <<"Resource Discovery Bonus">>,
          range => {0.0, 0.5},
          default => 0.1,
          description => <<"Bonus for finding resources">>},

        #{id => inflation_control,
          name => <<"Inflation Control">>,
          range => {0.0, 0.1},
          default => 0.02,
          description => <<"Control fitness inflation">>},

        #{id => debt_penalty,
          name => <<"Debt Penalty">>,
          range => {0.0, 0.2},
          default => 0.05,
          description => <<"Fitness penalty for debt">>}
    ].

-spec actuator_count() -> pos_integer().
actuator_count() -> 10.

-spec apply_actuators(actuator_vector(), economic_state()) -> economic_state().
apply_actuators(Outputs, State) when length(Outputs) =:= 10 ->
    [AllocStrat, BudgetInd, TaxRate, Redistrib, TradeInc,
     BankThresh, InvestHor, DiscBonus, InflCtrl, DebtPen] = Outputs,

    State1 = apply_computation_allocation_strategy(AllocStrat, State),
    State2 = apply_budget_per_individual(BudgetInd, State1),
    State3 = apply_energy_tax_rate(TaxRate, State2),
    State4 = apply_wealth_redistribution_rate(Redistrib, State3),
    State5 = apply_trade_incentive(TradeInc, State4),
    State6 = apply_bankruptcy_threshold(BankThresh, State5),
    State7 = apply_investment_horizon(InvestHor, State6),
    State8 = apply_resource_discovery_bonus(DiscBonus, State7),
    State9 = apply_inflation_control(InflCtrl, State8),
    apply_debt_penalty(DebtPen, State9).

%%====================================================================
%% Individual Actuator Application
%%====================================================================

apply_computation_allocation_strategy(Output, State) ->
    State#economic_state{computation_allocation_strategy = Output}.

apply_budget_per_individual(Output, State) ->
    Budget = 0.1 + Output * 9.9,
    State#economic_state{budget_per_individual = Budget}.

apply_energy_tax_rate(Output, State) ->
    Rate = Output * 0.3,
    State#economic_state{energy_tax_rate = Rate}.

apply_wealth_redistribution_rate(Output, State) ->
    Rate = Output * 0.5,
    State#economic_state{wealth_redistribution_rate = Rate}.

apply_trade_incentive(Output, State) ->
    Incentive = Output * 0.5,
    State#economic_state{trade_incentive = Incentive}.

apply_bankruptcy_threshold(Output, State) ->
    Threshold = Output * 0.3,
    State#economic_state{bankruptcy_threshold = Threshold}.

apply_investment_horizon(Output, State) ->
    Horizon = round(1 + Output * 19),
    State#economic_state{investment_horizon = Horizon}.

apply_resource_discovery_bonus(Output, State) ->
    Bonus = Output * 0.5,
    State#economic_state{resource_discovery_bonus = Bonus}.

apply_inflation_control(Output, State) ->
    Control = Output * 0.1,
    State#economic_state{inflation_control = Control}.

apply_debt_penalty(Output, State) ->
    Penalty = Output * 0.2,
    State#economic_state{debt_penalty = Penalty}.
```

---

## 5. Record Definitions

```erlang
%%%-------------------------------------------------------------------
%%% @doc Economic Silo Header
%%% Record definitions for economic dynamics management.
%%% @end
%%%-------------------------------------------------------------------

-ifndef(ECONOMIC_SILO_HRL).
-define(ECONOMIC_SILO_HRL, true).

%%====================================================================
%% Types
%%====================================================================

-type sensor_vector() :: [float()].
-type actuator_vector() :: [float()].
-type generation() :: non_neg_integer().
-type energy_units() :: float().
-type compute_units() :: float().

%%====================================================================
%% Individual Account Record
%%====================================================================

-record(individual_account, {
    individual_id :: binary(),

    %% Resources
    energy = 100.0 :: energy_units(),
    compute_budget = 1.0 :: compute_units(),

    %% Wealth
    wealth = 0.0 :: float(),
    debt = 0.0 :: float(),

    %% History
    income_history = [] :: [{generation(), float()}],
    expenditure_history = [] :: [{generation(), float()}],
    trade_history = [] :: [trade()],

    %% Efficiency
    fitness_earned = 0.0 :: float(),
    cost_incurred = 0.0 :: float()
}).

-type individual_account() :: #individual_account{}.

%%====================================================================
%% Trade Record
%%====================================================================

-record(trade, {
    trade_id :: binary(),
    generation :: generation(),

    %% Parties
    buyer_id :: binary(),
    seller_id :: binary(),

    %% Exchange
    item_type :: atom(),
    quantity :: float(),
    price :: float(),

    %% Status
    completed = false :: boolean()
}).

-type trade() :: #trade{}.

%%====================================================================
%% Context Record (Input to Sensors)
%%====================================================================

-record(economic_context, {
    %% Budget
    budget_remaining = 1000.0 :: float(),
    total_budget = 1000.0 :: float(),
    budget_history = [] :: [float()],

    %% Energy
    total_energy = 10000.0 :: energy_units(),
    max_energy = 10000.0 :: energy_units(),
    energy_income_rate = 100.0 :: float(),
    max_income_rate = 500.0 :: float(),
    energy_expenditure_rate = 50.0 :: float(),
    max_expenditure_rate = 500.0 :: float(),

    %% Trade
    trade_volume = 0.0 :: float(),
    max_trade_volume = 1000.0 :: float(),
    exports = 0.0 :: float(),
    imports = 0.0 :: float(),

    %% Market
    cost_per_fitness_point = 10.0 :: float(),
    baseline_cost = 10.0 :: float(),

    %% Wealth distribution
    individual_wealth = [] :: [float()],
    total_debt = 0.0 :: float(),
    total_assets = 10000.0 :: float(),

    %% Efficiency
    fitness_gained = 0.0 :: float(),
    cost_incurred = 1.0 :: float(),

    %% Resources
    available_resources = 100.0 :: float(),
    demanded_resources = 100.0 :: float()
}).

-type economic_context() :: #economic_context{}.

%%====================================================================
%% State Record
%%====================================================================

-record(economic_state, {
    %% Configuration
    config :: economic_config(),

    %% Actuator outputs
    computation_allocation_strategy = 0.5 :: float(),
    budget_per_individual = 1.0 :: float(),
    energy_tax_rate = 0.1 :: float(),
    wealth_redistribution_rate = 0.1 :: float(),
    trade_incentive = 0.2 :: float(),
    bankruptcy_threshold = 0.05 :: float(),
    investment_horizon = 5 :: pos_integer(),
    resource_discovery_bonus = 0.1 :: float(),
    inflation_control = 0.02 :: float(),
    debt_penalty = 0.05 :: float(),

    %% Accounts
    accounts = #{} :: #{binary() => individual_account()},

    %% Market
    pending_trades = [] :: [trade()],
    completed_trades = [] :: [trade()],
    market_price = 10.0 :: float(),

    %% Tracking
    current_generation = 0 :: generation(),
    total_tax_collected = 0.0 :: float(),
    total_redistributed = 0.0 :: float(),
    bankruptcies = 0 :: non_neg_integer(),

    %% L2 integration
    l2_enabled = false :: boolean(),
    l2_guidance = undefined :: l2_guidance() | undefined
}).

-type economic_state() :: #economic_state{}.

%%====================================================================
%% Configuration Record
%%====================================================================

-record(economic_config, {
    enabled = true :: boolean(),
    initial_energy = 100.0 :: energy_units(),
    initial_budget = 1.0 :: compute_units(),
    enable_trade = true :: boolean(),
    enable_taxation = true :: boolean(),
    emit_events = true :: boolean()
}).

-type economic_config() :: #economic_config{}.

%%====================================================================
%% L2 Guidance Record
%%====================================================================

-record(l2_guidance, {
    budget_pressure = 0.5 :: float(),
    efficiency_emphasis = 0.5 :: float(),
    trade_encouragement = 0.5 :: float(),
    redistribution_level = 0.5 :: float()
}).

-type l2_guidance() :: #l2_guidance{}.

-endif.
```

---

## 6. Core Silo Implementation

```erlang
%%%-------------------------------------------------------------------
%%% @doc Economic Silo
%%% Manages resource allocation and economic dynamics for neuroevolution.
%%% @end
%%%-------------------------------------------------------------------
-module(economic_silo).

-behaviour(gen_server).

%% API
-export([start_link/1,
         get_economic_params/1,
         update_context/2,
         allocate_budget/2,
         charge_energy/3,
         execute_trade/4,
         get_account/2,
         redistribute_wealth/1,
         collect_taxes/1,
         get_state/1,
         enable/1,
         disable/1,
         is_enabled/1]).

%% Cross-silo signals
-export([signal_economic_pressure/1,
         signal_budget_available/1,
         signal_efficiency_requirement/1,
         receive_resource_level/2,
         receive_complexity_cost/2,
         receive_evaluation_time/2]).

%% gen_server callbacks
-export([init/1, handle_call/3, handle_cast/2, handle_info/2,
         terminate/2, code_change/3]).

-include("economic_silo.hrl").

%%====================================================================
%% API
%%====================================================================

-spec start_link(economic_config()) -> {ok, pid()} | {error, term()}.
start_link(Config) ->
    gen_server:start_link({local, ?MODULE}, ?MODULE, Config, []).

-spec get_economic_params(pid()) -> map().
get_economic_params(Pid) ->
    gen_server:call(Pid, get_economic_params).

-spec update_context(pid(), economic_context()) -> ok.
update_context(Pid, Context) ->
    gen_server:cast(Pid, {update_context, Context}).

-spec allocate_budget(pid(), binary()) -> {ok, compute_units()} | {error, term()}.
allocate_budget(Pid, IndividualId) ->
    gen_server:call(Pid, {allocate_budget, IndividualId}).

-spec charge_energy(pid(), binary(), energy_units()) -> ok | {error, term()}.
charge_energy(Pid, IndividualId, Amount) ->
    gen_server:call(Pid, {charge_energy, IndividualId, Amount}).

-spec execute_trade(pid(), binary(), binary(), float()) -> ok | {error, term()}.
execute_trade(Pid, BuyerId, SellerId, Amount) ->
    gen_server:call(Pid, {execute_trade, BuyerId, SellerId, Amount}).

-spec get_account(pid(), binary()) -> individual_account() | undefined.
get_account(Pid, IndividualId) ->
    gen_server:call(Pid, {get_account, IndividualId}).

-spec redistribute_wealth(pid()) -> ok.
redistribute_wealth(Pid) ->
    gen_server:cast(Pid, redistribute_wealth).

-spec collect_taxes(pid()) -> ok.
collect_taxes(Pid) ->
    gen_server:cast(Pid, collect_taxes).

-spec get_state(pid()) -> economic_state().
get_state(Pid) ->
    gen_server:call(Pid, get_state).

-spec enable(pid()) -> ok.
enable(Pid) ->
    gen_server:call(Pid, enable).

-spec disable(pid()) -> ok.
disable(Pid) ->
    gen_server:call(Pid, disable).

-spec is_enabled(pid()) -> boolean().
is_enabled(Pid) ->
    gen_server:call(Pid, is_enabled).

%%====================================================================
%% Cross-Silo Signal API
%%====================================================================

-spec signal_economic_pressure(pid()) -> float().
signal_economic_pressure(Pid) ->
    gen_server:call(Pid, signal_economic_pressure).

-spec signal_budget_available(pid()) -> float().
signal_budget_available(Pid) ->
    gen_server:call(Pid, signal_budget_available).

-spec signal_efficiency_requirement(pid()) -> float().
signal_efficiency_requirement(Pid) ->
    gen_server:call(Pid, signal_efficiency_requirement).

-spec receive_resource_level(pid(), float()) -> ok.
receive_resource_level(Pid, Level) ->
    gen_server:cast(Pid, {cross_silo, resource_level, Level}).

-spec receive_complexity_cost(pid(), float()) -> ok.
receive_complexity_cost(Pid, Cost) ->
    gen_server:cast(Pid, {cross_silo, complexity_cost, Cost}).

-spec receive_evaluation_time(pid(), float()) -> ok.
receive_evaluation_time(Pid, Time) ->
    gen_server:cast(Pid, {cross_silo, evaluation_time, Time}).

%%====================================================================
%% gen_server Callbacks
%%====================================================================

init(Config) ->
    State = #economic_state{config = Config},
    {ok, State}.

handle_call(get_economic_params, _From, State) ->
    Params = #{
        computation_allocation_strategy => State#economic_state.computation_allocation_strategy,
        budget_per_individual => State#economic_state.budget_per_individual,
        energy_tax_rate => State#economic_state.energy_tax_rate,
        wealth_redistribution_rate => State#economic_state.wealth_redistribution_rate,
        trade_incentive => State#economic_state.trade_incentive,
        bankruptcy_threshold => State#economic_state.bankruptcy_threshold,
        investment_horizon => State#economic_state.investment_horizon,
        resource_discovery_bonus => State#economic_state.resource_discovery_bonus,
        debt_penalty => State#economic_state.debt_penalty
    },
    {reply, Params, State};

handle_call({allocate_budget, IndividualId}, _From, State) ->
    Budget = State#economic_state.budget_per_individual,
    Account = get_or_create_account(IndividualId, State),
    NewAccount = Account#individual_account{compute_budget = Budget},
    NewAccounts = maps:put(IndividualId, NewAccount, State#economic_state.accounts),

    maybe_emit_event(budget_allocated, #{
        individual_id => IndividualId,
        amount => Budget
    }, State),

    {reply, {ok, Budget}, State#economic_state{accounts = NewAccounts}};

handle_call({charge_energy, IndividualId, Amount}, _From, State) ->
    case maps:get(IndividualId, State#economic_state.accounts, undefined) of
        undefined ->
            {reply, {error, not_found}, State};
        Account ->
            case Account#individual_account.energy >= Amount of
                false ->
                    {reply, {error, insufficient_energy}, State};
                true ->
                    NewEnergy = Account#individual_account.energy - Amount,
                    NewAccount = Account#individual_account{
                        energy = NewEnergy,
                        cost_incurred = Account#individual_account.cost_incurred + Amount
                    },
                    NewAccounts = maps:put(IndividualId, NewAccount, State#economic_state.accounts),
                    {reply, ok, State#economic_state{accounts = NewAccounts}}
            end
    end;

handle_call({execute_trade, BuyerId, SellerId, Amount}, _From, State) ->
    #economic_state{config = Config} = State,
    case Config#economic_config.enable_trade of
        false ->
            {reply, {error, trade_disabled}, State};
        true ->
            case {maps:get(BuyerId, State#economic_state.accounts, undefined),
                  maps:get(SellerId, State#economic_state.accounts, undefined)} of
                {undefined, _} -> {reply, {error, buyer_not_found}, State};
                {_, undefined} -> {reply, {error, seller_not_found}, State};
                {BuyerAcc, SellerAcc} ->
                    Price = State#economic_state.market_price * Amount,
                    case BuyerAcc#individual_account.energy >= Price of
                        false ->
                            {reply, {error, insufficient_funds}, State};
                        true ->
                            %% Execute trade
                            NewBuyer = BuyerAcc#individual_account{
                                energy = BuyerAcc#individual_account.energy - Price
                            },
                            NewSeller = SellerAcc#individual_account{
                                energy = SellerAcc#individual_account.energy + Price
                            },

                            Trade = #trade{
                                trade_id = generate_trade_id(),
                                generation = State#economic_state.current_generation,
                                buyer_id = BuyerId,
                                seller_id = SellerId,
                                quantity = Amount,
                                price = Price,
                                completed = true
                            },

                            NewAccounts = maps:put(BuyerId, NewBuyer,
                                          maps:put(SellerId, NewSeller,
                                                  State#economic_state.accounts)),

                            maybe_emit_event(trade_completed, #{
                                buyer_id => BuyerId,
                                seller_id => SellerId,
                                amount => Amount,
                                price => Price
                            }, State),

                            {reply, ok, State#economic_state{
                                accounts = NewAccounts,
                                completed_trades = [Trade | State#economic_state.completed_trades]
                            }}
                    end
            end
    end;

handle_call({get_account, IndividualId}, _From, State) ->
    Account = maps:get(IndividualId, State#economic_state.accounts, undefined),
    {reply, Account, State};

handle_call(get_state, _From, State) ->
    {reply, State, State};

handle_call(enable, _From, State) ->
    Config = State#economic_state.config,
    NewConfig = Config#economic_config{enabled = true},
    {reply, ok, State#economic_state{config = NewConfig}};

handle_call(disable, _From, State) ->
    Config = State#economic_state.config,
    NewConfig = Config#economic_config{enabled = false},
    {reply, ok, State#economic_state{config = NewConfig}};

handle_call(is_enabled, _From, State) ->
    {reply, State#economic_state.config#economic_config.enabled, State};

handle_call(signal_economic_pressure, _From, State) ->
    %% Pressure based on scarcity and debt
    Accounts = maps:values(State#economic_state.accounts),
    case Accounts of
        [] -> {reply, 0.5, State};
        _ ->
            TotalDebt = lists:sum([A#individual_account.debt || A <- Accounts]),
            TotalWealth = lists:sum([A#individual_account.wealth || A <- Accounts]),
            Pressure = case TotalWealth of
                0 -> 0.8;
                _ -> clamp(TotalDebt / TotalWealth, 0.0, 1.0)
            end,
            {reply, Pressure, State}
    end;

handle_call(signal_budget_available, _From, State) ->
    %% Average budget availability
    Accounts = maps:values(State#economic_state.accounts),
    case Accounts of
        [] -> {reply, 1.0, State};
        _ ->
            AvgBudget = lists:sum([A#individual_account.compute_budget || A <- Accounts])
                       / length(Accounts),
            Available = clamp(AvgBudget / State#economic_state.budget_per_individual, 0.0, 1.0),
            {reply, Available, State}
    end;

handle_call(signal_efficiency_requirement, _From, State) ->
    %% Higher when budget is low
    {reply, 1.0 - State#economic_state.budget_per_individual / 10.0, State};

handle_call(_Request, _From, State) ->
    {reply, {error, unknown_request}, State}.

handle_cast({update_context, Context}, State) ->
    #economic_state{config = Config} = State,
    case Config#economic_config.enabled of
        false -> {noreply, State};
        true ->
            SensorVector = economic_silo_sensors:collect_sensors(Context),
            ActuatorVector = process_through_controller(SensorVector, State),
            NewState = economic_silo_actuators:apply_actuators(ActuatorVector, State),
            FinalState = NewState#economic_state{
                current_generation = State#economic_state.current_generation + 1
            },
            {noreply, FinalState}
    end;

handle_cast(redistribute_wealth, State) ->
    Rate = State#economic_state.wealth_redistribution_rate,
    Accounts = maps:to_list(State#economic_state.accounts),

    case length(Accounts) < 2 of
        true -> {noreply, State};
        false ->
            %% Calculate total and average wealth
            TotalWealth = lists:sum([A#individual_account.wealth || {_, A} <- Accounts]),
            AvgWealth = TotalWealth / length(Accounts),

            %% Redistribute
            NewAccounts = lists:foldl(fun({Id, Acc}, AccMap) ->
                Diff = Acc#individual_account.wealth - AvgWealth,
                Transfer = Diff * Rate,
                NewAcc = Acc#individual_account{
                    wealth = Acc#individual_account.wealth - Transfer
                },
                maps:put(Id, NewAcc, AccMap)
            end, #{}, Accounts),

            maybe_emit_event(wealth_redistributed, #{
                rate => Rate,
                total_transferred => TotalWealth * Rate
            }, State),

            {noreply, State#economic_state{
                accounts = NewAccounts,
                total_redistributed = State#economic_state.total_redistributed + TotalWealth * Rate
            }}
    end;

handle_cast(collect_taxes, State) ->
    #economic_state{config = Config} = State,
    case Config#economic_config.enable_taxation of
        false -> {noreply, State};
        true ->
            Rate = State#economic_state.energy_tax_rate,
            Accounts = maps:to_list(State#economic_state.accounts),

            {NewAccounts, TotalTax} = lists:foldl(fun({Id, Acc}, {AccMap, Tax}) ->
                Income = lists:sum([I || {_, I} <- Acc#individual_account.income_history]),
                TaxAmount = Income * Rate,
                NewAcc = Acc#individual_account{
                    energy = Acc#individual_account.energy - TaxAmount
                },
                {maps:put(Id, NewAcc, AccMap), Tax + TaxAmount}
            end, {#{}, 0.0}, Accounts),

            maybe_emit_event(energy_taxed, #{
                rate => Rate,
                total_collected => TotalTax
            }, State),

            {noreply, State#economic_state{
                accounts = NewAccounts,
                total_tax_collected = State#economic_state.total_tax_collected + TotalTax
            }}
    end;

handle_cast({cross_silo, resource_level, Level}, State) ->
    %% Adjust budget based on resource level
    NewBudget = State#economic_state.budget_per_individual * Level,
    {noreply, State#economic_state{budget_per_individual = max(0.1, NewBudget)}};

handle_cast({cross_silo, complexity_cost, Cost}, State) ->
    %% Update market price based on complexity
    NewPrice = State#economic_state.market_price * (1.0 + Cost * 0.5),
    {noreply, State#economic_state{market_price = NewPrice}};

handle_cast({cross_silo, evaluation_time, Time}, State) ->
    %% Time affects budget consumption
    {noreply, State};

handle_cast(_Msg, State) ->
    {noreply, State}.

handle_info(_Info, State) ->
    {noreply, State}.

terminate(_Reason, _State) ->
    ok.

code_change(_OldVsn, State, _Extra) ->
    {ok, State}.

%%====================================================================
%% Internal Functions
%%====================================================================

get_or_create_account(IndividualId, State) ->
    case maps:get(IndividualId, State#economic_state.accounts, undefined) of
        undefined ->
            #economic_state{config = Config} = State,
            #individual_account{
                individual_id = IndividualId,
                energy = Config#economic_config.initial_energy,
                compute_budget = Config#economic_config.initial_budget
            };
        Account ->
            Account
    end.

generate_trade_id() ->
    list_to_binary(integer_to_list(erlang:unique_integer([positive]))).

process_through_controller(SensorVector, State) ->
    case State#economic_state.l2_enabled of
        true -> apply_l2_guidance(SensorVector, State);
        false -> lists:duplicate(economic_silo_actuators:actuator_count(), 0.5)
    end.

apply_l2_guidance(_SensorVector, State) ->
    case State#economic_state.l2_guidance of
        undefined ->
            lists:duplicate(economic_silo_actuators:actuator_count(), 0.5);
        #l2_guidance{
            budget_pressure = BudgetPress,
            efficiency_emphasis = EffEmph,
            trade_encouragement = TradeEnc,
            redistribution_level = RedistLvl
        } ->
            [
                EffEmph,           % computation_allocation_strategy
                1.0 - BudgetPress, % budget_per_individual (inverse)
                0.1,               % energy_tax_rate
                RedistLvl,         % wealth_redistribution_rate
                TradeEnc,          % trade_incentive
                0.05,              % bankruptcy_threshold
                0.5,               % investment_horizon
                0.1,               % resource_discovery_bonus
                0.02,              % inflation_control
                BudgetPress * 0.2  % debt_penalty
            ]
    end.

maybe_emit_event(EventType, Payload, State) ->
    case State#economic_state.config#economic_config.emit_events of
        true ->
            Event = #{
                type => EventType,
                silo => economic,
                timestamp => erlang:system_time(millisecond),
                generation => State#economic_state.current_generation,
                payload => Payload
            },
            event_bus:publish(economic_silo_events, Event);
        false ->
            ok
    end.

clamp(Value, Min, Max) ->
    max(Min, min(Max, Value)).
```

---

## 7. Cross-Silo Signal Matrix

### 7.1 Outgoing Signals

| Signal | To Silo | Description |
|--------|---------|-------------|
| `economic_pressure` | Task | High pressure suggests simpler solutions |
| `budget_available` | Temporal | Budget constrains evaluation time |
| `efficiency_requirement` | Morphological | Efficiency targets for network size |

### 7.2 Incoming Signals

| Signal | From Silo | Effect |
|--------|-----------|--------|
| `resource_level` | Ecological | Resources affect available budget |
| `complexity_cost` | Morphological | Network size affects energy cost |
| `evaluation_time` | Temporal | Time affects compute cost |

---

## 8. Events Emitted

| Event | Trigger | Payload |
|-------|---------|---------|
| `budget_allocated` | Compute assigned | `{individual_id, amount}` |
| `budget_exhausted` | Ran out of compute | `{individual_id, needed, available}` |
| `trade_completed` | Exchange occurred | `{buyer_id, seller_id, amount, price}` |
| `bankruptcy_declared` | Out of resources | `{individual_id, debt, assets}` |
| `wealth_redistributed` | Transfer occurred | `{rate, total_transferred}` |
| `energy_taxed` | Tax collected | `{rate, total_collected}` |

---

## 9. Implementation Phases

### Phase 1: Core Infrastructure
- [ ] Create `economic_silo.hrl`
- [ ] Implement sensors and actuators
- [ ] Basic gen_server

### Phase 2: Budget & Energy
- [ ] Budget allocation system
- [ ] Energy tracking
- [ ] Cost accounting

### Phase 3: Trade & Market
- [ ] Trade execution
- [ ] Market pricing
- [ ] Trade incentives

### Phase 4: Taxation & Redistribution
- [ ] Tax collection
- [ ] Wealth redistribution
- [ ] Bankruptcy handling

### Phase 5: Integration
- [ ] Cross-silo signals
- [ ] Event emission
- [ ] Testing

---

## 10. Success Criteria

1. **Budget Awareness**: All individuals operate within budgets
2. **Efficiency Pressure**: Higher fitness/cost ratio rewarded
3. **Trade Function**: Successful trades between individuals
4. **Redistribution**: Wealth inequality managed
5. **Cost Tracking**: Full cost attribution for training
