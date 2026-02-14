%% @doc Domain Signal Provider behaviour.
%%
%% Domains implement this behaviour to emit signals that inform silo
%% decision-making. Unlike sensors (which feed L0 networks), signals
%% provide meta-level information to the self-tuning silos.
%%
%% Signals are routed to silos by category via the signal_router.
%% Each silo declares which categories it handles.
%%
%% == Signal Categories ==
%%
%% Categories map to silo types:
%%
%%   ecological    - Resource pressure, carrying capacity, scarcity
%%   competitive   - Predator/prey ratios, conflict rates, rankings
%%   morphological - Topology patterns, connectivity metrics
%%   regulatory    - Threshold violations, homeostatic state
%%   task          - Task difficulty, learning progress
%%   resource      - Compute usage, memory pressure
%%   distribution  - Node load, network latency
%%   temporal      - Time patterns, seasonal effects
%%   developmental - Growth stage, maturation progress
%%   cultural      - Behavioral diversity, meme spread
%%   social        - Group structures, cooperation levels
%%   communication - Signal quality, message rates
%%   economic      - Resource trading, market conditions
%%
%% == Example Implementation ==
%%
%% A domain bridge module implementing this behaviour:
%%
%%   -module(my_domain_bridge).
%%   -behaviour(domain_signals).
%%
%%   signal_spec() ->
%%       [#{name => food_scarcity, category => ecological, level => l0,
%%          range => {0.0, 1.0}, description => "Food availability"},
%%        #{name => predator_ratio, category => competitive, level => l0,
%%          range => {0.0, 1.0}, description => "Predator to prey ratio"}].
%%
%%   emit_signals(DomainState, Metrics) ->
%%       FoodCount = maps:get(food_count, DomainState, 0),
%%       MaxFood = maps:get(max_food, DomainState, 100),
%%       Scarcity = 1.0 - (FoodCount / max(1, MaxFood)),
%%
%%       PredatorCount = maps:get(predators, Metrics, 0),
%%       PreyCount = maps:get(prey, Metrics, 1),
%%       Ratio = PredatorCount / max(1, PreyCount),
%%
%%       [{ecological, food_scarcity, Scarcity},
%%        {competitive, predator_ratio, min(1.0, Ratio)}].
%%
%% @author R.G. Lefever
%% @copyright 2024-2026 R.G. Lefever
-module(domain_signals).

%% Behaviour callbacks
-callback signal_spec() -> [signal_definition()].
-callback emit_signals(DomainState :: term(), Metrics :: map()) -> [signal()].

-type signal_category() ::
    ecological |
    competitive |
    morphological |
    regulatory |
    task |
    resource |
    distribution |
    temporal |
    developmental |
    cultural |
    social |
    communication |
    economic.

-type signal_level() :: l0 | l1 | l2.

-type signal_definition() :: #{
    name := atom(),
    category := signal_category(),
    level := signal_level(),
    range := {Min :: float(), Max :: float()},
    description := binary() | string()
}.

-type signal() :: {Category :: signal_category(), Name :: atom(), Value :: float()}.

-export_type([
    signal_category/0,
    signal_level/0,
    signal_definition/0,
    signal/0
]).
