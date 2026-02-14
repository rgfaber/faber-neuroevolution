# PLAN: Communication Silo

**Status:** Planning
**Created:** 2025-12-23
**Last Updated:** 2025-12-23
**Related:** PLAN_L2_L1_HIERARCHICAL_INTERFACE.md, PLAN_BEHAVIORAL_EVENTS.md

---

## Overview

The Communication Silo manages signaling and language evolution: signal repertoires, protocol evolution, coordination messages, and deception detection. Multi-agent systems need communication to coordinate, and evolved communication is more robust than hard-coded protocols.

## Motivation

**Business Value:**
- **Human-AI communication**: Agents develop interpretable signals
- **Multi-agent systems**: Agents coordinate without hard-coded protocols
- **Emergent protocols**: Discover communication strategies
- **Explainability**: Signals reveal agent intentions

**Technical Benefits:**
- Language evolves to fit the problem domain
- Honest signaling emerges through evolutionary pressure
- Coordination without centralized control
- Protocol discovery through optimization

### Training Velocity & Inference Impact

| Metric | Without Communication Silo | With Communication Silo |
|--------|---------------------------|------------------------|
| Multi-agent coordination | Hard-coded only | Evolved, adaptive |
| Training velocity | Baseline (1.0x) | Slight overhead (0.85-0.95x) |
| Inference latency | No communication overhead | +5-15ms per message exchange |
| Coordination success rate | ~40% (rigid protocols) | ~75% (adaptive protocols) |
| Protocol interpretability | Opaque | Traceable signal meanings |

**Note:** Training velocity has slight overhead due to language evolution and message passing. However, for multi-agent tasks, evolved communication dramatically improves coordination success, making overall task completion faster despite per-generation overhead.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                      COMMUNICATION SILO                              │
├─────────────────────────────────────────────────────────────────────┤
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │                      L0 SENSORS (10)                         │    │
│  │                                                              │    │
│  │  Vocabulary       Message        Honesty      Coordination  │    │
│  │  ┌─────────┐     ┌─────────┐    ┌─────────┐  ┌─────────┐   │    │
│  │  │vocabulary│    │message_ │    │signal_  │  │coord_   │   │    │
│  │  │_size    │     │complexity│   │honesty_ │  │success_ │   │    │
│  │  │vocabulary│    │communic_│    │rate     │  │rate     │   │    │
│  │  │_growth  │     │frequency│    │deception│  └─────────┘   │    │
│  │  └─────────┘     └─────────┘    │_detection│              │    │
│  │                                 └─────────┘               │    │
│  │  Language         Dialects                                 │    │
│  │  ┌─────────┐     ┌─────────┐                               │    │
│  │  │language_│     │dialect_ │                               │    │
│  │  │stability│     │count    │                               │    │
│  │  └─────────┘     └─────────┘                               │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                                │                                     │
│                       ┌────────▼────────┐                           │
│                       │ TWEANN Controller│                          │
│                       │   (online ES)   │                           │
│                       └────────┬────────┘                           │
│                                │                                     │
│  ┌─────────────────────────────▼───────────────────────────────┐    │
│  │                      L0 ACTUATORS (8)                        │    │
│  │                                                              │    │
│  │  Vocabulary       Honesty        Coordination   Dialect     │    │
│  │  ┌─────────┐     ┌─────────┐    ┌─────────┐   ┌─────────┐  │    │
│  │  │vocab_   │     │lying_   │    │coord_   │   │dialect_ │  │    │
│  │  │growth_  │     │penalty  │    │reward   │   │isolation│  │    │
│  │  │rate     │     │deception│    │message_ │   └─────────┘  │    │
│  │  │communic_│     │_detection│   │compression│              │    │
│  │  │_cost    │     │_bonus   │    └─────────┘               │    │
│  │  └─────────┘     └─────────┘                               │    │
│  └─────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────┘
```

---

## L0 Sensors

### Sensor Specifications

| ID | Range | Description |
|----|-------|-------------|
| `vocabulary_size` | [0.0, 1.0] | Distinct signals / max possible |
| `vocabulary_growth_rate` | [-1.0, 1.0] | Rate of new signal creation |
| `message_complexity_mean` | [0.0, 1.0] | Average message length/structure |
| `communication_frequency` | [0.0, 1.0] | Messages per interaction |
| `signal_honesty_rate` | [0.0, 1.0] | Proportion of honest signals |
| `deception_detection_rate` | [0.0, 1.0] | Success at detecting lies |
| `coordination_success_rate` | [0.0, 1.0] | Success of coordinated actions |
| `language_stability` | [0.0, 1.0] | How stable the language is |
| `dialect_count` | [0.0, 1.0] | Number of distinct dialects (normalized) |
| `compression_ratio` | [0.0, 1.0] | Information per signal unit |

### Erlang Implementation

```erlang
%%%-------------------------------------------------------------------
%%% @doc Communication Silo L0 Sensors
%%% Collects metrics about population communication dynamics.
%%% @end
%%%-------------------------------------------------------------------
-module(communication_silo_sensors).
-behaviour(l0_sensor_behaviour).

-export([
    init/1,
    collect/2,
    normalize/2,
    get_sensor_ids/0,
    get_sensor_spec/1
]).

-include("communication_silo.hrl").

%%====================================================================
%% Behavior Callbacks
%%====================================================================

-spec init(Config :: map()) -> {ok, State :: map()}.
init(Config) ->
    State = #{
        config => Config,
        signal_registry => #{},
        message_history => [],
        deception_log => [],
        coordination_log => [],
        dialect_map => #{},
        last_collection => erlang:system_time(millisecond)
    },
    {ok, State}.

-spec get_sensor_ids() -> [atom()].
get_sensor_ids() ->
    [vocabulary_size, vocabulary_growth_rate, message_complexity_mean,
     communication_frequency, signal_honesty_rate, deception_detection_rate,
     coordination_success_rate, language_stability, dialect_count,
     compression_ratio].

-spec get_sensor_spec(SensorId :: atom()) -> sensor_spec().
get_sensor_spec(vocabulary_size) ->
    #sensor_spec{
        id = vocabulary_size,
        range = {0.0, 1.0},
        description = <<"Distinct signals / max possible signals">>
    };
get_sensor_spec(vocabulary_growth_rate) ->
    #sensor_spec{
        id = vocabulary_growth_rate,
        range = {-1.0, 1.0},
        description = <<"Rate of new signal creation">>
    };
get_sensor_spec(message_complexity_mean) ->
    #sensor_spec{
        id = message_complexity_mean,
        range = {0.0, 1.0},
        description = <<"Average message length/structure">>
    };
get_sensor_spec(communication_frequency) ->
    #sensor_spec{
        id = communication_frequency,
        range = {0.0, 1.0},
        description = <<"Messages per interaction">>
    };
get_sensor_spec(signal_honesty_rate) ->
    #sensor_spec{
        id = signal_honesty_rate,
        range = {0.0, 1.0},
        description = <<"Proportion of honest signals">>
    };
get_sensor_spec(deception_detection_rate) ->
    #sensor_spec{
        id = deception_detection_rate,
        range = {0.0, 1.0},
        description = <<"Success at detecting lies">>
    };
get_sensor_spec(coordination_success_rate) ->
    #sensor_spec{
        id = coordination_success_rate,
        range = {0.0, 1.0},
        description = <<"Success of coordinated actions">>
    };
get_sensor_spec(language_stability) ->
    #sensor_spec{
        id = language_stability,
        range = {0.0, 1.0},
        description = <<"How stable the language is over time">>
    };
get_sensor_spec(dialect_count) ->
    #sensor_spec{
        id = dialect_count,
        range = {0.0, 1.0},
        description = <<"Number of distinct dialects (normalized)">>
    };
get_sensor_spec(compression_ratio) ->
    #sensor_spec{
        id = compression_ratio,
        range = {0.0, 1.0},
        description = <<"Information per signal unit">>
    }.

-spec collect(SensorId :: atom(), State :: map()) ->
    {Value :: float(), NewState :: map()}.
collect(vocabulary_size, State) ->
    #{signal_registry := Registry, config := Config} = State,
    MaxSignals = maps:get(max_vocabulary_size, Config, 1000),
    CurrentSignals = maps:size(Registry),
    Value = min(1.0, CurrentSignals / max(1, MaxSignals)),
    {Value, State};

collect(vocabulary_growth_rate, State) ->
    #{signal_registry := Registry} = State,
    PrevSize = maps:get(prev_vocabulary_size, State, maps:size(Registry)),
    CurrentSize = maps:size(Registry),
    Growth = (CurrentSize - PrevSize) / max(1, PrevSize),
    Value = clamp(Growth, -1.0, 1.0),
    NewState = State#{prev_vocabulary_size => CurrentSize},
    {Value, NewState};

collect(message_complexity_mean, State) ->
    #{message_history := History, config := Config} = State,
    MaxComplexity = maps:get(max_message_complexity, Config, 100),
    Value = case History of
        [] -> 0.5;
        Messages ->
            Complexities = [calculate_complexity(M) || M <- Messages],
            MeanComplexity = lists:sum(Complexities) / length(Complexities),
            min(1.0, MeanComplexity / MaxComplexity)
    end,
    {Value, State};

collect(communication_frequency, State) ->
    #{message_history := History, config := Config} = State,
    WindowMs = maps:get(frequency_window_ms, Config, 60000),
    Now = erlang:system_time(millisecond),
    RecentMessages = [M || M <- History,
                      maps:get(timestamp, M, 0) > Now - WindowMs],
    MaxFrequency = maps:get(max_frequency, Config, 100),
    Value = min(1.0, length(RecentMessages) / MaxFrequency),
    {Value, State};

collect(signal_honesty_rate, State) ->
    #{message_history := History} = State,
    Value = case History of
        [] -> 1.0;
        Messages ->
            HonestCount = length([M || M <- Messages,
                                  maps:get(honest, M, true) =:= true]),
            HonestCount / length(Messages)
    end,
    {Value, State};

collect(deception_detection_rate, State) ->
    #{deception_log := Log} = State,
    Value = case Log of
        [] -> 0.5;
        Attempts ->
            Detected = length([A || A <- Attempts,
                              maps:get(detected, A, false) =:= true]),
            Detected / length(Attempts)
    end,
    {Value, State};

collect(coordination_success_rate, State) ->
    #{coordination_log := Log} = State,
    Value = case Log of
        [] -> 0.5;
        Attempts ->
            Successes = length([A || A <- Attempts,
                               maps:get(success, A, false) =:= true]),
            Successes / length(Attempts)
    end,
    {Value, State};

collect(language_stability, State) ->
    #{signal_registry := Registry} = State,
    PrevRegistry = maps:get(prev_signal_registry, State, Registry),
    Stability = calculate_language_stability(PrevRegistry, Registry),
    NewState = State#{prev_signal_registry => Registry},
    {Stability, NewState};

collect(dialect_count, State) ->
    #{dialect_map := Dialects, config := Config} = State,
    MaxDialects = maps:get(max_dialects, Config, 10),
    Value = min(1.0, maps:size(Dialects) / max(1, MaxDialects)),
    {Value, State};

collect(compression_ratio, State) ->
    #{message_history := History} = State,
    Value = case History of
        [] -> 0.5;
        Messages ->
            Ratios = [calculate_compression(M) || M <- Messages],
            lists:sum(Ratios) / length(Ratios)
    end,
    {Value, State}.

-spec normalize(SensorId :: atom(), RawValue :: number()) -> float().
normalize(vocabulary_growth_rate, Value) ->
    (Value + 1.0) / 2.0;
normalize(_SensorId, Value) ->
    clamp(Value, 0.0, 1.0).

%%====================================================================
%% Internal Functions
%%====================================================================

calculate_complexity(Message) ->
    Signals = maps:get(signals, Message, []),
    BaseComplexity = length(Signals),
    StructureBonus = case maps:get(structure, Message, flat) of
        flat -> 0;
        nested -> 5;
        recursive -> 10
    end,
    BaseComplexity + StructureBonus.

calculate_language_stability(PrevRegistry, CurrentRegistry) ->
    PrevKeys = maps:keys(PrevRegistry),
    CurrentKeys = maps:keys(CurrentRegistry),
    Shared = length([K || K <- PrevKeys, lists:member(K, CurrentKeys)]),
    Total = length(lists:usort(PrevKeys ++ CurrentKeys)),
    case Total of
        0 -> 1.0;
        _ -> Shared / Total
    end.

calculate_compression(Message) ->
    Information = maps:get(information_bits, Message, 10),
    SignalLength = maps:get(signal_length, Message, 10),
    case SignalLength of
        0 -> 0.5;
        _ -> min(1.0, Information / (SignalLength * 2))
    end.

clamp(Value, Min, Max) ->
    max(Min, min(Max, Value)).
```

---

## L0 Actuators

### Actuator Specifications

| ID | Range | Default | Description |
|----|-------|---------|-------------|
| `vocabulary_growth_rate` | [0.0, 0.1] | 0.02 | Speed of vocabulary expansion |
| `communication_cost` | [0.0, 0.2] | 0.05 | Energy cost per message |
| `lying_penalty` | [0.0, 0.5] | 0.2 | Fitness penalty for detected lies |
| `deception_detection_bonus` | [0.0, 0.3] | 0.1 | Bonus for catching liars |
| `coordination_reward` | [0.0, 0.5] | 0.2 | Bonus for successful coordination |
| `message_compression_pressure` | [0.0, 0.5] | 0.1 | Pressure to compress messages |
| `dialect_isolation` | [0.0, 1.0] | 0.3 | How isolated dialects become |
| `language_mutation_rate` | [0.0, 0.1] | 0.02 | Rate of signal meaning change |

### Erlang Implementation

```erlang
%%%-------------------------------------------------------------------
%%% @doc Communication Silo L0 Actuators
%%% Controls communication parameters for the population.
%%% @end
%%%-------------------------------------------------------------------
-module(communication_silo_actuators).
-behaviour(l0_actuator_behaviour).

-export([
    init/1,
    apply/3,
    get_actuator_ids/0,
    get_actuator_spec/1,
    validate_value/2
]).

-include("communication_silo.hrl").

%%====================================================================
%% Behavior Callbacks
%%====================================================================

-spec init(Config :: map()) -> {ok, State :: map()}.
init(Config) ->
    State = #{
        config => Config,
        current_values => default_values(),
        application_history => []
    },
    {ok, State}.

-spec get_actuator_ids() -> [atom()].
get_actuator_ids() ->
    [vocabulary_growth_rate, communication_cost, lying_penalty,
     deception_detection_bonus, coordination_reward,
     message_compression_pressure, dialect_isolation, language_mutation_rate].

-spec get_actuator_spec(ActuatorId :: atom()) -> actuator_spec().
get_actuator_spec(vocabulary_growth_rate) ->
    #actuator_spec{
        id = vocabulary_growth_rate,
        range = {0.0, 0.1},
        default = 0.02,
        description = <<"Speed of vocabulary expansion">>
    };
get_actuator_spec(communication_cost) ->
    #actuator_spec{
        id = communication_cost,
        range = {0.0, 0.2},
        default = 0.05,
        description = <<"Energy cost per message">>
    };
get_actuator_spec(lying_penalty) ->
    #actuator_spec{
        id = lying_penalty,
        range = {0.0, 0.5},
        default = 0.2,
        description = <<"Fitness penalty for detected lies">>
    };
get_actuator_spec(deception_detection_bonus) ->
    #actuator_spec{
        id = deception_detection_bonus,
        range = {0.0, 0.3},
        default = 0.1,
        description = <<"Bonus for catching liars">>
    };
get_actuator_spec(coordination_reward) ->
    #actuator_spec{
        id = coordination_reward,
        range = {0.0, 0.5},
        default = 0.2,
        description = <<"Bonus for successful coordination">>
    };
get_actuator_spec(message_compression_pressure) ->
    #actuator_spec{
        id = message_compression_pressure,
        range = {0.0, 0.5},
        default = 0.1,
        description = <<"Pressure to compress messages">>
    };
get_actuator_spec(dialect_isolation) ->
    #actuator_spec{
        id = dialect_isolation,
        range = {0.0, 1.0},
        default = 0.3,
        description = <<"How isolated dialects become">>
    };
get_actuator_spec(language_mutation_rate) ->
    #actuator_spec{
        id = language_mutation_rate,
        range = {0.0, 0.1},
        default = 0.02,
        description = <<"Rate of signal meaning change">>
    }.

-spec apply(ActuatorId :: atom(), Value :: float(), State :: map()) ->
    {ok, NewState :: map()} | {error, Reason :: term()}.
apply(ActuatorId, Value, State) ->
    case validate_value(ActuatorId, Value) of
        ok ->
            #{current_values := CurrentValues} = State,
            NewValues = CurrentValues#{ActuatorId => Value},
            HistoryEntry = #{
                actuator => ActuatorId,
                value => Value,
                timestamp => erlang:system_time(millisecond)
            },
            #{application_history := History} = State,
            NewHistory = [HistoryEntry | lists:sublist(History, 99)],
            NewState = State#{
                current_values => NewValues,
                application_history => NewHistory
            },
            apply_to_population(ActuatorId, Value),
            {ok, NewState};
        {error, Reason} ->
            {error, Reason}
    end.

-spec validate_value(ActuatorId :: atom(), Value :: float()) ->
    ok | {error, Reason :: term()}.
validate_value(ActuatorId, Value) ->
    Spec = get_actuator_spec(ActuatorId),
    {Min, Max} = Spec#actuator_spec.range,
    case Value >= Min andalso Value =< Max of
        true -> ok;
        false -> {error, {out_of_range, ActuatorId, Value, {Min, Max}}}
    end.

%%====================================================================
%% Internal Functions
%%====================================================================

default_values() ->
    #{
        vocabulary_growth_rate => 0.02,
        communication_cost => 0.05,
        lying_penalty => 0.2,
        deception_detection_bonus => 0.1,
        coordination_reward => 0.2,
        message_compression_pressure => 0.1,
        dialect_isolation => 0.3,
        language_mutation_rate => 0.02
    }.

apply_to_population(vocabulary_growth_rate, Value) ->
    population_manager:set_vocab_growth_rate(Value);
apply_to_population(communication_cost, Value) ->
    population_manager:set_communication_cost(Value);
apply_to_population(lying_penalty, Value) ->
    population_manager:set_lying_penalty(Value);
apply_to_population(deception_detection_bonus, Value) ->
    population_manager:set_detection_bonus(Value);
apply_to_population(coordination_reward, Value) ->
    population_manager:set_coordination_reward(Value);
apply_to_population(message_compression_pressure, Value) ->
    population_manager:set_compression_pressure(Value);
apply_to_population(dialect_isolation, Value) ->
    population_manager:set_dialect_isolation(Value);
apply_to_population(language_mutation_rate, Value) ->
    population_manager:set_language_mutation_rate(Value).
```

---

## Record Definitions

```erlang
%%%-------------------------------------------------------------------
%%% @doc Communication Silo Record Definitions
%%% @end
%%%-------------------------------------------------------------------

-ifndef(COMMUNICATION_SILO_HRL).
-define(COMMUNICATION_SILO_HRL, true).

%% Sensor specification
-record(sensor_spec, {
    id :: atom(),
    range :: {float(), float()},
    description :: binary()
}).
-type sensor_spec() :: #sensor_spec{}.

%% Actuator specification
-record(actuator_spec, {
    id :: atom(),
    range :: {float(), float()},
    default :: float(),
    description :: binary()
}).
-type actuator_spec() :: #actuator_spec{}.

%% Signal definition
-record(signal, {
    id :: binary(),
    meaning :: term(),
    creator_id :: binary(),
    created_at :: pos_integer(),
    adopters :: [binary()],
    mutation_count :: non_neg_integer(),
    stability :: float()
}).
-type signal() :: #signal{}.

%% Message record
-record(message, {
    id :: binary(),
    sender_id :: binary(),
    receiver_ids :: [binary()],
    signals :: [binary()],
    payload :: term(),
    honest :: boolean(),
    timestamp :: pos_integer(),
    information_bits :: non_neg_integer(),
    signal_length :: non_neg_integer(),
    structure :: flat | nested | recursive
}).
-type message() :: #message{}.

%% Dialect record
-record(dialect, {
    id :: binary(),
    members :: [binary()],
    vocabulary :: #{binary() => term()},
    created_at :: pos_integer(),
    parent_dialect :: binary() | undefined,
    isolation_score :: float()
}).
-type dialect() :: #dialect{}.

%% Coordination attempt
-record(coordination_attempt, {
    id :: binary(),
    participant_ids :: [binary()],
    action :: term(),
    messages_exchanged :: [binary()],
    success :: boolean(),
    timestamp :: pos_integer(),
    outcome :: term()
}).
-type coordination_attempt() :: #coordination_attempt{}.

%% Deception event
-record(deception_event, {
    id :: binary(),
    liar_id :: binary(),
    target_id :: binary(),
    signal :: binary(),
    actual_state :: term(),
    claimed_state :: term(),
    detected :: boolean(),
    detector_id :: binary() | undefined,
    timestamp :: pos_integer()
}).
-type deception_event() :: #deception_event{}.

%% Communication silo state
-record(communication_state, {
    enabled = true :: boolean(),
    population_id :: binary(),
    signal_registry :: #{binary() => signal()},
    dialects :: #{binary() => dialect()},
    message_history :: [message()],
    coordination_history :: [coordination_attempt()],
    deception_history :: [deception_event()],
    current_actuator_values :: map(),
    l2_guidance :: map(),
    last_update :: pos_integer()
}).
-type communication_state() :: #communication_state{}.

%% Configuration
-record(communication_config, {
    enabled = true :: boolean(),
    max_vocabulary_size = 1000 :: pos_integer(),
    max_message_complexity = 100 :: pos_integer(),
    max_dialects = 10 :: pos_integer(),
    frequency_window_ms = 60000 :: pos_integer(),
    max_frequency = 100 :: pos_integer(),
    history_limit = 1000 :: pos_integer(),
    l2_update_interval_ms = 30000 :: pos_integer()
}).
-type communication_config() :: #communication_config{}.

-endif.
```

---

## Core Silo Implementation

```erlang
%%%-------------------------------------------------------------------
%%% @doc Communication Silo Core
%%% Manages signaling and language evolution for the population.
%%% @end
%%%-------------------------------------------------------------------
-module(communication_silo).
-behaviour(gen_server).

%% API
-export([
    start_link/1,
    enable/1,
    disable/1,
    is_enabled/1,
    get_state/1,
    update_from_l2/2,
    record_message/2,
    record_signal_invention/2,
    record_coordination_attempt/2,
    record_deception_event/2,
    get_signal_registry/1,
    get_dialect_info/1
]).

%% gen_server callbacks
-export([
    init/1,
    handle_call/3,
    handle_cast/2,
    handle_info/2,
    terminate/2,
    code_change/3
]).

-include("communication_silo.hrl").

-define(SERVER(PopId), {via, gproc, {n, l, {?MODULE, PopId}}}).

%%====================================================================
%% API
%%====================================================================

-spec start_link(Config :: map()) -> {ok, pid()} | {error, term()}.
start_link(Config) ->
    PopulationId = maps:get(population_id, Config),
    gen_server:start_link(?SERVER(PopulationId), ?MODULE, Config, []).

-spec enable(PopulationId :: binary()) -> ok.
enable(PopulationId) ->
    gen_server:call(?SERVER(PopulationId), enable).

-spec disable(PopulationId :: binary()) -> ok.
disable(PopulationId) ->
    gen_server:call(?SERVER(PopulationId), disable).

-spec is_enabled(PopulationId :: binary()) -> boolean().
is_enabled(PopulationId) ->
    gen_server:call(?SERVER(PopulationId), is_enabled).

-spec get_state(PopulationId :: binary()) -> communication_state().
get_state(PopulationId) ->
    gen_server:call(?SERVER(PopulationId), get_state).

-spec update_from_l2(PopulationId :: binary(), L2Guidance :: map()) -> ok.
update_from_l2(PopulationId, L2Guidance) ->
    gen_server:cast(?SERVER(PopulationId), {update_from_l2, L2Guidance}).

-spec record_message(PopulationId :: binary(), Message :: message()) -> ok.
record_message(PopulationId, Message) ->
    gen_server:cast(?SERVER(PopulationId), {record_message, Message}).

-spec record_signal_invention(PopulationId :: binary(), Signal :: signal()) -> ok.
record_signal_invention(PopulationId, Signal) ->
    gen_server:cast(?SERVER(PopulationId), {record_signal_invention, Signal}).

-spec record_coordination_attempt(PopulationId :: binary(),
                                  Attempt :: coordination_attempt()) -> ok.
record_coordination_attempt(PopulationId, Attempt) ->
    gen_server:cast(?SERVER(PopulationId), {record_coordination_attempt, Attempt}).

-spec record_deception_event(PopulationId :: binary(),
                             Event :: deception_event()) -> ok.
record_deception_event(PopulationId, Event) ->
    gen_server:cast(?SERVER(PopulationId), {record_deception_event, Event}).

-spec get_signal_registry(PopulationId :: binary()) ->
    #{binary() => signal()}.
get_signal_registry(PopulationId) ->
    gen_server:call(?SERVER(PopulationId), get_signal_registry).

-spec get_dialect_info(PopulationId :: binary()) ->
    #{binary() => dialect()}.
get_dialect_info(PopulationId) ->
    gen_server:call(?SERVER(PopulationId), get_dialect_info).

%%====================================================================
%% gen_server Callbacks
%%====================================================================

init(Config) ->
    PopulationId = maps:get(population_id, Config),
    {ok, SensorState} = communication_silo_sensors:init(Config),
    {ok, ActuatorState} = communication_silo_actuators:init(Config),

    State = #communication_state{
        enabled = maps:get(enabled, Config, true),
        population_id = PopulationId,
        signal_registry = #{},
        dialects = #{},
        message_history = [],
        coordination_history = [],
        deception_history = [],
        current_actuator_values = maps:get(current_values, ActuatorState, #{}),
        l2_guidance = #{},
        last_update = erlang:system_time(millisecond)
    },

    schedule_collection(),
    {ok, #{state => State, sensors => SensorState, actuators => ActuatorState}}.

handle_call(enable, _From, #{state := State} = FullState) ->
    NewState = State#communication_state{enabled = true},
    emit_event(communication_silo_enabled, #{
        population_id => State#communication_state.population_id
    }),
    {reply, ok, FullState#{state => NewState}};

handle_call(disable, _From, #{state := State} = FullState) ->
    NewState = State#communication_state{enabled = false},
    emit_event(communication_silo_disabled, #{
        population_id => State#communication_state.population_id
    }),
    {reply, ok, FullState#{state => NewState}};

handle_call(is_enabled, _From, #{state := State} = FullState) ->
    {reply, State#communication_state.enabled, FullState};

handle_call(get_state, _From, #{state := State} = FullState) ->
    {reply, State, FullState};

handle_call(get_signal_registry, _From, #{state := State} = FullState) ->
    {reply, State#communication_state.signal_registry, FullState};

handle_call(get_dialect_info, _From, #{state := State} = FullState) ->
    {reply, State#communication_state.dialects, FullState};

handle_call(_Request, _From, State) ->
    {reply, {error, unknown_request}, State}.

handle_cast({update_from_l2, L2Guidance}, #{state := State} = FullState) ->
    NewState = apply_l2_guidance(State, L2Guidance),
    {noreply, FullState#{state => NewState}};

handle_cast({record_message, Message}, #{state := State} = FullState) ->
    History = State#communication_state.message_history,
    TrimmedHistory = lists:sublist([Message | History], 1000),
    NewState = State#communication_state{message_history = TrimmedHistory},
    emit_event(message_sent, #{
        sender_id => Message#message.sender_id,
        receiver_ids => Message#message.receiver_ids,
        signal_count => length(Message#message.signals)
    }),
    {noreply, FullState#{state => NewState}};

handle_cast({record_signal_invention, Signal}, #{state := State} = FullState) ->
    Registry = State#communication_state.signal_registry,
    NewRegistry = Registry#{Signal#signal.id => Signal},
    NewState = State#communication_state{signal_registry = NewRegistry},
    emit_event(signal_invented, #{
        signal_id => Signal#signal.id,
        creator_id => Signal#signal.creator_id,
        meaning => Signal#signal.meaning
    }),
    {noreply, FullState#{state => NewState}};

handle_cast({record_coordination_attempt, Attempt},
            #{state := State} = FullState) ->
    History = State#communication_state.coordination_history,
    TrimmedHistory = lists:sublist([Attempt | History], 500),
    NewState = State#communication_state{coordination_history = TrimmedHistory},
    EventType = case Attempt#coordination_attempt.success of
        true -> coordination_succeeded;
        false -> coordination_failed
    end,
    emit_event(EventType, #{
        participant_ids => Attempt#coordination_attempt.participant_ids,
        action => Attempt#coordination_attempt.action,
        outcome => Attempt#coordination_attempt.outcome
    }),
    {noreply, FullState#{state => NewState}};

handle_cast({record_deception_event, Event}, #{state := State} = FullState) ->
    History = State#communication_state.deception_history,
    TrimmedHistory = lists:sublist([Event | History], 500),
    NewState = State#communication_state{deception_history = TrimmedHistory},
    case Event#deception_event.detected of
        true ->
            emit_event(deception_detected, #{
                liar_id => Event#deception_event.liar_id,
                detector_id => Event#deception_event.detector_id,
                signal => Event#deception_event.signal
            });
        false ->
            ok
    end,
    {noreply, FullState#{state => NewState}};

handle_cast(_Msg, State) ->
    {noreply, State}.

handle_info(collect_and_control, #{state := State, sensors := SensorState} = FullState) ->
    case State#communication_state.enabled of
        true ->
            {SensorValues, NewSensorState} = collect_all_sensors(SensorState),
            ActuatorOutputs = run_controller(SensorValues, State),
            NewState = apply_actuator_outputs(State, ActuatorOutputs),
            send_cross_silo_signals(SensorValues, State),
            schedule_collection(),
            {noreply, FullState#{state => NewState, sensors => NewSensorState}};
        false ->
            schedule_collection(),
            {noreply, FullState}
    end;

handle_info(_Info, State) ->
    {noreply, State}.

terminate(_Reason, _State) ->
    ok.

code_change(_OldVsn, State, _Extra) ->
    {ok, State}.

%%====================================================================
%% Internal Functions
%%====================================================================

schedule_collection() ->
    erlang:send_after(1000, self(), collect_and_control).

collect_all_sensors(SensorState) ->
    SensorIds = communication_silo_sensors:get_sensor_ids(),
    lists:foldl(fun(SensorId, {Acc, State}) ->
        {Value, NewState} = communication_silo_sensors:collect(SensorId, State),
        NormValue = communication_silo_sensors:normalize(SensorId, Value),
        {Acc#{SensorId => NormValue}, NewState}
    end, {#{}, SensorState}, SensorIds).

run_controller(SensorValues, State) ->
    L2Guidance = State#communication_state.l2_guidance,
    SensorList = maps:to_list(SensorValues),
    InputVector = [V || {_, V} <- lists:sort(SensorList)],
    OutputVector = tweann_controller:forward(communication_silo_nn, InputVector),
    ActuatorIds = communication_silo_actuators:get_actuator_ids(),
    RawOutputs = lists:zip(ActuatorIds, OutputVector),
    apply_l2_modulation(RawOutputs, L2Guidance).

apply_l2_modulation(RawOutputs, L2Guidance) ->
    lists:map(fun({ActuatorId, RawValue}) ->
        Spec = communication_silo_actuators:get_actuator_spec(ActuatorId),
        {Min, Max} = Spec#actuator_spec.range,
        BaseValue = Min + RawValue * (Max - Min),
        ModulatedValue = case maps:get(ActuatorId, L2Guidance, undefined) of
            undefined -> BaseValue;
            Modulation -> clamp(BaseValue * Modulation, Min, Max)
        end,
        {ActuatorId, ModulatedValue}
    end, RawOutputs).

apply_actuator_outputs(State, Outputs) ->
    CurrentValues = State#communication_state.current_actuator_values,
    NewValues = lists:foldl(fun({ActuatorId, Value}, Acc) ->
        apply_to_population(ActuatorId, Value),
        Acc#{ActuatorId => Value}
    end, CurrentValues, Outputs),
    State#communication_state{
        current_actuator_values = NewValues,
        last_update = erlang:system_time(millisecond)
    }.

apply_to_population(ActuatorId, Value) ->
    communication_silo_actuators:apply(ActuatorId, Value, #{}).

apply_l2_guidance(State, L2Guidance) ->
    emit_event(l2_guidance_received, #{
        guidance => L2Guidance,
        population_id => State#communication_state.population_id
    }),
    State#communication_state{l2_guidance = L2Guidance}.

send_cross_silo_signals(SensorValues, State) ->
    PopId = State#communication_state.population_id,

    %% Signal to Task Silo: coordination capability
    CoordSuccess = maps:get(coordination_success_rate, SensorValues, 0.5),
    task_silo:receive_signal(PopId, coordination_capability, CoordSuccess),

    %% Signal to Cultural Silo: information sharing
    CommFreq = maps:get(communication_frequency, SensorValues, 0.5),
    cultural_silo:receive_signal(PopId, information_sharing, CommFreq),

    %% Signal to Social Silo: trust network
    Honesty = maps:get(signal_honesty_rate, SensorValues, 0.5),
    social_silo:receive_signal(PopId, trust_network, Honesty).

emit_event(EventType, Payload) ->
    Event = #{
        type => EventType,
        payload => Payload,
        timestamp => erlang:system_time(millisecond),
        silo => communication
    },
    erl_esdb:append(communication_silo_events, Event).

clamp(Value, Min, Max) ->
    max(Min, min(Max, Value)).
```

---

## Target

**Population** - Communication is a population-level phenomenon requiring multiple interacting agents.

---

## Cross-Silo Signals

### Complete 13-Silo Interaction Matrix

| From ↓ To → | Task | Resource | Distrib | Social | Cultural | Ecological | Morpho | Temporal | Competitive | Develop | Regulatory | Economic | Comm |
|-------------|------|----------|---------|--------|----------|------------|--------|----------|-------------|---------|------------|----------|------|
| **Communication** | coord | - | - | trust | sharing | - | - | - | signals | learning | - | trade | - |

### Outgoing Signals

| Signal | To Silo | Meaning | Implementation |
|--------|---------|---------|----------------|
| `coordination_capability` | Task | High coordination = team tasks possible | `coordination_success_rate` sensor value |
| `information_sharing` | Cultural | Communication enables cultural transfer | `communication_frequency` sensor value |
| `trust_network` | Social | Honest signals build trust | `signal_honesty_rate` sensor value |
| `language_complexity` | Developmental | Complex language may extend learning | `message_complexity_mean` sensor |
| `trade_communication` | Economic | Communication enables trade | `coordination_success_rate * communication_frequency` |
| `competitive_signals` | Competitive | Strategic signaling in competition | Signals used in competitive contexts |

### Incoming Signals

| Signal | From Silo | Effect | Implementation |
|--------|-----------|--------|----------------|
| `social_network_density` | Social | Dense networks = more communication | Increases `communication_frequency` baseline |
| `coalition_structure` | Social | Coalitions develop shared signals | Increases `dialect_isolation` |
| `cultural_diversity` | Cultural | Diversity creates dialects | Affects dialect formation rate |
| `population_structure` | Distribution | Isolated groups form dialects | Modulates `dialect_isolation` |
| `stress_level` | Ecological | Stress may increase honest signaling | Modulates `lying_penalty` |

### Signal API

```erlang
%% Receiving signals from other silos
-spec receive_signal(PopulationId :: binary(),
                     SignalType :: atom(),
                     Value :: float()) -> ok.
receive_signal(PopulationId, social_network_density, Value) ->
    gen_server:cast(?SERVER(PopulationId),
                    {external_signal, social_network_density, Value});
receive_signal(PopulationId, coalition_structure, Value) ->
    gen_server:cast(?SERVER(PopulationId),
                    {external_signal, coalition_structure, Value});
receive_signal(PopulationId, cultural_diversity, Value) ->
    gen_server:cast(?SERVER(PopulationId),
                    {external_signal, cultural_diversity, Value});
receive_signal(PopulationId, stress_level, Value) ->
    gen_server:cast(?SERVER(PopulationId),
                    {external_signal, stress_level, Value}).
```

---

## Events Emitted

### Event Types

| Event | Trigger | Payload |
|-------|---------|---------|
| `signal_invented` | New signal created | `{inventor_id, signal, meaning}` |
| `signal_adopted` | Signal spread to another | `{adopter_id, signal, source_id}` |
| `message_sent` | Communication occurred | `{sender_id, receiver_ids, signal_count}` |
| `deception_detected` | Lie caught | `{detector_id, liar_id, signal}` |
| `coordination_succeeded` | Team action worked | `{participant_ids, action, outcome}` |
| `coordination_failed` | Team action failed | `{participant_ids, action, reason}` |
| `dialect_emerged` | New dialect formed | `{dialect_id, members, vocabulary}` |
| `language_merged` | Dialects combined | `{dialect_a, dialect_b, result}` |
| `communication_silo_enabled` | Silo turned on | `{population_id}` |
| `communication_silo_disabled` | Silo turned off | `{population_id}` |

### Event Payloads

```erlang
%% Signal invented
#{
    type => signal_invented,
    signal_id => <<"sig_001">>,
    creator_id => <<"ind_123">>,
    meaning => {food_location, {10, 20}},
    vocabulary_size => 45,
    timestamp => 1703318400000
}

%% Coordination succeeded
#{
    type => coordination_succeeded,
    participant_ids => [<<"ind_001">>, <<"ind_002">>, <<"ind_003">>],
    action => group_hunt,
    messages_exchanged => 12,
    outcome => #{prey_captured => true, food_shared => 100},
    timestamp => 1703318400000
}

%% Deception detected
#{
    type => deception_detected,
    liar_id => <<"ind_007">>,
    detector_id => <<"ind_042">>,
    signal => <<"sig_food_here">>,
    claimed_state => food_present,
    actual_state => no_food,
    timestamp => 1703318400000
}

%% Dialect emerged
#{
    type => dialect_emerged,
    dialect_id => <<"dialect_003">>,
    parent_dialect => <<"dialect_001">>,
    members => [<<"ind_010">>, <<"ind_011">>, <<"ind_012">>],
    vocabulary_size => 23,
    isolation_score => 0.65,
    timestamp => 1703318400000
}
```

---

## Value of Event Storage

### Analysis Capabilities

| Analysis Type | Events Used | Insight |
|--------------|-------------|---------|
| **Language Evolution** | `signal_invented`, `signal_adopted` | Track how communication protocols emerge |
| **Coordination Patterns** | `coordination_succeeded`, `coordination_failed` | Identify successful coordination strategies |
| **Deception Dynamics** | `deception_detected`, honesty rates | Understand arms race between lying and detection |
| **Protocol Discovery** | All message events | Mine events for useful protocols |
| **Interpretability** | Signal registry, meanings | Map signals to meanings for explainability |
| **Dialect Formation** | `dialect_emerged`, `language_merged` | Track subgroup communication divergence |

### Replay Value

- **Language reconstruction**: Replay to understand how protocols emerged
- **What-if analysis**: Test different honesty pressures
- **Transfer learning**: Extract successful protocols for new populations
- **Debugging**: Trace coordination failures to communication breakdowns

---

## Multi-Layer Hierarchy

| Level | Role | Communication Responsibilities |
|-------|------|-------------------------------|
| **L0** | Hard limits | Minimum vocabulary, maximum message cost, honesty floor |
| **L1** | Tactical | Adjust coordination rewards based on current task demands |
| **L2** | Strategic | Learn optimal communication policies across task classes |

### L2 Guidance Integration

```erlang
%% L2 provides strategic communication policies
-record(l2_communication_guidance, {
    coordination_emphasis :: float(),     % 0-2x multiplier
    honesty_pressure :: float(),          % 0-2x multiplier
    dialect_tolerance :: float(),         % 0-2x multiplier
    compression_priority :: float(),      % 0-2x multiplier
    vocabulary_budget :: float()          % 0-2x multiplier
}).

%% Example L2 guidance for cooperative tasks
#{
    coordination_reward => 1.5,           % Emphasize coordination
    lying_penalty => 1.2,                 % Stricter honesty
    dialect_isolation => 0.5,             % Discourage fragmentation
    message_compression_pressure => 0.8,  % Allow verbose messages
    vocabulary_growth_rate => 1.2         % Encourage new signals
}

%% Example L2 guidance for competitive tasks
#{
    coordination_reward => 0.8,           % Less coordination emphasis
    lying_penalty => 0.7,                 % Allow strategic deception
    dialect_isolation => 1.3,             % Allow secret codes
    message_compression_pressure => 1.2,  % Efficient signals
    vocabulary_growth_rate => 0.9         % Stable vocabulary
}
```

---

## Enable/Disable Effects

### When OFF

| Aspect | Effect |
|--------|--------|
| **Communication** | No evolved communication (hard-coded only) |
| **Coordination** | No coordination through signals |
| **Deception** | No deception/detection dynamics |
| **Protocols** | No emergent protocols |
| **Dialects** | No dialect formation |
| **Vocabulary** | Fixed signal meanings |

### When ON

| Aspect | Effect |
|--------|--------|
| **Communication** | Signals evolve meanings |
| **Coordination** | Coordination through communication |
| **Deception** | Honesty/deception dynamics |
| **Protocols** | Protocols emerge naturally |
| **Dialects** | Subgroups develop distinct languages |
| **Vocabulary** | Vocabulary grows and adapts |

### Switching Effects

**Turning ON mid-evolution:**
- Initial vocabulary bootstrap required
- Coordination may be poor until signals stabilize
- Fitness may temporarily drop as communication overhead added

**Turning OFF mid-evolution:**
- Evolved protocols frozen (no further adaptation)
- Coordination reverts to hard-coded mechanisms
- May lose adaptation to current task

---

## Implementation Phases

### Phase 1: Core Infrastructure
- [ ] Basic signal registry
- [ ] Message recording
- [ ] Sensor implementation
- [ ] Actuator implementation

### Phase 2: Language Evolution
- [ ] Signal invention mechanism
- [ ] Signal adoption tracking
- [ ] Vocabulary growth control
- [ ] Language stability monitoring

### Phase 3: Coordination & Honesty
- [ ] Coordination attempt tracking
- [ ] Deception detection system
- [ ] Honesty rate calculation
- [ ] Coordination reward distribution

### Phase 4: Dialect System
- [ ] Dialect identification
- [ ] Dialect membership tracking
- [ ] Dialect merger detection
- [ ] Isolation scoring

### Phase 5: Integration
- [ ] Cross-silo signal routing
- [ ] L2 guidance integration
- [ ] Event storage (erl-esdb)
- [ ] Controller neural network

---

## Success Criteria

### Functional Requirements
- [ ] All 10 L0 sensors collecting valid data
- [ ] All 8 L0 actuators applying parameters
- [ ] Signal registry tracking invented signals
- [ ] Deception detection functioning
- [ ] Coordination success measured accurately

### Performance Requirements
- [ ] Sensor collection < 10ms per cycle
- [ ] Actuator application < 5ms per parameter
- [ ] Event storage latency < 50ms
- [ ] Cross-silo signal latency < 100ms

### Integration Requirements
- [ ] Signals sent to Task, Cultural, Social silos
- [ ] Signals received from Social, Cultural, Distribution, Ecological silos
- [ ] L2 guidance modulates actuator outputs
- [ ] Enable/disable affects all communication dynamics

### Emergent Behavior Requirements
- [ ] Vocabulary size increases over evolution
- [ ] Honesty rate stable when lying penalty appropriate
- [ ] Coordination success improves with communication
- [ ] Dialects emerge in isolated subpopulations

---

## References

- Nowak, M.A., & Krakauer, D.C. (1999). The evolution of language. PNAS.
- Steels, L. (2011). Modeling the cultural evolution of language. Physics of Life Reviews.
- Skyrms, B. (2010). Signals: Evolution, Learning, and Information.
- Wagner, K. et al. (2003). Progress in the simulation of emergent communication and language.
