# PLAN_MORPHOLOGICAL_SILO.md

**Status:** Planning
**Created:** 2025-12-23
**Last Updated:** 2025-12-23
**Related:** PLAN_BEHAVIORAL_EVENTS.md, PLAN_L2_L1_HIERARCHICAL_INTERFACE.md, PLAN_ECONOMIC_SILO.md

---

## Overview

This document specifies the **Morphological Silo** for the Liquid Conglomerate (LC) meta-controller. The Morphological Silo manages the "body plan" of neural networks - not just weights and topology, but the fundamental structural constraints: neuron budgets, connection limits, sensor/actuator counts, and complexity penalties.

### Purpose

The Morphological Silo extends the LC architecture to control structural evolution:

1. **Size Management**: Control maximum neurons and connections
2. **Efficiency Pressure**: Penalize bloated networks
3. **Pruning Control**: Remove unused structure automatically
4. **Modularity**: Encourage modular, symmetric architectures
5. **Hardware Targeting**: Evolve networks for specific deployment constraints

### Business Value

- **Edge deployment**: Evolve minimal networks for resource-constrained devices
- **Cost optimization**: Smaller networks = cheaper inference
- **Hardware adaptation**: Different "bodies" for different deployment targets
- **AutoML extension**: Architecture search beyond hyperparameters

### Training Velocity & Inference Impact

| Metric | Without Morphological Silo | With Morphological Silo |
|--------|---------------------------|------------------------|
| Network bloat | Unbounded growth | Constrained by budgets |
| Inference latency | Variable, often high | Predictable, optimized |
| Training velocity | Neutral (1.0x) | Neutral to slight slowdown (0.9-1.0x) |
| Memory footprint | Unpredictable | Bounded by max_neurons/max_conns |
| Deployment success rate | ~60% fit hardware | ~95% fit hardware |

**Note:** Training velocity is neutral because size constraints require additional fitness evaluations but eliminate wasted evolution of oversized networks. Inference speed improves significantly due to smaller, more efficient networks.

### LC Architecture Context

```
┌─────────────────────────────────────────────────────────────────────┐
│                    LIQUID CONGLOMERATE                               │
├─────────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐ │
│  │  Task Silo  │  │Resource Silo│  │Morphological│  │Economic Silo│ │
│  │  (existing) │  │  (existing) │  │    (NEW)    │  │    (NEW)    │ │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘ │
│         │                │                │                │        │
│         └────────────────┴────────────────┴────────────────┘        │
│                                   │                                  │
│                          ┌────────▼────────┐                        │
│                          │  LC Controller  │                        │
│                          │    (TWEANN)     │                        │
│                          └─────────────────┘                        │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Silo Architecture

### Morphological Silo Structure

```
┌─────────────────────────────────────────────────────────────────────┐
│                      MORPHOLOGICAL SILO                              │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │                      L0 SENSORS (10)                         │    │
│  │                                                              │    │
│  │  Size Metrics      Efficiency     Complexity    Growth       │    │
│  │  ┌─────────┐      ┌─────────┐    ┌─────────┐  ┌─────────┐   │    │
│  │  │neuron_  │      │fitness_ │    │modularity│ │growth_  │   │    │
│  │  │count    │      │per_param│    │_score    │ │rate     │   │    │
│  │  │conn_    │      │param_   │    │symmetry_ │ │pruning_ │   │    │
│  │  │count    │      │efficiency│   │index     │ │pressure │   │    │
│  │  │sensor_  │      └─────────┘    └─────────┘  └─────────┘   │    │
│  │  │count    │                                                 │    │
│  │  │actuator_│                                                 │    │
│  │  │count    │                                                 │    │
│  │  └─────────┘                                                 │    │
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
│  │  Size Limits       Rates          Penalties                  │    │
│  │  ┌─────────┐      ┌─────────┐    ┌─────────┐                │    │
│  │  │max_     │      │sensor_  │    │complexity│                │    │
│  │  │neurons  │      │add_rate │    │_penalty  │                │    │
│  │  │max_     │      │actuator_│    │size_     │                │    │
│  │  │conns    │      │add_rate │    │penalty   │                │    │
│  │  │min_     │      │pruning_ │    └─────────┘                │    │
│  │  │neurons  │      │threshold│                                │    │
│  │  └─────────┘      └─────────┘                                │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## L0 Sensors

### Sensor Specifications

```erlang
-module(morphological_silo_sensors).
-behaviour(l0_sensor_behaviour).

%% Sensor definitions
-export([sensor_specs/0, read_sensors/1]).

sensor_specs() ->
    [
        %% Size metrics
        #{
            id => neuron_count_norm,
            range => {0.0, 1.0},
            description => "Current neurons / max allowed neurons"
        },
        #{
            id => connection_count_norm,
            range => {0.0, 1.0},
            description => "Current connections / max possible connections"
        },
        #{
            id => sensor_count_norm,
            range => {0.0, 1.0},
            description => "Input neurons / max input budget"
        },
        #{
            id => actuator_count_norm,
            range => {0.0, 1.0},
            description => "Output neurons / max output budget"
        },

        %% Efficiency metrics
        #{
            id => fitness_per_parameter,
            range => {0.0, 1.0},
            description => "Fitness / total_parameters (efficiency metric)"
        },
        #{
            id => parameter_efficiency_trend,
            range => {-1.0, 1.0},
            description => "Change in efficiency over generations"
        },

        %% Complexity metrics
        #{
            id => modularity_score,
            range => {0.0, 1.0},
            description => "Clustering coefficient of network graph"
        },
        #{
            id => symmetry_index,
            range => {0.0, 1.0},
            description => "Structural symmetry (bilateral, radial)"
        },

        %% Growth metrics
        #{
            id => growth_rate,
            range => {-1.0, 1.0},
            description => "Rate of size increase per generation"
        },
        #{
            id => pruning_pressure,
            range => {0.0, 1.0},
            description => "How much unused structure exists"
        }
    ].

read_sensors(#morphological_state{} = State) ->
    [
        normalize(neuron_count_norm,
            State#morphological_state.neuron_count /
            State#morphological_state.max_neurons),
        normalize(connection_count_norm,
            State#morphological_state.connection_count /
            State#morphological_state.max_connections),
        normalize(sensor_count_norm,
            State#morphological_state.sensor_count /
            State#morphological_state.max_sensors),
        normalize(actuator_count_norm,
            State#morphological_state.actuator_count /
            State#morphological_state.max_actuators),
        normalize(fitness_per_parameter,
            State#morphological_state.fitness_per_param),
        normalize(parameter_efficiency_trend,
            State#morphological_state.efficiency_trend),
        normalize(modularity_score,
            State#morphological_state.modularity),
        normalize(symmetry_index,
            State#morphological_state.symmetry),
        normalize(growth_rate,
            State#morphological_state.growth_rate),
        normalize(pruning_pressure,
            State#morphological_state.pruning_pressure)
    ].

normalize(SensorId, Value) ->
    {Min, Max} = get_range(SensorId),
    clamp((Value - Min) / (Max - Min), 0.0, 1.0).

get_range(SensorId) ->
    Spec = lists:keyfind(SensorId, #{id := _}, sensor_specs()),
    maps:get(range, Spec).

clamp(V, Min, Max) -> max(Min, min(Max, V)).
```

---

## L0 Actuators

### Actuator Specifications

```erlang
-module(morphological_silo_actuators).
-behaviour(l0_actuator_behaviour).

-export([actuator_specs/0, apply_actuators/2]).

actuator_specs() ->
    [
        %% Size limits
        #{
            id => max_neurons,
            range => {10, 1000},
            default => 100,
            description => "Maximum hidden neurons allowed"
        },
        #{
            id => max_connections,
            range => {20, 10000},
            default => 500,
            description => "Maximum synaptic connections"
        },
        #{
            id => min_neurons,
            range => {1, 50},
            default => 5,
            description => "Minimum hidden neurons required"
        },

        %% Addition rates
        #{
            id => sensor_addition_rate,
            range => {0.0, 0.1},
            default => 0.01,
            description => "Probability of adding input neuron"
        },
        #{
            id => actuator_addition_rate,
            range => {0.0, 0.1},
            default => 0.01,
            description => "Probability of adding output neuron"
        },

        %% Pruning
        #{
            id => pruning_threshold,
            range => {0.0, 0.5},
            default => 0.1,
            description => "Weight magnitude below which to prune"
        },

        %% Penalties
        #{
            id => complexity_penalty,
            range => {0.0, 0.1},
            default => 0.01,
            description => "Fitness penalty per parameter"
        },
        #{
            id => size_penalty_exponent,
            range => {1.0, 3.0},
            default => 1.5,
            description => "How aggressively to penalize size"
        }
    ].

apply_actuators(Outputs, #morphological_config{} = Config) ->
    [MaxNeurons, MaxConns, MinNeurons, SensorRate, ActuatorRate,
     PruneThresh, ComplexityPen, SizePenExp] = Outputs,

    Config#morphological_config{
        max_neurons = denormalize(max_neurons, MaxNeurons),
        max_connections = denormalize(max_connections, MaxConns),
        min_neurons = denormalize(min_neurons, MinNeurons),
        sensor_addition_rate = denormalize(sensor_addition_rate, SensorRate),
        actuator_addition_rate = denormalize(actuator_addition_rate, ActuatorRate),
        pruning_threshold = denormalize(pruning_threshold, PruneThresh),
        complexity_penalty = denormalize(complexity_penalty, ComplexityPen),
        size_penalty_exponent = denormalize(size_penalty_exponent, SizePenExp)
    }.

denormalize(ActuatorId, NormalizedValue) ->
    Spec = lists:keyfind(ActuatorId, #{id := _}, actuator_specs()),
    {Min, Max} = maps:get(range, Spec),
    Min + NormalizedValue * (Max - Min).
```

---

## Records

### State Records

```erlang
%% In morphological_silo.hrl

-record(morphological_state, {
    %% Current size metrics
    neuron_count :: non_neg_integer(),
    connection_count :: non_neg_integer(),
    sensor_count :: non_neg_integer(),
    actuator_count :: non_neg_integer(),

    %% Limits (from config)
    max_neurons :: non_neg_integer(),
    max_connections :: non_neg_integer(),
    max_sensors :: non_neg_integer(),
    max_actuators :: non_neg_integer(),

    %% Efficiency metrics
    fitness_per_param :: float(),
    efficiency_trend :: float(),

    %% Complexity metrics
    modularity :: float(),
    symmetry :: float(),

    %% Growth metrics
    growth_rate :: float(),
    pruning_pressure :: float(),

    %% History
    size_history :: [{integer(), non_neg_integer()}],  % {gen, total_params}
    efficiency_history :: [{integer(), float()}]       % {gen, efficiency}
}).

-record(morphological_config, {
    enabled = true :: boolean(),

    %% Size limits
    max_neurons = 100 :: non_neg_integer(),
    max_connections = 500 :: non_neg_integer(),
    min_neurons = 5 :: non_neg_integer(),

    %% Addition rates
    sensor_addition_rate = 0.01 :: float(),
    actuator_addition_rate = 0.01 :: float(),

    %% Pruning
    pruning_threshold = 0.1 :: float(),

    %% Penalties
    complexity_penalty = 0.01 :: float(),
    size_penalty_exponent = 1.5 :: float()
}).

-record(network_morphology, {
    individual_id :: binary(),
    neuron_count :: non_neg_integer(),
    connection_count :: non_neg_integer(),
    input_count :: non_neg_integer(),
    output_count :: non_neg_integer(),
    hidden_count :: non_neg_integer(),
    layer_depths :: [non_neg_integer()],
    modularity_coefficient :: float(),
    symmetry_score :: float(),
    parameter_count :: non_neg_integer(),
    measured_at :: integer()
}).
```

---

## Core Systems

### Size Management

```erlang
-module(morphological_size).

%% Check if network is within size limits
-spec check_size_limits(NetworkId, Config) -> ok | {violation, Reason}
    when NetworkId :: binary(),
         Config :: #morphological_config{},
         Reason :: atom().

check_size_limits(NetworkId, Config) ->
    Morphology = get_network_morphology(NetworkId),

    Violations = lists:filtermap(fun
        ({neurons, Count, Max}) when Count > Max ->
            {true, {neurons_exceeded, Count, Max}};
        ({connections, Count, Max}) when Count > Max ->
            {true, {connections_exceeded, Count, Max}};
        (_) ->
            false
    end, [
        {neurons, Morphology#network_morphology.hidden_count,
            Config#morphological_config.max_neurons},
        {connections, Morphology#network_morphology.connection_count,
            Config#morphological_config.max_connections}
    ]),

    case Violations of
        [] -> ok;
        [First | _] -> {violation, First}
    end.

%% Enforce size limits during mutation
-spec enforce_limits(MutationOp, NetworkId, Config) -> allow | deny
    when MutationOp :: add_neuron | add_connection | any(),
         NetworkId :: binary(),
         Config :: #morphological_config{}.

enforce_limits(add_neuron, NetworkId, Config) ->
    Morphology = get_network_morphology(NetworkId),
    case Morphology#network_morphology.hidden_count <
         Config#morphological_config.max_neurons of
        true -> allow;
        false -> deny
    end;

enforce_limits(add_connection, NetworkId, Config) ->
    Morphology = get_network_morphology(NetworkId),
    case Morphology#network_morphology.connection_count <
         Config#morphological_config.max_connections of
        true -> allow;
        false -> deny
    end;

enforce_limits(_OtherOp, _NetworkId, _Config) ->
    allow.
```

### Efficiency Calculation

```erlang
-module(morphological_efficiency).

%% Calculate fitness per parameter
-spec calculate_efficiency(Fitness, ParameterCount) -> float()
    when Fitness :: float(),
         ParameterCount :: non_neg_integer().

calculate_efficiency(_Fitness, 0) -> 0.0;
calculate_efficiency(Fitness, ParameterCount) ->
    Fitness / ParameterCount.

%% Calculate efficiency trend over generations
-spec calculate_efficiency_trend(EfficiencyHistory) -> float()
    when EfficiencyHistory :: [{integer(), float()}].

calculate_efficiency_trend([]) -> 0.0;
calculate_efficiency_trend([_Single]) -> 0.0;
calculate_efficiency_trend(History) when length(History) >= 2 ->
    %% Use linear regression on last 10 generations
    Recent = lists:sublist(lists:reverse(History), 10),
    {Xs, Ys} = lists:unzip([{float(G), E} || {G, E} <- Recent]),
    linear_regression_slope(Xs, Ys).

linear_regression_slope(Xs, Ys) ->
    N = length(Xs),
    SumX = lists:sum(Xs),
    SumY = lists:sum(Ys),
    SumXY = lists:sum([X * Y || {X, Y} <- lists:zip(Xs, Ys)]),
    SumX2 = lists:sum([X * X || X <- Xs]),

    Denom = N * SumX2 - SumX * SumX,
    case Denom of
        0.0 -> 0.0;
        _ -> (N * SumXY - SumX * SumY) / Denom
    end.

%% Apply complexity penalty to fitness
-spec apply_complexity_penalty(Fitness, Morphology, Config) -> AdjustedFitness
    when Fitness :: float(),
         Morphology :: #network_morphology{},
         Config :: #morphological_config{},
         AdjustedFitness :: float().

apply_complexity_penalty(Fitness, Morphology, Config) ->
    ParamCount = Morphology#network_morphology.parameter_count,
    Penalty = Config#morphological_config.complexity_penalty,
    Exponent = Config#morphological_config.size_penalty_exponent,

    %% Penalty = penalty_rate * (param_count ^ exponent) / normalization
    PenaltyAmount = Penalty * math:pow(ParamCount, Exponent) / 1000,

    max(0.0, Fitness - PenaltyAmount).
```

### Pruning System

```erlang
-module(morphological_pruning).

%% Identify connections to prune
-spec identify_prunable_connections(NetworkId, Config) -> [ConnectionId]
    when NetworkId :: binary(),
         Config :: #morphological_config{},
         ConnectionId :: binary().

identify_prunable_connections(NetworkId, Config) ->
    Threshold = Config#morphological_config.pruning_threshold,
    Connections = get_all_connections(NetworkId),

    [Conn#connection.id || Conn <- Connections,
        abs(Conn#connection.weight) < Threshold].

%% Identify neurons with no active connections
-spec identify_orphan_neurons(NetworkId) -> [NeuronId]
    when NetworkId :: binary(),
         NeuronId :: binary().

identify_orphan_neurons(NetworkId) ->
    Neurons = get_all_neurons(NetworkId),
    Connections = get_all_connections(NetworkId),

    %% Find neurons with no incoming or outgoing connections
    ConnectedNeurons = lists:usort(
        lists:flatmap(fun(Conn) ->
            [Conn#connection.from_neuron_id,
             Conn#connection.to_neuron_id]
        end, Connections)
    ),

    [N#neuron.id || N <- Neurons,
        not lists:member(N#neuron.id, ConnectedNeurons),
        N#neuron.layer /= input,
        N#neuron.layer /= output].

%% Execute pruning
-spec prune_network(NetworkId, Config) -> PruningResult
    when NetworkId :: binary(),
         Config :: #morphological_config{},
         PruningResult :: #{connections_removed := non_neg_integer(),
                           neurons_removed := non_neg_integer()}.

prune_network(NetworkId, Config) ->
    %% Prune weak connections
    PrunableConns = identify_prunable_connections(NetworkId, Config),
    lists:foreach(fun(ConnId) ->
        remove_connection(NetworkId, ConnId)
    end, PrunableConns),

    %% Prune orphan neurons
    OrphanNeurons = identify_orphan_neurons(NetworkId),
    lists:foreach(fun(NeuronId) ->
        remove_neuron(NetworkId, NeuronId)
    end, OrphanNeurons),

    %% Emit event
    neuroevolution_events:emit(network_pruned, #{
        event_type => network_pruned,
        individual_id => NetworkId,
        neurons_removed => length(OrphanNeurons),
        connections_removed => length(PrunableConns),
        pruning_threshold => Config#morphological_config.pruning_threshold,
        pruned_at => erlang:system_time(microsecond)
    }),

    #{
        connections_removed => length(PrunableConns),
        neurons_removed => length(OrphanNeurons)
    }.

%% Calculate pruning pressure (amount of unused structure)
-spec calculate_pruning_pressure(NetworkId, Config) -> float()
    when NetworkId :: binary(),
         Config :: #morphological_config{}.

calculate_pruning_pressure(NetworkId, Config) ->
    Threshold = Config#morphological_config.pruning_threshold,
    Connections = get_all_connections(NetworkId),
    TotalConns = length(Connections),

    WeakConns = length([C || C <- Connections,
                        abs(C#connection.weight) < Threshold * 2]),

    case TotalConns of
        0 -> 0.0;
        _ -> WeakConns / TotalConns
    end.
```

### Modularity Calculation

```erlang
-module(morphological_modularity).

%% Calculate modularity score (clustering coefficient)
-spec calculate_modularity(NetworkId) -> float()
    when NetworkId :: binary().

calculate_modularity(NetworkId) ->
    Neurons = get_all_neurons(NetworkId),
    Connections = get_all_connections(NetworkId),

    %% Build adjacency map
    AdjMap = build_adjacency_map(Connections),

    %% Calculate local clustering coefficient for each neuron
    Coefficients = [local_clustering_coefficient(N#neuron.id, AdjMap)
                    || N <- Neurons, N#neuron.layer == hidden],

    case Coefficients of
        [] -> 0.0;
        _ -> lists:sum(Coefficients) / length(Coefficients)
    end.

build_adjacency_map(Connections) ->
    lists:foldl(fun(Conn, Acc) ->
        FromId = Conn#connection.from_neuron_id,
        ToId = Conn#connection.to_neuron_id,

        Acc1 = maps:update_with(FromId,
            fun(Neighbors) -> [ToId | Neighbors] end,
            [ToId], Acc),
        maps:update_with(ToId,
            fun(Neighbors) -> [FromId | Neighbors] end,
            [FromId], Acc1)
    end, #{}, Connections).

local_clustering_coefficient(NeuronId, AdjMap) ->
    Neighbors = maps:get(NeuronId, AdjMap, []),
    K = length(Neighbors),

    case K < 2 of
        true -> 0.0;
        false ->
            %% Count edges between neighbors
            EdgeCount = count_edges_between(Neighbors, AdjMap),
            MaxEdges = K * (K - 1) / 2,
            EdgeCount / MaxEdges
    end.

count_edges_between(Neighbors, AdjMap) ->
    Pairs = [{A, B} || A <- Neighbors, B <- Neighbors, A < B],
    length([{A, B} || {A, B} <- Pairs,
            lists:member(B, maps:get(A, AdjMap, []))]).
```

---

## Morphological Silo Server

### Module: `morphological_silo.erl`

```erlang
-module(morphological_silo).
-behaviour(gen_server).
-behaviour(lc_silo_behaviour).

-include("morphological_silo.hrl").

%% API
-export([
    start_link/1,
    enable/1,
    disable/1,
    is_enabled/1,
    read_sensors/1,
    apply_actuators/2,
    get_state/1,
    check_mutation/3,
    apply_penalty/3,
    trigger_pruning/2
]).

%% Silo behaviour callbacks
-export([
    init_silo/1,
    step/2,
    get_sensor_count/0,
    get_actuator_count/0
]).

%% gen_server callbacks
-export([init/1, handle_call/3, handle_cast/2, handle_info/2]).

-record(state, {
    population_id :: binary(),
    morphological_state :: #morphological_state{},
    config :: #morphological_config{},
    controller :: pid(),
    morphologies :: #{binary() => #network_morphology{}},
    generation :: non_neg_integer()
}).

%%====================================================================
%% API
%%====================================================================

start_link(Config) ->
    gen_server:start_link(?MODULE, Config, []).

enable(Pid) ->
    gen_server:call(Pid, enable).

disable(Pid) ->
    gen_server:call(Pid, disable).

is_enabled(Pid) ->
    gen_server:call(Pid, is_enabled).

check_mutation(Pid, MutationOp, NetworkId) ->
    gen_server:call(Pid, {check_mutation, MutationOp, NetworkId}).

apply_penalty(Pid, NetworkId, Fitness) ->
    gen_server:call(Pid, {apply_penalty, NetworkId, Fitness}).

trigger_pruning(Pid, NetworkId) ->
    gen_server:call(Pid, {trigger_pruning, NetworkId}).

%%====================================================================
%% Silo Behaviour Implementation
%%====================================================================

init_silo(Config) ->
    PopulationId = maps:get(population_id, Config),
    {ok, #state{
        population_id = PopulationId,
        morphological_state = initial_morphological_state(PopulationId),
        config = default_morphological_config(),
        morphologies = #{},
        generation = 0
    }}.

step(#state{config = #morphological_config{enabled = false}} = State, _Gen) ->
    {ok, State};

step(#state{} = State, Generation) ->
    %% 1. Update morphology metrics for all individuals
    Morphologies = update_all_morphologies(State#state.population_id),

    %% 2. Aggregate population morphology state
    NewMorphState = aggregate_morphological_state(
        Morphologies, State#state.morphological_state, Generation),

    %% 3. Read sensors
    SensorInputs = morphological_silo_sensors:read_sensors(NewMorphState),

    %% 4. Get controller outputs
    ActuatorOutputs = lc_controller:evaluate(
        State#state.controller, SensorInputs),

    %% 5. Apply actuator changes
    NewConfig = morphological_silo_actuators:apply_actuators(
        ActuatorOutputs, State#state.config),

    %% 6. Prune networks if needed
    prune_networks_if_needed(Morphologies, NewConfig),

    %% 7. Emit events for significant changes
    emit_morphology_events(State#state.morphological_state, NewMorphState),

    {ok, State#state{
        morphological_state = NewMorphState,
        config = NewConfig,
        morphologies = maps:from_list([
            {M#network_morphology.individual_id, M} || M <- Morphologies
        ]),
        generation = Generation
    }}.

get_sensor_count() -> 10.
get_actuator_count() -> 8.

%%====================================================================
%% gen_server callbacks
%%====================================================================

init(Config) ->
    {ok, State} = init_silo(Config),
    {ok, State}.

handle_call(enable, _From, State) ->
    NewConfig = (State#state.config)#morphological_config{enabled = true},
    {reply, ok, State#state{config = NewConfig}};

handle_call(disable, _From, State) ->
    NewConfig = (State#state.config)#morphological_config{enabled = false},
    {reply, ok, State#state{config = NewConfig}};

handle_call(is_enabled, _From, State) ->
    {reply, State#state.config#morphological_config.enabled, State};

handle_call({check_mutation, MutationOp, NetworkId}, _From, State) ->
    case State#state.config#morphological_config.enabled of
        false ->
            {reply, allow, State};
        true ->
            Result = morphological_size:enforce_limits(
                MutationOp, NetworkId, State#state.config),
            {reply, Result, State}
    end;

handle_call({apply_penalty, NetworkId, Fitness}, _From, State) ->
    case State#state.config#morphological_config.enabled of
        false ->
            {reply, Fitness, State};
        true ->
            Morphology = maps:get(NetworkId, State#state.morphologies,
                                  get_network_morphology(NetworkId)),
            AdjustedFitness = morphological_efficiency:apply_complexity_penalty(
                Fitness, Morphology, State#state.config),
            {reply, AdjustedFitness, State}
    end;

handle_call({trigger_pruning, NetworkId}, _From, State) ->
    case State#state.config#morphological_config.enabled of
        false ->
            {reply, #{connections_removed => 0, neurons_removed => 0}, State};
        true ->
            Result = morphological_pruning:prune_network(
                NetworkId, State#state.config),
            {reply, Result, State}
    end;

handle_call(_Request, _From, State) ->
    {reply, {error, unknown_request}, State}.

handle_cast(_Msg, State) ->
    {noreply, State}.

handle_info(_Info, State) ->
    {noreply, State}.

%%====================================================================
%% Internal Functions
%%====================================================================

initial_morphological_state(PopulationId) ->
    %% Initialize from current population
    Individuals = get_population_individuals(PopulationId),
    Morphologies = [get_network_morphology(I#individual.id) || I <- Individuals],

    aggregate_morphological_state(Morphologies, #morphological_state{
        max_neurons = 100,
        max_connections = 500,
        max_sensors = 50,
        max_actuators = 20,
        size_history = [],
        efficiency_history = []
    }, 0).

default_morphological_config() ->
    #morphological_config{
        enabled = true,
        max_neurons = 100,
        max_connections = 500,
        min_neurons = 5,
        sensor_addition_rate = 0.01,
        actuator_addition_rate = 0.01,
        pruning_threshold = 0.1,
        complexity_penalty = 0.01,
        size_penalty_exponent = 1.5
    }.

aggregate_morphological_state(Morphologies, PrevState, Generation) ->
    %% Calculate aggregates
    NeuronCounts = [M#network_morphology.neuron_count || M <- Morphologies],
    ConnCounts = [M#network_morphology.connection_count || M <- Morphologies],
    ParamCounts = [M#network_morphology.parameter_count || M <- Morphologies],
    Modularities = [M#network_morphology.modularity_coefficient || M <- Morphologies],

    AvgNeurons = safe_avg(NeuronCounts),
    AvgConns = safe_avg(ConnCounts),
    AvgParams = safe_avg(ParamCounts),
    AvgModularity = safe_avg(Modularities),

    %% Calculate efficiency (need fitness data)
    %% For now, use previous or calculate from available data

    %% Calculate growth rate from history
    NewSizeHistory = [{Generation, trunc(AvgParams)} |
                      lists:sublist(PrevState#morphological_state.size_history, 20)],
    GrowthRate = calculate_growth_rate(NewSizeHistory),

    PrevState#morphological_state{
        neuron_count = trunc(AvgNeurons),
        connection_count = trunc(AvgConns),
        sensor_count = safe_avg([M#network_morphology.input_count || M <- Morphologies]),
        actuator_count = safe_avg([M#network_morphology.output_count || M <- Morphologies]),
        modularity = AvgModularity,
        growth_rate = GrowthRate,
        size_history = NewSizeHistory
    }.

calculate_growth_rate([]) -> 0.0;
calculate_growth_rate([_]) -> 0.0;
calculate_growth_rate([{_G1, S1}, {_G2, S2} | _]) ->
    case S2 of
        0 -> 0.0;
        _ -> (S1 - S2) / S2
    end.

safe_avg([]) -> 0.0;
safe_avg(List) -> lists:sum(List) / length(List).

update_all_morphologies(PopulationId) ->
    Individuals = get_population_individuals(PopulationId),
    [get_network_morphology(I#individual.id) || I <- Individuals].

prune_networks_if_needed(Morphologies, Config) ->
    %% Prune networks over the threshold
    lists:foreach(fun(M) ->
        case M#network_morphology.neuron_count >
             Config#morphological_config.max_neurons of
            true ->
                morphological_pruning:prune_network(
                    M#network_morphology.individual_id, Config);
            false ->
                ok
        end
    end, Morphologies).

emit_morphology_events(OldState, NewState) ->
    %% Check for significant changes
    case NewState#morphological_state.neuron_count >
         OldState#morphological_state.neuron_count * 1.1 of
        true ->
            neuroevolution_events:emit(morphology_expanded, #{
                event_type => morphology_expanded,
                old_neuron_count => OldState#morphological_state.neuron_count,
                new_neuron_count => NewState#morphological_state.neuron_count,
                growth_rate => NewState#morphological_state.growth_rate,
                expanded_at => erlang:system_time(microsecond)
            });
        false ->
            ok
    end.
```

---

## Cross-Silo Signals

### Signals to Other Silos

| Signal | To Silo | Meaning |
|--------|---------|---------|
| `network_complexity` | Task | Complex networks may need different mutation rates |
| `size_budget_remaining` | Resource | Available growth room |
| `efficiency_score` | Economic | Cost-effectiveness of current morphology |

### Signals from Other Silos

| Signal | From Silo | Effect |
|--------|-----------|--------|
| `resource_pressure` | Resource | High pressure = tighter size limits |
| `stagnation_severity` | Task | Stagnation may warrant structural expansion |
| `computation_budget` | Economic | Budget constrains max size |

### Signal Implementation

```erlang
-module(morphological_signals).

%% Outgoing signals
-export([send_complexity_signal/2, send_efficiency_signal/2]).

%% Incoming signal handlers
-export([handle_resource_pressure/2, handle_stagnation/2, handle_budget/2]).

send_complexity_signal(MorphState, TaskSilo) ->
    Complexity = MorphState#morphological_state.neuron_count *
                 MorphState#morphological_state.connection_count / 10000,
    lc_cross_silo:send_signal(TaskSilo, network_complexity, Complexity).

send_efficiency_signal(MorphState, EconomicSilo) ->
    Efficiency = MorphState#morphological_state.fitness_per_param,
    lc_cross_silo:send_signal(EconomicSilo, efficiency_score, Efficiency).

handle_resource_pressure(Pressure, Config) when Pressure > 0.7 ->
    %% High resource pressure = reduce limits by 20%
    Config#morphological_config{
        max_neurons = trunc(Config#morphological_config.max_neurons * 0.8),
        max_connections = trunc(Config#morphological_config.max_connections * 0.8)
    };
handle_resource_pressure(_Pressure, Config) ->
    Config.

handle_stagnation(Severity, Config) when Severity > 0.5 ->
    %% High stagnation = allow more growth
    Config#morphological_config{
        max_neurons = trunc(Config#morphological_config.max_neurons * 1.2),
        complexity_penalty = Config#morphological_config.complexity_penalty * 0.5
    };
handle_stagnation(_Severity, Config) ->
    Config.

handle_budget(Budget, Config) when Budget < 0.3 ->
    %% Low budget = enforce smaller networks
    Config#morphological_config{
        complexity_penalty = Config#morphological_config.complexity_penalty * 2,
        size_penalty_exponent = min(3.0,
            Config#morphological_config.size_penalty_exponent + 0.5)
    };
handle_budget(_Budget, Config) ->
    Config.
```

---

## Events Emitted

| Event | Trigger |
|-------|---------|
| `morphology_constrained` | Size limits changed |
| `sensor_added` | New input neuron added |
| `actuator_added` | New output neuron added |
| `network_pruned` | Pruning occurred |
| `complexity_penalized` | Penalty applied to fitness |
| `efficiency_improved` | Better fitness/param ratio |
| `size_limit_reached` | Hit max neurons/connections |
| `morphology_expanded` | Significant size increase |

---

## Multi-Layer Considerations

| Level | Role |
|-------|------|
| L0 | Hard bounds: absolute min/max neurons, connections |
| L1 | Tactical: adjust limits based on current efficiency |
| L2 | Strategic: learn optimal size profiles for task classes |

### L2 Guidance for Morphological Silo

```erlang
%% In meta_controller.hrl - add morphological guidance

-record(l2_morphological_guidance, {
    %% Size control aggressiveness
    size_aggression = 0.5 :: float(),           % [0.0, 2.0]

    %% Pruning sensitivity
    pruning_sensitivity = 0.5 :: float(),       % [0.0, 1.0]

    %% Efficiency vs capability tradeoff
    efficiency_weight = 0.5 :: float(),         % [0.0, 1.0]

    %% Growth allowance
    growth_tolerance = 0.5 :: float()           % [0.0, 1.0]
}).
```

---

## On/Off Switching

**When OFF:**
- Network size evolves without constraint (can explode)
- No efficiency pressure (bloated networks)
- No pruning (dead connections accumulate)
- Complexity not penalized in fitness

**When ON:**
- Size bounded by learned limits
- Efficiency rewarded
- Automatic pruning of unused structure
- Smaller, faster networks emerge

**Switching Effect:** Turning ON mid-evolution will prune existing networks and constrain future growth. Turning OFF allows unconstrained exploration but may waste resources.

---

## Implementation Phases

- [ ] **Phase 1:** State records and morphology measurement
- [ ] **Phase 2:** L0 Sensors for morphological state
- [ ] **Phase 3:** L0 Actuators for morphological parameters
- [ ] **Phase 4:** Size limit enforcement
- [ ] **Phase 5:** Efficiency calculation and penalty
- [ ] **Phase 6:** Pruning system
- [ ] **Phase 7:** Modularity and symmetry metrics
- [ ] **Phase 8:** Morphological silo server with TWEANN controller
- [ ] **Phase 9:** Cross-silo signal integration
- [ ] **Phase 10:** L2 guidance integration

---

## Success Criteria

- [ ] Size limits enforced during mutation
- [ ] Complexity penalty reduces bloat
- [ ] Pruning removes unused structure
- [ ] Efficiency metrics tracked correctly
- [ ] TWEANN controller adapts morphological parameters
- [ ] All morphology events emitted correctly
- [ ] Cross-silo signals flow correctly
- [ ] Networks converge to efficient architectures
- [ ] Performance: < 5% overhead per generation

---

## References

- PLAN_BEHAVIORAL_EVENTS.md - Event definitions
- PLAN_L2_L1_HIERARCHICAL_INTERFACE.md - LC architecture
- PLAN_ECONOMIC_SILO.md - Economic integration
- task_silo.erl - Reference silo implementation
- "Neuroevolution" - Stanley & Miikkulainen
- "Network Pruning" - Han et al.
