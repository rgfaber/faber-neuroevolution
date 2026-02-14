%% @doc Morphological Silo - Network structure and complexity management.
%%
%% Part of the Liquid Conglomerate v2 architecture. The Morphological Silo manages:
%%   Network size constraints (neurons, connections)
%%   Complexity tracking and penalties
%%   Pruning thresholds
%%   Parameter efficiency optimization
%%   Sensor/actuator addition rates
%%
%% == Time Constant ==
%%
%% τ = 30 (medium adaptation for structural changes)
%%
%% == Cross-Silo Signals ==
%%
%% Outgoing:
%%   complexity_signal to task: Current network complexity level
%%   size_budget to resource: Network size requirements
%%   efficiency_score to economic: Parameter efficiency metric
%%   growth_stage to developmental: Structural development stage
%%
%% Incoming:
%%   pressure_signal from resource: Resource constraint
%%   complexity_target from task: Target complexity level
%%   efficiency_requirement from economic: Efficiency targets
%%   expression_cost from regulatory: Cost of gene expression
%%
%% @author R.G. Lefever
%% @copyright 2024-2026 R.G. Lefever
-module(morphological_silo).
-behaviour(gen_server).
-behaviour(lc_silo_behavior).

-include("lc_silos.hrl").
-include("lc_signals.hrl").

%% API
-export([
    start_link/0,
    start_link/1,
    get_params/1,
    record_network_size/4,
    get_complexity_stats/1,
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
-define(TIME_CONSTANT, 30.0).
-define(HISTORY_SIZE, 100).

%% Default actuator values
-define(DEFAULT_PARAMS, #{
    max_neurons => 100,
    max_connections => 500,
    min_neurons => 5,
    pruning_threshold => 0.1,
    complexity_penalty => 0.01,
    sensor_addition_rate => 0.01,
    actuator_addition_rate => 0.01,
    size_penalty_exponent => 1.5
}).

%% Actuator bounds
-define(ACTUATOR_BOUNDS, #{
    max_neurons => {10, 1000},
    max_connections => {20, 10000},
    min_neurons => {1, 50},
    pruning_threshold => {0.0, 0.5},
    complexity_penalty => {0.0, 0.1},
    sensor_addition_rate => {0.0, 0.1},
    actuator_addition_rate => {0.0, 0.1},
    size_penalty_exponent => {1.0, 3.0}
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

    %% Aggregate statistics
    neuron_count_history :: [non_neg_integer()],
    connection_count_history :: [non_neg_integer()],
    efficiency_history :: [float()],

    %% Cross-silo signal cache
    incoming_signals :: map(),

    %% Previous values for smoothing
    prev_complexity_signal :: float(),
    prev_efficiency_score :: float()
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

-spec record_network_size(pid(), term(), non_neg_integer(), non_neg_integer()) -> ok.
record_network_size(Pid, IndividualId, NeuronCount, ConnectionCount) ->
    gen_server:cast(Pid, {record_network_size, IndividualId, NeuronCount, ConnectionCount}).

-spec get_complexity_stats(pid()) -> map().
get_complexity_stats(Pid) ->
    gen_server:call(Pid, get_complexity_stats).

-spec get_state(pid()) -> map().
get_state(Pid) ->
    gen_server:call(Pid, get_state).

-spec reset(pid()) -> ok.
reset(Pid) ->
    gen_server:call(Pid, reset).

%%% ============================================================================
%%% lc_silo_behavior Callbacks
%%% ============================================================================

get_silo_type() -> morphological.

get_time_constant() -> ?TIME_CONSTANT.

init_silo(Config) ->
    Realm = maps:get(realm, Config, <<"default">>),
    EtsTables = lc_ets_utils:create_tables(morphological, Realm, [
        {network_sizes, [{keypos, 1}]}
    ]),
    {ok, #{
        ets_tables => EtsTables,
        realm => Realm
    }}.

collect_sensors(State) ->
    #state{
        neuron_count_history = NeuronHistory,
        connection_count_history = ConnectionHistory,
        efficiency_history = EfficiencyHistory,
        ets_tables = EtsTables,
        current_params = Params,
        incoming_signals = InSignals
    } = State,

    %% Neuron statistics
    NeuronMean = safe_mean(NeuronHistory),
    MaxNeurons = maps:get(max_neurons, Params, 100),
    NormNeuronMean = lc_silo_behavior:normalize(NeuronMean, 0, MaxNeurons),

    %% Connection statistics
    ConnectionMean = safe_mean(ConnectionHistory),
    MaxConnections = maps:get(max_connections, Params, 500),
    NormConnectionMean = lc_silo_behavior:normalize(ConnectionMean, 0, MaxConnections),

    %% Efficiency metrics
    EfficiencyMean = safe_mean(EfficiencyHistory),
    EfficiencyTrend = compute_trend(EfficiencyHistory),

    %% Network structure metrics from ETS
    NetworkTable = maps:get(network_sizes, EtsTables),
    {Modularity, Symmetry} = compute_structure_metrics(NetworkTable),

    %% Growth and pruning metrics
    GrowthRate = compute_growth_rate(NeuronHistory),
    PruningPressure = compute_pruning_pressure(State),
    ComplexityVariance = compute_complexity_variance(NeuronHistory, ConnectionHistory),

    %% Cross-silo signals as sensors
    ResourcePressure = maps:get(pressure_signal, InSignals, 0.0),
    ComplexityTarget = maps:get(complexity_target, InSignals, 0.5),

    #{
        neuron_count_mean => NormNeuronMean,
        connection_count_mean => NormConnectionMean,
        parameter_efficiency => EfficiencyMean,
        efficiency_trend => EfficiencyTrend,
        modularity_score => Modularity,
        symmetry_index => Symmetry,
        growth_rate => GrowthRate,
        pruning_pressure => PruningPressure,
        complexity_variance => ComplexityVariance,
        resource_pressure => ResourcePressure,
        %% External signals
        complexity_target => ComplexityTarget
    }.

apply_actuators(Actuators, State) ->
    BoundedParams = apply_bounds(Actuators, ?ACTUATOR_BOUNDS),
    NewState = State#state{current_params = BoundedParams},
    emit_cross_silo_signals(NewState),
    {ok, NewState}.

compute_reward(State) ->
    Sensors = collect_sensors(State),

    %% Reward components:
    %% 1. Good parameter efficiency
    Efficiency = maps:get(parameter_efficiency, Sensors, 0.5),

    %% 2. Positive efficiency trend
    EfficiencyTrend = maps:get(efficiency_trend, Sensors, 0.5),
    TrendBonus = lc_silo_behavior:normalize(EfficiencyTrend, 0.0, 1.0),

    %% 3. Moderate complexity (not too simple, not too complex)
    NeuronMean = maps:get(neuron_count_mean, Sensors, 0.5),
    ConnectionMean = maps:get(connection_count_mean, Sensors, 0.5),
    ComplexityScore = (NeuronMean + ConnectionMean) / 2,
    ComplexityOptimality = 1.0 - abs(ComplexityScore - 0.5) * 2,

    %% 4. Low variance (consistent network sizes)
    Variance = maps:get(complexity_variance, Sensors, 0.5),
    ConsistencyBonus = 1.0 - Variance,

    %% 5. Modularity bonus
    Modularity = maps:get(modularity_score, Sensors, 0.5),

    %% Combined reward
    Reward = 0.3 * Efficiency +
             0.2 * TrendBonus +
             0.2 * ComplexityOptimality +
             0.15 * ConsistencyBonus +
             0.15 * Modularity,

    lc_silo_behavior:clamp(Reward, 0.0, 1.0).

handle_cross_silo_signals(Signals, State) ->
    CurrentSignals = State#state.incoming_signals,
    UpdatedSignals = maps:merge(CurrentSignals, Signals),
    {ok, State#state{incoming_signals = UpdatedSignals}}.

emit_cross_silo_signals(State) ->
    Sensors = collect_sensors(State),

    %% Complexity signal: average of neuron and connection usage
    NeuronMean = maps:get(neuron_count_mean, Sensors, 0.5),
    ConnectionMean = maps:get(connection_count_mean, Sensors, 0.5),
    ComplexitySignal = (NeuronMean + ConnectionMean) / 2,

    %% Size budget: how much of max capacity is being used
    SizeBudget = ComplexitySignal,

    %% Efficiency score
    EfficiencyScore = maps:get(parameter_efficiency, Sensors, 0.5),

    %% Growth stage: based on trend and current size
    GrowthStage = compute_growth_stage(Sensors),

    %% Emit signals
    emit_signal(task, complexity_signal, ComplexitySignal),
    emit_signal(resource, size_budget, SizeBudget),
    emit_signal(economic, efficiency_score, EfficiencyScore),
    emit_signal(developmental, growth_stage, GrowthStage),
    ok.

%%% ============================================================================
%%% gen_server Callbacks
%%% ============================================================================

init(Config) ->
    Realm = maps:get(realm, Config, <<"default">>),
    EnabledLevels = maps:get(enabled_levels, Config, [l0, l1]),
    L0TweannEnabled = maps:get(l0_tweann_enabled, Config, false),
    L2Enabled = maps:get(l2_enabled, Config, false),

    %% Create ETS tables
    EtsTables = lc_ets_utils:create_tables(morphological, Realm, [
        {network_sizes, [{keypos, 1}]}
    ]),

    State = #state{
        realm = Realm,
        enabled_levels = EnabledLevels,
        l0_tweann_enabled = L0TweannEnabled,
        l2_enabled = L2Enabled,
        current_params = ?DEFAULT_PARAMS,
        ets_tables = EtsTables,
        neuron_count_history = [],
        connection_count_history = [],
        efficiency_history = [],
        incoming_signals = #{},
        prev_complexity_signal = 0.5,
        prev_efficiency_score = 0.5
    },

    %% Schedule periodic cross-silo signal update
    erlang:send_after(1000, self(), update_signals),

    {ok, State}.

handle_call(get_params, _From, State) ->
    {reply, State#state.current_params, State};

handle_call(get_complexity_stats, _From, State) ->
    Stats = #{
        neuron_mean => safe_mean(State#state.neuron_count_history),
        connection_mean => safe_mean(State#state.connection_count_history),
        efficiency_mean => safe_mean(State#state.efficiency_history),
        sample_count => length(State#state.neuron_count_history)
    },
    {reply, Stats, State};

handle_call(get_state, _From, State) ->
    StateMap = #{
        realm => State#state.realm,
        enabled_levels => State#state.enabled_levels,
        current_params => State#state.current_params,
        neuron_history_size => length(State#state.neuron_count_history),
        connection_history_size => length(State#state.connection_count_history),
        sensors => collect_sensors(State)
    },
    {reply, StateMap, State};

handle_call(reset, _From, State) ->
    %% Clear ETS tables
    NetworkTable = maps:get(network_sizes, State#state.ets_tables),
    ets:delete_all_objects(NetworkTable),

    NewState = State#state{
        current_params = ?DEFAULT_PARAMS,
        neuron_count_history = [],
        connection_count_history = [],
        efficiency_history = [],
        incoming_signals = #{},
        prev_complexity_signal = 0.5,
        prev_efficiency_score = 0.5
    },
    {reply, ok, NewState};

handle_call(_Request, _From, State) ->
    {reply, {error, unknown_request}, State}.

handle_cast({record_network_size, IndividualId, NeuronCount, ConnectionCount}, State) ->
    NetworkTable = maps:get(network_sizes, State#state.ets_tables),

    %% Store in ETS
    lc_ets_utils:insert(NetworkTable, IndividualId, #{
        neurons => NeuronCount,
        connections => ConnectionCount,
        params => NeuronCount + ConnectionCount
    }),

    %% Update histories
    NewNeuronHistory = truncate_history(
        [NeuronCount | State#state.neuron_count_history],
        ?HISTORY_SIZE
    ),
    NewConnectionHistory = truncate_history(
        [ConnectionCount | State#state.connection_count_history],
        ?HISTORY_SIZE
    ),

    %% Compute and record efficiency
    Efficiency = compute_individual_efficiency(NeuronCount, ConnectionCount, State),
    NewEfficiencyHistory = truncate_history(
        [Efficiency | State#state.efficiency_history],
        ?HISTORY_SIZE
    ),

    NewState = State#state{
        neuron_count_history = NewNeuronHistory,
        connection_count_history = NewConnectionHistory,
        efficiency_history = NewEfficiencyHistory
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
%%% Internal Functions - Statistics
%%% ============================================================================

safe_mean([]) -> 0.0;
safe_mean(Values) -> lists:sum(Values) / length(Values).

compute_trend([]) -> 0.5;
compute_trend([_]) -> 0.5;
compute_trend(Values) when length(Values) < 3 -> 0.5;
compute_trend(Values) ->
    Recent = lists:sublist(Values, 5),
    Older = lists:sublist(Values, 6, 5),
    RecentMean = safe_mean(Recent),
    OlderMean = safe_mean(Older),
    Trend = safe_ratio(RecentMean - OlderMean, OlderMean + 0.001),
    lc_silo_behavior:normalize(Trend, -0.5, 0.5).

compute_structure_metrics(Table) ->
    %% Simplified modularity and symmetry computation
    %% In a real implementation, this would analyze network topology
    AllSizes = lc_ets_utils:fold(
        fun({_Id, Data, _Ts}, Acc) ->
            Neurons = maps:get(neurons, Data, 0),
            Connections = maps:get(connections, Data, 0),
            [{Neurons, Connections} | Acc]
        end,
        [],
        Table
    ),
    compute_modularity_and_symmetry(AllSizes).

compute_modularity_and_symmetry([]) ->
    {0.5, 0.5};
compute_modularity_and_symmetry(Sizes) ->
    %% Simplified: modularity based on connection density
    %% Symmetry based on variance in sizes
    {NeuronList, ConnectionList} = lists:unzip(Sizes),

    %% Modularity: ratio of connections to max possible
    AvgNeurons = safe_mean(NeuronList),
    AvgConnections = safe_mean(ConnectionList),
    MaxConnections = AvgNeurons * AvgNeurons,
    Density = safe_ratio(AvgConnections, MaxConnections),
    %% Modularity is higher when density is moderate (not too sparse, not too dense)
    Modularity = 1.0 - abs(Density - 0.3) * 2,

    %% Symmetry: lower variance = more symmetric
    NeuronVar = compute_variance(NeuronList),
    ConnectionVar = compute_variance(ConnectionList),
    TotalVar = (NeuronVar + ConnectionVar) / 2,
    Symmetry = 1.0 - lc_silo_behavior:normalize(TotalVar, 0, 1000),

    {lc_silo_behavior:clamp(Modularity, 0.0, 1.0),
     lc_silo_behavior:clamp(Symmetry, 0.0, 1.0)}.

compute_variance([]) -> 0.0;
compute_variance([_]) -> 0.0;
compute_variance(Values) ->
    Mean = safe_mean(Values),
    SumSquares = lists:foldl(
        fun(V, Acc) -> Acc + (V - Mean) * (V - Mean) end,
        0.0,
        Values
    ),
    SumSquares / length(Values).

compute_growth_rate([]) -> 0.5;
compute_growth_rate([_]) -> 0.5;
compute_growth_rate(History) ->
    Recent = lists:sublist(History, 5),
    Older = lists:sublist(History, 6, 5),
    compute_growth_from_windows(Recent, Older).

compute_growth_from_windows([], _) -> 0.5;
compute_growth_from_windows(_, []) -> 0.5;
compute_growth_from_windows(Recent, Older) ->
    RecentMean = safe_mean(Recent),
    OlderMean = safe_mean(Older),
    Growth = safe_ratio(RecentMean - OlderMean, OlderMean + 1),
    lc_silo_behavior:normalize(Growth, -0.5, 0.5).

compute_pruning_pressure(State) ->
    %% Pressure increases when networks exceed target complexity
    Params = State#state.current_params,
    MaxNeurons = maps:get(max_neurons, Params, 100),
    MaxConnections = maps:get(max_connections, Params, 500),

    NeuronMean = safe_mean(State#state.neuron_count_history),
    ConnectionMean = safe_mean(State#state.connection_count_history),

    NeuronPressure = safe_ratio(NeuronMean, MaxNeurons),
    ConnectionPressure = safe_ratio(ConnectionMean, MaxConnections),

    %% Combined pressure
    (NeuronPressure + ConnectionPressure) / 2.

compute_complexity_variance(NeuronHistory, ConnectionHistory) ->
    NeuronVar = compute_variance(NeuronHistory),
    ConnectionVar = compute_variance(ConnectionHistory),
    TotalVar = (NeuronVar + ConnectionVar) / 2,
    lc_silo_behavior:normalize(TotalVar, 0, 1000).

compute_individual_efficiency(NeuronCount, ConnectionCount, State) ->
    %% Efficiency = fitness / complexity
    %% Since we don't have fitness here, use inverse complexity as proxy
    Params = State#state.current_params,
    MaxNeurons = maps:get(max_neurons, Params, 100),
    MaxConnections = maps:get(max_connections, Params, 500),

    NormNeurons = safe_ratio(NeuronCount, MaxNeurons),
    NormConnections = safe_ratio(ConnectionCount, MaxConnections),
    Complexity = (NormNeurons + NormConnections) / 2,

    %% Optimal efficiency at moderate complexity
    1.0 - abs(Complexity - 0.3) * 1.5.

compute_growth_stage(Sensors) ->
    %% Growth stage based on current size and trend
    NeuronMean = maps:get(neuron_count_mean, Sensors, 0.5),
    ConnectionMean = maps:get(connection_count_mean, Sensors, 0.5),
    GrowthRate = maps:get(growth_rate, Sensors, 0.5),

    %% Stage: 0 = juvenile (small, growing), 1 = mature (large, stable)
    SizeComponent = (NeuronMean + ConnectionMean) / 2,
    StabilityComponent = 1.0 - abs(GrowthRate - 0.5) * 2,

    0.6 * SizeComponent + 0.4 * StabilityComponent.

safe_ratio(_Num, Denom) when Denom == 0.0; Denom == 0 -> 0.0;
safe_ratio(Num, Denom) -> Num / Denom.

%%% ============================================================================
%%% Internal Functions - History Management
%%% ============================================================================

truncate_history(List, MaxSize) when length(List) > MaxSize ->
    lists:sublist(List, MaxSize);
truncate_history(List, _MaxSize) ->
    List.

%%% ============================================================================
%%% Internal Functions - Cross-Silo
%%% ============================================================================

emit_signal(_ToSilo, SignalName, Value) ->
    %% Event-driven: publish signal, lc_cross_silo routes to valid destinations
    silo_events:publish_signal(morphological, SignalName, Value).

fetch_incoming_signals() ->
    case whereis(lc_cross_silo) of
        undefined -> #{};
        _Pid -> lc_cross_silo:get_signals_for(morphological)
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
