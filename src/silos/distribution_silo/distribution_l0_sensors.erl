%% @doc Distribution Silo L0 Sensors - Collects and normalizes sensor inputs for TWEANN.
%%
%% Part of the Liquid Conglomerate v2 architecture. This module collects
%% network and distribution metrics for the L0 load balancing TWEANN.
%%
%% == Responsibilities ==
%%
%% 1. Collect network metrics (peer load, latency, bandwidth)
%% 2. Track island topology state
%% 3. Monitor migration success rates
%% 4. Normalize all values for TWEANN input
%% 5. Handle cross-silo input signals
%%
%% == Usage ==
%%
%% %% Start the sensor collector
%% {ok, Pid} = distribution_l0_sensors:start_link(Config),
%%
%% %% Update with peer metrics
%% distribution_l0_sensors:update_peer_metrics(Pid, PeerMetrics),
%%
%% %% Get current sensor vector
%% SensorVector = distribution_l0_sensors:get_sensor_vector(Pid),
%% %% Returns: [0.45, 0.23, ...] (14 values)
%%
%% @author R.G. Lefever
%% @copyright 2024-2026 R.G. Lefever
-module(distribution_l0_sensors).
-behaviour(gen_server).

%% API
-export([
    start_link/0,
    start_link/1,
    get_sensor_vector/0,
    get_sensor_vector/1,
    get_sensors/0,
    get_sensors/1,
    update_peer_metrics/1,
    update_island_topology/1,
    update_migration_result/1,
    update_cross_silo_signal/2,
    update_evaluation_queue/1,
    record_migration_time/0
]).

%% gen_server callbacks
-export([
    init/1,
    handle_call/3,
    handle_cast/2,
    handle_info/2,
    terminate/2
]).

-define(SERVER, ?MODULE).
-define(DEFAULT_SAMPLE_INTERVAL, 1000).  % Sample every 1 second
-define(MIGRATION_HISTORY_SIZE, 50).     % Track last 50 migrations

-record(state, {
    %% Configuration
    sample_interval :: pos_integer(),
    max_peers :: pos_integer(),
    max_islands :: pos_integer(),
    migration_cooldown_ms :: pos_integer(),
    max_queue_depth :: pos_integer(),

    %% Current sensor values
    current_sensors :: map(),

    %% Peer metrics (from mesh)
    peer_loads :: map(),        % #{PeerId => Load}
    peer_latencies :: map(),    % #{PeerId => LatencyMs}
    peer_bandwidths :: map(),   % #{PeerId => BandwidthMbps}

    %% Island topology
    island_count :: non_neg_integer(),
    species_per_island :: map(), % #{IslandId => [SpeciesIds]}

    %% Migration tracking
    migration_history :: [boolean()], % Recent success/failure
    last_migration_time :: integer(),

    %% Cross-silo signals
    resource_silo_pressure :: float(),
    task_silo_exploration :: float(),

    %% Queue state
    evaluation_queue_depth :: non_neg_integer(),

    %% Timing
    timer_ref :: reference() | undefined
}).

%%% ============================================================================
%%% API Functions
%%% ============================================================================

%% @doc Start the sensor collector with default configuration.
-spec start_link() -> {ok, pid()} | ignore | {error, term()}.
start_link() ->
    start_link(#{}).

%% @doc Start the sensor collector with custom configuration.
-spec start_link(map()) -> {ok, pid()} | ignore | {error, term()}.
start_link(Config) ->
    gen_server:start_link({local, ?SERVER}, ?MODULE, Config, []).

%% @doc Get ordered sensor vector for TWEANN input.
-spec get_sensor_vector() -> [float()].
get_sensor_vector() ->
    gen_server:call(?SERVER, get_sensor_vector).

%% @doc Get ordered sensor vector from specific server.
-spec get_sensor_vector(pid()) -> [float()].
get_sensor_vector(Pid) ->
    gen_server:call(Pid, get_sensor_vector).

%% @doc Get named sensor map.
-spec get_sensors() -> map().
get_sensors() ->
    gen_server:call(?SERVER, get_sensors).

%% @doc Get named sensor map from specific server.
-spec get_sensors(pid()) -> map().
get_sensors(Pid) ->
    gen_server:call(Pid, get_sensors).

%% @doc Update peer metrics from mesh.
%%
%% PeerMetrics should be a map: #{peer_id => #{load => L, latency => Ms, bandwidth => Mbps}}
-spec update_peer_metrics(map()) -> ok.
update_peer_metrics(PeerMetrics) ->
    gen_server:cast(?SERVER, {peer_metrics, PeerMetrics}).

%% @doc Update island topology information.
-spec update_island_topology(map()) -> ok.
update_island_topology(Topology) ->
    gen_server:cast(?SERVER, {island_topology, Topology}).

%% @doc Record a migration result (success or failure).
-spec update_migration_result(boolean()) -> ok.
update_migration_result(Success) ->
    gen_server:cast(?SERVER, {migration_result, Success}).

%% @doc Update cross-silo signal from another silo.
-spec update_cross_silo_signal(atom(), float()) -> ok.
update_cross_silo_signal(SignalName, Value) ->
    gen_server:cast(?SERVER, {cross_silo_signal, SignalName, Value}).

%% @doc Update evaluation queue depth.
-spec update_evaluation_queue(non_neg_integer()) -> ok.
update_evaluation_queue(Depth) ->
    gen_server:cast(?SERVER, {eval_queue, Depth}).

%% @doc Record that a migration just happened.
-spec record_migration_time() -> ok.
record_migration_time() ->
    gen_server:cast(?SERVER, record_migration_time).

%%% ============================================================================
%%% gen_server Callbacks
%%% ============================================================================

init(Config) ->
    SampleInterval = maps:get(sample_interval, Config, ?DEFAULT_SAMPLE_INTERVAL),
    MaxPeers = maps:get(max_peers, Config, 100),
    MaxIslands = maps:get(max_islands, Config, 8),
    MigrationCooldownMs = maps:get(migration_cooldown_ms, Config, 5000),
    MaxQueueDepth = maps:get(max_queue_depth, Config, 1000),

    %% Start sampling timer
    TimerRef = erlang:send_after(SampleInterval, self(), sample),

    State = #state{
        sample_interval = SampleInterval,
        max_peers = MaxPeers,
        max_islands = MaxIslands,
        migration_cooldown_ms = MigrationCooldownMs,
        max_queue_depth = MaxQueueDepth,
        current_sensors = initial_sensors(),
        peer_loads = #{},
        peer_latencies = #{},
        peer_bandwidths = #{},
        island_count = 1,
        species_per_island = #{},
        migration_history = [],
        last_migration_time = 0,
        resource_silo_pressure = 0.0,
        task_silo_exploration = 0.0,
        evaluation_queue_depth = 0,
        timer_ref = TimerRef
    },

    {ok, State}.

handle_call(get_sensor_vector, _From, State) ->
    Vector = sensors_to_vector(State#state.current_sensors),
    {reply, Vector, State};

handle_call(get_sensors, _From, State) ->
    {reply, State#state.current_sensors, State};

handle_call(_Request, _From, State) ->
    {reply, {error, unknown_request}, State}.

handle_cast({peer_metrics, PeerMetrics}, State) ->
    {Loads, Latencies, Bandwidths} = extract_peer_metrics(PeerMetrics),
    {noreply, State#state{
        peer_loads = Loads,
        peer_latencies = Latencies,
        peer_bandwidths = Bandwidths
    }};

handle_cast({island_topology, Topology}, State) ->
    IslandCount = maps:get(island_count, Topology, State#state.island_count),
    SpeciesPerIsland = maps:get(species_per_island, Topology, State#state.species_per_island),
    {noreply, State#state{
        island_count = IslandCount,
        species_per_island = SpeciesPerIsland
    }};

handle_cast({migration_result, Success}, State) ->
    NewHistory = lists:sublist([Success | State#state.migration_history], ?MIGRATION_HISTORY_SIZE),
    {noreply, State#state{migration_history = NewHistory}};

handle_cast(record_migration_time, State) ->
    Now = erlang:monotonic_time(millisecond),
    {noreply, State#state{last_migration_time = Now}};

handle_cast({cross_silo_signal, resource_silo_pressure, Value}, State) ->
    {noreply, State#state{resource_silo_pressure = clamp(Value, 0.0, 1.0)}};

handle_cast({cross_silo_signal, task_silo_exploration, Value}, State) ->
    {noreply, State#state{task_silo_exploration = clamp(Value, 0.0, 1.0)}};

handle_cast({eval_queue, Depth}, State) ->
    {noreply, State#state{evaluation_queue_depth = Depth}};

handle_cast(_Msg, State) ->
    {noreply, State}.

handle_info(sample, State) ->
    %% Recompute sensors
    NewSensors = compute_sensors(State),

    %% Schedule next sample
    TimerRef = erlang:send_after(State#state.sample_interval, self(), sample),

    {noreply, State#state{
        current_sensors = NewSensors,
        timer_ref = TimerRef
    }};

handle_info(_Info, State) ->
    {noreply, State}.

terminate(_Reason, #state{timer_ref = TimerRef}) ->
    case TimerRef of
        undefined -> ok;
        Ref -> erlang:cancel_timer(Ref)
    end,
    ok.

%%% ============================================================================
%%% Internal Functions - Sensor Computation
%%% ============================================================================

%% @private Compute all sensors from current state.
compute_sensors(State) ->
    #state{
        max_peers = MaxPeers,
        max_islands = MaxIslands,
        migration_cooldown_ms = MigrationCooldownMs,
        max_queue_depth = MaxQueueDepth,
        peer_loads = PeerLoads,
        peer_latencies = PeerLatencies,
        peer_bandwidths = PeerBandwidths,
        island_count = IslandCount,
        species_per_island = SpeciesPerIsland,
        migration_history = MigrationHistory,
        last_migration_time = LastMigration,
        resource_silo_pressure = ResourcePressure,
        task_silo_exploration = TaskExploration,
        evaluation_queue_depth = QueueDepth
    } = State,

    %% Local load (from resource_monitor)
    LocalLoad = get_local_load(),

    %% Queue depth normalized
    LocalQueueDepth = clamp(QueueDepth / max(1, MaxQueueDepth), 0.0, 1.0),

    %% Peer count
    PeerCount = maps:size(PeerLoads),
    PeerCountRatio = clamp(PeerCount / max(1, MaxPeers), 0.0, 1.0),

    %% Peer load statistics
    {AvgPeerLoad, MinPeerLoad} = compute_peer_load_stats(PeerLoads),

    %% Network latency (average)
    NetworkLatencyAvg = compute_avg_latency(PeerLatencies),

    %% Network bandwidth ratio
    NetworkBandwidthRatio = compute_bandwidth_ratio(PeerBandwidths),

    %% Island count ratio
    IslandCountRatio = clamp(IslandCount / max(1, MaxIslands), 0.0, 1.0),

    %% Migration success rate
    MigrationSuccessRate = compute_migration_rate(MigrationHistory),

    %% Species distribution entropy
    SpeciesDistEntropy = compute_species_entropy(SpeciesPerIsland),

    %% Evaluation batch pending
    EvalBatchPending = clamp(QueueDepth / max(1, MaxQueueDepth), 0.0, 1.0),

    %% Time since last migration
    Now = erlang:monotonic_time(millisecond),
    TimeSinceMigration = Now - LastMigration,
    TimeSinceLastMigration = clamp(TimeSinceMigration / max(1, MigrationCooldownMs), 0.0, 1.0),

    #{
        local_load => LocalLoad,
        local_queue_depth => LocalQueueDepth,
        peer_count => PeerCountRatio,
        avg_peer_load => AvgPeerLoad,
        min_peer_load => MinPeerLoad,
        network_latency_avg => NetworkLatencyAvg,
        network_bandwidth_ratio => NetworkBandwidthRatio,
        island_count => IslandCountRatio,
        migration_success_rate => MigrationSuccessRate,
        species_distribution_entropy => SpeciesDistEntropy,
        resource_silo_pressure => ResourcePressure,
        task_silo_exploration => TaskExploration,
        evaluation_batch_pending => EvalBatchPending,
        time_since_last_migration => TimeSinceLastMigration
    }.

%% @private Get local load from resource_monitor.
get_local_load() ->
    try
        Metrics = resource_monitor:get_normalized_metrics(),
        MemPressure = maps:get(memory_pressure, Metrics, 0.0),
        CpuPressure = maps:get(cpu_pressure, Metrics, 0.0),
        (MemPressure + CpuPressure) / 2.0
    catch
        _:_ -> 0.5
    end.

%% @private Compute peer load statistics.
compute_peer_load_stats(PeerLoads) when map_size(PeerLoads) =:= 0 ->
    {0.5, 0.5};  % Default when no peers
compute_peer_load_stats(PeerLoads) ->
    Loads = maps:values(PeerLoads),
    Avg = lists:sum(Loads) / max(1, length(Loads)),
    Min = lists:min(Loads),
    {clamp(Avg, 0.0, 1.0), clamp(Min, 0.0, 1.0)}.

%% @private Compute average latency (normalized).
compute_avg_latency(Latencies) when map_size(Latencies) =:= 0 ->
    0.0;
compute_avg_latency(Latencies) ->
    Values = maps:values(Latencies),
    Avg = lists:sum(Values) / max(1, length(Values)),
    %% Normalize: assume 1000ms is "maximum" latency
    clamp(Avg / 1000.0, 0.0, 1.0).

%% @private Compute bandwidth ratio (normalized).
compute_bandwidth_ratio(Bandwidths) when map_size(Bandwidths) =:= 0 ->
    1.0;  % Assume full bandwidth when no data
compute_bandwidth_ratio(Bandwidths) ->
    Values = maps:values(Bandwidths),
    Avg = lists:sum(Values) / max(1, length(Values)),
    %% Normalize: assume 100 Mbps is "maximum"
    clamp(Avg / 100.0, 0.0, 1.0).

%% @private Compute migration success rate.
compute_migration_rate([]) ->
    0.5;  % Default to neutral
compute_migration_rate(History) ->
    SuccessCount = length([X || X <- History, X =:= true]),
    clamp(SuccessCount / max(1, length(History)), 0.0, 1.0).

%% @private Compute species distribution entropy across islands.
compute_species_entropy(SpeciesPerIsland) when map_size(SpeciesPerIsland) =< 1 ->
    0.0;  % Single island = no distribution
compute_species_entropy(SpeciesPerIsland) ->
    %% Count species per island and compute entropy
    Counts = [length(Species) || Species <- maps:values(SpeciesPerIsland)],
    Total = lists:sum(Counts),
    case Total > 0 of
        true ->
            Probs = [C / Total || C <- Counts, C > 0],
            Entropy = -lists:sum([P * math:log(P) || P <- Probs]),
            %% Normalize by max entropy (log of island count)
            MaxEntropy = math:log(max(1, maps:size(SpeciesPerIsland))),
            case MaxEntropy > 0 of
                true -> clamp(Entropy / MaxEntropy, 0.0, 1.0);
                false -> 0.0
            end;
        false -> 0.0
    end.

%%% ============================================================================
%%% Internal Functions - Utilities
%%% ============================================================================

%% @private Extract peer metrics into separate maps.
extract_peer_metrics(PeerMetrics) ->
    maps:fold(
        fun(PeerId, Metrics, {Loads, Lats, BWs}) ->
            Load = maps:get(load, Metrics, 0.5),
            Lat = maps:get(latency, Metrics, 100),
            BW = maps:get(bandwidth, Metrics, 50),
            {
                maps:put(PeerId, Load, Loads),
                maps:put(PeerId, Lat, Lats),
                maps:put(PeerId, BW, BWs)
            }
        end,
        {#{}, #{}, #{}},
        PeerMetrics
    ).

%% @private Initial sensor values.
initial_sensors() ->
    #{
        local_load => 0.0,
        local_queue_depth => 0.0,
        peer_count => 0.0,
        avg_peer_load => 0.5,
        min_peer_load => 0.5,
        network_latency_avg => 0.0,
        network_bandwidth_ratio => 1.0,
        island_count => 0.1,
        migration_success_rate => 0.5,
        species_distribution_entropy => 0.0,
        resource_silo_pressure => 0.0,
        task_silo_exploration => 0.0,
        evaluation_batch_pending => 0.0,
        time_since_last_migration => 1.0
    }.

%% @private Convert sensor map to ordered vector.
sensors_to_vector(Sensors) ->
    SensorNames = distribution_l0_morphology:sensor_names(),
    [maps:get(Name, Sensors, 0.0) || Name <- SensorNames].

%% @private Clamp value to range.
clamp(Value, Min, Max) ->
    max(Min, min(Max, Value)).
