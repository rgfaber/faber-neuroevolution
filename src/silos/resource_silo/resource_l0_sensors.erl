%% @doc Resource Silo L0 Sensors - Collects and normalizes sensor inputs for TWEANN.
%%
%% Part of the Liquid Conglomerate v2 architecture. This module collects raw
%% system metrics and transforms them into normalized inputs for the L0 TWEANN.
%%
%% == Responsibilities ==
%%
%% 1. Collect raw metrics from resource_monitor and other sources
%% 2. Compute derived metrics (velocities, ratios)
%% 3. Normalize all values to appropriate ranges for TWEANN input
%% 4. Handle cross-silo input signals
%% 5. Maintain state for velocity calculations
%%
%% == Usage ==
%%
%% %% Start the sensor collector
%% {ok, Pid} = resource_l0_sensors:start_link(Config),
%%
%% %% Get current sensor vector (ordered list)
%% SensorVector = resource_l0_sensors:get_sensor_vector(Pid),
%% %% Returns: [0.45, 0.12, 0.33, ...] (13 values)
%%
%% %% Get sensor map (named values)
%% SensorMap = resource_l0_sensors:get_sensors(Pid),
%% %% Returns: #{memory_pressure => 0.45, memory_velocity => 0.12, ...}
%%
%% @author R.G. Lefever
%% @copyright 2024-2026 R.G. Lefever
-module(resource_l0_sensors).
-behaviour(gen_server).

%% API
-export([
    start_link/0,
    start_link/1,
    get_sensor_vector/0,
    get_sensor_vector/1,
    get_sensors/0,
    get_sensors/1,
    update_cross_silo_signal/2,
    update_evaluation_metrics/2,
    update_archive_memory_stats/1,
    set_realm/1
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
-define(DEFAULT_SAMPLE_INTERVAL, 500).  % Sample every 500ms
-define(VELOCITY_WINDOW_SIZE, 5).       % Samples for velocity calculation
-define(MIN_PUBLISH_INTERVAL_MS, 100).  % Max 10Hz event publishing
-define(CHANGE_THRESHOLD, 0.01).        % Minimum change to trigger event

-record(state, {
    %% Configuration
    realm :: binary(),  % Realm for event publishing
    sample_interval :: pos_integer(),
    max_concurrency :: pos_integer(),
    gc_cooldown_ms :: pos_integer(),

    %% Current sensor values
    current_sensors :: map(),

    %% History for velocity calculations
    memory_history :: [float()],
    cpu_history :: [float()],

    %% Cross-silo signals (updated externally)
    task_silo_exploration :: float(),

    %% Evaluation metrics (updated externally)
    current_concurrency :: non_neg_integer(),
    evaluation_throughput :: float(),  % evals/second

    %% GC timing
    last_gc_time :: integer(),

    %% Timing
    timer_ref :: reference() | undefined,

    %% Self-play archive memory stats
    archive_memory_stats :: map(),

    %% Event publishing state
    last_published_sensors :: map(),  % For change detection
    last_publish_time :: integer()    % For throttling (milliseconds)
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
%%
%% Returns a list of 15 float values in the order defined by
%% resource_l0_morphology:sensor_names/0.
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

%% @doc Update cross-silo signal from another silo.
-spec update_cross_silo_signal(atom(), float()) -> ok.
update_cross_silo_signal(SignalName, Value) ->
    gen_server:cast(?SERVER, {cross_silo_signal, SignalName, Value}).

%% @doc Update evaluation metrics from neuroevolution_server.
-spec update_evaluation_metrics(non_neg_integer(), float()) -> ok.
update_evaluation_metrics(CurrentConcurrency, Throughput) ->
    gen_server:cast(?SERVER, {eval_metrics, CurrentConcurrency, Throughput}).

%% @doc Update archive memory statistics from opponent_archive.
%%
%% Expected keys in Stats map:
%% - archive_memory_bytes: Memory used by archive entries
%% - crdt_state_bytes: Memory used by CRDT sync state
%% - total_entries: Number of archive entries
-spec update_archive_memory_stats(map()) -> ok.
update_archive_memory_stats(Stats) ->
    gen_server:cast(?SERVER, {archive_memory_stats, Stats}).

%% @doc Set the realm for event publishing.
%%
%% Use this to dynamically update the realm when a training session starts.
%% Events will be published to the topic for the specified realm.
-spec set_realm(binary()) -> ok.
set_realm(Realm) when is_binary(Realm) ->
    gen_server:cast(?SERVER, {set_realm, Realm}).

%%% ============================================================================
%%% gen_server Callbacks
%%% ============================================================================

init(Config) ->
    Realm = maps:get(realm, Config, <<"default">>),
    SampleInterval = maps:get(sample_interval, Config, ?DEFAULT_SAMPLE_INTERVAL),
    MaxConcurrency = maps:get(max_concurrency, Config, 20),
    GcCooldownMs = maps:get(gc_cooldown_ms, Config, 5000),

    %% Initial sample
    InitialSensors = sample_sensors(#state{
        realm = Realm,
        max_concurrency = MaxConcurrency,
        gc_cooldown_ms = GcCooldownMs,
        memory_history = [],
        cpu_history = [],
        task_silo_exploration = 0.0,
        current_concurrency = 0,
        evaluation_throughput = 0.0,
        last_gc_time = 0,
        archive_memory_stats = #{}
    }),

    %% Start sampling timer
    TimerRef = erlang:send_after(SampleInterval, self(), sample),

    State = #state{
        realm = Realm,
        sample_interval = SampleInterval,
        max_concurrency = MaxConcurrency,
        gc_cooldown_ms = GcCooldownMs,
        current_sensors = InitialSensors,
        memory_history = [],
        cpu_history = [],
        task_silo_exploration = 0.0,
        current_concurrency = 0,
        evaluation_throughput = 0.0,
        last_gc_time = 0,
        timer_ref = TimerRef,
        archive_memory_stats = #{},
        last_published_sensors = InitialSensors,
        last_publish_time = 0
    },

    {ok, State}.

handle_call(get_sensor_vector, _From, State) ->
    Vector = sensors_to_vector(State#state.current_sensors),
    {reply, Vector, State};

handle_call(get_sensors, _From, State) ->
    {reply, State#state.current_sensors, State};

handle_call(_Request, _From, State) ->
    {reply, {error, unknown_request}, State}.

handle_cast({cross_silo_signal, task_silo_exploration, Value}, State) ->
    {noreply, State#state{task_silo_exploration = clamp(Value, 0.0, 1.0)}};

handle_cast({eval_metrics, Concurrency, Throughput}, State) ->
    {noreply, State#state{
        current_concurrency = Concurrency,
        evaluation_throughput = Throughput
    }};

handle_cast({archive_memory_stats, ArchiveStats}, State) ->
    {noreply, State#state{archive_memory_stats = ArchiveStats}};

handle_cast({set_realm, Realm}, State) ->
    error_logger:info_msg("[resource_l0_sensors] Realm updated to: ~p~n", [Realm]),
    {noreply, State#state{realm = Realm}};

handle_cast(_Msg, State) ->
    {noreply, State}.

handle_info(sample, State) ->
    %% Sample new sensor values
    NewSensors = sample_sensors(State),

    %% Update history for velocity calculations
    MemoryPressure = maps:get(memory_pressure, NewSensors, 0.0),
    CpuPressure = maps:get(cpu_pressure, NewSensors, 0.0),

    NewMemHistory = update_history(MemoryPressure, State#state.memory_history),
    NewCpuHistory = update_history(CpuPressure, State#state.cpu_history),

    %% Compute velocities and update sensors
    MemVelocity = compute_velocity(NewMemHistory),
    CpuVelocity = compute_velocity(NewCpuHistory),

    FinalSensors = NewSensors#{
        memory_velocity => MemVelocity,
        cpu_velocity => CpuVelocity
    },

    %% Schedule next sample
    TimerRef = erlang:send_after(State#state.sample_interval, self(), sample),

    %% Update state and potentially publish event
    UpdatedState = State#state{
        current_sensors = FinalSensors,
        memory_history = NewMemHistory,
        cpu_history = NewCpuHistory,
        timer_ref = TimerRef
    },

    %% Publish event if sensors changed significantly
    FinalState = maybe_publish_sensors_event(UpdatedState),

    {noreply, FinalState};

handle_info(_Info, State) ->
    {noreply, State}.

terminate(_Reason, #state{timer_ref = TimerRef}) ->
    case TimerRef of
        undefined -> ok;
        Ref -> erlang:cancel_timer(Ref)
    end,
    ok.

%%% ============================================================================
%%% Internal Functions - Sensor Sampling
%%% ============================================================================

%% @private Sample all sensors and return normalized map.
sample_sensors(State) ->
    %% Get base metrics from resource_monitor
    RawMetrics = resource_monitor:get_metrics(),
    NormMetrics = resource_monitor:get_normalized_metrics(),

    %% Extract values with defaults
    MemoryPressure = maps:get(memory_pressure, NormMetrics, 0.0),
    CpuPressure = maps:get(cpu_pressure, NormMetrics, 0.0),
    ProcessPressure = maps:get(process_pressure, NormMetrics, 0.0),
    MessageQueuePressure = maps:get(message_queue_pressure, NormMetrics, 0.0),

    %% Run queue pressure
    RunQueue = maps:get(run_queue, RawMetrics, 0),
    Schedulers = erlang:system_info(schedulers),
    RunQueuePressure = clamp(RunQueue / max(1, Schedulers * 2), 0.0, 1.0),

    %% Binary memory ratio
    BinaryMem = maps:get(memory_binary, RawMetrics, 0),
    TotalMem = maps:get(memory_total, RawMetrics, 1),
    BinaryMemRatio = clamp(BinaryMem / max(1, TotalMem), 0.0, 1.0),

    %% GC frequency (normalized based on expected GC rate)
    GcFrequency = estimate_gc_frequency(),

    %% Current concurrency ratio
    MaxConc = State#state.max_concurrency,
    CurrentConc = State#state.current_concurrency,
    ConcurrencyRatio = clamp(CurrentConc / max(1, MaxConc), 0.0, 1.0),

    %% Evaluation throughput (normalized to expected max ~100 evals/sec)
    EvalThroughput = clamp(State#state.evaluation_throughput / 100.0, 0.0, 1.0),

    %% Time since last GC (normalized to cooldown)
    Now = erlang:monotonic_time(millisecond),
    TimeSinceGc = Now - State#state.last_gc_time,
    GcCooldown = State#state.gc_cooldown_ms,
    TimeSinceGcNorm = clamp(TimeSinceGc / max(1, GcCooldown), 0.0, 1.0),

    %% Archive memory sensors
    ArchiveStats = State#state.archive_memory_stats,
    ArchiveMemBytes = maps:get(archive_memory_bytes, ArchiveStats, 0),
    CrdtStateBytes = maps:get(crdt_state_bytes, ArchiveStats, 0),
    ArchiveMemRatio = clamp(ArchiveMemBytes / max(1, TotalMem), 0.0, 1.0),
    CrdtSizeRatio = case ArchiveMemBytes > 0 of
        true -> clamp(CrdtStateBytes / max(1, ArchiveMemBytes), 0.0, 1.0);
        false -> 0.0
    end,

    #{
        %% Resource sensors
        memory_pressure => MemoryPressure,
        memory_velocity => 0.0,  % Will be updated after history calculation
        cpu_pressure => CpuPressure,
        cpu_velocity => 0.0,     % Will be updated after history calculation
        run_queue_pressure => RunQueuePressure,
        process_pressure => ProcessPressure,
        message_queue_pressure => MessageQueuePressure,
        binary_memory_ratio => BinaryMemRatio,
        gc_frequency => GcFrequency,
        current_concurrency_ratio => ConcurrencyRatio,
        task_silo_exploration => State#state.task_silo_exploration,
        evaluation_throughput => EvalThroughput,
        time_since_last_gc => TimeSinceGcNorm,
        %% Self-play archive sensors
        archive_memory_ratio => ArchiveMemRatio,
        crdt_state_size_ratio => CrdtSizeRatio
    }.

%% @private Estimate GC frequency from statistics.
estimate_gc_frequency() ->
    try
        {GcCount, _, _} = erlang:statistics(garbage_collection),
        %% Normalize: assume 1000 GCs in sample window is "high"
        clamp(GcCount / 1000.0, 0.0, 1.0)
    catch
        _:_ -> 0.0
    end.

%%% ============================================================================
%%% Internal Functions - Velocity Calculation
%%% ============================================================================

%% @private Update history with new value.
update_history(Value, History) ->
    lists:sublist([Value | History], ?VELOCITY_WINDOW_SIZE).

%% @private Compute velocity from history.
%%
%% Returns value in range -1.0 to 1.0 where:
%% - Negative = value is decreasing
%% - Zero = stable
%% - Positive = value is increasing
compute_velocity([]) -> 0.0;
compute_velocity([_]) -> 0.0;
compute_velocity([Latest, Previous | _]) ->
    %% Simple difference, clamped to -1.0 to 1.0
    %% Scale by 10 to make small changes visible
    Diff = (Latest - Previous) * 10.0,
    clamp(Diff, -1.0, 1.0).

%%% ============================================================================
%%% Internal Functions - Utilities
%%% ============================================================================

%% @private Convert sensor map to ordered vector.
sensors_to_vector(Sensors) ->
    SensorNames = resource_l0_morphology:sensor_names(),
    [maps:get(Name, Sensors, 0.0) || Name <- SensorNames].

%% @private Clamp value to range.
clamp(Value, Min, Max) ->
    max(Min, min(Max, Value)).

%%% ============================================================================
%%% Internal Functions - Event Publishing
%%% ============================================================================

%% @private Publish sensors event if significant change or enough time passed.
maybe_publish_sensors_event(State) ->
    Now = erlang:system_time(millisecond),
    TimeSinceLastPublish = Now - State#state.last_publish_time,
    CurrentSensors = State#state.current_sensors,
    LastPublished = State#state.last_published_sensors,

    ShouldPublish = (TimeSinceLastPublish >= ?MIN_PUBLISH_INTERVAL_MS) andalso
                    sensors_changed_significantly(CurrentSensors, LastPublished),

    case ShouldPublish of
        true ->
            publish_sensors_event(State#state.realm, CurrentSensors),
            State#state{
                last_published_sensors = CurrentSensors,
                last_publish_time = Now
            };
        false ->
            State
    end.

%% @private Check if any sensor changed by more than threshold.
sensors_changed_significantly(Current, Last) ->
    maps:fold(
        fun(Key, Value, Acc) ->
            case Acc of
                true -> true;
                false ->
                    OldValue = maps:get(Key, Last, 0.0),
                    abs(Value - OldValue) > ?CHANGE_THRESHOLD
            end
        end,
        false,
        Current
    ).

%% @private Publish resource_sensors_updated event.
publish_sensors_event(Realm, Sensors) ->
    Topic = neuroevolution_events:topic_silo_sensors(Realm),
    Event = {resource_sensors_updated, #{
        realm => Realm,
        source => resource_l0_sensors,
        timestamp => erlang:system_time(millisecond),
        sensors => Sensors
    }},
    neuroevolution_events:publish(Topic, Event).
