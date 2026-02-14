%% @doc Resource Silo L0 Morphology - TWEANN sensor/actuator definitions.
%%
%% Part of the Liquid Conglomerate v2 architecture. Defines the neural network
%% morphology for the Resource Silo's L0 reactive controller.
%%
%% == Architecture ==
%%
%% L0 is a TWEANN (Topology and Weight Evolving Artificial Neural Network) that:
%% - Takes 15 sensor inputs (13 resource + 2 self-play archive)
%% - Produces 10 actuator outputs (8 resource + 1 archive control + 1 timeout control)
%% - Has 8 hyperparameters that L1 can tune
%% - Has 5 L1 hyperparameters that L2 can tune
%%
%% == Time Constant ==
%%
%% tau_L0 = 5 seconds (fast adaptation for resource management)
%%
%% @author R.G. Lefever
%% @copyright 2024-2026 R.G. Lefever
-module(resource_l0_morphology).

-export([
    %% Morphology definitions
    sensor_count/0,
    actuator_count/0,
    sensor_names/0,
    actuator_names/0,
    sensor_spec/1,
    actuator_spec/1,

    %% Hyperparameter definitions
    l0_hyperparameters/0,
    l0_hyperparameter_spec/1,
    l1_hyperparameters/0,
    l1_hyperparameter_spec/1,

    %% Bounds and defaults
    get_l0_defaults/0,
    get_l0_bounds/0,
    get_l1_defaults/0,
    get_l1_bounds/0,

    %% Time constants
    tau_l0/0,
    tau_l1/0,
    tau_l2/0
]).

%%% ============================================================================
%%% Time Constants
%%% ============================================================================

%% @doc L0 time constant - 5 seconds for reactive resource control.
-spec tau_l0() -> pos_integer().
tau_l0() -> 5000.  % milliseconds

%% @doc L1 time constant - 30 seconds for tactical adaptation.
-spec tau_l1() -> pos_integer().
tau_l1() -> 30000.  % milliseconds

%% @doc L2 time constant - 5 minutes for strategic learning.
-spec tau_l2() -> pos_integer().
tau_l2() -> 300000.  % milliseconds

%%% ============================================================================
%%% Sensor Definitions (15 inputs: 13 resource + 2 self-play archive)
%%% ============================================================================

%% @doc Number of sensors (neural network inputs).
-spec sensor_count() -> pos_integer().
sensor_count() -> 15.

%% @doc Ordered list of sensor names.
-spec sensor_names() -> [atom()].
sensor_names() ->
    [
        %% Resource sensors (1-13)
        memory_pressure,           % 1. Primary memory constraint
        memory_velocity,           % 2. Rate of memory change
        cpu_pressure,              % 3. CPU saturation
        cpu_velocity,              % 4. Rate of CPU change
        run_queue_pressure,        % 5. Work backlog
        process_pressure,          % 6. Process count
        message_queue_pressure,    % 7. Backpressure indicator
        binary_memory_ratio,       % 8. Binary heap stress
        gc_frequency,              % 9. GC activity level
        current_concurrency_ratio, % 10. How much headroom
        task_silo_exploration,     % 11. Cross-silo: Task Silo exploration state
        evaluation_throughput,     % 12. Performance feedback
        time_since_last_gc,        % 13. GC timing
        %% Self-play archive sensors (14-15)
        archive_memory_ratio,      % 14. Archive memory / total memory
        crdt_state_size_ratio      % 15. CRDT sync state overhead
    ].

%% @doc Get specification for a sensor.
%%
%% Returns a map with:
%% - name: Sensor name
%% - range: {Min, Max} normalized range
%% - source: Where the data comes from
%% - description: Human-readable description
-spec sensor_spec(atom()) -> map() | undefined.
sensor_spec(memory_pressure) ->
    #{
        name => memory_pressure,
        range => {0.0, 1.0},
        source => resource_monitor,
        description => <<"Primary memory constraint (0=free, 1=full)">>
    };
sensor_spec(memory_velocity) ->
    #{
        name => memory_velocity,
        range => {-1.0, 1.0},
        source => computed,
        description => <<"Rate of memory change (-1=freeing, +1=filling fast)">>
    };
sensor_spec(cpu_pressure) ->
    #{
        name => cpu_pressure,
        range => {0.0, 1.0},
        source => resource_monitor,
        description => <<"CPU saturation from scheduler utilization">>
    };
sensor_spec(cpu_velocity) ->
    #{
        name => cpu_velocity,
        range => {-1.0, 1.0},
        source => computed,
        description => <<"Rate of CPU change (-1=freeing, +1=saturating)">>
    };
sensor_spec(run_queue_pressure) ->
    #{
        name => run_queue_pressure,
        range => {0.0, 1.0},
        source => resource_monitor,
        description => <<"Work backlog (run_queue / schedulers)">>
    };
sensor_spec(process_pressure) ->
    #{
        name => process_pressure,
        range => {0.0, 1.0},
        source => resource_monitor,
        description => <<"Process count relative to limit">>
    };
sensor_spec(message_queue_pressure) ->
    #{
        name => message_queue_pressure,
        range => {0.0, 1.0},
        source => resource_monitor,
        description => <<"Total message queue length / 10000">>
    };
sensor_spec(binary_memory_ratio) ->
    #{
        name => binary_memory_ratio,
        range => {0.0, 1.0},
        source => resource_monitor,
        description => <<"Binary heap memory / total memory">>
    };
sensor_spec(gc_frequency) ->
    #{
        name => gc_frequency,
        range => {0.0, 1.0},
        source => computed,
        description => <<"GC count in sample window (normalized)">>
    };
sensor_spec(current_concurrency_ratio) ->
    #{
        name => current_concurrency_ratio,
        range => {0.0, 1.0},
        source => internal,
        description => <<"Current concurrency / max concurrency">>
    };
sensor_spec(task_silo_exploration) ->
    #{
        name => task_silo_exploration,
        range => {0.0, 1.0},
        source => cross_silo,
        description => <<"Task Silo's exploration boost state">>
    };
sensor_spec(evaluation_throughput) ->
    #{
        name => evaluation_throughput,
        range => {0.0, 1.0},
        source => computed,
        description => <<"Evaluations/second (normalized to expected max)">>
    };
sensor_spec(time_since_last_gc) ->
    #{
        name => time_since_last_gc,
        range => {0.0, 1.0},
        source => computed,
        description => <<"Time since last GC (normalized)">>
    };
%% Self-play archive sensors
sensor_spec(archive_memory_ratio) ->
    #{
        name => archive_memory_ratio,
        range => {0.0, 1.0},
        source => opponent_archive,
        description => <<"Archive memory usage / total memory">>
    };
sensor_spec(crdt_state_size_ratio) ->
    #{
        name => crdt_state_size_ratio,
        range => {0.0, 1.0},
        source => opponent_archive,
        description => <<"CRDT sync state size / archive data size">>
    };
sensor_spec(_) ->
    undefined.

%%% ============================================================================
%%% Actuator Definitions (10 outputs: 8 resource + 1 self-play archive + 1 timeout)
%%% ============================================================================

%% @doc Number of actuators (neural network outputs).
-spec actuator_count() -> pos_integer().
actuator_count() -> 10.

%% @doc Ordered list of actuator names.
-spec actuator_names() -> [atom()].
actuator_names() ->
    [
        %% Resource actuators (1-8)
        max_concurrent_evaluations, % 1. Worker parallelism
        evaluation_batch_size,      % 2. Batch granularity
        gc_trigger_threshold,       % 3. When to force GC
        pause_threshold,            % 4. When to pause evolution
        throttle_intensity,         % 5. How aggressively to throttle
        max_evals_per_individual,   % 6. Statistical confidence vs speed
        task_silo_pressure_signal,  % 7. Cross-silo: Tell Task Silo to back off
        gc_aggressiveness,          % 8. GC strategy (gentle vs aggressive)
        %% Self-play archive actuator (9)
        archive_gc_pressure,        % 9. Force archive cleanup under memory pressure
        %% Timeout control actuator (10)
        evaluation_timeout          % 10. Worker timeout in milliseconds (1000-10000ms)
    ].

%% @doc Get specification for an actuator.
%%
%% Returns a map with:
%% - name: Actuator name
%% - range: {Min, Max} target range (for denormalization)
%% - target: What system this affects
%% - description: Human-readable description
-spec actuator_spec(atom()) -> map() | undefined.
actuator_spec(max_concurrent_evaluations) ->
    #{
        name => max_concurrent_evaluations,
        range => {1, 1000000},  % This is Erlang - we can handle millions of processes
        target => neuroevolution_server,
        description => <<"Maximum concurrent evaluations">>
    };
actuator_spec(evaluation_batch_size) ->
    #{
        name => evaluation_batch_size,
        range => {1, 50},
        target => batch_evaluator,
        description => <<"Number of individuals per evaluation batch">>
    };
actuator_spec(gc_trigger_threshold) ->
    #{
        name => gc_trigger_threshold,
        range => {0.5, 0.95},
        target => l0_emergency_check,
        description => <<"Memory pressure threshold to trigger GC">>
    };
actuator_spec(pause_threshold) ->
    #{
        name => pause_threshold,
        range => {0.7, 0.99},
        target => l0_pause_decision,
        description => <<"Memory pressure threshold to pause evolution">>
    };
actuator_spec(throttle_intensity) ->
    #{
        name => throttle_intensity,
        range => {0.0, 1.0},
        target => concurrency_scaling,
        description => <<"How aggressively to reduce concurrency (0=none, 1=max)">>
    };
actuator_spec(max_evals_per_individual) ->
    #{
        name => max_evals_per_individual,
        range => {1, 20},
        target => evaluation_config,
        description => <<"Resource-constrained max evals per individual">>
    };
actuator_spec(task_silo_pressure_signal) ->
    #{
        name => task_silo_pressure_signal,
        range => {0.0, 1.0},
        target => cross_silo,
        description => <<"Signal to Task Silo to back off (0=ok, 1=critical)">>
    };
actuator_spec(gc_aggressiveness) ->
    #{
        name => gc_aggressiveness,
        range => {0.0, 1.0},
        target => gc_strategy,
        description => <<"GC strategy: gentle (0) vs aggressive (1)">>
    };
%% Self-play archive actuator
actuator_spec(archive_gc_pressure) ->
    #{
        name => archive_gc_pressure,
        range => {0.0, 1.0},
        target => opponent_archive,
        description => <<"Archive pruning pressure: 0=no pruning, 1=aggressive pruning">>
    };
%% Timeout control actuator
actuator_spec(evaluation_timeout) ->
    #{
        name => evaluation_timeout,
        range => {1000, 10000},
        target => neuroevolution_server,
        description => <<"Worker timeout in ms: 1000=aggressive kill, 10000=patient">>
    };
actuator_spec(_) ->
    undefined.

%%% ============================================================================
%%% L0 Hyperparameters (8 params, tuned by L1)
%%% ============================================================================

%% @doc List of L0 hyperparameter names.
-spec l0_hyperparameters() -> [atom()].
l0_hyperparameters() ->
    [
        memory_high_threshold,      % 1. When to start throttling
        memory_critical_threshold,  % 2. When to pause
        cpu_high_threshold,         % 3. CPU throttle trigger
        base_concurrency,           % 4. Starting worker count
        concurrency_scale_factor,   % 5. Max reduction factor
        pressure_smoothing_alpha,   % 6. EMA smoothing for pressure
        gc_cooldown_ms,             % 7. Min time between GC triggers
        velocity_weight             % 8. How much velocity affects decisions
    ].

%% @doc Get specification for an L0 hyperparameter.
-spec l0_hyperparameter_spec(atom()) -> map() | undefined.
l0_hyperparameter_spec(memory_high_threshold) ->
    #{
        name => memory_high_threshold,
        default => 0.70,
        range => {0.5, 0.85},
        description => <<"Memory pressure threshold to start throttling">>
    };
l0_hyperparameter_spec(memory_critical_threshold) ->
    #{
        name => memory_critical_threshold,
        default => 0.90,
        range => {0.8, 0.98},
        description => <<"Memory pressure threshold to pause evolution">>
    };
l0_hyperparameter_spec(cpu_high_threshold) ->
    #{
        name => cpu_high_threshold,
        default => 0.80,
        range => {0.6, 0.95},
        description => <<"CPU pressure threshold for throttling">>
    };
l0_hyperparameter_spec(base_concurrency) ->
    #{
        name => base_concurrency,
        default => 10,
        range => {2, 50},
        description => <<"Base number of concurrent evaluations">>
    };
l0_hyperparameter_spec(concurrency_scale_factor) ->
    #{
        name => concurrency_scale_factor,
        default => 0.9,
        range => {0.5, 0.99},
        description => <<"Maximum concurrency reduction factor">>
    };
l0_hyperparameter_spec(pressure_smoothing_alpha) ->
    #{
        name => pressure_smoothing_alpha,
        default => 0.3,
        range => {0.1, 0.9},
        description => <<"EMA alpha for smoothing pressure readings">>
    };
l0_hyperparameter_spec(gc_cooldown_ms) ->
    #{
        name => gc_cooldown_ms,
        default => 5000,
        range => {1000, 30000},
        description => <<"Minimum milliseconds between GC triggers">>
    };
l0_hyperparameter_spec(velocity_weight) ->
    #{
        name => velocity_weight,
        default => 0.3,
        range => {0.0, 0.8},
        description => <<"Weight of velocity signals in decisions">>
    };
l0_hyperparameter_spec(_) ->
    undefined.

%% @doc Get default values for L0 hyperparameters.
-spec get_l0_defaults() -> map().
get_l0_defaults() ->
    #{
        memory_high_threshold => 0.70,
        memory_critical_threshold => 0.90,
        cpu_high_threshold => 0.80,
        base_concurrency => 10,
        concurrency_scale_factor => 0.9,
        pressure_smoothing_alpha => 0.3,
        gc_cooldown_ms => 5000,
        velocity_weight => 0.3
    }.

%% @doc Get bounds for L0 hyperparameters.
-spec get_l0_bounds() -> map().
get_l0_bounds() ->
    #{
        memory_high_threshold => {0.5, 0.85},
        memory_critical_threshold => {0.8, 0.98},
        cpu_high_threshold => {0.6, 0.95},
        base_concurrency => {2, 50},
        concurrency_scale_factor => {0.5, 0.99},
        pressure_smoothing_alpha => {0.1, 0.9},
        gc_cooldown_ms => {1000, 30000},
        velocity_weight => {0.0, 0.8}
    }.

%%% ============================================================================
%%% L1 Hyperparameters (5 params, tuned by L2)
%%% ============================================================================

%% @doc List of L1 hyperparameter names.
%%
%% Note: These are called "meta-parameters" from L1's perspective,
%% but "hyperparameters" from L2's perspective.
-spec l1_hyperparameters() -> [atom()].
l1_hyperparameters() ->
    [
        threshold_adaptation_rate,  % 1. How fast L1 adjusts thresholds
        pressure_sensitivity,       % 2. Amplify/dampen pressure signals
        recovery_patience,          % 3. Samples before increasing concurrency
        proactive_gc_tendency,      % 4. Prefer early GC vs late GC
        cross_silo_coupling         % 5. How much to influence Task Silo
    ].

%% @doc Get specification for an L1 hyperparameter.
-spec l1_hyperparameter_spec(atom()) -> map() | undefined.
l1_hyperparameter_spec(threshold_adaptation_rate) ->
    #{
        name => threshold_adaptation_rate,
        default => 0.05,
        range => {0.01, 0.2},
        description => <<"Rate at which L1 adjusts L0's thresholds">>
    };
l1_hyperparameter_spec(pressure_sensitivity) ->
    #{
        name => pressure_sensitivity,
        default => 1.0,
        range => {0.5, 2.0},
        description => <<"Amplification of pressure signals to L0">>
    };
l1_hyperparameter_spec(recovery_patience) ->
    #{
        name => recovery_patience,
        default => 10,
        range => {3, 30},
        description => <<"Samples to wait before increasing concurrency">>
    };
l1_hyperparameter_spec(proactive_gc_tendency) ->
    #{
        name => proactive_gc_tendency,
        default => 0.5,
        range => {0.0, 1.0},
        description => <<"Preference for early GC (1) vs late GC (0)">>
    };
l1_hyperparameter_spec(cross_silo_coupling) ->
    #{
        name => cross_silo_coupling,
        default => 0.5,
        range => {0.0, 1.0},
        description => <<"How much L1 signals influence Task Silo">>
    };
l1_hyperparameter_spec(_) ->
    undefined.

%% @doc Get default values for L1 hyperparameters.
-spec get_l1_defaults() -> map().
get_l1_defaults() ->
    #{
        threshold_adaptation_rate => 0.05,
        pressure_sensitivity => 1.0,
        recovery_patience => 10,
        proactive_gc_tendency => 0.5,
        cross_silo_coupling => 0.5
    }.

%% @doc Get bounds for L1 hyperparameters.
-spec get_l1_bounds() -> map().
get_l1_bounds() ->
    #{
        threshold_adaptation_rate => {0.01, 0.2},
        pressure_sensitivity => {0.5, 2.0},
        recovery_patience => {3, 30},
        proactive_gc_tendency => {0.0, 1.0},
        cross_silo_coupling => {0.0, 1.0}
    }.
