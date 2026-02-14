%% @doc Resource Silo - System stability controller for neuroevolution.
%%
%% Part of the Liquid Conglomerate v2 architecture. The Resource Silo monitors
%% system resources and provides recommended parameters to maintain stability.
%%
%% == Operating Modes ==
%%
%% The Resource Silo can operate in two modes:
%%
%% training: Full control with aggressive resource management (default)
%%   - 1 second sample interval
%%   - Can pause training on critical memory pressure
%%   - Can trigger GC on high memory usage
%%
%% inference: Lightweight monitoring for production inference
%%   - 5 second sample interval (reduced overhead)
%%   - Never pauses (should_pause always returns false)
%%   - No forced GC (prioritizes latency)
%%   - Still publishes events for observability
%%
%% == Hierarchical Levels ==
%%
%% L0 (Emergency): Hard limits, GC triggers, pause if critical - ALWAYS ACTIVE
%% L1 (Reactive): Adjust concurrency based on current pressure
%% L2 (Predictive): Learn resource patterns, anticipate needs (future)
%%
%% == Time Constant ==
%%
%% tau = 5 (fast adaptation for system stability)
%%
%% @author R.G. Lefever
%% @copyright 2024-2026 R.G. Lefever
-module(resource_silo).
-behaviour(gen_server).

%% API
-export([
    start_link/0,
    start_link/1,
    get_recommendations/1,
    get_recommendations/0,
    should_pause/1,
    should_pause/0,
    get_state/0,
    get_state/1,
    force_gc/0,
    set_enabled_levels/2,
    %% Inference mode
    set_mode/1,
    get_mode/0
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

-include("meta_controller.hrl").

-define(SERVER, ?MODULE).
-define(DEFAULT_SAMPLE_INTERVAL, 1000).  % Sample every 1 second
-define(TIME_CONSTANT, 5).               % tau = 5 generations (fast)

-record(state, {
    %% Configuration
    enabled_levels :: [l0 | l1 | l2],
    base_concurrency :: pos_integer(),
    memory_high_threshold :: float(),
    memory_critical_threshold :: float(),
    cpu_high_threshold :: float(),
    sample_interval :: pos_integer(),
    realm :: binary(),  % Realm for event publishing

    %% Mode: training (full control) or inference (monitoring only)
    %% In inference mode:
    %% - Longer sample interval (5s vs 1s)
    %% - No forced GC
    %% - No pause triggers (always continue)
    %% - Just monitoring and event publishing
    mode = training :: training | inference,

    %% Current observations (L1 reactive state)
    current_metrics :: map(),
    pressure_history :: [float()],      % Rolling window of pressure values
    history_size :: pos_integer(),

    %% L1 adaptive state
    current_concurrency :: pos_integer(),
    gc_triggered_count :: non_neg_integer(),
    pause_count :: non_neg_integer(),

    %% State tracking for change detection
    prev_action :: continue | throttle | pause,

    %% L2 Strategic guidance (from meta_controller)
    %% These thresholds are LC-controlled when L2 is enabled
    l2_enabled = false :: boolean(),
    l2_guidance :: #l2_guidance{} | undefined,
    pressure_scale_factor = 0.9 :: float(),   % Max reduction (from L2: pressure_scale_factor)
    min_scale_factor = 0.1 :: float(),        % Min concurrency (from L2: min_scale_factor)
    pressure_change_threshold = 0.05 :: float(), % Threshold for significant pressure change

    %% Timing
    last_sample_time :: integer(),
    timer_ref :: reference() | undefined
}).

%%% ============================================================================
%%% API Functions
%%% ============================================================================

%% @doc Start the resource silo with default configuration.
-spec start_link() -> {ok, pid()} | ignore | {error, term()}.
start_link() ->
    start_link(#{}).

%% @doc Start the resource silo with custom configuration.
%%
%% Options:
%% - enabled_levels: [l0, l1, l2] (default: [l0, l1])
%% - base_concurrency: Base concurrent evaluations (default: 10)
%% - memory_high_threshold: Memory pressure for throttling (default: 0.7)
%% - memory_critical_threshold: Memory pressure for pausing (default: 0.9)
%% - cpu_high_threshold: CPU pressure for throttling (default: 0.9)
%% - sample_interval: Metrics sample interval in ms (default: 1000)
-spec start_link(map()) -> {ok, pid()} | ignore | {error, term()}.
start_link(Config) ->
    gen_server:start_link({local, ?SERVER}, ?MODULE, Config, []).

%% @doc Get recommended parameters based on current resource state.
%%
%% Returns a map with:
%% - max_concurrent: Recommended max concurrent evaluations
%% - action: continue | throttle | pause
%% - memory_pressure: Current memory pressure (0.0-1.0)
%% - cpu_pressure: Current CPU pressure (0.0-1.0)
%% - reason: Explanation for the recommendation
-spec get_recommendations(pid()) -> map().
get_recommendations(Pid) ->
    gen_server:call(Pid, get_recommendations).

%% @doc Get recommendations from the registered server.
-spec get_recommendations() -> map().
get_recommendations() ->
    gen_server:call(?SERVER, get_recommendations).

%% @doc Check if evolution should pause due to resource pressure.
-spec should_pause(pid()) -> boolean().
should_pause(Pid) ->
    gen_server:call(Pid, should_pause).

%% @doc Check pause status from registered server.
-spec should_pause() -> boolean().
should_pause() ->
    gen_server:call(?SERVER, should_pause).

%% @doc Get current silo state from the registered server.
-spec get_state() -> map().
get_state() ->
    gen_server:call(?SERVER, get_state).

%% @doc Get current silo state for debugging/monitoring.
-spec get_state(pid()) -> map().
get_state(Pid) ->
    gen_server:call(Pid, get_state).

%% @doc Force garbage collection on all processes.
-spec force_gc() -> ok.
force_gc() ->
    gen_server:cast(?SERVER, force_gc).

%% @doc Set which levels are enabled.
-spec set_enabled_levels(pid(), [l0 | l1 | l2]) -> ok.
set_enabled_levels(Pid, Levels) ->
    gen_server:cast(Pid, {set_enabled_levels, Levels}).

%% @doc Set the operating mode (training or inference).
%%
%% In inference mode:
%% - should_pause() always returns false (never block inference)
%% - Longer sample interval (5s vs 1s)
%% - force_gc has no effect (don't interfere with inference latency)
%% - Still publishes monitoring events for observability
-spec set_mode(training | inference) -> ok.
set_mode(Mode) when Mode =:= training; Mode =:= inference ->
    gen_server:cast(?SERVER, {set_mode, Mode}).

%% @doc Get the current operating mode.
-spec get_mode() -> training | inference.
get_mode() ->
    gen_server:call(?SERVER, get_mode).

%%% ============================================================================
%%% gen_server Callbacks
%%% ============================================================================

init(Config) ->
    %% Parse configuration
    EnabledLevels = maps:get(enabled_levels, Config, [l0, l1]),
    %% Default to 100K - this is Erlang, we can handle millions of processes
    %% Resource silo will throttle down dynamically based on memory pressure
    BaseConcurrency = maps:get(base_concurrency, Config, 100000),
    MemoryHighThreshold = maps:get(memory_high_threshold, Config, 0.7),
    MemoryCriticalThreshold = maps:get(memory_critical_threshold, Config, 0.9),
    CpuHighThreshold = maps:get(cpu_high_threshold, Config, 0.9),
    HistorySize = maps:get(history_size, Config, 10),
    Realm = maps:get(realm, Config, <<"default">>),

    %% Mode configuration: training (default) or inference
    %% Inference mode uses longer sample interval and lighter-weight monitoring
    Mode = maps:get(mode, Config, training),
    DefaultSampleInterval = case Mode of
        inference -> 5000;  %% 5 seconds in inference mode
        training -> ?DEFAULT_SAMPLE_INTERVAL  %% 1 second in training mode
    end,
    SampleInterval = maps:get(sample_interval, Config, DefaultSampleInterval),

    %% L2 configuration
    L2Enabled = lists:member(l2, EnabledLevels),
    DefaultL2Guidance = ?L2_GUIDANCE_DEFAULTS,

    %% Initial metrics sample
    InitialMetrics = resource_monitor:get_normalized_metrics(),

    %% Start sampling timer
    TimerRef = erlang:send_after(SampleInterval, self(), sample_metrics),

    State = #state{
        enabled_levels = EnabledLevels,
        base_concurrency = BaseConcurrency,
        memory_high_threshold = MemoryHighThreshold,
        memory_critical_threshold = MemoryCriticalThreshold,
        cpu_high_threshold = CpuHighThreshold,
        sample_interval = SampleInterval,
        realm = Realm,
        mode = Mode,
        current_metrics = InitialMetrics,
        pressure_history = [],
        history_size = HistorySize,
        current_concurrency = BaseConcurrency,
        gc_triggered_count = 0,
        pause_count = 0,
        prev_action = continue,
        %% L2 guidance initialization
        l2_enabled = L2Enabled,
        l2_guidance = DefaultL2Guidance,
        pressure_scale_factor = DefaultL2Guidance#l2_guidance.pressure_scale_factor,
        min_scale_factor = DefaultL2Guidance#l2_guidance.min_scale_factor,
        pressure_change_threshold = DefaultL2Guidance#l2_guidance.pressure_change_threshold,
        last_sample_time = erlang:monotonic_time(millisecond),
        timer_ref = TimerRef
    },

    error_logger:info_msg("[resource_silo] Started with mode=~p, levels=~p, base_concurrency=~p, sample_interval=~pms~n",
        [Mode, EnabledLevels, BaseConcurrency, SampleInterval]),

    {ok, State}.

handle_call(get_recommendations, _From, State) ->
    Recommendations = compute_recommendations(State),
    {reply, Recommendations, State};

handle_call(should_pause, _From, #state{mode = inference} = State) ->
    %% In inference mode, never pause - we prioritize throughput
    {reply, false, State};

handle_call(should_pause, _From, State) ->
    #state{current_metrics = Metrics, memory_critical_threshold = Critical} = State,
    MemoryPressure = maps:get(memory_pressure, Metrics, 0.0),
    ShouldPause = MemoryPressure >= Critical,
    {reply, ShouldPause, State};

handle_call(get_mode, _From, State) ->
    {reply, State#state.mode, State};

handle_call(get_state, _From, State) ->
    %% Get full sensor data from resource_l0_sensors (if running)
    Sensors = try
        resource_l0_sensors:get_sensors()
    catch
        _:_ -> #{}
    end,

    %% Get actuator values from resource_l0_actuators (if running)
    Actuators = try
        resource_l0_actuators:get_actuator_values()
    catch
        _:_ -> #{}
    end,

    %% Build comprehensive state map for UI
    StateMap = #{
        mode => State#state.mode,
        enabled_levels => State#state.enabled_levels,
        base_concurrency => State#state.base_concurrency,
        current_concurrency => State#state.current_concurrency,
        current_metrics => State#state.current_metrics,
        pressure_history => State#state.pressure_history,
        gc_triggered_count => State#state.gc_triggered_count,
        pause_count => State#state.pause_count,
        sample_interval => State#state.sample_interval,
        thresholds => #{
            memory_high => State#state.memory_high_threshold,
            memory_critical => State#state.memory_critical_threshold,
            cpu_high => State#state.cpu_high_threshold
        },
        %% Full L0 sensor data (15 sensors)
        sensors => Sensors,
        %% Full L0 actuator data (10 actuators)
        actuators => Actuators
    },
    {reply, StateMap, State};

handle_call(_Request, _From, State) ->
    {reply, {error, unknown_request}, State}.

handle_cast(force_gc, #state{mode = inference} = State) ->
    %% In inference mode, don't trigger GC - prioritize latency
    {noreply, State};

handle_cast(force_gc, State) ->
    trigger_gc(),
    NewState = State#state{gc_triggered_count = State#state.gc_triggered_count + 1},
    {noreply, NewState};

handle_cast({set_enabled_levels, Levels}, State) ->
    error_logger:info_msg("[resource_silo] Enabled levels changed to ~p~n", [Levels]),
    {noreply, State#state{enabled_levels = Levels}};

handle_cast({set_mode, Mode}, State) ->
    OldMode = State#state.mode,
    NewSampleInterval = case Mode of
        inference -> 5000;
        training -> ?DEFAULT_SAMPLE_INTERVAL
    end,
    %% Cancel existing timer and start new one with updated interval
    case State#state.timer_ref of
        undefined -> ok;
        OldRef -> erlang:cancel_timer(OldRef)
    end,
    NewTimerRef = erlang:send_after(NewSampleInterval, self(), sample_metrics),
    error_logger:info_msg("[resource_silo] Mode changed from ~p to ~p (sample interval: ~pms)~n",
        [OldMode, Mode, NewSampleInterval]),
    {noreply, State#state{
        mode = Mode,
        sample_interval = NewSampleInterval,
        timer_ref = NewTimerRef
    }};

handle_cast(_Msg, State) ->
    {noreply, State}.

handle_info(sample_metrics, State) ->
    %% Sample current metrics
    NewMetrics = resource_monitor:get_normalized_metrics(),
    OverallPressure = maps:get(overall_pressure, NewMetrics, 0.0),

    %% Update pressure history
    History = State#state.pressure_history,
    MaxHistory = State#state.history_size,
    NewHistory = lists:sublist([OverallPressure | History], MaxHistory),

    %% L2: Query meta_controller for updated guidance (if L2 enabled)
    StateWithL2 = maybe_query_l2_guidance(NewMetrics, State),

    %% L1: Compute adaptive concurrency (uses L2-controlled thresholds)
    NewConcurrency = compute_l1_concurrency(NewMetrics, StateWithL2),

    %% Compute current action and check for changes
    CurrentAction = determine_action(NewMetrics, StateWithL2),
    maybe_emit_resource_alert(StateWithL2#state.realm, StateWithL2#state.prev_action, CurrentAction, NewMetrics),

    %% L0: Check for emergency conditions
    NewState0 = check_l0_emergency(NewMetrics, StateWithL2),

    %% Update state
    NewState = NewState0#state{
        current_metrics = NewMetrics,
        pressure_history = NewHistory,
        current_concurrency = NewConcurrency,
        prev_action = CurrentAction,
        last_sample_time = erlang:monotonic_time(millisecond)
    },

    %% Publish recommendations when they change (event-driven read model)
    maybe_publish_recommendations(State, NewState),

    %% Schedule next sample
    TimerRef = erlang:send_after(State#state.sample_interval, self(), sample_metrics),

    {noreply, NewState#state{timer_ref = TimerRef}};

handle_info(_Info, State) ->
    {noreply, State}.

terminate(_Reason, #state{timer_ref = TimerRef}) ->
    case TimerRef of
        undefined -> ok;
        Ref -> erlang:cancel_timer(Ref)
    end,
    ok.

code_change(_OldVsn, State, _Extra) ->
    {ok, State}.

%%% ============================================================================
%%% Internal Functions - L0 Emergency
%%% ============================================================================

%% @private Check L0 emergency conditions and trigger protective actions.
check_l0_emergency(Metrics, State) ->
    #state{
        enabled_levels = Levels,
        memory_critical_threshold = Critical,
        memory_high_threshold = High
    } = State,

    case lists:member(l0, Levels) of
        false -> State;
        true ->
            MemoryPressure = maps:get(memory_pressure, Metrics, 0.0),

            if
                MemoryPressure >= Critical ->
                    %% Critical: Force GC, increment pause counter
                    error_logger:warning_msg(
                        "[resource_silo:L0] CRITICAL memory pressure (~.1f%) - forcing GC~n",
                        [MemoryPressure * 100]),
                    trigger_gc(),
                    State#state{
                        gc_triggered_count = State#state.gc_triggered_count + 1,
                        pause_count = State#state.pause_count + 1
                    };

                MemoryPressure >= High ->
                    %% High: Log warning, may trigger GC if rising
                    case is_pressure_rising(State#state.pressure_history, State) of
                        true ->
                            error_logger:warning_msg(
                                "[resource_silo:L0] HIGH memory pressure (~.1f%) rising - forcing GC~n",
                                [MemoryPressure * 100]),
                            trigger_gc(),
                            State#state{gc_triggered_count = State#state.gc_triggered_count + 1};
                        false ->
                            State
                    end;

                true ->
                    State
            end
    end.

%% @private Trigger garbage collection.
trigger_gc() ->
    %% First, GC our own process
    erlang:garbage_collect(),

    %% Then request GC on all processes (non-blocking)
    spawn(fun() ->
        lists:foreach(
            fun(Pid) ->
                catch erlang:garbage_collect(Pid)
            end,
            erlang:processes()
        )
    end),
    ok.

%% @private Check if pressure is rising based on history.
%% Uses L2-controlled pressure_change_threshold from state.
is_pressure_rising([], _State) -> false;
is_pressure_rising([_], _State) -> false;
is_pressure_rising([Latest, Previous | _], State) ->
    %% Use L2-controlled threshold instead of hardcoded 0.05
    Threshold = State#state.pressure_change_threshold,
    Latest > Previous andalso (Latest - Previous) > Threshold.

%%% ============================================================================
%%% Internal Functions - L1 Reactive
%%% ============================================================================

%% @private Compute L1 reactive concurrency adjustment.
%%
%% Uses L2-controlled scale factors from state instead of hardcoded values.
%% - pressure_scale_factor: How aggressively to reduce concurrency (0.5-0.99)
%% - min_scale_factor: Minimum concurrency as fraction of base (0.1-0.5)
compute_l1_concurrency(Metrics, State) ->
    #state{
        enabled_levels = Levels,
        base_concurrency = Base,
        memory_high_threshold = MemHigh,
        cpu_high_threshold = CpuHigh,
        pressure_scale_factor = PressureScale,
        min_scale_factor = MinScale
    } = State,

    case lists:member(l1, Levels) of
        false ->
            %% L1 disabled, use base concurrency
            Base;
        true ->
            MemoryPressure = maps:get(memory_pressure, Metrics, 0.0),
            CpuPressure = maps:get(cpu_pressure, Metrics, 0.0),

            %% Compute pressure factor (0.0 = no pressure, 1.0 = max pressure)
            MemFactor = pressure_to_factor(MemoryPressure, MemHigh),
            CpuFactor = pressure_to_factor(CpuPressure, CpuHigh),

            %% Use the more restrictive factor
            PressureFactor = max(MemFactor, CpuFactor),

            %% Scale concurrency using L2-controlled factors:
            %% Concurrency = Base * (1 - PressureFactor * PressureScale)
            %% MinScale ensures we always have at least MinScale% of base concurrency
            ScaleFactor = max(MinScale, 1.0 - (PressureFactor * PressureScale)),
            NewConcurrency = max(1, round(Base * ScaleFactor)),

            NewConcurrency
    end.

%% @private Convert pressure to a factor between 0.0 and 1.0.
%% Below threshold = 0.0, at threshold = 0.0, above = linear to 1.0 at 100%
pressure_to_factor(Pressure, Threshold) when Pressure =< Threshold ->
    0.0;
pressure_to_factor(Pressure, Threshold) ->
    %% Linear scale from threshold to 1.0
    min(1.0, (Pressure - Threshold) / (1.0 - Threshold)).

%%% ============================================================================
%%% Internal Functions - Recommendations
%%% ============================================================================

%% @private Compute recommendations based on current state.
compute_recommendations(State) ->
    #state{
        current_metrics = Metrics,
        current_concurrency = Concurrency,
        enabled_levels = Levels
    } = State,

    MemoryPressure = maps:get(memory_pressure, Metrics, 0.0),
    CpuPressure = maps:get(cpu_pressure, Metrics, 0.0),

    %% Determine action
    Action = determine_action(Metrics, State),
    Reason = action_reason(Action),

    #{
        max_concurrent => Concurrency,
        action => Action,
        memory_pressure => MemoryPressure,
        cpu_pressure => CpuPressure,
        overall_pressure => maps:get(overall_pressure, Metrics, 0.0),
        reason => Reason,
        enabled_levels => Levels,
        gc_triggered_count => State#state.gc_triggered_count,
        pause_count => State#state.pause_count
    }.

%% @private Determine the recommended action based on metrics.
-spec determine_action(map(), #state{}) -> continue | throttle | pause.
determine_action(Metrics, State) ->
    #state{
        memory_high_threshold = MemHigh,
        memory_critical_threshold = MemCritical,
        cpu_high_threshold = CpuHigh
    } = State,

    MemoryPressure = maps:get(memory_pressure, Metrics, 0.0),
    CpuPressure = maps:get(cpu_pressure, Metrics, 0.0),

    if
        MemoryPressure >= MemCritical -> pause;
        MemoryPressure >= MemHigh -> throttle;
        CpuPressure >= CpuHigh -> throttle;
        true -> continue
    end.

%% @private Get reason string for action.
-spec action_reason(continue | throttle | pause) -> binary().
action_reason(pause) -> <<"Memory critical - evolution should pause">>;
action_reason(throttle) -> <<"High resource pressure - reducing concurrency">>;
action_reason(continue) -> <<"Resources healthy">>.

%%% ============================================================================
%%% Internal Functions - Resource Alert Events
%%% ============================================================================

%% @private Emit resource alert events when action changes.
maybe_emit_resource_alert(_Realm, Same, Same, _Metrics) ->
    %% No change in action
    ok;

maybe_emit_resource_alert(Realm, PrevAction, CurrentAction, Metrics) ->
    MemoryPressure = maps:get(memory_pressure, Metrics, 0.0),
    CpuPressure = maps:get(cpu_pressure, Metrics, 0.0),

    Event = {resource_alert, #{
        realm => Realm,
        source => resource_silo,
        previous_action => PrevAction,
        action => CurrentAction,
        memory_pressure => MemoryPressure,
        cpu_pressure => CpuPressure,
        message => action_change_message(PrevAction, CurrentAction),
        timestamp => erlang:system_time(millisecond)
    }},

    %% Publish to resource_alert topic via pg-based pubsub
    Topic = neuroevolution_events:topic_resource_alert(Realm),
    neuroevolution_events:publish(Topic, Event),

    error_logger:info_msg(
        "[resource_silo] ACTION CHANGED: ~p -> ~p (mem=~.1f%, cpu=~.1f%)~n",
        [PrevAction, CurrentAction, MemoryPressure * 100, CpuPressure * 100]
    ).

%% @private Generate message for action change.
-spec action_change_message(atom(), atom()) -> binary().
action_change_message(continue, throttle) ->
    <<"Resource pressure detected, reducing concurrency">>;
action_change_message(continue, pause) ->
    <<"Critical resource pressure, evolution paused">>;
action_change_message(throttle, pause) ->
    <<"Resource pressure critical, evolution paused">>;
action_change_message(throttle, continue) ->
    <<"Resource pressure recovered, resuming normal operation">>;
action_change_message(pause, throttle) ->
    <<"Resource pressure easing, resuming with reduced concurrency">>;
action_change_message(pause, continue) ->
    <<"Resource pressure recovered, resuming normal operation">>;
action_change_message(_, _) ->
    <<"Resource state changed">>.

%%% ============================================================================
%%% Internal Functions - L2 Guidance
%%% ============================================================================

%% @private Query meta_controller for L2 guidance if L2 is enabled.
%%
%% When L2 is enabled:
%%   1. Check if meta_controller is registered and alive
%%   2. Call meta_controller:get_l1_guidance/2 with resource stats
%%   3. Update state with new L2 guidance and derived thresholds
%%
%% When L2 is disabled or meta_controller unavailable:
%%   Return state unchanged (using existing config defaults)
maybe_query_l2_guidance(_Metrics, #state{l2_enabled = false} = State) ->
    State;
maybe_query_l2_guidance(Metrics, #state{l2_enabled = true, l2_guidance = OldGuidance} = State) ->
    case whereis(meta_controller) of
        undefined ->
            %% meta_controller not running, use defaults
            State;
        Pid when is_pid(Pid) ->
            try
                %% Build resource stats for meta_controller
                Stats = build_resource_stats(Metrics, State),
                NewGuidance = meta_controller:get_l1_guidance(Pid, Stats),
                %% Update state with L2-controlled thresholds
                apply_l2_guidance(State, OldGuidance, NewGuidance)
            catch
                _:Reason ->
                    error_logger:warning_msg(
                        "[resource_silo] Failed to query L2 guidance: ~p~n", [Reason]),
                    State
            end
    end.

%% @private Build resource stats map for meta_controller.
build_resource_stats(Metrics, State) ->
    #{
        memory_pressure => maps:get(memory_pressure, Metrics, 0.0),
        cpu_pressure => maps:get(cpu_pressure, Metrics, 0.0),
        overall_pressure => maps:get(overall_pressure, Metrics, 0.0),
        current_concurrency => State#state.current_concurrency,
        base_concurrency => State#state.base_concurrency,
        gc_triggered_count => State#state.gc_triggered_count,
        pause_count => State#state.pause_count,
        prev_action => State#state.prev_action
    }.

%% @private Apply L2 guidance to update state thresholds.
apply_l2_guidance(State, OldGuidance, NewGuidance) ->
    %% Log significant changes
    case significant_guidance_change(OldGuidance, NewGuidance) of
        true ->
            error_logger:info_msg(
                "[resource_silo] L2 guidance updated: mem_high=~.2f, mem_crit=~.2f, cpu_high=~.2f~n",
                [NewGuidance#l2_guidance.memory_high_threshold,
                 NewGuidance#l2_guidance.memory_critical_threshold,
                 NewGuidance#l2_guidance.cpu_high_threshold]);
        false ->
            ok
    end,
    %% Apply L2-controlled thresholds to state
    State#state{
        l2_guidance = NewGuidance,
        memory_high_threshold = NewGuidance#l2_guidance.memory_high_threshold,
        memory_critical_threshold = NewGuidance#l2_guidance.memory_critical_threshold,
        cpu_high_threshold = NewGuidance#l2_guidance.cpu_high_threshold,
        pressure_scale_factor = NewGuidance#l2_guidance.pressure_scale_factor,
        min_scale_factor = NewGuidance#l2_guidance.min_scale_factor,
        pressure_change_threshold = NewGuidance#l2_guidance.pressure_change_threshold
    }.

%% @private Check if L2 guidance changed significantly.
significant_guidance_change(undefined, _New) ->
    true;
significant_guidance_change(Old, New) ->
    %% Check if any threshold changed by more than 0.01
    abs(Old#l2_guidance.memory_high_threshold - New#l2_guidance.memory_high_threshold) > 0.01 orelse
    abs(Old#l2_guidance.memory_critical_threshold - New#l2_guidance.memory_critical_threshold) > 0.01 orelse
    abs(Old#l2_guidance.cpu_high_threshold - New#l2_guidance.cpu_high_threshold) > 0.01 orelse
    abs(Old#l2_guidance.pressure_scale_factor - New#l2_guidance.pressure_scale_factor) > 0.01.

%%% ============================================================================
%%% Internal Functions - Event-Driven Recommendations
%%% ============================================================================

%% @private Publish recommendations when they change significantly.
%%
%% This implements the "cached read model" pattern for event-driven architecture.
%% Recommendations are published when:
%% - Action changes (continue -> throttle -> pause)
%% - Concurrency changes by more than 10%
%%
%% Subscribers cache the latest recommendations locally and use them
%% instead of making blocking get_recommendations() calls.
maybe_publish_recommendations(OldState, NewState) ->
    ActionChanged = OldState#state.prev_action =/= NewState#state.prev_action,
    OldConc = OldState#state.current_concurrency,
    NewConc = NewState#state.current_concurrency,
    ConcurrencyChanged = OldConc > 0 andalso abs(NewConc - OldConc) / OldConc > 0.1,

    case ActionChanged orelse ConcurrencyChanged of
        true ->
            Recommendations = compute_recommendations(NewState),
            silo_events:publish_recommendations(resource, Recommendations);
        false ->
            ok
    end.
