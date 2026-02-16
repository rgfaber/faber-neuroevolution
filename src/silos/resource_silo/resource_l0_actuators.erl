%% @doc Resource Silo L0 Actuators - Denormalizes and applies TWEANN outputs.
%%
%% Part of the Liquid Conglomerate v2 architecture. This module takes the
%% normalized output vector from the L0 TWEANN and converts it into actual
%% control signals that affect system behavior.
%%
%% == Responsibilities ==
%%
%% 1. Convert TWEANN outputs (0.0-1.0) to target ranges
%% 2. Apply outputs to neuroevolution_server and other targets
%% 3. Emit cross-silo signals to Task Silo
%% 5. Track applied values for debugging/monitoring
%%
%% == Actuators (10 total) ==
%%
%% Resource control actuators (1-8):
%% - max_concurrent_evaluations, evaluation_batch_size
%% - gc_trigger_threshold, pause_threshold, throttle_intensity
%% - max_evals_per_individual, task_silo_pressure_signal, gc_aggressiveness
%%
%% Self-play archive actuator (9):
%% - archive_gc_pressure: Force archive cleanup under memory pressure (0=none, 1=aggressive)
%%
%% Timeout control actuator (10):
%% - evaluation_timeout: Worker timeout in ms (1000=fast kill, 10000=patient)
%%
%% == Usage ==
%%
%% %% Start the actuator controller
%% {ok, Pid} = resource_l0_actuators:start_link(Config),
%%
%% %% Apply TWEANN output vector
%% resource_l0_actuators:apply_outputs(Pid, OutputVector),
%%
%% %% Get current actuator values
%% Values = resource_l0_actuators:get_actuator_values(Pid),
%% %% Returns: #{max_concurrent_evaluations => 15, ...}
%%
%% @author R.G. Lefever
%% @copyright 2024-2026 R.G. Lefever
-module(resource_l0_actuators).
-behaviour(gen_server).

%% API
-export([
    start_link/0,
    start_link/1,
    apply_outputs/1,
    apply_outputs/2,
    apply_output_vector/1,
    apply_output_vector/2,
    get_actuator_values/0,
    get_actuator_values/1,
    get_raw_outputs/0,
    get_raw_outputs/1
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

-record(state, {
    %% Configuration
    hyperparameters :: map(),  % L0 hyperparameters (from L1)

    %% Current values
    raw_outputs :: [float()],           % Raw TWEANN outputs (0.0-1.0)
    actuator_values :: map(),           % Denormalized actuator values

    %% Cross-silo output callback
    task_silo_callback :: fun((float()) -> ok) | undefined,

    %% Reserved for future archive callback
    archive_callback :: fun((float()) -> ok) | undefined
}).

%%% ============================================================================
%%% API Functions
%%% ============================================================================

%% @doc Start the actuator controller with default configuration.
-spec start_link() -> {ok, pid()} | ignore | {error, term()}.
start_link() ->
    start_link(#{}).

%% @doc Start the actuator controller with custom configuration.
-spec start_link(map()) -> {ok, pid()} | ignore | {error, term()}.
start_link(Config) ->
    gen_server:start_link({local, ?SERVER}, ?MODULE, Config, []).

%% @doc Apply TWEANN outputs as a map.
-spec apply_outputs(map()) -> ok.
apply_outputs(OutputMap) ->
    gen_server:cast(?SERVER, {apply_outputs, OutputMap}).

%% @doc Apply TWEANN outputs as a map (specific server).
-spec apply_outputs(pid(), map()) -> ok.
apply_outputs(Pid, OutputMap) ->
    gen_server:cast(Pid, {apply_outputs, OutputMap}).

%% @doc Apply TWEANN output vector (ordered list).
-spec apply_output_vector([float()]) -> ok.
apply_output_vector(OutputVector) ->
    gen_server:cast(?SERVER, {apply_output_vector, OutputVector}).

%% @doc Apply TWEANN output vector (specific server).
-spec apply_output_vector(pid(), [float()]) -> ok.
apply_output_vector(Pid, OutputVector) ->
    gen_server:cast(Pid, {apply_output_vector, OutputVector}).

%% @doc Get current denormalized actuator values.
-spec get_actuator_values() -> map().
get_actuator_values() ->
    gen_server:call(?SERVER, get_actuator_values).

%% @doc Get current denormalized actuator values (specific server).
-spec get_actuator_values(pid()) -> map().
get_actuator_values(Pid) ->
    gen_server:call(Pid, get_actuator_values).

%% @doc Get raw TWEANN outputs (before denormalization).
-spec get_raw_outputs() -> [float()].
get_raw_outputs() ->
    gen_server:call(?SERVER, get_raw_outputs).

%% @doc Get raw TWEANN outputs (specific server).
-spec get_raw_outputs(pid()) -> [float()].
get_raw_outputs(Pid) ->
    gen_server:call(Pid, get_raw_outputs).

%%% ============================================================================
%%% gen_server Callbacks
%%% ============================================================================

init(Config) ->
    Hyperparams = maps:get(hyperparameters, Config,
                           resource_l0_morphology:get_l0_defaults()),
    TaskSiloCallback = maps:get(task_silo_callback, Config, undefined),
    ArchiveCallback = maps:get(archive_callback, Config, undefined),

    State = #state{
        hyperparameters = Hyperparams,
        raw_outputs = lists:duplicate(resource_l0_morphology:actuator_count(), 0.5),
        actuator_values = initial_actuator_values(Hyperparams),
        task_silo_callback = TaskSiloCallback,
        archive_callback = ArchiveCallback
    },

    {ok, State}.

handle_call(get_actuator_values, _From, State) ->
    {reply, State#state.actuator_values, State};

handle_call(get_raw_outputs, _From, State) ->
    {reply, State#state.raw_outputs, State};

handle_call(_Request, _From, State) ->
    {reply, {error, unknown_request}, State}.

handle_cast({apply_output_vector, OutputVector}, State) ->
    NewState = process_output_vector(OutputVector, State),
    {noreply, NewState};

handle_cast({apply_outputs, OutputMap}, State) ->
    %% Convert map to ordered vector
    ActuatorNames = resource_l0_morphology:actuator_names(),
    OutputVector = [maps:get(Name, OutputMap, 0.5) || Name <- ActuatorNames],
    NewState = process_output_vector(OutputVector, State),
    {noreply, NewState};

handle_cast(_Msg, State) ->
    {noreply, State}.

handle_info(_Info, State) ->
    {noreply, State}.

terminate(_Reason, _State) ->
    ok.

%%% ============================================================================
%%% Internal Functions - Output Processing
%%% ============================================================================

%% @private Process output vector and apply to targets.
process_output_vector(OutputVector, State) ->
    ActuatorNames = resource_l0_morphology:actuator_names(),
    Hyperparams = State#state.hyperparameters,

    %% Denormalize each output
    ActuatorValues = denormalize_outputs(ActuatorNames, OutputVector, Hyperparams),

    %% Apply to targets
    apply_to_targets(ActuatorValues, State),

    State#state{
        raw_outputs = OutputVector,
        actuator_values = ActuatorValues
    }.

%% @private Denormalize outputs based on actuator specs.
denormalize_outputs(Names, Values, Hyperparams) ->
    lists:foldl(
        fun({Name, RawValue}, Acc) ->
            DenormValue = denormalize_actuator(Name, RawValue, Hyperparams),
            maps:put(Name, DenormValue, Acc)
        end,
        #{},
        lists:zip(Names, Values)
    ).

%% @private Denormalize a single actuator value.
denormalize_actuator(max_concurrent_evaluations, RawValue, Hyperparams) ->
    %% Range: 1 to base_concurrency * 2
    BaseConcurrency = maps:get(base_concurrency, Hyperparams, 10),
    MaxConc = BaseConcurrency * 2,
    round(lerp(RawValue, 1.0, float(MaxConc)));

denormalize_actuator(evaluation_batch_size, RawValue, _Hyperparams) ->
    %% Range: 1 to 50
    round(lerp(RawValue, 1.0, 50.0));

denormalize_actuator(gc_trigger_threshold, RawValue, _Hyperparams) ->
    %% Range: 0.5 to 0.95
    lerp(RawValue, 0.5, 0.95);

denormalize_actuator(pause_threshold, RawValue, _Hyperparams) ->
    %% Range: 0.7 to 0.99
    lerp(RawValue, 0.7, 0.99);

denormalize_actuator(throttle_intensity, RawValue, _Hyperparams) ->
    %% Range: 0.0 to 1.0 (already normalized)
    clamp(RawValue, 0.0, 1.0);

denormalize_actuator(max_evals_per_individual, RawValue, _Hyperparams) ->
    %% Range: 1 to 20
    round(lerp(RawValue, 1.0, 20.0));

denormalize_actuator(task_silo_pressure_signal, RawValue, _Hyperparams) ->
    %% Range: 0.0 to 1.0 (already normalized)
    clamp(RawValue, 0.0, 1.0);

denormalize_actuator(gc_aggressiveness, RawValue, _Hyperparams) ->
    %% Range: 0.0 to 1.0 (already normalized)
    clamp(RawValue, 0.0, 1.0);

%% Self-play archive actuator (9)
denormalize_actuator(archive_gc_pressure, RawValue, _Hyperparams) ->
    %% Range: 0.0 to 1.0 (0=no pressure, 1=aggressive cleanup)
    clamp(RawValue, 0.0, 1.0);

%% Timeout control actuator (10)
denormalize_actuator(evaluation_timeout, RawValue, _Hyperparams) ->
    %% Range: 1000ms to 10000ms (1s to 10s)
    %% Lower timeout = faster batch completion but may kill slow workers
    %% Higher timeout = patient with slow workers but batch waits longer
    round(lerp(RawValue, 1000.0, 10000.0));

denormalize_actuator(_Unknown, RawValue, _Hyperparams) ->
    RawValue.

%% @private Apply actuator values via event publication.
%%
%% Event-driven pattern: Publish resource parameters as events.
%% Interested parties (neuroevolution_server, resource_silo) subscribe.
apply_to_targets(ActuatorValues, State) ->
    %% Build resource parameters
    ResourceParams = #{
        max_concurrent_evaluations => maps:get(max_concurrent_evaluations, ActuatorValues, 10),
        evaluation_batch_size => maps:get(evaluation_batch_size, ActuatorValues, 10),
        max_evals_per_individual => maps:get(max_evals_per_individual, ActuatorValues, 5),
        evaluation_timeout => maps:get(evaluation_timeout, ActuatorValues, 3000),
        throttle_intensity => maps:get(throttle_intensity, ActuatorValues, 0.0),
        gc_trigger_threshold => maps:get(gc_trigger_threshold, ActuatorValues, 0.85),
        pause_threshold => maps:get(pause_threshold, ActuatorValues, 0.95),
        gc_aggressiveness => maps:get(gc_aggressiveness, ActuatorValues, 0.5)
    },

    %% Publish resource params event
    ResourceEvent = #{
        event_type => <<"l0_resource_params_computed">>,
        timestamp => erlang:system_time(millisecond),
        source => resource_l0_actuators,
        params => ResourceParams
    },
    neuroevolution_events:publish(<<"l0.resource_params">>, ResourceEvent),

    %% Publish cross-silo pressure signal event
    PressureSignal = maps:get(task_silo_pressure_signal, ActuatorValues, 0.0),
    PressureEvent = #{
        event_type => <<"resource_pressure_signal">>,
        timestamp => erlang:system_time(millisecond),
        source => resource_l0_actuators,
        signal_value => PressureSignal
    },
    neuroevolution_events:publish(<<"silo.resource.pressure">>, PressureEvent),

    %% Publish archive GC pressure event
    ArchiveGcPressure = maps:get(archive_gc_pressure, ActuatorValues, 0.0),
    ArchiveEvent = #{
        event_type => <<"archive_gc_pressure">>,
        timestamp => erlang:system_time(millisecond),
        source => resource_l0_actuators,
        pressure => ArchiveGcPressure
    },
    neuroevolution_events:publish(<<"l0.archive_gc_pressure">>, ArchiveEvent),

    %% Legacy callback support (for custom handlers)
    emit_task_silo_signal(PressureSignal, State),

    ok.

%% @private Emit pressure signal to Task Silo.
emit_task_silo_signal(PressureSignal, #state{task_silo_callback = undefined}) ->
    %% Try to send via task_l0_sensors if running
    case whereis(task_l0_sensors) of
        undefined -> ok;
        _Pid ->
            task_l0_sensors:update_cross_silo_signal(resource_pressure_signal, PressureSignal)
    end;
emit_task_silo_signal(PressureSignal, #state{task_silo_callback = Callback}) ->
    Callback(PressureSignal).

%%% ============================================================================
%%% Internal Functions - Utilities
%%% ============================================================================

%% @private Initial actuator values based on hyperparameters.
initial_actuator_values(Hyperparams) ->
    BaseConcurrency = maps:get(base_concurrency, Hyperparams, 10),
    #{
        %% Resource control actuators (1-8)
        max_concurrent_evaluations => BaseConcurrency,
        evaluation_batch_size => 10,
        gc_trigger_threshold => 0.85,
        pause_threshold => 0.95,
        throttle_intensity => 0.0,
        max_evals_per_individual => 5,
        task_silo_pressure_signal => 0.0,
        gc_aggressiveness => 0.5,
        %% Self-play archive actuator (9)
        archive_gc_pressure => 0.0,
        %% Timeout control actuator (10) - default 3000ms (current YAML value)
        evaluation_timeout => 3000
    }.

%% @private Linear interpolation from 0.0-1.0 to target range.
lerp(T, Min, Max) ->
    ClampedT = clamp(T, 0.0, 1.0),
    Min + ClampedT * (Max - Min).

%% @private Clamp value to range.
clamp(Value, Min, Max) ->
    max(Min, min(Max, Value)).
