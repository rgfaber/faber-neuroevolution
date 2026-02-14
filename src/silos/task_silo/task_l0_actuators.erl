%% @doc Task Silo L0 Actuators - Denormalizes and applies TWEANN outputs.
%%
%% Part of the Liquid Conglomerate v2 architecture. This module takes the
%% normalized output vector from the L0 TWEANN and converts it into actual
%% evolution parameters that control the neuroevolution process.
%%
%% == Responsibilities ==
%%
%% 1. Convert TWEANN outputs (0.0-1.0) to evolution parameter ranges
%% 2. Apply outputs to neuroevolution_server (evolution params)
%% 3. Apply outputs to opponent_archive (self-play archive params)
%% 4. Handle delta outputs (population_size_delta, compatibility_threshold_delta, archive_max_size_delta)
%% 5. Track applied values for debugging/monitoring
%%
%% == Actuators (20 total) ==
%%
%% Evolution actuators (1-12):
%% - mutation_rate, mutation_strength, selection_ratio
%% - add_node_rate, add_connection_rate, delete_connection_rate
%% - weight_perturb_vs_replace, crossover_rate, interspecies_crossover_rate
%% - elitism_count, population_size_delta, compatibility_threshold_delta
%%
%% Layer-specific mutation actuators (13-16):
%% - reservoir_mutation_rate: Hidden layer mutation probability
%% - reservoir_mutation_strength: Hidden layer weight perturbation strength
%% - readout_mutation_rate: Output layer mutation probability
%% - readout_mutation_strength: Output layer weight perturbation strength
%%
%% Self-play archive actuators (17-20):
%% - archive_threshold_percentile: Entry threshold for adding to archive
%% - archive_sampling_temperature: Opponent selection strategy (0=uniform, 1=fitness-weighted)
%% - archive_prune_ratio: Keep top X% when pruning (0.5-1.0)
%% - archive_max_size_delta: Adjust archive max size (-5 to +5)
%%
%% == Usage ==
%%
%% %% Start the actuator controller
%% {ok, Pid} = task_l0_actuators:start_link(Config),
%%
%% %% Apply TWEANN output vector
%% task_l0_actuators:apply_output_vector(OutputVector),
%%
%% %% Get current actuator values
%% Values = task_l0_actuators:get_actuator_values(),
%% %% Returns: #{mutation_rate => 0.15, crossover_rate => 0.7, ...}
%%
%% @author R.G. Lefever
%% @copyright 2024-2026 R.G. Lefever
-module(task_l0_actuators).
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
    get_raw_outputs/1,
    get_evolution_params/0,
    get_evolution_params/1
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

    %% Current evolution state (for delta calculations)
    current_population_size :: pos_integer(),
    current_compatibility_threshold :: float(),

    %% Self-play archive state (for delta calculations)
    current_archive_max_size :: pos_integer(),

    %% Target callback (for integration with neuroevolution_server)
    evolution_callback :: fun((map()) -> ok) | undefined,
    archive_callback :: fun((map()) -> ok) | undefined
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

%% @doc Get evolution parameters ready for neuroevolution_server.
-spec get_evolution_params() -> map().
get_evolution_params() ->
    gen_server:call(?SERVER, get_evolution_params).

%% @doc Get evolution parameters (specific server).
-spec get_evolution_params(pid()) -> map().
get_evolution_params(Pid) ->
    gen_server:call(Pid, get_evolution_params).

%%% ============================================================================
%%% gen_server Callbacks
%%% ============================================================================

init(Config) ->
    Hyperparams = maps:get(hyperparameters, Config,
                           task_l0_morphology:get_l0_defaults()),
    EvolutionCallback = maps:get(evolution_callback, Config, undefined),
    ArchiveCallback = maps:get(archive_callback, Config, undefined),
    InitialPopSize = maps:get(initial_population_size, Config, 100),
    InitialCompatThreshold = maps:get(initial_compatibility_threshold, Config, 3.0),
    InitialArchiveMaxSize = maps:get(initial_archive_max_size, Config, 50),

    State = #state{
        hyperparameters = Hyperparams,
        raw_outputs = lists:duplicate(task_l0_morphology:actuator_count(), 0.5),
        actuator_values = initial_actuator_values(),
        current_population_size = InitialPopSize,
        current_compatibility_threshold = InitialCompatThreshold,
        current_archive_max_size = InitialArchiveMaxSize,
        evolution_callback = EvolutionCallback,
        archive_callback = ArchiveCallback
    },

    {ok, State}.

handle_call(get_actuator_values, _From, State) ->
    {reply, State#state.actuator_values, State};

handle_call(get_raw_outputs, _From, State) ->
    {reply, State#state.raw_outputs, State};

handle_call(get_evolution_params, _From, State) ->
    Params = build_evolution_params(State),
    {reply, Params, State};

handle_call(_Request, _From, State) ->
    {reply, {error, unknown_request}, State}.

handle_cast({apply_output_vector, OutputVector}, State) ->
    NewState = process_output_vector(OutputVector, State),
    {noreply, NewState};

handle_cast({apply_outputs, OutputMap}, State) ->
    %% Convert map to ordered vector
    ActuatorNames = task_l0_morphology:actuator_names(),
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
    ActuatorNames = task_l0_morphology:actuator_names(),
    Hyperparams = State#state.hyperparameters,

    %% Denormalize each output
    ActuatorValues = denormalize_outputs(ActuatorNames, OutputVector, Hyperparams, State),

    %% Apply delta values to current state
    PopDelta = maps:get(population_size_delta, ActuatorValues, 0),
    CompatDelta = maps:get(compatibility_threshold_delta, ActuatorValues, 0.0),
    ArchiveDelta = maps:get(archive_max_size_delta, ActuatorValues, 0),

    NewPopSize = max(10, State#state.current_population_size + PopDelta),
    NewCompatThreshold = max(0.5, State#state.current_compatibility_threshold + CompatDelta),
    NewArchiveMaxSize = max(10, State#state.current_archive_max_size + ArchiveDelta),

    %% Update actuator values with computed absolute values
    FinalActuatorValues = ActuatorValues#{
        current_population_size => NewPopSize,
        current_compatibility_threshold => NewCompatThreshold,
        current_archive_max_size => NewArchiveMaxSize
    },

    %% Apply to targets
    NewState = State#state{
        raw_outputs = OutputVector,
        actuator_values = FinalActuatorValues,
        current_population_size = NewPopSize,
        current_compatibility_threshold = NewCompatThreshold,
        current_archive_max_size = NewArchiveMaxSize
    },

    apply_to_targets(FinalActuatorValues, NewState),

    NewState.

%% @private Denormalize outputs based on actuator specs.
denormalize_outputs(Names, Values, Hyperparams, State) ->
    lists:foldl(
        fun({Name, RawValue}, Acc) ->
            DenormValue = denormalize_actuator(Name, RawValue, Hyperparams, State),
            maps:put(Name, DenormValue, Acc)
        end,
        #{},
        lists:zip(Names, Values)
    ).

%% @private Denormalize a single actuator value.
denormalize_actuator(mutation_rate, RawValue, Hyperparams, _State) ->
    %% Range determined by L0 hyperparameters
    MinRate = maps:get(mutation_rate_min, Hyperparams, 0.01),
    MaxRate = maps:get(mutation_rate_max, Hyperparams, 0.50),
    lerp(RawValue, MinRate, MaxRate);

denormalize_actuator(mutation_strength, RawValue, _Hyperparams, _State) ->
    %% Range: 0.05 to 1.0
    lerp(RawValue, 0.05, 1.0);

denormalize_actuator(selection_ratio, RawValue, _Hyperparams, _State) ->
    %% Range: 0.05 to 0.50
    lerp(RawValue, 0.05, 0.50);

denormalize_actuator(add_node_rate, RawValue, _Hyperparams, _State) ->
    %% Range: 0.0 to 0.15
    lerp(RawValue, 0.0, 0.15);

denormalize_actuator(add_connection_rate, RawValue, _Hyperparams, _State) ->
    %% Range: 0.0 to 0.25
    lerp(RawValue, 0.0, 0.25);

denormalize_actuator(delete_connection_rate, RawValue, _Hyperparams, _State) ->
    %% Range: 0.0 to 0.10
    lerp(RawValue, 0.0, 0.10);

denormalize_actuator(weight_perturb_vs_replace, RawValue, _Hyperparams, _State) ->
    %% Range: 0.5 to 1.0 (higher = more perturbation, less replacement)
    lerp(RawValue, 0.5, 1.0);

denormalize_actuator(crossover_rate, RawValue, _Hyperparams, _State) ->
    %% Range: 0.0 to 0.9
    lerp(RawValue, 0.0, 0.9);

denormalize_actuator(interspecies_crossover_rate, RawValue, _Hyperparams, _State) ->
    %% Range: 0.0 to 0.3
    lerp(RawValue, 0.0, 0.3);

denormalize_actuator(elitism_count, RawValue, _Hyperparams, _State) ->
    %% Range: 1 to 10
    round(lerp(RawValue, 1.0, 10.0));

denormalize_actuator(population_size_delta, RawValue, _Hyperparams, _State) ->
    %% Range: -10 to +10 (centered at 0.5)
    %% 0.0 -> -10, 0.5 -> 0, 1.0 -> +10
    round(lerp(RawValue, -10.0, 10.0));

denormalize_actuator(compatibility_threshold_delta, RawValue, _Hyperparams, _State) ->
    %% Range: -0.5 to +0.5 (centered at 0.5)
    lerp(RawValue, -0.5, 0.5);

%% Layer-specific mutation actuators (13-16)
%% Enable reservoir/readout training strategies where hidden layers can have
%% different mutation rates than output layers. See guides/training-strategies.md
denormalize_actuator(reservoir_mutation_rate, RawValue, Hyperparams, _State) ->
    %% Range determined by L0 hyperparameters (same as mutation_rate)
    MinRate = maps:get(mutation_rate_min, Hyperparams, 0.01),
    MaxRate = maps:get(mutation_rate_max, Hyperparams, 0.50),
    lerp(RawValue, MinRate, MaxRate);

denormalize_actuator(reservoir_mutation_strength, RawValue, _Hyperparams, _State) ->
    %% Range: 0.05 to 1.0 (same as mutation_strength)
    lerp(RawValue, 0.05, 1.0);

denormalize_actuator(readout_mutation_rate, RawValue, Hyperparams, _State) ->
    %% Range determined by L0 hyperparameters (same as mutation_rate)
    MinRate = maps:get(mutation_rate_min, Hyperparams, 0.01),
    MaxRate = maps:get(mutation_rate_max, Hyperparams, 0.50),
    lerp(RawValue, MinRate, MaxRate);

denormalize_actuator(readout_mutation_strength, RawValue, _Hyperparams, _State) ->
    %% Range: 0.05 to 1.0 (same as mutation_strength)
    lerp(RawValue, 0.05, 1.0);

%% Self-play archive actuators (17-20)
denormalize_actuator(archive_threshold_percentile, RawValue, Hyperparams, _State) ->
    %% Range determined by L0 hyperparameters
    MinThreshold = maps:get(archive_threshold_min, Hyperparams, 0.3),
    MaxThreshold = maps:get(archive_threshold_max, Hyperparams, 0.95),
    lerp(RawValue, MinThreshold, MaxThreshold);

denormalize_actuator(archive_sampling_temperature, RawValue, _Hyperparams, _State) ->
    %% Range: 0.0 to 1.0 (0=uniform sampling, 1=fitness-weighted)
    lerp(RawValue, 0.0, 1.0);

denormalize_actuator(archive_prune_ratio, RawValue, _Hyperparams, _State) ->
    %% Range: 0.5 to 1.0 (keep top X% when pruning)
    lerp(RawValue, 0.5, 1.0);

denormalize_actuator(archive_max_size_delta, RawValue, _Hyperparams, _State) ->
    %% Range: -5 to +5 (centered at 0.5)
    %% 0.0 -> -5, 0.5 -> 0, 1.0 -> +5
    round(lerp(RawValue, -5.0, 5.0));

denormalize_actuator(_Unknown, RawValue, _Hyperparams, _State) ->
    RawValue.

%% @private Apply actuator values to their targets.
apply_to_targets(ActuatorValues, State) ->
    %% Build and apply evolution parameters
    EvolutionParams = build_evolution_params_from_values(ActuatorValues, State),
    apply_evolution_params(EvolutionParams, State),

    %% Build and apply archive parameters
    ArchiveParams = build_archive_params_from_values(ActuatorValues, State),
    apply_archive_params(ArchiveParams, State),

    ok.

%% @private Apply evolution parameters via event publication.
%%
%% Event-driven pattern: Instead of directly calling neuroevolution_server,
%% we publish an event that interested parties can subscribe to.
%% This decouples the L0 actuators from the evolution engine.
%%
%% Event: <<"l0.evolution_params">>
%% Payload: #{mutation_rate => ..., selection_ratio => ..., ...}
apply_evolution_params(EvolutionParams, _State) ->
    %% Publish event - neuroevolution_server subscribes and reacts
    Event = #{
        event_type => <<"l0_evolution_params_computed">>,
        timestamp => erlang:system_time(millisecond),
        source => task_l0_actuators,
        params => EvolutionParams
    },
    neuroevolution_events:publish(<<"l0.evolution_params">>, Event),
    ok.

%% @private Apply archive parameters via event publication.
%%
%% Event-driven pattern: Publish archive hyperparameter events.
%% When self_play_manager is implemented, it will subscribe to this topic.
%%
%% Event: <<"l0.archive_params">>
%% Payload: #{threshold_percentile => ..., sampling_temperature => ..., ...}
apply_archive_params(ArchiveParams, _State) ->
    %% Publish event - self_play_manager (when implemented) will subscribe
    Event = #{
        event_type => <<"l0_archive_params_computed">>,
        timestamp => erlang:system_time(millisecond),
        source => task_l0_actuators,
        params => ArchiveParams
    },
    neuroevolution_events:publish(<<"l0.archive_params">>, Event),
    ok.

%% @private Build evolution parameters from actuator values.
build_evolution_params_from_values(ActuatorValues, State) ->
    #{
        %% Mutation parameters (fallback rates)
        mutation_rate => maps:get(mutation_rate, ActuatorValues, 0.1),
        mutation_strength => maps:get(mutation_strength, ActuatorValues, 0.3),
        add_node_rate => maps:get(add_node_rate, ActuatorValues, 0.03),
        add_connection_rate => maps:get(add_connection_rate, ActuatorValues, 0.05),
        delete_connection_rate => maps:get(delete_connection_rate, ActuatorValues, 0.02),
        weight_perturb_probability => maps:get(weight_perturb_vs_replace, ActuatorValues, 0.9),

        %% Layer-specific mutation rates
        %% These override the fallback rates when layer-specific mutation is enabled
        reservoir_mutation_rate => maps:get(reservoir_mutation_rate, ActuatorValues, 0.08),
        reservoir_mutation_strength => maps:get(reservoir_mutation_strength, ActuatorValues, 0.25),
        readout_mutation_rate => maps:get(readout_mutation_rate, ActuatorValues, 0.15),
        readout_mutation_strength => maps:get(readout_mutation_strength, ActuatorValues, 0.40),

        %% Selection parameters
        selection_ratio => maps:get(selection_ratio, ActuatorValues, 0.2),
        elitism_count => maps:get(elitism_count, ActuatorValues, 2),

        %% Crossover parameters
        crossover_rate => maps:get(crossover_rate, ActuatorValues, 0.75),
        interspecies_crossover_rate => maps:get(interspecies_crossover_rate, ActuatorValues, 0.05),

        %% Population parameters
        population_size => State#state.current_population_size,
        compatibility_threshold => State#state.current_compatibility_threshold
    }.

%% @private Build evolution params for get_evolution_params call.
build_evolution_params(State) ->
    build_evolution_params_from_values(State#state.actuator_values, State).

%% @private Build archive parameters from actuator values.
build_archive_params_from_values(ActuatorValues, State) ->
    #{
        %% Archive entry threshold
        threshold_percentile => maps:get(archive_threshold_percentile, ActuatorValues, 0.7),

        %% Opponent sampling strategy
        sampling_temperature => maps:get(archive_sampling_temperature, ActuatorValues, 0.5),

        %% Archive pruning
        prune_ratio => maps:get(archive_prune_ratio, ActuatorValues, 0.8),

        %% Archive size
        max_size => State#state.current_archive_max_size
    }.

%%% ============================================================================
%%% Internal Functions - Utilities
%%% ============================================================================

%% @private Initial actuator values.
initial_actuator_values() ->
    #{
        %% Evolution actuators (1-12)
        mutation_rate => 0.10,
        mutation_strength => 0.30,
        selection_ratio => 0.20,
        add_node_rate => 0.03,
        add_connection_rate => 0.05,
        delete_connection_rate => 0.02,
        weight_perturb_vs_replace => 0.90,
        crossover_rate => 0.75,
        interspecies_crossover_rate => 0.05,
        elitism_count => 2,
        population_size_delta => 0,
        compatibility_threshold_delta => 0.0,
        %% Layer-specific mutation actuators (13-16)
        %% Initial values: reservoir slightly lower rate than readout
        %% This reflects the reservoir computing principle where hidden layers
        %% benefit from stability while readout adapts faster
        reservoir_mutation_rate => 0.08,
        reservoir_mutation_strength => 0.25,
        readout_mutation_rate => 0.15,
        readout_mutation_strength => 0.40,
        %% Self-play archive actuators (17-20)
        archive_threshold_percentile => 0.70,
        archive_sampling_temperature => 0.50,
        archive_prune_ratio => 0.80,
        archive_max_size_delta => 0
    }.

%% @private Linear interpolation from 0.0-1.0 to target range.
lerp(T, Min, Max) ->
    ClampedT = clamp(T, 0.0, 1.0),
    Min + ClampedT * (Max - Min).

%% @private Clamp value to range.
clamp(Value, Min, Max) ->
    max(Min, min(Max, Value)).
