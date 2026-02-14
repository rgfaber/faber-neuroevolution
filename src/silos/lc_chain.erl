%% @doc Chained LTC Controller for Liquid Conglomerate.
%%
%% This gen_server manages the L2→L1→L0 chain of LTC TWEANN networks.
%% Each level is a separate neural network that learns at different timescales.
%%
%% == Architecture ==
%%
%% L2 (Strategic, τ=100):
%%   Inputs: Evolution metrics (fitness, stagnation, diversity)
%%   Outputs: 4 strategic signals → L1 inputs
%%
%% L1 (Tactical, τ=50):
%%   Inputs: L2's 4 outputs
%%   Outputs: 5 tactical signals → L0 inputs
%%
%% L0 (Reactive, τ=10):
%%   Inputs: L1's 5 outputs + emergent metrics (evolvable sensors)
%%   Outputs: Final hyperparameters
%%
%% == Usage ==
%%
%% %% Start the chain
%% {ok, Pid} = lc_chain:start_link(Config),
%%
%% %% Forward pass: compute hyperparameters
%% EvoMetrics = #evolution_metrics{best_fitness = 0.85, ...},
%% EmergentMetrics = #emergent_metrics{convergence_rate = 0.02, ...},
%% Hyperparams = lc_chain:forward(Pid, EvoMetrics, EmergentMetrics),
%%
%% %% Train: provide reward signal
%% lc_chain:train(Pid, Reward).
%%
%% @author R.G. Lefever
%% @copyright 2024-2026 R.G. Lefever
-module(lc_chain).
-behaviour(gen_server).

-include("lc_chain.hrl").
-include_lib("faber_tweann/include/records.hrl").

%% API
-export([
    start_link/0,
    start_link/1,
    forward/3,
    forward/2,
    get_hyperparams/1,
    train/2,
    get_state/1,
    reset/1
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

-define(SERVER, ?MODULE).

%%% ============================================================================
%%% API Functions
%%% ============================================================================

%% @doc Start the LC chain with default configuration.
-spec start_link() -> {ok, pid()} | {error, term()}.
start_link() ->
    start_link(#lc_chain_config{}).

%% @doc Start the LC chain with custom configuration.
-spec start_link(lc_chain_config()) -> {ok, pid()} | {error, term()}.
start_link(Config) ->
    gen_server:start_link({local, ?SERVER}, ?MODULE, Config, []).

%% @doc Forward pass through the chain.
%%
%% Computes hyperparameters by cascading through L2→L1→L0.
%%
%% @param ServerRef Server reference
%% @param EvoMetrics Evolution metrics (L2 inputs)
%% @param EmergentMetrics Emergent metrics (L0 additional inputs)
%% @returns Computed hyperparameters
-spec forward(pid() | atom(), evolution_metrics(), emergent_metrics()) -> lc_hyperparams().
forward(ServerRef, EvoMetrics, EmergentMetrics) ->
    gen_server:call(ServerRef, {forward, EvoMetrics, EmergentMetrics}).

%% @doc Forward pass with only evolution metrics.
%%
%% Uses default emergent metrics.
-spec forward(pid() | atom(), evolution_metrics()) -> lc_hyperparams().
forward(ServerRef, EvoMetrics) ->
    forward(ServerRef, EvoMetrics, #emergent_metrics{}).

%% @doc Get last computed hyperparameters.
-spec get_hyperparams(pid() | atom()) -> lc_hyperparams().
get_hyperparams(ServerRef) ->
    gen_server:call(ServerRef, get_hyperparams).

%% @doc Train the chain with a reward signal.
%%
%% Backpropagates reward through all three levels.
%%
%% @param ServerRef Server reference
%% @param Reward Reward signal (higher = better performance)
-spec train(pid() | atom(), float()) -> ok.
train(ServerRef, Reward) ->
    gen_server:cast(ServerRef, {train, Reward}).

%% @doc Get current chain state (for debugging/visualization).
-spec get_state(pid() | atom()) -> {ok, map()}.
get_state(ServerRef) ->
    gen_server:call(ServerRef, get_state).

%% @doc Reset the chain state.
-spec reset(pid() | atom()) -> ok.
reset(ServerRef) ->
    gen_server:cast(ServerRef, reset).

%%% ============================================================================
%%% gen_server Callbacks
%%% ============================================================================

init(Config) ->
    error_logger:info_msg("[lc_chain] Initializing with tau: L2=~p, L1=~p, L0=~p~n",
        [Config#lc_chain_config.l2_tau,
         Config#lc_chain_config.l1_tau,
         Config#lc_chain_config.l0_tau]),

    %% Initialize genotype database (idempotent)
    ok = genotype:init_db(),

    %% Register LC morphologies
    ok = lc_morphologies:register_all(),

    %% Initialize each level
    {ok, L2State} = init_level(l2, Config),
    {ok, L1State} = init_level(l1, Config),
    {ok, L0State} = init_level(l0, Config),

    State = #lc_chain_state{
        config = Config,
        l2_state = L2State,
        l1_state = L1State,
        l0_state = L0State,
        last_hyperparams = hyperparams_to_map(?LC_DEFAULT_HYPERPARAMS),
        running = true
    },

    %% Subscribe to reward events (event-driven pattern)
    controller_events:subscribe_to_reward(),

    error_logger:info_msg("[lc_chain] Initialized all levels~n"),
    {ok, State}.

handle_call({forward, EvoMetrics, EmergentMetrics}, _From, State) ->
    {Hyperparams, NewState} = do_forward(EvoMetrics, EmergentMetrics, State),
    {reply, Hyperparams, NewState};

handle_call(get_hyperparams, _From, State) ->
    Hyperparams = map_to_hyperparams(State#lc_chain_state.last_hyperparams),
    {reply, Hyperparams, State};

handle_call(get_state, _From, State) ->
    Info = #{
        config => State#lc_chain_state.config,
        l2_outputs => (State#lc_chain_state.l2_state)#lc_level_state.last_outputs,
        l1_outputs => (State#lc_chain_state.l1_state)#lc_level_state.last_outputs,
        l0_outputs => (State#lc_chain_state.l0_state)#lc_level_state.last_outputs,
        last_hyperparams => State#lc_chain_state.last_hyperparams,
        cumulative_reward => State#lc_chain_state.cumulative_reward,
        running => State#lc_chain_state.running
    },
    {reply, {ok, Info}, State};

handle_call(_Request, _From, State) ->
    {reply, {error, unknown_request}, State}.

handle_cast({train, Reward}, State) ->
    %% DEPRECATED: Use controller_events:publish_reward/2 instead
    logger:warning("[DEPRECATED] lc_chain:train/2 called. "
                   "Use controller_events:publish_reward/2 instead."),
    NewState = do_train(Reward, State),
    {noreply, NewState};

handle_cast(reset, State) ->
    %% Reset neuron states but keep agent IDs
    L2Reset = reset_level_state(State#lc_chain_state.l2_state),
    L1Reset = reset_level_state(State#lc_chain_state.l1_state),
    L0Reset = reset_level_state(State#lc_chain_state.l0_state),
    NewState = State#lc_chain_state{
        l2_state = L2Reset,
        l1_state = L1Reset,
        l0_state = L0Reset,
        last_hyperparams = hyperparams_to_map(?LC_DEFAULT_HYPERPARAMS),
        reward_history = [],
        cumulative_reward = 0.0
    },
    {noreply, NewState};

handle_cast(_Msg, State) ->
    {noreply, State}.

%% Handle reward signal event (event-driven pattern)
handle_info({neuro_event, <<"controller.reward">>, #{reward := Reward}}, State) ->
    NewState = do_train(Reward, State),
    {noreply, NewState};

handle_info(_Info, State) ->
    {noreply, State}.

terminate(_Reason, _State) ->
    ok.

code_change(_OldVsn, State, _Extra) ->
    {ok, State}.

%%% ============================================================================
%%% Internal Functions - Initialization
%%% ============================================================================

%% @private Initialize a single LC level.
%%
%% Note: Hidden layer configuration (l2_hidden_layers, etc.) is reserved for
%% future enhancement. Currently, faber_tweann's construct_Agent creates a
%% minimal topology (sensors → actuators). Hidden neurons are added through
%% topology evolution via add_node mutations during training.
-spec init_level(l0 | l1 | l2, lc_chain_config()) -> {ok, lc_level_state()}.
init_level(Level, Config) ->
    Morphology = level_to_morphology(Level),
    Tau = level_to_tau(Level, Config),
    %% Hidden layer sizes stored for potential future use with custom topology
    _HiddenLayers = level_to_hidden(Level, Config),

    %% Create constraint for agent construction
    Constraint = #constraint{
        morphology = Morphology,
        neural_afs = [tanh],
        neural_pfns = [none],
        neural_aggr_fs = [dot_product]
    },

    %% Generate unique IDs for species and agent
    SpecieId = {lc_chain, Level, genotype:generate_UniqueId()},
    AgentId = {lc_agent, Level, genotype:generate_UniqueId()},

    %% Construct the agent genotype using faber_tweann API
    %% genotype:construct_Agent/3 expects (SpecieId, AgentId, Constraint)
    try
        _Result = genotype:construct_Agent(SpecieId, AgentId, Constraint),
        error_logger:info_msg("[lc_chain] Created ~p agent: ~p~n", [Level, AgentId]),

        %% Initialize neuron states to zero
        NeuronStates = init_neuron_states(AgentId),

        State = #lc_level_state{
            level = Level,
            agent_id = AgentId,
            tau = Tau,
            neuron_states = NeuronStates,
            last_outputs = lists:duplicate(output_count(Level), 0.0),
            connected_sensors = get_initial_sensors(Level),
            generation = 0
        },
        {ok, State}
    catch
        Error:Reason:Stacktrace ->
            error_logger:error_msg("[lc_chain] Failed to create ~p agent: ~p:~p~n~p~n",
                                   [Level, Error, Reason, Stacktrace]),
            {error, {Error, Reason}}
    end.

%% @private Map level to morphology name.
level_to_morphology(l2) -> lc_l2;
level_to_morphology(l1) -> lc_l1;
level_to_morphology(l0) -> lc_l0.

%% @private Get tau for level from config.
level_to_tau(l2, Config) -> Config#lc_chain_config.l2_tau;
level_to_tau(l1, Config) -> Config#lc_chain_config.l1_tau;
level_to_tau(l0, Config) -> Config#lc_chain_config.l0_tau.

%% @private Get hidden layers for level from config.
level_to_hidden(l2, Config) -> Config#lc_chain_config.l2_hidden_layers;
level_to_hidden(l1, Config) -> Config#lc_chain_config.l1_hidden_layers;
level_to_hidden(l0, Config) -> Config#lc_chain_config.l0_hidden_layers.

%% @private Get output count for level.
output_count(l2) -> 4;  % Strategic signals
output_count(l1) -> 5;  % Tactical signals
output_count(l0) -> 5.  % Hyperparameters

%% @private Get initially connected sensors for level.
get_initial_sensors(l2) ->
    [l2_best_fitness, l2_avg_fitness, l2_fitness_improvement,
     l2_fitness_variance, l2_stagnation_counter, l2_generation_progress,
     l2_population_diversity, l2_species_count];
get_initial_sensors(l1) ->
    [l1_from_l2_signal_1, l1_from_l2_signal_2,
     l1_from_l2_signal_3, l1_from_l2_signal_4];
get_initial_sensors(l0) ->
    %% Only fixed sensors initially - emergent can be added via topology evolution
    [l0_from_l1_signal_1, l0_from_l1_signal_2, l0_from_l1_signal_3,
     l0_from_l1_signal_4, l0_from_l1_signal_5].

%% @private Initialize neuron states to zero.
init_neuron_states(AgentId) ->
    Agent = genotype:dirty_read({agent, AgentId}),
    Cortex = genotype:dirty_read({cortex, Agent#agent.cx_id}),
    NeuronIds = Cortex#cortex.neuron_ids,
    maps:from_list([{NId, 0.0} || NId <- NeuronIds]).

%% @private Reset level state (keep agent, reset neuron states).
reset_level_state(LevelState) ->
    NeuronStates = init_neuron_states(LevelState#lc_level_state.agent_id),
    OutputCount = output_count(LevelState#lc_level_state.level),
    LevelState#lc_level_state{
        neuron_states = NeuronStates,
        last_outputs = lists:duplicate(OutputCount, 0.0),
        generation = 0
    }.

%%% ============================================================================
%%% Internal Functions - Forward Pass
%%% ============================================================================

%% @private Execute forward pass through the chain.
do_forward(EvoMetrics, EmergentMetrics, State) ->
    Config = State#lc_chain_state.config,

    %% L2: Evolution metrics → Strategic signals
    L2Inputs = evolution_metrics_to_inputs(EvoMetrics),
    {L2Outputs, L2NewState} = ltc_forward(
        State#lc_chain_state.l2_state, L2Inputs, Config),

    %% L1: L2 outputs → Tactical signals
    L1Inputs = L2Outputs,
    {L1Outputs, L1NewState} = ltc_forward(
        State#lc_chain_state.l1_state, L1Inputs, Config),

    %% L0: L1 outputs + emergent metrics → Hyperparameters
    L0Inputs = L1Outputs ++ get_emergent_inputs(
        State#lc_chain_state.l0_state, EmergentMetrics),
    {L0Outputs, L0NewState} = ltc_forward(
        State#lc_chain_state.l0_state, L0Inputs, Config),

    %% Scale L0 outputs to hyperparameter ranges
    Hyperparams = outputs_to_hyperparams(L0Outputs),

    NewState = State#lc_chain_state{
        l2_state = L2NewState,
        l1_state = L1NewState,
        l0_state = L0NewState,
        last_hyperparams = hyperparams_to_map(Hyperparams)
    },

    {Hyperparams, NewState}.

%% @private LTC forward pass for a single level.
%%
%% Implements the LTC dynamics:
%%   dx/dt = -x/τ + f(x, I)
%%   x(t+Δt) = x(t) + dx/dt * Δt
%%
%% For now, uses simple Euler integration with Δt=1 (one generation).
ltc_forward(LevelState, Inputs, _Config) ->
    #lc_level_state{
        agent_id = AgentId,
        tau = Tau,
        neuron_states = OldStates
    } = LevelState,

    %% Get network structure
    Agent = genotype:dirty_read({agent, AgentId}),
    Cortex = genotype:dirty_read({cortex, Agent#agent.cx_id}),

    %% Propagate through network with LTC dynamics
    {NewStates, Outputs} = propagate_ltc(
        Cortex, Inputs, OldStates, Tau),

    NewLevelState = LevelState#lc_level_state{
        neuron_states = NewStates,
        last_outputs = Outputs,
        generation = LevelState#lc_level_state.generation + 1
    },

    {Outputs, NewLevelState}.

%% @private Propagate inputs through LTC network.
%%
%% Full implementation:
%% 1. Map sensor inputs to sensor IDs
%% 2. For each neuron in topological order, compute weighted input and LTC update
%% 3. Collect outputs from actuators
propagate_ltc(Cortex, Inputs, OldStates, Tau) ->
    SensorIds = Cortex#cortex.sensor_ids,
    NeuronIds = Cortex#cortex.neuron_ids,
    ActuatorIds = Cortex#cortex.actuator_ids,

    %% Build sensor activation map from inputs
    SensorActivations = build_sensor_activations(SensorIds, Inputs),

    %% Process neurons in layer order (topological sort by layer coordinate)
    SortedNeurons = sort_neurons_by_layer(NeuronIds),

    %% For each neuron, compute LTC update
    DeltaT = 1.0,  % One generation = one time step
    {NewStates, NeuronOutputs} = lists:foldl(
        fun(NeuronId, {AccStates, AccOutputs}) ->
            Neuron = genotype:dirty_read({neuron, NeuronId}),
            OldState = maps:get(NeuronId, AccStates, 0.0),

            %% Compute weighted input from all sources (sensors + other neurons)
            NetInput = compute_weighted_input(Neuron, SensorActivations, AccOutputs),

            %% LTC dynamics: dx/dt = -x/τ + f(net_input)
            %% Use neuron's time_constant if LTC, otherwise use level tau
            NeuronTau = case Neuron#neuron.neuron_type of
                ltc -> Neuron#neuron.time_constant;
                cfc -> Neuron#neuron.time_constant;
                _ -> Tau
            end,

            Decay = -OldState / NeuronTau,
            Activation = activation(Neuron#neuron.af, NetInput),
            NewState = OldState + (Decay + Activation) * DeltaT,

            %% Clamp state to bounds
            StateBound = Neuron#neuron.state_bound,
            ClampedState = max(-StateBound, min(StateBound, NewState)),

            %% Output is the tanh of state (standard) or state itself (LTC)
            Output = case Neuron#neuron.neuron_type of
                standard -> math:tanh(ClampedState);
                _ -> ClampedState
            end,

            {maps:put(NeuronId, ClampedState, AccStates),
             maps:put(NeuronId, Output, AccOutputs)}
        end,
        {OldStates, #{}},
        SortedNeurons
    ),

    %% Collect outputs from actuators
    Outputs = [get_actuator_output(AId, NeuronOutputs) || AId <- ActuatorIds],

    {NewStates, Outputs}.

%% @private Build sensor activation map from input vector.
build_sensor_activations(SensorIds, Inputs) ->
    %% Pair each sensor ID with its corresponding input value
    %% If more inputs than sensors, truncate; if fewer, pad with 0.0
    PaddedInputs = Inputs ++ lists:duplicate(max(0, length(SensorIds) - length(Inputs)), 0.0),
    Pairs = lists:zip(SensorIds, lists:sublist(PaddedInputs, length(SensorIds))),
    maps:from_list(Pairs).

%% @private Sort neurons by layer coordinate for topological processing.
sort_neurons_by_layer(NeuronIds) ->
    lists:sort(
        fun(Id1, Id2) ->
            {{Layer1, _}, neuron} = Id1,
            {{Layer2, _}, neuron} = Id2,
            Layer1 =< Layer2
        end,
        NeuronIds
    ).

%% @private Compute weighted input to a neuron from its input_idps.
compute_weighted_input(Neuron, SensorActivations, NeuronOutputs) ->
    InputIdps = Neuron#neuron.input_idps,
    WeightedSum = lists:foldl(
        fun({SourceId, WeightSpecs}, Acc) ->
            %% Get source activation (from sensor or neuron)
            SourceValue = get_source_activation(SourceId, SensorActivations, NeuronOutputs),
            %% Get first weight from weight specs (simplified: ignore delta, lr, params)
            Weight = extract_weight(WeightSpecs),
            Acc + SourceValue * Weight
        end,
        0.0,
        InputIdps
    ),
    %% Add bias (last element if present, typically weight from 'bias' source)
    WeightedSum.

%% @private Get activation from a source (sensor or neuron).
get_source_activation(SourceId, SensorActivations, NeuronOutputs) ->
    case SourceId of
        bias -> 1.0;  % Bias is always 1.0
        _ ->
            %% Try sensor first, then neuron
            case maps:get(SourceId, SensorActivations, undefined) of
                undefined -> maps:get(SourceId, NeuronOutputs, 0.0);
                Value -> Value
            end
    end.

%% @private Extract weight value from weight specs.
%% Weight format: [{Weight, DeltaWeight, LR, Params}, ...] or just Weight
extract_weight([]) -> 0.0;
extract_weight([{W, _, _, _} | _]) -> W;
extract_weight([W | _]) when is_number(W) -> W;
extract_weight(W) when is_number(W) -> W;
extract_weight(_) -> 0.0.

%% @private Apply activation function.
activation(tanh, X) -> math:tanh(X);
activation(sigmoid, X) -> 1.0 / (1.0 + math:exp(-X));
activation(relu, X) -> max(0.0, X);
activation(linear, X) -> X;
activation(sin, X) -> math:sin(X);
activation(cos, X) -> math:cos(X);
activation(gaussian, X) -> math:exp(-(X * X));
activation(_, X) -> math:tanh(X).  % Default to tanh

%% @private Get output from actuator by reading connected neurons.
get_actuator_output(ActuatorId, NeuronOutputs) ->
    Actuator = genotype:dirty_read({actuator, ActuatorId}),
    FaninIds = Actuator#actuator.fanin_ids,
    case FaninIds of
        [] ->
            %% No inputs, return 0.0
            0.0;
        _ ->
            %% Average of all connected neuron outputs
            Values = [maps:get(NId, NeuronOutputs, 0.0) || NId <- FaninIds],
            lists:sum(Values) / length(Values)
    end.

%%% ============================================================================
%%% Internal Functions - Input/Output Conversion
%%% ============================================================================

%% @private Convert evolution metrics to normalized input vector.
evolution_metrics_to_inputs(#evolution_metrics{} = M) ->
    [
        normalize(M#evolution_metrics.best_fitness, 0.0, 1.0),
        normalize(M#evolution_metrics.avg_fitness, 0.0, 1.0),
        normalize(M#evolution_metrics.fitness_improvement, -0.1, 0.1),
        normalize(M#evolution_metrics.fitness_variance, 0.0, 0.5),
        normalize(M#evolution_metrics.stagnation_counter, 0, 50),
        M#evolution_metrics.generation_progress,  % Already 0-1
        normalize(M#evolution_metrics.population_diversity, 0.0, 1.0),
        normalize(M#evolution_metrics.species_count, 1, 20)
    ].

%% @private Get emergent metric inputs for connected sensors.
get_emergent_inputs(L0State, EmergentMetrics) ->
    ConnectedSensors = L0State#lc_level_state.connected_sensors,
    %% Only include emergent sensors (not l0_from_l1_* which are fixed)
    EmergentSensors = [S || S <- ConnectedSensors,
                            is_emergent_sensor(S)],
    [get_emergent_value(S, EmergentMetrics) || S <- EmergentSensors].

%% @private Check if sensor is an emergent sensor (not fixed L1 input).
is_emergent_sensor(l0_from_l1_signal_1) -> false;
is_emergent_sensor(l0_from_l1_signal_2) -> false;
is_emergent_sensor(l0_from_l1_signal_3) -> false;
is_emergent_sensor(l0_from_l1_signal_4) -> false;
is_emergent_sensor(l0_from_l1_signal_5) -> false;
is_emergent_sensor(_) -> true.

%% @private Get emergent metric value for a sensor.
get_emergent_value(l0_convergence_rate, M) ->
    normalize(M#emergent_metrics.convergence_rate, -0.1, 0.1);
get_emergent_value(l0_fitness_plateau_duration, M) ->
    normalize(M#emergent_metrics.fitness_plateau_duration, 0, 100);
get_emergent_value(l0_current_mutation_rate, M) ->
    normalize(M#emergent_metrics.current_mutation_rate, 0.01, 0.5);
get_emergent_value(l0_current_selection_ratio, M) ->
    normalize(M#emergent_metrics.current_selection_ratio, 0.1, 0.5);
get_emergent_value(l0_survival_rate, M) ->
    M#emergent_metrics.survival_rate;
get_emergent_value(l0_offspring_rate, M) ->
    M#emergent_metrics.offspring_rate;
get_emergent_value(l0_elite_age, M) ->
    normalize(M#emergent_metrics.elite_age, 0, 50);
get_emergent_value(l0_complexity_trend, M) ->
    normalize(M#emergent_metrics.complexity_trend, -1.0, 1.0);
get_emergent_value(l0_avg_network_size, M) ->
    normalize(M#emergent_metrics.avg_network_size, 0, 100);
get_emergent_value(l0_species_extinction_rate, M) ->
    normalize(M#emergent_metrics.species_extinction_rate, 0.0, 1.0);
get_emergent_value(l0_species_creation_rate, M) ->
    normalize(M#emergent_metrics.species_creation_rate, 0.0, 1.0);
get_emergent_value(l0_innovation_rate, M) ->
    normalize(M#emergent_metrics.innovation_rate, 0.0, 1.0);
get_emergent_value(l0_diversity_index, M) ->
    M#emergent_metrics.diversity_index;
get_emergent_value(_, _) ->
    0.5.  % Unknown sensor, return neutral value

%% @private Convert L0 outputs to hyperparameters.
outputs_to_hyperparams(Outputs) ->
    %% Scale outputs from [0,1] to parameter ranges
    [MR, MS, SR, ANR, ACR | _] = Outputs ++ [0.5, 0.5, 0.5, 0.5, 0.5],
    #lc_hyperparams{
        mutation_rate = scale(MR, 0.01, 0.5),
        mutation_strength = scale(MS, 0.05, 1.0),
        selection_ratio = scale(SR, 0.1, 0.5),
        add_node_rate = scale(ANR, 0.0, 0.1),
        add_connection_rate = scale(ACR, 0.0, 0.2)
    }.

%% @private Convert hyperparams record to map.
hyperparams_to_map(#lc_hyperparams{} = H) ->
    #{
        mutation_rate => H#lc_hyperparams.mutation_rate,
        mutation_strength => H#lc_hyperparams.mutation_strength,
        selection_ratio => H#lc_hyperparams.selection_ratio,
        add_node_rate => H#lc_hyperparams.add_node_rate,
        add_connection_rate => H#lc_hyperparams.add_connection_rate
    }.

%% @private Convert map to hyperparams record.
map_to_hyperparams(Map) ->
    #lc_hyperparams{
        mutation_rate = maps:get(mutation_rate, Map, 0.1),
        mutation_strength = maps:get(mutation_strength, Map, 0.3),
        selection_ratio = maps:get(selection_ratio, Map, 0.2),
        add_node_rate = maps:get(add_node_rate, Map, 0.03),
        add_connection_rate = maps:get(add_connection_rate, Map, 0.05)
    }.

%%% ============================================================================
%%% Internal Functions - Training
%%% ============================================================================

%% @private Train the LC chain using reward signal.
%%
%% Uses a simple policy gradient approach:
%% - Positive reward: reinforce recent weight perturbations
%% - Negative reward: reverse recent weight perturbations
%%
%% The learning rate is scaled by tau to provide temporal credit assignment:
%% - L0 (fast, tau=10): Learns quickly, adapts to immediate feedback
%% - L1 (medium, tau=50): Moderate learning, medium-term patterns
%% - L2 (slow, tau=100): Slow learning, long-term strategic changes
do_train(Reward, State) ->
    Config = State#lc_chain_state.config,
    LearningRate = Config#lc_chain_config.learning_rate,

    %% Update weights for each level (reverse order: L0 first, then L1, L2)
    %% This allows credit to flow from immediate to strategic
    L0NewState = update_level_weights(State#lc_chain_state.l0_state, Reward, LearningRate),
    L1NewState = update_level_weights(State#lc_chain_state.l1_state, Reward, LearningRate),
    L2NewState = update_level_weights(State#lc_chain_state.l2_state, Reward, LearningRate),

    %% Track reward history
    NewCumulative = State#lc_chain_state.cumulative_reward + Reward,
    Gen = L0NewState#lc_level_state.generation,
    NewHistory = [{Gen, Reward} | State#lc_chain_state.reward_history],

    State#lc_chain_state{
        l0_state = L0NewState,
        l1_state = L1NewState,
        l2_state = L2NewState,
        reward_history = lists:sublist(NewHistory, 100),
        cumulative_reward = NewCumulative
    }.

%% @private Update weights for a single level based on reward.
update_level_weights(LevelState, Reward, BaseLearningRate) ->
    AgentId = LevelState#lc_level_state.agent_id,
    Tau = LevelState#lc_level_state.tau,

    %% Scale learning rate by tau (faster adaptation for lower tau)
    %% LR_effective = LR_base * (10 / tau)
    EffectiveLR = BaseLearningRate * (10.0 / Tau),

    %% Get agent's cortex to find all neurons
    Agent = genotype:dirty_read({agent, AgentId}),
    Cortex = genotype:dirty_read({cortex, Agent#agent.cx_id}),
    NeuronIds = Cortex#cortex.neuron_ids,

    %% Update each neuron's weights
    lists:foreach(
        fun(NeuronId) ->
            update_neuron_weights(NeuronId, Reward, EffectiveLR)
        end,
        NeuronIds
    ),

    LevelState.

%% @private Update weights for a single neuron using reward signal.
%%
%% Simple weight update rule:
%%   w_new = w_old + lr * reward * sign(w_old) * |output|
%%
%% This reinforces the current weight direction when reward is positive
%% and reverses it when negative.
update_neuron_weights(NeuronId, Reward, LearningRate) ->
    Neuron = genotype:dirty_read({neuron, NeuronId}),
    InputIdps = Neuron#neuron.input_idps,

    %% Update each input connection weight
    NewInputIdps = lists:map(
        fun({SourceId, WeightSpecs}) ->
            NewWeightSpecs = update_weight_specs(WeightSpecs, Reward, LearningRate),
            {SourceId, NewWeightSpecs}
        end,
        InputIdps
    ),

    %% Write updated neuron back to genotype
    UpdatedNeuron = Neuron#neuron{input_idps = NewInputIdps},
    genotype:write(UpdatedNeuron),
    ok.

%% @private Update weight specs with reward signal.
update_weight_specs([], _Reward, _LR) -> [];
update_weight_specs([{W, DW, WLR, Params} | Rest], Reward, LR) ->
    %% Update weight using reward and a small noise term for exploration
    Delta = LR * Reward * sign(W),
    NewW = clamp_weight(W + Delta),
    %% Update delta weight for momentum (optional)
    NewDW = 0.9 * DW + 0.1 * Delta,
    [{NewW, NewDW, WLR, Params} | update_weight_specs(Rest, Reward, LR)];
update_weight_specs([W | Rest], Reward, LR) when is_number(W) ->
    %% Simple weight format (just a number)
    Delta = LR * Reward * sign(W),
    NewW = clamp_weight(W + Delta),
    [NewW | update_weight_specs(Rest, Reward, LR)];
update_weight_specs([Other | Rest], Reward, LR) ->
    [Other | update_weight_specs(Rest, Reward, LR)].

%% @private Sign function.
sign(X) when X > 0 -> 1.0;
sign(X) when X < 0 -> -1.0;
sign(_) -> 0.0.

%% @private Clamp weight to reasonable range.
clamp_weight(W) ->
    max(-10.0, min(10.0, W)).

%%% ============================================================================
%%% Internal Functions - Utilities
%%% ============================================================================

%% @private Normalize value to [0, 1] range.
normalize(Value, Min, Max) when Max > Min ->
    Clamped = max(Min, min(Max, Value)),
    (Clamped - Min) / (Max - Min);
normalize(_, _, _) ->
    0.5.

%% @private Scale value from [0, 1] to [Min, Max] range.
scale(Value, Min, Max) ->
    Clamped = max(0.0, min(1.0, Value)),
    Min + Clamped * (Max - Min).
