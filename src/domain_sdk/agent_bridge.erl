%% @doc Agent Bridge - Orchestrates the Sense→Think→Act Cycle.
%%
%% This module ties together agent definition, sensors, actuators, and environment
%% to run complete evaluation episodes. It is the integration point between
%% domain-defined behaviours and the neuroevolution engine.
%%
%% The bridge answers the question: "HOW do all the pieces fit together?"
%%
%% == Overview ==
%%
%% The agent bridge:
%% <ul>
%%   <li><b>Registers</b> sensors and actuators for an agent type</li>
%%   <li><b>Validates</b> that topology matches I/O counts</li>
%%   <li><b>Orchestrates</b> the sense→think→act cycle</li>
%%   <li><b>Slices</b> inputs/outputs correctly for each sensor/actuator</li>
%% </ul>
%%
%% == Bridge Configuration ==
%%
%% A bridge config specifies all components of an agent:
%%
%% ```
%% Config = #{
%%     definition => my_agent_definition,
%%     sensors => [vision_sensor, hearing_sensor, energy_sensor],
%%     actuators => [movement_actuator, signal_actuator],
%%     environment => hex_arena_env
%% }
%% '''
%%
%% == Sense→Think→Act Cycle ==
%%
%% Each tick, the bridge executes:
%%
%% ```
%% 1. SENSE: For each sensor, call read/2 → collect inputs
%% 2. THINK: Feed inputs to neural network → get outputs
%% 3. ACT: For each actuator, slice outputs → call act/3 → collect actions
%% 4. APPLY: For each action, call environment:apply_action/3
%% '''
%%
%% == Input/Output Slicing ==
%%
%% Sensors and actuators are processed in registration order:
%%
%% ```
%% Sensors: [vision(18), hearing(4), energy(1)] → Inputs: [0..17, 18..21, 22]
%% Actuators: [movement(7), signal(1)] → Outputs: [0..6, 7]
%% '''
%%
%% == Topology Validation ==
%%
%% The bridge validates that:
%% <ul>
%%   <li>Sum of sensor input_counts == topology inputs</li>
%%   <li>Sum of actuator output_counts == topology outputs</li>
%% </ul>
%%
%% This catches configuration errors before training begins.
%%
%% @author R.G. Lefever
%% @copyright 2024-2026 R.G. Lefever
%% @see agent_definition
%% @see agent_sensor
%% @see agent_actuator
%% @see agent_environment
-module(agent_bridge).

%%% ============================================================================
%%% Types
%%% ============================================================================

-type bridge_config() :: #{
    definition := module(),
    sensors := [module()],
    actuators := [module()],
    environment := module(),
    evaluator => module()  %% Optional: for fitness calculation
}.
%% Configuration specifying all components of an agent.
%% The evaluator is optional - if provided, run_episode returns fitness.

-type validated_bridge() :: #{
    definition := module(),
    sensors := [{module(), non_neg_integer(), pos_integer()}],  %% {Module, Offset, Count}
    actuators := [{module(), non_neg_integer(), pos_integer()}],  %% {Module, Offset, Count}
    environment := module(),
    evaluator => module(),  %% Optional: for fitness calculation
    total_inputs := pos_integer(),
    total_outputs := pos_integer(),
    topology := {pos_integer(), [pos_integer()], pos_integer()}
}.
%% Validated bridge with computed offsets and counts.

-type network() :: term().
%% Neural network (from faber_tweann).

-type agent_state() :: map().
-type env_state() :: map().

-export_type([
    bridge_config/0,
    validated_bridge/0,
    network/0
]).

%%% ============================================================================
%%% API Functions
%%% ============================================================================

-export([
    new/1,
    validate/1,
    sense/3,
    act/4,
    sense_think_act/4,
    run_episode/3,
    run_episode/4
]).

%% @doc Creates and validates a new bridge configuration.
%%
%% Returns a validated bridge with computed I/O offsets, or an error
%% if validation fails.
%%
%% Example:
%% ```
%% Config = #{
%%     definition => my_agent,
%%     sensors => [vision_sensor, energy_sensor],
%%     actuators => [movement_actuator],
%%     environment => arena_env
%% },
%% {ok, Bridge} = agent_bridge:new(Config).
%% '''
-spec new(Config) -> {ok, ValidatedBridge} | {error, Reason} when
    Config :: bridge_config(),
    ValidatedBridge :: validated_bridge(),
    Reason :: term().
new(Config) ->
    validate(Config).

%% @doc Validates a bridge configuration.
%%
%% Checks:
%% <ul>
%%   <li>All modules implement their respective behaviours</li>
%%   <li>Sum of sensor inputs matches topology inputs</li>
%%   <li>Sum of actuator outputs matches topology outputs</li>
%% </ul>
-spec validate(Config) -> {ok, ValidatedBridge} | {error, Reason} when
    Config :: bridge_config(),
    ValidatedBridge :: validated_bridge(),
    Reason :: term().
validate(Config) ->
    try
        %% Extract modules
        Definition = maps:get(definition, Config),
        Sensors = maps:get(sensors, Config),
        Actuators = maps:get(actuators, Config),
        Environment = maps:get(environment, Config),
        Evaluator = maps:get(evaluator, Config, undefined),

        %% Check if already validated (sensors are tuples with offsets)
        case is_already_validated(Sensors) of
            true ->
                %% Already validated, just return it
                {ok, Config};
            false ->
                validate_fresh(Definition, Sensors, Actuators, Environment, Evaluator)
        end
    catch
        throw:Error -> {error, Error};
        error:Reason -> {error, {validation_failed, Reason}}
    end.

%% @private
%% Check if bridge is already validated by examining sensor format
is_already_validated([]) -> false;
is_already_validated([{_Module, _Offset, _Count} | _]) -> true;
is_already_validated([Module | _]) when is_atom(Module) -> false;
is_already_validated(_) -> false.

%% @private
%% Validate a fresh (unvalidated) bridge config
validate_fresh(Definition, Sensors, Actuators, Environment, Evaluator) ->
        %% Validate definition
        ok = validate_module(agent_definition, Definition),

        %% Validate and compute sensor offsets
        {SensorSpecs, TotalInputs} = validate_sensors(Sensors),

        %% Validate and compute actuator offsets
        {ActuatorSpecs, TotalOutputs} = validate_actuators(Actuators),

        %% Validate environment (try multispecies first, then standard)
        ok = validate_environment(Environment),

        %% Validate evaluator if provided
        ok = validate_optional_evaluator(Evaluator),

        %% Get topology and validate I/O counts
        {TopologyInputs, HiddenLayers, TopologyOutputs} = Definition:network_topology(),

        case TotalInputs =:= TopologyInputs of
            false ->
                throw({topology_mismatch, #{
                    type => inputs,
                    expected => TopologyInputs,
                    actual => TotalInputs,
                    sensors => [{M, M:input_count()} || M <- Sensors]
                }});
            true -> ok
        end,

        case TotalOutputs =:= TopologyOutputs of
            false ->
                throw({topology_mismatch, #{
                    type => outputs,
                    expected => TopologyOutputs,
                    actual => TotalOutputs,
                    actuators => [{M, M:output_count()} || M <- Actuators]
                }});
            true -> ok
        end,

        %% Build validated bridge
        BaseBridge = #{
            definition => Definition,
            sensors => SensorSpecs,
            actuators => ActuatorSpecs,
            environment => Environment,
            total_inputs => TotalInputs,
            total_outputs => TotalOutputs,
            topology => {TopologyInputs, HiddenLayers, TopologyOutputs}
        },
        ValidatedBridge = maybe_add_evaluator(BaseBridge, Evaluator),
        {ok, ValidatedBridge}.

%% @doc Collects inputs from all sensors.
%%
%% Calls each sensor's `read/2' in order and concatenates the results.
%% Returns a flat list of floats ready for the neural network.
-spec sense(Bridge, AgentState, EnvState) -> Inputs when
    Bridge :: validated_bridge(),
    AgentState :: agent_state(),
    EnvState :: env_state(),
    Inputs :: [float()].
sense(Bridge, AgentState, EnvState) ->
    Sensors = maps:get(sensors, Bridge),
    lists:flatmap(
        fun({Module, _Offset, _Count}) ->
            Module:read(AgentState, EnvState)
        end,
        Sensors
    ).

%% @doc Processes outputs through all actuators.
%%
%% Slices the output vector and calls each actuator's `act/3'.
%% Returns a list of actions to apply.
-spec act(Bridge, Outputs, AgentState, EnvState) -> Actions when
    Bridge :: validated_bridge(),
    Outputs :: [float()],
    AgentState :: agent_state(),
    EnvState :: env_state(),
    Actions :: [map()].
act(Bridge, Outputs, AgentState, EnvState) ->
    Actuators = maps:get(actuators, Bridge),
    OutputList = if is_list(Outputs) -> Outputs; true -> tuple_to_list(Outputs) end,
    lists:filtermap(
        fun({Module, Offset, Count}) ->
            %% Slice outputs for this actuator
            SlicedOutputs = lists:sublist(OutputList, Offset + 1, Count),
            case Module:act(SlicedOutputs, AgentState, EnvState) of
                {ok, Action} -> {true, Action};
                {error, _} -> false
            end
        end,
        Actuators
    ).

%% @doc Executes one complete sense→think→act cycle.
%%
%% This is the core function called each tick:
%% 1. Sense: Collect inputs from all sensors
%% 2. Think: Evaluate neural network
%% 3. Act: Convert outputs to actions
%%
%% Returns the list of actions to apply to the environment.
-spec sense_think_act(Bridge, Network, AgentState, EnvState) -> {Inputs, Outputs, Actions} when
    Bridge :: validated_bridge(),
    Network :: network(),
    AgentState :: agent_state(),
    EnvState :: env_state(),
    Inputs :: [float()],
    Outputs :: [float()],
    Actions :: [map()].
sense_think_act(Bridge, Network, AgentState, EnvState) ->
    %% 1. SENSE
    Inputs = sense(Bridge, AgentState, EnvState),

    %% 2. THINK (evaluate neural network)
    Outputs = evaluate_network(Network, Inputs),

    %% 3. ACT
    Actions = act(Bridge, Outputs, AgentState, EnvState),

    {Inputs, Outputs, Actions}.

%% @doc Runs a complete evaluation episode.
%%
%% Executes the full episode lifecycle:
%% 1. Initialize environment
%% 2. Spawn agent
%% 3. Loop: tick → sense → think → act → apply actions
%% 4. Extract metrics when terminal
%% 5. Calculate fitness if evaluator is configured
%%
%% Returns:
%% - `{ok, Fitness, Metrics}' if evaluator is configured
%% - `{ok, Metrics}' if no evaluator (backward compatible)
-spec run_episode(Bridge, Network, EnvConfig) -> Result when
    Bridge :: validated_bridge(),
    Network :: network(),
    EnvConfig :: map(),
    Result :: {ok, float(), map()} | {ok, map()} | {error, term()}.
run_episode(Bridge, Network, EnvConfig) ->
    %% For non-multispecies environments, use default spawning
    run_episode(Bridge, Network, EnvConfig, undefined).

%% @doc Runs an evaluation episode with optional species ID.
%%
%% For multi-species environments, the SpeciesId parameter is passed to
%% spawn_agent/3. For standard environments, spawn_agent/2 is used.
-spec run_episode(Bridge, Network, EnvConfig, SpeciesId) -> Result when
    Bridge :: validated_bridge(),
    Network :: network(),
    EnvConfig :: map(),
    SpeciesId :: atom() | undefined,
    Result :: {ok, float(), map()} | {ok, map()} | {error, term()}.
run_episode(Bridge, Network, EnvConfig, SpeciesId) ->
    EnvModule = maps:get(environment, Bridge),
    Evaluator = maps:get(evaluator, Bridge, undefined),

    %% Initialize environment
    case EnvModule:init(EnvConfig) of
        {ok, EnvState0} ->
            %% Spawn agent (use spawn_agent/3 for multispecies, /2 for standard)
            SpawnResult = spawn_agent_for_env(EnvModule, SpeciesId, EnvState0),
            case SpawnResult of
                {ok, AgentState0, EnvState1} ->
                    %% Run episode loop
                    {FinalAgent, FinalEnv} = episode_loop(Bridge, Network, AgentState0, EnvState1, EnvModule),
                    %% Extract metrics
                    Metrics = EnvModule:extract_metrics(FinalAgent, FinalEnv),
                    %% Calculate fitness if evaluator present
                    maybe_calculate_fitness(Evaluator, Metrics);
                {error, Reason} ->
                    {error, {spawn_failed, Reason}}
            end;
        {error, Reason} ->
            {error, {init_failed, Reason}}
    end.

%% @private
%% Spawn agent using the appropriate arity based on environment type
spawn_agent_for_env(EnvModule, undefined, EnvState) ->
    %% Standard environment - spawn_agent/2
    EnvModule:spawn_agent(make_ref(), EnvState);
spawn_agent_for_env(EnvModule, SpeciesId, EnvState) when is_atom(SpeciesId) ->
    %% Multi-species environment - spawn_agent/3
    EnvModule:spawn_agent(make_ref(), SpeciesId, EnvState).

%%% ============================================================================
%%% Internal Functions
%%% ============================================================================

%% @private
validate_module(Behaviour, Module) ->
    case Behaviour:validate(Module) of
        ok -> ok;
        {error, Reasons} -> throw({invalid_module, Module, Behaviour, Reasons})
    end.

%% @private
%% Validate environment - try multispecies first, then standard
validate_environment(Module) ->
    %% Ensure module is loaded
    _ = code:ensure_loaded(Module),

    %% Try multispecies_environment first (has spawn_agent/3)
    case multispecies_environment:validate(Module) of
        ok -> ok;
        {error, _MultiErr} ->
            %% Fall back to standard agent_environment (has spawn_agent/2)
            case agent_environment:validate(Module) of
                ok -> ok;
                {error, Reasons} ->
                    throw({invalid_module, Module, agent_environment, Reasons})
            end
    end.

%% @private
validate_sensors(Sensors) ->
    validate_sensors(Sensors, 0, []).

validate_sensors([], Offset, Acc) ->
    {lists:reverse(Acc), Offset};
validate_sensors([Module | Rest], Offset, Acc) ->
    ok = validate_module(agent_sensor, Module),
    Count = Module:input_count(),
    Spec = {Module, Offset, Count},
    validate_sensors(Rest, Offset + Count, [Spec | Acc]).

%% @private
validate_actuators(Actuators) ->
    validate_actuators(Actuators, 0, []).

validate_actuators([], Offset, Acc) ->
    {lists:reverse(Acc), Offset};
validate_actuators([Module | Rest], Offset, Acc) ->
    ok = validate_module(agent_actuator, Module),
    Count = Module:output_count(),
    Spec = {Module, Offset, Count},
    validate_actuators(Rest, Offset + Count, [Spec | Acc]).

%% @private
episode_loop(Bridge, Network, AgentState, EnvState, EnvModule) ->
    case EnvModule:is_terminal(AgentState, EnvState) of
        true ->
            {AgentState, EnvState};
        false ->
            %% Tick environment
            {ok, AgentState1, EnvState1} = EnvModule:tick(AgentState, EnvState),

            %% Sense→Think→Act
            {_Inputs, _Outputs, Actions} = sense_think_act(Bridge, Network, AgentState1, EnvState1),

            %% Apply all actions
            {AgentState2, EnvState2} = apply_actions(Actions, AgentState1, EnvState1, EnvModule),

            %% Continue loop
            episode_loop(Bridge, Network, AgentState2, EnvState2, EnvModule)
    end.

%% @private
apply_actions([], AgentState, EnvState, _EnvModule) ->
    {AgentState, EnvState};
apply_actions([Action | Rest], AgentState, EnvState, EnvModule) ->
    {ok, NewAgent, NewEnv} = EnvModule:apply_action(Action, AgentState, EnvState),
    apply_actions(Rest, NewAgent, NewEnv, EnvModule).

%% @private
%% Evaluate the neural network.
%% Supports multiple network formats:
%% - network_evaluator record (from faber_tweann)
%% - function/1 (for testing)
%% - map with output_count (for testing stubs)
evaluate_network(Network, Inputs) when is_function(Network, 1) ->
    %% Function-based network (for testing)
    Network(Inputs);
evaluate_network(Network, Inputs) when is_map(Network) ->
    %% Map-based network representation (for testing)
    OutputCount = maps:get(output_count, Network, length(Inputs)),
    lists:duplicate(OutputCount, 0.5);
evaluate_network(Network, Inputs) when is_tuple(Network), element(1, Network) =:= network ->
    %% network_evaluator record from faber_tweann
    network_evaluator:evaluate(Network, Inputs);
evaluate_network(Network, Inputs) ->
    %% Try network_evaluator first (faber_tweann)
    try
        network_evaluator:evaluate(Network, Inputs)
    catch
        error:undef ->
            %% Fallback: return zeros - should only happen without dependencies
            io:format("[agent_bridge] WARNING: network_evaluator not available~n"),
            lists:duplicate(9, 0.0)
    end.

%% @private
%% Validate evaluator if provided
validate_optional_evaluator(undefined) ->
    ok;
validate_optional_evaluator(Evaluator) ->
    validate_module(agent_evaluator, Evaluator).

%% @private
%% Add evaluator to bridge if provided
maybe_add_evaluator(Bridge, undefined) ->
    Bridge;
maybe_add_evaluator(Bridge, Evaluator) ->
    Bridge#{evaluator => Evaluator}.

%% @private
%% Calculate fitness using evaluator if present
maybe_calculate_fitness(undefined, Metrics) ->
    {ok, Metrics};
maybe_calculate_fitness(Evaluator, Metrics) ->
    Fitness = Evaluator:calculate_fitness(Metrics),
    {ok, Fitness, Metrics}.
