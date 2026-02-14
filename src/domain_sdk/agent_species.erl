%% @doc Agent Species Behaviour - Defines a species/culture of agents.
%%
%% This behaviour extends agent_definition to support multi-species
%% coevolution. Each species has its own network topology, sensors,
%% actuators, and fitness function.
%%
%% == Overview ==
%%
%% Species represent distinct agent types that coevolve in a shared
%% environment. Examples:
%% <ul>
%%   <li><b>Foragers</b> - Optimized for finding and consuming food</li>
%%   <li><b>Predators</b> - Optimized for hunting other agents</li>
%%   <li><b>Scavengers</b> - Optimized for following and opportunistic feeding</li>
%% </ul>
%%
%% == Two-Level Speciation ==
%%
%% ```
%% Level 1: EXPLICIT SPECIES (user-defined)
%%   - Different network topologies
%%   - Different sensors/actuators
%%   - Different fitness functions
%%
%% Level 2: BEHAVIORAL SUB-SPECIES (emergent)
%%   - Same topology, different behaviors
%%   - Identified via behavioral fingerprinting
%%   - Preserves diversity within species
%% ```
%%
%% == Example Implementation ==
%%
%% ```
%% -module(forager_species).
%% -behaviour(agent_species).
%%
%% name() -> <<"forager">>.
%% network_topology() -> {29, [32, 16], 9}.
%% sensors() -> [vision_sensor, smell_sensor, state_sensor].
%% actuators() -> [movement_actuator, signal_actuator].
%% evaluator() -> forager_evaluator.
%% spawn_config() -> #{energy => 150, spawn_zone => center}.
%% subspeciation_threshold() -> 1.5.
%% '''
%%
%% @author R.G. Lefever
%% @copyright 2024-2026 R.G. Lefever
%% @see species_registry
%% @see coevolution_trainer
-module(agent_species).

%%% ============================================================================
%%% Behaviour Callbacks
%%% ============================================================================

%% Returns the species name as a binary.
%% Used for identification and logging.
-callback name() -> binary().

%% Returns the species version.
-callback version() -> binary().

%% Returns the neural network topology.
%% Format: {InputCount, HiddenLayers, OutputCount}
-callback network_topology() -> {pos_integer(), [pos_integer()], pos_integer()}.

%% Returns the list of sensor modules for this species.
%% Each module must implement the agent_sensor behaviour.
-callback sensors() -> [module()].

%% Returns the list of actuator modules for this species.
%% Each module must implement the agent_actuator behaviour.
-callback actuators() -> [module()].

%% Returns the evaluator module for fitness calculation.
%% Must implement the agent_evaluator behaviour.
-callback evaluator() -> module().

%% Returns spawn configuration for this species.
%% Includes starting energy, spawn zone, and other species-specific params.
-callback spawn_config() -> map().

%% Returns the behavioral distance threshold for sub-speciation.
%% Lower values = more sub-species, higher diversity.
%% Set to infinity to disable sub-speciation.
-callback subspeciation_threshold() -> float() | infinity.

%% Optional: Returns species-specific mutation rates.
%% Default: use global rates from trainer config.
-callback mutation_config() -> map().

-optional_callbacks([mutation_config/0]).

%%% ============================================================================
%%% Types
%%% ============================================================================

-type species_id() :: atom() | binary().
-type subspecies_id() :: {species_id(), non_neg_integer()}.

-type species_config() :: #{
    name := binary(),
    topology := {pos_integer(), [pos_integer()], pos_integer()},
    sensors := [module()],
    actuators := [module()],
    evaluator := module(),
    spawn_config := map(),
    subspeciation_threshold := float() | infinity,
    mutation_config => map()
}.

-export_type([
    species_id/0,
    subspecies_id/0,
    species_config/0
]).

%%% ============================================================================
%%% API Functions
%%% ============================================================================

-export([
    validate/1,
    to_config/1,
    to_bridge_config/2
]).

%% @doc Validates that a module correctly implements agent_species.
-spec validate(Module) -> ok | {error, Reasons} when
    Module :: module(),
    Reasons :: [term()].
validate(Module) ->
    %% Ensure module is loaded (important for Elixir modules)
    _ = code:ensure_loaded(Module),

    Callbacks = [
        {name, 0},
        {version, 0},
        {network_topology, 0},
        {sensors, 0},
        {actuators, 0},
        {evaluator, 0},
        {spawn_config, 0},
        {subspeciation_threshold, 0}
    ],
    Missing = lists:filtermap(
        fun({Fun, Arity}) ->
            case erlang:function_exported(Module, Fun, Arity) of
                true -> false;
                false -> {true, {missing_callback, Fun, Arity}}
            end
        end,
        Callbacks
    ),
    case Missing of
        [] -> validate_species_consistency(Module);
        _ -> {error, Missing}
    end.

%% @doc Extracts species configuration from a module.
-spec to_config(Module) -> species_config() when
    Module :: module().
to_config(Module) ->
    BaseConfig = #{
        name => Module:name(),
        topology => Module:network_topology(),
        sensors => Module:sensors(),
        actuators => Module:actuators(),
        evaluator => Module:evaluator(),
        spawn_config => Module:spawn_config(),
        subspeciation_threshold => Module:subspeciation_threshold()
    },
    %% Add optional mutation config if defined
    case erlang:function_exported(Module, mutation_config, 0) of
        true -> BaseConfig#{mutation_config => Module:mutation_config()};
        false -> BaseConfig
    end.

%% @doc Converts species to bridge config for single-species training.
-spec to_bridge_config(Module, Environment) -> agent_bridge:bridge_config() when
    Module :: module(),
    Environment :: module().
to_bridge_config(Module, Environment) ->
    #{
        definition => Module,  %% Species doubles as definition
        sensors => Module:sensors(),
        actuators => Module:actuators(),
        environment => Environment,
        evaluator => Module:evaluator()
    }.

%%% ============================================================================
%%% Internal Functions
%%% ============================================================================

%% @private
validate_species_consistency(Module) ->
    try
        %% Validate topology matches sensors/actuators
        {Inputs, _Hidden, Outputs} = Module:network_topology(),
        Sensors = Module:sensors(),
        Actuators = Module:actuators(),

        %% Calculate expected I/O
        TotalInputs = lists:sum([S:input_count() || S <- Sensors]),
        TotalOutputs = lists:sum([A:output_count() || A <- Actuators]),

        Errors = [],
        Errors1 = case TotalInputs =:= Inputs of
            true -> Errors;
            false -> [{topology_mismatch, inputs, Inputs, TotalInputs} | Errors]
        end,
        Errors2 = case TotalOutputs =:= Outputs of
            true -> Errors1;
            false -> [{topology_mismatch, outputs, Outputs, TotalOutputs} | Errors1]
        end,

        %% Validate evaluator
        Evaluator = Module:evaluator(),
        Errors3 = case agent_evaluator:validate(Evaluator) of
            ok -> Errors2;
            {error, EvalErrors} -> [{invalid_evaluator, EvalErrors} | Errors2]
        end,

        case Errors3 of
            [] -> ok;
            _ -> {error, Errors3}
        end
    catch
        _:Reason -> {error, [{validation_crashed, Reason}]}
    end.
