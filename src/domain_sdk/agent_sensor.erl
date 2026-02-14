%% Agent Sensor Behaviour.
%%
%% This module defines the behaviour for sensors that read from the environment
%% and produce input values for the agent's neural network.
%%
%% A sensor answers the question: "WHAT can this agent perceive?"
%%
%% == Overview ==
%%
%% Sensors are the input side of an agent's sensorimotor interface. Each sensor:
%% <ul>
%%   <li><b>Reads</b> from the environment state</li>
%%   <li><b>Produces</b> normalized values (typically 0.0 to 1.0) for the neural network</li>
%%   <li><b>Declares</b> how many input nodes it requires</li>
%% </ul>
%%
%% The total number of sensor inputs across all registered sensors must match
%% the `Inputs' value in the agent's `network_topology()'.
%%
%% == Implementing a Sensor ==
%%
%% ```
%% -module(vision_sensor).
%% -behaviour(agent_sensor).
%%
%% -export([name/0, input_count/0, read/2]).
%%
%% name() -> <<"vision">>.
%%
%% %% 6 rays × 3 channels (agent, food, wall) = 18 inputs
%% input_count() -> 18.
%%
%% read(AgentState, EnvState) ->
%%     Hex = maps:get(hex, AgentState),
%%     Agents = maps:get(agents, EnvState),
%%     Food = maps:get(food, EnvState),
%%     Walls = maps:get(walls, EnvState),
%%     Radius = maps:get(arena_radius, EnvState),
%%
%%     %% Cast 6 rays and return 18 values
%%     cast_rays(Hex, Agents, Food, Walls, Radius).
%% '''
%%
%% == Standard Sensor Categories ==
%%
%% Agents typically have multiple sensors:
%% <ul>
%%   <li><b>Exteroceptive</b> - Vision, hearing, smell (environment perception)</li>
%%   <li><b>Proprioceptive</b> - Energy, age, position (self-awareness)</li>
%%   <li><b>Social</b> - Nearby agents, signals, reputation</li>
%% </ul>
%%
%% == Normalization ==
%%
%% Sensor values should be normalized to the range [0.0, 1.0] for best
%% neural network performance. Use inverse-linear scaling for distances:
%% `1.0 / (1.0 + Distance)' to convert unbounded distances to [0, 1].
%%
%% == Latent Capabilities ==
%%
%% Register all potential sensors upfront, even if initially disabled.
%% TWEANN topology mutations can evolve connections to "activate" sensors.
%% A sensor with all-zero outputs effectively becomes dormant until evolved.
%%
%% @author R.G. Lefever
%% @copyright 2024-2026 R.G. Lefever
%% @see agent_definition
%% @see agent_actuator
%% @see agent_bridge
-module(agent_sensor).

%%% ============================================================================
%%% Types
%%% ============================================================================

-type sensor_name() :: binary().
%% Unique identifier for the sensor type within an agent.
%% Examples: `<<"vision">>', `<<"hearing">>', `<<"energy">>'.

-type input_count() :: pos_integer().
%% Number of neural network input nodes this sensor provides.
%% Must be positive (at least 1 input).

-type sensor_values() :: [float()].
%% List of sensor readings, length must match `input_count/0'.
%% Values should be normalized to [0.0, 1.0] range.

-type agent_state() :: map().
%% Agent-specific state containing position, energy, etc.
%% Structure is domain-defined.

-type env_state() :: map().
%% Environment state containing world information.
%% Structure is domain-defined.

-export_type([
    sensor_name/0,
    input_count/0,
    sensor_values/0,
    agent_state/0,
    env_state/0
]).

%%% ============================================================================
%%% Behaviour Callbacks
%%% ============================================================================

%% Returns the unique name of this sensor.
%%
%% The name should be descriptive and unique within the agent.
%% It is used for logging, metrics, and configuration.
%%
%% Example:
%% ```
%% name() -> <<"vision">>.
%% '''
-callback name() -> sensor_name().

%% Returns the number of input nodes this sensor provides.
%%
%% This value is used to:
%% <ul>
%%   <li>Validate network topology matches sensor configuration</li>
%%   <li>Allocate input slots in the neural network</li>
%%   <li>Verify the length of values returned by `read/2'</li>
%% </ul>
%%
%% Example:
%% ```
%% %% Vision sensor: 6 directions × 3 channels = 18 inputs
%% input_count() -> 18.
%% '''
-callback input_count() -> input_count().

%% Reads sensory data and returns normalized values.
%%
%% This is the core sensing function. It:
%% <ul>
%%   <li>Extracts relevant information from agent and environment state</li>
%%   <li>Processes raw data into normalized neural network inputs</li>
%%   <li>Returns exactly `input_count()' float values</li>
%% </ul>
%%
%% Values should be normalized to [0.0, 1.0] for optimal neural network
%% performance. Common normalization strategies:
%% <ul>
%%   <li>Binary presence: 0.0 (absent) or 1.0 (present)</li>
%%   <li>Ratio: `Value / MaxValue'</li>
%%   <li>Distance: `1.0 / (1.0 + Distance)'</li>
%%   <li>Sigmoid: `1.0 / (1.0 + math:exp(-Value))'</li>
%% </ul>
%%
%% Example:
%% ```
%% read(AgentState, EnvState) ->
%%     Energy = maps:get(energy, AgentState),
%%     MaxEnergy = maps:get(max_energy, EnvState, 100.0),
%%     [Energy / MaxEnergy].  %% Single normalized energy value
%% '''
-callback read(AgentState, EnvState) -> sensor_values() when
    AgentState :: agent_state(),
    EnvState :: env_state().

%%% ============================================================================
%%% API Functions
%%% ============================================================================

-export([
    validate/1,
    get_info/1,
    validate_values/2
]).

%% Validates that a module correctly implements the agent_sensor behaviour.
%%
%% Checks:
%% <ul>
%%   <li>Module exports all required callbacks</li>
%%   <li>name/0 returns a non-empty binary</li>
%%   <li>input_count/0 returns a positive integer</li>
%% </ul>
%%
%% Note: Cannot validate read/2 without actual state arguments.
%%
%% Returns `ok' if valid, or `{error, Reasons}' with a list of validation errors.
-spec validate(Module) -> ok | {error, [Reason]} when
    Module :: module(),
    Reason :: term().
validate(Module) ->
    Checks = [
        fun() -> validate_exports(Module) end,
        fun() -> validate_name(Module) end,
        fun() -> validate_input_count(Module) end
    ],
    Errors = lists:filtermap(
        fun(Check) ->
            case Check() of
                ok -> false;
                {error, Reason} -> {true, Reason}
            end
        end,
        Checks
    ),
    case Errors of
        [] -> ok;
        _ -> {error, Errors}
    end.

%% Retrieves sensor info from a module.
%%
%% Returns a map with name and input_count if the module is valid.
%% Returns `{error, Reason}' if the module doesn't properly implement the behaviour.
-spec get_info(Module) -> {ok, Info} | {error, Reason} when
    Module :: module(),
    Info :: #{
        name := sensor_name(),
        input_count := input_count()
    },
    Reason :: term().
get_info(Module) ->
    case validate(Module) of
        ok ->
            Name = Module:name(),
            InputCount = Module:input_count(),
            {ok, #{
                name => Name,
                input_count => InputCount
            }};
        {error, _} = Error ->
            Error
    end.

%% Validates that sensor values match the declared input count.
%%
%% Checks:
%% <ul>
%%   <li>Values is a list</li>
%%   <li>Length matches input_count/0</li>
%%   <li>All values are numbers (integer or float)</li>
%% </ul>
%%
%% Returns `ok' or `{error, Reason}'.
-spec validate_values(Module, Values) -> ok | {error, Reason} when
    Module :: module(),
    Values :: sensor_values(),
    Reason :: term().
validate_values(Module, Values) when is_list(Values) ->
    ExpectedCount = Module:input_count(),
    ActualCount = length(Values),
    case ActualCount =:= ExpectedCount of
        false ->
            {error, {value_count_mismatch, #{expected => ExpectedCount, actual => ActualCount}}};
        true ->
            case lists:all(fun is_number/1, Values) of
                true -> ok;
                false -> {error, {non_numeric_values, Values}}
            end
    end;
validate_values(_Module, Values) ->
    {error, {values_not_list, Values}}.

%%% ============================================================================
%%% Internal Functions
%%% ============================================================================

%% @private
validate_exports(Module) ->
    RequiredExports = [{name, 0}, {input_count, 0}, {read, 2}],
    Exports = Module:module_info(exports),
    Missing = [F || F <- RequiredExports, not lists:member(F, Exports)],
    case Missing of
        [] -> ok;
        _ -> {error, {missing_exports, Missing}}
    end.

%% @private
validate_name(Module) ->
    try Module:name() of
        Name when is_binary(Name), byte_size(Name) > 0 ->
            ok;
        Name when is_binary(Name) ->
            {error, {invalid_name, empty_binary}};
        Other ->
            {error, {invalid_name, {expected_binary, Other}}}
    catch
        _:Reason ->
            {error, {name_callback_failed, Reason}}
    end.

%% @private
validate_input_count(Module) ->
    try Module:input_count() of
        Count when is_integer(Count), Count > 0 ->
            ok;
        Count when is_integer(Count) ->
            {error, {invalid_input_count, {must_be_positive, Count}}};
        Other ->
            {error, {invalid_input_count, {expected_integer, Other}}}
    catch
        _:Reason ->
            {error, {input_count_callback_failed, Reason}}
    end.
