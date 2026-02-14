%% Agent Definition Behaviour.
%%
%% This module defines the behaviour that domain applications must implement
%% to define an agent's identity and neural network topology.
%%
%% An agent definition answers the question: "WHO is this agent?"
%%
%% == Overview ==
%%
%% The agent definition is the simplest of the SDK behaviours. It provides:
%% <ul>
%%   <li><b>Identity</b> - A unique name and version for the agent type</li>
%%   <li><b>Network Topology</b> - The shape of the neural network (inputs, hidden, outputs)</li>
%% </ul>
%%
%% The network topology is determined by the sensors and actuators registered
%% with the agent, but this callback provides a declarative specification that
%% can be validated against the actual sensor/actuator counts.
%%
%% == Implementing an Agent Definition ==
%%
%% ```
%% -module(my_arena_agent).
%% -behaviour(agent_definition).
%%
%% -export([name/0, version/0, network_topology/0]).
%%
%% name() -> <<"my_arena_agent">>.
%%
%% version() -> <<"1.0.0">>.
%%
%% network_topology() ->
%%     %% 29 inputs from sensors, 2 hidden layers [32, 16], 9 outputs for actuators
%%     {29, [32, 16], 9}.
%% '''
%%
%% == Network Topology ==
%%
%% The topology tuple `{Inputs, HiddenLayers, Outputs}' specifies:
%% <ul>
%%   <li><b>Inputs</b> - Total number of input nodes (sum of all sensor inputs)</li>
%%   <li><b>HiddenLayers</b> - List of hidden layer sizes, e.g., `[32, 16]'</li>
%%   <li><b>Outputs</b> - Total number of output nodes (sum of all actuator outputs)</li>
%% </ul>
%%
%% The agent_bridge validates that the declared topology matches the registered
%% sensors and actuators.
%%
%% == Semantic Versioning ==
%%
%% Version strings should follow semantic versioning (MAJOR.MINOR.PATCH):
%% <ul>
%%   <li><b>MAJOR</b> - Incompatible changes (network topology changes)</li>
%%   <li><b>MINOR</b> - Backwards-compatible additions (new optional sensors)</li>
%%   <li><b>PATCH</b> - Bug fixes that don't affect the network</li>
%% </ul>
%%
%% @author R.G. Lefever
%% @copyright 2024-2026 R.G. Lefever
%% @see agent_sensor
%% @see agent_actuator
%% @see agent_bridge
-module(agent_definition).

%%% ============================================================================
%%% Types
%%% ============================================================================

-type agent_name() :: binary().
%% Unique identifier for the agent type.
%% Should be a descriptive name like `<<"hex_arena_agent">>'.

-type agent_version() :: binary().
%% Semantic version string, e.g., `<<"1.0.0">>'.

-type network_topology() :: {pos_integer(), [pos_integer()], pos_integer()}.
%% Neural network shape specification: `{Inputs, HiddenLayers, Outputs}'.
%% <ul>
%%   <li>`Inputs' - Total input count, must match sum of all sensor input counts</li>
%%   <li>`HiddenLayers' - List of hidden layer sizes (can be empty for direct connections)</li>
%%   <li>`Outputs' - Total output count, must match sum of all actuator output counts</li>
%% </ul>

-export_type([
    agent_name/0,
    agent_version/0,
    network_topology/0
]).

%%% ============================================================================
%%% Behaviour Callbacks
%%% ============================================================================

%% Returns the unique name of this agent type.
%%
%% The name should be descriptive and unique within the application.
%% It is used for registration, logging, and identification.
%%
%% Example:
%% ```
%% name() -> <<"hex_arena_agent">>.
%% '''
-callback name() -> agent_name().

%% Returns the semantic version of this agent definition.
%%
%% Version changes should follow semantic versioning:
%% <ul>
%%   <li>MAJOR - Network topology changed (incompatible)</li>
%%   <li>MINOR - New optional capabilities added</li>
%%   <li>PATCH - Bug fixes, no network changes</li>
%% </ul>
%%
%% Example:
%% ```
%% version() -> <<"1.2.0">>.
%% '''
-callback version() -> agent_version().

%% Returns the neural network topology for this agent.
%%
%% The topology specifies the shape of the neural network:
%% <ul>
%%   <li>Inputs - Total input count (must match sum of sensor inputs)</li>
%%   <li>HiddenLayers - List of hidden layer sizes</li>
%%   <li>Outputs - Total output count (must match sum of actuator outputs)</li>
%% </ul>
%%
%% Example:
%% ```
%% network_topology() ->
%%     %% 29 inputs (vision:18 + smell:3 + hearing:4 + internal:4)
%%     %% 2 hidden layers [32, 16]
%%     %% 9 outputs (movement:7 + signal:1 + eat:1)
%%     {29, [32, 16], 9}.
%% '''
-callback network_topology() -> network_topology().

%%% ============================================================================
%%% API Functions
%%% ============================================================================

-export([
    validate/1,
    get_info/1
]).

%% Validates that a module correctly implements the agent_definition behaviour.
%%
%% Checks:
%% <ul>
%%   <li>Module exports all required callbacks</li>
%%   <li>name/0 returns a non-empty binary</li>
%%   <li>version/0 returns a valid version string</li>
%%   <li>network_topology/0 returns a valid topology tuple</li>
%% </ul>
%%
%% Returns `ok' if valid, or `{error, Reasons}' with a list of validation errors.
-spec validate(Module) -> ok | {error, [Reason]} when
    Module :: module(),
    Reason :: term().
validate(Module) ->
    Checks = [
        fun() -> validate_exports(Module) end,
        fun() -> validate_name(Module) end,
        fun() -> validate_version(Module) end,
        fun() -> validate_topology(Module) end
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

%% Retrieves agent definition info from a module.
%%
%% Returns a map with name, version, and topology if the module is valid.
%% Returns `{error, Reason}' if the module doesn't properly implement the behaviour.
-spec get_info(Module) -> {ok, Info} | {error, Reason} when
    Module :: module(),
    Info :: #{
        name := agent_name(),
        version := agent_version(),
        topology := network_topology(),
        inputs := pos_integer(),
        hidden_layers := [pos_integer()],
        outputs := pos_integer()
    },
    Reason :: term().
get_info(Module) ->
    case validate(Module) of
        ok ->
            Name = Module:name(),
            Version = Module:version(),
            {Inputs, HiddenLayers, Outputs} = Module:network_topology(),
            {ok, #{
                name => Name,
                version => Version,
                topology => {Inputs, HiddenLayers, Outputs},
                inputs => Inputs,
                hidden_layers => HiddenLayers,
                outputs => Outputs
            }};
        {error, _} = Error ->
            Error
    end.

%%% ============================================================================
%%% Internal Functions
%%% ============================================================================

%% @private
validate_exports(Module) ->
    RequiredExports = [{name, 0}, {version, 0}, {network_topology, 0}],
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
validate_version(Module) ->
    try Module:version() of
        Version when is_binary(Version), byte_size(Version) > 0 ->
            %% Basic semantic version validation (X.Y.Z pattern)
            case re:run(Version, <<"^[0-9]+\\.[0-9]+\\.[0-9]+">>) of
                {match, _} -> ok;
                nomatch -> {error, {invalid_version_format, Version}}
            end;
        Version when is_binary(Version) ->
            {error, {invalid_version, empty_binary}};
        Other ->
            {error, {invalid_version, {expected_binary, Other}}}
    catch
        _:Reason ->
            {error, {version_callback_failed, Reason}}
    end.

%% @private
validate_topology(Module) ->
    try Module:network_topology() of
        {Inputs, HiddenLayers, Outputs}
          when is_integer(Inputs), Inputs > 0,
               is_list(HiddenLayers),
               is_integer(Outputs), Outputs > 0 ->
            %% Validate hidden layers are all positive integers
            case lists:all(fun(H) -> is_integer(H) andalso H > 0 end, HiddenLayers) of
                true -> ok;
                false -> {error, {invalid_hidden_layers, HiddenLayers}}
            end;
        Other ->
            {error, {invalid_topology, Other}}
    catch
        _:Reason ->
            {error, {topology_callback_failed, Reason}}
    end.
