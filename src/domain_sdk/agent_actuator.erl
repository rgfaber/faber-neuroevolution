%% Agent Actuator Behaviour.
%%
%% This module defines the behaviour for actuators that convert neural network
%% outputs into actions that affect the agent or environment.
%%
%% An actuator answers the question: "WHAT can this agent do?"
%%
%% == Overview ==
%%
%% Actuators are the output side of an agent's sensorimotor interface. Each actuator:
%% <ul>
%%   <li><b>Consumes</b> neural network output values</li>
%%   <li><b>Converts</b> continuous values into discrete or continuous actions</li>
%%   <li><b>Produces</b> action commands for the environment to execute</li>
%% </ul>
%%
%% The total number of actuator outputs across all registered actuators must match
%% the `Outputs' value in the agent's `network_topology()'.
%%
%% == Implementing an Actuator ==
%%
%% ```
%% -module(movement_actuator).
%% -behaviour(agent_actuator).
%%
%% -export([name/0, output_count/0, act/3]).
%%
%% name() -> <<"movement">>.
%%
%% %% 6 hex directions + 1 stay = 7 outputs
%% output_count() -> 7.
%%
%% act(Outputs, AgentState, EnvState) ->
%%     %% Find highest activation direction
%%     {MaxIdx, _MaxVal} = find_max_with_index(Outputs),
%%     Direction = idx_to_direction(MaxIdx),
%%
%%     %% Return action command
%%     {ok, #{type => move, direction => Direction}}.
%% '''
%%
%% == Standard Actuator Categories ==
%%
%% Agents typically have actuators for:
%% <ul>
%%   <li><b>Movement</b> - Locomotion in the environment</li>
%%   <li><b>Interaction</b> - Eating, attacking, trading</li>
%%   <li><b>Communication</b> - Signaling, broadcasting</li>
%%   <li><b>Internal</b> - Energy allocation, reproduction decisions</li>
%% </ul>
%%
%% == Action Types ==
%%
%% Actuators can produce different action types:
%% <ul>
%%   <li><b>Discrete</b> - One of N choices (movement direction)</li>
%%   <li><b>Continuous</b> - A value in a range (signal strength)</li>
%%   <li><b>Boolean</b> - Yes/no decision (attempt to eat)</li>
%% </ul>
%%
%% == Output Interpretation ==
%%
%% Neural network outputs are typically in range [-1, 1] or [0, 1].
%% Common interpretation strategies:
%% <ul>
%%   <li><b>Argmax</b> - Choose the output with highest value</li>
%%   <li><b>Threshold</b> - Activate if above threshold (e.g., 0.5)</li>
%%   <li><b>Proportional</b> - Use value directly as magnitude</li>
%%   <li><b>Softmax</b> - Convert to probability distribution</li>
%% </ul>
%%
%% == Latent Capabilities ==
%%
%% Register all potential actuators upfront, even if initially disabled.
%% TWEANN topology mutations can evolve connections to "activate" actuators.
%% An actuator with all-zero inputs effectively becomes dormant until evolved.
%%
%% @author R.G. Lefever
%% @copyright 2024-2026 R.G. Lefever
%% @see agent_definition
%% @see agent_sensor
%% @see agent_bridge
-module(agent_actuator).

%%% ============================================================================
%%% Types
%%% ============================================================================

-type actuator_name() :: binary().
%% Unique identifier for the actuator type within an agent.
%% Examples: `<<"movement">>', `<<"signal">>', `<<"eat">>'.

-type output_count() :: pos_integer().
%% Number of neural network output nodes this actuator consumes.
%% Must be positive (at least 1 output).

-type output_values() :: [float()].
%% List of neural network outputs to interpret.
%% Length must match `output_count/0'.

-type action() :: map().
%% Action command produced by the actuator.
%% Structure is domain-defined, typically includes:
%% `#{type => atom(), ...action_specific_fields}'.

-type agent_state() :: map().
%% Agent-specific state. See `agent_sensor:agent_state()'.

-type env_state() :: map().
%% Environment state. See `agent_sensor:env_state()'.

-export_type([
    actuator_name/0,
    output_count/0,
    output_values/0,
    action/0,
    agent_state/0,
    env_state/0
]).

%%% ============================================================================
%%% Behaviour Callbacks
%%% ============================================================================

%% Returns the unique name of this actuator.
%%
%% The name should be descriptive and unique within the agent.
%% It is used for logging, metrics, and configuration.
%%
%% Example:
%% ```
%% name() -> <<"movement">>.
%% '''
-callback name() -> actuator_name().

%% Returns the number of output nodes this actuator consumes.
%%
%% This value is used to:
%% <ul>
%%   <li>Validate network topology matches actuator configuration</li>
%%   <li>Allocate output slots in the neural network</li>
%%   <li>Slice the correct portion of outputs for this actuator</li>
%% </ul>
%%
%% Example:
%% ```
%% %% Movement: 6 directions + 1 stay option = 7 outputs
%% output_count() -> 7.
%% '''
-callback output_count() -> output_count().

%% Converts neural network outputs into an action command.
%%
%% This is the core actuation function. It:
%% <ul>
%%   <li>Interprets neural network outputs as intentions</li>
%%   <li>Considers current agent and environment state</li>
%%   <li>Produces an action command for the environment</li>
%% </ul>
%%
%% The function receives exactly `output_count()' values sliced from
%% the full network output vector.
%%
%% Return values:
%% <ul>
%%   <li>`{ok, Action}' - Action to execute</li>
%%   <li>`{error, Reason}' - Action failed validation</li>
%% </ul>
%%
%% Example:
%% ```
%% act(Outputs, AgentState, _EnvState) ->
%%     %% Find direction with highest activation
%%     {Direction, _Score} = best_direction(Outputs),
%%     Energy = maps:get(energy, AgentState),
%%
%%     case Energy > 0.5 of
%%         true -> {ok, #{type => move, direction => Direction}};
%%         false -> {ok, #{type => stay}}  %% Too tired to move
%%     end.
%% '''
-callback act(Outputs, AgentState, EnvState) -> {ok, Action} | {error, Reason} when
    Outputs :: output_values(),
    AgentState :: agent_state(),
    EnvState :: env_state(),
    Action :: action(),
    Reason :: term().

%%% ============================================================================
%%% API Functions
%%% ============================================================================

-export([
    validate/1,
    get_info/1,
    validate_outputs/2
]).

%% Validates that a module correctly implements the agent_actuator behaviour.
%%
%% Checks:
%% <ul>
%%   <li>Module exports all required callbacks</li>
%%   <li>name/0 returns a non-empty binary</li>
%%   <li>output_count/0 returns a positive integer</li>
%% </ul>
%%
%% Note: Cannot validate act/3 without actual state arguments.
%%
%% Returns `ok' if valid, or `{error, Reasons}' with a list of validation errors.
-spec validate(Module) -> ok | {error, [Reason]} when
    Module :: module(),
    Reason :: term().
validate(Module) ->
    Checks = [
        fun() -> validate_exports(Module) end,
        fun() -> validate_name(Module) end,
        fun() -> validate_output_count(Module) end
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

%% Retrieves actuator info from a module.
%%
%% Returns a map with name and output_count if the module is valid.
%% Returns `{error, Reason}' if the module doesn't properly implement the behaviour.
-spec get_info(Module) -> {ok, Info} | {error, Reason} when
    Module :: module(),
    Info :: #{
        name := actuator_name(),
        output_count := output_count()
    },
    Reason :: term().
get_info(Module) ->
    case validate(Module) of
        ok ->
            Name = Module:name(),
            OutputCount = Module:output_count(),
            {ok, #{
                name => Name,
                output_count => OutputCount
            }};
        {error, _} = Error ->
            Error
    end.

%% Validates that output values match the declared output count.
%%
%% Checks:
%% <ul>
%%   <li>Outputs is a list</li>
%%   <li>Length matches output_count/0</li>
%%   <li>All values are numbers (integer or float)</li>
%% </ul>
%%
%% Returns `ok' or `{error, Reason}'.
-spec validate_outputs(Module, Outputs) -> ok | {error, Reason} when
    Module :: module(),
    Outputs :: output_values(),
    Reason :: term().
validate_outputs(Module, Outputs) when is_list(Outputs) ->
    ExpectedCount = Module:output_count(),
    ActualCount = length(Outputs),
    case ActualCount =:= ExpectedCount of
        false ->
            {error, {output_count_mismatch, #{expected => ExpectedCount, actual => ActualCount}}};
        true ->
            case lists:all(fun is_number/1, Outputs) of
                true -> ok;
                false -> {error, {non_numeric_outputs, Outputs}}
            end
    end;
validate_outputs(_Module, Outputs) ->
    {error, {outputs_not_list, Outputs}}.

%%% ============================================================================
%%% Internal Functions
%%% ============================================================================

%% @private
validate_exports(Module) ->
    RequiredExports = [{name, 0}, {output_count, 0}, {act, 3}],
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
validate_output_count(Module) ->
    try Module:output_count() of
        Count when is_integer(Count), Count > 0 ->
            ok;
        Count when is_integer(Count) ->
            {error, {invalid_output_count, {must_be_positive, Count}}};
        Other ->
            {error, {invalid_output_count, {expected_integer, Other}}}
    catch
        _:Reason ->
            {error, {output_count_callback_failed, Reason}}
    end.
