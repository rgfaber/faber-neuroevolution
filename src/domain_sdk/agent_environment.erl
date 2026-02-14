%% Agent Environment Behaviour.
%%
%% This module defines the behaviour for environments where agents live,
%% perceive, and act. The environment manages world state and episode lifecycle.
%%
%% An environment answers the question: "WHERE does this agent exist?"
%%
%% == Overview ==
%%
%% The environment is the arena in which agents are evaluated. It provides:
%% <ul>
%%   <li><b>World State</b> - Physical space, objects, other entities</li>
%%   <li><b>Episode Lifecycle</b> - Initialize, step, terminate</li>
%%   <li><b>Action Execution</b> - Apply agent actions to world state</li>
%%   <li><b>Metrics Extraction</b> - Gather data for fitness calculation</li>
%% </ul>
%%
%% == Episode Flow ==
%%
%% ```
%% init/1 → spawn_agent/2 → [tick/2 → apply_action/3]* → is_terminal/1 → extract_metrics/1
%%                              ↑_____________________________|
%%                                    (repeat until terminal)
%% '''
%%
%% == Implementing an Environment ==
%%
%% ```
%% -module(hex_arena_env).
%% -behaviour(agent_environment).
%%
%% -export([name/0, init/1, spawn_agent/2, tick/2, apply_action/3,
%%          is_terminal/1, extract_metrics/1]).
%%
%% name() -> <<"hex_arena">>.
%%
%% init(Config) ->
%%     Radius = maps:get(arena_radius, Config, 10),
%%     Walls = generate_walls(Radius),
%%     Food = spawn_initial_food(Walls, 10),
%%     {ok, #{
%%         arena_radius => Radius,
%%         walls => Walls,
%%         food => Food,
%%         tick => 0,
%%         max_ticks => maps:get(max_ticks, Config, 500)
%%     }}.
%%
%% spawn_agent(AgentId, EnvState) ->
%%     %% Place agent at center
%%     Agent = #{id => AgentId, hex => {0, 0}, energy => 100.0},
%%     {ok, Agent, EnvState}.
%%
%% tick(AgentState, EnvState) ->
%%     %% Advance simulation one step
%%     NewEnvState = maybe_spawn_food(EnvState),
%%     {ok, AgentState, NewEnvState#{tick => maps:get(tick, EnvState) + 1}}.
%%
%% apply_action(Action, AgentState, EnvState) ->
%%     %% Execute agent's chosen action
%%     case maps:get(type, Action) of
%%         move -> execute_move(Action, AgentState, EnvState);
%%         eat -> execute_eat(AgentState, EnvState);
%%         _ -> {ok, AgentState, EnvState}
%%     end.
%%
%% is_terminal(EnvState) ->
%%     maps:get(tick, EnvState) >= maps:get(max_ticks, EnvState).
%%
%% extract_metrics(AgentState) ->
%%     #{
%%         ticks_survived => maps:get(age, AgentState, 0),
%%         food_eaten => maps:get(food_eaten, AgentState, 0),
%%         final_energy => maps:get(energy, AgentState, 0)
%%     }.
%% '''
%%
%% == Environment Categories ==
%%
%% Environments can be:
%% <ul>
%%   <li><b>Single-agent</b> - One agent per episode (training isolation)</li>
%%   <li><b>Multi-agent</b> - Multiple agents interact (competition/cooperation)</li>
%%   <li><b>Deterministic</b> - Same seed produces same results</li>
%%   <li><b>Stochastic</b> - Random elements affect outcomes</li>
%% </ul>
%%
%% == Configuration ==
%%
%% The `init/1' callback receives a configuration map. Common parameters:
%% <ul>
%%   <li>`max_ticks' - Maximum episode length</li>
%%   <li>`seed' - Random seed for reproducibility</li>
%%   <li>`arena_radius' - World size</li>
%%   <li>`initial_food' - Starting resources</li>
%% </ul>
%%
%% @author R.G. Lefever
%% @copyright 2024-2026 R.G. Lefever
%% @see agent_definition
%% @see agent_sensor
%% @see agent_actuator
%% @see agent_bridge
-module(agent_environment).

%%% ============================================================================
%%% Types
%%% ============================================================================

-type env_name() :: binary().
%% Unique identifier for the environment type.
%% Examples: `<<"hex_arena">>', `<<"maze">>', `<<"open_field">>'.

-type env_config() :: map().
%% Configuration passed to `init/1'.
%% Structure is domain-defined, typically includes:
%% `max_ticks', `seed', `arena_size', etc.

-type env_state() :: map().
%% Environment state containing world information.
%% Structure is domain-defined, typically includes:
%% `walls', `food', `tick', `max_ticks', etc.

-type agent_id() :: term().
%% Unique identifier for an agent within an episode.

-type agent_state() :: map().
%% Agent-specific state containing position, energy, etc.
%% See `agent_sensor:agent_state()'.

-type action() :: map().
%% Action command from an actuator.
%% See `agent_actuator:action()'.

-type metrics() :: map().
%% Performance metrics extracted after episode.
%% Used by evaluator for fitness calculation.

-export_type([
    env_name/0,
    env_config/0,
    env_state/0,
    agent_id/0,
    agent_state/0,
    action/0,
    metrics/0
]).

%%% ============================================================================
%%% Behaviour Callbacks
%%% ============================================================================

%% Returns the unique name of this environment.
%%
%% The name should be descriptive and unique within the application.
%% It is used for logging, metrics, and configuration.
%%
%% Example:
%% ```
%% name() -> <<"hex_arena">>.
%% '''
-callback name() -> env_name().

%% Initializes the environment with given configuration.
%%
%% Called once at the start of each episode. Should:
%% <ul>
%%   <li>Generate or load world layout (walls, obstacles)</li>
%%   <li>Initialize resources (food, items)</li>
%%   <li>Set episode parameters (max_ticks, etc.)</li>
%% </ul>
%%
%% Example:
%% ```
%% init(Config) ->
%%     Seed = maps:get(seed, Config, erlang:system_time()),
%%     rand:seed(exsss, Seed),
%%     Walls = generate_maze(),
%%     {ok, #{walls => Walls, tick => 0, max_ticks => 500}}.
%% '''
-callback init(Config) -> {ok, EnvState} | {error, Reason} when
    Config :: env_config(),
    EnvState :: env_state(),
    Reason :: term().

%% Spawns an agent into the environment.
%%
%% Called after `init/1' to place the agent in the world.
%% Should create initial agent state with position, energy, etc.
%%
%% For multi-agent environments, this is called once per agent.
%%
%% Example:
%% ```
%% spawn_agent(AgentId, EnvState) ->
%%     StartPos = find_spawn_position(EnvState),
%%     Agent = #{id => AgentId, pos => StartPos, energy => 100.0, age => 0},
%%     {ok, Agent, EnvState}.
%% '''
-callback spawn_agent(AgentId, EnvState) -> {ok, AgentState, EnvState} | {error, Reason} when
    AgentId :: agent_id(),
    EnvState :: env_state(),
    AgentState :: agent_state(),
    Reason :: term().

%% Advances the environment by one time step.
%%
%% Called once per simulation tick BEFORE action application.
%% Use this for:
%% <ul>
%%   <li>Spawning new resources</li>
%%   <li>Environmental hazards/changes</li>
%%   <li>Passive energy decay</li>
%%   <li>Incrementing tick counter</li>
%% </ul>
%%
%% Example:
%% ```
%% tick(AgentState, EnvState) ->
%%     NewEnv = maybe_spawn_food(EnvState),
%%     NewAgent = decay_energy(AgentState),
%%     {ok, NewAgent, NewEnv#{tick => maps:get(tick, EnvState) + 1}}.
%% '''
-callback tick(AgentState, EnvState) -> {ok, AgentState, EnvState} when
    AgentState :: agent_state(),
    EnvState :: env_state().

%% Applies an agent's action to the environment.
%%
%% Called after `tick/2' to execute the agent's chosen action.
%% Should validate and apply the action, updating both agent and env state.
%%
%% Example:
%% ```
%% apply_action(#{type := move, direction := Dir}, Agent, Env) ->
%%     NewPos = compute_new_position(Agent, Dir, Env),
%%     {ok, Agent#{pos => NewPos}, Env};
%% apply_action(#{type := eat}, Agent, Env) ->
%%     case try_eat(Agent, Env) of
%%         {ok, Energy, NewEnv} ->
%%             {ok, Agent#{energy => Energy}, NewEnv};
%%         none ->
%%             {ok, Agent, Env}
%%     end.
%% '''
-callback apply_action(Action, AgentState, EnvState) -> {ok, AgentState, EnvState} when
    Action :: action(),
    AgentState :: agent_state(),
    EnvState :: env_state().

%% Checks if the episode should terminate.
%%
%% Called after each tick to determine if the episode is over.
%% Terminal conditions may include:
%% <ul>
%%   <li>Agent died (energy <= 0)</li>
%%   <li>Maximum ticks reached</li>
%%   <li>Goal achieved</li>
%%   <li>All resources depleted</li>
%% </ul>
%%
%% Example:
%% ```
%% is_terminal(AgentState, EnvState) ->
%%     Energy = maps:get(energy, AgentState, 0),
%%     Tick = maps:get(tick, EnvState),
%%     MaxTicks = maps:get(max_ticks, EnvState),
%%     Energy =< 0 orelse Tick >= MaxTicks.
%% '''
-callback is_terminal(AgentState, EnvState) -> boolean() when
    AgentState :: agent_state(),
    EnvState :: env_state().

%% Extracts performance metrics from the completed episode.
%%
%% Called after episode termination to gather data for fitness calculation.
%% Metrics should capture all relevant performance indicators.
%%
%% Example:
%% ```
%% extract_metrics(AgentState, EnvState) ->
%%     #{
%%         ticks_survived => maps:get(age, AgentState, 0),
%%         food_eaten => maps:get(food_eaten, AgentState, 0),
%%         kills => maps:get(kills, AgentState, 0),
%%         final_energy => maps:get(energy, AgentState, 0),
%%         distance_traveled => maps:get(distance, AgentState, 0)
%%     }.
%% '''
-callback extract_metrics(AgentState, EnvState) -> metrics() when
    AgentState :: agent_state(),
    EnvState :: env_state().

%%% ============================================================================
%%% API Functions
%%% ============================================================================

-export([
    validate/1,
    get_info/1
]).

%% Validates that a module correctly implements the agent_environment behaviour.
%%
%% Checks:
%% <ul>
%%   <li>Module exports all required callbacks</li>
%%   <li>name/0 returns a non-empty binary</li>
%% </ul>
%%
%% Note: Cannot validate other callbacks without actual state arguments.
%%
%% Returns `ok' if valid, or `{error, Reasons}' with a list of validation errors.
-spec validate(Module) -> ok | {error, [Reason]} when
    Module :: module(),
    Reason :: term().
validate(Module) ->
    Checks = [
        fun() -> validate_exports(Module) end,
        fun() -> validate_name(Module) end
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

%% Retrieves environment info from a module.
%%
%% Returns a map with name if the module is valid.
%% Returns `{error, Reason}' if the module doesn't properly implement the behaviour.
-spec get_info(Module) -> {ok, Info} | {error, Reason} when
    Module :: module(),
    Info :: #{name := env_name()},
    Reason :: term().
get_info(Module) ->
    case validate(Module) of
        ok ->
            Name = Module:name(),
            {ok, #{name => Name}};
        {error, _} = Error ->
            Error
    end.

%%% ============================================================================
%%% Internal Functions
%%% ============================================================================

%% @private
validate_exports(Module) ->
    RequiredExports = [
        {name, 0},
        {init, 1},
        {spawn_agent, 2},
        {tick, 2},
        {apply_action, 3},
        {is_terminal, 2},
        {extract_metrics, 2}
    ],
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
