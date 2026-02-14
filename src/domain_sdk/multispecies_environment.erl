%% @doc Multi-Species Environment Behaviour.
%%
%% Extends agent_environment to support multiple species coevolving
%% in a shared environment.
%%
%% == Overview ==
%%
%% A multi-species environment manages:
%% <ul>
%%   <li>Multiple agent types with different capabilities</li>
%%   <li>Species-specific spawning rules</li>
%%   <li>Inter-species interactions (predation, cooperation)</li>
%%   <li>Per-species metrics extraction</li>
%% </ul>
%%
%% == Agent State Extension ==
%%
%% Each agent state must include:
%% ```
%% #{
%%     id := term(),
%%     species := atom(),          %% Species identifier
%%     species_data := map(),      %% Species-specific state
%%     ...                         %% Standard agent fields
%% }
%% '''
%%
%% == Interaction Types ==
%%
%% ```
%% ┌─────────────────────────────────────────────────────┐
%% │              Interaction Matrix                      │
%% ├─────────────────────────────────────────────────────┤
%% │              │ Forager │ Predator │ Scavenger       │
%% ├──────────────┼─────────┼──────────┼─────────────────┤
%% │ Forager      │ compete │ prey     │ ignore          │
%% │ Predator     │ hunt    │ compete  │ ignore          │
%% │ Scavenger    │ follow  │ avoid    │ compete         │
%% └─────────────────────────────────────────────────────┘
%% '''
%%
%% @author R.G. Lefever
%% @copyright 2024-2026 R.G. Lefever
%% @see agent_environment
%% @see agent_species
-module(multispecies_environment).

%%% ============================================================================
%%% Behaviour Callbacks (extends agent_environment)
%%% ============================================================================

%% Inherit all callbacks from agent_environment
-callback name() -> binary().
-callback init(Config :: map()) -> {ok, EnvState :: map()} | {error, term()}.
-callback tick(AgentState :: map(), EnvState :: map()) ->
    {ok, AgentState :: map(), EnvState :: map()}.
-callback apply_action(Action :: map(), AgentState :: map(), EnvState :: map()) ->
    {ok, AgentState :: map(), EnvState :: map()}.
-callback is_terminal(AgentState :: map(), EnvState :: map()) -> boolean().
-callback extract_metrics(AgentState :: map(), EnvState :: map()) -> map().

%% Multi-species specific callbacks

%% Spawns an agent of a specific species.
%% Uses species spawn_config for initial state.
-callback spawn_agent(AgentId :: term(), SpeciesId :: atom(), EnvState :: map()) ->
    {ok, AgentState :: map(), EnvState :: map()} | {error, term()}.

%% Returns list of supported species.
-callback supported_species() -> [atom()].

%% Handles interaction between two agents.
%% Returns updated states for both agents.
-callback handle_interaction(Agent1 :: map(), Agent2 :: map(), EnvState :: map()) ->
    {ok, Agent1New :: map(), Agent2New :: map(), EnvState :: map()}.

%% Returns interaction type between two species.
-callback interaction_type(Species1 :: atom(), Species2 :: atom()) ->
    hunt | prey | compete | cooperate | ignore | avoid | follow.

%% Extracts species-specific metrics.
-callback extract_species_metrics(SpeciesId :: atom(), AgentStates :: [map()], EnvState :: map()) ->
    map().

-optional_callbacks([
    handle_interaction/3,
    interaction_type/2,
    extract_species_metrics/3
]).

%%% ============================================================================
%%% Types
%%% ============================================================================

-type interaction() :: hunt | prey | compete | cooperate | ignore | avoid | follow.

-type species_spawn_config() :: #{
    energy := float(),
    spawn_zone := center | edge | random | {hex, {integer(), integer()}},
    max_count => pos_integer(),
    spawn_delay => non_neg_integer()
}.

-export_type([interaction/0, species_spawn_config/0]).

%%% ============================================================================
%%% API Functions
%%% ============================================================================

-export([
    validate/1,
    spawn_population/3,
    process_interactions/2,
    extract_all_metrics/2
]).

%% @doc Validates a multi-species environment module.
%%
%% Note: This does NOT call agent_environment:validate/1 because
%% multi-species environments use spawn_agent/3 instead of spawn_agent/2.
-spec validate(Module) -> ok | {error, Reasons} when
    Module :: module(),
    Reasons :: [term()].
validate(Module) ->
    %% Ensure module is loaded (important for Elixir modules)
    _ = code:ensure_loaded(Module),

    %% Validate all required callbacks (combines base + multispecies)
    RequiredExports = [
        {name, 0},
        {init, 1},
        {spawn_agent, 3},  %% Multi-species uses /3 instead of /2
        {tick, 2},
        {apply_action, 3},
        {is_terminal, 2},
        {extract_metrics, 2},
        {supported_species, 0}
    ],
    try
        Exports = Module:module_info(exports),
        Missing = [F || F <- RequiredExports, not lists:member(F, Exports)],
        case Missing of
            [] ->
                %% Validate name returns proper binary
                validate_name(Module);
            _ ->
                {error, {missing_exports, Missing}}
        end
    catch
        _:Reason -> {error, {validation_failed, Reason}}
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

%% @doc Spawns a population of agents for each species.
-spec spawn_population(Module, SpeciesConfigs, EnvState) -> {ok, AgentStates, EnvState} when
    Module :: module(),
    SpeciesConfigs :: #{atom() => {pos_integer(), species_spawn_config()}},
    EnvState :: map(),
    AgentStates :: [map()].
spawn_population(Module, SpeciesConfigs, EnvState) ->
    {Agents, FinalEnv} = maps:fold(
        fun(SpeciesId, {Count, _SpawnConfig}, {AccAgents, AccEnv}) ->
            {NewAgents, NewEnv} = spawn_species_agents(Module, SpeciesId, Count, AccEnv),
            {AccAgents ++ NewAgents, NewEnv}
        end,
        {[], EnvState},
        SpeciesConfigs
    ),
    {ok, Agents, FinalEnv}.

%% @doc Processes interactions between all agent pairs.
-spec process_interactions(Module, State) -> {ok, NewAgents, NewEnv} when
    Module :: module(),
    State :: {[map()], map()},
    NewAgents :: [map()],
    NewEnv :: map().
process_interactions(Module, {AgentStates, EnvState}) ->
    %% Check if module implements handle_interaction
    case erlang:function_exported(Module, handle_interaction, 3) of
        true ->
            process_all_interactions(Module, AgentStates, EnvState);
        false ->
            {ok, AgentStates, EnvState}
    end.

%% @doc Extracts metrics for all species.
-spec extract_all_metrics(Module, State) -> #{atom() => map()} when
    Module :: module(),
    State :: {[map()], map()}.
extract_all_metrics(Module, {AgentStates, EnvState}) ->
    %% Group agents by species
    BySpecies = lists:foldl(
        fun(Agent, Acc) ->
            Species = maps:get(species, Agent, unknown),
            Current = maps:get(Species, Acc, []),
            maps:put(Species, [Agent | Current], Acc)
        end,
        #{},
        AgentStates
    ),

    %% Extract metrics per species
    case erlang:function_exported(Module, extract_species_metrics, 3) of
        true ->
            maps:map(
                fun(SpeciesId, Agents) ->
                    Module:extract_species_metrics(SpeciesId, Agents, EnvState)
                end,
                BySpecies
            );
        false ->
            %% Fallback: use standard metrics
            maps:map(
                fun(_SpeciesId, Agents) ->
                    aggregate_metrics(Module, Agents, EnvState)
                end,
                BySpecies
            )
    end.

%%% ============================================================================
%%% Internal Functions
%%% ============================================================================

%% @private
spawn_species_agents(Module, SpeciesId, Count, EnvState) ->
    lists:foldl(
        fun(I, {Agents, Env}) ->
            AgentId = {SpeciesId, I, make_ref()},
            case Module:spawn_agent(AgentId, SpeciesId, Env) of
                {ok, Agent, NewEnv} ->
                    {[Agent | Agents], NewEnv};
                {error, _Reason} ->
                    {Agents, Env}
            end
        end,
        {[], EnvState},
        lists:seq(1, Count)
    ).

%% @private
process_all_interactions(Module, AgentStates, EnvState) ->
    %% Process each unique pair
    Pairs = [{A, B} || A <- AgentStates, B <- AgentStates, A =/= B],
    UniquePairs = lists:usort([{min(A, B), max(A, B)} || {A, B} <- Pairs]),

    {FinalAgents, FinalEnv} = lists:foldl(
        fun({Agent1, Agent2}, {Agents, Env}) ->
            case Module:handle_interaction(Agent1, Agent2, Env) of
                {ok, NewAgent1, NewAgent2, NewEnv} ->
                    %% Update agents in list
                    Agents1 = update_agent(Agents, Agent1, NewAgent1),
                    Agents2 = update_agent(Agents1, Agent2, NewAgent2),
                    {Agents2, NewEnv};
                _ ->
                    {Agents, Env}
            end
        end,
        {AgentStates, EnvState},
        UniquePairs
    ),
    {ok, FinalAgents, FinalEnv}.

%% @private
update_agent(Agents, OldAgent, NewAgent) ->
    OldId = maps:get(id, OldAgent),
    lists:map(
        fun(A) ->
            case maps:get(id, A) =:= OldId of
                true -> NewAgent;
                false -> A
            end
        end,
        Agents
    ).

%% @private
aggregate_metrics(Module, Agents, EnvState) ->
    AllMetrics = [Module:extract_metrics(A, EnvState) || A <- Agents],
    #{
        count => length(Agents),
        metrics => AllMetrics
    }.
