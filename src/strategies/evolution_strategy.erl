%% @doc Behaviour definition for pluggable evolution strategies.
%%
%% This behaviour allows different evolution paradigms to be implemented
%% as interchangeable modules. Each strategy controls:
%%
%% - When individuals are created and removed
%% - How selection and reproduction work
%% - What events are emitted
%% - What inputs are provided to meta-controllers
%%
%% == Implementing a Strategy ==
%%
%% A strategy module must implement these callbacks:
%%
%% - `init/1' - Initialize strategy state from configuration
%% - `handle_evaluation_result/3' - Process a single evaluation result
%% - `tick/1' - Periodic callback for time-based actions
%% - `get_population_snapshot/1' - Return current population state for UI
%% - `get_meta_inputs/1' - Return inputs for meta-controller
%% - `apply_meta_params/2' - Apply meta-controller parameter adjustments
%%
%% == Built-in Strategies ==
%%
%% - `generational_strategy' - Traditional (mu,lambda) batch evolution
%% - `steady_state_strategy' - Continuous replacement, no generations
%% - `island_strategy' - Parallel populations with migration
%% - `novelty_strategy' - Behavioral novelty search
%% - `map_elites_strategy' - Quality-diversity with niche grid
%%
%% == Actions and Events ==
%%
%% Strategies return `{Actions, Events, NewState}' tuples:
%%
%% Actions are requests for the server to perform:
%% - `{create_individual, ParentIds, Metadata}'
%% - `{remove_individual, Id, Reason}'
%% - `{evaluate_individual, Id}'
%% - `{evaluate_batch, [Id]}'
%%
%% Events are lifecycle notifications:
%% - `#individual_born{}'
%% - `#individual_died{}'
%% - `#generation_advanced{}'
%% - etc.
%%
%% @author R.G. Lefever
%% @copyright 2024-2026 R.G. Lefever
-module(evolution_strategy).

-include("evolution_strategy.hrl").
-include("lifecycle_events.hrl").

%% API exports
-export([
    init/2,
    handle_evaluation_result/4,
    tick/2,
    get_population_snapshot/2,
    get_meta_inputs/2,
    apply_meta_params/3
]).

%% Optional callback check
-export([
    has_callback/3
]).

%%% ============================================================================
%%% Behaviour Callbacks
%%% ============================================================================

%% Initialize the strategy with configuration.
%% Called once when the evolution server starts. The strategy should
%% parse strategy_params from the config, create initial population,
%% and set up any internal state needed.
%% Returns {ok, State} or {error, Reason}.
-callback init(Config :: map()) ->
    {ok, State :: strategy_state()} |
    {error, Reason :: term()}.

%% Handle the result of evaluating an individual.
%% Called after each individual is evaluated. The strategy decides
%% whether to trigger selection/reproduction, remove low-fitness
%% individuals, and what events to emit.
-callback handle_evaluation_result(
    IndividualId :: individual_id(),
    FitnessResult :: #{fitness := fitness(), metrics => map()},
    State :: strategy_state()
) -> strategy_result().

%% Periodic tick for time-based actions.
%% Called periodically by the server. Use for migration, archive
%% maintenance, age-based elimination, statistics emission.
%% Return {[], [], State} if no action needed.
-callback tick(State :: strategy_state()) -> strategy_result().

%% Get a snapshot of the current population state.
%% Used by the UI and monitoring systems. Should return population
%% size, individual summaries, fitness statistics, strategy-specific
%% metrics. Should be lightweight - don't include full network weights.
-callback get_population_snapshot(State :: strategy_state()) ->
    population_snapshot().

%% Get input values for the meta-controller.
%% Returns a list of floats representing the current evolution state.
%% The meta-controller uses these to decide parameter adjustments.
-callback get_meta_inputs(State :: strategy_state()) -> meta_inputs().

%% Apply parameter adjustments from the meta-controller.
%% The meta-controller periodically suggests parameter changes based
%% on evolution progress. The strategy should validate parameters,
%% update internal state, and return the updated state.
-callback apply_meta_params(
    Params :: meta_params(),
    State :: strategy_state()
) -> strategy_state().

%%% ============================================================================
%%% Optional Callbacks
%%% ============================================================================

%% Optional callbacks for specific strategy types.
%% Check with has_callback/3 before calling.

%% Handle migration between islands (island strategies).
-callback handle_migration(
    IndividualId :: individual_id(),
    FromIsland :: island_id(),
    ToIsland :: island_id(),
    State :: strategy_state()
) -> strategy_result().

%% Handle niche update (MAP-Elites).
-callback handle_niche_update(
    NicheId :: niche_id(),
    NewIndividualId :: individual_id(),
    State :: strategy_state()
) -> strategy_result().

%% Handle archive update (novelty search).
-callback handle_archive_update(
    IndividualId :: individual_id(),
    Novelty :: float(),
    State :: strategy_state()
) -> strategy_result().

%% Get island-specific state (island strategies).
-callback get_island_state(
    IslandId :: island_id(),
    State :: strategy_state()
) -> {ok, IslandState :: map()} | {error, not_found}.

%% Get archive state (novelty/MAP-Elites).
-callback get_archive_state(State :: strategy_state()) ->
    {ok, ArchiveState :: map()} | {error, not_supported}.

-optional_callbacks([
    handle_migration/4,
    handle_niche_update/3,
    handle_archive_update/3,
    get_island_state/2,
    get_archive_state/1
]).

%%% ============================================================================
%%% API Functions (dispatch to strategy module)
%%% ============================================================================

%% @doc Initialize a strategy.
-spec init(Module, Config) -> Result when
    Module :: strategy_module(),
    Config :: map(),
    Result :: {ok, strategy_state()} | {error, term()}.
init(Module, Config) ->
    Module:init(Config).

%% @doc Dispatch evaluation result to strategy.
-spec handle_evaluation_result(Module, IndividualId, FitnessResult, State) -> Result when
    Module :: strategy_module(),
    IndividualId :: individual_id(),
    FitnessResult :: map(),
    State :: strategy_state(),
    Result :: strategy_result().
handle_evaluation_result(Module, IndividualId, FitnessResult, State) ->
    Module:handle_evaluation_result(IndividualId, FitnessResult, State).

%% @doc Dispatch tick to strategy.
-spec tick(Module, State) -> Result when
    Module :: strategy_module(),
    State :: strategy_state(),
    Result :: strategy_result().
tick(Module, State) ->
    Module:tick(State).

%% @doc Get population snapshot from strategy.
-spec get_population_snapshot(Module, State) -> population_snapshot() when
    Module :: strategy_module(),
    State :: strategy_state().
get_population_snapshot(Module, State) ->
    Module:get_population_snapshot(State).

%% @doc Get meta-controller inputs from strategy.
-spec get_meta_inputs(Module, State) -> meta_inputs() when
    Module :: strategy_module(),
    State :: strategy_state().
get_meta_inputs(Module, State) ->
    Module:get_meta_inputs(State).

%% @doc Apply meta-controller parameters to strategy.
-spec apply_meta_params(Module, Params, State) -> strategy_state() when
    Module :: strategy_module(),
    Params :: meta_params(),
    State :: strategy_state().
apply_meta_params(Module, Params, State) ->
    Module:apply_meta_params(Params, State).

%%% ============================================================================
%%% Helper Functions
%%% ============================================================================

%% @doc Check if a strategy module exports an optional callback.
-spec has_callback(Module, Function, Arity) -> boolean() when
    Module :: strategy_module(),
    Function :: atom(),
    Arity :: non_neg_integer().
has_callback(Module, Function, Arity) ->
    erlang:function_exported(Module, Function, Arity).
