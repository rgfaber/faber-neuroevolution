%%%-------------------------------------------------------------------
%%% @doc Supervisor for mesh distribution components.
%%%
%%% Manages the lifecycle of distributed evaluation components:
%%% - macula_mesh: Macula integration facade
%%% - evaluator_pool_registry: Tracks remote evaluator capacity
%%% - distributed_evaluator: RPC-based evaluation dispatch
%%%
%%% @end
%%%-------------------------------------------------------------------
-module(mesh_sup).

-behaviour(supervisor).

%% API
-export([start_link/1, start_link/0]).

%% Supervisor callbacks
-export([init/1]).

-define(SERVER, ?MODULE).

%%% ============================================================================
%%% API
%%% ============================================================================

-spec start_link() -> {ok, pid()} | {error, term()}.
start_link() ->
    start_link(#{}).

-spec start_link(Config :: map()) -> {ok, pid()} | {error, term()}.
start_link(Config) ->
    supervisor:start_link({local, ?SERVER}, ?MODULE, Config).

%%% ============================================================================
%%% Supervisor Callbacks
%%% ============================================================================

init(Config) ->
    SupFlags = #{
        strategy => one_for_one,
        intensity => 5,
        period => 10
    },

    %% Only start mesh components if mesh is enabled
    MeshEnabled = maps:get(mesh_enabled, Config, false),

    Children = case MeshEnabled of
        true ->
            mesh_children(Config);
        false ->
            %% No children when mesh is disabled
            []
    end,

    {ok, {SupFlags, Children}}.

%%% ============================================================================
%%% Internal Functions
%%% ============================================================================

mesh_children(Config) ->
    [
        %% Evaluator pool registry - tracks remote evaluator capacity
        #{
            id => evaluator_pool_registry,
            start => {evaluator_pool_registry, start_link, [Config]},
            restart => permanent,
            shutdown => 5000,
            type => worker,
            modules => [evaluator_pool_registry]
        },
        %% Macula mesh facade - integration with macula_peer
        #{
            id => macula_mesh,
            start => {macula_mesh, start_link, [Config]},
            restart => permanent,
            shutdown => 5000,
            type => worker,
            modules => [macula_mesh]
        },
        %% Distributed evaluator - RPC-based evaluation dispatch
        #{
            id => distributed_evaluator,
            start => {distributed_evaluator, start_link, [Config]},
            restart => permanent,
            shutdown => 5000,
            type => worker,
            modules => [distributed_evaluator]
        }
    ].
