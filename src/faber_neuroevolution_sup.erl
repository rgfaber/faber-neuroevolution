%% @doc Top-level supervisor for faber_neuroevolution.
%%
%% This supervisor manages:
%% - Liquid Conglomerate (LC) supervisor for resource/task/distribution silos
%% - Dynamic neuroevolution server supervisor for training sessions
%%
%% == Usage ==
%%
%% Use the API functions to start and stop neuroevolution servers:
%%
%% Config = #neuro_config{...},
%% {ok, Pid} = faber_neuroevolution_sup:start_server(Config),
%% faber_neuroevolution_sup:stop_server(Pid).
%%
%% @author R.G. Lefever
%% @copyright 2024-2026 R.G. Lefever
-module(faber_neuroevolution_sup).
-behaviour(supervisor).

-include("neuroevolution.hrl").

%% API
-export([
    start_link/0,
    start_server/1,
    start_server/2,
    stop_server/1
]).

%% Supervisor callbacks
-export([init/1]).

-define(SUPERVISOR, ?MODULE).
-define(SERVER_SUP, neuro_server_sup).

%%% ============================================================================
%%% API Functions
%%% ============================================================================

%% @doc Start the supervisor.
-spec start_link() -> {ok, pid()} | {error, term()}.
start_link() ->
    supervisor:start_link({local, ?SUPERVISOR}, ?MODULE, []).

%% @doc Start a neuroevolution server with given configuration.
-spec start_server(Config) -> {ok, pid()} | {error, term()} when
    Config :: neuro_config().
start_server(Config) ->
    start_server(Config, []).

%% @doc Start a neuroevolution server with configuration and options.
-spec start_server(Config, Options) -> {ok, pid()} | {error, term()} when
    Config :: neuro_config(),
    Options :: proplists:proplist().
start_server(Config, Options) ->
    supervisor:start_child(?SERVER_SUP, [Config, Options]).

%% @doc Stop a neuroevolution server.
-spec stop_server(Pid) -> ok | {error, term()} when
    Pid :: pid().
stop_server(Pid) ->
    supervisor:terminate_child(?SERVER_SUP, Pid).

%%% ============================================================================
%%% Supervisor Callbacks
%%% ============================================================================

%% @private
init([]) ->
    %% Initialize the local event system
    neuroevolution_events_local:start(),

    SupFlags = #{
        strategy => rest_for_one,  %% LC must start before server supervisor
        intensity => 5,
        period => 10
    },

    %% Liquid Conglomerate supervisor (Resource, Task, Distribution silos)
    %% Read LC config from application environment
    LcConfig = application:get_env(faber_neuroevolution, lc_config, #{}),
    LcSupSpec = #{
        id => lc_supervisor,
        start => {lc_supervisor, start_link, [LcConfig]},
        restart => permanent,
        shutdown => infinity,
        type => supervisor,
        modules => [lc_supervisor]
    },

    %% Dynamic neuroevolution server supervisor
    ServerSupSpec = #{
        id => ?SERVER_SUP,
        start => {supervisor, start_link, [{local, ?SERVER_SUP}, ?MODULE, server_sup]},
        restart => permanent,
        shutdown => infinity,
        type => supervisor,
        modules => []
    },

    {ok, {SupFlags, [LcSupSpec, ServerSupSpec]}};

%% @private Initialize the server supervisor (simple_one_for_one for dynamic servers)
init(server_sup) ->
    SupFlags = #{
        strategy => simple_one_for_one,
        intensity => 5,
        period => 10
    },

    ChildSpec = #{
        id => neuroevolution_server,
        start => {neuroevolution_server, start_link, []},
        restart => temporary,
        shutdown => 5000,
        type => worker,
        modules => [neuroevolution_server]
    },

    {ok, {SupFlags, [ChildSpec]}}.
