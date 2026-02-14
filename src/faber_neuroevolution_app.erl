%% @doc OTP Application module for faber_neuroevolution.
%%
%% This module implements the OTP application behaviour and is responsible
%% for starting the application supervisor.
%%
%% @author R.G. Lefever
%% @copyright 2024-2026 R.G. Lefever
-module(faber_neuroevolution_app).
-behaviour(application).

%% Application callbacks
-export([start/2, stop/1]).

%%% ============================================================================
%%% Application Callbacks
%%% ============================================================================

%% @doc Start the application.
-spec start(StartType, StartArgs) -> {ok, pid()} | {error, term()} when
    StartType :: normal | {takeover, node()} | {failover, node()},
    StartArgs :: term().
start(_StartType, _StartArgs) ->
    faber_neuroevolution_sup:start_link().

%% @doc Stop the application.
-spec stop(State) -> ok when
    State :: term().
stop(_State) ->
    ok.
