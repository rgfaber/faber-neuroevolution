%% @doc Local event backend using Erlang's pg (process groups).
%%
%% This is the default backend for single-node or development use.
%% For distributed operation, use `neuroevolution_events_macula'.
%%
%% == Message Format ==
%%
%% Subscribers receive messages in the format:
%% `{neuro_event, Topic, Event}'
%%
%% == Example ==
%%
%% Subscribe to events:
%%   neuroevolution_events:subscribe(Topic)
%%
%% In handle_info:
%%   handle_info({neuro_event, Topic, Event}, State) -&gt;
%%       io:format("Received ~p on ~p~n", [Event, Topic]),
%%       {noreply, State}.
%%
%% @author R.G. Lefever
%% @copyright 2024-2026 R.G. Lefever
-module(neuroevolution_events_local).
-behaviour(neuroevolution_events).

-export([
    publish/2,
    subscribe/2,
    unsubscribe/2,
    start/0
]).

%% Process group scope
-define(SCOPE, neuro_events).

%%% ============================================================================
%%% API Functions
%%% ============================================================================

%% @doc Start the local event system.
%%
%% This must be called once at application startup to initialize
%% the pg scope. It's safe to call multiple times.
-spec start() -> ok.
start() ->
    case pg:start(?SCOPE) of
        {ok, _Pid} -> ok;
        {error, {already_started, _Pid}} -> ok
    end.

%% @doc Publish an event to all subscribers of a topic.
-spec publish(Topic, Event) -> ok when
    Topic :: binary(),
    Event :: term().
publish(Topic, Event) ->
    %% Ensure pg scope exists
    _ = start(),

    %% Get all subscribers for this topic
    Members = pg:get_members(?SCOPE, Topic),

    %% Send event to all subscribers
    Message = {neuro_event, Topic, Event},
    lists:foreach(
        fun(Pid) ->
            Pid ! Message
        end,
        Members
    ),
    ok.

%% @doc Subscribe a process to a topic.
-spec subscribe(Topic, Pid) -> ok when
    Topic :: binary(),
    Pid :: pid().
subscribe(Topic, Pid) ->
    %% Ensure pg scope exists
    _ = start(),

    %% Join the process group for this topic
    ok = pg:join(?SCOPE, Topic, Pid),
    ok.

%% @doc Unsubscribe a process from a topic.
-spec unsubscribe(Topic, Pid) -> ok when
    Topic :: binary(),
    Pid :: pid().
unsubscribe(Topic, Pid) ->
    %% Leave the process group
    ok = pg:leave(?SCOPE, Topic, Pid),
    ok.
