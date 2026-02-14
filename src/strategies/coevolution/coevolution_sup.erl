%%% @doc Supervisor for competitive coevolution components.
%%%
%%% Manages lifecycle of Red Team archives and coevolution managers for
%%% different realms. Each realm gets its own Red Team and manager.
%%%
%%% == Starting Competitive Coevolution for a Realm ==
%%%
%%% To enable competitive coevolution for a realm:
%%%
%%% {ok, ManagerPid} = coevolution_sup:start_coevolution(my_realm, #{
%%%     red_team_size => 30,
%%%     immigration_rate => 0.05
%%% }).
%%%
%%% The ManagerPid can then be passed to evaluators in their options.
%%%
%%% @end
-module(coevolution_sup).

-behaviour(supervisor).

%% API
-export([
    start_link/0,
    start_coevolution/2,
    stop_coevolution/1,
    get_manager/1,
    list_realms/0
]).

%% Supervisor callbacks
-export([init/1]).

-include_lib("kernel/include/logger.hrl").

-define(SERVER, ?MODULE).

%%====================================================================
%% API
%%====================================================================

%% @doc Start the competitive coevolution supervisor.
-spec start_link() -> {ok, pid()} | {error, term()}.
start_link() ->
    supervisor:start_link({local, ?SERVER}, ?MODULE, []).

%% @doc Start competitive coevolution for a realm.
%% Creates and manages a Red Team and coevolution manager.
%% Options are passed to coevolution_manager:start_link/2.
-spec start_coevolution(Realm :: atom() | binary(), Options :: map()) ->
    {ok, ManagerPid :: pid()} | {error, term()}.
start_coevolution(Realm, Options) ->
    ChildId = child_id(Realm),

    %% Check if already running
    case get_manager(Realm) of
        {ok, ExistingPid} ->
            ?LOG_INFO("[coevolution_sup] Competitive coevolution already running for ~p", [Realm]),
            {ok, ExistingPid};
        {error, not_found} ->
            %% Start new manager
            ChildSpec = #{
                id => ChildId,
                start => {coevolution_manager, start_link, [Realm, Options]},
                restart => transient,
                shutdown => 5000,
                type => worker,
                modules => [coevolution_manager]
            },

            case supervisor:start_child(?SERVER, ChildSpec) of
                {ok, Pid} ->
                    ?LOG_INFO("[coevolution_sup] Started competitive coevolution for ~p (Red Team vs Blue Team)", [Realm]),
                    {ok, Pid};
                {ok, Pid, _Info} ->
                    {ok, Pid};
                {error, {already_started, Pid}} ->
                    {ok, Pid};
                {error, Reason} = Error ->
                    ?LOG_ERROR("[coevolution_sup] Failed to start competitive coevolution for ~p: ~p",
                              [Realm, Reason]),
                    Error
            end
    end.

%% @doc Stop competitive coevolution for a realm.
-spec stop_coevolution(Realm :: atom() | binary()) -> ok | {error, not_found}.
stop_coevolution(Realm) ->
    ChildId = child_id(Realm),
    case supervisor:terminate_child(?SERVER, ChildId) of
        ok ->
            supervisor:delete_child(?SERVER, ChildId),
            ?LOG_INFO("[coevolution_sup] Stopped competitive coevolution for ~p", [Realm]),
            ok;
        {error, not_found} ->
            {error, not_found}
    end.

%% @doc Get the coevolution manager PID for a realm.
-spec get_manager(Realm :: atom() | binary()) -> {ok, pid()} | {error, not_found}.
get_manager(Realm) ->
    ChildId = child_id(Realm),
    Children = supervisor:which_children(?SERVER),
    case lists:keyfind(ChildId, 1, Children) of
        {ChildId, Pid, worker, _} when is_pid(Pid) ->
            {ok, Pid};
        {ChildId, restarting, worker, _} ->
            %% Child is restarting, wait and retry
            timer:sleep(100),
            get_manager(Realm);
        _ ->
            {error, not_found}
    end.

%% @doc List all realms with active competitive coevolution.
-spec list_realms() -> [atom() | binary()].
list_realms() ->
    Children = supervisor:which_children(?SERVER),
    lists:filtermap(
        fun({Id, Pid, worker, _}) when is_pid(Pid) ->
            case extract_realm(Id) of
                {ok, Realm} -> {true, Realm};
                error -> false
            end;
           (_) -> false
        end,
        Children
    ).

%%====================================================================
%% Supervisor callbacks
%%====================================================================

init([]) ->
    SupFlags = #{
        strategy => one_for_one,
        intensity => 5,
        period => 60
    },

    %% Start with no children - realms are added dynamically
    ChildSpecs = [],

    ?LOG_INFO("[coevolution_sup] Starting Competitive Coevolution supervisor"),
    {ok, {SupFlags, ChildSpecs}}.

%%====================================================================
%% Internal functions
%%====================================================================

child_id(Realm) when is_atom(Realm) ->
    list_to_atom("coevolution_" ++ atom_to_list(Realm));
child_id(Realm) when is_binary(Realm) ->
    list_to_atom("coevolution_" ++ binary_to_list(Realm)).

extract_realm(ChildId) when is_atom(ChildId) ->
    Name = atom_to_list(ChildId),
    case lists:prefix("coevolution_", Name) of
        true ->
            RealmStr = lists:nthtail(length("coevolution_"), Name),
            {ok, list_to_atom(RealmStr)};
        false ->
            error
    end;
extract_realm(_) ->
    error.
