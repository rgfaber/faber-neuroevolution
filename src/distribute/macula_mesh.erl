%%%-------------------------------------------------------------------
%%% @doc Macula mesh integration facade.
%%%
%%% Provides a unified API for interacting with the Macula mesh platform.
%%% Handles:
%%% - Service advertisement (register as evaluator)
%%% - RPC requests (distributed evaluation)
%%% - Pub/Sub (signal broadcasting)
%%% - DHT operations (service discovery)
%%%
%%% When macula is not available (no MACULA_MESH_ENABLED define),
%%% operations gracefully degrade to local-only mode.
%%%
%%% @end
%%%-------------------------------------------------------------------
-module(macula_mesh).

-behaviour(gen_server).

%% API
-export([
    start_link/1,
    is_mesh_available/0,
    advertise_evaluator/2,
    discover_evaluators/1,
    request_evaluation/4,
    publish_signal/3,
    subscribe_signals/2,
    get_state/0
]).

%% gen_server callbacks
-export([init/1, handle_call/3, handle_cast/2, handle_info/2, terminate/2]).

-define(SERVER, ?MODULE).

-record(state, {
    mesh_available :: boolean(),
    realm :: binary(),
    node_id :: binary(),
    peer_pid :: pid() | undefined,
    evaluator_module :: module() | undefined,
    evaluator_capacity :: pos_integer(),
    subscriptions :: [binary()]
}).

%%% ============================================================================
%%% API
%%% ============================================================================

-spec start_link(Config :: map()) -> {ok, pid()} | {error, term()}.
start_link(Config) ->
    gen_server:start_link({local, ?SERVER}, ?MODULE, Config, []).

-spec is_mesh_available() -> boolean().
is_mesh_available() ->
    gen_server:call(?SERVER, is_mesh_available).

-spec advertise_evaluator(EvaluatorModule :: module(), Capacity :: pos_integer()) -> ok | {error, term()}.
advertise_evaluator(EvaluatorModule, Capacity) ->
    gen_server:call(?SERVER, {advertise_evaluator, EvaluatorModule, Capacity}).

-spec discover_evaluators(Realm :: binary()) -> {ok, [map()]} | {error, term()}.
discover_evaluators(Realm) ->
    gen_server:call(?SERVER, {discover_evaluators, Realm}).

-spec request_evaluation(NodeId :: binary(), Individual :: term(), EvaluatorModule :: module(), Options :: map()) ->
    {ok, RequestId :: binary()} | {error, term()}.
request_evaluation(NodeId, Individual, EvaluatorModule, Options) ->
    gen_server:call(?SERVER, {request_evaluation, NodeId, Individual, EvaluatorModule, Options}).

-spec publish_signal(Topic :: binary(), Signal :: term(), Options :: map()) -> ok | {error, term()}.
publish_signal(Topic, Signal, Options) ->
    gen_server:cast(?SERVER, {publish_signal, Topic, Signal, Options}).

-spec subscribe_signals(Topic :: binary(), Callback :: fun()) -> ok | {error, term()}.
subscribe_signals(Topic, Callback) ->
    gen_server:call(?SERVER, {subscribe_signals, Topic, Callback}).

-spec get_state() -> map().
get_state() ->
    gen_server:call(?SERVER, get_state).

%%% ============================================================================
%%% gen_server Callbacks
%%% ============================================================================

init(Config) ->
    Realm = maps:get(realm, Config, <<"neuroevolution.default">>),
    NodeId = maps:get(node_id, Config, generate_node_id()),
    EvaluatorModule = maps:get(evaluator_module, Config, undefined),
    EvaluatorCapacity = maps:get(evaluator_capacity, Config, erlang:system_info(schedulers)),

    %% Check if macula is available
    MeshAvailable = check_macula_available(),

    State = #state{
        mesh_available = MeshAvailable,
        realm = Realm,
        node_id = NodeId,
        peer_pid = undefined,
        evaluator_module = EvaluatorModule,
        evaluator_capacity = EvaluatorCapacity,
        subscriptions = []
    },

    %% If mesh is available, start macula peer
    FinalState = case MeshAvailable of
        true -> start_macula_peer(State, Config);
        false -> State
    end,

    {ok, FinalState}.

handle_call(is_mesh_available, _From, State) ->
    {reply, State#state.mesh_available, State};

handle_call({advertise_evaluator, EvaluatorModule, Capacity}, _From, State) ->
    #state{mesh_available = MeshAvailable, realm = Realm, node_id = NodeId} = State,

    Result = case MeshAvailable of
        true ->
            %% Advertise evaluator service via macula
            ServiceName = evaluator_service_name(Realm),
            advertise_service(ServiceName, #{
                node_id => NodeId,
                evaluator_module => EvaluatorModule,
                capacity => Capacity
            });
        false ->
            %% Just register locally
            evaluator_pool_registry:register_evaluator(NodeId, #{
                endpoint => <<"local">>,
                capacity => Capacity,
                evaluator_module => EvaluatorModule
            })
    end,

    NewState = State#state{
        evaluator_module = EvaluatorModule,
        evaluator_capacity = Capacity
    },

    {reply, Result, NewState};

handle_call({discover_evaluators, Realm}, _From, State) ->
    #state{mesh_available = MeshAvailable} = State,

    Result = case MeshAvailable of
        true ->
            ServiceName = evaluator_service_name(Realm),
            discover_service(ServiceName);
        false ->
            %% Return local evaluators only
            Evaluators = evaluator_pool_registry:get_all_evaluators(),
            {ok, lists:map(fun evaluator_to_map/1, Evaluators)}
    end,

    {reply, Result, State};

handle_call({request_evaluation, NodeId, Individual, EvaluatorModule, Options}, From, State) ->
    #state{mesh_available = MeshAvailable, realm = Realm, node_id = LocalNodeId} = State,

    case {MeshAvailable, NodeId =:= LocalNodeId} of
        {_, true} ->
            %% Local evaluation
            spawn_link(fun() ->
                Result = evaluate_locally(Individual, EvaluatorModule, Options),
                gen_server:reply(From, Result)
            end),
            {noreply, State};
        {true, false} ->
            %% Remote evaluation via macula RPC
            RequestId = generate_request_id(),
            Callback = maps:get(callback, Options, undefined),

            spawn_link(fun() ->
                Result = request_remote_evaluation(NodeId, Individual, EvaluatorModule, Options, Realm),
                case Callback of
                    undefined -> gen_server:reply(From, Result);
                    Fun when is_function(Fun) -> Fun(Result);
                    Pid when is_pid(Pid) -> Pid ! {evaluation_result, RequestId, Result}
                end
            end),

            {reply, {ok, RequestId}, State};
        {false, false} ->
            %% Mesh not available, can't reach remote node
            {reply, {error, mesh_not_available}, State}
    end;

handle_call({subscribe_signals, Topic, Callback}, _From, State) ->
    #state{mesh_available = MeshAvailable, subscriptions = Subs} = State,

    Result = case MeshAvailable of
        true ->
            subscribe_topic(Topic, Callback);
        false ->
            {error, mesh_not_available}
    end,

    NewState = case Result of
        ok -> State#state{subscriptions = [Topic | Subs]};
        _ -> State
    end,

    {reply, Result, NewState};

handle_call(get_state, _From, State) ->
    #state{
        mesh_available = MeshAvailable,
        realm = Realm,
        node_id = NodeId,
        evaluator_module = EvaluatorModule,
        evaluator_capacity = Capacity,
        subscriptions = Subs
    } = State,

    Info = #{
        mesh_available => MeshAvailable,
        realm => Realm,
        node_id => NodeId,
        evaluator_module => EvaluatorModule,
        evaluator_capacity => Capacity,
        subscriptions => Subs
    },

    {reply, Info, State};

handle_call(_Request, _From, State) ->
    {reply, {error, unknown_request}, State}.

handle_cast({publish_signal, Topic, Signal, _Options}, State) ->
    #state{mesh_available = MeshAvailable, node_id = NodeId} = State,

    case MeshAvailable of
        true ->
            publish_topic(Topic, #{
                node_id => NodeId,
                signal => Signal,
                timestamp => erlang:system_time(millisecond)
            });
        false ->
            ok  % Silently ignore when mesh not available
    end,

    {noreply, State};

handle_cast(_Msg, State) ->
    {noreply, State}.

handle_info({macula_message, Topic, Message}, State) ->
    %% Handle incoming pub/sub messages
    handle_incoming_message(Topic, Message),
    {noreply, State};

handle_info({rpc_request, _Procedure, Args, ReplyTo}, State) ->
    %% Handle incoming RPC requests (we're acting as evaluator)
    #state{evaluator_module = EvaluatorModule} = State,

    spawn_link(fun() ->
        Result = handle_evaluation_request(Args, EvaluatorModule),
        reply_rpc(ReplyTo, Result)
    end),

    {noreply, State};

handle_info(_Info, State) ->
    {noreply, State}.

terminate(_Reason, State) ->
    #state{peer_pid = PeerPid} = State,

    case PeerPid of
        undefined -> ok;
        Pid -> catch stop_macula_peer(Pid)
    end,

    ok.

%%% ============================================================================
%%% Internal Functions - Macula Integration
%%% ============================================================================

check_macula_available() ->
    %% Check if macula application is available
    case code:which(macula_peer) of
        non_existing -> false;
        _ -> true
    end.

start_macula_peer(State, Config) ->
    %% This is a placeholder - actual macula integration would go here
    %% When macula is available, we'd start a macula_peer process
    #state{realm = Realm, node_id = NodeId} = State,

    case check_macula_available() of
        true ->
            %% Start macula peer (when macula is compiled in)
            SeedNodes = maps:get(seed_nodes, Config, []),
            TlsMode = maps:get(tls_mode, Config, development),

            case start_peer(Realm, NodeId, SeedNodes, TlsMode) of
                {ok, PeerPid} ->
                    State#state{peer_pid = PeerPid};
                {error, _Reason} ->
                    State#state{mesh_available = false}
            end;
        false ->
            State
    end.

%% Conditional compilation for macula integration
-ifdef(MACULA_MESH_ENABLED).

start_peer(Realm, NodeId, SeedNodes, TlsMode) ->
    macula_peer:start_link(#{
        realm => Realm,
        node_id => NodeId,
        seed_nodes => SeedNodes,
        tls_mode => TlsMode
    }).

stop_macula_peer(Pid) ->
    gen_server:stop(Pid).

advertise_service(ServiceName, Info) ->
    macula_peer:advertise(ServiceName, Info).

discover_service(ServiceName) ->
    macula_peer:discover(ServiceName).

subscribe_topic(Topic, Callback) ->
    macula_peer:subscribe(Topic, Callback).

publish_topic(Topic, Message) ->
    macula_peer:publish(Topic, Message).

request_remote_evaluation(NodeId, Individual, EvaluatorModule, Options, Realm) ->
    Procedure = <<"neuroevolution.evaluate">>,
    Args = #{
        individual => Individual,
        evaluator_module => EvaluatorModule,
        options => Options
    },
    Timeout = maps:get(timeout_ms, Options, 30000),

    case macula_peer:request(NodeId, Procedure, Args, #{timeout => Timeout}) of
        {ok, Response} -> Response;
        {error, Reason} -> {error, Reason}
    end.

reply_rpc(ReplyTo, Result) ->
    macula_peer:reply(ReplyTo, Result).

-else.

%% Stubs when macula is not available
start_peer(_Realm, _NodeId, _SeedNodes, _TlsMode) ->
    {error, macula_not_available}.

stop_macula_peer(_Pid) ->
    ok.

advertise_service(_ServiceName, _Info) ->
    {error, macula_not_available}.

discover_service(_ServiceName) ->
    {error, macula_not_available}.

subscribe_topic(_Topic, _Callback) ->
    {error, macula_not_available}.

publish_topic(_Topic, _Message) ->
    {error, macula_not_available}.

request_remote_evaluation(_NodeId, _Individual, _EvaluatorModule, _Options, _Realm) ->
    {error, macula_not_available}.

reply_rpc(_ReplyTo, _Result) ->
    {error, macula_not_available}.

-endif.

%%% ============================================================================
%%% Internal Functions - Evaluation
%%% ============================================================================

evaluate_locally(Individual, EvaluatorModule, Options) ->
    StartTime = erlang:system_time(millisecond),

    try
        Result = EvaluatorModule:evaluate(Individual, Options),
        EndTime = erlang:system_time(millisecond),
        LatencyMs = EndTime - StartTime,

        %% Report completion for local stats
        NodeId = get_local_node_id(),
        evaluator_pool_registry:report_evaluation_completed(NodeId, LatencyMs),

        {ok, Result}
    catch
        Class:Reason:Stacktrace ->
            {error, {Class, Reason, Stacktrace}}
    end.

handle_evaluation_request(Args, EvaluatorModule) ->
    Individual = maps:get(individual, Args),
    Options = maps:get(options, Args, #{}),
    RemoteModule = maps:get(evaluator_module, Args, EvaluatorModule),

    %% Use provided module or default
    Module = case RemoteModule of
        undefined -> EvaluatorModule;
        M -> M
    end,

    evaluate_locally(Individual, Module, Options).

handle_incoming_message(_Topic, _Message) ->
    %% Override in actual implementation
    ok.

%%% ============================================================================
%%% Internal Functions - Helpers
%%% ============================================================================

evaluator_service_name(Realm) ->
    <<"neuroevolution.evaluator.", Realm/binary>>.

generate_node_id() ->
    Bytes = crypto:strong_rand_bytes(8),
    <<"node_", (binary:encode_hex(Bytes))/binary>>.

generate_request_id() ->
    Bytes = crypto:strong_rand_bytes(16),
    binary:encode_hex(Bytes).

get_local_node_id() ->
    case whereis(?SERVER) of
        undefined -> <<"unknown">>;
        _Pid ->
            case get_state() of
                #{node_id := NodeId} -> NodeId;
                _ -> <<"unknown">>
            end
    end.

evaluator_to_map(EvaluatorRecord) ->
    %% Convert evaluator record to map
    #{
        node_id => element(2, EvaluatorRecord),
        endpoint => element(3, EvaluatorRecord),
        capacity => element(4, EvaluatorRecord),
        active => element(5, EvaluatorRecord),
        evaluator_module => element(6, EvaluatorRecord),
        latency_ms => element(7, EvaluatorRecord)
    }.
