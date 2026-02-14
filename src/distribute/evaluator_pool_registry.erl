%%%-------------------------------------------------------------------
%%% @doc Evaluator pool registry for distributed evaluation.
%%%
%%% Tracks the capacity and availability of evaluator nodes across the mesh.
%%% Provides load-balanced selection of evaluator nodes based on:
%%% - Current load (active evaluations)
%%% - Maximum capacity (CPU cores / configured limit)
%%% - Latency estimates
%%% - Node health
%%%
%%% @end
%%%-------------------------------------------------------------------
-module(evaluator_pool_registry).

-behaviour(gen_server).

%% API
-export([
    start_link/1,
    register_evaluator/2,
    unregister_evaluator/1,
    get_available_evaluator/0,
    get_available_evaluator/1,
    get_all_evaluators/0,
    report_evaluation_started/1,
    report_evaluation_completed/2,
    get_stats/0
]).

%% gen_server callbacks
-export([init/1, handle_call/3, handle_cast/2, handle_info/2, terminate/2]).

-define(SERVER, ?MODULE).
-define(CLEANUP_INTERVAL_MS, 30000).  % 30 seconds
-define(STALE_THRESHOLD_MS, 60000).   % 60 seconds without heartbeat

-record(evaluator_node, {
    node_id :: binary(),
    endpoint :: binary(),
    capacity :: pos_integer(),
    active :: non_neg_integer(),
    evaluator_module :: module(),
    latency_ms :: non_neg_integer(),
    last_heartbeat :: integer(),
    error_count :: non_neg_integer()
}).

-record(state, {
    evaluators :: ets:tid(),
    local_node_id :: binary(),
    realm :: binary(),
    cleanup_timer :: reference() | undefined
}).

%%% ============================================================================
%%% API
%%% ============================================================================

-spec start_link(Config :: map()) -> {ok, pid()} | {error, term()}.
start_link(Config) ->
    gen_server:start_link({local, ?SERVER}, ?MODULE, Config, []).

-spec register_evaluator(NodeId :: binary(), Info :: map()) -> ok.
register_evaluator(NodeId, Info) ->
    gen_server:call(?SERVER, {register_evaluator, NodeId, Info}).

-spec unregister_evaluator(NodeId :: binary()) -> ok.
unregister_evaluator(NodeId) ->
    gen_server:call(?SERVER, {unregister_evaluator, NodeId}).

-spec get_available_evaluator() -> {ok, #evaluator_node{}} | {error, no_evaluators}.
get_available_evaluator() ->
    get_available_evaluator(#{}).

-spec get_available_evaluator(Options :: map()) -> {ok, #evaluator_node{}} | {error, no_evaluators}.
get_available_evaluator(Options) ->
    gen_server:call(?SERVER, {get_available_evaluator, Options}).

-spec get_all_evaluators() -> [#evaluator_node{}].
get_all_evaluators() ->
    gen_server:call(?SERVER, get_all_evaluators).

-spec report_evaluation_started(NodeId :: binary()) -> ok.
report_evaluation_started(NodeId) ->
    gen_server:cast(?SERVER, {evaluation_started, NodeId}).

-spec report_evaluation_completed(NodeId :: binary(), LatencyMs :: non_neg_integer()) -> ok.
report_evaluation_completed(NodeId, LatencyMs) ->
    gen_server:cast(?SERVER, {evaluation_completed, NodeId, LatencyMs}).

-spec get_stats() -> map().
get_stats() ->
    gen_server:call(?SERVER, get_stats).

%%% ============================================================================
%%% gen_server Callbacks
%%% ============================================================================

init(Config) ->
    Realm = maps:get(realm, Config, <<"neuroevolution.default">>),
    LocalNodeId = maps:get(node_id, Config, generate_node_id()),

    %% Create ETS table for evaluator tracking
    Evaluators = ets:new(evaluator_nodes, [
        set,
        {keypos, #evaluator_node.node_id},
        {read_concurrency, true}
    ]),

    %% Schedule cleanup timer
    Timer = erlang:send_after(?CLEANUP_INTERVAL_MS, self(), cleanup_stale),

    State = #state{
        evaluators = Evaluators,
        local_node_id = LocalNodeId,
        realm = Realm,
        cleanup_timer = Timer
    },

    {ok, State}.

handle_call({register_evaluator, NodeId, Info}, _From, State) ->
    #state{evaluators = Evaluators} = State,

    Node = #evaluator_node{
        node_id = NodeId,
        endpoint = maps:get(endpoint, Info, <<>>),
        capacity = maps:get(capacity, Info, 1),
        active = 0,
        evaluator_module = maps:get(evaluator_module, Info, undefined),
        latency_ms = maps:get(latency_ms, Info, 100),
        last_heartbeat = erlang:system_time(millisecond),
        error_count = 0
    },

    ets:insert(Evaluators, Node),
    {reply, ok, State};

handle_call({unregister_evaluator, NodeId}, _From, State) ->
    #state{evaluators = Evaluators} = State,
    ets:delete(Evaluators, NodeId),
    {reply, ok, State};

handle_call({get_available_evaluator, Options}, _From, State) ->
    #state{evaluators = Evaluators, local_node_id = LocalNodeId} = State,

    PreferLocal = maps:get(prefer_local, Options, 0.2),

    %% Get all evaluators with available capacity
    Available = ets:foldl(
        fun(Node, Acc) ->
            case Node#evaluator_node.active < Node#evaluator_node.capacity of
                true -> [Node | Acc];
                false -> Acc
            end
        end,
        [],
        Evaluators
    ),

    Result = select_evaluator(Available, LocalNodeId, PreferLocal),
    {reply, Result, State};

handle_call(get_all_evaluators, _From, State) ->
    #state{evaluators = Evaluators} = State,
    All = ets:tab2list(Evaluators),
    {reply, All, State};

handle_call(get_stats, _From, State) ->
    #state{evaluators = Evaluators, realm = Realm} = State,

    InitialStats = #{
        realm => Realm,
        total_nodes => 0,
        total_capacity => 0,
        total_active => 0,
        total_errors => 0
    },

    Stats = ets:foldl(
        fun(Node, Acc) ->
            Acc#{
                total_nodes => maps:get(total_nodes, Acc, 0) + 1,
                total_capacity => maps:get(total_capacity, Acc, 0) + Node#evaluator_node.capacity,
                total_active => maps:get(total_active, Acc, 0) + Node#evaluator_node.active,
                total_errors => maps:get(total_errors, Acc, 0) + Node#evaluator_node.error_count
            }
        end,
        InitialStats,
        Evaluators
    ),

    {reply, Stats, State};

handle_call(_Request, _From, State) ->
    {reply, {error, unknown_request}, State}.

handle_cast({evaluation_started, NodeId}, State) ->
    #state{evaluators = Evaluators} = State,

    case ets:lookup(Evaluators, NodeId) of
        [Node] ->
            Updated = Node#evaluator_node{active = Node#evaluator_node.active + 1},
            ets:insert(Evaluators, Updated);
        [] ->
            ok
    end,

    {noreply, State};

handle_cast({evaluation_completed, NodeId, LatencyMs}, State) ->
    #state{evaluators = Evaluators} = State,

    case ets:lookup(Evaluators, NodeId) of
        [Node] ->
            %% Update active count and latency (exponential moving average)
            NewActive = max(0, Node#evaluator_node.active - 1),
            OldLatency = Node#evaluator_node.latency_ms,
            NewLatency = (OldLatency * 7 + LatencyMs * 3) div 10,  % EMA with alpha=0.3

            Updated = Node#evaluator_node{
                active = NewActive,
                latency_ms = NewLatency,
                last_heartbeat = erlang:system_time(millisecond)
            },
            ets:insert(Evaluators, Updated);
        [] ->
            ok
    end,

    {noreply, State};

handle_cast(_Msg, State) ->
    {noreply, State}.

handle_info(cleanup_stale, State) ->
    #state{evaluators = Evaluators} = State,

    Now = erlang:system_time(millisecond),
    Threshold = Now - ?STALE_THRESHOLD_MS,

    %% Find and remove stale evaluators
    Stale = ets:foldl(
        fun(Node, Acc) ->
            case Node#evaluator_node.last_heartbeat < Threshold of
                true -> [Node#evaluator_node.node_id | Acc];
                false -> Acc
            end
        end,
        [],
        Evaluators
    ),

    lists:foreach(fun(NodeId) -> ets:delete(Evaluators, NodeId) end, Stale),

    %% Schedule next cleanup
    Timer = erlang:send_after(?CLEANUP_INTERVAL_MS, self(), cleanup_stale),

    {noreply, State#state{cleanup_timer = Timer}};

handle_info(_Info, State) ->
    {noreply, State}.

terminate(_Reason, State) ->
    #state{evaluators = Evaluators, cleanup_timer = Timer} = State,

    case Timer of
        undefined -> ok;
        _ -> erlang:cancel_timer(Timer)
    end,

    ets:delete(Evaluators),
    ok.

%%% ============================================================================
%%% Internal Functions
%%% ============================================================================

select_evaluator([], _LocalNodeId, _PreferLocal) ->
    {error, no_evaluators};

select_evaluator(Available, LocalNodeId, PreferLocal) ->
    %% Check if local node is available and we should prefer it
    LocalNode = lists:keyfind(LocalNodeId, #evaluator_node.node_id, Available),

    case {LocalNode, rand:uniform()} of
        {#evaluator_node{} = Node, R} when R < PreferLocal ->
            {ok, Node};
        _ ->
            %% Select based on load and latency score
            Scored = lists:map(
                fun(Node) ->
                    %% Lower score is better
                    LoadRatio = Node#evaluator_node.active / Node#evaluator_node.capacity,
                    LatencyScore = Node#evaluator_node.latency_ms / 100,
                    ErrorPenalty = Node#evaluator_node.error_count * 0.5,
                    Score = LoadRatio + LatencyScore + ErrorPenalty,
                    {Score, Node}
                end,
                Available
            ),

            %% Sort by score and pick best
            [{_Score, Best} | _] = lists:sort(fun({S1, _}, {S2, _}) -> S1 =< S2 end, Scored),
            {ok, Best}
    end.

generate_node_id() ->
    Bytes = crypto:strong_rand_bytes(8),
    <<"node_", (binary:encode_hex(Bytes))/binary>>.
