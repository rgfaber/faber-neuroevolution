%%%-------------------------------------------------------------------
%%% @doc Distributed evaluator for mesh-based fitness evaluation.
%%%
%%% Provides a high-level API for evaluating individuals across the mesh.
%%% Features:
%%% - Load-balanced evaluation dispatch
%%% - Automatic retry on failure
%%% - Local preference for reduced latency
%%% - Batch evaluation with parallelism control
%%% - Graceful degradation when mesh unavailable
%%%
%%% @end
%%%-------------------------------------------------------------------
-module(distributed_evaluator).

-behaviour(gen_server).

%% API
-export([
    start_link/1,
    evaluate/2,
    evaluate/3,
    evaluate_batch/3,
    evaluate_batch/4,
    register_evaluator/1,
    register_evaluator/2,
    get_stats/0
]).

%% gen_server callbacks
-export([init/1, handle_call/3, handle_cast/2, handle_info/2, terminate/2]).

-define(SERVER, ?MODULE).
-define(DEFAULT_TIMEOUT_MS, 30000).
-define(DEFAULT_RETRY_COUNT, 2).
-define(DEFAULT_PREFER_LOCAL, 0.3).

-record(state, {
    evaluator_module :: module() | undefined,
    default_options :: map(),
    pending_evaluations :: ets:tid(),
    stats :: map()
}).

-record(pending_eval, {
    request_id :: binary(),
    individual :: term(),
    evaluator_module :: module(),
    options :: map(),
    retries_left :: non_neg_integer(),
    start_time :: integer(),
    caller :: pid() | fun()
}).

%%% ============================================================================
%%% API
%%% ============================================================================

-spec start_link(Config :: map()) -> {ok, pid()} | {error, term()}.
start_link(Config) ->
    gen_server:start_link({local, ?SERVER}, ?MODULE, Config, []).

-spec evaluate(Individual :: term(), EvaluatorModule :: module()) ->
    {ok, Fitness :: float()} | {error, term()}.
evaluate(Individual, EvaluatorModule) ->
    evaluate(Individual, EvaluatorModule, #{}).

-spec evaluate(Individual :: term(), EvaluatorModule :: module(), Options :: map()) ->
    {ok, Fitness :: float()} | {error, term()}.
evaluate(Individual, EvaluatorModule, Options) ->
    gen_server:call(?SERVER, {evaluate, Individual, EvaluatorModule, Options},
                    maps:get(timeout_ms, Options, ?DEFAULT_TIMEOUT_MS) + 5000).

-spec evaluate_batch(Individuals :: [term()], EvaluatorModule :: module(), Options :: map()) ->
    [{ok, Fitness :: float()} | {error, term()}].
evaluate_batch(Individuals, EvaluatorModule, Options) ->
    evaluate_batch(Individuals, EvaluatorModule, Options, #{}).

-spec evaluate_batch(Individuals :: [term()], EvaluatorModule :: module(),
                     EvalOptions :: map(), BatchOptions :: map()) ->
    [{ok, Fitness :: float()} | {error, term()}].
evaluate_batch(Individuals, EvaluatorModule, EvalOptions, BatchOptions) ->
    gen_server:call(?SERVER, {evaluate_batch, Individuals, EvaluatorModule, EvalOptions, BatchOptions},
                    infinity).

-spec register_evaluator(EvaluatorModule :: module()) -> ok | {error, term()}.
register_evaluator(EvaluatorModule) ->
    register_evaluator(EvaluatorModule, #{}).

-spec register_evaluator(EvaluatorModule :: module(), Options :: map()) -> ok | {error, term()}.
register_evaluator(EvaluatorModule, Options) ->
    gen_server:call(?SERVER, {register_evaluator, EvaluatorModule, Options}).

-spec get_stats() -> map().
get_stats() ->
    gen_server:call(?SERVER, get_stats).

%%% ============================================================================
%%% gen_server Callbacks
%%% ============================================================================

init(Config) ->
    EvaluatorModule = maps:get(evaluator_module, Config, undefined),

    DefaultOptions = #{
        timeout_ms => maps:get(timeout_ms, Config, ?DEFAULT_TIMEOUT_MS),
        retry_count => maps:get(retry_count, Config, ?DEFAULT_RETRY_COUNT),
        prefer_local => maps:get(prefer_local, Config, ?DEFAULT_PREFER_LOCAL)
    },

    Pending = ets:new(pending_evaluations, [
        set,
        {keypos, #pending_eval.request_id}
    ]),

    State = #state{
        evaluator_module = EvaluatorModule,
        default_options = DefaultOptions,
        pending_evaluations = Pending,
        stats = initial_stats()
    },

    {ok, State}.

handle_call({evaluate, Individual, EvaluatorModule, Options}, From, State) ->
    #state{default_options = Defaults} = State,

    MergedOptions = maps:merge(Defaults, Options),
    spawn_link(fun() ->
        Result = do_evaluate(Individual, EvaluatorModule, MergedOptions, State),
        gen_server:reply(From, Result)
    end),

    {noreply, update_stats(evaluation_started, State)};

handle_call({evaluate_batch, Individuals, EvaluatorModule, EvalOptions, BatchOptions}, From, State) ->
    #state{default_options = Defaults} = State,

    MergedOptions = maps:merge(Defaults, EvalOptions),
    MaxParallel = maps:get(max_parallel, BatchOptions, 10),

    spawn_link(fun() ->
        Results = do_evaluate_batch(Individuals, EvaluatorModule, MergedOptions, MaxParallel, State),
        gen_server:reply(From, Results)
    end),

    {noreply, update_stats(batch_started, State)};

handle_call({register_evaluator, EvaluatorModule, Options}, _From, State) ->
    Capacity = maps:get(capacity, Options, erlang:system_info(schedulers)),

    %% Register with mesh facade
    Result = macula_mesh:advertise_evaluator(EvaluatorModule, Capacity),

    %% Also register locally
    LocalNodeId = get_local_node_id(),
    evaluator_pool_registry:register_evaluator(LocalNodeId, #{
        endpoint => <<"local">>,
        capacity => Capacity,
        evaluator_module => EvaluatorModule
    }),

    NewState = State#state{evaluator_module = EvaluatorModule},

    {reply, Result, NewState};

handle_call(get_stats, _From, State) ->
    #state{stats = Stats} = State,

    %% Add pool stats
    PoolStats = evaluator_pool_registry:get_stats(),
    AllStats = maps:merge(Stats, #{pool => PoolStats}),

    {reply, AllStats, State};

handle_call(_Request, _From, State) ->
    {reply, {error, unknown_request}, State}.

handle_cast(_Msg, State) ->
    {noreply, State}.

handle_info({evaluation_complete, RequestId, Result}, State) ->
    #state{pending_evaluations = Pending} = State,

    case ets:lookup(Pending, RequestId) of
        [#pending_eval{caller = Caller}] ->
            deliver_result(Caller, RequestId, Result),
            ets:delete(Pending, RequestId);
        [] ->
            ok  % Already completed or timed out
    end,

    {noreply, update_stats(evaluation_completed, State)};

handle_info({evaluation_failed, RequestId, Reason}, State) ->
    #state{pending_evaluations = Pending} = State,

    case ets:lookup(Pending, RequestId) of
        [#pending_eval{retries_left = 0} = Eval] ->
            deliver_result(Eval#pending_eval.caller, RequestId, {error, Reason}),
            ets:delete(Pending, RequestId),
            {noreply, update_stats(evaluation_failed, State)};
        [#pending_eval{retries_left = N} = Eval] ->
            %% Retry
            spawn_link(fun() ->
                retry_evaluation(Eval#pending_eval{retries_left = N - 1}, State)
            end),
            {noreply, update_stats(evaluation_retried, State)};
        [] ->
            {noreply, State}
    end;

handle_info(_Info, State) ->
    {noreply, State}.

terminate(_Reason, State) ->
    #state{pending_evaluations = Pending} = State,
    ets:delete(Pending),
    ok.

%%% ============================================================================
%%% Internal Functions - Evaluation
%%% ============================================================================

do_evaluate(Individual, EvaluatorModule, Options, _State) ->
    PreferLocal = maps:get(prefer_local, Options, ?DEFAULT_PREFER_LOCAL),
    TimeoutMs = maps:get(timeout_ms, Options, ?DEFAULT_TIMEOUT_MS),
    RetryCount = maps:get(retry_count, Options, ?DEFAULT_RETRY_COUNT),

    do_evaluate_with_retry(Individual, EvaluatorModule, Options, PreferLocal, TimeoutMs, RetryCount).

do_evaluate_with_retry(_Individual, _EvaluatorModule, _Options, _PreferLocal, _TimeoutMs, 0) ->
    {error, max_retries_exceeded};

do_evaluate_with_retry(Individual, EvaluatorModule, Options, PreferLocal, TimeoutMs, RetriesLeft) ->
    %% Get available evaluator
    case evaluator_pool_registry:get_available_evaluator(#{prefer_local => PreferLocal}) of
        {ok, Evaluator} ->
            NodeId = element(2, Evaluator),  % node_id field
            evaluator_pool_registry:report_evaluation_started(NodeId),

            StartTime = erlang:system_time(millisecond),

            Result = macula_mesh:request_evaluation(NodeId, Individual, EvaluatorModule, Options#{
                timeout_ms => TimeoutMs
            }),

            EndTime = erlang:system_time(millisecond),
            LatencyMs = EndTime - StartTime,

            case Result of
                {ok, {ok, Fitness}} ->
                    evaluator_pool_registry:report_evaluation_completed(NodeId, LatencyMs),
                    {ok, Fitness};
                {ok, {error, Reason}} ->
                    evaluator_pool_registry:report_evaluation_completed(NodeId, LatencyMs),
                    {error, Reason};
                {error, timeout} ->
                    %% Retry on timeout
                    do_evaluate_with_retry(Individual, EvaluatorModule, Options, PreferLocal, TimeoutMs, RetriesLeft - 1);
                {error, _OtherReason} ->
                    %% Retry on other errors
                    do_evaluate_with_retry(Individual, EvaluatorModule, Options, PreferLocal, TimeoutMs, RetriesLeft - 1)
            end;
        {error, no_evaluators} ->
            %% No evaluators available, try local evaluation
            evaluate_locally(Individual, EvaluatorModule, Options)
    end.

do_evaluate_batch(Individuals, EvaluatorModule, Options, MaxParallel, State) ->
    %% Create work queue
    WorkQueue = lists:zip(lists:seq(1, length(Individuals)), Individuals),

    %% Process in parallel with concurrency limit
    Results = pmap_limited(
        fun({_Idx, Individual}) ->
            do_evaluate(Individual, EvaluatorModule, Options, State)
        end,
        WorkQueue,
        MaxParallel
    ),

    Results.

evaluate_locally(Individual, EvaluatorModule, Options) ->
    try
        Result = EvaluatorModule:evaluate(Individual, Options),
        {ok, Result}
    catch
        Class:Reason:_Stacktrace ->
            {error, {Class, Reason}}
    end.

retry_evaluation(#pending_eval{individual = Individual, evaluator_module = Module, options = Options} = Eval, State) ->
    Result = do_evaluate(Individual, Module, Options, State),

    case Result of
        {ok, Fitness} ->
            self() ! {evaluation_complete, Eval#pending_eval.request_id, {ok, Fitness}};
        {error, Reason} ->
            self() ! {evaluation_failed, Eval#pending_eval.request_id, Reason}
    end.

deliver_result(Caller, RequestId, Result) when is_pid(Caller) ->
    Caller ! {evaluation_result, RequestId, Result};
deliver_result(Caller, _RequestId, Result) when is_function(Caller, 1) ->
    Caller(Result);
deliver_result(_Caller, _RequestId, _Result) ->
    ok.

%%% ============================================================================
%%% Internal Functions - Parallel Map
%%% ============================================================================

pmap_limited(Fun, Items, MaxParallel) ->
    %% Simple parallel map with concurrency limit
    Parent = self(),
    Ref = make_ref(),

    %% Start worker pool
    Workers = [spawn_link(fun() -> pmap_worker(Parent, Ref, Fun) end) || _ <- lists:seq(1, MaxParallel)],

    %% Send work items
    IndexedItems = lists:zip(lists:seq(1, length(Items)), Items),
    lists:foreach(fun({Idx, Item}) ->
        receive
            {ready, Worker} ->
                Worker ! {work, Idx, Item}
        end
    end, IndexedItems),

    %% Signal workers to stop
    lists:foreach(fun(Worker) ->
        receive
            {ready, Worker} ->
                Worker ! stop
        end
    end, Workers),

    %% Collect results
    collect_pmap_results(Ref, length(Items), #{}).

pmap_worker(Parent, Ref, Fun) ->
    Parent ! {ready, self()},
    receive
        {work, Idx, Item} ->
            Result = Fun(Item),
            Parent ! {result, Ref, Idx, Result},
            pmap_worker(Parent, Ref, Fun);
        stop ->
            ok
    end.

collect_pmap_results(_Ref, 0, Results) ->
    %% Sort by index and extract results
    Sorted = lists:sort(maps:to_list(Results)),
    [R || {_Idx, R} <- Sorted];

collect_pmap_results(Ref, Remaining, Results) ->
    receive
        {result, Ref, Idx, Result} ->
            collect_pmap_results(Ref, Remaining - 1, maps:put(Idx, Result, Results))
    after 60000 ->
        %% Timeout waiting for results
        Sorted = lists:sort(maps:to_list(Results)),
        [R || {_Idx, R} <- Sorted]
    end.

%%% ============================================================================
%%% Internal Functions - Helpers
%%% ============================================================================

get_local_node_id() ->
    case macula_mesh:get_state() of
        #{node_id := NodeId} -> NodeId;
        _ -> <<"local">>
    end.

initial_stats() ->
    #{
        evaluations_started => 0,
        evaluations_completed => 0,
        evaluations_failed => 0,
        evaluations_retried => 0,
        batches_started => 0
    }.

update_stats(evaluation_started, #state{stats = Stats} = State) ->
    State#state{stats = Stats#{evaluations_started => maps:get(evaluations_started, Stats, 0) + 1}};
update_stats(evaluation_completed, #state{stats = Stats} = State) ->
    State#state{stats = Stats#{evaluations_completed => maps:get(evaluations_completed, Stats, 0) + 1}};
update_stats(evaluation_failed, #state{stats = Stats} = State) ->
    State#state{stats = Stats#{evaluations_failed => maps:get(evaluations_failed, Stats, 0) + 1}};
update_stats(evaluation_retried, #state{stats = Stats} = State) ->
    State#state{stats = Stats#{evaluations_retried => maps:get(evaluations_retried, Stats, 0) + 1}};
update_stats(batch_started, #state{stats = Stats} = State) ->
    State#state{stats = Stats#{batches_started => maps:get(batches_started, Stats, 0) + 1}}.
