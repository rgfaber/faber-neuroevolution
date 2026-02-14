%% @doc Event-driven evaluator worker for distributed neuroevolution.
%%
%% This gen_server subscribes to evaluation request events and runs
%% evaluations using a configured evaluator module. Results are published
%% back to the evaluated topic.
%%
%% == Usage ==
%%
%% Start a worker for a specific realm:
%%
%%   {ok, Pid} = neuroevolution_evaluator_worker:start_link(#{
%%       realm =&gt; RealmBinary,
%%       evaluator_module =&gt; my_evaluator,
%%       evaluator_options =&gt; #{}
%%   }).
%%
%% The worker will automatically subscribe to `neuro.<realm>.evaluate'
%% and publish results to `neuro.<realm>.evaluated'.
%%
%% == Message Format ==
%%
%% The worker expects messages in the format:
%% `{neuro_event, Topic, {evaluate_request, RequestMap}}'
%%
%% Where RequestMap contains:
%% - `request_id' - Correlation ID for tracking
%% - `individual_id' - The individual's ID
%% - `network' - The neural network to evaluate
%% - `options' - Domain-specific evaluation options
%%
%% == Distributed Operation ==
%%
%% Multiple workers can subscribe to the same realm topic. The event
%% backend determines load distribution:
%% - Local backend (pg): All workers receive all requests
%% - Macula backend: DHT-based routing (load balanced)
%%
%% @author R.G. Lefever
%% @copyright 2024-2026 R.G. Lefever
-module(neuroevolution_evaluator_worker).
-behaviour(gen_server).

-include("neuroevolution.hrl").

%% API
-export([
    start_link/1,
    stop/1
]).

%% gen_server callbacks
-export([
    init/1,
    handle_call/3,
    handle_cast/2,
    handle_info/2,
    terminate/2
]).

%%% ============================================================================
%%% Types
%%% ============================================================================

-type worker_config() :: #{
    realm := binary(),
    evaluator_module := module(),
    evaluator_options => map(),
    max_concurrent => pos_integer()
}.

-record(state, {
    realm :: binary(),
    evaluator_module :: module(),
    evaluator_options :: map(),
    max_concurrent :: pos_integer(),
    active_evaluations :: non_neg_integer(),
    pending_queue :: queue:queue()
}).

%%% ============================================================================
%%% API Functions
%%% ============================================================================

%% @doc Start a linked evaluator worker.
%%
%% Config must contain:
%% - `realm' - The realm to subscribe to
%% - `evaluator_module' - Module implementing neuroevolution_evaluator behaviour
%%
%% Optional:
%% - `evaluator_options' - Options passed to evaluator (default: #{})
%% - `max_concurrent' - Max concurrent evaluations (default: 10)
-spec start_link(Config) -> {ok, pid()} | {error, term()} when
    Config :: worker_config().
start_link(Config) ->
    gen_server:start_link(?MODULE, Config, []).

%% @doc Stop a worker.
-spec stop(Pid) -> ok when
    Pid :: pid().
stop(Pid) ->
    gen_server:stop(Pid).

%%% ============================================================================
%%% gen_server Callbacks
%%% ============================================================================

%% @private
init(Config) ->
    Realm = maps:get(realm, Config),
    EvaluatorModule = maps:get(evaluator_module, Config),
    EvaluatorOptions = maps:get(evaluator_options, Config, #{}),
    MaxConcurrent = maps:get(max_concurrent, Config, 10),

    %% Subscribe to evaluation requests for this realm
    Topic = neuroevolution_events:evaluate_topic(Realm),
    ok = neuroevolution_events:subscribe(Topic),

    State = #state{
        realm = Realm,
        evaluator_module = EvaluatorModule,
        evaluator_options = EvaluatorOptions,
        max_concurrent = MaxConcurrent,
        active_evaluations = 0,
        pending_queue = queue:new()
    },

    error_logger:info_msg(
        "[evaluator_worker] Started for realm ~s, evaluator: ~p~n",
        [Realm, EvaluatorModule]
    ),

    {ok, State}.

%% @private
%% Introspection calls - this worker is primarily event-driven
handle_call(get_state, _From, State) ->
    Info = #{
        realm => State#state.realm,
        evaluator_module => State#state.evaluator_module,
        max_concurrent => State#state.max_concurrent,
        active_evaluations => State#state.active_evaluations,
        pending_count => queue:len(State#state.pending_queue)
    },
    {reply, {ok, Info}, State};
handle_call(get_stats, _From, State) ->
    Stats = #{
        active => State#state.active_evaluations,
        pending => queue:len(State#state.pending_queue),
        capacity => State#state.max_concurrent - State#state.active_evaluations
    },
    {reply, {ok, Stats}, State};
handle_call(_Request, _From, State) ->
    {reply, {error, unknown_request}, State}.

%% @private
handle_cast(_Request, State) ->
    {noreply, State}.

%% @private
%% Handle incoming evaluation request events
handle_info({neuro_event, _Topic, {evaluate_request, Request}}, State) ->
    NewState = handle_evaluate_request(Request, State),
    {noreply, NewState};

%% Handle evaluation completion from spawned process
handle_info({evaluation_complete, _Ref}, State) ->
    NewState = State#state{active_evaluations = State#state.active_evaluations - 1},
    %% Try to process queued requests
    FinalState = maybe_process_queue(NewState),
    {noreply, FinalState};

handle_info(_Info, State) ->
    {noreply, State}.

%% @private
terminate(_Reason, #state{realm = Realm}) ->
    Topic = neuroevolution_events:evaluate_topic(Realm),
    neuroevolution_events:unsubscribe(Topic),
    ok.

%%% ============================================================================
%%% Internal Functions
%%% ============================================================================

%% @private
%% Handle an evaluation request - either process immediately or queue
handle_evaluate_request(Request, #state{active_evaluations = Active,
                                         max_concurrent = Max} = State)
  when Active >= Max ->
    %% Queue the request
    NewQueue = queue:in(Request, State#state.pending_queue),
    State#state{pending_queue = NewQueue};
handle_evaluate_request(Request, State) ->
    %% Process immediately
    spawn_evaluation(Request, State),
    State#state{active_evaluations = State#state.active_evaluations + 1}.

%% @private
%% Try to process queued requests if we have capacity
maybe_process_queue(#state{pending_queue = Queue,
                           active_evaluations = Active,
                           max_concurrent = Max} = State)
  when Active < Max ->
    case queue:out(Queue) of
        {{value, Request}, NewQueue} ->
            spawn_evaluation(Request, State),
            maybe_process_queue(State#state{
                pending_queue = NewQueue,
                active_evaluations = Active + 1
            });
        {empty, _} ->
            State
    end;
maybe_process_queue(State) ->
    State.

%% @private
%% Spawn a process to run the evaluation
spawn_evaluation(Request, State) ->
    ParentPid = self(),
    Ref = make_ref(),

    spawn_link(fun() ->
        try
            run_evaluation(Request, State),
            ParentPid ! {evaluation_complete, Ref}
        catch
            Class:Reason:Stacktrace ->
                error_logger:error_msg(
                    "[evaluator_worker] Evaluation crashed: ~p:~p~n~p~n",
                    [Class, Reason, Stacktrace]
                ),
                %% Still notify parent of completion
                ParentPid ! {evaluation_complete, Ref},
                %% Publish error result
                publish_error_result(Request, {crashed, Class, Reason}, State)
        end
    end).

%% @private
%% Run the actual evaluation and publish results
run_evaluation(Request, #state{evaluator_module = EvaluatorModule,
                               evaluator_options = BaseOptions,
                               realm = Realm}) ->
    #{
        request_id := RequestId,
        individual_id := IndividualId,
        network := Network
    } = Request,

    %% Get request-specific options and merge with base options
    RequestOptions = maps:get(options, Request, #{}),
    MergedOptions = maps:merge(BaseOptions, RequestOptions),

    %% Build individual record for evaluator
    Individual = #individual{
        id = IndividualId,
        network = Network
    },

    %% Run evaluation
    Result = case neuroevolution_evaluator:evaluate_individual(
                     Individual, EvaluatorModule, MergedOptions) of
        {ok, EvaluatedInd} ->
            #{
                request_id => RequestId,
                individual_id => IndividualId,
                metrics => EvaluatedInd#individual.metrics,
                evaluator_node => node()
            };
        {error, Reason} ->
            #{
                request_id => RequestId,
                individual_id => IndividualId,
                metrics => #{error => Reason, fitness => 0.0},
                evaluator_node => node()
            }
    end,

    %% Publish result
    ResultTopic = neuroevolution_events:evaluated_topic(Realm),
    neuroevolution_events:publish(ResultTopic, {evaluated, Result}).

%% @private
%% Publish an error result when evaluation crashes
publish_error_result(Request, Error, #state{realm = Realm}) ->
    RequestId = maps:get(request_id, Request, undefined),
    IndividualId = maps:get(individual_id, Request, undefined),

    Result = #{
        request_id => RequestId,
        individual_id => IndividualId,
        metrics => #{error => Error, fitness => 0.0},
        evaluator_node => node()
    },

    ResultTopic = neuroevolution_events:evaluated_topic(Realm),
    neuroevolution_events:publish(ResultTopic, {evaluated, Result}).
