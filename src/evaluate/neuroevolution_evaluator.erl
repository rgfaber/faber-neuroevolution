%% @doc Behaviour for domain-specific evaluators.
%%
%% This module defines the behaviour that domain-specific evaluators must
%% implement. Evaluators are responsible for:
%%
%% <ul>
%% <li>Running an individual through domain-specific tests/games/simulations</li>
%% <li>Calculating metrics (score, ticks survived, wins, etc.)</li>
%% <li>Optionally calculating fitness from metrics</li>
%% </ul>
%%
%% == Implementing an Evaluator ==
%%
%% To create a custom evaluator, implement the `evaluate/2' callback:
%%
%% -module(my_evaluator).
%% -behaviour(neuroevolution_evaluator).
%%
%% -export([evaluate/2]).
%%
%% evaluate(Individual, Options) ->
%%     Network = Individual#individual.network,
%%     %% Run your evaluation logic
%%     Score = run_game(Network),
%%     Ticks = get_survival_time(),
%%     %% Return updated individual with metrics
%%     UpdatedIndividual = Individual#individual{
%%         metrics = #{score => Score, ticks => Ticks}
%%     },
%%     {ok, UpdatedIndividual}.
%%
%% == Optional Fitness Calculation ==
%%
%% By default, fitness is calculated by the neuroevolution server using
%% a standard formula. You can override this by implementing `calculate_fitness/1':
%%
%% -export([calculate_fitness/1]).
%%
%% calculate_fitness(Metrics) ->
%%     Score = maps:get(score, Metrics, 0),
%%     Ticks = maps:get(ticks, Metrics, 0),
%%     Score * 50.0 + Ticks / 50.0.
%%
%% @author R.G. Lefever
%% @copyright 2024-2026 R.G. Lefever
-module(neuroevolution_evaluator).

-include("neuroevolution.hrl").

%%% ============================================================================
%%% Behaviour Callbacks
%%% ============================================================================

%% Callback: evaluate/2
%% Invoked for each individual during generation evaluation.
%% The evaluator should:
%% 1. Run the individual's network through domain-specific tests
%% 2. Collect performance metrics
%% 3. Return the individual with updated metrics field
%%
%% Options map contains evaluator_options from config, plus:
%% - games - number of evaluations to run
%% - notify_pid - optional PID to send progress notifications
%%
%% If evaluation fails, return {error, Reason}.
-callback evaluate(Individual, Options) -> Result when
    Individual :: individual(),
    Options :: map(),
    Result :: {ok, EvaluatedIndividual :: individual()} | {error, term()}.

%% Callback: calculate_fitness/1 (optional)
%% If not implemented, the server uses a default fitness calculation.
%% Implement this to customize how metrics are converted to fitness.
-callback calculate_fitness(Metrics) -> Fitness when
    Metrics :: map(),
    Fitness :: float().

-optional_callbacks([calculate_fitness/1]).

%%% ============================================================================
%%% API Functions
%%% ============================================================================

-export([
    evaluate_individual/3,
    evaluate_batch_parallel/3,
    evaluate_batch_distributed/3,
    get_worker_nodes/0,
    default_fitness/1
]).

%% @doc Evaluate an individual using the specified evaluator module.
%%
%% Delegates to the evaluator module's evaluate/2 callback.
%% Returns the evaluated individual with metrics populated.
%% Catches and logs any exceptions from the evaluator.
%%
%% PERFORMANCE: Compiles the network for NIF evaluation before calling the
%% evaluator. This provides 50-100x faster network evaluation compared to
%% pure Erlang. The compiled_ref is stripped by the caller after evaluation
%% to prevent memory leaks (see neuroevolution_server:strip_compiled_ref_from_individual/1).
-spec evaluate_individual(Individual, EvaluatorModule, Options) -> Result when
    Individual :: individual(),
    EvaluatorModule :: module(),
    Options :: map(),
    Result :: {ok, individual()} | {error, term()}.
evaluate_individual(Individual, EvaluatorModule, Options) ->
    try
        %% Compile network for NIF-accelerated evaluation (50-100x faster).
        %% The compiled_ref is stripped after evaluation by the caller.
        CompiledIndividual = compile_network_for_nif(Individual),
        EvaluatorModule:evaluate(CompiledIndividual, Options)
    catch
        Class:Reason:Stacktrace ->
            error_logger:error_msg(
                "[neuroevolution_evaluator] Evaluation failed: ~p:~p~n~p~n",
                [Class, Reason, Stacktrace]
            ),
            {error, {evaluation_failed, Class, Reason}}
    end.

%% @private Compile the individual's network for NIF evaluation.
%% Only compiles if the network doesn't already have a compiled_ref.
compile_network_for_nif(#individual{network = Network} = Individual) when Network =/= undefined ->
    CompiledNetwork = network_evaluator:compile_for_nif(Network),
    Individual#individual{network = CompiledNetwork};
compile_network_for_nif(Individual) ->
    Individual.

%% @doc Default fitness calculation from standard metrics.
%%
%% Uses a formula that balances:
%% - Score (food eaten, points, etc.) - primary goal
%% - Survival (ticks lived) - secondary goal
%% - Wins - bonus
%%
%% Formula: Score * 50 + Ticks / 50 + Wins * 2
-spec default_fitness(Metrics) -> Fitness when
    Metrics :: map(),
    Fitness :: float().
default_fitness(Metrics) ->
    Score = maps:get(total_score, Metrics, 0),
    Ticks = maps:get(total_ticks, Metrics, 0),
    Wins = maps:get(wins, Metrics, 0),

    ScoreComponent = Score * 50.0,
    SurvivalComponent = Ticks / 50.0,
    WinComponent = Wins * 2.0,

    ScoreComponent + SurvivalComponent + WinComponent.

%% @doc Evaluate a batch of individuals in parallel using Elixir PartitionSupervisor.
%%
%% This function delegates batch evaluation to the Elixir EvaluationPool module,
%% which uses PartitionSupervisor to distribute work across all CPU cores.
%% This provides true multi-core parallelism compared to spawn_link which
%% creates processes on the same scheduler as the spawner.
%%
%% Falls back to sequential evaluation if the Elixir pool is not available.
-spec evaluate_batch_parallel(Population, EvaluatorModule, Options) -> Results when
    Population :: [individual()],
    EvaluatorModule :: module(),
    Options :: map(),
    Results :: [{ok, individual()} | {error, term()}].
evaluate_batch_parallel(Population, EvaluatorModule, Options) ->
    Timeout = maps:get(evaluation_timeout, Options, 30000),
    NumSchedulers = erlang:system_info(schedulers_online),

    %% Spawn evaluations distributed across schedulers using scheduler hints
    %% This is the key for multi-core parallelism in Erlang
    Tasks = lists:map(
        fun({Individual, Index}) ->
            %% Use spawn_opt with scheduler hint to distribute across cores
            %% The scheduler hint cycles through all schedulers
            SchedulerHint = (Index rem NumSchedulers) + 1,
            Self = self(),
            Ref = make_ref(),
            Pid = spawn_opt(
                fun() ->
                    Result = try
                        EvaluatorModule:evaluate(Individual, Options)
                    catch
                        Class:Reason:Stacktrace ->
                            error_logger:error_msg(
                                "[neuroevolution_evaluator] Batch eval failed: ~p:~p~n~p~n",
                                [Class, Reason, Stacktrace]
                            ),
                            {error, {evaluation_failed, Class, Reason}}
                    end,
                    Self ! {eval_batch_result, Ref, Result}
                end,
                [{scheduler, SchedulerHint}, link]
            ),
            {Ref, Pid, Individual}
        end,
        lists:zip(Population, lists:seq(0, length(Population) - 1))
    ),

    %% Collect all results with timeout
    collect_batch_results(Tasks, [], Timeout).

%% @private Collect results from parallel batch evaluation
collect_batch_results([], Acc, _Timeout) ->
    lists:reverse(Acc);
collect_batch_results(Tasks, Acc, Timeout) ->
    receive
        {eval_batch_result, Ref, Result} ->
            case lists:keytake(Ref, 1, Tasks) of
                {value, {Ref, _Pid, _Individual}, RemainingTasks} ->
                    collect_batch_results(RemainingTasks, [Result | Acc], Timeout);
                false ->
                    %% Stale result, ignore
                    collect_batch_results(Tasks, Acc, Timeout)
            end
    after Timeout ->
        %% Timeout - kill remaining workers and return what we have
        NumPending = length(Tasks),
        error_logger:warning_msg(
            "[neuroevolution_evaluator] Batch timeout - killing ~p workers~n",
            [NumPending]
        ),
        lists:foreach(
            fun({_Ref, Pid, _Ind}) ->
                unlink(Pid),
                exit(Pid, kill)
            end,
            Tasks
        ),
        %% Return error for timed-out individuals
        TimeoutResults = [{error, timeout} || _ <- Tasks],
        lists:reverse(Acc) ++ TimeoutResults
    end.

%% @doc Evaluate a batch distributed across all connected BEAM nodes.
%%
%% This function splits the population across all available nodes and uses
%% erpc to distribute work. When worker nodes are available, evaluations
%% are spread across all nodes for horizontal scaling. Falls back to local
%% parallel evaluation if no workers are connected.
%%
%% The coordinator splits the population, dispatches batches to workers via
%% erpc, evaluates a local batch in parallel, and collects all results.
%%
%% See `assets/distributed_evaluation.svg' for the architecture diagram.
%%
%% @param Population List of individuals to evaluate
%% @param EvaluatorModule Module implementing evaluate/2 callback
%% @param Options Evaluation options map
%% @returns List of evaluated individuals
-spec evaluate_batch_distributed(Population, EvaluatorModule, Options) -> Results when
    Population :: [individual()],
    EvaluatorModule :: module(),
    Options :: map(),
    Results :: [individual()].
evaluate_batch_distributed(Population, EvaluatorModule, Options) ->
    Workers = get_worker_nodes(),
    AllNodes = [node() | Workers],
    NumNodes = length(AllNodes),
    PopSize = length(Population),
    Timeout = maps:get(evaluation_timeout, Options, 60000),

    %% Log distributed evaluation start
    case Workers of
        [] ->
            error_logger:info_msg(
                "[neuroevolution_evaluator] No workers, using local parallel evaluation~n"
            );
        _ ->
            error_logger:info_msg(
                "[neuroevolution_evaluator] Distributing ~p individuals across ~p nodes: ~p~n",
                [PopSize, NumNodes, AllNodes]
            )
    end,

    %% If no workers, fall back to local parallel evaluation
    case Workers of
        [] ->
            %% Use local parallel evaluation
            Results = evaluate_batch_parallel(Population, EvaluatorModule, Options),
            extract_individuals(Results);
        _ ->
            %% Split population across nodes
            Batches = split_into_batches(Population, NumNodes),

            %% Spawn async tasks for each node's batch
            Parent = self(),
            Tasks = lists:zipwith(
                fun(Batch, Node) ->
                    Ref = make_ref(),
                    spawn_link(fun() ->
                        Result = evaluate_on_node(Batch, EvaluatorModule, Options, Node, Timeout),
                        Parent ! {distributed_eval_result, Ref, Result}
                    end),
                    {Ref, Node, length(Batch)}
                end,
                Batches,
                AllNodes
            ),

            %% Collect results
            collect_distributed_results(Tasks, [], Timeout)
    end.

%% @doc Get list of connected worker nodes.
-spec get_worker_nodes() -> [node()].
get_worker_nodes() ->
    nodes(connected).

%% @private Evaluate a batch on a specific node
evaluate_on_node([], _EvaluatorModule, _Options, _Node, _Timeout) ->
    [];
evaluate_on_node(Batch, EvaluatorModule, Options, Node, Timeout) ->
    case Node == node() of
        true ->
            %% Local evaluation
            Results = evaluate_batch_parallel(Batch, EvaluatorModule, Options),
            extract_individuals(Results);
        false ->
            %% Remote evaluation via erpc
            try
                erpc:call(
                    Node,
                    ?MODULE,
                    evaluate_batch_parallel,
                    [Batch, EvaluatorModule, Options],
                    Timeout
                )
            of
                Results when is_list(Results) ->
                    extract_individuals(Results)
            catch
                error:{erpc, noconnection} ->
                    error_logger:warning_msg(
                        "[neuroevolution_evaluator] Node ~p disconnected, retrying locally~n",
                        [Node]
                    ),
                    LocalResults = evaluate_batch_parallel(Batch, EvaluatorModule, Options),
                    extract_individuals(LocalResults);
                Class:Reason:Stacktrace ->
                    error_logger:error_msg(
                        "[neuroevolution_evaluator] erpc failed on ~p: ~p:~p~n~p~n",
                        [Node, Class, Reason, Stacktrace]
                    ),
                    %% Return original individuals on error
                    Batch
            end
    end.

%% @private Extract individuals from results, handling errors
extract_individuals(Results) ->
    lists:map(
        fun
            ({ok, Ind}) -> Ind;
            ({error, _}) -> undefined
        end,
        Results
    ).

%% @private Split a list into N roughly equal batches
split_into_batches(List, N) when N > 0 ->
    Len = length(List),
    BaseSize = Len div N,
    Remainder = Len rem N,
    split_into_batches(List, N, BaseSize, Remainder, []).

split_into_batches([], _N, _BaseSize, _Remainder, Acc) ->
    lists:reverse(Acc);
split_into_batches(_List, 0, _BaseSize, _Remainder, Acc) ->
    lists:reverse(Acc);
split_into_batches(List, N, BaseSize, Remainder, Acc) ->
    %% First 'Remainder' batches get one extra element
    Size = case Remainder > 0 of
        true -> BaseSize + 1;
        false -> BaseSize
    end,
    {Batch, Rest} = case Size >= length(List) of
        true -> {List, []};
        false -> lists:split(Size, List)
    end,
    NewRemainder = case Remainder > 0 of
        true -> Remainder - 1;
        false -> 0
    end,
    split_into_batches(Rest, N - 1, BaseSize, NewRemainder, [Batch | Acc]).

%% @private Collect results from distributed evaluation
collect_distributed_results([], Acc, _Timeout) ->
    lists:flatten(lists:reverse(Acc));
collect_distributed_results(Tasks, Acc, Timeout) ->
    receive
        {distributed_eval_result, Ref, Results} ->
            case lists:keytake(Ref, 1, Tasks) of
                {value, {Ref, _Node, _Count}, RemainingTasks} ->
                    collect_distributed_results(RemainingTasks, [Results | Acc], Timeout);
                false ->
                    %% Stale result, ignore
                    collect_distributed_results(Tasks, Acc, Timeout)
            end
    after Timeout ->
        NumPending = length(Tasks),
        error_logger:warning_msg(
            "[neuroevolution_evaluator] Distributed eval timeout - ~p nodes pending~n",
            [NumPending]
        ),
        %% Return what we have so far
        lists:flatten(lists:reverse(Acc))
    end.
