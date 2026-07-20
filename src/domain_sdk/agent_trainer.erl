%% @doc Agent Trainer - Integration between Agent SDK and Neuroevolution.
%%
%% This module bridges the gap between domain-defined agent behaviours
%% and the neuroevolution engine. It provides convenience functions to
%% configure, train, and evaluate agents without manual boilerplate.
%%
%% == Overview ==
%%
%% The trainer eliminates manual wiring between agent_bridge and
%% neuroevolution_server:
%%
%% ```
%% %% WITHOUT agent_trainer (manual boilerplate):
%% {ok, Bridge} = agent_bridge:new(Config),
%% {Inputs, _, Outputs} = my_agent:network_topology(),
%% FitnessFn = fun(Network) ->
%%     {ok, Fitness, _} = agent_bridge:run_episode(Bridge, Network, EnvConfig),
%%     Fitness
%% end,
%% NeuroConfig = neuro_config:new(#{
%%     population_size => 100,
%%     input_count => Inputs,
%%     output_count => Outputs,
%%     fitness_function => FitnessFn
%% }),
%% {ok, Pid} = neuroevolution_server:start_link(NeuroConfig),
%% neuroevolution_server:evolve(Pid, 100).
%%
%% %% WITH agent_trainer (one line):
%% {ok, Best, Stats} = agent_trainer:train(Bridge, #{generations => 100}).
%% '''
%%
%% == Quick Start ==
%%
%% ```
%% %% 1. Create bridge with all components (including evaluator!)
%% {ok, Bridge} = agent_bridge:new(#{
%%     definition => my_agent,
%%     sensors => [my_sensor],
%%     actuators => [my_actuator],
%%     environment => my_environment,
%%     evaluator => my_evaluator  %% Required for training!
%% }),
%%
%% %% 2. Train
%% {ok, BestNetwork, Stats} = agent_trainer:train(Bridge, #{
%%     generations => 100,
%%     population_size => 50
%% }).
%% '''
%%
%% == Configuration Options ==
%%
%% Training options passed to `train/2':
%% <ul>
%%   <li>`generations' - Number of generations (default: 100)</li>
%%   <li>`population_size' - Population size (default: 100)</li>
%%   <li>`strategy' - Evolution strategy (default: generational)</li>
%%   <li>`env_config' - Environment configuration (default: #{})</li>
%%   <li>`episodes_per_eval' - Episodes to average (default: 1)</li>
%%   <li>Any other neuro_config options</li>
%% </ul>
%%
%% @author R.G. Lefever
%% @copyright 2024-2026 R.G. Lefever
%% @see agent_bridge
%% @see neuroevolution_server
%% @see neuro_config
-module(agent_trainer).

-include("neuroevolution.hrl").

%%% ============================================================================
%%% Types
%%% ============================================================================

-type train_options() :: #{
    generations => pos_integer(),
    population_size => pos_integer(),
    strategy => atom(),
    env_config => map(),
    episodes_per_eval => pos_integer(),
    atom() => term()  %% Additional neuro_config options
}.
%% Training configuration options.

-type train_result() :: {ok, Network :: term(), Stats :: map()} | {error, term()}.
%% Result of training: best network and final statistics.

-type fitness_fn() :: fun((term()) -> float()).
%% Fitness function type.

-export_type([train_options/0, train_result/0, fitness_fn/0]).

%%% ============================================================================
%%% API Functions
%%% ============================================================================

-export([
    train/2,
    train/3,
    to_fitness_fn/2,
    to_neuro_config/2,
    to_neuro_config/3,
    evaluate/3,
    evaluate_many/4
]).

%% Callback invoked by neuroevolution_server via its event_handler config.
%% Not part of the public API.
-export([handle_event/2]).

%% @doc Trains an agent using neuroevolution.
%%
%% This is the main entry point for training. It:
%% 1. Creates a fitness function from the bridge
%% 2. Configures neuroevolution from the bridge topology
%% 3. Runs evolution for the specified generations
%% 4. Returns the best network and statistics
%%
%% Example:
%% ```
%% {ok, Bridge} = agent_bridge:new(#{
%%     definition => my_agent,
%%     sensors => [my_sensor],
%%     actuators => [my_actuator],
%%     environment => my_env,
%%     evaluator => my_evaluator
%% }),
%% {ok, BestNetwork, Stats} = agent_trainer:train(Bridge, #{
%%     generations => 100,
%%     population_size => 50
%% }).
%% '''
-spec train(Bridge, Options) -> train_result() when
    Bridge :: agent_bridge:validated_bridge(),
    Options :: train_options().
train(Bridge, Options) ->
    EnvConfig = maps:get(env_config, Options, #{}),
    train(Bridge, EnvConfig, Options).

%% @doc Trains an agent with explicit environment configuration.
%%
%% Same as `train/2' but with environment config as separate argument.
-spec train(Bridge, EnvConfig, Options) -> train_result() when
    Bridge :: agent_bridge:validated_bridge(),
    EnvConfig :: map(),
    Options :: train_options().
train(Bridge, EnvConfig, Options) ->
    %% Validate bridge has evaluator
    case maps:get(evaluator, Bridge, undefined) of
        undefined ->
            {error, {missing_evaluator, "Bridge must include evaluator for training"}};
        _Evaluator ->
            do_train(Bridge, EnvConfig, Options)
    end.

%% @doc Creates a fitness function from a bridge.
%%
%% The returned function can be used with neuro_config directly
%% for advanced users who want manual control.
%%
%% Example:
%% ```
%% FitnessFn = agent_trainer:to_fitness_fn(Bridge, EnvConfig),
%% Config = neuro_config:new(#{
%%     fitness_function => FitnessFn,
%%     ...
%% }).
%% '''
-spec to_fitness_fn(Bridge, EnvConfig) -> fitness_fn() when
    Bridge :: agent_bridge:validated_bridge(),
    EnvConfig :: map().
to_fitness_fn(Bridge, EnvConfig) ->
    EpisodesPerEval = 1,
    to_fitness_fn_impl(Bridge, EnvConfig, EpisodesPerEval).

%% @doc Creates a neuro_config from a bridge.
%%
%% Extracts topology from the bridge definition and creates
%% appropriate neuroevolution configuration.
%%
%% Example:
%% ```
%% {ok, Config} = agent_trainer:to_neuro_config(Bridge, EnvConfig),
%% {ok, Pid} = neuroevolution_server:start_link(Config).
%% '''
-spec to_neuro_config(Bridge, EnvConfig) -> {ok, term()} | {error, term()} when
    Bridge :: agent_bridge:validated_bridge(),
    EnvConfig :: map().
to_neuro_config(Bridge, EnvConfig) ->
    to_neuro_config(Bridge, EnvConfig, #{}).

%% @doc Creates a neuro_config with custom options.
%%
%% Options are merged with defaults extracted from the bridge.
%% Returns a #neuro_config{} record suitable for neuroevolution_server.
-spec to_neuro_config(Bridge, EnvConfig, Options) -> {ok, term()} | {error, term()} when
    Bridge :: agent_bridge:validated_bridge(),
    EnvConfig :: map(),
    Options :: map().
to_neuro_config(Bridge, EnvConfig, Options) ->
    case maps:get(evaluator, Bridge, undefined) of
        undefined ->
            {error, {missing_evaluator, "Bridge must include evaluator"}};
        _Evaluator ->
            {ok, build_neuro_config(Bridge, EnvConfig, Options)}
    end.

%% @doc Evaluates a single network using the bridge.
%%
%% Runs one episode and returns fitness and metrics.
%%
%% Example:
%% ```
%% {ok, Fitness, Metrics} = agent_trainer:evaluate(Bridge, Network, EnvConfig).
%% '''
-spec evaluate(Bridge, Network, EnvConfig) -> {ok, float(), map()} | {error, term()} when
    Bridge :: agent_bridge:validated_bridge(),
    Network :: term(),
    EnvConfig :: map().
evaluate(Bridge, Network, EnvConfig) ->
    case maps:get(evaluator, Bridge, undefined) of
        undefined ->
            {error, {missing_evaluator, "Bridge must include evaluator"}};
        _Evaluator ->
            agent_bridge:run_episode(Bridge, Network, EnvConfig)
    end.

%% @doc Evaluates a network over multiple episodes and averages fitness.
%%
%% Useful for stochastic environments where single-episode fitness
%% may have high variance.
%%
%% Example:
%% ```
%% {ok, AvgFitness, AllMetrics} = agent_trainer:evaluate_many(Bridge, Network, EnvConfig, 10).
%% '''
-spec evaluate_many(Bridge, Network, EnvConfig, Episodes) -> {ok, float(), [map()]} | {error, term()} when
    Bridge :: agent_bridge:validated_bridge(),
    Network :: term(),
    EnvConfig :: map(),
    Episodes :: pos_integer().
evaluate_many(Bridge, Network, EnvConfig, Episodes) ->
    case maps:get(evaluator, Bridge, undefined) of
        undefined ->
            {error, {missing_evaluator, "Bridge must include evaluator"}};
        _Evaluator ->
            Results = [agent_bridge:run_episode(Bridge, Network, EnvConfig)
                       || _ <- lists:seq(1, Episodes)],
            process_multi_episode_results(Results)
    end.

%%% ============================================================================
%%% Internal Functions
%%% ============================================================================

%% @private
do_train(Bridge, EnvConfig, Options) ->
    %% Rewritten against neuroevolution_server's actual export list.
    %%
    %% This function previously called neuroevolution_server:evolve/2,
    %% get_best/1 and stop/1. None of the three exist, so every call to
    %% train/2,3 died with undef. The server exports start_training/1,
    %% stop_training/1, get_stats/1 and get_last_evaluated_population/1.
    %%
    %% Training is asynchronous: start_training/1 returns immediately and the
    %% server runs until its own stop condition, announcing completion through
    %% the event_handler callback. We register ourselves as that handler and
    %% block on the message.
    Timeout = maps:get(train_timeout, Options, 300000),
    Config = build_neuro_config(Bridge, EnvConfig, Options),
    case neuroevolution_server:start_link(Config) of
        {ok, Pid} ->
            try
                %% start_training/1 replies {ok, started} or
                %% {ok, already_running}; it does not reply a bare ok.
                {ok, _} = neuroevolution_server:start_training(Pid),
                case await_training_complete(Pid, Timeout) of
                    ok ->
                        {ok, Stats} = neuroevolution_server:get_stats(Pid),
                        case best_network(Pid) of
                            {ok, Network} -> {ok, Network, Stats};
                            {error, Reason} -> {error, Reason}
                        end;
                    {error, Reason} ->
                        {error, Reason}
                end
            catch
                Class:CaughtReason:Stack ->
                    {error, {training_failed, Class, CaughtReason, Stack}}
            after
                stop_server(Pid)
            end;
        {error, Reason} ->
            {error, {start_failed, Reason}}
    end.

%% @private
%% @doc One evaluation for a deterministic environment, otherwise the
%% engine default of 10. Environments declare this via the optional
%% agent_environment:is_deterministic/0 callback.
default_evaluations_per_individual(Bridge) ->
    case maps:get(environment, Bridge, undefined) of
        undefined ->
            10;
        EnvModule ->
            _ = code:ensure_loaded(EnvModule),
            case erlang:function_exported(EnvModule, is_deterministic, 0) of
                true ->
                    case EnvModule:is_deterministic() of
                        true  -> 1;
                        false -> 10
                    end;
                false ->
                    10
            end
    end.

%% @private Event handler callback (neuroevolution_server calls
%% Module:handle_event/2 with the configured InitArg). We pass the caller's
%% pid as InitArg so training completion can be awaited synchronously.
handle_event(Event, Pid) when is_pid(Pid) ->
    Pid ! {?MODULE, neuro_event, Event},
    ok;
handle_event(_Event, _Other) ->
    ok.

%% @private
await_training_complete(Pid, Timeout) ->
    Monitor = erlang:monitor(process, Pid),
    Result = await_loop(Pid, Monitor, Timeout),
    erlang:demonitor(Monitor, [flush]),
    Result.

await_loop(Pid, Monitor, Timeout) ->
    receive
        {?MODULE, neuro_event, {training_complete, _Info}} ->
            ok;
        {?MODULE, neuro_event, _Other} ->
            await_loop(Pid, Monitor, Timeout);
        {'DOWN', Monitor, process, Pid, Reason} ->
            {error, {server_died, Reason}}
    after Timeout ->
        {error, {training_timeout, Timeout}}
    end.

%% @private Champion is the head of the last fully evaluated population,
%% which the server sorts by fitness descending.
best_network(Pid) ->
    case neuroevolution_server:get_last_evaluated_population(Pid) of
        {ok, [#individual{network = Network} | _]} ->
            {ok, Network};
        {ok, []} ->
            {error, no_evaluated_population};
        Other ->
            {error, {unexpected_population, Other}}
    end.

%% @private neuroevolution_server exports no stop/1; it is a gen_server.
stop_server(Pid) ->
    try
        gen_server:stop(Pid, normal, 5000)
    catch
        _:_ -> ok
    end.

%% @private
build_neuro_config(Bridge, EnvConfig, Options) ->
    %% Extract topology from bridge
    Topology = maps:get(topology, Bridge),

    %% Build evaluator options with bridge and env_config
    EpisodesPerEval = maps:get(episodes_per_eval, Options, 1),
    EvaluatorOptions = #{
        bridge => Bridge,
        env_config => EnvConfig,
        episodes_per_eval => EpisodesPerEval
    },

    %% Build base config map for neuro_config:from_map/1.
    %%
    %% generations and strategy were previously listed in SpecialKeys, which
    %% strips them from UserOptions, but neither was added back into
    %% BaseConfig. Both were therefore silently discarded: train/3 ran to
    %% max_generations = infinity regardless of the caller's generations, and
    %% always used the default strategy no matter what strategy was requested.
    BaseConfig0 = #{
        population_size => maps:get(population_size, Options, 100),
        network_topology => Topology,
        evaluator_module => bridge_evaluator,
        evaluator_options => EvaluatorOptions,
        max_generations => maps:get(generations, Options, 100),
        %% A deterministic environment gains nothing from repeated evaluation
        %% of the same individual: every repeat recomputes an identical
        %% result. Leaving the default of 10 in place inflated the first
        %% measured evaluations-to-solve figure tenfold (insight 006).
        %% An explicit caller setting always wins.
        evaluations_per_individual =>
            maps:get(evaluations_per_individual, Options,
                     default_evaluations_per_individual(Bridge)),
        %% Relay server events to the caller so training completion can be
        %% awaited. do_train/3 blocks on this.
        event_handler => {?MODULE, self()}
    },
    BaseConfig = case maps:get(strategy, Options, undefined) of
        undefined ->
            BaseConfig0;
        Strategy ->
            BaseConfig0#{strategy_config => #{strategy_module => Strategy}}
    end,

    %% Merge with user options (excluding keys handled explicitly above)
    SpecialKeys = [generations, env_config, episodes_per_eval, population_size,
                   strategy, train_timeout],
    %% rng_seed passes through in UserOptions to neuro_config:from_map/1.
    UserOptions = maps:without(SpecialKeys, Options),
    MergedConfig = maps:merge(BaseConfig, UserOptions),

    neuro_config:from_map(MergedConfig).

%% @private
to_fitness_fn_impl(Bridge, EnvConfig, 1) ->
    %% Single episode evaluation
    fun(Network) ->
        case agent_bridge:run_episode(Bridge, Network, EnvConfig) of
            {ok, Fitness, _Metrics} -> Fitness;
            {ok, _Metrics} -> 0.0;  %% No evaluator case (shouldn't happen in training)
            {error, _} -> 0.0
        end
    end;
to_fitness_fn_impl(Bridge, EnvConfig, Episodes) when Episodes > 1 ->
    %% Multi-episode evaluation (average fitness)
    fun(Network) ->
        Fitnesses = lists:map(
            fun(_) ->
                case agent_bridge:run_episode(Bridge, Network, EnvConfig) of
                    {ok, Fitness, _Metrics} -> Fitness;
                    {ok, _Metrics} -> 0.0;
                    {error, _} -> 0.0
                end
            end,
            lists:seq(1, Episodes)
        ),
        lists:sum(Fitnesses) / length(Fitnesses)
    end.

%% @private
process_multi_episode_results(Results) ->
    {Fitnesses, MetricsList} = lists:foldl(
        fun({ok, Fitness, Metrics}, {FAcc, MAcc}) ->
            {[Fitness | FAcc], [Metrics | MAcc]};
           ({ok, Metrics}, {FAcc, MAcc}) ->
            %% No evaluator case
            {FAcc, [Metrics | MAcc]};
           ({error, _}, {FAcc, MAcc}) ->
            {FAcc, MAcc}
        end,
        {[], []},
        Results
    ),
    case Fitnesses of
        [] ->
            {error, all_episodes_failed};
        _ ->
            AvgFitness = lists:sum(Fitnesses) / length(Fitnesses),
            {ok, AvgFitness, lists:reverse(MetricsList)}
    end.
