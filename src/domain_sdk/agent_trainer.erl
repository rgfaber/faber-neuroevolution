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
    Generations = maps:get(generations, Options, 100),
    Config = build_neuro_config(Bridge, EnvConfig, Options),

    %% Start neuroevolution server
    case neuroevolution_server:start_link(Config) of
        {ok, Pid} ->
            try
                %% Run evolution
                ok = neuroevolution_server:evolve(Pid, Generations),

                %% Get results
                Stats = neuroevolution_server:get_stats(Pid),
                Best = neuroevolution_server:get_best(Pid),

                %% Stop server
                neuroevolution_server:stop(Pid),

                {ok, Best, Stats}
            catch
                Class:Reason:Stack ->
                    neuroevolution_server:stop(Pid),
                    {error, {training_failed, Class, Reason, Stack}}
            end;
        {error, Reason} ->
            {error, {start_failed, Reason}}
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

    %% Build base config map for neuro_config:from_map/1
    BaseConfig = #{
        population_size => maps:get(population_size, Options, 100),
        network_topology => Topology,
        evaluator_module => bridge_evaluator,
        evaluator_options => EvaluatorOptions
    },

    %% Merge with user options (excluding our special keys)
    SpecialKeys = [generations, env_config, episodes_per_eval, population_size, strategy],
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
