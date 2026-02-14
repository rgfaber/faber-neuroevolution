%% @doc Bridge Evaluator - Adapts Agent SDK to neuroevolution_evaluator behaviour.
%%
%% This module bridges the Agent SDK's agent_bridge with the neuroevolution
%% system's evaluator interface. It allows agent_trainer to use agent_bridge
%% configurations with neuroevolution_server.
%%
%% == Usage ==
%%
%% This module is used internally by agent_trainer. You don't need to
%% use it directly. Configure your training via agent_trainer:train/2.
%%
%% == How It Works ==
%%
%% 1. agent_trainer stores the validated bridge in evaluator_options
%% 2. This module extracts the bridge and env_config from options
%% 3. For each individual, it extracts the network and runs an episode
%% 4. Returns the individual with populated metrics and fitness
%%
%% @author R.G. Lefever
%% @copyright 2024-2026 R.G. Lefever
%% @see agent_trainer
%% @see neuroevolution_evaluator
-module(bridge_evaluator).
-behaviour(neuroevolution_evaluator).

-include("neuroevolution.hrl").

%%% ============================================================================
%%% Behaviour Callbacks
%%% ============================================================================

-export([evaluate/2, calculate_fitness/1]).

%% @doc Evaluates an individual using the agent bridge.
%%
%% Extracts the network from the individual, runs it through the bridge's
%% sense-think-act cycle, and returns the individual with metrics populated.
-spec evaluate(Individual, Options) -> {ok, Individual} | {error, term()} when
    Individual :: individual(),
    Options :: map().
evaluate(Individual, Options) ->
    Bridge = maps:get(bridge, Options),
    EnvConfig = maps:get(env_config, Options, #{}),
    EpisodesPerEval = maps:get(episodes_per_eval, Options, 1),
    Network = Individual#individual.network,

    %% Run evaluation
    Result = run_evaluation(Bridge, Network, EnvConfig, EpisodesPerEval),

    case Result of
        {ok, Fitness, Metrics} ->
            UpdatedIndividual = Individual#individual{
                fitness = Fitness,
                metrics = Metrics
            },
            {ok, UpdatedIndividual};
        {ok, Metrics} when is_map(Metrics) ->
            %% Bridge without evaluator - shouldn't happen in agent_trainer
            %% but handle gracefully
            Fitness = default_fitness(Metrics),
            UpdatedIndividual = Individual#individual{
                fitness = Fitness,
                metrics = Metrics
            },
            {ok, UpdatedIndividual};
        {error, Reason} ->
            {error, Reason}
    end.

%% @doc Calculates fitness from metrics (not used - fitness comes from bridge).
%%
%% This is a fallback in case the neuroevolution server calls it directly.
%% Normally, fitness is calculated by agent_evaluator via the bridge.
-spec calculate_fitness(Metrics) -> float() when
    Metrics :: map().
calculate_fitness(Metrics) ->
    default_fitness(Metrics).

%%% ============================================================================
%%% Internal Functions
%%% ============================================================================

%% @private
%% Run single or multi-episode evaluation
run_evaluation(Bridge, Network, EnvConfig, 1) ->
    agent_bridge:run_episode(Bridge, Network, EnvConfig);
run_evaluation(Bridge, Network, EnvConfig, Episodes) when Episodes > 1 ->
    Results = [agent_bridge:run_episode(Bridge, Network, EnvConfig)
               || _ <- lists:seq(1, Episodes)],
    aggregate_results(Results).

%% @private
%% Aggregate multiple episode results
aggregate_results(Results) ->
    {Fitnesses, MetricsList} = lists:foldl(
        fun({ok, Fitness, Metrics}, {FAcc, MAcc}) when is_float(Fitness) ->
            {[Fitness | FAcc], [Metrics | MAcc]};
           ({ok, Metrics}, {FAcc, MAcc}) when is_map(Metrics) ->
            {FAcc, [Metrics | MAcc]};
           ({error, _}, {FAcc, MAcc}) ->
            {FAcc, MAcc}
        end,
        {[], []},
        Results
    ),
    case {Fitnesses, MetricsList} of
        {[], []} ->
            {error, all_episodes_failed};
        {[], [M | _]} ->
            %% No fitness (no evaluator), return first metrics
            {ok, M};
        {Fs, [M | _]} ->
            %% Average fitness, return first metrics (or could merge)
            AvgFitness = lists:sum(Fs) / length(Fs),
            {ok, AvgFitness, M}
    end.

%% @private
%% Default fitness calculation when none provided
default_fitness(Metrics) when is_map(Metrics) ->
    %% Sum all numeric values as a basic heuristic
    maps:fold(
        fun(_K, V, Acc) when is_number(V) -> Acc + V;
           (_K, _V, Acc) -> Acc
        end,
        0.0,
        Metrics
    ).
