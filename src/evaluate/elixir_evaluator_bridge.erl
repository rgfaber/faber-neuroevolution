%% @doc Bridge module for calling Elixir evaluators from Erlang.
%%
%% This module implements the neuroevolution_evaluator behaviour and provides
%% a bridge to call Elixir evaluator modules from the Erlang neuroevolution_server.
%%
%% == Usage ==
%%
%% Configure neuroevolution_server with this module as the evaluator.
%% Set `evaluator_module' to `elixir_evaluator_bridge' and provide
%% `elixir_module' and `elixir_options' in `evaluator_options'.
%%
%% The Elixir module must export:
%% - evaluate/2 - Takes (IndividualMap, Options), returns ok tuple or error
%%
%% @author R.G. Lefever
%% @copyright 2024-2026 R.G. Lefever
-module(elixir_evaluator_bridge).
-behaviour(neuroevolution_evaluator).

-include_lib("faber_neuroevolution/include/neuroevolution.hrl").

%% neuroevolution_evaluator callbacks
-export([evaluate/2, calculate_fitness/1]).

%% @doc Evaluate an individual using an Elixir module.
%%
%% The Options map must contain:
%% - `elixir_module': The Elixir module atom (e.g., 'Elixir.MyApp.Evaluator')
%% - `elixir_options': Options to pass to the Elixir evaluator (optional)
%% - `games': Number of evaluation games/episodes (optional, passed to Elixir)
%% - `notify_pid': PID to notify of progress (optional)
%%
%% @param Individual The individual record to evaluate
%% @param Options Map containing elixir_module and other options
%% @returns {ok, UpdatedIndividual} | {error, Reason}
-spec evaluate(individual(), map()) -> {ok, individual()} | {error, term()}.
evaluate(Individual, Options) when is_record(Individual, individual), is_map(Options) ->
    ElixirModule = maps:get(elixir_module, Options),
    ElixirOptions = maps:get(elixir_options, Options, #{}),

    %% Convert individual record to map for Elixir
    IndividualMap = individual_to_map(Individual),

    %% Merge evaluation-specific options
    FinalOptions = ElixirOptions#{
        games => maps:get(games, Options, 5),
        notify_pid => maps:get(notify_pid, Options, undefined)
    },

    %% Call Elixir module - it must export evaluate/2
    try
        case ElixirModule:evaluate(IndividualMap, FinalOptions) of
            {ok, Metrics} when is_map(Metrics) ->
                %% Extract fitness from metrics
                Fitness = maps:get(fitness, Metrics, 0.0),
                UpdatedIndividual = Individual#individual{
                    fitness = Fitness,
                    metrics = Metrics
                },
                {ok, UpdatedIndividual};

            {error, Reason} ->
                error_logger:warning_msg(
                    "[elixir_evaluator_bridge] Evaluation error: ~p~n",
                    [Reason]
                ),
                {error, Reason};

            Other ->
                error_logger:warning_msg(
                    "[elixir_evaluator_bridge] Unexpected result: ~p~n",
                    [Other]
                ),
                {error, {unexpected_result, Other}}
        end
    catch
        Class:Error:Stacktrace ->
            error_logger:error_msg(
                "[elixir_evaluator_bridge] Elixir evaluator crashed: ~p:~p~n~p~n",
                [Class, Error, Stacktrace]
            ),
            {error, {elixir_crash, Class, Error}}
    end.

%% @doc Calculate fitness from metrics.
%%
%% This is used when fitness needs to be computed from raw metrics.
%% The Elixir evaluator typically returns fitness directly in the metrics map.
%%
%% @param Metrics The metrics map from evaluation
%% @returns Fitness value as float
-spec calculate_fitness(map()) -> float().
calculate_fitness(Metrics) when is_map(Metrics) ->
    maps:get(fitness, Metrics, 0.0).

%% @private
%% @doc Convert an individual record to a map for Elixir consumption.
-spec individual_to_map(individual()) -> map().
individual_to_map(#individual{
    id = Id,
    network = Network,
    parent1_id = Parent1Id,
    parent2_id = Parent2Id,
    fitness = Fitness,
    metrics = Metrics,
    generation_born = GenerationBorn,
    is_survivor = IsSurvivor,
    is_offspring = IsOffspring
}) ->
    #{
        id => Id,
        network => Network,
        parent1_id => Parent1Id,
        parent2_id => Parent2Id,
        fitness => Fitness,
        metrics => Metrics,
        generation_born => GenerationBorn,
        is_survivor => IsSurvivor,
        is_offspring => IsOffspring
    }.
