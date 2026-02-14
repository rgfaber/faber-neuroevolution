# Custom Evaluators

The evaluator is where your domain-specific logic lives. This guide explains how to implement the `neuroevolution_evaluator` behaviour for different use cases.

## The Behaviour

```erlang
-callback evaluate(Individual, Options) -> {ok, UpdatedIndividual} | {error, Reason} when
    Individual :: individual(),
    Options :: map(),
    UpdatedIndividual :: individual(),
    Reason :: term().

-callback calculate_fitness(Metrics) -> Fitness when
    Metrics :: map(),
    Fitness :: float().
```

## Basic Structure

```erlang
-module(my_evaluator).
-behaviour(neuroevolution_evaluator).

-include_lib("faber_neuroevolution/include/neuroevolution.hrl").

-export([evaluate/2, calculate_fitness/1]).

evaluate(Individual, Options) ->
    %% 1. Extract the network
    Network = Individual#individual.network,

    %% 2. Run your evaluation logic
    Results = run_evaluation(Network, Options),

    %% 3. Update individual with metrics
    UpdatedIndividual = Individual#individual{
        metrics = results_to_metrics(Results)
    },
    {ok, UpdatedIndividual}.

calculate_fitness(Metrics) ->
    %% Convert metrics to a single fitness score
    %% Higher is better
    compute_fitness(Metrics).
```

## Example: Game AI Evaluator

Here's a complete example for a game-playing AI:

```erlang
-module(game_evaluator).
-behaviour(neuroevolution_evaluator).

-include_lib("faber_neuroevolution/include/neuroevolution.hrl").

-export([evaluate/2, calculate_fitness/1]).

%% Evaluate by playing multiple games
evaluate(Individual, Options) ->
    Network = Individual#individual.network,
    NumGames = maps:get(games_per_eval, Options, 10),

    %% Play multiple games to reduce variance
    GameResults = [play_game(Network) || _ <- lists:seq(1, NumGames)],

    %% Aggregate results
    TotalScore = lists:sum([R#game_result.score || R <- GameResults]),
    TotalMoves = lists:sum([R#game_result.moves || R <- GameResults]),
    Wins = length([R || R <- GameResults, R#game_result.won]),

    UpdatedIndividual = Individual#individual{
        metrics = #{
            total_score => TotalScore,
            total_moves => TotalMoves,
            wins => Wins,
            games_played => NumGames
        }
    },
    {ok, UpdatedIndividual}.

%% Multi-objective fitness
calculate_fitness(Metrics) ->
    Score = maps:get(total_score, Metrics, 0),
    Moves = maps:get(total_moves, Metrics, 0),
    Wins = maps:get(wins, Metrics, 0),

    %% Weighted combination
    Score * 10.0 + Moves * 0.1 + Wins * 100.0.

%% Internal: Play one game
play_game(Network) ->
    InitialState = game:new(),
    play_loop(Network, InitialState, 0, 0).

play_loop(Network, State, Score, Moves) ->
    case game:is_over(State) of
        true ->
            #game_result{
                score = Score,
                moves = Moves,
                won = game:is_won(State)
            };
        false ->
            %% Get game state as network input
            Input = game:to_input_vector(State),

            %% Get network decision
            Output = network_evaluator:propagate(Network, Input),
            Action = output_to_action(Output),

            %% Apply action
            {NewState, Reward} = game:step(State, Action),
            play_loop(Network, NewState, Score + Reward, Moves + 1)
    end.

output_to_action(Output) ->
    %% Convert network output to discrete action
    %% e.g., argmax for classification
    {MaxVal, MaxIdx} = lists:foldl(
        fun({Val, Idx}, {BestVal, BestIdx}) ->
            case Val > BestVal of
                true -> {Val, Idx};
                false -> {BestVal, BestIdx}
            end
        end,
        {hd(Output), 0},
        lists:zip(tl(Output), lists:seq(1, length(Output) - 1))
    ),
    MaxIdx.
```

## Example: Function Approximation

For supervised learning / function approximation:

```erlang
-module(regression_evaluator).
-behaviour(neuroevolution_evaluator).

-include_lib("faber_neuroevolution/include/neuroevolution.hrl").

-export([evaluate/2, calculate_fitness/1]).

evaluate(Individual, Options) ->
    Network = Individual#individual.network,
    Dataset = maps:get(dataset, Options),

    %% Evaluate on all samples
    Errors = lists:map(
        fun({Input, Expected}) ->
            Output = network_evaluator:propagate(Network, Input),
            mean_squared_error(Output, Expected)
        end,
        Dataset
    ),

    MeanError = lists:sum(Errors) / length(Errors),

    UpdatedIndividual = Individual#individual{
        metrics = #{
            mean_error => MeanError,
            samples => length(Dataset)
        }
    },
    {ok, UpdatedIndividual}.

%% Lower error = higher fitness
calculate_fitness(Metrics) ->
    Error = maps:get(mean_error, Metrics, 1.0),
    %% Convert to fitness (higher is better)
    1.0 / (1.0 + Error).

mean_squared_error(Output, Expected) ->
    Diffs = lists:zipwith(fun(O, E) -> (O - E) * (O - E) end, Output, Expected),
    lists:sum(Diffs) / length(Diffs).
```

## Example: Parallel Evaluation with Timeout

For expensive evaluations:

```erlang
-module(parallel_evaluator).
-behaviour(neuroevolution_evaluator).

-include_lib("faber_neuroevolution/include/neuroevolution.hrl").

-export([evaluate/2, calculate_fitness/1]).

evaluate(Individual, Options) ->
    Network = Individual#individual.network,
    Timeout = maps:get(timeout_ms, Options, 5000),

    %% Spawn evaluation in separate process
    Self = self(),
    Ref = make_ref(),

    spawn_link(fun() ->
        Result = run_expensive_evaluation(Network),
        Self ! {eval_result, Ref, Result}
    end),

    %% Wait with timeout
    receive
        {eval_result, Ref, Result} ->
            UpdatedIndividual = Individual#individual{
                metrics = Result
            },
            {ok, UpdatedIndividual}
    after Timeout ->
        %% Timeout - return poor fitness
        UpdatedIndividual = Individual#individual{
            metrics = #{timeout => true, score => 0}
        },
        {ok, UpdatedIndividual}
    end.

calculate_fitness(Metrics) ->
    case maps:get(timeout, Metrics, false) of
        true -> 0.0;  % Penalize timeouts
        false -> maps:get(score, Metrics, 0.0)
    end.

run_expensive_evaluation(Network) ->
    %% Your expensive computation here
    #{score => compute_score(Network)}.
```

## Best Practices

### 1. Multiple Evaluations
Run each network multiple times to reduce variance:
```erlang
NumTrials = maps:get(trials, Options, 10),
Results = [evaluate_once(Network) || _ <- lists:seq(1, NumTrials)],
AggregatedMetrics = aggregate(Results).
```

### 2. Normalize Fitness
Keep fitness values in a reasonable range:
```erlang
calculate_fitness(Metrics) ->
    RawScore = maps:get(score, Metrics, 0),
    %% Normalize to [0, 1] or similar range
    math:tanh(RawScore / 1000.0).
```

### 3. Multi-Objective Fitness
Combine multiple objectives carefully:
```erlang
calculate_fitness(Metrics) ->
    Score = maps:get(score, Metrics, 0),
    Efficiency = maps:get(efficiency, Metrics, 0),
    Safety = maps:get(safety, Metrics, 1),

    %% Weighted sum with constraints
    case Safety < 0.5 of
        true -> 0.0;  % Safety constraint
        false -> Score * 0.7 + Efficiency * 0.3
    end.
```

### 4. Handle Errors Gracefully
```erlang
evaluate(Individual, Options) ->
    try
        do_evaluation(Individual, Options)
    catch
        _:Reason ->
            error_logger:warning_msg("Evaluation failed: ~p~n", [Reason]),
            %% Return with zero fitness
            {ok, Individual#individual{metrics = #{error => true}}}
    end.
```

## Fitness Function Design

The fitness function is critical to evolutionary success:

- **Higher is better** - The selection algorithms expect higher fitness = better performance
- **Smooth gradients** - Avoid cliff functions where small changes cause big fitness jumps
- **Discriminating** - Even poor solutions should have slightly different fitness
- **Reward partial success** - Don't only reward complete solutions

Example of incremental fitness:
```erlang
calculate_fitness(Metrics) ->
    %% Reward any progress, not just winning
    DistanceTraveled = maps:get(distance, Metrics, 0),
    CollectedItems = maps:get(items, Metrics, 0),
    ReachedGoal = maps:get(goal_reached, Metrics, false),

    BaseFitness = DistanceTraveled * 0.1 + CollectedItems * 10.0,
    case ReachedGoal of
        true -> BaseFitness + 1000.0;  % Big bonus for goal
        false -> BaseFitness
    end.
```
