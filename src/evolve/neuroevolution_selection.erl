%% @doc Selection strategies for neuroevolution.
%%
%% This module provides different strategies for selecting which individuals
%% survive to the next generation and which are used as parents for breeding.
%%
%% == Selection Strategies ==
%%
%% <ul>
%% <li>`top_n/2' - Select top N% by fitness (truncation selection)</li>
%% <li>`tournament/3' - Tournament selection with configurable size</li>
%% <li>`roulette_wheel/1' - Fitness-proportionate selection</li>
%% <li>`random_select/1' - Uniform random selection</li>
%% </ul>
%%
%% All strategies expect individuals to have fitness values already calculated.
%%
%% @author R.G. Lefever
%% @copyright 2024-2026 R.G. Lefever
-module(neuroevolution_selection).

-include("neuroevolution.hrl").

%% API
-export([
    top_n/2,
    tournament/3,
    roulette_wheel/1,
    random_select/1,
    select_parents/2
]).

%%% ============================================================================
%%% API Functions
%%% ============================================================================

%% @doc Select top N individuals by fitness (truncation selection).
%%
%% Sorts population by fitness descending and returns top N.
%% This is elitist selection - only the best survive.
%%
%% Example:
%% %% Select top 10 individuals
%% Survivors = neuroevolution_selection:top_n(Population, 10).
-spec top_n(Population, N) -> Survivors when
    Population :: [individual()],
    N :: pos_integer(),
    Survivors :: [individual()].
top_n(Population, N) ->
    Sorted = lists:sort(
        fun(A, B) -> A#individual.fitness >= B#individual.fitness end,
        Population
    ),
    lists:sublist(Sorted, N).

%% @doc Tournament selection.
%%
%% Randomly samples TournamentSize individuals from the population,
%% returns the fittest one. Repeat to select multiple individuals.
%%
%% Tournament selection provides moderate selection pressure while
%% maintaining diversity.
%%
%% Uses tweann_nif:tournament_select/2 for NIF-accelerated selection.
%%
%% Example:
%% %% Tournament of size 3
%% Winner = neuroevolution_selection:tournament(Population, 3, 1).
-spec tournament(Population, TournamentSize, NumSelections) -> Selected when
    Population :: [individual()],
    TournamentSize :: pos_integer(),
    NumSelections :: pos_integer(),
    Selected :: [individual()].
tournament(Population, TournamentSize, NumSelections) ->
    %% Convert to array for O(1) access
    PopArray = list_to_tuple(Population),
    PopSize = length(Population),
    %% Extract fitness values for NIF
    Fitnesses = [Ind#individual.fitness || Ind <- Population],

    tournament_loop(PopArray, Fitnesses, TournamentSize, PopSize, NumSelections, []).

tournament_loop(_PopArray, _Fitnesses, _TournamentSize, _PopSize, 0, Acc) ->
    lists:reverse(Acc);
tournament_loop(PopArray, Fitnesses, TournamentSize, PopSize, Remaining, Acc) ->
    %% Generate random contestant indices (0-based)
    %% TournamentSize random indices from [0, PopSize-1]
    Contestants = [rand:uniform(PopSize) - 1 || _ <- lists:seq(1, TournamentSize)],
    %% Use NIF-accelerated tournament selection
    %% tournament_select(Contestants :: [non_neg_integer()], Fitnesses :: [float()])
    %% Returns 0-based index of winner
    WinnerIdx = tweann_nif:tournament_select(Contestants, Fitnesses),
    %% Convert to 1-based index for tuple access
    Winner = element(WinnerIdx + 1, PopArray),
    tournament_loop(PopArray, Fitnesses, TournamentSize, PopSize, Remaining - 1, [Winner | Acc]).

%% @doc Roulette wheel (fitness-proportionate) selection.
%%
%% Probability of selection is proportional to fitness.
%% Higher fitness = higher probability of being selected.
%%
%% Note: Handles negative fitness by shifting to positive range.
%%
%% Uses tweann_nif:roulette_select/3 for NIF-accelerated selection.
%%
%% Returns a single selected individual.
-spec roulette_wheel(Population) -> Selected when
    Population :: [individual()],
    Selected :: individual().
roulette_wheel(Population) ->
    %% Extract fitnesses
    Fitnesses = [I#individual.fitness || I <- Population],

    %% Use NIF-accelerated roulette selection
    %% build_cumulative_fitness handles shifting to positive and returns {Cumulative, Total}
    {Cumulative, Total} = tweann_nif:build_cumulative_fitness(Fitnesses),

    case Total =< 0 of
        true ->
            %% All fitness zero or negative - use random selection
            random_select(Population);
        false ->
            %% Generate random spin value
            Rand = rand:uniform(),
            %% roulette_select(Cumulative, Total, RandomVal) returns single 0-based index
            SelectedIdx = tweann_nif:roulette_select(Cumulative, Total, Rand),
            %% Return selected individual (convert 0-based to 1-based)
            lists:nth(SelectedIdx + 1, Population)
    end.

%% @doc Uniform random selection.
%%
%% Selects a single individual uniformly at random.
%% All individuals have equal probability regardless of fitness.
-spec random_select(Population) -> Selected when
    Population :: [individual()],
    Selected :: individual().
random_select(Population) ->
    Index = rand:uniform(length(Population)),
    lists:nth(Index, Population).

%% @doc Select two parents for breeding.
%%
%% Uses roulette wheel selection to choose parents, ensuring
%% that two different individuals are selected (if population allows).
-spec select_parents(Population, Config) -> {Parent1, Parent2} when
    Population :: [individual()],
    Config :: neuro_config(),
    Parent1 :: individual(),
    Parent2 :: individual().
select_parents(Population, _Config) when length(Population) < 2 ->
    %% Edge case: only one individual
    Single = hd(Population),
    {Single, Single};
select_parents(Population, _Config) ->
    Parent1 = roulette_wheel(Population),
    %% Select second parent, try to avoid same individual
    Parent2 = select_different(Population, Parent1, 5),
    {Parent1, Parent2}.

%%% ============================================================================
%%% Internal Functions
%%% ============================================================================

%% @private
%% @doc Try to select a different individual than the given one.
%% Makes MaxAttempts tries before giving up and returning any individual.
-spec select_different([individual()], individual(), non_neg_integer()) -> individual().
select_different(Population, _Exclude, 0) ->
    random_select(Population);
select_different(Population, Exclude, AttemptsLeft) ->
    Candidate = roulette_wheel(Population),
    case Candidate#individual.id =:= Exclude#individual.id of
        true -> select_different(Population, Exclude, AttemptsLeft - 1);
        false -> Candidate
    end.
