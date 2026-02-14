%% @doc Statistics calculation for neuroevolution.
%%
%% This module provides utility functions for calculating population
%% statistics such as average, min, max fitness, and standard deviation.
%%
%% @author R.G. Lefever
%% @copyright 2024-2026 R.G. Lefever
-module(neuroevolution_stats).

-include("neuroevolution.hrl").

%% API
-export([
    avg_fitness/1,
    min_fitness/1,
    max_fitness/1,
    fitness_std_dev/1,
    population_summary/1
]).

%%% ============================================================================
%%% API Functions
%%% ============================================================================

%% @doc Calculate average fitness of a population.
-spec avg_fitness(Population) -> AvgFitness when
    Population :: [individual()],
    AvgFitness :: float().
avg_fitness([]) ->
    0.0;
avg_fitness(Population) ->
    Fitnesses = [Ind#individual.fitness || Ind <- Population],
    lists:sum(Fitnesses) / length(Fitnesses).

%% @doc Find minimum fitness in a population.
-spec min_fitness(Population) -> MinFitness when
    Population :: [individual()],
    MinFitness :: float().
min_fitness([]) ->
    0.0;
min_fitness(Population) ->
    Fitnesses = [Ind#individual.fitness || Ind <- Population],
    lists:min(Fitnesses).

%% @doc Find maximum fitness in a population.
-spec max_fitness(Population) -> MaxFitness when
    Population :: [individual()],
    MaxFitness :: float().
max_fitness([]) ->
    0.0;
max_fitness(Population) ->
    Fitnesses = [Ind#individual.fitness || Ind <- Population],
    lists:max(Fitnesses).

%% @doc Calculate standard deviation of fitness.
-spec fitness_std_dev(Population) -> StdDev when
    Population :: [individual()],
    StdDev :: float().
fitness_std_dev([]) ->
    0.0;
fitness_std_dev(Population) when length(Population) < 2 ->
    0.0;
fitness_std_dev(Population) ->
    Fitnesses = [Ind#individual.fitness || Ind <- Population],
    Mean = lists:sum(Fitnesses) / length(Fitnesses),
    Variance = lists:sum([math:pow(F - Mean, 2) || F <- Fitnesses]) / length(Fitnesses),
    math:sqrt(Variance).

%% @doc Generate a summary of population statistics.
%%
%% Uses tweann_nif:fitness_stats/1 for NIF-accelerated computation of
%% min, max, mean, and std_dev in a single pass when available.
-spec population_summary(Population) -> Summary when
    Population :: [individual()],
    Summary :: map().
population_summary([]) ->
    #{
        count => 0,
        avg_fitness => 0.0,
        min_fitness => 0.0,
        max_fitness => 0.0,
        std_dev => 0.0,
        survivors => 0,
        offspring => 0
    };
population_summary(Population) ->
    %% Extract fitnesses once for NIF call
    Fitnesses = [Ind#individual.fitness || Ind <- Population],

    %% Use NIF-accelerated stats computation (single pass through data)
    %% tweann_nif handles fallback internally if NIF not loaded
    %% Returns {Min, Max, Mean, Variance, StdDev, Sum}
    {Min, Max, Mean, _Variance, StdDev, _Sum} = tweann_nif:fitness_stats(Fitnesses),

    #{
        count => length(Population),
        avg_fitness => Mean,
        min_fitness => Min,
        max_fitness => Max,
        std_dev => StdDev,
        survivors => length([I || I <- Population, I#individual.is_survivor]),
        offspring => length([I || I <- Population, I#individual.is_offspring])
    }.
