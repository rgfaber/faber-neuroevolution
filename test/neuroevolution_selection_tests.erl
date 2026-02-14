%% @doc EUnit tests for neuroevolution_selection module.
-module(neuroevolution_selection_tests).

-include_lib("eunit/include/eunit.hrl").
-include("neuroevolution.hrl").

%%% ============================================================================
%%% Test Fixtures
%%% ============================================================================

sample_population() ->
    [
        #individual{id = make_ref(), fitness = 100.0},
        #individual{id = make_ref(), fitness = 80.0},
        #individual{id = make_ref(), fitness = 60.0},
        #individual{id = make_ref(), fitness = 40.0},
        #individual{id = make_ref(), fitness = 20.0}
    ].

%%% ============================================================================
%%% Top-N Selection Tests
%%% ============================================================================

top_n_selects_correct_count_test() ->
    Population = sample_population(),
    %% Select top 2 individuals
    Selected = neuroevolution_selection:top_n(Population, 2),
    ?assertEqual(2, length(Selected)).

top_n_selects_highest_fitness_test() ->
    Population = sample_population(),
    Selected = neuroevolution_selection:top_n(Population, 2),

    %% Should select the two highest fitness individuals
    Fitnesses = [I#individual.fitness || I <- Selected],
    ?assertEqual([100.0, 80.0], lists:sort(fun(A, B) -> A >= B end, Fitnesses)).

top_n_more_than_population_test() ->
    Population = sample_population(),
    %% Request more than available
    Selected = neuroevolution_selection:top_n(Population, 10),
    ?assertEqual(5, length(Selected)).

top_n_empty_population_test() ->
    Selected = neuroevolution_selection:top_n([], 5),
    ?assertEqual([], Selected).

%%% ============================================================================
%%% Tournament Selection Tests
%%% ============================================================================

tournament_selects_requested_count_test() ->
    Population = sample_population(),
    Selected = neuroevolution_selection:tournament(Population, 3, 2),
    ?assertEqual(2, length(Selected)).

tournament_returns_valid_individuals_test() ->
    Population = sample_population(),
    Selected = neuroevolution_selection:tournament(Population, 3, 1),

    %% Selected should be from the population
    Ids = [I#individual.id || I <- Population],
    lists:foreach(fun(S) ->
        ?assert(lists:member(S#individual.id, Ids))
    end, Selected).

tournament_favors_higher_fitness_test() ->
    %% Run many tournaments and check that higher fitness wins more often
    Population = sample_population(),

    Selections = lists:flatten([
        neuroevolution_selection:tournament(Population, 3, 1)
        || _ <- lists:seq(1, 1000)
    ]),
    AvgFitness = lists:sum([S#individual.fitness || S <- Selections]) / 1000,

    %% Average fitness of random selection would be 60
    %% Tournament should produce higher average
    ?assert(AvgFitness > 60.0).

%%% ============================================================================
%%% Roulette Wheel Selection Tests
%%% ============================================================================

roulette_wheel_selects_one_test() ->
    Population = sample_population(),
    Selected = neuroevolution_selection:roulette_wheel(Population),
    ?assert(is_record(Selected, individual)).

roulette_wheel_returns_valid_individual_test() ->
    Population = sample_population(),
    Selected = neuroevolution_selection:roulette_wheel(Population),

    Ids = [I#individual.id || I <- Population],
    ?assert(lists:member(Selected#individual.id, Ids)).

roulette_wheel_favors_higher_fitness_test() ->
    Population = sample_population(),

    Selections = [neuroevolution_selection:roulette_wheel(Population) || _ <- lists:seq(1, 1000)],
    AvgFitness = lists:sum([S#individual.fitness || S <- Selections]) / 1000,

    %% With roulette wheel, higher fitness individuals should be selected more often
    %% Average of uniform random would be 60, roulette should be higher
    ?assert(AvgFitness > 60.0).

%%% ============================================================================
%%% Random Selection Tests
%%% ============================================================================

random_select_returns_individual_test() ->
    Population = sample_population(),
    Selected = neuroevolution_selection:random_select(Population),
    ?assert(is_record(Selected, individual)).

random_select_returns_valid_individual_test() ->
    Population = sample_population(),
    Selected = neuroevolution_selection:random_select(Population),

    Ids = [I#individual.id || I <- Population],
    ?assert(lists:member(Selected#individual.id, Ids)).

%%% ============================================================================
%%% Select Parents Tests
%%% ============================================================================

select_parents_returns_different_individuals_test() ->
    Population = sample_population(),
    Config = #neuro_config{},
    {Parent1, Parent2} = neuroevolution_selection:select_parents(Population, Config),

    ?assertNotEqual(Parent1#individual.id, Parent2#individual.id).

select_parents_returns_valid_individuals_test() ->
    Population = sample_population(),
    Config = #neuro_config{},
    {Parent1, Parent2} = neuroevolution_selection:select_parents(Population, Config),

    ?assert(is_record(Parent1, individual)),
    ?assert(is_record(Parent2, individual)).
