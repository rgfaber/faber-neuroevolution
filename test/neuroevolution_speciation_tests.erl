%% @doc Unit tests for neuroevolution_speciation module.
-module(neuroevolution_speciation_tests).

-include_lib("eunit/include/eunit.hrl").
-include("neuroevolution.hrl").

%%% ============================================================================
%%% Test Fixtures
%%% ============================================================================

default_speciation_config() ->
    #speciation_config{
        compatibility_threshold = 0.5,
        target_species = 5,
        threshold_adjustment_rate = 0.1,
        max_stagnation = 15,
        species_elitism = 0.2,
        interspecies_mating_rate = 0.01
    }.

mock_network(_Weights) ->
    %% Create a simple network structure for testing
    %% Note: network_evaluator:get_weights/1 will return [] for this mock,
    %% so compatibility_distance will return 1.0 (high distance)
    #{structure => [3, 4, 2]}.

sample_individual(Id, Fitness, Weights) ->
    #individual{
        id = Id,
        network = mock_network(Weights),
        fitness = Fitness,
        generation_born = 1
    }.

sample_species(Id, Members, BestFitness) ->
    Rep = sample_individual(rep, BestFitness, [0.5, 0.5, 0.5]),
    #species{
        id = Id,
        representative = Rep,
        members = Members,
        best_fitness = BestFitness,
        best_fitness_ever = BestFitness,
        generation_created = 1,
        age = 1,
        stagnant_generations = 0,
        offspring_quota = 0
    }.

%%% ============================================================================
%%% Compatibility Distance Tests
%%% ============================================================================

compatibility_distance_mock_returns_max_test() ->
    %% With mock networks, network_evaluator:get_weights returns [],
    %% so individuals without genomes get high distance (incompatible)
    Ind1 = sample_individual(1, 0.5, [0.1, 0.2, 0.3]),
    Ind2 = sample_individual(2, 0.6, [0.1, 0.2, 0.3]),

    Distance = neuroevolution_speciation:compatibility_distance(Ind1, Ind2),
    ?assertEqual(1.0, Distance).  %% Mock networks have no weights -> high distance

compatibility_distance_empty_weights_test() ->
    Ind1 = sample_individual(1, 0.5, []),
    Ind2 = sample_individual(2, 0.6, [0.1, 0.2]),

    Distance = neuroevolution_speciation:compatibility_distance(Ind1, Ind2),
    ?assertEqual(1.0, Distance).  %% High distance for empty weights

%%% ============================================================================
%%% Speciate Tests
%%% ============================================================================

speciate_empty_population_test() ->
    Config = default_speciation_config(),

    {Species, Events, NextId} = neuroevolution_speciation:speciate([], [], Config),

    ?assertEqual([], Species),
    ?assertEqual([], Events),
    ?assertEqual(1, NextId).

speciate_single_individual_test() ->
    Config = default_speciation_config(),
    Ind = sample_individual(1, 0.5, [0.1, 0.2, 0.3]),

    {Species, Events, NextId} = neuroevolution_speciation:speciate([Ind], [], Config),

    ?assertEqual(1, length(Species)),
    ?assertEqual(1, length(Events)),
    ?assertEqual(species_created, (hd(Events))#species_event.event_type),
    ?assertEqual(2, NextId).

speciate_mock_individuals_different_species_test() ->
    %% With mock networks (no weights), all individuals are maximally distant
    %% so each gets its own species
    Config = default_speciation_config(),
    Ind1 = sample_individual(1, 0.5, [0.1, 0.2, 0.3]),
    Ind2 = sample_individual(2, 0.6, [0.15, 0.25, 0.35]),

    {Species, Events, _NextId} = neuroevolution_speciation:speciate([Ind1, Ind2], [], Config),

    %% With mock networks, each individual creates a new species
    ?assertEqual(2, length(Species)),
    ?assertEqual(2, length(Events)).

speciate_creates_species_for_each_distant_individual_test() ->
    Config = default_speciation_config(),
    Ind1 = sample_individual(1, 0.5, [0.1, 0.2, 0.3]),
    Ind2 = sample_individual(2, 0.6, [0.9, 0.8, 0.7]),

    {Species, Events, _NextId} = neuroevolution_speciation:speciate([Ind1, Ind2], [], Config),

    %% Both individuals are maximally distant (mock networks)
    ?assertEqual(2, length(Species)),
    ?assertEqual(2, length(Events)).

%%% ============================================================================
%%% Assign Species Tests
%%% ============================================================================

assign_species_to_empty_test() ->
    Config = default_speciation_config(),
    Ind = sample_individual(1, 0.5, [0.1, 0.2, 0.3]),

    {Species, Events, NextId} = neuroevolution_speciation:assign_species(Ind, [], Config, 1),

    ?assertEqual(1, length(Species)),
    ?assertEqual(1, length(Events)),
    ?assertEqual(2, NextId).

assign_species_to_incompatible_creates_new_test() ->
    %% With mock networks (no weights), individual is incompatible with existing species
    Config = default_speciation_config(),
    ExistingSpecies = sample_species(1, [1], 0.5),
    Ind = sample_individual(2, 0.6, [0.5, 0.5, 0.5]),

    {Species, Events, NextId} = neuroevolution_speciation:assign_species(
        Ind, [ExistingSpecies], Config, 2
    ),

    %% Mock networks = max distance, so new species is created
    ?assertEqual(2, length(Species)),
    ?assertEqual(1, length(Events)),
    ?assertEqual(3, NextId).

%%% ============================================================================
%%% Species Fitness Update Tests
%%% ============================================================================

update_species_fitness_test() ->
    Ind1 = sample_individual(1, 0.8, [0.1, 0.2, 0.3]),
    Ind2 = sample_individual(2, 0.6, [0.4, 0.5, 0.6]),
    Population = [Ind1, Ind2],

    Species = sample_species(1, [1, 2], 0.5),

    [UpdatedSpecies] = neuroevolution_speciation:update_species_fitness([Species], Population),

    ?assertEqual(0.8, UpdatedSpecies#species.best_fitness),
    ?assertEqual(0.8, UpdatedSpecies#species.best_fitness_ever),
    ?assertEqual(2, UpdatedSpecies#species.age).

update_species_fitness_stagnation_test() ->
    Ind1 = sample_individual(1, 0.4, [0.1, 0.2, 0.3]),  %% Lower than best_ever
    Population = [Ind1],

    Species = #species{
        id = 1,
        representative = Ind1,
        members = [1],
        best_fitness = 0.8,
        best_fitness_ever = 0.8,
        generation_created = 1,
        age = 5,
        stagnant_generations = 2,
        offspring_quota = 0
    },

    [Updated] = neuroevolution_speciation:update_species_fitness([Species], Population),

    ?assertEqual(0.4, Updated#species.best_fitness),
    ?assertEqual(0.8, Updated#species.best_fitness_ever),  %% Unchanged
    ?assertEqual(3, Updated#species.stagnant_generations).  %% Incremented

%%% ============================================================================
%%% Offspring Quota Tests
%%% ============================================================================

calculate_offspring_quotas_single_species_test() ->
    Species = sample_species(1, [1, 2, 3], 0.8),

    [Updated] = neuroevolution_speciation:calculate_offspring_quotas([Species], 10),

    ?assertEqual(10, Updated#species.offspring_quota).

calculate_offspring_quotas_multiple_species_test() ->
    Species1 = sample_species(1, [1, 2], 0.8),    %% Higher fitness
    Species2 = sample_species(2, [3, 4], 0.4),    %% Lower fitness

    Updated = neuroevolution_speciation:calculate_offspring_quotas([Species1, Species2], 10),

    ?assertEqual(2, length(Updated)),
    TotalQuota = lists:sum([S#species.offspring_quota || S <- Updated]),
    ?assertEqual(10, TotalQuota).

calculate_offspring_quotas_empty_test() ->
    Result = neuroevolution_speciation:calculate_offspring_quotas([], 10),
    ?assertEqual([], Result).

%%% ============================================================================
%%% Threshold Adjustment Tests
%%% ============================================================================

adjust_threshold_too_many_species_test() ->
    Config = default_speciation_config(),
    CurrentThreshold = 0.5,
    ActualCount = 10,  %% More than target (5)

    NewThreshold = neuroevolution_speciation:adjust_compatibility_threshold(
        CurrentThreshold, Config, ActualCount
    ),

    ?assert(NewThreshold > CurrentThreshold).

adjust_threshold_too_few_species_test() ->
    Config = default_speciation_config(),
    CurrentThreshold = 0.5,
    ActualCount = 2,  %% Less than target (5)

    NewThreshold = neuroevolution_speciation:adjust_compatibility_threshold(
        CurrentThreshold, Config, ActualCount
    ),

    ?assert(NewThreshold < CurrentThreshold).

adjust_threshold_at_target_test() ->
    Config = default_speciation_config(),
    CurrentThreshold = 0.5,
    ActualCount = 5,  %% At target

    NewThreshold = neuroevolution_speciation:adjust_compatibility_threshold(
        CurrentThreshold, Config, ActualCount
    ),

    ?assertEqual(CurrentThreshold, NewThreshold).

adjust_threshold_minimum_test() ->
    Config = default_speciation_config(),
    CurrentThreshold = 0.1,  %% At minimum
    ActualCount = 2,  %% Less than target

    NewThreshold = neuroevolution_speciation:adjust_compatibility_threshold(
        CurrentThreshold, Config, ActualCount
    ),

    ?assert(NewThreshold >= 0.1).

%%% ============================================================================
%%% Stagnant Species Elimination Tests
%%% ============================================================================

eliminate_stagnant_none_stagnant_test() ->
    Config = default_speciation_config(),
    Species = sample_species(1, [1, 2], 0.8),

    {Remaining, Events} = neuroevolution_speciation:eliminate_stagnant_species(
        [Species], Config, 10
    ),

    ?assertEqual(1, length(Remaining)),
    ?assertEqual([], Events).

eliminate_stagnant_one_stagnant_test() ->
    Config = default_speciation_config(),
    StagnantSpecies = #species{
        id = 1,
        representative = sample_individual(1, 0.3, [0.1, 0.2, 0.3]),
        members = [1],
        best_fitness = 0.3,
        best_fitness_ever = 0.3,
        generation_created = 1,
        age = 20,
        stagnant_generations = 20,  %% Exceeds max_stagnation (15)
        offspring_quota = 0
    },
    GoodSpecies = sample_species(2, [2], 0.8),

    {Remaining, Events} = neuroevolution_speciation:eliminate_stagnant_species(
        [StagnantSpecies, GoodSpecies], Config, 10
    ),

    ?assertEqual(1, length(Remaining)),
    ?assertEqual(1, length(Events)),
    ?assertEqual(species_extinct, (hd(Events))#species_event.event_type).

eliminate_stagnant_keeps_best_test() ->
    Config = default_speciation_config(),
    %% Even if stagnant, should keep the best species
    StagnantBest = #species{
        id = 1,
        representative = sample_individual(1, 0.9, [0.1, 0.2, 0.3]),
        members = [1],
        best_fitness = 0.9,
        best_fitness_ever = 0.9,
        generation_created = 1,
        age = 20,
        stagnant_generations = 20,
        offspring_quota = 0
    },

    {Remaining, Events} = neuroevolution_speciation:eliminate_stagnant_species(
        [StagnantBest], Config, 10
    ),

    ?assertEqual(1, length(Remaining)),
    ?assertEqual([], Events).

%%% ============================================================================
%%% Utility Tests
%%% ============================================================================

get_species_by_id_found_test() ->
    Species1 = sample_species(1, [1], 0.5),
    Species2 = sample_species(2, [2], 0.6),

    {ok, Found} = neuroevolution_speciation:get_species_by_id(2, [Species1, Species2]),
    ?assertEqual(2, Found#species.id).

get_species_by_id_not_found_test() ->
    Species1 = sample_species(1, [1], 0.5),

    Result = neuroevolution_speciation:get_species_by_id(999, [Species1]),
    ?assertEqual(not_found, Result).

get_individual_species_found_test() ->
    Species = sample_species(1, [42, 43], 0.5),

    {ok, SpeciesId} = neuroevolution_speciation:get_individual_species(42, [Species]),
    ?assertEqual(1, SpeciesId).

get_individual_species_not_found_test() ->
    Species = sample_species(1, [1, 2], 0.5),

    Result = neuroevolution_speciation:get_individual_species(999, [Species]),
    ?assertEqual(not_found, Result).

species_summary_test() ->
    Species1 = sample_species(1, [1, 2, 3], 0.8),
    Species2 = sample_species(2, [4, 5], 0.6),

    Summary = neuroevolution_speciation:species_summary([Species1, Species2]),

    ?assertEqual(2, length(Summary)),
    [S1, S2] = Summary,
    ?assertEqual(1, maps:get(id, S1)),
    ?assertEqual(3, maps:get(size, S1)),
    ?assertEqual(2, maps:get(id, S2)),
    ?assertEqual(2, maps:get(size, S2)).

%%% ============================================================================
%%% Breeding Tests
%%% ============================================================================

select_within_species_test() ->
    Config = default_speciation_config(),
    Species = sample_species(1, [1, 2, 3, 4, 5], 0.8),

    {Survivors, Eliminated} = neuroevolution_speciation:select_within_species(Species, Config),

    %% With 0.2 elitism on 5 members, expect 1 survivor
    ?assert(length(Survivors) >= 1),
    ?assertEqual(5, length(Survivors) + length(Eliminated)).

breed_species_single_member_test() ->
    Config = default_speciation_config(),
    Species = sample_species(1, [42], 0.8),

    Pairs = neuroevolution_speciation:breed_species(Species, [Species], Config, 3),

    ?assertEqual(3, length(Pairs)),
    %% All pairs should be self-breeding
    lists:foreach(
        fun({P1, P2}) -> ?assertEqual(P1, P2) end,
        Pairs
    ).

breed_species_multiple_members_test() ->
    Config = default_speciation_config(),
    Species = sample_species(1, [1, 2, 3, 4, 5], 0.8),

    Pairs = neuroevolution_speciation:breed_species(Species, [Species], Config, 5),

    ?assertEqual(5, length(Pairs)),
    lists:foreach(
        fun({P1, P2}) ->
            ?assert(lists:member(P1, [1, 2, 3, 4, 5])),
            ?assert(lists:member(P2, [1, 2, 3, 4, 5]))
        end,
        Pairs
    ).

breed_species_empty_test() ->
    Config = default_speciation_config(),
    Species = sample_species(1, [], 0.0),

    Pairs = neuroevolution_speciation:breed_species(Species, [Species], Config, 3),

    ?assertEqual([], Pairs).
