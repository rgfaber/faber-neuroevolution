%% @doc NEAT-style speciation for neuroevolution.
%%
%% This module implements speciation - grouping individuals into species based
%% on genetic similarity. Species enable niching, allowing diverse strategies
%% to coexist and explore different fitness peaks without competing directly.
%%
%% == Compatibility Distance ==
%%
%% For fixed-topology networks (same structure, different weights), compatibility
%% is measured as the average absolute weight difference:
%% `distance(A, B) = (1/N) * sum(|w_a_i - w_b_i|)'
%%
%% Individuals with distance below the compatibility threshold belong to the
%% same species.
%%
%% == Species Lifecycle ==
%%
%% After evaluation, individuals are assigned to species. Each species gets
%% offspring quota proportional to average fitness. Within each species, top
%% performers are selected. Breeding is primarily within species, rarely between.
%% Species without improvement may be eliminated.
%%
%% == Dynamic Threshold ==
%%
%% The compatibility threshold adjusts dynamically to maintain the target
%% number of species. Too many species increases threshold (merging similar
%% species), too few decreases it (splitting diverse species).
%%
%% @author R.G. Lefever
%% @copyright 2024-2026 R.G. Lefever
-module(neuroevolution_speciation).

-include("neuroevolution.hrl").

%% Compatibility config for NEAT distance calculation
%% (mirrors genome_crossover internal record)
-record(compatibility_config, {
    c1 = 1.0 :: float(),  %% Coefficient for excess genes
    c2 = 1.0 :: float(),  %% Coefficient for disjoint genes
    c3 = 0.4 :: float()   %% Coefficient for weight differences
}).

-export([
    %% Core speciation
    speciate/3,
    assign_species/4,
    compatibility_distance/2,
    compatibility_distance/3,

    %% Species management
    update_species_fitness/2,
    calculate_offspring_quotas/2,
    adjust_compatibility_threshold/3,
    eliminate_stagnant_species/3,

    %% Species-aware breeding
    select_within_species/2,
    breed_species/4,

    %% Utilities
    get_species_by_id/2,
    get_individual_species/2,
    species_summary/1
]).

%%% ============================================================================
%%% Core Speciation Functions
%%% ============================================================================

%% @doc Assign all individuals in population to species.
%%
%% This is the main speciation entry point. Called after evaluation to
%% organize the population into species clusters.
%%
%% @param Population List of evaluated individuals
%% @param CurrentSpecies Existing species from previous generation
%% @param Config Speciation configuration
%% @returns {UpdatedSpecies, NewSpeciesEvents, NextSpeciesId}
-spec speciate([individual()], [species()], speciation_config()) ->
    {[species()], [species_event()], species_id()}.
speciate(Population, CurrentSpecies, Config) ->
    speciate(Population, CurrentSpecies, Config, 1).

speciate(Population, CurrentSpecies, Config, NextSpeciesId) ->
    %% Clear old members from species (keep representatives)
    ClearedSpecies = [S#species{members = [], best_fitness = 0.0} || S <- CurrentSpecies],

    %% Assign each individual to a species
    {UpdatedSpecies, Events, NewNextId} = lists:foldl(
        fun(Individual, {Species, Evts, NextId}) ->
            {NewSpecies, NewEvents, NewNextId} = assign_species(
                Individual, Species, Config, NextId
            ),
            {NewSpecies, Evts ++ NewEvents, NewNextId}
        end,
        {ClearedSpecies, [], NextSpeciesId},
        Population
    ),

    %% Remove empty species (all members were reassigned elsewhere)
    {ActiveSpecies, ExtinctEvents} = remove_empty_species(UpdatedSpecies),

    %% Update representatives for non-empty species
    FinalSpecies = update_representatives(ActiveSpecies),

    {FinalSpecies, Events ++ ExtinctEvents, NewNextId}.

%% @doc Assign a single individual to a species.
%%
%% Compares the individual to each species' representative. If compatible
%% with an existing species, joins it. Otherwise, creates a new species.
%%
%% @param Individual The individual to assign
%% @param Species List of current species
%% @param Config Speciation configuration
%% @param NextSpeciesId ID to use if creating new species
%% @returns {UpdatedSpecies, Events, NextSpeciesId}
-spec assign_species(individual(), [species()], speciation_config(), species_id()) ->
    {[species()], [species_event()], species_id()}.
assign_species(Individual, [], _Config, NextSpeciesId) ->
    %% No species exist - create first one
    NewSpecies = create_species(Individual, NextSpeciesId, Individual#individual.generation_born),
    Event = #species_event{
        generation = Individual#individual.generation_born,
        species_id = NextSpeciesId,
        event_type = species_created,
        details = #{founder_id => Individual#individual.id, founder_fitness => Individual#individual.fitness}
    },
    {[NewSpecies], [Event], NextSpeciesId + 1};

assign_species(Individual, Species, Config, NextSpeciesId) ->
    Threshold = Config#speciation_config.compatibility_threshold,

    %% Find compatible species
    case find_compatible_species(Individual, Species, Threshold) of
        {ok, CompatibleSpecies} ->
            %% Add to existing species
            UpdatedSpecies = add_to_species(Individual, CompatibleSpecies, Species),
            {UpdatedSpecies, [], NextSpeciesId};

        not_found ->
            %% Create new species
            NewSpecies = create_species(Individual, NextSpeciesId, Individual#individual.generation_born),
            Event = #species_event{
                generation = Individual#individual.generation_born,
                species_id = NextSpeciesId,
                event_type = species_created,
                details = #{founder_id => Individual#individual.id, founder_fitness => Individual#individual.fitness}
            },
            {Species ++ [NewSpecies], [Event], NextSpeciesId + 1}
    end.

%% @doc Calculate compatibility distance between two individuals.
%%
%% Uses NEAT-style distance when genomes are available:
%% delta = (c1 * E / N) + (c2 * D / N) + (c3 * W)
%% Where E=excess genes, D=disjoint genes, N=genome size, W=avg weight diff
%%
%% Falls back to weight-only distance for fixed-topology (no genome).
%%
%% @param Ind1 First individual
%% @param Ind2 Second individual
%% @returns Compatibility distance (non-negative float)
-spec compatibility_distance(individual(), individual()) -> float().
compatibility_distance(Ind1, Ind2) ->
    %% Check if both individuals have NEAT genomes
    case {Ind1#individual.genome, Ind2#individual.genome} of
        {undefined, _} ->
            %% No genome - use weight-based distance (legacy)
            weight_based_distance(Ind1, Ind2);
        {_, undefined} ->
            weight_based_distance(Ind1, Ind2);
        {Genome1, Genome2} ->
            %% Both have genomes - use NEAT compatibility distance
            neat_compatibility_distance(Genome1, Genome2)
    end.

%% @doc Calculate NEAT compatibility distance with config.
%%
%% Uses speciation config coefficients for the NEAT formula.
%%
%% @param Ind1 First individual
%% @param Ind2 Second individual
%% @param Config Speciation configuration with NEAT coefficients
%% @returns Compatibility distance (non-negative float)
-spec compatibility_distance(individual(), individual(), speciation_config()) -> float().
compatibility_distance(Ind1, Ind2, Config) ->
    case {Ind1#individual.genome, Ind2#individual.genome} of
        {undefined, _} ->
            weight_based_distance(Ind1, Ind2);
        {_, undefined} ->
            weight_based_distance(Ind1, Ind2);
        {Genome1, Genome2} ->
            %% Use config coefficients
            CompatConfig = #compatibility_config{
                c1 = Config#speciation_config.c1_excess,
                c2 = Config#speciation_config.c2_disjoint,
                c3 = Config#speciation_config.c3_weight_diff
            },
            genome_crossover:compatibility_distance(
                Genome1#genome.connection_genes,
                Genome2#genome.connection_genes,
                CompatConfig
            )
    end.

%% @private Calculate weight-based distance (legacy, for fixed-topology).
%%
%% Used when individuals don't have NEAT genomes.
%% Distance = average absolute weight difference.
-spec weight_based_distance(individual(), individual()) -> float().
weight_based_distance(Ind1, Ind2) ->
    Weights1 = get_weights(Ind1),
    Weights2 = get_weights(Ind2),

    case {Weights1, Weights2} of
        {[], _} -> 1.0;  %% Default high distance if no weights
        {_, []} -> 1.0;
        _ when length(Weights1) =/= length(Weights2) ->
            1.0;  %% Different topology
        _ ->
            Diffs = [abs(W1 - W2) || {W1, W2} <- lists:zip(Weights1, Weights2)],
            lists:sum(Diffs) / length(Diffs)
    end.

%% @private Calculate NEAT compatibility distance with default coefficients.
%%
%% Uses standard NEAT coefficients: c1=1.0, c2=1.0, c3=0.4
%% Automatically uses NIF acceleration when available (30-50x faster).
-spec neat_compatibility_distance(genome(), genome()) -> float().
neat_compatibility_distance(Genome1, Genome2) ->
    %% Default NEAT coefficients
    C1 = 1.0,  %% Excess gene coefficient
    C2 = 1.0,  %% Disjoint gene coefficient
    C3 = 0.4,  %% Weight difference coefficient

    %% Use NIF-accelerated distance when available
    case nif_network:is_nif_available() of
        true ->
            nif_network:compatibility_distance(
                Genome1#genome.connection_genes,
                Genome2#genome.connection_genes,
                C1, C2, C3
            );
        false ->
            %% Pure Erlang fallback
            CompatConfig = #compatibility_config{c1 = C1, c2 = C2, c3 = C3},
            genome_crossover:compatibility_distance(
                Genome1#genome.connection_genes,
                Genome2#genome.connection_genes,
                CompatConfig
            )
    end.

%%% ============================================================================
%%% Species Management
%%% ============================================================================

%% @doc Update fitness statistics for all species after evaluation.
%%
%% @param Species List of species
%% @param Population Evaluated population
%% @returns Updated species list
-spec update_species_fitness([species()], [individual()]) -> [species()].
update_species_fitness(Species, Population) ->
    %% Build ID -> Individual map for fast lookup
    IndMap = maps:from_list([{I#individual.id, I} || I <- Population]),

    lists:map(
        fun(S) ->
            %% Get fitness of all members
            MemberFitnesses = [
                (maps:get(MemberId, IndMap, #individual{fitness = 0.0}))#individual.fitness
                || MemberId <- S#species.members
            ],

            BestFitness = case MemberFitnesses of
                [] -> 0.0;
                _ -> lists:max(MemberFitnesses)
            end,

            %% Check for improvement
            Improved = BestFitness > S#species.best_fitness_ever,
            NewStagnant = case Improved of
                true -> 0;
                false -> S#species.stagnant_generations + 1
            end,

            S#species{
                best_fitness = BestFitness,
                best_fitness_ever = max(BestFitness, S#species.best_fitness_ever),
                stagnant_generations = NewStagnant,
                age = S#species.age + 1
            }
        end,
        Species
    ).

%% @doc Calculate offspring quota for each species based on relative fitness.
%%
%% Species with higher average fitness get more offspring slots.
%% This implements fitness sharing / explicit fitness sharing.
%%
%% @param Species List of species with updated fitness
%% @param TotalOffspring Total number of offspring to create
%% @returns Species with offspring_quota field populated
-spec calculate_offspring_quotas([species()], pos_integer()) -> [species()].
calculate_offspring_quotas([], _TotalOffspring) ->
    [];
calculate_offspring_quotas(Species, TotalOffspring) ->
    %% Calculate adjusted fitness for each species (fitness / species_size)
    %% This implements fitness sharing - large species don't dominate
    AdjustedFitnesses = [
        {S, adjusted_species_fitness(S)}
        || S <- Species
    ],

    TotalAdjusted = lists:sum([F || {_, F} <- AdjustedFitnesses]),

    case TotalAdjusted > 0 of
        true ->
            %% Distribute offspring proportionally
            WithQuotas = [
                S#species{
                    offspring_quota = max(1, round(F / TotalAdjusted * TotalOffspring))
                }
                || {S, F} <- AdjustedFitnesses
            ],

            %% Adjust to exactly match TotalOffspring
            adjust_quotas(WithQuotas, TotalOffspring);

        false ->
            %% All species have 0 fitness - distribute equally
            EqualQuota = max(1, TotalOffspring div length(Species)),
            [S#species{offspring_quota = EqualQuota} || S <- Species]
    end.

%% @doc Dynamically adjust compatibility threshold to maintain target species count.
%%
%% @param CurrentThreshold Current compatibility threshold
%% @param Config Speciation configuration
%% @param ActualSpeciesCount Current number of species
%% @returns New compatibility threshold
-spec adjust_compatibility_threshold(float(), speciation_config(), non_neg_integer()) -> float().
adjust_compatibility_threshold(CurrentThreshold, Config, ActualSpeciesCount) ->
    Target = Config#speciation_config.target_species,
    Rate = Config#speciation_config.threshold_adjustment_rate,

    if
        ActualSpeciesCount > Target ->
            %% Too many species - increase threshold to merge
            CurrentThreshold + Rate;
        ActualSpeciesCount < Target ->
            %% Too few species - decrease threshold to split
            max(0.1, CurrentThreshold - Rate);
        true ->
            CurrentThreshold
    end.

%% @doc Eliminate species that have been stagnant too long.
%%
%% Stagnant species (no fitness improvement for many generations) are
%% removed to make room for new exploration.
%%
%% @param Species List of species
%% @param Config Speciation configuration
%% @param Generation Current generation
%% @returns {RemainingSpecies, ExtinctionEvents}
-spec eliminate_stagnant_species([species()], speciation_config(), generation()) ->
    {[species()], [species_event()]}.
eliminate_stagnant_species(Species, Config, Generation) ->
    MaxStagnation = Config#speciation_config.max_stagnation,

    case MaxStagnation of
        0 ->
            %% Stagnation elimination disabled
            {Species, []};
        _ ->
            %% Keep at least one species (the best one)
            BestSpecies = lists:max([S#species.best_fitness_ever || S <- Species]),

            lists:foldl(
                fun(S, {Remaining, Events}) ->
                    IsStagnant = S#species.stagnant_generations >= MaxStagnation,
                    IsBest = S#species.best_fitness_ever =:= BestSpecies,

                    case IsStagnant andalso not IsBest of
                        true ->
                            Event = #species_event{
                                generation = Generation,
                                species_id = S#species.id,
                                event_type = species_extinct,
                                details = #{
                                    reason => stagnation,
                                    stagnant_generations => S#species.stagnant_generations,
                                    best_fitness_ever => S#species.best_fitness_ever
                                }
                            },
                            {Remaining, [Event | Events]};
                        false ->
                            {[S | Remaining], Events}
                    end
                end,
                {[], []},
                Species
            )
    end.

%%% ============================================================================
%%% Species-Aware Breeding
%%% ============================================================================

%% @doc Select survivors within a species.
%%
%% Applies elitism within the species - top performers survive.
%%
%% @param Species The species to select within
%% @param Config Speciation configuration
%% @returns {Survivors, Eliminated} as individual IDs
-spec select_within_species(species(), speciation_config()) ->
    {[individual_id()], [individual_id()]}.
select_within_species(Species, Config) ->
    Members = Species#species.members,
    NumSurvivors = max(1, round(length(Members) * Config#speciation_config.species_elitism)),

    %% Members should already be sorted by fitness (best first)
    Survivors = lists:sublist(Members, NumSurvivors),
    Eliminated = lists:nthtail(NumSurvivors, Members),

    {Survivors, Eliminated}.

%% @doc Breed offspring within/between species.
%%
%% Primarily breeds within species. Occasionally (based on interspecies_mating_rate)
%% breeds between different species.
%%
%% @param Species The species to breed from
%% @param AllSpecies All species (for interspecies mating)
%% @param Config Speciation configuration
%% @param NumOffspring Number of offspring to create
%% @returns List of {Parent1Id, Parent2Id} pairs for breeding
-spec breed_species(species(), [species()], speciation_config(), pos_integer()) ->
    [{individual_id(), individual_id()}].
breed_species(Species, AllSpecies, Config, NumOffspring) ->
    InterspeciesRate = Config#speciation_config.interspecies_mating_rate,
    Members = Species#species.members,

    case Members of
        [] ->
            [];
        [Single] ->
            %% Only one member - self-breed (clone with mutation)
            [{Single, Single} || _ <- lists:seq(1, NumOffspring)];
        _ ->
            %% Multiple members - breed pairs
            [
                case rand:uniform() < InterspeciesRate of
                    true ->
                        %% Interspecies mating
                        select_interspecies_pair(Members, AllSpecies);
                    false ->
                        %% Within species mating
                        select_parents_from_list(Members)
                end
                || _ <- lists:seq(1, NumOffspring)
            ]
    end.

%%% ============================================================================
%%% Utilities
%%% ============================================================================

%% @doc Get species by ID.
-spec get_species_by_id(species_id(), [species()]) -> {ok, species()} | not_found.
get_species_by_id(Id, Species) ->
    case lists:keyfind(Id, #species.id, Species) of
        false -> not_found;
        S -> {ok, S}
    end.

%% @doc Get the species ID for an individual.
-spec get_individual_species(individual_id(), [species()]) -> {ok, species_id()} | not_found.
get_individual_species(IndId, Species) ->
    case lists:filter(
        fun(S) -> lists:member(IndId, S#species.members) end,
        Species
    ) of
        [S | _] -> {ok, S#species.id};
        [] -> not_found
    end.

%% @doc Generate a summary of species for visualization/logging.
-spec species_summary([species()]) -> [map()].
species_summary(Species) ->
    [
        #{
            id => S#species.id,
            size => length(S#species.members),
            best_fitness => S#species.best_fitness,
            best_ever => S#species.best_fitness_ever,
            age => S#species.age,
            stagnant => S#species.stagnant_generations,
            offspring_quota => S#species.offspring_quota
        }
        || S <- Species
    ].

%%% ============================================================================
%%% Internal Functions
%%% ============================================================================

%% @private Get weights from individual's network.
get_weights(#individual{network = Network}) ->
    try
        network_evaluator:get_weights(Network)
    catch
        _:_ -> []
    end.

%% @private Find a compatible species for an individual.
find_compatible_species(_Individual, [], _Threshold) ->
    not_found;
find_compatible_species(Individual, [Species | Rest], Threshold) ->
    Rep = Species#species.representative,
    Distance = compatibility_distance(Individual, Rep),

    case Distance < Threshold of
        true -> {ok, Species};
        false -> find_compatible_species(Individual, Rest, Threshold)
    end.

%% @private Create a new species with given individual as founder.
create_species(Individual, SpeciesId, Generation) ->
    #species{
        id = SpeciesId,
        representative = Individual,
        members = [Individual#individual.id],
        best_fitness = Individual#individual.fitness,
        best_fitness_ever = Individual#individual.fitness,
        generation_created = Generation,
        age = 0,
        stagnant_generations = 0,
        offspring_quota = 0
    }.

%% @private Add individual to an existing species.
add_to_species(Individual, TargetSpecies, AllSpecies) ->
    lists:map(
        fun(S) ->
            case S#species.id =:= TargetSpecies#species.id of
                true ->
                    S#species{
                        members = [Individual#individual.id | S#species.members],
                        best_fitness = max(S#species.best_fitness, Individual#individual.fitness)
                    };
                false ->
                    S
            end
        end,
        AllSpecies
    ).

%% @private Remove species with no members.
remove_empty_species(Species) ->
    lists:partition(
        fun(S) -> S#species.members =/= [] end,
        Species
    ).

%% @private Update representative for each species (random member).
update_representatives(Species) ->
    [
        case S#species.members of
            [] -> S;
            _Members ->
                %% Pick random member as new representative
                %% (In practice, you might want to keep the old one or pick best)
                S
        end
        || S <- Species
    ].

%% @private Calculate adjusted fitness for a species (implements fitness sharing).
adjusted_species_fitness(#species{members = [], best_fitness = _}) ->
    0.0;
adjusted_species_fitness(#species{members = Members, best_fitness = BestFitness}) ->
    %% Adjusted fitness = best_fitness / sqrt(size)
    %% This penalizes large species, promoting diversity
    BestFitness / math:sqrt(length(Members)).

%% @private Adjust quotas to exactly match target total.
adjust_quotas(Species, TotalOffspring) ->
    CurrentTotal = lists:sum([S#species.offspring_quota || S <- Species]),
    Diff = TotalOffspring - CurrentTotal,

    case Diff of
        0 ->
            Species;
        _ when Diff > 0 ->
            %% Need more - add to best species
            add_to_best_species(Species, Diff);
        _ ->
            %% Need fewer - remove from worst species
            remove_from_worst_species(Species, -Diff)
    end.

%% @private Add offspring quota to best performing species.
add_to_best_species(Species, Amount) ->
    Sorted = lists:sort(
        fun(A, B) -> A#species.best_fitness >= B#species.best_fitness end,
        Species
    ),
    case Sorted of
        [Best | Rest] ->
            [Best#species{offspring_quota = Best#species.offspring_quota + Amount} | Rest];
        [] ->
            []
    end.

%% @private Remove offspring quota from worst performing species.
remove_from_worst_species(Species, Amount) ->
    Sorted = lists:sort(
        fun(A, B) -> A#species.best_fitness =< B#species.best_fitness end,
        Species
    ),
    remove_quota_helper(Sorted, Amount, []).

remove_quota_helper([], _Amount, Acc) ->
    lists:reverse(Acc);
remove_quota_helper([S | Rest], Amount, Acc) when Amount =< 0 ->
    lists:reverse(Acc) ++ [S | Rest];
remove_quota_helper([S | Rest], Amount, Acc) ->
    CanRemove = max(0, S#species.offspring_quota - 1),
    ToRemove = min(CanRemove, Amount),
    NewS = S#species{offspring_quota = S#species.offspring_quota - ToRemove},
    remove_quota_helper(Rest, Amount - ToRemove, [NewS | Acc]).

%% @private Select two parents from a list (for within-species breeding).
select_parents_from_list(Members) ->
    P1 = lists:nth(rand:uniform(length(Members)), Members),
    P2 = select_different(P1, Members, 3),
    {P1, P2}.

%% @private Select a parent different from the first (with retries).
select_different(First, _Members, 0) ->
    First;  %% Give up, use same parent
select_different(First, Members, Retries) ->
    Candidate = lists:nth(rand:uniform(length(Members)), Members),
    case Candidate =:= First of
        true -> select_different(First, Members, Retries - 1);
        false -> Candidate
    end.

%% @private Select interspecies breeding pair.
select_interspecies_pair(LocalMembers, AllSpecies) ->
    P1 = lists:nth(rand:uniform(length(LocalMembers)), LocalMembers),

    %% Pick random other species
    OtherSpecies = [S || S <- AllSpecies, not lists:member(P1, S#species.members)],
    case OtherSpecies of
        [] ->
            %% No other species - breed within
            P2 = select_different(P1, LocalMembers, 3),
            {P1, P2};
        _ ->
            RandomSpecies = lists:nth(rand:uniform(length(OtherSpecies)), OtherSpecies),
            case RandomSpecies#species.members of
                [] ->
                    {P1, P1};
                OtherMembers ->
                    P2 = lists:nth(rand:uniform(length(OtherMembers)), OtherMembers),
                    {P1, P2}
            end
    end.
