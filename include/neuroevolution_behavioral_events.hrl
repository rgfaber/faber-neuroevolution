%%%-------------------------------------------------------------------
%%% @doc Behavioral event type definitions for neuroevolution.
%%%
%%% Events follow behavioral naming conventions:
%%% - Past tense (facts that happened)
%%% - Domain language (evolutionary biology terminology)
%%% - No CRUD operations (offspring_born, not individual_created)
%%%
%%% @end
%%%-------------------------------------------------------------------
-ifndef(NEUROEVOLUTION_BEHAVIORAL_EVENTS_HRL).
-define(NEUROEVOLUTION_BEHAVIORAL_EVENTS_HRL, true).

%%% ============================================================================
%%% Common Types
%%% ============================================================================

-type individual_id() :: binary().
-type species_id() :: binary().
-type population_id() :: binary().
-type generation() :: non_neg_integer().
-type timestamp() :: integer().
-type fitness() :: float().

%%% ============================================================================
%%% Event Metadata (included in all events)
%%% ============================================================================

-record(event_meta, {
    event_id :: binary(),
    correlation_id :: binary() | undefined,
    causation_id :: binary() | undefined,
    timestamp :: timestamp(),
    version = 1 :: pos_integer(),
    emitter :: atom()
}).

%%% ============================================================================
%%% Birth Events
%%% ============================================================================

-record(offspring_born, {
    meta :: #event_meta{},
    individual_id :: individual_id(),
    parent_ids :: [individual_id()],
    generation :: generation(),
    species_id :: species_id(),
    population_id :: population_id(),
    genome_hash :: binary(),
    initial_fitness :: fitness() | undefined
}).

-record(pioneer_spawned, {
    meta :: #event_meta{},
    individual_id :: individual_id(),
    population_id :: population_id(),
    species_id :: species_id(),
    generation :: generation(),
    genome_hash :: binary()
}).

-record(clone_produced, {
    meta :: #event_meta{},
    individual_id :: individual_id(),
    parent_id :: individual_id(),
    generation :: generation(),
    species_id :: species_id(),
    population_id :: population_id(),
    mutation_applied :: boolean()
}).

-record(immigrant_arrived, {
    meta :: #event_meta{},
    individual_id :: individual_id(),
    source_population_id :: population_id(),
    target_population_id :: population_id(),
    source_species_id :: species_id(),
    target_species_id :: species_id(),
    fitness :: fitness()
}).

%%% ============================================================================
%%% Death Events
%%% ============================================================================

-record(individual_culled, {
    meta :: #event_meta{},
    individual_id :: individual_id(),
    population_id :: population_id(),
    species_id :: species_id(),
    final_fitness :: fitness(),
    age_generations :: non_neg_integer(),
    cause :: selection | stagnation | overcrowding,
    lineage_depth :: non_neg_integer()
}).

-record(lifespan_expired, {
    meta :: #event_meta{},
    individual_id :: individual_id(),
    population_id :: population_id(),
    age_generations :: non_neg_integer(),
    lifecycle_stage :: senescent,
    peak_fitness :: fitness(),
    offspring_count :: non_neg_integer()
}).

-record(individual_perished, {
    meta :: #event_meta{},
    individual_id :: individual_id(),
    population_id :: population_id(),
    cause :: evaluation_crash | simulation_death | timeout,
    generation :: generation()
}).

%%% ============================================================================
%%% Mutation Events
%%% ============================================================================

-record(mutation_applied, {
    meta :: #event_meta{},
    individual_id :: individual_id(),
    mutation_id :: binary(),
    mutation_type :: atom(),
    generation :: generation()
}).

-record(neuron_added, {
    meta :: #event_meta{},
    individual_id :: individual_id(),
    neuron_id :: binary(),
    layer :: input | hidden | output,
    layer_index :: float(),
    activation_function :: atom(),
    bias :: float(),
    mutation_id :: binary()
}).

-record(neuron_removed, {
    meta :: #event_meta{},
    individual_id :: individual_id(),
    neuron_id :: binary(),
    layer :: input | hidden | output,
    mutation_id :: binary()
}).

-record(connection_added, {
    meta :: #event_meta{},
    individual_id :: individual_id(),
    connection_id :: binary(),
    from_neuron_id :: binary(),
    to_neuron_id :: binary(),
    weight :: float(),
    mutation_id :: binary()
}).

-record(connection_removed, {
    meta :: #event_meta{},
    individual_id :: individual_id(),
    connection_id :: binary(),
    from_neuron_id :: binary(),
    to_neuron_id :: binary(),
    mutation_id :: binary()
}).

-record(weight_perturbed, {
    meta :: #event_meta{},
    individual_id :: individual_id(),
    connection_id :: binary(),
    old_weight :: float(),
    new_weight :: float(),
    mutation_id :: binary()
}).

-record(bias_perturbed, {
    meta :: #event_meta{},
    individual_id :: individual_id(),
    neuron_id :: binary(),
    old_bias :: float(),
    new_bias :: float(),
    mutation_id :: binary()
}).

-record(activation_changed, {
    meta :: #event_meta{},
    individual_id :: individual_id(),
    neuron_id :: binary(),
    old_activation :: atom(),
    new_activation :: atom(),
    mutation_id :: binary()
}).

%%% ============================================================================
%%% Fitness & Evaluation Events
%%% ============================================================================

-record(fitness_evaluated, {
    meta :: #event_meta{},
    individual_id :: individual_id(),
    population_id :: population_id(),
    fitness :: fitness(),
    fitness_components :: #{atom() => float()},
    evaluation_duration_ms :: non_neg_integer(),
    generation :: generation()
}).

-record(fitness_improved, {
    meta :: #event_meta{},
    individual_id :: individual_id(),
    population_id :: population_id(),
    previous_fitness :: fitness(),
    new_fitness :: fitness(),
    improvement :: float(),
    generation :: generation()
}).

-record(fitness_declined, {
    meta :: #event_meta{},
    individual_id :: individual_id(),
    population_id :: population_id(),
    previous_fitness :: fitness(),
    new_fitness :: fitness(),
    decline :: float(),
    generation :: generation()
}).

-record(champion_crowned, {
    meta :: #event_meta{},
    individual_id :: individual_id(),
    population_id :: population_id(),
    fitness :: fitness(),
    previous_champion_id :: individual_id() | undefined,
    previous_champion_fitness :: fitness() | undefined,
    generation :: generation()
}).

-record(evaluation_timeout, {
    meta :: #event_meta{},
    individual_id :: individual_id(),
    population_id :: population_id(),
    timeout_ms :: non_neg_integer(),
    generation :: generation()
}).

-record(stagnation_detected, {
    meta :: #event_meta{},
    population_id :: population_id(),
    generations_stagnant :: non_neg_integer(),
    best_fitness :: fitness(),
    generation :: generation()
}).

-record(breakthrough_achieved, {
    meta :: #event_meta{},
    individual_id :: individual_id(),
    population_id :: population_id(),
    previous_best :: fitness(),
    new_best :: fitness(),
    improvement_ratio :: float(),
    generation :: generation()
}).

%%% ============================================================================
%%% Generation & Population Events
%%% ============================================================================

-record(generation_completed, {
    meta :: #event_meta{},
    population_id :: population_id(),
    generation :: generation(),
    statistics :: #{
        population_size => non_neg_integer(),
        species_count => non_neg_integer(),
        mean_fitness => float(),
        max_fitness => float(),
        min_fitness => float(),
        std_fitness => float(),
        mean_complexity => float(),
        diversity_index => float()
    },
    champion_id :: individual_id(),
    duration_ms :: non_neg_integer()
}).

-record(population_initialized, {
    meta :: #event_meta{},
    population_id :: population_id(),
    initial_size :: non_neg_integer(),
    config :: map()
}).

-record(population_terminated, {
    meta :: #event_meta{},
    population_id :: population_id(),
    final_generation :: generation(),
    reason :: target_reached | max_generations | manual | error,
    final_champion_id :: individual_id(),
    final_champion_fitness :: fitness()
}).

%%% ============================================================================
%%% Species & Lineage Events
%%% ============================================================================

-record(lineage_diverged, {
    meta :: #event_meta{},
    new_species_id :: species_id(),
    parent_species_id :: species_id(),
    founder_individual_id :: individual_id(),
    divergence_cause :: structural | behavioral | geographic,
    compatibility_distance :: float(),
    generation :: generation()
}).

-record(species_emerged, {
    meta :: #event_meta{},
    species_id :: species_id(),
    population_id :: population_id(),
    founder_id :: individual_id(),
    generation :: generation()
}).

-record(lineage_ended, {
    meta :: #event_meta{},
    species_id :: species_id(),
    population_id :: population_id(),
    final_generation :: generation(),
    total_generations :: non_neg_integer(),
    peak_fitness :: fitness(),
    total_individuals :: non_neg_integer()
}).

%%% ============================================================================
%%% Breeding Events
%%% ============================================================================

-record(mating_occurred, {
    meta :: #event_meta{},
    parent_a_id :: individual_id(),
    parent_b_id :: individual_id(),
    offspring_ids :: [individual_id()],
    crossover_method :: atom(),
    compatibility_distance :: float(),
    relatedness_coefficient :: float()
}).

-record(crossover_performed, {
    meta :: #event_meta{},
    parent_a_id :: individual_id(),
    parent_b_id :: individual_id(),
    offspring_id :: individual_id(),
    crossover_point :: non_neg_integer() | multiple,
    genes_from_a :: non_neg_integer(),
    genes_from_b :: non_neg_integer()
}).

%%% ============================================================================
%%% Lifecycle Events
%%% ============================================================================

-record(individual_matured, {
    meta :: #event_meta{},
    individual_id :: individual_id(),
    population_id :: population_id(),
    previous_stage :: juvenile,
    new_stage :: fertile,
    age_generations :: non_neg_integer(),
    fitness_at_maturity :: fitness()
}).

-record(fertility_waned, {
    meta :: #event_meta{},
    individual_id :: individual_id(),
    population_id :: population_id(),
    previous_stage :: fertile,
    new_stage :: senescent,
    age_generations :: non_neg_integer(),
    offspring_count :: non_neg_integer()
}).

%%% ============================================================================
%%% Meta-Controller Events
%%% ============================================================================

-record(controller_adjusted, {
    meta :: #event_meta{},
    controller_id :: binary(),
    population_id :: population_id(),
    adjustments :: #{atom() => float()},
    trigger :: stagnation | diversity_crisis | breakthrough | scheduled
}).

-record(silo_activated, {
    meta :: #event_meta{},
    silo_type :: atom(),
    realm :: binary(),
    config :: map()
}).

-record(silo_deactivated, {
    meta :: #event_meta{},
    silo_type :: atom(),
    realm :: binary(),
    reason :: manual | dependency_removed | error
}).

%%% ============================================================================
%%% Event Type Exports
%%% ============================================================================

-type birth_event() :: #offspring_born{} | #pioneer_spawned{} | #clone_produced{} | #immigrant_arrived{}.
-type death_event() :: #individual_culled{} | #lifespan_expired{} | #individual_perished{}.
-type mutation_event() :: #mutation_applied{} | #neuron_added{} | #neuron_removed{} |
                          #connection_added{} | #connection_removed{} |
                          #weight_perturbed{} | #bias_perturbed{} | #activation_changed{}.
-type fitness_event() :: #fitness_evaluated{} | #fitness_improved{} | #fitness_declined{} |
                         #champion_crowned{} | #evaluation_timeout{} |
                         #stagnation_detected{} | #breakthrough_achieved{}.
-type generation_event() :: #generation_completed{} | #population_initialized{} | #population_terminated{}.
-type species_event() :: #lineage_diverged{} | #species_emerged{} | #lineage_ended{}.
-type breeding_event() :: #mating_occurred{} | #crossover_performed{}.
-type lifecycle_event() :: #individual_matured{} | #fertility_waned{}.
-type controller_event() :: #controller_adjusted{} | #silo_activated{} | #silo_deactivated{}.

-type behavioral_event() :: birth_event() | death_event() | mutation_event() |
                            fitness_event() | generation_event() | species_event() |
                            breeding_event() | lifecycle_event() | controller_event().

-endif. %% NEUROEVOLUTION_BEHAVIORAL_EVENTS_HRL
