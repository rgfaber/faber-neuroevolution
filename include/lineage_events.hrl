%% @doc Lineage event record definitions for genealogy tracking.
%%
%% These events capture evolutionary lineage information for persistent storage.
%% They complement lifecycle_events.hrl by focusing on ancestry, heredity,
%% and cross-generational analysis.
%%
%% == Stream Routing ==
%%
%% Events are routed to streams by entity type:
%% - `individual-{id}' - Birth, death, fitness, mutations, knowledge transfer
%% - `species-{id}' - Speciation, lineage divergence/merge
%% - `population-{id}' - Generation, capacity, catastrophe
%% - `coalition-{id}' - Coalition lifecycle
%%
%% == Domain Language ==
%%
%% Events use biological domain language (NOT CRUD):
%% - "offspring_born" (not "individual_created")
%% - "lineage_diverged" (not "species_created")
%% - "individual_culled" (not "individual_deleted")
%%
%% @author R.G. Lefever
%% @copyright 2024-2026 R.G. Lefever

-ifndef(LINEAGE_EVENTS_HRL).
-define(LINEAGE_EVENTS_HRL, true).

%%% ============================================================================
%%% Type Definitions
%%% ============================================================================

-type individual_id() :: binary().
-type species_id() :: binary().
-type population_id() :: binary().
-type coalition_id() :: binary().
-type causation_id() :: binary().
-type correlation_id() :: binary().
-type generation() :: non_neg_integer().
-type fitness() :: float().
-type age() :: non_neg_integer().

%% Age stages for lifecycle management
-type age_stage() :: juvenile | fertile | senescent.

%% Event metadata - included in all events
-type event_metadata() :: #{
    causation_id => causation_id(),
    correlation_id => correlation_id(),
    timestamp => integer(),
    source_node => atom()
}.

%%% ============================================================================
%%% Birth Events
%%% ============================================================================

%% @doc Sexual reproduction - two parents produce offspring via crossover.
-record(offspring_born, {
    individual_id :: individual_id(),
    parent_ids :: [individual_id()],  % Exactly 2 for sexual reproduction
    species_id :: species_id(),
    generation :: generation(),
    inherited_marks = [] :: [binary()],  % Epigenetic marks from parents
    crossover_point :: non_neg_integer() | undefined,
    metadata = #{} :: event_metadata()
}).

%% @doc Pioneer individual - created at population initialization.
-record(pioneer_spawned, {
    individual_id :: individual_id(),
    species_id :: species_id(),
    generation = 0 :: generation(),
    genome_template :: term(),
    metadata = #{} :: event_metadata()
}).

%% @doc Asexual reproduction - single parent produces identical copy.
-record(clone_produced, {
    individual_id :: individual_id(),
    parent_id :: individual_id(),
    species_id :: species_id(),
    generation :: generation(),
    mutation_applied = false :: boolean(),
    metadata = #{} :: event_metadata()
}).

%% @doc Individual migrated from another population/island.
-record(immigrant_arrived, {
    individual_id :: individual_id(),
    origin_population_id :: population_id(),
    origin_species_id :: species_id(),
    target_species_id :: species_id() | undefined,
    fitness_at_immigration :: fitness(),
    metadata = #{} :: event_metadata()
}).

%%% ============================================================================
%%% Death Events
%%% ============================================================================

%% @doc Individual eliminated through selection pressure.
-record(individual_culled, {
    individual_id :: individual_id(),
    species_id :: species_id(),
    generation :: generation(),
    final_fitness :: fitness(),
    culling_reason :: selection | stagnation | niche_competition,
    age_at_death :: age(),
    metadata = #{} :: event_metadata()
}).

%% @doc Individual died of old age (exceeded max lifespan).
-record(lifespan_expired, {
    individual_id :: individual_id(),
    species_id :: species_id(),
    final_age :: age(),
    max_lifespan :: age(),
    final_fitness :: fitness(),
    offspring_produced :: non_neg_integer(),
    metadata = #{} :: event_metadata()
}).

%% @doc General death event (starvation, disease, catastrophe).
-record(individual_perished, {
    individual_id :: individual_id(),
    species_id :: species_id(),
    cause :: starvation | disease | catastrophe | competition | other,
    age_at_death :: age(),
    final_fitness :: fitness(),
    metadata = #{} :: event_metadata()
}).

%%% ============================================================================
%%% Age-Based Lifecycle Events
%%% ============================================================================

%% @doc Individual reached reproductive maturity.
-record(individual_matured, {
    individual_id :: individual_id(),
    species_id :: species_id(),
    age_at_maturity :: age(),
    fitness_at_maturity :: fitness(),
    prev_stage = juvenile :: age_stage(),
    new_stage = fertile :: age_stage(),
    metadata = #{} :: event_metadata()
}).

%% @doc Individual's fertility has declined with age.
-record(fertility_waned, {
    individual_id :: individual_id(),
    species_id :: species_id(),
    age :: age(),
    prev_fertility :: float(),
    new_fertility :: float(),
    prev_stage :: age_stage(),
    new_stage :: age_stage(),
    metadata = #{} :: event_metadata()
}).

%%% ============================================================================
%%% Fitness Events
%%% ============================================================================

%% @doc Individual's fitness was evaluated.
-record(fitness_evaluated, {
    individual_id :: individual_id(),
    fitness :: fitness(),
    evaluation_count :: non_neg_integer(),
    metrics = #{} :: map(),
    evaluator_node :: atom() | undefined,
    evaluated_at :: integer(),
    metadata = #{} :: event_metadata()
}).

%% @doc Individual's fitness improved from previous evaluation.
-record(fitness_improved, {
    individual_id :: individual_id(),
    previous_fitness :: fitness(),
    new_fitness :: fitness(),
    improvement :: float(),  % Percentage or absolute delta
    metadata = #{} :: event_metadata()
}).

%% @doc Individual's fitness declined from previous evaluation.
-record(fitness_declined, {
    individual_id :: individual_id(),
    previous_fitness :: fitness(),
    new_fitness :: fitness(),
    decline :: float(),
    decline_reason :: environmental_change | competition | senescence | unknown,
    metadata = #{} :: event_metadata()
}).

%% @doc Individual became champion of its species or population.
-record(champion_crowned, {
    individual_id :: individual_id(),
    species_id :: species_id(),
    championship_scope :: species | population | global,
    fitness :: fitness(),
    previous_champion_id :: individual_id() | undefined,
    metadata = #{} :: event_metadata()
}).

%%% ============================================================================
%%% Mutation Events
%%% ============================================================================

%% @doc General mutation applied to individual.
-record(mutation_applied, {
    individual_id :: individual_id(),
    mutation_type :: atom(),  % weight, bias, connection, neuron, etc.
    mutation_details :: map(),
    fitness_before :: fitness() | undefined,
    fitness_after :: fitness() | undefined,
    metadata = #{} :: event_metadata()
}).

%% @doc Neuron added to network topology.
-record(neuron_added, {
    individual_id :: individual_id(),
    neuron_id :: term(),
    neuron_type :: hidden | output | ltc | standard,
    layer :: non_neg_integer() | undefined,
    split_connection :: {term(), term()} | undefined,  % {from, to}
    metadata = #{} :: event_metadata()
}).

%% @doc Neuron removed from network topology.
-record(neuron_removed, {
    individual_id :: individual_id(),
    neuron_id :: term(),
    neuron_type :: hidden | output | ltc | standard,
    connections_removed :: non_neg_integer(),
    metadata = #{} :: event_metadata()
}).

%% @doc Connection added between neurons.
-record(connection_added, {
    individual_id :: individual_id(),
    from_neuron :: term(),
    to_neuron :: term(),
    initial_weight :: float(),
    connection_type :: feedforward | recurrent | lateral,
    metadata = #{} :: event_metadata()
}).

%% @doc Connection removed between neurons.
-record(connection_removed, {
    individual_id :: individual_id(),
    from_neuron :: term(),
    to_neuron :: term(),
    final_weight :: float(),
    removal_reason :: pruning | structural_mutation | optimization,
    metadata = #{} :: event_metadata()
}).

%% @doc Connection weight perturbed.
-record(weight_perturbed, {
    individual_id :: individual_id(),
    from_neuron :: term(),
    to_neuron :: term(),
    old_weight :: float(),
    new_weight :: float(),
    delta :: float(),
    perturbation_type :: gaussian | uniform | cauchy | sign_flip,
    metadata = #{} :: event_metadata()
}).

%%% ============================================================================
%%% Species/Lineage Events
%%% ============================================================================

%% @doc A lineage has diverged - new species emerged from existing.
-record(lineage_diverged, {
    parent_species_id :: species_id(),
    new_species_id :: species_id(),
    founder_individual_id :: individual_id(),
    genetic_distance :: float(),
    generation :: generation(),
    metadata = #{} :: event_metadata()
}).

%% @doc New species emerged (no clear parent species).
-record(species_emerged, {
    species_id :: species_id(),
    founder_individual_id :: individual_id(),
    initial_size :: non_neg_integer(),
    representative_genome :: term() | undefined,
    metadata = #{} :: event_metadata()
}).

%% @doc Species lineage ended (extinction).
-record(lineage_ended, {
    species_id :: species_id(),
    extinction_reason :: stagnation | competition | catastrophe | merged,
    final_generation :: generation(),
    peak_size :: non_neg_integer(),
    peak_fitness :: fitness(),
    lifespan_generations :: non_neg_integer(),
    metadata = #{} :: event_metadata()
}).

%% @doc Two species lineages merged into one.
-record(lineage_merged, {
    absorbed_species_id :: species_id(),
    absorbing_species_id :: species_id(),
    merged_species_id :: species_id(),
    merge_generation :: generation(),
    combined_size :: non_neg_integer(),
    metadata = #{} :: event_metadata()
}).

%%% ============================================================================
%%% Knowledge Transfer Events
%%% ============================================================================

%% @doc Knowledge transferred from mentor to student.
-record(knowledge_transferred, {
    student_id :: individual_id(),
    mentor_id :: individual_id(),
    transfer_type :: behavioral_cloning | weight_grafting | structural_seeding,
    knowledge_domain :: binary() | undefined,
    transfer_fidelity :: float(),  % 0.0 to 1.0
    student_fitness_before :: fitness(),
    student_fitness_after :: fitness() | undefined,
    metadata = #{} :: event_metadata()
}).

%% @doc Skill imitated from another individual.
-record(skill_imitated, {
    imitator_id :: individual_id(),
    model_id :: individual_id(),
    skill_type :: binary(),
    imitation_fidelity :: float(),
    metadata = #{} :: event_metadata()
}).

%% @doc Full behavioral cloning from model.
-record(behavior_cloned, {
    student_id :: individual_id(),
    model_id :: individual_id(),
    behavior_samples :: non_neg_integer(),
    clone_accuracy :: float(),
    metadata = #{} :: event_metadata()
}).

%% @doc Neural weights grafted from donor.
-record(weights_grafted, {
    recipient_id :: individual_id(),
    donor_id :: individual_id(),
    layer_range :: {non_neg_integer(), non_neg_integer()},
    weights_transferred :: non_neg_integer(),
    compatibility_score :: float(),
    metadata = #{} :: event_metadata()
}).

%% @doc Network structure seeded from template.
-record(structure_seeded, {
    individual_id :: individual_id(),
    template_id :: individual_id() | binary(),
    neurons_seeded :: non_neg_integer(),
    connections_seeded :: non_neg_integer(),
    metadata = #{} :: event_metadata()
}).

%% @doc Mentorship relationship established.
-record(mentor_assigned, {
    student_id :: individual_id(),
    mentor_id :: individual_id(),
    mentorship_type :: passive | active | collaborative,
    expected_duration :: non_neg_integer() | undefined,
    metadata = #{} :: event_metadata()
}).

%% @doc Mentorship relationship ended.
-record(mentorship_concluded, {
    student_id :: individual_id(),
    mentor_id :: individual_id(),
    duration :: non_neg_integer(),
    success_score :: float(),
    conclusion_reason :: graduation | timeout | failure | mentor_death,
    metadata = #{} :: event_metadata()
}).

%%% ============================================================================
%%% Epigenetic Events
%%% ============================================================================

%% @doc Epigenetic mark acquired from environment.
-record(mark_acquired, {
    individual_id :: individual_id(),
    mark_type :: binary(),
    mark_source :: environment | stress | success | learning,
    mark_strength :: float(),  % 0.0 to 1.0
    affected_genes :: [binary()] | all,
    metadata = #{} :: event_metadata()
}).

%% @doc Epigenetic mark inherited from parent.
-record(mark_inherited, {
    individual_id :: individual_id(),
    parent_id :: individual_id(),
    mark_type :: binary(),
    inheritance_fidelity :: float(),
    mark_strength :: float(),
    metadata = #{} :: event_metadata()
}).

%% @doc Epigenetic mark decayed over time.
-record(mark_decayed, {
    individual_id :: individual_id(),
    mark_type :: binary(),
    old_strength :: float(),
    new_strength :: float(),
    decay_rate :: float(),
    metadata = #{} :: event_metadata()
}).

%%% ============================================================================
%%% Coalition Events
%%% ============================================================================

%% @doc Coalition formed by individuals.
-record(coalition_formed, {
    coalition_id :: coalition_id(),
    founder_ids :: [individual_id()],
    coalition_type :: breeding | hunting | defense | learning,
    formation_trigger :: kin_recognition | mutual_benefit | random,
    metadata = #{} :: event_metadata()
}).

%% @doc Coalition dissolved.
-record(coalition_dissolved, {
    coalition_id :: coalition_id(),
    member_ids :: [individual_id()],
    dissolution_reason :: achieved_goal | conflict | member_death | timeout,
    duration :: non_neg_integer(),
    metadata = #{} :: event_metadata()
}).

%% @doc Individual joined existing coalition.
-record(coalition_joined, {
    coalition_id :: coalition_id(),
    individual_id :: individual_id(),
    join_reason :: invitation | application | kin_recognition,
    metadata = #{} :: event_metadata()
}).

%%% ============================================================================
%%% Population Events
%%% ============================================================================

%% @doc Generation completed for population.
-record(generation_completed, {
    population_id :: population_id(),
    generation :: generation(),
    population_size :: non_neg_integer(),
    best_fitness :: fitness(),
    avg_fitness :: fitness(),
    species_count :: non_neg_integer(),
    stagnation_counter :: non_neg_integer(),
    metadata = #{} :: event_metadata()
}).

%% @doc Population initialized.
-record(population_initialized, {
    population_id :: population_id(),
    initial_size :: non_neg_integer(),
    config :: map(),
    seed :: integer() | undefined,
    metadata = #{} :: event_metadata()
}).

%% @doc Population terminated.
-record(population_terminated, {
    population_id :: population_id(),
    final_generation :: generation(),
    termination_reason :: goal_reached | max_generations | manual | error,
    final_best_fitness :: fitness(),
    total_evaluations :: non_neg_integer(),
    metadata = #{} :: event_metadata()
}).

%% @doc Stagnation detected in population.
-record(stagnation_detected, {
    population_id :: population_id(),
    stagnation_generations :: non_neg_integer(),
    current_best_fitness :: fitness(),
    species_affected :: [species_id()],
    intervention_triggered :: boolean(),
    metadata = #{} :: event_metadata()
}).

%% @doc Evolutionary breakthrough achieved.
-record(breakthrough_achieved, {
    population_id :: population_id(),
    breakthrough_individual_id :: individual_id(),
    previous_best_fitness :: fitness(),
    new_best_fitness :: fitness(),
    improvement_percentage :: float(),
    breakthrough_type :: fitness | capability | complexity,
    metadata = #{} :: event_metadata()
}).

%% @doc Carrying capacity reached.
-record(carrying_capacity_reached, {
    population_id :: population_id(),
    current_size :: non_neg_integer(),
    capacity :: non_neg_integer(),
    response_action :: culling | resource_competition | migration,
    metadata = #{} :: event_metadata()
}).

%% @doc Catastrophic event occurred.
-record(catastrophe_occurred, {
    population_id :: population_id(),
    catastrophe_type :: mass_extinction | environmental_shift | disease_outbreak,
    mortality_rate :: float(),
    survivors :: non_neg_integer(),
    affected_species :: [species_id()],
    metadata = #{} :: event_metadata()
}).

%%% ============================================================================
%%% Type Exports for Dialyzer
%%% ============================================================================

-type lineage_event() ::
    #offspring_born{} |
    #pioneer_spawned{} |
    #clone_produced{} |
    #immigrant_arrived{} |
    #individual_culled{} |
    #lifespan_expired{} |
    #individual_perished{} |
    #individual_matured{} |
    #fertility_waned{} |
    #fitness_evaluated{} |
    #fitness_improved{} |
    #fitness_declined{} |
    #champion_crowned{} |
    #mutation_applied{} |
    #neuron_added{} |
    #neuron_removed{} |
    #connection_added{} |
    #connection_removed{} |
    #weight_perturbed{} |
    #lineage_diverged{} |
    #species_emerged{} |
    #lineage_ended{} |
    #lineage_merged{} |
    #knowledge_transferred{} |
    #skill_imitated{} |
    #behavior_cloned{} |
    #weights_grafted{} |
    #structure_seeded{} |
    #mentor_assigned{} |
    #mentorship_concluded{} |
    #mark_acquired{} |
    #mark_inherited{} |
    #mark_decayed{} |
    #coalition_formed{} |
    #coalition_dissolved{} |
    #coalition_joined{} |
    #generation_completed{} |
    #population_initialized{} |
    #population_terminated{} |
    #stagnation_detected{} |
    #breakthrough_achieved{} |
    #carrying_capacity_reached{} |
    #catastrophe_occurred{}.

-endif.
