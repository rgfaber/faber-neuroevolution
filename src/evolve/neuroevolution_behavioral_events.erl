%%%-------------------------------------------------------------------
%%% @doc Behavioral event constructors for neuroevolution.
%%%
%%% This module provides helper functions to construct properly-formed
%%% behavioral events. All events include standard metadata and follow
%%% the behavioral naming conventions (past tense, domain language).
%%%
%%% == Usage ==
%%%
%%% ```
%%% Event = neuroevolution_behavioral_events:offspring_born(#{
%%%     individual_id => IndId,
%%%     parent_ids => [P1, P2],
%%%     generation => 42,
%%%     species_id => SpeciesId,
%%%     population_id => PopId
%%% }),
%%% neuroevolution_lineage_events:persist_event(Event, State).
%%% '''
%%%
%%% @end
%%%-------------------------------------------------------------------
-module(neuroevolution_behavioral_events).

-include("neuroevolution_behavioral_events.hrl").

%% Event constructors - Birth
-export([
    offspring_born/1,
    pioneer_spawned/1,
    clone_produced/1,
    immigrant_arrived/1
]).

%% Event constructors - Death
-export([
    individual_culled/1,
    lifespan_expired/1,
    individual_perished/1
]).

%% Event constructors - Mutation
-export([
    mutation_applied/1,
    neuron_added/1,
    neuron_removed/1,
    connection_added/1,
    connection_removed/1,
    weight_perturbed/1,
    bias_perturbed/1,
    activation_changed/1
]).

%% Event constructors - Fitness
-export([
    fitness_evaluated/1,
    fitness_improved/1,
    fitness_declined/1,
    champion_crowned/1,
    evaluation_timeout/1,
    stagnation_detected/1,
    breakthrough_achieved/1
]).

%% Event constructors - Generation
-export([
    generation_completed/1,
    population_initialized/1,
    population_terminated/1
]).

%% Event constructors - Species
-export([
    lineage_diverged/1,
    species_emerged/1,
    lineage_ended/1
]).

%% Event constructors - Breeding
-export([
    mating_occurred/1,
    crossover_performed/1
]).

%% Event constructors - Lifecycle
-export([
    individual_matured/1,
    fertility_waned/1
]).

%% Event constructors - Controller
-export([
    controller_adjusted/1,
    silo_activated/1,
    silo_deactivated/1
]).

%% Utility exports
-export([
    make_meta/1,
    make_meta/2,
    event_to_map/1,
    stream_id/2
]).

%%% ============================================================================
%%% Metadata Construction
%%% ============================================================================

%% @doc Create event metadata with default values.
-spec make_meta(Emitter :: atom()) -> #event_meta{}.
make_meta(Emitter) ->
    make_meta(Emitter, #{}).

%% @doc Create event metadata with options.
-spec make_meta(Emitter :: atom(), Opts :: map()) -> #event_meta{}.
make_meta(Emitter, Opts) ->
    #event_meta{
        event_id = maps:get(event_id, Opts, generate_id()),
        correlation_id = maps:get(correlation_id, Opts, undefined),
        causation_id = maps:get(causation_id, Opts, undefined),
        timestamp = maps:get(timestamp, Opts, erlang:system_time(microsecond)),
        version = maps:get(version, Opts, 1),
        emitter = Emitter
    }.

%%% ============================================================================
%%% Birth Event Constructors
%%% ============================================================================

%% @doc Create offspring_born event.
-spec offspring_born(map()) -> #offspring_born{}.
offspring_born(Data) ->
    #offspring_born{
        meta = make_meta(maps:get(emitter, Data, population_server), Data),
        individual_id = maps:get(individual_id, Data),
        parent_ids = maps:get(parent_ids, Data),
        generation = maps:get(generation, Data),
        species_id = maps:get(species_id, Data),
        population_id = maps:get(population_id, Data),
        genome_hash = maps:get(genome_hash, Data, <<>>),
        initial_fitness = maps:get(initial_fitness, Data, undefined)
    }.

%% @doc Create pioneer_spawned event.
-spec pioneer_spawned(map()) -> #pioneer_spawned{}.
pioneer_spawned(Data) ->
    #pioneer_spawned{
        meta = make_meta(maps:get(emitter, Data, population_server), Data),
        individual_id = maps:get(individual_id, Data),
        population_id = maps:get(population_id, Data),
        species_id = maps:get(species_id, Data),
        generation = maps:get(generation, Data, 0),
        genome_hash = maps:get(genome_hash, Data, <<>>)
    }.

%% @doc Create clone_produced event.
-spec clone_produced(map()) -> #clone_produced{}.
clone_produced(Data) ->
    #clone_produced{
        meta = make_meta(maps:get(emitter, Data, population_server), Data),
        individual_id = maps:get(individual_id, Data),
        parent_id = maps:get(parent_id, Data),
        generation = maps:get(generation, Data),
        species_id = maps:get(species_id, Data),
        population_id = maps:get(population_id, Data),
        mutation_applied = maps:get(mutation_applied, Data, false)
    }.

%% @doc Create immigrant_arrived event.
-spec immigrant_arrived(map()) -> #immigrant_arrived{}.
immigrant_arrived(Data) ->
    #immigrant_arrived{
        meta = make_meta(maps:get(emitter, Data, migration_server), Data),
        individual_id = maps:get(individual_id, Data),
        source_population_id = maps:get(source_population_id, Data),
        target_population_id = maps:get(target_population_id, Data),
        source_species_id = maps:get(source_species_id, Data),
        target_species_id = maps:get(target_species_id, Data),
        fitness = maps:get(fitness, Data)
    }.

%%% ============================================================================
%%% Death Event Constructors
%%% ============================================================================

%% @doc Create individual_culled event.
-spec individual_culled(map()) -> #individual_culled{}.
individual_culled(Data) ->
    #individual_culled{
        meta = make_meta(maps:get(emitter, Data, selection_server), Data),
        individual_id = maps:get(individual_id, Data),
        population_id = maps:get(population_id, Data),
        species_id = maps:get(species_id, Data),
        final_fitness = maps:get(final_fitness, Data),
        age_generations = maps:get(age_generations, Data, 0),
        cause = maps:get(cause, Data, selection),
        lineage_depth = maps:get(lineage_depth, Data, 0)
    }.

%% @doc Create lifespan_expired event.
-spec lifespan_expired(map()) -> #lifespan_expired{}.
lifespan_expired(Data) ->
    #lifespan_expired{
        meta = make_meta(maps:get(emitter, Data, lifecycle_server), Data),
        individual_id = maps:get(individual_id, Data),
        population_id = maps:get(population_id, Data),
        age_generations = maps:get(age_generations, Data),
        lifecycle_stage = senescent,
        peak_fitness = maps:get(peak_fitness, Data),
        offspring_count = maps:get(offspring_count, Data, 0)
    }.

%% @doc Create individual_perished event.
-spec individual_perished(map()) -> #individual_perished{}.
individual_perished(Data) ->
    #individual_perished{
        meta = make_meta(maps:get(emitter, Data, evaluation_server), Data),
        individual_id = maps:get(individual_id, Data),
        population_id = maps:get(population_id, Data),
        cause = maps:get(cause, Data),
        generation = maps:get(generation, Data)
    }.

%%% ============================================================================
%%% Mutation Event Constructors
%%% ============================================================================

%% @doc Create mutation_applied event.
-spec mutation_applied(map()) -> #mutation_applied{}.
mutation_applied(Data) ->
    #mutation_applied{
        meta = make_meta(maps:get(emitter, Data, genome_mutator), Data),
        individual_id = maps:get(individual_id, Data),
        mutation_id = maps:get(mutation_id, Data, generate_id()),
        mutation_type = maps:get(mutation_type, Data),
        generation = maps:get(generation, Data)
    }.

%% @doc Create neuron_added event.
-spec neuron_added(map()) -> #neuron_added{}.
neuron_added(Data) ->
    #neuron_added{
        meta = make_meta(maps:get(emitter, Data, topological_mutations), Data),
        individual_id = maps:get(individual_id, Data),
        neuron_id = maps:get(neuron_id, Data),
        layer = maps:get(layer, Data),
        layer_index = maps:get(layer_index, Data, 0.5),
        activation_function = maps:get(activation_function, Data, tanh),
        bias = maps:get(bias, Data, 0.0),
        mutation_id = maps:get(mutation_id, Data, generate_id())
    }.

%% @doc Create neuron_removed event.
-spec neuron_removed(map()) -> #neuron_removed{}.
neuron_removed(Data) ->
    #neuron_removed{
        meta = make_meta(maps:get(emitter, Data, topological_mutations), Data),
        individual_id = maps:get(individual_id, Data),
        neuron_id = maps:get(neuron_id, Data),
        layer = maps:get(layer, Data),
        mutation_id = maps:get(mutation_id, Data, generate_id())
    }.

%% @doc Create connection_added event.
-spec connection_added(map()) -> #connection_added{}.
connection_added(Data) ->
    #connection_added{
        meta = make_meta(maps:get(emitter, Data, topological_mutations), Data),
        individual_id = maps:get(individual_id, Data),
        connection_id = maps:get(connection_id, Data),
        from_neuron_id = maps:get(from_neuron_id, Data),
        to_neuron_id = maps:get(to_neuron_id, Data),
        weight = maps:get(weight, Data),
        mutation_id = maps:get(mutation_id, Data, generate_id())
    }.

%% @doc Create connection_removed event.
-spec connection_removed(map()) -> #connection_removed{}.
connection_removed(Data) ->
    #connection_removed{
        meta = make_meta(maps:get(emitter, Data, topological_mutations), Data),
        individual_id = maps:get(individual_id, Data),
        connection_id = maps:get(connection_id, Data),
        from_neuron_id = maps:get(from_neuron_id, Data),
        to_neuron_id = maps:get(to_neuron_id, Data),
        mutation_id = maps:get(mutation_id, Data, generate_id())
    }.

%% @doc Create weight_perturbed event.
-spec weight_perturbed(map()) -> #weight_perturbed{}.
weight_perturbed(Data) ->
    #weight_perturbed{
        meta = make_meta(maps:get(emitter, Data, weight_mutations), Data),
        individual_id = maps:get(individual_id, Data),
        connection_id = maps:get(connection_id, Data),
        old_weight = maps:get(old_weight, Data),
        new_weight = maps:get(new_weight, Data),
        mutation_id = maps:get(mutation_id, Data, generate_id())
    }.

%% @doc Create bias_perturbed event.
-spec bias_perturbed(map()) -> #bias_perturbed{}.
bias_perturbed(Data) ->
    #bias_perturbed{
        meta = make_meta(maps:get(emitter, Data, weight_mutations), Data),
        individual_id = maps:get(individual_id, Data),
        neuron_id = maps:get(neuron_id, Data),
        old_bias = maps:get(old_bias, Data),
        new_bias = maps:get(new_bias, Data),
        mutation_id = maps:get(mutation_id, Data, generate_id())
    }.

%% @doc Create activation_changed event.
-spec activation_changed(map()) -> #activation_changed{}.
activation_changed(Data) ->
    #activation_changed{
        meta = make_meta(maps:get(emitter, Data, topological_mutations), Data),
        individual_id = maps:get(individual_id, Data),
        neuron_id = maps:get(neuron_id, Data),
        old_activation = maps:get(old_activation, Data),
        new_activation = maps:get(new_activation, Data),
        mutation_id = maps:get(mutation_id, Data, generate_id())
    }.

%%% ============================================================================
%%% Fitness Event Constructors
%%% ============================================================================

%% @doc Create fitness_evaluated event.
-spec fitness_evaluated(map()) -> #fitness_evaluated{}.
fitness_evaluated(Data) ->
    #fitness_evaluated{
        meta = make_meta(maps:get(emitter, Data, evaluation_server), Data),
        individual_id = maps:get(individual_id, Data),
        population_id = maps:get(population_id, Data),
        fitness = maps:get(fitness, Data),
        fitness_components = maps:get(fitness_components, Data, #{}),
        evaluation_duration_ms = maps:get(evaluation_duration_ms, Data, 0),
        generation = maps:get(generation, Data)
    }.

%% @doc Create fitness_improved event.
-spec fitness_improved(map()) -> #fitness_improved{}.
fitness_improved(Data) ->
    #fitness_improved{
        meta = make_meta(maps:get(emitter, Data, evaluation_server), Data),
        individual_id = maps:get(individual_id, Data),
        population_id = maps:get(population_id, Data),
        previous_fitness = maps:get(previous_fitness, Data),
        new_fitness = maps:get(new_fitness, Data),
        improvement = maps:get(improvement, Data),
        generation = maps:get(generation, Data)
    }.

%% @doc Create fitness_declined event.
-spec fitness_declined(map()) -> #fitness_declined{}.
fitness_declined(Data) ->
    #fitness_declined{
        meta = make_meta(maps:get(emitter, Data, evaluation_server), Data),
        individual_id = maps:get(individual_id, Data),
        population_id = maps:get(population_id, Data),
        previous_fitness = maps:get(previous_fitness, Data),
        new_fitness = maps:get(new_fitness, Data),
        decline = maps:get(decline, Data),
        generation = maps:get(generation, Data)
    }.

%% @doc Create champion_crowned event.
-spec champion_crowned(map()) -> #champion_crowned{}.
champion_crowned(Data) ->
    #champion_crowned{
        meta = make_meta(maps:get(emitter, Data, population_server), Data),
        individual_id = maps:get(individual_id, Data),
        population_id = maps:get(population_id, Data),
        fitness = maps:get(fitness, Data),
        previous_champion_id = maps:get(previous_champion_id, Data, undefined),
        previous_champion_fitness = maps:get(previous_champion_fitness, Data, undefined),
        generation = maps:get(generation, Data)
    }.

%% @doc Create evaluation_timeout event.
-spec evaluation_timeout(map()) -> #evaluation_timeout{}.
evaluation_timeout(Data) ->
    #evaluation_timeout{
        meta = make_meta(maps:get(emitter, Data, evaluation_server), Data),
        individual_id = maps:get(individual_id, Data),
        population_id = maps:get(population_id, Data),
        timeout_ms = maps:get(timeout_ms, Data),
        generation = maps:get(generation, Data)
    }.

%% @doc Create stagnation_detected event.
-spec stagnation_detected(map()) -> #stagnation_detected{}.
stagnation_detected(Data) ->
    #stagnation_detected{
        meta = make_meta(maps:get(emitter, Data, population_server), Data),
        population_id = maps:get(population_id, Data),
        generations_stagnant = maps:get(generations_stagnant, Data),
        best_fitness = maps:get(best_fitness, Data),
        generation = maps:get(generation, Data)
    }.

%% @doc Create breakthrough_achieved event.
-spec breakthrough_achieved(map()) -> #breakthrough_achieved{}.
breakthrough_achieved(Data) ->
    #breakthrough_achieved{
        meta = make_meta(maps:get(emitter, Data, population_server), Data),
        individual_id = maps:get(individual_id, Data),
        population_id = maps:get(population_id, Data),
        previous_best = maps:get(previous_best, Data),
        new_best = maps:get(new_best, Data),
        improvement_ratio = maps:get(improvement_ratio, Data),
        generation = maps:get(generation, Data)
    }.

%%% ============================================================================
%%% Generation Event Constructors
%%% ============================================================================

%% @doc Create generation_completed event.
-spec generation_completed(map()) -> #generation_completed{}.
generation_completed(Data) ->
    #generation_completed{
        meta = make_meta(maps:get(emitter, Data, population_server), Data),
        population_id = maps:get(population_id, Data),
        generation = maps:get(generation, Data),
        statistics = maps:get(statistics, Data, #{}),
        champion_id = maps:get(champion_id, Data),
        duration_ms = maps:get(duration_ms, Data, 0)
    }.

%% @doc Create population_initialized event.
-spec population_initialized(map()) -> #population_initialized{}.
population_initialized(Data) ->
    #population_initialized{
        meta = make_meta(maps:get(emitter, Data, population_server), Data),
        population_id = maps:get(population_id, Data),
        initial_size = maps:get(initial_size, Data),
        config = maps:get(config, Data, #{})
    }.

%% @doc Create population_terminated event.
-spec population_terminated(map()) -> #population_terminated{}.
population_terminated(Data) ->
    #population_terminated{
        meta = make_meta(maps:get(emitter, Data, population_server), Data),
        population_id = maps:get(population_id, Data),
        final_generation = maps:get(final_generation, Data),
        reason = maps:get(reason, Data),
        final_champion_id = maps:get(final_champion_id, Data),
        final_champion_fitness = maps:get(final_champion_fitness, Data)
    }.

%%% ============================================================================
%%% Species Event Constructors
%%% ============================================================================

%% @doc Create lineage_diverged event.
-spec lineage_diverged(map()) -> #lineage_diverged{}.
lineage_diverged(Data) ->
    #lineage_diverged{
        meta = make_meta(maps:get(emitter, Data, speciation_server), Data),
        new_species_id = maps:get(new_species_id, Data),
        parent_species_id = maps:get(parent_species_id, Data),
        founder_individual_id = maps:get(founder_individual_id, Data),
        divergence_cause = maps:get(divergence_cause, Data, structural),
        compatibility_distance = maps:get(compatibility_distance, Data),
        generation = maps:get(generation, Data)
    }.

%% @doc Create species_emerged event.
-spec species_emerged(map()) -> #species_emerged{}.
species_emerged(Data) ->
    #species_emerged{
        meta = make_meta(maps:get(emitter, Data, speciation_server), Data),
        species_id = maps:get(species_id, Data),
        population_id = maps:get(population_id, Data),
        founder_id = maps:get(founder_id, Data),
        generation = maps:get(generation, Data)
    }.

%% @doc Create lineage_ended event.
-spec lineage_ended(map()) -> #lineage_ended{}.
lineage_ended(Data) ->
    #lineage_ended{
        meta = make_meta(maps:get(emitter, Data, speciation_server), Data),
        species_id = maps:get(species_id, Data),
        population_id = maps:get(population_id, Data),
        final_generation = maps:get(final_generation, Data),
        total_generations = maps:get(total_generations, Data),
        peak_fitness = maps:get(peak_fitness, Data),
        total_individuals = maps:get(total_individuals, Data)
    }.

%%% ============================================================================
%%% Breeding Event Constructors
%%% ============================================================================

%% @doc Create mating_occurred event.
-spec mating_occurred(map()) -> #mating_occurred{}.
mating_occurred(Data) ->
    #mating_occurred{
        meta = make_meta(maps:get(emitter, Data, breeding_server), Data),
        parent_a_id = maps:get(parent_a_id, Data),
        parent_b_id = maps:get(parent_b_id, Data),
        offspring_ids = maps:get(offspring_ids, Data),
        crossover_method = maps:get(crossover_method, Data),
        compatibility_distance = maps:get(compatibility_distance, Data),
        relatedness_coefficient = maps:get(relatedness_coefficient, Data, 0.0)
    }.

%% @doc Create crossover_performed event.
-spec crossover_performed(map()) -> #crossover_performed{}.
crossover_performed(Data) ->
    #crossover_performed{
        meta = make_meta(maps:get(emitter, Data, crossover_server), Data),
        parent_a_id = maps:get(parent_a_id, Data),
        parent_b_id = maps:get(parent_b_id, Data),
        offspring_id = maps:get(offspring_id, Data),
        crossover_point = maps:get(crossover_point, Data),
        genes_from_a = maps:get(genes_from_a, Data),
        genes_from_b = maps:get(genes_from_b, Data)
    }.

%%% ============================================================================
%%% Lifecycle Event Constructors
%%% ============================================================================

%% @doc Create individual_matured event.
-spec individual_matured(map()) -> #individual_matured{}.
individual_matured(Data) ->
    #individual_matured{
        meta = make_meta(maps:get(emitter, Data, lifecycle_server), Data),
        individual_id = maps:get(individual_id, Data),
        population_id = maps:get(population_id, Data),
        previous_stage = juvenile,
        new_stage = fertile,
        age_generations = maps:get(age_generations, Data),
        fitness_at_maturity = maps:get(fitness_at_maturity, Data)
    }.

%% @doc Create fertility_waned event.
-spec fertility_waned(map()) -> #fertility_waned{}.
fertility_waned(Data) ->
    #fertility_waned{
        meta = make_meta(maps:get(emitter, Data, lifecycle_server), Data),
        individual_id = maps:get(individual_id, Data),
        population_id = maps:get(population_id, Data),
        previous_stage = fertile,
        new_stage = senescent,
        age_generations = maps:get(age_generations, Data),
        offspring_count = maps:get(offspring_count, Data, 0)
    }.

%%% ============================================================================
%%% Controller Event Constructors
%%% ============================================================================

%% @doc Create controller_adjusted event.
-spec controller_adjusted(map()) -> #controller_adjusted{}.
controller_adjusted(Data) ->
    #controller_adjusted{
        meta = make_meta(maps:get(emitter, Data, lc_controller), Data),
        controller_id = maps:get(controller_id, Data),
        population_id = maps:get(population_id, Data),
        adjustments = maps:get(adjustments, Data),
        trigger = maps:get(trigger, Data)
    }.

%% @doc Create silo_activated event.
-spec silo_activated(map()) -> #silo_activated{}.
silo_activated(Data) ->
    #silo_activated{
        meta = make_meta(maps:get(emitter, Data, lc_supervisor), Data),
        silo_type = maps:get(silo_type, Data),
        realm = maps:get(realm, Data),
        config = maps:get(config, Data, #{})
    }.

%% @doc Create silo_deactivated event.
-spec silo_deactivated(map()) -> #silo_deactivated{}.
silo_deactivated(Data) ->
    #silo_deactivated{
        meta = make_meta(maps:get(emitter, Data, lc_supervisor), Data),
        silo_type = maps:get(silo_type, Data),
        realm = maps:get(realm, Data),
        reason = maps:get(reason, Data, manual)
    }.

%%% ============================================================================
%%% Utility Functions
%%% ============================================================================

%% @doc Convert an event record to a map for serialization.
-spec event_to_map(behavioral_event()) -> map().
event_to_map(Event) when is_tuple(Event) ->
    [RecordName | Fields] = tuple_to_list(Event),
    FieldNames = record_fields(RecordName),
    MetaMap = meta_to_map(hd(Fields)),
    DataMap = maps:from_list(lists:zip(tl(FieldNames), tl(Fields))),
    maps:merge(#{event_type => RecordName, meta => MetaMap}, DataMap).

%% @doc Generate stream ID for an entity.
-spec stream_id(Type :: atom(), Id :: binary()) -> binary().
stream_id(individual, Id) -> <<"individual-", Id/binary>>;
stream_id(species, Id) -> <<"species-", Id/binary>>;
stream_id(population, Id) -> <<"population-", Id/binary>>;
stream_id(coalition, Id) -> <<"coalition-", Id/binary>>;
stream_id(lineage, Id) -> <<"lineage-", Id/binary>>.

%%% ============================================================================
%%% Internal Functions
%%% ============================================================================

generate_id() ->
    Bytes = crypto:strong_rand_bytes(16),
    binary:encode_hex(Bytes).

meta_to_map(#event_meta{} = Meta) ->
    #{
        event_id => Meta#event_meta.event_id,
        correlation_id => Meta#event_meta.correlation_id,
        causation_id => Meta#event_meta.causation_id,
        timestamp => Meta#event_meta.timestamp,
        version => Meta#event_meta.version,
        emitter => Meta#event_meta.emitter
    }.

record_fields(offspring_born) -> [meta, individual_id, parent_ids, generation, species_id, population_id, genome_hash, initial_fitness];
record_fields(pioneer_spawned) -> [meta, individual_id, population_id, species_id, generation, genome_hash];
record_fields(clone_produced) -> [meta, individual_id, parent_id, generation, species_id, population_id, mutation_applied];
record_fields(immigrant_arrived) -> [meta, individual_id, source_population_id, target_population_id, source_species_id, target_species_id, fitness];
record_fields(individual_culled) -> [meta, individual_id, population_id, species_id, final_fitness, age_generations, cause, lineage_depth];
record_fields(lifespan_expired) -> [meta, individual_id, population_id, age_generations, lifecycle_stage, peak_fitness, offspring_count];
record_fields(individual_perished) -> [meta, individual_id, population_id, cause, generation];
record_fields(mutation_applied) -> [meta, individual_id, mutation_id, mutation_type, generation];
record_fields(neuron_added) -> [meta, individual_id, neuron_id, layer, layer_index, activation_function, bias, mutation_id];
record_fields(neuron_removed) -> [meta, individual_id, neuron_id, layer, mutation_id];
record_fields(connection_added) -> [meta, individual_id, connection_id, from_neuron_id, to_neuron_id, weight, mutation_id];
record_fields(connection_removed) -> [meta, individual_id, connection_id, from_neuron_id, to_neuron_id, mutation_id];
record_fields(weight_perturbed) -> [meta, individual_id, connection_id, old_weight, new_weight, mutation_id];
record_fields(bias_perturbed) -> [meta, individual_id, neuron_id, old_bias, new_bias, mutation_id];
record_fields(activation_changed) -> [meta, individual_id, neuron_id, old_activation, new_activation, mutation_id];
record_fields(fitness_evaluated) -> [meta, individual_id, population_id, fitness, fitness_components, evaluation_duration_ms, generation];
record_fields(fitness_improved) -> [meta, individual_id, population_id, previous_fitness, new_fitness, improvement, generation];
record_fields(fitness_declined) -> [meta, individual_id, population_id, previous_fitness, new_fitness, decline, generation];
record_fields(champion_crowned) -> [meta, individual_id, population_id, fitness, previous_champion_id, previous_champion_fitness, generation];
record_fields(evaluation_timeout) -> [meta, individual_id, population_id, timeout_ms, generation];
record_fields(stagnation_detected) -> [meta, population_id, generations_stagnant, best_fitness, generation];
record_fields(breakthrough_achieved) -> [meta, individual_id, population_id, previous_best, new_best, improvement_ratio, generation];
record_fields(generation_completed) -> [meta, population_id, generation, statistics, champion_id, duration_ms];
record_fields(population_initialized) -> [meta, population_id, initial_size, config];
record_fields(population_terminated) -> [meta, population_id, final_generation, reason, final_champion_id, final_champion_fitness];
record_fields(lineage_diverged) -> [meta, new_species_id, parent_species_id, founder_individual_id, divergence_cause, compatibility_distance, generation];
record_fields(species_emerged) -> [meta, species_id, population_id, founder_id, generation];
record_fields(lineage_ended) -> [meta, species_id, population_id, final_generation, total_generations, peak_fitness, total_individuals];
record_fields(mating_occurred) -> [meta, parent_a_id, parent_b_id, offspring_ids, crossover_method, compatibility_distance, relatedness_coefficient];
record_fields(crossover_performed) -> [meta, parent_a_id, parent_b_id, offspring_id, crossover_point, genes_from_a, genes_from_b];
record_fields(individual_matured) -> [meta, individual_id, population_id, previous_stage, new_stage, age_generations, fitness_at_maturity];
record_fields(fertility_waned) -> [meta, individual_id, population_id, previous_stage, new_stage, age_generations, offspring_count];
record_fields(controller_adjusted) -> [meta, controller_id, population_id, adjustments, trigger];
record_fields(silo_activated) -> [meta, silo_type, realm, config];
record_fields(silo_deactivated) -> [meta, silo_type, realm, reason].
