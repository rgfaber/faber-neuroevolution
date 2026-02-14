%%%-------------------------------------------------------------------
%%% @doc Unit tests for neuroevolution_behavioral_events.
%%% @end
%%%-------------------------------------------------------------------
-module(neuroevolution_behavioral_events_tests).

-include_lib("eunit/include/eunit.hrl").
-include("neuroevolution_behavioral_events.hrl").

%%% ============================================================================
%%% Metadata Tests
%%% ============================================================================

make_meta_test_() ->
    [
        {"make_meta creates valid metadata",
         fun() ->
             Meta = neuroevolution_behavioral_events:make_meta(test_emitter),
             ?assert(is_binary(Meta#event_meta.event_id)),
             ?assertEqual(test_emitter, Meta#event_meta.emitter),
             ?assertEqual(1, Meta#event_meta.version),
             ?assert(is_integer(Meta#event_meta.timestamp))
         end},
        {"make_meta with options uses provided values",
         fun() ->
             CorrelationId = <<"corr-123">>,
             Meta = neuroevolution_behavioral_events:make_meta(test_emitter, #{
                 correlation_id => CorrelationId
             }),
             ?assertEqual(CorrelationId, Meta#event_meta.correlation_id)
         end}
    ].

%%% ============================================================================
%%% Birth Event Tests
%%% ============================================================================

birth_events_test_() ->
    [
        {"offspring_born creates valid event",
         fun() ->
             Event = neuroevolution_behavioral_events:offspring_born(#{
                 individual_id => <<"ind-1">>,
                 parent_ids => [<<"p1">>, <<"p2">>],
                 generation => 10,
                 species_id => <<"species-1">>,
                 population_id => <<"pop-1">>
             }),
             ?assertEqual(<<"ind-1">>, Event#offspring_born.individual_id),
             ?assertEqual([<<"p1">>, <<"p2">>], Event#offspring_born.parent_ids),
             ?assertEqual(10, Event#offspring_born.generation),
             ?assertEqual(population_server, Event#offspring_born.meta#event_meta.emitter)
         end},
        {"pioneer_spawned creates valid event",
         fun() ->
             Event = neuroevolution_behavioral_events:pioneer_spawned(#{
                 individual_id => <<"pioneer-1">>,
                 population_id => <<"pop-1">>,
                 species_id => <<"species-1">>
             }),
             ?assertEqual(<<"pioneer-1">>, Event#pioneer_spawned.individual_id),
             ?assertEqual(0, Event#pioneer_spawned.generation)
         end},
        {"clone_produced creates valid event",
         fun() ->
             Event = neuroevolution_behavioral_events:clone_produced(#{
                 individual_id => <<"clone-1">>,
                 parent_id => <<"parent-1">>,
                 generation => 5,
                 species_id => <<"species-1">>,
                 population_id => <<"pop-1">>,
                 mutation_applied => true
             }),
             ?assertEqual(true, Event#clone_produced.mutation_applied)
         end},
        {"immigrant_arrived creates valid event",
         fun() ->
             Event = neuroevolution_behavioral_events:immigrant_arrived(#{
                 individual_id => <<"immigrant-1">>,
                 source_population_id => <<"pop-1">>,
                 target_population_id => <<"pop-2">>,
                 source_species_id => <<"species-1">>,
                 target_species_id => <<"species-2">>,
                 fitness => 0.85
             }),
             ?assertEqual(<<"pop-1">>, Event#immigrant_arrived.source_population_id),
             ?assertEqual(<<"pop-2">>, Event#immigrant_arrived.target_population_id)
         end}
    ].

%%% ============================================================================
%%% Death Event Tests
%%% ============================================================================

death_events_test_() ->
    [
        {"individual_culled creates valid event",
         fun() ->
             Event = neuroevolution_behavioral_events:individual_culled(#{
                 individual_id => <<"ind-1">>,
                 population_id => <<"pop-1">>,
                 species_id => <<"species-1">>,
                 final_fitness => 0.3,
                 cause => selection
             }),
             ?assertEqual(selection, Event#individual_culled.cause),
             ?assertEqual(0.3, Event#individual_culled.final_fitness)
         end},
        {"lifespan_expired creates valid event",
         fun() ->
             Event = neuroevolution_behavioral_events:lifespan_expired(#{
                 individual_id => <<"ind-1">>,
                 population_id => <<"pop-1">>,
                 age_generations => 100,
                 peak_fitness => 0.95
             }),
             ?assertEqual(senescent, Event#lifespan_expired.lifecycle_stage),
             ?assertEqual(100, Event#lifespan_expired.age_generations)
         end},
        {"individual_perished creates valid event",
         fun() ->
             Event = neuroevolution_behavioral_events:individual_perished(#{
                 individual_id => <<"ind-1">>,
                 population_id => <<"pop-1">>,
                 cause => timeout,
                 generation => 50
             }),
             ?assertEqual(timeout, Event#individual_perished.cause)
         end}
    ].

%%% ============================================================================
%%% Mutation Event Tests
%%% ============================================================================

mutation_events_test_() ->
    [
        {"neuron_added creates valid event",
         fun() ->
             Event = neuroevolution_behavioral_events:neuron_added(#{
                 individual_id => <<"ind-1">>,
                 neuron_id => <<"neuron-1">>,
                 layer => hidden,
                 activation_function => relu,
                 bias => 0.1
             }),
             ?assertEqual(hidden, Event#neuron_added.layer),
             ?assertEqual(relu, Event#neuron_added.activation_function)
         end},
        {"connection_added creates valid event",
         fun() ->
             Event = neuroevolution_behavioral_events:connection_added(#{
                 individual_id => <<"ind-1">>,
                 connection_id => <<"conn-1">>,
                 from_neuron_id => <<"n1">>,
                 to_neuron_id => <<"n2">>,
                 weight => 0.5
             }),
             ?assertEqual(0.5, Event#connection_added.weight)
         end},
        {"weight_perturbed creates valid event",
         fun() ->
             Event = neuroevolution_behavioral_events:weight_perturbed(#{
                 individual_id => <<"ind-1">>,
                 connection_id => <<"conn-1">>,
                 old_weight => 0.5,
                 new_weight => 0.6
             }),
             ?assertEqual(0.5, Event#weight_perturbed.old_weight),
             ?assertEqual(0.6, Event#weight_perturbed.new_weight)
         end}
    ].

%%% ============================================================================
%%% Fitness Event Tests
%%% ============================================================================

fitness_events_test_() ->
    [
        {"fitness_evaluated creates valid event",
         fun() ->
             Event = neuroevolution_behavioral_events:fitness_evaluated(#{
                 individual_id => <<"ind-1">>,
                 population_id => <<"pop-1">>,
                 fitness => 0.85,
                 generation => 42,
                 fitness_components => #{primary => 0.8, novelty => 0.05}
             }),
             ?assertEqual(0.85, Event#fitness_evaluated.fitness),
             ?assertEqual(42, Event#fitness_evaluated.generation)
         end},
        {"champion_crowned creates valid event",
         fun() ->
             Event = neuroevolution_behavioral_events:champion_crowned(#{
                 individual_id => <<"champ-1">>,
                 population_id => <<"pop-1">>,
                 fitness => 0.99,
                 generation => 100,
                 previous_champion_id => <<"old-champ">>
             }),
             ?assertEqual(<<"champ-1">>, Event#champion_crowned.individual_id),
             ?assertEqual(<<"old-champ">>, Event#champion_crowned.previous_champion_id)
         end},
        {"breakthrough_achieved creates valid event",
         fun() ->
             Event = neuroevolution_behavioral_events:breakthrough_achieved(#{
                 individual_id => <<"ind-1">>,
                 population_id => <<"pop-1">>,
                 previous_best => 0.5,
                 new_best => 0.9,
                 improvement_ratio => 0.8,
                 generation => 50
             }),
             ?assertEqual(0.8, Event#breakthrough_achieved.improvement_ratio)
         end}
    ].

%%% ============================================================================
%%% Generation Event Tests
%%% ============================================================================

generation_events_test_() ->
    [
        {"generation_completed creates valid event",
         fun() ->
             Event = neuroevolution_behavioral_events:generation_completed(#{
                 population_id => <<"pop-1">>,
                 generation => 100,
                 champion_id => <<"champ-1">>,
                 duration_ms => 5000,
                 statistics => #{
                     population_size => 100,
                     mean_fitness => 0.5,
                     max_fitness => 0.95
                 }
             }),
             ?assertEqual(100, Event#generation_completed.generation),
             Stats = Event#generation_completed.statistics,
             ?assertEqual(100, maps:get(population_size, Stats))
         end},
        {"population_initialized creates valid event",
         fun() ->
             Event = neuroevolution_behavioral_events:population_initialized(#{
                 population_id => <<"pop-1">>,
                 initial_size => 100
             }),
             ?assertEqual(100, Event#population_initialized.initial_size)
         end},
        {"population_terminated creates valid event",
         fun() ->
             Event = neuroevolution_behavioral_events:population_terminated(#{
                 population_id => <<"pop-1">>,
                 final_generation => 1000,
                 reason => target_reached,
                 final_champion_id => <<"champ-1">>,
                 final_champion_fitness => 0.99
             }),
             ?assertEqual(target_reached, Event#population_terminated.reason)
         end}
    ].

%%% ============================================================================
%%% Species Event Tests
%%% ============================================================================

species_events_test_() ->
    [
        {"lineage_diverged creates valid event",
         fun() ->
             Event = neuroevolution_behavioral_events:lineage_diverged(#{
                 new_species_id => <<"species-2">>,
                 parent_species_id => <<"species-1">>,
                 founder_individual_id => <<"ind-1">>,
                 compatibility_distance => 3.5,
                 generation => 50
             }),
             ?assertEqual(structural, Event#lineage_diverged.divergence_cause)
         end},
        {"species_emerged creates valid event",
         fun() ->
             Event = neuroevolution_behavioral_events:species_emerged(#{
                 species_id => <<"species-1">>,
                 population_id => <<"pop-1">>,
                 founder_id => <<"ind-1">>,
                 generation => 0
             }),
             ?assertEqual(<<"species-1">>, Event#species_emerged.species_id)
         end}
    ].

%%% ============================================================================
%%% Controller Event Tests
%%% ============================================================================

controller_events_test_() ->
    [
        {"silo_activated creates valid event",
         fun() ->
             Event = neuroevolution_behavioral_events:silo_activated(#{
                 silo_type => temporal,
                 realm => <<"default">>
             }),
             ?assertEqual(temporal, Event#silo_activated.silo_type),
             ?assertEqual(lc_supervisor, Event#silo_activated.meta#event_meta.emitter)
         end},
        {"silo_deactivated creates valid event",
         fun() ->
             Event = neuroevolution_behavioral_events:silo_deactivated(#{
                 silo_type => competitive,
                 realm => <<"default">>,
                 reason => manual
             }),
             ?assertEqual(manual, Event#silo_deactivated.reason)
         end}
    ].

%%% ============================================================================
%%% Utility Function Tests
%%% ============================================================================

utility_test_() ->
    [
        {"event_to_map converts record to map",
         fun() ->
             Event = neuroevolution_behavioral_events:offspring_born(#{
                 individual_id => <<"ind-1">>,
                 parent_ids => [<<"p1">>, <<"p2">>],
                 generation => 10,
                 species_id => <<"species-1">>,
                 population_id => <<"pop-1">>
             }),
             Map = neuroevolution_behavioral_events:event_to_map(Event),
             ?assertEqual(offspring_born, maps:get(event_type, Map)),
             ?assertEqual(<<"ind-1">>, maps:get(individual_id, Map)),
             ?assert(is_map(maps:get(meta, Map)))
         end},
        {"stream_id generates correct format",
         fun() ->
             ?assertEqual(<<"individual-abc123">>,
                          neuroevolution_behavioral_events:stream_id(individual, <<"abc123">>)),
             ?assertEqual(<<"species-xyz">>,
                          neuroevolution_behavioral_events:stream_id(species, <<"xyz">>)),
             ?assertEqual(<<"population-pop1">>,
                          neuroevolution_behavioral_events:stream_id(population, <<"pop1">>))
         end}
    ].
