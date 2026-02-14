%% @doc Unit tests for LC morphology modules (resource, task, distribution).
%%
%% Tests pure functions that define TWEANN sensor/actuator specifications.
%%
%% @author R.G. Lefever
%% @copyright 2024-2026 R.G. Lefever
-module(lc_morphology_tests).

-include_lib("eunit/include/eunit.hrl").

%%% ============================================================================
%%% Resource L0 Morphology Tests
%%% ============================================================================

resource_sensor_count_test() ->
    Count = resource_l0_morphology:sensor_count(),
    ?assertEqual(15, Count).  % 13 base + 2 archive (archive_memory_ratio, crdt_state_size_ratio)

resource_actuator_count_test() ->
    Count = resource_l0_morphology:actuator_count(),
    ?assertEqual(10, Count).  % 8 base + 1 archive (archive_gc_pressure) + 1 timeout (evaluation_timeout)

resource_sensor_names_test() ->
    Names = resource_l0_morphology:sensor_names(),
    ?assertEqual(15, length(Names)),
    %% Check some key sensors exist
    ?assert(lists:member(memory_pressure, Names)),
    ?assert(lists:member(cpu_pressure, Names)),
    ?assert(lists:member(task_silo_exploration, Names)),
    %% Check archive sensors exist
    ?assert(lists:member(archive_memory_ratio, Names)),
    ?assert(lists:member(crdt_state_size_ratio, Names)).

resource_actuator_names_test() ->
    Names = resource_l0_morphology:actuator_names(),
    ?assertEqual(10, length(Names)),
    %% Check key actuators exist
    ?assert(lists:member(max_concurrent_evaluations, Names)),
    ?assert(lists:member(gc_trigger_threshold, Names)),
    ?assert(lists:member(task_silo_pressure_signal, Names)),
    %% Check archive actuator exists
    ?assert(lists:member(archive_gc_pressure, Names)),
    %% Check timeout actuator exists
    ?assert(lists:member(evaluation_timeout, Names)).

resource_sensor_spec_valid_test() ->
    Spec = resource_l0_morphology:sensor_spec(memory_pressure),
    ?assert(is_map(Spec)),
    ?assertEqual(memory_pressure, maps:get(name, Spec)),
    ?assertEqual({0.0, 1.0}, maps:get(range, Spec)).

resource_sensor_spec_velocity_test() ->
    %% Velocity sensors have -1.0 to 1.0 range
    Spec = resource_l0_morphology:sensor_spec(memory_velocity),
    ?assertEqual({-1.0, 1.0}, maps:get(range, Spec)).

resource_sensor_spec_unknown_test() ->
    ?assertEqual(undefined, resource_l0_morphology:sensor_spec(nonexistent_sensor)).

resource_actuator_spec_valid_test() ->
    Spec = resource_l0_morphology:actuator_spec(max_concurrent_evaluations),
    ?assert(is_map(Spec)),
    ?assertEqual({1, 1000000}, maps:get(range, Spec)).

resource_actuator_spec_unknown_test() ->
    ?assertEqual(undefined, resource_l0_morphology:actuator_spec(nonexistent_actuator)).

resource_l0_hyperparameters_test() ->
    Params = resource_l0_morphology:l0_hyperparameters(),
    ?assertEqual(8, length(Params)),
    ?assert(lists:member(memory_high_threshold, Params)),
    ?assert(lists:member(base_concurrency, Params)).

resource_l0_hyperparameter_spec_test() ->
    Spec = resource_l0_morphology:l0_hyperparameter_spec(memory_high_threshold),
    ?assert(is_map(Spec)),
    ?assertEqual(0.70, maps:get(default, Spec)),
    ?assertEqual({0.5, 0.85}, maps:get(range, Spec)).

resource_l1_hyperparameters_test() ->
    Params = resource_l0_morphology:l1_hyperparameters(),
    ?assertEqual(5, length(Params)),
    ?assert(lists:member(threshold_adaptation_rate, Params)),
    ?assert(lists:member(cross_silo_coupling, Params)).

resource_get_l0_defaults_test() ->
    Defaults = resource_l0_morphology:get_l0_defaults(),
    ?assert(is_map(Defaults)),
    ?assertEqual(8, maps:size(Defaults)),
    ?assertEqual(0.70, maps:get(memory_high_threshold, Defaults)).

resource_get_l0_bounds_test() ->
    Bounds = resource_l0_morphology:get_l0_bounds(),
    ?assert(is_map(Bounds)),
    {Min, Max} = maps:get(memory_high_threshold, Bounds),
    ?assert(Min < Max).

resource_get_l1_defaults_test() ->
    Defaults = resource_l0_morphology:get_l1_defaults(),
    ?assertEqual(5, maps:size(Defaults)),
    ?assertEqual(0.05, maps:get(threshold_adaptation_rate, Defaults)).

resource_time_constants_test() ->
    ?assertEqual(5000, resource_l0_morphology:tau_l0()),
    ?assertEqual(30000, resource_l0_morphology:tau_l1()),
    ?assertEqual(300000, resource_l0_morphology:tau_l2()).

%%% ============================================================================
%%% Task L0 Morphology Tests
%%% ============================================================================

task_sensor_count_test() ->
    Count = task_l0_morphology:sensor_count(),
    ?assertEqual(21, Count).  % 16 base + 5 archive

task_actuator_count_test() ->
    Count = task_l0_morphology:actuator_count(),
    ?assertEqual(16, Count).  % 12 base + 4 archive

task_sensor_names_test() ->
    Names = task_l0_morphology:sensor_names(),
    ?assertEqual(21, length(Names)),
    %% Check key sensors
    ?assert(lists:member(best_fitness, Names)),
    ?assert(lists:member(improvement_velocity, Names)),
    ?assert(lists:member(resource_pressure_signal, Names)),
    %% Check archive sensors
    ?assert(lists:member(archive_fill_ratio, Names)),
    ?assert(lists:member(archive_staleness, Names)).

task_actuator_names_test() ->
    Names = task_l0_morphology:actuator_names(),
    ?assertEqual(16, length(Names)),
    ?assert(lists:member(mutation_rate, Names)),
    ?assert(lists:member(add_node_rate, Names)),
    %% Check archive actuators
    ?assert(lists:member(archive_threshold_percentile, Names)),
    ?assert(lists:member(archive_sampling_temperature, Names)).

task_sensor_spec_valid_test() ->
    Spec = task_l0_morphology:sensor_spec(best_fitness),
    ?assert(is_map(Spec)),
    ?assertEqual({0.0, 1.0}, maps:get(range, Spec)).

task_actuator_spec_valid_test() ->
    Spec = task_l0_morphology:actuator_spec(mutation_rate),
    ?assert(is_map(Spec)),
    %% Mutation rate has specific bounds
    {Min, Max} = maps:get(range, Spec),
    ?assert(Min >= 0.0),
    ?assert(Max =< 1.0).

task_l0_hyperparameters_test() ->
    Params = task_l0_morphology:l0_hyperparameters(),
    ?assertEqual(12, length(Params)),  % 8 base + 4 archive
    ?assert(lists:member(mutation_rate_min, Params)),
    %% Check archive hyperparameters
    ?assert(lists:member(archive_threshold_min, Params)),
    ?assert(lists:member(archive_diversity_weight, Params)).

task_l1_hyperparameters_test() ->
    Params = task_l0_morphology:l1_hyperparameters(),
    ?assertEqual(7, length(Params)),
    ?assert(lists:member(aggression_factor, Params)).

task_time_constants_evaluations_test() ->
    %% Task silo uses evaluations, not milliseconds
    %% tau_l0 = 10 generations = ~500 evaluations with pop_size 50
    TauL0 = task_l0_morphology:tau_l0(),
    TauL1 = task_l0_morphology:tau_l1(),
    TauL2 = task_l0_morphology:tau_l2(),
    ?assert(TauL0 < TauL1),
    ?assert(TauL1 < TauL2).

%%% ============================================================================
%%% Distribution L0 Morphology Tests
%%% ============================================================================

distribution_sensor_count_test() ->
    Count = distribution_l0_morphology:sensor_count(),
    ?assertEqual(14, Count).

distribution_actuator_count_test() ->
    Count = distribution_l0_morphology:actuator_count(),
    ?assertEqual(10, Count).

distribution_sensor_names_test() ->
    Names = distribution_l0_morphology:sensor_names(),
    ?assertEqual(14, length(Names)),
    ?assert(lists:member(local_load, Names)),
    ?assert(lists:member(peer_count, Names)).

distribution_actuator_names_test() ->
    Names = distribution_l0_morphology:actuator_names(),
    ?assertEqual(10, length(Names)),
    ?assert(lists:member(local_vs_remote_ratio, Names)),
    ?assert(lists:member(migration_rate, Names)).

distribution_sensor_spec_valid_test() ->
    Spec = distribution_l0_morphology:sensor_spec(local_load),
    ?assert(is_map(Spec)),
    ?assertEqual({0.0, 1.0}, maps:get(range, Spec)).

distribution_actuator_spec_valid_test() ->
    Spec = distribution_l0_morphology:actuator_spec(local_vs_remote_ratio),
    ?assert(is_map(Spec)),
    ?assertEqual({0.0, 1.0}, maps:get(range, Spec)).

distribution_l0_hyperparameters_test() ->
    Params = distribution_l0_morphology:l0_hyperparameters(),
    ?assertEqual(8, length(Params)),
    ?assert(lists:member(max_islands, Params)).

distribution_l1_hyperparameters_test() ->
    Params = distribution_l0_morphology:l1_hyperparameters(),
    ?assertEqual(5, length(Params)),
    ?assert(lists:member(load_sensitivity, Params)).

distribution_time_constants_test() ->
    %% Distribution silo is fastest (real-time)
    TauL0 = distribution_l0_morphology:tau_l0(),
    ?assertEqual(1000, TauL0).  % 1 second

%%% ============================================================================
%%% Cross-Module Consistency Tests
%%% ============================================================================

all_morphologies_have_matching_counts_test_() ->
    [
        {"resource sensors match names",
         fun() ->
             Count = resource_l0_morphology:sensor_count(),
             Names = resource_l0_morphology:sensor_names(),
             ?assertEqual(Count, length(Names))
         end},
        {"resource actuators match names",
         fun() ->
             Count = resource_l0_morphology:actuator_count(),
             Names = resource_l0_morphology:actuator_names(),
             ?assertEqual(Count, length(Names))
         end},
        {"task sensors match names",
         fun() ->
             Count = task_l0_morphology:sensor_count(),
             Names = task_l0_morphology:sensor_names(),
             ?assertEqual(Count, length(Names))
         end},
        {"task actuators match names",
         fun() ->
             Count = task_l0_morphology:actuator_count(),
             Names = task_l0_morphology:actuator_names(),
             ?assertEqual(Count, length(Names))
         end},
        {"distribution sensors match names",
         fun() ->
             Count = distribution_l0_morphology:sensor_count(),
             Names = distribution_l0_morphology:sensor_names(),
             ?assertEqual(Count, length(Names))
         end},
        {"distribution actuators match names",
         fun() ->
             Count = distribution_l0_morphology:actuator_count(),
             Names = distribution_l0_morphology:actuator_names(),
             ?assertEqual(Count, length(Names))
         end}
    ].

all_sensors_have_specs_test_() ->
    [
        {"resource sensors have specs",
         fun() ->
             Names = resource_l0_morphology:sensor_names(),
             lists:foreach(
                 fun(Name) ->
                     Spec = resource_l0_morphology:sensor_spec(Name),
                     ?assertNotEqual(undefined, Spec)
                 end,
                 Names
             )
         end},
        {"task sensors have specs",
         fun() ->
             Names = task_l0_morphology:sensor_names(),
             lists:foreach(
                 fun(Name) ->
                     Spec = task_l0_morphology:sensor_spec(Name),
                     ?assertNotEqual(undefined, Spec)
                 end,
                 Names
             )
         end},
        {"distribution sensors have specs",
         fun() ->
             Names = distribution_l0_morphology:sensor_names(),
             lists:foreach(
                 fun(Name) ->
                     Spec = distribution_l0_morphology:sensor_spec(Name),
                     ?assertNotEqual(undefined, Spec)
                 end,
                 Names
             )
         end}
    ].

all_actuators_have_specs_test_() ->
    [
        {"resource actuators have specs",
         fun() ->
             Names = resource_l0_morphology:actuator_names(),
             lists:foreach(
                 fun(Name) ->
                     Spec = resource_l0_morphology:actuator_spec(Name),
                     ?assertNotEqual(undefined, Spec)
                 end,
                 Names
             )
         end},
        {"task actuators have specs",
         fun() ->
             Names = task_l0_morphology:actuator_names(),
             lists:foreach(
                 fun(Name) ->
                     Spec = task_l0_morphology:actuator_spec(Name),
                     ?assertNotEqual(undefined, Spec)
                 end,
                 Names
             )
         end},
        {"distribution actuators have specs",
         fun() ->
             Names = distribution_l0_morphology:actuator_names(),
             lists:foreach(
                 fun(Name) ->
                     Spec = distribution_l0_morphology:actuator_spec(Name),
                     ?assertNotEqual(undefined, Spec)
                 end,
                 Names
             )
         end}
    ].
