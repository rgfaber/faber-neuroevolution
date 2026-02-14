%% @doc LC TWEANN Morphology - Sensor/actuator definitions for LC controllers.
%%
%% Implements morphology_behaviour for the Liquid Conglomerate hyperparameter
%% controllers. This allows LC controllers to be evolved using faber_tweann.
%%
%% == Morphology Names ==
%% - lc_task_controller: Task Silo L0 controller (21 sensors, 16 actuators)
%% - lc_resource_controller: Resource Silo L0 controller (15 sensors, 9 actuators)
%%
%% == Usage ==
%% 1. Register at application startup:
%%    morphology_registry:register(lc_task_controller, lc_tweann_morphology).
%%
%% 2. Create agent with this morphology:
%%    Constraint = #constraint{morphology = lc_task_controller},
%%    AgentId = genotype:construct_Agent(SpecieId, AgentId, Constraint).
%%
%% @author R.G. Lefever
%% @copyright 2024-2026 R.G. Lefever
-module(lc_tweann_morphology).
-behaviour(morphology_behaviour).

-include_lib("faber_tweann/include/records.hrl").

%% morphology_behaviour callbacks
-export([get_sensors/1, get_actuators/1]).

%% Utility
-export([register_morphologies/0]).

%%% ============================================================================
%%% Morphology Registration
%%% ============================================================================

%% @doc Register all LC morphologies with the morphology_registry.
%% Call this at application startup.
-spec register_morphologies() -> ok.
register_morphologies() ->
    morphology_registry:register(lc_task_controller, ?MODULE),
    morphology_registry:register(lc_resource_controller, ?MODULE),
    ok.

%%% ============================================================================
%%% morphology_behaviour Callbacks
%%% ============================================================================

%% @doc Get sensors for LC morphologies.
-spec get_sensors(atom()) -> [#sensor{}].
get_sensors(lc_task_controller) ->
    task_controller_sensors();
get_sensors(lc_resource_controller) ->
    resource_controller_sensors();
get_sensors(_) ->
    error({invalid_morphology, ?MODULE, "Use lc_task_controller or lc_resource_controller"}).

%% @doc Get actuators for LC morphologies.
-spec get_actuators(atom()) -> [#actuator{}].
get_actuators(lc_task_controller) ->
    task_controller_actuators();
get_actuators(lc_resource_controller) ->
    resource_controller_actuators();
get_actuators(_) ->
    error({invalid_morphology, ?MODULE, "Use lc_task_controller or lc_resource_controller"}).

%%% ============================================================================
%%% Task Controller Morphology (21 sensors, 16 actuators)
%%% ============================================================================

%% @private Task Silo L0 sensors.
%% Maps evolution state to neural network inputs.
task_controller_sensors() ->
    [
        %% Single sensor with 21 inputs (vector)
        #sensor{
            name = lc_task_sense,
            type = standard,
            vl = 21,  %% 21 sensor values as a vector
            scape = {private, lc_task_scape},
            format = {no_geo, [21]},
            parameters = #{
                inputs => [
                    %% Evolution sensors (1-16)
                    best_fitness,              % 1. Current best fitness (0-1)
                    avg_fitness,               % 2. Population average (0-1)
                    fitness_variance,          % 3. Population diversity (0-1)
                    improvement_velocity,      % 4. Rate of improvement (-1 to 1)
                    stagnation_severity,       % 5. How stagnant (0-1)
                    diversity_index,           % 6. Genetic diversity (0-1)
                    species_count_ratio,       % 7. Speciation level (0-1)
                    avg_network_complexity,    % 8. Topology complexity (0-1)
                    complexity_velocity,       % 9. Bloat detection (-1 to 1)
                    elite_dominance,           % 10. Elite vs avg (0-1)
                    crossover_success_rate,    % 11. Crossover effectiveness (0-1)
                    mutation_impact,           % 12. Mutation effectiveness (0-1)
                    resource_pressure_signal,  % 13. Resource Silo pressure (0-1)
                    evaluation_progress,       % 14. Training progress (0-1)
                    entropy,                   % 15. Population entropy (0-1)
                    convergence_trend,         % 16. Convergence direction (-1 to 1)
                    %% Self-play archive sensors (17-21)
                    archive_fill_ratio,        % 17. Archive size / max_size (0-1)
                    archive_fitness_mean,      % 18. Average fitness in archive (0-1)
                    archive_fitness_variance,  % 19. Fitness variance in archive (0-1)
                    archive_staleness,         % 20. Average age of archive entries (0-1)
                    population_vs_archive_ratio% 21. Population vs archive fitness (0-1)
                ]
            }
        }
    ].

%% @private Task Silo L0 actuators.
%% Maps neural network outputs to hyperparameter adjustments.
task_controller_actuators() ->
    [
        %% Single actuator with 16 outputs (vector)
        #actuator{
            name = lc_task_act,
            type = standard,
            vl = 16,  %% 16 actuator values as a vector
            scape = {private, lc_task_scape},
            format = {no_geo, [16]},
            parameters = #{
                outputs => [
                    %% Evolution actuators (1-12)
                    %% All outputs are sigmoid (0-1), scaled to actual ranges by task_silo
                    mutation_rate,              % 1. Per-gene mutation probability
                    mutation_strength,          % 2. Gaussian std dev for weights
                    selection_ratio,            % 3. Fraction surviving
                    add_node_rate,              % 4. Node addition probability
                    add_connection_rate,        % 5. Connection addition probability
                    delete_connection_rate,     % 6. Connection deletion probability
                    weight_perturb_vs_replace,  % 7. Perturb vs replace ratio
                    crossover_rate,             % 8. Probability of crossover
                    interspecies_crossover_rate,% 9. Cross-species breeding
                    elitism_count,              % 10. Guaranteed survivors (scaled to int)
                    population_size_delta,      % 11. Grow/shrink population (centered at 0.5)
                    compatibility_threshold_delta,% 12. Species separation (centered at 0.5)
                    %% Self-play archive actuators (13-16)
                    archive_threshold_percentile,  % 13. Entry threshold
                    archive_sampling_temperature,  % 14. Fitness-weighted sampling
                    archive_prune_ratio,           % 15. Keep top X%
                    archive_max_size_delta         % 16. Grow/shrink max size (centered at 0.5)
                ],
                %% Output scaling: neural network outputs 0-1, scaled to actual ranges
                output_ranges => #{
                    mutation_rate => {0.01, 0.50},
                    mutation_strength => {0.05, 1.0},
                    selection_ratio => {0.05, 0.50},
                    add_node_rate => {0.0, 0.15},
                    add_connection_rate => {0.0, 0.25},
                    delete_connection_rate => {0.0, 0.10},
                    weight_perturb_vs_replace => {0.5, 1.0},
                    crossover_rate => {0.0, 0.9},
                    interspecies_crossover_rate => {0.0, 0.3},
                    elitism_count => {1, 10},
                    population_size_delta => {-10, 10},
                    compatibility_threshold_delta => {-0.5, 0.5},
                    archive_threshold_percentile => {0.3, 0.95},
                    archive_sampling_temperature => {0.0, 1.0},
                    archive_prune_ratio => {0.5, 1.0},
                    archive_max_size_delta => {-5, 5}
                }
            }
        }
    ].

%%% ============================================================================
%%% Resource Controller Morphology (15 sensors, 9 actuators)
%%% ============================================================================

%% @private Resource Silo L0 sensors.
resource_controller_sensors() ->
    [
        #sensor{
            name = lc_resource_sense,
            type = standard,
            vl = 15,  %% 15 sensor values
            scape = {private, lc_resource_scape},
            format = {no_geo, [15]},
            parameters = #{
                inputs => [
                    memory_pressure,           % 1. Primary memory constraint (0-1)
                    memory_velocity,           % 2. Rate of memory change (-1 to 1)
                    cpu_pressure,              % 3. CPU saturation (0-1)
                    cpu_velocity,              % 4. Rate of CPU change (-1 to 1)
                    run_queue_pressure,        % 5. Work backlog (0-1)
                    process_pressure,          % 6. Process count ratio (0-1)
                    message_queue_pressure,    % 7. Backpressure indicator (0-1)
                    binary_memory_ratio,       % 8. Binary heap stress (0-1)
                    gc_frequency,              % 9. GC activity level (0-1)
                    current_concurrency_ratio, % 10. Headroom (0-1)
                    task_silo_exploration_boost,% 11. Task Silo's exploration state (0-1)
                    evaluation_throughput,     % 12. Performance feedback (0-1)
                    time_since_last_gc,        % 13. GC timing (0-1)
                    archive_memory_ratio,      % 14. Archive memory footprint (0-1)
                    crdt_state_size_ratio      % 15. CRDT sync overhead (0-1)
                ]
            }
        }
    ].

%% @private Resource Silo L0 actuators.
resource_controller_actuators() ->
    [
        #actuator{
            name = lc_resource_act,
            type = standard,
            vl = 9,  %% 9 actuator values
            scape = {private, lc_resource_scape},
            format = {no_geo, [9]},
            parameters = #{
                outputs => [
                    max_concurrent_evaluations,   % 1. Worker parallelism
                    evaluation_batch_size,        % 2. Batch granularity
                    gc_trigger_threshold,         % 3. When to force GC
                    pause_threshold,              % 4. When to pause evolution
                    throttle_intensity,           % 5. How aggressively to throttle
                    evaluations_per_individual,   % 6. Statistical confidence vs speed
                    task_silo_pressure_signal,    % 7. Tell Task Silo to back off
                    gc_aggressiveness,            % 8. Gentle (0) vs aggressive (1)
                    archive_gc_pressure           % 9. Force archive cleanup
                ],
                output_ranges => #{
                    max_concurrent_evaluations => {1, 1000000},  % Erlang handles millions of processes
                    evaluation_batch_size => {1, 50},
                    gc_trigger_threshold => {0.5, 0.95},
                    pause_threshold => {0.7, 0.99},
                    throttle_intensity => {0.0, 1.0},
                    evaluations_per_individual => {1, 20},
                    task_silo_pressure_signal => {0.0, 1.0},
                    gc_aggressiveness => {0.0, 1.0},
                    archive_gc_pressure => {0.0, 1.0}
                }
            }
        }
    ].
