%% @doc Task Silo L0 Morphology - TWEANN sensor/actuator definitions.
%%
%% Part of the Liquid Conglomerate v2 architecture. Defines the neural network
%% morphology for the Task Silo's L0 hyperparameter controller.
%%
%% == Architecture ==
%%
%% L0 is a TWEANN (Topology and Weight Evolving Artificial Neural Network) that:
%% - Takes 21 sensor inputs (16 evolution + 5 self-play archive)
%% - Produces 20 actuator outputs (12 evolution + 4 layer-specific + 4 archive)
%% - Has 12 hyperparameters that L1 can tune (8 evolution + 4 archive)
%% - Has 7 L1 hyperparameters that L2 can tune
%%
%% == Time Constant ==
%%
%% tau_L0 = 1000 evaluations (adaptation rate for evolution control)
%%
%% Note: The system uses evaluation-centric timing, not discrete generations.
%% All time constants are expressed in evaluation counts.
%%
%% @author R.G. Lefever
%% @copyright 2024-2026 R.G. Lefever
-module(task_l0_morphology).

-export([
    %% Morphology definitions
    sensor_count/0,
    actuator_count/0,
    sensor_names/0,
    actuator_names/0,
    sensor_spec/1,
    actuator_spec/1,

    %% Hyperparameter definitions
    l0_hyperparameters/0,
    l0_hyperparameter_spec/1,
    l1_hyperparameters/0,
    l1_hyperparameter_spec/1,

    %% Bounds and defaults
    get_l0_defaults/0,
    get_l0_bounds/0,
    get_l1_defaults/0,
    get_l1_bounds/0,

    %% Time constants
    tau_l0/0,
    tau_l1/0,
    tau_l2/0
]).

%%% ============================================================================
%%% Time Constants (in evaluations, not generations)
%%% ============================================================================

%% @doc L0 time constant - 1000 evaluations for hyperparameter adaptation.
%% With population ~100, this is roughly 10 evaluation cycles.
-spec tau_l0() -> pos_integer().
tau_l0() -> 1000.  % evaluations

%% @doc L1 time constant - 5000 evaluations for tactical adaptation.
%% With population ~100, this is roughly 50 evaluation cycles.
-spec tau_l1() -> pos_integer().
tau_l1() -> 5000.  % evaluations

%% @doc L2 time constant - 10000 evaluations for strategic learning.
%% With population ~100, this is roughly 100 evaluation cycles.
-spec tau_l2() -> pos_integer().
tau_l2() -> 10000.  % evaluations

%%% ============================================================================
%%% Sensor Definitions (21 inputs: 16 evolution + 5 self-play archive)
%%% ============================================================================

%% @doc Number of sensors (neural network inputs).
-spec sensor_count() -> pos_integer().
sensor_count() -> 21.

%% @doc Ordered list of sensor names.
-spec sensor_names() -> [atom()].
sensor_names() ->
    [
        %% Evolution sensors (1-16)
        best_fitness,              % 1. Current best fitness
        avg_fitness,               % 2. Population average
        fitness_variance,          % 3. Population diversity
        improvement_velocity,      % 4. Rate of improvement
        stagnation_severity,       % 5. How stagnant
        diversity_index,           % 6. Genetic diversity
        species_count_ratio,       % 7. Speciation level
        avg_network_complexity,    % 8. Topology complexity
        complexity_velocity,       % 9. Bloat detection
        elite_dominance,           % 10. How dominant elites are
        crossover_success_rate,    % 11. Crossover effectiveness
        mutation_impact,           % 12. Mutation effectiveness
        resource_pressure_signal,  % 13. Cross-silo: Resource Silo pressure
        evaluation_progress,       % 14. Training progress (evaluations done)
        entropy,                   % 15. Population entropy
        convergence_trend,         % 16. Convergence direction
        %% Self-play archive sensors (17-21)
        archive_fill_ratio,        % 17. Archive size / max_size
        archive_fitness_mean,      % 18. Average fitness in archive
        archive_fitness_variance,  % 19. Fitness variance in archive
        archive_staleness,         % 20. Average age of archive entries
        population_vs_archive_ratio% 21. Population fitness vs archive fitness
    ].

%% @doc Get specification for a sensor.
-spec sensor_spec(atom()) -> map() | undefined.
sensor_spec(best_fitness) ->
    #{
        name => best_fitness,
        range => {0.0, 1.0},
        source => evolution_stats,
        description => <<"Best fitness in population (normalized)">>
    };
sensor_spec(avg_fitness) ->
    #{
        name => avg_fitness,
        range => {0.0, 1.0},
        source => evolution_stats,
        description => <<"Average fitness of population (normalized)">>
    };
sensor_spec(fitness_variance) ->
    #{
        name => fitness_variance,
        range => {0.0, 1.0},
        source => evolution_stats,
        description => <<"Fitness variance (normalized, 0=uniform, 1=diverse)">>
    };
sensor_spec(improvement_velocity) ->
    #{
        name => improvement_velocity,
        range => {-1.0, 1.0},
        source => computed,
        description => <<"Rate of fitness improvement per evaluation (-1=declining, +1=improving fast)">>
    };
sensor_spec(stagnation_severity) ->
    #{
        name => stagnation_severity,
        range => {0.0, 1.0},
        source => computed,
        description => <<"How stagnant evolution is (0=progressing, 1=stuck)">>
    };
sensor_spec(diversity_index) ->
    #{
        name => diversity_index,
        range => {0.0, 1.0},
        source => evolution_stats,
        description => <<"Genetic diversity measure (0=homogeneous, 1=diverse)">>
    };
sensor_spec(species_count_ratio) ->
    #{
        name => species_count_ratio,
        range => {0.0, 1.0},
        source => evolution_stats,
        description => <<"Current species / max species">>
    };
sensor_spec(avg_network_complexity) ->
    #{
        name => avg_network_complexity,
        range => {0.0, 1.0},
        source => evolution_stats,
        description => <<"Average network complexity (normalized)">>
    };
sensor_spec(complexity_velocity) ->
    #{
        name => complexity_velocity,
        range => {-1.0, 1.0},
        source => computed,
        description => <<"Rate of complexity change (-1=simplifying, +1=bloating)">>
    };
sensor_spec(elite_dominance) ->
    #{
        name => elite_dominance,
        range => {0.0, 1.0},
        source => evolution_stats,
        description => <<"Elite fitness / average fitness (normalized)">>
    };
sensor_spec(crossover_success_rate) ->
    #{
        name => crossover_success_rate,
        range => {0.0, 1.0},
        source => evolution_stats,
        description => <<"Fraction of crossovers producing fitter offspring">>
    };
sensor_spec(mutation_impact) ->
    #{
        name => mutation_impact,
        range => {0.0, 1.0},
        source => evolution_stats,
        description => <<"Average fitness change from mutations (normalized)">>
    };
sensor_spec(resource_pressure_signal) ->
    #{
        name => resource_pressure_signal,
        range => {0.0, 1.0},
        source => cross_silo,
        description => <<"Resource Silo's pressure signal (0=healthy, 1=critical)">>
    };
sensor_spec(evaluation_progress) ->
    #{
        name => evaluation_progress,
        range => {0.0, 1.0},
        source => internal,
        description => <<"Current evaluations / max evaluations">>
    };
sensor_spec(entropy) ->
    #{
        name => entropy,
        range => {0.0, 1.0},
        source => computed,
        description => <<"Population information entropy (normalized)">>
    };
sensor_spec(convergence_trend) ->
    #{
        name => convergence_trend,
        range => {-1.0, 1.0},
        source => computed,
        description => <<"Convergence direction (-1=diverging, +1=converging)">>
    };
%% Self-play archive sensors
sensor_spec(archive_fill_ratio) ->
    #{
        name => archive_fill_ratio,
        range => {0.0, 1.0},
        source => opponent_archive,
        description => <<"Archive size / max_size (0=empty, 1=full)">>
    };
sensor_spec(archive_fitness_mean) ->
    #{
        name => archive_fitness_mean,
        range => {0.0, 1.0},
        source => opponent_archive,
        description => <<"Average fitness of archived opponents (normalized)">>
    };
sensor_spec(archive_fitness_variance) ->
    #{
        name => archive_fitness_variance,
        range => {0.0, 1.0},
        source => opponent_archive,
        description => <<"Fitness variance in archive (0=homogeneous, 1=diverse)">>
    };
sensor_spec(archive_staleness) ->
    #{
        name => archive_staleness,
        range => {0.0, 1.0},
        source => opponent_archive,
        description => <<"Average age of archive entries (0=fresh, 1=stale)">>
    };
sensor_spec(population_vs_archive_ratio) ->
    #{
        name => population_vs_archive_ratio,
        range => {0.0, 1.0},
        source => computed,
        description => <<"Population fitness / archive fitness (arms race progress)">>
    };
sensor_spec(_) ->
    undefined.

%%% ============================================================================
%%% Actuator Definitions (20 outputs: 12 evolution + 4 layer-specific + 4 archive)
%%% ============================================================================

%% @doc Number of actuators (neural network outputs).
%% 12 evolution + 4 layer-specific mutation + 4 self-play archive = 20
-spec actuator_count() -> pos_integer().
actuator_count() -> 20.

%% @doc Ordered list of actuator names.
-spec actuator_names() -> [atom()].
actuator_names() ->
    [
        %% Evolution actuators (1-12)
        mutation_rate,              % 1. Per-gene mutation probability (fallback)
        mutation_strength,          % 2. Gaussian std dev for weights (fallback)
        selection_ratio,            % 3. Fraction surviving
        add_node_rate,              % 4. Node addition probability
        add_connection_rate,        % 5. Connection addition probability
        delete_connection_rate,     % 6. Connection deletion probability
        weight_perturb_vs_replace,  % 7. Perturb vs replace ratio
        crossover_rate,             % 8. Probability of crossover
        interspecies_crossover_rate,% 9. Cross-species breeding
        elitism_count,              % 10. Guaranteed survivors
        population_size_delta,      % 11. Grow/shrink population
        compatibility_threshold_delta,% 12. Species separation
        %% Layer-specific mutation actuators (13-16)
        %% These enable reservoir/readout training strategies
        %% See guides/training-strategies.md for rationale
        reservoir_mutation_rate,    % 13. Hidden layer mutation rate
        reservoir_mutation_strength,% 14. Hidden layer mutation strength
        readout_mutation_rate,      % 15. Output layer mutation rate
        readout_mutation_strength,  % 16. Output layer mutation strength
        %% Self-play archive actuators (17-20)
        archive_threshold_percentile,  % 17. Entry threshold
        archive_sampling_temperature,  % 18. Fitness-weighted sampling
        archive_prune_ratio,           % 19. Keep top X%
        archive_max_size_delta         % 20. Grow/shrink max archive size
    ].

%% @doc Get specification for an actuator.
-spec actuator_spec(atom()) -> map() | undefined.
actuator_spec(mutation_rate) ->
    #{
        name => mutation_rate,
        range => {0.01, 0.50},
        target => evolution_params,
        description => <<"Per-gene mutation probability">>
    };
actuator_spec(mutation_strength) ->
    #{
        name => mutation_strength,
        range => {0.05, 1.0},
        target => evolution_params,
        description => <<"Gaussian std dev for weight perturbation">>
    };
actuator_spec(selection_ratio) ->
    #{
        name => selection_ratio,
        range => {0.05, 0.50},
        target => evolution_params,
        description => <<"Fraction of population surviving selection">>
    };
actuator_spec(add_node_rate) ->
    #{
        name => add_node_rate,
        range => {0.0, 0.15},
        target => topology_mutation,
        description => <<"Probability of adding a new node">>
    };
actuator_spec(add_connection_rate) ->
    #{
        name => add_connection_rate,
        range => {0.0, 0.25},
        target => topology_mutation,
        description => <<"Probability of adding a new connection">>
    };
actuator_spec(delete_connection_rate) ->
    #{
        name => delete_connection_rate,
        range => {0.0, 0.10},
        target => topology_mutation,
        description => <<"Probability of deleting a connection">>
    };
actuator_spec(weight_perturb_vs_replace) ->
    #{
        name => weight_perturb_vs_replace,
        range => {0.5, 1.0},
        target => weight_mutation,
        description => <<"Ratio of perturb (1.0) vs replace (0.5)">>
    };
actuator_spec(crossover_rate) ->
    #{
        name => crossover_rate,
        range => {0.0, 0.9},
        target => breeding,
        description => <<"Probability of using crossover vs cloning">>
    };
actuator_spec(interspecies_crossover_rate) ->
    #{
        name => interspecies_crossover_rate,
        range => {0.0, 0.3},
        target => speciation,
        description => <<"Probability of cross-species breeding">>
    };
actuator_spec(elitism_count) ->
    #{
        name => elitism_count,
        range => {1, 10},
        target => selection,
        description => <<"Number of elite individuals guaranteed to survive">>
    };
actuator_spec(population_size_delta) ->
    #{
        name => population_size_delta,
        range => {-10, 10},
        target => population,
        description => <<"Change to population size (-10 to +10)">>
    };
actuator_spec(compatibility_threshold_delta) ->
    #{
        name => compatibility_threshold_delta,
        range => {-0.5, 0.5},
        target => speciation,
        description => <<"Change to species compatibility threshold">>
    };
%% Layer-specific mutation actuators
%% These enable reservoir/readout training strategies where hidden layers
%% (reservoir) are mutated conservatively while output layers (readout)
%% can adapt faster. See guides/training-strategies.md for rationale.
actuator_spec(reservoir_mutation_rate) ->
    #{
        name => reservoir_mutation_rate,
        range => {0.01, 0.50},
        target => layer_mutation,
        description => <<"Hidden layer mutation probability">>
    };
actuator_spec(reservoir_mutation_strength) ->
    #{
        name => reservoir_mutation_strength,
        range => {0.05, 1.0},
        target => layer_mutation,
        description => <<"Hidden layer weight perturbation strength">>
    };
actuator_spec(readout_mutation_rate) ->
    #{
        name => readout_mutation_rate,
        range => {0.01, 0.50},
        target => layer_mutation,
        description => <<"Output layer mutation probability">>
    };
actuator_spec(readout_mutation_strength) ->
    #{
        name => readout_mutation_strength,
        range => {0.05, 1.0},
        target => layer_mutation,
        description => <<"Output layer weight perturbation strength">>
    };
%% Self-play archive actuators
actuator_spec(archive_threshold_percentile) ->
    #{
        name => archive_threshold_percentile,
        range => {0.3, 0.95},
        target => opponent_archive,
        description => <<"Minimum fitness percentile for archive entry">>
    };
actuator_spec(archive_sampling_temperature) ->
    #{
        name => archive_sampling_temperature,
        range => {0.0, 1.0},
        target => opponent_archive,
        description => <<"Sampling temperature: 0=uniform, 1=fitness-weighted">>
    };
actuator_spec(archive_prune_ratio) ->
    #{
        name => archive_prune_ratio,
        range => {0.5, 1.0},
        target => opponent_archive,
        description => <<"Keep top X% when pruning archive">>
    };
actuator_spec(archive_max_size_delta) ->
    #{
        name => archive_max_size_delta,
        range => {-5, 5},
        target => opponent_archive,
        description => <<"Change to maximum archive size">>
    };
actuator_spec(_) ->
    undefined.

%%% ============================================================================
%%% L0 Hyperparameters (12 params: 8 evolution + 4 archive, tuned by L1)
%%% ============================================================================

%% @doc List of L0 hyperparameter names.
-spec l0_hyperparameters() -> [atom()].
l0_hyperparameters() ->
    [
        %% Evolution hyperparameters (1-8)
        mutation_rate_min,             % 1. Floor for mutation rate
        mutation_rate_max,             % 2. Ceiling for mutation rate
        topology_mutation_boost,       % 3. Multiplier when stagnating
        exploitation_vs_exploration,   % 4. Balance factor
        stagnation_velocity_threshold, % 5. When to consider stagnant
        complexity_penalty_weight,     % 6. Penalize bloat
        diversity_bonus_weight,        % 7. Reward diversity
        resource_sensitivity,          % 8. How much to heed Resource Silo
        %% Self-play archive hyperparameters (9-12)
        archive_threshold_min,         % 9. Floor for archive entry threshold
        archive_threshold_max,         % 10. Ceiling for archive entry threshold
        archive_diversity_weight,      % 11. Fitness vs diversity in sampling
        archive_recency_decay          % 12. Age decay factor per generation
    ].

%% @doc Get specification for an L0 hyperparameter.
-spec l0_hyperparameter_spec(atom()) -> map() | undefined.
l0_hyperparameter_spec(mutation_rate_min) ->
    #{
        name => mutation_rate_min,
        default => 0.01,
        range => {0.001, 0.05},
        description => <<"Minimum allowed mutation rate">>
    };
l0_hyperparameter_spec(mutation_rate_max) ->
    #{
        name => mutation_rate_max,
        default => 0.50,
        range => {0.3, 0.8},
        description => <<"Maximum allowed mutation rate">>
    };
l0_hyperparameter_spec(topology_mutation_boost) ->
    #{
        name => topology_mutation_boost,
        default => 1.5,
        range => {1.0, 3.0},
        description => <<"Topology mutation multiplier when stagnating">>
    };
l0_hyperparameter_spec(exploitation_vs_exploration) ->
    #{
        name => exploitation_vs_exploration,
        default => 0.5,
        range => {0.0, 1.0},
        description => <<"Balance: 0=explore, 1=exploit">>
    };
l0_hyperparameter_spec(stagnation_velocity_threshold) ->
    #{
        name => stagnation_velocity_threshold,
        default => 0.001,
        range => {0.0001, 0.01},
        description => <<"Velocity below which evolution is stagnant">>
    };
l0_hyperparameter_spec(complexity_penalty_weight) ->
    #{
        name => complexity_penalty_weight,
        default => 0.1,
        range => {0.0, 0.5},
        description => <<"Weight for complexity penalty in fitness">>
    };
l0_hyperparameter_spec(diversity_bonus_weight) ->
    #{
        name => diversity_bonus_weight,
        default => 0.1,
        range => {0.0, 0.5},
        description => <<"Weight for diversity bonus in fitness">>
    };
l0_hyperparameter_spec(resource_sensitivity) ->
    #{
        name => resource_sensitivity,
        default => 0.5,
        range => {0.0, 1.0},
        description => <<"How much to respond to Resource Silo signals">>
    };
%% Self-play archive hyperparameters
l0_hyperparameter_spec(archive_threshold_min) ->
    #{
        name => archive_threshold_min,
        default => 0.3,
        range => {0.1, 0.5},
        description => <<"Minimum fitness percentile for archive entry">>
    };
l0_hyperparameter_spec(archive_threshold_max) ->
    #{
        name => archive_threshold_max,
        default => 0.95,
        range => {0.8, 0.99},
        description => <<"Maximum fitness percentile for archive entry">>
    };
l0_hyperparameter_spec(archive_diversity_weight) ->
    #{
        name => archive_diversity_weight,
        default => 0.3,
        range => {0.0, 0.7},
        description => <<"Balance: 0=fitness-only sampling, 0.7=diversity-weighted">>
    };
l0_hyperparameter_spec(archive_recency_decay) ->
    #{
        name => archive_recency_decay,
        default => 0.95,
        range => {0.8, 0.99},
        description => <<"Age decay factor per generation (higher=slower decay)">>
    };
l0_hyperparameter_spec(_) ->
    undefined.

%% @doc Get default values for L0 hyperparameters.
-spec get_l0_defaults() -> map().
get_l0_defaults() ->
    #{
        %% Evolution hyperparameters
        mutation_rate_min => 0.01,
        mutation_rate_max => 0.50,
        topology_mutation_boost => 1.5,
        exploitation_vs_exploration => 0.5,
        stagnation_velocity_threshold => 0.001,
        complexity_penalty_weight => 0.1,
        diversity_bonus_weight => 0.1,
        resource_sensitivity => 0.5,
        %% Self-play archive hyperparameters
        archive_threshold_min => 0.3,
        archive_threshold_max => 0.95,
        archive_diversity_weight => 0.3,
        archive_recency_decay => 0.95
    }.

%% @doc Get bounds for L0 hyperparameters.
-spec get_l0_bounds() -> map().
get_l0_bounds() ->
    #{
        %% Evolution hyperparameters
        mutation_rate_min => {0.001, 0.05},
        mutation_rate_max => {0.3, 0.8},
        topology_mutation_boost => {1.0, 3.0},
        exploitation_vs_exploration => {0.0, 1.0},
        stagnation_velocity_threshold => {0.0001, 0.01},
        complexity_penalty_weight => {0.0, 0.5},
        diversity_bonus_weight => {0.0, 0.5},
        resource_sensitivity => {0.0, 1.0},
        %% Self-play archive hyperparameters
        archive_threshold_min => {0.1, 0.5},
        archive_threshold_max => {0.8, 0.99},
        archive_diversity_weight => {0.0, 0.7},
        archive_recency_decay => {0.8, 0.99}
    }.

%%% ============================================================================
%%% L1 Hyperparameters (7 params, tuned by L2)
%%% ============================================================================

%% @doc List of L1 hyperparameter names.
%%
%% Note: These are called "meta-parameters" from L1's perspective,
%% but "hyperparameters" from L2's perspective.
-spec l1_hyperparameters() -> [atom()].
l1_hyperparameters() ->
    [
        aggression_factor,         % 1. Response intensity to stagnation
        exploration_step,          % 2. Max boost per tau cycle
        stagnation_sensitivity,    % 3. Detection threshold
        topology_aggression,       % 4. Topology mutation boost
        exploitation_weight,       % 5. Explore/exploit balance
        adaptation_momentum,       % 6. Smooth L1 adjustments
        improvement_patience       % 7. Tau cycles before escalating
    ].

%% @doc Get specification for an L1 hyperparameter.
-spec l1_hyperparameter_spec(atom()) -> map() | undefined.
l1_hyperparameter_spec(aggression_factor) ->
    #{
        name => aggression_factor,
        default => 0.5,
        range => {0.0, 2.0},
        description => <<"L1's response intensity to detected stagnation">>
    };
l1_hyperparameter_spec(exploration_step) ->
    #{
        name => exploration_step,
        default => 0.1,
        range => {0.05, 0.5},
        description => <<"Maximum parameter boost L1 can apply per tau cycle">>
    };
l1_hyperparameter_spec(stagnation_sensitivity) ->
    #{
        name => stagnation_sensitivity,
        default => 0.001,
        range => {0.0001, 0.01},
        description => <<"L1's threshold for detecting stagnation">>
    };
l1_hyperparameter_spec(topology_aggression) ->
    #{
        name => topology_aggression,
        default => 1.5,
        range => {1.0, 3.0},
        description => <<"L1's topology mutation boost factor">>
    };
l1_hyperparameter_spec(exploitation_weight) ->
    #{
        name => exploitation_weight,
        default => 0.5,
        range => {0.2, 0.8},
        description => <<"L1's preference: 0.2=explore, 0.8=exploit">>
    };
l1_hyperparameter_spec(adaptation_momentum) ->
    #{
        name => adaptation_momentum,
        default => 0.3,
        range => {0.0, 0.9},
        description => <<"Momentum for smoothing L1's adjustments">>
    };
l1_hyperparameter_spec(improvement_patience) ->
    #{
        name => improvement_patience,
        default => 5,
        range => {1, 20},
        description => <<"Tau cycles L1 waits before escalating response">>
    };
l1_hyperparameter_spec(_) ->
    undefined.

%% @doc Get default values for L1 hyperparameters.
-spec get_l1_defaults() -> map().
get_l1_defaults() ->
    #{
        aggression_factor => 0.5,
        exploration_step => 0.1,
        stagnation_sensitivity => 0.001,
        topology_aggression => 1.5,
        exploitation_weight => 0.5,
        adaptation_momentum => 0.3,
        improvement_patience => 5
    }.

%% @doc Get bounds for L1 hyperparameters.
-spec get_l1_bounds() -> map().
get_l1_bounds() ->
    #{
        aggression_factor => {0.0, 2.0},
        exploration_step => {0.05, 0.5},
        stagnation_sensitivity => {0.0001, 0.01},
        topology_aggression => {1.0, 3.0},
        exploitation_weight => {0.2, 0.8},
        adaptation_momentum => {0.0, 0.9},
        improvement_patience => {1, 20}
    }.
