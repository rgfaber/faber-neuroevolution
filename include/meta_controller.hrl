%% @doc Meta-Controller record definitions for LTC-based hyperparameter optimization
%%
%% This module implements a meta-learning system that uses Liquid Time-Constant
%% (LTC) networks to dynamically control neuroevolution hyperparameters.
%%
%% The meta-controller operates at a higher timescale than the task networks,
%% learning to "optimize the optimizer" by adapting parameters based on
%% training dynamics.
%%
%% @author R.G. Lefever
%% @copyright 2024-2026 R.G. Lefever

-ifndef(META_CONTROLLER_HRL).
-define(META_CONTROLLER_HRL, true).

%%% ============================================================================
%%% Types
%%% ============================================================================

-type meta_param() :: mutation_rate | mutation_strength | selection_ratio |
                      %% Resource-aware parameters
                      evaluations_per_individual | max_concurrent_evaluations |
                      population_size |
                      %% Topology control parameters
                      add_node_rate | add_connection_rate | complexity_penalty.
-type reward_component() :: convergence_speed | final_fitness | efficiency_ratio |
                           diversity_aware | normative_structure.

%%% ============================================================================
%%% Configuration Record
%%% ============================================================================

%% @doc Configuration for the meta-controller.
%%
%% Controls the LTC network architecture and reward signal composition.
%%
%% Inputs (11 total):
%%   1-8: Evolution metrics (fitness, improvement, variance, stagnation, progress, trend, diversity, entropy)
%%   9-11: Resource metrics (memory_pressure, cpu_pressure, process_pressure)
%%
%% Outputs (5 base, more if topology/population control enabled):
%%   1-3: Base evolution params (mutation_rate, mutation_strength, selection_ratio)
%%   4-5: Resource-aware params (evaluations_per_individual, max_concurrent_evaluations)
%%   6-8: Topology params if control_topology=true (add_node_rate, add_connection_rate, complexity_penalty)
%%   9: Population size if control_population_size=true
-record(meta_config, {
    %% Network topology: {Inputs, HiddenLayers, Outputs}
    %% Default: {11, [24, 16, 8], 5} - 11 inputs (8 evolution + 3 resource), 5 outputs (3 base + 2 resource)
    network_topology = {11, [24, 16, 8], 5} :: {pos_integer(), [pos_integer()], pos_integer()},

    %% LTC neuron type: cfc (fast) or ltc (accurate ODE)
    neuron_type = cfc :: ltc | cfc,

    %% Base time constant for meta-controller neurons
    %% Higher values = slower adaptation = more stable
    time_constant = 50.0 :: float(),

    %% State bound for LTC neurons
    state_bound = 1.0 :: float(),

    %% Reward component weights (must sum to ~1.0)
    reward_weights = #{
        convergence_speed => 0.25,
        final_fitness => 0.25,
        efficiency_ratio => 0.20,
        diversity_aware => 0.15,
        normative_structure => 0.15
    } :: #{reward_component() => float()},

    %% Learning rate for gradient-based meta-training
    learning_rate = 0.001 :: float(),

    %% Parameter bounds: {ParamName, {Min, Max}}
    param_bounds = #{
        %% Base evolution parameters
        mutation_rate => {0.01, 0.5},
        mutation_strength => {0.05, 1.0},
        selection_ratio => {0.10, 0.50},
        %% Resource-aware parameters (always controlled)
        evaluations_per_individual => {1.0, 20.0},   %% 1 = emergency low-memory mode
        max_concurrent_evaluations => {1.0, 50.0},   %% Limits parallelism under load
        %% Population control (when control_population_size=true)
        population_size => {10.0, 200.0},
        %% Topology control bounds (NEAT mode)
        add_node_rate => {0.0, 0.10},
        add_connection_rate => {0.0, 0.20},
        complexity_penalty => {0.0, 0.5}
    } :: #{meta_param() => {float(), float()}},

    %% Whether to include population_size as a controllable parameter
    %% (disabled by default as it affects computation more than other params)
    control_population_size = false :: boolean(),

    %% Whether to control topology mutation rates (NEAT mode)
    %% When true, adds 3 outputs: add_node_rate, add_connection_rate, complexity_penalty
    %% Requires additional inputs for topology metrics (complexity, diversity)
    control_topology = false :: boolean(),

    %% History window size for computing reward signals
    history_window = 10 :: pos_integer(),

    %% Momentum for parameter updates (smooths changes)
    momentum = 0.9 :: float()
}).

-type meta_config() :: #meta_config{}.

%%% ============================================================================
%%% State Record
%%% ============================================================================

%% @doc Internal state for the meta-controller gen_server.
-record(meta_state, {
    %% Server identifier
    id :: term(),

    %% Configuration
    config :: meta_config(),

    %% Current hyperparameter values
    current_params = #{
        mutation_rate => 0.10,
        mutation_strength => 0.30,
        selection_ratio => 0.20
    } :: #{meta_param() => float()},

    %% LTC network internal states (one per hidden neuron)
    %% Maps neuron_id => internal_state
    ltc_states = #{} :: #{term() => float()},

    %% LTC network weights
    %% Structure: #{layer => #{neuron_id => [{input_id, weight}]}}
    ltc_weights = #{} :: map(),

    %% Training metrics history (circular buffer of last N generations)
    metrics_history = [] :: [generation_metrics()],

    %% Parameter change momentum (for smooth updates)
    param_momentum = #{} :: #{meta_param() => float()},

    %% Generation counter
    generation = 0 :: non_neg_integer(),

    %% Cumulative reward for current training run
    cumulative_reward = 0.0 :: float(),

    %% Best fitness ever seen (for normalization)
    best_fitness_ever = 0.0 :: float(),

    %% Stagnation counter (generations without improvement)
    stagnation_count = 0 :: non_neg_integer(),

    %% Running state
    running = false :: boolean()
}).

-type meta_state() :: #meta_state{}.

%%% ============================================================================
%%% Generation Metrics Record
%%% ============================================================================

%% @doc Metrics captured from a single generation for meta-learning.
-record(generation_metrics, {
    %% Generation number
    generation :: pos_integer(),

    %% Fitness metrics
    best_fitness :: float(),
    avg_fitness :: float(),
    worst_fitness :: float(),
    fitness_std_dev :: float(),

    %% Improvement metrics
    fitness_delta :: float(),          %% Change from previous generation
    relative_improvement :: float(),   %% Percentage improvement

    %% Diversity metrics
    population_diversity :: float(),   %% Variance in fitness
    strategy_entropy :: float(),       %% Information content of strategies

    %% Efficiency metrics
    evaluations_used :: pos_integer(),
    fitness_per_evaluation :: float(),

    %% Normative structure metrics
    diversity_corridors :: float(),     %% Cluster distinctness
    adaptation_readiness :: float(),    %% Variance in adjacent traits

    %% Current parameters used
    params_used :: #{meta_param() => float()},

    %% Timestamp
    timestamp :: erlang:timestamp()
}).

-type generation_metrics() :: #generation_metrics{}.

%%% ============================================================================
%%% Reward Signal Record
%%% ============================================================================

%% @doc Computed reward signal for meta-controller training.
-record(meta_reward, {
    %% Individual reward components
    convergence_speed :: float(),
    final_fitness :: float(),
    efficiency_ratio :: float(),
    diversity_aware :: float(),
    normative_structure :: float(),

    %% Composite reward (weighted sum)
    total :: float(),

    %% Generation this reward is for
    generation :: pos_integer()
}).

-type meta_reward() :: #meta_reward{}.

%%% ============================================================================
%%% Network Structure Records
%%% ============================================================================

%% @doc LTC neuron state for the meta-controller network.
-record(meta_neuron, {
    %% Neuron identifier: {layer, index}
    id :: {pos_integer(), pos_integer()},

    %% Internal state x(t)
    internal_state = 0.0 :: float(),

    %% Base time constant
    time_constant :: float(),

    %% State bound
    state_bound :: float(),

    %% Input weights: [{source_id, weight}]
    input_weights = [] :: [{term(), float()}],

    %% Bias
    bias = 0.0 :: float(),

    %% Backbone weights for CfC f() function
    backbone_weights = [] :: [float()],

    %% Head weights for CfC h() function
    head_weights = [] :: [float()]
}).

-type meta_neuron() :: #meta_neuron{}.

%% @doc Output mapping for meta-controller.
%% Maps network outputs to parameter adjustments.
-record(output_mapping, {
    %% Output index -> parameter name
    index_to_param :: #{non_neg_integer() => meta_param()},

    %% Parameter name -> output index
    param_to_index :: #{meta_param() => non_neg_integer()},

    %% Activation functions for each output (e.g., sigmoid for bounded params)
    output_activations :: #{non_neg_integer() => atom()}
}).

-type output_mapping() :: #output_mapping{}.

%%% ============================================================================
%%% Training Event Record
%%% ============================================================================

%% @doc Training event for gradient-based meta-learning.
-record(meta_training_event, {
    %% Generation
    generation :: pos_integer(),

    %% Input features (normalized metrics)
    inputs :: [float()],

    %% Output parameters (raw network outputs before scaling)
    outputs :: [float()],

    %% Reward received
    reward :: float(),

    %% Gradient estimate for this step
    gradients :: #{term() => float()}
}).

-type meta_training_event() :: #meta_training_event{}.

%%% ============================================================================
%%% Structure Metrics Record (Normative Awareness)
%%% ============================================================================

%% @doc Structure metrics for normative awareness reward component.
%%
%% These metrics capture whether the population maintains the capacity
%% for future adaptation, not just current fitness.
-record(structure_metrics, {
    %% Number of distinct strategy clusters in population
    diversity_corridors = 0.0 :: float(),

    %% Variance in traits that are adjacent to fitness (exploration potential)
    adaptation_readiness = 0.0 :: float(),

    %% Estimated distance to unexplored fitness regions
    breakthrough_potential = 0.0 :: float(),

    %% Shannon entropy of strategy distribution
    strategy_entropy = 0.0 :: float()
}).

-type structure_metrics() :: #structure_metrics{}.

%%% ============================================================================
%%% L2 Guidance Record (L2→L1 Interface)
%%% ============================================================================

%% @doc L2 guidance output that controls L1 (task_silo) behavior.
%%
%% Instead of L2 directly outputting hyperparameters (mutation_rate, etc.),
%% it outputs meta-parameters that control HOW L1 adjusts hyperparameters.
%% This creates a proper hierarchical control system:
%%
%%   L2 (Strategic) → L1 Guidance → L1 (Tactical) → Hyperparams → L0 (Bounds)
%%
%% Example:
%%   - L2 outputs aggression_factor = 1.5
%%   - L1 uses this: AdjustedMR = BaseMR * (1 + NetFactor * 1.5)
%%   - Result: mutation_rate can increase by up to 150% instead of 50%
-record(l2_guidance, {
    %% =========================================================================
    %% Task Silo L1 Hyperparameters
    %% =========================================================================

    %% How aggressive L1 adjustments should be
    %% Higher = stronger response to stagnation
    %% Range: [0.0, 2.0], Default: 0.5
    aggression_factor = 0.5 :: float(),

    %% How fast exploration_boost increases per stagnating generation
    %% Higher = faster escalation when stuck
    %% Range: [0.05, 0.5], Default: 0.1
    exploration_step = 0.1 :: float(),

    %% Threshold for what counts as "improvement" (below = stagnating)
    %% Lower = more sensitive to small gains
    %% Range: [0.0001, 0.01], Default: 0.001
    stagnation_sensitivity = 0.001 :: float(),

    %% How much to boost topology mutations when heavily stagnating
    %% Higher = more aggressive structural changes
    %% Range: [1.0, 3.0], Default: 1.5
    topology_aggression = 1.5 :: float(),

    %% Weight for exploitation vs exploration in NetFactor calculation
    %% Higher = more exploitation when improving
    %% Range: [0.2, 0.8], Default: 0.5
    exploitation_weight = 0.5 :: float(),

    %% Momentum for smoothing exploration/exploitation boost updates
    %% Higher = smoother transitions, less oscillation
    %% Range: [0.0, 0.95], Default: 0.7
    adaptation_momentum = 0.7 :: float(),

    %% Intervention severity thresholds (stagnation severity -> action level)
    %% warning_threshold: When to emit warning events
    %% Range: [0.2, 0.5], Default: 0.3
    warning_threshold = 0.3 :: float(),

    %% intervention_threshold: When to actively intervene
    %% Range: [0.4, 0.8], Default: 0.6
    intervention_threshold = 0.6 :: float(),

    %% critical_threshold: When to apply maximum intervention
    %% Range: [0.7, 0.99], Default: 0.9
    critical_threshold = 0.9 :: float(),

    %% Velocity window size for stagnation detection
    %% Higher = more stable but slower reaction
    %% Range: [5, 30], Default: 10
    velocity_window_size = 10 :: pos_integer(),

    %% =========================================================================
    %% Resource Silo L1 Hyperparameters
    %% =========================================================================

    %% Memory pressure threshold for throttling
    %% Range: [0.5, 0.85], Default: 0.7
    memory_high_threshold = 0.7 :: float(),

    %% Memory pressure threshold for pausing evolution
    %% Range: [0.8, 0.98], Default: 0.9
    memory_critical_threshold = 0.9 :: float(),

    %% CPU pressure threshold for throttling
    %% Range: [0.6, 0.95], Default: 0.9
    cpu_high_threshold = 0.9 :: float(),

    %% Pressure scale factor for concurrency reduction
    %% Range: [0.5, 0.99], Default: 0.9
    pressure_scale_factor = 0.9 :: float(),

    %% Minimum concurrency scale (floor)
    %% Range: [0.05, 0.3], Default: 0.1
    min_scale_factor = 0.1 :: float(),

    %% Threshold for detecting significant pressure changes
    %% Range: [0.01, 0.2], Default: 0.05
    pressure_change_threshold = 0.05 :: float(),

    %% =========================================================================
    %% Metadata
    %% =========================================================================

    %% Generation this guidance was computed for
    generation = 0 :: non_neg_integer()
}).

-type l2_guidance() :: #l2_guidance{}.

%% @doc Parameter bounds for L2 guidance outputs.
-define(L2_GUIDANCE_BOUNDS, #{
    %% Task Silo L1 hyperparameters
    aggression_factor => {0.0, 2.0},
    exploration_step => {0.05, 0.5},
    stagnation_sensitivity => {0.0001, 0.01},
    topology_aggression => {1.0, 3.0},
    exploitation_weight => {0.2, 0.8},
    adaptation_momentum => {0.0, 0.95},
    warning_threshold => {0.2, 0.5},
    intervention_threshold => {0.4, 0.8},
    critical_threshold => {0.7, 0.99},
    velocity_window_size => {5, 30},
    %% Resource Silo L1 hyperparameters
    memory_high_threshold => {0.5, 0.85},
    memory_critical_threshold => {0.8, 0.98},
    cpu_high_threshold => {0.6, 0.95},
    pressure_scale_factor => {0.5, 0.99},
    min_scale_factor => {0.05, 0.3},
    pressure_change_threshold => {0.01, 0.2}
}).

%% @doc Default L2 guidance (used when L2 is disabled).
%% NOTE: These defaults are AGGRESSIVE to ensure responsiveness when L2 learning
%% is not yet active. Once L0 TWEANNs are wired in, L2 can learn optimal values.
-define(L2_GUIDANCE_DEFAULTS, #l2_guidance{
    %% Task Silo L1 hyperparameters
    %% AGGRESSIVE defaults for rule-based L1 (before TWEANN evolution is wired in)
    aggression_factor = 1.5,         %% Was 0.5: now 3x stronger response to stagnation
    exploration_step = 0.5,          %% Was 0.1: now 5x faster exploration boost ramp
    stagnation_sensitivity = 0.001,  %% Keep sensitive to stagnation
    topology_aggression = 2.5,       %% Was 1.5: more aggressive structural mutations
    exploitation_weight = 0.3,       %% Was 0.5: less exploitation, more exploration
    adaptation_momentum = 0.3,       %% Was 0.7: faster response (30% old, 70% new)
    warning_threshold = 0.2,         %% Was 0.3: warn earlier
    intervention_threshold = 0.4,    %% Was 0.6: intervene earlier
    critical_threshold = 0.7,        %% Was 0.9: critical earlier
    velocity_window_size = 5,        %% Was 10: shorter window = faster reaction
    %% Resource Silo L1 hyperparameters
    memory_high_threshold = 0.7,
    memory_critical_threshold = 0.9,
    cpu_high_threshold = 0.9,
    pressure_scale_factor = 0.9,
    min_scale_factor = 0.1,
    pressure_change_threshold = 0.05,
    %% Metadata
    generation = 0
}).

%%% ============================================================================
%%% Velocity-Based Stagnation State
%%% ============================================================================

%% @doc State for velocity-based stagnation detection.
%%
%% Tracks improvement velocity (fitness gain per 1000 evaluations) instead of
%% counting generations without improvement. This enables:
%%
%% 1. Reaction to slowdown BEFORE full plateau
%% 2. Continuous severity scale (0.0-1.0) instead of binary trigger
%% 3. Compatibility with variable batch sizes and steady-state evolution
%%
%% Algorithm:
%%   velocity = (current_fitness - prev_fitness) / (current_evals - prev_evals) * 1000
%%   avg_velocity = rolling_average(improvement_window)
%%   stagnation_severity = clamp((threshold - avg_velocity) / threshold, 0.0, 1.0)
%%
%% Reference: Evaluation-centric evolution architecture plan.
-record(velocity_state, {
    %% Rolling window of improvement velocities
    improvement_window = [] :: [float()],

    %% Number of checkpoints to track in window
    window_size = 10 :: pos_integer(),

    %% Current improvement velocity (fitness gain per 1000 evaluations)
    current_velocity = 0.0 :: float(),

    %% Velocity threshold: below this = stagnating
    %% Default: 0.001 = 0.1% fitness improvement per 1000 evaluations
    velocity_threshold = 0.001 :: float(),

    %% Fitness checkpoints: [{total_evaluations, best_fitness}]
    %% Used to compute velocity between checkpoints
    fitness_checkpoints = [] :: [{non_neg_integer(), float()}],

    %% Stagnation severity: 0.0 = healthy, 1.0 = critical
    %% Continuous scale enables proportional response
    stagnation_severity = 0.0 :: float(),

    %% Total evaluations at last update
    last_total_evaluations = 0 :: non_neg_integer()
}).

-type velocity_state() :: #velocity_state{}.

%% @doc Default velocity state initializer.
-define(VELOCITY_STATE_DEFAULTS, #velocity_state{
    improvement_window = [],
    window_size = 10,
    current_velocity = 0.0,
    velocity_threshold = 0.001,
    fitness_checkpoints = [],
    stagnation_severity = 0.0,
    last_total_evaluations = 0
}).

-endif. %% META_CONTROLLER_HRL
