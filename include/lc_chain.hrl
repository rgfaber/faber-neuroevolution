%% @doc Records for the Chained LTC Controller.
%%
%% The LC Chain consists of three LTC TWEANN levels that cascade:
%% L2 (Strategic) → L1 (Tactical) → L0 (Reactive) → Hyperparameters
%%
%% @author R.G. Lefever
%% @copyright 2024-2026 R.G. Lefever

-ifndef(LC_CHAIN_HRL).
-define(LC_CHAIN_HRL, true).

%%% ============================================================================
%%% Time Constants
%%% ============================================================================

%% Time constants for each level (generations as time unit)
%% Higher τ = slower adaptation = more stable
-define(LC_L2_TAU, 100.0).  % Strategic: very slow
-define(LC_L1_TAU, 50.0).   % Tactical: medium
-define(LC_L0_TAU, 10.0).   % Reactive: fast

%%% ============================================================================
%%% Chain Configuration
%%% ============================================================================

%% @doc Configuration for the LC chain.
-record(lc_chain_config, {
    %% Time constants (can override defaults)
    l2_tau = ?LC_L2_TAU :: float(),
    l1_tau = ?LC_L1_TAU :: float(),
    l0_tau = ?LC_L0_TAU :: float(),

    %% Learning rate for weight updates
    learning_rate = 0.001 :: float(),

    %% Whether to evolve LC network topologies
    evolve_topology = true :: boolean(),

    %% Initial hidden layer sizes for each level
    %% Format: [Layer1Size, Layer2Size, ...]
    l2_hidden_layers = [8, 4] :: [pos_integer()],
    l1_hidden_layers = [6, 4] :: [pos_integer()],
    l0_hidden_layers = [10, 6] :: [pos_integer()],

    %% Activation function for LTC neurons
    activation = tanh :: tanh | sigmoid | relu,

    %% Reward discount factor (for temporal credit assignment)
    gamma = 0.95 :: float()
}).

-type lc_chain_config() :: #lc_chain_config{}.

%%% ============================================================================
%%% Chain State
%%% ============================================================================

%% @doc State of a single LC level (L0, L1, or L2).
-record(lc_level_state, {
    %% Level identifier
    level :: l0 | l1 | l2,

    %% Agent ID in genotype DB (for TWEANN operations)
    agent_id :: term(),

    %% Time constant for this level
    tau :: float(),

    %% Internal LTC state for each neuron
    %% Map: NeuronId => InternalState (float)
    neuron_states = #{} :: #{term() => float()},

    %% Last output values (cached for chaining)
    last_outputs = [] :: [float()],

    %% Connected sensor IDs (for emergent sensor tracking in L0)
    connected_sensors = [] :: [atom()],

    %% Generation counter
    generation = 0 :: non_neg_integer()
}).

-type lc_level_state() :: #lc_level_state{}.

%% @doc State of the complete LC chain.
-record(lc_chain_state, {
    %% Configuration
    config :: lc_chain_config(),

    %% States for each level
    l2_state :: lc_level_state(),
    l1_state :: lc_level_state(),
    l0_state :: lc_level_state(),

    %% Last computed hyperparameters
    last_hyperparams = #{} :: #{atom() => float()},

    %% Training history for reward computation
    %% List of {Generation, Reward} tuples
    reward_history = [] :: [{non_neg_integer(), float()}],

    %% Cumulative reward for current episode
    cumulative_reward = 0.0 :: float(),

    %% Running flag
    running = false :: boolean()
}).

-type lc_chain_state() :: #lc_chain_state{}.

%%% ============================================================================
%%% Evolution Metrics (L2 Inputs)
%%% ============================================================================

%% @doc Evolution metrics passed to L2.
%% These are normalized to [0, 1] range before feeding to network.
-record(evolution_metrics, {
    best_fitness = 0.0 :: float(),
    avg_fitness = 0.0 :: float(),
    fitness_improvement = 0.0 :: float(),
    fitness_variance = 0.0 :: float(),
    stagnation_counter = 0 :: non_neg_integer(),
    generation_progress = 0.0 :: float(),  % current_gen / max_gen
    population_diversity = 0.0 :: float(),
    species_count = 1 :: pos_integer()
}).

-type evolution_metrics() :: #evolution_metrics{}.

%%% ============================================================================
%%% Emergent Metrics (L0 Additional Inputs)
%%% ============================================================================

%% @doc Emergent metrics from the model under training.
%% These are available for L0 to sense via topology evolution.
-record(emergent_metrics, {
    %% Convergence metrics
    convergence_rate = 0.0 :: float(),
    fitness_plateau_duration = 0 :: non_neg_integer(),

    %% Current hyperparameter feedback
    current_mutation_rate = 0.1 :: float(),
    current_selection_ratio = 0.2 :: float(),

    %% Population dynamics
    survival_rate = 0.0 :: float(),
    offspring_rate = 0.0 :: float(),
    elite_age = 0 :: non_neg_integer(),

    %% Topology metrics
    complexity_trend = 0.0 :: float(),
    avg_network_size = 0.0 :: float(),

    %% Species metrics
    species_extinction_rate = 0.0 :: float(),
    species_creation_rate = 0.0 :: float(),

    %% Innovation metrics
    innovation_rate = 0.0 :: float(),
    diversity_index = 0.0 :: float()
}).

-type emergent_metrics() :: #emergent_metrics{}.

%%% ============================================================================
%%% Hyperparameter Output
%%% ============================================================================

%% @doc Hyperparameters output by L0.
%% These control the model under training.
-record(lc_hyperparams, {
    mutation_rate = 0.1 :: float(),         % [0.01, 0.5]
    mutation_strength = 0.3 :: float(),     % [0.05, 1.0]
    selection_ratio = 0.2 :: float(),       % [0.1, 0.5]
    add_node_rate = 0.03 :: float(),        % [0.0, 0.1]
    add_connection_rate = 0.05 :: float()   % [0.0, 0.2]
}).

-type lc_hyperparams() :: #lc_hyperparams{}.

%%% ============================================================================
%%% Default Hyperparameters
%%% ============================================================================

-define(LC_DEFAULT_HYPERPARAMS, #lc_hyperparams{
    mutation_rate = 0.1,
    mutation_strength = 0.3,
    selection_ratio = 0.2,
    add_node_rate = 0.03,
    add_connection_rate = 0.05
}).

-endif. %% LC_CHAIN_HRL
