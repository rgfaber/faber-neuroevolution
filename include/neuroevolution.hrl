%% @doc Neuroevolution record definitions
%% @author R.G. Lefever
%% @copyright 2024-2026 R.G. Lefever

-ifndef(NEUROEVOLUTION_HRL).
-define(NEUROEVOLUTION_HRL, true).

%% Include faber_tweann records for NEAT topology evolution
%% Provides: #connection_gene{}, #sensor{}, #neuron{}, #actuator{}, etc.
-include_lib("faber_tweann/include/records.hrl").

%%% ============================================================================
%%% Types
%%% ============================================================================

%% Note: generation() and fitness() types are provided by faber_tweann/include/types.hrl
%% included via records.hrl above. We use their definitions for compatibility.

-type individual_id() :: term().
-type network() :: term().  %% network_evaluator:network()
-type metrics() :: map().
-type innovation_number() :: pos_integer().

%%% ============================================================================
%%% Genome Record (NEAT Topology Evolution)
%%% ============================================================================

%% @doc Represents the genetic makeup of an individual for NEAT evolution.
%%
%% The genome stores connection genes with innovation numbers that enable
%% meaningful crossover between networks with different topologies.
%%
%% Reference: Stanley, K.O. & Miikkulainen, R. (2002). "Evolving Neural
%% Networks through Augmenting Topologies." Evolutionary Computation, 10(2).
-record(genome, {
    %% Connection genes with innovation numbers (from faber_tweann)
    %% Each connection_gene has: {innovation, from_id, to_id, weight, enabled}
    connection_genes = [] :: [#connection_gene{}],

    %% Node count tracking (for complexity metrics)
    input_count = 0 :: non_neg_integer(),
    hidden_count = 0 :: non_neg_integer(),
    output_count = 0 :: non_neg_integer()
}).

-type genome() :: #genome{}.

%%% ============================================================================
%%% Individual Record
%%% ============================================================================

%% @doc Represents a single individual in the population.
%% Each individual has a neural network, genome, and fitness metrics.
-record(individual, {
    %% Unique identifier for this individual
    id :: individual_id(),

    %% Neural network (from network_evaluator) - used for evaluation
    network :: network(),

    %% Genome for NEAT topology evolution (connection genes with innovations)
    %% undefined for fixed-topology (backward compatibility)
    genome :: genome() | undefined,

    %% Lineage tracking
    parent1_id :: individual_id() | undefined,
    parent2_id :: individual_id() | undefined,

    %% Fitness and performance metrics
    fitness = 0.0 :: fitness(),
    metrics = #{} :: metrics(),

    %% Generation this individual was created in
    generation_born = 1 :: generation(),

    %% Lifespan fields for continuous/steady-state evolution
    %% birth_evaluation: Total evaluations counter when this individual was born
    %% Used to calculate current age: current_total_evaluations - birth_evaluation
    birth_evaluation = 0 :: non_neg_integer(),

    %% max_age: Maximum lifespan in evaluations (inherited, evolvable trait)
    %% When age >= max_age, the individual is eligible for replacement
    %% Inherited from parents with small mutations during breeding
    max_age = 5000 :: pos_integer(),

    %% Flags for visualization/tracking
    is_survivor = false :: boolean(),
    is_offspring = false :: boolean()
}).

-type individual() :: #individual{}.

%%% ============================================================================
%%% Speciation Records
%%% ============================================================================

-type species_id() :: pos_integer().

%% @doc Represents a species - a cluster of genetically similar individuals.
%%
%% Species enable niching: individuals compete primarily within their species,
%% allowing diverse strategies to coexist and explore different fitness peaks.
-record(species, {
    %% Unique species identifier
    id :: species_id(),

    %% Representative individual (used for compatibility comparisons)
    %% New individuals are compared against this to determine species membership
    representative :: individual(),

    %% Member individual IDs in this species
    members = [] :: [individual_id()],

    %% Best fitness achieved by any member this generation
    best_fitness = 0.0 :: fitness(),

    %% Best fitness ever achieved by this species
    best_fitness_ever = 0.0 :: fitness(),

    %% Generation this species was created
    generation_created :: generation(),

    %% Number of generations this species has existed
    age = 0 :: non_neg_integer(),

    %% Number of generations without fitness improvement
    stagnant_generations = 0 :: non_neg_integer(),

    %% Offspring quota for next generation (based on relative fitness)
    offspring_quota = 0 :: non_neg_integer()
}).

-type species() :: #species{}.

%% @doc Configuration for speciation behavior.
%%
%% Uses NEAT compatibility distance formula:
%% delta = (c1 * E / N) + (c2 * D / N) + (c3 * W)
%% Where:
%%   E = excess genes, D = disjoint genes, N = genome size, W = avg weight diff
%%
%% Reference: Stanley, K.O. & Miikkulainen, R. (2002). "Evolving Neural
%% Networks through Augmenting Topologies." Evolutionary Computation, 10(2).
-record(speciation_config, {
    %% Enable/disable speciation (default: disabled for backwards compatibility)
    enabled = false :: boolean(),

    %% Compatibility threshold - individuals with distance below this are same species
    %% Lower values = more species, higher values = fewer species
    compatibility_threshold = 3.0 :: float(),

    %% NEAT compatibility distance coefficients
    %% c1: Weight for excess genes (beyond the other's max innovation)
    c1_excess = 1.0 :: float(),

    %% c2: Weight for disjoint genes (within the other's range but missing)
    c2_disjoint = 1.0 :: float(),

    %% c3: Weight for average weight difference in matching genes
    c3_weight_diff = 0.4 :: float(),

    %% Target number of species (threshold adjusts dynamically to maintain)
    target_species = 5 :: pos_integer(),

    %% How much to adjust threshold when species count is off target
    threshold_adjustment_rate = 0.1 :: float(),

    %% Minimum species size (species below this may be eliminated)
    min_species_size = 2 :: pos_integer(),

    %% Maximum generations a species can stagnate before elimination
    %% (0 = never eliminate for stagnation)
    max_stagnation = 15 :: non_neg_integer(),

    %% Fraction of each species that survives selection
    species_elitism = 0.20 :: float(),

    %% Probability of interspecies breeding (usually low)
    interspecies_mating_rate = 0.001 :: float()
}).

-type speciation_config() :: #speciation_config{}.

%% @doc Event recording species lifecycle changes.
-record(species_event, {
    generation :: generation(),
    species_id :: species_id(),
    event_type :: species_created | species_extinct | species_stagnant | champion_emerged,
    details :: map()
}).

-type species_event() :: #species_event{}.

%%% ============================================================================
%%% Mutation Configuration Record (NEAT Topology Evolution)
%%% ============================================================================

%% @doc Configuration for genome mutations (structural + weight).
%%
%% Controls the probabilities and parameters for NEAT-style mutations.
%% Structural mutations (add_node, add_connection) are less frequent than
%% weight mutations to allow time for new structures to optimize.
%%
%% Reference: Stanley, K.O. & Miikkulainen, R. (2002).
-record(mutation_config, {
    %% Weight mutation parameters
    weight_mutation_rate = 0.80 :: float(),     %% 80% of offspring get weight mutation
    weight_perturb_rate = 0.90 :: float(),      %% 90% perturbed, 10% replaced entirely
    weight_perturb_strength = 0.3 :: float(),   %% Gaussian std dev for perturbation

    %% Structural mutation rates (per reproduction event)
    add_node_rate = 0.03 :: float(),            %% 3% chance to add node (split connection)
    add_connection_rate = 0.05 :: float(),      %% 5% chance to add new connection
    toggle_connection_rate = 0.01 :: float(),   %% 1% chance to enable/disable connection

    %% Sensor/actuator mutation (rare - changes morphology)
    add_sensor_rate = 0.001 :: float(),         %% 0.1% chance to add sensor
    add_actuator_rate = 0.001 :: float(),       %% 0.1% chance to add actuator

    %% LTC neuron mutations (when LTC neurons are present)
    mutate_neuron_type_rate = 0.01 :: float(),  %% 1% chance to switch standard/ltc/cfc
    mutate_time_constant_rate = 0.05 :: float() %% 5% chance to perturb tau
}).

-type mutation_config() :: #mutation_config{}.

%%% ============================================================================
%%% Self-Play Configuration Record
%%% ============================================================================

%% @doc Configuration for self-play evaluation mode.
%%
%% When enabled, individuals are evaluated against opponents from an archive
%% of previous top performers, creating arms-race dynamics.
%%
%% Pure self-play: No heuristic opponents. When archive is empty (first
%% generation), individuals compete against each other within the batch.
%% This is how systems like AlphaZero and OpenAI Five work.
%%
%% The archive is managed by self_play_manager (Erlang) and can optionally
%% sync across the mesh network for distributed self-play training.
-record(self_play_config, {
    %% Enable self-play mode (default: false)
    enabled = false :: boolean(),

    %% Maximum opponents in archive (default: 50)
    archive_size = 50 :: pos_integer(),

    %% Fitness threshold for archive entry: float() | auto (default: auto)
    %% - auto: Uses 50% of average archive fitness as threshold
    %% - float(): Fixed minimum fitness to enter archive
    archive_threshold = auto :: float() | auto,

    %% Minimum fitness percentile to enter archive (default: 0.5)
    %% Only top N% of population can be considered for archive
    min_fitness_percentile = 0.5 :: float()
}).

-type self_play_config() :: #self_play_config{}.

%%% ============================================================================
%%% Configuration Record
%%% ============================================================================

%% @doc Configuration for the neuroevolution server.
%% Controls population size, selection, mutation, and evaluation.
-record(neuro_config, {
    %% Population size (number of individuals)
    population_size = 50 :: pos_integer(),

    %% Number of evaluations per individual per generation
    evaluations_per_individual = 10 :: pos_integer(),

    %% Selection ratio (fraction of population that survives)
    %% e.g., 0.20 means top 20% survive
    selection_ratio = 0.20 :: float(),

    %% Legacy mutation parameters (used when mutation_config is undefined)
    mutation_rate = 0.10 :: float(),      %% Probability of mutating each weight
    mutation_strength = 0.3 :: float(),   %% Magnitude of weight perturbation

    %% Layer-specific mutation rates (optional - defaults to mutation_rate/strength)
    %% When set, reservoir (hidden) and readout (output) layers can have different rates.
    %% Rationale: Readout often benefits from faster adaptation while reservoir dynamics
    %% should be more stable. See guides/training-strategies.md for details.
    reservoir_mutation_rate :: float() | undefined,      %% Hidden layer mutation rate
    reservoir_mutation_strength :: float() | undefined,  %% Hidden layer mutation strength
    readout_mutation_rate :: float() | undefined,        %% Output layer mutation rate
    readout_mutation_strength :: float() | undefined,    %% Output layer mutation strength

    %% NEAT topology mutation configuration
    %% When set, enables full topology evolution with structural mutations
    %% When undefined, uses legacy weight-only mutation via mutation_rate/strength
    topology_mutation_config :: mutation_config() | undefined,

    %% Maximum evaluations (infinity for unlimited)
    %% PRIMARY stopping condition - preferred over max_generations
    %% Counts individual evaluations, not cohort/generation iterations
    max_evaluations = infinity :: pos_integer() | infinity,

    %% Maximum generations/cohorts (infinity for unlimited)
    %% DEPRECATED: Use max_evaluations instead for evaluation-centric control
    max_generations = infinity :: pos_integer() | infinity,

    %% Target fitness threshold (undefined for no threshold)
    %% Training stops when best fitness reaches or exceeds this value
    target_fitness = undefined :: float() | undefined,

    %% Network topology: {InputSize, HiddenLayers, OutputSize}
    %% e.g., {42, [16, 8], 6} for 42 inputs, 2 hidden layers (16, 8), 6 outputs
    network_topology :: {pos_integer(), [pos_integer()], pos_integer()},

    %% Evaluator module (must implement neuroevolution_evaluator behaviour)
    evaluator_module :: module(),

    %% Options passed to evaluator
    evaluator_options = #{} :: map(),

    %% Event handler for notifications: {Module, InitArg} | undefined
    %% Module must export handle_event/2
    %% DEPRECATED: Use event publishing instead (neuroevolution_events)
    event_handler :: {module(), term()} | undefined,

    %% Meta-controller configuration: meta_config() | undefined
    %% When set, an LTC meta-controller will dynamically adjust hyperparameters
    %% See meta_controller.hrl for meta_config record definition
    meta_controller_config :: term() | undefined,

    %% Speciation configuration: speciation_config() | undefined
    %% When set, enables NEAT-style speciation for niche-based evolution
    speciation_config :: speciation_config() | undefined,

    %% Self-play configuration: self_play_config() | undefined
    %% When set, enables self-play evaluation mode where individuals compete
    %% against archived opponents from previous generations (arms race dynamics)
    %% See self_play_manager.erl for the evaluation mode manager
    self_play_config :: self_play_config() | undefined,

    %% Realm for multi-tenancy and event topic scoping
    %% Events are published to "neuro.<realm>.<event_type>"
    realm = <<"default">> :: binary(),

    %% Whether to publish events via neuroevolution_events
    %% When true, events are published to topics in addition to (or instead of) callbacks
    publish_events = false :: boolean(),

    %% Evaluation mode: direct | distributed | mesh
    %% - direct: Call evaluator module directly (default, single-node)
    %% - distributed: Publish evaluate_request events, await results via events
    %%   Requires evaluator workers subscribed to the realm's evaluate topic
    %% - mesh: Use macula mesh for load-balanced distributed evaluation
    %%   Uses distributed_evaluator for automatic node selection and retry
    evaluation_mode = direct :: direct | distributed | mesh,

    %% Mesh configuration (for evaluation_mode = mesh)
    %% When set, enables mesh distribution with macula for distributed evaluation
    %% See mesh_sup.erl for the mesh supervisor configuration
    mesh_config :: map() | undefined,

    %% Timeout for distributed evaluation (milliseconds)
    %% How long to wait for all evaluation results before timing out
    evaluation_timeout = 30000 :: pos_integer(),

    %% Maximum concurrent evaluations (for memory/CPU management)
    %% Default: 2x the number of schedulers (CPU cores)
    %% Set lower if running into memory issues with large networks
    max_concurrent_evaluations = undefined :: pos_integer() | undefined,

    %% Strategy configuration for pluggable evolution strategies
    %% When set, the server delegates evolution logic to the strategy module
    %% See evolution_strategy.hrl for strategy_config record definition
    strategy_config :: term() | undefined,

    %% Chained LTC Controller configuration: lc_chain_config() | undefined
    %% When set, uses the L2→L1→L0 chained LTC architecture for hyperparameter control
    %% This supersedes both task_silo (L1) and meta_controller (legacy L2)
    %% See lc_chain.hrl for lc_chain_config record definition
    lc_chain_config :: term() | undefined,

    %% Checkpoint interval for continuous evolution mode
    %% How often to emit progress_checkpoint events (in evaluations)
    %% Default: population_size (emit after each full "cohort" worth of evaluations)
    %% Set to a fixed value for consistent checkpoint frequency
    checkpoint_interval :: pos_integer() | undefined,

    %% Network checkpoint configuration
    %% When set, networks are saved to disk at key milestones.
    %% Options:
    %% - checkpoint_dir: Directory for checkpoints (default: "_checkpoints")
    %% - save_on_fitness_record: Save when new best fitness (default: true)
    %% - generation_interval: Save every N generations (0 = disabled)
    %% - max_checkpoints_per_reason: Max checkpoints to keep per reason (default: 20)
    checkpoint_config :: map() | undefined
}).

-type neuro_config() :: #neuro_config{}.

%%% ============================================================================
%%% Statistics Records
%%% ============================================================================

%% @doc Statistics for a completed generation.
-record(generation_stats, {
    %% Generation number
    generation :: generation(),

    %% Fitness metrics
    best_fitness :: fitness(),
    avg_fitness :: fitness(),
    worst_fitness :: fitness(),

    %% Best individual ID
    best_individual_id :: individual_id(),

    %% Lists of individual IDs (NOT full individuals to save memory)
    survivors :: [individual_id()],
    eliminated :: [individual_id()],
    offspring :: [individual_id()],

    %% Population size (for stats without needing full lists)
    population_size = 0 :: non_neg_integer(),
    survivor_count = 0 :: non_neg_integer(),

    %% Top N individuals with summary info (for UI visualization)
    %% Contains lightweight summaries, not full network weights
    top_individuals = [] :: [#{id := individual_id(), fitness := fitness(), complexity := non_neg_integer()}]
}).

-type generation_stats() :: #generation_stats{}.

%% @doc Records a breeding event (two parents producing offspring).
-record(breeding_event, {
    parent1_id :: individual_id(),
    parent2_id :: individual_id(),
    child_id :: individual_id(),
    generation :: generation()
}).

-type breeding_event() :: #breeding_event{}.

%% @doc Competitive dynamics entry for arms race visualization.
%% Captures fitness distribution metrics for each generation.
-type competitive_entry() :: #{
    generation := generation(),
    best_fitness := fitness(),
    avg_fitness := fitness(),
    worst_fitness := fitness(),
    %% Top 10% elite performance
    top_10_avg := fitness(),
    %% Bottom 10% baseline performance
    bottom_10_avg := fitness(),
    %% Fitness variance (population diversity)
    fitness_variance := float(),
    %% Competitive pressure: gap between elite and baseline
    competitive_pressure := float(),
    %% Archive stats (0 for non-self-play scenarios)
    archive_size := non_neg_integer(),
    archive_avg := fitness()
}.

%%% ============================================================================
%%% Server State Record
%%% ============================================================================

%% @doc Internal state for neuroevolution_server.
-record(neuro_state, {
    %% Server identifier
    id :: term(),

    %% Configuration
    config :: neuro_config(),

    %% Current population
    population = [] :: [individual()],

    %% Total evaluations completed (PRIMARY progress metric)
    %% Counts individual evaluations across all cohorts
    total_evaluations = 0 :: non_neg_integer(),

    %% Current cohort number (formerly "generation")
    %% Retained for lineage tracking and backward compatibility
    %% DEPRECATED as primary progress metric - use total_evaluations instead
    generation = 1 :: generation(),

    %% Running state
    running = false :: boolean(),
    evaluating = false :: boolean(),

    %% Progress tracking
    games_completed = 0 :: non_neg_integer(),
    total_games = 0 :: non_neg_integer(),

    %% Historical stats
    best_fitness_ever = 0.0 :: fitness(),
    last_gen_best = 0.0 :: fitness(),
    last_gen_avg = 0.0 :: fitness(),
    generation_history = [] :: [{generation(), fitness(), fitness()}],

    %% Last generation results
    last_gen_results :: map() | undefined,
    breeding_events = [] :: [breeding_event()],

    %% Competitive dynamics history for arms race visualization
    %% Tracks fitness distribution metrics each generation
    competitive_history = [] :: [competitive_entry()],

    %% Evaluation task reference
    eval_task :: reference() | undefined,

    %% Meta-controller process (LTC-based hyperparameter optimizer)
    %% When active, this gen_server dynamically adjusts hyperparameters
    %% DEPRECATED: Use lc_chain for new implementations
    meta_controller :: pid() | undefined,

    %% Chained LTC Controller process (L2→L1→L0 hierarchy)
    %% When active, provides hyperparameters via cascaded LTC networks
    %% Supersedes both task_silo (L1) and meta_controller (legacy L2)
    lc_chain :: pid() | undefined,

    %% Self-play manager PID (when self_play_config is set)
    %% Manages opponent archive and provides opponents for evaluation
    self_play_manager :: pid() | undefined,

    %% Evolution strategy module and state
    %% The strategy module handles all evolution logic (selection, breeding, etc.)
    strategy_module :: module() | undefined,
    strategy_state :: term() | undefined,

    %% Speciation state
    species = [] :: [species()],
    species_events = [] :: [species_event()],
    next_species_id = 1 :: species_id(),

    %% Distributed evaluation state
    %% Maps request_id => {individual_id, individual_record}
    pending_evaluations = #{} :: #{reference() => {individual_id(), individual()}},
    %% Timer ref for evaluation timeout
    eval_timeout_ref :: reference() | undefined,

    %% Continuous evolution checkpoint tracking
    %% Timestamp of last progress_checkpoint emission (milliseconds)
    last_checkpoint_time :: non_neg_integer() | undefined,
    %% Total evaluations at last checkpoint (for evaluations_since_last calculation)
    last_checkpoint_evals = 0 :: non_neg_integer(),

    %% Cached recommendations from silos (event-driven read model pattern)
    %% These are updated via silo_events when silos publish changes,
    %% eliminating blocking get_recommendations() calls from the event flow.
    cached_resource_recommendations = #{} :: map(),
    cached_task_recommendations = #{} :: map()
}).

-type neuro_state() :: #neuro_state{}.

%%% ============================================================================
%%% Event Types
%%% ============================================================================

%% Events sent to event_handler
-type neuro_event() ::
    {generation_started, generation()} |
    {evaluation_progress, generation(), non_neg_integer(), non_neg_integer()} |
    {generation_complete, generation_stats()} |
    {training_started, neuro_config()} |
    {training_stopped, generation()} |
    {training_complete, training_result()} |
    {species_created, species_event()} |
    {species_extinct, species_event()} |
    {species_stagnant, species_event()} |
    {speciation_update, [species()]}.

%% @doc Training completion result.
%% Contains the reason for stopping and the best individual found.
-type training_result() :: #{
    reason := target_fitness_reached | max_evaluations_reached | max_generations_reached | stopped,
    total_evaluations := non_neg_integer(),
    generation := generation(),  %% Cohort number (backward compat)
    best_fitness := fitness(),
    best_individual := individual()
}.

-endif.
