%% @doc Evolution strategy type definitions and configuration records.
%%
%% This file defines the types and records used by the evolution_strategy
%% behaviour and its implementations.
%%
%% @author R.G. Lefever
%% @copyright 2024-2026 R.G. Lefever

-ifndef(EVOLUTION_STRATEGY_HRL).
-define(EVOLUTION_STRATEGY_HRL, true).

%% lifecycle_events.hrl includes neuroevolution.hrl
-include("lifecycle_events.hrl").

%%% ============================================================================
%%% Core Type Definitions
%%% ============================================================================

%% Strategy module implementing the evolution_strategy behaviour
-type strategy_module() :: module().

%% Note: individual_id(), fitness(), network() are already defined in neuroevolution.hrl

%% Actions that a strategy can request the server to perform
-type strategy_action() ::
    {create_individual, ParentIds :: [individual_id()], Metadata :: map()} |
    {remove_individual, individual_id(), Reason :: death_reason()} |
    {evaluate_individual, individual_id()} |
    {evaluate_batch, [individual_id()]} |
    {update_config, ConfigUpdates :: map()} |
    {migrate_individual, individual_id(), ToIsland :: island_id()} |
    {update_archive, ArchiveUpdate :: term()} |
    {emit_event, lifecycle_event()} |
    noop.

%% Population snapshot for UI and meta-controller
-type population_snapshot() :: #{
    size := non_neg_integer(),
    individuals := [individual_summary()],
    best_fitness := fitness(),
    avg_fitness := fitness(),
    worst_fitness := fitness(),
    species_count => non_neg_integer(),
    generation => pos_integer(),
    extra => map()
}.

%% Lightweight individual summary for snapshots
-type individual_summary() :: #{
    id := individual_id(),
    fitness := fitness(),
    is_survivor => boolean(),
    is_offspring => boolean(),
    species_id => species_id(),
    age => non_neg_integer()
}.

%%% ============================================================================
%%% Strategy Configuration
%%% ============================================================================

%% @doc Configuration for pluggable evolution strategies.
%%
%% This record is embedded in neuro_config to specify which strategy
%% to use and how to configure it.
-record(strategy_config, {
    %% Strategy module implementing evolution_strategy behaviour
    %% e.g., generational_strategy, steady_state_strategy, island_strategy
    strategy_module :: strategy_module(),

    %% Strategy-specific parameters (interpreted by the strategy)
    strategy_params = #{} :: map(),

    %% Population size bounds (strategy may vary within these)
    min_population = 10 :: pos_integer(),
    max_population = 1000 :: pos_integer(),
    initial_population = 50 :: pos_integer(),

    %% Meta-controller for this strategy (if any)
    %% Each strategy can have its own meta-controller configuration
    meta_controller_module :: module() | undefined,
    meta_controller_config = #{} :: map()
}).

-type strategy_config() :: #strategy_config{}.

%%% ============================================================================
%%% Generational Strategy Parameters
%%% ============================================================================

%% @doc Parameters specific to generational evolution.
%%
%% Use this in strategy_config.strategy_params when strategy_module
%% is generational_strategy.
-record(generational_params, {
    %% Selection method: top_n | tournament | roulette
    selection_method = top_n :: top_n | tournament | roulette,

    %% Fraction of population that survives selection (for top_n)
    selection_ratio = 0.20 :: float(),

    %% Tournament size (for tournament selection)
    tournament_size = 3 :: pos_integer(),

    %% Probability of mutating each weight
    mutation_rate = 0.10 :: float(),

    %% Magnitude of weight perturbation
    mutation_strength = 0.3 :: float(),

    %% Probability of crossover vs mutation-only reproduction
    crossover_rate = 0.75 :: float(),

    %% Whether to use elitism (best individual survives unchanged)
    elitism = true :: boolean(),

    %% Number of elite individuals to preserve unchanged
    elite_count = 1 :: pos_integer()
}).

-type generational_params() :: #generational_params{}.

%%% ============================================================================
%%% Steady-State Strategy Parameters
%%% ============================================================================

%% @doc Parameters specific to steady-state evolution.
-record(steady_state_params, {
    %% How many individuals to replace per tick
    replacement_count = 1 :: pos_integer(),

    %% Selection method for parents
    parent_selection = tournament :: tournament | fitness_proportional | random,

    %% Selection method for individuals to replace
    victim_selection = worst :: worst | oldest | random | tournament,

    %% Tournament size for selection
    tournament_size = 3 :: pos_integer(),

    %% Mutation parameters
    mutation_rate = 0.10 :: float(),
    mutation_strength = 0.3 :: float(),

    %% Default max_age for newly created organisms (0 = no limit)
    %% Individuals inherit max_age from parents with mutation
    default_max_age = 5000 :: non_neg_integer(),

    %% Max age mutation rate (probability of mutating during inheritance)
    max_age_mutation_rate = 0.10 :: float(),

    %% Max age mutation strength (relative change, e.g., 0.1 = +/- 10%)
    max_age_mutation_strength = 0.10 :: float()
}).

-type steady_state_params() :: #steady_state_params{}.

%%% ============================================================================
%%% Island Model Parameters
%%% ============================================================================

%% @doc Parameters specific to island-based evolution.
-record(island_params, {
    %% Number of islands
    island_count = 4 :: pos_integer(),

    %% Population per island (total = island_count * population_per_island)
    population_per_island = 25 :: pos_integer(),

    %% Migration frequency (every N evaluations per island)
    migration_interval = 50 :: pos_integer(),

    %% Number of individuals to migrate per event
    migration_count = 2 :: pos_integer(),

    %% How to select migrants: best | random | diverse
    migration_selection = best :: best | random | diverse,

    %% Topology: ring | full | random | custom
    topology = ring :: ring | full | random | custom,

    %% Custom topology (if topology = custom)
    %% List of {FromIsland, ToIsland} pairs
    custom_connections = [] :: [{island_id(), island_id()}],

    %% Sub-strategy for each island (defaults to generational)
    island_strategy = generational_strategy :: strategy_module(),
    island_strategy_params = #{} :: map()
}).

-type island_params() :: #island_params{}.

%%% ============================================================================
%%% Novelty Search Parameters
%%% ============================================================================

%% @doc Parameters specific to novelty search.
-record(novelty_params, {
    %% Archive size limit (oldest removed when exceeded)
    archive_size = 1000 :: pos_integer(),

    %% Probability of adding to archive
    archive_probability = 0.10 :: float(),

    %% Number of nearest neighbors for novelty calculation
    k_nearest = 15 :: pos_integer(),

    %% Whether to include fitness in selection (hybrid approach)
    include_fitness = false :: boolean(),

    %% Fitness weight if hybrid (0.0 = pure novelty, 1.0 = pure fitness)
    fitness_weight = 0.0 :: float(),

    %% Minimum novelty threshold for archive addition
    novelty_threshold = 0.0 :: float(),

    %% Behavior descriptor dimensions
    behavior_dimensions :: pos_integer() | undefined
}).

-type novelty_params() :: #novelty_params{}.

%%% ============================================================================
%%% MAP-Elites Parameters
%%% ============================================================================

%% @doc Parameters specific to MAP-Elites quality-diversity.
-record(map_elites_params, {
    %% Number of dimensions in behavior space
    behavior_dimensions = 2 :: pos_integer(),

    %% Number of bins per dimension (total niches = bins^dimensions)
    bins_per_dimension = 10 :: pos_integer(),

    %% Behavior space bounds per dimension
    %% List of {Min, Max} pairs
    behavior_bounds = [] :: [{float(), float()}],

    %% How many individuals to generate per iteration
    batch_size = 10 :: pos_integer(),

    %% Probability of generating from scratch vs mutating elite
    random_probability = 0.10 :: float(),

    %% Mutation parameters for elite modification
    mutation_rate = 0.10 :: float(),
    mutation_strength = 0.3 :: float()
}).

-type map_elites_params() :: #map_elites_params{}.

%%% ============================================================================
%%% Strategy State Types
%%% ============================================================================

%% Opaque state type returned by strategy:init/1
-type strategy_state() :: term().

%% Result of strategy tick or evaluation result handling
-type strategy_result() :: {
    Actions :: [strategy_action()],
    Events :: [lifecycle_event()],
    NewState :: strategy_state()
}.

%%% ============================================================================
%%% Meta-Controller Interface Types
%%% ============================================================================

%% Inputs provided by strategy to meta-controller
-type meta_inputs() :: [float()].

%% Parameters adjusted by meta-controller
-type meta_params() :: #{
    mutation_rate => float(),
    mutation_strength => float(),
    selection_ratio => float(),
    migration_rate => float(),
    novelty_weight => float(),
    atom() => number()
}.

-endif.
