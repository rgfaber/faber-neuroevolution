%% @doc Lifecycle event record definitions for evolution strategies.
%%
%% These events provide a universal vocabulary for all evolution strategies.
%% The UI and meta-controllers subscribe to these events rather than
%% strategy-specific mechanisms like "generation_complete".
%%
%% == Universal Events (all strategies emit these) ==
%%
%% - `individual_born' - When any individual is created (initial, offspring, migrant)
%% - `individual_died' - When any individual is removed from the population
%% - `individual_evaluated' - When an individual's fitness is computed
%% - `species_emerged' - When a new species is formed (if speciation enabled)
%% - `species_extinct' - When a species dies out
%%
%% == Strategy-Specific Events ==
%%
%% Generational:
%% - `cohort_evaluated' - All individuals in generation evaluated
%% - `breeding_complete' - Selection and breeding finished
%% - `generation_advanced' - New generation begins
%%
%% Steady-State:
%% - (uses individual_born/individual_died directly, no cohort events)
%%
%% Island Model:
%% - `island_migration' - Individual migrated between islands
%% - `island_topology_changed' - Island connectivity changed
%%
%% Novelty Search / MAP-Elites:
%% - `archive_updated' - Novelty archive changed
%% - `niche_discovered' - New behavioral niche found
%% - `niche_updated' - Better individual found for niche
%%
%% @author R.G. Lefever
%% @copyright 2024-2026 R.G. Lefever

-ifndef(LIFECYCLE_EVENTS_HRL).
-define(LIFECYCLE_EVENTS_HRL, true).

%% Include neuroevolution.hrl for individual_id() and species_id() types
-include("neuroevolution.hrl").

%%% ============================================================================
%%% Type Definitions
%%% ============================================================================

%% Note: individual_id() and species_id() are defined in neuroevolution.hrl
%% We only define types specific to lifecycle events here.

-type island_id() :: pos_integer() | atom().
-type niche_id() :: term().
-type timestamp() :: erlang:timestamp().

%% Birth origins - how the individual came to exist
-type birth_origin() ::
    initial |           % Created at population initialization
    crossover |         % Sexual reproduction from two parents
    mutation |          % Asexual mutation of single parent
    migration |         % Migrated from another island
    insertion.          % Externally inserted (e.g., from archive)

%% Death reasons - why the individual was removed
-type death_reason() ::
    selection_pressure |   % Lost in fitness-based selection
    stagnation |           % Removed due to lack of improvement
    age_limit |            % Exceeded maximum age
    niche_competition |    % Replaced by better individual in same niche
    migration |            % Migrated to another island (died here, born there)
    population_limit |     % Population size exceeded maximum
    extinction.            % Entire species/island eliminated

%%% ============================================================================
%%% Universal Lifecycle Events
%%% ============================================================================

%% @doc An individual has been created and added to the population.
%%
%% This event is emitted by ALL strategies when any individual is created,
%% regardless of how (initial population, breeding, migration, etc.).
-record(individual_born, {
    %% Unique identifier for the new individual
    id :: individual_id(),

    %% Parent IDs (empty for initial population, 1 for mutation, 2 for crossover)
    parent_ids = [] :: [individual_id()],

    %% When this individual was born
    timestamp :: timestamp(),

    %% How this individual was created
    origin :: birth_origin(),

    %% Strategy-specific metadata
    %% May include: island_id, niche_id, mutation_type, species_id, etc.
    metadata = #{} :: map()
}).

-type individual_born() :: #individual_born{}.


%% @doc An individual has been removed from the population.
%%
%% This event is emitted by ALL strategies when any individual is removed,
%% regardless of why (selection, stagnation, migration out, etc.).
-record(individual_died, {
    %% Identifier of the deceased individual
    id :: individual_id(),

    %% Why this individual was removed
    reason :: death_reason(),

    %% Final fitness at time of death (undefined if never evaluated)
    final_fitness :: float() | undefined,

    %% When this individual died
    timestamp :: timestamp(),

    %% Strategy-specific metadata
    metadata = #{} :: map()
}).

-type individual_died() :: #individual_died{}.


%% @doc An individual has been evaluated and assigned fitness.
%%
%% Emitted after fitness evaluation completes for any individual.
-record(individual_evaluated, {
    %% Which individual was evaluated
    id :: individual_id(),

    %% The computed fitness value
    fitness :: float(),

    %% Additional metrics from evaluation
    metrics = #{} :: map(),

    %% When evaluation completed
    timestamp :: timestamp(),

    %% Strategy-specific metadata (evaluator node, batch info, etc.)
    metadata = #{} :: map()
}).

-type individual_evaluated() :: #individual_evaluated{}.


%%% ============================================================================
%%% Species Lifecycle Events
%%% ============================================================================

%% @doc A new species has emerged from speciation.
%%
%% Emitted when an individual is sufficiently different from existing
%% species to warrant creating a new one.
-record(species_emerged, {
    %% Unique identifier for the new species
    species_id :: species_id(),

    %% The individual that founded this species (becomes representative)
    founder_id :: individual_id(),

    %% If this species split from an existing one, its ID (undefined for novel)
    parent_species_id :: species_id() | undefined,

    %% When this species was created
    timestamp :: timestamp(),

    %% Initial compatibility threshold, representative genome info, etc.
    metadata = #{} :: map()
}).

-type species_emerged() :: #species_emerged{}.


%% @doc A species has gone extinct.
%%
%% Emitted when a species has no remaining members or is eliminated
%% due to stagnation.
-record(species_extinct, {
    %% Which species died
    species_id :: species_id(),

    %% Why this species went extinct
    reason :: stagnation | empty | merged | eliminated,

    %% Final statistics at time of extinction
    final_stats :: map(),

    %% When this species went extinct
    timestamp :: timestamp()
}).

-type species_extinct() :: #species_extinct{}.


%%% ============================================================================
%%% Generational Strategy Events
%%% ============================================================================

%% @doc All individuals in the current cohort have been evaluated.
%%
%% Specific to generational strategies. Marks the end of the evaluation phase.
-record(cohort_evaluated, {
    %% Which generation completed evaluation
    generation :: pos_integer(),

    %% Statistics for the evaluated cohort
    best_fitness :: float(),
    avg_fitness :: float(),
    worst_fitness :: float(),

    %% Population size
    population_size :: pos_integer(),

    timestamp :: timestamp()
}).

-type cohort_evaluated() :: #cohort_evaluated{}.


%% @doc Selection and breeding have completed for a generation.
%%
%% Specific to generational strategies. Contains information about
%% survivors, eliminated, and new offspring.
-record(breeding_complete, {
    generation :: pos_integer(),

    %% How many survived selection
    survivor_count :: non_neg_integer(),

    %% How many were eliminated
    eliminated_count :: non_neg_integer(),

    %% How many offspring were created
    offspring_count :: non_neg_integer(),

    timestamp :: timestamp()
}).

-type breeding_complete() :: #breeding_complete{}.


%%% ============================================================================
%%% Steady-State Strategy Events
%%% ============================================================================

%% @doc A replacement cycle has occurred in steady-state evolution.
%%
%% Emitted when one or more individuals have been replaced.
-record(steady_state_replacement, {
    %% IDs of individuals that were replaced
    replaced_ids :: [individual_id()],

    %% IDs of the new offspring that replaced them
    offspring_ids :: [individual_id()],

    %% Current population statistics
    best_fitness :: float() | undefined,
    avg_fitness :: float() | undefined,

    timestamp :: timestamp()
}).

-type steady_state_replacement() :: #steady_state_replacement{}.


%% @doc A new generation is beginning.
%%
%% Replaces the old "generation_complete" event. This marks the transition
%% to a new generation after breeding is complete.
-record(generation_advanced, {
    %% The new generation number
    generation :: pos_integer(),

    %% Summary of what happened in the previous generation
    previous_best_fitness :: float(),
    previous_avg_fitness :: float(),

    %% Current population size
    population_size :: pos_integer(),

    %% Species count (if speciation enabled)
    species_count :: non_neg_integer(),

    timestamp :: timestamp()
}).

-type generation_advanced() :: #generation_advanced{}.


%%% ============================================================================
%%% Island Model Events
%%% ============================================================================

%% @doc An individual migrated from one island to another.
-record(island_migration, {
    %% Which individual migrated
    individual_id :: individual_id(),

    %% Source and destination islands
    from_island :: island_id(),
    to_island :: island_id(),

    %% Fitness at time of migration
    fitness :: float(),

    timestamp :: timestamp()
}).

-type island_migration() :: #island_migration{}.


%% @doc The island topology has changed.
%%
%% Emitted when islands are added, removed, or connectivity changes.
-record(island_topology_changed, {
    %% Current island IDs
    islands :: [island_id()],

    %% Connectivity graph (which islands can exchange with which)
    connections :: [{island_id(), island_id()}],

    %% What changed
    change_type :: island_added | island_removed | connection_changed,

    timestamp :: timestamp()
}).

-type island_topology_changed() :: #island_topology_changed{}.


%%% ============================================================================
%%% Quality-Diversity Events (Novelty Search, MAP-Elites)
%%% ============================================================================

%% @doc A new behavioral niche has been discovered.
%%
%% Emitted in MAP-Elites when an individual occupies a previously empty cell.
-record(niche_discovered, {
    %% Identifier for this niche (grid coordinates, behavior hash, etc.)
    niche_id :: niche_id(),

    %% The behavior descriptor that defines this niche
    behavior_descriptor :: [float()],

    %% The individual that discovered this niche
    individual_id :: individual_id(),

    %% Fitness of the discoverer
    fitness :: float(),

    timestamp :: timestamp()
}).

-type niche_discovered() :: #niche_discovered{}.


%% @doc A niche now has a better occupant.
%%
%% Emitted in MAP-Elites when a new individual beats the current elite.
-record(niche_updated, {
    %% Which niche was updated
    niche_id :: niche_id(),

    %% The old and new occupants
    old_individual_id :: individual_id(),
    new_individual_id :: individual_id(),

    %% Fitness improvement
    old_fitness :: float(),
    new_fitness :: float(),
    improvement :: float(),

    timestamp :: timestamp()
}).

-type niche_updated() :: #niche_updated{}.


%% @doc The novelty/QD archive has been updated.
%%
%% Emitted periodically with aggregate archive statistics.
-record(archive_updated, {
    %% Total individuals in archive
    size :: non_neg_integer(),

    %% Fraction of niches occupied (MAP-Elites)
    coverage :: float(),

    %% Quality-Diversity score (sum of elite fitnesses)
    qd_score :: float(),

    %% Number of updates since last event
    updates_since_last :: non_neg_integer(),

    timestamp :: timestamp()
}).

-type archive_updated() :: #archive_updated{}.


%%% ============================================================================
%%% Coevolution Events
%%% ============================================================================

%% @doc A competitor population has been updated.
-record(competitor_updated, {
    %% Which competitor population changed
    competitor_id :: term(),

    %% Summary of change
    change_type :: generation_advanced | champion_changed | strategy_shift,

    %% Current champion fitness (if applicable)
    champion_fitness :: float() | undefined,

    timestamp :: timestamp()
}).

-type competitor_updated() :: #competitor_updated{}.


%% @doc An arms race event occurred in coevolution.
%%
%% Tracks competitive dynamics like Red Queen effects.
-record(arms_race_event, {
    %% What type of arms race event
    event_type :: fitness_surge | counter_adaptation | stalemate | breakthrough,

    %% Which populations are involved
    populations :: [term()],

    %% Metrics about the competitive dynamics
    metrics :: map(),

    timestamp :: timestamp()
}).

-type arms_race_event() :: #arms_race_event{}.


%%% ============================================================================
%%% Open-Ended Evolution Events
%%% ============================================================================

%% @doc A new capability has emerged.
-record(capability_emerged, {
    %% Identifier for this capability
    capability_id :: term(),

    %% Description of what this capability enables
    description :: binary(),

    %% The individual(s) that first exhibited this capability
    exhibitors :: [individual_id()],

    timestamp :: timestamp()
}).

-type capability_emerged() :: #capability_emerged{}.


%% @doc Complexity has increased significantly.
-record(complexity_increased, {
    %% What complexity metric increased
    metric :: genome_size | network_depth | behavior_repertoire | term(),

    %% Previous and new values
    old_value :: number(),
    new_value :: number(),

    %% Percentage increase
    increase_pct :: float(),

    timestamp :: timestamp()
}).

-type complexity_increased() :: #complexity_increased{}.


%%% ============================================================================
%%% Strategy-Agnostic Progress Events
%%% ============================================================================

%% @doc Periodic progress checkpoint emitted every N evaluations.
%%
%% This event provides a strategy-agnostic way to track evolution progress.
%% It can be emitted by any strategy at configurable intervals, allowing
%% the UI to track progress regardless of whether the evolution is
%% generational or continuous.
%%
%% Updated for evaluation-centric evolution: total_evaluations is the
%% PRIMARY progress metric. cohort is the SECONDARY metric for lineage.
-record(progress_checkpoint, {
    %% PRIMARY: Total evaluations since training started
    total_evaluations :: non_neg_integer(),

    %% Evaluations in this batch/checkpoint interval
    evaluations_since_last = 0 :: non_neg_integer(),

    %% SECONDARY: Cohort number (formerly "generation" - kept for lineage)
    cohort = 0 :: non_neg_integer(),

    %% Current population statistics
    best_fitness = 0.0 :: float(),
    avg_fitness = 0.0 :: float(),
    worst_fitness = 0.0 :: float(),

    %% Population size
    population_size = 0 :: non_neg_integer(),

    %% Number of species (1 if no speciation)
    species_count = 1 :: pos_integer(),

    %% Improvement since last checkpoint (for velocity calculation)
    improvement = 0.0 :: float(),

    %% Timing information
    elapsed_ms = 0 :: non_neg_integer(),
    evals_per_second = 0.0 :: float(),

    %% Checkpoint interval (for reference)
    checkpoint_interval = 0 :: non_neg_integer(),

    timestamp :: timestamp()
}).

-type progress_checkpoint() :: #progress_checkpoint{}.


%%% ============================================================================
%%% Environmental Events
%%% ============================================================================

%% @doc Environment changed, affecting the fitness landscape.
%%
%% Emitted when environmental conditions change that affect how fitness
%% is calculated. This is useful for curriculum learning, adaptive
%% difficulty, or simulating changing real-world conditions.
-record(environment_changed, {
    %% Identifier for this environment or environment version
    environment_id :: term(),

    %% Type of change
    change_type :: difficulty_increased | difficulty_decreased |
                   task_shifted | condition_changed | curriculum_advanced,

    %% Human-readable description
    description :: binary(),

    %% Optional metrics about the change
    metrics = #{} :: map(),

    timestamp :: timestamp()
}).

-type environment_changed() :: #environment_changed{}.


%%% ============================================================================
%%% Lifetime Events (Continuous Evolution)
%%% ============================================================================

%% @doc Individual aged out due to max_lifetime parameter.
%%
%% In continuous evolution scenarios, individuals have a maximum lifetime
%% after which they are removed regardless of fitness. This event captures
%% the individual's lifetime statistics.
-record(individual_aged_out, {
    %% Individual that aged out
    id :: individual_id(),

    %% Final age (in evaluations or ticks)
    final_age :: pos_integer(),

    %% Final fitness at time of aging out
    final_fitness :: float(),

    %% Lifetime statistics
    lifetime_stats :: #{
        total_evaluations := non_neg_integer(),
        avg_fitness := float(),
        best_fitness := float(),
        offspring_count := non_neg_integer()
    },

    timestamp :: timestamp()
}).

-type individual_aged_out() :: #individual_aged_out{}.


%%% ============================================================================
%%% Competition Events (Coevolution)
%%% ============================================================================

%% @doc Result of a competition between individuals.
%%
%% Used in coevolution, game-based evaluation, and tournament selection
%% to record outcomes of head-to-head or multi-way competitions.
-record(competition_result, {
    %% Individuals that competed
    competitors :: [individual_id()],

    %% Scores for each competitor
    scores :: [{individual_id(), float()}],

    %% Winner (or 'draw' for ties)
    winner_id :: individual_id() | draw,

    %% Type of competition
    competition_type :: tournament | round_robin | elimination |
                        ranked_match | team_vs_team,

    %% Competition-specific metadata
    metadata = #{} :: map(),

    timestamp :: timestamp()
}).

-type competition_result() :: #competition_result{}.


%%% ============================================================================
%%% Aggregate Event Type
%%% ============================================================================

%% Union type for all lifecycle events
-type lifecycle_event() ::
    %% Universal events
    #individual_born{} |
    #individual_died{} |
    #individual_evaluated{} |
    %% Species events
    #species_emerged{} |
    #species_extinct{} |
    %% Generational strategy events
    #cohort_evaluated{} |
    #breeding_complete{} |
    #generation_advanced{} |
    %% Steady-state strategy events
    #steady_state_replacement{} |
    %% Island model events
    #island_migration{} |
    #island_topology_changed{} |
    %% Quality-diversity events
    #niche_discovered{} |
    #niche_updated{} |
    #archive_updated{} |
    %% Coevolution events
    #competitor_updated{} |
    #arms_race_event{} |
    #competition_result{} |
    %% Open-ended evolution events
    #capability_emerged{} |
    #complexity_increased{} |
    %% Strategy-agnostic progress events
    #progress_checkpoint{} |
    %% Environmental events
    #environment_changed{} |
    %% Lifetime events
    #individual_aged_out{}.

-endif.
