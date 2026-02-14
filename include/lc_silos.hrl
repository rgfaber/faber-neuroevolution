%% @doc Common record definitions for Liquid Conglomerate Silos.
%%
%% All 13 silos share a common state structure with silo-specific extensions.
%% This header provides the base records and types used across all silos.
%%
%% @author R.G. Lefever
%% @copyright 2024-2026 R.G. Lefever

-ifndef(LC_SILOS_HRL).
-define(LC_SILOS_HRL, true).

%%% ============================================================================
%%% Silo Types
%%% ============================================================================

-type silo_type() :: task | resource | distribution |
                     temporal | competitive | social | cultural |
                     ecological | morphological | developmental |
                     regulatory | economic | communication.

-type silo_level() :: l0 | l1 | l2.

%%% ============================================================================
%%% Base Silo State Record
%%% ============================================================================

%% @doc Base state record for all silos.
%%
%% Each silo extends this with silo-specific fields.
%% The common fields ensure consistent behavior across all silos.
-record(silo_base_state, {
    %% === Configuration ===

    %% Silo type identifier
    silo_type :: silo_type(),

    %% Which hierarchical levels are enabled
    enabled_levels = [l0, l1] :: [silo_level()],

    %% Realm for multi-tenancy
    realm = <<"default">> :: binary(),

    %% Whether L0 uses TWEANN control (vs rule-based)
    l0_tweann_enabled = false :: boolean(),

    %% Whether L2 strategic layer is enabled
    l2_enabled = false :: boolean(),

    %% Time constant for this silo (controls adaptation speed)
    %% Lower = faster adaptation, Higher = more stable
    time_constant = 50.0 :: float(),

    %% === Current Parameters (L0 Actuator Outputs) ===

    %% Current hyperparameter values
    current_params = #{} :: map(),

    %% === Evaluation-Centric Tracking ===

    %% PRIMARY: Total evaluations performed
    total_evaluations = 0 :: non_neg_integer(),

    %% SECONDARY: Cohort number (for lineage tracking)
    cohort = 0 :: non_neg_integer(),

    %% === History Windows (Sliding) ===

    %% Metric history for velocity computation
    metric_history = [] :: [float()],

    %% Maximum history size
    history_size = 20 :: pos_integer(),

    %% === Velocity-Based Stagnation ===

    %% Current improvement velocity
    current_velocity = 0.0 :: float(),

    %% Stagnation severity (0.0 = healthy, 1.0 = critical)
    stagnation_severity = 0.0 :: float(),

    %% Velocity threshold for stagnation detection
    velocity_threshold = 0.001 :: float(),

    %% === L1 Tactical State ===

    %% Exploration boost from stagnation
    exploration_boost = 0.0 :: float(),

    %% Exploitation boost from improvement
    exploitation_boost = 0.0 :: float(),

    %% Previous values for EMA smoothing
    prev_exploration_boost = 0.0 :: float(),
    prev_exploitation_boost = 0.0 :: float(),
    prev_stagnation_severity = 0.0 :: float(),

    %% === ETS Tables (for collections) ===

    %% Map of table_name => ets:tid()
    ets_tables = #{} :: #{atom() => ets:tid()},

    %% === L2 Guidance ===

    %% L2 guidance from meta_controller (or defaults)
    %% Uses #l2_guidance{} from meta_controller.hrl
    l2_guidance :: term()
}).

-type silo_base_state() :: #silo_base_state{}.

%%% ============================================================================
%%% Silo-Specific State Extensions
%%% ============================================================================

%% @doc Temporal Silo state extension.
-record(temporal_state, {
    base :: silo_base_state(),

    %% Episode tracking
    episode_length_history = [] :: [{non_neg_integer(), pos_integer()}],
    early_termination_count = 0 :: non_neg_integer(),
    timeout_count = 0 :: non_neg_integer(),

    %% Reaction time tracking
    reaction_times = [] :: [pos_integer()]
}).

%% @doc Competitive Silo state extension.
-record(competitive_state, {
    base :: silo_base_state(),

    %% ETS table references (via base.ets_tables):
    %% - opponents: {opponent_id, network_binary, elo, strategy_sig, games}
    %% - matches: {match_id, player_a, player_b, result, elo_changes}
    %% - elo_ratings: {individual_id, elo, wins, losses, draws}

    %% Archive configuration
    archive_max_size = 100 :: pos_integer(),
    archive_addition_threshold = 0.7 :: float(),

    %% Matchmaking config
    matchmaking_elo_range = 200 :: pos_integer(),
    self_play_ratio = 0.3 :: float()
}).

%% @doc Social Silo state extension.
-record(social_state, {
    base :: silo_base_state(),

    %% ETS table references (via base.ets_tables):
    %% - reputations: {individual_id, score, history}
    %% - coalitions: {coalition_id, members, fitness_avg, stability}
    %% - interactions: {interaction_id, from, to, type, outcome}

    %% Social configuration
    reputation_decay_rate = 0.05 :: float(),
    coalition_size_limit = 10 :: pos_integer()
}).

%% @doc Cultural Silo state extension.
-record(cultural_state, {
    base :: silo_base_state(),

    %% ETS table references (via base.ets_tables):
    %% - innovations: {innovation_id, originator, behavior_sig, fitness_delta}
    %% - traditions: {tradition_id, behavior, adopter_count, generations}
    %% - memes: {meme_id, encoding, spread_rate, fitness_correlation}

    %% Cultural configuration
    innovation_bonus = 0.15 :: float(),
    imitation_probability = 0.2 :: float(),
    tradition_threshold = 5 :: pos_integer()
}).

%% @doc Ecological Silo state extension.
-record(ecological_state, {
    base :: silo_base_state(),

    %% ETS table references (via base.ets_tables):
    %% - niches: {niche_id, occupants, capacity, fitness_range}
    %% - resource_pools: {resource_id, amount, regeneration_rate}

    %% Ecological configuration
    carrying_capacity = 100 :: pos_integer(),
    stress_level = 0.0 :: float()
}).

%% @doc Morphological Silo state extension.
-record(morphological_state, {
    base :: silo_base_state(),

    %% ETS table references (via base.ets_tables):
    %% - network_sizes: {individual_id, neurons, connections, params}

    %% Complexity tracking
    neuron_count_mean = 0.0 :: float(),
    connection_count_mean = 0.0 :: float(),
    parameter_efficiency = 0.0 :: float()
}).

%% @doc Developmental Silo state extension.
-record(developmental_state, {
    base :: silo_base_state(),

    %% ETS table references (via base.ets_tables):
    %% - developmental_states: {individual_id, stage, plasticity, age}
    %% - critical_periods: {individual_id, period_type, opened_at, closed_at}
    %% - milestones: {individual_id, milestone, achieved_at}

    %% Developmental configuration
    growth_rate = 0.05 :: float(),
    initial_plasticity = 0.9 :: float()
}).

%% @doc Regulatory Silo state extension.
-record(regulatory_state, {
    base :: silo_base_state(),

    %% ETS table references (via base.ets_tables):
    %% - gene_expression: {individual_id, #{gene_id => expressed?}}
    %% - module_states: {individual_id, #{module_id => active?}}
    %% - epigenetic_marks: {individual_id, #{gene_id => mark_type}}

    %% Regulatory configuration
    expression_threshold = 0.5 :: float(),
    context_sensitivity = 0.5 :: float()
}).

%% @doc Economic Silo state extension.
-record(economic_state, {
    base :: silo_base_state(),

    %% ETS table references (via base.ets_tables):
    %% - accounts: {individual_id, balance, income, expenditure}
    %% - transactions: {tx_id, from, to, amount, type, timestamp}
    %% - market_history: {timestamp, price_fitness, volume, gini}

    %% Economic configuration
    budget_per_individual = 1.0 :: float(),
    energy_tax_rate = 0.1 :: float(),
    wealth_redistribution_rate = 0.1 :: float()
}).

%% @doc Communication Silo state extension.
-record(communication_state, {
    base :: silo_base_state(),

    %% ETS table references (via base.ets_tables):
    %% - vocabulary: {signal_id, meaning, usage_count, inventors}
    %% - dialects: {dialect_id, signals, speakers}
    %% - messages: {msg_id, sender, receiver, content, honest?, timestamp}

    %% Communication configuration
    vocabulary_max_size = 1000 :: pos_integer(),
    lying_penalty = 0.2 :: float(),
    coordination_reward = 0.2 :: float()
}).

%% @doc Distribution Silo state extension.
-record(distribution_state, {
    base :: silo_base_state(),

    %% ETS table references (via base.ets_tables):
    %% - island_stats: {island_id, fitness_mean, diversity, load}
    %% - migration_history: {migration_id, from, to, individual_id}

    %% Distribution configuration
    migration_probability = 0.05 :: float(),
    load_balance_threshold = 0.2 :: float()
}).

%%% ============================================================================
%%% Time Constants by Silo
%%% ============================================================================

%% Default time constants for each silo type
%% Lower = faster adaptation, Higher = more stable
-define(SILO_TIME_CONSTANTS, #{
    resource => 5.0,       %% Very fast - system stability
    distribution => 1.0,   %% Fastest - network responsiveness
    temporal => 10.0,      %% Fast - episode management
    task => 50.0,          %% Medium - evolution optimization
    economic => 20.0,      %% Medium-fast - budget management
    morphological => 30.0, %% Medium - network structure
    competitive => 50.0,   %% Medium - opponent archives
    social => 50.0,        %% Medium - reputation dynamics
    cultural => 100.0,     %% Slow - tradition persistence
    ecological => 100.0,   %% Slow - niche stability
    developmental => 100.0, %% Slow - ontogeny stages
    regulatory => 50.0,    %% Medium - gene expression
    communication => 30.0  %% Medium - vocabulary evolution
}).

%%% ============================================================================
%%% Intervention Levels
%%% ============================================================================

-type intervention_level() :: healthy | warning | intervention | critical.

%% @doc Convert stagnation severity to intervention level.
%%
%% Uses thresholds from L2 guidance or defaults.
-define(severity_to_level(Severity, WarningT, InterventionT, CriticalT),
    case Severity of
        S when S >= CriticalT -> critical;
        S when S >= InterventionT -> intervention;
        S when S >= WarningT -> warning;
        _ -> healthy
    end
).

-endif. %% LC_SILOS_HRL
