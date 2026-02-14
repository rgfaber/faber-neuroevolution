%% @doc Cross-Silo Signal Type Definitions for Liquid Conglomerate.
%%
%% Defines all valid signals between the 13 silos. Signals are routed
%% through lc_cross_silo which validates signal names and routes.
%%
%% == Signal Naming Convention ==
%%
%% Signals are named with descriptive purpose:
%% - Pressure signals: constraint indicators (0-1, higher = more constrained)
%% - Boost signals: enhancement requests (0-1, higher = more boost)
%% - Score signals: quality metrics (0-1, higher = better)
%% - Rate signals: frequency indicators (0-1, higher = more frequent)
%%
%% @author R.G. Lefever
%% @copyright 2024-2026 R.G. Lefever

-ifndef(LC_SIGNALS_HRL).
-define(LC_SIGNALS_HRL, true).

%%% ============================================================================
%%% Signal Types
%%% ============================================================================

-type signal_name() :: atom().
-type signal_value() :: number().
%% Note: silo_type() is defined in lc_silos.hrl

%%% ============================================================================
%%% Signal Route Record
%%% ============================================================================

%% @doc Defines a valid signal route between silos.
-record(signal_route, {
    from :: atom(),        %% source silo type
    to :: atom(),          %% destination silo type
    signal :: signal_name(),
    %% Value constraints
    min_value = 0.0 :: number(),
    max_value = 1.0 :: number(),
    neutral_value = 0.5 :: number(),
    %% Description
    description :: binary()
}).

%%% ============================================================================
%%% All Valid Signals by Source Silo
%%% ============================================================================

%% ============================================================================
%% RESOURCE SILO SIGNALS (Outgoing)
%% ============================================================================

%% Resource → Task
-define(SIG_PRESSURE_SIGNAL, pressure_signal).           %% 0-1, overall resource constraint
-define(SIG_MAX_EVALS_PER_INDIVIDUAL, max_evals_per_individual).  %% 1-20, resource-limited eval count
-define(SIG_SHOULD_SIMPLIFY, should_simplify).           %% 0-1, hint to reduce complexity

%% Resource → Distribution
-define(SIG_OFFLOAD_PREFERENCE, offload_preference).     %% 0-1, prefer remote execution
-define(SIG_LOCAL_CAPACITY, local_capacity).             %% 0-1, available local resources

%% Resource → Temporal
-define(SIG_COMPUTE_AVAILABILITY, compute_availability). %% 0-1, computation budget available

%% Resource → Competitive
-define(SIG_ARMS_RACE_LOAD, arms_race_load).             %% 0-1, computational load from arms race

%% Resource → Ecological
-define(SIG_ABUNDANCE_SIGNAL, abundance_signal).         %% 0-1, resource abundance

%% Resource → Economic
-define(SIG_BUDGET_SIGNAL, budget_signal).               %% 0-1, compute budget available

%% Resource → Developmental
-define(SIG_SCARCITY_SIGNAL, scarcity_signal).           %% 0-1, resource scarcity level

%% ============================================================================
%% TASK SILO SIGNALS (Outgoing)
%% ============================================================================

%% Task → Resource
-define(SIG_EXPLORATION_BOOST, exploration_boost).       %% 0-1, exploration aggressiveness
-define(SIG_DESIRED_EVALS_PER_INDIVIDUAL, desired_evals_per_individual).  %% 1-50, fitness-based request
-define(SIG_EXPECTED_COMPLEXITY_GROWTH, expected_complexity_growth).      %% 0-1, anticipated memory needs

%% Task → Distribution
-define(SIG_DIVERSITY_NEED, diversity_need).             %% 0-1, migration diversity helps
-define(SIG_SPECIATION_PRESSURE, speciation_pressure).   %% 0-1, species splitting tendency

%% Task → Temporal
-define(SIG_STAGNATION_SEVERITY_OUT, stagnation_severity).  %% 0-1, how stagnated evolution is

%% Task → Competitive
-define(SIG_FITNESS_PRESSURE, fitness_pressure).         %% 0-1, fitness selection pressure

%% Task → Social
-define(SIG_SELECTION_PRESSURE, selection_pressure).     %% 0-1, selection intensity

%% Task → Cultural
-define(SIG_EXPLORATION_NEED, exploration_need).         %% 0-1, need for innovation

%% Task → Ecological
-define(SIG_ADAPTATION_PRESSURE, adaptation_pressure).   %% 0-1, adaptation urgency

%% Task → Morphological
-define(SIG_COMPLEXITY_TARGET, complexity_target).       %% 0-1, target network complexity

%% Task → Developmental
-define(SIG_MATURITY_TARGET, maturity_target).           %% 0-1, target maturity level

%% Task → Regulatory
-define(SIG_CONTEXT_COMPLEXITY, context_complexity).     %% 0-1, environmental complexity

%% Task → Economic
-define(SIG_BUDGET_CONSTRAINT, budget_constraint).       %% 0-1, budget pressure

%% Task → Communication
-define(SIG_COORDINATION_NEED, coordination_need).       %% 0-1, need for coordination

%% ============================================================================
%% DISTRIBUTION SILO SIGNALS (Outgoing)
%% ============================================================================

%% Distribution → Resource
-define(SIG_NETWORK_LOAD_CONTRIBUTION, network_load_contribution).  %% 0-1, load from distribution
-define(SIG_REMOTE_CAPACITY_AVAILABLE, remote_capacity_available).  %% 0-1, peers can help

%% Distribution → Task
-define(SIG_ISLAND_DIVERSITY_SCORE, island_diversity_score).  %% 0-1, cross-island diversity
-define(SIG_MIGRATION_ACTIVITY, migration_activity).          %% 0-1, recent migration level

%% Distribution → Temporal
-define(SIG_NETWORK_LATENCY, network_latency).           %% 0-1, network delay factor

%% ============================================================================
%% TEMPORAL SILO SIGNALS (Outgoing)
%% ============================================================================

%% Temporal → Task
-define(SIG_TIME_PRESSURE, time_pressure).               %% 0-1, temporal constraint

%% Temporal → Resource
-define(SIG_CONVERGENCE_STATUS, convergence_status).     %% 0-1, convergence progress

%% Temporal → Economic
-define(SIG_EPISODE_EFFICIENCY, episode_efficiency).     %% 0-1, episode cost efficiency

%% Temporal → Developmental
-define(SIG_CRITICAL_PERIOD_TIMING, critical_period_timing).  %% 0-1, timing in critical period

%% ============================================================================
%% COMPETITIVE SILO SIGNALS (Outgoing)
%% ============================================================================

%% Competitive → Task
-define(SIG_COMPETITIVE_PRESSURE, competitive_pressure). %% 0-1, competitive intensity

%% Competitive → Cultural
-define(SIG_STRATEGY_DIVERSITY_NEED, strategy_diversity_need).  %% 0-1, need for strategy variety

%% Competitive → Resource
-define(SIG_ARMS_RACE_ACTIVE, arms_race_active).         %% 0-1, arms race intensity

%% Competitive → Social
-define(SIG_COALITION_COMPETITION, coalition_competition).  %% 0-1, inter-coalition rivalry

%% ============================================================================
%% SOCIAL SILO SIGNALS (Outgoing)
%% ============================================================================

%% Social → Task
-define(SIG_SELECTION_INFLUENCE, selection_influence).   %% 0-1, social selection weight

%% Social → Cultural
-define(SIG_NORM_TRANSMISSION, norm_transmission).       %% 0-1, norm propagation rate

%% Social → Competitive
-define(SIG_COALITION_STRUCTURE, coalition_structure).   %% 0-1, coalition organization level

%% Social → Communication
-define(SIG_TRUST_NETWORK, trust_network).               %% 0-1, trust network density

%% ============================================================================
%% CULTURAL SILO SIGNALS (Outgoing)
%% ============================================================================

%% Cultural → Task
-define(SIG_INNOVATION_IMPACT, innovation_impact).       %% 0-1, innovation contribution

%% Cultural → Competitive
-define(SIG_STRATEGY_INNOVATION, strategy_innovation).   %% 0-1, strategic novelty rate

%% Cultural → Developmental
-define(SIG_PLASTICITY_INFLUENCE, plasticity_influence). %% 0-1, cultural effect on plasticity

%% Cultural → Communication
-define(SIG_INFORMATION_SHARING, information_sharing).   %% 0-1, information flow rate

%% ============================================================================
%% ECOLOGICAL SILO SIGNALS (Outgoing)
%% ============================================================================

%% Ecological → Task
-define(SIG_ENVIRONMENTAL_PRESSURE, environmental_pressure).  %% 0-1, environmental stress

%% Ecological → Resource
-define(SIG_RESOURCE_LEVEL, resource_level).             %% 0-1, ecological resource availability

%% Ecological → Developmental
-define(SIG_STRESS_SIGNAL, stress_signal).               %% 0-1, environmental stress level

%% Ecological → Regulatory
-define(SIG_ENVIRONMENTAL_CONTEXT, environmental_context).  %% 0-1, environmental complexity

%% ============================================================================
%% MORPHOLOGICAL SILO SIGNALS (Outgoing)
%% ============================================================================

%% Morphological → Task
-define(SIG_COMPLEXITY_SIGNAL, complexity_signal).       %% 0-1, current network complexity

%% Morphological → Resource
-define(SIG_SIZE_BUDGET, size_budget).                   %% 0-1, network size requirements

%% Morphological → Economic
-define(SIG_EFFICIENCY_SCORE, efficiency_score).         %% 0-1, parameter efficiency

%% Morphological → Developmental
-define(SIG_GROWTH_STAGE, growth_stage).                 %% 0-1, structural development stage

%% ============================================================================
%% DEVELOPMENTAL SILO SIGNALS (Outgoing)
%% ============================================================================

%% Developmental → Task
-define(SIG_MATURITY_DISTRIBUTION, maturity_distribution).  %% 0-1, population maturity

%% Developmental → Cultural
-define(SIG_PLASTICITY_AVAILABLE, plasticity_available). %% 0-1, learning capacity

%% Developmental → Ecological
-define(SIG_METAMORPHOSIS_RATE, metamorphosis_rate).     %% 0-1, stage transition rate

%% Developmental → Regulatory
-define(SIG_EXPRESSION_STAGE, expression_stage).         %% 0-1, developmental expression phase

%% ============================================================================
%% REGULATORY SILO SIGNALS (Outgoing)
%% ============================================================================

%% Regulatory → Task
-define(SIG_CONTEXT_AWARENESS, context_awareness).       %% 0-1, context sensitivity level

%% Regulatory → Cultural
-define(SIG_EXPRESSION_FLEXIBILITY, expression_flexibility).  %% 0-1, expression adaptability

%% Regulatory → Competitive
-define(SIG_DORMANT_POTENTIAL, dormant_potential).       %% 0-1, unexpressed capabilities

%% Regulatory → Morphological
-define(SIG_EXPRESSION_COST, expression_cost).           %% 0-1, cost of gene expression

%% ============================================================================
%% ECONOMIC SILO SIGNALS (Outgoing)
%% ============================================================================

%% Economic → Task
-define(SIG_ECONOMIC_PRESSURE, economic_pressure).       %% 0-1, economic constraint

%% Economic → Temporal
-define(SIG_BUDGET_AVAILABLE, budget_available).         %% 0-1, available budget ratio

%% Economic → Morphological
-define(SIG_EFFICIENCY_REQUIREMENT, efficiency_requirement).  %% 0-1, efficiency target

%% Economic → Social
-define(SIG_TRADE_OPPORTUNITY, trade_opportunity).       %% 0-1, trade possibilities

%% ============================================================================
%% COMMUNICATION SILO SIGNALS (Outgoing)
%% ============================================================================

%% Communication → Task
-define(SIG_COORDINATION_CAPABILITY, coordination_capability).  %% 0-1, coordination capacity

%% Communication → Cultural
-define(SIG_INFORMATION_TRANSFER, information_transfer). %% 0-1, info transmission rate

%% Communication → Social
-define(SIG_TRUST_SIGNAL, trust_signal).                 %% 0-1, communication trust level

%% Communication → Competitive
-define(SIG_STRATEGIC_SIGNALING, strategic_signaling).   %% 0-1, strategic communication level

%%% ============================================================================
%%% Neutral Values by Signal Type
%%% ============================================================================

%% Signals that decay to 0.0 (pressures, boosts that should turn off)
-define(SIGNALS_NEUTRAL_ZERO, [
    pressure_signal,
    should_simplify,
    exploration_boost,
    expected_complexity_growth,
    network_load_contribution,
    time_pressure,
    competitive_pressure,
    arms_race_active,
    environmental_pressure,
    stress_signal,
    economic_pressure
]).

%% Signals that decay to integer values
-define(SIGNALS_NEUTRAL_INTEGER, [
    {max_evals_per_individual, 10},
    {desired_evals_per_individual, 5}
]).

%% All other signals decay to 0.5 (neutral position)

%%% ============================================================================
%%% Signal Bounds
%%% ============================================================================

%% Default bounds: [0.0, 1.0]
%% Special bounds for specific signals:
-define(SIGNAL_BOUNDS, #{
    max_evals_per_individual => {1, 20},
    desired_evals_per_individual => {1, 50}
}).

-endif. %% LC_SIGNALS_HRL
