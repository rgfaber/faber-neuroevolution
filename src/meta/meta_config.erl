%% @doc Configuration builder for the LTC meta-controller.
%%
%% This module provides helper functions to construct #meta_config{} records
%% from maps, enabling clean integration with Elixir applications.
%%
%% == Usage ==
%%
%% From Elixir:
%% meta_config = :meta_config.from_map(%{
%%   network_topology: {11, [24, 16, 8], 5},
%%   neuron_type: :cfc,
%%   time_constant: 50.0
%% })
%% neuro_config = :neuro_config.from_map(%{
%%   meta_controller_config: meta_config,
%%   ...
%% })
%%
%% @author R.G. Lefever
%% @copyright 2024-2026 R.G. Lefever
-module(meta_config).

-include("meta_controller.hrl").

-export([
    from_map/1,
    to_map/1,
    default/0,
    default_reward_weights/0,
    default_param_bounds/0
]).

%%% ============================================================================
%%% API Functions
%%% ============================================================================

%% @doc Build a #meta_config{} record from a map.
%%
%% All fields are optional - missing fields use sensible defaults.
%% This function handles type coercion and validation.
-spec from_map(map()) -> #meta_config{}.
from_map(Map) when is_map(Map) ->
    #meta_config{
        %% Network architecture for the LTC meta-controller
        %% Default: 11 inputs (8 original + 3 resource metrics), 5 outputs
        network_topology = maps:get(network_topology, Map, {11, [24, 16, 8], 5}),

        %% LTC neuron type: cfc (fast) or ltc (accurate ODE)
        neuron_type = maps:get(neuron_type, Map, cfc),

        %% Base time constant for meta-controller neurons
        %% Higher values = slower adaptation = more stable
        time_constant = maps:get(time_constant, Map, 50.0),

        %% State bound for LTC neurons
        state_bound = maps:get(state_bound, Map, 1.0),

        %% Reward component weights (must sum to ~1.0)
        reward_weights = maps:get(reward_weights, Map, default_reward_weights()),

        %% Learning rate for gradient-based meta-training
        learning_rate = maps:get(learning_rate, Map, 0.001),

        %% Parameter bounds: {ParamName, {Min, Max}}
        param_bounds = maps:get(param_bounds, Map, default_param_bounds()),

        %% Whether to include population_size as a controllable parameter
        control_population_size = maps:get(control_population_size, Map, false),

        %% Whether to control topology mutation rates (NEAT mode)
        control_topology = maps:get(control_topology, Map, false),

        %% History window size for computing reward signals
        history_window = maps:get(history_window, Map, 10),

        %% Momentum for parameter updates (smooths changes)
        momentum = maps:get(momentum, Map, 0.9)
    }.

%% @doc Convert a #meta_config{} record to a map.
%%
%% Useful for serialization, logging, and passing to Elixir code.
-spec to_map(#meta_config{}) -> map().
to_map(Config) when is_record(Config, meta_config) ->
    #{
        network_topology => Config#meta_config.network_topology,
        neuron_type => Config#meta_config.neuron_type,
        time_constant => Config#meta_config.time_constant,
        state_bound => Config#meta_config.state_bound,
        reward_weights => Config#meta_config.reward_weights,
        learning_rate => Config#meta_config.learning_rate,
        param_bounds => Config#meta_config.param_bounds,
        control_population_size => Config#meta_config.control_population_size,
        control_topology => Config#meta_config.control_topology,
        history_window => Config#meta_config.history_window,
        momentum => Config#meta_config.momentum
    }.

%% @doc Create a default configuration with sensible defaults.
-spec default() -> #meta_config{}.
default() ->
    from_map(#{}).

%% @doc Default reward component weights.
%%
%% These weights balance different aspects of training quality:
%% - convergence_speed: How quickly fitness improves
%% - final_fitness: The ultimate fitness achieved
%% - efficiency_ratio: Fitness improvement per evaluation
%% - diversity_aware: Maintaining population diversity
%% - normative_structure: Preserving adaptation potential
-spec default_reward_weights() -> map().
default_reward_weights() ->
    #{
        convergence_speed => 0.25,
        final_fitness => 0.25,
        efficiency_ratio => 0.20,
        diversity_aware => 0.15,
        normative_structure => 0.15
    }.

%% @doc Default parameter bounds for meta-controller outputs.
%%
%% These bounds define the legal ranges for each hyperparameter
%% that the LTC meta-controller can adjust.
%%
%% Includes resource-aware parameters:
%% - evaluations_per_individual: Can drop to 1 under memory pressure
%% - max_concurrent_evaluations: Limits parallelism under load
-spec default_param_bounds() -> map().
default_param_bounds() ->
    #{
        %% Basic evolution parameters
        mutation_rate => {0.01, 0.5},
        mutation_strength => {0.05, 1.0},
        selection_ratio => {0.10, 0.50},

        %% Resource-aware parameters (NEW)
        evaluations_per_individual => {1, 20},
        max_concurrent_evaluations => {1, 1000000},  % This is Erlang - can handle millions of processes

        %% Population control (when enabled)
        population_size => {10, 200},

        %% Topology control (NEAT mode)
        add_node_rate => {0.0, 0.10},
        add_connection_rate => {0.0, 0.20},
        complexity_penalty => {0.0, 0.5}
    }.
