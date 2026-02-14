%% @doc L0 Reactive Layer Morphology for Liquid Conglomerate.
%%
%% The L0 layer operates at the lowest temporal abstraction (tau=10).
%% It receives tactical signals from L1 plus emergent metrics from the
%% model under training, and outputs the final hyperparameters.
%%
%% == Input Sensors ==
%%
%% Fixed inputs (always connected, from L1):
%% - l0_from_l1_signal_1 through l0_from_l1_signal_5
%%
%% Emergent metric sensors (available for topology evolution):
%% - convergence_rate, current_mutation_rate, survival_rate, etc.
%%
%% The emergent sensors are NOT initially connected. Topology evolution
%% can add them via add_sensor/1 when the network determines they are
%% useful for hyperparameter control.
%%
%% == Output Actuators ==
%%
%% Final hyperparameters for the model under training:
%% - mutation_rate: [0.01, 0.5]
%% - mutation_strength: [0.05, 1.0]
%% - selection_ratio: [0.1, 0.5]
%% - add_node_rate: [0.0, 0.1]
%% - add_connection_rate: [0.0, 0.2]
%%
%% @author R.G. Lefever
%% @copyright 2024-2026 R.G. Lefever
-module(lc_l0_morphology).

-behaviour(morphology_behaviour).

-include_lib("faber_tweann/include/records.hrl").

%% morphology_behaviour callbacks
-export([get_sensors/1, get_actuators/1]).

%% Additional exports for sensor classification
-export([get_fixed_sensors/0, get_emergent_sensors/0]).

%%==============================================================================
%% Callbacks
%%==============================================================================

%% @doc Get all sensors for L0 reactive layer.
%%
%% Returns both fixed sensors (from L1) and emergent sensors (from model).
%% Fixed sensors are always connected initially.
%% Emergent sensors are available but not initially connected - topology
%% evolution can add them via add_sensor/1.
%%
%% @param lc_l0 The morphology name
%% @returns List of all sensor records (fixed + emergent)
-spec get_sensors(lc_l0) -> [#sensor{}].
get_sensors(lc_l0) ->
    get_fixed_sensors() ++ get_emergent_sensors();
get_sensors(_) ->
    error(invalid_morphology).

%% @doc Get fixed sensors that receive L1 tactical outputs.
%%
%% These sensors are always connected in new agents.
-spec get_fixed_sensors() -> [#sensor{}].
get_fixed_sensors() ->
    [
        #sensor{
            name = l0_from_l1_signal_1,
            type = lc_chain_input,
            scape = {private, lc_chain},
            vl = 1,
            parameters = #{source_level => l1, input_index => 0, fixed => true}
        },
        #sensor{
            name = l0_from_l1_signal_2,
            type = lc_chain_input,
            scape = {private, lc_chain},
            vl = 1,
            parameters = #{source_level => l1, input_index => 1, fixed => true}
        },
        #sensor{
            name = l0_from_l1_signal_3,
            type = lc_chain_input,
            scape = {private, lc_chain},
            vl = 1,
            parameters = #{source_level => l1, input_index => 2, fixed => true}
        },
        #sensor{
            name = l0_from_l1_signal_4,
            type = lc_chain_input,
            scape = {private, lc_chain},
            vl = 1,
            parameters = #{source_level => l1, input_index => 3, fixed => true}
        },
        #sensor{
            name = l0_from_l1_signal_5,
            type = lc_chain_input,
            scape = {private, lc_chain},
            vl = 1,
            parameters = #{source_level => l1, input_index => 4, fixed => true}
        }
    ].

%% @doc Get emergent metric sensors.
%%
%% These sensors observe metrics from the model under training.
%% They are available for topology evolution to add, but NOT initially connected.
%% This allows the LC to evolve which emergent signals are useful.
-spec get_emergent_sensors() -> [#sensor{}].
get_emergent_sensors() ->
    [
        %% Convergence and progress metrics
        #sensor{
            name = l0_convergence_rate,
            type = lc_emergent_metric,
            scape = {private, lc_chain},
            vl = 1,
            parameters = #{
                metric => convergence_rate,
                fixed => false,
                description => "Rate of fitness improvement (smoothed)"
            }
        },
        #sensor{
            name = l0_fitness_plateau_duration,
            type = lc_emergent_metric,
            scape = {private, lc_chain},
            vl = 1,
            parameters = #{
                metric => fitness_plateau_duration,
                fixed => false,
                max_value => 100,
                description => "Generations at current best fitness"
            }
        },

        %% Current hyperparameter feedback
        #sensor{
            name = l0_current_mutation_rate,
            type = lc_emergent_metric,
            scape = {private, lc_chain},
            vl = 1,
            parameters = #{
                metric => current_mutation_rate,
                fixed => false,
                description => "Currently active mutation rate"
            }
        },
        #sensor{
            name = l0_current_selection_ratio,
            type = lc_emergent_metric,
            scape = {private, lc_chain},
            vl = 1,
            parameters = #{
                metric => current_selection_ratio,
                fixed => false,
                description => "Currently active selection ratio"
            }
        },

        %% Population dynamics
        #sensor{
            name = l0_survival_rate,
            type = lc_emergent_metric,
            scape = {private, lc_chain},
            vl = 1,
            parameters = #{
                metric => survival_rate,
                fixed => false,
                description => "Fraction of population surviving selection"
            }
        },
        #sensor{
            name = l0_offspring_rate,
            type = lc_emergent_metric,
            scape = {private, lc_chain},
            vl = 1,
            parameters = #{
                metric => offspring_rate,
                fixed => false,
                description => "New individuals / population size"
            }
        },
        #sensor{
            name = l0_elite_age,
            type = lc_emergent_metric,
            scape = {private, lc_chain},
            vl = 1,
            parameters = #{
                metric => elite_age,
                fixed => false,
                max_value => 50,
                description => "Generations the champion has been unchanged"
            }
        },

        %% Topology and complexity metrics
        #sensor{
            name = l0_complexity_trend,
            type = lc_emergent_metric,
            scape = {private, lc_chain},
            vl = 1,
            parameters = #{
                metric => complexity_trend,
                fixed => false,
                description => "Rate of network size change"
            }
        },
        #sensor{
            name = l0_avg_network_size,
            type = lc_emergent_metric,
            scape = {private, lc_chain},
            vl = 1,
            parameters = #{
                metric => avg_network_size,
                fixed => false,
                normalize => true,
                description => "Average neuron count in population"
            }
        },

        %% Species dynamics
        #sensor{
            name = l0_species_extinction_rate,
            type = lc_emergent_metric,
            scape = {private, lc_chain},
            vl = 1,
            parameters = #{
                metric => species_extinction_rate,
                fixed => false,
                description => "Species dying per generation"
            }
        },
        #sensor{
            name = l0_species_creation_rate,
            type = lc_emergent_metric,
            scape = {private, lc_chain},
            vl = 1,
            parameters = #{
                metric => species_creation_rate,
                fixed => false,
                description => "New species per generation"
            }
        },

        %% Innovation metrics
        #sensor{
            name = l0_innovation_rate,
            type = lc_emergent_metric,
            scape = {private, lc_chain},
            vl = 1,
            parameters = #{
                metric => innovation_rate,
                fixed => false,
                description => "New topology innovations per generation"
            }
        },
        #sensor{
            name = l0_diversity_index,
            type = lc_emergent_metric,
            scape = {private, lc_chain},
            vl = 1,
            parameters = #{
                metric => diversity_index,
                fixed => false,
                description => "Genotype diversity measure (0-1)"
            }
        }
    ].

%% @doc Get actuators for L0 reactive layer.
%%
%% Returns actuators that output the final hyperparameters for the
%% model under training. Each output is scaled to its valid range.
%%
%% @param lc_l0 The morphology name
%% @returns List of actuator records for hyperparameters
-spec get_actuators(lc_l0) -> [#actuator{}].
get_actuators(lc_l0) ->
    [
        #actuator{
            name = l0_mutation_rate,
            type = lc_hyperparam_output,
            scape = {private, lc_chain},
            vl = 1,
            parameters = #{
                hyperparam => mutation_rate,
                min_value => 0.01,
                max_value => 0.5,
                description => "Probability of mutation per gene"
            }
        },
        #actuator{
            name = l0_mutation_strength,
            type = lc_hyperparam_output,
            scape = {private, lc_chain},
            vl = 1,
            parameters = #{
                hyperparam => mutation_strength,
                min_value => 0.05,
                max_value => 1.0,
                description => "Magnitude of weight perturbations"
            }
        },
        #actuator{
            name = l0_selection_ratio,
            type = lc_hyperparam_output,
            scape = {private, lc_chain},
            vl = 1,
            parameters = #{
                hyperparam => selection_ratio,
                min_value => 0.1,
                max_value => 0.5,
                description => "Fraction of population selected as parents"
            }
        },
        #actuator{
            name = l0_add_node_rate,
            type = lc_hyperparam_output,
            scape = {private, lc_chain},
            vl = 1,
            parameters = #{
                hyperparam => add_node_rate,
                min_value => 0.0,
                max_value => 0.1,
                description => "Probability of adding a hidden node"
            }
        },
        #actuator{
            name = l0_add_connection_rate,
            type = lc_hyperparam_output,
            scape = {private, lc_chain},
            vl = 1,
            parameters = #{
                hyperparam => add_connection_rate,
                min_value => 0.0,
                max_value => 0.2,
                description => "Probability of adding a connection"
            }
        }
    ];
get_actuators(_) ->
    error(invalid_morphology).
