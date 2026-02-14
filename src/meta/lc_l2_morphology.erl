%% @doc L2 Strategic Layer Morphology for Liquid Conglomerate.
%%
%% The L2 layer operates at the highest level of temporal abstraction (tau=100).
%% It observes long-term evolution metrics and outputs strategic signals that
%% feed into the L1 tactical layer.
%%
%% == Input Sensors ==
%%
%% Evolution metrics from the neuroevolution process:
%% - best_fitness: Current best fitness in population
%% - avg_fitness: Average population fitness
%% - fitness_improvement: Delta from previous generation
%% - fitness_variance: Population fitness variance
%% - stagnation_counter: Generations without improvement
%% - generation_progress: Current gen / max gen ratio
%% - population_diversity: Genotype diversity measure
%% - species_count: Number of active species
%%
%% == Output Actuators ==
%%
%% Strategic signals (fed to L1 inputs):
%% - strategic_signal_1 through strategic_signal_4
%%
%% These outputs have no predefined semantics - the network learns what
%% information to pass to L1 for effective hyperparameter control.
%%
%% @author R.G. Lefever
%% @copyright 2024-2026 R.G. Lefever
-module(lc_l2_morphology).

-behaviour(morphology_behaviour).

-include_lib("faber_tweann/include/records.hrl").

%% morphology_behaviour callbacks
-export([get_sensors/1, get_actuators/1]).

%%==============================================================================
%% Callbacks
%%==============================================================================

%% @doc Get sensors for L2 strategic layer.
%%
%% Returns sensors that observe evolution metrics. All sensors are initially
%% connected to provide full observability of the evolution process.
%%
%% @param lc_l2 The morphology name
%% @returns List of sensor records for evolution metrics
-spec get_sensors(lc_l2) -> [#sensor{}].
get_sensors(lc_l2) ->
    [
        #sensor{
            name = l2_best_fitness,
            type = lc_evolution_metric,
            scape = {private, lc_chain},
            vl = 1,
            parameters = #{metric => best_fitness, normalize => true}
        },
        #sensor{
            name = l2_avg_fitness,
            type = lc_evolution_metric,
            scape = {private, lc_chain},
            vl = 1,
            parameters = #{metric => avg_fitness, normalize => true}
        },
        #sensor{
            name = l2_fitness_improvement,
            type = lc_evolution_metric,
            scape = {private, lc_chain},
            vl = 1,
            parameters = #{metric => fitness_improvement, normalize => true}
        },
        #sensor{
            name = l2_fitness_variance,
            type = lc_evolution_metric,
            scape = {private, lc_chain},
            vl = 1,
            parameters = #{metric => fitness_variance, normalize => true}
        },
        #sensor{
            name = l2_stagnation_counter,
            type = lc_evolution_metric,
            scape = {private, lc_chain},
            vl = 1,
            parameters = #{metric => stagnation_counter, max_value => 50}
        },
        #sensor{
            name = l2_generation_progress,
            type = lc_evolution_metric,
            scape = {private, lc_chain},
            vl = 1,
            parameters = #{metric => generation_progress}  % Already 0-1
        },
        #sensor{
            name = l2_population_diversity,
            type = lc_evolution_metric,
            scape = {private, lc_chain},
            vl = 1,
            parameters = #{metric => population_diversity, normalize => true}
        },
        #sensor{
            name = l2_species_count,
            type = lc_evolution_metric,
            scape = {private, lc_chain},
            vl = 1,
            parameters = #{metric => species_count, max_value => 20}
        }
    ];
get_sensors(_) ->
    error(invalid_morphology).

%% @doc Get actuators for L2 strategic layer.
%%
%% Returns actuators that output strategic signals. These signals become
%% the inputs to the L1 tactical layer.
%%
%% The number of outputs (4) was chosen to provide sufficient capacity
%% for strategic information while keeping the L1 input size manageable.
%%
%% @param lc_l2 The morphology name
%% @returns List of actuator records for strategic signals
-spec get_actuators(lc_l2) -> [#actuator{}].
get_actuators(lc_l2) ->
    [
        #actuator{
            name = l2_strategic_signal_1,
            type = lc_chain_output,
            scape = {private, lc_chain},
            vl = 1,
            parameters = #{target_level => l1, output_index => 0}
        },
        #actuator{
            name = l2_strategic_signal_2,
            type = lc_chain_output,
            scape = {private, lc_chain},
            vl = 1,
            parameters = #{target_level => l1, output_index => 1}
        },
        #actuator{
            name = l2_strategic_signal_3,
            type = lc_chain_output,
            scape = {private, lc_chain},
            vl = 1,
            parameters = #{target_level => l1, output_index => 2}
        },
        #actuator{
            name = l2_strategic_signal_4,
            type = lc_chain_output,
            scape = {private, lc_chain},
            vl = 1,
            parameters = #{target_level => l1, output_index => 3}
        }
    ];
get_actuators(_) ->
    error(invalid_morphology).
