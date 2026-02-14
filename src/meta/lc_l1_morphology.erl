%% @doc L1 Tactical Layer Morphology for Liquid Conglomerate.
%%
%% The L1 layer operates at medium temporal abstraction (tau=50).
%% It receives strategic signals from L2 and outputs tactical signals
%% that feed into the L0 reactive layer.
%%
%% == Input Sensors ==
%%
%% Strategic signals from L2 outputs:
%% - l1_from_l2_signal_1 through l1_from_l2_signal_4
%%
%% These inputs carry the strategic context computed by L2.
%%
%% == Output Actuators ==
%%
%% Tactical signals (fed to L0 inputs):
%% - tactical_signal_1 through tactical_signal_5
%%
%% Five outputs provide L0 with sufficient tactical context while
%% matching the expected input structure of the reactive layer.
%%
%% @author R.G. Lefever
%% @copyright 2024-2026 R.G. Lefever
-module(lc_l1_morphology).

-behaviour(morphology_behaviour).

-include_lib("faber_tweann/include/records.hrl").

%% morphology_behaviour callbacks
-export([get_sensors/1, get_actuators/1]).

%%==============================================================================
%% Callbacks
%%==============================================================================

%% @doc Get sensors for L1 tactical layer.
%%
%% Returns sensors that receive L2's strategic outputs. All sensors are
%% always connected since they carry essential information from L2.
%%
%% @param lc_l1 The morphology name
%% @returns List of sensor records for L2 strategic signals
-spec get_sensors(lc_l1) -> [#sensor{}].
get_sensors(lc_l1) ->
    [
        #sensor{
            name = l1_from_l2_signal_1,
            type = lc_chain_input,
            scape = {private, lc_chain},
            vl = 1,
            parameters = #{source_level => l2, input_index => 0, fixed => true}
        },
        #sensor{
            name = l1_from_l2_signal_2,
            type = lc_chain_input,
            scape = {private, lc_chain},
            vl = 1,
            parameters = #{source_level => l2, input_index => 1, fixed => true}
        },
        #sensor{
            name = l1_from_l2_signal_3,
            type = lc_chain_input,
            scape = {private, lc_chain},
            vl = 1,
            parameters = #{source_level => l2, input_index => 2, fixed => true}
        },
        #sensor{
            name = l1_from_l2_signal_4,
            type = lc_chain_input,
            scape = {private, lc_chain},
            vl = 1,
            parameters = #{source_level => l2, input_index => 3, fixed => true}
        }
    ];
get_sensors(_) ->
    error(invalid_morphology).

%% @doc Get actuators for L1 tactical layer.
%%
%% Returns actuators that output tactical signals. These signals become
%% the primary inputs to the L0 reactive layer.
%%
%% Five outputs are provided to give L0 sufficient tactical context for
%% computing final hyperparameters.
%%
%% @param lc_l1 The morphology name
%% @returns List of actuator records for tactical signals
-spec get_actuators(lc_l1) -> [#actuator{}].
get_actuators(lc_l1) ->
    [
        #actuator{
            name = l1_tactical_signal_1,
            type = lc_chain_output,
            scape = {private, lc_chain},
            vl = 1,
            parameters = #{target_level => l0, output_index => 0}
        },
        #actuator{
            name = l1_tactical_signal_2,
            type = lc_chain_output,
            scape = {private, lc_chain},
            vl = 1,
            parameters = #{target_level => l0, output_index => 1}
        },
        #actuator{
            name = l1_tactical_signal_3,
            type = lc_chain_output,
            scape = {private, lc_chain},
            vl = 1,
            parameters = #{target_level => l0, output_index => 2}
        },
        #actuator{
            name = l1_tactical_signal_4,
            type = lc_chain_output,
            scape = {private, lc_chain},
            vl = 1,
            parameters = #{target_level => l0, output_index => 3}
        },
        #actuator{
            name = l1_tactical_signal_5,
            type = lc_chain_output,
            scape = {private, lc_chain},
            vl = 1,
            parameters = #{target_level => l0, output_index => 4}
        }
    ];
get_actuators(_) ->
    error(invalid_morphology).
