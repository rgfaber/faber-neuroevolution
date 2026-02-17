%% @doc Tests for agent_bridge orchestration module.
%%
%% Tests validation, sensing, acting, and full episode execution
%% for the agent_bridge module.
%%
%% @author R.G. Lefever
%% @copyright 2024-2026 R.G. Lefever
-module(agent_bridge_tests).

-include_lib("eunit/include/eunit.hrl").

%%% ============================================================================
%%% Test Generators
%%% ============================================================================

validation_test_() ->
    [
        {"new/1 returns validated bridge for valid config", fun test_new_valid_config/0},
        {"new/1 detects missing definition", fun test_new_missing_definition/0},
        {"new/1 detects invalid sensor", fun test_new_invalid_sensor/0},
        {"new/1 detects invalid actuator", fun test_new_invalid_actuator/0},
        {"new/1 detects topology input mismatch", fun test_new_topology_input_mismatch/0},
        {"new/1 detects topology output mismatch", fun test_new_topology_output_mismatch/0}
    ].

sensing_test_() ->
    [
        {"sense/3 collects inputs from all sensors", fun test_sense_collects_inputs/0},
        {"sense/3 returns correct total count", fun test_sense_correct_count/0},
        {"sense/3 maintains sensor order", fun test_sense_maintains_order/0}
    ].

acting_test_() ->
    [
        {"act/4 slices outputs correctly", fun test_act_slices_correctly/0},
        {"act/4 returns actions from all actuators", fun test_act_returns_actions/0}
    ].

cycle_test_() ->
    [
        {"sense_think_act/4 executes full cycle", fun test_sense_think_act_cycle/0}
    ].

episode_test_() ->
    [
        {"run_episode/3 completes successfully", fun test_run_episode_completes/0},
        {"run_episode/3 returns metrics", fun test_run_episode_returns_metrics/0}
    ].

%%% ============================================================================
%%% Validation Test Cases
%%% ============================================================================

test_new_valid_config() ->
    Config = valid_bridge_config(),
    Result = agent_bridge:new(Config),
    ?assertMatch({ok, _}, Result),
    {ok, Bridge} = Result,
    ?assertEqual(test_bridge_agent, maps:get(definition, Bridge)),
    ?assertEqual(23, maps:get(total_inputs, Bridge)),  %% 18 + 4 + 1
    ?assertEqual(8, maps:get(total_outputs, Bridge)).   %% 7 + 1

test_new_missing_definition() ->
    Config = #{
        sensors => [test_bridge_vision_sensor],
        actuators => [test_bridge_movement_actuator],
        environment => test_valid_environment
    },
    Result = agent_bridge:new(Config),
    ?assertMatch({error, _}, Result).

test_new_invalid_sensor() ->
    Config = #{
        definition => test_bridge_agent,
        sensors => [nonexistent_sensor_module],
        actuators => [test_bridge_movement_actuator],
        environment => test_valid_environment
    },
    Result = agent_bridge:new(Config),
    ?assertMatch({error, _}, Result).

test_new_invalid_actuator() ->
    Config = #{
        definition => test_bridge_agent,
        sensors => [test_bridge_vision_sensor],
        actuators => [nonexistent_actuator_module],
        environment => test_valid_environment
    },
    Result = agent_bridge:new(Config),
    ?assertMatch({error, _}, Result).

test_new_topology_input_mismatch() ->
    %% Agent expects 23 inputs, but we only register vision (18)
    Config = #{
        definition => test_bridge_agent,
        sensors => [test_bridge_vision_sensor],  %% Only 18 inputs, need 23
        actuators => [test_bridge_movement_actuator, test_bridge_signal_actuator],
        environment => test_valid_environment
    },
    Result = agent_bridge:new(Config),
    ?assertMatch({error, {topology_mismatch, #{type := inputs}}}, Result).

test_new_topology_output_mismatch() ->
    %% Agent expects 8 outputs, but we only register movement (7)
    Config = #{
        definition => test_bridge_agent,
        sensors => [test_bridge_vision_sensor, test_bridge_hearing_sensor, test_bridge_energy_sensor],
        actuators => [test_bridge_movement_actuator],  %% Only 7 outputs, need 8
        environment => test_valid_environment
    },
    Result = agent_bridge:new(Config),
    ?assertMatch({error, {topology_mismatch, #{type := outputs}}}, Result).

%%% ============================================================================
%%% Sensing Test Cases
%%% ============================================================================

test_sense_collects_inputs() ->
    {ok, Bridge} = agent_bridge:new(valid_bridge_config()),
    AgentState = #{pos => {0, 0}, energy => 50.0},
    EnvState = #{},
    Inputs = agent_bridge:sense(Bridge, AgentState, EnvState),
    ?assert(is_list(Inputs)),
    ?assert(length(Inputs) > 0).

test_sense_correct_count() ->
    {ok, Bridge} = agent_bridge:new(valid_bridge_config()),
    AgentState = #{pos => {0, 0}, energy => 50.0},
    EnvState = #{},
    Inputs = agent_bridge:sense(Bridge, AgentState, EnvState),
    ?assertEqual(23, length(Inputs)).  %% 18 + 4 + 1

test_sense_maintains_order() ->
    {ok, Bridge} = agent_bridge:new(valid_bridge_config()),
    AgentState = #{pos => {0, 0}, energy => 75.0},  %% 75/100 = 0.75 normalized
    EnvState = #{},
    Inputs = agent_bridge:sense(Bridge, AgentState, EnvState),

    %% First 18 values from vision (all 0.0)
    VisionInputs = lists:sublist(Inputs, 1, 18),
    ?assert(lists:all(fun(V) -> V == 0.0 end, VisionInputs)),

    %% Next 4 values from hearing (all 0.0)
    HearingInputs = lists:sublist(Inputs, 19, 4),
    ?assert(lists:all(fun(V) -> V == 0.0 end, HearingInputs)),

    %% Last value from energy sensor (75.0 / 100.0 = 0.75)
    [EnergyInput] = lists:sublist(Inputs, 23, 1),
    ?assertEqual(0.75, EnergyInput).

%%% ============================================================================
%%% Acting Test Cases
%%% ============================================================================

test_act_slices_correctly() ->
    {ok, Bridge} = agent_bridge:new(valid_bridge_config()),
    AgentState = #{pos => {0, 0}, energy => 50.0},
    EnvState = #{},

    %% 8 outputs: 7 for movement, 1 for signal
    Outputs = [0.1, 0.2, 0.8, 0.1, 0.1, 0.1, 0.1, 0.9],

    Actions = agent_bridge:act(Bridge, Outputs, AgentState, EnvState),
    ?assertEqual(2, length(Actions)).

test_act_returns_actions() ->
    {ok, Bridge} = agent_bridge:new(valid_bridge_config()),
    AgentState = #{pos => {0, 0}, energy => 50.0},
    EnvState = #{},

    %% Direction 2 (NE) highest, signal 0.9
    Outputs = [0.1, 0.2, 0.8, 0.1, 0.1, 0.1, 0.1, 0.9],

    Actions = agent_bridge:act(Bridge, Outputs, AgentState, EnvState),

    %% First action should be movement
    [MoveAction, SignalAction] = Actions,
    ?assertEqual(move, maps:get(type, MoveAction)),
    ?assertEqual(2, maps:get(direction, MoveAction)),

    %% Second action should be signal
    ?assertEqual(signal, maps:get(type, SignalAction)),
    ?assertEqual(0.9, maps:get(strength, SignalAction)).

%%% ============================================================================
%%% Cycle Test Cases
%%% ============================================================================

test_sense_think_act_cycle() ->
    {ok, Bridge} = agent_bridge:new(valid_bridge_config()),
    AgentState = #{pos => {0, 0}, energy => 50.0},
    EnvState = #{},

    %% Simple network that returns fixed outputs
    Network = fun(_Inputs) -> [0.1, 0.2, 0.8, 0.1, 0.1, 0.1, 0.1, 0.5] end,

    {Inputs, Outputs, Actions, UpdatedNetwork} =
        agent_bridge:sense_think_act(Bridge, Network, AgentState, EnvState),

    ?assertEqual(23, length(Inputs)),
    ?assertEqual(8, length(Outputs)),
    ?assertEqual(2, length(Actions)),
    %% For function-based networks, the network is returned unchanged
    ?assertEqual(Network, UpdatedNetwork).

%%% ============================================================================
%%% Episode Test Cases
%%% ============================================================================

test_run_episode_completes() ->
    {ok, Bridge} = agent_bridge:new(valid_bridge_config()),

    %% Simple network that always stays
    Network = fun(_Inputs) -> [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.5] end,

    EnvConfig = #{max_ticks => 5},
    Result = agent_bridge:run_episode(Bridge, Network, EnvConfig),
    ?assertMatch({ok, _}, Result).

test_run_episode_returns_metrics() ->
    {ok, Bridge} = agent_bridge:new(valid_bridge_config()),

    %% Simple network that always stays
    Network = fun(_Inputs) -> [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.5] end,

    EnvConfig = #{max_ticks => 5},
    {ok, Metrics} = agent_bridge:run_episode(Bridge, Network, EnvConfig),

    ?assert(maps:is_key(ticks_survived, Metrics)),
    ?assert(maps:is_key(food_eaten, Metrics)),
    ?assert(maps:is_key(final_energy, Metrics)).

%%% ============================================================================
%%% Helper Functions
%%% ============================================================================

valid_bridge_config() ->
    #{
        definition => test_bridge_agent,
        sensors => [
            test_bridge_vision_sensor,   %% 18 inputs
            test_bridge_hearing_sensor,  %% 4 inputs
            test_bridge_energy_sensor    %% 1 input
        ],
        actuators => [
            test_bridge_movement_actuator,  %% 7 outputs
            test_bridge_signal_actuator     %% 1 output
        ],
        environment => test_valid_environment
    }.
