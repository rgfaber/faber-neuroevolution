%% @doc Tests for agent_actuator behaviour.
%%
%% Tests validation logic, callback contracts, and output validation
%% for the agent_actuator behaviour module.
%%
%% @author R.G. Lefever
%% @copyright 2024-2026 R.G. Lefever
-module(agent_actuator_tests).

-include_lib("eunit/include/eunit.hrl").

%%% ============================================================================
%%% Test Generators
%%% ============================================================================

agent_actuator_test_() ->
    [
        {"validate/1 returns ok for valid module", fun test_validate_valid_module/0},
        {"validate/1 detects missing exports", fun test_validate_missing_exports/0},
        {"validate/1 detects invalid name", fun test_validate_invalid_name/0},
        {"validate/1 detects empty name", fun test_validate_empty_name/0},
        {"validate/1 detects invalid output count", fun test_validate_invalid_output_count/0},
        {"validate/1 detects zero output count", fun test_validate_zero_output_count/0},
        {"get_info/1 returns info for valid module", fun test_get_info_valid/0},
        {"get_info/1 returns error for invalid module", fun test_get_info_invalid/0}
    ].

output_validation_test_() ->
    [
        {"validate_outputs/2 returns ok for correct count", fun test_validate_outputs_correct/0},
        {"validate_outputs/2 detects count mismatch", fun test_validate_outputs_mismatch/0},
        {"validate_outputs/2 detects non-numeric values", fun test_validate_outputs_non_numeric/0},
        {"validate_outputs/2 detects non-list", fun test_validate_outputs_non_list/0},
        {"validate_outputs/2 accepts integers", fun test_validate_outputs_integers/0},
        {"validate_outputs/2 accepts negative values", fun test_validate_outputs_negative/0}
    ].

act_callback_test_() ->
    [
        {"act/3 returns action for valid outputs", fun test_act_returns_action/0},
        {"act/3 produces move action", fun test_act_move_action/0},
        {"act/3 produces stay action", fun test_act_stay_action/0}
    ].

%%% ============================================================================
%%% Validation Test Cases
%%% ============================================================================

test_validate_valid_module() ->
    ?assertEqual(ok, agent_actuator:validate(test_valid_actuator)).

test_validate_missing_exports() ->
    Result = agent_actuator:validate(test_missing_exports_actuator),
    ?assertMatch({error, _}, Result),
    {error, Errors} = Result,
    ?assert(lists:any(
        fun({missing_exports, Missing}) ->
            lists:member({output_count, 0}, Missing) orelse
            lists:member({act, 3}, Missing);
           (_) -> false
        end,
        Errors
    )).

test_validate_invalid_name() ->
    Result = agent_actuator:validate(test_invalid_name_actuator),
    ?assertMatch({error, _}, Result),
    {error, Errors} = Result,
    ?assert(lists:any(
        fun({invalid_name, _}) -> true;
           (_) -> false
        end,
        Errors
    )).

test_validate_empty_name() ->
    Result = agent_actuator:validate(test_empty_name_actuator),
    ?assertMatch({error, _}, Result),
    {error, Errors} = Result,
    ?assert(lists:any(
        fun({invalid_name, empty_binary}) -> true;
           (_) -> false
        end,
        Errors
    )).

test_validate_invalid_output_count() ->
    Result = agent_actuator:validate(test_invalid_output_count_actuator),
    ?assertMatch({error, _}, Result),
    {error, Errors} = Result,
    ?assert(lists:any(
        fun({invalid_output_count, _}) -> true;
           (_) -> false
        end,
        Errors
    )).

test_validate_zero_output_count() ->
    Result = agent_actuator:validate(test_zero_output_count_actuator),
    ?assertMatch({error, _}, Result),
    {error, Errors} = Result,
    ?assert(lists:any(
        fun({invalid_output_count, {must_be_positive, 0}}) -> true;
           (_) -> false
        end,
        Errors
    )).

test_get_info_valid() ->
    Result = agent_actuator:get_info(test_valid_actuator),
    ?assertMatch({ok, _}, Result),
    {ok, Info} = Result,

    ?assertEqual(<<"movement">>, maps:get(name, Info)),
    ?assertEqual(7, maps:get(output_count, Info)).

test_get_info_invalid() ->
    Result = agent_actuator:get_info(test_invalid_name_actuator),
    ?assertMatch({error, _}, Result).

%%% ============================================================================
%%% Output Validation Test Cases
%%% ============================================================================

test_validate_outputs_correct() ->
    %% test_valid_actuator has output_count = 7
    Outputs = lists:duplicate(7, 0.5),
    ?assertEqual(ok, agent_actuator:validate_outputs(test_valid_actuator, Outputs)).

test_validate_outputs_mismatch() ->
    %% test_valid_actuator has output_count = 7, but we provide 3
    Outputs = lists:duplicate(3, 0.5),
    Result = agent_actuator:validate_outputs(test_valid_actuator, Outputs),
    ?assertMatch({error, {output_count_mismatch, _}}, Result).

test_validate_outputs_non_numeric() ->
    %% test_valid_actuator has output_count = 7
    Outputs = lists:duplicate(6, 0.5) ++ [not_a_number],
    Result = agent_actuator:validate_outputs(test_valid_actuator, Outputs),
    ?assertMatch({error, {non_numeric_outputs, _}}, Result).

test_validate_outputs_non_list() ->
    Result = agent_actuator:validate_outputs(test_valid_actuator, not_a_list),
    ?assertMatch({error, {outputs_not_list, _}}, Result).

test_validate_outputs_integers() ->
    %% Integers are valid numbers
    Outputs = lists:duplicate(7, 1),
    ?assertEqual(ok, agent_actuator:validate_outputs(test_valid_actuator, Outputs)).

test_validate_outputs_negative() ->
    %% Negative values are valid (neural network outputs can be negative)
    Outputs = [-0.5, -0.3, 0.0, 0.2, 0.4, 0.6, 0.8],
    ?assertEqual(ok, agent_actuator:validate_outputs(test_valid_actuator, Outputs)).

%%% ============================================================================
%%% Act Callback Test Cases
%%% ============================================================================

test_act_returns_action() ->
    AgentState = #{energy => 50.0},
    EnvState = #{},
    Outputs = [0.1, 0.2, 0.8, 0.1, 0.1, 0.1, 0.1],  %% Direction 2 (NE) highest
    Result = test_valid_actuator:act(Outputs, AgentState, EnvState),
    ?assertMatch({ok, _}, Result).

test_act_move_action() ->
    AgentState = #{energy => 50.0},
    EnvState = #{},
    Outputs = [0.1, 0.2, 0.8, 0.1, 0.1, 0.1, 0.1],  %% Direction 2 (NE) highest
    {ok, Action} = test_valid_actuator:act(Outputs, AgentState, EnvState),
    ?assertEqual(move, maps:get(type, Action)),
    ?assertEqual(2, maps:get(direction, Action)).

test_act_stay_action() ->
    AgentState = #{energy => 50.0},
    EnvState = #{},
    Outputs = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.9],  %% Stay (6) highest
    {ok, Action} = test_valid_actuator:act(Outputs, AgentState, EnvState),
    ?assertEqual(stay, maps:get(type, Action)).
