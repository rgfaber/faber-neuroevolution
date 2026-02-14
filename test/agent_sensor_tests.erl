%% @doc Tests for agent_sensor behaviour.
%%
%% Tests validation logic, callback contracts, and value validation
%% for the agent_sensor behaviour module.
%%
%% @author R.G. Lefever
%% @copyright 2024-2026 R.G. Lefever
-module(agent_sensor_tests).

-include_lib("eunit/include/eunit.hrl").

%%% ============================================================================
%%% Test Generators
%%% ============================================================================

agent_sensor_test_() ->
    [
        {"validate/1 returns ok for valid module", fun test_validate_valid_module/0},
        {"validate/1 detects missing exports", fun test_validate_missing_exports/0},
        {"validate/1 detects invalid name", fun test_validate_invalid_name/0},
        {"validate/1 detects empty name", fun test_validate_empty_name/0},
        {"validate/1 detects invalid input count", fun test_validate_invalid_input_count/0},
        {"validate/1 detects zero input count", fun test_validate_zero_input_count/0},
        {"get_info/1 returns info for valid module", fun test_get_info_valid/0},
        {"get_info/1 returns error for invalid module", fun test_get_info_invalid/0}
    ].

value_validation_test_() ->
    [
        {"validate_values/2 returns ok for correct count", fun test_validate_values_correct/0},
        {"validate_values/2 detects count mismatch", fun test_validate_values_mismatch/0},
        {"validate_values/2 detects non-numeric values", fun test_validate_values_non_numeric/0},
        {"validate_values/2 detects non-list", fun test_validate_values_non_list/0},
        {"validate_values/2 accepts integers", fun test_validate_values_integers/0}
    ].

read_callback_test_() ->
    [
        {"read/2 returns correct number of values", fun test_read_correct_count/0},
        {"read/2 returns normalized values", fun test_read_normalized/0}
    ].

%%% ============================================================================
%%% Validation Test Cases
%%% ============================================================================

test_validate_valid_module() ->
    ?assertEqual(ok, agent_sensor:validate(test_valid_sensor)).

test_validate_missing_exports() ->
    Result = agent_sensor:validate(test_missing_exports_sensor),
    ?assertMatch({error, _}, Result),
    {error, Errors} = Result,
    ?assert(lists:any(
        fun({missing_exports, Missing}) ->
            lists:member({input_count, 0}, Missing) orelse
            lists:member({read, 2}, Missing);
           (_) -> false
        end,
        Errors
    )).

test_validate_invalid_name() ->
    Result = agent_sensor:validate(test_invalid_name_sensor),
    ?assertMatch({error, _}, Result),
    {error, Errors} = Result,
    ?assert(lists:any(
        fun({invalid_name, _}) -> true;
           (_) -> false
        end,
        Errors
    )).

test_validate_empty_name() ->
    Result = agent_sensor:validate(test_empty_name_sensor),
    ?assertMatch({error, _}, Result),
    {error, Errors} = Result,
    ?assert(lists:any(
        fun({invalid_name, empty_binary}) -> true;
           (_) -> false
        end,
        Errors
    )).

test_validate_invalid_input_count() ->
    Result = agent_sensor:validate(test_invalid_input_count_sensor),
    ?assertMatch({error, _}, Result),
    {error, Errors} = Result,
    ?assert(lists:any(
        fun({invalid_input_count, _}) -> true;
           (_) -> false
        end,
        Errors
    )).

test_validate_zero_input_count() ->
    Result = agent_sensor:validate(test_zero_input_count_sensor),
    ?assertMatch({error, _}, Result),
    {error, Errors} = Result,
    ?assert(lists:any(
        fun({invalid_input_count, {must_be_positive, 0}}) -> true;
           (_) -> false
        end,
        Errors
    )).

test_get_info_valid() ->
    Result = agent_sensor:get_info(test_valid_sensor),
    ?assertMatch({ok, _}, Result),
    {ok, Info} = Result,

    ?assertEqual(<<"vision">>, maps:get(name, Info)),
    ?assertEqual(18, maps:get(input_count, Info)).

test_get_info_invalid() ->
    Result = agent_sensor:get_info(test_invalid_name_sensor),
    ?assertMatch({error, _}, Result).

%%% ============================================================================
%%% Value Validation Test Cases
%%% ============================================================================

test_validate_values_correct() ->
    %% test_valid_sensor has input_count = 18
    Values = lists:duplicate(18, 0.5),
    ?assertEqual(ok, agent_sensor:validate_values(test_valid_sensor, Values)).

test_validate_values_mismatch() ->
    %% test_valid_sensor has input_count = 18, but we provide 10
    Values = lists:duplicate(10, 0.5),
    Result = agent_sensor:validate_values(test_valid_sensor, Values),
    ?assertMatch({error, {value_count_mismatch, _}}, Result).

test_validate_values_non_numeric() ->
    %% test_valid_sensor has input_count = 18
    Values = lists:duplicate(17, 0.5) ++ [not_a_number],
    Result = agent_sensor:validate_values(test_valid_sensor, Values),
    ?assertMatch({error, {non_numeric_values, _}}, Result).

test_validate_values_non_list() ->
    Result = agent_sensor:validate_values(test_valid_sensor, not_a_list),
    ?assertMatch({error, {values_not_list, _}}, Result).

test_validate_values_integers() ->
    %% Integers are valid numbers
    Values = lists:duplicate(18, 1),
    ?assertEqual(ok, agent_sensor:validate_values(test_valid_sensor, Values)).

%%% ============================================================================
%%% Read Callback Test Cases
%%% ============================================================================

test_read_correct_count() ->
    AgentState = #{hex => {0, 0}, energy => 50.0},
    EnvState = #{arena_radius => 10, food => #{}, walls => #{}},
    Values = test_valid_sensor:read(AgentState, EnvState),
    ?assertEqual(18, length(Values)).

test_read_normalized() ->
    AgentState = #{hex => {0, 0}, energy => 50.0},
    EnvState = #{arena_radius => 10, food => #{}, walls => #{}},
    Values = test_valid_sensor:read(AgentState, EnvState),
    %% All values should be between 0.0 and 1.0
    ?assert(lists:all(fun(V) -> V >= 0.0 andalso V =< 1.0 end, Values)).
