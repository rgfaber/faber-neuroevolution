%% @doc Tests for agent_definition behaviour.
%%
%% Tests validation logic, callback contracts, and error handling
%% for the agent_definition behaviour module.
%%
%% @author R.G. Lefever
%% @copyright 2024-2026 R.G. Lefever
-module(agent_definition_tests).

-include_lib("eunit/include/eunit.hrl").

%%% ============================================================================
%%% Test Generators
%%% ============================================================================

agent_definition_test_() ->
    [
        {"validate/1 returns ok for valid module", fun test_validate_valid_module/0},
        {"validate/1 detects missing exports", fun test_validate_missing_exports/0},
        {"validate/1 detects invalid name", fun test_validate_invalid_name/0},
        {"validate/1 detects empty name", fun test_validate_empty_name/0},
        {"validate/1 detects invalid version format", fun test_validate_invalid_version_format/0},
        {"validate/1 detects invalid topology", fun test_validate_invalid_topology/0},
        {"validate/1 detects invalid hidden layers", fun test_validate_invalid_hidden_layers/0},
        {"get_info/1 returns info for valid module", fun test_get_info_valid/0},
        {"get_info/1 returns error for invalid module", fun test_get_info_invalid/0}
    ].

edge_case_test_() ->
    [
        {"empty hidden layers is valid", fun test_empty_hidden_layers/0},
        {"single hidden layer is valid", fun test_single_hidden_layer/0},
        {"many hidden layers is valid", fun test_many_hidden_layers/0},
        {"version with prerelease is valid", fun test_version_prerelease/0}
    ].

%%% ============================================================================
%%% Test Cases
%%% ============================================================================

test_validate_valid_module() ->
    ?assertEqual(ok, agent_definition:validate(test_valid_agent)).

test_validate_missing_exports() ->
    Result = agent_definition:validate(test_missing_exports_agent),
    ?assertMatch({error, _}, Result),
    {error, Errors} = Result,
    ?assert(lists:any(
        fun({missing_exports, Missing}) ->
            lists:member({version, 0}, Missing) andalso
            lists:member({network_topology, 0}, Missing);
           (_) -> false
        end,
        Errors
    )).

test_validate_invalid_name() ->
    Result = agent_definition:validate(test_invalid_name_agent),
    ?assertMatch({error, _}, Result),
    {error, Errors} = Result,
    ?assert(lists:any(
        fun({invalid_name, _}) -> true;
           (_) -> false
        end,
        Errors
    )).

test_validate_empty_name() ->
    Result = agent_definition:validate(test_empty_name_agent),
    ?assertMatch({error, _}, Result),
    {error, Errors} = Result,
    ?assert(lists:any(
        fun({invalid_name, empty_binary}) -> true;
           (_) -> false
        end,
        Errors
    )).

test_validate_invalid_version_format() ->
    Result = agent_definition:validate(test_bad_version_agent),
    ?assertMatch({error, _}, Result),
    {error, Errors} = Result,
    ?assert(lists:any(
        fun({invalid_version_format, _}) -> true;
           (_) -> false
        end,
        Errors
    )).

test_validate_invalid_topology() ->
    Result = agent_definition:validate(test_bad_topology_agent),
    ?assertMatch({error, _}, Result),
    {error, Errors} = Result,
    ?assert(lists:any(
        fun({invalid_topology, _}) -> true;
           (_) -> false
        end,
        Errors
    )).

test_validate_invalid_hidden_layers() ->
    Result = agent_definition:validate(test_bad_hidden_agent),
    ?assertMatch({error, _}, Result),
    {error, Errors} = Result,
    ?assert(lists:any(
        fun({invalid_hidden_layers, _}) -> true;
           (_) -> false
        end,
        Errors
    )).

test_get_info_valid() ->
    Result = agent_definition:get_info(test_valid_agent),
    ?assertMatch({ok, _}, Result),
    {ok, Info} = Result,

    ?assertEqual(<<"test_arena_agent">>, maps:get(name, Info)),
    ?assertEqual(<<"1.0.0">>, maps:get(version, Info)),
    ?assertEqual({29, [32, 16], 9}, maps:get(topology, Info)),
    ?assertEqual(29, maps:get(inputs, Info)),
    ?assertEqual([32, 16], maps:get(hidden_layers, Info)),
    ?assertEqual(9, maps:get(outputs, Info)).

test_get_info_invalid() ->
    Result = agent_definition:get_info(test_invalid_name_agent),
    ?assertMatch({error, _}, Result).

%%% ============================================================================
%%% Edge Case Tests
%%% ============================================================================

test_empty_hidden_layers() ->
    ?assertEqual(ok, agent_definition:validate(test_no_hidden_agent)).

test_single_hidden_layer() ->
    ?assertEqual(ok, agent_definition:validate(test_single_hidden_agent)).

test_many_hidden_layers() ->
    ?assertEqual(ok, agent_definition:validate(test_many_hidden_agent)).

test_version_prerelease() ->
    %% Prerelease versions (1.0.0-alpha) should still match the basic pattern
    %% Our regex only checks for X.Y.Z prefix
    ?assertEqual(ok, agent_definition:validate(test_prerelease_agent)).
