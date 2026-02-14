%% @doc Tests for agent_evaluator behaviour.
%%
%% Tests validation logic, fitness calculation, and component breakdown
%% for the agent_evaluator behaviour module.
%%
%% @author R.G. Lefever
%% @copyright 2024-2026 R.G. Lefever
-module(agent_evaluator_tests).

-include_lib("eunit/include/eunit.hrl").

%%% ============================================================================
%%% Test Generators
%%% ============================================================================

validation_test_() ->
    [
        {"validate/1 returns ok for valid module", fun test_validate_valid_module/0},
        {"validate/1 returns ok for module without components", fun test_validate_no_components/0},
        {"validate/1 detects missing exports", fun test_validate_missing_exports/0},
        {"validate/1 detects invalid name", fun test_validate_invalid_name/0},
        {"validate/1 detects empty name", fun test_validate_empty_name/0},
        {"get_info/1 returns info for valid module", fun test_get_info_valid/0},
        {"get_info/1 indicates components support", fun test_get_info_components/0}
    ].

fitness_test_() ->
    [
        {"calculate_fitness/1 returns fitness", fun test_calculate_fitness/0},
        {"calculate_fitness/1 handles zero metrics", fun test_calculate_fitness_zero/0},
        {"calculate_fitness/1 weights components correctly", fun test_calculate_fitness_weights/0}
    ].

evaluate_test_() ->
    [
        {"evaluate/2 returns fitness", fun test_evaluate_returns_fitness/0},
        {"evaluate/2 handles errors", fun test_evaluate_handles_errors/0},
        {"evaluate_with_breakdown/2 returns components", fun test_evaluate_with_breakdown/0},
        {"evaluate_with_breakdown/2 handles no components", fun test_evaluate_no_breakdown/0}
    ].

%%% ============================================================================
%%% Validation Test Cases
%%% ============================================================================

test_validate_valid_module() ->
    ?assertEqual(ok, agent_evaluator:validate(test_valid_evaluator)).

test_validate_no_components() ->
    ?assertEqual(ok, agent_evaluator:validate(test_simple_evaluator)).

test_validate_missing_exports() ->
    Result = agent_evaluator:validate(test_missing_exports_evaluator),
    ?assertMatch({error, _}, Result),
    {error, Errors} = Result,
    ?assert(lists:any(
        fun({missing_exports, Missing}) ->
            lists:member({calculate_fitness, 1}, Missing);
           (_) -> false
        end,
        Errors
    )).

test_validate_invalid_name() ->
    Result = agent_evaluator:validate(test_invalid_name_evaluator),
    ?assertMatch({error, _}, Result),
    {error, Errors} = Result,
    ?assert(lists:any(
        fun({invalid_name, _}) -> true;
           (_) -> false
        end,
        Errors
    )).

test_validate_empty_name() ->
    Result = agent_evaluator:validate(test_empty_name_evaluator),
    ?assertMatch({error, _}, Result),
    {error, Errors} = Result,
    ?assert(lists:any(
        fun({invalid_name, empty_binary}) -> true;
           (_) -> false
        end,
        Errors
    )).

test_get_info_valid() ->
    Result = agent_evaluator:get_info(test_valid_evaluator),
    ?assertMatch({ok, _}, Result),
    {ok, Info} = Result,
    ?assertEqual(<<"hex_arena_fitness">>, maps:get(name, Info)).

test_get_info_components() ->
    {ok, InfoWith} = agent_evaluator:get_info(test_valid_evaluator),
    ?assertEqual(true, maps:get(has_components, InfoWith)),

    {ok, InfoWithout} = agent_evaluator:get_info(test_simple_evaluator),
    ?assertEqual(false, maps:get(has_components, InfoWithout)).

%%% ============================================================================
%%% Fitness Calculation Test Cases
%%% ============================================================================

test_calculate_fitness() ->
    Metrics = #{
        ticks_survived => 100,
        food_eaten => 5,
        kills => 2
    },
    Fitness = test_valid_evaluator:calculate_fitness(Metrics),
    %% 100 * 0.1 + 5 * 150.0 + 2 * 100.0 = 10 + 750 + 200 = 960
    ?assertEqual(960.0, Fitness).

test_calculate_fitness_zero() ->
    Metrics = #{},
    Fitness = test_valid_evaluator:calculate_fitness(Metrics),
    ?assertEqual(0.0, Fitness).

test_calculate_fitness_weights() ->
    %% Test that food is weighted more than survival
    MetricsSurvival = #{ticks_survived => 1000, food_eaten => 0, kills => 0},
    MetricsFood = #{ticks_survived => 0, food_eaten => 1, kills => 0},

    FitnessSurvival = test_valid_evaluator:calculate_fitness(MetricsSurvival),
    FitnessFood = test_valid_evaluator:calculate_fitness(MetricsFood),

    %% 1000 * 0.1 = 100 vs 1 * 150 = 150
    ?assert(FitnessFood > FitnessSurvival).

%%% ============================================================================
%%% Evaluate API Test Cases
%%% ============================================================================

test_evaluate_returns_fitness() ->
    Metrics = #{ticks_survived => 50, food_eaten => 3, kills => 1},
    Result = agent_evaluator:evaluate(test_valid_evaluator, Metrics),
    ?assertMatch({ok, _}, Result),
    {ok, Fitness} = Result,
    %% 50 * 0.1 + 3 * 150.0 + 1 * 100.0 = 5 + 450 + 100 = 555
    ?assertEqual(555.0, Fitness).

test_evaluate_handles_errors() ->
    Result = agent_evaluator:evaluate(test_invalid_name_evaluator, #{}),
    ?assertMatch({error, _}, Result).

test_evaluate_with_breakdown() ->
    Metrics = #{ticks_survived => 100, food_eaten => 5, kills => 2},
    Result = agent_evaluator:evaluate_with_breakdown(test_valid_evaluator, Metrics),
    ?assertMatch({ok, _, _}, Result),
    {ok, Fitness, Components} = Result,

    ?assertEqual(960.0, Fitness),
    ?assertEqual(10.0, maps:get(survival, Components)),
    ?assertEqual(750.0, maps:get(food, Components)),
    ?assertEqual(200.0, maps:get(kills, Components)).

test_evaluate_no_breakdown() ->
    Metrics = #{ticks_survived => 100},
    Result = agent_evaluator:evaluate_with_breakdown(test_simple_evaluator, Metrics),
    ?assertMatch({ok, _, _}, Result),
    {ok, _Fitness, Components} = Result,

    %% Simple evaluator doesn't implement fitness_components
    ?assertEqual(#{}, Components).
