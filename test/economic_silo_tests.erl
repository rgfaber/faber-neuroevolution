%% @doc Unit tests for economic_silo module.
%%
%% Tests compute budget and resource allocation functionality.
%%
%% @author R.G. Lefever
%% @copyright 2024-2026 R.G. Lefever
-module(economic_silo_tests).

-include_lib("eunit/include/eunit.hrl").

%%% ============================================================================
%%% Setup/Teardown
%%% ============================================================================

setup() ->
    case whereis(economic_silo) of
        undefined -> ok;
        Pid ->
            catch gen_server:stop(Pid, normal, 1000),
            timer:sleep(50)
    end,
    ok.

cleanup(_) ->
    case whereis(economic_silo) of
        undefined -> ok;
        Pid ->
            catch gen_server:stop(Pid, normal, 1000),
            timer:sleep(50)
    end,
    ok.

%%% ============================================================================
%%% Test Suite
%%% ============================================================================

economic_silo_test_() ->
    {foreach,
     fun setup/0,
     fun cleanup/1,
     [
         {"start_link with default config", fun start_link_default_test/0},
         {"start_link with custom config", fun start_link_custom_config_test/0},
         {"get_params returns defaults", fun get_params_test/0},
         {"allocate_budget creates account", fun allocate_budget_test/0},
         {"record_expenditure updates balance", fun record_expenditure_test/0},
         {"record_income updates balance", fun record_income_test/0},
         {"get_balance returns account balance", fun get_balance_test/0},
         {"get_wealth_distribution returns metrics", fun get_wealth_distribution_test/0},
         {"reset clears state", fun reset_test/0},
         {"get_state returns full state", fun get_state_test/0}
     ]}.

%%% ============================================================================
%%% Individual Tests
%%% ============================================================================

start_link_default_test() ->
    {ok, Pid} = economic_silo:start_link(),
    ?assert(is_pid(Pid)),
    ?assert(is_process_alive(Pid)),
    ok = gen_server:stop(Pid).

start_link_custom_config_test() ->
    Config = #{
        realm => <<"test_realm">>,
        initial_budget => 5000.0,
        enabled_levels => [l0]
    },
    {ok, Pid} = economic_silo:start_link(Config),
    ?assert(is_pid(Pid)),
    ok = gen_server:stop(Pid).

get_params_test() ->
    {ok, Pid} = economic_silo:start_link(),
    Params = economic_silo:get_params(Pid),
    ?assert(is_map(Params)),
    ?assert(maps:is_key(budget_per_individual, Params)),
    ?assert(maps:is_key(energy_tax_rate, Params)),
    ?assert(maps:is_key(wealth_redistribution_rate, Params)),
    ?assertEqual(1.0, maps:get(budget_per_individual, Params)),
    ok = gen_server:stop(Pid).

allocate_budget_test() ->
    {ok, Pid} = economic_silo:start_link(),
    %% Allocate budget to an individual
    {ok, Balance} = economic_silo:allocate_budget(Pid, individual_1),
    ?assertEqual(1.0, Balance),

    %% Verify account was created
    {ok, StoredBalance} = economic_silo:get_balance(Pid, individual_1),
    ?assertEqual(1.0, StoredBalance),
    ok = gen_server:stop(Pid).

record_expenditure_test() ->
    {ok, Pid} = economic_silo:start_link(),
    %% First allocate budget
    {ok, _} = economic_silo:allocate_budget(Pid, individual_1),

    %% Record expenditure
    ok = economic_silo:record_expenditure(Pid, individual_1, 0.3),
    timer:sleep(50),

    %% Check balance decreased
    {ok, Balance} = economic_silo:get_balance(Pid, individual_1),
    ?assert(Balance < 1.0),
    ?assert(Balance >= 0.7),
    ok = gen_server:stop(Pid).

record_income_test() ->
    {ok, Pid} = economic_silo:start_link(),
    %% First allocate budget
    {ok, _} = economic_silo:allocate_budget(Pid, individual_1),

    %% Record income
    ok = economic_silo:record_income(Pid, individual_1, 0.5),
    timer:sleep(50),

    %% Check balance increased
    {ok, Balance} = economic_silo:get_balance(Pid, individual_1),
    ?assert(Balance >= 1.5),
    ok = gen_server:stop(Pid).

get_balance_test() ->
    {ok, Pid} = economic_silo:start_link(),

    %% Non-existent account
    ?assertEqual(not_found, economic_silo:get_balance(Pid, unknown_id)),

    %% Create account and check balance
    {ok, _} = economic_silo:allocate_budget(Pid, test_id),
    {ok, Balance} = economic_silo:get_balance(Pid, test_id),
    ?assertEqual(1.0, Balance),
    ok = gen_server:stop(Pid).

get_wealth_distribution_test() ->
    {ok, Pid} = economic_silo:start_link(),

    %% Create several accounts with different balances
    {ok, _} = economic_silo:allocate_budget(Pid, ind_1),
    {ok, _} = economic_silo:allocate_budget(Pid, ind_2),
    {ok, _} = economic_silo:allocate_budget(Pid, ind_3),

    %% Add income to create inequality
    ok = economic_silo:record_income(Pid, ind_1, 5.0),
    timer:sleep(50),

    {Mean, Gini} = economic_silo:get_wealth_distribution(Pid),
    ?assert(is_float(Mean)),
    ?assert(is_float(Gini)),
    ?assert(Mean > 0),
    ?assert(Gini >= 0.0),
    ?assert(Gini =< 1.0),
    ok = gen_server:stop(Pid).

reset_test() ->
    {ok, Pid} = economic_silo:start_link(),
    %% Add some data
    {ok, _} = economic_silo:allocate_budget(Pid, individual_1),
    ok = economic_silo:record_expenditure(Pid, individual_1, 0.2),
    timer:sleep(50),

    %% Reset
    ok = economic_silo:reset(Pid),

    %% Verify reset
    ?assertEqual(not_found, economic_silo:get_balance(Pid, individual_1)),
    ok = gen_server:stop(Pid).

get_state_test() ->
    {ok, Pid} = economic_silo:start_link(#{realm => <<"test">>}),
    State = economic_silo:get_state(Pid),
    ?assert(is_map(State)),
    ?assertEqual(<<"test">>, maps:get(realm, State)),
    ?assert(maps:is_key(current_params, State)),
    ?assert(maps:is_key(total_budget, State)),
    ?assert(maps:is_key(sensors, State)),
    ok = gen_server:stop(Pid).

%%% ============================================================================
%%% Behavior Callback Tests
%%% ============================================================================

behavior_callbacks_test_() ->
    [
        {"get_silo_type returns economic",
         ?_assertEqual(economic, economic_silo:get_silo_type())},
        {"get_time_constant returns 20.0",
         ?_assertEqual(20.0, economic_silo:get_time_constant())}
    ].
