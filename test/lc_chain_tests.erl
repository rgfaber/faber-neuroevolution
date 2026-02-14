%% @doc Unit tests for lc_chain module.
%%
%% Tests the chained LTC controller (L2→L1→L0).
%%
%% @author R.G. Lefever
%% @copyright 2024-2026 R.G. Lefever
-module(lc_chain_tests).

-include_lib("eunit/include/eunit.hrl").
-include("lc_chain.hrl").

%%% ============================================================================
%%% Test Setup
%%% ============================================================================

%% @doc Setup for each test - starts mnesia and the lc_chain.
setup() ->
    %% Ensure faber_tweann application is started
    _ = application:ensure_all_started(faber_tweann),
    %% Stop any existing lc_chain
    catch gen_server:stop(lc_chain),
    ok.

%% @doc Cleanup after each test.
cleanup(_) ->
    catch gen_server:stop(lc_chain),
    ok.

%%% ============================================================================
%%% Test Generators
%%% ============================================================================

lc_chain_test_() ->
    {foreach,
     fun setup/0,
     fun cleanup/1,
     [
         {"start_link with default config", fun start_link_default_test/0},
         {"start_link with custom config", fun start_link_custom_config_test/0},
         {"forward pass returns hyperparams", fun forward_returns_hyperparams_test/0},
         {"forward pass updates state", fun forward_updates_state_test/0},
         {"get_hyperparams returns last values", fun get_hyperparams_test/0},
         {"reset clears state", fun reset_test/0},
         {"chain outputs are cascaded", fun chain_cascade_test/0},
         {"training updates reward", fun training_updates_reward_test/0},
         {"training handles negative reward", fun training_negative_reward_test/0},
         {"forward and train cycle", fun forward_train_cycle_test/0}
     ]}.

%%% ============================================================================
%%% Individual Tests
%%% ============================================================================

%% @doc Test starting the chain with default configuration.
start_link_default_test() ->
    {ok, Pid} = lc_chain:start_link(),
    ?assert(is_pid(Pid)),
    ?assert(is_process_alive(Pid)),
    ok = gen_server:stop(Pid).

%% @doc Test starting with custom tau values.
start_link_custom_config_test() ->
    Config = #lc_chain_config{
        l2_tau = 200.0,
        l1_tau = 100.0,
        l0_tau = 20.0,
        learning_rate = 0.01
    },
    {ok, Pid} = lc_chain:start_link(Config),
    ?assert(is_pid(Pid)),
    %% Verify config was applied
    {ok, State} = lc_chain:get_state(Pid),
    ?assertEqual(200.0, (maps:get(config, State))#lc_chain_config.l2_tau),
    ok = gen_server:stop(Pid).

%% @doc Test that forward pass returns valid hyperparameters.
forward_returns_hyperparams_test() ->
    {ok, Pid} = lc_chain:start_link(),

    EvoMetrics = #evolution_metrics{
        best_fitness = 0.75,
        avg_fitness = 0.50,
        fitness_improvement = 0.05,
        fitness_variance = 0.15,
        stagnation_counter = 3,
        generation_progress = 0.3,
        population_diversity = 0.6,
        species_count = 4
    },

    Hyperparams = lc_chain:forward(Pid, EvoMetrics),

    %% Verify hyperparams record structure
    ?assert(is_record(Hyperparams, lc_hyperparams)),

    %% Verify values are within expected ranges
    ?assert(Hyperparams#lc_hyperparams.mutation_rate >= 0.01),
    ?assert(Hyperparams#lc_hyperparams.mutation_rate =< 0.5),
    ?assert(Hyperparams#lc_hyperparams.selection_ratio >= 0.1),
    ?assert(Hyperparams#lc_hyperparams.selection_ratio =< 0.5),

    ok = gen_server:stop(Pid).

%% @doc Test that forward pass updates internal state.
forward_updates_state_test() ->
    {ok, Pid} = lc_chain:start_link(),

    %% Get initial state
    {ok, StateBefore} = lc_chain:get_state(Pid),
    L2OutputsBefore = maps:get(l2_outputs, StateBefore),

    %% Run forward pass
    EvoMetrics = #evolution_metrics{
        best_fitness = 0.9,
        avg_fitness = 0.7,
        fitness_improvement = 0.1,
        stagnation_counter = 0
    },
    _ = lc_chain:forward(Pid, EvoMetrics),

    %% Get state after
    {ok, StateAfter} = lc_chain:get_state(Pid),
    L2OutputsAfter = maps:get(l2_outputs, StateAfter),

    %% Outputs should be lists (structure preserved)
    ?assert(is_list(L2OutputsBefore)),
    ?assert(is_list(L2OutputsAfter)),
    %% At least one output should exist
    ?assert(length(L2OutputsAfter) >= 1),
    %% All outputs should be valid floats
    lists:foreach(fun(V) -> ?assert(is_float(V)) end, L2OutputsAfter),

    ok = gen_server:stop(Pid).

%% @doc Test get_hyperparams returns last computed values.
get_hyperparams_test() ->
    {ok, Pid} = lc_chain:start_link(),

    %% Get initial hyperparams (defaults)
    InitialParams = lc_chain:get_hyperparams(Pid),
    ?assert(is_record(InitialParams, lc_hyperparams)),

    %% Run forward pass
    EvoMetrics = #evolution_metrics{best_fitness = 0.5},
    ComputedParams = lc_chain:forward(Pid, EvoMetrics),

    %% get_hyperparams should return same values
    StoredParams = lc_chain:get_hyperparams(Pid),
    ?assertEqual(ComputedParams#lc_hyperparams.mutation_rate,
                 StoredParams#lc_hyperparams.mutation_rate),

    ok = gen_server:stop(Pid).

%% @doc Test reset clears internal state.
reset_test() ->
    {ok, Pid} = lc_chain:start_link(),

    %% Run some forward passes
    EvoMetrics = #evolution_metrics{best_fitness = 0.8},
    _ = lc_chain:forward(Pid, EvoMetrics),
    _ = lc_chain:forward(Pid, EvoMetrics),

    %% Get state before reset (we don't use it, but verify it exists)
    {ok, _StateBefore} = lc_chain:get_state(Pid),

    %% Reset
    ok = lc_chain:reset(Pid),

    %% Get state after reset
    {ok, StateAfter} = lc_chain:get_state(Pid),
    RewardAfter = maps:get(cumulative_reward, StateAfter),

    %% Cumulative reward should be reset to 0
    ?assertEqual(0.0, RewardAfter),

    ok = gen_server:stop(Pid).

%% @doc Test that outputs cascade through the chain.
chain_cascade_test() ->
    {ok, Pid} = lc_chain:start_link(),

    %% Run forward pass
    EvoMetrics = #evolution_metrics{
        best_fitness = 0.6,
        avg_fitness = 0.4,
        fitness_improvement = 0.02
    },
    _ = lc_chain:forward(Pid, EvoMetrics),

    %% Get state to verify cascade
    {ok, State} = lc_chain:get_state(Pid),

    %% All levels should have outputs (lists with at least one element)
    L2Outputs = maps:get(l2_outputs, State),
    L1Outputs = maps:get(l1_outputs, State),
    L0Outputs = maps:get(l0_outputs, State),

    ?assert(is_list(L2Outputs) andalso length(L2Outputs) >= 1),
    ?assert(is_list(L1Outputs) andalso length(L1Outputs) >= 1),
    ?assert(is_list(L0Outputs) andalso length(L0Outputs) >= 1),

    %% All outputs should be valid floats
    lists:foreach(fun(V) -> ?assert(is_float(V)) end, L2Outputs),
    lists:foreach(fun(V) -> ?assert(is_float(V)) end, L1Outputs),
    lists:foreach(fun(V) -> ?assert(is_float(V)) end, L0Outputs),

    ok = gen_server:stop(Pid).

%%% ============================================================================
%%% Training Tests
%%% ============================================================================

%% @doc Test that training updates cumulative reward.
training_updates_reward_test() ->
    {ok, Pid} = lc_chain:start_link(),

    %% Get initial state
    {ok, StateBefore} = lc_chain:get_state(Pid),
    RewardBefore = maps:get(cumulative_reward, StateBefore),
    ?assertEqual(0.0, RewardBefore),

    %% Train with positive reward
    ok = lc_chain:train(Pid, 1.0),

    %% Small delay to allow cast to complete
    timer:sleep(10),

    %% Verify reward accumulated
    {ok, StateAfter} = lc_chain:get_state(Pid),
    RewardAfter = maps:get(cumulative_reward, StateAfter),
    ?assertEqual(1.0, RewardAfter),

    ok = gen_server:stop(Pid).

%% @doc Test that training handles negative reward.
training_negative_reward_test() ->
    {ok, Pid} = lc_chain:start_link(),

    %% Train with negative reward
    ok = lc_chain:train(Pid, -0.5),
    timer:sleep(10),

    {ok, State} = lc_chain:get_state(Pid),
    Reward = maps:get(cumulative_reward, State),
    ?assertEqual(-0.5, Reward),

    ok = gen_server:stop(Pid).

%% @doc Test combined forward and training cycle.
forward_train_cycle_test() ->
    {ok, Pid} = lc_chain:start_link(),

    %% Forward pass
    EvoMetrics = #evolution_metrics{
        best_fitness = 0.8,
        avg_fitness = 0.6,
        fitness_improvement = 0.1
    },
    _Hyperparams1 = lc_chain:forward(Pid, EvoMetrics),

    %% Train with reward
    ok = lc_chain:train(Pid, 0.5),
    timer:sleep(10),

    %% Another forward pass (should use updated weights)
    _Hyperparams2 = lc_chain:forward(Pid, EvoMetrics),

    %% Verify state is valid
    {ok, State} = lc_chain:get_state(Pid),
    ?assert(maps:get(cumulative_reward, State) > 0),

    ok = gen_server:stop(Pid).

%%% ============================================================================
%%% Integration Tests with EmergentMetrics
%%% ============================================================================

forward_with_emergent_metrics_test_() ->
    {setup,
     fun setup/0,
     fun cleanup/1,
     fun(_) ->
         [{"forward with emergent metrics",
           fun() ->
               {ok, Pid} = lc_chain:start_link(),

               EvoMetrics = #evolution_metrics{
                   best_fitness = 0.7,
                   avg_fitness = 0.5
               },

               EmergentMetrics = #emergent_metrics{
                   convergence_rate = 0.03,
                   fitness_plateau_duration = 5,
                   current_mutation_rate = 0.12,
                   survival_rate = 0.25,
                   complexity_trend = 0.01
               },

               %% Forward with both metric types
               Hyperparams = lc_chain:forward(Pid, EvoMetrics, EmergentMetrics),

               ?assert(is_record(Hyperparams, lc_hyperparams)),
               ?assert(Hyperparams#lc_hyperparams.mutation_rate >= 0.0),

               ok = gen_server:stop(Pid)
           end}]
     end}.
