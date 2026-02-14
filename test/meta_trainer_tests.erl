%% @doc Unit tests for meta_trainer module.
%%
%% Note: The compute_gradients and update_weights functions have known issues
%% with gradient accumulation. This test file focuses on the simpler functions.
-module(meta_trainer_tests).

-include_lib("eunit/include/eunit.hrl").
-include("meta_controller.hrl").

%%% ============================================================================
%%% Advantage Estimation Tests
%%% ============================================================================

estimate_advantage_empty_history_test() ->
    Reward = 0.5,
    Advantage = meta_trainer:estimate_advantage(Reward, []),
    ?assertEqual(0.5, Advantage).

estimate_advantage_with_history_test() ->
    Reward = 0.8,
    History = [0.4, 0.5, 0.6],
    Advantage = meta_trainer:estimate_advantage(Reward, History),
    %% Baseline = 0.5, Advantage = 0.8 - 0.5 = 0.3
    ?assert(abs(Advantage - 0.3) < 0.0001).

estimate_advantage_below_baseline_test() ->
    Reward = 0.2,
    History = [0.5, 0.5, 0.5],
    Advantage = meta_trainer:estimate_advantage(Reward, History),
    %% Baseline = 0.5, Advantage = 0.2 - 0.5 = -0.3
    ?assert(abs(Advantage - (-0.3)) < 0.0001).

estimate_advantage_at_baseline_test() ->
    Reward = 0.5,
    History = [0.4, 0.5, 0.6],
    Advantage = meta_trainer:estimate_advantage(Reward, History),
    %% Baseline = 0.5, Advantage = 0.5 - 0.5 = 0.0
    ?assertEqual(0.0, Advantage).

estimate_advantage_long_history_test() ->
    Reward = 1.0,
    History = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    %% Baseline = 0.55
    Advantage = meta_trainer:estimate_advantage(Reward, History),
    ?assert(abs(Advantage - 0.45) < 0.0001).

estimate_advantage_zero_reward_test() ->
    Reward = 0.0,
    History = [0.5, 0.5, 0.5],
    Advantage = meta_trainer:estimate_advantage(Reward, History),
    ?assert(abs(Advantage - (-0.5)) < 0.0001).

estimate_advantage_negative_reward_test() ->
    Reward = -1.0,
    History = [0.0, 0.0, 0.0],
    Advantage = meta_trainer:estimate_advantage(Reward, History),
    ?assertEqual(-1.0, Advantage).

estimate_advantage_high_reward_test() ->
    Reward = 100.0,
    History = [10.0, 20.0, 30.0],
    %% Baseline = 20.0
    Advantage = meta_trainer:estimate_advantage(Reward, History),
    ?assertEqual(80.0, Advantage).

%%% ============================================================================
%%% Gradient Computation Tests (Basic)
%%% ============================================================================

compute_gradients_empty_experience_test() ->
    %% Empty experience should return empty gradients map
    Weights = #{},
    Config = #meta_config{},
    Gradients = meta_trainer:compute_gradients([], Weights, Config),
    ?assert(is_map(Gradients)),
    ?assertEqual(0, maps:size(Gradients)).

%%% ============================================================================
%%% Apply Gradients Tests (Basic)
%%% ============================================================================

apply_gradients_empty_weights_test() ->
    Weights = #{},
    Gradients = #{},
    LearningRate = 0.01,

    UpdatedWeights = meta_trainer:apply_gradients(Weights, Gradients, LearningRate),
    ?assert(is_map(UpdatedWeights)),
    ?assertEqual(0, maps:size(UpdatedWeights)).

apply_gradients_empty_gradients_test() ->
    Neuron = #meta_neuron{
        id = {1, 1},
        internal_state = 0.0,
        time_constant = 10.0,
        state_bound = 1.0,
        input_weights = [{1, 0.5}],
        bias = 0.1
    },
    Weights = #{{1, 1} => Neuron},
    Gradients = #{},
    LearningRate = 0.01,

    UpdatedWeights = meta_trainer:apply_gradients(Weights, Gradients, LearningRate),
    ?assert(is_map(UpdatedWeights)),
    ?assertEqual(1, maps:size(UpdatedWeights)).
