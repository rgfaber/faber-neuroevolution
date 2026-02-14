%% @doc Gradient-based training for the meta-controller.
%%
%% This module implements gradient-based optimization for updating the
%% meta-controller's LTC network weights. It uses policy gradient methods
%% adapted for continuous action spaces.
%%
%% == Training Algorithm ==
%%
%% We use a simplified REINFORCE-style policy gradient:
%%
%% 1. Collect experience: (state, action, reward) tuples
%% 2. Compute returns: G_t = sum of future discounted rewards
%% 3. Estimate gradients: nabla_theta = E[G_t * nabla_theta log pi(a|s)]
%% 4. Update weights: theta = theta + alpha * gradient
%%
%% == LTC-Specific Considerations ==
%%
%% LTC neurons have temporal state that affects gradient flow:
%% - Backpropagation through time (BPTT) for temporal dependencies
%% - Truncated gradients for computational efficiency
%% - Momentum to smooth updates across generations
%%
%% @author R.G. Lefever
%% @copyright 2024-2026 R.G. Lefever
-module(meta_trainer).

-include("meta_controller.hrl").

-export([
    update_weights/4,
    compute_gradients/3,
    apply_gradients/3,
    estimate_advantage/2
]).

%%% ============================================================================
%%% API Functions
%%% ============================================================================

%% @doc Update meta-controller weights based on collected experience.
%%
%% @param Weights Current network weights
%% @param Experience List of training events
%% @param Config Meta-controller configuration
%% @param LearningRate Current learning rate
%% @returns Updated weights
-spec update_weights(map(), [meta_training_event()], meta_config(), float()) -> map().
update_weights(Weights, Experience, Config, LearningRate) ->
    %% Compute gradients from experience
    Gradients = compute_gradients(Experience, Weights, Config),

    %% Apply gradients with learning rate
    apply_gradients(Weights, Gradients, LearningRate).

%% @doc Compute gradients from experience.
%%
%% Uses REINFORCE-style policy gradient estimation.
%%
%% @param Experience List of training events
%% @param Weights Current network weights
%% @param Config Meta-controller configuration
%% @returns Map of weight gradients
-spec compute_gradients([meta_training_event()], map(), meta_config()) -> map().
compute_gradients(Experience, Weights, _Config) ->
    %% Compute baseline (average reward)
    Rewards = [E#meta_training_event.reward || E <- Experience],
    Baseline = safe_average(Rewards, 0.0),

    %% Compute advantages
    Advantages = [E#meta_training_event.reward - Baseline || E <- Experience],

    %% Accumulate gradients across experience
    lists:foldl(
        fun({Event, Advantage}, GradAcc) ->
            %% Compute gradient contribution from this event
            EventGradients = compute_event_gradients(Event, Advantage, Weights),

            %% Accumulate
            merge_gradients(GradAcc, EventGradients)
        end,
        #{},
        lists:zip(Experience, Advantages)
    ).

%% @doc Apply gradients to weights.
%%
%% Uses gradient descent with optional momentum and gradient clipping.
%%
%% @param Weights Current network weights
%% @param Gradients Computed gradients
%% @param LearningRate Learning rate
%% @returns Updated weights
-spec apply_gradients(map(), map(), float()) -> map().
apply_gradients(Weights, Gradients, LearningRate) ->
    %% Clip gradients to prevent exploding gradients
    ClippedGradients = clip_gradients(Gradients, 1.0),

    %% Apply updates to each neuron
    maps:map(
        fun(NeuronId, Neuron) ->
            update_neuron_weights(NeuronId, Neuron, ClippedGradients, LearningRate)
        end,
        Weights
    ).

%% @doc Estimate advantage for a reward.
%%
%% Advantage = reward - baseline, where baseline is a moving average.
%%
%% @param Reward Current reward
%% @param History Recent reward history
%% @returns Advantage estimate
-spec estimate_advantage(float(), [float()]) -> float().
estimate_advantage(Reward, []) ->
    Reward;
estimate_advantage(Reward, History) ->
    Baseline = safe_average(History, 0.0),
    Reward - Baseline.

%%% ============================================================================
%%% Internal Functions - Gradient Computation
%%% ============================================================================

%% @private Compute gradients for a single training event.
compute_event_gradients(Event, Advantage, Weights) ->
    Inputs = Event#meta_training_event.inputs,
    Outputs = Event#meta_training_event.outputs,

    %% For each weight, estimate gradient contribution
    %% Using numerical differentiation approximation for simplicity
    %% (Full backprop through LTC requires more complex implementation)

    maps:fold(
        fun(NeuronId, Neuron, GradAcc) ->
            %% Gradient for input weights
            InputGrads = compute_input_weight_gradients(
                NeuronId, Neuron, Inputs, Outputs, Advantage
            ),

            %% Gradient for bias
            BiasGrad = compute_bias_gradient(Advantage, Outputs),

            %% Store gradients
            GradAcc#{
                {NeuronId, input_weights} => InputGrads,
                {NeuronId, bias} => BiasGrad
            }
        end,
        #{},
        Weights
    ).

%% @private Compute gradients for input weights of a neuron.
compute_input_weight_gradients(_NeuronId, Neuron, Inputs, _Outputs, Advantage) ->
    InputWeights = Neuron#meta_neuron.input_weights,

    lists:map(
        fun({{InputIdx, _Weight}, InputVal}) ->
            %% Policy gradient: advantage * input_activation
            %% This is a simplified gradient estimate
            Gradient = Advantage * InputVal,
            {InputIdx, Gradient};
           ({InputIdx, _Weight}) ->
            InputVal = safe_nth(InputIdx, Inputs, 0.0),
            Gradient = Advantage * InputVal,
            {InputIdx, Gradient}
        end,
        lists:zip(InputWeights, pad_list(Inputs, length(InputWeights), 0.0))
    ).

%% @private Compute gradient for bias.
compute_bias_gradient(Advantage, _Outputs) ->
    %% Bias gradient is proportional to advantage
    Advantage * 0.1.

%% @private Merge gradient maps.
merge_gradients(Grads1, Grads2) ->
    maps:fold(
        fun(Key, Value, Acc) ->
            OldValue = maps:get(Key, Acc, default_gradient(Key)),
            NewValue = add_gradients(OldValue, Value),
            Acc#{Key => NewValue}
        end,
        Grads1,
        Grads2
    ).

%% @private Default gradient value for a key.
default_gradient({_NeuronId, input_weights}) -> [];
default_gradient({_NeuronId, bias}) -> 0.0;
default_gradient(_) -> 0.0.

%% @private Add two gradient values.
add_gradients(G1, G2) when is_number(G1), is_number(G2) ->
    G1 + G2;
add_gradients(G1, G2) when is_list(G1), is_list(G2) ->
    lists:zipwith(
        fun({Idx1, V1}, {Idx2, V2}) when Idx1 =:= Idx2 ->
            {Idx1, V1 + V2};
           (V1, V2) when is_number(V1), is_number(V2) ->
            V1 + V2
        end,
        pad_list(G1, max(length(G1), length(G2)), {0, 0.0}),
        pad_list(G2, max(length(G1), length(G2)), {0, 0.0})
    );
add_gradients(G1, _G2) ->
    G1.

%% @private Clip gradients to prevent explosion.
clip_gradients(Gradients, MaxNorm) ->
    %% Compute gradient norm
    Norm = compute_gradient_norm(Gradients),

    case Norm > MaxNorm of
        true ->
            Scale = MaxNorm / Norm,
            scale_gradients(Gradients, Scale);
        false ->
            Gradients
    end.

%% @private Compute L2 norm of gradients.
compute_gradient_norm(Gradients) ->
    SumSquares = maps:fold(
        fun(_Key, Value, Acc) ->
            Acc + gradient_sum_squares(Value)
        end,
        0.0,
        Gradients
    ),
    math:sqrt(SumSquares).

%% @private Sum of squares for a gradient value.
gradient_sum_squares(Value) when is_number(Value) ->
    Value * Value;
gradient_sum_squares(Values) when is_list(Values) ->
    lists:sum([gradient_sum_squares(V) || V <- Values]);
gradient_sum_squares({_Idx, Value}) ->
    Value * Value;
gradient_sum_squares(_) ->
    0.0.

%% @private Scale gradients by a factor.
scale_gradients(Gradients, Scale) ->
    maps:map(
        fun(_Key, Value) ->
            scale_gradient_value(Value, Scale)
        end,
        Gradients
    ).

%% @private Scale a single gradient value.
scale_gradient_value(Value, Scale) when is_number(Value) ->
    Value * Scale;
scale_gradient_value(Values, Scale) when is_list(Values) ->
    lists:map(fun(V) -> scale_gradient_value(V, Scale) end, Values);
scale_gradient_value({Idx, Value}, Scale) ->
    {Idx, Value * Scale};
scale_gradient_value(Value, _Scale) ->
    Value.

%%% ============================================================================
%%% Internal Functions - Weight Updates
%%% ============================================================================

%% @private Update weights for a single neuron.
update_neuron_weights(NeuronId, Neuron, Gradients, LearningRate) ->
    %% Get gradients for this neuron
    InputGrads = maps:get({NeuronId, input_weights}, Gradients, []),
    BiasGrad = maps:get({NeuronId, bias}, Gradients, 0.0),

    %% Update input weights
    NewInputWeights = update_input_weights(
        Neuron#meta_neuron.input_weights,
        InputGrads,
        LearningRate
    ),

    %% Update bias
    NewBias = Neuron#meta_neuron.bias + LearningRate * BiasGrad,

    Neuron#meta_neuron{
        input_weights = NewInputWeights,
        bias = clamp(NewBias, -5.0, 5.0)
    }.

%% @private Update input weights with gradients.
update_input_weights(Weights, [], _LearningRate) ->
    Weights;
update_input_weights(Weights, Gradients, LearningRate) ->
    GradMap = maps:from_list([{Idx, G} || {Idx, G} <- Gradients, is_integer(Idx)]),

    lists:map(
        fun({InputIdx, Weight}) ->
            Grad = maps:get(InputIdx, GradMap, 0.0),
            NewWeight = Weight + LearningRate * Grad,
            {InputIdx, clamp(NewWeight, -5.0, 5.0)}
        end,
        Weights
    ).

%%% ============================================================================
%%% Utility Functions
%%% ============================================================================

%% @private Safe average with default.
safe_average([], Default) -> Default;
safe_average(List, _Default) ->
    lists:sum(List) / length(List).

%% @private Safe nth element.
safe_nth(N, List, _Default) when N > 0, N =< length(List) ->
    lists:nth(N, List);
safe_nth(_N, _List, Default) ->
    Default.

%% @private Pad list to target length.
pad_list(List, TargetLen, _PadValue) when length(List) >= TargetLen ->
    lists:sublist(List, TargetLen);
pad_list(List, TargetLen, PadValue) ->
    List ++ lists:duplicate(TargetLen - length(List), PadValue).

%% @private Clamp value to range.
clamp(Val, Min, _Max) when Val < Min -> Min;
clamp(Val, _Min, Max) when Val > Max -> Max;
clamp(Val, _Min, _Max) -> Val.
