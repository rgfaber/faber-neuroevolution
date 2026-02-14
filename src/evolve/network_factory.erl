%% @doc Network factory wrapper for faber_tweann's network_evaluator.
%%
%% This module implements the network factory interface expected by evolution
%% strategies, delegating to the real network_evaluator from faber_tweann.
%%
%% The factory interface provides:
%% - create_feedforward/1 - Create a new feedforward network
%% - mutate/2 - Mutate a network's weights
%% - crossover/2 - Create offspring from two parent networks
%%
%% This abstraction enables:
%% - Dependency injection for testing (use mock_network_factory in tests)
%% - Clean separation between evolution logic and network implementation
%% - Future support for different network types (RNNs, LSTMs, etc.)
%%
%% @author R.G. Lefever
%% @copyright 2024-2026 R.G. Lefever
-module(network_factory).

-export([
    create_feedforward/1,
    create_compiled_feedforward/1,
    compile/1,
    mutate/2,
    crossover/2
]).

%% @doc Create a new feedforward neural network.
%%
%% Delegates to network_evaluator:create_feedforward/3 from faber_tweann.
%%
%% @param Topology Network topology as {InputSize, HiddenLayers, OutputSize}
%%        where HiddenLayers is a list of layer sizes
%% @returns A new feedforward network
-spec create_feedforward(Topology) -> network_evaluator:network() when
    Topology :: {pos_integer(), [pos_integer()], pos_integer()}.
create_feedforward({InputSize, HiddenLayers, OutputSize}) ->
    network_evaluator:create_feedforward(InputSize, HiddenLayers, OutputSize).

%% @doc Create a NIF-compiled feedforward network for fast evaluation.
%%
%% Combines network creation and NIF compilation in one step.
%% Uses NIF acceleration when available (50-100x faster evaluation).
%%
%% @param Topology Network topology as {InputSize, HiddenLayers, OutputSize}
%% @returns {ok, CompiledNetwork} | {error, Reason}
-spec create_compiled_feedforward(Topology) -> Result when
    Topology :: {pos_integer(), [pos_integer()], pos_integer()},
    Result :: {ok, nif_network:compiled_network()} | {error, term()}.
create_compiled_feedforward({InputSize, HiddenLayers, OutputSize}) ->
    nif_network:compile_feedforward(InputSize, HiddenLayers, OutputSize).

%% @doc Compile an existing network for NIF-accelerated evaluation.
%%
%% Takes a network from create_feedforward/1 and compiles it for
%% fast repeated evaluation via NIF.
%%
%% @param Network Network from create_feedforward/1
%% @returns {ok, CompiledNetwork} | {error, Reason}
-spec compile(Network) -> Result when
    Network :: network_evaluator:network(),
    Result :: {ok, nif_network:compiled_network()} | {error, term()}.
compile(Network) ->
    nif_network:compile(Network).

%% @doc Mutate a network's weights.
%%
%% Creates a copy of the network with mutated weights. The mutation applies
%% gaussian noise to each weight with the given strength.
%%
%% @param Network The network to mutate
%% @param MutationStrength Standard deviation of gaussian noise to add
%% @returns A new network with mutated weights
-spec mutate(Network, MutationStrength) -> network_evaluator:network() when
    Network :: network_evaluator:network(),
    MutationStrength :: float().
mutate(Network, MutationStrength) ->
    %% Get current weights
    Weights = network_evaluator:get_weights(Network),

    %% Apply gaussian mutation to each weight
    MutatedWeights = [W + (rand:normal() * MutationStrength) || W <- Weights],

    %% Create new network with mutated weights
    network_evaluator:set_weights(Network, MutatedWeights).

%% @doc Crossover two networks to produce offspring.
%%
%% Performs uniform crossover: each weight in the offspring is randomly
%% selected from either parent with equal probability.
%%
%% @param Parent1 First parent network
%% @param Parent2 Second parent network
%% @returns A new network combining weights from both parents
-spec crossover(Parent1, Parent2) -> network_evaluator:network() when
    Parent1 :: network_evaluator:network(),
    Parent2 :: network_evaluator:network().
crossover(Parent1, Parent2) ->
    %% Get weights from both parents
    Weights1 = network_evaluator:get_weights(Parent1),
    Weights2 = network_evaluator:get_weights(Parent2),

    %% Uniform crossover: randomly select each weight from either parent
    ChildWeights = lists:zipwith(
        fun(W1, W2) ->
            case rand:uniform() < 0.5 of
                true -> W1;
                false -> W2
            end
        end,
        Weights1,
        Weights2
    ),

    %% Create child network with crossed weights
    network_evaluator:set_weights(Parent1, ChildWeights).
