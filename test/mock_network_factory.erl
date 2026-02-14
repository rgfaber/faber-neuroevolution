%% @doc Mock network factory for testing evolution strategies.
%%
%% Provides a simple implementation of the network factory interface
%% that creates lightweight mock networks without requiring the full
%% faber_tweann dependency.
%%
%% @author R.G. Lefever
%% @copyright 2024-2026 R.G. Lefever
-module(mock_network_factory).

-export([
    create_feedforward/1,
    mutate/2,
    crossover/2
]).

%% @doc Create a mock feedforward network.
%%
%% Returns a simple map structure that mimics a network.
-spec create_feedforward(Topology :: term()) -> map().
create_feedforward(Topology) ->
    #{
        type => feedforward,
        topology => Topology,
        weights => random_weights(Topology),
        created_at => erlang:timestamp()
    }.

%% @doc Mutate a mock network.
-spec mutate(Network :: map(), MutationStrength :: float()) -> map().
mutate(Network, _MutationStrength) ->
    Network#{
        weights => random_weights(maps:get(topology, Network)),
        mutated => true
    }.

%% @doc Crossover two mock networks.
-spec crossover(Network1 :: map(), Network2 :: map()) -> map().
crossover(Network1, _Network2) ->
    Network1#{
        weights => random_weights(maps:get(topology, Network1)),
        crossed => true
    }.

%%% ============================================================================
%%% Internal Functions
%%% ============================================================================

%% @private Generate random weights based on topology.
random_weights({Inputs, Hidden, Outputs}) when is_list(Hidden) ->
    %% Calculate total number of connections
    HiddenSum = lists:sum(Hidden),
    TotalWeights = (Inputs * hd(Hidden)) + (lists:last(Hidden) * Outputs) + HiddenSum,
    [rand:uniform() || _ <- lists:seq(1, TotalWeights)];
random_weights({Inputs, Hidden, Outputs}) when is_integer(Hidden) ->
    random_weights({Inputs, [Hidden], Outputs});
random_weights(_) ->
    [rand:uniform() || _ <- lists:seq(1, 10)].
