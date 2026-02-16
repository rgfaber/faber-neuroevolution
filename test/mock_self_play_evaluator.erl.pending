%% @doc Mock evaluator for self-play integration testing.
%%
%% Simulates network-vs-network evaluation by:
%% 1. Getting opponent from self_play_manager (or batch fallback)
%% 2. Computing a simple "game result" based on network weights
%% 3. Returning fitness score
-module(mock_self_play_evaluator).
-behaviour(neuroevolution_evaluator).

-include("neuroevolution.hrl").

-export([evaluate/2, calculate_fitness/1]).

%% @doc Evaluate individual against opponent from self-play manager.
evaluate(Individual, Options) ->
    Network = Individual#individual.network,

    %% Get opponent network
    OpponentNetwork = get_opponent_network(Options),

    %% Simulate a "game" between networks
    Score = simulate_game(Network, OpponentNetwork),

    UpdatedIndividual = Individual#individual{
        metrics = #{
            total_score => Score,
            opponent_played => true,
            used_self_play => maps:is_key(self_play_manager, Options)
        }
    },
    {ok, UpdatedIndividual}.

%% @doc Extract fitness from metrics.
calculate_fitness(Metrics) ->
    maps:get(total_score, Metrics, 0.0).

%%% Internal functions

get_opponent_network(Options) ->
    case maps:get(self_play_manager, Options, undefined) of
        undefined ->
            %% No self-play - return a simple opponent
            #{weights => [0.5, 0.5, 0.5]};
        ManagerPid ->
            BatchNetworks = maps:get(batch_networks, Options, []),
            case self_play_manager:get_opponent(ManagerPid, BatchNetworks) of
                {ok, Network} -> Network;
                _ -> #{weights => [0.5, 0.5, 0.5]}
            end
    end.

%% @doc Simulate a game between two networks.
%% Simple heuristic: sum of weights determines "strength"
%% Winner is determined by comparing strengths with some randomness
simulate_game(Network1, Network2) ->
    Strength1 = network_strength(Network1),
    Strength2 = network_strength(Network2),

    %% Add randomness to make it interesting
    Noise = (rand:uniform() - 0.5) * 0.2,

    %% Compute relative performance
    TotalStrength = max(Strength1 + Strength2, 0.001),
    RelativePerformance = (Strength1 + Noise) / TotalStrength,

    %% Score based on performance (0-100 scale)
    BaseScore = RelativePerformance * 100,

    %% Clamp to 0-100
    max(0.0, min(100.0, BaseScore)).

network_strength(#{weights := Weights}) when is_list(Weights) ->
    lists:sum([abs(W) || W <- Weights]);
network_strength(#{topology := _}) ->
    %% TWEANN network - use random strength for now
    rand:uniform() * 10;
network_strength(_) ->
    %% Unknown format - default strength
    5.0.
