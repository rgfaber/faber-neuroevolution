%% @doc XOR fitness: inverse root-mean-square error.
%%
%% Follows Sher's xor_sim (Handbook Ch 7), which uses 1/(error + epsilon) so
%% that a perfect network yields a very large finite fitness rather than a
%% division by zero. Using RMSE rather than raw SSE keeps the scale
%% independent of the number of cases.
%%
%% Fitness is a search signal, not a success criterion. Whether a network has
%% actually solved XOR is answered by the `solved' flag in the metrics, which
%% requires all four cases on the correct side of zero by a clear margin.
%%
%% @copyright 2024-2026 R.G. Lefever
%% @license Apache-2.0
-module(xor_evaluator).
-behaviour(agent_evaluator).

-export([name/0, calculate_fitness/1, fitness_components/1, is_solved/1]).

-define(EPSILON, 1.0e-5).

name() -> <<"xor_inverse_rmse">>.

calculate_fitness(#{sse := Sse, cases := Cases}) when Cases > 0 ->
    Rmse = math:sqrt(Sse / Cases),
    1.0 / (Rmse + ?EPSILON).

fitness_components(#{sse := Sse, cases := Cases, correct := Correct}) ->
    #{sse => Sse, rmse => math:sqrt(Sse / Cases), correct => Correct}.

%% @doc Solved means every case on the correct side of zero by a clear margin.
%%
%% Deliberately not a fitness threshold: fitness is continuous and unbounded,
%% so any threshold would be arbitrary and would drift with the epsilon.
is_solved(#{correct := Correct, cases := Cases}) ->
    Correct =:= Cases;
is_solved(_) ->
    false.
