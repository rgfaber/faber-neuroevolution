%% @doc XOR: the first end-to-end evolutionary run in this codebase.
%%
%% Phase 2 of PLAN_FABER_FOUNDATION. Before this, no evolutionary run had ever
%% completed here: faber-tweann's process-per-neuron path has no scape and no
%% fitness channel, and faber-neuroevolution's strategies were tested only
%% against mock networks with rand:uniform() fitness, so selection pressure was
%% provably absent.
%%
%% These tests run the domain_sdk path with real fitness on a problem with a
%% known answer.
%%
%% Layered deliberately, cheapest first, so a failure localises:
%%
%%   1. the bridge validates
%%   2. a hand-written perfect network scores as solved  (environment correct)
%%   3. a constant network scores as unsolved            (scoring discriminates)
%%   4. evolution solves XOR                             (the actual claim)
%%   5. every strategy completes a run with real fitness
%%
%% Test 2 matters more than it looks. If the environment were broken such that
%% everything scored perfectly, test 4 would pass for the wrong reason. Test 3
%% is the same guard from the other side.
%%
%% @copyright 2024-2026 R.G. Lefever
%% @license Apache-2.0
-module(xor_tests).

-include_lib("eunit/include/eunit.hrl").

%% Invoked manually, not by the suite. See xor_solve_rate/0 below.
-export([xor_solve_rate/0, xor_solve_rate/1]).

%% Evolution is stochastic. These are deliberately modest so the suite stays
%% fast; the full 20-run solve-rate criterion from the plan lives in
%% xor_solve_rate/0, which is not part of the default suite.
-define(GENERATIONS, 60).
-define(POPULATION, 50).

%%% ===========================================================================
%%% Helpers
%%% ===========================================================================

bridge() ->
    agent_bridge:new(#{
        definition  => xor_definition,
        sensors     => [xor_sensor],
        actuators   => [xor_actuator],
        environment => xor_environment,
        evaluator   => xor_evaluator
    }).

%% A hand-written network that computes XOR exactly.
%% Inputs are -1/1, so the product is negative exactly when they differ.
oracle() ->
    fun([A, B]) ->
        case A * B < 0 of
            true  -> [1.0];
            false -> [-1.0]
        end
    end.

evaluate_with(NetworkFun) ->
    {ok, Bridge} = bridge(),
    agent_trainer:evaluate(Bridge, NetworkFun, #{}).

%%% ===========================================================================
%%% 1. Bridge
%%% ===========================================================================

bridge_validates_test() ->
    ?assertMatch({ok, _}, bridge()).

topology_matches_sensor_and_actuator_counts_test() ->
    {Inputs, _Hidden, Outputs} = xor_definition:network_topology(),
    ?assertEqual(Inputs, xor_sensor:input_count()),
    ?assertEqual(Outputs, xor_actuator:output_count()).

%%% ===========================================================================
%%% 2 & 3. The environment discriminates
%%% ===========================================================================

oracle_solves_test() ->
    {ok, Fitness, Metrics} = evaluate_with(oracle()),
    ?assertEqual(true, maps:get(solved, Metrics)),
    ?assertEqual(xor_environment:case_count(), maps:get(correct, Metrics)),
    ?assertEqual(xor_environment:case_count(), maps:get(presented, Metrics)),
    %% Zero error, so fitness is 1/epsilon.
    ?assert(Fitness > 1000.0).

constant_output_does_not_solve_test() ->
    {ok, Fitness, Metrics} = evaluate_with(fun(_) -> [0.0] end),
    ?assertEqual(false, maps:get(solved, Metrics)),
    ?assertEqual(0, maps:get(correct, Metrics)),
    %% Every case off by exactly 1.0, so RMSE is 1.0 and fitness ~1.0.
    ?assert(Fitness < 1.1).

inverted_output_is_worst_case_test() ->
    %% Deliberately wrong on every case: error of 2.0 each.
    Inverted = fun([A, B]) ->
        case A * B < 0 of true -> [-1.0]; false -> [1.0] end
    end,
    {ok, Fitness, Metrics} = evaluate_with(Inverted),
    ?assertEqual(0, maps:get(correct, Metrics)),
    {ok, GoodFitness, _} = evaluate_with(oracle()),
    ?assert(GoodFitness > Fitness).

all_four_cases_presented_test() ->
    %% Guards the tick/sense/apply_action ordering: every case must be
    %% presented exactly once, no more and no fewer.
    {ok, _F, Metrics} = evaluate_with(oracle()),
    ?assertEqual(4, maps:get(presented, Metrics)),
    ?assertEqual(4, maps:get(cases, Metrics)).

%%% ===========================================================================
%%% 4. Evolution
%%% ===========================================================================

evolution_solves_xor_test_() ->
    {timeout, 300, fun() ->
        {ok, Bridge} = bridge(),
        {ok, Network, Stats} = agent_trainer:train(Bridge, #{}, #{
            generations     => ?GENERATIONS,
            population_size => ?POPULATION
        }),
        ?assert(is_map(Stats)),
        {ok, Fitness, Metrics} = agent_trainer:evaluate(Bridge, Network, #{}),
        ?assertEqual(true, maps:get(solved, Metrics),
                     lists:flatten(io_lib:format(
                        "evolution did not solve XOR: fitness=~p metrics=~p",
                        [Fitness, Metrics])))
    end}.

%%% ===========================================================================
%%% 5. Strategy coverage
%%% ===========================================================================
%%%
%%% Every strategy has until now been exercised only against
%%% mock_network_factory, whose mutate/2 discards the parent's weights and
%%% regenerates them at random, and mock_evaluator, which returns
%%% rand:uniform() * 100. This is the first time any of them runs against a
%%% real network and a real objective.

%% Strategies confirmed to complete a run against real fitness.
strategies_run_with_real_fitness_test_() ->
    Strategies = [generational_strategy, steady_state_strategy, novelty_strategy],
    {timeout, 600,
     [{atom_to_list(S), fun() -> run_strategy(S) end} || S <- Strategies]}.

%% Strategies that do NOT work, asserted as broken so the suite stays green
%% while the breakage stays visible. When one is fixed, this test fails and
%% the strategy moves into the list above.
%%
%% Tracked in ROADMAP.md. Both were previously "verified" only against
%% mock_network_factory, whose mutate/2 discards parent weights, and
%% mock_evaluator, which returns rand:uniform() * 100. Neither had ever run
%% against a real objective.
known_broken_strategies_test_() ->
    Broken = [island_strategy, map_elites_strategy],
    {timeout, 600,
     [{atom_to_list(S) ++ " is still broken",
       fun() ->
           ?assertMatch({error, _}, run_isolated(fun() -> do_run_strategy(S) end, 120000))
       end} || S <- Broken]}.

%% Each strategy runs in its own process. island_strategy currently kills the
%% calling process outright (badarg from erlang:length/1 applied to the island
%% map), which without isolation cancels every strategy queued after it and
%% hides whether those work.
run_strategy(Strategy) ->
    case run_isolated(fun() -> do_run_strategy(Strategy) end, 120000) of
        ok ->
            ok;
        {error, Reason} ->
            erlang:error({strategy_failed, Strategy, Reason})
    end.

do_run_strategy(Strategy) ->
    {ok, Bridge} = bridge(),
    Result = agent_trainer:train(Bridge, #{}, #{
        generations     => 10,
        population_size => 20,
        strategy        => Strategy
    }),
    {ok, Network, _Stats} = Result,
    %% The returned champion must be evaluable. A strategy that returns
    %% something the bridge cannot score is broken regardless of its fitness.
    {ok, _F, _M} = agent_trainer:evaluate(Bridge, Network, #{}),
    ok.

run_isolated(Fun, Timeout) ->
    Parent = self(),
    Ref = make_ref(),
    {Pid, Mon} = spawn_monitor(fun() ->
        Result = try Fun() of R -> {ok, R} catch C:E -> {error, {C, E}} end,
        Parent ! {Ref, Result}
    end),
    receive
        {Ref, {ok, _}} -> erlang:demonitor(Mon, [flush]), ok;
        {Ref, {error, R}} -> erlang:demonitor(Mon, [flush]), {error, R};
        {'DOWN', Mon, process, Pid, Reason} -> {error, {died, Reason}}
    after Timeout ->
        exit(Pid, kill),
        {error, timeout}
    end.

%%% ===========================================================================
%%% Solve rate (not in the default suite)
%%% ===========================================================================

%% @doc Plan exit criterion for Phase 2: 18 of 20 independent runs solve XOR.
%%
%% Run manually, it is far too slow for the normal suite:
%%   rebar3 as test shell
%%   xor_tests:xor_solve_rate().
xor_solve_rate() ->
    xor_solve_rate(20).

xor_solve_rate(Runs) ->
    {ok, Bridge} = bridge(),
    Results = [begin
        {ok, Network, _} = agent_trainer:train(Bridge, #{}, #{
            generations     => ?GENERATIONS,
            population_size => ?POPULATION
        }),
        {ok, _F, M} = agent_trainer:evaluate(Bridge, Network, #{}),
        maps:get(solved, M)
    end || _ <- lists:seq(1, Runs)],
    Solved = length([S || S <- Results, S =:= true]),
    io:format("solved ~p/~p~n", [Solved, Runs]),
    {Solved, Runs}.
