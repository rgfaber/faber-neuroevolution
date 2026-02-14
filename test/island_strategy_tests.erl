%% @doc EUnit tests for island_strategy module.
%%
%% Tests the island model evolution strategy implementation including:
%% - Initialization with multiple islands
%% - Evaluation routing to correct island
%% - Migration topology building
%% - Population snapshots across islands
%% - Meta-controller interface
%%
%% @author R.G. Lefever
%% @copyright 2024-2026 R.G. Lefever
-module(island_strategy_tests).

-include_lib("eunit/include/eunit.hrl").
-include("neuroevolution.hrl").
-include("evolution_strategy.hrl").
-include("lifecycle_events.hrl").

%%% ============================================================================
%%% Test Fixtures
%%% ============================================================================

test_neuro_config() ->
    #neuro_config{
        population_size = 20,  % Will be overridden per island
        evaluations_per_individual = 1,
        selection_ratio = 0.4,
        mutation_rate = 0.1,
        mutation_strength = 0.2,
        max_generations = 100,
        network_topology = {4, [8], 2},
        evaluator_module = mock_evaluator
    }.

test_init_config() ->
    #{
        neuro_config => test_neuro_config(),
        strategy_params => #{
            island_count => 3,
            population_per_island => 5,
            migration_interval => 10,
            migration_count => 1,
            migration_selection => best,
            topology => ring,
            island_strategy => generational_strategy,
            island_strategy_params => #{
                selection_ratio => 0.4,
                mutation_rate => 0.1
            }
        },
        network_factory => mock_network_factory
    }.

%%% ============================================================================
%%% Initialization Tests
%%% ============================================================================

init_creates_multiple_islands_test() ->
    Config = test_init_config(),
    {ok, State, _Events} = island_strategy:init(Config),

    Snapshot = island_strategy:get_population_snapshot(State),

    %% Total population = 3 islands * 5 individuals = 15
    ?assertEqual(15, maps:get(size, Snapshot)),

    %% Should have 3 islands
    Extra = maps:get(extra, Snapshot),
    ?assertEqual(3, maps:get(island_count, Extra)).

init_returns_birth_events_for_all_islands_test() ->
    Config = test_init_config(),
    {ok, _State, Events} = island_strategy:init(Config),

    %% Should have birth events for all individuals (3 islands * 5 = 15)
    BirthEvents = [E || E <- Events, is_record(E, individual_born)],
    ?assertEqual(15, length(BirthEvents)),

    %% All should be initial births
    lists:foreach(fun(E) ->
        ?assertEqual(initial, E#individual_born.origin)
    end, BirthEvents).

init_tags_events_with_island_id_test() ->
    Config = test_init_config(),
    {ok, _State, Events} = island_strategy:init(Config),

    BirthEvents = [E || E <- Events, is_record(E, individual_born)],

    %% All events should have island_id in metadata
    lists:foreach(fun(E) ->
        Metadata = E#individual_born.metadata,
        ?assert(maps:is_key(island_id, Metadata))
    end, BirthEvents).

%%% ============================================================================
%%% Evaluation Routing Tests
%%% ============================================================================

eval_routes_to_correct_island_test() ->
    Config = test_init_config(),
    {ok, State, Events} = island_strategy:init(Config),

    %% Get first individual from events
    [FirstBirth | _] = [E || E <- Events, is_record(E, individual_born)],
    IndId = FirstBirth#individual_born.id,

    %% Evaluate it
    FitnessResult = #{fitness => 100.0, metrics => #{}},
    {_Actions, EvalEvents, _NewState} = island_strategy:handle_evaluation_result(IndId, FitnessResult, State),

    %% Should emit evaluation event
    EvaluatedEvents = [E || E <- EvalEvents, is_record(E, individual_evaluated)],
    ?assertEqual(1, length(EvaluatedEvents)),

    %% Event should have island_id in metadata
    [EvalEvent] = EvaluatedEvents,
    ?assert(maps:is_key(island_id, EvalEvent#individual_evaluated.metadata)).

%%% ============================================================================
%%% Snapshot Tests
%%% ============================================================================

snapshot_aggregates_all_islands_test() ->
    Config = test_init_config(),
    {ok, State, _Events} = island_strategy:init(Config),

    Snapshot = island_strategy:get_population_snapshot(State),

    ?assert(maps:is_key(size, Snapshot)),
    ?assert(maps:is_key(individuals, Snapshot)),
    ?assert(maps:is_key(best_fitness, Snapshot)),
    ?assert(maps:is_key(avg_fitness, Snapshot)),
    ?assert(maps:is_key(extra, Snapshot)),

    %% Individuals should have island_id
    Individuals = maps:get(individuals, Snapshot),
    lists:foreach(fun(Ind) ->
        ?assert(maps:is_key(island_id, Ind))
    end, Individuals).

snapshot_has_island_snapshots_test() ->
    Config = test_init_config(),
    {ok, State, _Events} = island_strategy:init(Config),

    Snapshot = island_strategy:get_population_snapshot(State),
    Extra = maps:get(extra, Snapshot),

    ?assert(maps:is_key(island_snapshots, Extra)),
    IslandSnapshots = maps:get(island_snapshots, Extra),
    ?assertEqual(3, length(IslandSnapshots)).

%%% ============================================================================
%%% Topology Tests
%%% ============================================================================

ring_topology_test() ->
    Config = #{
        neuro_config => test_neuro_config(),
        strategy_params => #{
            island_count => 4,
            population_per_island => 3,
            topology => ring,
            island_strategy => generational_strategy
        },
        network_factory => mock_network_factory
    },
    {ok, State, _Events} = island_strategy:init(Config),

    Snapshot = island_strategy:get_population_snapshot(State),
    Extra = maps:get(extra, Snapshot),

    %% Total should be 4 islands * 3 = 12
    ?assertEqual(12, maps:get(size, Snapshot)),
    ?assertEqual(4, maps:get(island_count, Extra)).

full_topology_test() ->
    Config = #{
        neuro_config => test_neuro_config(),
        strategy_params => #{
            island_count => 3,
            population_per_island => 4,
            topology => full,
            island_strategy => generational_strategy
        },
        network_factory => mock_network_factory
    },
    {ok, State, _Events} = island_strategy:init(Config),

    Snapshot = island_strategy:get_population_snapshot(State),
    ?assertEqual(12, maps:get(size, Snapshot)).

%%% ============================================================================
%%% Meta-Controller Interface Tests
%%% ============================================================================

meta_inputs_returns_floats_test() ->
    Config = test_init_config(),
    {ok, State, _Events} = island_strategy:init(Config),

    Inputs = island_strategy:get_meta_inputs(State),

    ?assert(is_list(Inputs)),
    ?assertEqual(4, length(Inputs)),
    lists:foreach(fun(V) ->
        ?assert(is_float(V))
    end, Inputs).

meta_inputs_normalized_test() ->
    Config = test_init_config(),
    {ok, State, _Events} = island_strategy:init(Config),

    Inputs = island_strategy:get_meta_inputs(State),

    lists:foreach(fun(V) ->
        ?assert(V >= 0.0),
        ?assert(V =< 1.0)
    end, Inputs).

apply_params_propagates_to_islands_test() ->
    Config = test_init_config(),
    {ok, State, _Events} = island_strategy:init(Config),

    %% Apply meta params
    NewState = island_strategy:apply_meta_params(#{mutation_rate => 0.25}, State),

    %% Check that state was updated (strategy continues to work)
    Snapshot = island_strategy:get_population_snapshot(NewState),
    ?assertEqual(15, maps:get(size, Snapshot)).

%%% ============================================================================
%%% Tick Tests
%%% ============================================================================

tick_propagates_to_islands_test() ->
    Config = test_init_config(),
    {ok, State, _Events} = island_strategy:init(Config),

    {Actions, Events, _NewState} = island_strategy:tick(State),

    %% Should complete without error
    ?assert(is_list(Actions)),
    ?assert(is_list(Events)).

%%% ============================================================================
%%% Behaviour Callback Tests
%%% ============================================================================

behaviour_init_dispatch_test() ->
    Config = test_init_config(),
    {ok, State, Events} = evolution_strategy:init(island_strategy, Config),

    ?assert(is_tuple(State)),
    ?assert(is_list(Events)),
    BirthEvents = [E || E <- Events, is_record(E, individual_born)],
    ?assertEqual(15, length(BirthEvents)).

%%% ============================================================================
%%% Different Sub-Strategy Tests
%%% ============================================================================

init_with_steady_state_sub_strategy_test() ->
    Config = #{
        neuro_config => test_neuro_config(),
        strategy_params => #{
            island_count => 2,
            population_per_island => 4,
            migration_interval => 10,
            topology => ring,
            island_strategy => steady_state_strategy,
            island_strategy_params => #{
                replacement_count => 1,
                victim_selection => worst
            }
        },
        network_factory => mock_network_factory
    },
    {ok, State, Events} = island_strategy:init(Config),

    BirthEvents = [E || E <- Events, is_record(E, individual_born)],
    ?assertEqual(8, length(BirthEvents)),  % 2 islands * 4

    Snapshot = island_strategy:get_population_snapshot(State),
    ?assertEqual(8, maps:get(size, Snapshot)).
