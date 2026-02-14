%%%-------------------------------------------------------------------
%%% @doc Unit tests for mesh distribution modules.
%%%
%%% Tests for:
%%% - mesh_sup: Supervisor lifecycle
%%% - evaluator_pool_registry: Evaluator tracking and selection
%%% - macula_mesh: Mesh integration facade
%%% - distributed_evaluator: Distributed evaluation dispatch
%%%
%%% @end
%%%-------------------------------------------------------------------
-module(mesh_tests).

-include_lib("eunit/include/eunit.hrl").

%%% ============================================================================
%%% Test Fixtures
%%% ============================================================================

setup() ->
    %% Start required applications
    application:ensure_all_started(crypto),
    ok.

cleanup(_) ->
    %% Stop any started processes
    catch gen_server:stop(evaluator_pool_registry),
    catch gen_server:stop(macula_mesh),
    catch gen_server:stop(distributed_evaluator),
    catch supervisor:terminate_child(mesh_sup, evaluator_pool_registry),
    catch supervisor:terminate_child(mesh_sup, macula_mesh),
    catch supervisor:terminate_child(mesh_sup, distributed_evaluator),
    catch gen_server:stop(mesh_sup),
    ok.

%%% ============================================================================
%%% Evaluator Pool Registry Tests
%%% ============================================================================

evaluator_pool_registry_test_() ->
    {setup,
     fun setup/0,
     fun cleanup/1,
     [
        {"Start evaluator pool registry",
         fun() ->
             {ok, Pid} = evaluator_pool_registry:start_link(#{}),
             ?assert(is_pid(Pid)),
             ?assert(is_process_alive(Pid)),
             gen_server:stop(Pid)
         end},

        {"Register and get evaluator",
         fun() ->
             {ok, Pid} = evaluator_pool_registry:start_link(#{}),

             %% Register an evaluator
             ok = evaluator_pool_registry:register_evaluator(<<"node1">>, #{
                 endpoint => <<"quic://localhost:4433">>,
                 capacity => 4,
                 evaluator_module => mock_evaluator
             }),

             %% Get available evaluator
             {ok, Evaluator} = evaluator_pool_registry:get_available_evaluator(),
             ?assertEqual(<<"node1">>, element(2, Evaluator)),
             ?assertEqual(4, element(4, Evaluator)),

             gen_server:stop(Pid)
         end},

        {"Get all evaluators",
         fun() ->
             {ok, Pid} = evaluator_pool_registry:start_link(#{}),

             %% Register multiple evaluators
             ok = evaluator_pool_registry:register_evaluator(<<"node1">>, #{capacity => 2}),
             ok = evaluator_pool_registry:register_evaluator(<<"node2">>, #{capacity => 4}),
             ok = evaluator_pool_registry:register_evaluator(<<"node3">>, #{capacity => 1}),

             All = evaluator_pool_registry:get_all_evaluators(),
             ?assertEqual(3, length(All)),

             gen_server:stop(Pid)
         end},

        {"Unregister evaluator",
         fun() ->
             {ok, Pid} = evaluator_pool_registry:start_link(#{}),

             ok = evaluator_pool_registry:register_evaluator(<<"node1">>, #{capacity => 2}),
             ok = evaluator_pool_registry:register_evaluator(<<"node2">>, #{capacity => 4}),

             ?assertEqual(2, length(evaluator_pool_registry:get_all_evaluators())),

             ok = evaluator_pool_registry:unregister_evaluator(<<"node1">>),

             ?assertEqual(1, length(evaluator_pool_registry:get_all_evaluators())),

             gen_server:stop(Pid)
         end},

        {"No evaluators returns error",
         fun() ->
             {ok, Pid} = evaluator_pool_registry:start_link(#{}),

             Result = evaluator_pool_registry:get_available_evaluator(),
             ?assertEqual({error, no_evaluators}, Result),

             gen_server:stop(Pid)
         end},

        {"Report evaluation started increases active count",
         fun() ->
             {ok, Pid} = evaluator_pool_registry:start_link(#{}),

             ok = evaluator_pool_registry:register_evaluator(<<"node1">>, #{capacity => 4}),

             %% Get evaluator before
             {ok, Before} = evaluator_pool_registry:get_available_evaluator(),
             ?assertEqual(0, element(5, Before)),  % active field

             %% Report started
             evaluator_pool_registry:report_evaluation_started(<<"node1">>),
             timer:sleep(10),

             %% Get evaluator after
             {ok, After} = evaluator_pool_registry:get_available_evaluator(),
             ?assertEqual(1, element(5, After)),

             gen_server:stop(Pid)
         end},

        {"Report evaluation completed decreases active count",
         fun() ->
             {ok, Pid} = evaluator_pool_registry:start_link(#{}),

             ok = evaluator_pool_registry:register_evaluator(<<"node1">>, #{capacity => 4}),

             evaluator_pool_registry:report_evaluation_started(<<"node1">>),
             timer:sleep(10),

             {ok, During} = evaluator_pool_registry:get_available_evaluator(),
             ?assertEqual(1, element(5, During)),

             evaluator_pool_registry:report_evaluation_completed(<<"node1">>, 100),
             timer:sleep(10),

             {ok, After} = evaluator_pool_registry:get_available_evaluator(),
             ?assertEqual(0, element(5, After)),

             gen_server:stop(Pid)
         end},

        {"Load balancing prefers lower load",
         fun() ->
             {ok, Pid} = evaluator_pool_registry:start_link(#{}),

             %% Register two evaluators
             ok = evaluator_pool_registry:register_evaluator(<<"node1">>, #{capacity => 4}),
             ok = evaluator_pool_registry:register_evaluator(<<"node2">>, #{capacity => 4}),

             %% Load node1 heavily
             evaluator_pool_registry:report_evaluation_started(<<"node1">>),
             evaluator_pool_registry:report_evaluation_started(<<"node1">>),
             evaluator_pool_registry:report_evaluation_started(<<"node1">>),
             timer:sleep(10),

             %% Selection should prefer node2 (lower load)
             Selections = lists:map(
                 fun(_) ->
                     {ok, E} = evaluator_pool_registry:get_available_evaluator(#{prefer_local => 0.0}),
                     element(2, E)
                 end,
                 lists:seq(1, 10)
             ),

             %% Most selections should be node2
             Node2Count = length([N || N <- Selections, N =:= <<"node2">>]),
             ?assert(Node2Count >= 7),

             gen_server:stop(Pid)
         end},

        {"Get stats returns aggregated data",
         fun() ->
             {ok, Pid} = evaluator_pool_registry:start_link(#{realm => <<"test.realm">>}),

             ok = evaluator_pool_registry:register_evaluator(<<"node1">>, #{capacity => 2}),
             ok = evaluator_pool_registry:register_evaluator(<<"node2">>, #{capacity => 4}),

             Stats = evaluator_pool_registry:get_stats(),

             ?assertEqual(<<"test.realm">>, maps:get(realm, Stats)),
             ?assertEqual(2, maps:get(total_nodes, Stats)),
             ?assertEqual(6, maps:get(total_capacity, Stats)),
             ?assertEqual(0, maps:get(total_active, Stats)),

             gen_server:stop(Pid)
         end}
     ]}.

%%% ============================================================================
%%% Macula Mesh Tests
%%% ============================================================================

macula_mesh_test_() ->
    {setup,
     fun setup/0,
     fun cleanup/1,
     [
        {"Start macula mesh without macula available",
         fun() ->
             {ok, Pid} = macula_mesh:start_link(#{}),
             ?assert(is_pid(Pid)),

             %% Mesh should not be available (macula not compiled)
             ?assertNot(macula_mesh:is_mesh_available()),

             gen_server:stop(Pid)
         end},

        {"Get state returns configuration",
         fun() ->
             {ok, Pid} = macula_mesh:start_link(#{
                 realm => <<"my.realm">>,
                 evaluator_capacity => 8
             }),

             State = macula_mesh:get_state(),

             ?assertEqual(<<"my.realm">>, maps:get(realm, State)),
             ?assertEqual(8, maps:get(evaluator_capacity, State)),
             ?assertNot(maps:get(mesh_available, State)),

             gen_server:stop(Pid)
         end},

        {"Advertise evaluator without mesh falls back to local registration",
         fun() ->
             %% Ensure clean state
             catch gen_server:stop(evaluator_pool_registry),
             catch gen_server:stop(macula_mesh),
             timer:sleep(10),

             {ok, PoolPid} = evaluator_pool_registry:start_link(#{}),
             {ok, MeshPid} = macula_mesh:start_link(#{}),

             %% When mesh is unavailable, it falls back to local registration (returns ok)
             Result = macula_mesh:advertise_evaluator(mock_evaluator, 4),
             ?assertEqual(ok, Result),

             %% Verify the evaluator was registered locally
             All = evaluator_pool_registry:get_all_evaluators(),
             ?assertEqual(1, length(All)),

             gen_server:stop(MeshPid),
             gen_server:stop(PoolPid)
         end},

        {"Discover evaluators without mesh returns local only",
         fun() ->
             %% Ensure clean state by stopping any existing processes
             catch gen_server:stop(evaluator_pool_registry),
             catch gen_server:stop(macula_mesh),
             timer:sleep(10),

             {ok, PoolPid} = evaluator_pool_registry:start_link(#{}),
             {ok, MeshPid} = macula_mesh:start_link(#{}),

             %% Register a local evaluator
             evaluator_pool_registry:register_evaluator(<<"local">>, #{
                 capacity => 4,
                 evaluator_module => mock_evaluator
             }),

             {ok, Evaluators} = macula_mesh:discover_evaluators(<<"test.realm">>),
             ?assertEqual(1, length(Evaluators)),

             gen_server:stop(MeshPid),
             gen_server:stop(PoolPid)
         end}
     ]}.

%%% ============================================================================
%%% Distributed Evaluator Tests
%%% ============================================================================

distributed_evaluator_test_() ->
    {setup,
     fun setup/0,
     fun cleanup/1,
     [
        {"Start distributed evaluator",
         fun() ->
             {ok, PoolPid} = evaluator_pool_registry:start_link(#{}),
             {ok, MeshPid} = macula_mesh:start_link(#{}),
             {ok, DistPid} = distributed_evaluator:start_link(#{}),

             ?assert(is_pid(DistPid)),
             ?assert(is_process_alive(DistPid)),

             gen_server:stop(DistPid),
             gen_server:stop(MeshPid),
             gen_server:stop(PoolPid)
         end},

        {"Register evaluator creates local entry",
         fun() ->
             {ok, PoolPid} = evaluator_pool_registry:start_link(#{}),
             {ok, MeshPid} = macula_mesh:start_link(#{}),
             {ok, DistPid} = distributed_evaluator:start_link(#{}),

             %% Register evaluator
             distributed_evaluator:register_evaluator(mock_evaluator, #{capacity => 4}),

             %% Should have local evaluator
             All = evaluator_pool_registry:get_all_evaluators(),
             ?assertEqual(1, length(All)),

             gen_server:stop(DistPid),
             gen_server:stop(MeshPid),
             gen_server:stop(PoolPid)
         end},

        {"Get stats returns evaluation statistics",
         fun() ->
             {ok, PoolPid} = evaluator_pool_registry:start_link(#{}),
             {ok, MeshPid} = macula_mesh:start_link(#{}),
             {ok, DistPid} = distributed_evaluator:start_link(#{}),

             Stats = distributed_evaluator:get_stats(),

             ?assert(is_map(Stats)),
             ?assertEqual(0, maps:get(evaluations_started, Stats)),
             ?assertEqual(0, maps:get(evaluations_completed, Stats)),
             ?assert(maps:is_key(pool, Stats)),

             gen_server:stop(DistPid),
             gen_server:stop(MeshPid),
             gen_server:stop(PoolPid)
         end}
     ]}.

%%% ============================================================================
%%% Mesh Supervisor Tests
%%% ============================================================================

mesh_sup_test_() ->
    {setup,
     fun setup/0,
     fun cleanup/1,
     [
        {"Start mesh supervisor with mesh disabled",
         fun() ->
             {ok, Pid} = mesh_sup:start_link(#{mesh_enabled => false}),
             ?assert(is_pid(Pid)),

             %% Should have no children when disabled
             Children = supervisor:which_children(Pid),
             ?assertEqual(0, length(Children)),

             gen_server:stop(Pid)
         end},

        {"Start mesh supervisor with mesh enabled",
         fun() ->
             {ok, Pid} = mesh_sup:start_link(#{mesh_enabled => true}),
             ?assert(is_pid(Pid)),

             %% Should have 3 children when enabled
             Children = supervisor:which_children(Pid),
             ?assertEqual(3, length(Children)),

             %% All children should be running
             lists:foreach(
                 fun({_Id, ChildPid, _Type, _Modules}) ->
                     ?assert(is_pid(ChildPid)),
                     ?assert(is_process_alive(ChildPid))
                 end,
                 Children
             ),

             gen_server:stop(Pid)
         end}
     ]}.

%%% ============================================================================
%%% Integration Tests (Local Only - No Macula)
%%% ============================================================================

local_evaluation_test_() ->
    {setup,
     fun setup/0,
     fun cleanup/1,
     [
        {"Local evaluation falls back correctly",
         fun() ->
             {ok, PoolPid} = evaluator_pool_registry:start_link(#{}),
             {ok, MeshPid} = macula_mesh:start_link(#{}),
             {ok, DistPid} = distributed_evaluator:start_link(#{}),

             %% No evaluators registered, should fail gracefully
             Result = distributed_evaluator:evaluate(
                 #{genome => [1, 2, 3]},
                 nonexistent_evaluator
             ),

             %% Should get error since no evaluators and local eval will fail
             ?assertMatch({error, _}, Result),

             gen_server:stop(DistPid),
             gen_server:stop(MeshPid),
             gen_server:stop(PoolPid)
         end}
     ]}.
