%% @doc Unit tests for faber_neuroevolution_sup module.
-module(faber_neuroevolution_sup_tests).

-include_lib("eunit/include/eunit.hrl").
-include("neuroevolution.hrl").
-include("evolution_strategy.hrl").

%%% ============================================================================
%%% Test Fixtures
%%% ============================================================================

default_config() ->
    #neuro_config{
        population_size = 5,
        evaluations_per_individual = 1,
        selection_ratio = 0.20,
        mutation_rate = 0.10,
        mutation_strength = 0.3,
        max_generations = 10,
        network_topology = {8, [4], 3},
        evaluator_module = mock_evaluator,
        strategy_config = #strategy_config{
            strategy_module = generational_strategy,
            strategy_params = #{network_factory => mock_network_factory}
        }
    }.

%%% ============================================================================
%%% Lifecycle Tests
%%% ============================================================================

start_link_test() ->
    %% Ensure supervisor is not already running
    case whereis(faber_neuroevolution_sup) of
        undefined -> ok;
        Pid -> gen_server:stop(Pid)
    end,
    timer:sleep(10),

    {ok, SupPid} = faber_neuroevolution_sup:start_link(),
    ?assert(is_pid(SupPid)),
    ?assert(is_process_alive(SupPid)),
    ?assertEqual(SupPid, whereis(faber_neuroevolution_sup)),

    gen_server:stop(SupPid).

%%% ============================================================================
%%% Child Management Tests
%%% ============================================================================

start_server_test() ->
    %% Start supervisor
    {ok, SupPid} = faber_neuroevolution_sup:start_link(),

    %% Start a server
    Config = default_config(),
    {ok, ServerPid} = faber_neuroevolution_sup:start_server(Config),

    ?assert(is_pid(ServerPid)),
    ?assert(is_process_alive(ServerPid)),

    %% Cleanup
    faber_neuroevolution_sup:stop_server(ServerPid),
    gen_server:stop(SupPid).

start_server_with_options_test() ->
    {ok, SupPid} = faber_neuroevolution_sup:start_link(),

    Config = default_config(),
    Options = [{name, {local, test_server}}],
    {ok, ServerPid} = faber_neuroevolution_sup:start_server(Config, Options),

    ?assert(is_pid(ServerPid)),
    ?assertEqual(ServerPid, whereis(test_server)),

    faber_neuroevolution_sup:stop_server(ServerPid),
    gen_server:stop(SupPid).

start_multiple_servers_test() ->
    {ok, SupPid} = faber_neuroevolution_sup:start_link(),

    Config = default_config(),
    {ok, Server1} = faber_neuroevolution_sup:start_server(Config),
    {ok, Server2} = faber_neuroevolution_sup:start_server(Config),
    {ok, Server3} = faber_neuroevolution_sup:start_server(Config),

    ?assert(is_pid(Server1)),
    ?assert(is_pid(Server2)),
    ?assert(is_pid(Server3)),
    ?assert(Server1 =/= Server2),
    ?assert(Server2 =/= Server3),

    faber_neuroevolution_sup:stop_server(Server1),
    faber_neuroevolution_sup:stop_server(Server2),
    faber_neuroevolution_sup:stop_server(Server3),
    gen_server:stop(SupPid).

stop_server_test() ->
    {ok, SupPid} = faber_neuroevolution_sup:start_link(),

    Config = default_config(),
    {ok, ServerPid} = faber_neuroevolution_sup:start_server(Config),
    ?assert(is_process_alive(ServerPid)),

    ok = faber_neuroevolution_sup:stop_server(ServerPid),
    timer:sleep(10),
    ?assertNot(is_process_alive(ServerPid)),

    gen_server:stop(SupPid).

%%% ============================================================================
%%% Supervisor Strategy Tests
%%% ============================================================================

server_crash_handled_test() ->
    {ok, SupPid} = faber_neuroevolution_sup:start_link(),

    Config = default_config(),
    {ok, ServerPid} = faber_neuroevolution_sup:start_server(Config),

    %% Kill the server
    exit(ServerPid, kill),
    timer:sleep(50),

    %% Server should be dead (temporary restart strategy)
    ?assertNot(is_process_alive(ServerPid)),

    %% Supervisor should still be running
    ?assert(is_process_alive(SupPid)),

    gen_server:stop(SupPid).
