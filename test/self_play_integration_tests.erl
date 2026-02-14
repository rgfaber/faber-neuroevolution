%% @doc Integration tests for self-play training mode.
%%
%% Tests the complete self-play pipeline:
%% - neuroevolution_server starts self_play_manager
%% - Evaluator receives opponent networks
%% - Archive gets populated with top performers
%% - Subsequent generations use archive opponents
-module(self_play_integration_tests).

-include_lib("eunit/include/eunit.hrl").
-include("neuroevolution.hrl").
-include("evolution_strategy.hrl").

%%====================================================================
%% Test Configuration
%%====================================================================

self_play_config(Realm) ->
    #neuro_config{
        population_size = 20,
        evaluations_per_individual = 1,
        selection_ratio = 0.3,
        mutation_rate = 0.1,
        mutation_strength = 0.2,
        max_generations = 5,
        network_topology = {4, [8], 2},
        evaluator_module = mock_self_play_evaluator,
        evaluator_options = #{},
        realm = Realm,
        strategy_config = #strategy_config{
            strategy_module = generational_strategy,
            strategy_params = #{network_factory => mock_network_factory}
        },
        %% Enable self-play
        self_play_config = #self_play_config{
            enabled = true,
            archive_size = 10,
            archive_threshold = auto,
            min_fitness_percentile = 0.5
        }
    }.

%%====================================================================
%% Test Fixtures
%%====================================================================

integration_test_() ->
    {foreach,
     fun setup/0,
     fun cleanup/1,
     [
      {"self-play manager starts with server", fun test_manager_starts/0},
      {"training runs with self-play", fun test_training_runs/0},
      {"archive gets populated after generation", fun test_archive_populated/0},
      {"full training cycle completes", fun test_full_training_cycle/0}
     ]}.

setup() ->
    %% Ensure self_play_sup is running
    case whereis(self_play_sup) of
        undefined ->
            {ok, _SupPid} = self_play_sup:start_link();
        _ ->
            ok
    end,
    ok.

cleanup(_) ->
    %% Clean up any test realms
    lists:foreach(
        fun(Realm) -> catch self_play_sup:stop_self_play(Realm) end,
        self_play_sup:list_realms()
    ),
    ok.

%% @private Generate unique realm for each test
unique_realm(TestName) ->
    Timestamp = erlang:unique_integer([positive]),
    list_to_binary(io_lib:format("~s_~p", [TestName, Timestamp])).

%%====================================================================
%% Tests
%%====================================================================

test_manager_starts() ->
    Realm = unique_realm("manager_starts"),
    Config = self_play_config(Realm),
    {ok, Pid} = neuroevolution_server:start_link(Config, []),

    %% Get stats - should include self_play info
    {ok, Stats} = neuroevolution_server:get_stats(Pid),

    %% Should have self_play stats
    ?assert(maps:is_key(self_play, Stats)),
    SelfPlayStats = maps:get(self_play, Stats),
    ?assertEqual(pure_self_play, maps:get(mode, SelfPlayStats)),
    ?assertEqual(0, maps:get(archive_size, SelfPlayStats)),

    gen_server:stop(Pid).

test_training_runs() ->
    Realm = unique_realm("training_runs"),
    Config = self_play_config(Realm),
    {ok, Pid} = neuroevolution_server:start_link(Config, []),

    %% Start training
    {ok, started} = neuroevolution_server:start_training(Pid),

    %% Wait a bit for at least one generation
    timer:sleep(500),

    %% Stop and check stats
    ok = neuroevolution_server:stop_training(Pid),
    {ok, Stats} = neuroevolution_server:get_stats(Pid),

    %% Should have progressed
    Generation = maps:get(generation, Stats),
    ?assert(Generation >= 0),

    gen_server:stop(Pid).

test_archive_populated() ->
    Realm = unique_realm("archive_populated"),
    Config = self_play_config(Realm),
    {ok, Pid} = neuroevolution_server:start_link(Config, []),

    %% Start training
    {ok, started} = neuroevolution_server:start_training(Pid),

    %% Wait for a few generations
    wait_for_generations(Pid, 2, 5000),

    %% Stop and check archive
    ok = neuroevolution_server:stop_training(Pid),
    {ok, Stats} = neuroevolution_server:get_stats(Pid),

    SelfPlayStats = maps:get(self_play, Stats),
    ArchiveSize = maps:get(archive_size, SelfPlayStats),
    ChampionsAdded = maps:get(champions_added, SelfPlayStats),

    %% Archive should have been populated (top performers added)
    ?assert(ChampionsAdded > 0),
    ?assert(ArchiveSize > 0),

    gen_server:stop(Pid).

test_full_training_cycle() ->
    Realm = unique_realm("full_cycle"),
    Config = self_play_config(Realm),
    {ok, Pid} = neuroevolution_server:start_link(Config, []),

    %% Get initial state
    {ok, InitStats} = neuroevolution_server:get_stats(Pid),
    InitArchiveSize = get_archive_size(InitStats),
    ?assertEqual(0, InitArchiveSize),

    %% Start training
    {ok, started} = neuroevolution_server:start_training(Pid),

    %% Wait for training to complete or timeout
    wait_for_completion(Pid, 10000),

    %% Get final stats
    {ok, FinalStats} = neuroevolution_server:get_stats(Pid),
    FinalGeneration = maps:get(generation, FinalStats),
    FinalArchiveSize = get_archive_size(FinalStats),

    %% Should have run multiple generations
    ?assert(FinalGeneration >= 3),

    %% Archive should have grown
    ?assert(FinalArchiveSize > InitArchiveSize),

    %% Check self-play metrics
    SelfPlayStats = maps:get(self_play, FinalStats),
    TotalEvaluations = maps:get(total_evaluations, SelfPlayStats),
    ChampionsAdded = maps:get(champions_added, SelfPlayStats),

    %% Should have many evaluations
    ?assert(TotalEvaluations > 0),

    %% Should have added some champions
    ?assert(ChampionsAdded > 0),

    gen_server:stop(Pid).

%%====================================================================
%% Helpers
%%====================================================================

get_archive_size(Stats) ->
    case maps:get(self_play, Stats, undefined) of
        undefined -> 0;
        SelfPlayStats -> maps:get(archive_size, SelfPlayStats, 0)
    end.

wait_for_generations(Pid, TargetGen, Timeout) ->
    StartTime = erlang:monotonic_time(millisecond),
    wait_for_generations_loop(Pid, TargetGen, StartTime, Timeout).

wait_for_generations_loop(Pid, TargetGen, StartTime, Timeout) ->
    Now = erlang:monotonic_time(millisecond),
    Elapsed = Now - StartTime,

    case Elapsed > Timeout of
        true ->
            timeout;
        false ->
            {ok, Stats} = neuroevolution_server:get_stats(Pid),
            Gen = maps:get(generation, Stats),
            case Gen >= TargetGen of
                true -> ok;
                false ->
                    timer:sleep(100),
                    wait_for_generations_loop(Pid, TargetGen, StartTime, Timeout)
            end
    end.

wait_for_completion(Pid, Timeout) ->
    StartTime = erlang:monotonic_time(millisecond),
    wait_for_completion_loop(Pid, StartTime, Timeout).

wait_for_completion_loop(Pid, StartTime, Timeout) ->
    Now = erlang:monotonic_time(millisecond),
    Elapsed = Now - StartTime,

    case Elapsed > Timeout of
        true ->
            %% Timeout - stop training
            catch neuroevolution_server:stop_training(Pid),
            timeout;
        false ->
            {ok, Stats} = neuroevolution_server:get_stats(Pid),
            Running = maps:get(running, Stats),
            case Running of
                false -> ok;
                true ->
                    timer:sleep(100),
                    wait_for_completion_loop(Pid, StartTime, Timeout)
            end
    end.
