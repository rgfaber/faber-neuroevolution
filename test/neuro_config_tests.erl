-module(neuro_config_tests).
-include_lib("eunit/include/eunit.hrl").
-include("neuroevolution.hrl").
-include("lc_chain.hrl").

%%% ============================================================================
%%% from_map/1 Tests
%%% ============================================================================

from_map_minimal_test() ->
    Config = neuro_config:from_map(#{
        network_topology => {4, [8], 2},
        evaluator_module => test_evaluator
    }),
    ?assertEqual(50, Config#neuro_config.population_size),
    ?assertEqual({4, [8], 2}, Config#neuro_config.network_topology),
    ?assertEqual(test_evaluator, Config#neuro_config.evaluator_module),
    %% Missing fields should be undefined
    ?assertEqual(undefined, Config#neuro_config.lc_chain_config),
    ?assertEqual(undefined, Config#neuro_config.checkpoint_interval),
    ?assertEqual(undefined, Config#neuro_config.checkpoint_config).

from_map_lc_chain_config_record_test() ->
    LcConfig = #lc_chain_config{l2_tau = 200.0, learning_rate = 0.01},
    Config = neuro_config:from_map(#{
        network_topology => {4, [8], 2},
        evaluator_module => test_evaluator,
        lc_chain_config => LcConfig
    }),
    ?assertMatch(#lc_chain_config{l2_tau = 200.0, learning_rate = 0.01},
                 Config#neuro_config.lc_chain_config).

from_map_lc_chain_config_map_test() ->
    Config = neuro_config:from_map(#{
        network_topology => {4, [8], 2},
        evaluator_module => test_evaluator,
        lc_chain_config => #{
            l2_tau => 150.0,
            l1_tau => 75.0,
            l0_tau => 15.0,
            learning_rate => 0.005,
            evolve_topology => false,
            gamma => 0.99
        }
    }),
    LcConfig = Config#neuro_config.lc_chain_config,
    ?assertMatch(#lc_chain_config{}, LcConfig),
    ?assertEqual(150.0, LcConfig#lc_chain_config.l2_tau),
    ?assertEqual(75.0, LcConfig#lc_chain_config.l1_tau),
    ?assertEqual(15.0, LcConfig#lc_chain_config.l0_tau),
    ?assertEqual(0.005, LcConfig#lc_chain_config.learning_rate),
    ?assertEqual(false, LcConfig#lc_chain_config.evolve_topology),
    ?assertEqual(0.99, LcConfig#lc_chain_config.gamma).

from_map_lc_chain_config_defaults_test() ->
    Config = neuro_config:from_map(#{
        network_topology => {4, [8], 2},
        evaluator_module => test_evaluator,
        lc_chain_config => #lc_chain_config{}
    }),
    LcConfig = Config#neuro_config.lc_chain_config,
    ?assertEqual(100.0, LcConfig#lc_chain_config.l2_tau),
    ?assertEqual(50.0, LcConfig#lc_chain_config.l1_tau),
    ?assertEqual(10.0, LcConfig#lc_chain_config.l0_tau),
    ?assertEqual(0.001, LcConfig#lc_chain_config.learning_rate),
    ?assertEqual(true, LcConfig#lc_chain_config.evolve_topology).

from_map_lc_chain_config_nil_test() ->
    Config = neuro_config:from_map(#{
        network_topology => {4, [8], 2},
        evaluator_module => test_evaluator,
        lc_chain_config => nil
    }),
    ?assertEqual(undefined, Config#neuro_config.lc_chain_config).

from_map_checkpoint_interval_test() ->
    Config = neuro_config:from_map(#{
        network_topology => {4, [8], 2},
        evaluator_module => test_evaluator,
        checkpoint_interval => 500
    }),
    ?assertEqual(500, Config#neuro_config.checkpoint_interval).

from_map_checkpoint_config_test() ->
    CheckpointConfig = #{checkpoint_dir => "/tmp/checkpoints", save_on_fitness_record => true},
    Config = neuro_config:from_map(#{
        network_topology => {4, [8], 2},
        evaluator_module => test_evaluator,
        checkpoint_config => CheckpointConfig
    }),
    ?assertEqual(CheckpointConfig, Config#neuro_config.checkpoint_config).

%%% ============================================================================
%%% to_map/1 Tests
%%% ============================================================================

to_map_roundtrip_lc_chain_test() ->
    LcConfig = #lc_chain_config{l2_tau = 200.0, learning_rate = 0.01},
    Config = neuro_config:from_map(#{
        network_topology => {4, [8], 2},
        evaluator_module => test_evaluator,
        lc_chain_config => LcConfig
    }),
    Map = neuro_config:to_map(Config),
    LcMap = maps:get(lc_chain_config, Map),
    ?assertEqual(200.0, maps:get(l2_tau, LcMap)),
    ?assertEqual(0.01, maps:get(learning_rate, LcMap)).

to_map_undefined_lc_chain_test() ->
    Config = neuro_config:from_map(#{
        network_topology => {4, [8], 2},
        evaluator_module => test_evaluator
    }),
    Map = neuro_config:to_map(Config),
    ?assertEqual(undefined, maps:get(lc_chain_config, Map)).

to_map_checkpoint_fields_test() ->
    Config = neuro_config:from_map(#{
        network_topology => {4, [8], 2},
        evaluator_module => test_evaluator,
        checkpoint_interval => 1000,
        checkpoint_config => #{checkpoint_dir => "/tmp"}
    }),
    Map = neuro_config:to_map(Config),
    ?assertEqual(1000, maps:get(checkpoint_interval, Map)),
    ?assertEqual(#{checkpoint_dir => "/tmp"}, maps:get(checkpoint_config, Map)).
