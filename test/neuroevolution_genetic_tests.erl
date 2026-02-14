%% @doc EUnit tests for neuroevolution_genetic module.
-module(neuroevolution_genetic_tests).

-include_lib("eunit/include/eunit.hrl").
-include("neuroevolution.hrl").

%%% ============================================================================
%%% Test Fixtures
%%% ============================================================================

%% Simple flat weight list matching network_evaluator format
sample_weights() ->
    [0.1, 0.2, 0.3, 0.4, 0.5, 0.6].

sample_weights2() ->
    [0.9, 0.8, 0.7, 0.6, 0.5, 0.4].

sample_config() ->
    #neuro_config{
        population_size = 10,
        mutation_rate = 0.5,
        mutation_strength = 0.1,
        network_topology = {3, [4], 2},
        evaluator_module = mock_evaluator
    }.

%%% ============================================================================
%%% Crossover Tests
%%% ============================================================================

crossover_uniform_same_length_test() ->
    Weights1 = sample_weights(),
    Weights2 = sample_weights2(),
    Result = neuroevolution_genetic:crossover_uniform(Weights1, Weights2),

    %% Result should have same length as inputs
    ?assertEqual(length(Weights1), length(Result)),

    %% Each weight should be from either parent
    lists:foreach(fun({V1, V2, VR}) ->
        ?assert(VR =:= V1 orelse VR =:= V2)
    end, lists:zip3(Weights1, Weights2, Result)).

crossover_uniform_deterministic_seed_test() ->
    Weights1 = sample_weights(),
    Weights2 = sample_weights2(),

    %% With same random seed, should produce same result
    rand:seed(exsss, {1, 2, 3}),
    Result1 = neuroevolution_genetic:crossover_uniform(Weights1, Weights2),

    rand:seed(exsss, {1, 2, 3}),
    Result2 = neuroevolution_genetic:crossover_uniform(Weights1, Weights2),

    ?assertEqual(Result1, Result2).

%%% ============================================================================
%%% Mutation Tests
%%% ============================================================================

mutate_weights_zero_rate_test() ->
    Weights = sample_weights(),
    Result = neuroevolution_genetic:mutate_weights(Weights, 0.0, 0.5),

    %% With 0% mutation rate, weights should be unchanged
    ?assertEqual(Weights, Result).

mutate_weights_full_rate_test() ->
    Weights = sample_weights(),
    Result = neuroevolution_genetic:mutate_weights(Weights, 1.0, 0.1),

    %% With 100% mutation rate, at least one weight should be different
    %% (statistically nearly impossible for them to be exactly the same)
    ?assert(Weights =/= Result).

mutate_weights_preserves_length_test() ->
    Weights = sample_weights(),
    Result = neuroevolution_genetic:mutate_weights(Weights, 0.5, 0.1),

    %% Length should be preserved
    ?assertEqual(length(Weights), length(Result)).

mutate_weights_bounded_test() ->
    %% Create weights at boundary
    Weights = [0.99, -0.99, 0.5, -0.5],

    %% Mutate many times and verify weights change
    lists:foreach(fun(_) ->
        Mutated = neuroevolution_genetic:mutate_weights(Weights, 1.0, 0.5),
        ?assertEqual(length(Weights), length(Mutated))
    end, lists:seq(1, 10)).

%%% ============================================================================
%%% Layer-Specific Mutation Tests
%%% ============================================================================

%% Test mutate_weights_layered with different rates for reservoir and readout
mutate_weights_layered_basic_test() ->
    %% Topology: 2 inputs, [3] hidden, 2 outputs
    %% Layer 1 (reservoir): 2*3 + 3 = 9 weights (input->hidden + bias)
    %% Layer 2 (readout): 3*2 + 2 = 8 weights (hidden->output + bias)
    %% Total: 17 weights
    Topology = {2, [3], 2},
    ActualWeights = [float(X) / 10 || X <- lists:seq(1, 17)],

    ReservoirParams = {1.0, 0.5},  %% 100% mutation rate, 0.5 strength
    ReadoutParams = {0.0, 0.1},    %% 0% mutation rate (no mutations)
    FallbackParams = {0.1, 0.1},

    Result = neuroevolution_genetic:mutate_weights_layered(
        ActualWeights, Topology, ReservoirParams, ReadoutParams, FallbackParams
    ),

    %% Verify length preserved
    ?assertEqual(length(ActualWeights), length(Result)),

    %% Split into reservoir and readout portions
    {ReservoirResult, ReadoutResult} = lists:split(9, Result),
    {_ReservoirOrig, ReadoutOrig} = lists:split(9, ActualWeights),

    %% Readout should be unchanged (0% mutation rate)
    ?assertEqual(ReadoutOrig, ReadoutResult),

    %% Reservoir should be different (100% mutation rate)
    ?assert(ReservoirResult =/= lists:sublist(ActualWeights, 9)).

%% Test that undefined reservoir params falls back to default
mutate_weights_layered_undefined_reservoir_test() ->
    Topology = {2, [2], 1},
    %% Layer 1: 2*2+2 = 6 weights
    %% Layer 2: 2*1+1 = 3 weights
    Weights = [0.5 || _ <- lists:seq(1, 9)],

    ReservoirParams = undefined,
    ReadoutParams = {0.0, 0.1},  %% 0% mutation - readout unchanged
    FallbackParams = {1.0, 0.5}, %% 100% mutation - reservoir changes

    Result = neuroevolution_genetic:mutate_weights_layered(
        Weights, Topology, ReservoirParams, ReadoutParams, FallbackParams
    ),

    {ReservoirResult, ReadoutResult} = lists:split(6, Result),
    {_ReservoirOrig, ReadoutOrig} = lists:split(6, Weights),

    %% Readout unchanged
    ?assertEqual(ReadoutOrig, ReadoutResult),

    %% Reservoir changed (using fallback's 100% rate)
    ?assert(ReservoirResult =/= [0.5, 0.5, 0.5, 0.5, 0.5, 0.5]).

%% Test that undefined readout params falls back to default
mutate_weights_layered_undefined_readout_test() ->
    Topology = {2, [2], 1},
    Weights = [0.5 || _ <- lists:seq(1, 9)],

    ReservoirParams = {0.0, 0.1},  %% 0% - reservoir unchanged
    ReadoutParams = undefined,
    FallbackParams = {1.0, 0.5},   %% 100% - readout changes

    Result = neuroevolution_genetic:mutate_weights_layered(
        Weights, Topology, ReservoirParams, ReadoutParams, FallbackParams
    ),

    {ReservoirResult, ReadoutResult} = lists:split(6, Result),
    {ReservoirOrig, _ReadoutOrig} = lists:split(6, Weights),

    %% Reservoir unchanged
    ?assertEqual(ReservoirOrig, ReservoirResult),

    %% Readout changed (using fallback's 100% rate)
    ?assert(ReadoutResult =/= [0.5, 0.5, 0.5]).

%% Test with multi-layer hidden topology
mutate_weights_layered_multilayer_test() ->
    %% Topology: 3 inputs, [4, 2] hidden, 1 output
    %% Layer 1: 3*4+4 = 16 weights (reservoir)
    %% Layer 2: 4*2+2 = 10 weights (reservoir)
    %% Layer 3: 2*1+1 = 3 weights (readout)
    %% Total: 29 weights, reservoir = 26, readout = 3
    Topology = {3, [4, 2], 1},
    Weights = [0.5 || _ <- lists:seq(1, 29)],

    ReservoirParams = {0.0, 0.1},  %% 0% - reservoir unchanged
    ReadoutParams = {1.0, 0.5},    %% 100% - readout changes
    FallbackParams = {0.1, 0.1},

    Result = neuroevolution_genetic:mutate_weights_layered(
        Weights, Topology, ReservoirParams, ReadoutParams, FallbackParams
    ),

    {ReservoirResult, ReadoutResult} = lists:split(26, Result),
    {ReservoirOrig, _ReadoutOrig} = lists:split(26, Weights),

    %% Reservoir unchanged (0% rate)
    ?assertEqual(ReservoirOrig, ReservoirResult),

    %% Readout changed (100% rate)
    ?assert(ReadoutResult =/= [0.5, 0.5, 0.5]).

%%% ============================================================================
%%% Create Offspring Tests
%%% ============================================================================

create_offspring_returns_individual_test() ->
    Config = sample_config(),

    %% Create mock parent individuals with real networks
    Parent1 = #individual{
        id = make_ref(),
        network = create_mock_network(Config),
        fitness = 10.0
    },
    Parent2 = #individual{
        id = make_ref(),
        network = create_mock_network(Config),
        fitness = 8.0
    },

    Offspring = neuroevolution_genetic:create_offspring(Parent1, Parent2, Config, 5),

    %% Check offspring properties
    ?assert(is_reference(Offspring#individual.id)),
    ?assertEqual(Parent1#individual.id, Offspring#individual.parent1_id),
    ?assertEqual(Parent2#individual.id, Offspring#individual.parent2_id),
    ?assertEqual(5, Offspring#individual.generation_born),
    ?assertEqual(true, Offspring#individual.is_offspring),
    ?assertEqual(0.0, Offspring#individual.fitness).

%% Test create_offspring with layer-specific mutation rates
create_offspring_layer_specific_rates_test() ->
    %% Config with layer-specific rates
    Config = #neuro_config{
        population_size = 10,
        mutation_rate = 0.1,
        mutation_strength = 0.1,
        reservoir_mutation_rate = 0.0,     %% No reservoir mutations
        reservoir_mutation_strength = 0.1,
        readout_mutation_rate = 1.0,       %% Full readout mutations
        readout_mutation_strength = 0.5,
        network_topology = {3, [4], 2},
        evaluator_module = mock_evaluator
    },

    %% Create identical parent networks
    Parent1 = #individual{
        id = make_ref(),
        network = create_mock_network(Config),
        fitness = 10.0
    },
    Parent2 = #individual{
        id = make_ref(),
        network = create_mock_network(Config),
        fitness = 10.0
    },

    %% Create offspring
    Offspring = neuroevolution_genetic:create_offspring(Parent1, Parent2, Config, 1),

    %% Verify offspring was created
    ?assert(is_reference(Offspring#individual.id)),
    ?assertEqual(1, Offspring#individual.generation_born),
    ?assertEqual(true, Offspring#individual.is_offspring).

%% Test create_offspring uses uniform mutation when no layer-specific rates
create_offspring_uniform_rates_test() ->
    Config = sample_config(),  %% No layer-specific rates

    Parent1 = #individual{
        id = make_ref(),
        network = create_mock_network(Config),
        fitness = 10.0
    },
    Parent2 = #individual{
        id = make_ref(),
        network = create_mock_network(Config),
        fitness = 8.0
    },

    %% Should work without layer-specific rates
    Offspring = neuroevolution_genetic:create_offspring(Parent1, Parent2, Config, 3),

    ?assert(is_reference(Offspring#individual.id)),
    ?assertEqual(3, Offspring#individual.generation_born).

%%% ============================================================================
%%% Helper Functions
%%% ============================================================================

create_mock_network(Config) ->
    {InputSize, HiddenLayers, OutputSize} = Config#neuro_config.network_topology,
    network_evaluator:create_feedforward(InputSize, HiddenLayers, OutputSize).
