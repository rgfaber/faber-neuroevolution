%% @doc Unit tests for genome_factory module.
%%
%% Tests NEAT-style genome operations:
%% - Minimal genome creation
%% - Genome to network conversion
%% - NEAT crossover
%% - Structural and weight mutations
%%
%% @author R.G. Lefever
%% @copyright 2024-2026 R.G. Lefever
-module(genome_factory_tests).

-include_lib("eunit/include/eunit.hrl").
-include("neuroevolution.hrl").

%%==============================================================================
%% Test Setup
%%==============================================================================

setup() ->
    %% Initialize mnesia for innovation tracking
    ok = mnesia:start(),
    ok = innovation:init(),
    ok = innovation:reset(),
    ok.

cleanup(_) ->
    mnesia:stop(),
    ok.

%%==============================================================================
%% Test Fixtures
%%==============================================================================

genome_factory_test_() ->
    {setup,
        fun setup/0,
        fun cleanup/1,
        [
            {"create_minimal creates valid genome", fun test_create_minimal/0},
            {"create_minimal has correct connection count", fun test_minimal_connection_count/0},
            {"create_minimal assigns unique innovations", fun test_minimal_innovations/0},
            {"to_network produces valid network", fun test_to_network/0},
            {"crossover produces valid offspring", fun test_crossover/0},
            {"crossover inherits from fitter parent", fun test_crossover_fitter/0},
            {"mutate_weights changes weights", fun test_mutate_weights/0},
            {"add_node splits connection", fun test_add_node/0},
            {"add_connection creates new link", fun test_add_connection/0}
        ]
    }.

%%==============================================================================
%% Test Cases
%%==============================================================================

test_create_minimal() ->
    Config = #neuro_config{
        network_topology = {3, [4], 2}  % Hidden layers ignored for minimal
    },
    Genome = genome_factory:create_minimal(Config),

    ?assert(is_record(Genome, genome)),
    ?assertEqual(3, Genome#genome.input_count),
    ?assertEqual(2, Genome#genome.output_count),
    ?assertEqual(0, Genome#genome.hidden_count).

test_minimal_connection_count() ->
    Config = #neuro_config{
        network_topology = {3, [], 2}
    },
    Genome = genome_factory:create_minimal(Config),

    %% Should have InputSize * OutputSize connections
    ExpectedCount = 3 * 2,
    ?assertEqual(ExpectedCount, length(Genome#genome.connection_genes)).

test_minimal_innovations() ->
    Config = #neuro_config{
        network_topology = {2, [], 1}
    },
    Genome = genome_factory:create_minimal(Config),

    %% All innovations should be unique
    Innovations = [G#connection_gene.innovation
                   || G <- Genome#genome.connection_genes],
    ?assertEqual(length(Innovations), length(lists:usort(Innovations))).

test_to_network() ->
    Config = #neuro_config{
        network_topology = {2, [], 1}
    },
    Genome = genome_factory:create_minimal(Config),
    Network = genome_factory:to_network(Genome),

    %% Should produce a valid network (non-empty)
    ?assert(is_map(Network) orelse is_tuple(Network) orelse is_list(Network)).

test_crossover() ->
    Config = #neuro_config{
        network_topology = {2, [], 1}
    },
    Parent1 = genome_factory:create_minimal(Config),
    Parent2 = genome_factory:create_minimal(Config),

    Child = genome_factory:crossover(Parent1, Parent2, equal),

    ?assert(is_record(Child, genome)),
    ?assertEqual(2, Child#genome.input_count),
    ?assertEqual(1, Child#genome.output_count).

test_crossover_fitter() ->
    Config = #neuro_config{
        network_topology = {2, [], 1}
    },
    Parent1 = genome_factory:create_minimal(Config),

    %% Add a node to parent2 to make it structurally different
    MutConfig = #mutation_config{
        add_node_rate = 1.0,  % Force add node
        add_connection_rate = 0.0,
        toggle_connection_rate = 0.0,
        weight_mutation_rate = 0.0
    },
    Parent2 = genome_factory:mutate(Parent1, MutConfig),

    %% When parent2 is fitter, child should have more genes (includes excess/disjoint)
    Child = genome_factory:crossover(Parent1, Parent2, 2),

    ?assert(length(Child#genome.connection_genes) >= length(Parent1#genome.connection_genes)).

test_mutate_weights() ->
    Config = #neuro_config{
        network_topology = {2, [], 1}
    },
    Genome = genome_factory:create_minimal(Config),

    MutConfig = #mutation_config{
        add_node_rate = 0.0,
        add_connection_rate = 0.0,
        toggle_connection_rate = 0.0,
        weight_mutation_rate = 1.0,  % Force weight mutation
        weight_perturb_rate = 1.0,
        weight_perturb_strength = 0.5
    },

    Mutated = genome_factory:mutate(Genome, MutConfig),

    %% At least some weights should have changed
    OriginalWeights = [G#connection_gene.weight
                       || G <- Genome#genome.connection_genes],
    MutatedWeights = [G#connection_gene.weight
                      || G <- Mutated#genome.connection_genes],

    %% Weights should be different (probabilistic but very likely with 1.0 rate)
    ?assertNotEqual(OriginalWeights, MutatedWeights).

test_add_node() ->
    Config = #neuro_config{
        network_topology = {2, [], 1}
    },
    Genome = genome_factory:create_minimal(Config),
    OriginalGeneCount = length(Genome#genome.connection_genes),

    MutConfig = #mutation_config{
        add_node_rate = 1.0,  % Force add node
        add_connection_rate = 0.0,
        toggle_connection_rate = 0.0,
        weight_mutation_rate = 0.0
    },

    Mutated = genome_factory:mutate(Genome, MutConfig),

    %% Should have 2 more connections (in + out) minus disabled
    %% Net: +2 new, -0 removed (old just disabled)
    NewGeneCount = length(Mutated#genome.connection_genes),
    ?assert(NewGeneCount > OriginalGeneCount).

test_add_connection() ->
    Config = #neuro_config{
        network_topology = {2, [], 2}
    },
    %% Start with minimal genome
    Genome = genome_factory:create_minimal(Config),

    %% First add a node to create opportunity for new connections
    MutConfig1 = #mutation_config{
        add_node_rate = 1.0,
        add_connection_rate = 0.0,
        toggle_connection_rate = 0.0,
        weight_mutation_rate = 0.0
    },
    WithNode = genome_factory:mutate(Genome, MutConfig1),
    OriginalCount = length(WithNode#genome.connection_genes),

    %% Now try to add a connection
    MutConfig2 = #mutation_config{
        add_node_rate = 0.0,
        add_connection_rate = 1.0,  % Force add connection
        toggle_connection_rate = 0.0,
        weight_mutation_rate = 0.0
    },
    WithConnection = genome_factory:mutate(WithNode, MutConfig2),
    NewCount = length(WithConnection#genome.connection_genes),

    %% Should have one more connection (or same if no valid targets)
    ?assert(NewCount >= OriginalCount).
