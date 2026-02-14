%% @doc Genetic operators for neuroevolution.
%%
%% This module provides crossover and mutation operators for evolving
%% neural networks. Supports two modes:
%%
%% == NEAT Topology Evolution ==
%%
%% When individuals have genomes (connection_genes with innovation numbers),
%% uses NEAT-style crossover and structural mutations via genome_factory.
%% This enables topology-evolving networks.
%%
%% == Fixed Topology (Legacy) ==
%%
%% For backward compatibility, when individuals don't have genomes,
%% uses weight-only evolution with uniform crossover and perturbation mutation.
%%
%% Reference: Stanley, K.O. and Miikkulainen, R. (2002). "Evolving Neural
%% Networks through Augmenting Topologies." Evolutionary Computation, 10(2).
%%
%% @author R.G. Lefever
%% @copyright 2024-2026 R.G. Lefever
-module(neuroevolution_genetic).

-include("neuroevolution.hrl").

%% API
-export([
    crossover_uniform/2,
    mutate_weights/3,
    mutate_weights_layered/5,
    create_offspring/4,
    create_offspring_neat/4
]).

%%% ============================================================================
%%% API Functions
%%% ============================================================================

%% @doc Uniform crossover of two weight lists.
%%
%% For each weight position, randomly selects from either parent with
%% equal probability (50/50).
%%
%% Both weight lists must be the same length.
%%
%% Example:
%% Parent1 = [1.0, 2.0, 3.0, 4.0],
%% Parent2 = [5.0, 6.0, 7.0, 8.0],
%% %% Might produce: [1.0, 6.0, 3.0, 8.0]
%% Child = neuroevolution_genetic:crossover_uniform(Parent1, Parent2).
-spec crossover_uniform(Weights1, Weights2) -> ChildWeights when
    Weights1 :: [float()],
    Weights2 :: [float()],
    ChildWeights :: [float()].
crossover_uniform(Weights1, Weights2) ->
    lists:zipwith(
        fun(W1, W2) ->
            case rand:uniform() < 0.5 of
                true -> W1;
                false -> W2
            end
        end,
        Weights1,
        Weights2
    ).

%% @doc Mutate weights with given rate and strength.
%%
%% Each weight has MutationRate probability of being perturbed.
%% When mutated, a random value in [-Strength, +Strength] is added.
%%
%% Example:
%% Weights = [1.0, 2.0, 3.0],
%% Rate = 0.1,      %% 10% of weights mutated
%% Strength = 0.3,  %% Changes up to +/- 0.3
%% Mutated = neuroevolution_genetic:mutate_weights(Weights, Rate, Strength).
-spec mutate_weights(Weights, MutationRate, MutationStrength) -> MutatedWeights when
    Weights :: [float()],
    MutationRate :: float(),
    MutationStrength :: float(),
    MutatedWeights :: [float()].
mutate_weights(Weights, MutationRate, MutationStrength) ->
    lists:map(
        fun(W) ->
            case rand:uniform() < MutationRate of
                true ->
                    %% Perturb: add random value in [-Strength, +Strength]
                    Delta = (rand:uniform() - 0.5) * 2 * MutationStrength,
                    W + Delta;
                false ->
                    W
            end
        end,
        Weights
    ).

%% @doc Mutate weights with layer-specific rates.
%%
%% Applies different mutation rates to reservoir (hidden) and readout (output) layers.
%% The reservoir typically benefits from lower mutation rates for stability,
%% while the readout can adapt faster with higher rates.
%%
%% Reference: See guides/training-strategies.md for rationale.
%%
%% Example:
%% Weights = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],  %% 4 reservoir, 2 readout
%% Topology = {2, [2], 2},                     %% 2 inputs, 2 hidden, 2 outputs
%% ReservoirParams = {0.05, 0.2},              %% Low rate for reservoir
%% ReadoutParams = {0.20, 0.5},                %% High rate for readout
%% Mutated = neuroevolution_genetic:mutate_weights_layered(
%%     Weights, Topology, ReservoirParams, ReadoutParams
%% ).
-spec mutate_weights_layered(Weights, Topology, ReservoirParams, ReadoutParams, FallbackParams) -> MutatedWeights when
    Weights :: [float()],
    Topology :: {pos_integer(), [pos_integer()], pos_integer()},
    ReservoirParams :: {float(), float()} | undefined,
    ReadoutParams :: {float(), float()} | undefined,
    FallbackParams :: {float(), float()},
    MutatedWeights :: [float()].
mutate_weights_layered(Weights, Topology, ReservoirParams, ReadoutParams, FallbackParams) ->
    %% Calculate weight counts per layer
    LayerWeightCounts = compute_layer_weight_counts(Topology),

    %% The last layer is the readout, all others are reservoir
    {ReservoirCounts, [_ReadoutCount]} = lists:split(length(LayerWeightCounts) - 1, LayerWeightCounts),
    ReservoirTotal = lists:sum(ReservoirCounts),

    %% Split weights into reservoir and readout portions
    {ReservoirWeights, ReadoutWeights} = lists:split(ReservoirTotal, Weights),

    %% Get effective mutation parameters
    {ReservoirRate, ReservoirStrength} = effective_params(ReservoirParams, FallbackParams),
    {ReadoutRate, ReadoutStrength} = effective_params(ReadoutParams, FallbackParams),

    %% Apply mutations with layer-specific rates
    MutatedReservoir = mutate_weights(ReservoirWeights, ReservoirRate, ReservoirStrength),
    MutatedReadout = mutate_weights(ReadoutWeights, ReadoutRate, ReadoutStrength),

    %% Combine back
    MutatedReservoir ++ MutatedReadout.

%% @doc Create offspring from two parent individuals.
%%
%% Automatically selects the appropriate mode:
%% - NEAT mode: When both parents have genomes and topology_mutation_config is set
%% - Legacy mode: Weight-only evolution with uniform crossover
%%
%% For NEAT mode:
%% 1. Determine fitter parent
%% 2. Perform NEAT crossover (gene alignment by innovation number)
%% 3. Apply structural and weight mutations
%% 4. Convert genome to network for evaluation
%%
%% For Legacy mode:
%% 1. Extract weights from both parent networks
%% 2. Perform uniform crossover
%% 3. Apply mutation
%% 4. Create new network with child weights
%%
%% Returns a new individual record with lineage tracking.
-spec create_offspring(Parent1, Parent2, Config, Generation) -> Offspring when
    Parent1 :: individual(),
    Parent2 :: individual(),
    Config :: neuro_config(),
    Generation :: generation(),
    Offspring :: individual().
create_offspring(Parent1, Parent2, Config, Generation) ->
    %% Check if NEAT mode should be used
    case should_use_neat(Parent1, Parent2, Config) of
        true ->
            create_offspring_neat(Parent1, Parent2, Config, Generation);
        false ->
            create_offspring_legacy(Parent1, Parent2, Config, Generation)
    end.

%% @doc Create offspring using NEAT topology evolution.
%%
%% Uses genome_factory to perform NEAT-style crossover and mutation.
%% Both parents must have genomes for this to work.
%%
%% @param Parent1 First parent individual (must have genome)
%% @param Parent2 Second parent individual (must have genome)
%% @param Config Configuration with topology_mutation_config
%% @param Generation Current generation number
%% @returns Offspring individual with genome
-spec create_offspring_neat(Parent1, Parent2, Config, Generation) -> Offspring when
    Parent1 :: individual(),
    Parent2 :: individual(),
    Config :: neuro_config(),
    Generation :: generation(),
    Offspring :: individual().
create_offspring_neat(Parent1, Parent2, Config, Generation) ->
    %% Determine fitter parent for NEAT crossover
    FitterParent = determine_fitter_parent(Parent1, Parent2),

    %% NEAT crossover via genome_factory (aligns genes by innovation number)
    ChildGenome = genome_factory:crossover(
        Parent1#individual.genome,
        Parent2#individual.genome,
        FitterParent
    ),

    %% Apply mutations (structural + weight)
    MutConfig = get_mutation_config(Config),
    MutatedGenome = genome_factory:mutate(ChildGenome, MutConfig),

    %% Convert genome to network for evaluation
    ChildNetwork = genome_factory:to_network(MutatedGenome),

    %% Create offspring individual with lineage and genome
    ChildId = make_ref(),
    #individual{
        id = ChildId,
        network = ChildNetwork,
        genome = MutatedGenome,
        parent1_id = Parent1#individual.id,
        parent2_id = Parent2#individual.id,
        generation_born = Generation,
        is_offspring = true
    }.

%%% ============================================================================
%%% Internal Functions
%%% ============================================================================

%% @private Check if NEAT mode should be used for offspring creation.
should_use_neat(Parent1, Parent2, Config) ->
    HasGenomes = Parent1#individual.genome =/= undefined andalso
                 Parent2#individual.genome =/= undefined,
    HasMutConfig = Config#neuro_config.topology_mutation_config =/= undefined,
    HasGenomes andalso HasMutConfig.

%% @private Determine which parent is fitter for NEAT crossover.
%%
%% Returns: 1 if Parent1 is fitter, 2 if Parent2 is fitter, equal if same
determine_fitter_parent(Parent1, Parent2) ->
    F1 = Parent1#individual.fitness,
    F2 = Parent2#individual.fitness,
    if
        F1 > F2 -> 1;
        F2 > F1 -> 2;
        true -> equal
    end.

%% @private Get mutation config, with defaults if not specified.
get_mutation_config(Config) ->
    case Config#neuro_config.topology_mutation_config of
        undefined ->
            %% Default mutation config based on legacy rates
            #mutation_config{
                weight_mutation_rate = Config#neuro_config.mutation_rate,
                weight_perturb_rate = 0.9,
                weight_perturb_strength = Config#neuro_config.mutation_strength,
                add_node_rate = 0.03,
                add_connection_rate = 0.05,
                toggle_connection_rate = 0.01
            };
        MutConfig ->
            MutConfig
    end.

%% @private Create offspring using legacy weight-only evolution.
create_offspring_legacy(Parent1, Parent2, Config, Generation) ->
    %% Extract weights from parent networks
    Weights1 = network_evaluator:get_weights(Parent1#individual.network),
    Weights2 = network_evaluator:get_weights(Parent2#individual.network),

    %% Crossover
    ChildWeights = crossover_uniform(Weights1, Weights2),

    %% Mutation - use layer-specific rates if configured
    MutatedWeights = apply_mutation(ChildWeights, Config),

    %% Create new network with child weights
    {InputSize, HiddenLayers, OutputSize} = Config#neuro_config.network_topology,
    Network = network_evaluator:create_feedforward(InputSize, HiddenLayers, OutputSize),
    ChildNetwork = network_evaluator:set_weights(Network, MutatedWeights),

    %% Create offspring individual with lineage
    ChildId = make_ref(),
    #individual{
        id = ChildId,
        network = ChildNetwork,
        parent1_id = Parent1#individual.id,
        parent2_id = Parent2#individual.id,
        generation_born = Generation,
        is_offspring = true
    }.

%% @private Apply mutation with layer-specific or uniform rates.
%%
%% Uses layer-specific rates if any are configured, otherwise falls back to uniform mutation.
apply_mutation(Weights, Config) ->
    case has_layer_specific_rates(Config) of
        true ->
            %% Use layer-specific mutation
            Topology = Config#neuro_config.network_topology,
            ReservoirParams = get_reservoir_params(Config),
            ReadoutParams = get_readout_params(Config),
            FallbackParams = {Config#neuro_config.mutation_rate, Config#neuro_config.mutation_strength},
            mutate_weights_layered(Weights, Topology, ReservoirParams, ReadoutParams, FallbackParams);
        false ->
            %% Use uniform mutation (original behavior)
            mutate_weights(
                Weights,
                Config#neuro_config.mutation_rate,
                Config#neuro_config.mutation_strength
            )
    end.

%% @private Check if layer-specific mutation rates are configured.
has_layer_specific_rates(Config) ->
    Config#neuro_config.reservoir_mutation_rate =/= undefined orelse
    Config#neuro_config.reservoir_mutation_strength =/= undefined orelse
    Config#neuro_config.readout_mutation_rate =/= undefined orelse
    Config#neuro_config.readout_mutation_strength =/= undefined.

%% @private Get reservoir mutation parameters.
get_reservoir_params(Config) ->
    case {Config#neuro_config.reservoir_mutation_rate,
          Config#neuro_config.reservoir_mutation_strength} of
        {undefined, undefined} -> undefined;
        {Rate, undefined} -> {Rate, Config#neuro_config.mutation_strength};
        {undefined, Strength} -> {Config#neuro_config.mutation_rate, Strength};
        {Rate, Strength} -> {Rate, Strength}
    end.

%% @private Get readout mutation parameters.
get_readout_params(Config) ->
    case {Config#neuro_config.readout_mutation_rate,
          Config#neuro_config.readout_mutation_strength} of
        {undefined, undefined} -> undefined;
        {Rate, undefined} -> {Rate, Config#neuro_config.mutation_strength};
        {undefined, Strength} -> {Config#neuro_config.mutation_rate, Strength};
        {Rate, Strength} -> {Rate, Strength}
    end.

%% @private Get effective parameters, using fallback if undefined.
effective_params(undefined, Fallback) -> Fallback;
effective_params(Params, _Fallback) -> Params.

%% @private Compute the number of weights (including biases) per layer.
%%
%% For a topology {Input, [H1, H2, ...], Output}, returns weight counts for each layer.
%% Each layer has: (prev_size * current_size) weights + current_size biases.
compute_layer_weight_counts({InputSize, HiddenLayers, OutputSize}) ->
    AllLayers = [InputSize | HiddenLayers] ++ [OutputSize],
    Pairs = lists:zip(lists:droplast(AllLayers), tl(AllLayers)),
    [FromSize * ToSize + ToSize || {FromSize, ToSize} <- Pairs].
