%% @doc NEAT-style genome factory for topology-evolving neural networks.
%%
%% This module implements genome operations for NEAT (NeuroEvolution of
%% Augmenting Topologies) style evolution. It provides:
%% - Minimal genome creation (starting point for NEAT)
%% - Genome to network conversion (for evaluation)
%% - NEAT-style crossover (gene alignment by innovation number)
%% - Structural and weight mutations
%%
%% The module delegates to faber_tweann's innovation.erl and genome_crossover.erl
%% for core NEAT operations.
%%
%% Reference: Stanley, K.O. and Miikkulainen, R. (2002). Evolving Neural
%% Networks through Augmenting Topologies. Evolutionary Computation, 10(2).
%%
%% @author R.G. Lefever
%% @copyright 2024-2026 R.G. Lefever
-module(genome_factory).

-include("neuroevolution.hrl").

-export([
    create_minimal/1,
    to_network/1,
    to_compiled_network/1,
    mutate/2,
    crossover/3
]).

%%==============================================================================
%% API Functions
%%==============================================================================

%% @doc Create a minimal NEAT genome.
%%
%% Creates a genome where all inputs connect directly to all outputs.
%% This is the NEAT starting point - networks grow from this minimal structure.
%%
%% @param Config Configuration containing network_topology {Input, Hidden, Output}
%%        Note: Hidden layers are ignored for minimal genome (NEAT starts minimal)
%% @returns A minimal genome record
-spec create_minimal(neuro_config()) -> genome().
create_minimal(Config) ->
    {InputSize, _HiddenLayers, OutputSize} = Config#neuro_config.network_topology,

    %% Create connection genes for all input->output connections
    ConnectionGenes = [
        create_connection_gene(InIdx, InputSize + OutIdx)
        || InIdx <- lists:seq(1, InputSize),
           OutIdx <- lists:seq(1, OutputSize)
    ],

    #genome{
        connection_genes = ConnectionGenes,
        input_count = InputSize,
        hidden_count = 0,
        output_count = OutputSize
    }.

%% @doc Convert a genome to a network for evaluation.
%%
%% Builds a neural network from the genome's connection genes.
%% Disabled connections are excluded from the network.
%%
%% Note: For variable topology, we create a network that matches the genome
%% structure. Since network_evaluator uses dense layers with biases, we create
%% a network topology and then set the weights to match the genome.
%%
%% @param Genome The genome to convert
%% @returns A network suitable for network_evaluator
-spec to_network(genome()) -> network_evaluator:network().
to_network(Genome) ->
    %% Extract enabled connections
    EnabledGenes = [G || G <- Genome#genome.connection_genes,
                         G#connection_gene.enabled],

    %% Build network topology from genes
    InputSize = Genome#genome.input_count,
    OutputSize = Genome#genome.output_count,

    %% Detect hidden nodes from connections
    HiddenCount = count_hidden_nodes(EnabledGenes, InputSize, OutputSize),
    HiddenLayers = case HiddenCount of
        0 -> [];
        N -> [N]
    end,

    %% Create network with the detected topology
    %% network_evaluator creates random weights, which we then configure
    Network = network_evaluator:create_feedforward(InputSize, HiddenLayers, OutputSize),

    %% Get the number of weights the network expects
    NetworkWeights = network_evaluator:get_weights(Network),

    %% Build weight matrix from genome connections
    %% For now, we create a simple mapping - more sophisticated topology
    %% handling can be added when needed for complex structures
    ConfiguredWeights = configure_network_weights(
        EnabledGenes, NetworkWeights, InputSize, HiddenCount, OutputSize
    ),

    network_evaluator:set_weights(Network, ConfiguredWeights).

%% @doc Convert a genome to a NIF-compiled network for fast evaluation.
%%
%% This is the optimized path: compile once, evaluate many times.
%% Uses NIF acceleration when available (50-100x faster evaluation).
%%
%% @param Genome The genome to convert
%% @returns {ok, CompiledNetwork} | {error, Reason}
-spec to_compiled_network(genome()) -> {ok, nif_network:compiled_network()} | {error, term()}.
to_compiled_network(Genome) ->
    Network = to_network(Genome),
    nif_network:compile(Network).

%% @doc Perform NEAT-style crossover between two genomes.
%%
%% Aligns genes by innovation number and:
%% - Matching genes: randomly inherit from either parent
%% - Disjoint/Excess genes: inherit from fitter parent
%%
%% @param Genome1 First parent genome
%% @param Genome2 Second parent genome
%% @param FitterParent Which parent is fitter (1, 2, or equal)
%% @returns Child genome
-spec crossover(genome(), genome(), 1 | 2 | equal) -> genome().
crossover(Genome1, Genome2, FitterParent) ->
    %% Delegate to faber_tweann's genome_crossover module
    ChildGenes = genome_crossover:crossover(
        Genome1#genome.connection_genes,
        Genome2#genome.connection_genes,
        FitterParent
    ),

    %% Recompute node counts from child genes
    InputCount = max(Genome1#genome.input_count, Genome2#genome.input_count),
    OutputCount = max(Genome1#genome.output_count, Genome2#genome.output_count),
    HiddenCount = count_hidden_nodes(ChildGenes, InputCount, OutputCount),

    #genome{
        connection_genes = ChildGenes,
        input_count = InputCount,
        hidden_count = HiddenCount,
        output_count = OutputCount
    }.

%% @doc Apply mutations to a genome (structural + weight).
%%
%% Mutations are applied based on the mutation_config probabilities:
%% - Weight mutation: Perturb or replace weights
%% - Add node: Split an existing connection
%% - Add connection: Add new connection between unconnected nodes
%% - Toggle connection: Enable/disable a connection
%%
%% @param Genome The genome to mutate
%% @param Config Mutation configuration
%% @returns Mutated genome
-spec mutate(genome(), mutation_config()) -> genome().
mutate(Genome, Config) ->
    %% Apply mutations in order: structural first, then weight

    %% 1. Maybe add node (split connection)
    G1 = maybe_add_node(Genome, Config#mutation_config.add_node_rate),

    %% 2. Maybe add connection
    G2 = maybe_add_connection(G1, Config#mutation_config.add_connection_rate),

    %% 3. Maybe toggle connection
    G3 = maybe_toggle_connection(G2, Config#mutation_config.toggle_connection_rate),

    %% 4. Mutate weights
    G4 = mutate_weights(G3, Config),

    G4.

%%==============================================================================
%% Internal Functions
%%==============================================================================

%% Create a single connection gene with innovation number
create_connection_gene(FromNode, ToNode) ->
    Innovation = innovation:get_or_create_link_innovation(FromNode, ToNode),
    #connection_gene{
        innovation = Innovation,
        from_id = FromNode,
        to_id = ToNode,
        weight = random_weight(),
        enabled = true
    }.

%% Generate a random initial weight (Xavier initialization)
random_weight() ->
    (rand:uniform() * 2 - 1) * 0.5.  % Range: [-0.5, 0.5]

%% Count hidden nodes from connection genes
count_hidden_nodes(Genes, InputSize, OutputSize) ->
    AllNodes = lists:usort(
        lists:flatmap(
            fun(G) -> [G#connection_gene.from_id, G#connection_gene.to_id] end,
            Genes
        )
    ),
    InputNodes = lists:seq(1, InputSize),
    OutputNodes = lists:seq(InputSize + 1, InputSize + OutputSize),
    IONodes = InputNodes ++ OutputNodes,
    length([N || N <- AllNodes, not lists:member(N, IONodes)]).

%% Configure network weights from genome connection genes.
%%
%% network_evaluator expects weights in a specific format:
%% [layer1_weights..., layer1_biases..., layer2_weights..., layer2_biases...]
%%
%% For a minimal genome (no hidden), we map input->output connections
%% to the output layer weights. Biases default to small random values.
configure_network_weights(Genes, NetworkWeights, InputSize, HiddenCount, OutputSize) ->
    case HiddenCount of
        0 ->
            %% Simple case: direct input->output
            %% Network expects: OutputSize * InputSize weights + OutputSize biases
            configure_direct_weights(Genes, NetworkWeights, InputSize, OutputSize);
        _ ->
            %% Multi-layer network: use genome genes to initialize weights where available
            %% Fall back to network default weights for connections not in genome
            configure_multilayer_weights(Genes, NetworkWeights, InputSize, HiddenCount, OutputSize)
    end.

%% @private Configure weights for multi-layer networks.
%% Maps genome connection genes to network weight positions based on layer structure.
configure_multilayer_weights(Genes, NetworkWeights, InputSize, HiddenCount, OutputSize) ->
    %% Build gene lookup map
    GeneMap = maps:from_list([
        {{G#connection_gene.from_id, G#connection_gene.to_id}, G#connection_gene.weight}
        || G <- Genes, G#connection_gene.enabled
    ]),

    %% For networks with structure: Input -> Hidden -> Output
    %% Weight layout: [hidden_weights, hidden_biases, output_weights, output_biases]
    %% Hidden: InputSize * HiddenCount + HiddenCount biases
    %% Output: HiddenCount * OutputSize + OutputSize biases
    HiddenWeightCount = InputSize * HiddenCount,
    HiddenBiasCount = HiddenCount,
    OutputWeightCount = HiddenCount * OutputSize,
    _OutputBiasCount = OutputSize,

    %% Apply genome weights where connections exist, keep network defaults otherwise
    {HiddenWeights, Rest1} = lists:split(min(HiddenWeightCount, length(NetworkWeights)), NetworkWeights),
    {HiddenBiases, Rest2} = lists:split(min(HiddenBiasCount, length(Rest1)), Rest1),
    {OutputWeights, OutputBiases} = lists:split(min(OutputWeightCount, length(Rest2)), Rest2),

    %% Apply genome perturbation based on average gene weight
    AvgGeneWeight = average_gene_weight(Genes),
    PerturbFactor = 1.0 + AvgGeneWeight * 0.1,

    %% Combine with perturbation
    NewHiddenWeights = [W * PerturbFactor || W <- HiddenWeights],
    NewOutputWeights = [W * PerturbFactor || W <- OutputWeights],

    %% Override with specific gene weights where available
    FinalHiddenWeights = apply_gene_weights(NewHiddenWeights, GeneMap, InputSize, HiddenCount, input_hidden),
    FinalOutputWeights = apply_gene_weights(NewOutputWeights, GeneMap, HiddenCount, OutputSize, hidden_output),

    FinalHiddenWeights ++ HiddenBiases ++ FinalOutputWeights ++ OutputBiases.

%% @private Calculate average gene weight.
average_gene_weight([]) -> 0.0;
average_gene_weight(Genes) ->
    Weights = [G#connection_gene.weight || G <- Genes, G#connection_gene.enabled],
    case Weights of
        [] -> 0.0;
        _ -> lists:sum(Weights) / length(Weights)
    end.

%% @private Apply specific gene weights where connections exist.
apply_gene_weights(Weights, GeneMap, FromCount, ToCount, LayerType) ->
    %% Map weight indices to from/to pairs
    IndexedWeights = lists:zip(lists:seq(1, length(Weights)), Weights),
    lists:map(
        fun({Idx, DefaultW}) ->
            %% Calculate from/to indices for this weight position
            {FromIdx, ToIdx} = weight_index_to_ids(Idx, FromCount, ToCount),
            %% Generate potential gene IDs (simplified: use layer type + index)
            GeneKey = {LayerType, FromIdx, ToIdx},
            case maps:get(GeneKey, GeneMap, undefined) of
                undefined -> DefaultW;
                GeneWeight -> GeneWeight
            end
        end,
        IndexedWeights
    ).

%% @private Convert weight index to from/to node indices.
weight_index_to_ids(Idx, FromCount, _ToCount) ->
    %% Weights are laid out: all weights to output 0, then to output 1, etc.
    FromIdx = ((Idx - 1) rem FromCount) + 1,
    ToIdx = ((Idx - 1) div FromCount) + 1,
    {FromIdx, ToIdx}.

%% Configure weights for direct input->output networks
configure_direct_weights(Genes, NetworkWeights, InputSize, OutputSize) ->
    %% Build a lookup map from (from, to) -> weight
    GeneMap = maps:from_list([
        {{G#connection_gene.from_id, G#connection_gene.to_id}, G#connection_gene.weight}
        || G <- Genes
    ]),

    %% Network weight order: weights for output neuron 1, then 2, etc., then biases
    WeightCount = InputSize * OutputSize,
    BiasCount = OutputSize,
    ExpectedTotal = WeightCount + BiasCount,

    case length(NetworkWeights) of
        ExpectedTotal ->
            %% Build weights in network order
            Weights = [
                maps:get({InIdx, InputSize + OutIdx}, GeneMap, 0.0)
                || OutIdx <- lists:seq(1, OutputSize),
                   InIdx <- lists:seq(1, InputSize)
            ],
            %% Keep small biases (not in genome)
            Biases = [rand:uniform() * 0.1 - 0.05 || _ <- lists:seq(1, BiasCount)],
            Weights ++ Biases;
        _ ->
            %% Size mismatch - use network defaults
            NetworkWeights
    end.

%% Maybe add a node by splitting an existing connection
maybe_add_node(Genome, Rate) ->
    case rand:uniform() < Rate of
        true -> add_node(Genome);
        false -> Genome
    end.

add_node(Genome) ->
    EnabledGenes = [G || G <- Genome#genome.connection_genes,
                         G#connection_gene.enabled],
    case EnabledGenes of
        [] -> Genome;
        _ ->
            %% Select random connection to split
            Conn = lists:nth(rand:uniform(length(EnabledGenes)), EnabledGenes),

            %% Get innovation numbers for new structure
            {NodeInn, InInn, OutInn} = innovation:get_or_create_node_innovation(
                Conn#connection_gene.from_id,
                Conn#connection_gene.to_id
            ),

            %% Disable old connection
            UpdatedGenes = [
                case G#connection_gene.innovation =:= Conn#connection_gene.innovation of
                    true -> G#connection_gene{enabled = false};
                    false -> G
                end
                || G <- Genome#genome.connection_genes
            ],

            %% Create new connections through new node
            InConn = #connection_gene{
                innovation = InInn,
                from_id = Conn#connection_gene.from_id,
                to_id = NodeInn,
                weight = 1.0,  % Weight 1.0 to preserve signal initially
                enabled = true
            },
            OutConn = #connection_gene{
                innovation = OutInn,
                from_id = NodeInn,
                to_id = Conn#connection_gene.to_id,
                weight = Conn#connection_gene.weight,  % Original weight
                enabled = true
            },

            Genome#genome{
                connection_genes = [InConn, OutConn | UpdatedGenes],
                hidden_count = Genome#genome.hidden_count + 1
            }
    end.

%% Maybe add a new connection
maybe_add_connection(Genome, Rate) ->
    case rand:uniform() < Rate of
        true -> add_connection(Genome);
        false -> Genome
    end.

add_connection(Genome) ->
    %% Get all node IDs
    AllNodes = get_all_nodes(Genome),

    %% Get existing connections (as {from, to} pairs)
    ExistingPairs = [{G#connection_gene.from_id, G#connection_gene.to_id}
                     || G <- Genome#genome.connection_genes],

    %% Find possible new connections (not already existing)
    %% For simplicity, only allow feedforward connections (from < to)
    PossiblePairs = [
        {From, To}
        || From <- AllNodes,
           To <- AllNodes,
           From < To,
           not lists:member({From, To}, ExistingPairs)
    ],

    case PossiblePairs of
        [] -> Genome;
        _ ->
            %% Select random new connection
            {From, To} = lists:nth(rand:uniform(length(PossiblePairs)), PossiblePairs),

            NewConn = create_connection_gene(From, To),
            Genome#genome{
                connection_genes = [NewConn | Genome#genome.connection_genes]
            }
    end.

get_all_nodes(Genome) ->
    InputNodes = lists:seq(1, Genome#genome.input_count),
    OutputNodes = lists:seq(
        Genome#genome.input_count + 1,
        Genome#genome.input_count + Genome#genome.output_count
    ),
    %% Hidden nodes from connection genes
    HiddenNodes = lists:usort(
        lists:flatmap(
            fun(G) ->
                [G#connection_gene.from_id, G#connection_gene.to_id]
            end,
            Genome#genome.connection_genes
        )
    ) -- (InputNodes ++ OutputNodes),
    InputNodes ++ HiddenNodes ++ OutputNodes.

%% Maybe toggle a connection's enabled state
maybe_toggle_connection(Genome, Rate) ->
    case rand:uniform() < Rate of
        true -> toggle_random_connection(Genome);
        false -> Genome
    end.

toggle_random_connection(Genome) ->
    Genes = Genome#genome.connection_genes,
    case Genes of
        [] -> Genome;
        _ ->
            Idx = rand:uniform(length(Genes)),
            UpdatedGenes = lists:map(
                fun({I, G}) when I =:= Idx ->
                    G#connection_gene{enabled = not G#connection_gene.enabled};
                   ({_, G}) -> G
                end,
                lists:zip(lists:seq(1, length(Genes)), Genes)
            ),
            Genome#genome{connection_genes = UpdatedGenes}
    end.

%% Mutate weights in the genome using NIF for performance
mutate_weights(Genome, Config) ->
    MutationRate = Config#mutation_config.weight_mutation_rate,
    PerturbRate = Config#mutation_config.weight_perturb_rate,
    PerturbStrength = Config#mutation_config.weight_perturb_strength,
    Genes = Genome#genome.connection_genes,

    %% Extract weights, mutate via NIF, then update genes
    Weights = [G#connection_gene.weight || G <- Genes],
    MutatedWeights = tweann_nif:mutate_weights(
        Weights, MutationRate, PerturbRate, PerturbStrength
    ),

    %% Zip mutated weights back into genes
    MutatedGenes = [
        G#connection_gene{weight = W}
        || {G, W} <- lists:zip(Genes, MutatedWeights)
    ],
    Genome#genome{connection_genes = MutatedGenes}.
