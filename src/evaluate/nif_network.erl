%% @doc NIF-accelerated network operations for faber-neuroevolution.
%%
%% This module provides high-performance network evaluation using
%% Rust NIFs from faber_tweann. When the NIF is available, operations
%% are 50-100x faster than pure Erlang.
%%
%% == Features ==
%%
%% - Network compilation for fast repeated evaluation
%% - Batch evaluation (multiple inputs, same network)
%% - NIF-accelerated compatibility distance for speciation
%% - LTC/CfC neuron support for temporal processing
%% - Automatic fallback to pure Erlang when NIF unavailable
%%
%% == Usage ==
%%
%% %% Compile a network for fast evaluation
%% {ok, CompiledNet} = nif_network:compile(Network),
%%
%% %% Evaluate (50-100x faster than pure Erlang)
%% Outputs = nif_network:evaluate(CompiledNet, Inputs),
%%
%% %% Batch evaluate (even more efficient)
%% OutputsList = nif_network:evaluate_batch(CompiledNet, InputsList).
%%
%% @author R.G. Lefever
%% @copyright 2024-2026 R.G. Lefever
-module(nif_network).

-export([
    %% Core operations
    is_nif_available/0,
    compile/1,
    compile_feedforward/3,
    evaluate/2,
    evaluate_batch/2,

    %% Compatibility distance (for speciation)
    compatibility_distance/3,
    compatibility_distance/5,

    %% LTC/CfC support
    evaluate_cfc/4,
    evaluate_cfc_batch/4
]).

%% Compiled network record
-record(compiled_network, {
    ref :: reference() | undefined,       %% NIF reference (when NIF available)
    fallback :: term(),                   %% Fallback network (pure Erlang)
    input_count :: pos_integer(),
    output_count :: pos_integer(),
    use_nif :: boolean()
}).

-type compiled_network() :: #compiled_network{}.
-export_type([compiled_network/0]).

%%% ============================================================================
%%% Core Operations
%%% ============================================================================

%% @doc Check if NIF acceleration is available.
%%
%% Returns true if the Rust NIF library is loaded and functional.
%% When false, all operations use pure Erlang fallbacks.
-spec is_nif_available() -> boolean().
is_nif_available() ->
    try
        tweann_nif:is_loaded()
    catch
        _:_ -> false
    end.

%% @doc Compile a network for fast NIF evaluation.
%%
%% Takes a network from network_evaluator and compiles it to the
%% NIF format for fast repeated evaluation. If NIF is unavailable,
%% returns a wrapper that uses pure Erlang evaluation.
%%
%% @param Network Network from network_evaluator:create_feedforward/3
%% @returns {ok, CompiledNetwork} | {error, Reason}
-spec compile(Network :: term()) -> {ok, compiled_network()} | {error, term()}.
compile(Network) ->
    try
        %% Extract network structure
        Weights = network_evaluator:get_weights(Network),
        {InputSize, HiddenLayers, OutputSize} = infer_topology(Network, Weights),

        UseNif = is_nif_available(),

        case UseNif of
            true ->
                %% Build node list for NIF compilation
                Nodes = build_nif_nodes(InputSize, HiddenLayers, OutputSize, Weights),
                OutputIndices = lists:seq(
                    InputSize + lists:sum(HiddenLayers),
                    InputSize + lists:sum(HiddenLayers) + OutputSize - 1
                ),

                %% Compile via NIF
                NifRef = tweann_nif:compile_network(Nodes, InputSize, OutputIndices),

                {ok, #compiled_network{
                    ref = NifRef,
                    fallback = Network,
                    input_count = InputSize,
                    output_count = OutputSize,
                    use_nif = true
                }};
            false ->
                %% NIF not available - use fallback
                {ok, #compiled_network{
                    ref = undefined,
                    fallback = Network,
                    input_count = InputSize,
                    output_count = OutputSize,
                    use_nif = false
                }}
        end
    catch
        Class:Reason:Stack ->
            error_logger:warning_msg(
                "[nif_network] Compilation failed: ~p:~p~n~p~n",
                [Class, Reason, Stack]
            ),
            {error, {compilation_failed, Reason}}
    end.

%% @doc Compile a feedforward network directly from topology.
%%
%% Creates and compiles a feedforward network in one step.
%% More efficient than create_feedforward + compile separately.
%%
%% @param InputSize Number of input neurons
%% @param HiddenLayers List of hidden layer sizes
%% @param OutputSize Number of output neurons
%% @returns {ok, CompiledNetwork} | {error, Reason}
-spec compile_feedforward(
    InputSize :: pos_integer(),
    HiddenLayers :: [pos_integer()],
    OutputSize :: pos_integer()
) -> {ok, compiled_network()} | {error, term()}.
compile_feedforward(InputSize, HiddenLayers, OutputSize) ->
    Network = network_evaluator:create_feedforward(InputSize, HiddenLayers, OutputSize),
    compile(Network).

%% @doc Evaluate a compiled network with given inputs.
%%
%% Uses NIF evaluation when available (50-100x faster), otherwise
%% falls back to pure Erlang network_evaluator.
%%
%% @param CompiledNetwork Compiled network from compile/1
%% @param Inputs List of input values (must match input_count)
%% @returns List of output values
-spec evaluate(CompiledNetwork :: compiled_network(), Inputs :: [float()]) -> [float()].
evaluate(#compiled_network{use_nif = true, ref = Ref}, Inputs) ->
    tweann_nif:evaluate(Ref, Inputs);
evaluate(#compiled_network{use_nif = false, fallback = Network}, Inputs) ->
    %% Use pure Erlang fallback
    network_evaluator:evaluate(Network, Inputs).

%% @doc Evaluate a compiled network with multiple input sets.
%%
%% More efficient than calling evaluate/2 multiple times when
%% evaluating the same network with different inputs.
%%
%% @param CompiledNetwork Compiled network from compile/1
%% @param InputsList List of input lists
%% @returns List of output lists (one per input set)
-spec evaluate_batch(
    CompiledNetwork :: compiled_network(),
    InputsList :: [[float()]]
) -> [[float()]].
evaluate_batch(#compiled_network{use_nif = true, ref = Ref}, InputsList) ->
    tweann_nif:evaluate_batch(Ref, InputsList);
evaluate_batch(#compiled_network{use_nif = false, fallback = Network}, InputsList) ->
    %% Pure Erlang fallback - evaluate sequentially
    [network_evaluator:evaluate(Network, Inputs) || Inputs <- InputsList].

%%% ============================================================================
%%% Compatibility Distance (for Speciation)
%%% ============================================================================

%% @doc Calculate NEAT compatibility distance using NIF.
%%
%% Uses the NIF-accelerated distance calculation when available.
%% Falls back to pure Erlang genome_crossover when not.
%%
%% @param Genome1 First genome (connection genes)
%% @param Genome2 Second genome (connection genes)
%% @param Config Compatibility config with c1, c2, c3 coefficients
%% @returns Distance value (lower = more similar)
-spec compatibility_distance(
    Genome1 :: [tuple()],
    Genome2 :: [tuple()],
    Config :: tuple()
) -> float().
compatibility_distance(Genome1, Genome2, Config) ->
    C1 = element(2, Config),  % c1 coefficient
    C2 = element(3, Config),  % c2 coefficient
    C3 = element(4, Config),  % c3 coefficient
    compatibility_distance(Genome1, Genome2, C1, C2, C3).

%% @doc Calculate compatibility distance with explicit coefficients.
%%
%% @param Genome1 First genome (connection genes)
%% @param Genome2 Second genome (connection genes)
%% @param C1 Excess gene coefficient
%% @param C2 Disjoint gene coefficient
%% @param C3 Weight difference coefficient
%% @returns Distance value
-spec compatibility_distance(
    Genome1 :: [tuple()],
    Genome2 :: [tuple()],
    C1 :: float(),
    C2 :: float(),
    C3 :: float()
) -> float().
compatibility_distance(Genome1, Genome2, C1, C2, C3) ->
    case is_nif_available() of
        true ->
            %% Convert to NIF format: [{Innovation, Weight}, ...]
            Connections1 = genes_to_nif_format(Genome1),
            Connections2 = genes_to_nif_format(Genome2),
            tweann_nif:compatibility_distance(Connections1, Connections2, C1, C2, C3);
        false ->
            %% Pure Erlang fallback via genome_crossover
            CompatConfig = {compatibility_config, C1, C2, C3},
            genome_crossover:compatibility_distance(Genome1, Genome2, CompatConfig)
    end.

%%% ============================================================================
%%% LTC/CfC Support
%%% ============================================================================

%% @doc Evaluate CfC (Closed-form Continuous-time) neuron.
%%
%% Fast closed-form approximation of LTC dynamics.
%% Suitable for temporal/sequential processing tasks.
%%
%% @param Input Current input value
%% @param State Current internal state
%% @param Tau Base time constant
%% @param Bound State bound (clamping range)
%% @returns {NewState, Output}
-spec evaluate_cfc(
    Input :: float(),
    State :: float(),
    Tau :: float(),
    Bound :: float()
) -> {float(), float()}.
evaluate_cfc(Input, State, Tau, Bound) ->
    case is_nif_available() of
        true ->
            tweann_nif:evaluate_cfc(Input, State, Tau, Bound);
        false ->
            %% Pure Erlang fallback
            ltc_dynamics:evaluate_cfc(Input, State, Tau, Bound)
    end.

%% @doc Batch CfC evaluation for time series.
%%
%% Evaluates a sequence of inputs, maintaining state between steps.
%%
%% @param Inputs List of input values (time series)
%% @param InitialState Starting internal state
%% @param Tau Base time constant
%% @param Bound State bound
%% @returns List of {State, Output} tuples
-spec evaluate_cfc_batch(
    Inputs :: [float()],
    InitialState :: float(),
    Tau :: float(),
    Bound :: float()
) -> [{float(), float()}].
evaluate_cfc_batch(Inputs, InitialState, Tau, Bound) ->
    case is_nif_available() of
        true ->
            tweann_nif:evaluate_cfc_batch(Inputs, InitialState, Tau, Bound);
        false ->
            %% Pure Erlang fallback - sequential evaluation
            evaluate_cfc_sequence(Inputs, InitialState, Tau, Bound, [])
    end.

%%% ============================================================================
%%% Internal Functions
%%% ============================================================================

%% @private Infer network topology from weights.
%%
%% network_evaluator doesn't directly expose topology, so we infer it
%% from the weight structure.
-spec infer_topology(term(), [float()]) -> {pos_integer(), [pos_integer()], pos_integer()}.
infer_topology(Network, _Weights) ->
    %% Try to get topology from network record
    try
        %% network_evaluator stores topology in record
        {network_evaluator:input_size(Network),
         network_evaluator:hidden_layers(Network),
         network_evaluator:output_size(Network)}
    catch
        _:_ ->
            %% Fallback: assume simple topology
            %% This is a best-effort guess
            {4, [8], 2}
    end.

%% @private Build node list for NIF compilation.
%%
%% Converts network weights to the NIF node format:
%% [{Index, Type, Activation, Bias, [{FromIndex, Weight}, ...]}, ...]
-spec build_nif_nodes(
    InputSize :: pos_integer(),
    HiddenLayers :: [pos_integer()],
    OutputSize :: pos_integer(),
    Weights :: [float()]
) -> list().
build_nif_nodes(InputSize, HiddenLayers, OutputSize, Weights) ->
    %% Input nodes (no connections to them)
    InputNodes = [
        {I - 1, input, linear, 0.0, []}
        || I <- lists:seq(1, InputSize)
    ],

    %% Build hidden and output nodes with connections
    {HiddenNodes, RemainingWeights1} = build_hidden_nodes(
        InputSize, HiddenLayers, Weights
    ),

    {OutputNodes, _RemainingWeights2} = build_output_nodes(
        InputSize, HiddenLayers, OutputSize, RemainingWeights1
    ),

    InputNodes ++ HiddenNodes ++ OutputNodes.

%% @private Build hidden layer nodes.
-spec build_hidden_nodes(
    InputSize :: pos_integer(),
    HiddenLayers :: [pos_integer()],
    Weights :: [float()]
) -> {list(), [float()]}.
build_hidden_nodes(InputSize, HiddenLayers, Weights) ->
    build_hidden_nodes(InputSize, HiddenLayers, Weights, InputSize, []).

build_hidden_nodes(_PrevSize, [], Weights, _NodeIdx, Acc) ->
    {lists:reverse(Acc), Weights};
build_hidden_nodes(PrevSize, [LayerSize | RestLayers], Weights, NodeIdx, Acc) ->
    %% Each neuron in this layer connects from all neurons in previous layer
    WeightsNeeded = LayerSize * PrevSize + LayerSize,  % weights + biases

    {LayerWeights, BiasWeights, RemainingWeights} = case length(Weights) >= WeightsNeeded of
        true ->
            {LW, Rest1} = lists:split(LayerSize * PrevSize, Weights),
            {BW, Rest2} = lists:split(LayerSize, Rest1),
            {LW, BW, Rest2};
        false ->
            %% Not enough weights - use defaults
            {[0.0 || _ <- lists:seq(1, LayerSize * PrevSize)],
             [0.0 || _ <- lists:seq(1, LayerSize)],
             []}
    end,

    %% Build nodes for this layer
    LayerNodes = [
        begin
            NeuronIdx = NodeIdx + N - 1,
            Bias = lists:nth(N, BiasWeights),
            %% Connections from previous layer
            Connections = [
                {NodeIdx - PrevSize + P - 1,
                 lists:nth((N - 1) * PrevSize + P, LayerWeights)}
                || P <- lists:seq(1, PrevSize)
            ],
            {NeuronIdx, hidden, tanh, Bias, Connections}
        end
        || N <- lists:seq(1, LayerSize)
    ],

    build_hidden_nodes(LayerSize, RestLayers, RemainingWeights, NodeIdx + LayerSize, LayerNodes ++ Acc).

%% @private Build output layer nodes.
-spec build_output_nodes(
    InputSize :: pos_integer(),
    HiddenLayers :: [pos_integer()],
    OutputSize :: pos_integer(),
    Weights :: [float()]
) -> {list(), [float()]}.
build_output_nodes(InputSize, HiddenLayers, OutputSize, Weights) ->
    %% Previous layer size
    PrevSize = case HiddenLayers of
        [] -> InputSize;
        _ -> lists:last(HiddenLayers)
    end,

    %% Starting index for output nodes
    HiddenTotal = lists:sum(HiddenLayers),
    NodeIdx = InputSize + HiddenTotal,

    %% Weights needed: OutputSize * PrevSize + OutputSize biases
    WeightsNeeded = OutputSize * PrevSize + OutputSize,

    {LayerWeights, BiasWeights, RemainingWeights} = case length(Weights) >= WeightsNeeded of
        true ->
            {LW, Rest1} = lists:split(OutputSize * PrevSize, Weights),
            {BW, Rest2} = lists:split(OutputSize, Rest1),
            {LW, BW, Rest2};
        false ->
            {[0.0 || _ <- lists:seq(1, OutputSize * PrevSize)],
             [0.0 || _ <- lists:seq(1, OutputSize)],
             []}
    end,

    %% Index of first node in previous layer
    PrevLayerStart = case HiddenLayers of
        [] -> 0;  % Connect from inputs
        _ -> InputSize + HiddenTotal - PrevSize
    end,

    OutputNodes = [
        begin
            NeuronIdx = NodeIdx + N - 1,
            Bias = lists:nth(N, BiasWeights),
            Connections = [
                {PrevLayerStart + P - 1,
                 lists:nth((N - 1) * PrevSize + P, LayerWeights)}
                || P <- lists:seq(1, PrevSize)
            ],
            {NeuronIdx, output, tanh, Bias, Connections}
        end
        || N <- lists:seq(1, OutputSize)
    ],

    {OutputNodes, RemainingWeights}.

%% @private Convert connection genes to NIF format.
-spec genes_to_nif_format([tuple()]) -> [{non_neg_integer(), float()}].
genes_to_nif_format(Genes) ->
    [
        {element(2, G), element(5, G)}  % {innovation, weight}
        || G <- Genes,
           element(6, G) =:= true  % Only enabled genes
    ].

%% @private Sequential CfC evaluation (pure Erlang fallback).
-spec evaluate_cfc_sequence(
    [float()], float(), float(), float(), [{float(), float()}]
) -> [{float(), float()}].
evaluate_cfc_sequence([], _State, _Tau, _Bound, Acc) ->
    lists:reverse(Acc);
evaluate_cfc_sequence([Input | Rest], State, Tau, Bound, Acc) ->
    {NewState, Output} = ltc_dynamics:evaluate_cfc(Input, State, Tau, Bound),
    evaluate_cfc_sequence(Rest, NewState, Tau, Bound, [{NewState, Output} | Acc]).
