%% @doc LTC-based meta-controller for adaptive hyperparameter optimization.
%%
%% This gen_server implements a meta-learning system that uses Liquid Time-Constant
%% (LTC) neural networks to dynamically control neuroevolution hyperparameters.
%%
%% == Architecture ==
%%
%% The meta-controller operates at a higher timescale than task networks.
%% It receives training metrics as inputs and outputs hyperparameters:
%% mutation_rate, mutation_strength, and selection_ratio.
%%
%% == LTC Advantage ==
%%
%% LTC neurons maintain internal state that evolves continuously.
%% This enables temporal memory of training dynamics, adaptive response
%% speed based on signal magnitude, and smooth parameter transitions.
%%
%% == Usage ==
%%
%% Create a config and start the meta-controller:
%%
%% Config = #meta_config{network_topology = {8, [16, 8], 4}},
%% {ok, Pid} = meta_controller:start_link(Config),
%% meta_controller:start_training(Pid),
%% NewParams = meta_controller:update(Pid, GenerationStats).
%%
%% @author R.G. Lefever
%% @copyright 2024-2026 R.G. Lefever
-module(meta_controller).
-behaviour(gen_server).

-include("neuroevolution.hrl").
-include("meta_controller.hrl").

%% API
-export([
    start_link/1,
    start_link/2,
    start_training/1,
    stop_training/1,
    update/2,
    get_l1_guidance/2,
    get_state/1,
    get_params/1,
    get_current_guidance/1,
    reset/1
]).

%% gen_server callbacks
-export([
    init/1,
    handle_call/3,
    handle_cast/2,
    handle_info/2,
    terminate/2,
    code_change/3
]).

%%% ============================================================================
%%% API Functions
%%% ============================================================================

%% @doc Start the meta-controller with given configuration.
-spec start_link(meta_config()) -> {ok, pid()} | {error, term()}.
start_link(Config) ->
    start_link(Config, []).

%% @doc Start the meta-controller with configuration and options.
%%
%% Options:
%% - `{id, Id}' - Server identifier (default: make_ref())
%% - `{name, Name}' - Register with given name
-spec start_link(meta_config(), proplists:proplist()) -> {ok, pid()} | {error, term()}.
start_link(Config, Options) ->
    Id = proplists:get_value(id, Options, make_ref()),
    case proplists:get_value(name, Options) of
        undefined ->
            gen_server:start_link(?MODULE, {Id, Config}, []);
        Name ->
            gen_server:start_link(Name, ?MODULE, {Id, Config}, [])
    end.

%% @doc Start the meta-learning training process.
-spec start_training(pid() | atom()) -> {ok, started | already_running}.
start_training(ServerRef) ->
    gen_server:call(ServerRef, start_training).

%% @doc Stop the meta-learning training process.
-spec stop_training(pid() | atom()) -> ok.
stop_training(ServerRef) ->
    gen_server:call(ServerRef, stop_training).

%% @doc Update the meta-controller with new generation stats.
%%
%% This is the main entry point called after each neuroevolution generation.
%% Returns new hyperparameters to use for the next generation.
-spec update(pid() | atom(), generation_stats() | map()) -> #{atom() => float()}.
update(ServerRef, GenerationStats) ->
    gen_server:call(ServerRef, {update, GenerationStats}).

%% @doc Get current meta-controller state (for visualization).
-spec get_state(pid() | atom()) -> {ok, map()}.
get_state(ServerRef) ->
    gen_server:call(ServerRef, get_state).

%% @doc Get current hyperparameter values (legacy API).
-spec get_params(pid() | atom()) -> #{atom() => float()}.
get_params(ServerRef) ->
    gen_server:call(ServerRef, get_params).

%% @doc Get L1 guidance based on generation statistics.
%%
%% This is the primary L2→L1 interface. Called by task_silo to get
%% meta-parameters that control L1's adjustment behavior.
%%
%% Returns an #l2_guidance{} record with:
%% - aggression_factor: How aggressive L1 adjustments should be
%% - exploration_step: How fast exploration_boost increases
%% - stagnation_sensitivity: Threshold for detecting stagnation
%% - topology_aggression: How much to boost topology mutations
%% - exploitation_weight: Balance between explore and exploit
-spec get_l1_guidance(pid() | atom(), map()) -> l2_guidance().
get_l1_guidance(ServerRef, GenStats) ->
    gen_server:call(ServerRef, {get_l1_guidance, GenStats}).

%% @doc Get current L1 guidance without updating (for monitoring).
-spec get_current_guidance(pid() | atom()) -> l2_guidance().
get_current_guidance(ServerRef) ->
    gen_server:call(ServerRef, get_current_guidance).

%% @doc Reset the meta-controller to initial state.
-spec reset(pid() | atom()) -> ok.
reset(ServerRef) ->
    gen_server:call(ServerRef, reset).

%%% ============================================================================
%%% gen_server Callbacks
%%% ============================================================================

%% @private
init({Id, Config}) ->
    error_logger:info_msg(
        "[meta_controller] Initializing with topology ~p, tau=~p~n",
        [Config#meta_config.network_topology, Config#meta_config.time_constant]
    ),

    %% Initialize LTC network
    {LtcWeights, LtcStates} = initialize_network(Config),

    %% Initialize output mapping
    OutputMapping = create_output_mapping(Config),

    State = #meta_state{
        id = Id,
        config = Config,
        ltc_weights = LtcWeights,
        ltc_states = LtcStates,
        current_params = default_params(Config),
        param_momentum = #{
            mutation_rate => 0.0,
            mutation_strength => 0.0,
            selection_ratio => 0.0
        }
    },

    %% Store output mapping in process dictionary for efficiency
    put(output_mapping, OutputMapping),

    {ok, State}.

%% @private
handle_call(start_training, _From, State = #meta_state{running = true}) ->
    {reply, {ok, already_running}, State};

handle_call(start_training, _From, State) ->
    error_logger:info_msg("[meta_controller] Starting meta-training~n"),
    NewState = State#meta_state{running = true},
    {reply, {ok, started}, NewState};

handle_call(stop_training, _From, State) ->
    NewState = State#meta_state{running = false},
    {reply, ok, NewState};

handle_call({update, GenStats}, _From, State) ->
    {NewParams, NewState} = process_generation(GenStats, State),
    {reply, NewParams, NewState};

handle_call(get_state, _From, State) ->
    StateMap = #{
        generation => State#meta_state.generation,
        current_params => State#meta_state.current_params,
        cumulative_reward => State#meta_state.cumulative_reward,
        best_fitness_ever => State#meta_state.best_fitness_ever,
        stagnation_count => State#meta_state.stagnation_count,
        running => State#meta_state.running,
        metrics_history_length => length(State#meta_state.metrics_history),
        ltc_states => State#meta_state.ltc_states
    },
    {reply, {ok, StateMap}, State};

handle_call(get_params, _From, State) ->
    {reply, State#meta_state.current_params, State};

handle_call({get_l1_guidance, GenStats}, _From, State) ->
    %% Process generation stats and compute L1 guidance
    {Guidance, NewState} = process_for_l1_guidance(GenStats, State),
    {reply, Guidance, NewState};

handle_call(get_current_guidance, _From, State) ->
    %% Return current guidance without updating
    Guidance = state_to_l1_guidance(State),
    {reply, Guidance, State};

handle_call(reset, _From, State) ->
    Config = State#meta_state.config,
    {LtcWeights, LtcStates} = initialize_network(Config),
    NewState = State#meta_state{
        ltc_weights = LtcWeights,
        ltc_states = LtcStates,
        current_params = default_params(Config),
        metrics_history = [],
        generation = 0,
        cumulative_reward = 0.0,
        best_fitness_ever = 0.0,
        stagnation_count = 0
    },
    {reply, ok, NewState};

handle_call(_Request, _From, State) ->
    {reply, {error, unknown_request}, State}.

%% @private
handle_cast(_Msg, State) ->
    {noreply, State}.

%% @private
handle_info(_Info, State) ->
    {noreply, State}.

%% @private
terminate(_Reason, _State) ->
    ok.

%% @private
code_change(_OldVsn, State, _Extra) ->
    {ok, State}.

%%% ============================================================================
%%% Internal Functions - Network Initialization
%%% ============================================================================

%% @private Initialize the LTC network structure.
initialize_network(Config) ->
    {InputSize, HiddenLayers, OutputSize} = Config#meta_config.network_topology,
    Tau = Config#meta_config.time_constant,
    Bound = Config#meta_config.state_bound,

    %% Build layer sizes: [InputSize | HiddenLayers] ++ [OutputSize]
    LayerSizes = [InputSize | HiddenLayers] ++ [OutputSize],

    %% Initialize weights and states for each layer connection
    {Weights, States} = initialize_layers(LayerSizes, Tau, Bound, 1, #{}, #{}),

    {Weights, States}.

%% @private Initialize weights between consecutive layers.
initialize_layers([_], _Tau, _Bound, _LayerIdx, Weights, States) ->
    {Weights, States};
initialize_layers([FromSize, ToSize | Rest], Tau, Bound, LayerIdx, Weights, States) ->
    %% Xavier initialization for weights
    Scale = math:sqrt(2.0 / (FromSize + ToSize)),

    %% Initialize neurons in this layer
    {LayerWeights, LayerStates} = lists:foldl(
        fun(NeuronIdx, {WAcc, SAcc}) ->
            NeuronId = {LayerIdx, NeuronIdx},

            %% Random weights for each input
            InputWeights = [{I, random_weight(Scale)} || I <- lists:seq(1, FromSize)],

            %% Random bias
            Bias = random_weight(Scale * 0.1),

            %% Random backbone/head weights for CfC
            BackboneWeights = [random_weight(0.1) || _ <- lists:seq(1, 3)],
            HeadWeights = [random_weight(0.1) || _ <- lists:seq(1, 3)],

            Neuron = #meta_neuron{
                id = NeuronId,
                internal_state = 0.0,
                time_constant = Tau,
                state_bound = Bound,
                input_weights = InputWeights,
                bias = Bias,
                backbone_weights = BackboneWeights,
                head_weights = HeadWeights
            },

            {WAcc#{NeuronId => Neuron}, SAcc#{NeuronId => 0.0}}
        end,
        {Weights, States},
        lists:seq(1, ToSize)
    ),

    initialize_layers([ToSize | Rest], Tau, Bound, LayerIdx + 1, LayerWeights, LayerStates).

%% @private Generate random weight with given scale.
random_weight(Scale) ->
    (rand:uniform() * 2.0 - 1.0) * Scale.

%% @private Create output mapping for parameters.
%%
%% Base outputs (always present):
%% - mutation_rate, mutation_strength, selection_ratio
%%
%% Resource-aware outputs (always present for safety):
%% - evaluations_per_individual: Reduces under memory pressure
%% - max_concurrent_evaluations: Limits parallelism under load
%%
%% Optional topology control (NEAT mode):
%% - add_node_rate: Controls structural growth
%% - add_connection_rate: Controls connectivity expansion
%% - complexity_penalty: Parsimony pressure to prevent bloat
create_output_mapping(Config) ->
    BaseParams = [mutation_rate, mutation_strength, selection_ratio],

    %% Resource-aware params (always enabled for safety)
    %% These allow the LTC to throttle evolution when resources are constrained
    ResourceParams = [evaluations_per_individual, max_concurrent_evaluations],

    %% Add topology control params when enabled
    TopologyParams = case Config#meta_config.control_topology of
        true -> [add_node_rate, add_connection_rate, complexity_penalty];
        _ -> []
    end,

    %% Add population size control when enabled
    PopParams = case Config#meta_config.control_population_size of
        true -> [population_size];
        false -> []
    end,

    Params = BaseParams ++ ResourceParams ++ TopologyParams ++ PopParams,

    IndexToParam = maps:from_list([{I, lists:nth(I, Params)} || I <- lists:seq(1, length(Params))]),
    ParamToIndex = maps:from_list([{lists:nth(I, Params), I} || I <- lists:seq(1, length(Params))]),

    %% All outputs use sigmoid activation for bounded parameters
    OutputActivations = maps:from_list([{I, sigmoid} || I <- lists:seq(1, length(Params))]),

    #output_mapping{
        index_to_param = IndexToParam,
        param_to_index = ParamToIndex,
        output_activations = OutputActivations
    }.

%% @private Default parameter values.
%%
%% Includes:
%% - Base evolution parameters
%% - Resource-aware parameters
%% - Topology parameters for NEAT mode (when control_topology=true)
default_params(_Config) ->
    #{
        %% Base evolution parameters
        mutation_rate => 0.10,
        mutation_strength => 0.30,
        selection_ratio => 0.20,
        %% Resource-aware parameters (always present)
        evaluations_per_individual => 10,
        max_concurrent_evaluations => 100000,  % This is Erlang - go big, resource_silo throttles if needed
        %% Population size (when control_population_size=true)
        population_size => 50,
        %% Topology control (NEAT mode)
        add_node_rate => 0.03,
        add_connection_rate => 0.05,
        complexity_penalty => 0.0
    }.

%%% ============================================================================
%%% Internal Functions - Generation Processing
%%% ============================================================================

%% @private Process a completed generation and compute new parameters.
process_generation(GenStats, State) ->
    Config = State#meta_state.config,

    %% Convert generation stats to metrics
    Metrics = stats_to_metrics(GenStats, State),

    %% Update metrics history
    NewHistory = update_history(Metrics, State#meta_state.metrics_history, Config),

    %% Compute input features for LTC network
    Inputs = compute_input_features(Metrics, NewHistory, State),

    %% Forward pass through LTC network
    {Outputs, NewLtcStates} = forward_pass(Inputs, State),

    %% Convert outputs to parameter values
    NewParams = outputs_to_params(Outputs, Config),

    %% Compute reward for this generation
    Reward = meta_reward:compute(Metrics, NewHistory, Config),

    %% Update training (gradient estimate)
    TrainingEvent = #meta_training_event{
        generation = State#meta_state.generation + 1,
        inputs = Inputs,
        outputs = Outputs,
        reward = Reward#meta_reward.total,
        gradients = #{}
    },

    %% Apply momentum smoothing to parameter changes
    SmoothedParams = apply_momentum(NewParams, State),

    %% Update stagnation counter
    NewStagnation = update_stagnation(Metrics, State),

    NewState = State#meta_state{
        generation = State#meta_state.generation + 1,
        metrics_history = NewHistory,
        ltc_states = NewLtcStates,
        current_params = SmoothedParams,
        cumulative_reward = State#meta_state.cumulative_reward + Reward#meta_reward.total,
        best_fitness_ever = max(State#meta_state.best_fitness_ever, Metrics#generation_metrics.best_fitness),
        stagnation_count = NewStagnation
    },

    %% Log progress periodically
    maybe_log_progress(NewState, TrainingEvent),

    {SmoothedParams, NewState}.

%% @private Convert generation stats to metrics record.
stats_to_metrics(GenStats, State) when is_record(GenStats, generation_stats) ->
    PrevMetrics = case State#meta_state.metrics_history of
        [] -> undefined;
        [H | _] -> H
    end,

    BestFitness = GenStats#generation_stats.best_fitness,
    AvgFitness = GenStats#generation_stats.avg_fitness,
    WorstFitness = GenStats#generation_stats.worst_fitness,

    FitnessDelta = case PrevMetrics of
        undefined -> 0.0;
        _ -> BestFitness - PrevMetrics#generation_metrics.best_fitness
    end,

    RelativeImprovement = case PrevMetrics of
        undefined -> 0.0;
        _ when PrevMetrics#generation_metrics.best_fitness > 0 ->
            FitnessDelta / PrevMetrics#generation_metrics.best_fitness;
        _ -> 0.0
    end,

    %% Compute fitness standard deviation
    FitnessStdDev = compute_fitness_std_dev(BestFitness, AvgFitness, WorstFitness),

    %% Compute structure metrics (simplified for now)
    StructureMetrics = compute_structure_metrics(GenStats),

    %% Evaluations per generation = population size (each individual evaluated once)
    EvaluationsUsed = GenStats#generation_stats.population_size,
    FitnessPerEval = case EvaluationsUsed > 0 of
        true -> BestFitness / EvaluationsUsed;
        false -> BestFitness
    end,

    #generation_metrics{
        generation = GenStats#generation_stats.generation,
        best_fitness = BestFitness,
        avg_fitness = AvgFitness,
        worst_fitness = WorstFitness,
        fitness_std_dev = FitnessStdDev,
        fitness_delta = FitnessDelta,
        relative_improvement = RelativeImprovement,
        population_diversity = FitnessStdDev,
        strategy_entropy = StructureMetrics#structure_metrics.strategy_entropy,
        evaluations_used = EvaluationsUsed,
        fitness_per_evaluation = FitnessPerEval,
        diversity_corridors = StructureMetrics#structure_metrics.diversity_corridors,
        adaptation_readiness = StructureMetrics#structure_metrics.adaptation_readiness,
        params_used = State#meta_state.current_params,
        timestamp = erlang:timestamp()
    };

%% Handle map-based stats (from Elixir)
stats_to_metrics(GenStatsMap, State) when is_map(GenStatsMap) ->
    PrevMetrics = case State#meta_state.metrics_history of
        [] -> undefined;
        [H | _] -> H
    end,

    BestFitness = maps:get(best_fitness, GenStatsMap, 0.0),
    AvgFitness = maps:get(avg_fitness, GenStatsMap, 0.0),
    WorstFitness = maps:get(worst_fitness, GenStatsMap, 0.0),
    Generation = maps:get(generation, GenStatsMap, 1),

    FitnessDelta = case PrevMetrics of
        undefined -> 0.0;
        _ -> BestFitness - PrevMetrics#generation_metrics.best_fitness
    end,

    RelativeImprovement = case PrevMetrics of
        undefined -> 0.0;
        _ when PrevMetrics#generation_metrics.best_fitness > 0 ->
            FitnessDelta / PrevMetrics#generation_metrics.best_fitness;
        _ -> 0.0
    end,

    FitnessStdDev = compute_fitness_std_dev(BestFitness, AvgFitness, WorstFitness),

    #generation_metrics{
        generation = Generation,
        best_fitness = BestFitness,
        avg_fitness = AvgFitness,
        worst_fitness = WorstFitness,
        fitness_std_dev = FitnessStdDev,
        fitness_delta = FitnessDelta,
        relative_improvement = RelativeImprovement,
        population_diversity = FitnessStdDev,
        strategy_entropy = 0.5,  %% Default
        evaluations_used = 1,
        fitness_per_evaluation = BestFitness,
        diversity_corridors = 0.5,
        adaptation_readiness = 0.5,
        params_used = State#meta_state.current_params,
        timestamp = erlang:timestamp()
    }.

%% @private Estimate fitness standard deviation from summary stats.
compute_fitness_std_dev(Best, _Avg, Worst) ->
    %% Simple approximation assuming roughly normal distribution
    Range = Best - Worst,
    %% Standard deviation is roughly range/4 for normal distribution
    max(0.0, Range / 4.0).

%% @private Compute structure metrics for normative awareness.
%%
%% When topology information is available (from top_individuals in gen stats),
%% computes actual topology metrics. Otherwise uses defaults.
compute_structure_metrics(GenStats) when is_record(GenStats, generation_stats) ->
    TopIndividuals = GenStats#generation_stats.top_individuals,
    compute_structure_metrics_from_individuals(TopIndividuals);
compute_structure_metrics(GenStatsMap) when is_map(GenStatsMap) ->
    TopIndividuals = maps:get(top_individuals, GenStatsMap, []),
    compute_structure_metrics_from_individuals(TopIndividuals).

%% @private Compute structure metrics from top individuals list.
compute_structure_metrics_from_individuals([]) ->
    %% No topology info - use defaults
    #structure_metrics{
        diversity_corridors = 0.5,
        adaptation_readiness = 0.5,
        breakthrough_potential = 0.5,
        strategy_entropy = 0.5
    };
compute_structure_metrics_from_individuals(TopIndividuals) ->
    %% Extract complexity values from individuals
    Complexities = [maps:get(complexity, Ind, 0) || Ind <- TopIndividuals],

    case Complexities of
        [] ->
            #structure_metrics{
                diversity_corridors = 0.5,
                adaptation_readiness = 0.5,
                breakthrough_potential = 0.5,
                strategy_entropy = 0.5
            };
        _ ->
            %% Compute topology-aware metrics
            AvgComplexity = lists:sum(Complexities) / length(Complexities),
            MaxComplexity = lists:max(Complexities),
            MinComplexity = lists:min(Complexities),
            Range = MaxComplexity - MinComplexity,

            %% Diversity corridors: variance in complexity (more diverse = higher)
            Variance = compute_variance(Complexities, AvgComplexity),
            DiversityCorridors = min(1.0, Variance / max(1.0, AvgComplexity)),

            %% Adaptation readiness: how much room for growth
            %% Higher when complexity is low (room to grow)
            MaxPossibleComplexity = 100.0,  %% Assume max complexity of 100
            AdaptationReadiness = max(0.0, 1.0 - (AvgComplexity / MaxPossibleComplexity)),

            %% Breakthrough potential: correlation between fitness and complexity
            %% High when diverse structures are being explored
            BreakthroughPotential = min(1.0, Range / max(1.0, AvgComplexity)),

            %% Strategy entropy: how evenly distributed complexities are
            StrategyEntropy = compute_complexity_entropy(Complexities),

            #structure_metrics{
                diversity_corridors = DiversityCorridors,
                adaptation_readiness = AdaptationReadiness,
                breakthrough_potential = BreakthroughPotential,
                strategy_entropy = StrategyEntropy
            }
    end.

%% @private Compute variance.
compute_variance([], _Mean) -> 0.0;
compute_variance([_], _Mean) -> 0.0;
compute_variance(Values, Mean) ->
    SumSquaredDiffs = lists:sum([(V - Mean) * (V - Mean) || V <- Values]),
    SumSquaredDiffs / length(Values).

%% @private Compute Shannon entropy of complexity distribution.
compute_complexity_entropy([]) -> 0.0;
compute_complexity_entropy(Values) ->
    %% Bin complexities into buckets and compute entropy
    Buckets = 10,
    MaxVal = max(1.0, lists:max(Values)),
    BinCounts = lists:foldl(
        fun(V, Acc) ->
            Bin = min(Buckets, max(1, trunc((V / MaxVal) * Buckets) + 1)),
            maps:update_with(Bin, fun(C) -> C + 1 end, 1, Acc)
        end,
        #{},
        Values
    ),

    Total = length(Values),
    Probs = [C / Total || C <- maps:values(BinCounts)],

    %% Shannon entropy normalized to [0, 1]
    MaxEntropy = math:log(Buckets),
    Entropy = -lists:sum([P * safe_log(P) || P <- Probs, P > 0]),
    case MaxEntropy > 0 of
        true -> Entropy / MaxEntropy;
        false -> 0.0
    end.

%% @private Safe log (avoid log(0)).
safe_log(X) when X > 0 -> math:log(X);
safe_log(_) -> 0.0.

%% @private Update metrics history (keep last N entries).
update_history(Metrics, History, Config) ->
    WindowSize = Config#meta_config.history_window,
    NewHistory = [Metrics | History],
    lists:sublist(NewHistory, WindowSize).

%% @private Compute input features for the LTC network.
%%
%% Features 1-8: Evolution metrics
%% Features 9-11: Resource metrics (memory, CPU, process pressure)
compute_input_features(Metrics, History, State) ->
    %% Normalize all features to roughly [-1, 1] or [0, 1] range

    %% Feature 1: Normalized best fitness
    BestFitness = Metrics#generation_metrics.best_fitness,
    MaxFitness = max(1.0, State#meta_state.best_fitness_ever),
    NormBestFitness = BestFitness / MaxFitness,

    %% Feature 2: Relative improvement (already normalized)
    RelImprovement = clamp(Metrics#generation_metrics.relative_improvement, -1.0, 1.0),

    %% Feature 3: Fitness variance (normalized by avg)
    AvgFitness = max(1.0, Metrics#generation_metrics.avg_fitness),
    NormVariance = clamp(Metrics#generation_metrics.fitness_std_dev / AvgFitness, 0.0, 1.0),

    %% Feature 4: Stagnation signal (increases with stagnation)
    StagnationSignal = sigmoid(State#meta_state.stagnation_count / 5.0),

    %% Feature 5: Generation progress (normalized)
    GenProgress = sigmoid(State#meta_state.generation / 100.0),

    %% Feature 6: Moving average improvement (trend)
    TrendSignal = compute_trend(History),

    %% Feature 7: Population diversity
    Diversity = clamp(Metrics#generation_metrics.population_diversity / MaxFitness, 0.0, 1.0),

    %% Feature 8: Strategy entropy
    Entropy = Metrics#generation_metrics.strategy_entropy,

    %% Features 9-11: Resource metrics (NEW)
    %% These allow the LTC to adapt hyperparameters based on system load
    ResourceMetrics = resource_monitor:get_normalized_metrics(),
    MemoryPressure = maps:get(memory_pressure, ResourceMetrics, 0.0),
    CpuPressure = maps:get(cpu_pressure, ResourceMetrics, 0.0),
    ProcessPressure = maps:get(process_pressure, ResourceMetrics, 0.0),

    [NormBestFitness, RelImprovement, NormVariance, StagnationSignal,
     GenProgress, TrendSignal, Diversity, Entropy,
     MemoryPressure, CpuPressure, ProcessPressure].

%% @private Compute improvement trend from history.
compute_trend([]) -> 0.0;
compute_trend([_]) -> 0.0;
compute_trend(History) ->
    %% Simple linear regression on fitness improvements
    Improvements = [M#generation_metrics.relative_improvement || M <- History],
    case length(Improvements) of
        N when N < 2 -> 0.0;
        N ->
            %% Weighted average with more weight on recent
            Weights = [math:pow(0.9, I) || I <- lists:seq(0, N - 1)],
            WeightSum = lists:sum(Weights),
            WeightedSum = lists:sum([W * V || {W, V} <- lists:zip(Weights, Improvements)]),
            clamp(WeightedSum / WeightSum, -1.0, 1.0)
    end.

%%% ============================================================================
%%% Internal Functions - LTC Forward Pass
%%% ============================================================================

%% @private Forward pass through LTC network.
forward_pass(Inputs, State) ->
    Config = State#meta_state.config,
    {_InputSize, HiddenLayers, OutputSize} = Config#meta_config.network_topology,
    NeuronType = Config#meta_config.neuron_type,

    LtcWeights = State#meta_state.ltc_weights,
    LtcStates = State#meta_state.ltc_states,

    %% Process through hidden layers
    {HiddenOutputs, NewStates1} = process_hidden_layers(
        Inputs, HiddenLayers, LtcWeights, LtcStates, NeuronType, 1
    ),

    %% Process output layer
    NumHiddenLayers = length(HiddenLayers),
    OutputLayerIdx = NumHiddenLayers + 1,
    {Outputs, NewStates2} = process_output_layer(
        HiddenOutputs, OutputSize, LtcWeights, NewStates1, NeuronType, OutputLayerIdx
    ),

    {Outputs, NewStates2}.

%% @private Process hidden layers.
process_hidden_layers(Inputs, [], _Weights, States, _NeuronType, _LayerIdx) ->
    {Inputs, States};
process_hidden_layers(Inputs, [LayerSize | Rest], Weights, States, NeuronType, LayerIdx) ->
    %% Process each neuron in this layer
    {LayerOutputs, NewStates} = lists:foldl(
        fun(NeuronIdx, {OutAcc, StateAcc}) ->
            NeuronId = {LayerIdx, NeuronIdx},
            Neuron = maps:get(NeuronId, Weights),
            OldState = maps:get(NeuronId, StateAcc, 0.0),

            %% Compute weighted input sum
            WeightedSum = compute_weighted_sum(Inputs, Neuron#meta_neuron.input_weights)
                        + Neuron#meta_neuron.bias,

            %% Apply LTC dynamics
            {NewState, Output} = apply_ltc_dynamics(
                NeuronType,
                WeightedSum,
                OldState,
                Neuron#meta_neuron.time_constant,
                Neuron#meta_neuron.state_bound,
                Neuron#meta_neuron.backbone_weights,
                Neuron#meta_neuron.head_weights
            ),

            {OutAcc ++ [Output], StateAcc#{NeuronId => NewState}}
        end,
        {[], States},
        lists:seq(1, LayerSize)
    ),

    process_hidden_layers(LayerOutputs, Rest, Weights, NewStates, NeuronType, LayerIdx + 1).

%% @private Process output layer.
process_output_layer(Inputs, OutputSize, Weights, States, NeuronType, LayerIdx) ->
    lists:foldl(
        fun(NeuronIdx, {OutAcc, StateAcc}) ->
            NeuronId = {LayerIdx, NeuronIdx},
            Neuron = maps:get(NeuronId, Weights),
            OldState = maps:get(NeuronId, StateAcc, 0.0),

            WeightedSum = compute_weighted_sum(Inputs, Neuron#meta_neuron.input_weights)
                        + Neuron#meta_neuron.bias,

            {NewState, Output} = apply_ltc_dynamics(
                NeuronType,
                WeightedSum,
                OldState,
                Neuron#meta_neuron.time_constant,
                Neuron#meta_neuron.state_bound,
                Neuron#meta_neuron.backbone_weights,
                Neuron#meta_neuron.head_weights
            ),

            {OutAcc ++ [Output], StateAcc#{NeuronId => NewState}}
        end,
        {[], States},
        lists:seq(1, OutputSize)
    ).

%% @private Compute weighted sum of inputs.
compute_weighted_sum(Inputs, InputWeights) ->
    lists:foldl(
        fun({{_InputIdx, Weight}, InputVal}, Acc) ->
            Acc + Weight * InputVal;
           ({InputIdx, Weight}, Acc) when is_integer(InputIdx) ->
            InputVal = lists:nth(InputIdx, Inputs),
            Acc + Weight * InputVal
        end,
        0.0,
        lists:zip(InputWeights, Inputs)
    ).

%% @private Apply LTC dynamics (CfC or ODE).
apply_ltc_dynamics(cfc, Input, State, Tau, Bound, BackboneWeights, HeadWeights) ->
    Params = #{
        backbone_weights => BackboneWeights,
        head_weights => HeadWeights
    },
    ltc_dynamics:evaluate_cfc(Input, State, Tau, Bound, Params);

apply_ltc_dynamics(ltc, Input, State, Tau, Bound, _BackboneWeights, _HeadWeights) ->
    Dt = 0.1,  %% Time step for ODE mode
    ltc_dynamics:evaluate_ode(Input, State, Tau, Bound, Dt).

%%% ============================================================================
%%% Internal Functions - Output Processing
%%% ============================================================================

%% @private Convert network outputs to parameter values.
outputs_to_params(Outputs, Config) ->
    OutputMapping = get(output_mapping),
    ParamBounds = Config#meta_config.param_bounds,

    maps:fold(
        fun(Index, ParamName, Acc) ->
            RawOutput = lists:nth(Index, Outputs),
            %% Apply sigmoid to bound output to [0, 1]
            BoundedOutput = sigmoid(RawOutput),
            %% Scale to parameter range
            {Min, Max} = maps:get(ParamName, ParamBounds, {0.0, 1.0}),
            ParamValue = Min + BoundedOutput * (Max - Min),
            Acc#{ParamName => ParamValue}
        end,
        #{},
        OutputMapping#output_mapping.index_to_param
    ).

%% @private Apply momentum smoothing to parameter changes.
apply_momentum(NewParams, State) ->
    Momentum = (State#meta_state.config)#meta_config.momentum,
    OldParams = State#meta_state.current_params,
    OldMomentum = State#meta_state.param_momentum,

    maps:fold(
        fun(Param, NewVal, Acc) ->
            OldVal = maps:get(Param, OldParams, NewVal),
            OldMom = maps:get(Param, OldMomentum, 0.0),

            %% Momentum update: v = momentum * v + (1 - momentum) * (new - old)
            Delta = NewVal - OldVal,
            NewMom = Momentum * OldMom + (1.0 - Momentum) * Delta,

            %% Apply momentum-smoothed change
            SmoothedVal = OldVal + NewMom,
            Acc#{Param => SmoothedVal}
        end,
        #{},
        NewParams
    ).

%% @private Update stagnation counter.
update_stagnation(Metrics, State) ->
    case Metrics#generation_metrics.relative_improvement of
        Imp when Imp > 0.01 ->
            %% Significant improvement, reset counter
            0;
        _ ->
            %% No improvement, increment counter
            State#meta_state.stagnation_count + 1
    end.

%% @private Log progress periodically.
maybe_log_progress(State, _TrainingEvent) ->
    case State#meta_state.generation rem 10 of
        0 ->
            Params = State#meta_state.current_params,
            error_logger:info_msg(
                "[meta_controller] Cohort ~p: mutation_rate=~.3f, mutation_strength=~.3f, "
                "selection_ratio=~.3f, reward=~.2f, stagnation=~p~n",
                [
                    State#meta_state.generation,
                    maps:get(mutation_rate, Params),
                    maps:get(mutation_strength, Params),
                    maps:get(selection_ratio, Params),
                    State#meta_state.cumulative_reward,
                    State#meta_state.stagnation_count
                ]
            );
        _ ->
            ok
    end.

%%% ============================================================================
%%% L2→L1 Guidance Functions
%%% ============================================================================

%% @private Process generation stats and compute L1 guidance.
%%
%% This is similar to process_generation but produces L1 guidance
%% (meta-parameters) instead of direct hyperparameters.
process_for_l1_guidance(GenStats, State) ->
    Config = State#meta_state.config,

    %% Convert generation stats to metrics
    Metrics = stats_to_metrics(GenStats, State),

    %% Update metrics history
    NewHistory = update_history(Metrics, State#meta_state.metrics_history, Config),

    %% Compute input features for LTC network
    Inputs = compute_input_features(Metrics, NewHistory, State),

    %% Forward pass through LTC network
    {Outputs, NewLtcStates} = forward_pass(Inputs, State),

    %% Convert outputs to L1 guidance (meta-parameters)
    Guidance = outputs_to_l1_guidance(Outputs, Config, State#meta_state.generation + 1),

    %% Compute reward for this generation
    Reward = meta_reward:compute(Metrics, NewHistory, Config),

    %% Update stagnation counter
    NewStagnation = update_stagnation(Metrics, State),

    NewState = State#meta_state{
        generation = State#meta_state.generation + 1,
        metrics_history = NewHistory,
        ltc_states = NewLtcStates,
        cumulative_reward = State#meta_state.cumulative_reward + Reward#meta_reward.total,
        best_fitness_ever = max(State#meta_state.best_fitness_ever, Metrics#generation_metrics.best_fitness),
        stagnation_count = NewStagnation
    },

    %% Log progress periodically
    maybe_log_l1_guidance(NewState, Guidance),

    {Guidance, NewState}.

%% @private Convert network outputs to L1 guidance parameters.
%%
%% The LTC network outputs 16 values controlling Task Silo and Resource Silo L1:
%%
%% Task Silo L1 (1-10):
%% 1. aggression_factor [0, 2]
%% 2. exploration_step [0.05, 0.5]
%% 3. stagnation_sensitivity [0.0001, 0.01]
%% 4. topology_aggression [1, 3]
%% 5. exploitation_weight [0.2, 0.8]
%% 6. adaptation_momentum [0, 0.95]
%% 7. warning_threshold [0.2, 0.5]
%% 8. intervention_threshold [0.4, 0.8]
%% 9. critical_threshold [0.7, 0.99]
%% 10. velocity_window_size [5, 30]
%%
%% Resource Silo L1 (11-16):
%% 11. memory_high_threshold [0.5, 0.85]
%% 12. memory_critical_threshold [0.8, 0.98]
%% 13. cpu_high_threshold [0.6, 0.95]
%% 14. pressure_scale_factor [0.5, 0.99]
%% 15. min_scale_factor [0.05, 0.3]
%% 16. pressure_change_threshold [0.01, 0.2]
outputs_to_l1_guidance(Outputs, _Config, Generation) ->
    %% Get bounded outputs (sigmoid already applied in forward pass for output neurons)
    %% We need at least 16 outputs for full L1 guidance
    [O1, O2, O3, O4, O5, O6, O7, O8, O9, O10, O11, O12, O13, O14, O15, O16 | _] =
        pad_outputs(Outputs, 16),

    %% Task Silo L1 hyperparameters
    AggressionFactor = scale_output(sigmoid(O1), 0.0, 2.0),
    ExplorationStep = scale_output(sigmoid(O2), 0.05, 0.5),
    StagnationSensitivity = scale_output(sigmoid(O3), 0.0001, 0.01),
    TopologyAggression = scale_output(sigmoid(O4), 1.0, 3.0),
    ExploitationWeight = scale_output(sigmoid(O5), 0.2, 0.8),
    AdaptationMomentum = scale_output(sigmoid(O6), 0.0, 0.95),
    WarningThreshold = scale_output(sigmoid(O7), 0.2, 0.5),
    InterventionThreshold = scale_output(sigmoid(O8), 0.4, 0.8),
    CriticalThreshold = scale_output(sigmoid(O9), 0.7, 0.99),
    VelocityWindowSize = round(scale_output(sigmoid(O10), 5.0, 30.0)),

    %% Resource Silo L1 hyperparameters
    MemoryHighThreshold = scale_output(sigmoid(O11), 0.5, 0.85),
    MemoryCriticalThreshold = scale_output(sigmoid(O12), 0.8, 0.98),
    CpuHighThreshold = scale_output(sigmoid(O13), 0.6, 0.95),
    PressureScaleFactor = scale_output(sigmoid(O14), 0.5, 0.99),
    MinScaleFactor = scale_output(sigmoid(O15), 0.05, 0.3),
    PressureChangeThreshold = scale_output(sigmoid(O16), 0.01, 0.2),

    #l2_guidance{
        %% Task Silo L1
        aggression_factor = AggressionFactor,
        exploration_step = ExplorationStep,
        stagnation_sensitivity = StagnationSensitivity,
        topology_aggression = TopologyAggression,
        exploitation_weight = ExploitationWeight,
        adaptation_momentum = AdaptationMomentum,
        warning_threshold = WarningThreshold,
        intervention_threshold = InterventionThreshold,
        critical_threshold = CriticalThreshold,
        velocity_window_size = VelocityWindowSize,
        %% Resource Silo L1
        memory_high_threshold = MemoryHighThreshold,
        memory_critical_threshold = MemoryCriticalThreshold,
        cpu_high_threshold = CpuHighThreshold,
        pressure_scale_factor = PressureScaleFactor,
        min_scale_factor = MinScaleFactor,
        pressure_change_threshold = PressureChangeThreshold,
        %% Metadata
        generation = Generation
    }.

%% @private Pad outputs list to minimum length.
pad_outputs(Outputs, MinLen) when length(Outputs) >= MinLen ->
    Outputs;
pad_outputs(Outputs, MinLen) ->
    Outputs ++ lists:duplicate(MinLen - length(Outputs), 0.0).

%% @private Scale output from [0,1] to [Min, Max].
scale_output(Value, Min, Max) ->
    Min + Value * (Max - Min).

%% @private Convert current state to L1 guidance (for monitoring).
state_to_l1_guidance(State) ->
    %% If we have current params, derive guidance from them
    %% Otherwise return defaults
    case State#meta_state.generation of
        0 ->
            ?L2_GUIDANCE_DEFAULTS;
        Gen ->
            %% Use the last computed guidance based on current state
            %% This is a simplified conversion - in practice, we'd store
            %% the last guidance in state
            Params = State#meta_state.current_params,

            %% Estimate guidance from current params
            %% (This is approximate - we map back from hyperparams to meta-params)
            MR = maps:get(mutation_rate, Params, 0.10),
            BaseMR = 0.10,
            EstAggression = if
                MR > BaseMR -> min(2.0, (MR / BaseMR - 1.0));
                true -> 0.5
            end,

            %% Use defaults with estimated aggression
            Defaults = ?L2_GUIDANCE_DEFAULTS,
            Defaults#l2_guidance{
                aggression_factor = EstAggression,
                generation = Gen
            }
    end.

%% @private Log L1 guidance periodically.
maybe_log_l1_guidance(State, Guidance) ->
    case State#meta_state.generation rem 10 of
        0 ->
            error_logger:info_msg(
                "[meta_controller] Cohort ~p L1 Guidance: aggression=~.2f, "
                "exploration_step=~.2f, topology_aggression=~.2f~n",
                [
                    State#meta_state.generation,
                    Guidance#l2_guidance.aggression_factor,
                    Guidance#l2_guidance.exploration_step,
                    Guidance#l2_guidance.topology_aggression
                ]
            );
        _ ->
            ok
    end.

%%% ============================================================================
%%% Utility Functions
%%% ============================================================================

%% @private Sigmoid function.
sigmoid(X) ->
    V = clamp(X, -10.0, 10.0),
    1.0 / (1.0 + math:exp(-V)).

%% @private Clamp value to range.
clamp(Val, Min, _Max) when Val < Min -> Min;
clamp(Val, _Min, Max) when Val > Max -> Max;
clamp(Val, _Min, _Max) -> Val.
