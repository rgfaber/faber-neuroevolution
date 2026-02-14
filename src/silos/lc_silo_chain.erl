%% @doc LC Silo Chain - Manages L0/L1/L2 LTC TWEANNs for a single silo.
%%
%% Each silo (Task, Resource, Distribution) has 3 chained LTC TWEANNs
%% that operate at different time scales. Training metrics flow to all
%% levels, while outputs cascade down: L2 outputs feed into L1 inputs,
%% and L1 outputs feed into L0 inputs.
%%
%% See `assets/lc_silo_chain.svg' for the architecture diagram.
%%
%% == Data Flow ==
%%
%% 1. Training metrics from NuT are fed to ALL levels
%% 2. L2 outputs become PART OF L1's inputs (L1 hyperparameters)
%% 3. L1 outputs become PART OF L0's inputs (L0 hyperparameters)
%% 4. L0 outputs are the actual hyperparameters applied to evolution
%%
%% == Time Constants ==
%%
%% Each level operates at a different time scale:
%% - L2: Slow (strategic) - updates every τ_L2 (e.g., 100 generations)
%% - L1: Medium (tactical) - updates every τ_L1 (e.g., 50 generations)
%% - L0: Fast (reactive) - updates every τ_L0 (e.g., 10 generations)
%%
%% @author R.G. Lefever
%% @copyright 2024-2026 R.G. Lefever
-module(lc_silo_chain).
-behaviour(gen_server).

%% API
-export([
    start_link/2,
    get_recommendations/2,
    report_reward/2,
    get_state/1
]).

%% gen_server callbacks
-export([init/1, handle_call/3, handle_cast/2, handle_info/2, terminate/2]).

-record(state, {
    %% Silo identity
    silo_type :: task | resource | distribution,
    name :: atom(),

    %% The 3 LTC TWEANNs (stored as in-memory networks, not separate processes)
    l0_network :: network(),
    l1_network :: network(),
    l2_network :: network(),

    %% Time constants (in evaluation counts or time units)
    tau_l0 :: pos_integer(),
    tau_l1 :: pos_integer(),
    tau_l2 :: pos_integer(),

    %% Update counters
    l0_counter :: non_neg_integer(),
    l1_counter :: non_neg_integer(),
    l2_counter :: non_neg_integer(),

    %% Cached outputs from each level (for chaining)
    l2_outputs :: [float()],
    l1_outputs :: [float()],
    l0_outputs :: [float()],

    %% Output ranges for final hyperparameters
    output_ranges :: #{atom() => {float(), float()}},

    %% Learning parameters
    learning_rate :: float(),
    noise_std :: float(),
    noise_decay :: float(),

    %% Reward tracking per level
    l0_reward_history :: [float()],
    l1_reward_history :: [float()],
    l2_reward_history :: [float()],

    %% Rate limiting to prevent message queue buildup
    last_reward_time :: integer(),  %% erlang:monotonic_time(millisecond)
    reward_throttle_ms :: pos_integer()  %% Minimum ms between reward updates
}).

%% Network record - in-memory TWEANN
-record(network, {
    input_size :: pos_integer(),
    hidden_sizes :: [pos_integer()],
    output_size :: pos_integer(),
    layers :: [layer()],
    %% For ES-style learning
    candidate_layers :: [layer()] | undefined,
    candidate_reward :: float()
}).

-type layer() :: #{weights := [[float()]], biases := [float()]}.
-type network() :: #network{}.

%%% ============================================================================
%%% API Functions
%%% ============================================================================

%% @doc Start the silo chain with configuration.
%%
%% Config options:
%% - silo_type: task | resource | distribution
%% - tau_l0, tau_l1, tau_l2: time constants for each level
%% - l0_topology, l1_topology, l2_topology: {InputSize, HiddenSizes, OutputSize}
%% - output_ranges: map of output name to {min, max}
-spec start_link(atom(), map()) -> {ok, pid()} | {error, term()}.
start_link(Name, Config) ->
    gen_server:start_link({local, Name}, ?MODULE, Config#{name => Name}, []).

%% @doc Get hyperparameter recommendations.
%%
%% TrainingMetrics is a map containing:
%% - best_fitness, avg_fitness, fitness_variance
%% - improvement_velocity, convergence_trend
%% - stagnation_severity, diversity_index
%% - memory_pressure, cpu_pressure
%% - etc.
%%
%% Returns a map of hyperparameter name => value.
-spec get_recommendations(atom() | pid(), map()) -> map().
get_recommendations(Chain, TrainingMetrics) ->
    gen_server:call(Chain, {get_recommendations, TrainingMetrics}).

%% @doc Report reward signal for learning.
%%
%% The reward is distributed to all levels based on their contribution.
-spec report_reward(atom() | pid(), float()) -> ok.
report_reward(Chain, Reward) ->
    gen_server:cast(Chain, {report_reward, Reward}).

%% @doc Get current state for debugging/monitoring.
-spec get_state(atom() | pid()) -> map().
get_state(Chain) ->
    gen_server:call(Chain, get_state).

%%% ============================================================================
%%% gen_server Callbacks
%%% ============================================================================

init(Config) ->
    SiloType = maps:get(silo_type, Config, task),
    Name = maps:get(name, Config, lc_silo_chain),

    %% Time constants
    TauL0 = maps:get(tau_l0, Config, 10),
    TauL1 = maps:get(tau_l1, Config, 50),
    TauL2 = maps:get(tau_l2, Config, 100),

    %% Get topologies for each level
    %% Default: Task silo with 21 training metric inputs
    {L0In, L0Hidden, L0Out} = maps:get(l0_topology, Config, {21 + 8, [32, 16], 16}),
    {L1In, L1Hidden, L1Out} = maps:get(l1_topology, Config, {21 + 5, [24, 12], 8}),
    {L2In, L2Hidden, L2Out} = maps:get(l2_topology, Config, {21, [16, 8], 5}),

    %% Initialize networks
    L0Network = init_network(L0In, L0Hidden, L0Out),
    L1Network = init_network(L1In, L1Hidden, L1Out),
    L2Network = init_network(L2In, L2Hidden, L2Out),

    %% Learning parameters
    LearningRate = maps:get(learning_rate, Config, 0.1),
    NoiseStd = maps:get(noise_std, Config, 0.1),
    NoiseDecay = maps:get(noise_decay, Config, 0.999),

    %% Rate limiting - minimum 100ms between reward updates to prevent queue buildup
    RewardThrottleMs = maps:get(reward_throttle_ms, Config, 100),

    %% Output ranges
    OutputRanges = maps:get(output_ranges, Config, default_task_output_ranges()),

    State = #state{
        silo_type = SiloType,
        name = Name,
        l0_network = L0Network,
        l1_network = L1Network,
        l2_network = L2Network,
        tau_l0 = TauL0,
        tau_l1 = TauL1,
        tau_l2 = TauL2,
        l0_counter = 0,
        l1_counter = 0,
        l2_counter = 0,
        l2_outputs = lists:duplicate(L2Out, 0.5),
        l1_outputs = lists:duplicate(L1Out, 0.5),
        l0_outputs = lists:duplicate(L0Out, 0.5),
        output_ranges = OutputRanges,
        learning_rate = LearningRate,
        noise_std = NoiseStd,
        noise_decay = NoiseDecay,
        l0_reward_history = [],
        l1_reward_history = [],
        l2_reward_history = [],
        last_reward_time = 0,
        reward_throttle_ms = RewardThrottleMs
    },

    error_logger:info_msg("[lc_silo_chain] Started ~p for ~p silo~n"
                         "  L2: ~p -> ~p -> ~p~n"
                         "  L1: ~p -> ~p -> ~p~n"
                         "  L0: ~p -> ~p -> ~p~n",
                         [Name, SiloType,
                          L2In, L2Hidden, L2Out,
                          L1In, L1Hidden, L1Out,
                          L0In, L0Hidden, L0Out]),

    %% Subscribe to reward events for this silo (event-driven pattern)
    controller_events:subscribe_to_reward(),

    {ok, State}.

handle_call({get_recommendations, TrainingMetrics}, _From, State) ->
    %% Increment counters
    NewL0Counter = State#state.l0_counter + 1,
    NewL1Counter = State#state.l1_counter + 1,
    NewL2Counter = State#state.l2_counter + 1,

    %% Convert training metrics to input vector
    MetricsVector = training_metrics_to_vector(TrainingMetrics),

    %% === L2 Forward Pass ===
    %% L2 only sees training metrics (strategic, long-term view)
    L2Outputs = case NewL2Counter >= State#state.tau_l2 of
        true ->
            forward_pass(MetricsVector, State#state.l2_network);
        false ->
            State#state.l2_outputs
    end,

    %% === L1 Forward Pass ===
    %% L1 sees: training metrics + L2 outputs
    L1Inputs = MetricsVector ++ L2Outputs,
    L1Outputs = case NewL1Counter >= State#state.tau_l1 of
        true ->
            forward_pass(L1Inputs, State#state.l1_network);
        false ->
            State#state.l1_outputs
    end,

    %% === L0 Forward Pass ===
    %% L0 sees: training metrics + L1 outputs
    L0Inputs = MetricsVector ++ L1Outputs,
    L0Outputs = forward_pass(L0Inputs, State#state.l0_network),

    %% Scale L0 outputs to actual hyperparameter ranges
    Recommendations = scale_outputs(L0Outputs, State#state.output_ranges),

    %% Reset counters if they reached tau
    NewState = State#state{
        l0_counter = NewL0Counter,
        l1_counter = case NewL1Counter >= State#state.tau_l1 of true -> 0; false -> NewL1Counter end,
        l2_counter = case NewL2Counter >= State#state.tau_l2 of true -> 0; false -> NewL2Counter end,
        l2_outputs = L2Outputs,
        l1_outputs = L1Outputs,
        l0_outputs = L0Outputs
    },

    {reply, Recommendations, NewState};

handle_call(get_state, _From, State) ->
    Info = #{
        silo_type => State#state.silo_type,
        counters => #{l0 => State#state.l0_counter, l1 => State#state.l1_counter, l2 => State#state.l2_counter},
        tau => #{l0 => State#state.tau_l0, l1 => State#state.tau_l1, l2 => State#state.tau_l2},
        noise_std => State#state.noise_std,
        l0_avg_reward => avg_list(State#state.l0_reward_history),
        l1_avg_reward => avg_list(State#state.l1_reward_history),
        l2_avg_reward => avg_list(State#state.l2_reward_history)
    },
    {reply, Info, State};

handle_call(_Request, _From, State) ->
    {reply, {error, unknown_request}, State}.

handle_cast({report_reward, Reward}, State) ->
    %% DEPRECATED: Use controller_events:publish_reward/2 instead
    logger:warning("[DEPRECATED] lc_silo_chain:report_reward/2 called. "
                   "Use controller_events:publish_reward/2 instead."),
    NewState = do_report_reward(Reward, State),
    {noreply, NewState};

handle_cast(_Msg, State) ->
    {noreply, State}.

%% Handle reward signal event (event-driven pattern)
handle_info({neuro_event, <<"controller.reward">>, #{silo := Silo, reward := Reward}}, State) ->
    %% Only process if reward is for our silo or global
    case Silo =:= State#state.silo_type orelse Silo =:= global of
        true ->
            NewState = do_report_reward(Reward, State),
            {noreply, NewState};
        false ->
            {noreply, State}
    end;

handle_info(_Info, State) ->
    {noreply, State}.

terminate(_Reason, _State) ->
    ok.

%%% ============================================================================
%%% Internal Functions - Reward Processing
%%% ============================================================================

%% @private Process reward signal (shared by cast and event handlers).
do_report_reward(Reward, State) ->
    %% Rate limiting to prevent message queue buildup
    Now = erlang:monotonic_time(millisecond),
    TimeSinceLastReward = Now - State#state.last_reward_time,

    case TimeSinceLastReward >= State#state.reward_throttle_ms of
        true ->
            %% Enough time has passed, process the reward
            AvgL0 = avg_list(State#state.l0_reward_history),
            error_logger:info_msg("[lc_silo_chain] ~p received reward=~.3f (avg_l0=~.3f, history_len=~p)~n",
                                  [State#state.silo_type, Reward, AvgL0, length(State#state.l0_reward_history)]),
            L0History = [Reward | lists:sublist(State#state.l0_reward_history, 99)],
            L1History = [Reward | lists:sublist(State#state.l1_reward_history, 99)],
            L2History = [Reward | lists:sublist(State#state.l2_reward_history, 99)],

            %% Maybe update weights (ES-style learning)
            NewState0 = maybe_update_network(l0, State#state{l0_reward_history = L0History}),
            NewState1 = maybe_update_network(l1, NewState0#state{l1_reward_history = L1History}),
            NewState2 = maybe_update_network(l2, NewState1#state{l2_reward_history = L2History}),

            NewState2#state{last_reward_time = Now};
        false ->
            %% Throttled - skip this reward update
            State
    end.

%%% ============================================================================
%%% Internal Functions - Network Operations
%%% ============================================================================

%% @private Initialize a network with random weights.
-spec init_network(pos_integer(), [pos_integer()], pos_integer()) -> network().
init_network(InputSize, HiddenSizes, OutputSize) ->
    Layers = init_layers(InputSize, HiddenSizes, OutputSize),
    #network{
        input_size = InputSize,
        hidden_sizes = HiddenSizes,
        output_size = OutputSize,
        layers = Layers,
        candidate_layers = undefined,
        candidate_reward = -999.0
    }.

%% @private Initialize layer weights.
init_layers(InputSize, HiddenSizes, OutputSize) ->
    AllSizes = [InputSize | HiddenSizes] ++ [OutputSize],
    init_layers_acc(AllSizes, []).

init_layers_acc([_], Acc) ->
    lists:reverse(Acc);
init_layers_acc([InSize, OutSize | Rest], Acc) ->
    Layer = #{
        weights => random_matrix(OutSize, InSize),
        biases => lists:duplicate(OutSize, 0.0)
    },
    init_layers_acc([OutSize | Rest], [Layer | Acc]).

%% @private Random weight matrix (Xavier initialization).
random_matrix(Rows, Cols) ->
    Scale = math:sqrt(2.0 / (Rows + Cols)),
    [[rand:normal() * Scale || _ <- lists:seq(1, Cols)] || _ <- lists:seq(1, Rows)].

%% @private Forward pass through network.
-spec forward_pass([float()], network()) -> [float()].
forward_pass(Input, #network{layers = Layers}) ->
    forward_pass_layers(Input, Layers).

forward_pass_layers(Input, []) ->
    Input;
forward_pass_layers(Input, [Layer]) ->
    %% Last layer: sigmoid activation for 0-1 outputs
    Weights = maps:get(weights, Layer),
    Biases = maps:get(biases, Layer),
    Linear = vec_add(mat_vec_mul(Weights, Input), Biases),
    [sigmoid(X) || X <- Linear];
forward_pass_layers(Input, [Layer | Rest]) ->
    %% Hidden layers: tanh activation
    Weights = maps:get(weights, Layer),
    Biases = maps:get(biases, Layer),
    Linear = vec_add(mat_vec_mul(Weights, Input), Biases),
    Activated = [math:tanh(X) || X <- Linear],
    forward_pass_layers(Activated, Rest).

%% @private Matrix-vector multiplication.
mat_vec_mul(Matrix, Vector) ->
    [dot_product(Row, Vector) || Row <- Matrix].

dot_product(V1, V2) ->
    lists:sum([A * B || {A, B} <- lists:zip(V1, V2)]).

vec_add(V1, V2) ->
    [A + B || {A, B} <- lists:zip(V1, V2)].

sigmoid(X) ->
    1.0 / (1.0 + math:exp(-X)).

%%% ============================================================================
%%% Internal Functions - Learning (ES-style)
%%% ============================================================================

%% @private Maybe update network weights based on reward.
maybe_update_network(Level, State) ->
    History = case Level of
        l0 -> State#state.l0_reward_history;
        l1 -> State#state.l1_reward_history;
        l2 -> State#state.l2_reward_history
    end,
    Tau = case Level of
        l0 -> State#state.tau_l0;
        l1 -> State#state.tau_l1;
        l2 -> State#state.tau_l2
    end,

    %% Only update at tau intervals
    Counter = case Level of
        l0 -> State#state.l0_counter;
        l1 -> State#state.l1_counter;
        l2 -> State#state.l2_counter
    end,

    case Counter rem Tau =:= 0 andalso length(History) >= 10 of
        true ->
            Network = case Level of
                l0 -> State#state.l0_network;
                l1 -> State#state.l1_network;
                l2 -> State#state.l2_network
            end,
            CurrentReward = avg_list(History),
            UpdatedNetwork = es_update(Network, CurrentReward, State#state.noise_std),
            case Level of
                l0 -> State#state{l0_network = UpdatedNetwork};
                l1 -> State#state{l1_network = UpdatedNetwork};
                l2 -> State#state{l2_network = UpdatedNetwork}
            end;
        false ->
            State
    end.

%% @private ES-style weight update.
es_update(#network{candidate_layers = undefined} = Network, CurrentReward, NoiseStd) ->
    %% No candidate yet, create one
    Candidate = perturb_layers(Network#network.layers, NoiseStd),
    Network#network{
        candidate_layers = Candidate,
        candidate_reward = CurrentReward
    };
es_update(Network, CurrentReward, NoiseStd) ->
    case CurrentReward > Network#network.candidate_reward of
        true ->
            %% Improvement! Keep current weights
            NewCandidate = perturb_layers(Network#network.layers, NoiseStd),
            Network#network{
                candidate_layers = NewCandidate,
                candidate_reward = CurrentReward
            };
        false ->
            %% Try candidate weights
            NewCandidate = perturb_layers(Network#network.candidate_layers, NoiseStd),
            Network#network{
                layers = Network#network.candidate_layers,
                candidate_layers = NewCandidate,
                candidate_reward = CurrentReward
            }
    end.

%% @private Perturb layer weights with Gaussian noise.
perturb_layers(Layers, NoiseStd) ->
    [perturb_layer(L, NoiseStd) || L <- Layers].

perturb_layer(Layer, NoiseStd) ->
    Weights = maps:get(weights, Layer),
    Biases = maps:get(biases, Layer),
    #{
        weights => [[W + rand:normal() * NoiseStd || W <- Row] || Row <- Weights],
        biases => [B + rand:normal() * NoiseStd || B <- Biases]
    }.

%%% ============================================================================
%%% Internal Functions - Input/Output Processing
%%% ============================================================================

%% @private Convert training metrics map to ordered vector.
training_metrics_to_vector(Metrics) ->
    %% Standard order for training metrics (21 inputs)
    Order = [
        best_fitness, avg_fitness, fitness_variance, improvement_velocity,
        stagnation_severity, diversity_index, species_count_ratio,
        avg_network_complexity, complexity_velocity, elite_dominance,
        crossover_success_rate, mutation_impact, resource_pressure_signal,
        evaluation_progress, entropy, convergence_trend,
        archive_fill_ratio, archive_fitness_mean, archive_fitness_variance,
        archive_staleness, population_vs_archive_ratio
    ],
    [maps:get(Name, Metrics, 0.5) || Name <- Order].

%% @private Scale network outputs to hyperparameter ranges.
scale_outputs(Outputs, Ranges) ->
    OutputOrder = [
        mutation_rate, mutation_strength, selection_ratio,
        add_node_rate, add_connection_rate, delete_connection_rate,
        weight_perturb_vs_replace, crossover_rate, interspecies_crossover_rate,
        elitism_count, population_size_delta, compatibility_threshold_delta,
        archive_threshold_percentile, archive_sampling_temperature,
        archive_prune_ratio, archive_max_size_delta
    ],
    Pairs = lists:zip(OutputOrder, Outputs),
    maps:from_list([
        {Name, scale_value(Value, maps:get(Name, Ranges, {0.0, 1.0}))}
        || {Name, Value} <- Pairs
    ]).

scale_value(Value, {Min, Max}) ->
    Min + Value * (Max - Min).

%% @private Default output ranges for task silo.
default_task_output_ranges() ->
    #{
        mutation_rate => {0.01, 0.50},
        mutation_strength => {0.05, 1.0},
        selection_ratio => {0.05, 0.50},
        add_node_rate => {0.0, 0.15},
        add_connection_rate => {0.0, 0.25},
        delete_connection_rate => {0.0, 0.10},
        weight_perturb_vs_replace => {0.5, 1.0},
        crossover_rate => {0.0, 0.9},
        interspecies_crossover_rate => {0.0, 0.3},
        elitism_count => {1.0, 10.0},
        population_size_delta => {-10.0, 10.0},
        compatibility_threshold_delta => {-0.5, 0.5},
        archive_threshold_percentile => {0.3, 0.95},
        archive_sampling_temperature => {0.0, 1.0},
        archive_prune_ratio => {0.5, 1.0},
        archive_max_size_delta => {-5.0, 5.0}
    }.

%% @private Average of a list.
avg_list([]) -> 0.0;
avg_list(List) -> lists:sum(List) / length(List).
