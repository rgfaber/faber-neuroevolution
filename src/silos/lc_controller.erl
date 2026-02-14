%% @doc LC Controller - Simple online-learning TWEANN for hyperparameter control.
%%
%% Unlike the main TWEANN population which evolves competitively, LC controllers
%% are single networks that learn "on the job" through online weight updates.
%%
%% Architecture:
%% - Input layer: Sensor values (21 for task silo, 15 for resource silo)
%% - Hidden layers: Configurable (default: [32, 16])
%% - Output layer: Actuator values (16 for task silo, 9 for resource silo)
%%
%% Learning: Evolution Strategies (ES) style weight perturbation
%% - Perturb weights with Gaussian noise
%% - Keep perturbation if reward improves
%% - Decay noise over time as performance stabilizes
%%
%% @author R.G. Lefever
%% @copyright 2024-2026 R.G. Lefever
-module(lc_controller).
-behaviour(gen_server).

%% API
-export([
    start_link/1,
    start_link/2,
    get_recommendations/2,
    report_reward/2,
    get_state/1
]).

%% gen_server callbacks
-export([init/1, handle_call/3, handle_cast/2, handle_info/2, terminate/2]).

-define(DEFAULT_HIDDEN_LAYERS, [32, 16]).
-define(DEFAULT_LEARNING_RATE, 0.1).
-define(DEFAULT_NOISE_STD, 0.1).
-define(DEFAULT_NOISE_DECAY, 0.999).
-define(MIN_NOISE_STD, 0.01).
-define(UPDATE_INTERVAL, 100).  %% Update weights every N reward reports

-record(state, {
    %% Network topology
    input_size :: pos_integer(),
    hidden_sizes :: [pos_integer()],
    output_size :: pos_integer(),

    %% Network weights (list of layer weight matrices)
    %% Each layer is a map: #{weights => [[float()]], biases => [float()]}
    layers :: [map()],

    %% Learning state
    learning_rate :: float(),
    noise_std :: float(),
    noise_decay :: float(),

    %% Reward tracking
    current_reward :: float(),
    reward_history :: [float()],
    reward_count :: non_neg_integer(),

    %% Candidate weights for ES
    candidate_layers :: [map()] | undefined,
    candidate_reward :: float(),

    %% Output scaling
    output_ranges :: #{atom() => {float(), float()}}
}).

%%% ============================================================================
%%% API Functions
%%% ============================================================================

%% @doc Start the LC controller with default options.
-spec start_link(atom()) -> {ok, pid()} | ignore | {error, term()}.
start_link(Name) ->
    start_link(Name, #{}).

%% @doc Start the LC controller with custom configuration.
%%
%% Config options:
%% - input_size: Number of sensor inputs (default: 21)
%% - output_size: Number of actuator outputs (default: 16)
%% - hidden_sizes: List of hidden layer sizes (default: [32, 16])
%% - learning_rate: Weight update rate (default: 0.1)
%% - noise_std: Initial noise standard deviation (default: 0.1)
%% - output_ranges: Map of output name to {min, max} ranges
-spec start_link(atom(), map()) -> {ok, pid()} | ignore | {error, term()}.
start_link(Name, Config) ->
    gen_server:start_link({local, Name}, ?MODULE, Config, []).

%% @doc Get hyperparameter recommendations from the controller.
%%
%% SensorInputs is a map of sensor name => value (all normalized to 0-1 or -1 to 1).
%% Returns a map of actuator name => value (scaled to actual ranges).
-spec get_recommendations(atom() | pid(), map()) -> map().
get_recommendations(Controller, SensorInputs) ->
    gen_server:call(Controller, {get_recommendations, SensorInputs}).

%% @doc Report reward signal to the controller for learning.
%%
%% The reward should be in range [-1, 1].
-spec report_reward(atom() | pid(), float()) -> ok.
report_reward(Controller, Reward) ->
    gen_server:cast(Controller, {report_reward, Reward}).

%% @doc Get current controller state for debugging/monitoring.
-spec get_state(atom() | pid()) -> map().
get_state(Controller) ->
    gen_server:call(Controller, get_state).

%%% ============================================================================
%%% gen_server Callbacks
%%% ============================================================================

init(Config) ->
    %% Get topology from config
    InputSize = maps:get(input_size, Config, 21),
    OutputSize = maps:get(output_size, Config, 16),
    HiddenSizes = maps:get(hidden_sizes, Config, ?DEFAULT_HIDDEN_LAYERS),

    %% Learning parameters
    LearningRate = maps:get(learning_rate, Config, ?DEFAULT_LEARNING_RATE),
    NoiseStd = maps:get(noise_std, Config, ?DEFAULT_NOISE_STD),
    NoiseDecay = maps:get(noise_decay, Config, ?DEFAULT_NOISE_DECAY),

    %% Output ranges (for scaling network outputs to actual hyperparameter ranges)
    OutputRanges = maps:get(output_ranges, Config, default_output_ranges()),

    %% Initialize network weights randomly
    Layers = initialize_layers(InputSize, HiddenSizes, OutputSize),

    State = #state{
        input_size = InputSize,
        hidden_sizes = HiddenSizes,
        output_size = OutputSize,
        layers = Layers,
        learning_rate = LearningRate,
        noise_std = NoiseStd,
        noise_decay = NoiseDecay,
        current_reward = 0.0,
        reward_history = [],
        reward_count = 0,
        candidate_layers = undefined,
        candidate_reward = -999.0,
        output_ranges = OutputRanges
    },

    error_logger:info_msg("[lc_controller] Started with topology ~p -> ~p -> ~p~n",
                         [InputSize, HiddenSizes, OutputSize]),

    {ok, State}.

handle_call({get_recommendations, SensorInputs}, _From, State) ->
    %% Convert sensor map to input vector
    InputVector = sensor_map_to_vector(SensorInputs),

    %% Forward pass through network
    OutputVector = forward_pass(InputVector, State#state.layers),

    %% Scale outputs to hyperparameter ranges
    Recommendations = scale_outputs(OutputVector, State#state.output_ranges),

    {reply, Recommendations, State};

handle_call(get_state, _From, State) ->
    Info = #{
        topology => {State#state.input_size, State#state.hidden_sizes, State#state.output_size},
        noise_std => State#state.noise_std,
        current_reward => State#state.current_reward,
        reward_count => State#state.reward_count,
        avg_recent_reward => avg_recent_reward(State#state.reward_history)
    },
    {reply, Info, State};

handle_call(_Request, _From, State) ->
    {reply, {error, unknown_request}, State}.

handle_cast({report_reward, Reward}, State) ->
    %% Update reward tracking
    NewHistory = [Reward | lists:sublist(State#state.reward_history, 99)],
    NewCount = State#state.reward_count + 1,

    %% ES-style learning: periodically try weight perturbations
    NewState = case NewCount rem ?UPDATE_INTERVAL of
        0 ->
            maybe_update_weights(State#state{
                reward_history = NewHistory,
                reward_count = NewCount,
                current_reward = avg_recent_reward(NewHistory)
            });
        _ ->
            State#state{
                reward_history = NewHistory,
                reward_count = NewCount
            }
    end,

    {noreply, NewState};

handle_cast(_Msg, State) ->
    {noreply, State}.

handle_info(_Info, State) ->
    {noreply, State}.

terminate(_Reason, _State) ->
    ok.

%%% ============================================================================
%%% Internal Functions - Network Operations
%%% ============================================================================

%% @private Initialize network layers with random weights.
-spec initialize_layers(pos_integer(), [pos_integer()], pos_integer()) -> [map()].
initialize_layers(InputSize, HiddenSizes, OutputSize) ->
    AllSizes = [InputSize | HiddenSizes] ++ [OutputSize],
    initialize_layers_acc(AllSizes, []).

initialize_layers_acc([_], Acc) ->
    lists:reverse(Acc);
initialize_layers_acc([InSize, OutSize | Rest], Acc) ->
    Layer = #{
        weights => random_matrix(OutSize, InSize),
        biases => random_vector(OutSize)
    },
    initialize_layers_acc([OutSize | Rest], [Layer | Acc]).

%% @private Generate random weight matrix using Xavier initialization.
-spec random_matrix(pos_integer(), pos_integer()) -> [[float()]].
random_matrix(Rows, Cols) ->
    Scale = math:sqrt(2.0 / (Rows + Cols)),
    [[rand:normal() * Scale || _ <- lists:seq(1, Cols)] || _ <- lists:seq(1, Rows)].

%% @private Generate random bias vector (initialized to zero).
-spec random_vector(pos_integer()) -> [float()].
random_vector(Size) ->
    [0.0 || _ <- lists:seq(1, Size)].

%% @private Forward pass through the network.
-spec forward_pass([float()], [map()]) -> [float()].
forward_pass(Input, Layers) ->
    lists:foldl(
        fun(Layer, Activations) ->
            Weights = maps:get(weights, Layer),
            Biases = maps:get(biases, Layer),
            %% Linear transformation: W * x + b
            Linear = mat_vec_add(mat_vec_mul(Weights, Activations), Biases),
            %% Apply activation (tanh for hidden, sigmoid for output)
            [tanh(X) || X <- Linear]
        end,
        Input,
        lists:droplast(Layers)
    ),
    %% Final layer uses sigmoid for 0-1 outputs
    LastLayer = lists:last(Layers),
    Weights = maps:get(weights, LastLayer),
    Biases = maps:get(biases, LastLayer),
    Linear = mat_vec_add(mat_vec_mul(Weights, forward_pass_hidden(Input, lists:droplast(Layers))), Biases),
    [sigmoid(X) || X <- Linear].

%% @private Forward pass through hidden layers only.
forward_pass_hidden(Input, []) ->
    Input;
forward_pass_hidden(Input, Layers) ->
    lists:foldl(
        fun(Layer, Activations) ->
            Weights = maps:get(weights, Layer),
            Biases = maps:get(biases, Layer),
            Linear = mat_vec_add(mat_vec_mul(Weights, Activations), Biases),
            [tanh(X) || X <- Linear]
        end,
        Input,
        Layers
    ).

%% @private Matrix-vector multiplication.
-spec mat_vec_mul([[float()]], [float()]) -> [float()].
mat_vec_mul(Matrix, Vector) ->
    [dot_product(Row, Vector) || Row <- Matrix].

%% @private Dot product of two vectors.
-spec dot_product([float()], [float()]) -> float().
dot_product(V1, V2) ->
    lists:sum([A * B || {A, B} <- lists:zip(V1, V2)]).

%% @private Vector addition.
-spec mat_vec_add([float()], [float()]) -> [float()].
mat_vec_add(V1, V2) ->
    [A + B || {A, B} <- lists:zip(V1, V2)].

%% @private Tanh activation.
-spec tanh(float()) -> float().
tanh(X) ->
    math:tanh(X).

%% @private Sigmoid activation.
-spec sigmoid(float()) -> float().
sigmoid(X) ->
    1.0 / (1.0 + math:exp(-X)).

%%% ============================================================================
%%% Internal Functions - Learning (ES-style)
%%% ============================================================================

%% @private Maybe update weights based on reward comparison.
-spec maybe_update_weights(#state{}) -> #state{}.
maybe_update_weights(#state{candidate_layers = undefined} = State) ->
    %% No candidate yet, create one by perturbing current weights
    Candidate = perturb_layers(State#state.layers, State#state.noise_std),
    State#state{
        candidate_layers = Candidate,
        candidate_reward = State#state.current_reward
    };
maybe_update_weights(State) ->
    %% Compare current reward with candidate's baseline reward
    CurrentReward = State#state.current_reward,
    CandidateBaseline = State#state.candidate_reward,

    case CurrentReward > CandidateBaseline of
        true ->
            %% Improvement! Keep current weights, decay noise
            NewNoiseStd = max(?MIN_NOISE_STD, State#state.noise_std * State#state.noise_decay),
            error_logger:info_msg("[lc_controller] Reward improved: ~.4f -> ~.4f, noise=~.4f~n",
                                 [CandidateBaseline, CurrentReward, NewNoiseStd]),
            %% Create new candidate from current (improved) weights
            NewCandidate = perturb_layers(State#state.layers, NewNoiseStd),
            State#state{
                noise_std = NewNoiseStd,
                candidate_layers = NewCandidate,
                candidate_reward = CurrentReward
            };
        false ->
            %% No improvement, try the candidate weights
            error_logger:info_msg("[lc_controller] Trying candidate weights (reward ~.4f vs ~.4f)~n",
                                 [CurrentReward, CandidateBaseline]),
            %% Swap to candidate weights
            NewCandidate = perturb_layers(State#state.candidate_layers, State#state.noise_std),
            State#state{
                layers = State#state.candidate_layers,
                candidate_layers = NewCandidate,
                candidate_reward = CurrentReward
            }
    end.

%% @private Perturb all layer weights with Gaussian noise.
-spec perturb_layers([map()], float()) -> [map()].
perturb_layers(Layers, NoiseStd) ->
    [perturb_layer(Layer, NoiseStd) || Layer <- Layers].

%% @private Perturb a single layer's weights and biases.
-spec perturb_layer(map(), float()) -> map().
perturb_layer(Layer, NoiseStd) ->
    Weights = maps:get(weights, Layer),
    Biases = maps:get(biases, Layer),
    #{
        weights => [[W + rand:normal() * NoiseStd || W <- Row] || Row <- Weights],
        biases => [B + rand:normal() * NoiseStd || B <- Biases]
    }.

%% @private Calculate average of recent rewards.
-spec avg_recent_reward([float()]) -> float().
avg_recent_reward([]) -> 0.0;
avg_recent_reward(History) ->
    lists:sum(History) / length(History).

%%% ============================================================================
%%% Internal Functions - Input/Output Processing
%%% ============================================================================

%% @private Convert sensor map to ordered input vector.
-spec sensor_map_to_vector(map()) -> [float()].
sensor_map_to_vector(SensorMap) ->
    %% Expected sensor order for task silo (21 inputs)
    SensorOrder = [
        best_fitness, avg_fitness, fitness_variance, improvement_velocity,
        stagnation_severity, diversity_index, species_count_ratio,
        avg_network_complexity, complexity_velocity, elite_dominance,
        crossover_success_rate, mutation_impact, resource_pressure_signal,
        evaluation_progress, entropy, convergence_trend,
        archive_fill_ratio, archive_fitness_mean, archive_fitness_variance,
        archive_staleness, population_vs_archive_ratio
    ],
    [maps:get(Name, SensorMap, 0.5) || Name <- SensorOrder].

%% @private Scale network outputs (0-1) to actual hyperparameter ranges.
-spec scale_outputs([float()], #{atom() => {float(), float()}}) -> map().
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

%% @private Scale a 0-1 value to a target range.
-spec scale_value(float(), {float(), float()}) -> float().
scale_value(Value, {Min, Max}) ->
    Min + Value * (Max - Min).

%% @private Default output ranges for task silo actuators.
-spec default_output_ranges() -> #{atom() => {float(), float()}}.
default_output_ranges() ->
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
