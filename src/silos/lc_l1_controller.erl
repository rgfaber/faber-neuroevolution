%% @doc L1 Hyperparameter Controller for Liquid Conglomerate Silos.
%%
%% Part of the Liquid Conglomerate v2 architecture. This module implements
%% the L1 tactical layer that learns to tune L0's hyperparameters based on
%% L0's performance metrics.
%%
%% == Architecture ==
%%
%% L1 is itself a TWEANN that:
%% - Observes L0 performance over τ_L1 time windows
%% - Outputs adjustments (deltas) to L0's hyperparameters
%% - Evolves slower than L0 to provide a stable adaptation platform
%%
%% == Hyperparameters Tuned (by Silo) ==
%%
%% Resource Silo:
%% - memory_high_threshold, pressure_smoothing_alpha
%%
%% Task Silo:
%% - mutation_rate_min/max, topology_mutation_boost, exploitation_vs_exploration
%% - archive_threshold_min/max, archive_diversity_weight, archive_recency_decay
%%
%% Distribution Silo:
%% - migration_cooldown_ms, load_imbalance_threshold
%%
%% == Learning Mechanism ==
%%
%% L1 learns through meta-evolution:
%% 1. Population of L1 TWEANNs (5-10 individuals)
%% 2. Each L1 variant manages L0 for N τ_L1 cycles
%% 3. Fitness = how well L0 performed under that hyperparameter regime
%% 4. Selection + mutation + crossover produces next generation
%%
%% == Usage ==
%%
%% %% Create L1 controller for a silo
%% Config = #{
%%     silo_type => resource,
%%     morphology_module => resource_l0_morphology,
%%     tau_l1 => 30000,  % 30 seconds
%%     l0_hyperparameters => resource_l0_morphology:l0_hyperparameters()
%% },
%% {ok, Pid} = lc_l1_controller:start_link(Config),
%%
%% %% Update with L0 performance (called every τ_L0)
%% lc_l1_controller:observe_l0_performance(Pid, L0Metrics),
%%
%% %% Get current hyperparameter adjustments for L0
%% Deltas = lc_l1_controller:get_hyperparameter_deltas(Pid),
%%
%% @author R.G. Lefever
%% @copyright 2024-2026 R.G. Lefever
-module(lc_l1_controller).
-behaviour(gen_server).

%% API
-export([
    start_link/1,
    observe_l0_performance/2,
    get_hyperparameter_deltas/1,
    get_current_hyperparameters/1,
    apply_deltas_to_hyperparameters/2,
    set_l2_hyperparameters/2
]).

%% gen_server callbacks
-export([
    init/1,
    handle_call/3,
    handle_cast/2,
    handle_info/2,
    terminate/2
]).

-record(state, {
    %% Configuration
    silo_type :: atom(),
    morphology_module :: module(),
    tau_l1 :: pos_integer(),
    l0_hyperparameter_names :: [atom()],
    l0_bounds :: map(),

    %% Current L0 hyperparameters
    current_hyperparameters :: map(),

    %% L1's own hyperparameters (tuned by L2)
    l1_hyperparameters :: map(),

    %% Performance observation window
    performance_history :: [map()],
    window_size :: pos_integer(),

    %% Current output deltas
    current_deltas :: map(),

    %% Timing
    last_adjustment_time :: integer(),
    observations_since_adjustment :: non_neg_integer()
}).

%%% ============================================================================
%%% API Functions
%%% ============================================================================

%% @doc Start L1 controller with configuration.
-spec start_link(map()) -> {ok, pid()} | ignore | {error, term()}.
start_link(Config) ->
    gen_server:start_link(?MODULE, Config, []).

%% @doc Observe L0 performance metrics.
%%
%% Called every τ_L0 with L0's performance metrics (reward, sensors, etc.)
-spec observe_l0_performance(pid(), map()) -> ok.
observe_l0_performance(Pid, L0Metrics) ->
    gen_server:cast(Pid, {observe_l0, L0Metrics}).

%% @doc Get current hyperparameter adjustment deltas.
%%
%% Returns map of {hyperparameter_name => delta_value}
-spec get_hyperparameter_deltas(pid()) -> map().
get_hyperparameter_deltas(Pid) ->
    gen_server:call(Pid, get_deltas).

%% @doc Get current absolute hyperparameters for L0.
-spec get_current_hyperparameters(pid()) -> map().
get_current_hyperparameters(Pid) ->
    gen_server:call(Pid, get_hyperparameters).

%% @doc Apply deltas to base hyperparameters (utility function).
-spec apply_deltas_to_hyperparameters(map(), map()) -> map().
apply_deltas_to_hyperparameters(BaseHyperparams, Deltas) ->
    maps:merge(BaseHyperparams, Deltas).

%% @doc Set L1's own hyperparameters (from L2).
-spec set_l2_hyperparameters(pid(), map()) -> ok.
set_l2_hyperparameters(Pid, L1Hyperparams) ->
    gen_server:cast(Pid, {set_l1_hyperparameters, L1Hyperparams}).

%%% ============================================================================
%%% gen_server Callbacks
%%% ============================================================================

init(Config) ->
    SiloType = maps:get(silo_type, Config, unknown),
    MorphologyModule = maps:get(morphology_module, Config),
    TauL1 = maps:get(tau_l1, Config, 30000),
    WindowSize = maps:get(window_size, Config, 10),

    %% Get L0 hyperparameter definitions
    L0HyperparamNames = MorphologyModule:l0_hyperparameters(),
    L0Defaults = MorphologyModule:get_l0_defaults(),
    L0Bounds = MorphologyModule:get_l0_bounds(),

    %% Get L1 hyperparameter definitions
    L1Defaults = MorphologyModule:get_l1_defaults(),

    State = #state{
        silo_type = SiloType,
        morphology_module = MorphologyModule,
        tau_l1 = TauL1,
        l0_hyperparameter_names = L0HyperparamNames,
        l0_bounds = L0Bounds,
        current_hyperparameters = L0Defaults,
        l1_hyperparameters = L1Defaults,
        performance_history = [],
        window_size = WindowSize,
        current_deltas = #{},
        last_adjustment_time = erlang:monotonic_time(millisecond),
        observations_since_adjustment = 0
    },

    %% Subscribe to L0 metrics and L2 guidance events (event-driven pattern)
    controller_events:subscribe_to_l0_metrics(SiloType),
    controller_events:subscribe_to_l2_guidance(SiloType),

    {ok, State}.

handle_call(get_deltas, _From, State) ->
    {reply, State#state.current_deltas, State};

handle_call(get_hyperparameters, _From, State) ->
    {reply, State#state.current_hyperparameters, State};

handle_call(_Request, _From, State) ->
    {reply, {error, unknown_request}, State}.

handle_cast({observe_l0, L0Metrics}, State) ->
    %% DEPRECATED: Use controller_events:publish_l0_metrics/2 instead
    logger:warning("[DEPRECATED] lc_l1_controller:observe_l0_performance/2 called. "
                   "Use controller_events:publish_l0_metrics/2 instead."),
    NewState = do_observe_l0(L0Metrics, State),
    {noreply, NewState};

handle_cast({set_l1_hyperparameters, L1Hyperparams}, State) ->
    %% DEPRECATED: Use controller_events:publish_l2_guidance/2 instead
    logger:warning("[DEPRECATED] lc_l1_controller:set_l2_hyperparameters/2 called. "
                   "Use controller_events:publish_l2_guidance/2 instead."),
    {noreply, State#state{l1_hyperparameters = L1Hyperparams}};

handle_cast(_Msg, State) ->
    {noreply, State}.

%% Handle L0 metrics event (event-driven pattern)
handle_info({neuro_event, Topic, #{silo := Silo, metrics := L0Metrics}}, State) ->
    %% Check if this is an L0 metrics topic for our silo
    ExpectedTopic = controller_events:l0_metrics_topic(State#state.silo_type),
    case Topic =:= ExpectedTopic orelse Silo =:= State#state.silo_type of
        true ->
            NewState = do_observe_l0(L0Metrics, State),
            {noreply, NewState};
        false ->
            {noreply, State}
    end;

%% Handle L2 guidance event (event-driven pattern)
handle_info({neuro_event, Topic, #{silo := Silo, guidance := Guidance}}, State) ->
    ExpectedTopic = controller_events:l2_guidance_topic(State#state.silo_type),
    case Topic =:= ExpectedTopic orelse Silo =:= State#state.silo_type of
        true ->
            %% Apply L2's hyperparameters
            L1Hyperparams = maps:get(l1_hyperparameters, Guidance, State#state.l1_hyperparameters),
            {noreply, State#state{l1_hyperparameters = L1Hyperparams}};
        false ->
            {noreply, State}
    end;

handle_info(_Info, State) ->
    {noreply, State}.

terminate(_Reason, _State) ->
    ok.

%%% ============================================================================
%%% Internal Functions - L0 Observation
%%% ============================================================================

%% @private Process L0 metrics observation (shared by cast and event handlers).
do_observe_l0(L0Metrics, State) ->
    %% Add to performance history
    NewHistory = add_to_history(L0Metrics, State#state.performance_history,
                                State#state.window_size),

    %% Check if it's time for an adjustment
    Now = erlang:monotonic_time(millisecond),
    TimeSinceAdjustment = Now - State#state.last_adjustment_time,
    NewObsCount = State#state.observations_since_adjustment + 1,

    case TimeSinceAdjustment >= State#state.tau_l1 of
        true ->
            %% Time to compute new adjustments
            NewState = compute_adjustment(State#state{
                performance_history = NewHistory,
                observations_since_adjustment = 0,
                last_adjustment_time = Now
            }),
            %% Publish L1 metrics (event-driven pattern)
            L1Metrics = #{
                cumulative_reward => compute_cumulative_reward(NewHistory),
                hyperparameter_deltas => NewState#state.current_deltas,
                observations => length(NewHistory)
            },
            controller_events:publish_l1_metrics(State#state.silo_type, L1Metrics),
            NewState;
        false ->
            State#state{
                performance_history = NewHistory,
                observations_since_adjustment = NewObsCount
            }
    end.

%% @private Compute cumulative reward from history.
compute_cumulative_reward([]) -> 0.0;
compute_cumulative_reward(History) ->
    lists:foldl(
        fun(Metrics, Acc) ->
            Acc + maps:get(reward, Metrics, 0.0)
        end,
        0.0,
        History
    ).

%%% ============================================================================
%%% Internal Functions - Adjustment Computation
%%% ============================================================================

%% @private Compute hyperparameter adjustments based on performance history.
compute_adjustment(State) ->
    L1Hyperparams = State#state.l1_hyperparameters,
    PerformanceHistory = State#state.performance_history,

    %% Extract L1 hyperparameters that control adjustment behavior
    AdaptationRate = maps:get(threshold_adaptation_rate, L1Hyperparams,
                              maps:get(aggression_factor, L1Hyperparams, 0.1)),

    %% Analyze performance trends
    Trends = analyze_performance_trends(PerformanceHistory),

    %% Compute deltas based on trends and L1 hyperparameters
    Deltas = compute_deltas_for_silo(State#state.silo_type, Trends,
                                      L1Hyperparams, State),

    %% Apply deltas to current hyperparameters (with bounds)
    NewHyperparams = apply_bounded_deltas(
        State#state.current_hyperparameters,
        Deltas,
        State#state.l0_bounds,
        AdaptationRate
    ),

    State#state{
        current_deltas = Deltas,
        current_hyperparameters = NewHyperparams
    }.

%% @private Analyze trends from performance history.
analyze_performance_trends([]) ->
    #{avg_reward => 0.0, reward_trend => 0.0, stability => 1.0};
analyze_performance_trends(History) ->
    Rewards = [maps:get(reward, M, 0.0) || M <- History],

    %% Average reward
    AvgReward = lists:sum(Rewards) / max(1, length(Rewards)),

    %% Reward trend (simple linear regression slope)
    RewardTrend = compute_trend(Rewards),

    %% Stability (inverse of variance)
    Variance = compute_variance(Rewards, AvgReward),
    Stability = 1.0 / max(0.01, 1.0 + Variance * 10.0),

    %% Stagnation detection
    Stagnation = case abs(RewardTrend) < 0.001 andalso length(History) > 5 of
        true -> 1.0;
        false -> 0.0
    end,

    #{
        avg_reward => AvgReward,
        reward_trend => RewardTrend,
        stability => Stability,
        stagnation => Stagnation
    }.

%% @private Compute deltas specific to silo type.
compute_deltas_for_silo(resource, Trends, L1Hyperparams, _State) ->
    %% Resource Silo L1 adjustments
    %% - If performance is low, might need to adjust thresholds
    %% - If stability is low, increase smoothing
    AvgReward = maps:get(avg_reward, Trends, 0.0),
    Stability = maps:get(stability, Trends, 1.0),
    PressureSensitivity = maps:get(pressure_sensitivity, L1Hyperparams, 1.0),

    #{
        %% Lower memory threshold if reward is consistently low
        memory_high_threshold => case AvgReward < 0.3 of
            true -> -0.02 * PressureSensitivity;
            false -> 0.0
        end,
        %% Increase smoothing if stability is low
        pressure_smoothing_alpha => case Stability < 0.5 of
            true -> 0.05;
            false -> 0.0
        end
    };

compute_deltas_for_silo(task, Trends, L1Hyperparams, _State) ->
    %% Task Silo L1 adjustments
    %% - If stagnating, boost exploration (mutation rates, topology mutation)
    %% - If improving rapidly, exploit (reduce exploration)
    %% - Archive params adjust based on learning progress
    RewardTrend = maps:get(reward_trend, Trends, 0.0),
    Stagnation = maps:get(stagnation, Trends, 0.0),
    AggressionFactor = maps:get(aggression_factor, L1Hyperparams, 0.5),
    ExplorationStep = maps:get(exploration_step, L1Hyperparams, 0.1),
    TopologyAggression = maps:get(topology_aggression, L1Hyperparams, 1.5),

    %% Compute exploration boost based on stagnation
    ExplorationBoost = case Stagnation > 0.5 of
        true -> ExplorationStep * AggressionFactor;
        false -> case RewardTrend > 0.01 of
            true -> -ExplorationStep * 0.5;  % Exploit when improving
            false -> 0.0
        end
    end,

    %% Archive adjustments:
    %% - When stagnating: lower threshold (more diverse archive), increase diversity weight
    %% - When improving: higher threshold (more selective), fitness-weighted sampling
    ArchiveThresholdDelta = case Stagnation > 0.5 of
        true -> -0.05 * AggressionFactor;   % Lower threshold = more entries
        false -> case RewardTrend > 0.01 of
            true -> 0.03;                    % Higher threshold = selective
            false -> 0.0
        end
    end,

    ArchiveDiversityDelta = case Stagnation > 0.5 of
        true -> 0.1 * AggressionFactor;     % More diversity when stagnating
        false -> case RewardTrend > 0.01 of
            true -> -0.05;                   % Less diversity when improving
            false -> 0.0
        end
    end,

    #{
        %% Mutation rate adjustments
        mutation_rate_min => ExplorationBoost * 0.5,
        mutation_rate_max => ExplorationBoost,
        %% Topology mutation boost
        topology_mutation_boost => case Stagnation > 0.5 of
            true -> 0.2 * TopologyAggression;
            false -> 0.0
        end,
        %% Exploration vs exploitation balance
        exploitation_vs_exploration => case RewardTrend > 0.01 of
            true -> 0.1;   % More exploitation
            false -> -0.1  % More exploration
        end,
        %% Self-play archive hyperparameters
        archive_threshold_min => ArchiveThresholdDelta,
        archive_threshold_max => ArchiveThresholdDelta,
        archive_diversity_weight => ArchiveDiversityDelta,
        %% Recency decay: slower decay (higher value) when stagnating to keep old opponents longer
        archive_recency_decay => case Stagnation > 0.5 of
            true -> 0.01;   % Slower decay, keep old entries longer
            false -> -0.01  % Faster decay when improving
        end
    };

compute_deltas_for_silo(distribution, Trends, L1Hyperparams, _State) ->
    %% Distribution Silo L1 adjustments
    LoadSensitivity = maps:get(load_sensitivity, L1Hyperparams, 1.0),
    MigrationAdaptRate = maps:get(migration_adaptation_rate, L1Hyperparams, 0.1),
    AvgReward = maps:get(avg_reward, Trends, 0.0),

    #{
        %% Adjust migration cooldown based on performance
        migration_cooldown_ms => case AvgReward < 0.5 of
            true -> -1000 * MigrationAdaptRate;  % Faster migration
            false -> 0
        end,
        %% Adjust load imbalance threshold
        load_imbalance_threshold => case AvgReward < 0.3 of
            true -> -0.05 * LoadSensitivity;  % More sensitive
            false -> 0.0
        end
    };

compute_deltas_for_silo(_Unknown, _Trends, _L1Hyperparams, _State) ->
    #{}.

%% @private Apply bounded deltas with adaptation rate.
apply_bounded_deltas(CurrentHyperparams, Deltas, Bounds, AdaptationRate) ->
    maps:fold(
        fun(Name, Delta, Acc) ->
            case maps:get(Name, Acc, undefined) of
                undefined -> Acc;
                CurrentValue ->
                    ScaledDelta = Delta * AdaptationRate,
                    NewValue = CurrentValue + ScaledDelta,
                    BoundedValue = apply_bounds(Name, NewValue, Bounds),
                    maps:put(Name, BoundedValue, Acc)
            end
        end,
        CurrentHyperparams,
        Deltas
    ).

%% @private Apply bounds to a hyperparameter value.
apply_bounds(Name, Value, Bounds) ->
    case maps:get(Name, Bounds, undefined) of
        undefined -> Value;
        {Min, Max} -> max(Min, min(Max, Value))
    end.

%%% ============================================================================
%%% Internal Functions - Utilities
%%% ============================================================================

%% @private Add observation to history with max size.
add_to_history(Observation, History, MaxSize) ->
    lists:sublist([Observation | History], MaxSize).

%% @private Compute simple trend (slope) from list of values.
compute_trend([]) -> 0.0;
compute_trend([_]) -> 0.0;
compute_trend(Values) ->
    N = length(Values),
    Indexed = lists:zip(lists:seq(1, N), lists:reverse(Values)),
    SumX = N * (N + 1) / 2,
    SumY = lists:sum(Values),
    SumXY = lists:foldl(fun({X, Y}, Acc) -> Acc + X * Y end, 0.0, Indexed),
    SumXX = N * (N + 1) * (2*N + 1) / 6,
    Denom = N * SumXX - SumX * SumX,
    case abs(Denom) < 0.0001 of
        true -> 0.0;
        false -> (N * SumXY - SumX * SumY) / Denom
    end.

%% @private Compute variance from list of values.
compute_variance([], _Mean) -> 0.0;
compute_variance(Values, Mean) ->
    SumSqDiff = lists:foldl(fun(V, Acc) -> Acc + (V - Mean) * (V - Mean) end,
                            0.0, Values),
    SumSqDiff / max(1, length(Values)).
