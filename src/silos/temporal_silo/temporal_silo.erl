%% @doc Temporal Silo - Episode timing and temporal control for neuroevolution.
%%
%% Part of the Liquid Conglomerate v2 architecture. The Temporal Silo manages:
%%   Episode length control (min/max/target)
%%   Early termination detection
%%   Evaluation timeouts
%%   Learning rate scheduling
%%   Reaction time budgets
%%
%% == Time Constant ==
%%
%% τ = 10 (fast adaptation for responsive timing control)
%%
%% == Cross-Silo Signals ==
%%
%% Outgoing:
%%   time_pressure to task: Temporal constraint urgency
%%   convergence_status to resource: Progress toward convergence
%%   episode_efficiency to economic: Cost efficiency of episodes
%%   critical_period_timing to developmental: Timing in developmental windows
%%
%% Incoming:
%%   stagnation_severity from task: Evolution stagnation level
%%   budget_available from economic: Computation budget available
%%   network_latency from distribution: Network delay factor
%%
%% @author R.G. Lefever
%% @copyright 2024-2026 R.G. Lefever
-module(temporal_silo).
-behaviour(gen_server).
-behaviour(lc_silo_behavior).

-include("lc_silos.hrl").
-include("lc_signals.hrl").
-include("lc_events.hrl").

%% API
-export([
    start_link/0,
    start_link/1,
    get_params/1,
    update_episode/3,
    record_reaction_time/2,
    record_timeout/1,
    record_early_termination/1,
    get_state/1,
    reset/1
]).

%% gen_server callbacks
-export([
    init/1,
    handle_call/3,
    handle_cast/2,
    handle_info/2,
    terminate/2
]).

%% lc_silo_behavior callbacks
-export([
    init_silo/1,
    collect_sensors/1,
    apply_actuators/2,
    compute_reward/1,
    get_silo_type/0,
    get_time_constant/0,
    handle_cross_silo_signals/2,
    emit_cross_silo_signals/1,
    emit_silo_events/2
]).

-define(SERVER, ?MODULE).
-define(TIME_CONSTANT, 10.0).
-define(HISTORY_SIZE, 50).

%% Default actuator values
-define(DEFAULT_PARAMS, #{
    episode_length_target => 1000,
    evaluation_timeout_ms => 5000,
    reaction_time_limit_ms => 100,
    early_termination_threshold => 0.1,
    patience_threshold => 20,
    learning_rate_multiplier => 1.0,
    discount_factor => 0.99,
    eligibility_trace_decay => 0.9,
    episode_variance_allowed => 0.3,
    learning_rate_decay => 0.99
}).

%% Actuator bounds
-define(ACTUATOR_BOUNDS, #{
    episode_length_target => {10, 10000},
    evaluation_timeout_ms => {100, 60000},
    reaction_time_limit_ms => {1, 1000},
    early_termination_threshold => {0.0, 0.5},
    patience_threshold => {5, 100},
    learning_rate_multiplier => {0.1, 10.0},
    discount_factor => {0.9, 0.999},
    eligibility_trace_decay => {0.0, 1.0},
    episode_variance_allowed => {0.0, 1.0},
    learning_rate_decay => {0.9, 1.0}
}).

-record(state, {
    %% Core silo state
    realm :: binary(),
    enabled_levels :: [l0 | l1 | l2],
    l0_tweann_enabled :: boolean(),
    l2_enabled :: boolean(),

    %% Current parameters (actuator outputs)
    current_params :: map(),

    %% ETS tables
    ets_tables :: #{atom() => ets:tid()},

    %% Episode tracking
    episode_count :: non_neg_integer(),
    episode_length_history :: [pos_integer()],
    early_termination_count :: non_neg_integer(),
    timeout_count :: non_neg_integer(),

    %% Reaction time tracking
    reaction_times :: [pos_integer()],

    %% Convergence tracking
    fitness_checkpoints :: [{non_neg_integer(), float()}],
    patience_counter :: non_neg_integer(),

    %% Cross-silo signal cache
    incoming_signals :: map(),

    %% Previous values for smoothing
    prev_time_pressure :: float(),
    prev_convergence_status :: float()
}).

%%% ============================================================================
%%% API Functions
%%% ============================================================================

-spec start_link() -> {ok, pid()} | ignore | {error, term()}.
start_link() ->
    start_link(#{}).

-spec start_link(map()) -> {ok, pid()} | ignore | {error, term()}.
start_link(Config) ->
    gen_server:start_link({local, ?SERVER}, ?MODULE, Config, []).

-spec get_params(pid()) -> map().
get_params(Pid) ->
    gen_server:call(Pid, get_params).

-spec update_episode(pid(), non_neg_integer(), float()) -> ok.
update_episode(Pid, EpisodeLength, FinalFitness) ->
    gen_server:cast(Pid, {update_episode, EpisodeLength, FinalFitness}).

-spec record_reaction_time(pid(), pos_integer()) -> ok.
record_reaction_time(Pid, ReactionTimeMs) ->
    gen_server:cast(Pid, {record_reaction_time, ReactionTimeMs}).

-spec record_timeout(pid()) -> ok.
record_timeout(Pid) ->
    gen_server:cast(Pid, record_timeout).

-spec record_early_termination(pid()) -> ok.
record_early_termination(Pid) ->
    gen_server:cast(Pid, record_early_termination).

-spec get_state(pid()) -> map().
get_state(Pid) ->
    gen_server:call(Pid, get_state).

-spec reset(pid()) -> ok.
reset(Pid) ->
    gen_server:call(Pid, reset).

%%% ============================================================================
%%% lc_silo_behavior Callbacks
%%% ============================================================================

get_silo_type() -> temporal.

get_time_constant() -> ?TIME_CONSTANT.

init_silo(Config) ->
    Realm = maps:get(realm, Config, <<"default">>),
    EtsTables = lc_ets_utils:create_tables(temporal, Realm, [
        {episode_history, [{keypos, 1}]}
    ]),
    {ok, #{
        ets_tables => EtsTables,
        realm => Realm
    }}.

collect_sensors(State) ->
    #state{
        episode_length_history = EpisodeLengths,
        early_termination_count = EarlyTermCount,
        timeout_count = TimeoutCount,
        episode_count = EpisodeCount,
        reaction_times = ReactionTimes,
        patience_counter = PatienceCounter,
        current_params = Params,
        fitness_checkpoints = Checkpoints,
        incoming_signals = InSignals
    } = State,

    %% Episode metrics
    {EpisodeMean, EpisodeVariance} = compute_episode_stats(EpisodeLengths),
    EarlyTermRate = safe_ratio(EarlyTermCount, EpisodeCount),
    TimeoutRate = safe_ratio(TimeoutCount, EpisodeCount),

    %% Reaction time metrics
    ReactionMean = safe_mean(ReactionTimes),
    ReactionLimit = maps:get(reaction_time_limit_ms, Params, 100),
    ReactionBudgetUsed = safe_ratio(ReactionMean, ReactionLimit),

    %% Convergence metrics
    ConvergenceRate = compute_convergence_rate(Checkpoints),
    PatienceThreshold = maps:get(patience_threshold, Params, 20),
    PatienceRemaining = lc_silo_behavior:normalize(
        PatienceThreshold - PatienceCounter, 0, PatienceThreshold
    ),

    %% Learning rate
    LRMultiplier = maps:get(learning_rate_multiplier, Params, 1.0),
    LRCurrent = lc_silo_behavior:normalize(LRMultiplier, 0.1, 10.0),

    %% Credit horizon (from discount factor)
    DiscountFactor = maps:get(discount_factor, Params, 0.99),
    CreditHorizon = lc_silo_behavior:normalize(DiscountFactor, 0.9, 0.999),

    %% Cross-silo signals as sensors
    StagnationSeverity = maps:get(stagnation_severity, InSignals, 0.5),
    ComputationBudget = maps:get(budget_available, InSignals, 0.5),

    #{
        episode_length_mean => EpisodeMean,
        episode_length_variance => EpisodeVariance,
        early_termination_rate => EarlyTermRate,
        reaction_time_mean => lc_silo_behavior:normalize(ReactionMean, 0, 1000),
        reaction_time_budget_used => ReactionBudgetUsed,
        convergence_rate => ConvergenceRate,
        patience_remaining => PatienceRemaining,
        timeout_rate => TimeoutRate,
        learning_rate_current => LRCurrent,
        credit_horizon => CreditHorizon,
        stagnation_severity => StagnationSeverity,
        computation_budget => ComputationBudget
    }.

apply_actuators(Actuators, State) ->
    BoundedParams = apply_bounds(Actuators, ?ACTUATOR_BOUNDS),
    NewState = State#state{current_params = BoundedParams},
    emit_cross_silo_signals(NewState),
    %% Emit event for hyperparameter adjustment
    emit_silo_events(?EVENT_HYPERPARAMS_ADJUSTED, NewState),
    %% Check if learning rate specifically changed
    OldLR = maps:get(learning_rate_multiplier, State#state.current_params, 1.0),
    NewLR = maps:get(learning_rate_multiplier, BoundedParams, 1.0),
    maybe_emit_learning_rate_event(OldLR, NewLR, NewState),
    {ok, NewState}.

compute_reward(State) ->
    Sensors = collect_sensors(State),

    %% Reward components:
    %% 1. Low early termination rate (efficient episodes)
    EarlyTermPenalty = maps:get(early_termination_rate, Sensors, 0.0),

    %% 2. Low timeout rate (completing evaluations)
    TimeoutPenalty = maps:get(timeout_rate, Sensors, 0.0),

    %% 3. Good convergence progress
    ConvergenceBonus = maps:get(convergence_rate, Sensors, 0.5),

    %% 4. Efficient reaction times
    ReactionEfficiency = 1.0 - maps:get(reaction_time_budget_used, Sensors, 0.5),

    %% Combined reward
    Reward = 0.3 * (1.0 - EarlyTermPenalty) +
             0.2 * (1.0 - TimeoutPenalty) +
             0.3 * ConvergenceBonus +
             0.2 * ReactionEfficiency,

    lc_silo_behavior:clamp(Reward, 0.0, 1.0).

handle_cross_silo_signals(Signals, State) ->
    %% Merge new signals with existing
    CurrentSignals = State#state.incoming_signals,
    UpdatedSignals = maps:merge(CurrentSignals, Signals),
    {ok, State#state{incoming_signals = UpdatedSignals}}.

emit_cross_silo_signals(State) ->
    Sensors = collect_sensors(State),

    %% Time pressure: high when timeouts or early terminations are frequent
    TimePressure = compute_time_pressure(Sensors),

    %% Convergence status: progress toward solution
    ConvergenceStatus = maps:get(convergence_rate, Sensors, 0.5),

    %% Episode efficiency: fitness gain per computation
    EpisodeEfficiency = compute_episode_efficiency(State),

    %% Critical period timing: normalized episode position
    CriticalPeriodTiming = compute_critical_period_timing(State),

    %% Emit signals
    emit_signal(task, time_pressure, TimePressure),
    emit_signal(resource, convergence_status, ConvergenceStatus),
    emit_signal(economic, episode_efficiency, EpisodeEfficiency),
    emit_signal(developmental, critical_period_timing, CriticalPeriodTiming),
    ok.

%% @doc Emit silo events for persistence (zero-config).
%%
%% Events are automatically persisted if faber_neuroevolution_esdb is
%% available. If not, this is a no-op.
emit_silo_events(?EVENT_HYPERPARAMS_ADJUSTED, State) ->
    lc_event_emitter:emit(temporal, ?EVENT_HYPERPARAMS_ADJUSTED, #{
        realm => State#state.realm,
        params => State#state.current_params,
        episode_count => State#state.episode_count
    });
emit_silo_events(?EVENT_EPISODE_TIMEOUT, State) ->
    lc_event_emitter:emit(temporal, ?EVENT_EPISODE_TIMEOUT, #{
        realm => State#state.realm,
        timeout_count => State#state.timeout_count,
        episode_count => State#state.episode_count
    });
emit_silo_events(?EVENT_EARLY_TERMINATION, State) ->
    lc_event_emitter:emit(temporal, ?EVENT_EARLY_TERMINATION, #{
        realm => State#state.realm,
        early_termination_count => State#state.early_termination_count,
        episode_count => State#state.episode_count
    });
emit_silo_events(?EVENT_PATIENCE_EXHAUSTED, State) ->
    lc_event_emitter:emit(temporal, ?EVENT_PATIENCE_EXHAUSTED, #{
        realm => State#state.realm,
        patience_counter => State#state.patience_counter,
        episode_count => State#state.episode_count
    });
emit_silo_events(?EVENT_LEARNING_RATE_ADJUSTED, State) ->
    LRMultiplier = maps:get(learning_rate_multiplier, State#state.current_params, 1.0),
    lc_event_emitter:emit(temporal, ?EVENT_LEARNING_RATE_ADJUSTED, #{
        realm => State#state.realm,
        learning_rate_multiplier => LRMultiplier,
        episode_count => State#state.episode_count
    });
emit_silo_events(_EventType, _State) ->
    ok.

%%% ============================================================================
%%% gen_server Callbacks
%%% ============================================================================

init(Config) ->
    Realm = maps:get(realm, Config, <<"default">>),
    EnabledLevels = maps:get(enabled_levels, Config, [l0, l1]),
    L0TweannEnabled = maps:get(l0_tweann_enabled, Config, false),
    L2Enabled = maps:get(l2_enabled, Config, false),

    %% Create ETS tables
    EtsTables = lc_ets_utils:create_tables(temporal, Realm, [
        {episode_history, [{keypos, 1}]}
    ]),

    State = #state{
        realm = Realm,
        enabled_levels = EnabledLevels,
        l0_tweann_enabled = L0TweannEnabled,
        l2_enabled = L2Enabled,
        current_params = ?DEFAULT_PARAMS,
        ets_tables = EtsTables,
        episode_count = 0,
        episode_length_history = [],
        early_termination_count = 0,
        timeout_count = 0,
        reaction_times = [],
        fitness_checkpoints = [],
        patience_counter = 0,
        incoming_signals = #{},
        prev_time_pressure = 0.0,
        prev_convergence_status = 0.5
    },

    %% Schedule periodic cross-silo signal update
    erlang:send_after(1000, self(), update_signals),

    {ok, State}.

handle_call(get_params, _From, State) ->
    {reply, State#state.current_params, State};

handle_call(get_state, _From, State) ->
    StateMap = #{
        realm => State#state.realm,
        enabled_levels => State#state.enabled_levels,
        current_params => State#state.current_params,
        episode_count => State#state.episode_count,
        early_termination_count => State#state.early_termination_count,
        timeout_count => State#state.timeout_count,
        patience_counter => State#state.patience_counter,
        sensors => collect_sensors(State)
    },
    {reply, StateMap, State};

handle_call(reset, _From, State) ->
    NewState = State#state{
        current_params = ?DEFAULT_PARAMS,
        episode_count = 0,
        episode_length_history = [],
        early_termination_count = 0,
        timeout_count = 0,
        reaction_times = [],
        fitness_checkpoints = [],
        patience_counter = 0,
        incoming_signals = #{},
        prev_time_pressure = 0.0,
        prev_convergence_status = 0.5
    },
    {reply, ok, NewState};

handle_call(_Request, _From, State) ->
    {reply, {error, unknown_request}, State}.

handle_cast({update_episode, EpisodeLength, FinalFitness}, State) ->
    %% Update episode history
    NewHistory = truncate_history(
        [EpisodeLength | State#state.episode_length_history],
        ?HISTORY_SIZE
    ),

    %% Update fitness checkpoints for convergence tracking
    NewCheckpoints = add_checkpoint(
        State#state.fitness_checkpoints,
        State#state.episode_count + 1,
        FinalFitness
    ),

    %% Update patience counter based on improvement
    NewPatience = update_patience(
        State#state.patience_counter,
        State#state.fitness_checkpoints,
        FinalFitness
    ),

    %% Store in ETS
    EpisodeTable = maps:get(episode_history, State#state.ets_tables),
    lc_ets_utils:insert(EpisodeTable, State#state.episode_count + 1, #{
        length => EpisodeLength,
        fitness => FinalFitness
    }),

    NewState = State#state{
        episode_count = State#state.episode_count + 1,
        episode_length_history = NewHistory,
        fitness_checkpoints = NewCheckpoints,
        patience_counter = NewPatience
    },
    %% Check if patience was just exhausted
    PatienceThreshold = maps:get(patience_threshold, State#state.current_params, 20),
    maybe_emit_patience_exhausted(NewPatience, PatienceThreshold, NewState),
    {noreply, NewState};

handle_cast({record_reaction_time, ReactionTimeMs}, State) ->
    NewTimes = truncate_history(
        [ReactionTimeMs | State#state.reaction_times],
        ?HISTORY_SIZE
    ),
    {noreply, State#state{reaction_times = NewTimes}};

handle_cast(record_timeout, State) ->
    NewState = State#state{timeout_count = State#state.timeout_count + 1},
    emit_silo_events(?EVENT_EPISODE_TIMEOUT, NewState),
    {noreply, NewState};

handle_cast(record_early_termination, State) ->
    NewState = State#state{early_termination_count = State#state.early_termination_count + 1},
    emit_silo_events(?EVENT_EARLY_TERMINATION, NewState),
    {noreply, NewState};

handle_cast(_Msg, State) ->
    {noreply, State}.

handle_info(update_signals, State) ->
    %% Fetch incoming signals from cross-silo coordinator
    fetch_incoming_signals(State),

    %% Emit outgoing signals
    emit_cross_silo_signals(State),

    %% Reschedule
    erlang:send_after(1000, self(), update_signals),
    {noreply, State};

handle_info(_Info, State) ->
    {noreply, State}.

terminate(_Reason, State) ->
    lc_ets_utils:delete_tables(State#state.ets_tables),
    ok.

%%% ============================================================================
%%% Internal Functions - Statistics
%%% ============================================================================

compute_episode_stats([]) ->
    {0.5, 0.0};
compute_episode_stats(Lengths) ->
    Mean = lists:sum(Lengths) / length(Lengths),
    Variance = compute_variance(Lengths, Mean),
    %% Normalize to 0-1 range
    NormMean = lc_silo_behavior:normalize(Mean, 10, 10000),
    NormVariance = lc_silo_behavior:normalize(Variance, 0, 1000000),
    {NormMean, NormVariance}.

compute_variance([], _Mean) -> 0.0;
compute_variance(Values, Mean) ->
    SumSquares = lists:foldl(
        fun(V, Acc) -> Acc + (V - Mean) * (V - Mean) end,
        0.0,
        Values
    ),
    SumSquares / length(Values).

safe_ratio(_Num, 0) -> 0.0;
safe_ratio(Num, Denom) -> Num / Denom.

safe_mean([]) -> 0.0;
safe_mean(Values) -> lists:sum(Values) / length(Values).

compute_convergence_rate([]) -> 0.5;
compute_convergence_rate([_]) -> 0.5;
compute_convergence_rate(Checkpoints) ->
    %% Get recent fitness values
    RecentFitnesses = [F || {_, F} <- lists:sublist(Checkpoints, 10)],
    compute_trend(RecentFitnesses).

compute_trend([]) -> 0.5;
compute_trend([_]) -> 0.5;
compute_trend(Values) ->
    %% Simple linear trend: positive = improving
    N = length(Values),
    Indices = lists:seq(1, N),
    Pairs = lists:zip(Indices, Values),

    %% Calculate slope
    MeanX = (N + 1) / 2,
    MeanY = lists:sum(Values) / N,

    {SumXY, SumXX} = lists:foldl(
        fun({X, Y}, {AccXY, AccXX}) ->
            DX = X - MeanX,
            DY = Y - MeanY,
            {AccXY + DX * DY, AccXX + DX * DX}
        end,
        {0.0, 0.0},
        Pairs
    ),

    Slope = safe_ratio(SumXY, SumXX),
    %% Normalize slope to 0-1 (assuming reasonable bounds)
    lc_silo_behavior:normalize(Slope, -0.1, 0.1).

%%% ============================================================================
%%% Internal Functions - Cross-Silo
%%% ============================================================================

compute_time_pressure(Sensors) ->
    TimeoutRate = maps:get(timeout_rate, Sensors, 0.0),
    EarlyTermRate = maps:get(early_termination_rate, Sensors, 0.0),
    PatienceUsed = 1.0 - maps:get(patience_remaining, Sensors, 1.0),

    %% Weighted combination
    0.4 * TimeoutRate + 0.3 * EarlyTermRate + 0.3 * PatienceUsed.

compute_episode_efficiency(State) ->
    Checkpoints = State#state.fitness_checkpoints,
    compute_fitness_per_episode(Checkpoints).

compute_fitness_per_episode([]) -> 0.5;
compute_fitness_per_episode([{_, F}]) -> lc_silo_behavior:normalize(F, 0.0, 1.0);
compute_fitness_per_episode(Checkpoints) ->
    [{_, FirstF} | _] = lists:reverse(Checkpoints),
    [{_, LastF} | _] = Checkpoints,
    FitnessGain = LastF - FirstF,
    EpisodeSpan = length(Checkpoints),
    GainPerEpisode = safe_ratio(FitnessGain, EpisodeSpan),
    lc_silo_behavior:normalize(GainPerEpisode, -0.01, 0.1).

compute_critical_period_timing(State) ->
    %% Normalized position in training (0 = start, 1 = mature)
    EpisodeCount = State#state.episode_count,
    %% Assume 1000 episodes is "mature"
    lc_silo_behavior:normalize(EpisodeCount, 0, 1000).

emit_signal(_ToSilo, SignalName, Value) ->
    %% Event-driven: publish signal, lc_cross_silo routes to valid destinations
    silo_events:publish_signal(temporal, SignalName, Value).

fetch_incoming_signals(State) ->
    case whereis(lc_cross_silo) of
        undefined ->
            State#state.incoming_signals;
        _Pid ->
            lc_cross_silo:get_signals_for(temporal)
    end.

%%% ============================================================================
%%% Internal Functions - History Management
%%% ============================================================================

truncate_history(List, MaxSize) when length(List) > MaxSize ->
    lists:sublist(List, MaxSize);
truncate_history(List, _MaxSize) ->
    List.

add_checkpoint(Checkpoints, EpisodeNum, Fitness) ->
    NewCheckpoints = [{EpisodeNum, Fitness} | Checkpoints],
    truncate_history(NewCheckpoints, 100).

update_patience(CurrentPatience, [], _NewFitness) ->
    CurrentPatience;
update_patience(CurrentPatience, [{_, PrevFitness} | _], NewFitness) ->
    update_patience_based_on_improvement(CurrentPatience, NewFitness > PrevFitness).

update_patience_based_on_improvement(_Patience, true) -> 0;
update_patience_based_on_improvement(Patience, false) -> Patience + 1.

%%% ============================================================================
%%% Internal Functions - Event Emission
%%% ============================================================================

%% @private Only emit patience exhausted event when threshold is reached
maybe_emit_patience_exhausted(PatienceCounter, PatienceThreshold, State)
    when PatienceCounter >= PatienceThreshold ->
    emit_silo_events(?EVENT_PATIENCE_EXHAUSTED, State);
maybe_emit_patience_exhausted(_PatienceCounter, _PatienceThreshold, _State) ->
    ok.

%% @private Only emit learning rate event when value actually changed
maybe_emit_learning_rate_event(OldLR, NewLR, State) when abs(NewLR - OldLR) > 0.001 ->
    emit_silo_events(?EVENT_LEARNING_RATE_ADJUSTED, State);
maybe_emit_learning_rate_event(_OldLR, _NewLR, _State) ->
    ok.

%%% ============================================================================
%%% Internal Functions - Bounds
%%% ============================================================================

apply_bounds(Params, Bounds) ->
    maps:fold(
        fun(Key, Value, Acc) ->
            BoundedValue = apply_single_bound(Key, Value, Bounds),
            maps:put(Key, BoundedValue, Acc)
        end,
        #{},
        Params
    ).

apply_single_bound(Key, Value, Bounds) ->
    case maps:get(Key, Bounds, undefined) of
        undefined -> Value;
        {Min, Max} -> lc_silo_behavior:clamp(Value, Min, Max)
    end.
