%% @doc Developmental Silo - Ontogeny, plasticity, and critical periods.
%%
%% Part of the Liquid Conglomerate v2 architecture. The Developmental Silo manages:
%%   Developmental stages and maturation
%%   Plasticity levels and decay
%%   Critical period timing
%%   Metamorphosis triggers
%%   Developmental noise
%%
%% == Time Constant ==
%%
%% τ = 40 (slow adaptation for developmental dynamics)
%%
%% == Cross-Silo Signals ==
%%
%% Outgoing:
%%   maturity_distribution to task: Population maturity level
%%   plasticity_available to cultural: Learning capacity
%%   metamorphosis_rate to ecological: Stage transition rate
%%   expression_stage to regulatory: Developmental expression phase
%%
%% Incoming:
%%   maturity_target from task: Target maturity level
%%   plasticity_influence from cultural: Cultural effect on plasticity
%%   stress_signal from ecological: Environmental stress level
%%   critical_period_timing from temporal: Timing in critical period
%%
%% @author R.G. Lefever
%% @copyright 2024-2026 R.G. Lefever
-module(developmental_silo).
-behaviour(gen_server).
-behaviour(lc_silo_behavior).

-include("lc_silos.hrl").
-include("lc_signals.hrl").

%% API
-export([
    start_link/0,
    start_link/1,
    get_params/1,
    update_developmental_state/4,
    get_developmental_state/2,
    trigger_metamorphosis/2,
    open_critical_period/3,
    close_critical_period/2,
    get_developmental_stats/1,
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
    emit_cross_silo_signals/1
]).

-define(SERVER, ?MODULE).
-define(TIME_CONSTANT, 40.0).
-define(HISTORY_SIZE, 100).

%% Default actuator values
-define(DEFAULT_PARAMS, #{
    growth_rate => 0.05,
    maturation_speed => 0.02,
    critical_period_duration => 5,
    plasticity_decay_rate => 0.05,
    initial_plasticity => 0.9,
    developmental_noise_level => 0.1,
    metamorphosis_trigger => 0.8,
    metamorphosis_severity => 0.5
}).

%% Actuator bounds
-define(ACTUATOR_BOUNDS, #{
    growth_rate => {0.0, 0.2},
    maturation_speed => {0.0, 0.1},
    critical_period_duration => {1, 20},
    plasticity_decay_rate => {0.0, 0.2},
    initial_plasticity => {0.5, 1.0},
    developmental_noise_level => {0.0, 0.3},
    metamorphosis_trigger => {0.5, 0.95},
    metamorphosis_severity => {0.0, 1.0}
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

    %% Aggregate statistics
    maturity_history :: [float()],
    plasticity_history :: [float()],
    metamorphosis_count :: non_neg_integer(),

    %% Cross-silo signal cache
    incoming_signals :: map(),

    %% Previous values for smoothing
    prev_maturity_distribution :: float(),
    prev_plasticity_available :: float()
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

-spec update_developmental_state(pid(), term(), float(), float()) -> ok.
update_developmental_state(Pid, IndividualId, Stage, Plasticity) ->
    gen_server:cast(Pid, {update_developmental_state, IndividualId, Stage, Plasticity}).

-spec get_developmental_state(pid(), term()) -> {ok, map()} | not_found.
get_developmental_state(Pid, IndividualId) ->
    gen_server:call(Pid, {get_developmental_state, IndividualId}).

-spec trigger_metamorphosis(pid(), term()) -> ok.
trigger_metamorphosis(Pid, IndividualId) ->
    gen_server:cast(Pid, {trigger_metamorphosis, IndividualId}).

-spec open_critical_period(pid(), term(), atom()) -> ok.
open_critical_period(Pid, IndividualId, PeriodType) ->
    gen_server:cast(Pid, {open_critical_period, IndividualId, PeriodType}).

-spec close_critical_period(pid(), term()) -> ok.
close_critical_period(Pid, IndividualId) ->
    gen_server:cast(Pid, {close_critical_period, IndividualId}).

-spec get_developmental_stats(pid()) -> map().
get_developmental_stats(Pid) ->
    gen_server:call(Pid, get_developmental_stats).

-spec get_state(pid()) -> map().
get_state(Pid) ->
    gen_server:call(Pid, get_state).

-spec reset(pid()) -> ok.
reset(Pid) ->
    gen_server:call(Pid, reset).

%%% ============================================================================
%%% lc_silo_behavior Callbacks
%%% ============================================================================

get_silo_type() -> developmental.

get_time_constant() -> ?TIME_CONSTANT.

init_silo(Config) ->
    Realm = maps:get(realm, Config, <<"default">>),
    EtsTables = lc_ets_utils:create_tables(developmental, Realm, [
        {developmental_states, [{keypos, 1}]},
        {critical_periods, [{keypos, 1}]},
        {milestones, [{keypos, 1}]}
    ]),
    {ok, #{
        ets_tables => EtsTables,
        realm => Realm
    }}.

collect_sensors(State) ->
    #state{
        plasticity_history = PlasticityHistory,
        ets_tables = EtsTables,
        metamorphosis_count = MetaCount,
        incoming_signals = InSignals
    } = State,

    %% Developmental state metrics
    StatesTable = maps:get(developmental_states, EtsTables),
    PeriodsTable = maps:get(critical_periods, EtsTables),

    AllStates = lc_ets_utils:all(StatesTable),
    {StageMean, MaturityMean, PlasticityMean} = compute_state_means(AllStates),

    %% Plasticity trend
    PlasticityTrend = compute_trend(PlasticityHistory),

    %% Critical period metrics
    AllPeriods = lc_ets_utils:all(PeriodsTable),
    CriticalPeriodActiveRatio = compute_active_period_ratio(AllPeriods),

    %% Canalization and heterochrony
    Canalization = compute_canalization(AllStates),
    Heterochrony = compute_heterochrony(AllStates),

    %% Metamorphosis rate
    MetamorphosisRate = lc_silo_behavior:normalize(MetaCount, 0, 50),

    %% Developmental noise
    DevelopmentalNoise = compute_developmental_noise(AllStates),

    %% Cross-silo signals as sensors
    MaturityTarget = maps:get(maturity_target, InSignals, 0.5),
    StressLevel = maps:get(stress_signal, InSignals, 0.0),

    #{
        developmental_stage_mean => StageMean,
        maturation_level_mean => MaturityMean,
        plasticity_level_mean => PlasticityMean,
        plasticity_trend => PlasticityTrend,
        critical_period_active_ratio => CriticalPeriodActiveRatio,
        canalization_strength => Canalization,
        heterochrony_index => Heterochrony,
        metamorphosis_rate => MetamorphosisRate,
        developmental_noise => DevelopmentalNoise,
        %% External signals
        maturity_target => MaturityTarget,
        stress_level => StressLevel
    }.

apply_actuators(Actuators, State) ->
    BoundedParams = apply_bounds(Actuators, ?ACTUATOR_BOUNDS),
    NewState = State#state{current_params = BoundedParams},
    emit_cross_silo_signals(NewState),
    {ok, NewState}.

compute_reward(State) ->
    Sensors = collect_sensors(State),

    %% Reward components:
    %% 1. Moderate plasticity (not too rigid, not too unstable)
    Plasticity = maps:get(plasticity_level_mean, Sensors, 0.5),
    PlasticityOptimality = 1.0 - abs(Plasticity - 0.5) * 2,

    %% 2. Progress toward maturity target
    MaturityMean = maps:get(maturation_level_mean, Sensors, 0.5),
    MaturityTarget = maps:get(maturity_target, Sensors, 0.5),
    MaturityProgress = 1.0 - abs(MaturityMean - MaturityTarget),

    %% 3. Low developmental noise
    Noise = maps:get(developmental_noise, Sensors, 0.5),
    NoiseOptimality = 1.0 - Noise,

    %% 4. High canalization (robustness)
    Canalization = maps:get(canalization_strength, Sensors, 0.5),

    %% 5. Moderate heterochrony
    Heterochrony = maps:get(heterochrony_index, Sensors, 0.5),
    HeterochronyOptimality = 1.0 - abs(Heterochrony - 0.3) * 2,

    %% Combined reward
    Reward = 0.20 * PlasticityOptimality +
             0.25 * MaturityProgress +
             0.20 * NoiseOptimality +
             0.20 * Canalization +
             0.15 * HeterochronyOptimality,

    lc_silo_behavior:clamp(Reward, 0.0, 1.0).

handle_cross_silo_signals(Signals, State) ->
    CurrentSignals = State#state.incoming_signals,
    UpdatedSignals = maps:merge(CurrentSignals, Signals),
    {ok, State#state{incoming_signals = UpdatedSignals}}.

emit_cross_silo_signals(State) ->
    Sensors = collect_sensors(State),

    %% Maturity distribution
    MaturityMean = maps:get(maturation_level_mean, Sensors, 0.5),
    MaturityDistribution = MaturityMean,

    %% Plasticity available
    PlasticityMean = maps:get(plasticity_level_mean, Sensors, 0.5),
    PlasticityAvailable = PlasticityMean,

    %% Metamorphosis rate
    MetamorphosisRate = maps:get(metamorphosis_rate, Sensors, 0.0),

    %% Expression stage: based on developmental stage
    StageMean = maps:get(developmental_stage_mean, Sensors, 0.5),
    ExpressionStage = StageMean,

    %% Emit signals
    emit_signal(task, maturity_distribution, MaturityDistribution),
    emit_signal(cultural, plasticity_available, PlasticityAvailable),
    emit_signal(ecological, metamorphosis_rate, MetamorphosisRate),
    emit_signal(regulatory, expression_stage, ExpressionStage),
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
    EtsTables = lc_ets_utils:create_tables(developmental, Realm, [
        {developmental_states, [{keypos, 1}]},
        {critical_periods, [{keypos, 1}]},
        {milestones, [{keypos, 1}]}
    ]),

    State = #state{
        realm = Realm,
        enabled_levels = EnabledLevels,
        l0_tweann_enabled = L0TweannEnabled,
        l2_enabled = L2Enabled,
        current_params = ?DEFAULT_PARAMS,
        ets_tables = EtsTables,
        maturity_history = [],
        plasticity_history = [],
        metamorphosis_count = 0,
        incoming_signals = #{},
        prev_maturity_distribution = 0.5,
        prev_plasticity_available = 0.5
    },

    %% Schedule periodic cross-silo signal update
    erlang:send_after(1000, self(), update_signals),

    {ok, State}.

handle_call(get_params, _From, State) ->
    {reply, State#state.current_params, State};

handle_call({get_developmental_state, IndividualId}, _From, State) ->
    StatesTable = maps:get(developmental_states, State#state.ets_tables),
    Result = lc_ets_utils:lookup(StatesTable, IndividualId),
    {reply, Result, State};

handle_call(get_developmental_stats, _From, State) ->
    StatesTable = maps:get(developmental_states, State#state.ets_tables),
    PeriodsTable = maps:get(critical_periods, State#state.ets_tables),

    Stats = #{
        individual_count => lc_ets_utils:count(StatesTable),
        active_critical_periods => lc_ets_utils:count(PeriodsTable),
        metamorphosis_count => State#state.metamorphosis_count
    },
    {reply, Stats, State};

handle_call(get_state, _From, State) ->
    StateMap = #{
        realm => State#state.realm,
        enabled_levels => State#state.enabled_levels,
        current_params => State#state.current_params,
        metamorphosis_count => State#state.metamorphosis_count,
        sensors => collect_sensors(State)
    },
    {reply, StateMap, State};

handle_call(reset, _From, State) ->
    %% Clear all ETS tables
    maps:foreach(
        fun(_Name, Table) -> ets:delete_all_objects(Table) end,
        State#state.ets_tables
    ),

    NewState = State#state{
        current_params = ?DEFAULT_PARAMS,
        maturity_history = [],
        plasticity_history = [],
        metamorphosis_count = 0,
        incoming_signals = #{},
        prev_maturity_distribution = 0.5,
        prev_plasticity_available = 0.5
    },
    {reply, ok, NewState};

handle_call(_Request, _From, State) ->
    {reply, {error, unknown_request}, State}.

handle_cast({update_developmental_state, IndividualId, Stage, Plasticity}, State) ->
    StatesTable = maps:get(developmental_states, State#state.ets_tables),
    Params = State#state.current_params,

    %% Get or create state
    Age = get_current_age(StatesTable, IndividualId),
    Maturity = compute_maturity(Stage, Age, Params),

    lc_ets_utils:insert(StatesTable, IndividualId, #{
        stage => Stage,
        plasticity => Plasticity,
        age => Age + 1,
        maturity => Maturity
    }),

    %% Update histories with numeric values
    NewMaturityHistory = truncate_history(
        [Maturity | State#state.maturity_history],
        ?HISTORY_SIZE
    ),
    NewPlasticityHistory = truncate_history(
        [Plasticity | State#state.plasticity_history],
        ?HISTORY_SIZE
    ),

    NewState = State#state{
        maturity_history = NewMaturityHistory,
        plasticity_history = NewPlasticityHistory
    },
    {noreply, NewState};

handle_cast({trigger_metamorphosis, IndividualId}, State) ->
    StatesTable = maps:get(developmental_states, State#state.ets_tables),
    MilestonesTable = maps:get(milestones, State#state.ets_tables),
    Params = State#state.current_params,

    %% Record metamorphosis
    case lc_ets_utils:lookup(StatesTable, IndividualId) of
        {ok, CurrentState} ->
            Severity = maps:get(metamorphosis_severity, Params, 0.5),
            OldStage = maps:get(stage, CurrentState, juvenile),
            OldStageNum = stage_to_number(OldStage),
            NewStageNum = min(1.0, OldStageNum + Severity * 0.5),
            NewPlasticity = maps:get(initial_plasticity, Params, 0.9),

            lc_ets_utils:insert(StatesTable, IndividualId, #{
                stage => NewStageNum,
                plasticity => NewPlasticity,
                age => maps:get(age, CurrentState, 0) + 1,
                maturity => NewStageNum
            }),

            %% Record milestone
            MilestoneId = erlang:unique_integer([positive]),
            lc_ets_utils:insert(MilestonesTable, MilestoneId, #{
                individual => IndividualId,
                type => metamorphosis,
                stage_before => OldStageNum,
                stage_after => NewStageNum
            });
        not_found ->
            ok
    end,

    NewState = State#state{
        metamorphosis_count = State#state.metamorphosis_count + 1
    },
    {noreply, NewState};

handle_cast({open_critical_period, IndividualId, PeriodType}, State) ->
    PeriodsTable = maps:get(critical_periods, State#state.ets_tables),

    lc_ets_utils:insert(PeriodsTable, IndividualId, #{
        period_type => PeriodType,
        opened_at => erlang:system_time(millisecond),
        closed_at => undefined,
        active => true
    }),
    {noreply, State};

handle_cast({close_critical_period, IndividualId}, State) ->
    PeriodsTable = maps:get(critical_periods, State#state.ets_tables),

    case lc_ets_utils:lookup(PeriodsTable, IndividualId) of
        {ok, PeriodData} ->
            lc_ets_utils:insert(PeriodsTable, IndividualId, PeriodData#{
                closed_at => erlang:system_time(millisecond),
                active => false
            });
        not_found ->
            ok
    end,
    {noreply, State};

handle_cast(_Msg, State) ->
    {noreply, State}.

handle_info(update_signals, State) ->
    %% Fetch incoming signals from cross-silo coordinator
    NewSignals = fetch_incoming_signals(),
    UpdatedState = State#state{
        incoming_signals = maps:merge(State#state.incoming_signals, NewSignals)
    },

    %% Emit outgoing signals
    emit_cross_silo_signals(UpdatedState),

    %% Reschedule
    erlang:send_after(1000, self(), update_signals),
    {noreply, UpdatedState};

handle_info(_Info, State) ->
    {noreply, State}.

terminate(_Reason, State) ->
    lc_ets_utils:delete_tables(State#state.ets_tables),
    ok.

%%% ============================================================================
%%% Internal Functions - Developmental Metrics
%%% ============================================================================

get_current_age(StatesTable, IndividualId) ->
    case lc_ets_utils:lookup(StatesTable, IndividualId) of
        {ok, Data} -> maps:get(age, Data, 0);
        not_found -> 0
    end.

compute_maturity(Stage, Age, Params) ->
    MaturationSpeed = maps:get(maturation_speed, Params, 0.02),
    BaseMaturity = stage_to_number(Stage),
    AgeContribution = min(1.0, Age * MaturationSpeed),
    min(1.0, BaseMaturity * 0.7 + AgeContribution * 0.3).

%% Convert developmental stage atoms to numerical values
stage_to_number(embryonic) -> 0.1;
stage_to_number(larval) -> 0.2;
stage_to_number(juvenile) -> 0.5;
stage_to_number(adolescent) -> 0.7;
stage_to_number(adult) -> 0.9;
stage_to_number(mature) -> 1.0;
stage_to_number(N) when is_number(N) -> N;
stage_to_number(_) -> 0.5.

compute_state_means(AllStates) ->
    compute_means_from_states(AllStates).

compute_means_from_states([]) -> {0.5, 0.5, 0.5};
compute_means_from_states(States) ->
    Stages = [stage_to_number(maps:get(stage, Data, 0.5)) || {_Id, Data, _Ts} <- States],
    Maturities = [maps:get(maturity, Data, 0.5) || {_Id, Data, _Ts} <- States],
    Plasticities = [maps:get(plasticity, Data, 0.5) || {_Id, Data, _Ts} <- States],
    {safe_mean(Stages), safe_mean(Maturities), safe_mean(Plasticities)}.

compute_active_period_ratio(AllPeriods) ->
    compute_ratio_from_periods(AllPeriods).

compute_ratio_from_periods([]) -> 0.0;
compute_ratio_from_periods(Periods) ->
    ActiveCount = length([P || {_Id, P, _Ts} <- Periods, maps:get(active, P, false)]),
    ActiveCount / length(Periods).

compute_canalization(AllStates) ->
    %% Canalization = resistance to perturbation (low variance in outcomes)
    compute_canalization_from_states(AllStates).

compute_canalization_from_states([]) -> 0.5;
compute_canalization_from_states(States) ->
    Stages = [maps:get(stage, Data, 0.5) || {_Id, Data, _Ts} <- States],
    Variance = compute_variance(Stages),
    1.0 - lc_silo_behavior:normalize(Variance, 0, 0.25).

compute_heterochrony(AllStates) ->
    %% Heterochrony = variance in developmental timing
    compute_heterochrony_from_states(AllStates).

compute_heterochrony_from_states([]) -> 0.0;
compute_heterochrony_from_states(States) ->
    Ages = [maps:get(age, Data, 0) || {_Id, Data, _Ts} <- States],
    Stages = [maps:get(stage, Data, 0.5) || {_Id, Data, _Ts} <- States],
    %% Compute variance in stage/age ratios
    Ratios = [safe_ratio(S, max(1, A)) || {S, A} <- lists:zip(Stages, Ages)],
    Variance = compute_variance(Ratios),
    lc_silo_behavior:normalize(Variance, 0, 0.5).

compute_developmental_noise(AllStates) ->
    %% Noise = random variation in development
    compute_noise_from_states(AllStates).

compute_noise_from_states([]) -> 0.0;
compute_noise_from_states(States) ->
    Plasticities = [maps:get(plasticity, Data, 0.5) || {_Id, Data, _Ts} <- States],
    Variance = compute_variance(Plasticities),
    lc_silo_behavior:normalize(Variance, 0, 0.25).

safe_mean([]) -> 0.0;
safe_mean(Values) -> lists:sum(Values) / length(Values).

compute_variance([]) -> 0.0;
compute_variance([_]) -> 0.0;
compute_variance(Values) ->
    Mean = safe_mean(Values),
    SumSquares = lists:foldl(
        fun(V, Acc) -> Acc + (V - Mean) * (V - Mean) end,
        0.0,
        Values
    ),
    SumSquares / length(Values).

compute_trend([]) -> 0.5;
compute_trend([_]) -> 0.5;
compute_trend(Values) when length(Values) < 3 -> 0.5;
compute_trend(Values) ->
    Recent = lists:sublist(Values, 5),
    Older = lists:sublist(Values, 6, 5),
    RecentMean = safe_mean(Recent),
    OlderMean = safe_mean(Older),
    Trend = safe_ratio(RecentMean - OlderMean, OlderMean + 0.001),
    lc_silo_behavior:normalize(Trend, -0.5, 0.5).

safe_ratio(_Num, Denom) when Denom == 0.0; Denom == 0 -> 0.0;
safe_ratio(Num, Denom) -> Num / Denom.

%%% ============================================================================
%%% Internal Functions - History Management
%%% ============================================================================

truncate_history(List, MaxSize) when length(List) > MaxSize ->
    lists:sublist(List, MaxSize);
truncate_history(List, _MaxSize) ->
    List.

%%% ============================================================================
%%% Internal Functions - Cross-Silo
%%% ============================================================================

emit_signal(_ToSilo, SignalName, Value) ->
    %% Event-driven: publish signal, lc_cross_silo routes to valid destinations
    silo_events:publish_signal(developmental, SignalName, Value).

fetch_incoming_signals() ->
    case whereis(lc_cross_silo) of
        undefined -> #{};
        _Pid -> lc_cross_silo:get_signals_for(developmental)
    end.

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
