%% @doc Cultural Silo - Innovations, traditions, and meme propagation.
%%
%% Part of the Liquid Conglomerate v2 architecture. The Cultural Silo manages:
%%   Innovation tracking and rewards
%%   Tradition formation and maintenance
%%   Meme spread and mutation
%%   Imitation and cultural learning
%%   Cultural diversity metrics
%%
%% == Time Constant ==
%%
%% τ = 35 (slow adaptation for cultural dynamics)
%%
%% == Cross-Silo Signals ==
%%
%% Outgoing:
%%   innovation_impact to task: Innovation contribution to fitness
%%   strategy_innovation to competitive: Strategic novelty rate
%%   plasticity_influence to developmental: Cultural effect on plasticity
%%   information_sharing to communication: Information flow rate
%%
%% Incoming:
%%   exploration_need from task: Need for innovation
%%   strategy_diversity_need from competitive: Need for strategy variety
%%   plasticity_available from developmental: Learning capacity
%%   norm_transmission from social: Norm propagation rate
%%
%% @author R.G. Lefever
%% @copyright 2024-2026 R.G. Lefever
-module(cultural_silo).
-behaviour(gen_server).
-behaviour(lc_silo_behavior).

-include("lc_silos.hrl").
-include("lc_signals.hrl").

%% API
-export([
    start_link/0,
    start_link/1,
    get_params/1,
    record_innovation/4,
    record_imitation/4,
    promote_to_tradition/2,
    get_tradition/2,
    get_innovation_stats/1,
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
-define(TIME_CONSTANT, 35.0).
-define(HISTORY_SIZE, 100).

%% Default actuator values
-define(DEFAULT_PARAMS, #{
    innovation_bonus => 0.15,
    imitation_probability => 0.2,
    tradition_threshold => 5,
    meme_mutation_rate => 0.05,
    conformity_bonus => 0.1,
    cultural_memory_depth => 20,
    innovation_spread_radius => 3,
    tradition_decay_rate => 0.02
}).

%% Actuator bounds
-define(ACTUATOR_BOUNDS, #{
    innovation_bonus => {0.0, 0.5},
    imitation_probability => {0.0, 0.5},
    tradition_threshold => {3, 20},
    meme_mutation_rate => {0.0, 0.2},
    conformity_bonus => {0.0, 0.3},
    cultural_memory_depth => {5, 50},
    innovation_spread_radius => {1, 10},
    tradition_decay_rate => {0.0, 0.1}
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
    innovation_history :: [float()],
    imitation_history :: [float()],
    tradition_count_history :: [non_neg_integer()],

    %% Counters
    innovation_count :: non_neg_integer(),
    imitation_count :: non_neg_integer(),

    %% Cross-silo signal cache
    incoming_signals :: map(),

    %% Previous values for smoothing
    prev_innovation_impact :: float(),
    prev_information_sharing :: float()
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

-spec record_innovation(pid(), term(), term(), float()) -> ok.
record_innovation(Pid, InnovatorId, BehaviorSignature, FitnessDelta) ->
    gen_server:cast(Pid, {record_innovation, InnovatorId, BehaviorSignature, FitnessDelta}).

-spec record_imitation(pid(), term(), term(), boolean()) -> ok.
record_imitation(Pid, ImitatorId, SourceId, Success) ->
    gen_server:cast(Pid, {record_imitation, ImitatorId, SourceId, Success}).

-spec promote_to_tradition(pid(), term()) -> ok | {error, term()}.
promote_to_tradition(Pid, InnovationId) ->
    gen_server:call(Pid, {promote_to_tradition, InnovationId}).

-spec get_tradition(pid(), term()) -> {ok, map()} | not_found.
get_tradition(Pid, TraditionId) ->
    gen_server:call(Pid, {get_tradition, TraditionId}).

-spec get_innovation_stats(pid()) -> map().
get_innovation_stats(Pid) ->
    gen_server:call(Pid, get_innovation_stats).

-spec get_state(pid()) -> map().
get_state(Pid) ->
    gen_server:call(Pid, get_state).

-spec reset(pid()) -> ok.
reset(Pid) ->
    gen_server:call(Pid, reset).

%%% ============================================================================
%%% lc_silo_behavior Callbacks
%%% ============================================================================

get_silo_type() -> cultural.

get_time_constant() -> ?TIME_CONSTANT.

init_silo(Config) ->
    Realm = maps:get(realm, Config, <<"default">>),
    EtsTables = lc_ets_utils:create_tables(cultural, Realm, [
        {innovations, [{keypos, 1}]},
        {traditions, [{keypos, 1}]},
        {memes, [{keypos, 1}]},
        {cultural_lineage, [{keypos, 1}]}
    ]),
    {ok, #{
        ets_tables => EtsTables,
        realm => Realm
    }}.

collect_sensors(State) ->
    #state{
        innovation_history = InnovHistory,
        imitation_history = ImitHistory,
        ets_tables = EtsTables,
        current_params = Params,
        innovation_count = InnovCount,
        imitation_count = ImitCount,
        incoming_signals = InSignals
    } = State,

    %% Innovation metrics
    InnovationsTable = maps:get(innovations, EtsTables),
    TraditionsTable = maps:get(traditions, EtsTables),

    InnovationRate = compute_innovation_rate(InnovCount, InnovHistory),
    TraditionStrength = compute_tradition_strength(TraditionsTable),
    TraditionCount = lc_ets_utils:count(TraditionsTable),
    MaxTraditions = maps:get(cultural_memory_depth, Params, 20),
    NormTraditionCount = lc_silo_behavior:normalize(TraditionCount, 0, MaxTraditions),

    %% Cultural diversity
    CulturalDiversity = compute_cultural_diversity(InnovationsTable),

    %% Imitation metrics
    ImitationSuccessRate = compute_imitation_success_rate(ImitHistory),

    %% Meme dynamics
    MemesTable = maps:get(memes, EtsTables),
    MemeSpreadVelocity = compute_meme_spread_velocity(MemesTable),

    %% Convergence and conformity
    CulturalConvergence = compute_cultural_convergence(InnovationsTable),
    ConformityPressure = 1.0 - CulturalDiversity,

    %% Fitness correlation
    FitnessCulturalCorr = compute_fitness_correlation(InnovationsTable),

    %% Cross-silo signals as sensors
    ExplorationNeed = maps:get(exploration_need, InSignals, 0.5),
    PlasticityAvailable = maps:get(plasticity_available, InSignals, 0.5),

    %% Generic evolutionary signals (behavioral_diversity from any domain)
    %% Phenotype variation in population (0=uniform, 1=diverse)
    BehavioralDiversity = maps:get(behavioral_diversity, InSignals, 0.5),

    #{
        innovation_rate => InnovationRate,
        tradition_strength => TraditionStrength,
        cultural_diversity => CulturalDiversity,
        imitation_success_rate => ImitationSuccessRate,
        meme_spread_velocity => MemeSpreadVelocity,
        cultural_convergence => CulturalConvergence,
        fitness_cultural_correlation => FitnessCulturalCorr,
        tradition_count => NormTraditionCount,
        conformity_pressure => ConformityPressure,
        innovation_count => lc_silo_behavior:normalize(InnovCount, 0, 100),
        imitation_count => lc_silo_behavior:normalize(ImitCount, 0, 100),
        %% Cross-silo signals
        exploration_need => ExplorationNeed,
        cultural_plasticity => PlasticityAvailable,
        %% Generic evolutionary metric from evaluator
        behavioral_diversity => BehavioralDiversity
    }.

apply_actuators(Actuators, State) ->
    BoundedParams = apply_bounds(Actuators, ?ACTUATOR_BOUNDS),
    NewState = State#state{current_params = BoundedParams},
    emit_cross_silo_signals(NewState),
    {ok, NewState}.

compute_reward(State) ->
    Sensors = collect_sensors(State),

    %% Reward components:
    %% 1. High innovation rate (novelty generation)
    InnovationRate = maps:get(innovation_rate, Sensors, 0.5),

    %% 2. Moderate cultural diversity (not too uniform, not chaotic)
    Diversity = maps:get(cultural_diversity, Sensors, 0.5),
    DiversityOptimality = 1.0 - abs(Diversity - 0.5) * 2,

    %% 3. High imitation success (cultural transmission works)
    ImitationSuccess = maps:get(imitation_success_rate, Sensors, 0.5),

    %% 4. Positive fitness-culture correlation
    FitnessCorr = maps:get(fitness_cultural_correlation, Sensors, 0.5),

    %% 5. Tradition stability
    TraditionStrength = maps:get(tradition_strength, Sensors, 0.5),

    %% 6. External alignment: internal cultural diversity should track observed behavioral diversity
    ObservedDiversity = maps:get(behavioral_diversity, Sensors, 0.5),
    DiversityAlignment = 1.0 - abs(Diversity - ObservedDiversity),

    %% Combined reward
    Reward = 0.20 * InnovationRate +
             0.15 * DiversityOptimality +
             0.20 * ImitationSuccess +
             0.15 * FitnessCorr +
             0.15 * TraditionStrength +
             0.15 * DiversityAlignment,

    lc_silo_behavior:clamp(Reward, 0.0, 1.0).

handle_cross_silo_signals(Signals, State) ->
    CurrentSignals = State#state.incoming_signals,
    UpdatedSignals = maps:merge(CurrentSignals, Signals),
    {ok, State#state{incoming_signals = UpdatedSignals}}.

emit_cross_silo_signals(State) ->
    Sensors = collect_sensors(State),

    %% Innovation impact: based on innovation rate and fitness correlation
    InnovationRate = maps:get(innovation_rate, Sensors, 0.5),
    FitnessCorr = maps:get(fitness_cultural_correlation, Sensors, 0.5),
    InnovationImpact = (InnovationRate + FitnessCorr) / 2,

    %% Strategy innovation: based on diversity and innovation
    Diversity = maps:get(cultural_diversity, Sensors, 0.5),
    StrategyInnovation = (InnovationRate + Diversity) / 2,

    %% Plasticity influence: based on imitation and tradition
    ImitationSuccess = maps:get(imitation_success_rate, Sensors, 0.5),
    TraditionStrength = maps:get(tradition_strength, Sensors, 0.5),
    PlasticityInfluence = ImitationSuccess * (1.0 - TraditionStrength * 0.5),

    %% Information sharing: based on meme spread and imitation
    MemeSpread = maps:get(meme_spread_velocity, Sensors, 0.5),
    InformationSharing = (MemeSpread + ImitationSuccess) / 2,

    %% Emit signals
    emit_signal(task, innovation_impact, InnovationImpact),
    emit_signal(competitive, strategy_innovation, StrategyInnovation),
    emit_signal(developmental, plasticity_influence, PlasticityInfluence),
    emit_signal(communication, information_sharing, InformationSharing),
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
    EtsTables = lc_ets_utils:create_tables(cultural, Realm, [
        {innovations, [{keypos, 1}]},
        {traditions, [{keypos, 1}]},
        {memes, [{keypos, 1}]},
        {cultural_lineage, [{keypos, 1}]}
    ]),

    State = #state{
        realm = Realm,
        enabled_levels = EnabledLevels,
        l0_tweann_enabled = L0TweannEnabled,
        l2_enabled = L2Enabled,
        current_params = ?DEFAULT_PARAMS,
        ets_tables = EtsTables,
        innovation_history = [],
        imitation_history = [],
        tradition_count_history = [],
        innovation_count = 0,
        imitation_count = 0,
        incoming_signals = #{},
        prev_innovation_impact = 0.5,
        prev_information_sharing = 0.5
    },

    %% Schedule periodic cross-silo signal update
    erlang:send_after(1000, self(), update_signals),

    {ok, State}.

handle_call(get_params, _From, State) ->
    {reply, State#state.current_params, State};

handle_call({promote_to_tradition, InnovationId}, _From, State) ->
    InnovationsTable = maps:get(innovations, State#state.ets_tables),
    TraditionsTable = maps:get(traditions, State#state.ets_tables),

    Result = promote_innovation_to_tradition(InnovationsTable, TraditionsTable, InnovationId),

    %% Update tradition count history
    NewTraditionHistory = truncate_history(
        [lc_ets_utils:count(TraditionsTable) | State#state.tradition_count_history],
        ?HISTORY_SIZE
    ),

    NewState = State#state{tradition_count_history = NewTraditionHistory},
    {reply, Result, NewState};

handle_call({get_tradition, TraditionId}, _From, State) ->
    TraditionsTable = maps:get(traditions, State#state.ets_tables),
    Result = lc_ets_utils:lookup(TraditionsTable, TraditionId),
    {reply, Result, State};

handle_call(get_innovation_stats, _From, State) ->
    InnovationsTable = maps:get(innovations, State#state.ets_tables),
    TraditionsTable = maps:get(traditions, State#state.ets_tables),
    MemesTable = maps:get(memes, State#state.ets_tables),

    Stats = #{
        innovation_count => State#state.innovation_count,
        imitation_count => State#state.imitation_count,
        tradition_count => lc_ets_utils:count(TraditionsTable),
        active_innovations => lc_ets_utils:count(InnovationsTable),
        active_memes => lc_ets_utils:count(MemesTable)
    },
    {reply, Stats, State};

handle_call(get_state, _From, State) ->
    StateMap = #{
        realm => State#state.realm,
        enabled_levels => State#state.enabled_levels,
        current_params => State#state.current_params,
        innovation_count => State#state.innovation_count,
        imitation_count => State#state.imitation_count,
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
        innovation_history = [],
        imitation_history = [],
        tradition_count_history = [],
        innovation_count = 0,
        imitation_count = 0,
        incoming_signals = #{},
        prev_innovation_impact = 0.5,
        prev_information_sharing = 0.5
    },
    {reply, ok, NewState};

handle_call(_Request, _From, State) ->
    {reply, {error, unknown_request}, State}.

handle_cast({record_innovation, InnovatorId, BehaviorSignature, FitnessDelta}, State) ->
    InnovationsTable = maps:get(innovations, State#state.ets_tables),
    MemesTable = maps:get(memes, State#state.ets_tables),

    %% Record innovation
    InnovationId = erlang:unique_integer([positive]),
    lc_ets_utils:insert(InnovationsTable, InnovationId, #{
        innovator => InnovatorId,
        behavior_signature => BehaviorSignature,
        fitness_delta => FitnessDelta,
        adopter_count => 1
    }),

    %% Create meme from innovation
    MemeId = erlang:unique_integer([positive]),
    lc_ets_utils:insert(MemesTable, MemeId, #{
        source_innovation => InnovationId,
        encoding => BehaviorSignature,
        spread_count => 1,
        fitness_correlation => FitnessDelta
    }),

    %% Update history
    NormFitnessDelta = lc_silo_behavior:normalize(FitnessDelta, -1.0, 1.0),
    NewInnovHistory = truncate_history(
        [NormFitnessDelta | State#state.innovation_history],
        ?HISTORY_SIZE
    ),

    NewState = State#state{
        innovation_count = State#state.innovation_count + 1,
        innovation_history = NewInnovHistory
    },
    {noreply, NewState};

handle_cast({record_imitation, ImitatorId, SourceId, Success}, State) ->
    LineageTable = maps:get(cultural_lineage, State#state.ets_tables),

    %% Record lineage
    lc_ets_utils:insert(LineageTable, ImitatorId, #{
        cultural_parent => SourceId,
        success => Success
    }),

    %% Update history
    SuccessValue = case Success of true -> 1.0; false -> 0.0 end,
    NewImitHistory = truncate_history(
        [SuccessValue | State#state.imitation_history],
        ?HISTORY_SIZE
    ),

    NewState = State#state{
        imitation_count = State#state.imitation_count + 1,
        imitation_history = NewImitHistory
    },
    {noreply, NewState};

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
%%% Internal Functions - Innovation/Tradition
%%% ============================================================================

promote_innovation_to_tradition(InnovationsTable, TraditionsTable, InnovationId) ->
    case lc_ets_utils:lookup(InnovationsTable, InnovationId) of
        {ok, InnovData} ->
            TraditionId = erlang:unique_integer([positive]),
            lc_ets_utils:insert(TraditionsTable, TraditionId, #{
                source_innovation => InnovationId,
                behavior => maps:get(behavior_signature, InnovData),
                adopter_count => maps:get(adopter_count, InnovData, 1),
                generations => 1,
                stability => 1.0
            }),
            ok;
        not_found ->
            {error, innovation_not_found}
    end.

%%% ============================================================================
%%% Internal Functions - Metrics
%%% ============================================================================

safe_mean([]) -> 0.0;
safe_mean(Values) -> lists:sum(Values) / length(Values).

compute_innovation_rate(InnovCount, History) ->
    %% Rate based on recent innovation frequency
    RecentInnovations = length([V || V <- lists:sublist(History, 10), V > 0]),
    BaseRate = lc_silo_behavior:normalize(RecentInnovations, 0, 10),
    TotalRate = lc_silo_behavior:normalize(InnovCount, 0, 100),
    (BaseRate + TotalRate) / 2.

compute_tradition_strength(TraditionsTable) ->
    AllTraditions = lc_ets_utils:all(TraditionsTable),
    compute_strength_from_traditions(AllTraditions).

compute_strength_from_traditions([]) -> 0.0;
compute_strength_from_traditions(Traditions) ->
    Stabilities = [maps:get(stability, Data, 0.5) || {_Id, Data, _Ts} <- Traditions],
    safe_mean(Stabilities).

compute_cultural_diversity(InnovationsTable) ->
    AllInnovations = lc_ets_utils:all(InnovationsTable),
    compute_diversity_from_innovations(AllInnovations).

compute_diversity_from_innovations([]) -> 0.0;
compute_diversity_from_innovations(Innovations) ->
    Signatures = [maps:get(behavior_signature, Data, 0) || {_Id, Data, _Ts} <- Innovations],
    UniqueCount = length(lists:usort(Signatures)),
    TotalCount = length(Signatures),
    lc_silo_behavior:normalize(UniqueCount, 0, max(1, TotalCount)).

compute_imitation_success_rate(History) ->
    safe_mean(History).

compute_meme_spread_velocity(MemesTable) ->
    AllMemes = lc_ets_utils:all(MemesTable),
    compute_velocity_from_memes(AllMemes).

compute_velocity_from_memes([]) -> 0.0;
compute_velocity_from_memes(Memes) ->
    SpreadCounts = [maps:get(spread_count, Data, 1) || {_Id, Data, _Ts} <- Memes],
    AvgSpread = safe_mean(SpreadCounts),
    lc_silo_behavior:normalize(AvgSpread, 0, 10).

compute_cultural_convergence(InnovationsTable) ->
    AllInnovations = lc_ets_utils:all(InnovationsTable),
    compute_convergence_from_innovations(AllInnovations).

compute_convergence_from_innovations([]) -> 0.5;
compute_convergence_from_innovations(Innovations) ->
    %% Convergence is inverse of diversity
    Signatures = [maps:get(behavior_signature, Data, 0) || {_Id, Data, _Ts} <- Innovations],
    UniqueCount = length(lists:usort(Signatures)),
    TotalCount = length(Signatures),
    1.0 - lc_silo_behavior:normalize(UniqueCount, 0, max(1, TotalCount)).

compute_fitness_correlation(InnovationsTable) ->
    AllInnovations = lc_ets_utils:all(InnovationsTable),
    compute_correlation_from_innovations(AllInnovations).

compute_correlation_from_innovations([]) -> 0.5;
compute_correlation_from_innovations(Innovations) ->
    FitnessDeltas = [maps:get(fitness_delta, Data, 0.0) || {_Id, Data, _Ts} <- Innovations],
    PositiveRatio = length([D || D <- FitnessDeltas, D > 0]) / max(1, length(FitnessDeltas)),
    PositiveRatio.

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
    silo_events:publish_signal(cultural, SignalName, Value).

fetch_incoming_signals() ->
    case whereis(lc_cross_silo) of
        undefined -> #{};
        _Pid -> lc_cross_silo:get_signals_for(cultural)
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
