%% @doc Communication Silo - Signaling, vocabulary evolution, and coordination.
%%
%% Part of the Liquid Conglomerate v2 architecture. The Communication Silo manages:
%%   Vocabulary and signal definitions
%%   Dialect formation
%%   Message exchange tracking
%%   Coordination success
%%   Honesty/deception dynamics
%%
%% == Time Constant ==
%%
%% τ = 55 (slow adaptation for communication dynamics)
%%
%% == Cross-Silo Signals ==
%%
%% Outgoing:
%%   coordination_capability to task: Coordination ability
%%   information_transfer to cultural: Information sharing rate
%%   trust_signal to social: Trust level
%%   strategic_signaling to competitive: Signaling strategy
%%
%% Incoming:
%%   coordination_need from task: Need for coordination
%%   norm_transmission from social: Social norms influence
%%   information_sharing from cultural: Cultural sharing patterns
%%   social_structure from competitive: Competition influence
%%
%% @author R.G. Lefever
%% @copyright 2024-2026 R.G. Lefever
-module(communication_silo).
-behaviour(gen_server).
-behaviour(lc_silo_behavior).

-include("lc_silos.hrl").
-include("lc_signals.hrl").

%% API
-export([
    start_link/0,
    start_link/1,
    get_params/1,
    register_signal/3,
    send_message/4,
    record_coordination/3,
    get_vocabulary/1,
    get_communication_stats/1,
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
-define(TIME_CONSTANT, 55.0).
-define(HISTORY_SIZE, 100).

%% Default actuator values
-define(DEFAULT_PARAMS, #{
    vocabulary_growth_rate => 0.02,
    communication_cost => 0.05,
    lying_penalty => 0.2,
    deception_detection_bonus => 0.1,
    coordination_reward => 0.2,
    message_compression_pressure => 0.1,
    dialect_isolation => 0.3,
    language_mutation_rate => 0.02
}).

%% Actuator bounds
-define(ACTUATOR_BOUNDS, #{
    vocabulary_growth_rate => {0.0, 0.1},
    communication_cost => {0.0, 0.2},
    lying_penalty => {0.0, 0.5},
    deception_detection_bonus => {0.0, 0.3},
    coordination_reward => {0.0, 0.5},
    message_compression_pressure => {0.0, 0.5},
    dialect_isolation => {0.0, 1.0},
    language_mutation_rate => {0.0, 0.1}
}).

-record(state, {
    %% Configuration
    realm :: binary(),
    enabled_levels :: [l0 | l1 | l2],
    l0_tweann_enabled :: boolean(),
    l2_enabled :: boolean(),

    %% Current params (actuators)
    current_params :: map(),

    %% ETS tables for collections
    ets_tables :: #{atom() => ets:tid()},

    %% History windows
    honesty_history :: [float()],
    coordination_history :: [float()],

    %% Tracking
    message_count :: non_neg_integer(),
    coordination_count :: non_neg_integer(),
    successful_coordinations :: non_neg_integer(),

    %% Cross-silo signals
    incoming_signals :: map()
}).

%%% ============================================================================
%%% API Functions
%%% ============================================================================

-spec start_link() -> {ok, pid()} | {error, term()}.
start_link() ->
    start_link(#{}).

-spec start_link(map()) -> {ok, pid()} | {error, term()}.
start_link(Config) ->
    gen_server:start_link({local, ?SERVER}, ?MODULE, Config, []).

-spec get_params(pid()) -> map().
get_params(Pid) ->
    gen_server:call(Pid, get_params).

-spec register_signal(pid(), term(), map()) -> ok.
register_signal(Pid, SignalId, SignalData) ->
    gen_server:cast(Pid, {register_signal, SignalId, SignalData}).

-spec send_message(pid(), term(), term(), map()) -> ok.
send_message(Pid, Sender, Receiver, MessageData) ->
    gen_server:cast(Pid, {send_message, Sender, Receiver, MessageData}).

-spec record_coordination(pid(), [term()], boolean()) -> ok.
record_coordination(Pid, Participants, Success) ->
    gen_server:cast(Pid, {record_coordination, Participants, Success}).

-spec get_vocabulary(pid()) -> [map()].
get_vocabulary(Pid) ->
    gen_server:call(Pid, get_vocabulary).

-spec get_communication_stats(pid()) -> map().
get_communication_stats(Pid) ->
    gen_server:call(Pid, get_communication_stats).

-spec get_state(pid()) -> map().
get_state(Pid) ->
    gen_server:call(Pid, get_state).

-spec reset(pid()) -> ok.
reset(Pid) ->
    gen_server:call(Pid, reset).

%%% ============================================================================
%%% lc_silo_behavior Callbacks
%%% ============================================================================

get_silo_type() -> communication.

get_time_constant() -> ?TIME_CONSTANT.

init_silo(Config) ->
    Realm = maps:get(realm, Config, <<"default">>),
    EnabledLevels = maps:get(enabled_levels, Config, [l0, l1]),

    EtsTables = create_ets_tables(Realm),

    #state{
        realm = Realm,
        enabled_levels = EnabledLevels,
        l0_tweann_enabled = false,
        l2_enabled = false,
        current_params = ?DEFAULT_PARAMS,
        ets_tables = EtsTables,
        honesty_history = [],
        coordination_history = [],
        message_count = 0,
        coordination_count = 0,
        successful_coordinations = 0,
        incoming_signals = #{}
    }.

collect_sensors(State) ->
    VocabTable = maps:get(vocabulary, State#state.ets_tables),
    MessagesTable = maps:get(messages, State#state.ets_tables),
    CoordTable = maps:get(coordination_history, State#state.ets_tables),

    AllSignals = lc_ets_utils:all(VocabTable),
    AllMessages = lc_ets_utils:all(MessagesTable),
    _AllCoordinations = lc_ets_utils:all(CoordTable),

    %% Vocabulary size
    VocabSize = length(AllSignals),
    MaxVocab = 100,
    VocabSizeNorm = lc_silo_behavior:normalize(VocabSize, 0, MaxVocab),

    %% Vocabulary growth rate
    VocabGrowthRate = compute_vocab_growth_rate(AllSignals),

    %% Message complexity
    MessageComplexity = compute_message_complexity(AllMessages),

    %% Communication frequency
    CommFrequency = lc_silo_behavior:normalize(length(AllMessages), 0, 500),

    %% Signal honesty rate
    HonestyRate = compute_honesty_rate(AllMessages),

    %% Deception detection rate
    DeceptionDetectionRate = compute_deception_detection_rate(AllMessages),

    %% Coordination success rate
    CoordSuccessRate = case State#state.coordination_count of
        0 -> 0.5;
        N -> State#state.successful_coordinations / N
    end,

    %% Language stability
    LanguageStability = compute_language_stability(AllSignals),

    %% Dialect count
    DialectCount = compute_dialect_count(AllSignals),
    DialectCountNorm = lc_silo_behavior:normalize(DialectCount, 0, 10),

    %% Compression ratio
    CompressionRatio = compute_compression_ratio(AllMessages),

    %% Cross-silo signals as sensors
    InSignals = State#state.incoming_signals,
    CoordinationNeed = maps:get(coordination_need, InSignals, 0.5),
    NormTransmission = maps:get(norm_transmission, InSignals, 0.5),

    #{
        vocabulary_size => VocabSizeNorm,
        vocabulary_growth_rate => VocabGrowthRate,
        message_complexity => MessageComplexity,
        communication_frequency => CommFrequency,
        signal_honesty_rate => HonestyRate,
        deception_detection_rate => DeceptionDetectionRate,
        coordination_success_rate => CoordSuccessRate,
        language_stability => LanguageStability,
        dialect_count => DialectCountNorm,
        compression_ratio => CompressionRatio,
        %% External signals
        coordination_need => CoordinationNeed,
        norm_transmission => NormTransmission
    }.

apply_actuators(Actuators, State) ->
    BoundedParams = apply_bounds(Actuators, ?ACTUATOR_BOUNDS),
    NewState = State#state{current_params = BoundedParams},
    emit_cross_silo_signals(NewState),
    {ok, NewState}.

compute_reward(State) ->
    Sensors = collect_sensors(State),

    %% Reward components:
    %% 1. High coordination success
    CoordSuccess = maps:get(coordination_success_rate, Sensors, 0.5),

    %% 2. Signal honesty (promotes trust)
    Honesty = maps:get(signal_honesty_rate, Sensors, 0.5),

    %% 3. Efficient communication (good compression)
    Compression = maps:get(compression_ratio, Sensors, 0.5),

    %% 4. Language stability (consistent meanings)
    Stability = maps:get(language_stability, Sensors, 0.5),

    %% 5. Moderate vocabulary size (enough to express but not too complex)
    VocabSize = maps:get(vocabulary_size, Sensors, 0.5),
    VocabOptimality = 1.0 - abs(VocabSize - 0.5) * 2,

    %% Combined reward
    Reward = (CoordSuccess * 0.3 +
              Honesty * 0.25 +
              Compression * 0.15 +
              Stability * 0.15 +
              max(0.0, VocabOptimality) * 0.15),

    {ok, Reward}.

handle_cross_silo_signals(Signals, State) ->
    NewState = State#state{incoming_signals = Signals},
    {ok, NewState}.

emit_cross_silo_signals(State) ->
    Sensors = collect_sensors(State),

    Signals = #{
        coordination_capability => maps:get(coordination_success_rate, Sensors, 0.5),
        information_transfer => maps:get(communication_frequency, Sensors, 0.5),
        trust_signal => maps:get(signal_honesty_rate, Sensors, 0.5),
        strategic_signaling => maps:get(compression_ratio, Sensors, 0.5)
    },

    %% Event-driven: publish once, lc_cross_silo routes to valid destinations
    silo_events:publish_signals(communication, Signals),
    ok.

%%% ============================================================================
%%% gen_server Callbacks
%%% ============================================================================

init(Config) ->
    State = init_silo(Config),
    {ok, State}.

handle_call(get_params, _From, State) ->
    {reply, State#state.current_params, State};

handle_call(get_vocabulary, _From, State) ->
    VocabTable = maps:get(vocabulary, State#state.ets_tables),
    AllSignals = lc_ets_utils:all(VocabTable),
    Vocab = [Data || {_Id, Data, _Ts} <- AllSignals],
    {reply, Vocab, State};

handle_call(get_communication_stats, _From, State) ->
    VocabTable = maps:get(vocabulary, State#state.ets_tables),
    MessagesTable = maps:get(messages, State#state.ets_tables),

    CoordSuccessRate = case State#state.coordination_count of
        0 -> 0.0;
        N -> State#state.successful_coordinations / N
    end,

    Stats = #{
        vocabulary_size => lc_ets_utils:count(VocabTable),
        message_count => lc_ets_utils:count(MessagesTable),
        total_messages_sent => State#state.message_count,
        coordination_count => State#state.coordination_count,
        successful_coordinations => State#state.successful_coordinations,
        coordination_success_rate => CoordSuccessRate
    },
    {reply, Stats, State};

handle_call(get_state, _From, State) ->
    StateMap = #{
        realm => State#state.realm,
        enabled_levels => State#state.enabled_levels,
        current_params => State#state.current_params,
        message_count => State#state.message_count,
        coordination_count => State#state.coordination_count,
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
        honesty_history = [],
        coordination_history = [],
        message_count = 0,
        coordination_count = 0,
        successful_coordinations = 0
    },
    {reply, ok, NewState};

handle_call(_Request, _From, State) ->
    {reply, {error, unknown_request}, State}.

handle_cast({register_signal, SignalId, SignalData}, State) ->
    VocabTable = maps:get(vocabulary, State#state.ets_tables),

    CompleteData = maps:merge(#{
        meaning => undefined,
        usage_count => 0,
        inventors => [],
        dialect => default
    }, SignalData),

    lc_ets_utils:insert(VocabTable, SignalId, CompleteData),
    {noreply, State};

handle_cast({send_message, Sender, Receiver, MessageData}, State) ->
    MessagesTable = maps:get(messages, State#state.ets_tables),

    MessageId = erlang:unique_integer([positive]),
    CompleteData = maps:merge(#{
        sender => Sender,
        receiver => Receiver,
        honest => true,
        detected => false,
        content_length => 1
    }, MessageData),

    lc_ets_utils:insert(MessagesTable, MessageId, CompleteData),

    %% Update honesty history
    Honest = maps:get(honest, CompleteData, true),
    HonestyValue = case Honest of true -> 1.0; false -> 0.0 end,
    NewHonestyHistory = truncate_history(
        [HonestyValue | State#state.honesty_history],
        ?HISTORY_SIZE
    ),

    NewState = State#state{
        message_count = State#state.message_count + 1,
        honesty_history = NewHonestyHistory
    },
    {noreply, NewState};

handle_cast({record_coordination, Participants, Success}, State) ->
    CoordTable = maps:get(coordination_history, State#state.ets_tables),

    CoordId = erlang:unique_integer([positive]),
    lc_ets_utils:insert(CoordTable, CoordId, #{
        participants => Participants,
        success => Success,
        participant_count => length(Participants)
    }),

    %% Update coordination history
    SuccessValue = case Success of true -> 1.0; false -> 0.0 end,
    NewCoordHistory = truncate_history(
        [SuccessValue | State#state.coordination_history],
        ?HISTORY_SIZE
    ),

    NewSuccessful = case Success of
        true -> State#state.successful_coordinations + 1;
        false -> State#state.successful_coordinations
    end,

    NewState = State#state{
        coordination_count = State#state.coordination_count + 1,
        successful_coordinations = NewSuccessful,
        coordination_history = NewCoordHistory
    },
    {noreply, NewState};

handle_cast(_Msg, State) ->
    {noreply, State}.

handle_info(_Info, State) ->
    {noreply, State}.

terminate(_Reason, State) ->
    %% Clean up ETS tables
    maps:foreach(
        fun(_Name, Table) ->
            catch ets:delete(Table)
        end,
        State#state.ets_tables
    ),
    ok.

%%% ============================================================================
%%% Internal Functions
%%% ============================================================================

create_ets_tables(Realm) ->
    RealmStr = binary_to_list(Realm),
    #{
        vocabulary => ets:new(
            list_to_atom("communication_vocab_" ++ RealmStr),
            [set, public, {keypos, 1}, {read_concurrency, true}]
        ),
        messages => ets:new(
            list_to_atom("communication_messages_" ++ RealmStr),
            [set, public, {keypos, 1}, {read_concurrency, true}]
        ),
        dialects => ets:new(
            list_to_atom("communication_dialects_" ++ RealmStr),
            [set, public, {keypos, 1}, {read_concurrency, true}]
        ),
        coordination_history => ets:new(
            list_to_atom("communication_coord_" ++ RealmStr),
            [set, public, {keypos, 1}, {read_concurrency, true}]
        )
    }.

apply_bounds(Params, Bounds) ->
    maps:fold(
        fun(Key, Value, Acc) ->
            case maps:get(Key, Bounds, undefined) of
                {Min, Max} ->
                    BoundedValue = max(Min, min(Max, Value)),
                    maps:put(Key, BoundedValue, Acc);
                undefined ->
                    maps:put(Key, Value, Acc)
            end
        end,
        #{},
        Params
    ).

truncate_history(History, MaxSize) ->
    lists:sublist(History, MaxSize).

compute_vocab_growth_rate(Signals) ->
    case length(Signals) of
        0 -> 0.0;
        _ ->
            Now = erlang:system_time(millisecond),
            RecentThreshold = 60000, %% 1 minute
            RecentCount = length([1 || {_Id, _Data, Ts} <- Signals,
                                  Now - Ts < RecentThreshold]),
            min(1.0, RecentCount / 10)
    end.

compute_message_complexity(Messages) ->
    case length(Messages) of
        0 -> 0.0;
        N ->
            Lengths = [maps:get(content_length, Data, 1) ||
                      {_Id, Data, _Ts} <- Messages],
            AvgLength = lists:sum(Lengths) / N,
            min(1.0, AvgLength / 10)
    end.

compute_honesty_rate(Messages) ->
    case length(Messages) of
        0 -> 1.0;
        N ->
            HonestCount = length([1 || {_Id, Data, _Ts} <- Messages,
                                  maps:get(honest, Data, true) =:= true]),
            HonestCount / N
    end.

compute_deception_detection_rate(Messages) ->
    %% Of dishonest messages, how many were detected
    DishonestMessages = [{Id, Data, Ts} || {Id, Data, Ts} <- Messages,
                         maps:get(honest, Data, true) =:= false],
    case length(DishonestMessages) of
        0 -> 1.0;
        N ->
            DetectedCount = length([1 || {_Id, Data, _Ts} <- DishonestMessages,
                                    maps:get(detected, Data, false) =:= true]),
            DetectedCount / N
    end.

compute_language_stability(Signals) ->
    case length(Signals) of
        0 -> 1.0;
        N ->
            %% Signals with consistent usage are stable
            UsageCounts = [maps:get(usage_count, Data, 0) ||
                          {_Id, Data, _Ts} <- Signals],
            case lists:sum(UsageCounts) of
                0 -> 0.5;
                Total ->
                    %% Higher average usage = more stable
                    AvgUsage = Total / N,
                    min(1.0, AvgUsage / 10)
            end
    end.

compute_dialect_count(Signals) ->
    Dialects = lists:usort([maps:get(dialect, Data, default) ||
                           {_Id, Data, _Ts} <- Signals]),
    length(Dialects).

compute_compression_ratio(Messages) ->
    case length(Messages) of
        0 -> 0.5;
        N ->
            %% Shorter messages with more content = better compression
            Lengths = [maps:get(content_length, Data, 1) ||
                      {_Id, Data, _Ts} <- Messages],
            AvgLength = lists:sum(Lengths) / N,
            %% Assume optimal length is around 3
            1.0 - abs(AvgLength - 3) / 10
    end.
