%% @doc Competitive Silo - Opponent archives, Elo ratings, matchmaking.
%%
%% Part of the Liquid Conglomerate v2 architecture. The Competitive Silo manages:
%%   Opponent archive maintenance and selection
%%   Elo rating system for skill tracking
%%   Matchmaking based on skill levels
%%   Arms race detection and mitigation
%%   Strategy diversity monitoring
%%
%% == Time Constant ==
%%
%% τ = 15 (medium-fast adaptation for competitive dynamics)
%%
%% == Cross-Silo Signals ==
%%
%% Outgoing:
%%   competitive_pressure to task: Competitive intensity level
%%   strategy_diversity_need to cultural: Need for strategy variety
%%   arms_race_active to resource: Arms race intensity
%%   coalition_competition to social: Inter-coalition rivalry
%%
%% Incoming:
%%   fitness_pressure from task: Fitness selection pressure
%%   strategy_innovation from cultural: Strategic novelty rate
%%   resource_level from ecological: Resource availability
%%   coalition_structure from social: Coalition organization level
%%
%% @author R.G. Lefever
%% @copyright 2024-2026 R.G. Lefever
-module(competitive_silo).
-behaviour(gen_server).
-behaviour(lc_silo_behavior).

-include("lc_silos.hrl").
-include("lc_signals.hrl").

%% API
-export([
    start_link/0,
    start_link/1,
    get_params/1,
    record_match/5,
    add_to_archive/3,
    get_elo/2,
    update_elo/4,
    get_archive_stats/1,
    select_opponent/2,
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
-define(TIME_CONSTANT, 15.0).
-define(HISTORY_SIZE, 100).
-define(DEFAULT_ELO, 1500.0).
-define(K_FACTOR, 32.0).

%% Default actuator values
-define(DEFAULT_PARAMS, #{
    archive_addition_threshold => 0.7,
    archive_max_size => 100,
    matchmaking_elo_range => 200,
    self_play_ratio => 0.3,
    archive_play_ratio => 0.5,
    exploit_reward => 0.1,
    counter_strategy_reward => 0.2,
    novelty_bonus => 0.1,
    diversity_bonus => 0.1,
    anti_cycle_penalty => 0.05
}).

%% Actuator bounds
-define(ACTUATOR_BOUNDS, #{
    archive_addition_threshold => {0.5, 0.95},
    archive_max_size => {10, 1000},
    matchmaking_elo_range => {50, 500},
    self_play_ratio => {0.0, 1.0},
    archive_play_ratio => {0.0, 1.0},
    exploit_reward => {0.0, 1.0},
    counter_strategy_reward => {0.0, 1.0},
    novelty_bonus => {0.0, 0.5},
    diversity_bonus => {0.0, 0.5},
    anti_cycle_penalty => {0.0, 0.3}
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
    elo_history :: [float()],
    win_rate_history :: [float()],
    match_count :: non_neg_integer(),

    %% Cross-silo signal cache
    incoming_signals :: map(),

    %% Previous values for smoothing
    prev_competitive_pressure :: float(),
    prev_arms_race :: float()
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

-spec record_match(pid(), term(), term(), win | loss | draw, float()) -> ok.
record_match(Pid, PlayerId, OpponentId, Result, EloChange) ->
    gen_server:cast(Pid, {record_match, PlayerId, OpponentId, Result, EloChange}).

-spec add_to_archive(pid(), term(), binary()) -> ok | {error, term()}.
add_to_archive(Pid, IndividualId, NetworkBinary) ->
    gen_server:call(Pid, {add_to_archive, IndividualId, NetworkBinary}).

-spec get_elo(pid(), term()) -> {ok, float()} | not_found.
get_elo(Pid, IndividualId) ->
    gen_server:call(Pid, {get_elo, IndividualId}).

-spec update_elo(pid(), term(), term(), win | loss | draw) -> {ok, float(), float()}.
update_elo(Pid, PlayerId, OpponentId, Result) ->
    gen_server:call(Pid, {update_elo, PlayerId, OpponentId, Result}).

-spec get_archive_stats(pid()) -> map().
get_archive_stats(Pid) ->
    gen_server:call(Pid, get_archive_stats).

-spec select_opponent(pid(), term()) -> {ok, term()} | no_opponents.
select_opponent(Pid, PlayerId) ->
    gen_server:call(Pid, {select_opponent, PlayerId}).

-spec get_state(pid()) -> map().
get_state(Pid) ->
    gen_server:call(Pid, get_state).

-spec reset(pid()) -> ok.
reset(Pid) ->
    gen_server:call(Pid, reset).

%%% ============================================================================
%%% lc_silo_behavior Callbacks
%%% ============================================================================

get_silo_type() -> competitive.

get_time_constant() -> ?TIME_CONSTANT.

init_silo(Config) ->
    Realm = maps:get(realm, Config, <<"default">>),
    EtsTables = lc_ets_utils:create_tables(competitive, Realm, [
        {opponents, [{keypos, 1}]},
        {matches, [{keypos, 1}]},
        {elo_ratings, [{keypos, 1}]},
        {dominance_matrix, [{keypos, 1}]}
    ]),
    {ok, #{
        ets_tables => EtsTables,
        realm => Realm
    }}.

collect_sensors(State) ->
    #state{
        elo_history = EloHistory,
        win_rate_history = WinHistory,
        ets_tables = EtsTables,
        current_params = Params,
        match_count = MatchCount,
        incoming_signals = InSignals
    } = State,

    %% Elo statistics
    EloMean = safe_mean(EloHistory),
    NormEloMean = lc_silo_behavior:normalize(EloMean, 1000, 2000),
    EloVariance = compute_variance(EloHistory),
    NormEloVariance = lc_silo_behavior:normalize(EloVariance, 0, 100000),
    EloTrend = compute_trend(EloHistory),

    %% Win rate statistics
    WinRateMean = safe_mean(WinHistory),

    %% Archive metrics
    OpponentsTable = maps:get(opponents, EtsTables),
    ArchiveSize = lc_ets_utils:count(OpponentsTable),
    MaxArchive = maps:get(archive_max_size, Params, 100),
    ArchiveSizeRatio = lc_silo_behavior:normalize(ArchiveSize, 0, MaxArchive),

    %% Strategy diversity from archive
    StrategyDiversity = compute_strategy_diversity(OpponentsTable),

    %% Match dynamics
    DrawRate = compute_draw_rate(EtsTables),

    %% Arms race detection
    ArmsRaceVelocity = compute_arms_race_velocity(EloHistory),

    %% Cycle detection (rock-paper-scissors patterns)
    CycleStrength = compute_cycle_strength(EtsTables),

    %% Coverage and exploitability
    ArchiveCoverage = compute_archive_coverage(EtsTables),
    Exploitability = compute_exploitability(EtsTables),

    %% Cross-silo signals as sensors
    FitnessPressure = maps:get(fitness_pressure, InSignals, 0.5),
    StrategyInnovation = maps:get(strategy_innovation, InSignals, 0.5),

    #{
        elo_rating_mean => NormEloMean,
        elo_variance => NormEloVariance,
        elo_trend => EloTrend,
        win_rate_vs_archive => WinRateMean,
        win_rate_vs_current => WinRateMean,
        draw_rate => DrawRate,
        strategy_diversity => StrategyDiversity,
        exploitability_score => Exploitability,
        arms_race_velocity => ArmsRaceVelocity,
        cycle_strength => CycleStrength,
        archive_coverage => ArchiveCoverage,
        archive_size_ratio => ArchiveSizeRatio,
        match_count => lc_silo_behavior:normalize(MatchCount, 0, 1000),
        %% External signals
        fitness_pressure => FitnessPressure,
        strategy_innovation => StrategyInnovation
    }.

apply_actuators(Actuators, State) ->
    BoundedParams = apply_bounds(Actuators, ?ACTUATOR_BOUNDS),
    NewState = State#state{current_params = BoundedParams},
    emit_cross_silo_signals(NewState),
    {ok, NewState}.

compute_reward(State) ->
    Sensors = collect_sensors(State),

    %% Reward components:
    %% 1. High strategy diversity (avoid convergence)
    Diversity = maps:get(strategy_diversity, Sensors, 0.5),

    %% 2. Moderate Elo variance (healthy competition)
    EloVariance = maps:get(elo_variance, Sensors, 0.5),
    VarianceOptimality = 1.0 - abs(EloVariance - 0.3) * 2,

    %% 3. Low exploitability
    Exploitability = maps:get(exploitability_score, Sensors, 0.5),
    RobustnessScore = 1.0 - Exploitability,

    %% 4. Low cycle strength (avoid RPS dynamics)
    CycleStrength = maps:get(cycle_strength, Sensors, 0.5),
    NoCycleBonus = 1.0 - CycleStrength,

    %% 5. Good archive coverage
    Coverage = maps:get(archive_coverage, Sensors, 0.5),

    %% Combined reward
    Reward = 0.25 * Diversity +
             0.20 * VarianceOptimality +
             0.20 * RobustnessScore +
             0.15 * NoCycleBonus +
             0.20 * Coverage,

    lc_silo_behavior:clamp(Reward, 0.0, 1.0).

handle_cross_silo_signals(Signals, State) ->
    CurrentSignals = State#state.incoming_signals,
    UpdatedSignals = maps:merge(CurrentSignals, Signals),
    {ok, State#state{incoming_signals = UpdatedSignals}}.

emit_cross_silo_signals(State) ->
    Sensors = collect_sensors(State),

    %% Competitive pressure: based on Elo variance and match activity
    EloVariance = maps:get(elo_variance, Sensors, 0.5),
    MatchActivity = maps:get(match_count, Sensors, 0.5),
    CompetitivePressure = (EloVariance + MatchActivity) / 2,

    %% Strategy diversity need: inverse of current diversity
    Diversity = maps:get(strategy_diversity, Sensors, 0.5),
    DiversityNeed = 1.0 - Diversity,

    %% Arms race active: based on Elo trend and velocity
    ArmsRaceVelocity = maps:get(arms_race_velocity, Sensors, 0.0),
    EloTrend = maps:get(elo_trend, Sensors, 0.5),
    ArmsRaceActive = (ArmsRaceVelocity + abs(EloTrend - 0.5)) / 2,

    %% Coalition competition: based on archive dynamics
    CoalitionCompetition = maps:get(exploitability_score, Sensors, 0.5),

    %% Emit signals
    emit_signal(task, competitive_pressure, CompetitivePressure),
    emit_signal(cultural, strategy_diversity_need, DiversityNeed),
    emit_signal(resource, arms_race_active, ArmsRaceActive),
    emit_signal(social, coalition_competition, CoalitionCompetition),
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
    EtsTables = lc_ets_utils:create_tables(competitive, Realm, [
        {opponents, [{keypos, 1}]},
        {matches, [{keypos, 1}]},
        {elo_ratings, [{keypos, 1}]},
        {dominance_matrix, [{keypos, 1}]}
    ]),

    State = #state{
        realm = Realm,
        enabled_levels = EnabledLevels,
        l0_tweann_enabled = L0TweannEnabled,
        l2_enabled = L2Enabled,
        current_params = ?DEFAULT_PARAMS,
        ets_tables = EtsTables,
        elo_history = [],
        win_rate_history = [],
        match_count = 0,
        incoming_signals = #{},
        prev_competitive_pressure = 0.5,
        prev_arms_race = 0.0
    },

    %% Schedule periodic cross-silo signal update
    erlang:send_after(1000, self(), update_signals),

    {ok, State}.

handle_call(get_params, _From, State) ->
    {reply, State#state.current_params, State};

handle_call({add_to_archive, IndividualId, NetworkBinary}, _From, State) ->
    OpponentsTable = maps:get(opponents, State#state.ets_tables),
    EloTable = maps:get(elo_ratings, State#state.ets_tables),
    Params = State#state.current_params,
    MaxSize = maps:get(archive_max_size, Params, 100),

    %% Check archive size
    CurrentSize = lc_ets_utils:count(OpponentsTable),
    Result = add_to_archive_internal(
        CurrentSize, MaxSize, IndividualId, NetworkBinary,
        OpponentsTable, EloTable
    ),
    {reply, Result, State};

handle_call({get_elo, IndividualId}, _From, State) ->
    EloTable = maps:get(elo_ratings, State#state.ets_tables),
    Result = lookup_elo(EloTable, IndividualId),
    {reply, Result, State};

handle_call({update_elo, PlayerId, OpponentId, Result}, _From, State) ->
    EloTable = maps:get(elo_ratings, State#state.ets_tables),
    {PlayerElo, OpponentElo} = update_elo_internal(EloTable, PlayerId, OpponentId, Result),

    %% Update Elo history
    NewEloHistory = truncate_history(
        [PlayerElo | State#state.elo_history],
        ?HISTORY_SIZE
    ),

    NewState = State#state{elo_history = NewEloHistory},
    {reply, {ok, PlayerElo, OpponentElo}, NewState};

handle_call(get_archive_stats, _From, State) ->
    OpponentsTable = maps:get(opponents, State#state.ets_tables),
    EloTable = maps:get(elo_ratings, State#state.ets_tables),

    Stats = #{
        archive_size => lc_ets_utils:count(OpponentsTable),
        elo_count => lc_ets_utils:count(EloTable),
        elo_mean => compute_elo_mean(EloTable),
        match_count => State#state.match_count
    },
    {reply, Stats, State};

handle_call({select_opponent, PlayerId}, _From, State) ->
    EloTable = maps:get(elo_ratings, State#state.ets_tables),
    OpponentsTable = maps:get(opponents, State#state.ets_tables),
    Params = State#state.current_params,
    EloRange = maps:get(matchmaking_elo_range, Params, 200),

    Result = select_opponent_internal(PlayerId, EloTable, OpponentsTable, EloRange),
    {reply, Result, State};

handle_call(get_state, _From, State) ->
    StateMap = #{
        realm => State#state.realm,
        enabled_levels => State#state.enabled_levels,
        current_params => State#state.current_params,
        match_count => State#state.match_count,
        elo_history_size => length(State#state.elo_history),
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
        elo_history = [],
        win_rate_history = [],
        match_count = 0,
        incoming_signals = #{},
        prev_competitive_pressure = 0.5,
        prev_arms_race = 0.0
    },
    {reply, ok, NewState};

handle_call(_Request, _From, State) ->
    {reply, {error, unknown_request}, State}.

handle_cast({record_match, PlayerId, OpponentId, Result, EloChange}, State) ->
    MatchesTable = maps:get(matches, State#state.ets_tables),
    DominanceTable = maps:get(dominance_matrix, State#state.ets_tables),

    %% Record match
    MatchId = erlang:unique_integer([positive]),
    lc_ets_utils:insert(MatchesTable, MatchId, #{
        player => PlayerId,
        opponent => OpponentId,
        result => Result,
        elo_change => EloChange
    }),

    %% Update dominance matrix
    update_dominance(DominanceTable, PlayerId, OpponentId, Result),

    %% Update win rate history
    WinValue = result_to_win_value(Result),
    NewWinHistory = truncate_history(
        [WinValue | State#state.win_rate_history],
        ?HISTORY_SIZE
    ),

    NewState = State#state{
        match_count = State#state.match_count + 1,
        win_rate_history = NewWinHistory
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
%%% Internal Functions - Elo System
%%% ============================================================================

lookup_elo(EloTable, IndividualId) ->
    case lc_ets_utils:lookup(EloTable, IndividualId) of
        {ok, Data} -> {ok, maps:get(elo, Data, ?DEFAULT_ELO)};
        not_found -> not_found
    end.

update_elo_internal(EloTable, PlayerId, OpponentId, Result) ->
    PlayerElo = get_or_create_elo(EloTable, PlayerId),
    OpponentElo = get_or_create_elo(EloTable, OpponentId),

    %% Calculate expected scores
    PlayerExpected = expected_score(PlayerElo, OpponentElo),
    OpponentExpected = 1.0 - PlayerExpected,

    %% Actual scores
    {PlayerActual, OpponentActual} = result_to_scores(Result),

    %% New Elo ratings
    NewPlayerElo = PlayerElo + ?K_FACTOR * (PlayerActual - PlayerExpected),
    NewOpponentElo = OpponentElo + ?K_FACTOR * (OpponentActual - OpponentExpected),

    %% Store updated ratings
    store_elo(EloTable, PlayerId, NewPlayerElo),
    store_elo(EloTable, OpponentId, NewOpponentElo),

    {NewPlayerElo, NewOpponentElo}.

get_or_create_elo(EloTable, IndividualId) ->
    case lc_ets_utils:lookup(EloTable, IndividualId) of
        {ok, Data} -> maps:get(elo, Data, ?DEFAULT_ELO);
        not_found ->
            store_elo(EloTable, IndividualId, ?DEFAULT_ELO),
            ?DEFAULT_ELO
    end.

store_elo(EloTable, IndividualId, Elo) ->
    lc_ets_utils:insert(EloTable, IndividualId, #{
        elo => Elo,
        games => get_game_count(EloTable, IndividualId) + 1
    }).

get_game_count(EloTable, IndividualId) ->
    case lc_ets_utils:lookup(EloTable, IndividualId) of
        {ok, Data} -> maps:get(games, Data, 0);
        not_found -> 0
    end.

expected_score(PlayerElo, OpponentElo) ->
    1.0 / (1.0 + math:pow(10, (OpponentElo - PlayerElo) / 400)).

result_to_scores(win) -> {1.0, 0.0};
result_to_scores(loss) -> {0.0, 1.0};
result_to_scores(draw) -> {0.5, 0.5}.

result_to_win_value(win) -> 1.0;
result_to_win_value(loss) -> 0.0;
result_to_win_value(draw) -> 0.5.

compute_elo_mean(EloTable) ->
    Elos = lc_ets_utils:fold(
        fun({_Id, Data, _Ts}, Acc) ->
            [maps:get(elo, Data, ?DEFAULT_ELO) | Acc]
        end,
        [],
        EloTable
    ),
    safe_mean(Elos).

%%% ============================================================================
%%% Internal Functions - Archive Management
%%% ============================================================================

add_to_archive_internal(CurrentSize, MaxSize, _Id, _Network, _OppTable, _EloTable)
  when CurrentSize >= MaxSize ->
    {error, archive_full};
add_to_archive_internal(_CurrentSize, _MaxSize, IndividualId, NetworkBinary, OpponentsTable, EloTable) ->
    %% Get Elo for this individual
    Elo = get_or_create_elo(EloTable, IndividualId),

    %% Store in archive
    lc_ets_utils:insert(OpponentsTable, IndividualId, #{
        network => NetworkBinary,
        elo => Elo,
        strategy_signature => compute_strategy_signature(NetworkBinary),
        games_played => 0
    }),
    ok.

compute_strategy_signature(NetworkBinary) ->
    %% Simple hash-based signature for strategy clustering
    erlang:phash2(NetworkBinary, 1000000).

select_opponent_internal(PlayerId, EloTable, OpponentsTable, EloRange) ->
    PlayerElo = get_or_create_elo(EloTable, PlayerId),

    %% Find opponents within Elo range
    Candidates = lc_ets_utils:fold(
        fun({Id, Data, _Ts}, Acc) ->
            OpponentElo = maps:get(elo, Data, ?DEFAULT_ELO),
            EloDiff = abs(OpponentElo - PlayerElo),
            collect_candidate(Id, PlayerId, EloDiff, EloRange, Acc)
        end,
        [],
        OpponentsTable
    ),
    select_from_candidates(Candidates).

collect_candidate(Id, PlayerId, _EloDiff, _EloRange, Acc) when Id =:= PlayerId ->
    Acc;
collect_candidate(Id, _PlayerId, EloDiff, EloRange, Acc) when EloDiff =< EloRange ->
    [Id | Acc];
collect_candidate(_Id, _PlayerId, _EloDiff, _EloRange, Acc) ->
    Acc.

select_from_candidates([]) -> no_opponents;
select_from_candidates(Candidates) ->
    Index = rand:uniform(length(Candidates)),
    {ok, lists:nth(Index, Candidates)}.

%%% ============================================================================
%%% Internal Functions - Metrics
%%% ============================================================================

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

compute_strategy_diversity(OpponentsTable) ->
    Signatures = lc_ets_utils:fold(
        fun({_Id, Data, _Ts}, Acc) ->
            Sig = maps:get(strategy_signature, Data, 0),
            [Sig | Acc]
        end,
        [],
        OpponentsTable
    ),
    compute_diversity_from_signatures(Signatures).

compute_diversity_from_signatures([]) -> 0.0;
compute_diversity_from_signatures(Signatures) ->
    UniqueCount = length(lists:usort(Signatures)),
    TotalCount = length(Signatures),
    lc_silo_behavior:normalize(UniqueCount, 0, TotalCount).

compute_draw_rate(EtsTables) ->
    MatchesTable = maps:get(matches, EtsTables),
    AllMatches = lc_ets_utils:all(MatchesTable),
    compute_draw_rate_from_matches(AllMatches).

compute_draw_rate_from_matches([]) -> 0.0;
compute_draw_rate_from_matches(Matches) ->
    Draws = length([M || {_Id, M, _Ts} <- Matches, maps:get(result, M) =:= draw]),
    Draws / length(Matches).

compute_arms_race_velocity(EloHistory) ->
    %% Arms race detected by rapid Elo inflation
    compute_elo_inflation_rate(EloHistory).

compute_elo_inflation_rate([]) -> 0.0;
compute_elo_inflation_rate(History) when length(History) < 10 -> 0.0;
compute_elo_inflation_rate(History) ->
    Recent = lists:sublist(History, 10),
    Older = lists:sublist(History, 11, 10),
    RecentMean = safe_mean(Recent),
    OlderMean = safe_mean(Older),
    InflationRate = safe_ratio(RecentMean - OlderMean, 100),
    lc_silo_behavior:clamp(InflationRate, 0.0, 1.0).

compute_cycle_strength(EtsTables) ->
    %% Detect rock-paper-scissors patterns in dominance matrix
    DominanceTable = maps:get(dominance_matrix, EtsTables),
    AllDominance = lc_ets_utils:all(DominanceTable),
    detect_cycles(AllDominance).

detect_cycles([]) -> 0.0;
detect_cycles(DominanceEntries) when length(DominanceEntries) < 3 -> 0.0;
detect_cycles(_DominanceEntries) ->
    %% Simplified: return low cycle strength
    %% Full implementation would analyze transitivity violations
    0.1.

compute_archive_coverage(EtsTables) ->
    OpponentsTable = maps:get(opponents, EtsTables),
    Signatures = lc_ets_utils:fold(
        fun({_Id, Data, _Ts}, Acc) ->
            Sig = maps:get(strategy_signature, Data, 0),
            [Sig | Acc]
        end,
        [],
        OpponentsTable
    ),
    compute_coverage_from_signatures(Signatures).

compute_coverage_from_signatures([]) -> 0.0;
compute_coverage_from_signatures(Signatures) ->
    %% Coverage based on spread of strategy signatures
    UniqueCount = length(lists:usort(Signatures)),
    %% Assume 100 is good coverage
    lc_silo_behavior:normalize(UniqueCount, 0, 100).

compute_exploitability(EtsTables) ->
    %% Exploitability based on how often top strategies are beaten
    DominanceTable = maps:get(dominance_matrix, EtsTables),
    AllDominance = lc_ets_utils:all(DominanceTable),
    compute_exploitability_from_dominance(AllDominance).

compute_exploitability_from_dominance([]) -> 0.5;
compute_exploitability_from_dominance(_Entries) ->
    %% Simplified: moderate exploitability
    0.3.

update_dominance(DominanceTable, PlayerId, OpponentId, Result) ->
    Key = {PlayerId, OpponentId},
    case lc_ets_utils:lookup(DominanceTable, Key) of
        {ok, Data} ->
            {PWins, OWins} = {maps:get(player_wins, Data, 0), maps:get(opponent_wins, Data, 0)},
            {NewPWins, NewOWins} = update_win_counts(PWins, OWins, Result),
            lc_ets_utils:insert(DominanceTable, Key, #{
                player_wins => NewPWins,
                opponent_wins => NewOWins
            });
        not_found ->
            {NewPWins, NewOWins} = update_win_counts(0, 0, Result),
            lc_ets_utils:insert(DominanceTable, Key, #{
                player_wins => NewPWins,
                opponent_wins => NewOWins
            })
    end.

update_win_counts(PWins, OWins, win) -> {PWins + 1, OWins};
update_win_counts(PWins, OWins, loss) -> {PWins, OWins + 1};
update_win_counts(PWins, OWins, draw) -> {PWins, OWins}.

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
    silo_events:publish_signal(competitive, SignalName, Value).

fetch_incoming_signals() ->
    case whereis(lc_cross_silo) of
        undefined -> #{};
        _Pid -> lc_cross_silo:get_signals_for(competitive)
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
