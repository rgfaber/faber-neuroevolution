%% @doc Ecological Silo - Niches, resource competition, and environmental dynamics.
%%
%% Part of the Liquid Conglomerate v2 architecture. The Ecological Silo manages:
%%   Niche formation and occupancy
%%   Resource pools and regeneration
%%   Carrying capacity
%%   Environmental stress
%%   Ecosystem stability
%%
%% == Time Constant ==
%%
%% τ = 50 (slow adaptation for ecological dynamics)
%%
%% == Cross-Silo Signals ==
%%
%% Outgoing:
%%   environmental_pressure to task: Environmental stress level
%%   resource_level to resource: Available resources
%%   stress_signal to developmental: Environmental stress
%%   environmental_context to regulatory: Context for gene expression
%%
%% Incoming:
%%   adaptation_pressure from task: Need for adaptation
%%   abundance_signal from resource: Resource availability
%%   metamorphosis_rate from developmental: Stage transitions
%%   efficiency_score from morphological: Network efficiency
%%
%% @author R.G. Lefever
%% @copyright 2024-2026 R.G. Lefever
-module(ecological_silo).
-behaviour(gen_server).
-behaviour(lc_silo_behavior).

-include("lc_silos.hrl").
-include("lc_signals.hrl").

%% API
-export([
    start_link/0,
    start_link/1,
    get_params/1,
    register_niche/3,
    update_niche/3,
    get_niche/2,
    add_to_niche/3,
    remove_from_niche/3,
    update_resource_pool/3,
    get_ecological_stats/1,
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
-define(TIME_CONSTANT, 50.0).
-define(HISTORY_SIZE, 100).

%% Default actuator values
-define(DEFAULT_PARAMS, #{
    carrying_capacity => 100,
    niche_formation_threshold => 0.5,
    resource_regeneration_rate => 0.05,
    stress_injection_rate => 0.1,
    extinction_threshold => 0.05,
    niche_competition_factor => 0.5,
    environmental_variance => 0.1,
    adaptation_bonus => 0.1
}).

%% Actuator bounds
-define(ACTUATOR_BOUNDS, #{
    carrying_capacity => {10, 1000},
    niche_formation_threshold => {0.3, 0.9},
    resource_regeneration_rate => {0.0, 0.2},
    stress_injection_rate => {0.0, 0.3},
    extinction_threshold => {0.01, 0.2},
    niche_competition_factor => {0.0, 1.0},
    environmental_variance => {0.0, 0.5},
    adaptation_bonus => {0.0, 0.3}
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
    stress_history :: [float()],
    diversity_history :: [float()],

    %% Tracking
    extinction_count :: non_neg_integer(),

    %% Cross-silo signals
    incoming_signals :: map(),

    %% Computed values
    current_stress :: float(),
    current_diversity :: float()
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

-spec register_niche(pid(), term(), map()) -> ok.
register_niche(Pid, NicheId, NicheData) ->
    gen_server:cast(Pid, {register_niche, NicheId, NicheData}).

-spec update_niche(pid(), term(), map()) -> ok.
update_niche(Pid, NicheId, Updates) ->
    gen_server:cast(Pid, {update_niche, NicheId, Updates}).

-spec get_niche(pid(), term()) -> {ok, map()} | not_found.
get_niche(Pid, NicheId) ->
    gen_server:call(Pid, {get_niche, NicheId}).

-spec add_to_niche(pid(), term(), term()) -> ok.
add_to_niche(Pid, NicheId, IndividualId) ->
    gen_server:cast(Pid, {add_to_niche, NicheId, IndividualId}).

-spec remove_from_niche(pid(), term(), term()) -> ok.
remove_from_niche(Pid, NicheId, IndividualId) ->
    gen_server:cast(Pid, {remove_from_niche, NicheId, IndividualId}).

-spec update_resource_pool(pid(), term(), float()) -> ok.
update_resource_pool(Pid, ResourceId, Amount) ->
    gen_server:cast(Pid, {update_resource_pool, ResourceId, Amount}).

-spec get_ecological_stats(pid()) -> map().
get_ecological_stats(Pid) ->
    gen_server:call(Pid, get_ecological_stats).

-spec get_state(pid()) -> map().
get_state(Pid) ->
    gen_server:call(Pid, get_state).

-spec reset(pid()) -> ok.
reset(Pid) ->
    gen_server:call(Pid, reset).

%%% ============================================================================
%%% lc_silo_behavior Callbacks
%%% ============================================================================

get_silo_type() -> ecological.

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
        stress_history = [],
        diversity_history = [],
        extinction_count = 0,
        incoming_signals = #{},
        current_stress = 0.0,
        current_diversity = 0.5
    }.

collect_sensors(State) ->
    NichesTable = maps:get(niches, State#state.ets_tables),
    ResourcesTable = maps:get(resource_pools, State#state.ets_tables),
    Params = State#state.current_params,

    AllNiches = lc_ets_utils:all(NichesTable),
    AllResources = lc_ets_utils:all(ResourcesTable),

    %% Compute niche metrics
    NicheCount = length(AllNiches),
    MaxNiches = 20,
    NicheDiversity = lc_silo_behavior:normalize(NicheCount, 0, MaxNiches),

    %% Compute niche overlap
    NicheOverlap = compute_niche_overlap(AllNiches),

    %% Carrying capacity ratio
    CarryingCapacity = maps:get(carrying_capacity, Params, 100),
    TotalOccupancy = compute_total_occupancy(AllNiches),
    CapacityRatio = lc_silo_behavior:normalize(TotalOccupancy, 0, CarryingCapacity),

    %% Resource abundance
    TotalResources = compute_total_resources(AllResources),
    ResourceAbundance = lc_silo_behavior:normalize(TotalResources, 0, 1000),

    %% Stress level
    StressLevel = State#state.current_stress,

    %% Extinction risk
    ExtinctionRisk = compute_extinction_risk(AllNiches, Params),

    %% Invasion pressure
    InvasionPressure = compute_invasion_pressure(AllNiches),

    %% Ecosystem stability (inverse of variance)
    EcosystemStability = compute_ecosystem_stability(State#state.stress_history),

    %% Adaptation rate
    AdaptationRate = compute_adaptation_rate(AllNiches),

    %% Biodiversity index (Shannon entropy)
    BiodiversityIndex = compute_biodiversity_index(AllNiches),

    %% Cross-silo signals as sensors
    InSignals = State#state.incoming_signals,
    AdaptationPressure = maps:get(adaptation_pressure, InSignals, 0.5),
    AbundanceSignal = maps:get(abundance_signal, InSignals, 0.5),

    #{
        niche_diversity => NicheDiversity,
        niche_overlap => NicheOverlap,
        carrying_capacity_ratio => CapacityRatio,
        resource_abundance => ResourceAbundance,
        stress_level => StressLevel,
        extinction_risk => ExtinctionRisk,
        invasion_pressure => InvasionPressure,
        ecosystem_stability => EcosystemStability,
        adaptation_rate => AdaptationRate,
        biodiversity_index => BiodiversityIndex,
        %% External signals
        adaptation_pressure => AdaptationPressure,
        abundance_signal => AbundanceSignal
    }.

apply_actuators(Actuators, State) ->
    BoundedParams = apply_bounds(Actuators, ?ACTUATOR_BOUNDS),
    NewState = State#state{current_params = BoundedParams},
    emit_cross_silo_signals(NewState),
    {ok, NewState}.

compute_reward(State) ->
    Sensors = collect_sensors(State),

    %% Reward components:
    %% 1. High biodiversity
    Biodiversity = maps:get(biodiversity_index, Sensors, 0.5),

    %% 2. Ecosystem stability
    Stability = maps:get(ecosystem_stability, Sensors, 0.5),

    %% 3. Low extinction risk
    ExtinctionRisk = maps:get(extinction_risk, Sensors, 0.5),
    ExtinctionSafety = 1.0 - ExtinctionRisk,

    %% 4. Moderate stress (some pressure drives adaptation)
    StressLevel = maps:get(stress_level, Sensors, 0.0),
    StressOptimality = 1.0 - abs(StressLevel - 0.3) * 2,

    %% 5. Resource sustainability
    ResourceAbundance = maps:get(resource_abundance, Sensors, 0.5),

    %% Combined reward
    Reward = (Biodiversity * 0.25 +
              Stability * 0.25 +
              ExtinctionSafety * 0.2 +
              max(0.0, StressOptimality) * 0.15 +
              ResourceAbundance * 0.15),

    {ok, Reward}.

handle_cross_silo_signals(Signals, State) ->
    NewState = State#state{incoming_signals = Signals},
    {ok, NewState}.

emit_cross_silo_signals(State) ->
    Sensors = collect_sensors(State),

    Signals = #{
        environmental_pressure => maps:get(stress_level, Sensors, 0.0),
        resource_level => maps:get(resource_abundance, Sensors, 0.5),
        stress_signal => maps:get(stress_level, Sensors, 0.0),
        environmental_context => maps:get(biodiversity_index, Sensors, 0.5)
    },

    %% Event-driven: publish once, lc_cross_silo routes to valid destinations
    silo_events:publish_signals(ecological, Signals),
    ok.

%%% ============================================================================
%%% gen_server Callbacks
%%% ============================================================================

init(Config) ->
    State = init_silo(Config),
    {ok, State}.

handle_call(get_params, _From, State) ->
    {reply, State#state.current_params, State};

handle_call({get_niche, NicheId}, _From, State) ->
    NichesTable = maps:get(niches, State#state.ets_tables),
    Result = lc_ets_utils:lookup(NichesTable, NicheId),
    {reply, Result, State};

handle_call(get_ecological_stats, _From, State) ->
    NichesTable = maps:get(niches, State#state.ets_tables),
    ResourcesTable = maps:get(resource_pools, State#state.ets_tables),

    Stats = #{
        niche_count => lc_ets_utils:count(NichesTable),
        resource_pool_count => lc_ets_utils:count(ResourcesTable),
        extinction_count => State#state.extinction_count,
        current_stress => State#state.current_stress,
        current_diversity => State#state.current_diversity
    },
    {reply, Stats, State};

handle_call(get_state, _From, State) ->
    StateMap = #{
        realm => State#state.realm,
        enabled_levels => State#state.enabled_levels,
        current_params => State#state.current_params,
        extinction_count => State#state.extinction_count,
        current_stress => State#state.current_stress,
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
        stress_history = [],
        diversity_history = [],
        extinction_count = 0,
        current_stress = 0.0,
        current_diversity = 0.5
    },
    {reply, ok, NewState};

handle_call(_Request, _From, State) ->
    {reply, {error, unknown_request}, State}.

handle_cast({register_niche, NicheId, NicheData}, State) ->
    NichesTable = maps:get(niches, State#state.ets_tables),

    CompleteData = maps:merge(#{
        occupants => [],
        capacity => 10,
        fitness_range => {0.0, 1.0},
        resource_type => general
    }, NicheData),

    lc_ets_utils:insert(NichesTable, NicheId, CompleteData),

    %% Update diversity
    AllNiches = lc_ets_utils:all(NichesTable),
    NewDiversity = compute_biodiversity_index(AllNiches),

    NewState = State#state{current_diversity = NewDiversity},
    {noreply, NewState};

handle_cast({update_niche, NicheId, Updates}, State) ->
    NichesTable = maps:get(niches, State#state.ets_tables),

    case lc_ets_utils:lookup(NichesTable, NicheId) of
        {ok, CurrentData} ->
            NewData = maps:merge(CurrentData, Updates),
            lc_ets_utils:insert(NichesTable, NicheId, NewData);
        not_found ->
            ok
    end,
    {noreply, State};

handle_cast({add_to_niche, NicheId, IndividualId}, State) ->
    NichesTable = maps:get(niches, State#state.ets_tables),

    case lc_ets_utils:lookup(NichesTable, NicheId) of
        {ok, NicheData} ->
            Occupants = maps:get(occupants, NicheData, []),
            NewOccupants = [IndividualId | lists:delete(IndividualId, Occupants)],
            NewData = NicheData#{occupants => NewOccupants},
            lc_ets_utils:insert(NichesTable, NicheId, NewData);
        not_found ->
            ok
    end,
    {noreply, State};

handle_cast({remove_from_niche, NicheId, IndividualId}, State) ->
    NichesTable = maps:get(niches, State#state.ets_tables),

    case lc_ets_utils:lookup(NichesTable, NicheId) of
        {ok, NicheData} ->
            Occupants = maps:get(occupants, NicheData, []),
            NewOccupants = lists:delete(IndividualId, Occupants),
            NewData = NicheData#{occupants => NewOccupants},
            lc_ets_utils:insert(NichesTable, NicheId, NewData);
        not_found ->
            ok
    end,
    {noreply, State};

handle_cast({update_resource_pool, ResourceId, Amount}, State) ->
    ResourcesTable = maps:get(resource_pools, State#state.ets_tables),

    lc_ets_utils:insert(ResourcesTable, ResourceId, #{
        amount => Amount,
        regeneration_rate => maps:get(resource_regeneration_rate,
                                      State#state.current_params, 0.05)
    }),
    {noreply, State};

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
        niches => ets:new(
            list_to_atom("ecological_niches_" ++ RealmStr),
            [set, public, {keypos, 1}, {read_concurrency, true}]
        ),
        resource_pools => ets:new(
            list_to_atom("ecological_resources_" ++ RealmStr),
            [set, public, {keypos, 1}, {read_concurrency, true}]
        ),
        environmental_history => ets:new(
            list_to_atom("ecological_history_" ++ RealmStr),
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

compute_niche_overlap(Niches) ->
    case length(Niches) of
        N when N < 2 -> 0.0;
        _ ->
            %% Simplified overlap based on occupancy patterns
            Occupancies = [length(maps:get(occupants, Data, [])) ||
                          {_Id, Data, _Ts} <- Niches],
            TotalOccupancy = lists:sum(Occupancies),
            case TotalOccupancy of
                0 -> 0.0;
                _ ->
                    %% More niches with similar occupancy = more overlap potential
                    Variance = compute_variance(Occupancies),
                    max(0.0, 1.0 - Variance / 10.0)
            end
    end.

compute_total_occupancy(Niches) ->
    lists:sum([length(maps:get(occupants, Data, [])) ||
               {_Id, Data, _Ts} <- Niches]).

compute_total_resources(Resources) ->
    lists:sum([maps:get(amount, Data, 0.0) ||
               {_Id, Data, _Ts} <- Resources]).

compute_extinction_risk(Niches, _Params) ->
    case length(Niches) of
        0 -> 0.0;
        N ->
            %% Count niches with very few occupants
            AtRisk = length([1 || {_Id, Data, _Ts} <- Niches,
                            length(maps:get(occupants, Data, [])) < 2]),
            AtRisk / N
    end.

compute_invasion_pressure(Niches) ->
    case length(Niches) of
        0 -> 0.0;
        _ ->
            %% Recent niches indicate invasion
            Now = erlang:system_time(millisecond),
            RecentThreshold = 60000, %% 1 minute
            RecentCount = length([1 || {_Id, _Data, Ts} <- Niches,
                                  Now - Ts < RecentThreshold]),
            min(1.0, RecentCount / 5)
    end.

compute_ecosystem_stability(StressHistory) ->
    case length(StressHistory) of
        0 -> 1.0;
        _ ->
            Variance = compute_variance(StressHistory),
            max(0.0, 1.0 - Variance * 5)
    end.

compute_adaptation_rate(Niches) ->
    case length(Niches) of
        0 -> 0.0;
        N ->
            %% Niches with changing occupancy indicate adaptation
            ActiveNiches = length([N || {_Id, Data, _Ts} <- Niches,
                                   length(maps:get(occupants, Data, [])) > 0]),
            ActiveNiches / N
    end.

compute_biodiversity_index(Niches) ->
    case length(Niches) of
        0 -> 0.0;
        N ->
            %% Shannon entropy approximation
            Occupancies = [length(maps:get(occupants, Data, [])) ||
                          {_Id, Data, _Ts} <- Niches],
            Total = lists:sum(Occupancies),
            case Total of
                0 -> 0.0;
                _ ->
                    Proportions = [O / Total || O <- Occupancies, O > 0],
                    Entropy = -lists:sum([P * math:log(P) || P <- Proportions]),
                    %% Normalize by max entropy
                    MaxEntropy = math:log(N),
                    case MaxEntropy > 0.0 of
                        false -> 0.0;
                        true -> min(1.0, Entropy / MaxEntropy)
                    end
            end
    end.

compute_variance([]) -> 0.0;
compute_variance(Values) ->
    Mean = lists:sum(Values) / length(Values),
    SumSquares = lists:sum([(V - Mean) * (V - Mean) || V <- Values]),
    SumSquares / length(Values).
