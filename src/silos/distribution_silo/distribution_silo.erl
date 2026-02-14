%% @doc Distribution Silo - Mesh networking, island migration, and load balancing.
%%
%% Part of the Liquid Conglomerate v2 architecture. The Distribution Silo manages:
%%   Island statistics and topology
%%   Migration between islands
%%   Load balancing across nodes
%%   Network connectivity
%%   Remote capacity tracking
%%
%% == Time Constant ==
%%
%% τ = 60 (slow adaptation for distribution dynamics)
%%
%% == Cross-Silo Signals ==
%%
%% Outgoing:
%%   network_latency to temporal: Network delay information
%%   remote_capacity to resource: Available remote compute
%%   migration_pressure to competitive: Migration opportunity
%%   island_diversity to ecological: Cross-island diversity
%%
%% Incoming:
%%   local_pressure from resource: Local resource pressure
%%   compute_availability from temporal: Time budget available
%%   arms_race_load from competitive: Competition intensity
%%   abundance_signal from ecological: Resource levels
%%
%% @author R.G. Lefever
%% @copyright 2024-2026 R.G. Lefever
-module(distribution_silo).
-behaviour(gen_server).
-behaviour(lc_silo_behavior).

-include("lc_silos.hrl").
-include("lc_signals.hrl").

%% API
-export([
    start_link/0,
    start_link/1,
    get_params/1,
    register_island/2,
    update_island_stats/3,
    get_island/2,
    record_migration/4,
    get_migrations/1,
    get_distribution_stats/1,
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
-define(TIME_CONSTANT, 60.0).
-define(HISTORY_SIZE, 100).

%% Default actuator values
-define(DEFAULT_PARAMS, #{
    migration_probability => 0.05,
    migration_elite_ratio => 0.1,
    load_balance_threshold => 0.2,
    offload_preference => 0.5,
    island_formation_threshold => 0.8,
    topology_update_interval => 10
}).

%% Actuator bounds
-define(ACTUATOR_BOUNDS, #{
    migration_probability => {0.0, 0.2},
    migration_elite_ratio => {0.0, 0.5},
    load_balance_threshold => {0.1, 0.5},
    offload_preference => {0.0, 1.0},
    island_formation_threshold => {0.5, 0.95},
    topology_update_interval => {1, 50}
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
    load_history :: [float()],
    migration_history :: [float()],

    %% Tracking
    total_migrations :: non_neg_integer(),
    successful_migrations :: non_neg_integer(),

    %% Cross-silo signals
    incoming_signals :: map(),

    %% Computed values
    current_load_balance :: float(),
    avg_latency :: float()
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

-spec register_island(pid(), term()) -> ok.
register_island(Pid, IslandId) ->
    gen_server:cast(Pid, {register_island, IslandId}).

-spec update_island_stats(pid(), term(), map()) -> ok.
update_island_stats(Pid, IslandId, Stats) ->
    gen_server:cast(Pid, {update_island_stats, IslandId, Stats}).

-spec get_island(pid(), term()) -> {ok, map()} | not_found.
get_island(Pid, IslandId) ->
    gen_server:call(Pid, {get_island, IslandId}).

-spec record_migration(pid(), term(), term(), term()) -> ok.
record_migration(Pid, FromIsland, ToIsland, IndividualId) ->
    gen_server:cast(Pid, {record_migration, FromIsland, ToIsland, IndividualId}).

-spec get_migrations(pid()) -> [map()].
get_migrations(Pid) ->
    gen_server:call(Pid, get_migrations).

-spec get_distribution_stats(pid()) -> map().
get_distribution_stats(Pid) ->
    gen_server:call(Pid, get_distribution_stats).

-spec get_state(pid()) -> map().
get_state(Pid) ->
    gen_server:call(Pid, get_state).

-spec reset(pid()) -> ok.
reset(Pid) ->
    gen_server:call(Pid, reset).

%%% ============================================================================
%%% lc_silo_behavior Callbacks
%%% ============================================================================

get_silo_type() -> distribution.

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
        load_history = [],
        migration_history = [],
        total_migrations = 0,
        successful_migrations = 0,
        incoming_signals = #{},
        current_load_balance = 1.0,
        avg_latency = 0.0
    }.

collect_sensors(State) ->
    IslandsTable = maps:get(island_stats, State#state.ets_tables),
    MigrationsTable = maps:get(migration_history, State#state.ets_tables),

    AllIslands = lc_ets_utils:all(IslandsTable),
    AllMigrations = lc_ets_utils:all(MigrationsTable),

    %% Island count
    IslandCount = length(AllIslands),
    MaxIslands = 20,
    IslandCountNorm = lc_silo_behavior:normalize(IslandCount, 0, MaxIslands),

    %% Island diversity (variance in fitness across islands)
    IslandDiversity = compute_island_diversity(AllIslands),

    %% Migration rate
    MigrationRate = compute_migration_rate(AllMigrations),

    %% Load balance (1.0 = perfectly balanced, 0.0 = highly imbalanced)
    LoadBalance = compute_load_balance(AllIslands),

    %% Network latency (normalized)
    NetworkLatency = State#state.avg_latency,

    %% Connectivity (proportion of reachable islands)
    Connectivity = compute_connectivity(AllIslands),

    %% Remote capacity
    RemoteCapacity = compute_remote_capacity(AllIslands),

    %% Local pressure from cross-silo signals
    InSignals = State#state.incoming_signals,
    LocalPressure = maps:get(local_pressure, InSignals, 0.5),
    ComputeAvailability = maps:get(compute_availability, InSignals, 0.5),

    #{
        island_count => IslandCountNorm,
        island_diversity => IslandDiversity,
        migration_rate => MigrationRate,
        load_balance => LoadBalance,
        network_latency => NetworkLatency,
        connectivity => Connectivity,
        remote_capacity => RemoteCapacity,
        %% External signals
        local_pressure => LocalPressure,
        compute_availability => ComputeAvailability
    }.

apply_actuators(Actuators, State) ->
    BoundedParams = apply_bounds(Actuators, ?ACTUATOR_BOUNDS),
    NewState = State#state{current_params = BoundedParams},
    emit_cross_silo_signals(NewState),
    {ok, NewState}.

compute_reward(State) ->
    Sensors = collect_sensors(State),

    %% Reward components:
    %% 1. Good load balance
    LoadBalance = maps:get(load_balance, Sensors, 0.5),

    %% 2. High connectivity
    Connectivity = maps:get(connectivity, Sensors, 0.5),

    %% 3. Low latency (inverted)
    Latency = maps:get(network_latency, Sensors, 0.5),
    LatencyScore = 1.0 - Latency,

    %% 4. Moderate migration (too much = churn, too little = stagnation)
    MigrationRate = maps:get(migration_rate, Sensors, 0.0),
    MigrationOptimality = 1.0 - abs(MigrationRate - 0.1) * 5,

    %% 5. Island diversity (genetic diversity across islands)
    Diversity = maps:get(island_diversity, Sensors, 0.5),

    %% Combined reward
    Reward = (LoadBalance * 0.25 +
              Connectivity * 0.25 +
              LatencyScore * 0.2 +
              max(0.0, MigrationOptimality) * 0.15 +
              Diversity * 0.15),

    {ok, Reward}.

handle_cross_silo_signals(Signals, State) ->
    NewState = State#state{incoming_signals = Signals},
    {ok, NewState}.

emit_cross_silo_signals(State) ->
    Sensors = collect_sensors(State),

    Signals = #{
        network_latency => maps:get(network_latency, Sensors, 0.0),
        remote_capacity => maps:get(remote_capacity, Sensors, 0.5),
        migration_pressure => maps:get(migration_rate, Sensors, 0.0),
        island_diversity => maps:get(island_diversity, Sensors, 0.5)
    },

    %% Event-driven: publish once, lc_cross_silo routes to valid destinations
    silo_events:publish_signals(distribution, Signals),
    ok.

%%% ============================================================================
%%% gen_server Callbacks
%%% ============================================================================

init(Config) ->
    State = init_silo(Config),
    {ok, State}.

handle_call(get_params, _From, State) ->
    {reply, State#state.current_params, State};

handle_call({get_island, IslandId}, _From, State) ->
    IslandsTable = maps:get(island_stats, State#state.ets_tables),
    Result = lc_ets_utils:lookup(IslandsTable, IslandId),
    {reply, Result, State};

handle_call(get_migrations, _From, State) ->
    MigrationsTable = maps:get(migration_history, State#state.ets_tables),
    AllMigrations = lc_ets_utils:all(MigrationsTable),
    Migrations = [Data || {_Id, Data, _Ts} <- AllMigrations],
    {reply, Migrations, State};

handle_call(get_distribution_stats, _From, State) ->
    IslandsTable = maps:get(island_stats, State#state.ets_tables),
    MigrationsTable = maps:get(migration_history, State#state.ets_tables),

    MigrationSuccessRate = case State#state.total_migrations of
        0 -> 0.0;
        N -> State#state.successful_migrations / N
    end,

    Stats = #{
        island_count => lc_ets_utils:count(IslandsTable),
        migration_count => lc_ets_utils:count(MigrationsTable),
        total_migrations => State#state.total_migrations,
        successful_migrations => State#state.successful_migrations,
        migration_success_rate => MigrationSuccessRate,
        current_load_balance => State#state.current_load_balance,
        avg_latency => State#state.avg_latency
    },
    {reply, Stats, State};

handle_call(get_state, _From, State) ->
    StateMap = #{
        realm => State#state.realm,
        enabled_levels => State#state.enabled_levels,
        current_params => State#state.current_params,
        total_migrations => State#state.total_migrations,
        current_load_balance => State#state.current_load_balance,
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
        load_history = [],
        migration_history = [],
        total_migrations = 0,
        successful_migrations = 0,
        current_load_balance = 1.0,
        avg_latency = 0.0
    },
    {reply, ok, NewState};

handle_call(_Request, _From, State) ->
    {reply, {error, unknown_request}, State}.

handle_cast({register_island, IslandId}, State) ->
    IslandsTable = maps:get(island_stats, State#state.ets_tables),

    lc_ets_utils:insert(IslandsTable, IslandId, #{
        fitness_mean => 0.0,
        fitness_variance => 0.0,
        population_size => 0,
        load => 0.0,
        latency => 0.0,
        reachable => true
    }),
    {noreply, State};

handle_cast({update_island_stats, IslandId, Stats}, State) ->
    IslandsTable = maps:get(island_stats, State#state.ets_tables),

    case lc_ets_utils:lookup(IslandsTable, IslandId) of
        {ok, CurrentStats} ->
            NewStats = maps:merge(CurrentStats, Stats),
            lc_ets_utils:insert(IslandsTable, IslandId, NewStats);
        not_found ->
            lc_ets_utils:insert(IslandsTable, IslandId, Stats)
    end,

    %% Update load balance
    AllIslands = lc_ets_utils:all(IslandsTable),
    NewLoadBalance = compute_load_balance(AllIslands),

    NewState = State#state{current_load_balance = NewLoadBalance},
    {noreply, NewState};

handle_cast({record_migration, FromIsland, ToIsland, IndividualId}, State) ->
    MigrationsTable = maps:get(migration_history, State#state.ets_tables),

    MigrationId = erlang:unique_integer([positive]),
    lc_ets_utils:insert(MigrationsTable, MigrationId, #{
        from => FromIsland,
        to => ToIsland,
        individual => IndividualId,
        success => true
    }),

    %% Update migration stats
    NewTotalMigrations = State#state.total_migrations + 1,
    NewSuccessfulMigrations = State#state.successful_migrations + 1,

    %% Update migration history
    NewMigrationHistory = truncate_history(
        [1.0 | State#state.migration_history],
        ?HISTORY_SIZE
    ),

    NewState = State#state{
        total_migrations = NewTotalMigrations,
        successful_migrations = NewSuccessfulMigrations,
        migration_history = NewMigrationHistory
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
        island_stats => ets:new(
            list_to_atom("distribution_islands_" ++ RealmStr),
            [set, public, {keypos, 1}, {read_concurrency, true}]
        ),
        migration_history => ets:new(
            list_to_atom("distribution_migrations_" ++ RealmStr),
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

compute_island_diversity(Islands) ->
    case length(Islands) of
        N when N < 2 -> 0.0;
        _ ->
            FitnessMeans = [maps:get(fitness_mean, Data, 0.0) ||
                           {_Id, Data, _Ts} <- Islands],
            %% Diversity is variance in fitness across islands
            Variance = compute_variance(FitnessMeans),
            %% Normalize to 0-1 range
            min(1.0, Variance * 10)
    end.

compute_migration_rate(Migrations) ->
    case length(Migrations) of
        0 -> 0.0;
        _ ->
            Now = erlang:system_time(millisecond),
            RecentThreshold = 60000, %% 1 minute
            RecentCount = length([1 || {_Id, _Data, Ts} <- Migrations,
                                  Now - Ts < RecentThreshold]),
            min(1.0, RecentCount / 20)
    end.

compute_load_balance(Islands) ->
    case length(Islands) of
        N when N < 2 -> 1.0;
        _ ->
            Loads = [maps:get(load, Data, 0.0) ||
                    {_Id, Data, _Ts} <- Islands],
            TotalLoad = lists:sum(Loads),
            case TotalLoad > 0.0 of
                false -> 1.0;
                true ->
                    Variance = compute_variance(Loads),
                    %% Lower variance = better balance
                    max(0.0, 1.0 - Variance * 5)
            end
    end.

compute_connectivity(Islands) ->
    case length(Islands) of
        0 -> 1.0;
        N ->
            ReachableCount = length([1 || {_Id, Data, _Ts} <- Islands,
                                     maps:get(reachable, Data, true) =:= true]),
            ReachableCount / N
    end.

compute_remote_capacity(Islands) ->
    case length(Islands) of
        0 -> 0.0;
        N ->
            %% Sum of available capacity across islands
            Loads = [maps:get(load, Data, 0.0) ||
                    {_Id, Data, _Ts} <- Islands],
            AvgLoad = lists:sum(Loads) / N,
            %% Remote capacity is inverse of average load
            max(0.0, 1.0 - AvgLoad)
    end.

compute_variance([]) -> 0.0;
compute_variance(Values) ->
    Mean = lists:sum(Values) / length(Values),
    SumSquares = lists:sum([(V - Mean) * (V - Mean) || V <- Values]),
    SumSquares / length(Values).
