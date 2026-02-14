%% @doc Social Silo - Reputation, coalitions, and social networks.
%%
%% Part of the Liquid Conglomerate v2 architecture. The Social Silo manages:
%%   Reputation tracking and dynamics
%%   Coalition formation and stability
%%   Social network structure and metrics
%%   Mentoring relationships
%%   Reciprocity and cooperation patterns
%%
%% == Time Constant ==
%%
%% τ = 25 (medium adaptation for social dynamics)
%%
%% == Cross-Silo Signals ==
%%
%% Outgoing:
%%   selection_influence to task: Social selection weight
%%   norm_transmission to cultural: Norm propagation rate
%%   coalition_structure to competitive: Coalition organization level
%%   trust_network to communication: Trust network density
%%
%% Incoming:
%%   selection_pressure from task: Selection intensity
%%   coalition_competition from competitive: Inter-coalition rivalry
%%   trust_signal from communication: Communication trust level
%%   information_sharing from cultural: Information flow rate
%%
%% @author R.G. Lefever
%% @copyright 2024-2026 R.G. Lefever
-module(social_silo).
-behaviour(gen_server).
-behaviour(lc_silo_behavior).

-include("lc_silos.hrl").
-include("lc_signals.hrl").

%% API
-export([
    start_link/0,
    start_link/1,
    get_params/1,
    update_reputation/3,
    get_reputation/2,
    record_interaction/4,
    form_coalition/3,
    dissolve_coalition/2,
    get_coalition/2,
    get_social_metrics/1,
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
-define(TIME_CONSTANT, 25.0).
-define(HISTORY_SIZE, 100).
-define(DEFAULT_REPUTATION, 0.5).

%% Default actuator values
-define(DEFAULT_PARAMS, #{
    reputation_decay_rate => 0.05,
    coalition_formation_threshold => 0.6,
    mentoring_bonus => 0.2,
    defection_penalty => 0.3,
    cooperation_reward => 0.2,
    network_connection_cost => 0.02,
    coalition_size_limit => 10,
    interaction_memory_depth => 20
}).

%% Actuator bounds
-define(ACTUATOR_BOUNDS, #{
    reputation_decay_rate => {0.0, 0.2},
    coalition_formation_threshold => {0.3, 0.9},
    mentoring_bonus => {0.0, 0.5},
    defection_penalty => {0.0, 0.5},
    cooperation_reward => {0.0, 0.5},
    network_connection_cost => {0.0, 0.1},
    coalition_size_limit => {2, 20},
    interaction_memory_depth => {5, 50}
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
    reputation_history :: [float()],
    coalition_count_history :: [non_neg_integer()],
    interaction_count :: non_neg_integer(),

    %% Cross-silo signal cache
    incoming_signals :: map(),

    %% Previous values for smoothing
    prev_selection_influence :: float(),
    prev_trust_network :: float()
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

-spec update_reputation(pid(), term(), float()) -> ok.
update_reputation(Pid, IndividualId, Delta) ->
    gen_server:cast(Pid, {update_reputation, IndividualId, Delta}).

-spec get_reputation(pid(), term()) -> {ok, float()} | not_found.
get_reputation(Pid, IndividualId) ->
    gen_server:call(Pid, {get_reputation, IndividualId}).

-spec record_interaction(pid(), term(), term(), atom()) -> ok.
record_interaction(Pid, FromId, ToId, InteractionType) ->
    gen_server:cast(Pid, {record_interaction, FromId, ToId, InteractionType}).

-spec form_coalition(pid(), term(), [term()]) -> {ok, term()} | {error, term()}.
form_coalition(Pid, CoalitionId, MemberIds) ->
    gen_server:call(Pid, {form_coalition, CoalitionId, MemberIds}).

-spec dissolve_coalition(pid(), term()) -> ok | not_found.
dissolve_coalition(Pid, CoalitionId) ->
    gen_server:call(Pid, {dissolve_coalition, CoalitionId}).

-spec get_coalition(pid(), term()) -> {ok, map()} | not_found.
get_coalition(Pid, CoalitionId) ->
    gen_server:call(Pid, {get_coalition, CoalitionId}).

-spec get_social_metrics(pid()) -> map().
get_social_metrics(Pid) ->
    gen_server:call(Pid, get_social_metrics).

-spec get_state(pid()) -> map().
get_state(Pid) ->
    gen_server:call(Pid, get_state).

-spec reset(pid()) -> ok.
reset(Pid) ->
    gen_server:call(Pid, reset).

%%% ============================================================================
%%% lc_silo_behavior Callbacks
%%% ============================================================================

get_silo_type() -> social.

get_time_constant() -> ?TIME_CONSTANT.

init_silo(Config) ->
    Realm = maps:get(realm, Config, <<"default">>),
    EtsTables = lc_ets_utils:create_tables(social, Realm, [
        {reputations, [{keypos, 1}]},
        {coalitions, [{keypos, 1}]},
        {interactions, [{keypos, 1}]},
        {social_graph, [{keypos, 1}]}
    ]),
    {ok, #{
        ets_tables => EtsTables,
        realm => Realm
    }}.

collect_sensors(State) ->
    #state{
        reputation_history = RepHistory,
        ets_tables = EtsTables,
        current_params = Params,
        interaction_count = InteractionCount,
        incoming_signals = InSignals
    } = State,

    %% Reputation statistics
    RepTable = maps:get(reputations, EtsTables),
    AllReputations = get_all_reputations(RepTable),
    RepMean = safe_mean(AllReputations),
    RepVariance = compute_variance(AllReputations),
    NormRepMean = lc_silo_behavior:normalize(RepMean, 0.0, 1.0),
    NormRepVariance = lc_silo_behavior:normalize(RepVariance, 0.0, 0.25),

    %% Coalition metrics
    CoalitionsTable = maps:get(coalitions, EtsTables),
    CoalitionCount = lc_ets_utils:count(CoalitionsTable),
    MaxCoalitions = maps:get(coalition_size_limit, Params, 10) * 2,
    NormCoalitionCount = lc_silo_behavior:normalize(CoalitionCount, 0, MaxCoalitions),
    CoalitionStability = compute_coalition_stability(CoalitionsTable),

    %% Network metrics
    GraphTable = maps:get(social_graph, EtsTables),
    NetworkDensity = compute_network_density(GraphTable),
    ClusteringCoeff = compute_clustering_coefficient(GraphTable),

    %% Interaction metrics
    InteractionsTable = maps:get(interactions, EtsTables),
    MentoringSuccess = compute_mentoring_success(InteractionsTable),
    ReciprocityIndex = compute_reciprocity_index(InteractionsTable),

    %% Hierarchy and mobility
    DominanceHierarchy = compute_dominance_hierarchy(AllReputations),
    SocialMobility = compute_social_mobility(RepHistory),

    %% Cross-silo signals as sensors
    SelectionPressure = maps:get(selection_pressure, InSignals, 0.5),
    CoalitionCompetition = maps:get(coalition_competition, InSignals, 0.5),

    #{
        reputation_mean => NormRepMean,
        reputation_variance => NormRepVariance,
        coalition_count => NormCoalitionCount,
        coalition_stability => CoalitionStability,
        network_density => NetworkDensity,
        clustering_coefficient => ClusteringCoeff,
        mentoring_success_rate => MentoringSuccess,
        reciprocity_index => ReciprocityIndex,
        dominance_hierarchy => DominanceHierarchy,
        social_mobility => SocialMobility,
        interaction_count => lc_silo_behavior:normalize(InteractionCount, 0, 1000),
        %% External signals
        selection_pressure => SelectionPressure,
        coalition_competition => CoalitionCompetition
    }.

apply_actuators(Actuators, State) ->
    BoundedParams = apply_bounds(Actuators, ?ACTUATOR_BOUNDS),
    NewState = State#state{current_params = BoundedParams},
    emit_cross_silo_signals(NewState),
    {ok, NewState}.

compute_reward(State) ->
    Sensors = collect_sensors(State),

    %% Reward components:
    %% 1. Moderate reputation variance (not too equal, not too unequal)
    RepVariance = maps:get(reputation_variance, Sensors, 0.5),
    VarianceOptimality = 1.0 - abs(RepVariance - 0.3) * 2,

    %% 2. High coalition stability
    Stability = maps:get(coalition_stability, Sensors, 0.5),

    %% 3. Good network density (connected but not overcrowded)
    Density = maps:get(network_density, Sensors, 0.5),
    DensityOptimality = 1.0 - abs(Density - 0.4) * 2,

    %% 4. High reciprocity
    Reciprocity = maps:get(reciprocity_index, Sensors, 0.5),

    %% 5. High social mobility
    Mobility = maps:get(social_mobility, Sensors, 0.5),

    %% Combined reward
    Reward = 0.20 * VarianceOptimality +
             0.25 * Stability +
             0.20 * DensityOptimality +
             0.20 * Reciprocity +
             0.15 * Mobility,

    lc_silo_behavior:clamp(Reward, 0.0, 1.0).

handle_cross_silo_signals(Signals, State) ->
    CurrentSignals = State#state.incoming_signals,
    UpdatedSignals = maps:merge(CurrentSignals, Signals),
    {ok, State#state{incoming_signals = UpdatedSignals}}.

emit_cross_silo_signals(State) ->
    Sensors = collect_sensors(State),

    %% Selection influence: based on reputation variance and network structure
    RepVariance = maps:get(reputation_variance, Sensors, 0.5),
    Density = maps:get(network_density, Sensors, 0.5),
    SelectionInfluence = (RepVariance + Density) / 2,

    %% Norm transmission: based on coalition stability and clustering
    Stability = maps:get(coalition_stability, Sensors, 0.5),
    Clustering = maps:get(clustering_coefficient, Sensors, 0.5),
    NormTransmission = (Stability + Clustering) / 2,

    %% Coalition structure: based on coalition count and stability
    CoalitionCount = maps:get(coalition_count, Sensors, 0.5),
    CoalitionStructure = (CoalitionCount + Stability) / 2,

    %% Trust network: based on reciprocity and density
    Reciprocity = maps:get(reciprocity_index, Sensors, 0.5),
    TrustNetwork = (Reciprocity + Density) / 2,

    %% Emit signals
    emit_signal(task, selection_influence, SelectionInfluence),
    emit_signal(cultural, norm_transmission, NormTransmission),
    emit_signal(competitive, coalition_structure, CoalitionStructure),
    emit_signal(communication, trust_network, TrustNetwork),
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
    EtsTables = lc_ets_utils:create_tables(social, Realm, [
        {reputations, [{keypos, 1}]},
        {coalitions, [{keypos, 1}]},
        {interactions, [{keypos, 1}]},
        {social_graph, [{keypos, 1}]}
    ]),

    State = #state{
        realm = Realm,
        enabled_levels = EnabledLevels,
        l0_tweann_enabled = L0TweannEnabled,
        l2_enabled = L2Enabled,
        current_params = ?DEFAULT_PARAMS,
        ets_tables = EtsTables,
        reputation_history = [],
        coalition_count_history = [],
        interaction_count = 0,
        incoming_signals = #{},
        prev_selection_influence = 0.5,
        prev_trust_network = 0.5
    },

    %% Schedule periodic cross-silo signal update
    erlang:send_after(1000, self(), update_signals),

    {ok, State}.

handle_call(get_params, _From, State) ->
    {reply, State#state.current_params, State};

handle_call({get_reputation, IndividualId}, _From, State) ->
    RepTable = maps:get(reputations, State#state.ets_tables),
    Result = lookup_reputation(RepTable, IndividualId),
    {reply, Result, State};

handle_call({form_coalition, CoalitionId, MemberIds}, _From, State) ->
    CoalitionsTable = maps:get(coalitions, State#state.ets_tables),
    Params = State#state.current_params,
    MaxSize = maps:get(coalition_size_limit, Params, 10),

    Result = form_coalition_internal(CoalitionsTable, CoalitionId, MemberIds, MaxSize),

    %% Update coalition history
    NewCoalitionHistory = truncate_history(
        [lc_ets_utils:count(CoalitionsTable) | State#state.coalition_count_history],
        ?HISTORY_SIZE
    ),

    NewState = State#state{coalition_count_history = NewCoalitionHistory},
    {reply, Result, NewState};

handle_call({dissolve_coalition, CoalitionId}, _From, State) ->
    CoalitionsTable = maps:get(coalitions, State#state.ets_tables),
    Result = dissolve_coalition_internal(CoalitionsTable, CoalitionId),
    {reply, Result, State};

handle_call({get_coalition, CoalitionId}, _From, State) ->
    CoalitionsTable = maps:get(coalitions, State#state.ets_tables),
    Result = lc_ets_utils:lookup(CoalitionsTable, CoalitionId),
    {reply, Result, State};

handle_call(get_social_metrics, _From, State) ->
    Metrics = #{
        reputation_count => lc_ets_utils:count(maps:get(reputations, State#state.ets_tables)),
        coalition_count => lc_ets_utils:count(maps:get(coalitions, State#state.ets_tables)),
        interaction_count => State#state.interaction_count,
        graph_node_count => lc_ets_utils:count(maps:get(social_graph, State#state.ets_tables))
    },
    {reply, Metrics, State};

handle_call(get_state, _From, State) ->
    StateMap = #{
        realm => State#state.realm,
        enabled_levels => State#state.enabled_levels,
        current_params => State#state.current_params,
        interaction_count => State#state.interaction_count,
        reputation_history_size => length(State#state.reputation_history),
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
        reputation_history = [],
        coalition_count_history = [],
        interaction_count = 0,
        incoming_signals = #{},
        prev_selection_influence = 0.5,
        prev_trust_network = 0.5
    },
    {reply, ok, NewState};

handle_call(_Request, _From, State) ->
    {reply, {error, unknown_request}, State}.

handle_cast({update_reputation, IndividualId, Delta}, State) ->
    RepTable = maps:get(reputations, State#state.ets_tables),
    NewRep = update_reputation_internal(RepTable, IndividualId, Delta),

    %% Update reputation history
    NewRepHistory = truncate_history(
        [NewRep | State#state.reputation_history],
        ?HISTORY_SIZE
    ),

    NewState = State#state{reputation_history = NewRepHistory},
    {noreply, NewState};

handle_cast({record_interaction, FromId, ToId, InteractionType}, State) ->
    InteractionsTable = maps:get(interactions, State#state.ets_tables),
    GraphTable = maps:get(social_graph, State#state.ets_tables),

    %% Record interaction
    InteractionId = erlang:unique_integer([positive]),
    lc_ets_utils:insert(InteractionsTable, InteractionId, #{
        from => FromId,
        to => ToId,
        type => InteractionType
    }),

    %% Update social graph
    update_social_graph(GraphTable, FromId, ToId),

    NewState = State#state{
        interaction_count = State#state.interaction_count + 1
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
%%% Internal Functions - Reputation
%%% ============================================================================

lookup_reputation(RepTable, IndividualId) ->
    case lc_ets_utils:lookup(RepTable, IndividualId) of
        {ok, Data} -> {ok, maps:get(reputation, Data, ?DEFAULT_REPUTATION)};
        not_found -> not_found
    end.

update_reputation_internal(RepTable, IndividualId, Delta) ->
    CurrentRep = get_or_create_reputation(RepTable, IndividualId),
    NewRep = lc_silo_behavior:clamp(CurrentRep + Delta, 0.0, 1.0),
    store_reputation(RepTable, IndividualId, NewRep),
    NewRep.

get_or_create_reputation(RepTable, IndividualId) ->
    case lc_ets_utils:lookup(RepTable, IndividualId) of
        {ok, Data} -> maps:get(reputation, Data, ?DEFAULT_REPUTATION);
        not_found ->
            store_reputation(RepTable, IndividualId, ?DEFAULT_REPUTATION),
            ?DEFAULT_REPUTATION
    end.

store_reputation(RepTable, IndividualId, Reputation) ->
    lc_ets_utils:insert(RepTable, IndividualId, #{
        reputation => Reputation,
        interactions => get_interaction_count_for(RepTable, IndividualId) + 1
    }).

get_interaction_count_for(RepTable, IndividualId) ->
    case lc_ets_utils:lookup(RepTable, IndividualId) of
        {ok, Data} -> maps:get(interactions, Data, 0);
        not_found -> 0
    end.

get_all_reputations(RepTable) ->
    lc_ets_utils:fold(
        fun({_Id, Data, _Ts}, Acc) ->
            [maps:get(reputation, Data, ?DEFAULT_REPUTATION) | Acc]
        end,
        [],
        RepTable
    ).

%%% ============================================================================
%%% Internal Functions - Coalitions
%%% ============================================================================

form_coalition_internal(_Table, _CoalitionId, MemberIds, MaxSize)
  when length(MemberIds) > MaxSize ->
    {error, coalition_too_large};
form_coalition_internal(_Table, _CoalitionId, MemberIds, _MaxSize)
  when length(MemberIds) < 2 ->
    {error, need_at_least_two_members};
form_coalition_internal(Table, CoalitionId, MemberIds, _MaxSize) ->
    lc_ets_utils:insert(Table, CoalitionId, #{
        members => MemberIds,
        formed_at => erlang:system_time(millisecond),
        stability => 1.0
    }),
    {ok, CoalitionId}.

dissolve_coalition_internal(Table, CoalitionId) ->
    case lc_ets_utils:lookup(Table, CoalitionId) of
        {ok, _} ->
            lc_ets_utils:delete(Table, CoalitionId),
            ok;
        not_found ->
            not_found
    end.

compute_coalition_stability(CoalitionsTable) ->
    AllCoalitions = lc_ets_utils:all(CoalitionsTable),
    compute_stability_from_coalitions(AllCoalitions).

compute_stability_from_coalitions([]) -> 0.5;
compute_stability_from_coalitions(Coalitions) ->
    Stabilities = [maps:get(stability, Data, 1.0) || {_Id, Data, _Ts} <- Coalitions],
    safe_mean(Stabilities).

%%% ============================================================================
%%% Internal Functions - Social Graph
%%% ============================================================================

update_social_graph(GraphTable, FromId, ToId) ->
    %% Update FromId's neighbors
    update_neighbors(GraphTable, FromId, ToId),
    %% Update ToId's neighbors
    update_neighbors(GraphTable, ToId, FromId).

update_neighbors(GraphTable, NodeId, NeighborId) ->
    case lc_ets_utils:lookup(GraphTable, NodeId) of
        {ok, Data} ->
            Neighbors = maps:get(neighbors, Data, []),
            NewNeighbors = add_unique(NeighborId, Neighbors),
            lc_ets_utils:insert(GraphTable, NodeId, #{
                neighbors => NewNeighbors,
                centrality => compute_centrality(length(NewNeighbors))
            });
        not_found ->
            lc_ets_utils:insert(GraphTable, NodeId, #{
                neighbors => [NeighborId],
                centrality => compute_centrality(1)
            })
    end.

add_unique(Item, List) ->
    case lists:member(Item, List) of
        true -> List;
        false -> [Item | List]
    end.

compute_centrality(NeighborCount) ->
    %% Simple degree centrality normalized
    lc_silo_behavior:normalize(NeighborCount, 0, 20).

compute_network_density(GraphTable) ->
    AllNodes = lc_ets_utils:all(GraphTable),
    compute_density_from_nodes(AllNodes).

compute_density_from_nodes([]) -> 0.0;
compute_density_from_nodes(Nodes) when length(Nodes) < 2 -> 0.0;
compute_density_from_nodes(Nodes) ->
    N = length(Nodes),
    TotalEdges = lists:foldl(
        fun({_Id, Data, _Ts}, Acc) ->
            Acc + length(maps:get(neighbors, Data, []))
        end,
        0,
        Nodes
    ),
    %% Density = 2 * edges / (n * (n-1))
    MaxEdges = N * (N - 1),
    safe_ratio(TotalEdges, MaxEdges).

compute_clustering_coefficient(GraphTable) ->
    AllNodes = lc_ets_utils:all(GraphTable),
    compute_clustering_from_nodes(AllNodes, GraphTable).

compute_clustering_from_nodes([], _) -> 0.0;
compute_clustering_from_nodes(Nodes, GraphTable) ->
    Coefficients = [compute_local_clustering(Data, GraphTable) || {_Id, Data, _Ts} <- Nodes],
    safe_mean(Coefficients).

compute_local_clustering(NodeData, GraphTable) ->
    Neighbors = maps:get(neighbors, NodeData, []),
    compute_local_clustering_for_neighbors(Neighbors, GraphTable).

compute_local_clustering_for_neighbors(Neighbors, _) when length(Neighbors) < 2 -> 0.0;
compute_local_clustering_for_neighbors(Neighbors, GraphTable) ->
    %% Count edges between neighbors
    NeighborEdges = count_edges_between_neighbors(Neighbors, GraphTable),
    MaxPossible = length(Neighbors) * (length(Neighbors) - 1) / 2,
    safe_ratio(NeighborEdges, MaxPossible).

count_edges_between_neighbors(Neighbors, GraphTable) ->
    lists:foldl(
        fun(N1, Acc) ->
            case lc_ets_utils:lookup(GraphTable, N1) of
                {ok, Data} ->
                    N1Neighbors = maps:get(neighbors, Data, []),
                    CommonCount = length([N || N <- N1Neighbors, lists:member(N, Neighbors), N =/= N1]),
                    Acc + CommonCount;
                not_found ->
                    Acc
            end
        end,
        0,
        Neighbors
    ) div 2.  %% Each edge counted twice

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

compute_mentoring_success(InteractionsTable) ->
    AllInteractions = lc_ets_utils:all(InteractionsTable),
    MentoringInteractions = [I || {_Id, I, _Ts} <- AllInteractions,
                                  maps:get(type, I) =:= mentoring],
    compute_success_rate_for_mentoring(MentoringInteractions, AllInteractions).

compute_success_rate_for_mentoring([], _) -> 0.5;
compute_success_rate_for_mentoring(_, []) -> 0.5;
compute_success_rate_for_mentoring(Mentoring, All) ->
    lc_silo_behavior:normalize(length(Mentoring), 0, length(All)).

compute_reciprocity_index(InteractionsTable) ->
    AllInteractions = lc_ets_utils:all(InteractionsTable),
    compute_reciprocity_from_interactions(AllInteractions).

compute_reciprocity_from_interactions([]) -> 0.5;
compute_reciprocity_from_interactions(Interactions) ->
    %% Build pair counts
    Pairs = [{maps:get(from, I), maps:get(to, I)} || {_Id, I, _Ts} <- Interactions],
    ReciprocatedCount = count_reciprocated(Pairs),
    TotalPairs = length(lists:usort(Pairs)),
    safe_ratio(ReciprocatedCount, TotalPairs).

count_reciprocated(Pairs) ->
    UniqueForward = lists:usort(Pairs),
    lists:foldl(
        fun({A, B}, Acc) ->
            case lists:member({B, A}, UniqueForward) of
                true -> Acc + 1;
                false -> Acc
            end
        end,
        0,
        UniqueForward
    ) div 2.

compute_dominance_hierarchy(Reputations) ->
    %% Gini coefficient of reputations
    compute_gini(Reputations).

compute_gini([]) -> 0.0;
compute_gini([_]) -> 0.0;
compute_gini(Values) ->
    Sorted = lists:sort(Values),
    N = length(Sorted),
    Sum = lists:sum(Values),
    compute_gini_sum(Sorted, N, Sum).

compute_gini_sum(_Sorted, _N, Sum) when Sum == 0.0; Sum == 0 -> 0.0;
compute_gini_sum(Sorted, N, Sum) ->
    IndexedSum = lists:foldl(
        fun({I, V}, Acc) -> Acc + (2 * I - N - 1) * V end,
        0.0,
        lists:zip(lists:seq(1, N), Sorted)
    ),
    IndexedSum / (N * Sum).

compute_social_mobility(RepHistory) ->
    %% Mobility based on variance in reputation changes over time
    compute_mobility_from_history(RepHistory).

compute_mobility_from_history([]) -> 0.5;
compute_mobility_from_history(History) when length(History) < 5 -> 0.5;
compute_mobility_from_history(History) ->
    Recent = lists:sublist(History, 10),
    Changes = compute_changes(Recent),
    ChangeVariance = compute_variance(Changes),
    lc_silo_behavior:normalize(ChangeVariance, 0, 0.1).

compute_changes([]) -> [];
compute_changes([_]) -> [];
compute_changes([A, B | Rest]) ->
    [abs(A - B) | compute_changes([B | Rest])].

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
    silo_events:publish_signal(social, SignalName, Value).

fetch_incoming_signals() ->
    case whereis(lc_cross_silo) of
        undefined -> #{};
        _Pid -> lc_cross_silo:get_signals_for(social)
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
