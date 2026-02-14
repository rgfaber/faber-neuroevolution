%% @doc Regulatory Silo - Gene expression and module activation.
%%
%% Part of the Liquid Conglomerate v2 architecture. The Regulatory Silo manages:
%%   Gene expression states
%%   Module activation/deactivation
%%   Epigenetic marks
%%   Context-dependent expression
%%   Dormant capability management
%%
%% == Time Constant ==
%%
%% τ = 45 (slow adaptation for regulatory dynamics)
%%
%% == Cross-Silo Signals ==
%%
%% Outgoing:
%%   context_awareness to task: Context sensitivity level
%%   expression_flexibility to cultural: Expression adaptability
%%   dormant_potential to competitive: Unexpressed capabilities
%%   expression_cost to morphological: Cost of gene expression
%%
%% Incoming:
%%   context_complexity from task: Environmental complexity
%%   expression_stage from developmental: Developmental expression phase
%%   environmental_context from ecological: Environmental complexity
%%   efficiency_requirement from economic: Efficiency targets
%%
%% @author R.G. Lefever
%% @copyright 2024-2026 R.G. Lefever
-module(regulatory_silo).
-behaviour(gen_server).
-behaviour(lc_silo_behavior).

-include("lc_silos.hrl").
-include("lc_signals.hrl").

%% API
-export([
    start_link/0,
    start_link/1,
    get_params/1,
    update_expression/3,
    get_expression/2,
    activate_module/2,
    deactivate_module/2,
    add_epigenetic_mark/3,
    get_regulatory_stats/1,
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
-define(TIME_CONSTANT, 45.0).
-define(HISTORY_SIZE, 100).

%% Default actuator values
-define(DEFAULT_PARAMS, #{
    expression_threshold => 0.5,
    regulatory_mutation_rate => 0.02,
    context_sensitivity => 0.5,
    module_switching_cost => 0.1,
    dormancy_maintenance_cost => 0.01,
    epigenetic_inheritance_strength => 0.5,
    constitutive_expression_bonus => 0.05,
    regulatory_complexity_penalty => 0.02
}).

%% Actuator bounds
-define(ACTUATOR_BOUNDS, #{
    expression_threshold => {0.0, 1.0},
    regulatory_mutation_rate => {0.0, 0.1},
    context_sensitivity => {0.0, 1.0},
    module_switching_cost => {0.0, 0.5},
    dormancy_maintenance_cost => {0.0, 0.1},
    epigenetic_inheritance_strength => {0.0, 1.0},
    constitutive_expression_bonus => {0.0, 0.2},
    regulatory_complexity_penalty => {0.0, 0.1}
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
    expression_history :: [float()],
    switch_count :: non_neg_integer(),

    %% Cross-silo signal cache
    incoming_signals :: map(),

    %% Previous values for smoothing
    prev_context_awareness :: float(),
    prev_dormant_potential :: float()
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

-spec update_expression(pid(), term(), map()) -> ok.
update_expression(Pid, IndividualId, GeneExpressionMap) ->
    gen_server:cast(Pid, {update_expression, IndividualId, GeneExpressionMap}).

-spec get_expression(pid(), term()) -> {ok, map()} | not_found.
get_expression(Pid, IndividualId) ->
    gen_server:call(Pid, {get_expression, IndividualId}).

-spec activate_module(pid(), term()) -> ok.
activate_module(Pid, ModuleId) ->
    gen_server:cast(Pid, {activate_module, ModuleId}).

-spec deactivate_module(pid(), term()) -> ok.
deactivate_module(Pid, ModuleId) ->
    gen_server:cast(Pid, {deactivate_module, ModuleId}).

-spec add_epigenetic_mark(pid(), term(), atom()) -> ok.
add_epigenetic_mark(Pid, GeneId, MarkType) ->
    gen_server:cast(Pid, {add_epigenetic_mark, GeneId, MarkType}).

-spec get_regulatory_stats(pid()) -> map().
get_regulatory_stats(Pid) ->
    gen_server:call(Pid, get_regulatory_stats).

-spec get_state(pid()) -> map().
get_state(Pid) ->
    gen_server:call(Pid, get_state).

-spec reset(pid()) -> ok.
reset(Pid) ->
    gen_server:call(Pid, reset).

%%% ============================================================================
%%% lc_silo_behavior Callbacks
%%% ============================================================================

get_silo_type() -> regulatory.

get_time_constant() -> ?TIME_CONSTANT.

init_silo(Config) ->
    Realm = maps:get(realm, Config, <<"default">>),
    EtsTables = lc_ets_utils:create_tables(regulatory, Realm, [
        {gene_expression, [{keypos, 1}]},
        {module_states, [{keypos, 1}]},
        {epigenetic_marks, [{keypos, 1}]}
    ]),
    {ok, #{
        ets_tables => EtsTables,
        realm => Realm
    }}.

collect_sensors(State) ->
    #state{
        expression_history = ExprHistory,
        ets_tables = EtsTables,
        switch_count = SwitchCount,
        incoming_signals = InSignals
    } = State,

    %% Gene expression metrics
    ExpressionTable = maps:get(gene_expression, EtsTables),
    ModuleTable = maps:get(module_states, EtsTables),
    EpigeneticTable = maps:get(epigenetic_marks, EtsTables),

    AllExpressions = lc_ets_utils:all(ExpressionTable),
    {ActiveRatio, DormantRatio} = compute_expression_ratios(AllExpressions),

    %% Module metrics
    AllModules = lc_ets_utils:all(ModuleTable),
    ModuleEntropy = compute_module_entropy(AllModules),

    %% Context switching
    ContextSwitchFreq = lc_silo_behavior:normalize(SwitchCount, 0, 100),

    %% Regulatory complexity
    RegulatoryComplexity = compute_regulatory_complexity(AllExpressions, AllModules),

    %% Expression noise
    ExpressionNoise = compute_expression_noise(ExprHistory),

    %% Epigenetic metrics
    AllMarks = lc_ets_utils:all(EpigeneticTable),
    EpigeneticDensity = compute_epigenetic_density(AllMarks, AllExpressions),

    %% Conditional vs constitutive expression
    ConditionalRatio = compute_conditional_ratio(AllExpressions),

    %% Fitness contribution
    FitnessContribution = compute_fitness_contribution(AllExpressions),

    %% Cross-silo signals as sensors
    ContextComplexity = maps:get(context_complexity, InSignals, 0.5),
    ExpressionStage = maps:get(expression_stage, InSignals, 0.5),

    #{
        active_gene_ratio => ActiveRatio,
        dormant_capability_ratio => DormantRatio,
        module_activation_entropy => ModuleEntropy,
        context_switch_frequency => ContextSwitchFreq,
        regulatory_complexity => RegulatoryComplexity,
        expression_noise => ExpressionNoise,
        epigenetic_mark_density => EpigeneticDensity,
        conditional_expression_ratio => ConditionalRatio,
        regulatory_fitness_contribution => FitnessContribution,
        %% External signals
        environmental_context => ContextComplexity,
        developmental_stage => ExpressionStage
    }.

apply_actuators(Actuators, State) ->
    BoundedParams = apply_bounds(Actuators, ?ACTUATOR_BOUNDS),
    NewState = State#state{current_params = BoundedParams},
    emit_cross_silo_signals(NewState),
    {ok, NewState}.

compute_reward(State) ->
    Sensors = collect_sensors(State),

    %% Reward components:
    %% 1. Moderate active gene ratio (not all on, not all off)
    ActiveRatio = maps:get(active_gene_ratio, Sensors, 0.5),
    ActiveOptimality = 1.0 - abs(ActiveRatio - 0.5) * 2,

    %% 2. Low expression noise
    ExpressionNoise = maps:get(expression_noise, Sensors, 0.5),
    NoiseOptimality = 1.0 - ExpressionNoise,

    %% 3. Moderate regulatory complexity
    Complexity = maps:get(regulatory_complexity, Sensors, 0.5),
    ComplexityOptimality = 1.0 - abs(Complexity - 0.4) * 2,

    %% 4. High fitness contribution
    FitnessContribution = maps:get(regulatory_fitness_contribution, Sensors, 0.5),

    %% 5. Moderate context switching (responsive but stable)
    SwitchFreq = maps:get(context_switch_frequency, Sensors, 0.5),
    SwitchOptimality = 1.0 - abs(SwitchFreq - 0.3) * 2,

    %% Combined reward
    Reward = 0.20 * ActiveOptimality +
             0.20 * NoiseOptimality +
             0.20 * ComplexityOptimality +
             0.25 * FitnessContribution +
             0.15 * SwitchOptimality,

    lc_silo_behavior:clamp(Reward, 0.0, 1.0).

handle_cross_silo_signals(Signals, State) ->
    CurrentSignals = State#state.incoming_signals,
    UpdatedSignals = maps:merge(CurrentSignals, Signals),
    {ok, State#state{incoming_signals = UpdatedSignals}}.

emit_cross_silo_signals(State) ->
    Sensors = collect_sensors(State),

    %% Context awareness: based on conditional expression and context sensitivity
    ConditionalRatio = maps:get(conditional_expression_ratio, Sensors, 0.5),
    Params = State#state.current_params,
    ContextSensitivity = maps:get(context_sensitivity, Params, 0.5),
    ContextAwareness = (ConditionalRatio + ContextSensitivity) / 2,

    %% Expression flexibility: inverse of regulatory complexity
    Complexity = maps:get(regulatory_complexity, Sensors, 0.5),
    ExpressionFlexibility = 1.0 - Complexity * 0.5,

    %% Dormant potential
    DormantRatio = maps:get(dormant_capability_ratio, Sensors, 0.5),
    DormantPotential = DormantRatio,

    %% Expression cost
    ActiveRatio = maps:get(active_gene_ratio, Sensors, 0.5),
    ExpressionCost = ActiveRatio * 0.5 + Complexity * 0.5,

    %% Emit signals
    emit_signal(task, context_awareness, ContextAwareness),
    emit_signal(cultural, expression_flexibility, ExpressionFlexibility),
    emit_signal(competitive, dormant_potential, DormantPotential),
    emit_signal(morphological, expression_cost, ExpressionCost),
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
    EtsTables = lc_ets_utils:create_tables(regulatory, Realm, [
        {gene_expression, [{keypos, 1}]},
        {module_states, [{keypos, 1}]},
        {epigenetic_marks, [{keypos, 1}]}
    ]),

    State = #state{
        realm = Realm,
        enabled_levels = EnabledLevels,
        l0_tweann_enabled = L0TweannEnabled,
        l2_enabled = L2Enabled,
        current_params = ?DEFAULT_PARAMS,
        ets_tables = EtsTables,
        expression_history = [],
        switch_count = 0,
        incoming_signals = #{},
        prev_context_awareness = 0.5,
        prev_dormant_potential = 0.5
    },

    %% Schedule periodic cross-silo signal update
    erlang:send_after(1000, self(), update_signals),

    {ok, State}.

handle_call(get_params, _From, State) ->
    {reply, State#state.current_params, State};

handle_call({get_expression, IndividualId}, _From, State) ->
    ExpressionTable = maps:get(gene_expression, State#state.ets_tables),
    Result = lc_ets_utils:lookup(ExpressionTable, IndividualId),
    {reply, Result, State};

handle_call(get_regulatory_stats, _From, State) ->
    ExpressionTable = maps:get(gene_expression, State#state.ets_tables),
    ModuleTable = maps:get(module_states, State#state.ets_tables),
    EpigeneticTable = maps:get(epigenetic_marks, State#state.ets_tables),

    Stats = #{
        expression_count => lc_ets_utils:count(ExpressionTable),
        module_count => lc_ets_utils:count(ModuleTable),
        epigenetic_mark_count => lc_ets_utils:count(EpigeneticTable),
        switch_count => State#state.switch_count
    },
    {reply, Stats, State};

handle_call(get_state, _From, State) ->
    StateMap = #{
        realm => State#state.realm,
        enabled_levels => State#state.enabled_levels,
        current_params => State#state.current_params,
        switch_count => State#state.switch_count,
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
        expression_history = [],
        switch_count = 0,
        incoming_signals = #{},
        prev_context_awareness = 0.5,
        prev_dormant_potential = 0.5
    },
    {reply, ok, NewState};

handle_call(_Request, _From, State) ->
    {reply, {error, unknown_request}, State}.

handle_cast({update_expression, IndividualId, GeneExpressionMap}, State) ->
    ExpressionTable = maps:get(gene_expression, State#state.ets_tables),

    %% Compute expression level
    ActiveCount = maps:fold(
        fun(_Gene, Expressed, Acc) ->
            case Expressed of true -> Acc + 1; false -> Acc end
        end,
        0,
        GeneExpressionMap
    ),
    TotalGenes = maps:size(GeneExpressionMap),
    ExpressionLevel = safe_ratio(ActiveCount, TotalGenes),

    lc_ets_utils:insert(ExpressionTable, IndividualId, #{
        genes => GeneExpressionMap,
        active_count => ActiveCount,
        total_count => TotalGenes,
        expression_level => ExpressionLevel
    }),

    %% Update history
    NewExprHistory = truncate_history(
        [ExpressionLevel | State#state.expression_history],
        ?HISTORY_SIZE
    ),

    NewState = State#state{expression_history = NewExprHistory},
    {noreply, NewState};

handle_cast({activate_module, ModuleId}, State) ->
    ModuleTable = maps:get(module_states, State#state.ets_tables),

    lc_ets_utils:insert(ModuleTable, ModuleId, #{
        active => true,
        activated_at => erlang:system_time(millisecond)
    }),

    NewState = State#state{switch_count = State#state.switch_count + 1},
    {noreply, NewState};

handle_cast({deactivate_module, ModuleId}, State) ->
    ModuleTable = maps:get(module_states, State#state.ets_tables),

    case lc_ets_utils:lookup(ModuleTable, ModuleId) of
        {ok, _} ->
            lc_ets_utils:insert(ModuleTable, ModuleId, #{
                active => false,
                deactivated_at => erlang:system_time(millisecond)
            });
        not_found ->
            ok
    end,

    NewState = State#state{switch_count = State#state.switch_count + 1},
    {noreply, NewState};

handle_cast({add_epigenetic_mark, GeneId, MarkType}, State) ->
    EpigeneticTable = maps:get(epigenetic_marks, State#state.ets_tables),

    lc_ets_utils:insert(EpigeneticTable, GeneId, #{
        mark_type => MarkType,
        added_at => erlang:system_time(millisecond)
    }),
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
%%% Internal Functions - Regulatory Metrics
%%% ============================================================================

compute_expression_ratios(AllExpressions) ->
    compute_ratios_from_expressions(AllExpressions).

compute_ratios_from_expressions([]) -> {0.5, 0.5};
compute_ratios_from_expressions(Expressions) ->
    TotalActive = lists:sum([maps:get(active_count, Data, 0) || {_Id, Data, _Ts} <- Expressions]),
    TotalGenes = lists:sum([maps:get(total_count, Data, 1) || {_Id, Data, _Ts} <- Expressions]),
    ActiveRatio = safe_ratio(TotalActive, TotalGenes),
    DormantRatio = 1.0 - ActiveRatio,
    {ActiveRatio, DormantRatio}.

compute_module_entropy(AllModules) ->
    compute_entropy_from_modules(AllModules).

compute_entropy_from_modules([]) -> 0.5;
compute_entropy_from_modules(Modules) ->
    ActiveCount = length([M || {_Id, M, _Ts} <- Modules, maps:get(active, M, false)]),
    TotalCount = length(Modules),
    ActiveRatio = safe_ratio(ActiveCount, TotalCount),
    %% Binary entropy
    compute_binary_entropy(ActiveRatio).

compute_binary_entropy(P) when P =< 0.0; P >= 1.0 -> 0.0;
compute_binary_entropy(P) ->
    -P * math:log2(P) - (1.0 - P) * math:log2(1.0 - P).

compute_regulatory_complexity(AllExpressions, AllModules) ->
    %% Complexity based on number of regulatory elements
    ExprCount = length(AllExpressions),
    ModCount = length(AllModules),
    TotalElements = ExprCount + ModCount,
    lc_silo_behavior:normalize(TotalElements, 0, 100).

compute_expression_noise(ExprHistory) ->
    compute_variance(ExprHistory).

compute_epigenetic_density(AllMarks, AllExpressions) ->
    MarkCount = length(AllMarks),
    GeneCount = lists:sum([maps:get(total_count, Data, 1) || {_Id, Data, _Ts} <- AllExpressions]),
    safe_ratio(MarkCount, max(1, GeneCount)).

compute_conditional_ratio(AllExpressions) ->
    %% Simplified: assume 50% are conditional by default
    ExprLevels = [maps:get(expression_level, Data, 0.5) || {_Id, Data, _Ts} <- AllExpressions],
    Variance = compute_variance(ExprLevels),
    %% High variance suggests more conditional expression
    lc_silo_behavior:normalize(Variance, 0, 0.25).

compute_fitness_contribution(AllExpressions) ->
    %% Simplified: moderate expression levels contribute best
    ExprLevels = [maps:get(expression_level, Data, 0.5) || {_Id, Data, _Ts} <- AllExpressions],
    AvgLevel = safe_mean(ExprLevels),
    1.0 - abs(AvgLevel - 0.5) * 2.

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
    silo_events:publish_signal(regulatory, SignalName, Value).

fetch_incoming_signals() ->
    case whereis(lc_cross_silo) of
        undefined -> #{};
        _Pid -> lc_cross_silo:get_signals_for(regulatory)
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
