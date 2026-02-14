%% @doc Distribution Silo L0 Actuators - Denormalizes and applies TWEANN outputs.
%%
%% Part of the Liquid Conglomerate v2 architecture. This module takes the
%% normalized output vector from the L0 TWEANN and converts it into actual
%% distribution control signals affecting load balancing and migration.
%%
%% == Responsibilities ==
%%
%% 1. Convert TWEANN outputs (0.0-1.0) to distribution parameter ranges
%% 2. Apply outputs to island topology and load balancer
%% 3. Control migration rates and strategies
%% 4. Track applied values for debugging/monitoring
%%
%% == Usage ==
%%
%% %% Start the actuator controller
%% {ok, Pid} = distribution_l0_actuators:start_link(Config),
%%
%% %% Apply TWEANN output vector
%% distribution_l0_actuators:apply_output_vector(OutputVector),
%%
%% %% Get current actuator values
%% Values = distribution_l0_actuators:get_actuator_values(),
%% %% Returns: #{local_vs_remote_ratio => 0.7, migration_rate => 0.05, ...}
%%
%% @author R.G. Lefever
%% @copyright 2024-2026 R.G. Lefever
-module(distribution_l0_actuators).
-behaviour(gen_server).

%% API
-export([
    start_link/0,
    start_link/1,
    apply_outputs/1,
    apply_outputs/2,
    apply_output_vector/1,
    apply_output_vector/2,
    get_actuator_values/0,
    get_actuator_values/1,
    get_raw_outputs/0,
    get_raw_outputs/1,
    get_distribution_params/0,
    get_distribution_params/1
]).

%% gen_server callbacks
-export([
    init/1,
    handle_call/3,
    handle_cast/2,
    handle_info/2,
    terminate/2
]).

-define(SERVER, ?MODULE).

-record(state, {
    %% Configuration
    hyperparameters :: map(),  % L0 hyperparameters (from L1)

    %% Current values
    raw_outputs :: [float()],           % Raw TWEANN outputs (0.0-1.0)
    actuator_values :: map(),           % Denormalized actuator values

    %% Target callbacks
    load_balancer_callback :: fun((map()) -> ok) | undefined,
    migration_callback :: fun((map()) -> ok) | undefined,
    topology_callback :: fun((map()) -> ok) | undefined
}).

%%% ============================================================================
%%% API Functions
%%% ============================================================================

%% @doc Start the actuator controller with default configuration.
-spec start_link() -> {ok, pid()} | ignore | {error, term()}.
start_link() ->
    start_link(#{}).

%% @doc Start the actuator controller with custom configuration.
-spec start_link(map()) -> {ok, pid()} | ignore | {error, term()}.
start_link(Config) ->
    gen_server:start_link({local, ?SERVER}, ?MODULE, Config, []).

%% @doc Apply TWEANN outputs as a map.
-spec apply_outputs(map()) -> ok.
apply_outputs(OutputMap) ->
    gen_server:cast(?SERVER, {apply_outputs, OutputMap}).

%% @doc Apply TWEANN outputs as a map (specific server).
-spec apply_outputs(pid(), map()) -> ok.
apply_outputs(Pid, OutputMap) ->
    gen_server:cast(Pid, {apply_outputs, OutputMap}).

%% @doc Apply TWEANN output vector (ordered list).
-spec apply_output_vector([float()]) -> ok.
apply_output_vector(OutputVector) ->
    gen_server:cast(?SERVER, {apply_output_vector, OutputVector}).

%% @doc Apply TWEANN output vector (specific server).
-spec apply_output_vector(pid(), [float()]) -> ok.
apply_output_vector(Pid, OutputVector) ->
    gen_server:cast(Pid, {apply_output_vector, OutputVector}).

%% @doc Get current denormalized actuator values.
-spec get_actuator_values() -> map().
get_actuator_values() ->
    gen_server:call(?SERVER, get_actuator_values).

%% @doc Get current denormalized actuator values (specific server).
-spec get_actuator_values(pid()) -> map().
get_actuator_values(Pid) ->
    gen_server:call(Pid, get_actuator_values).

%% @doc Get raw TWEANN outputs (before denormalization).
-spec get_raw_outputs() -> [float()].
get_raw_outputs() ->
    gen_server:call(?SERVER, get_raw_outputs).

%% @doc Get raw TWEANN outputs (specific server).
-spec get_raw_outputs(pid()) -> [float()].
get_raw_outputs(Pid) ->
    gen_server:call(Pid, get_raw_outputs).

%% @doc Get distribution parameters ready for distribution components.
-spec get_distribution_params() -> map().
get_distribution_params() ->
    gen_server:call(?SERVER, get_distribution_params).

%% @doc Get distribution parameters (specific server).
-spec get_distribution_params(pid()) -> map().
get_distribution_params(Pid) ->
    gen_server:call(Pid, get_distribution_params).

%%% ============================================================================
%%% gen_server Callbacks
%%% ============================================================================

init(Config) ->
    Hyperparams = maps:get(hyperparameters, Config,
                           distribution_l0_morphology:get_l0_defaults()),
    LoadBalancerCallback = maps:get(load_balancer_callback, Config, undefined),
    MigrationCallback = maps:get(migration_callback, Config, undefined),
    TopologyCallback = maps:get(topology_callback, Config, undefined),

    State = #state{
        hyperparameters = Hyperparams,
        raw_outputs = lists:duplicate(distribution_l0_morphology:actuator_count(), 0.5),
        actuator_values = initial_actuator_values(Hyperparams),
        load_balancer_callback = LoadBalancerCallback,
        migration_callback = MigrationCallback,
        topology_callback = TopologyCallback
    },

    {ok, State}.

handle_call(get_actuator_values, _From, State) ->
    {reply, State#state.actuator_values, State};

handle_call(get_raw_outputs, _From, State) ->
    {reply, State#state.raw_outputs, State};

handle_call(get_distribution_params, _From, State) ->
    Params = build_distribution_params(State#state.actuator_values, State),
    {reply, Params, State};

handle_call(_Request, _From, State) ->
    {reply, {error, unknown_request}, State}.

handle_cast({apply_output_vector, OutputVector}, State) ->
    NewState = process_output_vector(OutputVector, State),
    {noreply, NewState};

handle_cast({apply_outputs, OutputMap}, State) ->
    %% Convert map to ordered vector
    ActuatorNames = distribution_l0_morphology:actuator_names(),
    OutputVector = [maps:get(Name, OutputMap, 0.5) || Name <- ActuatorNames],
    NewState = process_output_vector(OutputVector, State),
    {noreply, NewState};

handle_cast(_Msg, State) ->
    {noreply, State}.

handle_info(_Info, State) ->
    {noreply, State}.

terminate(_Reason, _State) ->
    ok.

%%% ============================================================================
%%% Internal Functions - Output Processing
%%% ============================================================================

%% @private Process output vector and apply to targets.
process_output_vector(OutputVector, State) ->
    ActuatorNames = distribution_l0_morphology:actuator_names(),
    Hyperparams = State#state.hyperparameters,

    %% Denormalize each output
    ActuatorValues = denormalize_outputs(ActuatorNames, OutputVector, Hyperparams),

    %% Apply to targets
    apply_to_targets(ActuatorValues, State),

    State#state{
        raw_outputs = OutputVector,
        actuator_values = ActuatorValues
    }.

%% @private Denormalize outputs based on actuator specs.
denormalize_outputs(Names, Values, Hyperparams) ->
    lists:foldl(
        fun({Name, RawValue}, Acc) ->
            DenormValue = denormalize_actuator(Name, RawValue, Hyperparams),
            maps:put(Name, DenormValue, Acc)
        end,
        #{},
        lists:zip(Names, Values)
    ).

%% @private Denormalize a single actuator value.
denormalize_actuator(local_vs_remote_ratio, RawValue, Hyperparams) ->
    %% Range: 0.0 to 1.0, biased by local_preference_base
    LocalPrefBase = maps:get(local_preference_base, Hyperparams, 0.8),
    %% Blend towards local preference
    BaseValue = lerp(RawValue, 0.0, 1.0),
    %% Weight with base preference
    (BaseValue + LocalPrefBase) / 2.0;

denormalize_actuator(migration_rate, RawValue, _Hyperparams) ->
    %% Range: 0.0 to 0.2
    lerp(RawValue, 0.0, 0.2);

denormalize_actuator(migration_selection_pressure, RawValue, _Hyperparams) ->
    %% Range: 0.0 to 1.0 (0=random, 1=best only)
    clamp(RawValue, 0.0, 1.0);

denormalize_actuator(target_island_selection, RawValue, _Hyperparams) ->
    %% Range: 0.0 to 1.0 (0=random, 1=most different)
    clamp(RawValue, 0.0, 1.0);

denormalize_actuator(island_split_threshold, RawValue, _Hyperparams) ->
    %% Range: 0.5 to 0.95
    lerp(RawValue, 0.5, 0.95);

denormalize_actuator(island_merge_threshold, RawValue, _Hyperparams) ->
    %% Range: 0.1 to 0.5
    lerp(RawValue, 0.1, 0.5);

denormalize_actuator(load_balance_aggressiveness, RawValue, _Hyperparams) ->
    %% Range: 0.0 to 1.0
    clamp(RawValue, 0.0, 1.0);

denormalize_actuator(peer_selection_strategy, RawValue, _Hyperparams) ->
    %% Range: 0.0 to 1.0 (0=nearest, 1=least loaded)
    clamp(RawValue, 0.0, 1.0);

denormalize_actuator(batch_size_for_remote, RawValue, _Hyperparams) ->
    %% Range: 1 to 20
    round(lerp(RawValue, 1.0, 20.0));

denormalize_actuator(topology_change_rate, RawValue, Hyperparams) ->
    %% Range: 0.0 to 0.1, influenced by stability weight
    StabilityWeight = maps:get(topology_stability_weight, Hyperparams, 0.5),
    BaseRate = lerp(RawValue, 0.0, 0.1),
    %% Higher stability = lower change rate
    BaseRate * (1.0 - StabilityWeight * 0.5);

denormalize_actuator(_Unknown, RawValue, _Hyperparams) ->
    RawValue.

%% @private Apply actuator values to their targets.
apply_to_targets(ActuatorValues, State) ->
    %% Apply load balancing parameters
    apply_load_balancer_params(ActuatorValues, State),

    %% Apply migration parameters
    apply_migration_params(ActuatorValues, State),

    %% Apply topology parameters
    apply_topology_params(ActuatorValues, State),

    ok.

%% @private Apply load balancing parameters.
apply_load_balancer_params(ActuatorValues, State) ->
    LoadBalancerParams = #{
        local_vs_remote_ratio => maps:get(local_vs_remote_ratio, ActuatorValues, 0.8),
        load_balance_aggressiveness => maps:get(load_balance_aggressiveness, ActuatorValues, 0.5),
        peer_selection_strategy => maps:get(peer_selection_strategy, ActuatorValues, 0.5),
        batch_size_for_remote => maps:get(batch_size_for_remote, ActuatorValues, 5)
    },

    case State#state.load_balancer_callback of
        undefined -> ok;
        Callback -> Callback(LoadBalancerParams)
    end.

%% @private Apply migration parameters.
apply_migration_params(ActuatorValues, State) ->
    MigrationParams = #{
        migration_rate => maps:get(migration_rate, ActuatorValues, 0.05),
        migration_selection_pressure => maps:get(migration_selection_pressure, ActuatorValues, 0.5),
        target_island_selection => maps:get(target_island_selection, ActuatorValues, 0.5)
    },

    case State#state.migration_callback of
        undefined -> ok;
        Callback -> Callback(MigrationParams)
    end.

%% @private Apply topology parameters.
apply_topology_params(ActuatorValues, State) ->
    TopologyParams = #{
        island_split_threshold => maps:get(island_split_threshold, ActuatorValues, 0.8),
        island_merge_threshold => maps:get(island_merge_threshold, ActuatorValues, 0.3),
        topology_change_rate => maps:get(topology_change_rate, ActuatorValues, 0.02)
    },

    case State#state.topology_callback of
        undefined -> ok;
        Callback -> Callback(TopologyParams)
    end.

%% @private Build distribution parameters map.
build_distribution_params(ActuatorValues, _State) ->
    #{
        %% Load balancing
        local_vs_remote_ratio => maps:get(local_vs_remote_ratio, ActuatorValues, 0.8),
        load_balance_aggressiveness => maps:get(load_balance_aggressiveness, ActuatorValues, 0.5),
        peer_selection_strategy => maps:get(peer_selection_strategy, ActuatorValues, 0.5),
        batch_size_for_remote => maps:get(batch_size_for_remote, ActuatorValues, 5),

        %% Migration
        migration_rate => maps:get(migration_rate, ActuatorValues, 0.05),
        migration_selection_pressure => maps:get(migration_selection_pressure, ActuatorValues, 0.5),
        target_island_selection => maps:get(target_island_selection, ActuatorValues, 0.5),

        %% Topology
        island_split_threshold => maps:get(island_split_threshold, ActuatorValues, 0.8),
        island_merge_threshold => maps:get(island_merge_threshold, ActuatorValues, 0.3),
        topology_change_rate => maps:get(topology_change_rate, ActuatorValues, 0.02)
    }.

%%% ============================================================================
%%% Internal Functions - Utilities
%%% ============================================================================

%% @private Initial actuator values based on hyperparameters.
initial_actuator_values(Hyperparams) ->
    LocalPrefBase = maps:get(local_preference_base, Hyperparams, 0.8),
    #{
        local_vs_remote_ratio => LocalPrefBase,
        migration_rate => 0.05,
        migration_selection_pressure => 0.5,
        target_island_selection => 0.5,
        island_split_threshold => 0.8,
        island_merge_threshold => 0.3,
        load_balance_aggressiveness => 0.5,
        peer_selection_strategy => 0.5,
        batch_size_for_remote => 5,
        topology_change_rate => 0.02
    }.

%% @private Linear interpolation from 0.0-1.0 to target range.
lerp(T, Min, Max) ->
    ClampedT = clamp(T, 0.0, 1.0),
    Min + ClampedT * (Max - Min).

%% @private Clamp value to range.
clamp(Value, Min, Max) ->
    max(Min, min(Max, Value)).
