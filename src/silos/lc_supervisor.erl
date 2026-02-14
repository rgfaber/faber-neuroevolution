%% @doc Liquid Conglomerate Supervisor.
%%
%% Supervises all 13 silos of the Liquid Conglomerate v2 architecture:
%%
%% **Core Silos (Always Enabled):**
%%   Task Silo: Evolution optimization (τ = 50)
%%   Resource Silo: System stability (τ = 5)
%%
%% **Optional Silos (Enable via Config):**
%%   Distribution Silo: Mesh networking (τ = 1)
%%   Temporal Silo: Episode timing (τ = 10)
%%   Competitive Silo: Opponent archives, Elo (τ = 50)
%%   Social Silo: Reputation, coalitions (τ = 50)
%%   Cultural Silo: Innovations, traditions (τ = 100)
%%   Ecological Silo: Niches, stress (τ = 100)
%%   Morphological Silo: Network complexity (τ = 30)
%%   Developmental Silo: Ontogeny, plasticity (τ = 100)
%%   Regulatory Silo: Gene expression (τ = 50)
%%   Economic Silo: Compute budgets (τ = 20)
%%   Communication Silo: Vocabulary, coordination (τ = 30)
%%
%% Additionally supervises the meta_controller (L2 strategic layer) when enabled:
%%   Meta Controller: LTC-based hierarchical control (outputs L1 guidance)
%%
%% Each silo operates independently with its own hierarchical levels (L0/L1/L2).
%% When meta_controller is enabled, it provides L2 guidance to silos.
%%
%% @author R.G. Lefever
%% @copyright 2024-2026 R.G. Lefever
-module(lc_supervisor).
-behaviour(supervisor).

-include("meta_controller.hrl").

%% API
-export([start_link/0, start_link/1]).

%% Runtime silo control
-export([
    enable_silo/1,
    enable_silo/2,
    disable_silo/1,
    is_silo_enabled/1,
    list_enabled_silos/0,
    list_available_silos/0,
    get_silo_config/1,
    reconfigure_silo/2,
    silo_module/1,
    all_silo_types/0,
    silo_dependencies/1
]).

%% Config validation
-export([
    validate_silo_config/2
]).

%% Supervisor callbacks
-export([init/1]).

-define(SERVER, ?MODULE).
-define(SILO_CONFIG_TABLE, lc_silo_configs).

%%% ============================================================================
%%% API Functions
%%% ============================================================================

%% @doc Start the Liquid Conglomerate supervisor with default configuration.
-spec start_link() -> {ok, pid()} | ignore | {error, term()}.
start_link() ->
    start_link(#{}).

%% @doc Start the Liquid Conglomerate supervisor with custom configuration.
%%
%% Configuration is merged from multiple sources (in priority order):
%% 1. Config map passed to start_link/1 (highest priority)
%% 2. Application env: application:get_env(faber_neuroevolution, lc_supervisor)
%% 3. Default values (lowest priority)
%%
%% If the supervisor is already running, returns the existing pid.
-spec start_link(map()) -> {ok, pid()} | ignore | {error, term()}.
start_link(Config) ->
    case whereis(?SERVER) of
        undefined ->
            %% Create ETS table for silo configs before starting supervisor
            init_config_table(),
            %% Merge with application env
            MergedConfig = merge_app_env_config(Config),
            supervisor:start_link({local, ?SERVER}, ?MODULE, MergedConfig);
        Pid when is_pid(Pid) ->
            %% Already running, return existing pid
            {ok, Pid}
    end.

%%% ============================================================================
%%% Runtime Silo Control API
%%% ============================================================================

%% @doc Enable an extension silo at runtime with default configuration.
-spec enable_silo(atom()) -> ok | {error, term()}.
enable_silo(SiloType) ->
    enable_silo(SiloType, #{}).

%% @doc Enable an extension silo at runtime with custom configuration.
%%
%% Valid silo types: temporal, competitive, social, cultural, ecological,
%% morphological, developmental, regulatory, economic, communication
%%
%% Before enabling, this function:
%%   1. Validates the silo type
%%   2. Checks dependencies are satisfied
%%   3. Validates the configuration
%%   4. Stores config in ETS for later retrieval
-spec enable_silo(atom(), map()) -> ok | {error, term()}.
enable_silo(SiloType, Config) ->
    case silo_module(SiloType) of
        {error, _} = Err ->
            Err;
        ModuleName ->
            case is_silo_enabled(SiloType) of
                true ->
                    {error, already_enabled};
                false ->
                    %% Check dependencies
                    case check_dependencies(SiloType) of
                        ok ->
                            %% Validate config
                            case validate_silo_config(SiloType, Config) of
                                ok ->
                                    do_enable_silo(SiloType, ModuleName, Config);
                                {error, _} = ValidationErr ->
                                    ValidationErr
                            end;
                        {error, _} = DepErr ->
                            DepErr
                    end
            end
    end.

%% @private
do_enable_silo(SiloType, ModuleName, Config) ->
    Spec = #{
        id => ModuleName,
        start => {ModuleName, start_link, [Config]},
        restart => permanent,
        shutdown => 5000,
        type => worker,
        modules => [ModuleName]
    },
    case supervisor:start_child(?SERVER, Spec) of
        {ok, _Pid} ->
            %% Store config in ETS
            store_silo_config(SiloType, Config),
            %% Notify sensor publisher to start polling this silo
            catch lc_sensor_publisher:enable_silo(SiloType),
            %% Publish silo activation event
            publish_silo_event(SiloType, enabled),
            ok;
        {error, Reason} ->
            {error, Reason}
    end.

%% @doc Disable an extension silo at runtime.
%%
%% Core silos (task, resource) cannot be disabled.
%% Also checks that no other enabled silos depend on this one.
-spec disable_silo(atom()) -> ok | {error, term()}.
disable_silo(SiloType) when SiloType =:= task; SiloType =:= resource ->
    {error, cannot_disable_core_silo};
disable_silo(SiloType) ->
    case silo_module(SiloType) of
        {error, _} = Err ->
            Err;
        ModuleName ->
            %% Check if other silos depend on this one
            case check_dependents(SiloType) of
                ok ->
                    case supervisor:terminate_child(?SERVER, ModuleName) of
                        ok ->
                            %% Remove config from ETS
                            delete_silo_config(SiloType),
                            %% Notify sensor publisher to stop polling this silo
                            catch lc_sensor_publisher:disable_silo(SiloType),
                            %% Publish silo deactivation event
                            publish_silo_event(SiloType, disabled),
                            supervisor:delete_child(?SERVER, ModuleName);
                        {error, not_found} ->
                            {error, not_enabled};
                        {error, Reason} ->
                            {error, Reason}
                    end;
                {error, _} = DepErr ->
                    DepErr
            end
    end.

%% @doc Check if a silo is currently enabled.
-spec is_silo_enabled(atom()) -> boolean().
is_silo_enabled(SiloType) ->
    case silo_module(SiloType) of
        {error, _} ->
            false;
        ModuleName ->
            Children = supervisor:which_children(?SERVER),
            lists:any(fun({Id, _, _, _}) -> Id =:= ModuleName end, Children)
    end.

%% @doc List all currently enabled silos.
-spec list_enabled_silos() -> [atom()].
list_enabled_silos() ->
    Children = supervisor:which_children(?SERVER),
    ChildIds = [Id || {Id, _, _, _} <- Children],
    AllSilos = all_silo_types(),
    lists:filter(
        fun(SiloType) ->
            case silo_module(SiloType) of
                {error, _} -> false;
                ModuleName -> lists:member(ModuleName, ChildIds)
            end
        end,
        AllSilos
    ).

%% @doc List all available silo types.
-spec list_available_silos() -> [atom()].
list_available_silos() ->
    all_silo_types().

%% @doc Get the current configuration for a silo.
%%
%% Returns the config that was used to start the silo, or {error, not_enabled}
%% if the silo is not running.
-spec get_silo_config(atom()) -> {ok, map()} | {error, not_enabled | unknown_silo}.
get_silo_config(SiloType) ->
    case silo_module(SiloType) of
        {error, _} = Err ->
            Err;
        _ModuleName ->
            case is_silo_enabled(SiloType) of
                false ->
                    {error, not_enabled};
                true ->
                    case ets:lookup(?SILO_CONFIG_TABLE, SiloType) of
                        [{SiloType, Config}] -> {ok, Config};
                        [] -> {ok, #{}}  %% Core silos started without explicit config
                    end
            end
    end.

%% @doc Reconfigure a running silo with new configuration.
%%
%% This stops the silo, validates the new config, and restarts with new config.
%% Core silos (task, resource) cannot be reconfigured this way.
-spec reconfigure_silo(atom(), map()) -> ok | {error, term()}.
reconfigure_silo(SiloType, _NewConfig) when SiloType =:= task; SiloType =:= resource ->
    {error, cannot_reconfigure_core_silo};
reconfigure_silo(SiloType, NewConfig) ->
    case is_silo_enabled(SiloType) of
        false ->
            {error, not_enabled};
        true ->
            case validate_silo_config(SiloType, NewConfig) of
                ok ->
                    %% Disable and re-enable with new config
                    case disable_silo(SiloType) of
                        ok ->
                            enable_silo(SiloType, NewConfig);
                        {error, _} = Err ->
                            Err
                    end;
                {error, _} = ValidationErr ->
                    ValidationErr
            end
    end.

%% @doc Validate configuration for a silo type.
%%
%% Each silo type has specific configuration requirements. This function
%% checks that required fields are present and values are within valid ranges.
-spec validate_silo_config(atom(), map()) -> ok | {error, term()}.
validate_silo_config(SiloType, Config) ->
    %% Get required and optional fields for this silo type
    case silo_config_schema(SiloType) of
        {error, _} = Err ->
            Err;
        Schema ->
            validate_against_schema(Config, Schema)
    end.

%% @doc Get the dependencies for a silo type.
%%
%% Some silos require other silos to be enabled first.
%% Returns a list of silo types that must be enabled before this one.
-spec silo_dependencies(atom()) -> [atom()].
silo_dependencies(social) -> [competitive];       %% Social needs competitive for coalition competition
silo_dependencies(cultural) -> [social];          %% Cultural needs social for norm transmission
silo_dependencies(communication) -> [social];    %% Communication needs social for trust network
silo_dependencies(developmental) -> [temporal];  %% Developmental needs temporal for critical periods
silo_dependencies(regulatory) -> [developmental]; %% Regulatory needs developmental for expression stages
silo_dependencies(_) -> [].

%%% ============================================================================
%%% Supervisor Callbacks
%%% ============================================================================

init(Config) ->
    SupFlags = #{
        strategy => one_for_one,
        intensity => 5,
        period => 10
    },

    %% ========================================================================
    %% LC v2 Infrastructure - Cross-Silo Communication
    %% ========================================================================

    %% Cross-silo coordinator (must start first for signal routing)
    CrossSiloConfig = maps:get(cross_silo, Config, #{}),
    CrossSiloSpec = #{
        id => lc_cross_silo,
        start => {lc_cross_silo, start_link, [CrossSiloConfig]},
        restart => permanent,
        shutdown => 5000,
        type => worker,
        modules => [lc_cross_silo]
    },

    %% ========================================================================
    %% Resource Silo Components
    %% ========================================================================

    ResourceSiloConfig = maps:get(resource_silo, Config, #{}),

    %% Resource L0 Sensors - Collect system metrics
    ResourceSensorsSpec = #{
        id => resource_l0_sensors,
        start => {resource_l0_sensors, start_link, [ResourceSiloConfig]},
        restart => permanent,
        shutdown => 5000,
        type => worker,
        modules => [resource_l0_sensors]
    },

    %% Resource L0 Actuators - Apply TWEANN outputs
    ResourceActuatorsSpec = #{
        id => resource_l0_actuators,
        start => {resource_l0_actuators, start_link, [ResourceSiloConfig]},
        restart => permanent,
        shutdown => 5000,
        type => worker,
        modules => [resource_l0_actuators]
    },

    %% Resource Silo - Main controller
    ResourceSiloSpec = #{
        id => resource_silo,
        start => {resource_silo, start_link, [ResourceSiloConfig]},
        restart => permanent,
        shutdown => 5000,
        type => worker,
        modules => [resource_silo]
    },

    %% ========================================================================
    %% Task Silo Components
    %% ========================================================================

    TaskSiloConfig = maps:get(task_silo, Config, #{}),

    %% Task L0 Sensors - Collect evolution statistics
    TaskSensorsSpec = #{
        id => task_l0_sensors,
        start => {task_l0_sensors, start_link, [TaskSiloConfig]},
        restart => permanent,
        shutdown => 5000,
        type => worker,
        modules => [task_l0_sensors]
    },

    %% Task L0 Actuators - Apply TWEANN outputs
    TaskActuatorsSpec = #{
        id => task_l0_actuators,
        start => {task_l0_actuators, start_link, [TaskSiloConfig]},
        restart => permanent,
        shutdown => 5000,
        type => worker,
        modules => [task_l0_actuators]
    },

    %% Meta Controller (L2) - Strategic layer (optional)
    MetaControllerEnabled = maps:get(enable_meta_controller, Config, false),
    MetaControllerSpecs = case MetaControllerEnabled of
        true ->
            MetaControllerConfig = build_meta_config(Config),
            [#{
                id => meta_controller,
                start => {meta_controller, start_link, [MetaControllerConfig, [{name, {local, meta_controller}}]]},
                restart => permanent,
                shutdown => 5000,
                type => worker,
                modules => [meta_controller]
            }];
        false ->
            []
    end,

    %% Task Silo - Main controller
    %% L0 TWEANN mode: when enabled, task_silo uses lc_silo_chain for neural network-based
    %% hyperparameter control instead of rule-based L1 adjustments
    %% Enabled by default for LC v2 architecture
    L0TweannEnabled = maps:get(enable_l0_tweann, Config, true),

    TaskSiloConfigWithFlags = TaskSiloConfig#{
        l2_enabled => MetaControllerEnabled,
        l0_tweann_enabled => L0TweannEnabled
    },
    TaskSiloConfigWithL2 = TaskSiloConfigWithFlags,
    TaskSiloSpec = #{
        id => task_silo,
        start => {task_silo, start_link, [TaskSiloConfigWithL2]},
        restart => permanent,
        shutdown => 5000,
        type => worker,
        modules => [task_silo]
    },

    %% ========================================================================
    %% Distribution Silo Components (optional, for mesh networking)
    %% ========================================================================

    DistributionEnabled = maps:get(enable_distribution_silo, Config, false),
    DistSiloConfig = maps:get(distribution_silo, Config, #{}),

    DistributionSpecs = case DistributionEnabled of
        true ->
            [
                %% Distribution L0 Sensors
                #{
                    id => distribution_l0_sensors,
                    start => {distribution_l0_sensors, start_link, [DistSiloConfig]},
                    restart => permanent,
                    shutdown => 5000,
                    type => worker,
                    modules => [distribution_l0_sensors]
                },
                %% Distribution L0 Actuators
                #{
                    id => distribution_l0_actuators,
                    start => {distribution_l0_actuators, start_link, [DistSiloConfig]},
                    restart => permanent,
                    shutdown => 5000,
                    type => worker,
                    modules => [distribution_l0_actuators]
                }
                %% Distribution Silo main controller would go here
            ];
        false ->
            []
    end,

    %% ========================================================================
    %% Extension Silos (LC v2 - Optional)
    %% ========================================================================
    %%
    %% These silos are disabled by default and enabled via configuration.
    %% Each silo follows the same pattern: sensors + actuators + main controller.

    ExtensionSiloSpecs = build_extension_silo_specs(Config),

    %% ========================================================================
    %% Sensor Publisher (polls extension silos and publishes events)
    %% ========================================================================

    SensorPublisherConfig = Config#{
        realm => maps:get(realm, Config, <<"default">>),
        poll_interval_ms => maps:get(sensor_poll_interval_ms, Config, 100)
    },
    SensorPublisherSpec = #{
        id => lc_sensor_publisher,
        start => {lc_sensor_publisher, start_link, [SensorPublisherConfig]},
        restart => permanent,
        shutdown => 5000,
        type => worker,
        modules => [lc_sensor_publisher]
    },

    %% ========================================================================
    %% Child Ordering
    %% ========================================================================
    %% Order matters for dependency:
    %% 1. Cross-silo coordinator (signal routing)
    %% 2. Sensor Publisher (polls extension silos)
    %% 3. Sensors (collect inputs)
    %% 4. Actuators (apply outputs)
    %% 5. Meta controller (L2 guidance)
    %% 6. Main silo controllers
    %% 7. Extension silos (optional)

    ChildSpecs = [CrossSiloSpec, SensorPublisherSpec,
                  ResourceSensorsSpec, ResourceActuatorsSpec,
                  TaskSensorsSpec, TaskActuatorsSpec]
                 ++ MetaControllerSpecs
                 ++ [ResourceSiloSpec, TaskSiloSpec]
                 ++ DistributionSpecs
                 ++ ExtensionSiloSpecs,

    %% Count enabled extension silos
    EnabledExtensions = count_enabled_extensions(Config),

    error_logger:info_msg("[lc_supervisor] Starting Liquid Conglomerate v2 with ~p children "
                         "(L2=~p, Distribution=~p, Extensions=~p)~n",
                         [length(ChildSpecs), MetaControllerEnabled, DistributionEnabled,
                          EnabledExtensions]),

    {ok, {SupFlags, ChildSpecs}}.

%%% ============================================================================
%%% Internal Functions
%%% ============================================================================

%% @doc Build meta_config from supervisor config.
%% Uses defaults with overrides from config map.
-spec build_meta_config(map()) -> meta_config().
build_meta_config(Config) ->
    MetaOpts = maps:get(meta_controller, Config, #{}),
    %% Default topology: 11 inputs (8 evolution + 3 resource), 5 outputs (L1 guidance)
    Topology = maps:get(network_topology, MetaOpts, {11, [24, 16, 8], 5}),
    TimeConstant = maps:get(time_constant, MetaOpts, 50.0),
    NeuronType = maps:get(neuron_type, MetaOpts, cfc),
    LearningRate = maps:get(learning_rate, MetaOpts, 0.001),
    #meta_config{
        network_topology = Topology,
        time_constant = TimeConstant,
        neuron_type = NeuronType,
        learning_rate = LearningRate
    }.

%% @doc Build child specs for extension silos based on configuration.
%%
%% Extension silos are disabled by default. Enable via:
%%
%%   Config = #{
%%       enable_temporal_silo => true,
%%       enable_competitive_silo => true,
%%       ...
%%   }
%%
%% Each enabled silo adds its child spec to the supervision tree.
%% Silos that aren't implemented yet are logged as warnings.
-spec build_extension_silo_specs(map()) -> [supervisor:child_spec()].
build_extension_silo_specs(Config) ->
    %% List of extension silos with their configuration keys and modules
    ExtensionSilos = [
        {temporal, enable_temporal_silo, temporal_silo, temporal_silo},
        {competitive, enable_competitive_silo, competitive_silo, competitive_silo},
        {social, enable_social_silo, social_silo, social_silo},
        {cultural, enable_cultural_silo, cultural_silo, cultural_silo},
        {ecological, enable_ecological_silo, ecological_silo, ecological_silo},
        {morphological, enable_morphological_silo, morphological_silo, morphological_silo},
        {developmental, enable_developmental_silo, developmental_silo, developmental_silo},
        {regulatory, enable_regulatory_silo, regulatory_silo, regulatory_silo},
        {economic, enable_economic_silo, economic_silo, economic_silo},
        {communication, enable_communication_silo, communication_silo, communication_silo}
    ],

    %% Build specs for enabled silos
    lists:filtermap(
        fun({SiloType, EnableKey, ModuleName, _Module}) ->
            Enabled = maps:get(EnableKey, Config, false),
            case Enabled of
                true ->
                    case code:which(ModuleName) of
                        non_existing ->
                            error_logger:warning_msg(
                                "[lc_supervisor] ~p enabled but ~p module not found~n",
                                [SiloType, ModuleName]
                            ),
                            false;
                        _ ->
                            SiloConfig = maps:get(
                                list_to_atom(atom_to_list(SiloType) ++ "_silo"),
                                Config,
                                #{}
                            ),
                            Spec = #{
                                id => ModuleName,
                                start => {ModuleName, start_link, [SiloConfig]},
                                restart => permanent,
                                shutdown => 5000,
                                type => worker,
                                modules => [ModuleName]
                            },
                            {true, Spec}
                    end;
                false ->
                    false
            end
        end,
        ExtensionSilos
    ).

%% @doc Count enabled extension silos in configuration.
-spec count_enabled_extensions(map()) -> non_neg_integer().
count_enabled_extensions(Config) ->
    EnableKeys = [
        enable_temporal_silo,
        enable_competitive_silo,
        enable_social_silo,
        enable_cultural_silo,
        enable_ecological_silo,
        enable_morphological_silo,
        enable_developmental_silo,
        enable_regulatory_silo,
        enable_economic_silo,
        enable_communication_silo
    ],
    lists:foldl(
        fun(Key, Count) ->
            case maps:get(Key, Config, false) of
                true -> Count + 1;
                false -> Count
            end
        end,
        0,
        EnableKeys
    ).

%% @private Publish silo activation/deactivation event to event bus.
%% This allows the UI (via EventBridge) to update silo status in real-time.
%%
%% Publishes to both:
%% 1. Legacy topic (for UI/EventBridge) - silo_status_changed
%% 2. New lifecycle topic (for event-driven architecture) - silo_events:publish_lifecycle/2
-spec publish_silo_event(atom(), enabled | disabled) -> ok.
publish_silo_event(SiloType, Status) ->
    Realm = get_realm(),
    Topic = neuroevolution_events:topic_silo_sensors(Realm),
    Event = {silo_status_changed, #{
        silo => SiloType,
        status => Status,
        realm => Realm,
        timestamp => erlang:system_time(millisecond)
    }},
    neuroevolution_events:publish(Topic, Event),

    %% Also publish to new lifecycle topic (event-driven architecture)
    LifecycleEvent = status_to_lifecycle_event(Status),
    silo_events:publish_lifecycle(SiloType, LifecycleEvent),
    ok.

%% @private Convert enabled/disabled status to lifecycle event atom.
status_to_lifecycle_event(enabled) -> activated;
status_to_lifecycle_event(disabled) -> deactivated.

%% @private Get the realm from application config or default.
-spec get_realm() -> binary().
get_realm() ->
    case application:get_env(faber_neuroevolution, realm) of
        {ok, Realm} when is_binary(Realm) -> Realm;
        _ -> <<"default">>
    end.

%% @doc Map silo type atom to its module name.
-spec silo_module(atom()) -> atom() | {error, unknown_silo}.
silo_module(task) -> task_silo;
silo_module(resource) -> resource_silo;
silo_module(temporal) -> temporal_silo;
silo_module(competitive) -> competitive_silo;
silo_module(social) -> social_silo;
silo_module(cultural) -> cultural_silo;
silo_module(ecological) -> ecological_silo;
silo_module(morphological) -> morphological_silo;
silo_module(developmental) -> developmental_silo;
silo_module(regulatory) -> regulatory_silo;
silo_module(economic) -> economic_silo;
silo_module(communication) -> communication_silo;
silo_module(distribution) -> distribution_silo;
silo_module(_) -> {error, unknown_silo}.

%% @doc List all available silo types.
-spec all_silo_types() -> [atom()].
all_silo_types() ->
    [task, resource, temporal, competitive, social, cultural,
     ecological, morphological, developmental, regulatory,
     economic, communication, distribution].

%%% ============================================================================
%%% Configuration Management
%%% ============================================================================

%% @private Initialize the ETS table for storing silo configurations.
-spec init_config_table() -> ok.
init_config_table() ->
    case ets:whereis(?SILO_CONFIG_TABLE) of
        undefined ->
            ets:new(?SILO_CONFIG_TABLE, [
                named_table,
                public,
                set,
                {read_concurrency, true}
            ]);
        _Tid ->
            ok
    end,
    ok.

%% @private Merge configuration from application environment.
%%
%% Reads from application:get_env(faber_neuroevolution, lc_supervisor) and
%% merges with the provided config map. The provided config takes precedence.
%%
%% Also converts the new silos map format to the legacy enable_X_silo format
%% for backward compatibility.
-spec merge_app_env_config(map()) -> map().
merge_app_env_config(Config) ->
    AppEnvConfig = case application:get_env(faber_neuroevolution, lc_supervisor) of
        {ok, EnvConfig} when is_map(EnvConfig) -> EnvConfig;
        _ -> #{}
    end,
    %% Convert silos map to enable flags if present
    SilosConfig = maps:get(silos, AppEnvConfig, #{}),
    EnableFlags = convert_silos_to_flags(SilosConfig),
    %% Merge: defaults < app_env < provided config
    maps:merge(maps:merge(EnableFlags, AppEnvConfig), Config).

%% @private Convert silos map format to enable flags.
%% Transforms silo configs into enable_*_silo flags and *_silo config maps.
-spec convert_silos_to_flags(map()) -> map().
convert_silos_to_flags(SilosConfig) ->
    maps:fold(
        fun(SiloType, SiloConfig, Acc) when is_map(SiloConfig) ->
            Enabled = maps:get(enabled, SiloConfig, false),
            EnableKey = list_to_atom("enable_" ++ atom_to_list(SiloType) ++ "_silo"),
            ConfigKey = list_to_atom(atom_to_list(SiloType) ++ "_silo"),
            %% Remove 'enabled' from config passed to silo
            CleanConfig = maps:remove(enabled, SiloConfig),
            Acc#{EnableKey => Enabled, ConfigKey => CleanConfig};
           (_, _, Acc) ->
            Acc
        end,
        #{},
        SilosConfig
    ).

%% @private Store silo configuration in ETS.
-spec store_silo_config(atom(), map()) -> true.
store_silo_config(SiloType, Config) ->
    ets:insert(?SILO_CONFIG_TABLE, {SiloType, Config}).

%% @private Delete silo configuration from ETS.
-spec delete_silo_config(atom()) -> true.
delete_silo_config(SiloType) ->
    ets:delete(?SILO_CONFIG_TABLE, SiloType).

%%% ============================================================================
%%% Dependency Management
%%% ============================================================================

%% @private Check that all dependencies for a silo are enabled.
-spec check_dependencies(atom()) -> ok | {error, {missing_dependency, atom()}}.
check_dependencies(SiloType) ->
    Dependencies = silo_dependencies(SiloType),
    case lists:filter(fun(Dep) -> not is_silo_enabled(Dep) end, Dependencies) of
        [] -> ok;
        [Missing | _] -> {error, {missing_dependency, Missing}}
    end.

%% @private Check that no enabled silos depend on this silo.
-spec check_dependents(atom()) -> ok | {error, {has_dependents, [atom()]}}.
check_dependents(SiloType) ->
    EnabledSilos = list_enabled_silos(),
    Dependents = lists:filter(
        fun(Silo) ->
            lists:member(SiloType, silo_dependencies(Silo))
        end,
        EnabledSilos
    ),
    case Dependents of
        [] -> ok;
        _ -> {error, {has_dependents, Dependents}}
    end.

%%% ============================================================================
%%% Configuration Validation
%%% ============================================================================

%% @private Get the configuration schema for a silo type.
%%
%% Schema format: #{required => [atom()], optional => [{atom(), term()}]}
%% where optional contains {Key, DefaultValue} tuples.
-spec silo_config_schema(atom()) -> map() | {error, unknown_silo}.
silo_config_schema(task) ->
    #{required => [], optional => [
        {velocity_threshold, 0.001},
        {enabled_levels, [l0, l1]},
        {l2_enabled, false},
        {l0_tweann_enabled, true}
    ]};
silo_config_schema(resource) ->
    #{required => [], optional => [
        {base_concurrency, 100000},
        {enabled_levels, [l0, l1]},
        {l2_enabled, false}
    ]};
silo_config_schema(temporal) ->
    #{required => [], optional => [
        {realm, <<"default">>},
        {episode_length_target, 1000},
        {evaluation_timeout_ms, 5000}
    ]};
silo_config_schema(competitive) ->
    #{required => [], optional => [
        {realm, <<"default">>},
        {archive_max_size, 100},
        {elo_k_factor, 32}
    ]};
silo_config_schema(social) ->
    #{required => [], optional => [
        {realm, <<"default">>},
        {reputation_decay_rate, 0.05},
        {coalition_size_limit, 10}
    ]};
silo_config_schema(cultural) ->
    #{required => [], optional => [
        {realm, <<"default">>},
        {innovation_bonus, 0.15},
        {tradition_threshold, 5}
    ]};
silo_config_schema(ecological) ->
    #{required => [], optional => [
        {realm, <<"default">>},
        {carrying_capacity, 100},
        {stress_injection_rate, 0.1}
    ]};
silo_config_schema(morphological) ->
    #{required => [], optional => [
        {realm, <<"default">>},
        {max_neurons, 100},
        {max_connections, 500}
    ]};
silo_config_schema(developmental) ->
    #{required => [], optional => [
        {realm, <<"default">>},
        {initial_plasticity, 0.9},
        {critical_period_duration, 5}
    ]};
silo_config_schema(regulatory) ->
    #{required => [], optional => [
        {realm, <<"default">>},
        {expression_threshold, 0.5},
        {context_sensitivity, 0.5}
    ]};
silo_config_schema(economic) ->
    #{required => [], optional => [
        {realm, <<"default">>},
        {budget_per_individual, 1.0},
        {energy_tax_rate, 0.1}
    ]};
silo_config_schema(communication) ->
    #{required => [], optional => [
        {realm, <<"default">>},
        {vocabulary_growth_rate, 0.02},
        {communication_cost, 0.05}
    ]};
silo_config_schema(distribution) ->
    #{required => [], optional => [
        {realm, <<"default">>},
        {migration_probability, 0.05},
        {load_balance_threshold, 0.2}
    ]};
silo_config_schema(_) ->
    {error, unknown_silo}.

%% @private Validate config against a schema.
-spec validate_against_schema(map(), map()) -> ok | {error, term()}.
validate_against_schema(Config, Schema) ->
    RequiredKeys = maps:get(required, Schema, []),
    %% Check all required keys are present
    MissingKeys = lists:filter(
        fun(Key) -> not maps:is_key(Key, Config) end,
        RequiredKeys
    ),
    case MissingKeys of
        [] -> ok;
        _ -> {error, {missing_required_keys, MissingKeys}}
    end.
