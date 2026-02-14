%% @doc LC Sensor Publisher - Publishes sensor events for all enabled silos.
%%
%% Part of the Liquid Conglomerate v2 architecture. This module:
%%   - Polls all enabled extension silos for sensor data
%%   - Publishes sensor events to the silo_sensors topic
%%   - Throttles publishing to avoid overwhelming the UI
%%   - Supports runtime silo enable/disable
%%
%% == Event Format ==
%%
%% Each silo publishes events like:
%%   {temporal_sensors_updated, #{realm => Realm, sensors => SensorMap}}
%%   {competitive_sensors_updated, #{realm => Realm, sensors => SensorMap}}
%%   ...
%%
%% == Configuration ==
%%
%% Started by lc_supervisor with realm and poll_interval_ms options.
%%
%% @author R.G. Lefever
%% @copyright 2024-2026 R.G. Lefever
-module(lc_sensor_publisher).
-behaviour(gen_server).

%% API
-export([
    start_link/0,
    start_link/1,
    enable_silo/1,
    disable_silo/1,
    get_enabled_silos/0,
    force_publish/0,
    force_publish/1
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
-define(DEFAULT_POLL_INTERVAL_MS, 100).  % 10Hz polling
-define(MIN_PUBLISH_INTERVAL_MS, 100).   % Max 10Hz publishing per silo
-define(CHANGE_THRESHOLD, 0.005).        % Minimum change to trigger event

%% Extension silos that can be enabled/disabled
-define(EXTENSION_SILOS, [
    temporal,
    competitive,
    economic,
    social,
    morphological,
    communication,
    developmental,
    cultural,
    regulatory,
    ecological
]).

-record(state, {
    realm :: binary(),
    poll_interval_ms :: pos_integer(),
    timer_ref :: reference() | undefined,
    enabled_silos :: #{atom() => boolean()},
    last_sensors :: #{atom() => map()},
    last_publish_times :: #{atom() => integer()}
}).

%%% ============================================================================
%%% API Functions
%%% ============================================================================

%% @doc Start the sensor publisher with default configuration.
-spec start_link() -> {ok, pid()} | ignore | {error, term()}.
start_link() ->
    start_link(#{}).

%% @doc Start the sensor publisher with custom configuration.
-spec start_link(map()) -> {ok, pid()} | ignore | {error, term()}.
start_link(Config) ->
    gen_server:start_link({local, ?SERVER}, ?MODULE, Config, []).

%% @doc Enable sensor publishing for a silo.
-spec enable_silo(atom()) -> ok.
enable_silo(SiloType) when is_atom(SiloType) ->
    gen_server:cast(?SERVER, {enable_silo, SiloType}).

%% @doc Disable sensor publishing for a silo.
-spec disable_silo(atom()) -> ok.
disable_silo(SiloType) when is_atom(SiloType) ->
    gen_server:cast(?SERVER, {disable_silo, SiloType}).

%% @doc Get list of enabled silos.
-spec get_enabled_silos() -> [atom()].
get_enabled_silos() ->
    gen_server:call(?SERVER, get_enabled_silos).

%% @doc Force immediate publish of all enabled silo sensors.
-spec force_publish() -> ok.
force_publish() ->
    gen_server:cast(?SERVER, force_publish).

%% @doc Force immediate publish of a specific silo's sensors.
-spec force_publish(atom()) -> ok.
force_publish(SiloType) when is_atom(SiloType) ->
    gen_server:cast(?SERVER, {force_publish, SiloType}).

%%% ============================================================================
%%% gen_server Callbacks
%%% ============================================================================

init(Config) ->
    Realm = maps:get(realm, Config, <<"default">>),
    PollInterval = maps:get(poll_interval_ms, Config, ?DEFAULT_POLL_INTERVAL_MS),

    %% Initialize enabled silos from config
    EnabledSilos = init_enabled_silos(Config),

    %% Start polling timer
    TimerRef = erlang:send_after(PollInterval, self(), poll_sensors),

    State = #state{
        realm = Realm,
        poll_interval_ms = PollInterval,
        timer_ref = TimerRef,
        enabled_silos = EnabledSilos,
        last_sensors = #{},
        last_publish_times = #{}
    },

    logger:notice("[lc_sensor_publisher] Started with ~p enabled silos, poll_interval=~pms",
                  [count_enabled(EnabledSilos), PollInterval]),

    {ok, State}.

handle_call(get_enabled_silos, _From, State) ->
    Enabled = [S || {S, true} <- maps:to_list(State#state.enabled_silos)],
    {reply, Enabled, State};

handle_call(_Request, _From, State) ->
    {reply, {error, unknown_request}, State}.

handle_cast({enable_silo, SiloType}, State) ->
    case lists:member(SiloType, ?EXTENSION_SILOS) of
        true ->
            NewEnabled = maps:put(SiloType, true, State#state.enabled_silos),
            logger:info("[lc_sensor_publisher] Enabled ~p silo", [SiloType]),
            {noreply, State#state{enabled_silos = NewEnabled}};
        false ->
            logger:warning("[lc_sensor_publisher] Unknown silo type: ~p", [SiloType]),
            {noreply, State}
    end;

handle_cast({disable_silo, SiloType}, State) ->
    NewEnabled = maps:put(SiloType, false, State#state.enabled_silos),
    logger:info("[lc_sensor_publisher] Disabled ~p silo", [SiloType]),
    {noreply, State#state{enabled_silos = NewEnabled}};

handle_cast(force_publish, State) ->
    NewState = publish_all_sensors(State, true),
    {noreply, NewState};

handle_cast({force_publish, SiloType}, State) ->
    NewState = publish_silo_sensors(SiloType, State, true),
    {noreply, NewState};

handle_cast(_Msg, State) ->
    {noreply, State}.

handle_info(poll_sensors, State) ->
    %% Poll all enabled silos and publish changes
    NewState = publish_all_sensors(State, false),

    %% Schedule next poll
    TimerRef = erlang:send_after(State#state.poll_interval_ms, self(), poll_sensors),

    {noreply, NewState#state{timer_ref = TimerRef}};

handle_info(_Info, State) ->
    {noreply, State}.

terminate(_Reason, State) ->
    case State#state.timer_ref of
        undefined -> ok;
        Ref -> erlang:cancel_timer(Ref)
    end,
    ok.

%%% ============================================================================
%%% Internal Functions
%%% ============================================================================

%% @private Initialize enabled silos from config.
init_enabled_silos(Config) ->
    lists:foldl(
        fun(SiloType, Acc) ->
            ConfigKey = list_to_atom("enable_" ++ atom_to_list(SiloType) ++ "_silo"),
            Enabled = maps:get(ConfigKey, Config, false),
            maps:put(SiloType, Enabled, Acc)
        end,
        #{},
        ?EXTENSION_SILOS
    ).

%% @private Count enabled silos.
count_enabled(EnabledMap) ->
    length([S || {S, true} <- maps:to_list(EnabledMap)]).

%% @private Publish sensors for all enabled silos.
publish_all_sensors(State, Force) ->
    EnabledSilos = [S || {S, true} <- maps:to_list(State#state.enabled_silos)],
    lists:foldl(
        fun(SiloType, AccState) ->
            publish_silo_sensors(SiloType, AccState, Force)
        end,
        State,
        EnabledSilos
    ).

%% @private Publish sensors for a specific silo.
publish_silo_sensors(SiloType, State, Force) ->
    Now = erlang:system_time(millisecond),
    LastPublishTime = maps:get(SiloType, State#state.last_publish_times, 0),
    TimeSincePublish = Now - LastPublishTime,

    %% Rate limit unless forced
    case Force orelse TimeSincePublish >= ?MIN_PUBLISH_INTERVAL_MS of
        true ->
            try_publish_silo(SiloType, State, Now, Force);
        false ->
            State
    end.

%% @private Try to get sensors from silo and publish if changed.
try_publish_silo(SiloType, State, Now, Force) ->
    Module = silo_module(SiloType),
    case whereis(Module) of
        undefined ->
            %% Silo not running
            State;
        Pid ->
            try
                case gen_server:call(Pid, get_state, 1000) of
                    #{sensors := Sensors} when is_map(Sensors) ->
                        maybe_publish(SiloType, Sensors, State, Now, Force);
                    StateMap when is_map(StateMap) ->
                        %% Some silos return sensors directly
                        Sensors = extract_sensors(SiloType, StateMap),
                        maybe_publish(SiloType, Sensors, State, Now, Force);
                    _ ->
                        State
                end
            catch
                _:_ ->
                    %% Silo busy or errored, skip this cycle
                    State
            end
    end.

%% @private Extract sensors from silo state map.
extract_sensors(SiloType, StateMap) ->
    %% Try to get sensors key, or compute from state
    case maps:get(sensors, StateMap, undefined) of
        undefined ->
            %% Call collect_sensors on the module
            Module = silo_module(SiloType),
            case erlang:function_exported(Module, collect_sensors, 1) of
                true ->
                    %% This requires internal state, which we don't have
                    %% Fall back to extracting known keys from StateMap
                    extract_known_sensors(SiloType, StateMap);
                false ->
                    extract_known_sensors(SiloType, StateMap)
            end;
        Sensors when is_map(Sensors) ->
            Sensors
    end.

%% @private Extract known sensor keys from state map for each silo type.
extract_known_sensors(temporal, State) ->
    #{
        episode_length_avg => maps:get(episode_length_avg, State, 0.0),
        episode_variance => maps:get(episode_variance, State, 0.0),
        time_pressure => maps:get(time_pressure, State, 0.0),
        convergence_progress => maps:get(convergence_progress, State, 0.0),
        patience_remaining => maps:get(patience_remaining, State, 1.0),
        learning_rate => maps:get(learning_rate, State, 1.0),
        timeout_ratio => maps:get(timeout_ratio, State, 0.0),
        early_termination_ratio => maps:get(early_termination_ratio, State, 0.0),
        reaction_time_avg => maps:get(reaction_time_avg, State, 0.0)
    };
extract_known_sensors(competitive, State) ->
    #{
        elo_mean => maps:get(elo_mean, State, 1000.0),
        elo_variance => maps:get(elo_variance, State, 0.0),
        win_rate_mean => maps:get(win_rate_mean, State, 0.5),
        arms_race_progress => maps:get(arms_race_progress, State, 0.0),
        opponent_diversity => maps:get(opponent_diversity, State, 0.0),
        counter_adaptation_rate => maps:get(counter_adaptation_rate, State, 0.0),
        dominance_index => maps:get(dominance_index, State, 0.0)
    };
extract_known_sensors(economic, State) ->
    #{
        budget_utilization => maps:get(budget_utilization, State, 0.0),
        gini_coefficient => maps:get(gini_coefficient, State, 0.0),
        trade_volume => maps:get(trade_volume, State, 0.0),
        bankruptcy_rate => maps:get(bankruptcy_rate, State, 0.0),
        economic_pressure => maps:get(economic_pressure, State, 0.0),
        inflation_rate => maps:get(inflation_rate, State, 0.0)
    };
extract_known_sensors(social, State) ->
    #{
        cooperation_rate => maps:get(cooperation_rate, State, 0.5),
        defection_rate => maps:get(defection_rate, State, 0.0),
        reputation_mean => maps:get(reputation_mean, State, 0.5),
        reputation_variance => maps:get(reputation_variance, State, 0.0),
        coalition_count => maps:get(coalition_count, State, 0),
        coalition_stability => maps:get(coalition_stability, State, 1.0),
        trust_network_density => maps:get(trust_network_density, State, 0.0)
    };
extract_known_sensors(morphological, State) ->
    #{
        avg_network_size => maps:get(avg_network_size, State, 0.0),
        avg_connection_count => maps:get(avg_connection_count, State, 0.0),
        topology_diversity => maps:get(topology_diversity, State, 0.0),
        bloat_ratio => maps:get(bloat_ratio, State, 0.0),
        modularity_score => maps:get(modularity_score, State, 0.0),
        complexity_trend => maps:get(complexity_trend, State, 0.0),
        depth_avg => maps:get(depth_avg, State, 0.0)
    };
extract_known_sensors(communication, State) ->
    #{
        vocabulary_size => maps:get(vocabulary_size, State, 0),
        signal_diversity => maps:get(signal_diversity, State, 0.0),
        communication_efficiency => maps:get(communication_efficiency, State, 0.0),
        coordination_success_rate => maps:get(coordination_success_rate, State, 0.0),
        message_complexity => maps:get(message_complexity, State, 0.0),
        language_drift_rate => maps:get(language_drift_rate, State, 0.0)
    };
extract_known_sensors(developmental, State) ->
    #{
        plasticity_mean => maps:get(plasticity_mean, State, 0.5),
        plasticity_variance => maps:get(plasticity_variance, State, 0.0),
        maturation_rate => maps:get(maturation_rate, State, 0.0),
        critical_period_active => maps:get(critical_period_active, State, false),
        growth_efficiency => maps:get(growth_efficiency, State, 0.0)
    };
extract_known_sensors(cultural, State) ->
    #{
        innovation_rate => maps:get(innovation_rate, State, 0.0),
        tradition_stability => maps:get(tradition_stability, State, 0.0),
        meme_diversity => maps:get(meme_diversity, State, 0.0),
        cultural_drift_rate => maps:get(cultural_drift_rate, State, 0.0),
        knowledge_transfer_rate => maps:get(knowledge_transfer_rate, State, 0.0),
        teaching_effectiveness => maps:get(teaching_effectiveness, State, 0.0)
    };
extract_known_sensors(regulatory, State) ->
    #{
        expression_diversity => maps:get(expression_diversity, State, 0.0),
        epigenetic_stability => maps:get(epigenetic_stability, State, 0.0),
        regulatory_complexity => maps:get(regulatory_complexity, State, 0.0),
        environmental_sensitivity => maps:get(environmental_sensitivity, State, 0.0),
        inheritance_fidelity => maps:get(inheritance_fidelity, State, 0.0)
    };
extract_known_sensors(ecological, State) ->
    #{
        niche_count => maps:get(niche_count, State, 1),
        niche_overlap => maps:get(niche_overlap, State, 0.0),
        environmental_stress => maps:get(environmental_stress, State, 0.0),
        carrying_capacity_ratio => maps:get(carrying_capacity_ratio, State, 0.5),
        extinction_risk => maps:get(extinction_risk, State, 0.0),
        adaptation_rate => maps:get(adaptation_rate, State, 0.0)
    };
extract_known_sensors(_, _State) ->
    #{}.

%% @private Maybe publish sensors if changed or forced.
maybe_publish(SiloType, Sensors, State, Now, Force) ->
    LastSensors = maps:get(SiloType, State#state.last_sensors, #{}),

    case Force orelse sensors_changed(Sensors, LastSensors) of
        true ->
            publish_event(SiloType, Sensors, State#state.realm),
            State#state{
                last_sensors = maps:put(SiloType, Sensors, State#state.last_sensors),
                last_publish_times = maps:put(SiloType, Now, State#state.last_publish_times)
            };
        false ->
            State
    end.

%% @private Check if sensors changed significantly.
sensors_changed(_New, Old) when map_size(Old) == 0 ->
    true;
sensors_changed(New, Old) ->
    maps:fold(
        fun(Key, Value, Acc) ->
            case Acc of
                true -> true;
                false ->
                    OldValue = maps:get(Key, Old, undefined),
                    value_changed(Value, OldValue)
            end
        end,
        false,
        New
    ).

%% @private Check if a value changed significantly.
value_changed(V, V) -> false;
value_changed(New, Old) when is_number(New), is_number(Old) ->
    abs(New - Old) > ?CHANGE_THRESHOLD;
value_changed(_, _) -> true.

%% @private Publish sensor event.
publish_event(SiloType, Sensors, Realm) ->
    Topic = neuroevolution_events:topic_silo_sensors(Realm),
    EventType = event_type(SiloType),
    Event = {EventType, #{
        realm => Realm,
        source => ?MODULE,
        silo => SiloType,
        sensors => Sensors,
        timestamp => erlang:system_time(millisecond)
    }},
    neuroevolution_events:publish(Topic, Event).

%% @private Get event type for silo.
event_type(temporal) -> temporal_sensors_updated;
event_type(competitive) -> competitive_sensors_updated;
event_type(economic) -> economic_sensors_updated;
event_type(social) -> social_sensors_updated;
event_type(morphological) -> morphological_sensors_updated;
event_type(communication) -> communication_sensors_updated;
event_type(developmental) -> developmental_sensors_updated;
event_type(cultural) -> cultural_sensors_updated;
event_type(regulatory) -> regulatory_sensors_updated;
event_type(ecological) -> ecological_sensors_updated.

%% @private Get module name for silo type.
silo_module(temporal) -> temporal_silo;
silo_module(competitive) -> competitive_silo;
silo_module(economic) -> economic_silo;
silo_module(social) -> social_silo;
silo_module(morphological) -> morphological_silo;
silo_module(communication) -> communication_silo;
silo_module(developmental) -> developmental_silo;
silo_module(cultural) -> cultural_silo;
silo_module(regulatory) -> regulatory_silo;
silo_module(ecological) -> ecological_silo.
