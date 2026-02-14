%% @doc Cross-Silo Communication Coordinator for Liquid Conglomerate.
%%
%% Part of the Liquid Conglomerate v2 event-driven architecture. This module
%% subscribes to silo signal events and aggregates them for consumers.
%%
%% == Event-Driven Architecture ==
%%
%% Silos publish signals via silo_events:publish_signal/3 to topic-based
%% pub/sub. This module subscribes to all silo topics and maintains an
%% aggregated view for backward compatibility and efficient querying.
%%
%% == Architecture ==
%%
%% The LC routes ~60+ named signals between 13 specialized silos:
%%
%% Core Silos (Original):
%%   Task Silo: Evolution optimization (tau = 50)
%%   Resource Silo: System stability (tau = 5)
%%   Distribution Silo: Mesh networking (tau = 1)
%%
%% Extension Silos (LC v2):
%%   Temporal Silo: Episode timing, early termination (tau = 10)
%%   Competitive Silo: Opponent archives, Elo ratings (tau = 50)
%%   Social Silo: Reputation, coalitions, mentoring (tau = 50)
%%   Cultural Silo: Innovations, traditions, imitation (tau = 100)
%%   Ecological Silo: Niches, carrying capacity, stress (tau = 100)
%%   Morphological Silo: Network size, pruning (tau = 30)
%%   Developmental Silo: Ontogeny, critical periods (tau = 100)
%%   Regulatory Silo: Gene expression, module activation (tau = 50)
%%   Economic Silo: Compute budgets, energy economics (tau = 20)
%%   Communication Silo: Vocabulary, coordination (tau = 30)
%%
%% == Signal Naming Convention ==
%%
%% Pressure signals: constraint indicators (0-1, higher = more constrained)
%% Boost signals: enhancement requests (0-1, higher = more boost)
%% Score signals: quality metrics (0-1, higher = better)
%% Rate signals: frequency indicators (0-1, higher = more frequent)
%%
%% == Signal Decay ==
%%
%% Signals not updated within decay_ms (default 30s) decay toward neutral:
%% Pressure/boost signals decay to 0.0, balance signals decay to 0.5.
%%
%% @author R.G. Lefever
%% @copyright 2024-2026 R.G. Lefever
-module(lc_cross_silo).
-behaviour(gen_server).

%% API
-export([
    start_link/0,
    start_link/1,
    emit/4,
    emit_batch/3,
    get_signals_for/1,
    get_all_signals/0,
    subscribe/2,
    unsubscribe/1,
    get_effective_evals_per_individual/0
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
    %% Signals organized by destination silo
    %% #{task => #{pressure_signal => 0.7, ...}, ...}
    signals :: #{atom() => map()},

    %% Subscribers (silo => callback)
    subscribers :: #{atom() => fun((map()) -> ok)},

    %% Configuration
    signal_decay_ms :: pos_integer(),  % How fast signals decay if not updated
    last_update :: #{atom() => integer()}  % Last update time per signal
}).

%%% ============================================================================
%%% API Functions
%%% ============================================================================

%% @doc Start the cross-silo coordinator with default configuration.
-spec start_link() -> {ok, pid()} | ignore | {error, term()}.
start_link() ->
    start_link(#{}).

%% @doc Start the cross-silo coordinator with custom configuration.
%% If the coordinator is already running, returns the existing pid.
-spec start_link(map()) -> {ok, pid()} | ignore | {error, term()}.
start_link(Config) ->
    case whereis(?SERVER) of
        undefined ->
            gen_server:start_link({local, ?SERVER}, ?MODULE, Config, []);
        Pid when is_pid(Pid) ->
            %% Already running, return existing pid
            {ok, Pid}
    end.

%% @doc Emit a signal from one silo to another.
%%
%% Example: emit(resource, task, pressure_signal, 0.7)
-spec emit(atom(), atom(), atom(), number()) -> ok.
emit(FromSilo, ToSilo, SignalName, Value) ->
    gen_server:cast(?SERVER, {emit, FromSilo, ToSilo, SignalName, Value}).

%% @doc Emit multiple signals from one silo to another.
-spec emit_batch(atom(), atom(), map()) -> ok.
emit_batch(FromSilo, ToSilo, Signals) ->
    gen_server:cast(?SERVER, {emit_batch, FromSilo, ToSilo, Signals}).

%% @doc Get all signals destined for a specific silo.
-spec get_signals_for(atom()) -> map().
get_signals_for(Silo) ->
    gen_server:call(?SERVER, {get_signals_for, Silo}).

%% @doc Get all signals for all silos.
-spec get_all_signals() -> map().
get_all_signals() ->
    gen_server:call(?SERVER, get_all_signals).

%% @doc Subscribe to signals for a specific silo.
%%
%% The callback will be invoked whenever signals change.
-spec subscribe(atom(), fun((map()) -> ok)) -> ok.
subscribe(Silo, Callback) ->
    gen_server:cast(?SERVER, {subscribe, Silo, Callback}).

%% @doc Unsubscribe from signals for a specific silo.
-spec unsubscribe(atom()) -> ok.
unsubscribe(Silo) ->
    gen_server:cast(?SERVER, {unsubscribe, Silo}).

%% @doc Get effective evaluations per individual (negotiated between silos).
%%
%% This resolves the shared resource conflict:
%% - Resource Silo sets max based on pressure
%% - Task Silo requests desired based on fitness variance
%% - Effective = min(resource_max, task_desired)
-spec get_effective_evals_per_individual() -> pos_integer().
get_effective_evals_per_individual() ->
    gen_server:call(?SERVER, get_effective_evals_per_individual).

%%% ============================================================================
%%% gen_server Callbacks
%%% ============================================================================

init(Config) ->
    SignalDecayMs = maps:get(signal_decay_ms, Config, 30000),

    State = #state{
        signals = initial_signals(),
        subscribers = #{},
        signal_decay_ms = SignalDecayMs,
        last_update = #{}
    },

    %% Subscribe to all silo signal topics (event-driven pattern)
    subscribe_to_silo_topics(),

    %% Start periodic decay check
    erlang:send_after(SignalDecayMs, self(), check_decay),

    {ok, State}.

%% @private Subscribe to all silo signal topics for event-driven aggregation.
subscribe_to_silo_topics() ->
    lists:foreach(
        fun(SiloName) ->
            Topic = silo_events:signal_topic(SiloName),
            neuroevolution_events:subscribe(Topic, self())
        end,
        silo_events:all_silo_names()
    ),
    ok.

handle_call({get_signals_for, Silo}, _From, State) ->
    Signals = maps:get(Silo, State#state.signals, #{}),
    {reply, Signals, State};

handle_call(get_all_signals, _From, State) ->
    {reply, State#state.signals, State};

handle_call(get_effective_evals_per_individual, _From, State) ->
    %% Get signals from both Resource and Task silos
    TaskSignals = maps:get(task, State#state.signals, #{}),
    ResourceMax = maps:get(max_evals_per_individual, TaskSignals, 20),

    ResourceSignals = maps:get(resource, State#state.signals, #{}),
    TaskDesired = maps:get(desired_evals_per_individual, ResourceSignals, 5),

    %% Effective = min(resource_max, task_desired)
    Effective = max(1, min(ResourceMax, TaskDesired)),
    {reply, Effective, State};

handle_call(_Request, _From, State) ->
    {reply, {error, unknown_request}, State}.

handle_cast({emit, FromSilo, ToSilo, SignalName, Value}, State) ->
    %% DEPRECATED: Use silo_events:publish_signal/3 instead
    log_deprecation_warning(emit, FromSilo, SignalName),
    NewState = store_signal(FromSilo, ToSilo, SignalName, Value, State),
    notify_subscriber(ToSilo, NewState),
    {noreply, NewState};

handle_cast({emit_batch, FromSilo, ToSilo, Signals}, State) ->
    %% DEPRECATED: Use silo_events:publish_signals/2 instead
    log_deprecation_warning(emit_batch, FromSilo, maps:keys(Signals)),
    NewState = maps:fold(
        fun(SignalName, Value, AccState) ->
            store_signal(FromSilo, ToSilo, SignalName, Value, AccState)
        end,
        State,
        Signals
    ),
    notify_subscriber(ToSilo, NewState),
    {noreply, NewState};

handle_cast({subscribe, Silo, Callback}, State) ->
    NewSubscribers = maps:put(Silo, Callback, State#state.subscribers),
    {noreply, State#state{subscribers = NewSubscribers}};

handle_cast({unsubscribe, Silo}, State) ->
    NewSubscribers = maps:remove(Silo, State#state.subscribers),
    {noreply, State#state{subscribers = NewSubscribers}};

handle_cast(_Msg, State) ->
    {noreply, State}.

handle_info(check_decay, State) ->
    NewState = apply_decay(State),
    erlang:send_after(State#state.signal_decay_ms, self(), check_decay),
    {noreply, NewState};

%% Handle single silo signal event (event-driven pattern)
handle_info({neuro_event, _Topic, #{event_type := <<"silo_signal">>,
                                     from := FromSilo,
                                     signal := SignalName,
                                     value := Value}}, State) ->
    %% Route signal to all valid destinations
    NewState = route_signal_to_destinations(FromSilo, SignalName, Value, State),
    {noreply, NewState};

%% Handle batch silo signals event (event-driven pattern)
handle_info({neuro_event, _Topic, #{event_type := <<"silo_signals">>,
                                     from := FromSilo,
                                     signals := Signals}}, State) ->
    %% Route all signals to their destinations
    NewState = maps:fold(
        fun(SignalName, Value, AccState) ->
            route_signal_to_destinations(FromSilo, SignalName, Value, AccState)
        end,
        State,
        Signals
    ),
    {noreply, NewState};

handle_info(_Info, State) ->
    {noreply, State}.

terminate(_Reason, _State) ->
    ok.

%%% ============================================================================
%%% Internal Functions - Event-Driven Signal Routing
%%% ============================================================================

%% @private Route a signal from source silo to all valid destinations.
%%
%% In event-driven mode, signals are published to a topic without specifying
%% a destination. This function finds all valid destinations for the signal
%% based on is_valid_signal/3 and stores/notifies accordingly.
route_signal_to_destinations(FromSilo, SignalName, Value, State) ->
    %% Find all valid destinations for this signal
    ValidDestinations = find_valid_destinations(FromSilo, SignalName),

    %% Store signal for each valid destination and notify
    lists:foldl(
        fun(ToSilo, AccState) ->
            NewAccState = store_signal(FromSilo, ToSilo, SignalName, Value, AccState),
            notify_subscriber(ToSilo, NewAccState),
            NewAccState
        end,
        State,
        ValidDestinations
    ).

%% @private Find all valid destination silos for a signal from a source.
find_valid_destinations(FromSilo, SignalName) ->
    AllSilos = silo_events:all_silo_names(),
    lists:filter(
        fun(ToSilo) ->
            is_valid_signal(FromSilo, ToSilo, SignalName)
        end,
        AllSilos
    ).

%% @private Log deprecation warning for direct emit calls.
%%
%% These warnings help track migration progress to event-driven pattern.
log_deprecation_warning(emit, FromSilo, SignalName) ->
    logger:warning("[DEPRECATED] lc_cross_silo:emit/4 called from ~p for signal ~p. "
                   "Use silo_events:publish_signal/3 instead.",
                   [FromSilo, SignalName]);
log_deprecation_warning(emit_batch, FromSilo, SignalNames) ->
    logger:warning("[DEPRECATED] lc_cross_silo:emit_batch/3 called from ~p for signals ~p. "
                   "Use silo_events:publish_signals/2 instead.",
                   [FromSilo, SignalNames]).

%%% ============================================================================
%%% Internal Functions - Signal Storage
%%% ============================================================================

%% @private Store a signal value.
store_signal(FromSilo, ToSilo, SignalName, Value, State) ->
    %% Validate the signal is allowed
    case is_valid_signal(FromSilo, ToSilo, SignalName) of
        true ->
            %% Get current signals for destination
            CurrentSignals = maps:get(ToSilo, State#state.signals, #{}),

            %% Update the specific signal
            NewSignals = maps:put(SignalName, clamp_signal(SignalName, Value), CurrentSignals),

            %% Update last update time
            SignalKey = {FromSilo, ToSilo, SignalName},
            Now = erlang:monotonic_time(millisecond),
            NewLastUpdate = maps:put(SignalKey, Now, State#state.last_update),

            State#state{
                signals = maps:put(ToSilo, NewSignals, State#state.signals),
                last_update = NewLastUpdate
            };
        false ->
            %% Log invalid signal attempt
            logger:warning("Invalid cross-silo signal: ~p -> ~p: ~p",
                          [FromSilo, ToSilo, SignalName]),
            State
    end.

%% @private Check if a signal is valid for the given route.
is_valid_signal(resource, task, pressure_signal) -> true;
is_valid_signal(resource, task, max_evals_per_individual) -> true;
is_valid_signal(resource, task, should_simplify) -> true;
is_valid_signal(resource, distribution, offload_preference) -> true;
is_valid_signal(resource, distribution, local_capacity) -> true;

is_valid_signal(task, resource, exploration_boost) -> true;
is_valid_signal(task, resource, desired_evals_per_individual) -> true;
is_valid_signal(task, resource, expected_complexity_growth) -> true;
is_valid_signal(task, distribution, diversity_need) -> true;
is_valid_signal(task, distribution, speciation_pressure) -> true;

is_valid_signal(distribution, resource, network_load_contribution) -> true;
is_valid_signal(distribution, resource, remote_capacity_available) -> true;
is_valid_signal(distribution, task, island_diversity_score) -> true;
is_valid_signal(distribution, task, migration_activity) -> true;
is_valid_signal(distribution, temporal, network_latency) -> true;

%%% ============================================================================
%%% New Silo Signals (LC v2 - 13 Silos)
%%% ============================================================================

%% Resource Silo → New Silos
is_valid_signal(resource, temporal, compute_availability) -> true;
is_valid_signal(resource, competitive, arms_race_load) -> true;
is_valid_signal(resource, ecological, abundance_signal) -> true;
is_valid_signal(resource, economic, budget_signal) -> true;
is_valid_signal(resource, developmental, scarcity_signal) -> true;

%% Task Silo → New Silos
is_valid_signal(task, temporal, stagnation_severity) -> true;
is_valid_signal(task, competitive, fitness_pressure) -> true;
is_valid_signal(task, social, selection_pressure) -> true;
is_valid_signal(task, cultural, exploration_need) -> true;
is_valid_signal(task, ecological, adaptation_pressure) -> true;
is_valid_signal(task, morphological, complexity_target) -> true;
is_valid_signal(task, developmental, maturity_target) -> true;
is_valid_signal(task, regulatory, context_complexity) -> true;
is_valid_signal(task, economic, budget_constraint) -> true;
is_valid_signal(task, communication, coordination_need) -> true;

%% Temporal Silo Signals
is_valid_signal(temporal, task, time_pressure) -> true;
is_valid_signal(temporal, resource, convergence_status) -> true;
is_valid_signal(temporal, economic, episode_efficiency) -> true;
is_valid_signal(temporal, developmental, critical_period_timing) -> true;

%% Competitive Silo Signals
is_valid_signal(competitive, task, competitive_pressure) -> true;
is_valid_signal(competitive, cultural, strategy_diversity_need) -> true;
is_valid_signal(competitive, resource, arms_race_active) -> true;
is_valid_signal(competitive, social, coalition_competition) -> true;

%% Social Silo Signals
is_valid_signal(social, task, selection_influence) -> true;
is_valid_signal(social, cultural, norm_transmission) -> true;
is_valid_signal(social, competitive, coalition_structure) -> true;
is_valid_signal(social, communication, trust_network) -> true;

%% Cultural Silo Signals
is_valid_signal(cultural, task, innovation_impact) -> true;
is_valid_signal(cultural, competitive, strategy_innovation) -> true;
is_valid_signal(cultural, developmental, plasticity_influence) -> true;
is_valid_signal(cultural, communication, information_sharing) -> true;

%% Ecological Silo Signals
is_valid_signal(ecological, task, environmental_pressure) -> true;
is_valid_signal(ecological, resource, resource_level) -> true;
is_valid_signal(ecological, developmental, stress_signal) -> true;
is_valid_signal(ecological, regulatory, environmental_context) -> true;

%% Morphological Silo Signals
is_valid_signal(morphological, task, complexity_signal) -> true;
is_valid_signal(morphological, resource, size_budget) -> true;
is_valid_signal(morphological, economic, efficiency_score) -> true;
is_valid_signal(morphological, developmental, growth_stage) -> true;

%% Developmental Silo Signals
is_valid_signal(developmental, task, maturity_distribution) -> true;
is_valid_signal(developmental, cultural, plasticity_available) -> true;
is_valid_signal(developmental, ecological, metamorphosis_rate) -> true;
is_valid_signal(developmental, regulatory, expression_stage) -> true;

%% Regulatory Silo Signals
is_valid_signal(regulatory, task, context_awareness) -> true;
is_valid_signal(regulatory, cultural, expression_flexibility) -> true;
is_valid_signal(regulatory, competitive, dormant_potential) -> true;
is_valid_signal(regulatory, morphological, expression_cost) -> true;

%% Economic Silo Signals
is_valid_signal(economic, task, economic_pressure) -> true;
is_valid_signal(economic, temporal, budget_available) -> true;
is_valid_signal(economic, morphological, efficiency_requirement) -> true;
is_valid_signal(economic, social, trade_opportunity) -> true;

%% Communication Silo Signals
is_valid_signal(communication, task, coordination_capability) -> true;
is_valid_signal(communication, cultural, information_transfer) -> true;
is_valid_signal(communication, social, trust_signal) -> true;
is_valid_signal(communication, competitive, strategic_signaling) -> true;

%% Domain Signals (from external applications via signal_router)
%% Domain signals are prefixed with 'domain_' and can target any silo
%% that matches their category.
is_valid_signal(domain, ToSilo, SignalName) ->
    %% Accept any domain_* signal to matching silo category
    is_domain_signal(SignalName) andalso is_valid_silo(ToSilo);

is_valid_signal(_, _, _) -> false.

%% @private Check if signal name has domain_ prefix.
is_domain_signal(SignalName) when is_atom(SignalName) ->
    case atom_to_list(SignalName) of
        "domain_" ++ _ -> true;
        _ -> false
    end;
is_domain_signal(_) -> false.

%% @private Check if destination is a valid silo.
is_valid_silo(task) -> true;
is_valid_silo(resource) -> true;
is_valid_silo(distribution) -> true;
is_valid_silo(temporal) -> true;
is_valid_silo(competitive) -> true;
is_valid_silo(social) -> true;
is_valid_silo(cultural) -> true;
is_valid_silo(ecological) -> true;
is_valid_silo(morphological) -> true;
is_valid_silo(developmental) -> true;
is_valid_silo(regulatory) -> true;
is_valid_silo(economic) -> true;
is_valid_silo(communication) -> true;
is_valid_silo(_) -> false.

%% @private Clamp signal value to appropriate range.
clamp_signal(max_evals_per_individual, Value) ->
    round(max(1, min(20, Value)));
clamp_signal(desired_evals_per_individual, Value) ->
    round(max(1, min(50, Value)));
clamp_signal(_Name, Value) when is_number(Value) ->
    max(0.0, min(1.0, Value));
clamp_signal(_Name, Value) ->
    Value.

%% @private Notify subscriber about signal changes.
notify_subscriber(Silo, State) ->
    case maps:get(Silo, State#state.subscribers, undefined) of
        undefined -> ok;
        Callback ->
            Signals = maps:get(Silo, State#state.signals, #{}),
            try
                Callback(Signals)
            catch
                _:_ -> ok
            end
    end.

%% @private Apply decay to old signals.
apply_decay(State) ->
    Now = erlang:monotonic_time(millisecond),
    DecayMs = State#state.signal_decay_ms,

    %% Find signals that haven't been updated recently
    OldSignals = maps:filter(
        fun(_Key, LastTime) ->
            Now - LastTime > DecayMs
        end,
        State#state.last_update
    ),

    %% Decay old signals towards neutral (0.5 for ratios, 0.0 for pressures)
    NewSignals = maps:fold(
        fun({_From, ToSilo, SignalName}, _Time, AccSignals) ->
            SiloSignals = maps:get(ToSilo, AccSignals, #{}),
            case maps:get(SignalName, SiloSignals, undefined) of
                undefined -> AccSignals;
                CurrentValue ->
                    NeutralValue = get_neutral_value(SignalName),
                    %% Decay 10% towards neutral
                    DecayedValue = CurrentValue + (NeutralValue - CurrentValue) * 0.1,
                    NewSiloSignals = maps:put(SignalName, DecayedValue, SiloSignals),
                    maps:put(ToSilo, NewSiloSignals, AccSignals)
            end
        end,
        State#state.signals,
        OldSignals
    ),

    State#state{signals = NewSignals}.

%% @private Get neutral value for a signal type.
%% Signals decay toward these values when not updated.
%%
%% Neutral = 0.0: Pressures and boosts that should turn off
%% Neutral = 0.5: Balanced/neutral state signals
%% Neutral = integer: Count-based signals

%% Existing signals (original 3 silos)
get_neutral_value(pressure_signal) -> 0.0;
get_neutral_value(should_simplify) -> 0.0;
get_neutral_value(exploration_boost) -> 0.0;
get_neutral_value(expected_complexity_growth) -> 0.0;
get_neutral_value(network_load_contribution) -> 0.0;
get_neutral_value(max_evals_per_individual) -> 10;
get_neutral_value(desired_evals_per_individual) -> 5;

%% Temporal Silo signals
get_neutral_value(time_pressure) -> 0.0;
get_neutral_value(convergence_status) -> 0.5;
get_neutral_value(episode_efficiency) -> 0.5;
get_neutral_value(critical_period_timing) -> 0.5;
get_neutral_value(compute_availability) -> 0.5;
get_neutral_value(network_latency) -> 0.0;
get_neutral_value(stagnation_severity) -> 0.0;

%% Competitive Silo signals
get_neutral_value(competitive_pressure) -> 0.0;
get_neutral_value(strategy_diversity_need) -> 0.5;
get_neutral_value(arms_race_active) -> 0.0;
get_neutral_value(arms_race_load) -> 0.0;
get_neutral_value(coalition_competition) -> 0.5;
get_neutral_value(fitness_pressure) -> 0.5;

%% Social Silo signals
get_neutral_value(selection_influence) -> 0.5;
get_neutral_value(norm_transmission) -> 0.5;
get_neutral_value(coalition_structure) -> 0.5;
get_neutral_value(trust_network) -> 0.5;
get_neutral_value(selection_pressure) -> 0.5;
get_neutral_value(trade_opportunity) -> 0.5;

%% Cultural Silo signals
get_neutral_value(innovation_impact) -> 0.5;
get_neutral_value(strategy_innovation) -> 0.5;
get_neutral_value(plasticity_influence) -> 0.5;
get_neutral_value(information_sharing) -> 0.5;
get_neutral_value(exploration_need) -> 0.5;

%% Ecological Silo signals
get_neutral_value(environmental_pressure) -> 0.0;
get_neutral_value(resource_level) -> 0.5;
get_neutral_value(stress_signal) -> 0.0;
get_neutral_value(environmental_context) -> 0.5;
get_neutral_value(abundance_signal) -> 0.5;
get_neutral_value(adaptation_pressure) -> 0.5;

%% Morphological Silo signals
get_neutral_value(complexity_signal) -> 0.5;
get_neutral_value(size_budget) -> 0.5;
get_neutral_value(efficiency_score) -> 0.5;
get_neutral_value(growth_stage) -> 0.5;
get_neutral_value(complexity_target) -> 0.5;

%% Developmental Silo signals
get_neutral_value(maturity_distribution) -> 0.5;
get_neutral_value(plasticity_available) -> 0.5;
get_neutral_value(metamorphosis_rate) -> 0.0;
get_neutral_value(expression_stage) -> 0.5;
get_neutral_value(maturity_target) -> 0.5;
get_neutral_value(scarcity_signal) -> 0.0;

%% Regulatory Silo signals
get_neutral_value(context_awareness) -> 0.5;
get_neutral_value(expression_flexibility) -> 0.5;
get_neutral_value(dormant_potential) -> 0.5;
get_neutral_value(expression_cost) -> 0.5;
get_neutral_value(context_complexity) -> 0.5;

%% Economic Silo signals
get_neutral_value(economic_pressure) -> 0.0;
get_neutral_value(budget_available) -> 0.5;
get_neutral_value(efficiency_requirement) -> 0.5;
get_neutral_value(budget_signal) -> 0.5;
get_neutral_value(budget_constraint) -> 0.0;

%% Communication Silo signals
get_neutral_value(coordination_capability) -> 0.5;
get_neutral_value(information_transfer) -> 0.5;
get_neutral_value(trust_signal) -> 0.5;
get_neutral_value(strategic_signaling) -> 0.5;
get_neutral_value(coordination_need) -> 0.5;

%% Default for any unspecified signal
get_neutral_value(_) -> 0.5.

%% @private Initial signal values for all silos.
%%
%% Each silo has a map of signals it receives from other silos.
%% Values start at neutral and are updated by emitting silos.
initial_signals() ->
    #{
        %% === Original 3 Silos ===

        resource => #{
            %% From Task
            exploration_boost => 0.0,
            desired_evals_per_individual => 5,
            expected_complexity_growth => 0.0,
            %% From Distribution
            network_load_contribution => 0.0,
            remote_capacity_available => 0.5,
            %% From Temporal
            convergence_status => 0.5,
            %% From Competitive
            arms_race_active => 0.0,
            %% From Ecological
            resource_level => 0.5,
            %% From Morphological
            size_budget => 0.5
        },
        task => #{
            %% From Resource
            pressure_signal => 0.0,
            max_evals_per_individual => 20,
            should_simplify => 0.0,
            %% From Distribution
            island_diversity_score => 0.5,
            migration_activity => 0.0,
            %% From Temporal
            time_pressure => 0.0,
            %% From Competitive
            competitive_pressure => 0.0,
            %% From Social
            selection_influence => 0.5,
            %% From Cultural
            innovation_impact => 0.5,
            %% From Ecological
            environmental_pressure => 0.0,
            %% From Morphological
            complexity_signal => 0.5,
            %% From Developmental
            maturity_distribution => 0.5,
            %% From Regulatory
            context_awareness => 0.5,
            %% From Economic
            economic_pressure => 0.0,
            %% From Communication
            coordination_capability => 0.5
        },
        distribution => #{
            %% From Resource
            offload_preference => 0.0,
            local_capacity => 1.0,
            %% From Task
            diversity_need => 0.5,
            speciation_pressure => 0.5
        },

        %% === New Silos (LC v2) ===

        temporal => #{
            %% From Task
            stagnation_severity => 0.0,
            %% From Resource
            compute_availability => 0.5,
            %% From Distribution
            network_latency => 0.0,
            %% From Economic
            budget_available => 0.5
        },
        competitive => #{
            %% From Task
            fitness_pressure => 0.5,
            %% From Resource
            arms_race_load => 0.0,
            %% From Social
            coalition_structure => 0.5,
            %% From Cultural
            strategy_innovation => 0.5,
            %% From Regulatory
            dormant_potential => 0.5,
            %% From Communication
            strategic_signaling => 0.5
        },
        social => #{
            %% From Task
            selection_pressure => 0.5,
            %% From Competitive
            coalition_competition => 0.5,
            %% From Economic
            trade_opportunity => 0.5,
            %% From Communication
            trust_signal => 0.5
        },
        cultural => #{
            %% From Task
            exploration_need => 0.5,
            %% From Social
            norm_transmission => 0.5,
            %% From Competitive
            strategy_diversity_need => 0.5,
            %% From Developmental
            plasticity_available => 0.5,
            %% From Regulatory
            expression_flexibility => 0.5,
            %% From Communication
            information_transfer => 0.5
        },
        ecological => #{
            %% From Task
            adaptation_pressure => 0.5,
            %% From Resource
            abundance_signal => 0.5,
            %% From Developmental
            metamorphosis_rate => 0.0
        },
        morphological => #{
            %% From Task
            complexity_target => 0.5,
            %% From Developmental
            growth_stage => 0.5,
            %% From Regulatory
            expression_cost => 0.5,
            %% From Economic
            efficiency_requirement => 0.5
        },
        developmental => #{
            %% From Task
            maturity_target => 0.5,
            %% From Resource
            scarcity_signal => 0.0,
            %% From Temporal
            critical_period_timing => 0.5,
            %% From Cultural
            plasticity_influence => 0.5,
            %% From Ecological
            stress_signal => 0.0,
            %% From Regulatory
            expression_stage => 0.5
        },
        regulatory => #{
            %% From Task
            context_complexity => 0.5,
            %% From Ecological
            environmental_context => 0.5,
            %% From Developmental
            expression_stage => 0.5
        },
        economic => #{
            %% From Task
            budget_constraint => 0.0,
            %% From Resource
            budget_signal => 0.5,
            %% From Temporal
            episode_efficiency => 0.5,
            %% From Morphological
            efficiency_score => 0.5
        },
        communication => #{
            %% From Task
            coordination_need => 0.5,
            %% From Social
            trust_network => 0.5,
            %% From Cultural
            information_sharing => 0.5
        }
    }.
