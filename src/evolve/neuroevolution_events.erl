%% @doc Event publishing abstraction for neuroevolution.
%%
%% This module provides a pluggable event system that can use different
%% backends for event distribution:
%%
%% - `neuroevolution_events_local' - Local pg-based pubsub (default)
%% - `neuroevolution_events_macula' - Distributed via Macula mesh (future)
%%
%% == Event Types ==
%%
%% Commands (requests for work):
%% - `{evaluate_request, #{request_id, realm, individual_id, network, options}}'
%% - `{evaluate_batch_request, #{request_id, realm, individuals, options}}'
%%
%% Events (facts that happened):
%% - `{evaluated, #{request_id, individual_id, metrics, evaluator_node}}'
%% - `{generation_started, #{realm, generation, population_size, timestamp}}'
%% - `{generation_completed, #{realm, generation, best_fitness, avg_fitness, ...}}'
%% - `{training_started, #{realm, config}}'
%% - `{training_stopped, #{realm, generation, reason}}'
%%
%% == Topic Design ==
%%
%% Topics follow the pattern: `evolution:<realm>:<type>'
%% - `evolution:default:generation_complete' - Generation lifecycle events
%% - `evolution:default:training' - Training lifecycle events
%% - `evolution:default:intervention' - LC intervention events
%% - `evolution:default:resource_alert' - Resource pressure alerts
%%
%% Legacy topics (for distributed evaluation):
%% - `neuro.default.evaluate' - Evaluation requests
%% - `neuro.default.evaluated' - Evaluation results
%%
%% @author R.G. Lefever
%% @copyright 2024-2026 R.G. Lefever
-module(neuroevolution_events).

-export([
    publish/2,
    subscribe/1,
    subscribe/2,
    unsubscribe/1,
    set_backend/1,
    get_backend/0,
    %% Standardized event helpers (v4)
    make_event/3,
    make_event/4,
    publish_event/5,
    %% Legacy topic helpers (deprecated)
    evaluate_topic/1,
    evaluated_topic/1,
    events_topic/1,
    %% Multi-channel topic constructors (v2)
    topic_generation/1,
    topic_population/1,
    topic_species/1,
    topic_training/1,
    topic_meta/1,
    topic_individual/1,
    %% Strategy-agnostic topic constructors (v2.1)
    topic_progress/1,
    topic_island/1,
    topic_novelty/1,
    topic_qd/1,
    topic_competition/1,
    topic_environment/1,
    %% LC Silo event topics (v3 - direct Phoenix.PubSub)
    topic_intervention/1,
    topic_resource_alert/1,
    topic_silo_sensors/1
]).

%% Behaviour callbacks that backends must implement
-callback publish(Topic :: binary(), Event :: term()) -> ok.
-callback subscribe(Topic :: binary(), Pid :: pid()) -> ok.
-callback unsubscribe(Topic :: binary(), Pid :: pid()) -> ok.

%%% ============================================================================
%%% API Functions
%%% ============================================================================

%% @doc Publish an event to a topic.
%%
%% The event will be delivered to all subscribers of the topic.
%% Event format is typically `{EventType, EventData}' where EventData is a map.
-spec publish(Topic, Event) -> ok when
    Topic :: binary(),
    Event :: term().
publish(Topic, Event) ->
    Backend = get_backend(),
    Backend:publish(Topic, Event).

%% @doc Subscribe the calling process to a topic.
-spec subscribe(Topic) -> ok when
    Topic :: binary().
subscribe(Topic) ->
    subscribe(Topic, self()).

%% @doc Subscribe a specific process to a topic.
-spec subscribe(Topic, Pid) -> ok when
    Topic :: binary(),
    Pid :: pid().
subscribe(Topic, Pid) ->
    Backend = get_backend(),
    Backend:subscribe(Topic, Pid).

%% @doc Unsubscribe the calling process from a topic.
-spec unsubscribe(Topic) -> ok when
    Topic :: binary().
unsubscribe(Topic) ->
    Backend = get_backend(),
    Backend:unsubscribe(Topic, self()).

%% @doc Set the event backend module.
%%
%% The backend module must implement the neuroevolution_events behaviour.
%% Default is `neuroevolution_events_local'.
-spec set_backend(Module) -> ok when
    Module :: module().
set_backend(Module) ->
    persistent_term:put({?MODULE, backend}, Module),
    ok.

%% @doc Get the current event backend module.
-spec get_backend() -> module().
get_backend() ->
    persistent_term:get({?MODULE, backend}, neuroevolution_events_local).

%%% ============================================================================
%%% Standardized Event Helpers (v4)
%%% ============================================================================
%%
%% All events SHOULD include these standard fields:
%% - `realm' - Session/domain identifier (required)
%% - `source' - Publisher module atom (required)
%% - `timestamp' - Milliseconds since epoch (auto-added)
%% - `payload' - Event-specific data (optional, prefer flat structure)
%%
%% Using these helpers ensures consistent event structure across all publishers.

%% @doc Create a standardized event map with realm, source, and timestamp.
%%
%% Example:
%% ```
%% Event = neuroevolution_events:make_event(<<"snake_duel">>, task_silo, #{
%%     event_type => task_silo_intervention_started,
%%     stagnation_severity => 0.75,
%%     exploration_boost => 0.3
%% }).
%% '''
-spec make_event(Realm, Source, Payload) -> {atom(), map()} when
    Realm :: binary(),
    Source :: atom(),
    Payload :: map().
make_event(Realm, Source, Payload) when is_map(Payload) ->
    EventType = maps:get(event_type, Payload, undefined),
    BaseEvent = #{
        realm => Realm,
        source => Source,
        timestamp => erlang:system_time(millisecond)
    },
    %% Merge payload into base event (flat structure preferred)
    FinalEvent = maps:merge(BaseEvent, maps:remove(event_type, Payload)),
    {EventType, FinalEvent}.

%% @doc Create a standardized event with explicit event type.
%%
%% Example:
%% ```
%% Event = neuroevolution_events:make_event(generation_complete, <<"snake_duel">>,
%%     neuroevolution_server, #{
%%         cohort => 42,
%%         best_fitness => 0.95
%%     }).
%% '''
-spec make_event(EventType, Realm, Source, Payload) -> {atom(), map()} when
    EventType :: atom(),
    Realm :: binary(),
    Source :: atom(),
    Payload :: map().
make_event(EventType, Realm, Source, Payload) when is_map(Payload) ->
    BaseEvent = #{
        realm => Realm,
        source => Source,
        timestamp => erlang:system_time(millisecond)
    },
    {EventType, maps:merge(BaseEvent, Payload)}.

%% @doc Create and publish a standardized event in one call.
%%
%% Example:
%% ```
%% neuroevolution_events:publish_event(
%%     topic_intervention(Realm),
%%     task_silo_intervention_started,
%%     Realm,
%%     task_silo,
%%     #{stagnation_severity => 0.75, exploration_boost => 0.3}
%% ).
%% '''
-spec publish_event(Topic, EventType, Realm, Source, Payload) -> ok when
    Topic :: binary(),
    EventType :: atom(),
    Realm :: binary(),
    Source :: atom(),
    Payload :: map().
publish_event(Topic, EventType, Realm, Source, Payload) ->
    Event = make_event(EventType, Realm, Source, Payload),
    publish(Topic, Event).


%%% ============================================================================
%%% Helper Functions for Topic Construction
%%% ============================================================================

%% @doc Get the evaluation request topic for a realm.
-spec evaluate_topic(Realm) -> binary() when
    Realm :: binary().
evaluate_topic(Realm) ->
    <<"neuro.", Realm/binary, ".evaluate">>.

%% @doc Get the evaluation results topic for a realm.
-spec evaluated_topic(Realm) -> binary() when
    Realm :: binary().
evaluated_topic(Realm) ->
    <<"neuro.", Realm/binary, ".evaluated">>.

%% @doc Get the training events topic for a realm.
%% @deprecated Use topic_generation/1, topic_population/1, etc. instead.
-spec events_topic(Realm) -> binary() when
    Realm :: binary().
events_topic(Realm) ->
    <<"evolution:", Realm/binary, ":events">>.

%%% ============================================================================
%%% Multi-Channel Topic Constructors (v2 - Event-Carried State)
%%% ============================================================================
%%
%% These topic constructors enable subscribers to receive only the events they
%% care about, reducing unnecessary message processing. Events on these topics
%% carry rich payloads with all data needed for visualization.

%% @doc Topic for generation lifecycle events (low frequency, rich payload).
%%
%% Events:
%% - `{generation_started, #{realm, generation, population_size, timestamp}}'
%% - `{generation_complete, #{realm, generation, generation_stats, history_point,
%%                            meta_params, silo_state, convergence_status, ...}}'
-spec topic_generation(Realm) -> binary() when
    Realm :: binary().
topic_generation(Realm) ->
    <<"evolution:", Realm/binary, ":generation">>.

%% @doc Topic for population-level events (medium frequency).
%%
%% Events:
%% - `{population_evaluated, #{realm, generation, population, top_individuals, ...}}'
%% - `{selection_complete, #{realm, generation, survivors, offspring}}'
-spec topic_population(Realm) -> binary() when
    Realm :: binary().
topic_population(Realm) ->
    <<"evolution:", Realm/binary, ":population">>.

%% @doc Topic for species lifecycle events (low frequency).
%%
%% Events:
%% - `{species_updated, #{realm, generation, species, species_count}}'
%% - `{species_emerged, #{realm, species_id, founding_member}}'
%% - `{species_extinct, #{realm, species_id, final_generation}}'
-spec topic_species(Realm) -> binary() when
    Realm :: binary().
topic_species(Realm) ->
    <<"evolution:", Realm/binary, ":species">>.

%% @doc Topic for training lifecycle events (rare, important).
%%
%% Events:
%% - `{training_started, #{realm, config, timestamp}}'
%% - `{training_paused, #{realm, generation, reason}}'
%% - `{training_resumed, #{realm, generation}}'
%% - `{training_stopped, #{realm, generation, reason, final_stats}}'
%% - `{training_complete, #{realm, generation, best_individual, elapsed_ms}}'
-spec topic_training(Realm) -> binary() when
    Realm :: binary().
topic_training(Realm) ->
    <<"evolution:", Realm/binary, ":training">>.

%% @doc Topic for meta-controller parameter updates (low frequency).
%%
%% Events:
%% - `{meta_params_updated, #{realm, generation, params, silo_states}}'
-spec topic_meta(Realm) -> binary() when
    Realm :: binary().
topic_meta(Realm) ->
    <<"evolution:", Realm/binary, ":meta">>.

%% @doc Topic for individual-level events (high frequency, use sparingly).
%%
%% Events:
%% - `{individual_evaluated, #{realm, individual_id, fitness, metrics}}'
%%
%% Note: For UI purposes, prefer subscribing to topic_population/1 which
%% provides aggregated data. This topic is for distributed evaluation workers.
-spec topic_individual(Realm) -> binary() when
    Realm :: binary().
topic_individual(Realm) ->
    <<"evolution:", Realm/binary, ":individual">>.

%%% ============================================================================
%%% Strategy-Agnostic Topic Constructors (v2.1)
%%% ============================================================================
%%
%% These topics support different evolution strategies with strategy-agnostic
%% event types. They enable the UI to work with both generational and
%% continuous evolution approaches.

%% @doc Topic for strategy-agnostic progress events (configurable frequency).
%%
%% Events:
%% - `{progress_checkpoint, #progress_checkpoint{}}' - Periodic progress updates
%%
%% This topic is useful for continuous evolution strategies (steady-state)
%% where traditional "generation" boundaries don't apply. Progress checkpoints
%% are emitted every N evaluations (configurable).
-spec topic_progress(Realm) -> binary() when
    Realm :: binary().
topic_progress(Realm) ->
    <<"evolution:", Realm/binary, ":progress">>.

%% @doc Topic for island model events (medium frequency).
%%
%% Events:
%% - `{island_migration, #island_migration{}}' - Individual migrated between islands
%% - `{island_topology_changed, #island_topology_changed{}}' - Island connectivity changed
-spec topic_island(Realm) -> binary() when
    Realm :: binary().
topic_island(Realm) ->
    <<"evolution:", Realm/binary, ":island">>.

%% @doc Topic for novelty search events (medium frequency).
%%
%% Events:
%% - `{archive_updated, #archive_updated{}}' - Novelty archive changed
%% - `{niche_discovered, #niche_discovered{}}' - New behavioral niche found
-spec topic_novelty(Realm) -> binary() when
    Realm :: binary().
topic_novelty(Realm) ->
    <<"evolution:", Realm/binary, ":novelty">>.

%% @doc Topic for quality-diversity (MAP-Elites) events (medium frequency).
%%
%% Events:
%% - `{niche_discovered, #niche_discovered{}}' - New cell in grid occupied
%% - `{niche_updated, #niche_updated{}}' - Better individual in existing cell
%% - `{archive_updated, #archive_updated{}}' - Grid coverage/QD-score updated
-spec topic_qd(Realm) -> binary() when
    Realm :: binary().
topic_qd(Realm) ->
    <<"evolution:", Realm/binary, ":qd">>.

%% @doc Topic for competitive coevolution events (medium frequency).
%%
%% Competitive Coevolution (Red Team vs Blue Team) events:
%% - `{red_team_updated, #{realm, red_team_size, red_team_avg_fitness,
%%                         red_team_max_fitness, members_added}}'
%% - `{blue_team_evaluated, #{realm, blue_team_best_fitness,
%%                            red_team_opponent_id, result}}'
%% - `{immigration_occurred, #{realm, direction, count, individuals}}'
%%   direction: `blue_to_red' or `red_to_blue'
%% - `{arms_race_progress, #{realm, generation, blue_best, red_best,
%%                           blue_improvement_rate, red_improvement_rate}}'
%%
%% The arms race dynamic is key: both populations must continuously improve
%% to beat each other, preventing overfitting to static opponents.
-spec topic_competition(Realm) -> binary() when
    Realm :: binary().
topic_competition(Realm) ->
    <<"evolution:", Realm/binary, ":competition">>.

%% @doc Topic for environment change events (low frequency).
%%
%% Events:
%% - `{environment_changed, #environment_changed{}}' - Fitness landscape changed
%%
%% Used for curriculum learning, adaptive difficulty, or simulating
%% changing real-world conditions.
-spec topic_environment(Realm) -> binary() when
    Realm :: binary().
topic_environment(Realm) ->
    <<"evolution:", Realm/binary, ":environment">>.

%%% ============================================================================
%%% LC Silo Event Topics (v3 - Direct Phoenix.PubSub Integration)
%%% ============================================================================
%%
%% These topics are designed for direct publishing to Phoenix.PubSub via
%% MaculaNeurolab.EventBridge, bypassing the local pg-based pubsub for
%% low-latency UI updates.

%% @doc Topic for LC intervention events (Task Silo actions).
%%
%% Events:
%% - `{task_silo_intervention_started, #{realm, generation, exploration_boost, ...}}'
%% - `{task_silo_intervention_ended, #{realm, generation, prev_boost}}'
%% - `{task_silo_intervention_intensified, #{realm, generation, old_boost, new_boost}}'
%%
%% These events are emitted when the Task Silo intervenes to address stagnation.
%% The dashboard subscribes directly to this topic for immediate UI updates.
-spec topic_intervention(Realm) -> binary() when
    Realm :: binary().
topic_intervention(Realm) ->
    <<"evolution:", Realm/binary, ":intervention">>.

%% @doc Topic for resource pressure alerts (Resource Silo warnings).
%%
%% Events:
%% - `{resource_alert, #{realm, action, memory_pressure, message}}'
%%
%% Emitted when resource pressure changes from `continue' to `throttle' or `pause',
%% or when recovering from a resource-constrained state.
-spec topic_resource_alert(Realm) -> binary() when
    Realm :: binary().
topic_resource_alert(Realm) ->
    <<"evolution:", Realm/binary, ":resource_alert">>.

%% @doc Topic for real-time LC silo sensor updates (high frequency).
%%
%% Events:
%% - `{task_sensors_updated, #{realm, sensors}}'
%%   Emitted when Task Silo L0 sensors change significantly.
%% - `{resource_sensors_updated, #{realm, sensors}}'
%%   Emitted when Resource Silo L0 sensors change significantly.
%% - `{distribution_sensors_updated, #{realm, sensors}}'
%%   Emitted when Distribution Silo L0 sensors change significantly.
%%
%% Sensors map contains all current L0 sensor values. Events are throttled
%% to avoid flooding - only emitted when significant changes occur or
%% at a maximum rate of 10Hz.
-spec topic_silo_sensors(Realm) -> binary() when
    Realm :: binary().
topic_silo_sensors(Realm) ->
    <<"evolution:", Realm/binary, ":silo_sensors">>.
