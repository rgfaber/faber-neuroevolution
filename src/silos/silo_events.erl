%% @doc Silo Event Publishing for Liquid Conglomerate.
%%
%% Part of the Liquid Conglomerate v2 event-driven architecture. This module
%% provides topic definitions and publishing helpers for cross-silo communication.
%%
%% == Event-Driven Architecture ==
%%
%% Instead of direct lc_cross_silo:emit() calls (imperative push), silos now
%% publish events to topics. Interested parties subscribe and react.
%%
%% == Topic Hierarchy ==
%%
%%   silo.SILONAME.signals     - Cross-silo signals from a specific silo
%%   silo.SILONAME.lifecycle   - Silo lifecycle events (activated, deactivated)
%%   silo.aggregated.signals   - Aggregated view from lc_cross_silo
%%
%% @author R.G. Lefever
%% @copyright 2024-2026 R.G. Lefever
-module(silo_events).

%% API - Publishing
-export([
    publish_signal/3,
    publish_signals/2,
    publish_lifecycle/2,
    publish_recommendations/2
]).

%% API - Subscriptions
-export([
    subscribe_to_silo/1,
    subscribe_to_silo/2,
    unsubscribe_from_silo/1,
    subscribe_to_all_silos/0,
    subscribe_to_all_silos/1,
    subscribe_to_recommendations/1,
    subscribe_to_recommendations/2,
    unsubscribe_from_recommendations/1
]).

%% API - Topic Helpers
-export([
    signal_topic/1,
    lifecycle_topic/1,
    recommendations_topic/1,
    all_silo_names/0
]).

%%% ============================================================================
%%% Type Definitions
%%% ============================================================================

-type silo_name() :: task | resource | distribution | temporal | competitive |
                     social | cultural | ecological | morphological |
                     developmental | regulatory | economic | communication.

-type signal_name() :: atom().
-type signal_value() :: number().
-type lifecycle_event() :: activated | deactivated | config_changed.

-export_type([silo_name/0, signal_name/0, signal_value/0, lifecycle_event/0]).

%%% ============================================================================
%%% API - Publishing
%%% ============================================================================

%% @doc Publish a single signal from a silo.
%%
%% Event format (map with keys):
%%   event_type - binary "silo_signal"
%%   timestamp  - millisecond timestamp
%%   from       - source silo name
%%   signal     - signal name atom
%%   value      - numeric signal value
-spec publish_signal(silo_name(), signal_name(), signal_value()) -> ok.
publish_signal(FromSilo, SignalName, Value) ->
    Topic = signal_topic(FromSilo),
    Event = #{
        event_type => <<"silo_signal">>,
        timestamp => erlang:system_time(millisecond),
        from => FromSilo,
        signal => SignalName,
        value => Value
    },
    neuroevolution_events:publish(Topic, Event),
    ok.

%% @doc Publish multiple signals from a silo.
%%
%% More efficient than multiple single publishes.
%% Event format (map with keys):
%%   event_type - binary "silo_signals"
%%   timestamp  - millisecond timestamp
%%   from       - source silo name
%%   signals    - map of signal_name to value
-spec publish_signals(silo_name(), #{signal_name() => signal_value()}) -> ok.
publish_signals(FromSilo, Signals) when map_size(Signals) > 0 ->
    Topic = signal_topic(FromSilo),
    Event = #{
        event_type => <<"silo_signals">>,
        timestamp => erlang:system_time(millisecond),
        from => FromSilo,
        signals => Signals
    },
    neuroevolution_events:publish(Topic, Event),
    ok;
publish_signals(_FromSilo, _EmptySignals) ->
    ok.

%% @doc Publish a silo lifecycle event.
%%
%% Used for silo activation, deactivation, and configuration changes.
-spec publish_lifecycle(silo_name(), lifecycle_event()) -> ok.
publish_lifecycle(SiloName, LifecycleEvent) ->
    Topic = lifecycle_topic(SiloName),
    Event = #{
        event_type => <<"silo_lifecycle">>,
        timestamp => erlang:system_time(millisecond),
        silo => SiloName,
        lifecycle_event => LifecycleEvent
    },
    neuroevolution_events:publish(Topic, Event),
    ok.

%% @doc Publish recommendations from a silo.
%%
%% Used for event-driven read models. Silos publish their recommendations
%% whenever they change, and consumers cache the latest values locally.
%% This replaces blocking get_recommendations() calls with cached lookups.
%%
%% Event format (map with keys):
%%   event_type      - binary "silo_recommendations"
%%   timestamp       - millisecond timestamp
%%   silo            - silo name atom
%%   recommendations - map of recommendation data
%%
%% Silo usage: Call publish_recommendations/2 when recommendations change.
%%
%% Consumer usage: Subscribe via subscribe_to_recommendations/1, then
%% handle {silo_recommendations, SiloName, Recs} messages in handle_info.
-spec publish_recommendations(silo_name(), map()) -> ok.
publish_recommendations(SiloName, Recommendations) when is_map(Recommendations) ->
    Topic = recommendations_topic(SiloName),
    Event = #{
        event_type => <<"silo_recommendations">>,
        timestamp => erlang:system_time(millisecond),
        silo => SiloName,
        recommendations => Recommendations
    },
    neuroevolution_events:publish(Topic, Event),
    ok.

%%% ============================================================================
%%% API - Subscriptions
%%% ============================================================================

%% @doc Subscribe the calling process to signals from a specific silo.
-spec subscribe_to_silo(silo_name()) -> ok.
subscribe_to_silo(SiloName) ->
    subscribe_to_silo(SiloName, self()).

%% @doc Subscribe a specific process to signals from a silo.
-spec subscribe_to_silo(silo_name(), pid()) -> ok.
subscribe_to_silo(SiloName, Pid) ->
    Topic = signal_topic(SiloName),
    neuroevolution_events:subscribe(Topic, Pid),
    ok.

%% @doc Unsubscribe from a specific silo's signals.
-spec unsubscribe_from_silo(silo_name()) -> ok.
unsubscribe_from_silo(SiloName) ->
    Topic = signal_topic(SiloName),
    neuroevolution_events:unsubscribe(Topic, self()),
    ok.

%% @doc Subscribe the calling process to signals from all silos.
-spec subscribe_to_all_silos() -> ok.
subscribe_to_all_silos() ->
    subscribe_to_all_silos(self()).

%% @doc Subscribe a specific process to signals from all silos.
-spec subscribe_to_all_silos(pid()) -> ok.
subscribe_to_all_silos(Pid) ->
    lists:foreach(
        fun(SiloName) ->
            subscribe_to_silo(SiloName, Pid)
        end,
        all_silo_names()
    ),
    ok.

%% @doc Subscribe the calling process to recommendations from a specific silo.
%%
%% Events are delivered as: {silo_recommendations, SiloName, RecommendationsMap}
-spec subscribe_to_recommendations(silo_name()) -> ok.
subscribe_to_recommendations(SiloName) ->
    subscribe_to_recommendations(SiloName, self()).

%% @doc Subscribe a specific process to recommendations from a silo.
-spec subscribe_to_recommendations(silo_name(), pid()) -> ok.
subscribe_to_recommendations(SiloName, Pid) ->
    Topic = recommendations_topic(SiloName),
    neuroevolution_events:subscribe(Topic, Pid),
    ok.

%% @doc Unsubscribe from a specific silo's recommendations.
-spec unsubscribe_from_recommendations(silo_name()) -> ok.
unsubscribe_from_recommendations(SiloName) ->
    Topic = recommendations_topic(SiloName),
    neuroevolution_events:unsubscribe(Topic, self()),
    ok.

%%% ============================================================================
%%% API - Topic Helpers
%%% ============================================================================

%% @doc Get the signal topic for a silo.
%%
%% Returns: binary "silo.NAME.signals"
-spec signal_topic(silo_name()) -> binary().
signal_topic(SiloName) when is_atom(SiloName) ->
    iolist_to_binary([<<"silo.">>, atom_to_binary(SiloName, utf8), <<".signals">>]).

%% @doc Get the lifecycle topic for a silo.
%%
%% Returns: binary "silo.NAME.lifecycle"
-spec lifecycle_topic(silo_name()) -> binary().
lifecycle_topic(SiloName) when is_atom(SiloName) ->
    iolist_to_binary([<<"silo.">>, atom_to_binary(SiloName, utf8), <<".lifecycle">>]).

%% @doc Get the recommendations topic for a silo.
%%
%% Returns: binary "silo.NAME.recommendations"
-spec recommendations_topic(silo_name()) -> binary().
recommendations_topic(SiloName) when is_atom(SiloName) ->
    iolist_to_binary([<<"silo.">>, atom_to_binary(SiloName, utf8), <<".recommendations">>]).

%% @doc Get all silo names in the Liquid Conglomerate.
-spec all_silo_names() -> [silo_name()].
all_silo_names() ->
    [
        %% Original 3 silos
        task,
        resource,
        distribution,
        %% LC v2 extension silos
        temporal,
        competitive,
        social,
        cultural,
        ecological,
        morphological,
        developmental,
        regulatory,
        economic,
        communication
    ].
