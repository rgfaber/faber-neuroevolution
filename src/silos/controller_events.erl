%% @doc Controller Event Publishing for Liquid Conglomerate.
%%
%% Part of the Liquid Conglomerate v2 event-driven architecture. This module
%% provides topic definitions and publishing helpers for L0/L1/L2 controller
%% chain communication.
%%
%% == Event-Driven Controller Chain ==
%%
%% Instead of direct function calls between controller levels, each level
%% publishes events to topics. Higher levels subscribe and react.
%%
%% == Topic Hierarchy ==
%%
%%   controller.reward                - Reward signals for learning
%%   controller.SILO.l0.metrics       - L0 performance metrics
%%   controller.SILO.l1.metrics       - L1 performance metrics
%%   controller.SILO.l2.guidance      - L2 strategic guidance
%%   controller.population.metrics    - Population-level metrics
%%
%% == Data Flow (Event-Driven) ==
%%
%% 1. Evolution engine publishes controller.reward with reward signal
%% 2. L0 publishes controller.SILO.l0.metrics after each update
%% 3. L1 subscribes to L0 metrics, publishes controller.SILO.l1.metrics
%% 4. L2 subscribes to L1 metrics, publishes controller.SILO.l2.guidance
%% 5. L0/L1 subscribe to guidance to adjust their behavior
%%
%% @author R.G. Lefever
%% @copyright 2024-2026 R.G. Lefever
-module(controller_events).

%% API - Publishing
-export([
    publish_reward/2,
    publish_l0_metrics/2,
    publish_l1_metrics/2,
    publish_l2_guidance/2,
    publish_population_metrics/1
]).

%% API - Subscriptions
-export([
    subscribe_to_reward/0,
    subscribe_to_reward/1,
    subscribe_to_l0_metrics/1,
    subscribe_to_l0_metrics/2,
    subscribe_to_l1_metrics/1,
    subscribe_to_l1_metrics/2,
    subscribe_to_l2_guidance/1,
    subscribe_to_l2_guidance/2,
    subscribe_to_population_metrics/0,
    subscribe_to_population_metrics/1
]).

%% API - Topic Helpers
-export([
    reward_topic/0,
    l0_metrics_topic/1,
    l1_metrics_topic/1,
    l2_guidance_topic/1,
    population_metrics_topic/0
]).

%%% ============================================================================
%%% Type Definitions
%%% ============================================================================

-type silo_name() :: task | resource | distribution | temporal | competitive |
                     social | cultural | ecological | morphological |
                     developmental | regulatory | economic | communication.

-export_type([silo_name/0]).

%%% ============================================================================
%%% API - Publishing
%%% ============================================================================

%% @doc Publish a reward signal for controller learning.
%%
%% This replaces imperative calls like:
%% - lc_chain:train(ChainPid, Reward)
%% - lc_silo_chain:report_reward(Chain, Reward)
%%
%% Event format (map with keys):
%%   event_type - binary "reward_signal"
%%   timestamp  - millisecond timestamp
%%   silo       - silo name atom
%%   reward     - float reward value
%%   source     - caller module atom
-spec publish_reward(silo_name() | global, float()) -> ok.
publish_reward(Silo, Reward) ->
    Topic = reward_topic(),
    Event = #{
        event_type => <<"reward_signal">>,
        timestamp => erlang:system_time(millisecond),
        silo => Silo,
        reward => Reward,
        source => caller_module()
    },
    neuroevolution_events:publish(Topic, Event),
    ok.

%% @doc Publish L0 controller metrics.
%%
%% Called by L0 after each update cycle to report performance.
%% L1 subscribes to adjust its hyperparameter deltas.
%%
%% Metrics typically include:
%% - reward: Recent reward signal
%% - hyperparameters: Current L0 output values
%% - update_count: Number of L0 updates
-spec publish_l0_metrics(silo_name(), map()) -> ok.
publish_l0_metrics(Silo, Metrics) ->
    Topic = l0_metrics_topic(Silo),
    Event = #{
        event_type => <<"l0_metrics">>,
        timestamp => erlang:system_time(millisecond),
        silo => Silo,
        metrics => Metrics,
        source => caller_module()
    },
    neuroevolution_events:publish(Topic, Event),
    ok.

%% @doc Publish L1 controller metrics.
%%
%% Called by L1 after processing L0 metrics.
%% L2 subscribes to adjust its strategic parameters.
%%
%% Metrics typically include:
%% - cumulative_reward: Sum of L0 rewards over L1 window
%% - hyperparameter_deltas: Current L1 output values
%% - observations: Number of L0 observations
-spec publish_l1_metrics(silo_name(), map()) -> ok.
publish_l1_metrics(Silo, Metrics) ->
    Topic = l1_metrics_topic(Silo),
    Event = #{
        event_type => <<"l1_metrics">>,
        timestamp => erlang:system_time(millisecond),
        silo => Silo,
        metrics => Metrics,
        source => caller_module()
    },
    neuroevolution_events:publish(Topic, Event),
    ok.

%% @doc Publish L2 strategic guidance.
%%
%% Called by L2 to provide guidance to L1 and L0.
%%
%% Guidance typically includes:
%% - l1_hyperparameters: Hyperparameters for L1 controller
%% - exploration_rate: How much L1 should explore
%% - adaptation_speed: How fast L1 should adapt
-spec publish_l2_guidance(silo_name(), map()) -> ok.
publish_l2_guidance(Silo, Guidance) ->
    Topic = l2_guidance_topic(Silo),
    Event = #{
        event_type => <<"l2_guidance">>,
        timestamp => erlang:system_time(millisecond),
        silo => Silo,
        guidance => Guidance,
        source => caller_module()
    },
    neuroevolution_events:publish(Topic, Event),
    ok.

%% @doc Publish population-level metrics.
%%
%% Called by lc_population after processing training metrics.
%%
%% Metrics typically include:
%% - generation: Current generation number
%% - active_agent: Currently active controller
%% - fitness_scores: All agent fitness scores
%% - trial_progress: Progress through current trial
-spec publish_population_metrics(map()) -> ok.
publish_population_metrics(Metrics) ->
    Topic = population_metrics_topic(),
    Event = #{
        event_type => <<"population_metrics">>,
        timestamp => erlang:system_time(millisecond),
        metrics => Metrics,
        source => caller_module()
    },
    neuroevolution_events:publish(Topic, Event),
    ok.

%%% ============================================================================
%%% API - Subscriptions
%%% ============================================================================

%% @doc Subscribe the calling process to reward signals.
-spec subscribe_to_reward() -> ok.
subscribe_to_reward() ->
    subscribe_to_reward(self()).

%% @doc Subscribe a specific process to reward signals.
-spec subscribe_to_reward(pid()) -> ok.
subscribe_to_reward(Pid) ->
    Topic = reward_topic(),
    neuroevolution_events:subscribe(Topic, Pid),
    ok.

%% @doc Subscribe to L0 metrics for a specific silo.
-spec subscribe_to_l0_metrics(silo_name()) -> ok.
subscribe_to_l0_metrics(Silo) ->
    subscribe_to_l0_metrics(Silo, self()).

%% @doc Subscribe a specific process to L0 metrics for a silo.
-spec subscribe_to_l0_metrics(silo_name(), pid()) -> ok.
subscribe_to_l0_metrics(Silo, Pid) ->
    Topic = l0_metrics_topic(Silo),
    neuroevolution_events:subscribe(Topic, Pid),
    ok.

%% @doc Subscribe to L1 metrics for a specific silo.
-spec subscribe_to_l1_metrics(silo_name()) -> ok.
subscribe_to_l1_metrics(Silo) ->
    subscribe_to_l1_metrics(Silo, self()).

%% @doc Subscribe a specific process to L1 metrics for a silo.
-spec subscribe_to_l1_metrics(silo_name(), pid()) -> ok.
subscribe_to_l1_metrics(Silo, Pid) ->
    Topic = l1_metrics_topic(Silo),
    neuroevolution_events:subscribe(Topic, Pid),
    ok.

%% @doc Subscribe to L2 guidance for a specific silo.
-spec subscribe_to_l2_guidance(silo_name()) -> ok.
subscribe_to_l2_guidance(Silo) ->
    subscribe_to_l2_guidance(Silo, self()).

%% @doc Subscribe a specific process to L2 guidance for a silo.
-spec subscribe_to_l2_guidance(silo_name(), pid()) -> ok.
subscribe_to_l2_guidance(Silo, Pid) ->
    Topic = l2_guidance_topic(Silo),
    neuroevolution_events:subscribe(Topic, Pid),
    ok.

%% @doc Subscribe to population-level metrics.
-spec subscribe_to_population_metrics() -> ok.
subscribe_to_population_metrics() ->
    subscribe_to_population_metrics(self()).

%% @doc Subscribe a specific process to population-level metrics.
-spec subscribe_to_population_metrics(pid()) -> ok.
subscribe_to_population_metrics(Pid) ->
    Topic = population_metrics_topic(),
    neuroevolution_events:subscribe(Topic, Pid),
    ok.

%%% ============================================================================
%%% API - Topic Helpers
%%% ============================================================================

%% @doc Get the reward signal topic.
-spec reward_topic() -> binary().
reward_topic() ->
    <<"controller.reward">>.

%% @doc Get the L0 metrics topic for a silo.
-spec l0_metrics_topic(silo_name()) -> binary().
l0_metrics_topic(Silo) when is_atom(Silo) ->
    iolist_to_binary([<<"controller.">>, atom_to_binary(Silo, utf8), <<".l0.metrics">>]).

%% @doc Get the L1 metrics topic for a silo.
-spec l1_metrics_topic(silo_name()) -> binary().
l1_metrics_topic(Silo) when is_atom(Silo) ->
    iolist_to_binary([<<"controller.">>, atom_to_binary(Silo, utf8), <<".l1.metrics">>]).

%% @doc Get the L2 guidance topic for a silo.
-spec l2_guidance_topic(silo_name()) -> binary().
l2_guidance_topic(Silo) when is_atom(Silo) ->
    iolist_to_binary([<<"controller.">>, atom_to_binary(Silo, utf8), <<".l2.guidance">>]).

%% @doc Get the population metrics topic.
-spec population_metrics_topic() -> binary().
population_metrics_topic() ->
    <<"controller.population.metrics">>.

%%% ============================================================================
%%% Internal Functions
%%% ============================================================================

%% @private Get the calling module for event tracing.
caller_module() ->
    case erlang:process_info(self(), current_stacktrace) of
        {current_stacktrace, [{_, _, _, _}, {_, _, _, _}, {Module, _, _, _} | _]} ->
            Module;
        _ ->
            unknown
    end.
