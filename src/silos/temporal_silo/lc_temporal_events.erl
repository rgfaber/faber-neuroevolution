%%%-------------------------------------------------------------------
%%% @doc Behavioral event constructors for Temporal Silo.
%%%
%%% Events related to timing, episodes, and learning rate adaptation.
%%% Temporal Silo operates at timescale τ=10 (fastest adaptation).
%%%
%%% @end
%%%-------------------------------------------------------------------
-module(lc_temporal_events).

-include("lc_temporal_events.hrl").

%% Event constructors
-export([
    episode_started/1,
    episode_completed/1,
    timing_adjusted/1,
    learning_rate_adapted/1,
    patience_exhausted/1,
    convergence_detected/1,
    timeout_threshold_reached/1
]).

%% Utility exports
-export([
    make_meta/1,
    make_meta/2,
    event_to_map/1
]).

%%% ============================================================================
%%% Metadata Construction
%%% ============================================================================

-spec make_meta(Emitter :: atom()) -> #lc_event_meta{}.
make_meta(Emitter) ->
    make_meta(Emitter, #{}).

-spec make_meta(Emitter :: atom(), Opts :: map()) -> #lc_event_meta{}.
make_meta(Emitter, Opts) ->
    #lc_event_meta{
        event_id = maps:get(event_id, Opts, generate_id()),
        correlation_id = maps:get(correlation_id, Opts, undefined),
        causation_id = maps:get(causation_id, Opts, undefined),
        timestamp = maps:get(timestamp, Opts, erlang:system_time(microsecond)),
        version = maps:get(version, Opts, ?LC_EVENT_VERSION),
        emitter = Emitter
    }.

%%% ============================================================================
%%% Episode Event Constructors
%%% ============================================================================

-spec episode_started(map()) -> #episode_started{}.
episode_started(Data) ->
    #episode_started{
        meta = make_meta(maps:get(emitter, Data, temporal_silo), Data),
        individual_id = maps:get(individual_id, Data),
        episode_number = maps:get(episode_number, Data),
        expected_duration_ms = maps:get(expected_duration_ms, Data, undefined),
        generation = maps:get(generation, Data, 0)
    }.

-spec episode_completed(map()) -> #episode_completed{}.
episode_completed(Data) ->
    #episode_completed{
        meta = make_meta(maps:get(emitter, Data, temporal_silo), Data),
        individual_id = maps:get(individual_id, Data),
        episode_number = maps:get(episode_number, Data),
        duration_ms = maps:get(duration_ms, Data),
        outcome = maps:get(outcome, Data, success),
        fitness_delta = maps:get(fitness_delta, Data, 0.0),
        steps_taken = maps:get(steps_taken, Data, 0)
    }.

%%% ============================================================================
%%% Timing Event Constructors
%%% ============================================================================

-spec timing_adjusted(map()) -> #timing_adjusted{}.
timing_adjusted(Data) ->
    #timing_adjusted{
        meta = make_meta(maps:get(emitter, Data, temporal_silo), Data),
        population_id = maps:get(population_id, Data),
        previous_timeout_ms = maps:get(previous_timeout_ms, Data),
        new_timeout_ms = maps:get(new_timeout_ms, Data),
        adjustment_reason = maps:get(adjustment_reason, Data, convergence)
    }.

-spec learning_rate_adapted(map()) -> #learning_rate_adapted{}.
learning_rate_adapted(Data) ->
    #learning_rate_adapted{
        meta = make_meta(maps:get(emitter, Data, temporal_silo), Data),
        population_id = maps:get(population_id, Data),
        previous_rate = maps:get(previous_rate, Data),
        new_rate = maps:get(new_rate, Data),
        adaptation_reason = maps:get(adaptation_reason, Data, scheduled),
        generation = maps:get(generation, Data)
    }.

%%% ============================================================================
%%% Termination Event Constructors
%%% ============================================================================

-spec patience_exhausted(map()) -> #patience_exhausted{}.
patience_exhausted(Data) ->
    #patience_exhausted{
        meta = make_meta(maps:get(emitter, Data, temporal_silo), Data),
        population_id = maps:get(population_id, Data),
        generations_waited = maps:get(generations_waited, Data),
        best_fitness = maps:get(best_fitness, Data),
        action_taken = maps:get(action_taken, Data, terminate)
    }.

-spec convergence_detected(map()) -> #convergence_detected{}.
convergence_detected(Data) ->
    #convergence_detected{
        meta = make_meta(maps:get(emitter, Data, temporal_silo), Data),
        population_id = maps:get(population_id, Data),
        fitness_variance = maps:get(fitness_variance, Data),
        generations_stable = maps:get(generations_stable, Data),
        generation = maps:get(generation, Data)
    }.

-spec timeout_threshold_reached(map()) -> #timeout_threshold_reached{}.
timeout_threshold_reached(Data) ->
    #timeout_threshold_reached{
        meta = make_meta(maps:get(emitter, Data, temporal_silo), Data),
        individual_id = maps:get(individual_id, Data),
        elapsed_ms = maps:get(elapsed_ms, Data),
        threshold_ms = maps:get(threshold_ms, Data),
        action = maps:get(action, Data, warn)
    }.

%%% ============================================================================
%%% Utility Functions
%%% ============================================================================

-spec event_to_map(temporal_event()) -> map().
event_to_map(Event) when is_tuple(Event) ->
    [RecordName | Fields] = tuple_to_list(Event),
    FieldNames = record_fields(RecordName),
    MetaMap = meta_to_map(hd(Fields)),
    DataMap = maps:from_list(lists:zip(tl(FieldNames), tl(Fields))),
    maps:merge(#{event_type => RecordName, meta => MetaMap}, DataMap).

%%% ============================================================================
%%% Internal Functions
%%% ============================================================================

generate_id() ->
    Bytes = crypto:strong_rand_bytes(16),
    binary:encode_hex(Bytes).

meta_to_map(#lc_event_meta{} = Meta) ->
    #{
        event_id => Meta#lc_event_meta.event_id,
        correlation_id => Meta#lc_event_meta.correlation_id,
        causation_id => Meta#lc_event_meta.causation_id,
        timestamp => Meta#lc_event_meta.timestamp,
        version => Meta#lc_event_meta.version,
        emitter => Meta#lc_event_meta.emitter
    }.

record_fields(episode_started) -> [meta, individual_id, episode_number, expected_duration_ms, generation];
record_fields(episode_completed) -> [meta, individual_id, episode_number, duration_ms, outcome, fitness_delta, steps_taken];
record_fields(timing_adjusted) -> [meta, population_id, previous_timeout_ms, new_timeout_ms, adjustment_reason];
record_fields(learning_rate_adapted) -> [meta, population_id, previous_rate, new_rate, adaptation_reason, generation];
record_fields(patience_exhausted) -> [meta, population_id, generations_waited, best_fitness, action_taken];
record_fields(convergence_detected) -> [meta, population_id, fitness_variance, generations_stable, generation];
record_fields(timeout_threshold_reached) -> [meta, individual_id, elapsed_ms, threshold_ms, action].
