%%%-------------------------------------------------------------------
%%% @doc Behavioral event constructors for Competitive Silo.
%%%
%%% Events related to Elo ratings, opponent archives, and matchmaking.
%%% Competitive Silo operates at timescale τ=15.
%%%
%%% @end
%%%-------------------------------------------------------------------
-module(lc_competitive_events).

-include("lc_competitive_events.hrl").

%% Event constructors
-export([
    opponent_archived/1,
    opponent_retired/1,
    match_completed/1,
    elo_updated/1,
    strategy_countered/1,
    arms_race_detected/1,
    matchmaking_adjusted/1,
    dominance_matrix_updated/1
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
%%% Archive Event Constructors
%%% ============================================================================

-spec opponent_archived(map()) -> #opponent_archived{}.
opponent_archived(Data) ->
    #opponent_archived{
        meta = make_meta(maps:get(emitter, Data, competitive_silo), Data),
        opponent_id = maps:get(opponent_id, Data),
        archive_id = maps:get(archive_id, Data),
        fitness_at_archive = maps:get(fitness_at_archive, Data),
        elo_at_archive = maps:get(elo_at_archive, Data, 1500.0),
        generation = maps:get(generation, Data),
        archive_size_after = maps:get(archive_size_after, Data)
    }.

-spec opponent_retired(map()) -> #opponent_retired{}.
opponent_retired(Data) ->
    #opponent_retired{
        meta = make_meta(maps:get(emitter, Data, competitive_silo), Data),
        opponent_id = maps:get(opponent_id, Data),
        archive_id = maps:get(archive_id, Data),
        retirement_reason = maps:get(retirement_reason, Data, age),
        games_played = maps:get(games_played, Data, 0),
        final_elo = maps:get(final_elo, Data)
    }.

%%% ============================================================================
%%% Match Event Constructors
%%% ============================================================================

-spec match_completed(map()) -> #match_completed{}.
match_completed(Data) ->
    #match_completed{
        meta = make_meta(maps:get(emitter, Data, competitive_silo), Data),
        match_id = maps:get(match_id, Data, generate_id()),
        player_a_id = maps:get(player_a_id, Data),
        player_b_id = maps:get(player_b_id, Data),
        winner_id = maps:get(winner_id, Data),
        player_a_elo_before = maps:get(player_a_elo_before, Data),
        player_b_elo_before = maps:get(player_b_elo_before, Data),
        player_a_elo_after = maps:get(player_a_elo_after, Data),
        player_b_elo_after = maps:get(player_b_elo_after, Data),
        match_duration_ms = maps:get(match_duration_ms, Data, 0)
    }.

-spec elo_updated(map()) -> #elo_updated{}.
elo_updated(Data) ->
    #elo_updated{
        meta = make_meta(maps:get(emitter, Data, competitive_silo), Data),
        individual_id = maps:get(individual_id, Data),
        elo_before = maps:get(elo_before, Data),
        elo_after = maps:get(elo_after, Data),
        k_factor = maps:get(k_factor, Data, 32.0),
        match_result = maps:get(match_result, Data)
    }.

%%% ============================================================================
%%% Strategy Event Constructors
%%% ============================================================================

-spec strategy_countered(map()) -> #strategy_countered{}.
strategy_countered(Data) ->
    #strategy_countered{
        meta = make_meta(maps:get(emitter, Data, competitive_silo), Data),
        counter_id = maps:get(counter_id, Data),
        countered_id = maps:get(countered_id, Data),
        win_rate_against = maps:get(win_rate_against, Data),
        strategy_signature = maps:get(strategy_signature, Data, <<>>),
        generation = maps:get(generation, Data)
    }.

-spec arms_race_detected(map()) -> #arms_race_detected{}.
arms_race_detected(Data) ->
    #arms_race_detected{
        meta = make_meta(maps:get(emitter, Data, competitive_silo), Data),
        population_id = maps:get(population_id, Data),
        cycle_length = maps:get(cycle_length, Data),
        participants = maps:get(participants, Data, []),
        intensity = maps:get(intensity, Data),
        generation = maps:get(generation, Data)
    }.

%%% ============================================================================
%%% Matchmaking Event Constructors
%%% ============================================================================

-spec matchmaking_adjusted(map()) -> #matchmaking_adjusted{}.
matchmaking_adjusted(Data) ->
    #matchmaking_adjusted{
        meta = make_meta(maps:get(emitter, Data, competitive_silo), Data),
        population_id = maps:get(population_id, Data),
        elo_range_before = maps:get(elo_range_before, Data),
        elo_range_after = maps:get(elo_range_after, Data),
        self_play_ratio_before = maps:get(self_play_ratio_before, Data),
        self_play_ratio_after = maps:get(self_play_ratio_after, Data),
        adjustment_reason = maps:get(adjustment_reason, Data, diversity)
    }.

-spec dominance_matrix_updated(map()) -> #dominance_matrix_updated{}.
dominance_matrix_updated(Data) ->
    #dominance_matrix_updated{
        meta = make_meta(maps:get(emitter, Data, competitive_silo), Data),
        population_id = maps:get(population_id, Data),
        matrix_size = maps:get(matrix_size, Data),
        strongest_counter = maps:get(strongest_counter, Data),
        cycle_detected = maps:get(cycle_detected, Data, false),
        update_generation = maps:get(update_generation, Data)
    }.

%%% ============================================================================
%%% Utility Functions
%%% ============================================================================

-spec event_to_map(competitive_event()) -> map().
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

record_fields(opponent_archived) -> [meta, opponent_id, archive_id, fitness_at_archive, elo_at_archive, generation, archive_size_after];
record_fields(opponent_retired) -> [meta, opponent_id, archive_id, retirement_reason, games_played, final_elo];
record_fields(match_completed) -> [meta, match_id, player_a_id, player_b_id, winner_id, player_a_elo_before, player_b_elo_before, player_a_elo_after, player_b_elo_after, match_duration_ms];
record_fields(elo_updated) -> [meta, individual_id, elo_before, elo_after, k_factor, match_result];
record_fields(strategy_countered) -> [meta, counter_id, countered_id, win_rate_against, strategy_signature, generation];
record_fields(arms_race_detected) -> [meta, population_id, cycle_length, participants, intensity, generation];
record_fields(matchmaking_adjusted) -> [meta, population_id, elo_range_before, elo_range_after, self_play_ratio_before, self_play_ratio_after, adjustment_reason];
record_fields(dominance_matrix_updated) -> [meta, population_id, matrix_size, strongest_counter, cycle_detected, update_generation].
