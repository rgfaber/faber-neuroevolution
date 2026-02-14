%%%-------------------------------------------------------------------
%%% @doc Behavioral event constructors for Cultural Silo.
%%%
%%% Events related to behavioral innovations, traditions, and memes.
%%% Cultural Silo operates at timescale τ=35.
%%%
%%% @end
%%%-------------------------------------------------------------------
-module(lc_cultural_events).

-include("lc_cultural_events.hrl").

%% Event constructors
-export([
    innovation_discovered/1,
    innovation_spread/1,
    tradition_established/1,
    tradition_abandoned/1,
    meme_created/1,
    meme_spread/1,
    meme_mutated/1,
    cultural_convergence/1,
    cultural_divergence/1
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
%%% Innovation Event Constructors
%%% ============================================================================

-spec innovation_discovered(map()) -> #innovation_discovered{}.
innovation_discovered(Data) ->
    #innovation_discovered{
        meta = make_meta(maps:get(emitter, Data, cultural_silo), Data),
        innovator_id = maps:get(innovator_id, Data),
        innovation_id = maps:get(innovation_id, Data, generate_id()),
        population_id = maps:get(population_id, Data),
        novelty_score = maps:get(novelty_score, Data),
        fitness_advantage = maps:get(fitness_advantage, Data),
        innovation_type = maps:get(innovation_type, Data, behavioral)
    }.

-spec innovation_spread(map()) -> #innovation_spread{}.
innovation_spread(Data) ->
    #innovation_spread{
        meta = make_meta(maps:get(emitter, Data, cultural_silo), Data),
        innovation_id = maps:get(innovation_id, Data),
        source_id = maps:get(source_id, Data),
        adopter_id = maps:get(adopter_id, Data),
        spread_fidelity = maps:get(spread_fidelity, Data, 1.0),
        total_adopters = maps:get(total_adopters, Data)
    }.

%%% ============================================================================
%%% Tradition Event Constructors
%%% ============================================================================

-spec tradition_established(map()) -> #tradition_established{}.
tradition_established(Data) ->
    #tradition_established{
        meta = make_meta(maps:get(emitter, Data, cultural_silo), Data),
        tradition_id = maps:get(tradition_id, Data, generate_id()),
        population_id = maps:get(population_id, Data),
        behavior_signature = maps:get(behavior_signature, Data, <<>>),
        initial_practitioners = maps:get(initial_practitioners, Data),
        establishment_threshold = maps:get(establishment_threshold, Data)
    }.

-spec tradition_abandoned(map()) -> #tradition_abandoned{}.
tradition_abandoned(Data) ->
    #tradition_abandoned{
        meta = make_meta(maps:get(emitter, Data, cultural_silo), Data),
        tradition_id = maps:get(tradition_id, Data),
        population_id = maps:get(population_id, Data),
        abandonment_reason = maps:get(abandonment_reason, Data, obsolescence),
        final_practitioners = maps:get(final_practitioners, Data, 0),
        duration_generations = maps:get(duration_generations, Data, 0)
    }.

%%% ============================================================================
%%% Meme Event Constructors
%%% ============================================================================

-spec meme_created(map()) -> #meme_created{}.
meme_created(Data) ->
    #meme_created{
        meta = make_meta(maps:get(emitter, Data, cultural_silo), Data),
        meme_id = maps:get(meme_id, Data, generate_id()),
        creator_id = maps:get(creator_id, Data),
        population_id = maps:get(population_id, Data),
        meme_encoding = maps:get(meme_encoding, Data),
        initial_fitness_correlation = maps:get(initial_fitness_correlation, Data, 0.0)
    }.

-spec meme_spread(map()) -> #meme_spread{}.
meme_spread(Data) ->
    #meme_spread{
        meta = make_meta(maps:get(emitter, Data, cultural_silo), Data),
        meme_id = maps:get(meme_id, Data),
        source_id = maps:get(source_id, Data),
        target_id = maps:get(target_id, Data),
        spread_fidelity = maps:get(spread_fidelity, Data, 1.0),
        fitness_correlation = maps:get(fitness_correlation, Data, 0.0),
        total_adopters = maps:get(total_adopters, Data)
    }.

-spec meme_mutated(map()) -> #meme_mutated{}.
meme_mutated(Data) ->
    #meme_mutated{
        meta = make_meta(maps:get(emitter, Data, cultural_silo), Data),
        original_meme_id = maps:get(original_meme_id, Data),
        mutated_meme_id = maps:get(mutated_meme_id, Data, generate_id()),
        mutator_id = maps:get(mutator_id, Data),
        mutation_magnitude = maps:get(mutation_magnitude, Data),
        fitness_change = maps:get(fitness_change, Data, 0.0)
    }.

%%% ============================================================================
%%% Cultural Dynamics Event Constructors
%%% ============================================================================

-spec cultural_convergence(map()) -> #cultural_convergence{}.
cultural_convergence(Data) ->
    #cultural_convergence{
        meta = make_meta(maps:get(emitter, Data, cultural_silo), Data),
        population_id = maps:get(population_id, Data),
        behavioral_variance_before = maps:get(behavioral_variance_before, Data),
        behavioral_variance_after = maps:get(behavioral_variance_after, Data),
        convergence_driver = maps:get(convergence_driver, Data, selection)
    }.

-spec cultural_divergence(map()) -> #cultural_divergence{}.
cultural_divergence(Data) ->
    #cultural_divergence{
        meta = make_meta(maps:get(emitter, Data, cultural_silo), Data),
        population_id = maps:get(population_id, Data),
        behavioral_variance_before = maps:get(behavioral_variance_before, Data),
        behavioral_variance_after = maps:get(behavioral_variance_after, Data),
        divergence_driver = maps:get(divergence_driver, Data, exploration)
    }.

%%% ============================================================================
%%% Utility Functions
%%% ============================================================================

-spec event_to_map(cultural_event()) -> map().
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

record_fields(innovation_discovered) -> [meta, innovator_id, innovation_id, population_id, novelty_score, fitness_advantage, innovation_type];
record_fields(innovation_spread) -> [meta, innovation_id, source_id, adopter_id, spread_fidelity, total_adopters];
record_fields(tradition_established) -> [meta, tradition_id, population_id, behavior_signature, initial_practitioners, establishment_threshold];
record_fields(tradition_abandoned) -> [meta, tradition_id, population_id, abandonment_reason, final_practitioners, duration_generations];
record_fields(meme_created) -> [meta, meme_id, creator_id, population_id, meme_encoding, initial_fitness_correlation];
record_fields(meme_spread) -> [meta, meme_id, source_id, target_id, spread_fidelity, fitness_correlation, total_adopters];
record_fields(meme_mutated) -> [meta, original_meme_id, mutated_meme_id, mutator_id, mutation_magnitude, fitness_change];
record_fields(cultural_convergence) -> [meta, population_id, behavioral_variance_before, behavioral_variance_after, convergence_driver];
record_fields(cultural_divergence) -> [meta, population_id, behavioral_variance_before, behavioral_variance_after, divergence_driver].
