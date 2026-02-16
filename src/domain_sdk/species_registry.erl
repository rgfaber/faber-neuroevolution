%% @doc Species Registry - Manages multiple species for coevolution.
%%
%% The registry maintains a collection of species that coevolve in a
%% shared environment. Each species has its own population, topology,
%% and fitness function.
%%
%% == Overview ==
%%
%% ```
%% ┌─────────────────────────────────────────────────┐
%% │              Species Registry                    │
%% ├─────────────────────────────────────────────────┤
%% │  ┌─────────┐  ┌─────────┐  ┌─────────┐         │
%% │  │ Forager │  │ Predator│  │Scavenger│         │
%% │  │ Pop:100 │  │ Pop:30  │  │ Pop:20  │         │
%% │  └────┬────┘  └────┬────┘  └────┬────┘         │
%% │       │            │            │               │
%% │  ┌────┴────────────┴────────────┴────┐         │
%% │  │        Shared Environment         │         │
%% │  └───────────────────────────────────┘         │
%% └─────────────────────────────────────────────────┘
%% '''
%%
%% == Usage ==
%%
%% ```
%% %% Create registry with multiple species
%% {ok, Registry} = species_registry:new(#{
%%     species => [forager_species, predator_species],
%%     population_sizes => #{forager => 100, predator => 30},
%%     environment => hex_arena_env
%% }).
%%
%% %% Get all species
%% Species = species_registry:list_species(Registry).
%%
%% %% Create bridge for a specific species
%% {ok, Bridge} = species_registry:create_bridge(Registry, forager).
%% '''
%%
%% @author R.G. Lefever
%% @copyright 2024-2026 R.G. Lefever
%% @see agent_species
-module(species_registry).

%%% ============================================================================
%%% Types
%%% ============================================================================

-type registry() :: #{
    species := #{agent_species:species_id() => species_entry()},
    environment := module(),
    population_sizes := #{agent_species:species_id() => pos_integer()},
    subspecies := #{agent_species:subspecies_id() => [term()]}  %% Agent IDs
}.

-type species_entry() :: #{
    module := module(),
    config := agent_species:species_config(),
    bridge := agent_bridge:validated_bridge() | undefined,
    population := [term()],  %% Network IDs
    subspecies_map := #{non_neg_integer() => [term()]}  %% Behavioral clusters
}.

-export_type([registry/0, species_entry/0]).

%%% ============================================================================
%%% API Functions
%%% ============================================================================

-export([
    new/1,
    register_species/2,
    unregister_species/2,
    list_species/1,
    get_species/2,
    get_config/2,
    get_bridge/2,
    create_bridge/2,
    get_population_size/2,
    set_population/3,
    get_population/2,
    get_all_populations/1,
    update_subspecies/3,
    get_subspecies/2,
    total_population/1
]).

%% @doc Creates a new species registry.
%%
%% Options:
%% <ul>
%%   <li>`species' - List of species modules (required)</li>
%%   <li>`environment' - Shared environment module (required)</li>
%%   <li>`population_sizes' - Map of species_id => size (optional)</li>
%% </ul>
-spec new(Options) -> {ok, Registry} | {error, Reason} when
    Options :: #{
        species := [module()],
        environment := module(),
        population_sizes => #{atom() => pos_integer()}
    },
    Registry :: registry(),
    Reason :: term().
new(Options) ->
    try
        SpeciesModules = maps:get(species, Options),
        Environment = maps:get(environment, Options),
        PopSizes = maps:get(population_sizes, Options, #{}),

        %% Validate environment (use multispecies_environment for multi-species support)
        ok = multispecies_environment:validate(Environment),

        %% Register each species
        SpeciesMap = lists:foldl(
            fun(Module, Acc) ->
                case validate_and_register(Module, Environment, PopSizes) of
                    {ok, Id, Entry} -> maps:put(Id, Entry, Acc);
                    {error, Reason} -> throw({species_error, Module, Reason})
                end
            end,
            #{},
            SpeciesModules
        ),

        Registry = #{
            species => SpeciesMap,
            environment => Environment,
            population_sizes => PopSizes,
            subspecies => #{}
        },

        {ok, Registry}
    catch
        throw:Error -> {error, Error};
        _:Reason -> {error, {registry_creation_failed, Reason}}
    end.

%% @doc Registers a new species in the registry.
-spec register_species(Registry, Module) -> {ok, NewRegistry} | {error, Reason} when
    Registry :: registry(),
    Module :: module(),
    NewRegistry :: registry(),
    Reason :: term().
register_species(Registry, Module) ->
    Environment = maps:get(environment, Registry),
    PopSizes = maps:get(population_sizes, Registry),
    case validate_and_register(Module, Environment, PopSizes) of
        {ok, Id, Entry} ->
            SpeciesMap = maps:get(species, Registry),
            NewSpeciesMap = maps:put(Id, Entry, SpeciesMap),
            {ok, Registry#{species := NewSpeciesMap}};
        {error, Reason} ->
            {error, Reason}
    end.

%% @doc Unregisters a species from the registry.
-spec unregister_species(Registry, SpeciesId) -> {ok, NewRegistry} | {error, not_found} when
    Registry :: registry(),
    SpeciesId :: agent_species:species_id(),
    NewRegistry :: registry().
unregister_species(Registry, SpeciesId) ->
    SpeciesMap = maps:get(species, Registry),
    case maps:is_key(SpeciesId, SpeciesMap) of
        true ->
            NewSpeciesMap = maps:remove(SpeciesId, SpeciesMap),
            {ok, Registry#{species := NewSpeciesMap}};
        false ->
            {error, not_found}
    end.

%% @doc Lists all registered species IDs.
-spec list_species(Registry) -> [agent_species:species_id()] when
    Registry :: registry().
list_species(Registry) ->
    maps:keys(maps:get(species, Registry)).

%% @doc Gets a species entry by ID.
-spec get_species(Registry, SpeciesId) -> {ok, Entry} | {error, not_found} when
    Registry :: registry(),
    SpeciesId :: agent_species:species_id(),
    Entry :: species_entry().
get_species(Registry, SpeciesId) ->
    SpeciesMap = maps:get(species, Registry),
    case maps:get(SpeciesId, SpeciesMap, undefined) of
        undefined -> {error, not_found};
        Entry -> {ok, Entry}
    end.

%% @doc Gets species configuration.
-spec get_config(Registry, SpeciesId) -> {ok, Config} | {error, not_found} when
    Registry :: registry(),
    SpeciesId :: agent_species:species_id(),
    Config :: agent_species:species_config().
get_config(Registry, SpeciesId) ->
    case get_species(Registry, SpeciesId) of
        {ok, Entry} -> {ok, maps:get(config, Entry)};
        Error -> Error
    end.

%% @doc Gets or creates a bridge for a species.
-spec get_bridge(Registry, SpeciesId) -> {ok, Bridge} | {error, Reason} when
    Registry :: registry(),
    SpeciesId :: agent_species:species_id(),
    Bridge :: agent_bridge:validated_bridge(),
    Reason :: term().
get_bridge(Registry, SpeciesId) ->
    case get_species(Registry, SpeciesId) of
        {ok, Entry} ->
            case maps:get(bridge, Entry, undefined) of
                undefined -> create_bridge(Registry, SpeciesId);
                Bridge -> {ok, Bridge}
            end;
        Error -> Error
    end.

%% @doc Creates a new bridge for a species.
-spec create_bridge(Registry, SpeciesId) -> {ok, Bridge} | {error, Reason} when
    Registry :: registry(),
    SpeciesId :: agent_species:species_id(),
    Bridge :: agent_bridge:validated_bridge(),
    Reason :: term().
create_bridge(Registry, SpeciesId) ->
    case get_species(Registry, SpeciesId) of
        {ok, Entry} ->
            Module = maps:get(module, Entry),
            Environment = maps:get(environment, Registry),
            BridgeConfig = agent_species:to_bridge_config(Module, Environment),
            agent_bridge:new(BridgeConfig);
        Error -> Error
    end.

%% @doc Gets population size for a species.
-spec get_population_size(Registry, SpeciesId) -> pos_integer() when
    Registry :: registry(),
    SpeciesId :: agent_species:species_id().
get_population_size(Registry, SpeciesId) ->
    PopSizes = maps:get(population_sizes, Registry),
    maps:get(SpeciesId, PopSizes, 50).  %% Default 50

%% @doc Sets the population (network IDs) for a species.
-spec set_population(Registry, SpeciesId, Population) -> registry() when
    Registry :: registry(),
    SpeciesId :: agent_species:species_id(),
    Population :: [term()].
set_population(Registry, SpeciesId, Population) ->
    SpeciesMap = maps:get(species, Registry),
    case maps:get(SpeciesId, SpeciesMap, undefined) of
        undefined -> Registry;
        Entry ->
            NewEntry = Entry#{population := Population},
            NewSpeciesMap = maps:put(SpeciesId, NewEntry, SpeciesMap),
            Registry#{species := NewSpeciesMap}
    end.

%% @doc Gets the population for a species.
-spec get_population(Registry, SpeciesId) -> [term()] when
    Registry :: registry(),
    SpeciesId :: agent_species:species_id().
get_population(Registry, SpeciesId) ->
    case get_species(Registry, SpeciesId) of
        {ok, Entry} -> maps:get(population, Entry, []);
        _ -> []
    end.

%% @doc Gets all populations keyed by species ID.
-spec get_all_populations(Registry) -> #{agent_species:species_id() => [term()]} when
    Registry :: registry().
get_all_populations(Registry) ->
    SpeciesMap = maps:get(species, Registry),
    maps:map(fun(_Id, Entry) -> maps:get(population, Entry, []) end, SpeciesMap).

%% @doc Updates subspecies assignments for a species.
-spec update_subspecies(Registry, SpeciesId, SubspeciesMap) -> registry() when
    Registry :: registry(),
    SpeciesId :: agent_species:species_id(),
    SubspeciesMap :: #{non_neg_integer() => [term()]}.
update_subspecies(Registry, SpeciesId, SubspeciesMap) ->
    SpeciesMapOuter = maps:get(species, Registry),
    case maps:get(SpeciesId, SpeciesMapOuter, undefined) of
        undefined -> Registry;
        Entry ->
            NewEntry = Entry#{subspecies_map := SubspeciesMap},
            NewSpeciesMap = maps:put(SpeciesId, NewEntry, SpeciesMapOuter),
            Registry#{species := NewSpeciesMap}
    end.

%% @doc Gets subspecies assignments for a species.
-spec get_subspecies(Registry, SpeciesId) -> #{non_neg_integer() => [term()]} when
    Registry :: registry(),
    SpeciesId :: agent_species:species_id().
get_subspecies(Registry, SpeciesId) ->
    case get_species(Registry, SpeciesId) of
        {ok, Entry} -> maps:get(subspecies_map, Entry, #{});
        _ -> #{}
    end.

%% @doc Returns total population across all species.
-spec total_population(Registry) -> non_neg_integer() when
    Registry :: registry().
total_population(Registry) ->
    Populations = get_all_populations(Registry),
    lists:sum([length(P) || P <- maps:values(Populations)]).

%%% ============================================================================
%%% Internal Functions
%%% ============================================================================

%% @private
validate_and_register(Module, _Environment, _PopSizes) ->
    case agent_species:validate(Module) of
        ok ->
            Config = agent_species:to_config(Module),
            Name = maps:get(name, Config),
            Id = binary_to_atom(Name, utf8),

            %% Create bridge config (don't validate yet - lazy)
            Entry = #{
                module => Module,
                config => Config,
                bridge => undefined,
                population => [],
                subspecies_map => #{}
            },

            {ok, Id, Entry};
        {error, Reasons} ->
            {error, {invalid_species, Module, Reasons}}
    end.
