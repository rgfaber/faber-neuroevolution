%% @doc Configuration builder for neuroevolution server.
%%
%% This module provides helper functions to construct #neuro_config{} records
%% from maps, enabling clean integration with Elixir applications without
%% requiring manual tuple construction.
%%
%% == Usage ==
%%
%% From Elixir:
%% config = :neuro_config.from_map(%{
%%   population_size: 50,
%%   network_topology: {42, [24], 6},
%%   evaluator_module: :my_evaluator
%% })
%% {:ok, pid} = :neuroevolution_server.start_link(config)
%%
%% @author R.G. Lefever
%% @copyright 2024-2026 R.G. Lefever
-module(neuro_config).

-include("neuroevolution.hrl").
-include("evolution_strategy.hrl").

-export([
    from_map/1,
    to_map/1,
    default/0,
    default/1,
    with_l0_params/1,
    with_l0_params/2
]).

%%% ============================================================================
%%% API Functions
%%% ============================================================================

%% @doc Build a #neuro_config{} record from a map.
%%
%% All fields are optional - missing fields use sensible defaults.
%% This function handles type coercion and validation.
%%
%% Required fields (no reasonable defaults):
%% - `network_topology' - {Inputs, HiddenLayers, Outputs}
%% - `evaluator_module' - Module implementing neuroevolution_evaluator behaviour
%%
%% Throws `{missing_required_field, FieldName}' if required field is missing.
-spec from_map(map()) -> #neuro_config{}.
from_map(Map) when is_map(Map) ->
    %% Validate required fields
    NetworkTopology = get_required(network_topology, Map),
    EvaluatorModule = get_required(evaluator_module, Map),

    #neuro_config{
        %% Population and evaluation settings
        population_size = maps:get(population_size, Map, 50),
        evaluations_per_individual = maps:get(evaluations_per_individual, Map, 10),
        selection_ratio = maps:get(selection_ratio, Map, 0.20),

        %% Mutation parameters (legacy weight-only mutation)
        mutation_rate = maps:get(mutation_rate, Map, 0.10),
        mutation_strength = maps:get(mutation_strength, Map, 0.3),

        %% Layer-specific mutation rates (optional)
        reservoir_mutation_rate = normalize_nil(maps:get(reservoir_mutation_rate, Map, undefined)),
        reservoir_mutation_strength = normalize_nil(maps:get(reservoir_mutation_strength, Map, undefined)),
        readout_mutation_rate = normalize_nil(maps:get(readout_mutation_rate, Map, undefined)),
        readout_mutation_strength = normalize_nil(maps:get(readout_mutation_strength, Map, undefined)),

        %% NEAT topology mutation config (optional)
        topology_mutation_config = get_mutation_config(Map),

        %% Stopping criteria
        max_generations = maps:get(max_generations, Map, infinity),
        target_fitness = normalize_nil(maps:get(target_fitness, Map, undefined)),

        %% Network architecture
        network_topology = NetworkTopology,

        %% Evaluator configuration
        evaluator_module = EvaluatorModule,
        evaluator_options = maps:get(evaluator_options, Map, #{}),

        %% Event handling
        event_handler = get_event_handler(Map),

        %% Meta-controller configuration
        meta_controller_config = get_meta_config(Map),

        %% Speciation configuration
        speciation_config = get_speciation_config(Map),

        %% Event publishing
        realm = maps:get(realm, Map, <<"default">>),
        publish_events = maps:get(publish_events, Map, false),

        %% Evaluation mode
        evaluation_mode = maps:get(evaluation_mode, Map, direct),
        evaluation_timeout = maps:get(evaluation_timeout, Map, 30000),
        max_concurrent_evaluations = normalize_nil(maps:get(max_concurrent_evaluations, Map, undefined)),

        %% Evolution strategy
        strategy_config = get_strategy_config(Map)
    }.

%% @doc Convert a #neuro_config{} record to a map.
%%
%% Useful for serialization, logging, and passing to Elixir code.
-spec to_map(#neuro_config{}) -> map().
to_map(Config) when is_record(Config, neuro_config) ->
    #{
        population_size => Config#neuro_config.population_size,
        evaluations_per_individual => Config#neuro_config.evaluations_per_individual,
        selection_ratio => Config#neuro_config.selection_ratio,
        mutation_rate => Config#neuro_config.mutation_rate,
        mutation_strength => Config#neuro_config.mutation_strength,
        reservoir_mutation_rate => Config#neuro_config.reservoir_mutation_rate,
        reservoir_mutation_strength => Config#neuro_config.reservoir_mutation_strength,
        readout_mutation_rate => Config#neuro_config.readout_mutation_rate,
        readout_mutation_strength => Config#neuro_config.readout_mutation_strength,
        topology_mutation_config => mutation_config_to_map(Config#neuro_config.topology_mutation_config),
        max_generations => Config#neuro_config.max_generations,
        target_fitness => Config#neuro_config.target_fitness,
        network_topology => Config#neuro_config.network_topology,
        evaluator_module => Config#neuro_config.evaluator_module,
        evaluator_options => Config#neuro_config.evaluator_options,
        event_handler => Config#neuro_config.event_handler,
        meta_controller_config => meta_config_to_map(Config#neuro_config.meta_controller_config),
        speciation_config => speciation_config_to_map(Config#neuro_config.speciation_config),
        realm => Config#neuro_config.realm,
        publish_events => Config#neuro_config.publish_events,
        evaluation_mode => Config#neuro_config.evaluation_mode,
        evaluation_timeout => Config#neuro_config.evaluation_timeout,
        max_concurrent_evaluations => Config#neuro_config.max_concurrent_evaluations,
        strategy_config => Config#neuro_config.strategy_config
    }.

%% @doc Create a default configuration.
%%
%% Note: This creates a config with placeholder topology and evaluator.
%% In practice, you should use from_map/1 with your actual values.
-spec default() -> #neuro_config{}.
default() ->
    default(#{
        network_topology => {4, [8], 2},
        evaluator_module => undefined
    }).

%% @doc Create a configuration with the given overrides.
-spec default(map()) -> #neuro_config{}.
default(Overrides) ->
    from_map(Overrides).

%%% ============================================================================
%%% Internal Functions
%%% ============================================================================

%% @private Get required field or throw error.
get_required(Field, Map) ->
    case maps:get(Field, Map, undefined) of
        undefined ->
            error({missing_required_field, Field});
        Value ->
            Value
    end.

%% @private Extract event handler from map.
get_event_handler(Map) ->
    case maps:get(event_handler, Map, undefined) of
        undefined -> undefined;
        nil -> undefined;  %% Elixir nil
        {Module, Arg} when is_atom(Module) -> {Module, Arg};
        Module when is_atom(Module) -> {Module, undefined};
        _ -> undefined
    end.

%% @private Extract meta-controller config from map.
%% If it's already a record (passed through), use it directly.
%% If it's a map, convert using meta_config:from_map/1.
get_meta_config(Map) ->
    case maps:get(meta_controller_config, Map, undefined) of
        undefined -> undefined;
        nil -> undefined;  %% Elixir nil
        Config when is_tuple(Config), element(1, Config) =:= meta_config -> Config;
        ConfigMap when is_map(ConfigMap) -> meta_config:from_map(ConfigMap);
        _ -> undefined
    end.

%% @private Extract mutation config from map.
get_mutation_config(Map) ->
    case maps:get(topology_mutation_config, Map, undefined) of
        undefined -> undefined;
        nil -> undefined;  %% Elixir nil
        Config when is_tuple(Config), element(1, Config) =:= mutation_config -> Config;
        ConfigMap when is_map(ConfigMap) -> mutation_config_from_map(ConfigMap);
        _ -> undefined
    end.

%% @private Extract speciation config from map.
get_speciation_config(Map) ->
    case maps:get(speciation_config, Map, undefined) of
        undefined -> undefined;
        nil -> undefined;  %% Elixir nil
        Config when is_tuple(Config), element(1, Config) =:= speciation_config -> Config;
        ConfigMap when is_map(ConfigMap) -> speciation_config_from_map(ConfigMap);
        _ -> undefined
    end.

%% @private Extract strategy config from map.
%% Handles Elixir's nil -> Erlang undefined conversion.
get_strategy_config(Map) ->
    case maps:get(strategy_config, Map, undefined) of
        undefined -> undefined;
        nil -> undefined;  %% Elixir nil
        Config when is_tuple(Config), element(1, Config) =:= strategy_config -> Config;
        ConfigMap when is_map(ConfigMap) -> strategy_config_from_map(ConfigMap);
        _ -> undefined
    end.

%% @private Convert strategy_config map to record.
strategy_config_from_map(Map) ->
    #strategy_config{
        strategy_module = maps:get(strategy_module, Map, generational_strategy),
        strategy_params = maps:get(strategy_params, Map, #{}),
        min_population = maps:get(min_population, Map, 10),
        max_population = maps:get(max_population, Map, 1000),
        initial_population = maps:get(initial_population, Map, 50),
        meta_controller_module = normalize_nil(maps:get(meta_controller_module, Map, undefined)),
        meta_controller_config = maps:get(meta_controller_config, Map, #{})
    }.

%% @private Convert Elixir nil to Erlang undefined.
normalize_nil(nil) -> undefined;
normalize_nil(Value) -> Value.

%% @private Convert mutation_config map to record.
mutation_config_from_map(Map) ->
    #mutation_config{
        weight_mutation_rate = maps:get(weight_mutation_rate, Map, 0.80),
        weight_perturb_rate = maps:get(weight_perturb_rate, Map, 0.90),
        weight_perturb_strength = maps:get(weight_perturb_strength, Map, 0.3),
        add_node_rate = maps:get(add_node_rate, Map, 0.03),
        add_connection_rate = maps:get(add_connection_rate, Map, 0.05),
        toggle_connection_rate = maps:get(toggle_connection_rate, Map, 0.01),
        add_sensor_rate = maps:get(add_sensor_rate, Map, 0.001),
        add_actuator_rate = maps:get(add_actuator_rate, Map, 0.001),
        mutate_neuron_type_rate = maps:get(mutate_neuron_type_rate, Map, 0.01),
        mutate_time_constant_rate = maps:get(mutate_time_constant_rate, Map, 0.05)
    }.

%% @private Convert speciation_config map to record.
speciation_config_from_map(Map) ->
    #speciation_config{
        enabled = maps:get(enabled, Map, false),
        compatibility_threshold = maps:get(compatibility_threshold, Map, 3.0),
        c1_excess = maps:get(c1_excess, Map, 1.0),
        c2_disjoint = maps:get(c2_disjoint, Map, 1.0),
        c3_weight_diff = maps:get(c3_weight_diff, Map, 0.4),
        target_species = maps:get(target_species, Map, 5),
        threshold_adjustment_rate = maps:get(threshold_adjustment_rate, Map, 0.1),
        min_species_size = maps:get(min_species_size, Map, 2),
        max_stagnation = maps:get(max_stagnation, Map, 15),
        species_elitism = maps:get(species_elitism, Map, 0.20),
        interspecies_mating_rate = maps:get(interspecies_mating_rate, Map, 0.001)
    }.

%% @private Convert mutation_config record to map.
mutation_config_to_map(undefined) -> undefined;
mutation_config_to_map(Config) when is_record(Config, mutation_config) ->
    #{
        weight_mutation_rate => Config#mutation_config.weight_mutation_rate,
        weight_perturb_rate => Config#mutation_config.weight_perturb_rate,
        weight_perturb_strength => Config#mutation_config.weight_perturb_strength,
        add_node_rate => Config#mutation_config.add_node_rate,
        add_connection_rate => Config#mutation_config.add_connection_rate,
        toggle_connection_rate => Config#mutation_config.toggle_connection_rate,
        add_sensor_rate => Config#mutation_config.add_sensor_rate,
        add_actuator_rate => Config#mutation_config.add_actuator_rate,
        mutate_neuron_type_rate => Config#mutation_config.mutate_neuron_type_rate,
        mutate_time_constant_rate => Config#mutation_config.mutate_time_constant_rate
    };
mutation_config_to_map(_) -> undefined.

%% @private Convert speciation_config record to map.
speciation_config_to_map(undefined) -> undefined;
speciation_config_to_map(Config) when is_record(Config, speciation_config) ->
    #{
        enabled => Config#speciation_config.enabled,
        compatibility_threshold => Config#speciation_config.compatibility_threshold,
        c1_excess => Config#speciation_config.c1_excess,
        c2_disjoint => Config#speciation_config.c2_disjoint,
        c3_weight_diff => Config#speciation_config.c3_weight_diff,
        target_species => Config#speciation_config.target_species,
        threshold_adjustment_rate => Config#speciation_config.threshold_adjustment_rate,
        min_species_size => Config#speciation_config.min_species_size,
        max_stagnation => Config#speciation_config.max_stagnation,
        species_elitism => Config#speciation_config.species_elitism,
        interspecies_mating_rate => Config#speciation_config.interspecies_mating_rate
    };
speciation_config_to_map(_) -> undefined.

%% @private Convert meta_config record to map (delegated to meta_config module).
meta_config_to_map(undefined) -> undefined;
meta_config_to_map(Config) when is_tuple(Config), element(1, Config) =:= meta_config ->
    meta_config:to_map(Config);
meta_config_to_map(_) -> undefined.

%%% ============================================================================
%%% L0 Integration Functions
%%% ============================================================================

%% @doc Merge L0 actuator values into config.
%%
%% When Liquid Conglomerate (LC) is enabled, this function gets the current
%% hyperparameter values from the L0 controller and updates the config.
%% This enables dynamic adaptation of mutation rates during training.
%%
%% If task_l0_actuators is not running, returns the config unchanged.
%%
%% Example:
%% Config = neuro_config:from_map(#{...}),
%% DynamicConfig = neuro_config:with_l0_params(Config),
%% %% DynamicConfig now has L0-controlled mutation rates
-spec with_l0_params(#neuro_config{}) -> #neuro_config{}.
with_l0_params(Config) ->
    case whereis(task_l0_actuators) of
        undefined ->
            %% L0 not running, return config unchanged
            Config;
        _Pid ->
            L0Params = task_l0_actuators:get_evolution_params(),
            merge_l0_params(Config, L0Params)
    end.

%% @doc Merge specific L0 params into config.
%%
%% This variant takes the L0 params directly, useful when you already have them.
-spec with_l0_params(#neuro_config{}, map()) -> #neuro_config{}.
with_l0_params(Config, L0Params) ->
    merge_l0_params(Config, L0Params).

%% @private Merge L0 params map into config record.
merge_l0_params(Config, L0Params) ->
    Config#neuro_config{
        %% Fallback mutation rates (from L0)
        mutation_rate = maps:get(mutation_rate, L0Params, Config#neuro_config.mutation_rate),
        mutation_strength = maps:get(mutation_strength, L0Params, Config#neuro_config.mutation_strength),

        %% Layer-specific rates (from L0)
        reservoir_mutation_rate = maps:get(reservoir_mutation_rate, L0Params,
                                           Config#neuro_config.reservoir_mutation_rate),
        reservoir_mutation_strength = maps:get(reservoir_mutation_strength, L0Params,
                                               Config#neuro_config.reservoir_mutation_strength),
        readout_mutation_rate = maps:get(readout_mutation_rate, L0Params,
                                         Config#neuro_config.readout_mutation_rate),
        readout_mutation_strength = maps:get(readout_mutation_strength, L0Params,
                                             Config#neuro_config.readout_mutation_strength),

        %% Selection ratio (from L0)
        selection_ratio = maps:get(selection_ratio, L0Params, Config#neuro_config.selection_ratio)
    }.
