%% @doc Task Silo L0: Safe Default Parameters.
%%
%% Part of the Liquid Conglomerate v2 Task Silo. Provides safe, conservative
%% default parameters that are always active and never disabled.
%%
%% == Purpose ==
%%
%% L0 serves as the fallback baseline when L1/L2 are disabled or fail.
%% These defaults are designed to be safe across a wide range of problems:
%% - Moderate mutation rates (not too aggressive, not too conservative)
%% - Balanced selection pressure
%% - Conservative topology mutation (when enabled)
%%
%% == Usage ==
%%
%% %% Get default parameters
%% Defaults = task_l0_defaults:get_defaults(),
%% MutationRate = maps:get(mutation_rate, Defaults).
%%
%% %% Clamp a value to safe bounds
%% SafeRate = task_l0_defaults:clamp(mutation_rate, 0.8).  % Returns 0.5 (max)
%%
%% %% Apply bounds to a full parameter map
%% SafeParams = task_l0_defaults:apply_bounds(UnsafeParams).
%%
%% @author R.G. Lefever
%% @copyright 2024-2026 R.G. Lefever
-module(task_l0_defaults).

-export([
    get_defaults/0,
    get_bounds/0,
    get_bound/1,
    clamp/2,
    apply_bounds/1,
    merge_with_defaults/1
]).

%%% ============================================================================
%%% API Functions
%%% ============================================================================

%% @doc Get default parameter values.
%%
%% These are safe starting values for neuroevolution that work across
%% a wide range of problem domains.
-spec get_defaults() -> map().
get_defaults() ->
    #{
        %% Core evolution parameters
        mutation_rate => 0.10,           % 10% of genes mutated
        mutation_strength => 0.30,       % Moderate perturbation
        selection_ratio => 0.20,         % Top 20% survive

        %% NEAT topology mutation (conservative)
        add_node_rate => 0.03,           % 3% chance to add node
        add_connection_rate => 0.05,     % 5% chance to add connection
        toggle_connection_rate => 0.01,  % 1% chance to toggle

        %% Weight mutation details
        weight_perturb_rate => 0.90,     % 90% perturb, 10% replace
        weight_perturb_strength => 0.30, % Moderate perturbation

        %% Population control
        min_population => 10,
        max_population => 500,

        %% Evaluation
        evaluations_per_individual => 10
    }.

%% @doc Get parameter bounds (min, max).
%%
%% These bounds ensure parameters stay within safe, meaningful ranges
%% regardless of what L1/L2 recommend.
-spec get_bounds() -> map().
get_bounds() ->
    #{
        mutation_rate => {0.01, 0.50},
        mutation_strength => {0.05, 1.0},
        selection_ratio => {0.05, 0.50},
        add_node_rate => {0.0, 0.10},
        add_connection_rate => {0.0, 0.20},
        toggle_connection_rate => {0.0, 0.05},
        weight_perturb_rate => {0.5, 1.0},
        weight_perturb_strength => {0.05, 1.0},
        min_population => {5, 50},
        max_population => {50, 2000},
        evaluations_per_individual => {1, 50}
    }.

%% @doc Get bounds for a specific parameter.
-spec get_bound(atom()) -> {float(), float()} | undefined.
get_bound(Param) ->
    maps:get(Param, get_bounds(), undefined).

%% @doc Clamp a value to its safe bounds.
%%
%% Returns the value constrained to [min, max] for the given parameter.
%% Returns the value unchanged if no bounds are defined.
-spec clamp(atom(), number()) -> number().
clamp(Param, Value) ->
    case get_bound(Param) of
        undefined -> Value;
        {Min, Max} -> max(Min, min(Max, Value))
    end.

%% @doc Apply bounds to all parameters in a map.
%%
%% Any parameter with defined bounds is clamped to [min, max].
%% Parameters without defined bounds are passed through unchanged.
-spec apply_bounds(map()) -> map().
apply_bounds(Params) when is_map(Params) ->
    Bounds = get_bounds(),
    maps:map(
        fun(Key, Value) ->
            case maps:get(Key, Bounds, undefined) of
                undefined -> Value;
                {Min, Max} when is_number(Value) -> max(Min, min(Max, Value));
                _ -> Value
            end
        end,
        Params
    ).

%% @doc Merge provided parameters with defaults.
%%
%% Missing parameters are filled in from defaults.
%% All parameters are then clamped to safe bounds.
-spec merge_with_defaults(map()) -> map().
merge_with_defaults(Params) when is_map(Params) ->
    Defaults = get_defaults(),
    Merged = maps:merge(Defaults, Params),
    apply_bounds(Merged);
merge_with_defaults(_) ->
    get_defaults().

%%% ============================================================================
%%% Parameter Documentation
%%% ============================================================================

%% Parameter reference.
%%
%% == Core Evolution ==
%%
%% `mutation_rate' (0.01 - 0.50, default 0.10):
%%   Probability that each gene in an individual is mutated.
%%   Higher = more exploration, lower = more exploitation.
%%
%% `mutation_strength' (0.05 - 1.0, default 0.30):
%%   Magnitude of weight perturbations (gaussian std dev).
%%   Higher = larger jumps in weight space.
%%
%% `selection_ratio' (0.05 - 0.50, default 0.20):
%%   Fraction of population that survives to next generation.
%%   Lower = stronger selection pressure.
%%
%% == NEAT Topology ==
%%
%% `add_node_rate' (0.0 - 0.10, default 0.03):
%%   Probability of adding a new hidden node (splits an existing connection).
%%   Keep low to avoid complexity explosion.
%%
%% `add_connection_rate' (0.0 - 0.20, default 0.05):
%%   Probability of adding a new connection between existing nodes.
%%   Moderate values allow gradual topology growth.
%%
%% `toggle_connection_rate' (0.0 - 0.05, default 0.01):
%%   Probability of enabling/disabling an existing connection.
%%   Keep low for stable topology exploration.
%%
%% == Weight Mutation ==
%%
%% `weight_perturb_rate' (0.5 - 1.0, default 0.90):
%%   When mutating a weight, probability of perturbing vs. replacing.
%%   Higher = more stable evolution (small changes).
%%
%% `weight_perturb_strength' (0.05 - 1.0, default 0.30):
%%   Standard deviation of weight perturbation gaussian.
%%   Higher = larger weight changes.
