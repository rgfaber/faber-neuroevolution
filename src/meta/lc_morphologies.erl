%% @doc Registration module for Liquid Conglomerate morphologies.
%%
%% This module provides functions to register all LC morphologies with
%% the morphology_registry. Call register_all/0 during application startup.
%%
%% == Morphology Hierarchy ==
%%
%% The LC uses three chained LTC networks:
%%
%% L2 (Strategic, tau=100):
%%   - Inputs: Evolution metrics (fitness, stagnation, diversity, etc.)
%%   - Outputs: 4 strategic signals
%%
%% L1 (Tactical, tau=50):
%%   - Inputs: L2's 4 strategic signals
%%   - Outputs: 5 tactical signals
%%
%% L0 (Reactive, tau=10):
%%   - Inputs: L1's 5 tactical signals + emergent metrics (evolvable)
%%   - Outputs: Final hyperparameters (mutation_rate, selection_ratio, etc.)
%%
%% @author R.G. Lefever
%% @copyright 2024-2026 R.G. Lefever
-module(lc_morphologies).

-export([
    register_all/0,
    unregister_all/0,
    list_morphologies/0
]).

%% @doc Register all LC morphologies with the morphology registry.
%%
%% This should be called during application startup, after faber_tweann
%% is started and morphology_registry is available.
%%
%% @returns ok
-spec register_all() -> ok.
register_all() ->
    ok = morphology_registry:register(lc_l2, lc_l2_morphology),
    ok = morphology_registry:register(lc_l1, lc_l1_morphology),
    ok = morphology_registry:register(lc_l0, lc_l0_morphology),
    error_logger:info_msg("[lc_morphologies] Registered LC morphologies: ~p~n",
        [list_morphologies()]),
    ok.

%% @doc Unregister all LC morphologies.
%%
%% Useful for testing or cleanup.
%%
%% @returns ok
-spec unregister_all() -> ok.
unregister_all() ->
    morphology_registry:unregister(lc_l2),
    morphology_registry:unregister(lc_l1),
    morphology_registry:unregister(lc_l0),
    ok.

%% @doc List all LC morphology names.
%%
%% @returns List of morphology atoms
-spec list_morphologies() -> [atom()].
list_morphologies() ->
    [lc_l2, lc_l1, lc_l0].
