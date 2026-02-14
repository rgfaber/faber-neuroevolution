-module(signal_router_tests).
-include_lib("eunit/include/eunit.hrl").

%%% ============================================================================
%%% Test Cases
%%% ============================================================================

route_empty_list_test() ->
    ?assertEqual(ok, signal_router:route([])).

route_single_signal_test() ->
    %% Routes without error (lc_cross_silo may not be running)
    ?assertEqual(ok, signal_router:route([{ecological, food_scarcity, 0.5}])).

route_multiple_signals_test() ->
    Signals = [
        {ecological, food_scarcity, 0.7},
        {competitive, predator_ratio, 0.3},
        {cultural, behavioral_diversity, 0.8}
    ],
    ?assertEqual(ok, signal_router:route(Signals)).

route_signal_clamps_value_test() ->
    %% Values outside [0,1] should be clamped
    ?assertEqual(ok, signal_router:route([{ecological, test_signal, 1.5}])),
    ?assertEqual(ok, signal_router:route([{ecological, test_signal, -0.5}])).

route_signal_invalid_format_test() ->
    %% Invalid signal format should log warning but not crash
    ?assertEqual(ok, signal_router:route_signal(invalid)),
    ?assertEqual(ok, signal_router:route_signal({not_atom, 123, "string"})).

register_domain_module_test() ->
    ?assertEqual(ok, signal_router:register_domain_module(test_domain_module)),
    ?assertEqual({ok, test_domain_module}, signal_router:get_domain_module()).

emit_from_domain_not_registered_test() ->
    %% Clear any registered module
    persistent_term:erase(signal_router_domain_module),
    %% Should return ok when no module registered
    ?assertEqual(ok, signal_router:emit_from_domain(#{}, #{})).

category_mapping_test() ->
    %% All 13 categories should route without error
    Categories = [
        ecological, competitive, morphological, regulatory,
        task, resource, distribution, temporal,
        developmental, cultural, social, communication, economic
    ],
    lists:foreach(
        fun(Cat) ->
            ?assertEqual(ok, signal_router:route([{Cat, test_signal, 0.5}]))
        end,
        Categories
    ).
