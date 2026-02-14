%% @doc Unit tests for regulatory_silo module.
%%
%% Tests gene expression, module activation, and epigenetic functionality.
%%
%% @author R.G. Lefever
%% @copyright 2024-2026 R.G. Lefever
-module(regulatory_silo_tests).

-include_lib("eunit/include/eunit.hrl").

%%% ============================================================================
%%% Setup/Teardown
%%% ============================================================================

setup() ->
    case whereis(regulatory_silo) of
        undefined -> ok;
        Pid ->
            catch gen_server:stop(Pid, normal, 1000),
            timer:sleep(50)
    end,
    ok.

cleanup(_) ->
    case whereis(regulatory_silo) of
        undefined -> ok;
        Pid ->
            catch gen_server:stop(Pid, normal, 1000),
            timer:sleep(50)
    end,
    ok.

%%% ============================================================================
%%% Test Suite
%%% ============================================================================

regulatory_silo_test_() ->
    {foreach,
     fun setup/0,
     fun cleanup/1,
     [
         {"start_link with default config", fun start_link_default_test/0},
         {"start_link with custom config", fun start_link_custom_config_test/0},
         {"get_params returns defaults", fun get_params_test/0},
         {"update_expression tracks expression", fun update_expression_test/0},
         {"get_expression retrieves expression", fun get_expression_test/0},
         {"activate_module activates", fun activate_module_test/0},
         {"deactivate_module deactivates", fun deactivate_module_test/0},
         {"add_epigenetic_mark tracks marks", fun add_epigenetic_mark_test/0},
         {"get_regulatory_stats returns metrics", fun get_regulatory_stats_test/0},
         {"reset clears state", fun reset_test/0},
         {"get_state returns full state", fun get_state_test/0}
     ]}.

%%% ============================================================================
%%% Individual Tests
%%% ============================================================================

start_link_default_test() ->
    {ok, Pid} = regulatory_silo:start_link(),
    ?assert(is_pid(Pid)),
    ?assert(is_process_alive(Pid)),
    ok = gen_server:stop(Pid).

start_link_custom_config_test() ->
    Config = #{
        realm => <<"test_realm">>,
        enabled_levels => [l0]
    },
    {ok, Pid} = regulatory_silo:start_link(Config),
    ?assert(is_pid(Pid)),
    ok = gen_server:stop(Pid).

get_params_test() ->
    {ok, Pid} = regulatory_silo:start_link(),
    Params = regulatory_silo:get_params(Pid),
    ?assert(is_map(Params)),
    ?assert(maps:is_key(expression_threshold, Params)),
    ?assert(maps:is_key(context_sensitivity, Params)),
    ?assert(maps:is_key(epigenetic_inheritance_strength, Params)),
    ?assertEqual(0.5, maps:get(expression_threshold, Params)),
    ?assertEqual(0.5, maps:get(context_sensitivity, Params)),
    ok = gen_server:stop(Pid).

update_expression_test() ->
    {ok, Pid} = regulatory_silo:start_link(),

    %% Update expression (3 args: Pid, IndividualId, GeneExpressionMap)
    ok = regulatory_silo:update_expression(Pid, ind_1, #{gene_a => true, gene_b => false}),
    ok = regulatory_silo:update_expression(Pid, ind_2, #{gene_a => false, gene_b => true}),
    timer:sleep(50),

    %% Check stats
    Stats = regulatory_silo:get_regulatory_stats(Pid),
    ?assertEqual(2, maps:get(expression_count, Stats)),
    ok = gen_server:stop(Pid).

get_expression_test() ->
    {ok, Pid} = regulatory_silo:start_link(),

    %% Add expression
    ok = regulatory_silo:update_expression(Pid, ind_1, #{gene_a => true, gene_b => false}),
    timer:sleep(50),

    %% Retrieve it
    {ok, Expression} = regulatory_silo:get_expression(Pid, ind_1),
    ?assert(is_map(Expression)),

    %% Non-existent individual
    ?assertEqual(not_found, regulatory_silo:get_expression(Pid, non_existent)),
    ok = gen_server:stop(Pid).

activate_module_test() ->
    {ok, Pid} = regulatory_silo:start_link(),

    %% Activate modules (2 args: Pid, ModuleId)
    ok = regulatory_silo:activate_module(Pid, module_1),
    ok = regulatory_silo:activate_module(Pid, module_2),
    timer:sleep(50),

    %% Check stats
    Stats = regulatory_silo:get_regulatory_stats(Pid),
    ?assertEqual(2, maps:get(module_count, Stats)),
    ok = gen_server:stop(Pid).

deactivate_module_test() ->
    {ok, Pid} = regulatory_silo:start_link(),

    %% First activate
    ok = regulatory_silo:activate_module(Pid, module_1),
    timer:sleep(50),

    %% Then deactivate (2 args: Pid, ModuleId)
    ok = regulatory_silo:deactivate_module(Pid, module_1),
    timer:sleep(50),

    %% Module should be removed or marked inactive
    ok = gen_server:stop(Pid).

add_epigenetic_mark_test() ->
    {ok, Pid} = regulatory_silo:start_link(),

    %% Add epigenetic marks (3 args: Pid, GeneId, MarkType)
    ok = regulatory_silo:add_epigenetic_mark(Pid, gene_a, methylation),
    ok = regulatory_silo:add_epigenetic_mark(Pid, gene_b, acetylation),
    ok = regulatory_silo:add_epigenetic_mark(Pid, gene_c, methylation),
    timer:sleep(50),

    %% Check stats
    Stats = regulatory_silo:get_regulatory_stats(Pid),
    ?assertEqual(3, maps:get(epigenetic_mark_count, Stats)),
    ok = gen_server:stop(Pid).

get_regulatory_stats_test() ->
    {ok, Pid} = regulatory_silo:start_link(),

    %% Empty stats
    Stats1 = regulatory_silo:get_regulatory_stats(Pid),
    ?assertEqual(0, maps:get(expression_count, Stats1)),
    ?assertEqual(0, maps:get(module_count, Stats1)),
    ?assertEqual(0, maps:get(epigenetic_mark_count, Stats1)),
    ?assertEqual(0, maps:get(switch_count, Stats1)),

    %% Add some data
    ok = regulatory_silo:update_expression(Pid, ind_1, #{gene_a => true}),
    ok = regulatory_silo:activate_module(Pid, module_1),
    ok = regulatory_silo:add_epigenetic_mark(Pid, gene_a, methylation),
    timer:sleep(50),

    Stats2 = regulatory_silo:get_regulatory_stats(Pid),
    ?assertEqual(1, maps:get(expression_count, Stats2)),
    ?assertEqual(1, maps:get(module_count, Stats2)),
    ?assertEqual(1, maps:get(epigenetic_mark_count, Stats2)),
    ok = gen_server:stop(Pid).

reset_test() ->
    {ok, Pid} = regulatory_silo:start_link(),

    %% Add data
    ok = regulatory_silo:update_expression(Pid, ind_1, #{gene_a => true}),
    ok = regulatory_silo:add_epigenetic_mark(Pid, gene_a, methylation),
    timer:sleep(50),

    %% Verify data exists
    Stats1 = regulatory_silo:get_regulatory_stats(Pid),
    ?assert(maps:get(expression_count, Stats1) > 0),

    %% Reset
    ok = regulatory_silo:reset(Pid),

    %% Verify reset
    Stats2 = regulatory_silo:get_regulatory_stats(Pid),
    ?assertEqual(0, maps:get(expression_count, Stats2)),
    ?assertEqual(0, maps:get(epigenetic_mark_count, Stats2)),
    ok = gen_server:stop(Pid).

get_state_test() ->
    {ok, Pid} = regulatory_silo:start_link(#{realm => <<"test">>}),
    State = regulatory_silo:get_state(Pid),
    ?assert(is_map(State)),
    ?assertEqual(<<"test">>, maps:get(realm, State)),
    ?assert(maps:is_key(current_params, State)),
    ?assert(maps:is_key(switch_count, State)),
    ?assert(maps:is_key(sensors, State)),
    ok = gen_server:stop(Pid).

%%% ============================================================================
%%% Behavior Callback Tests
%%% ============================================================================

behavior_callbacks_test_() ->
    [
        {"get_silo_type returns regulatory",
         ?_assertEqual(regulatory, regulatory_silo:get_silo_type())},
        {"get_time_constant returns 45.0",
         ?_assertEqual(45.0, regulatory_silo:get_time_constant())}
    ].
