-module(lc_supervisor_tests).

-include_lib("eunit/include/eunit.hrl").

%%====================================================================
%% Runtime Silo Control Tests
%%====================================================================

silo_module_test_() ->
    {"silo_module/1 maps silo types to modules", [
        ?_assertEqual(task_silo, lc_supervisor:silo_module(task)),
        ?_assertEqual(resource_silo, lc_supervisor:silo_module(resource)),
        ?_assertEqual(temporal_silo, lc_supervisor:silo_module(temporal)),
        ?_assertEqual(competitive_silo, lc_supervisor:silo_module(competitive)),
        ?_assertEqual(social_silo, lc_supervisor:silo_module(social)),
        ?_assertEqual(cultural_silo, lc_supervisor:silo_module(cultural)),
        ?_assertEqual(ecological_silo, lc_supervisor:silo_module(ecological)),
        ?_assertEqual(morphological_silo, lc_supervisor:silo_module(morphological)),
        ?_assertEqual(developmental_silo, lc_supervisor:silo_module(developmental)),
        ?_assertEqual(regulatory_silo, lc_supervisor:silo_module(regulatory)),
        ?_assertEqual(economic_silo, lc_supervisor:silo_module(economic)),
        ?_assertEqual(communication_silo, lc_supervisor:silo_module(communication)),
        ?_assertEqual(distribution_silo, lc_supervisor:silo_module(distribution)),
        ?_assertEqual({error, unknown_silo}, lc_supervisor:silo_module(unknown))
    ]}.

all_silo_types_test() ->
    AllTypes = lc_supervisor:all_silo_types(),
    ?assertEqual(13, length(AllTypes)),
    ?assert(lists:member(task, AllTypes)),
    ?assert(lists:member(resource, AllTypes)),
    ?assert(lists:member(temporal, AllTypes)),
    ?assert(lists:member(distribution, AllTypes)).

list_available_silos_test() ->
    Available = lc_supervisor:list_available_silos(),
    ?assertEqual(13, length(Available)),
    ?assert(lists:member(task, Available)),
    ?assert(lists:member(temporal, Available)).

%%====================================================================
%% Integration Tests (require supervisor running)
%%====================================================================

supervisor_runtime_control_test_() ->
    {setup,
     fun setup_supervisor/0,
     fun cleanup_supervisor/1,
     fun(Pid) ->
         {"Runtime silo control", [
             {"Core silos are enabled by default",
              fun() -> core_silos_enabled(Pid) end},
             {"Cannot disable core silos",
              fun() -> cannot_disable_core_silos(Pid) end},
             {"Enable and disable extension silo",
              fun() -> enable_disable_extension_silo(Pid) end},
             {"Double enable returns error",
              fun() -> double_enable_error(Pid) end},
             {"Disable not-enabled returns error",
              fun() -> disable_not_enabled_error(Pid) end},
             {"Unknown silo returns error",
              fun() -> unknown_silo_error(Pid) end}
         ]}
     end}.

setup_supervisor() ->
    %% Ensure lc_cross_silo module is available for the supervisor
    case whereis(lc_supervisor) of
        undefined ->
            {ok, Pid} = lc_supervisor:start_link(#{}),
            Pid;
        Pid ->
            Pid
    end.

cleanup_supervisor(Pid) ->
    case is_process_alive(Pid) of
        true ->
            %% Disable any extension silos we enabled
            lists:foreach(
                fun(Silo) -> catch lc_supervisor:disable_silo(Silo) end,
                [temporal, competitive, social, cultural, ecological,
                 morphological, developmental, regulatory, economic,
                 communication, distribution]
            );
        false ->
            ok
    end.

core_silos_enabled(_Pid) ->
    ?assert(lc_supervisor:is_silo_enabled(task)),
    ?assert(lc_supervisor:is_silo_enabled(resource)).

cannot_disable_core_silos(_Pid) ->
    ?assertEqual({error, cannot_disable_core_silo}, lc_supervisor:disable_silo(task)),
    ?assertEqual({error, cannot_disable_core_silo}, lc_supervisor:disable_silo(resource)).

enable_disable_extension_silo(_Pid) ->
    %% Initially not enabled
    ?assertNot(lc_supervisor:is_silo_enabled(temporal)),

    %% Enable it
    ?assertEqual(ok, lc_supervisor:enable_silo(temporal)),
    ?assert(lc_supervisor:is_silo_enabled(temporal)),

    %% Should be in enabled list
    Enabled = lc_supervisor:list_enabled_silos(),
    ?assert(lists:member(temporal, Enabled)),

    %% Disable it
    ?assertEqual(ok, lc_supervisor:disable_silo(temporal)),
    ?assertNot(lc_supervisor:is_silo_enabled(temporal)).

double_enable_error(_Pid) ->
    %% Enable first
    lc_supervisor:enable_silo(competitive),
    %% Try to enable again
    ?assertEqual({error, already_enabled}, lc_supervisor:enable_silo(competitive)),
    %% Clean up
    lc_supervisor:disable_silo(competitive).

disable_not_enabled_error(_Pid) ->
    ?assertNot(lc_supervisor:is_silo_enabled(social)),
    ?assertEqual({error, not_enabled}, lc_supervisor:disable_silo(social)).

unknown_silo_error(_Pid) ->
    ?assertEqual({error, unknown_silo}, lc_supervisor:enable_silo(unknown_silo)),
    ?assertEqual({error, unknown_silo}, lc_supervisor:disable_silo(unknown_silo)).

%%====================================================================
%% Configuration Management Tests
%%====================================================================

get_silo_config_test_() ->
    {setup,
     fun setup_supervisor/0,
     fun cleanup_supervisor/1,
     fun(_Pid) ->
         {"get_silo_config/1", [
             {"Core silo returns empty config",
              fun() ->
                  ?assertEqual({ok, #{}}, lc_supervisor:get_silo_config(task))
              end},
             {"Not-enabled silo returns error",
              fun() ->
                  ?assertEqual({error, not_enabled}, lc_supervisor:get_silo_config(temporal))
              end},
             {"Enabled silo returns its config",
              fun() ->
                  Config = #{realm => <<"test">>},
                  lc_supervisor:enable_silo(temporal, Config),
                  ?assertEqual({ok, Config}, lc_supervisor:get_silo_config(temporal)),
                  lc_supervisor:disable_silo(temporal)
              end},
             {"Unknown silo returns error",
              fun() ->
                  ?assertEqual({error, unknown_silo}, lc_supervisor:get_silo_config(unknown))
              end}
         ]}
     end}.

validate_silo_config_test_() ->
    {"validate_silo_config/2", [
        ?_assertEqual(ok, lc_supervisor:validate_silo_config(temporal, #{})),
        ?_assertEqual(ok, lc_supervisor:validate_silo_config(temporal, #{realm => <<"test">>})),
        ?_assertEqual(ok, lc_supervisor:validate_silo_config(competitive, #{archive_max_size => 50})),
        ?_assertEqual({error, unknown_silo}, lc_supervisor:validate_silo_config(unknown, #{}))
    ]}.

silo_dependencies_test_() ->
    {"silo_dependencies/1", [
        ?_assertEqual([], lc_supervisor:silo_dependencies(task)),
        ?_assertEqual([], lc_supervisor:silo_dependencies(temporal)),
        ?_assertEqual([competitive], lc_supervisor:silo_dependencies(social)),
        ?_assertEqual([social], lc_supervisor:silo_dependencies(cultural)),
        ?_assertEqual([social], lc_supervisor:silo_dependencies(communication)),
        ?_assertEqual([temporal], lc_supervisor:silo_dependencies(developmental)),
        ?_assertEqual([developmental], lc_supervisor:silo_dependencies(regulatory))
    ]}.

dependency_enforcement_test_() ->
    {setup,
     fun setup_supervisor/0,
     fun cleanup_supervisor/1,
     fun(_Pid) ->
         {"Dependency enforcement", [
             {"Cannot enable social without competitive",
              fun() ->
                  ?assertEqual({error, {missing_dependency, competitive}},
                               lc_supervisor:enable_silo(social))
              end},
             {"Can enable social after competitive",
              fun() ->
                  ok = lc_supervisor:enable_silo(competitive),
                  ?assertEqual(ok, lc_supervisor:enable_silo(social)),
                  %% Clean up in reverse order
                  lc_supervisor:disable_silo(social),
                  lc_supervisor:disable_silo(competitive)
              end},
             {"Cannot disable competitive while social is enabled",
              fun() ->
                  ok = lc_supervisor:enable_silo(competitive),
                  ok = lc_supervisor:enable_silo(social),
                  ?assertEqual({error, {has_dependents, [social]}},
                               lc_supervisor:disable_silo(competitive)),
                  %% Clean up properly
                  lc_supervisor:disable_silo(social),
                  lc_supervisor:disable_silo(competitive)
              end}
         ]}
     end}.

reconfigure_silo_test_() ->
    {setup,
     fun setup_supervisor/0,
     fun cleanup_supervisor/1,
     fun(_Pid) ->
         {"reconfigure_silo/2", [
             {"Cannot reconfigure core silos",
              fun() ->
                  ?assertEqual({error, cannot_reconfigure_core_silo},
                               lc_supervisor:reconfigure_silo(task, #{})),
                  ?assertEqual({error, cannot_reconfigure_core_silo},
                               lc_supervisor:reconfigure_silo(resource, #{}))
              end},
             {"Cannot reconfigure not-enabled silo",
              fun() ->
                  ?assertEqual({error, not_enabled},
                               lc_supervisor:reconfigure_silo(temporal, #{}))
              end},
             {"Can reconfigure enabled silo",
              fun() ->
                  ok = lc_supervisor:enable_silo(temporal, #{realm => <<"old">>}),
                  ?assertEqual({ok, #{realm => <<"old">>}},
                               lc_supervisor:get_silo_config(temporal)),
                  ok = lc_supervisor:reconfigure_silo(temporal, #{realm => <<"new">>}),
                  ?assertEqual({ok, #{realm => <<"new">>}},
                               lc_supervisor:get_silo_config(temporal)),
                  lc_supervisor:disable_silo(temporal)
              end}
         ]}
     end}.
