%% @doc Test environment with missing exports for agent_environment tests.
-module(test_missing_exports_environment).

%% Only exports name/0, missing other callbacks
-export([name/0]).

name() -> <<"incomplete_env">>.
