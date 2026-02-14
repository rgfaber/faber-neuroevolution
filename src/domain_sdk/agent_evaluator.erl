%% Agent Evaluator Behaviour.
%%
%% This module defines the behaviour for evaluators that calculate fitness
%% from episode metrics. The evaluator is the bridge between domain-specific
%% performance metrics and the neuroevolution engine's fitness scores.
%%
%% An evaluator answers the question: "HOW WELL did this agent perform?"
%%
%% == Overview ==
%%
%% The evaluator:
%% <ul>
%%   <li><b>Receives</b> metrics from a completed episode</li>
%%   <li><b>Calculates</b> a fitness score (higher = better)</li>
%%   <li><b>Optionally</b> provides component breakdown for analysis</li>
%% </ul>
%%
%% == Implementing an Evaluator ==
%%
%% ```
%% -module(hex_arena_evaluator).
%% -behaviour(agent_evaluator).
%%
%% -export([name/0, calculate_fitness/1, fitness_components/1]).
%%
%% name() -> <<"hex_arena_fitness">>.
%%
%% calculate_fitness(Metrics) ->
%%     Survival = maps:get(ticks_survived, Metrics, 0) * 0.1,
%%     Food = maps:get(food_eaten, Metrics, 0) * 150.0,
%%     Kills = maps:get(kills, Metrics, 0) * 100.0,
%%     Survival + Food + Kills.
%%
%% fitness_components(Metrics) ->
%%     #{
%%         survival => maps:get(ticks_survived, Metrics, 0) * 0.1,
%%         food => maps:get(food_eaten, Metrics, 0) * 150.0,
%%         kills => maps:get(kills, Metrics, 0) * 100.0
%%     }.
%% '''
%%
%% == Fitness Design ==
%%
%% Good fitness functions:
%% <ul>
%%   <li><b>Reward the goal</b> - Primary objective gets highest weight</li>
%%   <li><b>Provide gradient</b> - Partial progress gets partial reward</li>
%%   <li><b>Avoid local optima</b> - Don't reward degenerate strategies</li>
%%   <li><b>Stay positive</b> - Fitness >= 0 simplifies selection</li>
%% </ul>
%%
%% == Component Weights ==
%%
%% Example weight ratios for different goals:
%% <ul>
%%   <li><b>Survival game</b>: survival:food:combat = 1:10:5</li>
%%   <li><b>Foraging</b>: survival:food:efficiency = 1:50:10</li>
%%   <li><b>Combat arena</b>: survival:kills:damage = 1:100:20</li>
%% </ul>
%%
%% == Shaped Rewards ==
%%
%% Beyond terminal metrics, evaluators can incorporate:
%% <ul>
%%   <li><b>Approach rewards</b> - Getting closer to food</li>
%%   <li><b>Exploration bonus</b> - Visiting new areas</li>
%%   <li><b>Efficiency penalty</b> - Energy spent per action</li>
%% </ul>
%%
%% @author R.G. Lefever
%% @copyright 2024-2026 R.G. Lefever
%% @see agent_environment
%% @see agent_bridge
-module(agent_evaluator).

%%% ============================================================================
%%% Types
%%% ============================================================================

-type evaluator_name() :: binary().
%% Unique identifier for the evaluator.
%% Examples: `<<"hex_arena_fitness">>', `<<"foraging_score">>'.

-type metrics() :: map().
%% Performance metrics from a completed episode.
%% See `agent_environment:metrics()'.

-type fitness() :: float().
%% Scalar fitness score. Higher is better.
%% Should be non-negative for compatibility with selection algorithms.

-type fitness_components() :: #{atom() => float()}.
%% Breakdown of fitness into named components.
%% Useful for analysis and debugging.

-export_type([
    evaluator_name/0,
    metrics/0,
    fitness/0,
    fitness_components/0
]).

%%% ============================================================================
%%% Behaviour Callbacks
%%% ============================================================================

%% Returns the unique name of this evaluator.
%%
%% The name should be descriptive and indicate the evaluation strategy.
%% It is used for logging and configuration.
%%
%% Example:
%% ```
%% name() -> <<"hex_arena_fitness">>.
%% '''
-callback name() -> evaluator_name().

%% Calculates fitness from episode metrics.
%%
%% This is the core evaluation function. It should:
%% <ul>
%%   <li>Extract relevant metrics</li>
%%   <li>Apply weights and transformations</li>
%%   <li>Return a single scalar fitness value</li>
%% </ul>
%%
%% The returned fitness should be non-negative. Higher values indicate
%% better performance.
%%
%% Example:
%% ```
%% calculate_fitness(Metrics) ->
%%     Ticks = maps:get(ticks_survived, Metrics, 0),
%%     Food = maps:get(food_eaten, Metrics, 0),
%%     Ticks * 0.1 + Food * 150.0.
%% '''
-callback calculate_fitness(Metrics) -> Fitness when
    Metrics :: metrics(),
    Fitness :: fitness().

%% Returns a breakdown of fitness components.
%%
%% Optional callback for analysis. Returns a map showing how each
%% component contributes to the total fitness.
%%
%% Example:
%% ```
%% fitness_components(Metrics) ->
%%     #{
%%         survival => maps:get(ticks_survived, Metrics, 0) * 0.1,
%%         food => maps:get(food_eaten, Metrics, 0) * 150.0
%%     }.
%% '''
-callback fitness_components(Metrics) -> Components when
    Metrics :: metrics(),
    Components :: fitness_components().

%% Make fitness_components optional
-optional_callbacks([fitness_components/1]).

%%% ============================================================================
%%% API Functions
%%% ============================================================================

-export([
    validate/1,
    get_info/1,
    evaluate/2,
    evaluate_with_breakdown/2
]).

%% Validates that a module correctly implements the agent_evaluator behaviour.
%%
%% Checks:
%% <ul>
%%   <li>Module exports all required callbacks</li>
%%   <li>name/0 returns a non-empty binary</li>
%% </ul>
%%
%% Returns `ok' if valid, or `{error, Reasons}' with a list of validation errors.
-spec validate(Module) -> ok | {error, [Reason]} when
    Module :: module(),
    Reason :: term().
validate(Module) ->
    Checks = [
        fun() -> validate_exports(Module) end,
        fun() -> validate_name(Module) end
    ],
    Errors = lists:filtermap(
        fun(Check) ->
            case Check() of
                ok -> false;
                {error, Reason} -> {true, Reason}
            end
        end,
        Checks
    ),
    case Errors of
        [] -> ok;
        _ -> {error, Errors}
    end.

%% Retrieves evaluator info from a module.
%%
%% Returns a map with name and whether fitness_components is supported.
-spec get_info(Module) -> {ok, Info} | {error, Reason} when
    Module :: module(),
    Info :: #{
        name := evaluator_name(),
        has_components := boolean()
    },
    Reason :: term().
get_info(Module) ->
    case validate(Module) of
        ok ->
            Name = Module:name(),
            HasComponents = erlang:function_exported(Module, fitness_components, 1),
            {ok, #{
                name => Name,
                has_components => HasComponents
            }};
        {error, _} = Error ->
            Error
    end.

%% Evaluates metrics using the given evaluator module.
%%
%% Convenience function that validates the evaluator and calculates fitness.
-spec evaluate(Module, Metrics) -> {ok, Fitness} | {error, Reason} when
    Module :: module(),
    Metrics :: metrics(),
    Fitness :: fitness(),
    Reason :: term().
evaluate(Module, Metrics) ->
    case validate(Module) of
        ok ->
            try
                Fitness = Module:calculate_fitness(Metrics),
                {ok, Fitness}
            catch
                _:Error -> {error, {calculation_failed, Error}}
            end;
        {error, _} = Error ->
            Error
    end.

%% Evaluates metrics and returns both fitness and component breakdown.
%%
%% If the evaluator implements fitness_components/1, returns the breakdown.
%% Otherwise, returns an empty component map.
-spec evaluate_with_breakdown(Module, Metrics) -> {ok, Fitness, Components} | {error, Reason} when
    Module :: module(),
    Metrics :: metrics(),
    Fitness :: fitness(),
    Components :: fitness_components(),
    Reason :: term().
evaluate_with_breakdown(Module, Metrics) ->
    case validate(Module) of
        ok ->
            try
                Fitness = Module:calculate_fitness(Metrics),
                Components = case erlang:function_exported(Module, fitness_components, 1) of
                    true -> Module:fitness_components(Metrics);
                    false -> #{}
                end,
                {ok, Fitness, Components}
            catch
                _:Error -> {error, {calculation_failed, Error}}
            end;
        {error, _} = Error ->
            Error
    end.

%%% ============================================================================
%%% Internal Functions
%%% ============================================================================

%% @private
validate_exports(Module) ->
    RequiredExports = [{name, 0}, {calculate_fitness, 1}],
    Exports = Module:module_info(exports),
    Missing = [F || F <- RequiredExports, not lists:member(F, Exports)],
    case Missing of
        [] -> ok;
        _ -> {error, {missing_exports, Missing}}
    end.

%% @private
validate_name(Module) ->
    try Module:name() of
        Name when is_binary(Name), byte_size(Name) > 0 ->
            ok;
        Name when is_binary(Name) ->
            {error, {invalid_name, empty_binary}};
        Other ->
            {error, {invalid_name, {expected_binary, Other}}}
    catch
        _:Reason ->
            {error, {name_callback_failed, Reason}}
    end.
