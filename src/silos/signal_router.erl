%% @doc Signal Router for Domain-to-Silo Communication.
%%
%% Routes domain signals (emitted by domain_signals behaviour implementations)
%% to the appropriate silos based on signal category.
%%
%% == Architecture ==
%%
%% Domain signals flow from external applications to internal silos:
%%
%%   Domain App (swai-node, etc.)
%%       │
%%       ├── domain_signals:emit_signals/2
%%       │
%%       ▼
%%   signal_router:route/1
%%       │
%%       ├── ecological signals → ecological_silo
%%       ├── competitive signals → competitive_silo
%%       ├── cultural signals → cultural_silo
%%       └── ... (13 silo categories)
%%
%% The router uses lc_cross_silo infrastructure to deliver signals,
%% treating the domain as a special "domain" source silo.
%%
%% == Usage ==
%%
%% Called by the evaluation loop after each evaluation:
%%
%%   Signals = DomainModule:emit_signals(DomainState, Metrics),
%%   signal_router:route(Signals).
%%
%% @author R.G. Lefever
%% @copyright 2024-2026 R.G. Lefever
-module(signal_router).

%% API
-export([
    route/1,
    route_signal/1,
    register_domain_module/1,
    get_domain_module/0,
    emit_from_domain/2
]).

%% Internal state
-define(DOMAIN_MODULE_KEY, signal_router_domain_module).

%%% ============================================================================
%%% API Functions
%%% ============================================================================

%% @doc Route a list of domain signals to their destination silos.
%%
%% Each signal is a tuple: {Category, Name, Value}
%% Category determines which silo receives the signal.
-spec route([domain_signals:signal()]) -> ok.
route(Signals) when is_list(Signals) ->
    lists:foreach(fun route_signal/1, Signals),
    ok.

%% @doc Route a single domain signal to its destination silo.
-spec route_signal(domain_signals:signal()) -> ok.
route_signal({Category, Name, Value}) when is_atom(Category), is_atom(Name), is_number(Value) ->
    %% Map category to destination silo
    case category_to_silo(Category) of
        skip ->
            %% domain_local signals are not routed
            ok;
        DestSilo ->
            %% Get signal name (generic metrics use natural names)
            SignalName = domain_signal_name(Name),

            %% Clamp value to [0.0, 1.0]
            ClampedValue = max(0.0, min(1.0, Value)),

            %% Route via lc_cross_silo with 'domain' as source
            case whereis(lc_cross_silo) of
                undefined ->
                    %% LC not running, skip
                    ok;
                _Pid ->
                    lc_cross_silo:emit(domain, DestSilo, SignalName, ClampedValue)
            end,
            ok
    end;
route_signal(InvalidSignal) ->
    logger:warning("[signal_router] Invalid signal format: ~p", [InvalidSignal]),
    ok.

%% @doc Register the domain module implementing domain_signals behaviour.
%%
%% Called during application startup to configure signal emission.
-spec register_domain_module(module()) -> ok.
register_domain_module(Module) when is_atom(Module) ->
    persistent_term:put(?DOMAIN_MODULE_KEY, Module),
    ok.

%% @doc Get the registered domain module.
-spec get_domain_module() -> {ok, module()} | {error, not_registered}.
get_domain_module() ->
    try
        Module = persistent_term:get(?DOMAIN_MODULE_KEY),
        {ok, Module}
    catch
        error:badarg ->
            {error, not_registered}
    end.

%% @doc Emit signals from domain state using the registered domain module.
%%
%% Convenience function that:
%% 1. Gets the registered domain module
%% 2. Calls emit_signals/2 on it
%% 3. Routes the resulting signals
-spec emit_from_domain(DomainState :: term(), Metrics :: map()) -> ok | {error, term()}.
emit_from_domain(DomainState, Metrics) ->
    case get_domain_module() of
        {ok, Module} ->
            try
                Signals = Module:emit_signals(DomainState, Metrics),
                route(Signals)
            catch
                Error:Reason ->
                    logger:warning("[signal_router] Error emitting signals: ~p:~p", [Error, Reason]),
                    {error, {emit_failed, Error, Reason}}
            end;
        {error, not_registered} ->
            %% No domain module registered, silently skip
            ok
    end.

%%% ============================================================================
%%% Internal Functions
%%% ============================================================================

%% @private Map signal category to destination silo.
%%
%% Categories are designed to match silo types 1:1.
-spec category_to_silo(domain_signals:signal_category()) -> atom().
category_to_silo(ecological) -> ecological;
category_to_silo(competitive) -> competitive;
category_to_silo(morphological) -> morphological;
category_to_silo(regulatory) -> regulatory;
category_to_silo(task) -> task;
category_to_silo(resource) -> resource;
category_to_silo(distribution) -> distribution;
category_to_silo(temporal) -> temporal;
category_to_silo(developmental) -> developmental;
category_to_silo(cultural) -> cultural;
category_to_silo(social) -> social;
category_to_silo(communication) -> communication;
category_to_silo(economic) -> economic;
category_to_silo(domain_local) ->
    %% domain_local signals are NOT routed to silos
    skip;
category_to_silo(Unknown) ->
    logger:warning("[signal_router] Unknown category: ~p, routing to task silo", [Unknown]),
    task.

%% @private Get signal name for routing.
%%
%% Domain signals are prefixed with 'domain_' because lc_cross_silo requires
%% this prefix for validation (see is_domain_signal/1 in lc_cross_silo.erl).
%% This keeps domain signals namespaced separately from internal silo signals.
-spec domain_signal_name(atom()) -> atom().
domain_signal_name(Name) ->
    %% Add domain_ prefix for lc_cross_silo validation
    list_to_atom("domain_" ++ atom_to_list(Name)).
