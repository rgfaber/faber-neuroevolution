%% @doc Domain Actuator Consumer behaviour.
%%
%% Domains implement this behaviour to declare what actuators they accept
%% and how to apply actuator outputs to the domain state.
%%
%% == Example Implementation ==
%%
%% A domain bridge module implementing this behaviour:
%%
%%   -module(my_domain_bridge).
%%   -behaviour(domain_actuators).
%%
%%   actuator_spec() ->
%%       [#{name => turn, dimension => 1, range => {-1.0, 1.0},
%%          level => l0, category => motor,
%%          description => "Rotation amount"}].
%%
%%   apply_actuators(Outputs, DomainState) ->
%%       Turn = maps:get(turn, Outputs, [0.0]),
%%       update_rotation(DomainState, hd(Turn)).
%%
%% @author R.G. Lefever
%% @copyright 2024-2026 R.G. Lefever
-module(domain_actuators).

%% Behaviour callbacks
-callback actuator_spec() -> [actuator_definition()].
-callback apply_actuators(actuator_outputs(), DomainState :: term()) -> NewDomainState :: term().

-type actuator_definition() :: #{
    name := atom(),
    dimension := pos_integer(),
    range := {Min :: float(), Max :: float()},
    level := l0 | l1 | l2,
    category := atom(),
    description := binary()
}.

-type actuator_outputs() :: #{atom() => [float()]}.

-export_type([actuator_definition/0, actuator_outputs/0]).
