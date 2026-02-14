%% @doc Domain Sensor Provider behaviour.
%%
%% Domains implement this behaviour to declare what sensors they provide
%% and how to read sensor values from the domain state.
%%
%% == Example Implementation ==
%%
%% A domain bridge module implementing this behaviour:
%%
%%   -module(my_domain_bridge).
%%   -behaviour(domain_sensors).
%%
%%   sensor_spec() ->
%%       [#{name => vision_food, dimension => 8, range => {0.0, 1.0},
%%          level => l0, category => ecological,
%%          description => "Distance to food in 8 directions"}].
%%
%%   read_sensors(DomainState) ->
%%       #{vision_food => calculate_vision(DomainState)}.
%%
%% @author R.G. Lefever
%% @copyright 2024-2026 R.G. Lefever
-module(domain_sensors).

%% Behaviour callbacks
-callback sensor_spec() -> [sensor_definition()].
-callback read_sensors(DomainState :: term()) -> sensor_readings().

-type sensor_definition() :: #{
    name := atom(),
    dimension := pos_integer(),
    range := {Min :: float(), Max :: float()},
    level := l0 | l1 | l2,
    category := atom(),
    description := binary()
}.

-type sensor_readings() :: #{atom() => [float()]}.

-export_type([sensor_definition/0, sensor_readings/0]).
