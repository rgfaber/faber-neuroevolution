%% @doc Domain Reward Provider behaviour.
%%
%% Domains implement this behaviour to declare what reward signals they emit
%% and how to compute rewards from domain state and metrics.
%%
%% == Example Implementation ==
%%
%% A domain bridge module implementing this behaviour:
%%
%%   -module(my_domain_bridge).
%%   -behaviour(domain_rewards).
%%
%%   reward_spec() ->
%%       [#{name => survival, weight => 1.0, level => l0,
%%          sign => reward, category => temporal,
%%          description => "Ticks survived"},
%%        #{name => death, weight => 100.0, level => l0,
%%          sign => punishment, category => temporal,
%%          description => "Death penalty"}].
%%
%%   compute_rewards(DomainState, Metrics) ->
%%       Ticks = maps:get(ticks, Metrics, 0),
%%       Died = maps:get(died, Metrics, false),
%%       #{survival => Ticks * 1.0,
%%         death => case Died of true -> -100.0; false -> 0.0 end}.
%%
%% @author R.G. Lefever
%% @copyright 2024-2026 R.G. Lefever
-module(domain_rewards).

%% Behaviour callbacks
-callback reward_spec() -> [reward_definition()].
-callback compute_rewards(DomainState :: term(), Metrics :: map()) -> reward_signals().

-type reward_definition() :: #{
    name := atom(),
    weight := float(),
    level := l0 | l1 | l2,
    sign := reward | punishment,
    category := atom(),
    description := binary()
}.

-type reward_signals() :: #{atom() => float()}.

-export_type([reward_definition/0, reward_signals/0]).
