%% @doc Common behavior module for Liquid Conglomerate Silos.
%%
%% All 13 silos implement this common pattern derived from task_silo.erl:
%%   gen_server behavior
%%   L0/L1/L2 hierarchical control levels
%%   Evaluation-centric tracking (total_evaluations as primary dimension)
%%   Velocity-based stagnation detection
%%   Cross-silo signal integration via lc_cross_silo
%%   ETS tables for persistent collections (where needed)
%%
%% == Implementing a New Silo ==
%%
%% 1. Create module with -behaviour(lc_silo_behavior)
%% 2. Include lc_silos.hrl for common records
%% 3. Implement required callbacks:
%%      init_silo/1: Initialize silo-specific state
%%      collect_sensors/1: Gather L0 sensor values
%%      apply_actuators/2: Apply L0 actuator outputs
%%      compute_reward/1: Compute reward for LC learning
%%
%% @author R.G. Lefever
%% @copyright 2024-2026 R.G. Lefever
-module(lc_silo_behavior).

%% Behavior callbacks
-callback init_silo(Config :: map()) ->
    {ok, SiloState :: map()} | {error, Reason :: term()}.

-callback collect_sensors(SiloState :: map()) -> Sensors :: map().

-callback apply_actuators(Actuators :: map(), SiloState :: map()) ->
    {ok, NewSiloState :: map()}.

-callback compute_reward(SiloState :: map()) -> Reward :: float().

%% Required: silo identity and time constant
-callback get_silo_type() -> atom().
-callback get_time_constant() -> float().

%% Optional: cross-silo communication
-callback handle_cross_silo_signals(Signals :: map(), SiloState :: map()) ->
    {ok, NewSiloState :: map()}.
-callback emit_cross_silo_signals(SiloState :: map()) -> ok.

%% Optional: event persistence (call lc_event_emitter:emit/3)
-callback emit_silo_events(EventType :: atom(), SiloState :: map()) -> ok.

-optional_callbacks([
    handle_cross_silo_signals/2,
    emit_cross_silo_signals/1,
    emit_silo_events/2
]).

%% API exports for helper functions
-export([
    normalize/3,
    clamp/3,
    compute_velocity/4,
    compute_stagnation_severity/2,
    ema_smooth/3,
    asymmetric_ema_smooth/5
]).

%%% ============================================================================
%%% Helper Functions for Silo Implementations
%%% ============================================================================

%% @doc Normalize a value to [0,1] range given min/max bounds.
-spec normalize(Value :: number(), Min :: number(), Max :: number()) -> float().
normalize(Value, Min, Max) when Max > Min ->
    clamp((Value - Min) / (Max - Min), 0.0, 1.0);
normalize(_Value, _Min, _Max) ->
    0.5.

%% @doc Clamp a value to a specified range.
-spec clamp(Value :: number(), Min :: number(), Max :: number()) -> number().
clamp(Value, Min, _Max) when Value < Min -> Min;
clamp(Value, _Min, Max) when Value > Max -> Max;
clamp(Value, _Min, _Max) -> Value.

%% @doc Compute improvement velocity from checkpoints.
%%
%% Velocity = (delta_fitness / delta_evaluations) * 1000
%%
%% Returns velocity in fitness improvement per 1000 evaluations.
-spec compute_velocity(
    CurrentFitness :: float(),
    CurrentEvals :: non_neg_integer(),
    PrevFitness :: float(),
    PrevEvals :: non_neg_integer()
) -> float().
compute_velocity(CurrentFitness, CurrentEvals, PrevFitness, PrevEvals)
    when CurrentEvals > PrevEvals ->
    DeltaFitness = CurrentFitness - PrevFitness,
    DeltaEvals = CurrentEvals - PrevEvals,
    (DeltaFitness / DeltaEvals) * 1000;
compute_velocity(_CurrentFitness, _CurrentEvals, _PrevFitness, _PrevEvals) ->
    0.0.

%% @doc Compute stagnation severity from velocity.
%%
%% Severity = clamp((threshold - avg_velocity) / threshold, 0.0, 1.0)
%%
%% Returns 0.0 = healthy, 1.0 = critical stagnation.
-spec compute_stagnation_severity(
    AvgVelocity :: float(),
    VelocityThreshold :: float()
) -> float().
compute_stagnation_severity(AvgVelocity, VelocityThreshold) when VelocityThreshold > 0 ->
    RawSeverity = (VelocityThreshold - AvgVelocity) / VelocityThreshold,
    clamp(RawSeverity, 0.0, 1.0);
compute_stagnation_severity(_AvgVelocity, _VelocityThreshold) ->
    0.0.

%% @doc Apply exponential moving average smoothing.
%%
%% SmoothedValue = Momentum * PreviousValue + (1 - Momentum) * NewValue
%%
%% Higher momentum = smoother but slower response.
-spec ema_smooth(
    NewValue :: float(),
    PreviousValue :: float(),
    Momentum :: float()
) -> float().
ema_smooth(NewValue, PreviousValue, Momentum) when Momentum >= 0, Momentum =< 1 ->
    Momentum * PreviousValue + (1.0 - Momentum) * NewValue;
ema_smooth(NewValue, _PreviousValue, _Momentum) ->
    NewValue.

%% @doc Apply asymmetric EMA smoothing for fast escalation, slow de-escalation.
%%
%% When escalating (new value higher): use low momentum (fast response)
%% When de-escalating (new value lower): use high momentum (slow recovery)
%%
%% This prevents oscillation while ensuring responsive intervention.
-spec asymmetric_ema_smooth(
    NewValue :: float(),
    PreviousValue :: float(),
    BaseMomentum :: float(),
    EscalationFactor :: float(),
    DeescalationOffset :: float()
) -> float().
asymmetric_ema_smooth(NewValue, PreviousValue, BaseMomentum, EscalationFactor, DeescalationOffset) ->
    Momentum = compute_asymmetric_momentum(NewValue > PreviousValue, BaseMomentum, EscalationFactor, DeescalationOffset),
    ema_smooth(NewValue, PreviousValue, Momentum).

%% @private Escalating: fast response (low momentum)
compute_asymmetric_momentum(true, BaseMomentum, EscalationFactor, _DeescalationOffset) ->
    BaseMomentum * EscalationFactor;
%% @private De-escalating: slow recovery (high momentum, capped at 0.9)
compute_asymmetric_momentum(false, BaseMomentum, _EscalationFactor, DeescalationOffset) ->
    min(0.9, BaseMomentum + DeescalationOffset).
