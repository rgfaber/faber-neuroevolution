%% @doc Unit tests for lc_silo_behavior helper functions.
%%
%% Tests utility functions used by all Liquid Conglomerate silos.
%%
%% @author R.G. Lefever
%% @copyright 2024-2026 R.G. Lefever
-module(lc_silo_behavior_tests).

-include_lib("eunit/include/eunit.hrl").

%%% ============================================================================
%%% normalize/3 Tests
%%% ============================================================================

normalize_test_() ->
    [
        {"normalizes value to 0-1 range",
         ?_assertEqual(0.5, lc_silo_behavior:normalize(50, 0, 100))},
        {"returns 0.0 for value at min",
         ?_assertEqual(0.0, lc_silo_behavior:normalize(0, 0, 100))},
        {"returns 1.0 for value at max",
         ?_assertEqual(1.0, lc_silo_behavior:normalize(100, 0, 100))},
        {"clamps value below min to 0.0",
         ?_assertEqual(0.0, lc_silo_behavior:normalize(-10, 0, 100))},
        {"clamps value above max to 1.0",
         ?_assertEqual(1.0, lc_silo_behavior:normalize(150, 0, 100))},
        {"handles negative range",
         ?_assertEqual(0.5, lc_silo_behavior:normalize(0, -100, 100))},
        {"returns 0.5 when max equals min",
         ?_assertEqual(0.5, lc_silo_behavior:normalize(50, 100, 100))},
        {"returns 0.5 when max less than min",
         ?_assertEqual(0.5, lc_silo_behavior:normalize(50, 100, 0))}
    ].

%%% ============================================================================
%%% clamp/3 Tests
%%% ============================================================================

clamp_test_() ->
    [
        {"returns value when in range",
         ?_assertEqual(50, lc_silo_behavior:clamp(50, 0, 100))},
        {"clamps to min when below",
         ?_assertEqual(0, lc_silo_behavior:clamp(-10, 0, 100))},
        {"clamps to max when above",
         ?_assertEqual(100, lc_silo_behavior:clamp(150, 0, 100))},
        {"returns min when value equals min",
         ?_assertEqual(0, lc_silo_behavior:clamp(0, 0, 100))},
        {"returns max when value equals max",
         ?_assertEqual(100, lc_silo_behavior:clamp(100, 0, 100))},
        {"works with float values",
         ?_assertEqual(0.5, lc_silo_behavior:clamp(0.5, 0.0, 1.0))},
        {"works with negative range",
         ?_assertEqual(-50, lc_silo_behavior:clamp(-50, -100, 0))}
    ].

%%% ============================================================================
%%% compute_velocity/4 Tests
%%% ============================================================================

compute_velocity_test_() ->
    [
        {"computes positive velocity",
         fun() ->
             V = lc_silo_behavior:compute_velocity(100.0, 1000, 50.0, 500),
             ?assertEqual(100.0, V)  % (100-50)/(1000-500)*1000 = 100
         end},
        {"computes negative velocity",
         fun() ->
             V = lc_silo_behavior:compute_velocity(50.0, 1000, 100.0, 500),
             ?assertEqual(-100.0, V)  % (50-100)/(1000-500)*1000 = -100
         end},
        {"returns 0 when no progress",
         fun() ->
             V = lc_silo_behavior:compute_velocity(50.0, 1000, 50.0, 500),
             ?assertEqual(0.0, V)
         end},
        {"returns 0 when evals not increased",
         fun() ->
             V = lc_silo_behavior:compute_velocity(100.0, 500, 50.0, 500),
             ?assertEqual(0.0, V)
         end},
        {"returns 0 when current evals less than prev",
         fun() ->
             V = lc_silo_behavior:compute_velocity(100.0, 400, 50.0, 500),
             ?assertEqual(0.0, V)
         end}
    ].

%%% ============================================================================
%%% compute_stagnation_severity/2 Tests
%%% ============================================================================

stagnation_severity_test_() ->
    [
        {"returns 0.0 when velocity exceeds threshold",
         fun() ->
             S = lc_silo_behavior:compute_stagnation_severity(2.0, 1.0),
             ?assertEqual(0.0, S)
         end},
        {"returns 1.0 when velocity is 0",
         fun() ->
             S = lc_silo_behavior:compute_stagnation_severity(0.0, 1.0),
             ?assertEqual(1.0, S)
         end},
        {"returns 0.5 when velocity is half threshold",
         fun() ->
             S = lc_silo_behavior:compute_stagnation_severity(0.5, 1.0),
             ?assertEqual(0.5, S)
         end},
        {"clamps negative velocity to 1.0",
         fun() ->
             S = lc_silo_behavior:compute_stagnation_severity(-0.5, 1.0),
             ?assertEqual(1.0, S)
         end},
        {"returns 0.0 when threshold is 0",
         fun() ->
             S = lc_silo_behavior:compute_stagnation_severity(1.0, 0.0),
             ?assertEqual(0.0, S)
         end},
        {"returns 0.0 when threshold is negative",
         fun() ->
             S = lc_silo_behavior:compute_stagnation_severity(1.0, -1.0),
             ?assertEqual(0.0, S)
         end}
    ].

%%% ============================================================================
%%% ema_smooth/3 Tests
%%% ============================================================================

ema_smooth_test_() ->
    [
        {"momentum 0 returns new value",
         fun() ->
             Result = lc_silo_behavior:ema_smooth(1.0, 0.0, 0.0),
             ?assertEqual(1.0, Result)
         end},
        {"momentum 1 returns previous value",
         fun() ->
             Result = lc_silo_behavior:ema_smooth(1.0, 0.0, 1.0),
             ?assertEqual(0.0, Result)
         end},
        {"momentum 0.5 returns average",
         fun() ->
             Result = lc_silo_behavior:ema_smooth(1.0, 0.0, 0.5),
             ?assertEqual(0.5, Result)
         end},
        {"typical momentum 0.9",
         fun() ->
             Result = lc_silo_behavior:ema_smooth(1.0, 0.0, 0.9),
             ?assert(abs(Result - 0.1) < 0.001)
         end},
        {"invalid momentum returns new value",
         fun() ->
             Result = lc_silo_behavior:ema_smooth(1.0, 0.0, 1.5),
             ?assertEqual(1.0, Result)
         end},
        {"negative momentum returns new value",
         fun() ->
             Result = lc_silo_behavior:ema_smooth(1.0, 0.0, -0.5),
             ?assertEqual(1.0, Result)
         end}
    ].

%%% ============================================================================
%%% asymmetric_ema_smooth/5 Tests
%%% ============================================================================

asymmetric_ema_smooth_test_() ->
    [
        {"escalating uses low momentum (fast response)",
         fun() ->
             %% NewValue > PreviousValue => escalating
             %% Momentum = BaseMomentum * EscalationFactor = 0.5 * 0.5 = 0.25
             %% Result = 0.25 * 0.0 + 0.75 * 1.0 = 0.75
             Result = lc_silo_behavior:asymmetric_ema_smooth(1.0, 0.0, 0.5, 0.5, 0.2),
             ?assertEqual(0.75, Result)
         end},
        {"de-escalating uses high momentum (slow response)",
         fun() ->
             %% NewValue < PreviousValue => de-escalating
             %% Momentum = min(0.9, BaseMomentum + DeescalationOffset) = min(0.9, 0.5 + 0.2) = 0.7
             %% Result = 0.7 * 1.0 + 0.3 * 0.0 = 0.7
             Result = lc_silo_behavior:asymmetric_ema_smooth(0.0, 1.0, 0.5, 0.5, 0.2),
             ?assertEqual(0.7, Result)
         end},
        {"de-escalating momentum capped at 0.9",
         fun() ->
             %% De-escalating with high base + offset
             %% Momentum = min(0.9, 0.8 + 0.5) = 0.9
             %% Result = 0.9 * 1.0 + 0.1 * 0.0 = 0.9
             Result = lc_silo_behavior:asymmetric_ema_smooth(0.0, 1.0, 0.8, 0.5, 0.5),
             ?assert(abs(Result - 0.9) < 0.001)
         end},
        {"equal values treated as de-escalating",
         fun() ->
             %% NewValue = PreviousValue => false path (de-escalating)
             Result = lc_silo_behavior:asymmetric_ema_smooth(0.5, 0.5, 0.5, 0.5, 0.2),
             %% Momentum = min(0.9, 0.7) = 0.7
             %% Result = 0.7 * 0.5 + 0.3 * 0.5 = 0.5
             ?assertEqual(0.5, Result)
         end}
    ].

%%% ============================================================================
%%% Integration Tests
%%% ============================================================================

integration_test_() ->
    [
        {"stagnation detection workflow",
         fun() ->
             %% Simulate velocity computation over time
             V1 = lc_silo_behavior:compute_velocity(100.0, 1000, 90.0, 900),
             V2 = lc_silo_behavior:compute_velocity(101.0, 1100, 100.0, 1000),
             V3 = lc_silo_behavior:compute_velocity(101.5, 1200, 101.0, 1100),

             %% Average velocity
             AvgV = (V1 + V2 + V3) / 3,

             %% Compute severity with threshold of 50 fitness/1000 evals
             Severity = lc_silo_behavior:compute_stagnation_severity(AvgV, 50.0),

             %% V1 = 100, V2 = 10, V3 = 5 => AvgV ~ 38.3
             %% Severity = (50 - 38.3) / 50 ~ 0.23
             ?assert(Severity > 0.0),
             ?assert(Severity < 0.5)
         end},
        {"pressure smoothing workflow",
         fun() ->
             %% Simulate smoothing pressure signal over updates
             P0 = 0.0,
             P1 = lc_silo_behavior:asymmetric_ema_smooth(0.8, P0, 0.5, 0.5, 0.3),
             %% Escalating: fast response
             ?assert(P1 > 0.5),

             P2 = lc_silo_behavior:asymmetric_ema_smooth(0.9, P1, 0.5, 0.5, 0.3),
             %% Still escalating
             ?assert(P2 > P1),

             P3 = lc_silo_behavior:asymmetric_ema_smooth(0.5, P2, 0.5, 0.5, 0.3),
             %% De-escalating: slow response, should still be relatively high
             ?assert(P3 > 0.5),
             ?assert(P3 < P2)
         end}
    ].
