# Plan: Competitive Silo for Liquid Conglomerate

**Status:** Planning
**Created:** 2025-12-23
**Last Updated:** 2025-12-23
**Related:** PLAN_SOCIAL_SILO.md, PLAN_CULTURAL_SILO.md, PLAN_ECOLOGICAL_SILO.md, PLAN_MORPHOLOGICAL_SILO.md, PLAN_TEMPORAL_SILO.md

---

## Overview

The Competitive Silo manages adversarial dynamics in neuroevolution: opponent archives, skill ratings (Elo), arms race detection, and competitive matchmaking. Without this, populations overfit to current opponents and fail against novel strategies.

---

## 1. Motivation

### Problem Statement

Competitive neuroevolution faces unique challenges:
- **Overfitting to current opponents**: Agents specialize against specific opponents, failing against novel strategies
- **No skill measurement**: Without Elo or similar, can't assess true agent capability
- **Poor matchmaking**: Random opponents waste evaluation time on mismatches
- **Arms race cycling**: Populations cycle through strategies without progress (rock-paper-scissors dynamics)
- **Archive management**: How to maintain useful historical opponents

### Business Value

| Benefit | Impact |
|---------|--------|
| Gaming AI | NPCs that adapt to diverse player strategies |
| Security | Adversarial robustness for ML systems |
| Negotiation | Agents that can't be easily exploited |
| Red-teaming | Automatic discovery of system weaknesses |
| Benchmarking | Calibrated skill ratings for comparison |

### Training Velocity Impact

| Metric | Without Competitive Silo | With Competitive Silo |
|--------|-------------------------|----------------------|
| Strategy diversity | Low (convergent) | High (maintained) |
| Exploit vulnerability | High | Low |
| Match quality | Random | Skill-matched |
| Skill measurement | None | Calibrated Elo |

---

## 2. Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                       COMPETITIVE SILO                               │
├─────────────────────────────────────────────────────────────────────┤
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │                      L0 SENSORS (12)                         │    │
│  │                                                              │    │
│  │  Rating           Performance     Strategy     Arms Race    │    │
│  │  ┌─────────┐     ┌─────────┐     ┌─────────┐  ┌─────────┐  │    │
│  │  │elo_     │     │win_rate_│     │strategy_│  │arms_race│  │    │
│  │  │rating   │     │archive  │     │diversity│  │velocity │  │    │
│  │  │elo_     │     │win_rate_│     │exploit_ │  │cycle_   │  │    │
│  │  │variance │     │current  │     │ability  │  │strength │  │    │
│  │  │elo_trend│     │draw_rate│     └─────────┘  └─────────┘  │    │
│  │  └─────────┘     └─────────┘                                │    │
│  │                                                              │    │
│  │  Archive          Matchmaking                               │    │
│  │  ┌─────────┐     ┌─────────┐                                │    │
│  │  │archive_ │     │match_   │                                │    │
│  │  │coverage │     │quality  │                                │    │
│  │  │archive_ │     └─────────┘                                │    │
│  │  │age      │                                                │    │
│  │  └─────────┘                                                │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                                │                                     │
│                       ┌────────▼────────┐                           │
│                       │ TWEANN Controller│                          │
│                       │   (online ES)   │                           │
│                       └────────┬────────┘                           │
│                                │                                     │
│  ┌─────────────────────────────▼───────────────────────────────┐    │
│  │                      L0 ACTUATORS (10)                       │    │
│  │                                                              │    │
│  │  Archive          Matchmaking     Rewards      Diversity    │    │
│  │  ┌─────────┐     ┌─────────┐     ┌─────────┐  ┌─────────┐  │    │
│  │  │archive_ │     │elo_range│     │exploit_ │  │novelty_ │  │    │
│  │  │add_     │     │self_play│     │reward   │  │bonus    │  │    │
│  │  │threshold│     │_ratio   │     │counter_ │  │diversity│  │    │
│  │  │archive_ │     │archive_ │     │strategy_│  │_bonus   │  │    │
│  │  │max_size │     │play_    │     │reward   │  └─────────┘  │    │
│  │  └─────────┘     │ratio    │     └─────────┘               │    │
│  │                  └─────────┘                                │    │
│  └─────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 3. L0 Sensors

### 3.1 Sensor Specifications

| ID | Name | Range | Description |
|----|------|-------|-------------|
| 1 | `elo_rating` | [0.0, 1.0] | Population's average Elo (normalized 0-3000) |
| 2 | `elo_variance` | [0.0, 1.0] | Variance in population Elo ratings |
| 3 | `elo_trend` | [-1.0, 1.0] | Direction of Elo change over generations |
| 4 | `win_rate_vs_archive` | [0.0, 1.0] | Win rate against archived opponents |
| 5 | `win_rate_vs_current` | [0.0, 1.0] | Win rate against current population |
| 6 | `draw_rate` | [0.0, 1.0] | Proportion of draws (stalemate indicator) |
| 7 | `strategy_diversity_index` | [0.0, 1.0] | Diversity of strategies in population |
| 8 | `exploitability_score` | [0.0, 1.0] | How exploitable by counter-strategies |
| 9 | `arms_race_velocity` | [0.0, 1.0] | Rate of counter-adaptation |
| 10 | `cycle_strength` | [0.0, 1.0] | Rock-paper-scissors cycling intensity |
| 11 | `archive_coverage` | [0.0, 1.0] | How well archive covers strategy space |
| 12 | `match_quality` | [0.0, 1.0] | Quality of recent matchups (close games) |

### 3.2 Sensor Module

```erlang
%%%-------------------------------------------------------------------
%%% @doc Competitive Silo L0 Sensors
%%% Monitors adversarial dynamics: Elo ratings, win rates, strategy
%%% diversity, and arms race patterns.
%%% @end
%%%-------------------------------------------------------------------
-module(competitive_silo_sensors).

-behaviour(l0_sensor_behaviour).

%% API
-export([sensor_specs/0,
         collect_sensors/1,
         sensor_count/0]).

%% Sensor collection
-export([collect_elo_rating/1,
         collect_elo_variance/1,
         collect_elo_trend/1,
         collect_win_rate_vs_archive/1,
         collect_win_rate_vs_current/1,
         collect_draw_rate/1,
         collect_strategy_diversity_index/1,
         collect_exploitability_score/1,
         collect_arms_race_velocity/1,
         collect_cycle_strength/1,
         collect_archive_coverage/1,
         collect_match_quality/1]).

-include("competitive_silo.hrl").

%%====================================================================
%% Behaviour Callbacks
%%====================================================================

-spec sensor_specs() -> [l0_sensor_spec()].
sensor_specs() ->
    [
        #{id => elo_rating,
          name => <<"Elo Rating">>,
          range => {0.0, 1.0},
          description => <<"Population average Elo (normalized 0-3000)">>},

        #{id => elo_variance,
          name => <<"Elo Variance">>,
          range => {0.0, 1.0},
          description => <<"Variance in population Elo ratings">>},

        #{id => elo_trend,
          name => <<"Elo Trend">>,
          range => {-1.0, 1.0},
          description => <<"Direction of Elo change">>},

        #{id => win_rate_vs_archive,
          name => <<"Win Rate vs Archive">>,
          range => {0.0, 1.0},
          description => <<"Win rate against archived opponents">>},

        #{id => win_rate_vs_current,
          name => <<"Win Rate vs Current">>,
          range => {0.0, 1.0},
          description => <<"Win rate against current population">>},

        #{id => draw_rate,
          name => <<"Draw Rate">>,
          range => {0.0, 1.0},
          description => <<"Proportion of draws">>},

        #{id => strategy_diversity_index,
          name => <<"Strategy Diversity">>,
          range => {0.0, 1.0},
          description => <<"Diversity of strategies in population">>},

        #{id => exploitability_score,
          name => <<"Exploitability">>,
          range => {0.0, 1.0},
          description => <<"How exploitable by counter-strategies">>},

        #{id => arms_race_velocity,
          name => <<"Arms Race Velocity">>,
          range => {0.0, 1.0},
          description => <<"Rate of counter-adaptation">>},

        #{id => cycle_strength,
          name => <<"Cycle Strength">>,
          range => {0.0, 1.0},
          description => <<"Rock-paper-scissors cycling intensity">>},

        #{id => archive_coverage,
          name => <<"Archive Coverage">>,
          range => {0.0, 1.0},
          description => <<"Strategy space coverage by archive">>},

        #{id => match_quality,
          name => <<"Match Quality">>,
          range => {0.0, 1.0},
          description => <<"Quality of recent matchups">>}
    ].

-spec sensor_count() -> pos_integer().
sensor_count() -> 12.

-spec collect_sensors(competitive_context()) -> sensor_vector().
collect_sensors(Context) ->
    [
        collect_elo_rating(Context),
        collect_elo_variance(Context),
        collect_elo_trend(Context),
        collect_win_rate_vs_archive(Context),
        collect_win_rate_vs_current(Context),
        collect_draw_rate(Context),
        collect_strategy_diversity_index(Context),
        collect_exploitability_score(Context),
        collect_arms_race_velocity(Context),
        collect_cycle_strength(Context),
        collect_archive_coverage(Context),
        collect_match_quality(Context)
    ].

%%====================================================================
%% Individual Sensor Collection
%%====================================================================

%% @doc Average Elo rating normalized to [0,1]
-spec collect_elo_rating(competitive_context()) -> float().
collect_elo_rating(#competitive_context{elo_ratings = Ratings}) ->
    case Ratings of
        [] -> 0.5;
        _ ->
            AvgElo = lists:sum(Ratings) / length(Ratings),
            %% Normalize: typical Elo range 0-3000
            clamp(AvgElo / 3000.0, 0.0, 1.0)
    end.

%% @doc Variance in Elo ratings
-spec collect_elo_variance(competitive_context()) -> float().
collect_elo_variance(#competitive_context{elo_ratings = Ratings}) ->
    case length(Ratings) >= 2 of
        false -> 0.5;
        true ->
            Variance = variance(Ratings),
            %% Normalize assuming max variance ~500000 (std dev ~700)
            clamp(Variance / 500000.0, 0.0, 1.0)
    end.

%% @doc Trend in Elo ratings
-spec collect_elo_trend(competitive_context()) -> float().
collect_elo_trend(#competitive_context{elo_history = History}) ->
    case length(History) >= 3 of
        false -> 0.0;
        true ->
            Recent = lists:sublist(History, 10),
            %% History is newest first, so reverse for trend calculation
            calculate_trend(lists:reverse(Recent))
    end.

%% @doc Win rate against archived opponents
-spec collect_win_rate_vs_archive(competitive_context()) -> float().
collect_win_rate_vs_archive(#competitive_context{
    matches_vs_archive = Matches
}) ->
    calculate_win_rate(Matches).

%% @doc Win rate against current population
-spec collect_win_rate_vs_current(competitive_context()) -> float().
collect_win_rate_vs_current(#competitive_context{
    matches_vs_current = Matches
}) ->
    calculate_win_rate(Matches).

%% @doc Draw rate in recent matches
-spec collect_draw_rate(competitive_context()) -> float().
collect_draw_rate(#competitive_context{recent_matches = Matches}) ->
    case Matches of
        [] -> 0.0;
        _ ->
            Draws = length([M || M <- Matches, M#match_result.outcome =:= draw]),
            Draws / length(Matches)
    end.

%% @doc Strategy diversity using behavior fingerprints
-spec collect_strategy_diversity_index(competitive_context()) -> float().
collect_strategy_diversity_index(#competitive_context{
    strategy_fingerprints = Fingerprints
}) ->
    case length(Fingerprints) >= 2 of
        false -> 1.0;  % Single strategy is maximally diverse relative to itself
        true ->
            %% Calculate average pairwise distance
            Distances = pairwise_distances(Fingerprints),
            AvgDistance = lists:sum(Distances) / max(1, length(Distances)),
            %% Normalize assuming max distance of 1.0
            clamp(AvgDistance, 0.0, 1.0)
    end.

%% @doc Exploitability by counter-strategies
-spec collect_exploitability_score(competitive_context()) -> float().
collect_exploitability_score(#competitive_context{
    exploitability_tests = Tests
}) ->
    case Tests of
        [] -> 0.5;
        _ ->
            %% Average loss against counter-strategies
            Losses = [Loss || #{loss := Loss} <- Tests],
            clamp(lists:sum(Losses) / length(Losses), 0.0, 1.0)
    end.

%% @doc Rate of counter-adaptation (arms race velocity)
-spec collect_arms_race_velocity(competitive_context()) -> float().
collect_arms_race_velocity(#competitive_context{
    counter_adaptations = Adaptations
}) ->
    case length(Adaptations) >= 2 of
        false -> 0.0;
        true ->
            %% Measure rate of strategy dominance changes
            Recent = lists:sublist(Adaptations, 10),
            DominanceChanges = count_dominance_changes(Recent),
            clamp(DominanceChanges / length(Recent), 0.0, 1.0)
    end.

%% @doc Rock-paper-scissors cycle strength
-spec collect_cycle_strength(competitive_context()) -> float().
collect_cycle_strength(#competitive_context{
    dominance_matrix = Matrix
}) ->
    %% Analyze transitive dominance relationships
    case Matrix of
        undefined -> 0.0;
        _ -> calculate_intransitivity(Matrix)
    end.

%% @doc Archive coverage of strategy space
-spec collect_archive_coverage(competitive_context()) -> float().
collect_archive_coverage(#competitive_context{
    archive_fingerprints = ArchiveFP,
    known_strategy_space = KnownSpace
}) ->
    case {ArchiveFP, KnownSpace} of
        {[], _} -> 0.0;
        {_, []} -> 0.5;
        _ ->
            Coverage = calculate_coverage(ArchiveFP, KnownSpace),
            clamp(Coverage, 0.0, 1.0)
    end.

%% @doc Quality of recent matchups
-spec collect_match_quality(competitive_context()) -> float().
collect_match_quality(#competitive_context{recent_matches = Matches}) ->
    case Matches of
        [] -> 0.5;
        _ ->
            %% High quality = close games (Elo difference matters)
            Qualities = [match_quality_score(M) || M <- Matches],
            lists:sum(Qualities) / length(Qualities)
    end.

%%====================================================================
%% Internal Functions
%%====================================================================

calculate_win_rate(Matches) ->
    case Matches of
        [] -> 0.5;
        _ ->
            Wins = length([M || M <- Matches, M#match_result.outcome =:= win]),
            Wins / length(Matches)
    end.

calculate_trend(Values) ->
    N = length(Values),
    case N >= 2 of
        false -> 0.0;
        true ->
            Indices = lists:seq(1, N),
            MeanX = (N + 1) / 2,
            MeanY = lists:sum(Values) / N,
            Numerator = lists:sum([((I - MeanX) * (V - MeanY))
                                   || {I, V} <- lists:zip(Indices, Values)]),
            Denominator = lists:sum([math:pow(I - MeanX, 2) || I <- Indices]),
            case Denominator of
                0.0 -> 0.0;
                _ ->
                    Slope = Numerator / Denominator,
                    %% Normalize slope to [-1, 1]
                    clamp(Slope / 100.0, -1.0, 1.0)
            end
    end.

pairwise_distances([]) -> [];
pairwise_distances([_]) -> [];
pairwise_distances([H | T]) ->
    Distances = [fingerprint_distance(H, X) || X <- T],
    Distances ++ pairwise_distances(T).

fingerprint_distance(FP1, FP2) when length(FP1) =:= length(FP2) ->
    %% Euclidean distance normalized
    SumSq = lists:sum([math:pow(A - B, 2) || {A, B} <- lists:zip(FP1, FP2)]),
    math:sqrt(SumSq) / math:sqrt(length(FP1)).

count_dominance_changes([]) -> 0;
count_dominance_changes([_]) -> 0;
count_dominance_changes([A, B | Rest]) ->
    Change = case A#adaptation.dominant_strategy =/= B#adaptation.dominant_strategy of
        true -> 1;
        false -> 0
    end,
    Change + count_dominance_changes([B | Rest]).

calculate_intransitivity(Matrix) ->
    %% Detect A > B > C > A cycles
    %% Returns 0-1 where 1 = strong cycles
    Size = length(Matrix),
    case Size < 3 of
        true -> 0.0;
        false ->
            Cycles = detect_cycles(Matrix),
            clamp(Cycles / (Size * Size), 0.0, 1.0)
    end.

detect_cycles(Matrix) ->
    %% Simple cycle detection: count non-transitive triples
    Size = length(Matrix),
    Indices = lists:seq(0, Size - 1),
    Triples = [{I, J, K} || I <- Indices, J <- Indices, K <- Indices,
                           I =/= J, J =/= K, I =/= K],
    lists:foldl(fun({I, J, K}, Acc) ->
        case is_cycle(Matrix, I, J, K) of
            true -> Acc + 1;
            false -> Acc
        end
    end, 0, Triples).

is_cycle(Matrix, I, J, K) ->
    %% A > B > C > A pattern
    get_dominance(Matrix, I, J) > 0.5 andalso
    get_dominance(Matrix, J, K) > 0.5 andalso
    get_dominance(Matrix, K, I) > 0.5.

get_dominance(Matrix, I, J) ->
    Row = lists:nth(I + 1, Matrix),
    lists:nth(J + 1, Row).

calculate_coverage(ArchiveFP, KnownSpace) ->
    %% Proportion of known space covered by archive
    CoveredPoints = lists:filter(fun(Point) ->
        lists:any(fun(ArchivePoint) ->
            fingerprint_distance(Point, ArchivePoint) < 0.2
        end, ArchiveFP)
    end, KnownSpace),
    length(CoveredPoints) / max(1, length(KnownSpace)).

match_quality_score(#match_result{
    player_a_elo = EloA,
    player_b_elo = EloB,
    outcome = Outcome,
    margin = Margin
}) ->
    %% Quality based on Elo closeness and game closeness
    EloDiff = abs(EloA - EloB),
    EloQuality = 1.0 - clamp(EloDiff / 400.0, 0.0, 1.0),

    %% Close games are higher quality
    MarginQuality = case Outcome of
        draw -> 1.0;
        _ -> 1.0 - clamp(Margin, 0.0, 1.0)
    end,

    (EloQuality + MarginQuality) / 2.0.

variance([]) -> 0.0;
variance(Values) ->
    Mean = lists:sum(Values) / length(Values),
    lists:sum([math:pow(V - Mean, 2) || V <- Values]) / length(Values).

clamp(Value, Min, Max) ->
    max(Min, min(Max, Value)).
```

---

## 4. L0 Actuators

### 4.1 Actuator Specifications

| ID | Name | Range | Default | Description |
|----|------|-------|---------|-------------|
| 1 | `archive_addition_threshold` | [0.5, 0.95] | 0.7 | Win rate to qualify for archive |
| 2 | `archive_max_size` | [10, 1000] | 100 | Maximum archived opponents |
| 3 | `matchmaking_elo_range` | [50, 500] | 200 | Elo difference for fair matches |
| 4 | `self_play_ratio` | [0.0, 1.0] | 0.3 | Fraction of games vs self |
| 5 | `archive_play_ratio` | [0.0, 1.0] | 0.5 | Fraction of games vs archive |
| 6 | `exploit_reward` | [0.0, 1.0] | 0.1 | Fitness bonus for exploiting weakness |
| 7 | `counter_strategy_reward` | [0.0, 1.0] | 0.2 | Bonus for beating counters |
| 8 | `novelty_bonus` | [0.0, 0.5] | 0.1 | Bonus for novel strategies |
| 9 | `diversity_bonus` | [0.0, 0.5] | 0.1 | Bonus for diverse population |
| 10 | `anti_cycle_penalty` | [0.0, 0.3] | 0.05 | Penalty for cycling without progress |

### 4.2 Actuator Module

```erlang
%%%-------------------------------------------------------------------
%%% @doc Competitive Silo L0 Actuators
%%% Controls competitive dynamics: archive management, matchmaking,
%%% and reward structures for strategic diversity.
%%% @end
%%%-------------------------------------------------------------------
-module(competitive_silo_actuators).

-behaviour(l0_actuator_behaviour).

%% API
-export([actuator_specs/0,
         apply_actuators/2,
         actuator_count/0]).

%% Individual actuator application
-export([apply_archive_addition_threshold/2,
         apply_archive_max_size/2,
         apply_matchmaking_elo_range/2,
         apply_self_play_ratio/2,
         apply_archive_play_ratio/2,
         apply_exploit_reward/2,
         apply_counter_strategy_reward/2,
         apply_novelty_bonus/2,
         apply_diversity_bonus/2,
         apply_anti_cycle_penalty/2]).

-include("competitive_silo.hrl").

%%====================================================================
%% Behaviour Callbacks
%%====================================================================

-spec actuator_specs() -> [l0_actuator_spec()].
actuator_specs() ->
    [
        #{id => archive_addition_threshold,
          name => <<"Archive Addition Threshold">>,
          range => {0.5, 0.95},
          default => 0.7,
          description => <<"Win rate to qualify for archive">>},

        #{id => archive_max_size,
          name => <<"Archive Max Size">>,
          range => {10, 1000},
          default => 100,
          description => <<"Maximum archived opponents">>},

        #{id => matchmaking_elo_range,
          name => <<"Matchmaking Elo Range">>,
          range => {50, 500},
          default => 200,
          description => <<"Elo difference for fair matches">>},

        #{id => self_play_ratio,
          name => <<"Self-Play Ratio">>,
          range => {0.0, 1.0},
          default => 0.3,
          description => <<"Fraction of games vs self">>},

        #{id => archive_play_ratio,
          name => <<"Archive Play Ratio">>,
          range => {0.0, 1.0},
          default => 0.5,
          description => <<"Fraction of games vs archive">>},

        #{id => exploit_reward,
          name => <<"Exploit Reward">>,
          range => {0.0, 1.0},
          default => 0.1,
          description => <<"Fitness bonus for exploiting weakness">>},

        #{id => counter_strategy_reward,
          name => <<"Counter-Strategy Reward">>,
          range => {0.0, 1.0},
          default => 0.2,
          description => <<"Bonus for beating counters">>},

        #{id => novelty_bonus,
          name => <<"Novelty Bonus">>,
          range => {0.0, 0.5},
          default => 0.1,
          description => <<"Bonus for novel strategies">>},

        #{id => diversity_bonus,
          name => <<"Diversity Bonus">>,
          range => {0.0, 0.5},
          default => 0.1,
          description => <<"Bonus for population diversity">>},

        #{id => anti_cycle_penalty,
          name => <<"Anti-Cycle Penalty">>,
          range => {0.0, 0.3},
          default => 0.05,
          description => <<"Penalty for cycling without progress">>}
    ].

-spec actuator_count() -> pos_integer().
actuator_count() -> 10.

-spec apply_actuators(actuator_vector(), competitive_state()) -> competitive_state().
apply_actuators(Outputs, State) when length(Outputs) =:= 10 ->
    [ArchiveThresh, ArchiveSize, EloRange, SelfPlay, ArchivePlay,
     ExploitReward, CounterReward, NoveltyBonus, DiversityBonus, AntiCycle] = Outputs,

    State1 = apply_archive_addition_threshold(ArchiveThresh, State),
    State2 = apply_archive_max_size(ArchiveSize, State1),
    State3 = apply_matchmaking_elo_range(EloRange, State2),
    State4 = apply_self_play_ratio(SelfPlay, State3),
    State5 = apply_archive_play_ratio(ArchivePlay, State4),
    State6 = apply_exploit_reward(ExploitReward, State5),
    State7 = apply_counter_strategy_reward(CounterReward, State6),
    State8 = apply_novelty_bonus(NoveltyBonus, State7),
    State9 = apply_diversity_bonus(DiversityBonus, State8),
    apply_anti_cycle_penalty(AntiCycle, State9).

%%====================================================================
%% Individual Actuator Application
%%====================================================================

%% @doc Apply archive addition threshold
-spec apply_archive_addition_threshold(float(), competitive_state()) -> competitive_state().
apply_archive_addition_threshold(Output, State) ->
    %% Output in [0,1] -> Threshold in [0.5, 0.95]
    Threshold = 0.5 + Output * 0.45,
    State#competitive_state{archive_addition_threshold = Threshold}.

%% @doc Apply archive max size
-spec apply_archive_max_size(float(), competitive_state()) -> competitive_state().
apply_archive_max_size(Output, State) ->
    %% Output in [0,1] -> Size in [10, 1000]
    Size = round(10 + Output * 990),
    %% Prune archive if needed
    CurrentArchive = State#competitive_state.opponent_archive,
    PrunedArchive = case length(CurrentArchive) > Size of
        true -> prune_archive(CurrentArchive, Size);
        false -> CurrentArchive
    end,
    State#competitive_state{
        archive_max_size = Size,
        opponent_archive = PrunedArchive
    }.

%% @doc Apply matchmaking Elo range
-spec apply_matchmaking_elo_range(float(), competitive_state()) -> competitive_state().
apply_matchmaking_elo_range(Output, State) ->
    %% Output in [0,1] -> Range in [50, 500]
    Range = round(50 + Output * 450),
    State#competitive_state{matchmaking_elo_range = Range}.

%% @doc Apply self-play ratio
-spec apply_self_play_ratio(float(), competitive_state()) -> competitive_state().
apply_self_play_ratio(Output, State) ->
    State#competitive_state{self_play_ratio = Output}.

%% @doc Apply archive play ratio
-spec apply_archive_play_ratio(float(), competitive_state()) -> competitive_state().
apply_archive_play_ratio(Output, State) ->
    State#competitive_state{archive_play_ratio = Output}.

%% @doc Apply exploit reward
-spec apply_exploit_reward(float(), competitive_state()) -> competitive_state().
apply_exploit_reward(Output, State) ->
    State#competitive_state{exploit_reward = Output}.

%% @doc Apply counter-strategy reward
-spec apply_counter_strategy_reward(float(), competitive_state()) -> competitive_state().
apply_counter_strategy_reward(Output, State) ->
    %% Output in [0,1] -> Reward in [0.0, 1.0]
    State#competitive_state{counter_strategy_reward = Output}.

%% @doc Apply novelty bonus
-spec apply_novelty_bonus(float(), competitive_state()) -> competitive_state().
apply_novelty_bonus(Output, State) ->
    %% Output in [0,1] -> Bonus in [0.0, 0.5]
    Bonus = Output * 0.5,
    State#competitive_state{novelty_bonus = Bonus}.

%% @doc Apply diversity bonus
-spec apply_diversity_bonus(float(), competitive_state()) -> competitive_state().
apply_diversity_bonus(Output, State) ->
    %% Output in [0,1] -> Bonus in [0.0, 0.5]
    Bonus = Output * 0.5,
    State#competitive_state{diversity_bonus = Bonus}.

%% @doc Apply anti-cycle penalty
-spec apply_anti_cycle_penalty(float(), competitive_state()) -> competitive_state().
apply_anti_cycle_penalty(Output, State) ->
    %% Output in [0,1] -> Penalty in [0.0, 0.3]
    Penalty = Output * 0.3,
    State#competitive_state{anti_cycle_penalty = Penalty}.

%%====================================================================
%% Internal Functions
%%====================================================================

prune_archive(Archive, TargetSize) ->
    %% Keep most diverse and highest-rated opponents
    Sorted = lists:sort(fun(A, B) ->
        %% Score based on Elo and diversity contribution
        score_opponent(A) > score_opponent(B)
    end, Archive),
    lists:sublist(Sorted, TargetSize).

score_opponent(#archived_opponent{elo = Elo, diversity_contribution = Diversity}) ->
    %% Combine Elo and diversity
    (Elo / 3000.0) * 0.5 + Diversity * 0.5.
```

---

## 5. Record Definitions

```erlang
%%%-------------------------------------------------------------------
%%% @doc Competitive Silo Header
%%% Record definitions for adversarial dynamics management.
%%% @end
%%%-------------------------------------------------------------------

-ifndef(COMPETITIVE_SILO_HRL).
-define(COMPETITIVE_SILO_HRL, true).

%%====================================================================
%% Types
%%====================================================================

-type sensor_vector() :: [float()].
-type actuator_vector() :: [float()].
-type elo_rating() :: non_neg_integer().
-type strategy_fingerprint() :: [float()].
-type generation() :: non_neg_integer().
-type win_rate() :: float().

%%====================================================================
%% Match Result Record
%%====================================================================

-record(match_result, {
    %% Match identity
    match_id :: binary(),
    generation :: generation(),
    timestamp :: non_neg_integer(),

    %% Players
    player_a_id :: binary(),
    player_b_id :: binary(),
    player_a_elo :: elo_rating(),
    player_b_elo :: elo_rating(),

    %% Outcome
    outcome :: win | loss | draw,
    winner_id :: binary() | undefined,
    margin :: float(),  % How close the game was [0,1]

    %% Elo changes
    elo_change_a :: integer(),
    elo_change_b :: integer(),

    %% Strategy info
    player_a_strategy :: strategy_fingerprint(),
    player_b_strategy :: strategy_fingerprint()
}).

-type match_result() :: #match_result{}.

%%====================================================================
%% Archived Opponent Record
%%====================================================================

-record(archived_opponent, {
    %% Identity
    opponent_id :: binary(),
    archived_at :: non_neg_integer(),
    source_generation :: generation(),

    %% Rating
    elo :: elo_rating(),
    elo_at_archive :: elo_rating(),

    %% Strategy
    strategy_fingerprint :: strategy_fingerprint(),
    strategy_type :: atom(),  % aggressive, defensive, balanced, etc.

    %% Performance
    games_played :: non_neg_integer(),
    win_rate_vs_population :: win_rate(),

    %% Diversity
    diversity_contribution :: float(),
    unique_counters :: [binary()],  % IDs of strategies that counter this

    %% Network weights (for evaluation)
    network_weights :: binary()
}).

-type archived_opponent() :: #archived_opponent{}.

%%====================================================================
%% Adaptation Record
%%====================================================================

-record(adaptation, {
    generation :: generation(),
    dominant_strategy :: atom(),
    dominance_strength :: float(),
    counter_strategies :: [atom()]
}).

-type adaptation() :: #adaptation{}.

%%====================================================================
%% Context Record (Input to Sensors)
%%====================================================================

-record(competitive_context, {
    %% Elo ratings
    elo_ratings = [] :: [elo_rating()],
    elo_history = [] :: [{generation(), float()}],  % Avg Elo per generation

    %% Match results
    recent_matches = [] :: [match_result()],
    matches_vs_archive = [] :: [match_result()],
    matches_vs_current = [] :: [match_result()],

    %% Strategy analysis
    strategy_fingerprints = [] :: [strategy_fingerprint()],
    archive_fingerprints = [] :: [strategy_fingerprint()],
    known_strategy_space = [] :: [strategy_fingerprint()],

    %% Exploitability
    exploitability_tests = [] :: [map()],

    %% Arms race tracking
    counter_adaptations = [] :: [adaptation()],
    dominance_matrix :: [[float()]] | undefined
}).

-type competitive_context() :: #competitive_context{}.

%%====================================================================
%% State Record (Silo Internal State)
%%====================================================================

-record(competitive_state, {
    %% Configuration
    config :: competitive_config(),

    %% Opponent archive
    opponent_archive = [] :: [archived_opponent()],
    archive_max_size = 100 :: pos_integer(),
    archive_addition_threshold = 0.7 :: float(),

    %% Matchmaking
    matchmaking_elo_range = 200 :: pos_integer(),
    self_play_ratio = 0.3 :: float(),
    archive_play_ratio = 0.5 :: float(),

    %% Reward structure
    exploit_reward = 0.1 :: float(),
    counter_strategy_reward = 0.2 :: float(),
    novelty_bonus = 0.1 :: float(),
    diversity_bonus = 0.1 :: float(),
    anti_cycle_penalty = 0.05 :: float(),

    %% Tracking
    current_generation = 0 :: generation(),
    total_matches = 0 :: non_neg_integer(),
    elo_updates = [] :: [{binary(), integer()}],

    %% Dominance tracking
    dominance_matrix = [] :: [[float()]],
    cycle_detected = false :: boolean(),
    cycle_length = 0 :: non_neg_integer(),

    %% L2 integration
    l2_enabled = false :: boolean(),
    l2_guidance = undefined :: l2_guidance() | undefined
}).

-type competitive_state() :: #competitive_state{}.

%%====================================================================
%% Configuration Record
%%====================================================================

-record(competitive_config, {
    %% Enable/disable
    enabled = true :: boolean(),

    %% Elo system
    initial_elo = 1500 :: elo_rating(),
    k_factor = 32 :: pos_integer(),

    %% Archive
    min_archive_size = 10 :: pos_integer(),
    max_archive_size = 1000 :: pos_integer(),
    archive_diversity_weight = 0.5 :: float(),

    %% Matchmaking
    min_games_per_generation = 5 :: pos_integer(),

    %% Strategy fingerprinting
    fingerprint_dimensions = 16 :: pos_integer(),

    %% Event emission
    emit_events = true :: boolean()
}).

-type competitive_config() :: #competitive_config{}.

%%====================================================================
%% L2 Guidance Record
%%====================================================================

-record(l2_guidance, {
    %% Competitive pressure
    competitive_pressure = 0.5 :: float(),

    %% Archive aggressiveness
    archive_aggressiveness = 0.5 :: float(),

    %% Diversity pressure
    diversity_pressure = 0.5 :: float(),

    %% Counter-strategy incentive
    counter_incentive = 0.5 :: float()
}).

-type l2_guidance() :: #l2_guidance{}.

%%====================================================================
%% Constants
%%====================================================================

-define(DEFAULT_ELO, 1500).
-define(K_FACTOR, 32).
-define(MAX_ARCHIVE_SIZE, 1000).
-define(MIN_GAMES_FOR_ARCHIVE, 10).

-endif.
```

---

## 6. Core Silo Implementation

```erlang
%%%-------------------------------------------------------------------
%%% @doc Competitive Silo
%%% Manages adversarial dynamics for neuroevolution: opponent archives,
%%% Elo ratings, matchmaking, and strategic diversity.
%%% @end
%%%-------------------------------------------------------------------
-module(competitive_silo).

-behaviour(gen_server).

%% API
-export([start_link/1,
         get_competitive_params/1,
         update_context/2,
         record_match/2,
         get_opponent/2,
         calculate_fitness_bonus/3,
         add_to_archive/2,
         get_archive/1,
         get_state/1,
         enable/1,
         disable/1,
         is_enabled/1]).

%% Elo API
-export([get_elo/2,
         update_elo/4]).

%% Cross-silo signals
-export([signal_competitive_pressure/1,
         signal_strategy_diversity/1,
         signal_arms_race_active/1,
         receive_innovation_rate/2,
         receive_resource_abundance/2,
         receive_social_structure/2]).

%% gen_server callbacks
-export([init/1, handle_call/3, handle_cast/2, handle_info/2,
         terminate/2, code_change/3]).

-include("competitive_silo.hrl").

%%====================================================================
%% API
%%====================================================================

-spec start_link(competitive_config()) -> {ok, pid()} | {error, term()}.
start_link(Config) ->
    gen_server:start_link({local, ?MODULE}, ?MODULE, Config, []).

-spec get_competitive_params(pid()) -> map().
get_competitive_params(Pid) ->
    gen_server:call(Pid, get_competitive_params).

-spec update_context(pid(), competitive_context()) -> ok.
update_context(Pid, Context) ->
    gen_server:cast(Pid, {update_context, Context}).

-spec record_match(pid(), match_result()) -> ok.
record_match(Pid, Match) ->
    gen_server:cast(Pid, {record_match, Match}).

-spec get_opponent(pid(), binary()) -> {ok, archived_opponent() | self_play} | {error, term()}.
get_opponent(Pid, PlayerId) ->
    gen_server:call(Pid, {get_opponent, PlayerId}).

-spec calculate_fitness_bonus(pid(), binary(), match_result()) -> float().
calculate_fitness_bonus(Pid, IndividualId, MatchResult) ->
    gen_server:call(Pid, {calculate_fitness_bonus, IndividualId, MatchResult}).

-spec add_to_archive(pid(), archived_opponent()) -> ok | {error, term()}.
add_to_archive(Pid, Opponent) ->
    gen_server:call(Pid, {add_to_archive, Opponent}).

-spec get_archive(pid()) -> [archived_opponent()].
get_archive(Pid) ->
    gen_server:call(Pid, get_archive).

-spec get_state(pid()) -> competitive_state().
get_state(Pid) ->
    gen_server:call(Pid, get_state).

-spec enable(pid()) -> ok.
enable(Pid) ->
    gen_server:call(Pid, enable).

-spec disable(pid()) -> ok.
disable(Pid) ->
    gen_server:call(Pid, disable).

-spec is_enabled(pid()) -> boolean().
is_enabled(Pid) ->
    gen_server:call(Pid, is_enabled).

%%====================================================================
%% Elo API
%%====================================================================

-spec get_elo(pid(), binary()) -> elo_rating().
get_elo(Pid, IndividualId) ->
    gen_server:call(Pid, {get_elo, IndividualId}).

-spec update_elo(pid(), binary(), binary(), atom()) -> {integer(), integer()}.
update_elo(Pid, WinnerId, LoserId, Outcome) ->
    gen_server:call(Pid, {update_elo, WinnerId, LoserId, Outcome}).

%%====================================================================
%% Cross-Silo Signal API
%%====================================================================

%% @doc Get competitive pressure for other silos
-spec signal_competitive_pressure(pid()) -> float().
signal_competitive_pressure(Pid) ->
    gen_server:call(Pid, signal_competitive_pressure).

%% @doc Get strategy diversity for cultural silo
-spec signal_strategy_diversity(pid()) -> float().
signal_strategy_diversity(Pid) ->
    gen_server:call(Pid, signal_strategy_diversity).

%% @doc Get arms race status for resource silo
-spec signal_arms_race_active(pid()) -> float().
signal_arms_race_active(Pid) ->
    gen_server:call(Pid, signal_arms_race_active).

%% @doc Receive innovation rate from cultural silo
-spec receive_innovation_rate(pid(), float()) -> ok.
receive_innovation_rate(Pid, Rate) ->
    gen_server:cast(Pid, {cross_silo, innovation_rate, Rate}).

%% @doc Receive resource abundance from ecological silo
-spec receive_resource_abundance(pid(), float()) -> ok.
receive_resource_abundance(Pid, Abundance) ->
    gen_server:cast(Pid, {cross_silo, resource_abundance, Abundance}).

%% @doc Receive social structure from social silo
-spec receive_social_structure(pid(), float()) -> ok.
receive_social_structure(Pid, Structure) ->
    gen_server:cast(Pid, {cross_silo, social_structure, Structure}).

%%====================================================================
%% gen_server Callbacks
%%====================================================================

init(Config) ->
    State = #competitive_state{
        config = Config
    },
    {ok, State}.

handle_call(get_competitive_params, _From, State) ->
    Params = #{
        archive_addition_threshold => State#competitive_state.archive_addition_threshold,
        archive_max_size => State#competitive_state.archive_max_size,
        matchmaking_elo_range => State#competitive_state.matchmaking_elo_range,
        self_play_ratio => State#competitive_state.self_play_ratio,
        archive_play_ratio => State#competitive_state.archive_play_ratio,
        exploit_reward => State#competitive_state.exploit_reward,
        counter_strategy_reward => State#competitive_state.counter_strategy_reward,
        novelty_bonus => State#competitive_state.novelty_bonus,
        diversity_bonus => State#competitive_state.diversity_bonus,
        anti_cycle_penalty => State#competitive_state.anti_cycle_penalty
    },
    {reply, Params, State};

handle_call({get_opponent, PlayerId}, _From, State) ->
    #competitive_state{
        self_play_ratio = SelfPlayRatio,
        archive_play_ratio = ArchivePlayRatio,
        opponent_archive = Archive
    } = State,

    %% Decide opponent type
    Rand = rand:uniform(),
    OpponentType = if
        Rand < SelfPlayRatio -> self_play;
        Rand < SelfPlayRatio + ArchivePlayRatio, Archive =/= [] ->
            select_archive_opponent(PlayerId, Archive, State);
        true -> current_population
    end,

    {reply, {ok, OpponentType}, State};

handle_call({calculate_fitness_bonus, IndividualId, MatchResult}, _From, State) ->
    Bonus = calculate_bonus(IndividualId, MatchResult, State),
    {reply, Bonus, State};

handle_call({add_to_archive, Opponent}, _From, State) ->
    #competitive_state{
        opponent_archive = Archive,
        archive_max_size = MaxSize,
        archive_addition_threshold = Threshold
    } = State,

    %% Check if qualifies
    case Opponent#archived_opponent.win_rate_vs_population >= Threshold of
        false ->
            {reply, {error, below_threshold}, State};
        true ->
            NewArchive = [Opponent | Archive],
            PrunedArchive = prune_archive_if_needed(NewArchive, MaxSize),
            maybe_emit_event(opponent_archived, #{
                opponent_id => Opponent#archived_opponent.opponent_id,
                elo => Opponent#archived_opponent.elo,
                strategy_type => Opponent#archived_opponent.strategy_type
            }, State),
            NewState = State#competitive_state{opponent_archive = PrunedArchive},
            {reply, ok, NewState}
    end;

handle_call(get_archive, _From, State) ->
    {reply, State#competitive_state.opponent_archive, State};

handle_call(get_state, _From, State) ->
    {reply, State, State};

handle_call(enable, _From, State) ->
    Config = State#competitive_state.config,
    NewConfig = Config#competitive_config{enabled = true},
    {reply, ok, State#competitive_state{config = NewConfig}};

handle_call(disable, _From, State) ->
    Config = State#competitive_state.config,
    NewConfig = Config#competitive_config{enabled = false},
    {reply, ok, State#competitive_state{config = NewConfig}};

handle_call(is_enabled, _From, State) ->
    {reply, State#competitive_state.config#competitive_config.enabled, State};

handle_call({get_elo, IndividualId}, _From, State) ->
    Elo = get_individual_elo(IndividualId, State),
    {reply, Elo, State};

handle_call({update_elo, WinnerId, LoserId, Outcome}, _From, State) ->
    {DeltaWinner, DeltaLoser, NewState} = do_update_elo(WinnerId, LoserId, Outcome, State),
    {reply, {DeltaWinner, DeltaLoser}, NewState};

handle_call(signal_competitive_pressure, _From, State) ->
    %% Pressure based on arms race velocity and cycle strength
    Pressure = case State#competitive_state.cycle_detected of
        true -> 0.8;  % High pressure if cycling
        false ->
            %% Base pressure on archive win rate variance
            Archive = State#competitive_state.opponent_archive,
            case Archive of
                [] -> 0.5;
                _ ->
                    WinRates = [O#archived_opponent.win_rate_vs_population || O <- Archive],
                    Variance = variance(WinRates),
                    clamp(Variance * 4, 0.0, 1.0)
            end
    end,
    {reply, Pressure, State};

handle_call(signal_strategy_diversity, _From, State) ->
    %% Based on archive diversity
    Archive = State#competitive_state.opponent_archive,
    case Archive of
        [] -> {reply, 1.0, State};
        _ ->
            Fingerprints = [O#archived_opponent.strategy_fingerprint || O <- Archive],
            Diversity = calculate_fingerprint_diversity(Fingerprints),
            {reply, Diversity, State}
    end;

handle_call(signal_arms_race_active, _From, State) ->
    Active = case State#competitive_state.cycle_detected of
        true -> 0.9;
        false -> 0.3
    end,
    {reply, Active, State};

handle_call(_Request, _From, State) ->
    {reply, {error, unknown_request}, State}.

handle_cast({update_context, Context}, State) ->
    #competitive_state{config = Config} = State,
    case Config#competitive_config.enabled of
        false ->
            {noreply, State};
        true ->
            %% Collect sensors
            SensorVector = competitive_silo_sensors:collect_sensors(Context),

            %% Process through TWEANN controller
            ActuatorVector = process_through_controller(SensorVector, State),

            %% Apply actuators
            NewState = competitive_silo_actuators:apply_actuators(ActuatorVector, State),

            %% Update cycle detection
            FinalState = detect_cycles(Context, NewState),

            {noreply, FinalState}
    end;

handle_cast({record_match, Match}, State) ->
    %% Update Elo ratings
    NewState = record_match_result(Match, State),

    %% Emit event
    maybe_emit_event(match_completed, #{
        player_a => Match#match_result.player_a_id,
        player_b => Match#match_result.player_b_id,
        outcome => Match#match_result.outcome,
        elo_change_a => Match#match_result.elo_change_a,
        elo_change_b => Match#match_result.elo_change_b
    }, State),

    %% Check for counter-strategy emergence
    check_counter_strategy(Match, NewState),

    {noreply, NewState#competitive_state{
        total_matches = State#competitive_state.total_matches + 1
    }};

handle_cast({cross_silo, innovation_rate, Rate}, State) ->
    %% High innovation -> add to archive faster
    CurrentThreshold = State#competitive_state.archive_addition_threshold,
    Adjustment = (1.0 - Rate) * 0.1,
    NewThreshold = clamp(CurrentThreshold - Adjustment, 0.5, 0.95),
    {noreply, State#competitive_state{archive_addition_threshold = NewThreshold}};

handle_cast({cross_silo, resource_abundance, Abundance}, State) ->
    %% Abundance allows larger archive
    CurrentMax = State#competitive_state.archive_max_size,
    #competitive_state{config = Config} = State,
    MaxAllowed = Config#competitive_config.max_archive_size,
    MinAllowed = Config#competitive_config.min_archive_size,

    NewMax = round(MinAllowed + Abundance * (MaxAllowed - MinAllowed)),
    NewState = case NewMax < CurrentMax of
        true ->
            Archive = State#competitive_state.opponent_archive,
            PrunedArchive = prune_archive_if_needed(Archive, NewMax),
            State#competitive_state{
                archive_max_size = NewMax,
                opponent_archive = PrunedArchive
            };
        false ->
            State#competitive_state{archive_max_size = NewMax}
    end,
    {noreply, NewState};

handle_cast({cross_silo, social_structure, _Structure}, State) ->
    %% Coalitions may compete as groups - future enhancement
    {noreply, State};

handle_cast(_Msg, State) ->
    {noreply, State}.

handle_info(_Info, State) ->
    {noreply, State}.

terminate(_Reason, _State) ->
    ok.

code_change(_OldVsn, State, _Extra) ->
    {ok, State}.

%%====================================================================
%% Internal Functions
%%====================================================================

get_individual_elo(_IndividualId, State) ->
    %% TODO: Lookup from elo_updates or return default
    State#competitive_state.config#competitive_config.initial_elo.

do_update_elo(WinnerId, LoserId, Outcome, State) ->
    #competitive_state{config = Config} = State,
    K = Config#competitive_config.k_factor,

    WinnerElo = get_individual_elo(WinnerId, State),
    LoserElo = get_individual_elo(LoserId, State),

    %% Expected scores
    ExpectedWinner = 1.0 / (1.0 + math:pow(10, (LoserElo - WinnerElo) / 400.0)),
    ExpectedLoser = 1.0 - ExpectedWinner,

    %% Actual scores
    {ActualWinner, ActualLoser} = case Outcome of
        win -> {1.0, 0.0};
        draw -> {0.5, 0.5};
        loss -> {0.0, 1.0}
    end,

    %% Elo changes
    DeltaWinner = round(K * (ActualWinner - ExpectedWinner)),
    DeltaLoser = round(K * (ActualLoser - ExpectedLoser)),

    %% Update state
    Updates = [{WinnerId, DeltaWinner}, {LoserId, DeltaLoser} |
               State#competitive_state.elo_updates],
    NewState = State#competitive_state{elo_updates = Updates},

    maybe_emit_event(elo_updated, #{
        winner_id => WinnerId,
        loser_id => LoserId,
        winner_delta => DeltaWinner,
        loser_delta => DeltaLoser
    }, State),

    {DeltaWinner, DeltaLoser, NewState}.

select_archive_opponent(PlayerId, Archive, State) ->
    %% Select opponent within Elo range
    PlayerElo = get_individual_elo(PlayerId, State),
    EloRange = State#competitive_state.matchmaking_elo_range,

    ValidOpponents = [O || O <- Archive,
                      abs(O#archived_opponent.elo - PlayerElo) =< EloRange],

    case ValidOpponents of
        [] ->
            %% No opponents in range, pick closest
            Sorted = lists:sort(fun(A, B) ->
                abs(A#archived_opponent.elo - PlayerElo) <
                abs(B#archived_opponent.elo - PlayerElo)
            end, Archive),
            hd(Sorted);
        _ ->
            %% Random from valid
            lists:nth(rand:uniform(length(ValidOpponents)), ValidOpponents)
    end.

calculate_bonus(IndividualId, MatchResult, State) ->
    #competitive_state{
        exploit_reward = ExploitReward,
        counter_strategy_reward = CounterReward,
        novelty_bonus = NoveltyBonus,
        diversity_bonus = DiversityBonus,
        anti_cycle_penalty = AntiCyclePenalty
    } = State,

    %% Base bonus from winning
    WinBonus = case MatchResult#match_result.winner_id of
        IndividualId -> 0.1;
        _ -> 0.0
    end,

    %% Exploit bonus (beating weaker opponent by large margin)
    ExploitBonus = case MatchResult#match_result.margin > 0.7 of
        true -> ExploitReward;
        false -> 0.0
    end,

    %% Counter-strategy bonus (beating an opponent that previously dominated)
    CounterBonus = check_counter_bonus(IndividualId, MatchResult, State) * CounterReward,

    %% Novelty bonus (using new strategy)
    Novelty = check_novelty(IndividualId, State) * NoveltyBonus,

    %% Diversity contribution
    Diversity = check_diversity_contribution(IndividualId, State) * DiversityBonus,

    %% Cycle penalty
    CyclePenalty = case State#competitive_state.cycle_detected of
        true -> AntiCyclePenalty;
        false -> 0.0
    end,

    WinBonus + ExploitBonus + CounterBonus + Novelty + Diversity - CyclePenalty.

check_counter_bonus(_IndividualId, _MatchResult, _State) ->
    %% TODO: Check if this is a counter-strategy win
    0.0.

check_novelty(_IndividualId, _State) ->
    %% TODO: Check strategy novelty
    0.0.

check_diversity_contribution(_IndividualId, _State) ->
    %% TODO: Calculate diversity contribution
    0.5.

record_match_result(Match, State) ->
    %% Update Elo based on match result
    {_, _, NewState} = case Match#match_result.outcome of
        win ->
            do_update_elo(Match#match_result.player_a_id,
                         Match#match_result.player_b_id, win, State);
        loss ->
            do_update_elo(Match#match_result.player_b_id,
                         Match#match_result.player_a_id, win, State);
        draw ->
            do_update_elo(Match#match_result.player_a_id,
                         Match#match_result.player_b_id, draw, State)
    end,
    NewState.

check_counter_strategy(Match, State) ->
    %% Check if winner used a counter-strategy
    #match_result{
        winner_id = WinnerId,
        player_a_strategy = StratA,
        player_b_strategy = StratB
    } = Match,

    WinnerStrategy = case WinnerId of
        undefined -> undefined;
        _ when WinnerId =:= Match#match_result.player_a_id -> StratA;
        _ -> StratB
    end,

    case WinnerStrategy of
        undefined -> ok;
        _ ->
            %% TODO: Compare with historical dominant strategies
            maybe_emit_event(counter_strategy_emerged, #{
                winner_id => WinnerId,
                strategy => WinnerStrategy
            }, State)
    end.

detect_cycles(Context, State) ->
    case Context#competitive_context.dominance_matrix of
        undefined -> State;
        Matrix ->
            CycleStrength = calculate_intransitivity(Matrix),
            CycleDetected = CycleStrength > 0.3,
            State#competitive_state{
                cycle_detected = CycleDetected,
                dominance_matrix = Matrix
            }
    end.

calculate_intransitivity(_Matrix) ->
    %% TODO: Implement cycle detection
    0.0.

prune_archive_if_needed(Archive, MaxSize) when length(Archive) =< MaxSize ->
    Archive;
prune_archive_if_needed(Archive, MaxSize) ->
    %% Sort by score and keep top
    Sorted = lists:sort(fun(A, B) ->
        score_opponent(A) > score_opponent(B)
    end, Archive),
    Pruned = lists:sublist(Sorted, MaxSize),

    %% Emit event for retired opponents
    Retired = Archive -- Pruned,
    [maybe_emit_event_simple(opponent_retired, #{
        opponent_id => O#archived_opponent.opponent_id,
        reason => pruned
    }) || O <- Retired],

    Pruned.

score_opponent(#archived_opponent{elo = Elo, diversity_contribution = Diversity}) ->
    (Elo / 3000.0) * 0.5 + Diversity * 0.5.

calculate_fingerprint_diversity(Fingerprints) ->
    case length(Fingerprints) >= 2 of
        false -> 1.0;
        true ->
            Distances = pairwise_distances(Fingerprints),
            lists:sum(Distances) / max(1, length(Distances))
    end.

pairwise_distances([]) -> [];
pairwise_distances([_]) -> [];
pairwise_distances([H | T]) ->
    [fingerprint_distance(H, X) || X <- T] ++ pairwise_distances(T).

fingerprint_distance(FP1, FP2) when length(FP1) =:= length(FP2) ->
    SumSq = lists:sum([math:pow(A - B, 2) || {A, B} <- lists:zip(FP1, FP2)]),
    math:sqrt(SumSq) / math:sqrt(length(FP1));
fingerprint_distance(_, _) ->
    1.0.

process_through_controller(SensorVector, State) ->
    case State#competitive_state.l2_enabled of
        true ->
            apply_l2_guidance(SensorVector, State);
        false ->
            lists:duplicate(competitive_silo_actuators:actuator_count(), 0.5)
    end.

apply_l2_guidance(_SensorVector, State) ->
    case State#competitive_state.l2_guidance of
        undefined ->
            lists:duplicate(competitive_silo_actuators:actuator_count(), 0.5);
        #l2_guidance{
            competitive_pressure = Pressure,
            archive_aggressiveness = ArchAgg,
            diversity_pressure = DivPress,
            counter_incentive = CounterInc
        } ->
            [
                1.0 - ArchAgg,    % archive_addition_threshold (lower = more additions)
                ArchAgg,          % archive_max_size
                0.5,              % matchmaking_elo_range
                0.3,              % self_play_ratio
                0.5,              % archive_play_ratio
                Pressure,         % exploit_reward
                CounterInc,       % counter_strategy_reward
                DivPress,         % novelty_bonus
                DivPress,         % diversity_bonus
                Pressure * 0.5    % anti_cycle_penalty
            ]
    end.

maybe_emit_event(EventType, Payload, State) ->
    case State#competitive_state.config#competitive_config.emit_events of
        true ->
            Event = #{
                type => EventType,
                silo => competitive,
                timestamp => erlang:system_time(millisecond),
                generation => State#competitive_state.current_generation,
                payload => Payload
            },
            event_bus:publish(competitive_silo_events, Event);
        false ->
            ok
    end.

maybe_emit_event_simple(EventType, Payload) ->
    Event = #{
        type => EventType,
        silo => competitive,
        timestamp => erlang:system_time(millisecond),
        payload => Payload
    },
    event_bus:publish(competitive_silo_events, Event).

variance([]) -> 0.0;
variance(Values) ->
    Mean = lists:sum(Values) / length(Values),
    lists:sum([math:pow(V - Mean, 2) || V <- Values]) / length(Values).

clamp(Value, Min, Max) ->
    max(Min, min(Max, Value)).
```

---

## 7. Cross-Silo Signal Matrix

### 7.1 Outgoing Signals

| Signal | To Silo | Type | Description |
|--------|---------|------|-------------|
| `competitive_pressure` | Task | float() | High pressure suggests need for faster adaptation |
| `strategy_diversity` | Cultural | float() | Low diversity encourages cultural innovation |
| `arms_race_active` | Resource | float() | Arms race needs more compute resources |
| `strategy_diversity` | Distribution | float() | Affects population structuring |
| `hidden_capabilities` | Regulatory | float() | Strategies with dormant capabilities |

### 7.2 Incoming Signals

| Signal | From Silo | Effect |
|--------|-----------|--------|
| `innovation_rate` | Cultural | High innovation adds to archive faster |
| `resource_abundance` | Ecological | Abundance allows larger archive |
| `social_structure` | Social | Coalitions may compete as groups |
| `population_diversity` | Distribution | Affects matchmaking strategies |
| `resource_pressure` | Resource | Pressure may limit archive size |

### 7.3 Complete 13-Silo Matrix (Competitive Row)

| To → | Task | Resource | Distrib | Social | Cultural | Ecological | Morpho | Temporal | Competitive | Develop | Regulatory | Economic | Comm |
|------|------|----------|---------|--------|----------|------------|--------|----------|-------------|---------|------------|----------|------|
| **From Competitive** | pressure | arms_race | diversity | coalitions | strategy | scarcity | - | eval_time | - | - | hidden | - | signals |

---

## 8. Events Emitted

### 8.1 Event Specifications

| Event | Trigger | Payload |
|-------|---------|---------|
| `opponent_archived` | Added to archive | `{opponent_id, elo, strategy_signature}` |
| `opponent_retired` | Removed from archive | `{opponent_id, reason, games_played}` |
| `elo_updated` | Rating changed | `{individual_id, old_elo, new_elo, opponent_elo}` |
| `match_completed` | Game finished | `{player_a, player_b, result, elo_changes}` |
| `counter_strategy_emerged` | Beat previous champion | `{counter_id, target_id, win_margin}` |
| `arms_race_detected` | Cycling strategies | `{cycle_participants, cycle_length}` |
| `exploitable_strategy_found` | Weakness discovered | `{strategy_id, exploit_id, win_rate}` |
| `nash_equilibrium_approached` | Stable strategy mix | `{strategy_distribution, exploitability}` |
| `archive_threshold_changed` | Threshold adjusted | `{old_threshold, new_threshold, reason}` |
| `matchmaking_adjusted` | Elo range changed | `{old_range, new_range, match_quality}` |

### 8.2 Event Payload Types

```erlang
%% Event type specifications
-type competitive_event() ::
    opponent_archived_event() |
    opponent_retired_event() |
    elo_updated_event() |
    match_completed_event() |
    counter_strategy_emerged_event() |
    arms_race_detected_event() |
    exploitable_strategy_found_event() |
    nash_equilibrium_approached_event().

-type opponent_archived_event() :: #{
    type := opponent_archived,
    silo := competitive,
    timestamp := non_neg_integer(),
    generation := generation(),
    payload := #{
        opponent_id := binary(),
        elo := elo_rating(),
        strategy_type := atom(),
        diversity_contribution := float()
    }
}.

-type match_completed_event() :: #{
    type := match_completed,
    silo := competitive,
    timestamp := non_neg_integer(),
    generation := generation(),
    payload := #{
        player_a := binary(),
        player_b := binary(),
        outcome := win | loss | draw,
        margin := float(),
        elo_change_a := integer(),
        elo_change_b := integer()
    }
}.

-type arms_race_detected_event() :: #{
    type := arms_race_detected,
    silo := competitive,
    timestamp := non_neg_integer(),
    generation := generation(),
    payload := #{
        cycle_participants := [binary()],
        cycle_length := pos_integer(),
        cycle_strength := float(),
        strategies_involved := [atom()]
    }
}.
```

---

## 9. Value of Event Storage

### 9.1 Analysis Capabilities

| Stored Events | Analysis Enabled |
|---------------|------------------|
| `match_completed` | Win rate analysis, matchmaking optimization |
| `elo_updated` | Skill progression tracking, difficulty calibration |
| `opponent_archived` | Strategy evolution timeline |
| `counter_strategy_emerged` | Arms race dynamics, meta-game evolution |
| `arms_race_detected` | Identify productive vs cycling arms races |

### 9.2 Business Intelligence

- **Strategy Evolution**: Track how strategies counter each other over time
- **Elo Calibration**: Understand skill progression for difficulty scaling
- **Counter Discovery**: Replay to find counters to specific strategies
- **Arms Race Analysis**: Identify productive vs cycling arms races
- **Deployment Selection**: Choose robust strategies from historical performance
- **Meta-game Tracking**: Monitor strategy distribution over time

### 9.3 Replay Scenarios

| Scenario | Value |
|----------|-------|
| "Find all strategies that beat aggressive play" | Counter-strategy development |
| "Show Elo progression for champion agents" | Training curriculum design |
| "Identify strategies never countered" | Find robust strategies |
| "Trace arms race evolution over 100 generations" | Meta-game analysis |

---

## 10. Multi-Layer Hierarchy

### 10.1 Layer Responsibilities

| Level | Role | Controls |
|-------|------|----------|
| **L0** | Hard limits | Max archive size, minimum match quality, Elo bounds |
| **L1** | Tactical | Adjust matchmaking based on recent performance |
| **L2** | Strategic | Learn optimal competitive pressure for convergence |

### 10.2 L2 Integration

```erlang
%% L2 guidance for competitive silo
-record(competitive_l2_guidance, {
    %% How aggressive competition should be
    competitive_pressure = 0.5 :: float(),  % [0.0, 1.0]

    %% Archive management
    archive_aggressiveness = 0.5 :: float(),  % [0.0, 1.0]

    %% Diversity incentives
    diversity_pressure = 0.5 :: float(),  % [0.0, 1.0]

    %% Counter-strategy rewards
    counter_incentive = 0.5 :: float()  % [0.0, 1.0]
}).

%% L2 queries competitive silo state and provides guidance
-spec get_l2_guidance(competitive_state(), l2_context()) -> competitive_l2_guidance().
get_l2_guidance(State, L2Context) ->
    #competitive_l2_guidance{
        competitive_pressure = compute_optimal_pressure(State, L2Context),
        archive_aggressiveness = compute_archive_policy(State, L2Context),
        diversity_pressure = compute_diversity_need(State, L2Context),
        counter_incentive = compute_counter_incentive(State, L2Context)
    }.
```

---

## 11. Enable/Disable Effects

### 11.1 When Disabled

| Aspect | Behavior |
|--------|----------|
| Opponent archive | No archive maintained |
| Elo tracking | No skill ratings |
| Matchmaking | Random opponents |
| Counter-strategy rewards | No incentive for counters |
| Diversity bonus | No diversity pressure |
| Arms race detection | No cycle detection |

### 11.2 When Enabled

| Aspect | Behavior |
|--------|----------|
| Opponent archive | Diverse historical opponents maintained |
| Elo tracking | Calibrated skill ratings |
| Matchmaking | Skill-matched opponents |
| Counter-strategy rewards | Innovation rewarded |
| Diversity bonus | Population diversity maintained |
| Arms race detection | Cycling detected and penalized |

### 11.3 Switching Effects

**Enabling mid-run:**
- Archive starts building from current population
- Elo ratings initialized at default
- Matchmaking begins skill-based selection
- Historical events not available

**Disabling mid-run:**
- Archive preserved but not updated
- Elo ratings frozen
- Random matchmaking resumes
- Competitive bonuses removed from fitness

---

## 12. Implementation Phases

### Phase 1: Core Infrastructure
- [ ] Create `competitive_silo.hrl` with record definitions
- [ ] Implement `competitive_silo_sensors.erl` with all 12 sensors
- [ ] Implement `competitive_silo_actuators.erl` with all 10 actuators
- [ ] Basic `competitive_silo.erl` gen_server

### Phase 2: Elo System
- [ ] Implement Elo rating calculation
- [ ] Elo update on match completion
- [ ] Elo-based matchmaking
- [ ] Elo history tracking

### Phase 3: Archive Management
- [ ] Opponent archiving logic
- [ ] Archive pruning strategies
- [ ] Strategy fingerprinting
- [ ] Diversity contribution calculation

### Phase 4: Competitive Dynamics
- [ ] Counter-strategy detection
- [ ] Arms race / cycle detection
- [ ] Fitness bonus calculation
- [ ] Exploitability testing

### Phase 5: Cross-Silo Integration
- [ ] Outgoing signal implementations
- [ ] Incoming signal handlers
- [ ] Event emission to event store
- [ ] Integration with cultural and resource silos

### Phase 6: Testing & Tuning
- [ ] Unit tests for Elo system
- [ ] Unit tests for archive management
- [ ] Integration tests with adversarial tasks
- [ ] Tune K-factor and thresholds

---

## 13. Success Criteria

1. **Strategy Diversity**: Maintain > 5 distinct strategies in archive
2. **Elo Accuracy**: Elo predicts match outcomes with > 70% accuracy
3. **Matchmaking Quality**: > 60% of matches within Elo range
4. **Counter Detection**: Detect counter-strategies within 5 generations
5. **Cycle Prevention**: Reduce unproductive cycling by > 50%
6. **Archive Coverage**: Archive covers > 70% of known strategy space
7. **Cross-Silo Integration**: All signals documented and functional

---

## Appendix A: Elo System Details

### A.1 Standard Elo Calculation

```erlang
%% Expected score based on Elo difference
expected_score(EloA, EloB) ->
    1.0 / (1.0 + math:pow(10, (EloB - EloA) / 400.0)).

%% New Elo after match
new_elo(OldElo, Expected, Actual, K) ->
    round(OldElo + K * (Actual - Expected)).

%% K-factor selection (can be adaptive)
select_k_factor(GamesPlayed, Elo) when GamesPlayed < 30 ->
    40;  % New players
select_k_factor(_GamesPlayed, Elo) when Elo >= 2400 ->
    10;  % Masters
select_k_factor(_, _) ->
    20.  % Standard
```

### A.2 Strategy Fingerprinting

```erlang
%% Create fingerprint from agent behavior
create_fingerprint(Agent, TestScenarios) ->
    Behaviors = [observe_behavior(Agent, Scenario) || Scenario <- TestScenarios],
    normalize_fingerprint(Behaviors).

observe_behavior(Agent, Scenario) ->
    %% Run agent through scenario, collect metrics
    #{
        aggression => measure_aggression(Agent, Scenario),
        defense => measure_defense(Agent, Scenario),
        exploration => measure_exploration(Agent, Scenario),
        timing => measure_timing(Agent, Scenario)
    }.

normalize_fingerprint(Behaviors) ->
    %% Flatten to vector and normalize
    Values = lists:flatmap(fun(B) ->
        [maps:get(aggression, B), maps:get(defense, B),
         maps:get(exploration, B), maps:get(timing, B)]
    end, Behaviors),
    normalize_vector(Values).
```

---

## Appendix B: Configuration Examples

### B.1 Tournament Mode Config

```erlang
#competitive_config{
    enabled = true,
    initial_elo = 1500,
    k_factor = 32,
    min_archive_size = 50,
    max_archive_size = 500,
    archive_diversity_weight = 0.3,
    min_games_per_generation = 20,
    fingerprint_dimensions = 32,
    emit_events = true
}.
```

### B.2 Exploration Mode Config

```erlang
#competitive_config{
    enabled = true,
    initial_elo = 1500,
    k_factor = 16,  % Slower Elo changes
    min_archive_size = 100,
    max_archive_size = 1000,
    archive_diversity_weight = 0.7,  % Prioritize diversity
    min_games_per_generation = 5,
    fingerprint_dimensions = 64,
    emit_events = true
}.
```

### B.3 Rapid Development Config

```erlang
#competitive_config{
    enabled = true,
    initial_elo = 1500,
    k_factor = 64,  % Fast Elo updates
    min_archive_size = 10,
    max_archive_size = 50,
    archive_diversity_weight = 0.5,
    min_games_per_generation = 3,
    fingerprint_dimensions = 8,
    emit_events = true
}.
```
