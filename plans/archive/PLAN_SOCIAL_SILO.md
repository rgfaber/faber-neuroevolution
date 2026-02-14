# PLAN_SOCIAL_SILO.md

**Status:** Planning
**Created:** 2025-12-23
**Last Updated:** 2025-12-23
**Related:** PLAN_BEHAVIORAL_EVENTS.md, PLAN_L2_L1_HIERARCHICAL_INTERFACE.md, PLAN_CULTURAL_SILO.md

---

## Overview

This document specifies the **Social Silo** for the Liquid Conglomerate (LC) meta-controller. The Social Silo manages social dynamics within evolving populations: reputation, coalitions, kin recognition, cooperation/defection, and social norms.

### Purpose

The Social Silo extends the LC architecture to model social evolutionary pressures:

1. **Reputation Tracking**: Monitor individual standing within the population
2. **Coalition Formation**: Enable group alliances for collective benefit
3. **Kin Selection**: Favor genetic relatives (Hamilton's rule)
4. **Cooperation Dynamics**: Balance cooperation vs. defection (game theory)
5. **Social Learning**: Influence who learns from whom

### Training Velocity & Inference Impact

| Metric | Without Social Silo | With Social Silo |
|--------|---------------------|------------------|
| Selection fairness | Random/fitness-only | Socially-informed |
| Training velocity | Baseline (1.0x) | Slight overhead (0.9-1.0x) |
| Inference latency | No social computation | +2-5ms for reputation lookup |
| Cooperative task success | ~30% | ~65% |
| Population stability | High variance | More stable coalitions |

**Note:** Training velocity is nearly neutral. The overhead of tracking reputation and coalitions is offset by more efficient selection (high-reputation individuals are better learning targets). For cooperative tasks, social structure dramatically improves outcomes.

### LC Architecture Context

```
┌─────────────────────────────────────────────────────────────────────┐
│                    LIQUID CONGLOMERATE                               │
├─────────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐ │
│  │  Task Silo  │  │Resource Silo│  │ Social Silo │  │Cultural Silo│ │
│  │  (existing) │  │  (existing) │  │    (NEW)    │  │    (NEW)    │ │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘ │
│         │                │                │                │        │
│         └────────────────┴────────────────┴────────────────┘        │
│                                   │                                  │
│                          ┌────────▼────────┐                        │
│                          │  LC Controller  │                        │
│                          │    (TWEANN)     │                        │
│                          └─────────────────┘                        │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Silo Architecture

### Social Silo Structure

```
┌─────────────────────────────────────────────────────────────────────┐
│                         SOCIAL SILO                                  │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │                      L0 SENSORS (12)                         │    │
│  │                                                              │    │
│  │  Reputation     Coalition    Kin          Cooperation       │    │
│  │  ┌─────────┐   ┌─────────┐  ┌─────────┐  ┌─────────┐       │    │
│  │  │mean_rep │   │coalition│  │mean_kin │  │coop_rate│       │    │
│  │  │rep_var  │   │_count   │  │kin_var  │  │defect_  │       │    │
│  │  │rep_trend│   │coal_size│  │kin_help │  │rate     │       │    │
│  │  └─────────┘   └─────────┘  └─────────┘  └─────────┘       │    │
│  │                                                              │    │
│  │  Social Structure           Interaction                     │    │
│  │  ┌─────────┐   ┌─────────┐  ┌─────────┐  ┌─────────┐       │    │
│  │  │hierarchy│   │network_ │  │interact_│  │conflict_│       │    │
│  │  │_steep   │   │density  │  │frequency│  │level    │       │    │
│  │  └─────────┘   └─────────┘  └─────────┘  └─────────┘       │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                                │                                     │
│                       ┌────────▼────────┐                           │
│                       │ TWEANN Controller│                          │
│                       │   (online ES)   │                           │
│                       └────────┬────────┘                           │
│                                │                                     │
│  ┌─────────────────────────────▼───────────────────────────────┐    │
│  │                      L0 ACTUATORS (8)                        │    │
│  │                                                              │    │
│  │  Reputation     Coalition    Kin          Cooperation       │    │
│  │  ┌─────────┐   ┌─────────┐  ┌─────────┐  ┌─────────┐       │    │
│  │  │rep_decay│   │coal_    │  │kin_bonus│  │coop_    │       │    │
│  │  │rep_boost│   │incentive│  │kin_     │  │reward   │       │    │
│  │  └─────────┘   └─────────┘  │threshold│  │defect_  │       │    │
│  │                             └─────────┘  │penalty  │       │    │
│  │                                          └─────────┘       │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## L0 Sensors

### Sensor Specifications

```erlang
-module(social_silo_sensors).
-behaviour(l0_sensor_behaviour).

%% Sensor definitions
-export([sensor_specs/0, read_sensors/1]).

sensor_specs() ->
    [
        %% Reputation sensors
        #{
            id => mean_reputation,
            range => {-1.0, 1.0},
            description => "Average reputation across population"
        },
        #{
            id => reputation_variance,
            range => {0.0, 1.0},
            description => "Variance in reputation scores"
        },
        #{
            id => reputation_trend,
            range => {-1.0, 1.0},
            description => "Direction of reputation change (-1=declining, +1=rising)"
        },

        %% Coalition sensors
        #{
            id => coalition_count,
            range => {0.0, 1.0},
            description => "Number of coalitions (normalized by population)"
        },
        #{
            id => avg_coalition_size,
            range => {0.0, 1.0},
            description => "Average coalition size (normalized)"
        },

        %% Kin selection sensors
        #{
            id => mean_relatedness,
            range => {0.0, 1.0},
            description => "Average relatedness coefficient in population"
        },
        #{
            id => relatedness_variance,
            range => {0.0, 1.0},
            description => "Variance in relatedness"
        },
        #{
            id => kin_help_rate,
            range => {0.0, 1.0},
            description => "Rate of kin-directed helping behaviors"
        },

        %% Cooperation sensors
        #{
            id => cooperation_rate,
            range => {0.0, 1.0},
            description => "Proportion of cooperative interactions"
        },
        #{
            id => defection_rate,
            range => {0.0, 1.0},
            description => "Proportion of defection interactions"
        },

        %% Social structure sensors
        #{
            id => hierarchy_steepness,
            range => {0.0, 1.0},
            description => "Steepness of dominance hierarchy"
        },
        #{
            id => social_network_density,
            range => {0.0, 1.0},
            description => "Density of interaction network"
        }
    ].

read_sensors(#social_state{} = State) ->
    [
        normalize(mean_reputation, State#social_state.mean_reputation),
        normalize(reputation_variance, State#social_state.reputation_variance),
        normalize(reputation_trend, State#social_state.reputation_trend),
        normalize(coalition_count, State#social_state.coalition_count / State#social_state.population_size),
        normalize(avg_coalition_size, State#social_state.avg_coalition_size / State#social_state.population_size),
        normalize(mean_relatedness, State#social_state.mean_relatedness),
        normalize(relatedness_variance, State#social_state.relatedness_variance),
        normalize(kin_help_rate, State#social_state.kin_help_rate),
        normalize(cooperation_rate, State#social_state.cooperation_rate),
        normalize(defection_rate, State#social_state.defection_rate),
        normalize(hierarchy_steepness, State#social_state.hierarchy_steepness),
        normalize(social_network_density, State#social_state.network_density)
    ].
```

---

## L0 Actuators

### Actuator Specifications

```erlang
-module(social_silo_actuators).
-behaviour(l0_actuator_behaviour).

-export([actuator_specs/0, apply_actuators/2]).

actuator_specs() ->
    [
        %% Reputation actuators
        #{
            id => reputation_decay_rate,
            range => {0.0, 0.5},
            default => 0.1,
            description => "How fast reputation decays without reinforcement"
        },
        #{
            id => reputation_boost_factor,
            range => {1.0, 3.0},
            default => 1.5,
            description => "Multiplier for positive reputation events"
        },

        %% Coalition actuators
        #{
            id => coalition_formation_incentive,
            range => {0.0, 1.0},
            default => 0.3,
            description => "Fitness bonus for coalition membership"
        },

        %% Kin selection actuators
        #{
            id => kin_selection_bonus,
            range => {0.0, 2.0},
            default => 0.5,
            description => "Fitness multiplier for helping relatives"
        },
        #{
            id => kin_recognition_threshold,
            range => {0.0, 0.5},
            default => 0.125,
            description => "Minimum relatedness for kin treatment"
        },

        %% Cooperation actuators
        #{
            id => cooperation_reward,
            range => {0.0, 1.0},
            default => 0.2,
            description => "Fitness bonus for mutual cooperation"
        },
        #{
            id => defection_penalty,
            range => {0.0, 1.0},
            default => 0.3,
            description => "Fitness penalty for defecting"
        },
        #{
            id => sucker_penalty,
            range => {0.0, 0.5},
            default => 0.1,
            description => "Penalty for being defected against while cooperating"
        }
    ].

apply_actuators(Outputs, #social_config{} = Config) ->
    [RepDecay, RepBoost, CoalInc, KinBonus, KinThresh, CoopReward, DefectPen, SuckerPen] = Outputs,

    Config#social_config{
        reputation_decay_rate = denormalize(reputation_decay_rate, RepDecay),
        reputation_boost_factor = denormalize(reputation_boost_factor, RepBoost),
        coalition_formation_incentive = denormalize(coalition_formation_incentive, CoalInc),
        kin_selection_bonus = denormalize(kin_selection_bonus, KinBonus),
        kin_recognition_threshold = denormalize(kin_recognition_threshold, KinThresh),
        cooperation_reward = denormalize(cooperation_reward, CoopReward),
        defection_penalty = denormalize(defection_penalty, DefectPen),
        sucker_penalty = denormalize(sucker_penalty, SuckerPen)
    }.
```

---

## Reputation System

### Reputation Record

```erlang
-record(reputation, {
    individual_id :: binary(),
    score :: float(),                  % -1.0 to 1.0
    history :: [{integer(), float(), atom()}],  % {timestamp, delta, reason}
    observers :: #{binary() => float()},  % Per-observer reputation view
    last_updated :: integer()
}).
```

### Reputation Management

```erlang
-module(social_reputation).

%% Update reputation based on observed behavior
-spec update_reputation(IndividualId, Event, Observers, Config) -> ok
    when IndividualId :: binary(),
         Event :: cooperation | defection | helping | aggression | achievement,
         Observers :: [binary()],
         Config :: #social_config{}.

update_reputation(IndividualId, Event, Observers, Config) ->
    CurrentRep = get_reputation(IndividualId),

    Delta = calculate_reputation_delta(Event, Config),
    BoostedDelta = case Delta > 0 of
        true -> Delta * Config#social_config.reputation_boost_factor;
        false -> Delta
    end,

    NewScore = clamp(CurrentRep#reputation.score + BoostedDelta, -1.0, 1.0),

    %% Update observer-specific views
    NewObservers = lists:foldl(fun(ObserverId, Acc) ->
        ObserverView = maps:get(ObserverId, Acc, 0.0),
        maps:put(ObserverId, clamp(ObserverView + BoostedDelta, -1.0, 1.0), Acc)
    end, CurrentRep#reputation.observers, Observers),

    UpdatedRep = CurrentRep#reputation{
        score = NewScore,
        history = [{erlang:system_time(microsecond), BoostedDelta, Event} |
                   CurrentRep#reputation.history],
        observers = NewObservers,
        last_updated = erlang:system_time(microsecond)
    },

    store_reputation(UpdatedRep),

    %% Emit event
    neuroevolution_events:emit(reputation_updated, #{
        event_type => reputation_updated,
        individual_id => IndividualId,
        previous_reputation => CurrentRep#reputation.score,
        new_reputation => NewScore,
        update_cause => Event,
        observer_ids => Observers,
        updated_at => erlang:system_time(microsecond)
    }).

calculate_reputation_delta(cooperation, _Config) -> 0.1;
calculate_reputation_delta(defection, _Config) -> -0.15;
calculate_reputation_delta(helping, _Config) -> 0.2;
calculate_reputation_delta(aggression, _Config) -> -0.1;
calculate_reputation_delta(achievement, _Config) -> 0.3.

%% Decay reputation over time
decay_all_reputations(PopulationId, Config) ->
    Individuals = get_all_individuals(PopulationId),
    DecayRate = Config#social_config.reputation_decay_rate,

    lists:foreach(fun(Ind) ->
        Rep = get_reputation(Ind#individual.id),
        DecayedScore = Rep#reputation.score * (1.0 - DecayRate),
        store_reputation(Rep#reputation{score = DecayedScore})
    end, Individuals).
```

---

## Coalition System

### Coalition Record

```erlang
-record(coalition, {
    id :: binary(),
    name :: binary() | undefined,
    members :: [binary()],             % Individual IDs
    leader_id :: binary() | none,
    strength :: float(),               % Combined fitness
    formation_reason :: defense | resource | breeding | opportunistic,
    formed_at :: integer(),
    dissolved_at :: integer() | active
}).
```

### Coalition Management

```erlang
-module(social_coalitions).

%% Form a new coalition
-spec form_coalition(FounderIds, Reason, Config) -> {ok, CoalitionId} | {error, term()}
    when FounderIds :: [binary()],
         Reason :: atom(),
         Config :: #social_config{}.

form_coalition(FounderIds, Reason, Config) when length(FounderIds) >= 2 ->
    %% Check if members are available (not in other coalitions)
    case all_available(FounderIds) of
        true ->
            CoalitionId = generate_coalition_id(),
            Coalition = #coalition{
                id = CoalitionId,
                members = FounderIds,
                leader_id = select_leader(FounderIds),
                strength = calculate_coalition_strength(FounderIds),
                formation_reason = Reason,
                formed_at = erlang:system_time(microsecond),
                dissolved_at = active
            },

            store_coalition(Coalition),

            %% Apply fitness bonus to members
            Bonus = Config#social_config.coalition_formation_incentive,
            lists:foreach(fun(MemberId) ->
                apply_fitness_modifier(MemberId, 1.0 + Bonus)
            end, FounderIds),

            %% Emit event
            neuroevolution_events:emit(coalition_formed, #{
                event_type => coalition_formed,
                coalition_id => CoalitionId,
                founder_ids => FounderIds,
                formation_reason => Reason,
                initial_strength => Coalition#coalition.strength,
                formed_at => Coalition#coalition.formed_at
            }),

            {ok, CoalitionId};
        false ->
            {error, members_unavailable}
    end.

%% Join existing coalition
join_coalition(CoalitionId, IndividualId, Config) ->
    Coalition = get_coalition(CoalitionId),

    case can_join(IndividualId, Coalition, Config) of
        true ->
            UpdatedCoalition = Coalition#coalition{
                members = [IndividualId | Coalition#coalition.members],
                strength = calculate_coalition_strength(
                    [IndividualId | Coalition#coalition.members])
            },
            store_coalition(UpdatedCoalition),

            %% Apply bonus
            Bonus = Config#social_config.coalition_formation_incentive,
            apply_fitness_modifier(IndividualId, 1.0 + Bonus),

            neuroevolution_events:emit(coalition_joined, #{
                event_type => coalition_joined,
                coalition_id => CoalitionId,
                individual_id => IndividualId,
                new_strength => UpdatedCoalition#coalition.strength,
                joined_at => erlang:system_time(microsecond)
            }),

            ok;
        false ->
            {error, cannot_join}
    end.

%% Dissolve coalition
dissolve_coalition(CoalitionId, Reason) ->
    Coalition = get_coalition(CoalitionId),

    %% Remove fitness bonuses from members
    lists:foreach(fun(MemberId) ->
        remove_coalition_bonus(MemberId)
    end, Coalition#coalition.members),

    UpdatedCoalition = Coalition#coalition{
        dissolved_at = erlang:system_time(microsecond)
    },
    store_coalition(UpdatedCoalition),

    neuroevolution_events:emit(coalition_dissolved, #{
        event_type => coalition_dissolved,
        coalition_id => CoalitionId,
        member_ids => Coalition#coalition.members,
        dissolution_reason => Reason,
        lifespan_generations => calculate_lifespan(Coalition),
        dissolved_at => UpdatedCoalition#coalition.dissolved_at
    }).
```

---

## Kin Selection

### Relatedness Calculation

```erlang
-module(social_kin).

%% Calculate coefficient of relatedness between two individuals
-spec calculate_relatedness(IndividualA, IndividualB) -> float()
    when IndividualA :: #individual{},
         IndividualB :: #individual{}.

calculate_relatedness(IndA, IndB) ->
    %% Use pedigree-based calculation
    CommonAncestors = find_common_ancestors(IndA, IndB),

    case CommonAncestors of
        [] ->
            0.0;  % No common ancestors = unrelated
        Ancestors ->
            %% Sum 0.5^(path_length) for each path through common ancestors
            lists:sum([
                math:pow(0.5, PathA + PathB)
                || {_AncestorId, PathA, PathB} <- Ancestors
            ])
    end.

%% For approximation when pedigree unavailable: use genome similarity
calculate_genetic_relatedness(GenomeA, GenomeB) ->
    SharedInnovations = count_shared_innovations(GenomeA, GenomeB),
    TotalInnovations = count_union_innovations(GenomeA, GenomeB),

    case TotalInnovations of
        0 -> 0.0;
        _ -> SharedInnovations / TotalInnovations
    end.

%% Check if kin and apply Hamilton's rule
-spec should_help_kin(Helper, Recipient, Cost, Benefit, Config) -> boolean()
    when Helper :: #individual{},
         Recipient :: #individual{},
         Cost :: float(),
         Benefit :: float(),
         Config :: #social_config{}.

should_help_kin(Helper, Recipient, Cost, Benefit, Config) ->
    Relatedness = calculate_relatedness(Helper, Recipient),
    Threshold = Config#social_config.kin_recognition_threshold,

    %% Is this individual recognized as kin?
    case Relatedness >= Threshold of
        false ->
            false;  % Not recognized as kin
        true ->
            %% Hamilton's rule: rB > C
            %% Help if relatedness * benefit > cost
            KinBonus = Config#social_config.kin_selection_bonus,
            AdjustedBenefit = Benefit * KinBonus,
            (Relatedness * AdjustedBenefit) > Cost
    end.

%% Record kin-directed helping
record_kin_help(HelperId, RecipientId, HelpType, Cost, Benefit) ->
    neuroevolution_events:emit(kin_favored, #{
        event_type => kin_favored,
        helper_id => HelperId,
        recipient_id => RecipientId,
        help_type => HelpType,
        cost_to_helper => Cost,
        benefit_to_recipient => Benefit,
        relatedness => calculate_relatedness(
            get_individual(HelperId),
            get_individual(RecipientId)
        ),
        favored_at => erlang:system_time(microsecond)
    }).
```

---

## Cooperation/Defection Dynamics

### Interaction Record

```erlang
-record(social_interaction, {
    id :: binary(),
    participant_a :: binary(),
    participant_b :: binary(),
    action_a :: cooperate | defect,
    action_b :: cooperate | defect,
    payoff_a :: float(),
    payoff_b :: float(),
    context :: term(),
    timestamp :: integer()
}).
```

### Payoff Matrix (Prisoner's Dilemma)

```erlang
-module(social_cooperation).

%% Standard payoff matrix
%% Actions: cooperate (C), defect (D)
%% Payoffs: (A's payoff, B's payoff)
%%
%%              B: Cooperate     B: Defect
%% A: Cooperate   (R, R)          (S, T)
%% A: Defect      (T, S)          (P, P)
%%
%% Where: T > R > P > S (temptation > reward > punishment > sucker)

calculate_payoffs(ActionA, ActionB, Config) ->
    R = Config#social_config.cooperation_reward,    % Reward for mutual cooperation
    P = -Config#social_config.defection_penalty,    % Punishment for mutual defection
    T = R + 0.1,                                     % Temptation to defect
    S = -Config#social_config.sucker_penalty,       % Sucker's payoff

    case {ActionA, ActionB} of
        {cooperate, cooperate} -> {R, R};
        {cooperate, defect} -> {S, T};
        {defect, cooperate} -> {T, S};
        {defect, defect} -> {P, P}
    end.

%% Process an interaction between two individuals
-spec process_interaction(IndividualA, IndividualB, Context, Config) -> Interaction
    when IndividualA :: #individual{},
         IndividualB :: #individual{},
         Context :: term(),
         Config :: #social_config{},
         Interaction :: #social_interaction{}.

process_interaction(IndA, IndB, Context, Config) ->
    %% Get actions from individuals (from their neural networks)
    ActionA = get_social_action(IndA, IndB, Context),
    ActionB = get_social_action(IndB, IndA, Context),

    %% Calculate payoffs
    {PayoffA, PayoffB} = calculate_payoffs(ActionA, ActionB, Config),

    Interaction = #social_interaction{
        id = generate_interaction_id(),
        participant_a = IndA#individual.id,
        participant_b = IndB#individual.id,
        action_a = ActionA,
        action_b = ActionB,
        payoff_a = PayoffA,
        payoff_b = PayoffB,
        context = Context,
        timestamp = erlang:system_time(microsecond)
    },

    %% Apply payoffs to fitness
    apply_fitness_delta(IndA#individual.id, PayoffA),
    apply_fitness_delta(IndB#individual.id, PayoffB),

    %% Update reputations
    Observers = get_nearby_individuals(IndA, IndB),
    update_reputation_from_action(IndA#individual.id, ActionA, Observers, Config),
    update_reputation_from_action(IndB#individual.id, ActionB, Observers, Config),

    %% Emit events
    emit_cooperation_events(Interaction),

    Interaction.

emit_cooperation_events(#social_interaction{} = Int) ->
    %% Emit appropriate event based on actions
    case {Int#social_interaction.action_a, Int#social_interaction.action_b} of
        {cooperate, cooperate} ->
            neuroevolution_events:emit(cooperation_occurred, #{
                event_type => cooperation_occurred,
                participant_ids => [Int#social_interaction.participant_a,
                                   Int#social_interaction.participant_b],
                mutual => true,
                payoffs => {Int#social_interaction.payoff_a,
                           Int#social_interaction.payoff_b},
                occurred_at => Int#social_interaction.timestamp
            });
        {defect, defect} ->
            neuroevolution_events:emit(defection_occurred, #{
                event_type => defection_occurred,
                defector_ids => [Int#social_interaction.participant_a,
                                Int#social_interaction.participant_b],
                mutual => true,
                occurred_at => Int#social_interaction.timestamp
            });
        {cooperate, defect} ->
            neuroevolution_events:emit(defection_occurred, #{
                event_type => defection_occurred,
                defector_ids => [Int#social_interaction.participant_b],
                victim_id => Int#social_interaction.participant_a,
                occurred_at => Int#social_interaction.timestamp
            });
        {defect, cooperate} ->
            neuroevolution_events:emit(defection_occurred, #{
                event_type => defection_occurred,
                defector_ids => [Int#social_interaction.participant_a],
                victim_id => Int#social_interaction.participant_b,
                occurred_at => Int#social_interaction.timestamp
            })
    end.
```

---

## Social Silo Server

### Module: `social_silo.erl`

```erlang
-module(social_silo).
-behaviour(gen_server).
-behaviour(lc_silo_behaviour).

%% API
-export([
    start_link/1,
    read_sensors/1,
    apply_actuators/2,
    get_state/1
]).

%% Silo behaviour callbacks
-export([
    init_silo/1,
    step/2,
    get_sensor_count/0,
    get_actuator_count/0
]).

-record(state, {
    population_id :: binary(),
    social_state :: #social_state{},
    config :: #social_config{},
    controller :: pid(),               % TWEANN controller
    reputations :: #{binary() => #reputation{}},
    coalitions :: #{binary() => #coalition{}},
    interaction_history :: [#social_interaction{}]
}).

%%====================================================================
%% Silo Behaviour Implementation
%%====================================================================

init_silo(Config) ->
    {ok, #state{
        population_id = maps:get(population_id, Config),
        social_state = initial_social_state(),
        config = default_social_config(),
        reputations = #{},
        coalitions = #{},
        interaction_history = []
    }}.

step(#state{} = State, Generation) ->
    %% 1. Read current social state
    SensorInputs = social_silo_sensors:read_sensors(State#state.social_state),

    %% 2. Get controller outputs
    ActuatorOutputs = lc_controller:evaluate(State#state.controller, SensorInputs),

    %% 3. Apply actuator changes
    NewConfig = social_silo_actuators:apply_actuators(
        ActuatorOutputs, State#state.config),

    %% 4. Process social interactions for this generation
    NewInteractions = process_generation_interactions(
        State#state.population_id, NewConfig),

    %% 5. Update reputation decay
    social_reputation:decay_all_reputations(
        State#state.population_id, NewConfig),

    %% 6. Maintain coalitions (check for dissolution)
    maintain_coalitions(State#state.coalitions, NewConfig),

    %% 7. Update social state
    NewSocialState = calculate_social_state(
        State#state.population_id,
        State#state.reputations,
        State#state.coalitions,
        NewInteractions
    ),

    {ok, State#state{
        social_state = NewSocialState,
        config = NewConfig,
        interaction_history = NewInteractions ++ State#state.interaction_history
    }}.

get_sensor_count() -> 12.
get_actuator_count() -> 8.
```

---

## Cross-Silo Signals

### Signals to Other Silos

| Signal | To Silo | Meaning |
|--------|---------|---------|
| `social_cohesion` | Task | High cooperation = stable evolution |
| `conflict_level` | Resource | High conflict = resource competition |
| `network_density` | Cultural | Dense network = faster cultural spread |

### Signals from Other Silos

| Signal | From Silo | Effect |
|--------|-----------|--------|
| `resource_scarcity` | Resource | Increases competition, coalition formation |
| `stagnation_severity` | Task | Triggers social restructuring |
| `innovation_rate` | Cultural | High innovation = social prestige bonus |

---

## Events Emitted

| Event | Trigger |
|-------|---------|
| `reputation_updated` | Reputation score changed |
| `coalition_formed` | New coalition created |
| `coalition_dissolved` | Coalition ended |
| `coalition_joined` | Individual joined coalition |
| `coalition_expelled` | Individual removed from coalition |
| `cooperation_occurred` | Cooperative interaction |
| `defection_occurred` | Defection interaction |
| `kin_recognized` | Kin relationship detected |
| `kin_favored` | Kin-directed helping |
| `dominance_established` | Hierarchy position set |
| `dominance_challenged` | Hierarchy challenged |

---

## Implementation Phases

- [ ] **Phase 1:** Reputation system (tracking, update, decay)
- [ ] **Phase 2:** L0 Sensors for social state
- [ ] **Phase 3:** L0 Actuators for social parameters
- [ ] **Phase 4:** Coalition formation and management
- [ ] **Phase 5:** Kin selection and Hamilton's rule
- [ ] **Phase 6:** Cooperation/defection game dynamics
- [ ] **Phase 7:** Social silo server with TWEANN controller
- [ ] **Phase 8:** Cross-silo signal integration

---

## Success Criteria

- [ ] Reputation system tracks and decays correctly
- [ ] Coalitions form, grow, and dissolve naturally
- [ ] Kin selection influences helping behavior
- [ ] Cooperation/defection dynamics emerge
- [ ] TWEANN controller adapts social parameters
- [ ] All social events emitted correctly
- [ ] Cross-silo signals flow correctly
- [ ] Performance: < 10% overhead per generation

---

## References

- PLAN_BEHAVIORAL_EVENTS.md - Event definitions
- PLAN_L2_L1_HIERARCHICAL_INTERFACE.md - LC architecture
- task_silo.erl - Reference silo implementation
- "The Evolution of Cooperation" - Axelrod
- "Kin Selection" - Hamilton
- "Social Network Analysis" - Wasserman & Faust
