%% @doc Task Silo - Evolution optimization controller for neuroevolution.
%%
%% Part of the Liquid Conglomerate v2 architecture. The Task Silo optimizes
%% evolution hyperparameters to maximize fitness improvement.
%%
%% == Hierarchical Levels ==
%%
%% - L0 (Defaults): Safe starting parameters - ALWAYS ACTIVE
%% - L1 (Tactical): Per-generation adjustments based on recent fitness
%% - L2 (Strategic): Multi-generation learning via LTC network (future)
%%
%% == Time Constant ==
%%
%% τ = 50 (slow adaptation for stable evolution)
%%
%% == Usage ==
%%
%% %% Start task silo
%% {ok, Pid} = task_silo:start_link(#{
%%     enabled_levels => [l0, l1],
%%     stagnation_threshold => 5
%% }),
%%
%% %% Get recommended parameters after a generation
%% Stats = #{best_fitness => 0.85, improvement => 0.02, stagnation => 0},
%% #{mutation_rate := MR} = task_silo:get_recommendations(Pid, Stats).
%%
%% @author R.G. Lefever
%% @copyright 2024-2026 R.G. Lefever
-module(task_silo).
-behaviour(gen_server).

-include("meta_controller.hrl").

%% API
-export([
    start_link/0,
    start_link/1,
    get_recommendations/1,
    get_recommendations/2,
    update_stats/2,
    get_state/1,
    set_enabled_levels/2,
    set_l2_guidance/2,
    reset/1
]).

%% gen_server callbacks
-export([
    init/1,
    handle_call/3,
    handle_cast/2,
    handle_info/2,
    terminate/2,
    code_change/3
]).

-define(SERVER, ?MODULE).
-define(TIME_CONSTANT, 50).  % tau = 50 generations (slow)

-record(state, {
    %% Configuration
    enabled_levels :: [l0 | l1 | l2],
    realm :: binary(),  % Realm for event publishing

    %% L0 TWEANN Configuration
    %% When true, task_silo uses lc_controller for TWEANN-based hyperparameter control
    %% When false, uses rule-based L1 adjustments
    l0_tweann_enabled :: boolean(),

    %% L2 Configuration
    %% When true, task_silo queries meta_controller for L2 guidance
    l2_enabled :: boolean(),

    %% Current parameters (output)
    current_params :: map(),

    %% L1 Tactical state (evaluation-centric)
    total_evaluations :: non_neg_integer(),   %% PRIMARY: Total evaluations (replaces generation)
    cohort :: non_neg_integer(),              %% SECONDARY: Cohort number for lineage (formerly generation)
    fitness_history :: [float()],
    improvement_history :: [float()],
    history_size :: pos_integer(),

    %% Velocity-based stagnation detection (replaces counter-based)
    velocity_state :: velocity_state(),

    %% L1 Tactical tuning factors
    exploration_boost :: float(),     % Based on stagnation_severity from velocity
    exploitation_boost :: float(),    % Increased when improving

    %% Previous state for EMA smoothing (L2 controls adaptation_momentum)
    prev_exploration_boost :: float(),
    prev_exploitation_boost :: float(),
    prev_stagnation_severity :: float(),

    %% L2 Strategic guidance (controls L1 behavior)
    %% When L2 is enabled, these values come from meta_controller
    %% When L2 is disabled, defaults from ?L2_GUIDANCE_DEFAULTS are used
    l2_guidance :: l2_guidance(),

    %% Population stats from neuroevolution_server (for sensor inputs)
    %% These are updated via update_stats/2 each generation
    diversity_index = 0.5 :: float(),
    species_count_ratio = 0.5 :: float(),
    avg_network_complexity = 0.5 :: float(),
    prev_complexity = 0.5 :: float(),  % For computing complexity_velocity
    resource_pressure_signal = 0.0 :: float()
}).

%%% ============================================================================
%%% API Functions
%%% ============================================================================

%% @doc Start the task silo with default configuration.
-spec start_link() -> {ok, pid()} | ignore | {error, term()}.
start_link() ->
    start_link(#{}).

%% @doc Start the task silo with custom configuration.
%%
%% Options:
%% - enabled_levels: [l0, l1, l2] (default: [l0, l1])
%% - stagnation_threshold: Generations without improvement before boosting (default: 5)
%% - history_size: Number of generations to track (default: 20)
-spec start_link(map()) -> {ok, pid()} | ignore | {error, term()}.
start_link(Config) ->
    gen_server:start_link({local, ?SERVER}, ?MODULE, Config, []).

%% @doc Get recommended parameters based on current evolution state.
%%
%% Stats should include:
%% - best_fitness: Current best fitness
%% - avg_fitness: Average population fitness
%% - improvement: Fitness improvement from last generation
%% - generation: Current generation number
%%
%% Returns a map of recommended hyperparameters.
-spec get_recommendations(pid(), map()) -> map().
get_recommendations(Pid, Stats) ->
    gen_server:call(Pid, {get_recommendations, Stats}).

%% @doc Get recommendations from the registered server using cached state.
-spec get_recommendations(pid()) -> map().
get_recommendations(Pid) ->
    gen_server:call(Pid, get_recommendations).

%% @doc Update internal state with generation statistics.
-spec update_stats(pid(), map()) -> ok.
update_stats(Pid, Stats) ->
    gen_server:cast(Pid, {update_stats, Stats}).

%% @doc Get current silo state for debugging/monitoring.
-spec get_state(pid()) -> map().
get_state(Pid) ->
    gen_server:call(Pid, get_state).

%% @doc Set which levels are enabled.
-spec set_enabled_levels(pid(), [l0 | l1 | l2]) -> ok.
set_enabled_levels(Pid, Levels) ->
    gen_server:cast(Pid, {set_enabled_levels, Levels}).

%% @doc Set L2 guidance from meta_controller.
%%
%% Called by meta_controller to update the L1 control parameters.
%% This is the L2→L1 communication channel.
-spec set_l2_guidance(pid(), l2_guidance()) -> ok.
set_l2_guidance(Pid, Guidance) when is_record(Guidance, l2_guidance) ->
    gen_server:cast(Pid, {set_l2_guidance, Guidance}).

%% @doc Reset the silo state (for new training session).
-spec reset(pid()) -> ok.
reset(Pid) ->
    gen_server:cast(Pid, reset).

%%% ============================================================================
%%% gen_server Callbacks
%%% ============================================================================

init(Config) ->
    EnabledLevels = maps:get(enabled_levels, Config, [l0, l1]),
    HistorySize = maps:get(history_size, Config, 20),
    Realm = maps:get(realm, Config, <<"default">>),
    L2Enabled = maps:get(l2_enabled, Config, false),

    %% L0 TWEANN mode: when true, use lc_controller for neural network-based control
    %% When false, use rule-based L1 adjustments
    L0TweannEnabled = maps:get(l0_tweann_enabled, Config, false),

    %% Velocity-based stagnation configuration
    VelocityThreshold = maps:get(velocity_threshold, Config, 0.001),
    VelocityWindowSize = maps:get(velocity_window_size, Config, 10),

    %% Start with L0 defaults
    DefaultParams = task_l0_defaults:get_defaults(),

    %% Initialize L2 guidance with defaults
    %% When L2 is enabled, these will be updated by querying meta_controller
    L2Guidance = ?L2_GUIDANCE_DEFAULTS,

    %% Initialize velocity state for evaluation-centric stagnation detection
    VelocityState = #velocity_state{
        improvement_window = [],
        window_size = VelocityWindowSize,
        current_velocity = 0.0,
        velocity_threshold = VelocityThreshold,
        fitness_checkpoints = [],
        stagnation_severity = 0.0,
        last_total_evaluations = 0
    },

    %% Start lc_silo_chain if TWEANN mode is enabled (3 chained LTC TWEANNs)
    maybe_start_lc_silo_chain(L0TweannEnabled, Config),

    State = #state{
        enabled_levels = EnabledLevels,
        realm = Realm,
        l0_tweann_enabled = L0TweannEnabled,
        l2_enabled = L2Enabled,
        current_params = DefaultParams,
        total_evaluations = 0,
        cohort = 0,
        fitness_history = [],
        improvement_history = [],
        history_size = HistorySize,
        velocity_state = VelocityState,
        exploration_boost = 0.0,
        exploitation_boost = 0.0,
        prev_exploration_boost = 0.0,
        prev_exploitation_boost = 0.0,
        prev_stagnation_severity = 0.0,
        l2_guidance = L2Guidance
    },

    error_logger:info_msg("[task_silo] Started with levels=~p, velocity_threshold=~.4f, l2_enabled=~p, l0_tweann=~p~n",
        [EnabledLevels, VelocityThreshold, L2Enabled, L0TweannEnabled]),

    {ok, State}.

handle_call({get_recommendations, Stats}, _From, State) ->
    %% Query L2 for guidance if enabled
    StateWithL2 = maybe_query_l2_guidance(Stats, State),
    %% Update state with new stats and compute recommendations
    NewState = update_internal_state(Stats, StateWithL2),
    Params = compute_recommendations(NewState),
    %% Publish recommendations when they change (event-driven read model)
    maybe_publish_recommendations(State#state.current_params, Params),
    {reply, Params, NewState#state{current_params = Params}};

handle_call(get_recommendations, _From, State) ->
    {reply, State#state.current_params, State};

handle_call(get_state, _From, State) ->
    L2Guidance = State#state.l2_guidance,
    VelocityState = State#state.velocity_state,
    StateMap = #{
        enabled_levels => State#state.enabled_levels,
        current_params => State#state.current_params,
        %% PRIMARY: Evaluation-centric metrics
        total_evaluations => State#state.total_evaluations,
        %% SECONDARY: Cohort number for backward compatibility
        cohort => State#state.cohort,
        generation => State#state.cohort,  %% Deprecated alias
        %% Velocity-based stagnation metrics
        improvement_velocity => VelocityState#velocity_state.current_velocity,
        stagnation_severity => VelocityState#velocity_state.stagnation_severity,
        velocity_threshold => VelocityState#velocity_state.velocity_threshold,
        %% L1 tactical state
        exploration_boost => State#state.exploration_boost,
        exploitation_boost => State#state.exploitation_boost,
        fitness_history_size => length(State#state.fitness_history),
        %% L2 guidance for visualization
        l2_guidance => #{
            aggression_factor => L2Guidance#l2_guidance.aggression_factor,
            exploration_step => L2Guidance#l2_guidance.exploration_step,
            stagnation_sensitivity => L2Guidance#l2_guidance.stagnation_sensitivity,
            topology_aggression => L2Guidance#l2_guidance.topology_aggression,
            exploitation_weight => L2Guidance#l2_guidance.exploitation_weight
        }
    },
    {reply, StateMap, State};

handle_call(_Request, _From, State) ->
    {reply, {error, unknown_request}, State}.

handle_cast({update_stats, Stats}, State) ->
    NewState = update_internal_state(Stats, State),
    {noreply, NewState};

handle_cast({set_enabled_levels, Levels}, State) ->
    error_logger:info_msg("[task_silo] Enabled levels changed to ~p~n", [Levels]),
    {noreply, State#state{enabled_levels = Levels}};

handle_cast({set_l2_guidance, Guidance}, State) ->
    %% Log significant changes in L2 guidance
    OldGuidance = State#state.l2_guidance,
    case significant_guidance_change(OldGuidance, Guidance) of
        true ->
            error_logger:info_msg(
                "[task_silo] L2 guidance updated: aggression=~.2f, exploration_step=~.2f, "
                "topology_aggression=~.2f~n",
                [Guidance#l2_guidance.aggression_factor,
                 Guidance#l2_guidance.exploration_step,
                 Guidance#l2_guidance.topology_aggression]
            );
        false ->
            ok
    end,
    {noreply, State#state{l2_guidance = Guidance}};

handle_cast(reset, State) ->
    DefaultParams = task_l0_defaults:get_defaults(),
    %% Reset velocity state but preserve threshold configuration
    OldVelocityState = State#state.velocity_state,
    NewVelocityState = #velocity_state{
        improvement_window = [],
        window_size = OldVelocityState#velocity_state.window_size,
        current_velocity = 0.0,
        velocity_threshold = OldVelocityState#velocity_state.velocity_threshold,
        fitness_checkpoints = [],
        stagnation_severity = 0.0,
        last_total_evaluations = 0
    },
    NewState = State#state{
        current_params = DefaultParams,
        total_evaluations = 0,
        cohort = 0,
        fitness_history = [],
        improvement_history = [],
        velocity_state = NewVelocityState,
        exploration_boost = 0.0,
        exploitation_boost = 0.0,
        prev_exploration_boost = 0.0,
        prev_stagnation_severity = 0.0,
        l2_guidance = ?L2_GUIDANCE_DEFAULTS
    },
    error_logger:info_msg("[task_silo] Reset to initial state~n"),
    {noreply, NewState};

handle_cast(_Msg, State) ->
    {noreply, State}.

handle_info(_Info, State) ->
    {noreply, State}.

terminate(_Reason, _State) ->
    ok.

code_change(_OldVsn, State, _Extra) ->
    {ok, State}.

%%% ============================================================================
%%% Internal Functions - State Updates
%%% ============================================================================

%% @private Update internal state with evaluation-centric statistics.
%%
%% Stats should include:
%% - best_fitness OR red_team_max_fitness: Current best fitness
%% - total_evaluations: Total evaluations so far (PRIMARY)
%% - generation/cohort: Current cohort number (SECONDARY, for lineage)
%% - improvement: Fitness improvement from previous (optional)
update_internal_state(Stats, State) ->
    %% Support both best_fitness (standard) and red_team_max_fitness (self-play mode)
    BestFitness = maps:get(best_fitness, Stats,
                    maps:get(red_team_max_fitness, Stats, 0.0)),
    Improvement = maps:get(improvement, Stats, 0.0),
    %% PRIMARY: Total evaluations (evaluation-centric progress)
    TotalEvaluations = maps:get(total_evaluations, Stats, State#state.total_evaluations),
    %% SECONDARY: Cohort number (accept both 'cohort' and 'generation' for backward compat)
    Cohort = maps:get(cohort, Stats, maps:get(generation, Stats, State#state.cohort + 1)),

    %% Update fitness history
    FitnessHistory = lists:sublist(
        [BestFitness | State#state.fitness_history],
        State#state.history_size
    ),

    %% Update improvement history
    ImprovementHistory = lists:sublist(
        [Improvement | State#state.improvement_history],
        State#state.history_size
    ),

    %% Update velocity state for stagnation detection
    NewVelocityState = update_velocity_state(
        BestFitness,
        TotalEvaluations,
        State#state.velocity_state
    ),

    %% Get L2 guidance (controls boost intensity AND adaptation momentum)
    L2Guidance = State#state.l2_guidance,

    %% Compute L1 tactical boosts from stagnation severity (continuous 0.0-1.0)
    StagnationSeverity = NewVelocityState#velocity_state.stagnation_severity,
    {RawExplorationBoost, RawExploitationBoost} = compute_l1_boosts_from_severity(
        StagnationSeverity,
        ImprovementHistory,
        L2Guidance
    ),

    %% Apply ASYMMETRIC EMA smoothing for fast escalation, slow de-escalation
    %% When stagnating (RawExplorationBoost > Previous): use LOW momentum (fast response)
    %% When recovering (RawExplorationBoost < Previous): use HIGH momentum (slow de-escalation)
    %% This prevents oscillation while ensuring responsive intervention
    BaseMomentum = L2Guidance#l2_guidance.adaptation_momentum,

    %% Asymmetric momentum: fast up (BaseMomentum * 0.3), slow down (BaseMomentum + 0.5)
    ExplorationMomentum = case RawExplorationBoost > State#state.prev_exploration_boost of
        true -> BaseMomentum * 0.3;           %% Escalating: very fast (low momentum)
        false -> min(0.9, BaseMomentum + 0.5)  %% De-escalating: slow (high momentum)
    end,
    ExploitationMomentum = case RawExploitationBoost > State#state.prev_exploitation_boost of
        true -> BaseMomentum * 0.5;           %% Entering exploitation: medium speed
        false -> min(0.9, BaseMomentum + 0.3)  %% Leaving exploitation: slow
    end,

    SmoothedExplorationBoost = ExplorationMomentum * State#state.prev_exploration_boost +
                               (1.0 - ExplorationMomentum) * RawExplorationBoost,
    SmoothedExploitationBoost = ExploitationMomentum * State#state.prev_exploitation_boost +
                                (1.0 - ExploitationMomentum) * RawExploitationBoost,

    %% Emit intervention events when stagnation_severity changes significantly
    %% Pass L2 guidance so thresholds are LC-controlled, not hardcoded
    maybe_emit_velocity_intervention_event(
        State#state.realm,
        TotalEvaluations,
        Cohort,
        State#state.prev_stagnation_severity,
        StagnationSeverity,
        NewVelocityState#velocity_state.current_velocity,
        L2Guidance
    ),

    %% Extract population stats from neuroevolution_server
    DiversityIndex = maps:get(diversity_index, Stats, State#state.diversity_index),
    SpeciesCountRatio = maps:get(species_count_ratio, Stats, State#state.species_count_ratio),
    AvgNetworkComplexity = maps:get(avg_network_complexity, Stats, State#state.avg_network_complexity),
    ResourcePressureSignal = maps:get(resource_pressure_signal, Stats, State#state.resource_pressure_signal),

    NewState = State#state{
        total_evaluations = TotalEvaluations,
        cohort = Cohort,
        fitness_history = FitnessHistory,
        improvement_history = ImprovementHistory,
        velocity_state = NewVelocityState,
        exploration_boost = SmoothedExplorationBoost,
        exploitation_boost = SmoothedExploitationBoost,
        prev_exploration_boost = SmoothedExplorationBoost,
        prev_exploitation_boost = SmoothedExploitationBoost,
        prev_stagnation_severity = StagnationSeverity,
        %% Population stats for sensor inputs
        diversity_index = DiversityIndex,
        species_count_ratio = SpeciesCountRatio,
        avg_network_complexity = AvgNetworkComplexity,
        prev_complexity = State#state.avg_network_complexity,  % Store previous for velocity
        resource_pressure_signal = ResourcePressureSignal
    },

    %% Report reward to lc_silo_chain for online learning (if TWEANN mode enabled)
    maybe_report_reward_to_lc_silo_chain(State#state.l0_tweann_enabled, NewState),

    NewState.

%% @private Update velocity state with new fitness checkpoint.
%%
%% Computes improvement velocity as: (delta_fitness / delta_evals) * 1000
%% Then calculates stagnation_severity from velocity vs threshold.
update_velocity_state(BestFitness, TotalEvaluations, VelocityState) ->
    #velocity_state{
        fitness_checkpoints = OldCheckpoints,
        improvement_window = OldWindow,
        window_size = WindowSize,
        velocity_threshold = VelocityThreshold,
        last_total_evaluations = LastEvaluations
    } = VelocityState,

    %% Skip if no evaluations since last update
    case TotalEvaluations > LastEvaluations of
        false ->
            VelocityState;
        true ->
            %% Add new checkpoint
            NewCheckpoint = {TotalEvaluations, BestFitness},
            Checkpoints = lists:sublist([NewCheckpoint | OldCheckpoints], WindowSize + 1),

            %% Calculate velocity from previous checkpoint
            Velocity = case OldCheckpoints of
                [{PrevEvaluations, PrevFitness} | _] when TotalEvaluations > PrevEvaluations ->
                    DeltaFitness = BestFitness - PrevFitness,
                    DeltaEvaluations = TotalEvaluations - PrevEvaluations,
                    %% Velocity = fitness improvement per 1000 evaluations
                    (DeltaFitness / DeltaEvaluations) * 1000;
                _ ->
                    0.0
            end,

            %% Update rolling window of velocities
            NewWindow = lists:sublist([Velocity | OldWindow], WindowSize),

            %% Calculate average velocity
            AvgVelocity = case NewWindow of
                [] -> 0.0;
                Velocities -> lists:sum(Velocities) / length(Velocities)
            end,

            %% Calculate stagnation severity: 0.0 = healthy, 1.0 = critical
            %% severity = clamp((threshold - avg_velocity) / threshold, 0.0, 1.0)
            Severity = case VelocityThreshold > 0 of
                true ->
                    RawSeverity = (VelocityThreshold - AvgVelocity) / VelocityThreshold,
                    max(0.0, min(1.0, RawSeverity));
                false ->
                    0.0
            end,

            VelocityState#velocity_state{
                fitness_checkpoints = Checkpoints,
                improvement_window = NewWindow,
                current_velocity = Velocity,
                stagnation_severity = Severity,
                last_total_evaluations = TotalEvaluations
            }
    end.

%% @private Compute L1 tactical boosts from stagnation severity.
%%
%% Uses continuous stagnation_severity (0.0-1.0) instead of binary counter threshold.
%% L2 guidance controls the exploration_step (intensity of response).
compute_l1_boosts_from_severity(StagnationSeverity, ImprovementHistory, L2Guidance) ->
    %% Get L2-controlled exploration step (default: 0.1, range: 0.05-0.5)
    ExplorationStep = L2Guidance#l2_guidance.exploration_step,

    %% Exploration boost: proportional to stagnation severity
    %% At severity=0.5, boost is 0.5 * exploration_step
    %% At severity=1.0, boost is exploration_step (max response)
    ExplorationBoost = StagnationSeverity * ExplorationStep,

    %% Exploitation boost: increases when consistently improving
    ExploitationBoost = case length(ImprovementHistory) >= 3 of
        true ->
            RecentImprovements = lists:sublist(ImprovementHistory, 3),
            AvgImprovement = lists:sum(RecentImprovements) / 3,
            case AvgImprovement > 0.01 of
                true -> min(1.0, AvgImprovement * 10);
                false -> 0.0
            end;
        false ->
            0.0
    end,

    {ExplorationBoost, ExploitationBoost}.

%%% ============================================================================
%%% Internal Functions - Recommendations
%%% ============================================================================

%% @private Compute recommended parameters based on current state.
%%
%% When l0_tweann_enabled=true: Query lc_silo_chain for 3-level TWEANN-based recommendations
%% When l0_tweann_enabled=false: Use rule-based L1 adjustments
compute_recommendations(#state{l0_tweann_enabled = true} = State) ->
    %% TWEANN mode: query lc_silo_chain for hierarchical neural network output
    SensorInputs = build_sensor_inputs(State),
    case whereis(lc_task_silo_chain) of
        undefined ->
            %% lc_silo_chain not running, fall back to rule-based
            error_logger:warning_msg("[task_silo] L0 TWEANN enabled but lc_silo_chain not found, using rule-based~n"),
            compute_recommendations_rule_based(State);
        Pid when is_pid(Pid) ->
            try
                TweannParams = lc_silo_chain:get_recommendations(Pid, SensorInputs),
                %% Merge with L0 defaults for any missing params and apply bounds
                L0Defaults = task_l0_defaults:get_defaults(),
                MergedParams = maps:merge(L0Defaults, TweannParams),
                task_l0_defaults:apply_bounds(MergedParams)
            catch
                _:Reason ->
                    error_logger:warning_msg("[task_silo] TWEANN query failed: ~p, using rule-based~n", [Reason]),
                    compute_recommendations_rule_based(State)
            end
    end;
compute_recommendations(State) ->
    %% Rule-based mode (L1 adjustments)
    compute_recommendations_rule_based(State).

%% @private Rule-based recommendations using L1 tactical adjustments.
compute_recommendations_rule_based(State) ->
    #state{enabled_levels = Levels} = State,

    %% Start with L0 defaults
    L0Params = task_l0_defaults:get_defaults(),

    %% Apply L1 tactical adjustments if enabled
    L1Params = case lists:member(l1, Levels) of
        true -> apply_l1_adjustments(L0Params, State);
        false -> L0Params
    end,

    %% L2 would apply strategic adjustments here (future)
    %% L2Params = case lists:member(l2, Levels) of
    %%     true -> apply_l2_adjustments(L1Params, State);
    %%     false -> L1Params
    %% end,

    %% Always apply bounds from L0 (safety net)
    task_l0_defaults:apply_bounds(L1Params).

%% @private Build sensor inputs for the LC TWEANN from current state.
build_sensor_inputs(State) ->
    #state{
        fitness_history = FitnessHistory,
        improvement_history = ImprovementHistory,
        velocity_state = VelocityState,
        total_evaluations = TotalEvals,
        %% Population stats from neuroevolution_server
        diversity_index = DiversityIndex,
        species_count_ratio = SpeciesCountRatio,
        avg_network_complexity = AvgNetworkComplexity,
        prev_complexity = PrevComplexity,
        resource_pressure_signal = ResourcePressureSignal
    } = State,

    %% Extract current fitness metrics
    BestFitness = case FitnessHistory of
        [F | _] -> F;
        [] -> 0.0
    end,

    AvgFitness = case FitnessHistory of
        [] -> 0.0;
        Fs -> lists:sum(Fs) / length(Fs)
    end,

    FitnessVariance = case length(FitnessHistory) > 1 of
        true ->
            Mean = AvgFitness,
            SumSq = lists:sum([math:pow(F - Mean, 2) || F <- FitnessHistory]),
            SumSq / length(FitnessHistory);
        false -> 0.0
    end,

    %% Get velocity and stagnation from velocity state
    ImprovementVelocity = VelocityState#velocity_state.current_velocity,
    StagnationSeverity = VelocityState#velocity_state.stagnation_severity,

    %% Compute derived metrics
    RecentImprovement = case ImprovementHistory of
        [I | _] -> I;
        [] -> 0.0
    end,

    %% Compute complexity velocity (rate of change in network complexity)
    ComplexityVelocity = AvgNetworkComplexity - PrevComplexity,

    %% Build sensor input map
    #{
        %% Evolution sensors (1-16)
        best_fitness => clamp(BestFitness, 0.0, 1.0),
        avg_fitness => clamp(AvgFitness, 0.0, 1.0),
        fitness_variance => clamp(FitnessVariance, 0.0, 1.0),
        improvement_velocity => clamp(ImprovementVelocity, -1.0, 1.0),
        stagnation_severity => clamp(StagnationSeverity, 0.0, 1.0),
        diversity_index => clamp(DiversityIndex, 0.0, 1.0),
        species_count_ratio => clamp(SpeciesCountRatio, 0.0, 1.0),
        avg_network_complexity => clamp(AvgNetworkComplexity, 0.0, 1.0),
        complexity_velocity => clamp(ComplexityVelocity, -1.0, 1.0),
        elite_dominance => case AvgFitness > 0 of
            true -> clamp(BestFitness / AvgFitness, 0.0, 1.0);
            false -> 0.5
        end,
        crossover_success_rate => 0.5,  %% Future: track crossover outcomes
        mutation_impact => clamp(RecentImprovement * 10, 0.0, 1.0),
        resource_pressure_signal => clamp(ResourcePressureSignal, 0.0, 1.0),
        evaluation_progress => clamp(TotalEvals / 100000.0, 0.0, 1.0),
        entropy => clamp(FitnessVariance * 2, 0.0, 1.0),
        convergence_trend => clamp(-ImprovementVelocity * 5, -1.0, 1.0),
        %% Self-play archive sensors (17-21) - future integration with archive
        archive_fill_ratio => 0.0,
        archive_fitness_mean => 0.5,
        archive_fitness_variance => 0.0,
        archive_staleness => 0.0,
        population_vs_archive_ratio => 0.5
    }.

%% @private Clamp value to range.
clamp(Value, Min, Max) ->
    max(Min, min(Max, Value)).

%% @private Apply L1 tactical adjustments.
%%
%% L1 adjusts parameters based on recent evolution dynamics:
%% - Stagnation -> increase mutation rate, add_node_rate (exploration)
%% - Improvement -> decrease mutation rate (exploitation)
%%
%% L2 guidance controls HOW AGGRESSIVE these adjustments are:
%% - aggression_factor: Multiplier for mutation adjustments (0.0-2.0)
%% - exploitation_weight: Balance between explore/exploit (0.2-0.8)
%% - topology_aggression: Multiplier for add_node_rate boost (1.0-3.0)
%%
%% NOTE: Small noise is added to outputs to prevent deterministic patterns
%% while waiting for L0 TWEANN integration. This simulates the stochastic
%% nature a neural network would provide.
apply_l1_adjustments(Params, State) ->
    #state{
        exploration_boost = ExplorationBoost,
        exploitation_boost = ExploitationBoost,
        l2_guidance = L2Guidance
    } = State,

    %% Get L2 guidance factors
    Aggression = L2Guidance#l2_guidance.aggression_factor,
    ExploitWeight = L2Guidance#l2_guidance.exploitation_weight,
    TopologyAggression = L2Guidance#l2_guidance.topology_aggression,

    %% Net adjustment factor: positive = explore, negative = exploit
    %% L2 controls the exploitation weight (how much to value exploitation)
    NetFactor = ExplorationBoost - (ExploitationBoost * ExploitWeight),

    %% Add small noise to simulate stochastic behavior (±10% variation)
    %% This prevents the deterministic sawtooth pattern in the UI
    Noise = (rand:uniform() - 0.5) * 0.2,
    NoisyNetFactor = NetFactor + (NetFactor * Noise),

    %% Adjust mutation rate using L2's aggression factor
    %% With Aggression=2.0 and NetFactor=1.0: BaseMR * (1 + 2.0) = BaseMR * 3.0 (+200%)
    %% With Aggression=0.5 and NetFactor=1.0: BaseMR * (1 + 0.5) = BaseMR * 1.5 (+50%)
    BaseMR = maps:get(mutation_rate, Params, 0.10),
    AdjustedMR = BaseMR * (1.0 + NoisyNetFactor * Aggression),

    %% Adjust mutation strength (slightly less aggressive than mutation rate)
    %% Add independent noise for less correlated outputs
    BaseMS = maps:get(mutation_strength, Params, 0.30),
    MSNoise = (rand:uniform() - 0.5) * 0.15,
    AdjustedMS = BaseMS * (1.0 + (NetFactor + NetFactor * MSNoise) * Aggression * 0.6),

    %% Adjust topology mutation using L2's topology aggression
    %% With TopologyAggression=3.0: BaseANR * 3.0 (+200% when heavily stagnating)
    BaseANR = maps:get(add_node_rate, Params, 0.03),
    AdjustedANR = case ExplorationBoost > 0.5 of
        true -> BaseANR * TopologyAggression;  % L2-controlled boost
        false -> BaseANR
    end,

    %% Adjust selection ratio (tighter when exploiting)
    %% Add independent noise
    BaseSR = maps:get(selection_ratio, Params, 0.20),
    SRNoise = (rand:uniform() - 0.5) * 0.1,
    AdjustedSR = BaseSR * (1.0 - ExploitationBoost * 0.2) * (1.0 + SRNoise),

    Params#{
        mutation_rate => AdjustedMR,
        mutation_strength => AdjustedMS,
        add_node_rate => AdjustedANR,
        selection_ratio => AdjustedSR
    }.

%%% ============================================================================
%%% Internal Functions - Intervention Events (Velocity-Based)
%%% ============================================================================

%% @private Emit velocity-based intervention events.
%%
%% Uses stagnation_severity (0.0-1.0) instead of counter-based threshold.
%% Events trigger on severity threshold crossings for clearer state transitions.
%%
%% Thresholds are LC-controlled via L2 guidance:
%% - severity < warning_threshold: Healthy (no intervention)
%% - severity >= warning_threshold: Warning (mild intervention)
%% - severity >= intervention_threshold: Active intervention
%% - severity >= critical_threshold: Critical intervention
maybe_emit_velocity_intervention_event(_Realm, _TotalEvaluations, _Cohort, Prev, Current, _Velocity, _L2Guidance)
    when abs(Prev - Current) < 0.1 ->
    %% No significant change in severity
    ok;

maybe_emit_velocity_intervention_event(Realm, TotalEvaluations, Cohort, PrevSeverity, CurrentSeverity, Velocity, L2Guidance) ->
    %% Extract LC-controlled thresholds from L2 guidance
    WarningThreshold = L2Guidance#l2_guidance.warning_threshold,
    InterventionThreshold = L2Guidance#l2_guidance.intervention_threshold,
    CriticalThreshold = L2Guidance#l2_guidance.critical_threshold,

    %% Delegate to threshold-based matching
    emit_velocity_event_with_thresholds(
        Realm, TotalEvaluations, Cohort, PrevSeverity, CurrentSeverity, Velocity,
        WarningThreshold, InterventionThreshold, CriticalThreshold
    ).

%% @private Emit events based on LC-controlled threshold crossings.
emit_velocity_event_with_thresholds(Realm, TotalEvaluations, Cohort, PrevSeverity, CurrentSeverity, Velocity,
                                     WarningThreshold, _InterventionThreshold, _CriticalThreshold)
    when PrevSeverity < WarningThreshold, CurrentSeverity >= WarningThreshold ->
    %% Crossed into warning territory
    Event = {task_silo_stagnation_warning, #{
        realm => Realm,
        source => task_silo,
        timestamp => erlang:system_time(millisecond),
        total_evaluations => TotalEvaluations,
        cohort => Cohort,
        stagnation_severity => CurrentSeverity,
        improvement_velocity => Velocity,
        threshold => WarningThreshold,
        message => <<"Improvement velocity declining, monitoring closely">>
    }},
    Topic = neuroevolution_events:topic_intervention(Realm),
    neuroevolution_events:publish(Topic, Event),
    error_logger:info_msg(
        "[task_silo] Evals ~p: STAGNATION WARNING - severity=~.2f >= ~.2f, velocity=~.4f~n",
        [TotalEvaluations, CurrentSeverity, WarningThreshold, Velocity]
    );

emit_velocity_event_with_thresholds(Realm, TotalEvaluations, Cohort, PrevSeverity, CurrentSeverity, Velocity,
                                     _WarningThreshold, InterventionThreshold, _CriticalThreshold)
    when PrevSeverity < InterventionThreshold, CurrentSeverity >= InterventionThreshold ->
    %% Crossed into active intervention
    Event = {task_silo_intervention_started, #{
        realm => Realm,
        source => task_silo,
        timestamp => erlang:system_time(millisecond),
        total_evaluations => TotalEvaluations,
        cohort => Cohort,
        stagnation_severity => CurrentSeverity,
        improvement_velocity => Velocity,
        threshold => InterventionThreshold,
        message => <<"Stagnation detected, boosting exploration">>
    }},
    Topic = neuroevolution_events:topic_intervention(Realm),
    neuroevolution_events:publish(Topic, Event),
    error_logger:info_msg(
        "[task_silo] Evals ~p: INTERVENTION STARTED - severity=~.2f >= ~.2f, velocity=~.4f~n",
        [TotalEvaluations, CurrentSeverity, InterventionThreshold, Velocity]
    );

emit_velocity_event_with_thresholds(Realm, TotalEvaluations, Cohort, PrevSeverity, CurrentSeverity, Velocity,
                                     _WarningThreshold, _InterventionThreshold, CriticalThreshold)
    when PrevSeverity < CriticalThreshold, CurrentSeverity >= CriticalThreshold ->
    %% Crossed into critical intervention
    Event = {task_silo_intervention_critical, #{
        realm => Realm,
        source => task_silo,
        timestamp => erlang:system_time(millisecond),
        total_evaluations => TotalEvaluations,
        cohort => Cohort,
        stagnation_severity => CurrentSeverity,
        improvement_velocity => Velocity,
        threshold => CriticalThreshold,
        message => <<"Critical stagnation, maximum exploration boost">>
    }},
    Topic = neuroevolution_events:topic_intervention(Realm),
    neuroevolution_events:publish(Topic, Event),
    error_logger:warning_msg(
        "[task_silo] Evals ~p: CRITICAL INTERVENTION - severity=~.2f >= ~.2f, velocity=~.4f~n",
        [TotalEvaluations, CurrentSeverity, CriticalThreshold, Velocity]
    );

emit_velocity_event_with_thresholds(Realm, TotalEvaluations, Cohort, PrevSeverity, CurrentSeverity, Velocity,
                                     WarningThreshold, _InterventionThreshold, _CriticalThreshold)
    when PrevSeverity >= WarningThreshold, CurrentSeverity < WarningThreshold ->
    %% Recovered to healthy
    Event = {task_silo_intervention_ended, #{
        realm => Realm,
        source => task_silo,
        timestamp => erlang:system_time(millisecond),
        total_evaluations => TotalEvaluations,
        cohort => Cohort,
        stagnation_severity => CurrentSeverity,
        improvement_velocity => Velocity,
        threshold => WarningThreshold,
        message => <<"Recovered from stagnation, normal operation resumed">>
    }},
    Topic = neuroevolution_events:topic_intervention(Realm),
    neuroevolution_events:publish(Topic, Event),
    error_logger:info_msg(
        "[task_silo] Evals ~p: INTERVENTION ENDED - severity=~.2f < ~.2f, velocity=~.4f~n",
        [TotalEvaluations, CurrentSeverity, WarningThreshold, Velocity]
    );

emit_velocity_event_with_thresholds(_Realm, _TotalEvaluations, _Cohort, _Prev, _Current, _Velocity,
                                     _WarningThreshold, _InterventionThreshold, _CriticalThreshold) ->
    %% Other changes (severity changed but no threshold crossing)
    ok.

%%% ============================================================================
%%% Internal Functions - L2 Guidance Helpers
%%% ============================================================================

%% @private Query meta_controller for L2 guidance if L2 is enabled.
%%
%% When L2 is enabled:
%%   1. Check if meta_controller is registered and alive
%%   2. Call meta_controller:get_l1_guidance/2 with generation stats
%%   3. Update state with new L2 guidance
%%
%% When L2 is disabled or meta_controller unavailable:
%%   Return state unchanged (using existing defaults)
maybe_query_l2_guidance(_Stats, #state{l2_enabled = false} = State) ->
    State;
maybe_query_l2_guidance(Stats, #state{l2_enabled = true, l2_guidance = OldGuidance} = State) ->
    case whereis(meta_controller) of
        undefined ->
            %% meta_controller not running, use defaults
            error_logger:warning_msg("[task_silo] L2 enabled but meta_controller not found, using defaults~n"),
            State;
        Pid when is_pid(Pid) ->
            try
                NewGuidance = meta_controller:get_l1_guidance(Pid, Stats),
                %% Log significant changes
                case significant_guidance_change(OldGuidance, NewGuidance) of
                    true ->
                        error_logger:info_msg("[task_silo] L2 guidance updated: aggression=~.2f, exploration_step=~.2f~n",
                            [NewGuidance#l2_guidance.aggression_factor,
                             NewGuidance#l2_guidance.exploration_step]);
                    false ->
                        ok
                end,
                State#state{l2_guidance = NewGuidance}
            catch
                _:Reason ->
                    error_logger:warning_msg("[task_silo] Failed to query L2 guidance: ~p~n", [Reason]),
                    State
            end
    end.

%% @private Check if L2 guidance changed significantly (for logging).
significant_guidance_change(Old, New) ->
    %% Check if any parameter changed by more than 10%
    abs(Old#l2_guidance.aggression_factor - New#l2_guidance.aggression_factor) > 0.1 orelse
    abs(Old#l2_guidance.exploration_step - New#l2_guidance.exploration_step) > 0.05 orelse
    abs(Old#l2_guidance.topology_aggression - New#l2_guidance.topology_aggression) > 0.2.

%%% ============================================================================
%%% Internal Functions - LC Silo Chain (3-level TWEANN) Helpers
%%% ============================================================================

%% @private Start lc_silo_chain if TWEANN mode is enabled.
%%
%% The silo chain manages 3 chained LTC TWEANNs:
%% - L2 (strategic): 21 inputs → 5 outputs (L1 hyperparameters)
%% - L1 (tactical): 21 + 5 = 26 inputs → 8 outputs (L0 hyperparameters)
%% - L0 (reactive): 21 + 8 = 29 inputs → 16 outputs (evolution hyperparameters)
maybe_start_lc_silo_chain(false, _Config) ->
    ok;
maybe_start_lc_silo_chain(true, Config) ->
    case whereis(lc_task_silo_chain) of
        undefined ->
            %% Start lc_silo_chain with config
            ChainConfig = #{
                silo_type => task,
                %% Time constants (in generations/evaluations)
                tau_l0 => maps:get(tau_l0, Config, 10),   %% Fast: every 10 updates
                tau_l1 => maps:get(tau_l1, Config, 50),   %% Medium: every 50 updates
                tau_l2 => maps:get(tau_l2, Config, 100),  %% Slow: every 100 updates
                %% L2 topology: 21 training metrics → 5 L1 hyperparameters
                l2_topology => {21, [16, 8], 5},
                %% L1 topology: 21 metrics + 5 L2 outputs = 26 → 8 L0 hyperparameters
                l1_topology => {26, [24, 12], 8},
                %% L0 topology: 21 metrics + 8 L1 outputs = 29 → 16 evolution hyperparameters
                l0_topology => {29, [32, 16], 16},
                %% Learning parameters
                learning_rate => maps:get(lc_learning_rate, Config, 0.1),
                noise_std => maps:get(lc_noise_std, Config, 0.1),
                noise_decay => maps:get(lc_noise_decay, Config, 0.999)
            },
            case lc_silo_chain:start_link(lc_task_silo_chain, ChainConfig) of
                {ok, _Pid} ->
                    error_logger:info_msg("[task_silo] Started lc_silo_chain for 3-level TWEANN mode~n");
                {error, Reason} ->
                    error_logger:error_msg("[task_silo] Failed to start lc_silo_chain: ~p~n", [Reason])
            end;
        _Pid ->
            error_logger:info_msg("[task_silo] lc_silo_chain already running~n")
    end.

%% @private Report reward to lc_silo_chain for online learning.
%%
%% Called after each update to provide the LC TWEANNs with feedback.
%% The reward is computed from improvement velocity and resource usage.
%% All 3 levels (L0, L1, L2) receive the reward for their own ES-style learning.
maybe_report_reward_to_lc_silo_chain(false, _State) ->
    ok;
maybe_report_reward_to_lc_silo_chain(true, State) ->
    case whereis(lc_task_silo_chain) of
        undefined ->
            error_logger:warning_msg("[task_silo] lc_task_silo_chain NOT FOUND - rewards not being reported~n"),
            ok;
        Pid when is_pid(Pid) ->
            %% Compute reward signal based on lc_reward module
            Metrics = build_reward_metrics(State),
            Reward = lc_reward:compute_task_reward(Metrics),
            %% Log detailed metrics for debugging
            StagnSev = maps:get(stagnation_severity, Metrics, 0.0),
            Velocity = maps:get(improvement_velocity, Metrics, 0.0),
            error_logger:info_msg("[task_silo] Reporting reward=~.3f (stag=~.2f, vel=~.3f) to lc_silo_chain~n",
                                  [Reward, StagnSev, Velocity]),
            lc_silo_chain:report_reward(Pid, Reward)
    end.

%% @private Build metrics map for reward computation.
build_reward_metrics(State) ->
    #state{
        fitness_history = FitnessHistory,
        velocity_state = VelocityState,
        total_evaluations = TotalEvals,
        diversity_index = DiversityIndex
    } = State,

    BestFitness = case FitnessHistory of
        [F | _] -> F;
        [] -> 0.0
    end,

    %% Compute dynamic fitness target based on observed history
    %% Use max fitness * 1.5 as target (always leave room to improve)
    FitnessTarget = case VelocityState#velocity_state.fitness_checkpoints of
        [] ->
            max(1.0, BestFitness * 1.5);
        Checkpoints ->
            MaxHistoric = lists:max([Fitness || {_, Fitness} <- Checkpoints]),
            max(1.0, MaxHistoric * 1.5)
    end,

    %% Get resource pressure from resource_l0_sensors
    {MemoryPressure, CpuPressure} = get_resource_pressures(),

    #{
        improvement_velocity => VelocityState#velocity_state.current_velocity,
        best_fitness => BestFitness,
        fitness_target => FitnessTarget,
        evaluations_used => TotalEvals,
        evaluations_budget => 100000,  %% Typical budget
        stagnation_severity => VelocityState#velocity_state.stagnation_severity,
        diversity_index => DiversityIndex,
        memory_pressure => MemoryPressure,
        cpu_pressure => CpuPressure
    }.

%% @private Get resource pressures from resource_l0_sensors.
get_resource_pressures() ->
    try
        Sensors = resource_l0_sensors:get_sensors(),
        MemPressure = maps:get(memory_pressure, Sensors, 0.0),
        CpuPressure = maps:get(cpu_pressure, Sensors, 0.0),
        {MemPressure, CpuPressure}
    catch
        _:_ -> {0.0, 0.0}  %% Fallback if sensors not available
    end.

%%% ============================================================================
%%% Internal Functions - Event-Driven Recommendations
%%% ============================================================================

%% @private Publish recommendations when they change significantly.
%%
%% This implements the "cached read model" pattern for event-driven architecture.
%% Recommendations are published when key parameters change by more than 5%.
%%
%% Subscribers cache the latest recommendations locally and use them
%% instead of making blocking get_recommendations() calls.
maybe_publish_recommendations(OldParams, NewParams) ->
    %% Check if key parameters changed significantly
    OldMR = maps:get(mutation_rate, OldParams, 0.0),
    NewMR = maps:get(mutation_rate, NewParams, 0.0),
    OldMS = maps:get(mutation_strength, OldParams, 0.0),
    NewMS = maps:get(mutation_strength, NewParams, 0.0),

    MutationRateChanged = OldMR > 0 andalso abs(NewMR - OldMR) / OldMR > 0.05,
    MutationStrengthChanged = OldMS > 0 andalso abs(NewMS - OldMS) / OldMS > 0.05,

    case MutationRateChanged orelse MutationStrengthChanged of
        true ->
            silo_events:publish_recommendations(task, NewParams);
        false ->
            ok
    end.
