%% @doc Task Silo L0 Sensors - Collects and normalizes sensor inputs for TWEANN.
%%
%% Part of the Liquid Conglomerate v2 architecture. This module collects
%% evolution statistics and transforms them into normalized inputs for the
%% L0 hyperparameter-tuning TWEANN.
%%
%% == Responsibilities ==
%%
%% 1. Collect evolution statistics from neuroevolution_server
%% 2. Compute derived metrics (velocities, trends)
%% 3. Normalize all values to appropriate ranges for TWEANN input
%% 4. Handle cross-silo input signals
%% 5. Maintain state for velocity and trend calculations
%%
%% == Usage ==
%%
%% %% Start the sensor collector
%% {ok, Pid} = task_l0_sensors:start_link(Config),
%%
%% %% Update with new evolution stats (called by neuroevolution_server)
%% task_l0_sensors:update_stats(Pid, Stats),
%%
%% %% Get current sensor vector (ordered list)
%% SensorVector = task_l0_sensors:get_sensor_vector(Pid),
%% %% Returns: [0.82, 0.65, 0.12, ...] (16 values)
%%
%% @author R.G. Lefever
%% @copyright 2024-2026 R.G. Lefever
-module(task_l0_sensors).
-behaviour(gen_server).

%% API
-export([
    start_link/0,
    start_link/1,
    get_sensor_vector/0,
    get_sensor_vector/1,
    get_sensors/0,
    get_sensors/1,
    update_stats/1,
    update_stats/2,
    update_cross_silo_signal/2,
    update_mutation_impact/1,
    update_crossover_success/1,
    update_archive_stats/1,
    set_realm/1
]).

%% gen_server callbacks
-export([
    init/1,
    handle_call/3,
    handle_cast/2,
    handle_info/2,
    terminate/2
]).

-define(SERVER, ?MODULE).
-define(VELOCITY_WINDOW_SIZE, 10).  % Stats samples for velocity
-define(DEFAULT_MAX_EVALS, 100000). % Default max evaluations for progress
-define(MIN_PUBLISH_INTERVAL_MS, 100). % Max 10Hz event publishing
-define(CHANGE_THRESHOLD, 0.01).  % Minimum change to trigger event

-record(state, {
    %% Configuration
    realm :: binary(),  % Realm for event publishing
    max_evaluations :: pos_integer(),
    max_species :: pos_integer(),
    max_complexity :: float(),
    max_archive_size :: pos_integer(),

    %% Current sensor values
    current_sensors :: map(),

    %% History for velocity/trend calculations
    fitness_history :: [float()],
    complexity_history :: [float()],
    diversity_history :: [float()],

    %% Latest raw stats
    current_stats :: map(),

    %% Cross-silo signals
    resource_pressure_signal :: float(),

    %% Mutation/crossover tracking
    mutation_impacts :: [float()],   % Recent mutation fitness changes
    crossover_successes :: [boolean()], % Recent crossover outcomes

    %% Progress tracking
    total_evaluations :: non_neg_integer(),

    %% Archive stats
    archive_stats :: map(),  % Archive metrics

    %% Event publishing state
    last_published_sensors :: map(),  % For change detection
    last_publish_time :: integer()  % For throttling (milliseconds)
}).

%%% ============================================================================
%%% API Functions
%%% ============================================================================

%% @doc Start the sensor collector with default configuration.
-spec start_link() -> {ok, pid()} | ignore | {error, term()}.
start_link() ->
    start_link(#{}).

%% @doc Start the sensor collector with custom configuration.
-spec start_link(map()) -> {ok, pid()} | ignore | {error, term()}.
start_link(Config) ->
    gen_server:start_link({local, ?SERVER}, ?MODULE, Config, []).

%% @doc Get ordered sensor vector for TWEANN input.
%%
%% Returns a list of 21 float values in the order defined by
%% task_l0_morphology:sensor_names/0.
-spec get_sensor_vector() -> [float()].
get_sensor_vector() ->
    gen_server:call(?SERVER, get_sensor_vector).

%% @doc Get ordered sensor vector from specific server.
-spec get_sensor_vector(pid()) -> [float()].
get_sensor_vector(Pid) ->
    gen_server:call(Pid, get_sensor_vector).

%% @doc Get named sensor map.
-spec get_sensors() -> map().
get_sensors() ->
    gen_server:call(?SERVER, get_sensors).

%% @doc Get named sensor map from specific server.
-spec get_sensors(pid()) -> map().
get_sensors(Pid) ->
    gen_server:call(Pid, get_sensors).

%% @doc Update with new evolution statistics.
-spec update_stats(map()) -> ok.
update_stats(Stats) ->
    gen_server:cast(?SERVER, {update_stats, Stats}).

%% @doc Update with new evolution statistics (specific server).
-spec update_stats(pid(), map()) -> ok.
update_stats(Pid, Stats) ->
    gen_server:cast(Pid, {update_stats, Stats}).

%% @doc Update cross-silo signal from another silo.
-spec update_cross_silo_signal(atom(), float()) -> ok.
update_cross_silo_signal(SignalName, Value) ->
    gen_server:cast(?SERVER, {cross_silo_signal, SignalName, Value}).

%% @doc Record a mutation impact (fitness change).
-spec update_mutation_impact(float()) -> ok.
update_mutation_impact(FitnessChange) ->
    gen_server:cast(?SERVER, {mutation_impact, FitnessChange}).

%% @doc Record a crossover outcome (success or failure).
-spec update_crossover_success(boolean()) -> ok.
update_crossover_success(Success) ->
    gen_server:cast(?SERVER, {crossover_success, Success}).

%% @doc Update archive statistics.
%%
%% Expected keys in Stats map:
%% - size: current archive size
%% - max_size: maximum archive size
%% - avg_fitness: average fitness in archive
%% - fitness_variance: fitness variance in archive
%% - avg_generation: average generation of entries (for staleness)
-spec update_archive_stats(map()) -> ok.
update_archive_stats(Stats) ->
    gen_server:cast(?SERVER, {archive_stats, Stats}).

%% @doc Set the realm for event publishing.
%%
%% Use this to dynamically update the realm when a training session starts.
%% Events will be published to the topic for the specified realm.
-spec set_realm(binary()) -> ok.
set_realm(Realm) when is_binary(Realm) ->
    gen_server:cast(?SERVER, {set_realm, Realm}).

%%% ============================================================================
%%% gen_server Callbacks
%%% ============================================================================

init(Config) ->
    Realm = maps:get(realm, Config, <<"default">>),
    MaxEvaluations = maps:get(max_evaluations, Config, ?DEFAULT_MAX_EVALS),
    MaxSpecies = maps:get(max_species, Config, 20),
    MaxComplexity = maps:get(max_complexity, Config, 100.0),
    MaxArchiveSize = maps:get(max_archive_size, Config, 100),
    InitialSensors = initial_sensors(),

    State = #state{
        realm = Realm,
        max_evaluations = MaxEvaluations,
        max_species = MaxSpecies,
        max_complexity = MaxComplexity,
        max_archive_size = MaxArchiveSize,
        current_sensors = InitialSensors,
        fitness_history = [],
        complexity_history = [],
        diversity_history = [],
        current_stats = #{},
        resource_pressure_signal = 0.0,
        mutation_impacts = [],
        crossover_successes = [],
        total_evaluations = 0,
        archive_stats = #{},
        last_published_sensors = InitialSensors,
        last_publish_time = 0
    },

    {ok, State}.

handle_call(get_sensor_vector, _From, State) ->
    Vector = sensors_to_vector(State#state.current_sensors),
    {reply, Vector, State};

handle_call(get_sensors, _From, State) ->
    {reply, State#state.current_sensors, State};

handle_call(_Request, _From, State) ->
    {reply, {error, unknown_request}, State}.

handle_cast({update_stats, Stats}, State) ->
    NewState = process_stats(Stats, State),
    {noreply, NewState};

handle_cast({cross_silo_signal, resource_pressure_signal, Value}, State) ->
    NewSensors = maps:put(resource_pressure_signal, clamp(Value, 0.0, 1.0),
                          State#state.current_sensors),
    {noreply, State#state{
        resource_pressure_signal = clamp(Value, 0.0, 1.0),
        current_sensors = NewSensors
    }};

handle_cast({mutation_impact, Change}, State) ->
    NewImpacts = lists:sublist([Change | State#state.mutation_impacts], 100),
    {noreply, State#state{mutation_impacts = NewImpacts}};

handle_cast({crossover_success, Success}, State) ->
    NewSuccesses = lists:sublist([Success | State#state.crossover_successes], 100),
    {noreply, State#state{crossover_successes = NewSuccesses}};

handle_cast({archive_stats, ArchiveStats}, State) ->
    %% Compute archive sensor values from stats
    ArchiveSensors = compute_archive_sensors(ArchiveStats, State),
    NewSensors = maps:merge(State#state.current_sensors, ArchiveSensors),
    {noreply, State#state{archive_stats = ArchiveStats, current_sensors = NewSensors}};

handle_cast({set_realm, Realm}, State) ->
    error_logger:info_msg("[task_l0_sensors] Realm updated to: ~p~n", [Realm]),
    {noreply, State#state{realm = Realm}};

handle_cast(_Msg, State) ->
    {noreply, State}.

handle_info(_Info, State) ->
    {noreply, State}.

terminate(_Reason, _State) ->
    ok.

%%% ============================================================================
%%% Internal Functions - Stats Processing
%%% ============================================================================

%% @private Process new stats and update sensors.
process_stats(Stats, State) ->
    %% Extract key values from stats
    BestFitness = maps:get(best_fitness, Stats, 0.0),
    AvgFitness = maps:get(avg_fitness, Stats, 0.0),
    FitnessVariance = maps:get(fitness_variance, Stats, 0.0),
    SpeciesCount = maps:get(species_count, Stats, 1),
    AvgComplexity = maps:get(avg_complexity, Stats, 0.0),
    DiversityIndex = maps:get(diversity_index, Stats, 0.5),
    TotalEvals = maps:get(total_evaluations, Stats, State#state.total_evaluations),

    %% Normalize fitness values (assume 0-1000 typical range, normalize to 0-1)
    FitnessNormFactor = maps:get(fitness_norm_factor, Stats, 1000.0),
    BestFitnessNorm = clamp(BestFitness / max(1.0, FitnessNormFactor), 0.0, 1.0),
    AvgFitnessNorm = clamp(AvgFitness / max(1.0, FitnessNormFactor), 0.0, 1.0),

    %% Update histories
    NewFitnessHistory = update_history(BestFitnessNorm, State#state.fitness_history),
    NewComplexityHistory = update_history(AvgComplexity, State#state.complexity_history),
    NewDiversityHistory = update_history(DiversityIndex, State#state.diversity_history),

    %% Compute velocities
    ImprovementVelocity = compute_velocity(NewFitnessHistory),
    ComplexityVelocity = compute_velocity(NewComplexityHistory),

    %% Compute stagnation severity
    StagnationSeverity = compute_stagnation(NewFitnessHistory),

    %% Species count ratio
    MaxSpecies = State#state.max_species,
    SpeciesCountRatio = clamp(SpeciesCount / max(1, MaxSpecies), 0.0, 1.0),

    %% Average complexity normalized
    MaxComplexity = State#state.max_complexity,
    AvgComplexityNorm = clamp(AvgComplexity / max(1.0, MaxComplexity), 0.0, 1.0),

    %% Elite dominance (best/avg ratio, normalized)
    EliteDominance = case AvgFitness > 0.0 of
        true -> clamp((BestFitness / AvgFitness - 1.0) / 2.0, 0.0, 1.0);
        false -> 0.0
    end,

    %% Fitness variance normalized
    FitnessVarianceNorm = clamp(FitnessVariance / max(1.0, FitnessNormFactor), 0.0, 1.0),

    %% Crossover success rate
    CrossoverSuccessRate = compute_crossover_rate(State#state.crossover_successes),

    %% Mutation impact
    MutationImpact = compute_mutation_impact(State#state.mutation_impacts),

    %% Evaluation progress
    MaxEvals = State#state.max_evaluations,
    EvalProgress = clamp(TotalEvals / max(1, MaxEvals), 0.0, 1.0),

    %% Population entropy (approximated from fitness variance and diversity)
    Entropy = clamp((FitnessVarianceNorm + DiversityIndex) / 2.0, 0.0, 1.0),

    %% Convergence trend (negative velocity = converging)
    ConvergenceTrend = compute_convergence_trend(NewDiversityHistory),

    %% Compute reward factors using lc_reward module
    RewardMetrics = #{
        improvement_velocity => ImprovementVelocity,
        stagnation_severity => StagnationSeverity,
        fitness_ratio => BestFitnessNorm,
        eval_ratio => EvalProgress,
        memory_pressure => 0.0,  % Will be updated by resource_l0_sensors cross-silo signal
        cpu_pressure => 0.0     % Will be updated by resource_l0_sensors cross-silo signal
    },
    RewardBreakdown = lc_reward:compute_task_reward_breakdown(RewardMetrics),

    %% Build sensor map
    NewSensors = #{
        best_fitness => BestFitnessNorm,
        avg_fitness => AvgFitnessNorm,
        fitness_variance => FitnessVarianceNorm,
        improvement_velocity => ImprovementVelocity,
        stagnation_severity => StagnationSeverity,
        diversity_index => DiversityIndex,
        species_count_ratio => SpeciesCountRatio,
        avg_network_complexity => AvgComplexityNorm,
        complexity_velocity => ComplexityVelocity,
        elite_dominance => EliteDominance,
        crossover_success_rate => CrossoverSuccessRate,
        mutation_impact => MutationImpact,
        resource_pressure_signal => State#state.resource_pressure_signal,
        evaluation_progress => EvalProgress,
        entropy => Entropy,
        convergence_trend => ConvergenceTrend,
        %% Reward factors from lc_reward
        total_reward => maps:get(total_reward, RewardBreakdown, 0.5),
        velocity_score => maps:get(velocity_score, RewardBreakdown, 0.5),
        efficiency_score => maps:get(efficiency_score, RewardBreakdown, 0.5),
        convergence_score => maps:get(convergence_score, RewardBreakdown, 1.0),
        resource_score => maps:get(resource_score, RewardBreakdown, 1.0)
    },

    UpdatedState = State#state{
        current_sensors = NewSensors,
        current_stats = Stats,
        fitness_history = NewFitnessHistory,
        complexity_history = NewComplexityHistory,
        diversity_history = NewDiversityHistory,
        total_evaluations = TotalEvals
    },
    %% Publish event if sensors changed significantly
    maybe_publish_sensors_event(UpdatedState).

%% @private Compute stagnation severity from fitness history.
compute_stagnation([]) -> 0.0;
compute_stagnation(History) when length(History) < 3 -> 0.0;
compute_stagnation(History) ->
    %% Count how many recent samples showed no improvement
    Recent = lists:sublist(History, ?VELOCITY_WINDOW_SIZE),
    case Recent of
        [Latest | Rest] ->
            NoImprove = length([X || X <- Rest, X >= Latest - 0.001]),
            clamp(NoImprove / max(1, length(Rest)), 0.0, 1.0);
        [] -> 0.0
    end.

%% @private Compute convergence trend from diversity history.
compute_convergence_trend([]) -> 0.0;
compute_convergence_trend([_]) -> 0.0;
compute_convergence_trend(History) ->
    %% Negative = diversity decreasing = converging
    %% Positive = diversity increasing = diverging
    Velocity = compute_velocity(History),
    -Velocity.  % Invert: low diversity velocity = converging = positive trend

%% @private Compute crossover success rate.
compute_crossover_rate([]) -> 0.5;  % Default to neutral
compute_crossover_rate(Successes) ->
    SuccessCount = length([X || X <- Successes, X =:= true]),
    clamp(SuccessCount / max(1, length(Successes)), 0.0, 1.0).

%% @private Compute average mutation impact.
compute_mutation_impact([]) -> 0.5;  % Default to neutral
compute_mutation_impact(Impacts) ->
    %% Average absolute impact, normalized
    AvgAbsImpact = lists:sum([abs(X) || X <- Impacts]) / max(1, length(Impacts)),
    clamp(AvgAbsImpact, 0.0, 1.0).

%%% ============================================================================
%%% Internal Functions - Utilities
%%% ============================================================================

%% @private Initial sensor values.
initial_sensors() ->
    #{
        %% Evolution sensors
        best_fitness => 0.0,
        avg_fitness => 0.0,
        fitness_variance => 0.5,
        improvement_velocity => 0.0,
        stagnation_severity => 0.0,
        diversity_index => 0.5,
        species_count_ratio => 0.1,
        avg_network_complexity => 0.1,
        complexity_velocity => 0.0,
        elite_dominance => 0.0,
        crossover_success_rate => 0.5,
        mutation_impact => 0.5,
        resource_pressure_signal => 0.0,
        evaluation_progress => 0.0,
        entropy => 0.5,
        convergence_trend => 0.0,
        %% Self-play archive sensors
        archive_fill_ratio => 0.0,
        archive_fitness_mean => 0.0,
        archive_fitness_variance => 0.5,
        archive_staleness => 0.0,
        population_vs_archive_ratio => 0.5,
        %% Reward factors (LC v2 Task Silo reward breakdown)
        total_reward => 0.5,
        velocity_score => 0.5,
        efficiency_score => 0.5,
        convergence_score => 1.0,
        resource_score => 1.0
    }.

%% @private Update history with new value.
update_history(Value, History) ->
    lists:sublist([Value | History], ?VELOCITY_WINDOW_SIZE).

%% @private Compute velocity from history.
compute_velocity([]) -> 0.0;
compute_velocity([_]) -> 0.0;
compute_velocity([Latest, Previous | _]) ->
    Diff = (Latest - Previous) * 5.0,  % Scale for visibility
    clamp(Diff, -1.0, 1.0).

%% @private Convert sensor map to ordered vector.
sensors_to_vector(Sensors) ->
    SensorNames = task_l0_morphology:sensor_names(),
    [maps:get(Name, Sensors, 0.0) || Name <- SensorNames].

%% @private Compute archive sensors from archive stats.
compute_archive_sensors(ArchiveStats, State) ->
    MaxArchiveSize = State#state.max_archive_size,
    Size = maps:get(size, ArchiveStats, 0),
    MaxSize = maps:get(max_size, ArchiveStats, MaxArchiveSize),
    AvgFitness = maps:get(avg_fitness, ArchiveStats, 0.0),
    FitnessVariance = maps:get(fitness_variance, ArchiveStats, 0.0),
    AvgGeneration = maps:get(avg_generation, ArchiveStats, 0),
    CurrentGen = maps:get(current_generation, ArchiveStats,
                          State#state.total_evaluations div 100),
    PopFitness = maps:get(avg_fitness, State#state.current_stats, 0.0),

    %% Compute derived values
    FillRatio = clamp(Size / max(1, MaxSize), 0.0, 1.0),

    %% Fitness mean normalized (assume 0-1000 range, normalize to 0-1)
    FitnessNormFactor = maps:get(fitness_norm_factor, ArchiveStats, 1000.0),
    FitnessMeanNorm = clamp(AvgFitness / max(1.0, FitnessNormFactor), 0.0, 1.0),

    %% Fitness variance normalized
    FitnessVarNorm = clamp(FitnessVariance / max(1.0, FitnessNormFactor), 0.0, 1.0),

    %% Staleness: higher = older entries
    Staleness = case CurrentGen > 0 of
        true ->
            AvgAge = max(0, CurrentGen - AvgGeneration),
            clamp(AvgAge / max(1, CurrentGen) * 2.0, 0.0, 1.0);
        false -> 0.0
    end,

    %% Population vs archive ratio (arms race indicator)
    %% >0.5 = population better than archive, <0.5 = archive better
    PopVsArchive = case AvgFitness > 0.0 of
        true -> clamp(PopFitness / (AvgFitness * 2.0), 0.0, 1.0);
        false -> 0.5
    end,

    #{
        archive_fill_ratio => FillRatio,
        archive_fitness_mean => FitnessMeanNorm,
        archive_fitness_variance => FitnessVarNorm,
        archive_staleness => Staleness,
        population_vs_archive_ratio => PopVsArchive
    }.

%% @private Clamp value to range.
clamp(Value, Min, Max) ->
    max(Min, min(Max, Value)).

%%% ============================================================================
%%% Internal Functions - Event Publishing
%%% ============================================================================

%% @private Publish sensors event if significant change or enough time passed.
maybe_publish_sensors_event(State) ->
    Now = erlang:system_time(millisecond),
    TimeSinceLastPublish = Now - State#state.last_publish_time,
    CurrentSensors = State#state.current_sensors,
    LastPublished = State#state.last_published_sensors,

    ShouldPublish = (TimeSinceLastPublish >= ?MIN_PUBLISH_INTERVAL_MS) andalso
                    sensors_changed_significantly(CurrentSensors, LastPublished),

    case ShouldPublish of
        true ->
            publish_sensors_event(State#state.realm, CurrentSensors),
            State#state{
                last_published_sensors = CurrentSensors,
                last_publish_time = Now
            };
        false ->
            State
    end.

%% @private Check if any sensor changed by more than threshold.
sensors_changed_significantly(Current, Last) ->
    maps:fold(
        fun(Key, Value, Acc) ->
            case Acc of
                true -> true;
                false ->
                    OldValue = maps:get(Key, Last, 0.0),
                    abs(Value - OldValue) > ?CHANGE_THRESHOLD
            end
        end,
        false,
        Current
    ).

%% @private Publish task_sensors_updated event.
publish_sensors_event(Realm, Sensors) ->
    Topic = neuroevolution_events:topic_silo_sensors(Realm),
    Event = {task_sensors_updated, #{
        realm => Realm,
        source => task_l0_sensors,
        timestamp => erlang:system_time(millisecond),
        sensors => Sensors
    }},
    neuroevolution_events:publish(Topic, Event).
