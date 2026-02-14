%% @doc LC Population Manager - Evolves Liquid Conglomerate controller TWEANNs.
%%
%% This gen_server manages a population of LC controller neural networks that
%% compete to be the active controller. Each controller is evaluated over a
%% "trial period" (e.g., 5000 evaluations) and rewarded based on:
%%
%% 1. Fitness improvement velocity (faster learning = better)
%% 2. Training efficiency (fewer evaluations to reach target)
%% 3. Avoiding premature convergence (not getting stuck at low fitness)
%% 4. Resource efficiency (low memory/CPU usage)
%%
%% == Evolution Strategy ==
%%
%% - Population of N LC TWEANNs (default: 10)
%% - Each generation: trial period for each controller
%% - Active controller is the champion from last generation
%% - After each trial, compute reward using lc_reward:compute_trial_reward/1
%% - Select top 50% as parents, mutate to create offspring
%% - Repeat
%%
%% == Time Constants ==
%%
%% - Trial period: 5000 evaluations (tau_l0 * 5)
%% - Generation: N * trial_period evaluations
%% - Champion persistence: 3 generations (give good controllers time)
%%
%% @author R.G. Lefever
%% @copyright 2024-2026 R.G. Lefever
-module(lc_population).
-behaviour(gen_server).

-include_lib("faber_tweann/include/records.hrl").

%% API
-export([
    start_link/0,
    start_link/1,
    get_active_controller/1,
    get_recommendations/2,
    report_metrics/2,
    get_state/1,
    reset/1
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

%% Default configuration
-define(DEFAULT_POPULATION_SIZE, 10).
-define(DEFAULT_TRIAL_PERIOD, 5000).     %% evaluations per trial
-define(DEFAULT_SURVIVAL_RATE, 0.5).     %% top 50% survive
-define(CHAMPION_PERSISTENCE, 3).        %% generations before champion can change

-record(state, {
    %% Configuration
    population_size :: pos_integer(),
    trial_period :: pos_integer(),
    survival_rate :: float(),
    morphology :: atom(),

    %% Population
    agents :: [term()],                  %% List of agent IDs
    fitness_scores :: #{term() => float()}, %% Agent -> cumulative reward
    generation :: non_neg_integer(),

    %% Active controller
    active_agent :: term() | undefined,  %% Currently active LC TWEANN
    active_exoself :: pid() | undefined, %% Running phenotype process
    champion_tenure :: non_neg_integer(),%% Generations as champion

    %% Trial state
    trial_start_fitness :: float(),
    trial_start_evals :: non_neg_integer(),
    trial_metrics_acc :: [map()],        %% Accumulated metrics during trial
    trial_stagnation_events :: non_neg_integer(),
    peak_memory :: float(),
    avg_cpu_acc :: float(),
    cpu_samples :: non_neg_integer(),

    %% Overall tracking
    total_evaluations :: non_neg_integer(),
    best_fitness_ever :: float()
}).

%%% ============================================================================
%%% API
%%% ============================================================================

%% @doc Start LC population manager with default config.
-spec start_link() -> {ok, pid()} | {error, term()}.
start_link() ->
    start_link(#{}).

%% @doc Start LC population manager with custom config.
-spec start_link(map()) -> {ok, pid()} | {error, term()}.
start_link(Config) ->
    gen_server:start_link({local, ?SERVER}, ?MODULE, Config, []).

%% @doc Get the active LC controller's recommendations.
%% This is called by task_silo to get hyperparameters from the LC TWEANN.
-spec get_active_controller(pid()) -> {ok, pid()} | {error, no_controller}.
get_active_controller(Pid) ->
    gen_server:call(Pid, get_active_controller).

%% @doc Get hyperparameter recommendations from the active LC TWEANN.
%% Sensors are fed to the network, outputs are scaled to hyperparameter ranges.
-spec get_recommendations(pid(), map()) -> map().
get_recommendations(Pid, SensorInputs) ->
    gen_server:call(Pid, {get_recommendations, SensorInputs}).

%% @doc Report evolution metrics for reward computation.
%% Called by task_silo after each cohort/generation.
-spec report_metrics(pid(), map()) -> ok.
report_metrics(Pid, Metrics) ->
    gen_server:cast(Pid, {report_metrics, Metrics}).

%% @doc Get current population state for monitoring.
-spec get_state(pid()) -> map().
get_state(Pid) ->
    gen_server:call(Pid, get_state).

%% @doc Reset population (start fresh evolution).
-spec reset(pid()) -> ok.
reset(Pid) ->
    gen_server:cast(Pid, reset).

%%% ============================================================================
%%% gen_server Callbacks
%%% ============================================================================

init(Config) ->
    %% Ensure morphologies are registered
    lc_tweann_morphology:register_morphologies(),

    %% Configuration
    PopSize = maps:get(population_size, Config, ?DEFAULT_POPULATION_SIZE),
    TrialPeriod = maps:get(trial_period, Config, ?DEFAULT_TRIAL_PERIOD),
    SurvivalRate = maps:get(survival_rate, Config, ?DEFAULT_SURVIVAL_RATE),
    Morphology = maps:get(morphology, Config, lc_task_controller),

    %% Initialize ETS for genotype storage
    genotype:init_db(),

    %% Create initial population
    Agents = create_initial_population(PopSize, Morphology),

    %% Select random initial champion
    ActiveAgent = hd(Agents),

    State = #state{
        population_size = PopSize,
        trial_period = TrialPeriod,
        survival_rate = SurvivalRate,
        morphology = Morphology,
        agents = Agents,
        fitness_scores = #{},
        generation = 0,
        active_agent = ActiveAgent,
        active_exoself = undefined,
        champion_tenure = 0,
        trial_start_fitness = 0.0,
        trial_start_evals = 0,
        trial_metrics_acc = [],
        trial_stagnation_events = 0,
        peak_memory = 0.0,
        avg_cpu_acc = 0.0,
        cpu_samples = 0,
        total_evaluations = 0,
        best_fitness_ever = 0.0
    },

    %% Start the active controller's phenotype
    NewState = start_active_phenotype(State),

    error_logger:info_msg("[lc_population] Started with ~p agents, morphology=~p~n",
                         [PopSize, Morphology]),

    %% Subscribe to population metrics events (event-driven pattern)
    controller_events:subscribe_to_population_metrics(),

    {ok, NewState}.

handle_call(get_active_controller, _From, #state{active_exoself = undefined} = State) ->
    {reply, {error, no_controller}, State};
handle_call(get_active_controller, _From, #state{active_exoself = Pid} = State) ->
    {reply, {ok, Pid}, State};

handle_call({get_recommendations, SensorInputs}, _From, State) ->
    Recommendations = compute_recommendations(SensorInputs, State),
    {reply, Recommendations, State};

handle_call(get_state, _From, State) ->
    StateMap = #{
        generation => State#state.generation,
        population_size => State#state.population_size,
        active_agent => State#state.active_agent,
        champion_tenure => State#state.champion_tenure,
        total_evaluations => State#state.total_evaluations,
        best_fitness_ever => State#state.best_fitness_ever,
        trial_progress => length(State#state.trial_metrics_acc),
        agents => State#state.agents,
        fitness_scores => State#state.fitness_scores
    },
    {reply, StateMap, State};

handle_call(_Request, _From, State) ->
    {reply, {error, unknown_request}, State}.

handle_cast({report_metrics, Metrics}, State) ->
    %% DEPRECATED: Use controller_events:publish_population_metrics/1 instead
    logger:warning("[DEPRECATED] lc_population:report_metrics/2 called. "
                   "Use controller_events:publish_population_metrics/1 instead."),
    NewState = handle_metrics_report(Metrics, State),
    {noreply, NewState};

handle_cast(reset, State) ->
    %% Stop active phenotype if running
    NewState0 = stop_active_phenotype(State),
    %% Delete all agents
    lists:foreach(fun(AgentId) ->
        genotype:delete_Agent(AgentId)
    end, State#state.agents),
    %% Create fresh population
    Agents = create_initial_population(State#state.population_size, State#state.morphology),
    NewState = NewState0#state{
        agents = Agents,
        fitness_scores = #{},
        generation = 0,
        active_agent = hd(Agents),
        champion_tenure = 0,
        trial_start_fitness = 0.0,
        trial_start_evals = 0,
        trial_metrics_acc = [],
        trial_stagnation_events = 0,
        peak_memory = 0.0,
        avg_cpu_acc = 0.0,
        cpu_samples = 0,
        total_evaluations = 0,
        best_fitness_ever = 0.0
    },
    FinalState = start_active_phenotype(NewState),
    {noreply, FinalState};

handle_cast(_Msg, State) ->
    {noreply, State}.

handle_info({'EXIT', Pid, Reason}, #state{active_exoself = Pid} = State) ->
    error_logger:warning_msg("[lc_population] Active controller exited: ~p~n", [Reason]),
    %% Restart the phenotype
    NewState = start_active_phenotype(State#state{active_exoself = undefined}),
    {noreply, NewState};

%% Handle population metrics event (event-driven pattern)
handle_info({neuro_event, <<"controller.population.metrics">>, #{metrics := Metrics}}, State) ->
    NewState = handle_metrics_report(Metrics, State),
    {noreply, NewState};

handle_info(_Info, State) ->
    {noreply, State}.

terminate(_Reason, State) ->
    stop_active_phenotype(State),
    ok.

%%% ============================================================================
%%% Internal Functions - Population Management
%%% ============================================================================

%% @private Create initial population of LC TWEANNs.
create_initial_population(PopSize, Morphology) ->
    %% Create a constraint for LC controllers
    Constraint = #constraint{
        morphology = Morphology,
        neural_afs = [tanh, sigmoid, relu],
        agent_encoding_types = [neural],
        substrate_plasticities = [none],
        substrate_linkforms = [l2l_feedforward]
    },

    %% Create a species for LC controllers
    %% Species ID format: {RandomFloat, specie} per species_identifier.erl
    SpecieId = {rand:uniform() * 1000000.0, specie},

    %% Create agents
    lists:map(
        fun(_) ->
            AgentId = genotype:generate_id(agent),
            genotype:construct_Agent(SpecieId, AgentId, Constraint),
            AgentId
        end,
        lists:seq(1, PopSize)
    ).

%% @private Start the active controller's phenotype.
start_active_phenotype(#state{active_agent = undefined} = State) ->
    State;
start_active_phenotype(#state{active_agent = AgentId} = State) ->
    %% We use self() as the active_exoself because we run the network directly
    %% in compute_recommendations/2 (synchronous forward pass), rather than
    %% spawning a separate exoself process. This simplifies the architecture
    %% and reduces latency for hyperparameter recommendations.
    error_logger:info_msg("[lc_population] Activated controller ~p~n", [AgentId]),
    State#state{active_exoself = self()}.

%% @private Stop the active phenotype.
stop_active_phenotype(#state{active_exoself = undefined} = State) ->
    State;
stop_active_phenotype(State) ->
    State#state{active_exoself = undefined}.

%%% ============================================================================
%%% Internal Functions - Recommendations (TWEANN Forward Pass)
%%% ============================================================================

%% @private Compute recommendations from the active LC TWEANN.
%%
%% This performs a forward pass through the neural network:
%% 1. Normalize sensor inputs to 0-1 range
%% 2. Feed through network
%% 3. Scale outputs to hyperparameter ranges
compute_recommendations(SensorInputs, #state{active_agent = AgentId}) when AgentId =/= undefined ->
    %% Read agent genotype
    case genotype:read(AgentId) of
        [Agent] ->
            %% Get the cortex
            CortexId = Agent#agent.cx_id,
            [Cortex] = genotype:read(CortexId),

            %% Get sensor and actuator IDs
            [SensorId | _] = Cortex#cortex.sensor_ids,
            [ActuatorId | _] = Cortex#cortex.actuator_ids,

            %% Read sensor and actuator records
            [Sensor] = genotype:read(SensorId),
            [Actuator] = genotype:read(ActuatorId),

            %% Normalize sensor inputs
            NormalizedInputs = normalize_sensor_inputs(SensorInputs, Sensor),

            %% Forward pass through network
            Outputs = forward_pass(NormalizedInputs, Cortex),

            %% Scale outputs to hyperparameter ranges
            scale_outputs(Outputs, Actuator);
        _ ->
            %% Agent not found, return defaults
            task_l0_defaults:get_defaults()
    end;
compute_recommendations(_SensorInputs, _State) ->
    %% No active agent, return defaults
    task_l0_defaults:get_defaults().

%% @private Normalize sensor inputs to 0-1 (or -1 to 1) range.
normalize_sensor_inputs(SensorInputs, _Sensor) ->
    %% Extract values in expected order
    InputNames = [
        best_fitness, avg_fitness, fitness_variance, improvement_velocity,
        stagnation_severity, diversity_index, species_count_ratio,
        avg_network_complexity, complexity_velocity, elite_dominance,
        crossover_success_rate, mutation_impact, resource_pressure_signal,
        evaluation_progress, entropy, convergence_trend,
        archive_fill_ratio, archive_fitness_mean, archive_fitness_variance,
        archive_staleness, population_vs_archive_ratio
    ],

    lists:map(
        fun(Name) ->
            Value = maps:get(Name, SensorInputs, 0.5),
            %% Clamp to valid range
            clamp(Value, -1.0, 1.0)
        end,
        InputNames
    ).

%% @private Forward pass through the neural network.
%%
%% Simplified feedforward implementation for LC controllers.
%% Uses the genotype's weights directly without spawning neuron processes.
forward_pass(Inputs, Cortex) ->
    %% Get all neurons
    NeuronIds = Cortex#cortex.neuron_ids,
    Neurons = lists:map(fun(NId) -> [N] = genotype:read(NId), N end, NeuronIds),

    %% Sort neurons by layer (input to output order)
    SortedNeurons = lists:sort(
        fun(N1, N2) ->
            {{L1, _}, _} = N1#neuron.id,
            {{L2, _}, _} = N2#neuron.id,
            L1 < L2
        end,
        Neurons
    ),

    %% Initialize activations with sensor inputs
    SensorIds = Cortex#cortex.sensor_ids,
    InitialActivations = maps:from_list(
        lists:zip(SensorIds, [Inputs])  %% Sensor outputs the input vector
    ),

    %% Propagate through neurons
    FinalActivations = lists:foldl(
        fun(Neuron, Activations) ->
            process_neuron(Neuron, Activations)
        end,
        InitialActivations,
        SortedNeurons
    ),

    %% Get actuator inputs (outputs of last-layer neurons)
    [ActuatorId | _] = Cortex#cortex.actuator_ids,
    [Actuator] = genotype:read(ActuatorId),
    FaninIds = Actuator#actuator.fanin_ids,

    lists:flatmap(
        fun(NId) ->
            case maps:get(NId, FinalActivations, [0.0]) of
                V when is_list(V) -> V;
                V -> [V]
            end
        end,
        FaninIds
    ).

%% @private Process a single neuron in the forward pass.
process_neuron(Neuron, Activations) ->
    %% Get input weights (input_idps = [{source_id, weights}, ...])
    InputIdps = Neuron#neuron.input_idps,

    %% Compute weighted sum of inputs
    WeightedSum = lists:foldl(
        fun({InputId, Weights}, Acc) ->
            InputValues = maps:get(InputId, Activations, [0.0]),
            %% Weights is [{Weight, DW, LR, Params}, ...]
            WeightValues = extract_weights(Weights),
            %% Dot product
            DotProduct = dot_product(InputValues, WeightValues),
            Acc + DotProduct
        end,
        0.0,
        InputIdps
    ),

    %% Apply activation function
    AF = Neuron#neuron.af,
    Output = apply_activation(AF, WeightedSum),

    %% Store in activations map
    maps:put(Neuron#neuron.id, [Output], Activations).

%% @private Extract weight values from weight tuples.
extract_weights(Weights) when is_list(Weights) ->
    lists:map(
        fun({W, _DW, _LR, _Params}) -> W;
           (W) when is_number(W) -> W
        end,
        Weights
    );
extract_weights(_) ->
    [0.0].

%% @private Dot product of two vectors.
dot_product(A, B) ->
    lists:sum(lists:zipwith(fun(X, Y) -> X * Y end, A, B)).

%% @private Apply activation function.
apply_activation(tanh, X) -> math:tanh(X);
apply_activation(sigmoid, X) -> 1.0 / (1.0 + math:exp(-X));
apply_activation(relu, X) when X > 0 -> X;
apply_activation(relu, _) -> 0.0;
apply_activation(_, X) -> math:tanh(X).  %% Default to tanh

%% @private Scale neural network outputs to hyperparameter ranges.
scale_outputs(Outputs, Actuator) ->
    %% Get output ranges from actuator parameters
    Params = Actuator#actuator.parameters,
    OutputRanges = maps:get(output_ranges, Params, #{}),
    OutputNames = maps:get(outputs, Params, []),

    %% Scale each output
    ScaledPairs = lists:zipwith(
        fun(Output, Name) ->
            {Min, Max} = maps:get(Name, OutputRanges, {0.0, 1.0}),
            %% Output is 0-1 (from sigmoid), scale to [Min, Max]
            ScaledValue = Min + Output * (Max - Min),
            {Name, ScaledValue}
        end,
        pad_list(Outputs, length(OutputNames), 0.5),
        OutputNames
    ),

    maps:from_list(ScaledPairs).

%% @private Pad list to required length.
pad_list(List, TargetLen, _PadValue) when length(List) >= TargetLen ->
    lists:sublist(List, TargetLen);
pad_list(List, TargetLen, PadValue) ->
    List ++ lists:duplicate(TargetLen - length(List), PadValue).

%%% ============================================================================
%%% Internal Functions - Metrics and Evolution
%%% ============================================================================

%% @private Handle metrics report from task_silo.
handle_metrics_report(Metrics, State) ->
    %% Update trial accumulator
    TotalEvals = maps:get(total_evaluations, Metrics, State#state.total_evaluations),
    BestFitness = maps:get(best_fitness, Metrics, 0.0),
    MemPressure = maps:get(memory_pressure, Metrics, 0.0),
    CpuPressure = maps:get(cpu_pressure, Metrics, 0.0),
    StagnationSeverity = maps:get(stagnation_severity, Metrics, 0.0),

    %% Track metrics
    NewMetricsAcc = [Metrics | State#state.trial_metrics_acc],
    NewPeakMemory = max(State#state.peak_memory, MemPressure),
    NewAvgCpuAcc = State#state.avg_cpu_acc + CpuPressure,
    NewCpuSamples = State#state.cpu_samples + 1,
    NewBestFitness = max(State#state.best_fitness_ever, BestFitness),

    %% Count stagnation events (severity > 0.7)
    NewStagnationEvents = case StagnationSeverity > 0.7 of
        true -> State#state.trial_stagnation_events + 1;
        false -> State#state.trial_stagnation_events
    end,

    NewState = State#state{
        total_evaluations = TotalEvals,
        best_fitness_ever = NewBestFitness,
        trial_metrics_acc = NewMetricsAcc,
        peak_memory = NewPeakMemory,
        avg_cpu_acc = NewAvgCpuAcc,
        cpu_samples = NewCpuSamples,
        trial_stagnation_events = NewStagnationEvents
    },

    %% Check if trial period is complete
    EvalsSinceTrialStart = TotalEvals - State#state.trial_start_evals,
    case EvalsSinceTrialStart >= State#state.trial_period of
        true ->
            complete_trial(BestFitness, NewState);
        false ->
            NewState
    end.

%% @private Complete a trial and potentially advance to next generation.
complete_trial(EndFitness, State) ->
    %% Compute trial reward
    AvgCpu = case State#state.cpu_samples of
        0 -> 0.5;
        N -> State#state.avg_cpu_acc / N
    end,

    TrialMetrics = #{
        start_fitness => State#state.trial_start_fitness,
        end_fitness => EndFitness,
        evaluations_used => State#state.total_evaluations - State#state.trial_start_evals,
        evaluation_budget => State#state.trial_period,
        peak_memory => State#state.peak_memory,
        avg_cpu => AvgCpu,
        stagnation_events => State#state.trial_stagnation_events,
        convergence_reached => EndFitness >= 0.95
    },

    TrialReward = lc_reward:compute_trial_reward(TrialMetrics),

    %% Update fitness score for active agent
    ActiveAgent = State#state.active_agent,
    OldScore = maps:get(ActiveAgent, State#state.fitness_scores, 0.0),
    NewScores = maps:put(ActiveAgent, OldScore + TrialReward, State#state.fitness_scores),

    error_logger:info_msg("[lc_population] Trial complete: agent=~p, reward=~.3f, total=~.3f~n",
                         [ActiveAgent, TrialReward, OldScore + TrialReward]),

    %% Check if all agents have been evaluated this generation
    AllAgents = State#state.agents,
    EvaluatedAgents = maps:keys(NewScores),
    RemainingAgents = AllAgents -- EvaluatedAgents,

    case RemainingAgents of
        [] ->
            %% All agents evaluated - advance generation
            advance_generation(State#state{fitness_scores = NewScores});
        [NextAgent | _] ->
            %% Start trial for next agent
            start_new_trial(NextAgent, State#state{fitness_scores = NewScores})
    end.

%% @private Start a new trial with a different agent.
start_new_trial(AgentId, State) ->
    NewState = stop_active_phenotype(State),
    State2 = NewState#state{
        active_agent = AgentId,
        trial_start_fitness = State#state.best_fitness_ever,
        trial_start_evals = State#state.total_evaluations,
        trial_metrics_acc = [],
        trial_stagnation_events = 0,
        peak_memory = 0.0,
        avg_cpu_acc = 0.0,
        cpu_samples = 0
    },
    start_active_phenotype(State2).

%% @private Advance to next generation after all agents evaluated.
advance_generation(State) ->
    %% Select survivors
    FitnessScores = State#state.fitness_scores,
    Agents = State#state.agents,
    SurvivalRate = State#state.survival_rate,

    %% Sort agents by fitness (descending)
    SortedAgents = lists:sort(
        fun(A1, A2) ->
            maps:get(A1, FitnessScores, 0.0) > maps:get(A2, FitnessScores, 0.0)
        end,
        Agents
    ),

    %% Keep top survivors
    NumSurvivors = max(2, round(length(Agents) * SurvivalRate)),
    Survivors = lists:sublist(SortedAgents, NumSurvivors),
    NonSurvivors = Agents -- Survivors,

    %% Delete non-survivors
    lists:foreach(fun(AgentId) ->
        genotype:delete_Agent(AgentId)
    end, NonSurvivors),

    %% Reproduce to fill population
    NumOffspring = State#state.population_size - length(Survivors),
    Offspring = lists:map(
        fun(_) ->
            %% Select random parent
            ParentId = lists:nth(rand:uniform(length(Survivors)), Survivors),
            %% Clone and mutate
            CloneId = genotype:clone_Agent(ParentId),
            genome_mutator:mutate(CloneId),
            CloneId
        end,
        lists:seq(1, NumOffspring)
    ),

    NewAgents = Survivors ++ Offspring,

    %% Select new champion (best performer)
    [NewChampion | _] = SortedAgents,
    ChampionScore = maps:get(NewChampion, FitnessScores, 0.0),

    error_logger:info_msg("[lc_population] Generation ~p complete: champion=~p (score=~.3f)~n",
                         [State#state.generation + 1, NewChampion, ChampionScore]),

    %% Start new generation
    NewState = State#state{
        agents = NewAgents,
        fitness_scores = #{},  %% Reset for new generation
        generation = State#state.generation + 1,
        champion_tenure = case NewChampion == State#state.active_agent of
            true -> State#state.champion_tenure + 1;
            false -> 0
        end
    },

    %% Start trial with champion
    start_new_trial(NewChampion, NewState).

%% @private Clamp value to range.
clamp(Value, Min, Max) ->
    max(Min, min(Max, Value)).
