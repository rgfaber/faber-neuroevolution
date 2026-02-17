%% @doc Population-based evolutionary training server.
%%
%% This gen_server manages a population of neural network individuals,
%% running them through domain-specific evaluation and evolving them
%% across generations.
%%
%% == Pluggable Evolution Strategies ==
%%
%% The server delegates all evolution logic to a configurable strategy module
%% that implements the `evolution_strategy' behaviour. Built-in strategies:
%%
%% - `generational_strategy' - Traditional (mu,lambda) batch evolution (default)
%% - `steady_state_strategy' - Continuous replacement, no generations
%% - `island_strategy' - Parallel populations with migration
%% - `novelty_strategy' - Behavioral novelty search
%% - `map_elites_strategy' - Quality-diversity with niche grid
%%
%% == Generation Lifecycle ==
%%
%% Each generation follows this cycle:
%% <ol>
%% <li>Evaluate all individuals in parallel using the configured evaluator</li>
%% <li>Strategy receives evaluation results and manages selection/breeding</li>
%% <li>Strategy emits lifecycle events (individual_born, individual_died, etc.)</li>
%% <li>Server orchestrates next evaluation round</li>
%% <li>Repeat</li>
%% </ol>
%%
%% == Configuration ==
%%
%% The server is configured via a `#neuro_config{}' record that specifies:
%% <ul>
%% <li>Population size and selection ratio</li>
%% <li>Mutation rate and strength</li>
%% <li>Network topology (inputs, hidden layers, outputs)</li>
%% <li>Evaluator module (implements `neuroevolution_evaluator' behaviour)</li>
%% <li>Strategy module (implements `evolution_strategy' behaviour)</li>
%% <li>Optional event handler for notifications</li>
%% </ul>
%%
%% @author R.G. Lefever
%% @copyright 2024-2026 R.G. Lefever
-module(neuroevolution_server).
-behaviour(gen_server).

-include("neuroevolution.hrl").
-include("evolution_strategy.hrl").
-include("lifecycle_events.hrl").
-include("lc_chain.hrl").

%% API
-export([
    start_link/1,
    start_link/2,
    start_training/1,
    stop_training/1,
    get_stats/1,
    get_population/1,
    get_last_evaluated_population/1,
    get_population_snapshot/1,
    update_config/2
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

%%% ============================================================================
%%% API Functions
%%% ============================================================================

%% @doc Start the neuroevolution server with given configuration.
-spec start_link(Config) -> {ok, pid()} | {error, term()} when
    Config :: neuro_config().
start_link(Config) ->
    start_link(Config, []).

%% @doc Start the neuroevolution server with configuration and options.
%%
%% Options:
%% - `{id, Id}' - Server identifier (default: make_ref())
%% - `{name, Name}' - Register with given name
-spec start_link(Config, Options) -> {ok, pid()} | {error, term()} when
    Config :: neuro_config(),
    Options :: proplists:proplist().
start_link(Config, Options) ->
    Id = proplists:get_value(id, Options, make_ref()),
    case proplists:get_value(name, Options) of
        undefined ->
            gen_server:start_link(?MODULE, {Id, Config}, []);
        Name ->
            gen_server:start_link(Name, ?MODULE, {Id, Config}, [])
    end.

%% @doc Start the evolutionary training process.
-spec start_training(ServerRef) -> {ok, started | already_running} when
    ServerRef :: pid() | atom().
start_training(ServerRef) ->
    gen_server:call(ServerRef, start_training).

%% @doc Stop the evolutionary training process.
-spec stop_training(ServerRef) -> ok when
    ServerRef :: pid() | atom().
stop_training(ServerRef) ->
    gen_server:call(ServerRef, stop_training).

%% @doc Get current training statistics.
-spec get_stats(ServerRef) -> {ok, Stats} when
    ServerRef :: pid() | atom(),
    Stats :: map().
get_stats(ServerRef) ->
    gen_server:call(ServerRef, get_stats).

%% @doc Get the current population (raw individuals).
%% NOTE: After training_complete, this returns the post-breeding population
%% where non-elite individuals have fitness reset to 0. For the fully
%% evaluated population with all fitness values, use
%% get_last_evaluated_population/1.
-spec get_population(ServerRef) -> {ok, [individual()]} when
    ServerRef :: pid() | atom().
get_population(ServerRef) ->
    gen_server:call(ServerRef, get_population).

%% @doc Get the last fully-evaluated population, sorted by fitness descending.
%% This is captured before the strategy replaces/resets individuals for the
%% next generation. All individuals retain their fitness values.
%% Returns empty list if no evaluation has completed yet.
-spec get_last_evaluated_population(ServerRef) -> {ok, [individual()]} when
    ServerRef :: pid() | atom().
get_last_evaluated_population(ServerRef) ->
    gen_server:call(ServerRef, get_last_evaluated_population).

%% @doc Get population snapshot from the strategy.
-spec get_population_snapshot(ServerRef) -> {ok, population_snapshot()} when
    ServerRef :: pid() | atom().
get_population_snapshot(ServerRef) ->
    gen_server:call(ServerRef, get_population_snapshot).

%% @doc Update configuration dynamically (used by external meta-controllers).
%%
%% Allows updating hyperparameters like mutation_rate, mutation_strength,
%% and selection_ratio between generations. This is used by the Elixir
%% Liquid Conglomerate meta-controller to feed its recommendations into
%% the Erlang neuroevolution server.
-spec update_config(ServerRef, Params) -> {ok, map()} when
    ServerRef :: pid() | atom(),
    Params :: #{atom() => number()}.
update_config(ServerRef, Params) ->
    gen_server:call(ServerRef, {update_config, Params}).

%%% ============================================================================
%%% gen_server Callbacks
%%% ============================================================================

%% @private
%% Handle map config (from Elixir integration)
init({Id, Config}) when is_map(Config) ->
    NeuroConfig = neuro_config:from_map(Config),
    init({Id, NeuroConfig});

%% Handle record config (native Erlang)
init({Id, Config}) when is_record(Config, neuro_config) ->
    error_logger:info_msg(
        "[neuroevolution_server] Initializing with population size ~p~n",
        [Config#neuro_config.population_size]
    ),

    %% Get strategy module (default to generational_strategy)
    StrategyModule = get_strategy_module(Config),

    %% Build strategy config
    %% Merge strategy_params into top-level so strategy can access options like network_factory
    StrategyParams = get_strategy_params(Config),
    StrategyConfig = maps:merge(StrategyParams, #{
        neuro_config => Config,
        strategy_params => StrategyParams
    }),

    %% Initialize strategy
    {ok, StrategyState, InitEvents} = evolution_strategy:init(StrategyModule, StrategyConfig),

    %% Process initial events (birth events for initial population)
    process_lifecycle_events(InitEvents, Config),

    %% Get population from strategy for state
    Snapshot = evolution_strategy:get_population_snapshot(StrategyModule, StrategyState),
    Population = extract_population_from_snapshot(Snapshot, StrategyState),

    TotalGames = Config#neuro_config.population_size *
                 Config#neuro_config.evaluations_per_individual,

    %% Start meta-controller if configured
    MetaController = maybe_start_meta_controller(Config),
    LcChain = maybe_start_lc_chain(Config),

    %% Start mesh supervisor if mesh evaluation is enabled
    maybe_start_mesh_sup(Config),

    %% Initialize checkpoint manager if configured
    maybe_init_checkpoint_manager(Config),

    State = #neuro_state{
        id = Id,
        config = Config,
        population = Population,
        total_games = TotalGames,
        meta_controller = MetaController,
        lc_chain = LcChain,
        strategy_module = StrategyModule,
        strategy_state = StrategyState,
        %% Initialize checkpoint tracking for continuous evolution
        last_checkpoint_time = erlang:system_time(millisecond),
        last_checkpoint_evals = 0
    },

    %% Subscribe to L0 actuator events (event-driven hyperparameter control)
    %% The L0 actuators publish events, we react by updating config
    subscribe_to_l0_events(),

    {ok, State}.

%% @private Subscribe to L0 actuator events for reactive hyperparameter control.
subscribe_to_l0_events() ->
    %% Subscribe to Task Silo L0 evolution params
    neuroevolution_events:subscribe(<<"l0.evolution_params">>, self()),
    %% Subscribe to Resource Silo L0 resource params
    neuroevolution_events:subscribe(<<"l0.resource_params">>, self()),
    %% Subscribe to silo recommendation events (event-driven read model pattern)
    %% These replace blocking get_recommendations() calls with cached lookups
    silo_events:subscribe_to_recommendations(resource),
    silo_events:subscribe_to_recommendations(task),
    ok.

%% @private
handle_call(start_training, _From, State = #neuro_state{running = true}) ->
    {reply, {ok, already_running}, State};

handle_call(start_training, _From, State) ->
    error_logger:info_msg(
        "[neuroevolution_server] Starting training - Generation ~p~n",
        [State#neuro_state.generation]
    ),

    NewState = State#neuro_state{
        running = true,
        evaluating = true,
        games_completed = 0
    },

    %% Notify event handler
    notify_event(NewState, {training_started, State#neuro_state.config}),

    %% Start evaluation
    self() ! evaluate_generation,

    {reply, {ok, started}, NewState};

handle_call(stop_training, _From, State) ->
    notify_event(State, {training_stopped, State#neuro_state.generation}),
    NewState = State#neuro_state{running = false, evaluating = false},
    {reply, ok, NewState};

handle_call(get_stats, _From, State) ->
    Stats = build_stats(State),
    {reply, {ok, Stats}, State};

handle_call(get_population, _From, State) ->
    {reply, {ok, State#neuro_state.population}, State};

handle_call(get_last_evaluated_population, _From, State) ->
    {reply, {ok, State#neuro_state.last_evaluated_population}, State};

handle_call(get_population_snapshot, _From, State) ->
    StrategyModule = State#neuro_state.strategy_module,
    StrategyState = State#neuro_state.strategy_state,
    Snapshot = evolution_strategy:get_population_snapshot(StrategyModule, StrategyState),
    {reply, {ok, Snapshot}, State};

handle_call({update_config, Params}, _From, State) ->
    %% Apply external meta-controller (LC) parameters via strategy
    StrategyModule = State#neuro_state.strategy_module,
    StrategyState = State#neuro_state.strategy_state,

    %% Update strategy state with new params
    NewStrategyState = evolution_strategy:apply_meta_params(StrategyModule, Params, StrategyState),

    %% Also update neuro_config for consistency
    Config = State#neuro_state.config,
    UpdatedConfig = apply_config_params(Config, Params),

    %% Return the current config values as confirmation
    ConfigMap = build_config_response(UpdatedConfig),

    NewState = State#neuro_state{
        config = UpdatedConfig,
        strategy_state = NewStrategyState
    },
    {reply, {ok, ConfigMap}, NewState};

handle_call(_Request, _From, State) ->
    {reply, {error, unknown_request}, State}.

%% @private
handle_cast(_Msg, State) ->
    {noreply, State}.

%% @private
%% ============================================================================
%% L0 Event Handlers (Event-Driven Hyperparameter Control)
%% ============================================================================

%% Handle L0 Task Silo evolution params event
handle_info({neuro_event, <<"l0.evolution_params">>, Event}, State) ->
    Params = maps:get(params, Event, #{}),
    NewState = apply_l0_evolution_params(Params, State),
    {noreply, NewState};

%% Handle L0 Resource Silo resource params event
handle_info({neuro_event, <<"l0.resource_params">>, Event}, State) ->
    Params = maps:get(params, Event, #{}),
    NewState = apply_l0_resource_params(Params, State),
    {noreply, NewState};

%% ============================================================================
%% Silo Recommendation Event Handlers (Event-Driven Read Model)
%% ============================================================================

%% Cache resource silo recommendations when they change
handle_info({neuro_event, <<"silo.resource.recommendations">>, Event}, State) ->
    Recs = maps:get(recommendations, Event, #{}),
    NewState = State#neuro_state{cached_resource_recommendations = Recs},
    {noreply, NewState};

%% Cache task silo recommendations when they change
handle_info({neuro_event, <<"silo.task.recommendations">>, Event}, State) ->
    Recs = maps:get(recommendations, Event, #{}),
    NewState = State#neuro_state{cached_task_recommendations = Recs},
    {noreply, NewState};

%% ============================================================================
%% Generation Evaluation Handlers
%% ============================================================================

handle_info(evaluate_generation, State = #neuro_state{running = false}) ->
    {noreply, State};

handle_info(evaluate_generation, State) ->
    %% Check memory pressure before evaluation (emergency protection)
    case check_memory_pressure() of
        critical ->
            error_logger:warning_msg(
                "[neuroevolution_server] Memory CRITICAL - pausing evolution~n"
            ),
            %% Force garbage collection
            erlang:garbage_collect(),
            %% Wait and retry
            erlang:send_after(5000, self(), evaluate_generation),
            {noreply, State};
        high ->
            error_logger:warning_msg(
                "[neuroevolution_server] Memory HIGH (~.1f%) - forcing GC~n",
                [get_memory_pressure_pct()]
            ),
            erlang:garbage_collect(),
            do_evaluate_generation(State);
        normal ->
            do_evaluate_generation(State)
    end;

%% Handle distributed evaluation result
handle_info({neuro_event, _Topic, {evaluated, Result}}, State)
  when State#neuro_state.evaluating =:= true,
       map_size(State#neuro_state.pending_evaluations) > 0 ->
    handle_distributed_eval_result(Result, State);

%% Handle evaluation timeout
handle_info({eval_timeout, TimeoutRef}, #neuro_state{eval_timeout_ref = TimeoutRef} = State)
  when State#neuro_state.evaluating =:= true ->
    handle_evaluation_timeout(State);

handle_info({game_completed, _Result}, State) ->
    %% Update progress counter
    NewCompleted = State#neuro_state.games_completed + 1,
    NewState = State#neuro_state{games_completed = NewCompleted},

    %% Notify progress every 10 games
    case NewCompleted rem 10 of
        0 ->
            notify_event(NewState, {evaluation_progress,
                State#neuro_state.generation,
                NewCompleted,
                State#neuro_state.total_games
            });
        _ ->
            ok
    end,

    {noreply, NewState};

handle_info({evaluation_complete, EvaluatedPopulation}, State) ->
    handle_evaluation_complete(State, EvaluatedPopulation);

handle_info(_Info, State) ->
    {noreply, State}.

%% @private
terminate(_Reason, _State) ->
    ok.

%% @private
code_change(_OldVsn, State, _Extra) ->
    {ok, State}.

%%% ============================================================================
%%% Internal Functions - Strategy Integration
%%% ============================================================================

%% @private Get strategy module from config.
%%
%% Strategy selection priority:
%% 1. Explicit strategy_module in strategy_config
%% 2. Auto-detect based on config (continuous vs generational indicators)
%% 3. Default to steady_state_strategy (continuous evolution)
%%
%% Auto-detection: If max_evaluations is set AND max_generations is infinity,
%% assume continuous evolution mode and use steady_state_strategy.
get_strategy_module(Config) ->
    case Config#neuro_config.strategy_config of
        #strategy_config{strategy_module = Module} when Module =/= undefined ->
            Module;
        _ ->
            %% Auto-detect: continuous evolution indicators trigger steady_state
            case detect_evolution_mode(Config) of
                continuous -> steady_state_strategy;
                generational -> generational_strategy
            end
    end.

%% @private Detect evolution mode from config indicators.
%%
%% Continuous mode when:
%% - max_evaluations is set (finite integer)
%% - AND max_generations is infinity or undefined
%%
%% Otherwise use generational mode for backward compatibility.
detect_evolution_mode(Config) ->
    MaxEvals = Config#neuro_config.max_evaluations,
    MaxGens = Config#neuro_config.max_generations,

    case {MaxEvals, MaxGens} of
        {N, infinity} when is_integer(N), N > 0 ->
            %% Clear continuous evolution setup
            continuous;
        {N, undefined} when is_integer(N), N > 0 ->
            %% max_evaluations set, no max_generations
            continuous;
        _ ->
            %% Default to generational for backward compat
            generational
    end.

%% @private Get strategy params from config.
get_strategy_params(Config) ->
    case Config#neuro_config.strategy_config of
        undefined ->
            #{};
        #strategy_config{strategy_params = Params} ->
            Params
    end.

%% @private Extract population list from strategy snapshot.
%% The strategy state contains the actual population.
extract_population_from_snapshot(_Snapshot, StrategyState) ->
    %% Access internal population from strategy state
    %% This is strategy-specific; generational_strategy uses #gen_state{}
    case StrategyState of
        {gen_state, _, _, _, Population, _, _, _, _, _, _, _, _, _, _, _} ->
            Population;
        _ ->
            %% Fallback: try to get from state if it's a record
            try
                element(5, StrategyState)  % Population is usually 5th element in gen_state
            catch
                _:_ -> []
            end
    end.

%% @private Process lifecycle events from strategy.
process_lifecycle_events(Events, Config) ->
    lists:foreach(
        fun(Event) ->
            maybe_publish_lifecycle_event(Config, Event)
        end,
        Events
    ).

%% @private Extract breeding events from lifecycle events.
%% Converts individual_born events with crossover origin to breeding_event records.
extract_breeding_events(Events, Generation) ->
    lists:filtermap(
        fun(#individual_born{id = ChildId, origin = Origin, parent_ids = Parents})
                when Origin =:= crossover, length(Parents) >= 2 ->
            [Parent1Id, Parent2Id | _] = Parents,
            {true, #breeding_event{
                parent1_id = Parent1Id,
                parent2_id = Parent2Id,
                child_id = ChildId,
                generation = Generation
            }};
           (_) ->
            false
        end,
        Events
    ).

%% @private Publish a lifecycle event if event publishing is enabled.
maybe_publish_lifecycle_event(Config, Event) ->
    case Config#neuro_config.publish_events of
        true ->
            Realm = Config#neuro_config.realm,
            Topic = neuroevolution_events:events_topic(Realm),
            EventMap = lifecycle_event_to_map(Event, Realm),
            neuroevolution_events:publish(Topic, EventMap);
        false ->
            ok
    end.

%% @private Convert lifecycle event to map for publishing.
lifecycle_event_to_map(#individual_born{id = Id, origin = Origin, parent_ids = Parents}, Realm) ->
    {individual_born, #{
        realm => Realm,
        individual_id => Id,
        origin => Origin,
        parent_ids => Parents,
        timestamp => erlang:system_time(millisecond)
    }};
lifecycle_event_to_map(#individual_died{id = Id, reason = Reason, final_fitness = Fitness}, Realm) ->
    {individual_died, #{
        realm => Realm,
        individual_id => Id,
        reason => Reason,
        final_fitness => Fitness,
        timestamp => erlang:system_time(millisecond)
    }};
lifecycle_event_to_map(#individual_evaluated{id = Id, fitness = Fitness, metrics = Metrics}, Realm) ->
    {individual_evaluated, #{
        realm => Realm,
        individual_id => Id,
        fitness => Fitness,
        metrics => Metrics,
        timestamp => erlang:system_time(millisecond)
    }};
lifecycle_event_to_map(#cohort_evaluated{generation = Gen, best_fitness = Best, avg_fitness = Avg}, Realm) ->
    {cohort_evaluated, #{
        realm => Realm,
        generation => Gen,
        best_fitness => Best,
        avg_fitness => Avg,
        timestamp => erlang:system_time(millisecond)
    }};
lifecycle_event_to_map(#breeding_complete{generation = Gen, survivor_count = Surv, offspring_count = Off}, Realm) ->
    {breeding_complete, #{
        realm => Realm,
        generation => Gen,
        survivor_count => Surv,
        offspring_count => Off,
        timestamp => erlang:system_time(millisecond)
    }};
lifecycle_event_to_map(#generation_advanced{generation = Gen, previous_best_fitness = Best}, Realm) ->
    {generation_advanced, #{
        realm => Realm,
        generation => Gen,
        previous_best_fitness => Best,
        timestamp => erlang:system_time(millisecond)
    }};
%% Progress checkpoint - strategy-agnostic progress event (continuous evolution)
lifecycle_event_to_map(#progress_checkpoint{} = Checkpoint, Realm) ->
    {progress_checkpoint, #{
        realm => Realm,
        %% PRIMARY: Evaluation-centric metrics
        total_evaluations => Checkpoint#progress_checkpoint.total_evaluations,
        evaluations_since_last => Checkpoint#progress_checkpoint.evaluations_since_last,
        evals_per_second => Checkpoint#progress_checkpoint.evals_per_second,
        elapsed_ms => Checkpoint#progress_checkpoint.elapsed_ms,
        %% SECONDARY: Cohort for lineage (formerly "generation")
        cohort => Checkpoint#progress_checkpoint.cohort,
        %% Fitness metrics
        best_fitness => Checkpoint#progress_checkpoint.best_fitness,
        avg_fitness => Checkpoint#progress_checkpoint.avg_fitness,
        worst_fitness => Checkpoint#progress_checkpoint.worst_fitness,
        improvement => Checkpoint#progress_checkpoint.improvement,
        %% Population info
        population_size => Checkpoint#progress_checkpoint.population_size,
        species_count => Checkpoint#progress_checkpoint.species_count,
        checkpoint_interval => Checkpoint#progress_checkpoint.checkpoint_interval,
        timestamp => erlang:system_time(millisecond)
    }};
%% Steady-state replacement event (continuous evolution)
lifecycle_event_to_map(#steady_state_replacement{} = Event, Realm) ->
    {steady_state_replacement, #{
        realm => Realm,
        replaced_ids => Event#steady_state_replacement.replaced_ids,
        offspring_ids => Event#steady_state_replacement.offspring_ids,
        best_fitness => Event#steady_state_replacement.best_fitness,
        avg_fitness => Event#steady_state_replacement.avg_fitness,
        timestamp => erlang:system_time(millisecond)
    }};
%% Individual aged out - organism lifespan death (continuous evolution)
lifecycle_event_to_map(#individual_aged_out{} = Event, Realm) ->
    {individual_aged_out, #{
        realm => Realm,
        individual_id => Event#individual_aged_out.id,
        final_age => Event#individual_aged_out.final_age,
        final_fitness => Event#individual_aged_out.final_fitness,
        lifetime_stats => Event#individual_aged_out.lifetime_stats,
        timestamp => erlang:system_time(millisecond)
    }};
lifecycle_event_to_map(Event, Realm) ->
    {lifecycle_event, #{
        realm => Realm,
        event => Event,
        timestamp => erlang:system_time(millisecond)
    }}.

%%% ============================================================================
%%% Internal Functions - Progress Checkpoints (Continuous Evolution)
%%% ============================================================================

%% @private Emit progress_checkpoint event for continuous evolution tracking.
%% Called after each evaluation batch to provide evaluation-centric metrics.
%% Returns updated state with new checkpoint tracking values.
-spec emit_progress_checkpoint(neuro_state(), [individual()]) -> neuro_state().
emit_progress_checkpoint(State, Sorted) ->
    Config = State#neuro_state.config,
    Now = erlang:system_time(millisecond),

    %% Calculate checkpoint metrics
    LastTime = State#neuro_state.last_checkpoint_time,
    LastEvals = State#neuro_state.last_checkpoint_evals,
    CurrentEvals = State#neuro_state.total_evaluations,
    EvalsSinceLast = CurrentEvals - LastEvals,
    ElapsedMs = Now - LastTime,

    %% Calculate throughput (evals per second)
    EvalsPerSecond = case ElapsedMs of
        0 -> 0.0;
        Ms -> (EvalsSinceLast * 1000.0) / Ms
    end,

    %% Calculate fitness improvement
    Best = hd(Sorted),
    LastBest = State#neuro_state.last_gen_best,
    Improvement = case LastBest of
        V when V == 0.0 -> 0.0;
        _ -> Best#individual.fitness - LastBest
    end,

    %% Get checkpoint interval (default to population size)
    CheckpointInterval = get_checkpoint_interval(Config),

    %% Get species count from strategy's population snapshot
    StrategyMod = State#neuro_state.strategy_module,
    StrategyState = State#neuro_state.strategy_state,
    Snapshot = StrategyMod:get_population_snapshot(StrategyState),
    SpeciesCount = maps:get(species_count, Snapshot, 1),

    %% Build the checkpoint record
    Checkpoint = #progress_checkpoint{
        total_evaluations = CurrentEvals,
        evaluations_since_last = EvalsSinceLast,
        cohort = State#neuro_state.generation,
        best_fitness = Best#individual.fitness,
        avg_fitness = neuroevolution_stats:avg_fitness(Sorted),
        worst_fitness = (lists:last(Sorted))#individual.fitness,
        population_size = length(Sorted),
        species_count = SpeciesCount,
        improvement = Improvement,
        elapsed_ms = ElapsedMs,
        evals_per_second = EvalsPerSecond,
        checkpoint_interval = CheckpointInterval,
        timestamp = Now
    },

    %% Emit the checkpoint event
    maybe_publish_lifecycle_event(Config, Checkpoint),

    %% Update state with new checkpoint tracking
    State#neuro_state{
        last_checkpoint_time = Now,
        last_checkpoint_evals = CurrentEvals
    }.

%% @private Get checkpoint interval from config or use default.
%% Default is population_size (emit after each cohort's worth of evaluations).
get_checkpoint_interval(Config) ->
    case Config#neuro_config.checkpoint_interval of
        undefined -> Config#neuro_config.population_size;
        Interval when is_integer(Interval), Interval > 0 -> Interval
    end.

%%% ============================================================================
%%% Internal Functions - Evaluation
%%% ============================================================================

%% @private
%% @doc Calculate max concurrent evaluations based on cached recommendations or config.
%% Priority: cached_resource_recommendations (event-driven) > config > system default.
%%
%% This version uses cached recommendations from the event-driven read model pattern.
%% The resource_silo publishes recommendations when they change, and we cache them
%% in State. This avoids blocking calls during evaluation.
get_max_concurrent(Config, CachedResourceRecs) ->
    %% Try cached recommendations first (event-driven read model)
    case CachedResourceRecs of
        #{max_concurrent_evaluations := N} when is_integer(N), N > 0 ->
            N;
        _ ->
            %% Fall back to config or system default
            get_max_concurrent_from_config(Config)
    end.

%% @private
%% @doc Calculate max concurrent evaluations from config only.
%% Used when cached recommendations are not available.
get_max_concurrent_from_config(Config) ->
    case Config#neuro_config.max_concurrent_evaluations of
        undefined ->
            erlang:system_info(schedulers_online) * 2;
        nil ->  %% Elixir nil
            erlang:system_info(schedulers_online) * 2;
        ConfigN when is_integer(ConfigN), ConfigN > 0 ->
            ConfigN
    end.

%% @private
%% @doc Evaluate all individuals with bounded concurrency.
%% MaxConcurrent is passed in from caller (computed before spawning using cached
%% resource silo recommendations via the event-driven read model pattern).
evaluate_population_parallel(Population, Config, MaxConcurrent, ServerPid, Generation) ->
    EvaluatorModule = Config#neuro_config.evaluator_module,
    EvaluatorOptions = Config#neuro_config.evaluator_options,
    GamesPerIndividual = Config#neuro_config.evaluations_per_individual,
    PopSize = length(Population),

    %% NOTE: batch_networks was removed to fix a critical memory leak!
    %% Previously, all 500 population networks were passed in Options to every spawned
    %% evaluation process. Erlang copies closure variables on spawn, so each of 500 spawns
    %% received a COPY of 500 networks = 250,000 network copies per generation (~2GB/gen).
    %% The batch_networks field was dead code - never used by any evaluator.
    %% If intra-batch pairing is ever needed, pass only the single opponent network per spawn.
    BaseOptions = maps:merge(EvaluatorOptions, #{
        games => GamesPerIndividual,
        notify_pid => ServerPid
    }),
    Options = BaseOptions,

    EventCtx = #{
        config => Config,
        generation => Generation,
        total => PopSize
    },

    %% Check if distributed evaluation is enabled
    UseDistributed = maps:get(distributed_evaluation, Options, false),
    case UseDistributed of
        true ->
            %% Use distributed evaluation across connected nodes
            neuroevolution_evaluator:evaluate_batch_distributed(
                Population, EvaluatorModule, Options
            );
        false ->
            %% Use local parallel evaluation
            evaluate_batches(Population, MaxConcurrent, EvaluatorModule, Options, [], EventCtx, 0)
    end.

%% @private
%% Evaluate population using spawn_opt with scheduler hints for multi-core parallelism.
%% The key insight: spawn_link creates processes on the same scheduler as the spawner,
%% but spawn_opt with {scheduler, N} distributes work across all CPU cores.
evaluate_batches([], _MaxConcurrent, _EvaluatorModule, _Options, Acc, _EventCtx, _Completed) ->
    lists:reverse(Acc);
evaluate_batches(Population, MaxConcurrent, EvaluatorModule, Options, Acc, EventCtx, Completed) ->
    {Batch, Remaining} = split_list(Population, MaxConcurrent),
    NumSchedulers = erlang:system_info(schedulers_online),
    %% Get timeout from options, default to 5000ms (was 30000ms hardcoded)
    %% Shorter timeout = faster throughput, slow individuals get penalized with min fitness
    Timeout = maps:get(evaluation_timeout, Options, 5000),

    %% Track both Ref and Pid so we can kill zombie processes on timeout
    %% Use spawn_opt with scheduler hints to distribute across CPU cores
    RefsWithPids = lists:map(
        fun({Individual, Index}) ->
            Ref = make_ref(),
            ParentPid = self(),
            %% Distribute across schedulers: Index mod NumSchedulers + 1
            SchedulerHint = (Index rem NumSchedulers) + 1,
            Pid = spawn_opt(
                fun() ->
                    Result = neuroevolution_evaluator:evaluate_individual(
                        Individual, EvaluatorModule, Options
                    ),
                    ParentPid ! {eval_result, Ref, Result}
                end,
                [{scheduler, SchedulerHint}, link]
            ),
            {Ref, Pid, Individual}
        end,
        lists:zip(Batch, lists:seq(0, length(Batch) - 1))
    ),

    {BatchResults, NewCompleted} = collect_eval_results(RefsWithPids, [], EventCtx, Completed, Timeout),

    evaluate_batches(Remaining, MaxConcurrent, EvaluatorModule, Options,
                     lists:reverse(BatchResults) ++ Acc, EventCtx, NewCompleted).

%% @private
split_list(List, N) when N >= length(List) ->
    {List, []};
split_list(List, N) ->
    lists:split(N, List).

%% @private
%% Collects evaluation results from spawned workers.
%% RefsWithPids is a list of {Ref, Pid, Individual} tuples.
%% On timeout, explicitly kills remaining worker processes to prevent zombie memory leaks.
%% Timeout is now configurable (was hardcoded 30000ms).
collect_eval_results([], Acc, _EventCtx, Completed, _Timeout) ->
    {lists:reverse(Acc), Completed};
collect_eval_results(RefsWithPids, Acc, EventCtx, Completed, Timeout) ->
    Total = maps:get(total, EventCtx, 0),
    receive
        {eval_result, Ref, {ok, EvaluatedIndividual}} ->
            NewRefs = lists:keydelete(Ref, 1, RefsWithPids),
            NewCompleted = Completed + 1,
            notify_individual_evaluated(EventCtx, EvaluatedIndividual, NewCompleted, Total),
            %% CRITICAL: Strip compiled_ref from network to prevent NIF memory leaks.
            %% The compiled_ref holds a Rust ResourceArc that keeps native memory alive.
            %% Networks will be recompiled on-demand in subsequent generations.
            CleanedIndividual = strip_compiled_ref_from_individual(EvaluatedIndividual),
            collect_eval_results(NewRefs, [CleanedIndividual | Acc], EventCtx, NewCompleted, Timeout);
        {eval_result, Ref, {error, _Reason}} ->
            case lists:keyfind(Ref, 1, RefsWithPids) of
                {Ref, _Pid, Original} ->
                    NewRefs = lists:keydelete(Ref, 1, RefsWithPids),
                    NewCompleted = Completed + 1,
                    collect_eval_results(NewRefs, [Original | Acc], EventCtx, NewCompleted, Timeout);
                false ->
                    collect_eval_results(RefsWithPids, Acc, EventCtx, Completed, Timeout)
            end
    after Timeout ->
        %% CRITICAL: Kill all remaining worker processes to prevent zombie memory leaks.
        %% Without this, timed-out workers continue running and consuming memory/CPU.
        NumPending = length(RefsWithPids),
        error_logger:warning_msg(
            "[neuroevolution_server] Evaluation timeout - killing ~p zombie workers~n",
            [NumPending]
        ),
        lists:foreach(
            fun({_Ref, Pid, _Ind}) ->
                %% Unlink first to prevent crash propagation, then kill
                unlink(Pid),
                exit(Pid, kill)
            end,
            RefsWithPids
        ),
        Remaining = [Ind || {_, _, Ind} <- RefsWithPids],
        {lists:reverse(Acc) ++ Remaining, Completed + NumPending}
    end.

%% @private
notify_individual_evaluated(EventCtx, Individual, Completed, Total) ->
    Config = maps:get(config, EventCtx),
    case Config#neuro_config.publish_events of
        true ->
            Realm = Config#neuro_config.realm,
            Topic = neuroevolution_events:events_topic(Realm),
            EventData = #{
                event_type => individual_evaluated,
                individual_id => Individual#individual.id,
                fitness => Individual#individual.fitness,
                metrics => Individual#individual.metrics,
                completed => Completed,
                total => Total
            },
            neuroevolution_events:publish(Topic, EventData);
        false ->
            ok
    end.

%%% ============================================================================
%%% Internal Functions - Direct Evaluation Mode
%%% ============================================================================

%% @private
start_direct_evaluation(Population, State) ->
    Config = State#neuro_state.config,
    ServerPid = self(),
    Generation = State#neuro_state.generation,
    CachedResourceRecs = State#neuro_state.cached_resource_recommendations,

    %% Compute MaxConcurrent BEFORE spawning using cached recommendations (event-driven)
    %% This avoids blocking calls in the spawned evaluation process
    MaxConcurrent = get_max_concurrent(Config, CachedResourceRecs),

    spawn_link(fun() ->
        Results = evaluate_population_parallel(Population, Config, MaxConcurrent, ServerPid, Generation),
        ServerPid ! {evaluation_complete, Results}
    end),

    State.

%%% ============================================================================
%%% Internal Functions - Distributed Evaluation Mode
%%% ============================================================================

%% @private
start_distributed_evaluation(Population, State) ->
    Config = State#neuro_state.config,
    Realm = Config#neuro_config.realm,
    Timeout = Config#neuro_config.evaluation_timeout,

    ResultTopic = neuroevolution_events:evaluated_topic(Realm),
    ok = neuroevolution_events:subscribe(ResultTopic),

    EvalTopic = neuroevolution_events:evaluate_topic(Realm),
    EvaluatorOptions = Config#neuro_config.evaluator_options,
    GamesPerIndividual = Config#neuro_config.evaluations_per_individual,

    PendingEvaluations = lists:foldl(
        fun(Individual, Acc) ->
            RequestId = make_ref(),
            Request = #{
                request_id => RequestId,
                realm => Realm,
                individual_id => Individual#individual.id,
                network => Individual#individual.network,
                options => maps:merge(EvaluatorOptions, #{
                    games => GamesPerIndividual
                })
            },
            ok = neuroevolution_events:publish(EvalTopic, {evaluate_request, Request}),
            maps:put(RequestId, {Individual#individual.id, Individual}, Acc)
        end,
        #{},
        Population
    ),

    TimeoutRef = make_ref(),
    erlang:send_after(Timeout, self(), {eval_timeout, TimeoutRef}),

    State#neuro_state{
        pending_evaluations = PendingEvaluations,
        eval_timeout_ref = TimeoutRef
    }.

%% @private
handle_distributed_eval_result(Result, State) ->
    RequestId = maps:get(request_id, Result),
    PendingEvaluations = State#neuro_state.pending_evaluations,

    case maps:find(RequestId, PendingEvaluations) of
        {ok, {_IndId, Original}} ->
            Metrics = maps:get(metrics, Result, #{}),
            EvaluatedInd = Original#individual{metrics = Metrics},

            NewPending = maps:remove(RequestId, PendingEvaluations),
            NewState = State#neuro_state{pending_evaluations = NewPending},

            case maps:size(NewPending) of
                0 ->
                    finish_distributed_evaluation(NewState, [EvaluatedInd]);
                _ ->
                    AccumulatedResults = get_accumulated_results(State),
                    NewAccumulated = [EvaluatedInd | AccumulatedResults],
                    {noreply, store_accumulated_results(NewState, NewAccumulated)}
            end;
        error ->
            {noreply, State}
    end.

%% @private
handle_evaluation_timeout(State) ->
    PendingEvaluations = State#neuro_state.pending_evaluations,
    PendingCount = maps:size(PendingEvaluations),

    error_logger:warning_msg(
        "[neuroevolution_server] Distributed evaluation timeout. ~p evaluations still pending.~n",
        [PendingCount]
    ),

    PendingIndividuals = [Ind || {_IndId, Ind} <- maps:values(PendingEvaluations)],
    AccumulatedResults = get_accumulated_results(State),
    AllResults = AccumulatedResults ++ PendingIndividuals,

    finish_distributed_evaluation(State, AllResults).

%% @private
finish_distributed_evaluation(State, AdditionalResults) ->
    Config = State#neuro_state.config,
    Realm = Config#neuro_config.realm,

    ResultTopic = neuroevolution_events:evaluated_topic(Realm),
    ok = neuroevolution_events:unsubscribe(ResultTopic),

    AccumulatedResults = get_accumulated_results(State),
    AllResults = AccumulatedResults ++ AdditionalResults,

    CleanState = State#neuro_state{
        pending_evaluations = #{},
        eval_timeout_ref = undefined
    },
    CleanState2 = clear_accumulated_results(CleanState),

    handle_evaluation_complete(CleanState2, AllResults).

%% @private
get_accumulated_results(#neuro_state{}) ->
    case get(distributed_eval_results) of
        undefined -> [];
        Results -> Results
    end.

%% @private
store_accumulated_results(State, Results) ->
    put(distributed_eval_results, Results),
    State.

%% @private
clear_accumulated_results(State) ->
    erase(distributed_eval_results),
    State.

%%% ============================================================================
%%% Internal Functions - Mesh Evaluation Mode
%%% ============================================================================

%% @private Start mesh supervisor if mesh evaluation mode is enabled.
%%
%% The mesh supervisor (mesh_sup) manages:
%% - evaluator_pool_registry: Tracks remote evaluator capacity
%% - macula_mesh: Macula integration facade
%% - distributed_evaluator: Load-balanced evaluation dispatch
maybe_start_mesh_sup(Config) ->
    case Config#neuro_config.evaluation_mode of
        mesh ->
            MeshConfig = case Config#neuro_config.mesh_config of
                undefined -> #{mesh_enabled => true, realm => Config#neuro_config.realm};
                M when is_map(M) -> M#{mesh_enabled => true, realm => Config#neuro_config.realm}
            end,
            case mesh_sup:start_link(MeshConfig) of
                {ok, Pid} ->
                    error_logger:info_msg(
                        "[neuroevolution_server] Started mesh supervisor ~p~n",
                        [Pid]
                    ),
                    ok;
                {error, {already_started, _}} ->
                    ok;  % Already running, that's fine
                {error, Reason} ->
                    error_logger:error_msg(
                        "[neuroevolution_server] Failed to start mesh supervisor: ~p~n",
                        [Reason]
                    ),
                    error
            end;
        _ ->
            ok
    end.

%% @private Start mesh-based evaluation using distributed_evaluator.
%%
%% Uses the distributed_evaluator module for load-balanced evaluation
%% across mesh nodes. Falls back to local evaluation if no remote
%% evaluators are available.
start_mesh_evaluation(Population, State) ->
    Config = State#neuro_state.config,
    EvaluatorModule = Config#neuro_config.evaluator_module,
    EvaluatorOptions = Config#neuro_config.evaluator_options,
    GamesPerIndividual = Config#neuro_config.evaluations_per_individual,
    Timeout = Config#neuro_config.evaluation_timeout,

    ServerPid = self(),

    spawn_link(fun() ->
        %% Build evaluation options
        Options = maps:merge(EvaluatorOptions, #{
            games => GamesPerIndividual,
            timeout_ms => Timeout,
            prefer_local => 0.3  % 30% local preference for latency
        }),

        %% Use distributed_evaluator for batch evaluation
        Results = distributed_evaluator:evaluate_batch(
            [Ind#individual.network || Ind <- Population],
            EvaluatorModule,
            Options
        ),

        %% Combine results with original individuals
        EvaluatedPopulation = lists:zipwith(
            fun(Ind, Result) ->
                case Result of
                    {ok, Fitness} when is_number(Fitness) ->
                        Ind#individual{
                            fitness = Fitness,
                            metrics = #{mesh_evaluated => true}
                        };
                    {ok, Metrics} when is_map(Metrics) ->
                        Ind#individual{metrics = Metrics};
                    {error, _Reason} ->
                        %% Keep original individual on error
                        Ind
                end
            end,
            Population,
            Results
        ),

        ServerPid ! {evaluation_complete, EvaluatedPopulation}
    end),

    State.

%%% ============================================================================
%%% Internal Functions - Generation Complete (Strategy Delegation)
%%% ============================================================================

%% @private
handle_evaluation_complete(State, EvaluatedPopulation) ->
    Config = State#neuro_state.config,
    StrategyModule = State#neuro_state.strategy_module,
    StrategyState = State#neuro_state.strategy_state,

    %% Calculate fitness for all individuals
    Population = lists:map(
        fun(Ind) ->
            Fitness = calculate_fitness(Ind, Config),
            Ind#individual{fitness = Fitness}
        end,
        EvaluatedPopulation
    ),

    %% Notify population_evaluated (before strategy processes results)
    Sorted = lists:sort(
        fun(A, B) -> A#individual.fitness >= B#individual.fitness end,
        Population
    ),
    Best = hd(Sorted),
    AvgFitness = neuroevolution_stats:avg_fitness(Sorted),

    error_logger:info_msg(
        "[neuroevolution_server] Cohort ~p complete - Best: ~.2f, Avg: ~.2f~n",
        [State#neuro_state.generation, Best#individual.fitness, AvgFitness]
    ),

    PopulationSummaryForEvent = [summarize_individual_for_grid(I, false, false) || I <- Sorted],
    notify_event(State, {population_evaluated, #{
        generation => State#neuro_state.generation,
        population => PopulationSummaryForEvent,
        best_fitness => Best#individual.fitness,
        avg_fitness => AvgFitness,
        worst_fitness => (lists:last(Sorted))#individual.fitness,
        population_size => length(Sorted)
    }}),

    %% Emit archive_candidates event (for subscribers)
    %% Top 20% of population WITH network data for opponent archive
    ArchiveCandidates = build_archive_candidates(Sorted, State#neuro_state.generation),
    notify_event(State, {archive_candidates, #{
        generation => State#neuro_state.generation,
        candidates => ArchiveCandidates,
        best_fitness => Best#individual.fitness,
        avg_fitness => AvgFitness
    }}),

    %% Feed each evaluation result to the strategy
    {FinalStrategyState, AllEvents, AllActions} = lists:foldl(
        fun(Ind, {StratState, Events, Actions}) ->
            FitnessResult = #{fitness => Ind#individual.fitness, metrics => Ind#individual.metrics},
            {NewActions, NewEvents, NewStratState} = evolution_strategy:handle_evaluation_result(
                StrategyModule, Ind#individual.id, FitnessResult, StratState
            ),
            {NewStratState, Events ++ NewEvents, Actions ++ NewActions}
        end,
        {StrategyState, [], []},
        Sorted
    ),

    %% Process lifecycle events from strategy
    process_lifecycle_events(AllEvents, Config),

    %% Extract breeding events from individual_born lifecycle events
    BreedingEvents = extract_breeding_events(AllEvents, State#neuro_state.generation),

    %% Process actions from strategy
    NextPopulation = process_strategy_actions(AllActions, FinalStrategyState, StrategyModule),

    %% Build generation stats for meta-controllers
    GenStats = build_generation_stats(State, Sorted),

    %% Build competitive dynamics entry for arms race visualization
    CompetitiveEntry = build_competitive_entry(State#neuro_state.generation, Sorted),
    UpdatedCompetitiveHistory = [CompetitiveEntry | lists:sublist(State#neuro_state.competitive_history, 99)],

    %% Update LC L0 sensors with evolution stats (event-driven, not polled)
    %% This pushes stats to task_l0_sensors which then publishes events to Phoenix.PubSub
    update_task_l0_sensors(State, Sorted, CompetitiveEntry),

    %% Update hyperparameters via LC chain or legacy path
    %% LC chain (L2→L1→L0 chained LTC TWEANNs) supersedes task_silo + meta_controller
    {UpdatedConfig, NewState1} = case State#neuro_state.lc_chain of
        undefined ->
            %% Legacy path: Task Silo already incorporates L2 guidance via maybe_query_l2_guidance()
            %% If task_silo is active, use its recommendations exclusively (it already queries meta_controller)
            %% Only fall back to meta_controller if task_silo is not running
            case whereis(task_silo) of
                undefined ->
                    %% No task_silo - use meta_controller directly
                    maybe_update_meta_controller(GenStats, Config, State);
                TaskSiloPid when is_pid(TaskSiloPid) ->
                    %% Task silo is active - it already incorporates L2 guidance
                    %% Don't also call meta_controller to avoid oscillation
                    maybe_update_task_silo(GenStats, Config, State)
            end;
        _LcChainPid ->
            %% New path: Chained LTC TWEANN controller
            maybe_update_lc_chain(GenStats, Config, State)
    end,

    %% Notify generation complete with silo state
    MetaParams = #{
        mutation_rate => UpdatedConfig#neuro_config.mutation_rate,
        mutation_strength => UpdatedConfig#neuro_config.mutation_strength,
        selection_ratio => UpdatedConfig#neuro_config.selection_ratio,
        population_size => UpdatedConfig#neuro_config.population_size
    },

    %% Gather LC v2 silo state for event-driven UI updates
    SiloState = get_silo_state(),

    %% Calculate batch evaluations for total_evaluations tracking
    BatchEvaluations = length(Population) * Config#neuro_config.evaluations_per_individual,
    NewTotalEvaluations = State#neuro_state.total_evaluations + BatchEvaluations,

    %% Emit progress_checkpoint event for continuous evolution tracking
    %% Update state temporarily with new evaluations count for checkpoint calculation
    TempState = NewState1#neuro_state{total_evaluations = NewTotalEvaluations},
    StateWithCheckpoint = emit_progress_checkpoint(TempState, Sorted),

    notify_event(StateWithCheckpoint, {generation_complete, #{
        generation_stats => GenStats,
        breeding_events => BreedingEvents,
        meta_params => MetaParams,
        silo_state => SiloState,
        %% Competitive dynamics for arms race visualization
        competitive_history => UpdatedCompetitiveHistory,
        %% Evaluation-centric progress metrics
        total_evaluations => NewTotalEvaluations,
        batch_evaluations => BatchEvaluations
    }}),

    %% Determine if this is a new fitness record
    IsNewFitnessRecord = Best#individual.fitness > State#neuro_state.best_fitness_ever,

    %% Update state (checkpoint tracking already updated by emit_progress_checkpoint)
    NewState = StateWithCheckpoint#neuro_state{
        config = UpdatedConfig,
        population = NextPopulation,
        generation = State#neuro_state.generation + 1,
        best_fitness_ever = max(State#neuro_state.best_fitness_ever, Best#individual.fitness),
        last_gen_best = Best#individual.fitness,
        last_gen_avg = AvgFitness,
        generation_history = [{State#neuro_state.generation, Best#individual.fitness, AvgFitness}
                              | lists:sublist(State#neuro_state.generation_history, 49)],
        competitive_history = UpdatedCompetitiveHistory,
        strategy_state = FinalStrategyState,
        evaluating = false,
        games_completed = 0,
        %% Preserve the evaluated population (sorted by fitness desc)
        %% before strategy replaces it. Used for multi-champion extraction.
        last_evaluated_population = Sorted
    },

    %% Save checkpoint on fitness record
    case IsNewFitnessRecord of
        true -> maybe_save_checkpoint(fitness_record, Best, NewState);
        false -> ok
    end,

    %% Save checkpoint at generation interval
    maybe_save_checkpoint(generation_interval, Best, NewState),

    %% Check stopping conditions and continue if appropriate
    case should_stop(NewState) of
        {true, Reason} ->
            %% Save final checkpoint on training complete
            maybe_save_checkpoint(training_complete, Best, NewState),
            notify_event(NewState, {training_complete, #{
                reason => Reason,
                %% PRIMARY: Total evaluations as progress metric
                total_evaluations => NewState#neuro_state.total_evaluations,
                %% SECONDARY: Cohort number (formerly "generation")
                generation => NewState#neuro_state.generation - 1,
                best_fitness => NewState#neuro_state.best_fitness_ever,
                best_individual => Best
            }}),
            StoppedState = NewState#neuro_state{running = false},
            {noreply, StoppedState};
        false ->
            _ = case NewState#neuro_state.running of
                true ->
                    erlang:send_after(500, self(), evaluate_generation);
                false ->
                    ok
            end,
            {noreply, NewState}
    end.

%% @private Process strategy actions and extract next population.
process_strategy_actions(Actions, StrategyState, StrategyModule) ->
    %% Look for evaluate_batch action which indicates new population ready
    case lists:keyfind(evaluate_batch, 1, Actions) of
        {evaluate_batch, _IndIds} ->
            %% Get updated population from strategy snapshot
            Snapshot = evolution_strategy:get_population_snapshot(StrategyModule, StrategyState),
            extract_population_from_snapshot(Snapshot, StrategyState);
        false ->
            %% No new population action, keep current (shouldn't happen normally)
            extract_population_from_snapshot(
                evolution_strategy:get_population_snapshot(StrategyModule, StrategyState),
                StrategyState
            )
    end.

%% @private Build generation stats for meta-controller.
build_generation_stats(State, Sorted) ->
    Best = hd(Sorted),
    AvgFitness = neuroevolution_stats:avg_fitness(Sorted),
    #generation_stats{
        generation = State#neuro_state.generation,
        best_fitness = Best#individual.fitness,
        avg_fitness = AvgFitness,
        worst_fitness = (lists:last(Sorted))#individual.fitness,
        best_individual_id = Best#individual.id,
        population_size = length(Sorted)
    }.

%% @private
%% @doc Build competitive dynamics entry for arms race visualization.
%% Calculates fitness distribution metrics: top/bottom 10% performance,
%% variance, and competitive pressure (gap between elite and baseline).
-spec build_competitive_entry(generation(), [individual()]) -> competitive_entry().
build_competitive_entry(Generation, Sorted) when is_list(Sorted), length(Sorted) > 0 ->
    Fitnesses = [I#individual.fitness || I <- Sorted],

    %% Population fitness stats
    BestFitness = hd(Fitnesses),
    AvgFitness = lists:sum(Fitnesses) / length(Fitnesses),
    WorstFitness = lists:last(Fitnesses),

    %% Top 10% elite performance (already sorted desc)
    Top10Count = max(1, length(Sorted) div 10),
    Top10 = lists:sublist(Fitnesses, Top10Count),
    Top10Avg = lists:sum(Top10) / length(Top10),

    %% Bottom 10% baseline performance
    Bottom10 = lists:sublist(lists:reverse(Fitnesses), Top10Count),
    Bottom10Avg = lists:sum(Bottom10) / length(Bottom10),

    %% Fitness variance (population diversity indicator)
    Variance = calculate_fitness_variance(Fitnesses, AvgFitness),

    %% Competitive pressure: gap between elite and baseline
    CompetitivePressure = Top10Avg - Bottom10Avg,

    #{
        generation => Generation,
        best_fitness => BestFitness,
        avg_fitness => AvgFitness,
        worst_fitness => WorstFitness,
        top_10_avg => Top10Avg,
        bottom_10_avg => Bottom10Avg,
        fitness_variance => Variance,
        competitive_pressure => CompetitivePressure,
        %% Archive stats (0 for non-self-play - Elixir SelfPlayMode handles these)
        archive_size => 0,
        archive_avg => 0.0
    };
build_competitive_entry(Generation, _) ->
    %% Empty population fallback
    #{
        generation => Generation,
        best_fitness => 0.0,
        avg_fitness => 0.0,
        worst_fitness => 0.0,
        top_10_avg => 0.0,
        bottom_10_avg => 0.0,
        fitness_variance => 0.0,
        competitive_pressure => 0.0,
        archive_size => 0,
        archive_avg => 0.0
    }.

%% @private
%% @doc Calculate fitness variance (measure of population diversity).
-spec calculate_fitness_variance([fitness()], fitness()) -> float().
calculate_fitness_variance([], _Mean) -> 0.0;
calculate_fitness_variance(Fitnesses, Mean) ->
    SumSqDiff = lists:foldl(
        fun(F, Acc) ->
            Diff = F - Mean,
            Acc + (Diff * Diff)
        end,
        0.0,
        Fitnesses
    ),
    SumSqDiff / length(Fitnesses).

%% @private
%% @doc Build archive candidates from top performers for self-play mode.
%% Returns top 20% of population WITH network data for opponent archive.
%% This is a separate event to keep population_evaluated lean.
%%
%% IMPORTANT: Strip compiled_ref from networks before archiving to prevent
%% NIF memory leaks. The compiled_ref holds a Rust ResourceArc that keeps
%% native memory alive. Networks will be recompiled on-demand when evaluated.
-spec build_archive_candidates([individual()], generation()) -> [map()].
build_archive_candidates([], _Generation) -> [];
build_archive_candidates(Sorted, Generation) when is_list(Sorted) ->
    %% Take top 20% as archive candidates
    CandidateCount = max(1, length(Sorted) div 5),
    TopPerformers = lists:sublist(Sorted, CandidateCount),

    %% Build candidate maps with network data (stripped of compiled_ref)
    [#{
        id => Ind#individual.id,
        network => network_evaluator:strip_compiled_ref(Ind#individual.network),
        fitness => Ind#individual.fitness,
        generation => Generation,
        metrics => Ind#individual.metrics
    } || Ind <- TopPerformers].

%% @private
calculate_fitness(Individual, Config) ->
    EvaluatorModule = Config#neuro_config.evaluator_module,
    Metrics = Individual#individual.metrics,

    try
        EvaluatorModule:calculate_fitness(Metrics)
    catch
        error:undef ->
            neuroevolution_evaluator:default_fitness(Metrics)
    end.

%% @private
-spec should_stop(neuro_state()) -> {true, atom()} | false.
should_stop(#neuro_state{running = false}) ->
    {true, stopped};
should_stop(#neuro_state{
    config = Config,
    total_evaluations = TotalEvaluations,
    generation = Generation,
    best_fitness_ever = BestFitness
}) ->
    case Config#neuro_config.target_fitness of
        TargetFitness when is_number(TargetFitness), BestFitness >= TargetFitness ->
            {true, target_fitness_reached};
        _ ->
            %% Check max_evaluations first (preferred stopping condition)
            case Config#neuro_config.max_evaluations of
                MaxEvals when is_integer(MaxEvals), TotalEvaluations >= MaxEvals ->
                    {true, max_evaluations_reached};
                _ ->
                    %% Fall back to max_generations (deprecated)
                    case Config#neuro_config.max_generations of
                        infinity ->
                            false;
                        MaxGen when is_integer(MaxGen), Generation > MaxGen ->
                            {true, max_generations_reached};
                        _ ->
                            false
                    end
            end
    end.

%% @private
summarize_individual_for_grid(Individual, IsSurvivor, IsOffspring) ->
    #{
        id => Individual#individual.id,
        index => Individual#individual.id,
        fitness => Individual#individual.fitness,
        is_survivor => IsSurvivor,
        is_offspring => IsOffspring,
        wins => maps:get(wins, Individual#individual.metrics, undefined)
    }.

%%% ============================================================================
%%% Internal Functions - Stats & Events
%%% ============================================================================

%% @private
build_stats(State) ->
    Config = State#neuro_state.config,
    StrategyModule = State#neuro_state.strategy_module,
    StrategyState = State#neuro_state.strategy_state,

    %% Get population snapshot from strategy
    Snapshot = evolution_strategy:get_population_snapshot(StrategyModule, StrategyState),

    BaseStats = #{
        generation => State#neuro_state.generation,
        population_size => Config#neuro_config.population_size,
        evaluations_per_individual => Config#neuro_config.evaluations_per_individual,
        games_completed => State#neuro_state.games_completed,
        total_games => State#neuro_state.total_games,
        best_fitness_ever => State#neuro_state.best_fitness_ever,
        running => State#neuro_state.running,
        evaluating => State#neuro_state.evaluating,
        last_gen_best => State#neuro_state.last_gen_best,
        last_gen_avg => State#neuro_state.last_gen_avg,
        generation_history => State#neuro_state.generation_history,
        %% Strategy-provided data
        strategy_module => StrategyModule,
        population_snapshot => Snapshot,
        species_count => maps:get(species_count, Snapshot, 0)
    },

    BaseStats.

%% @private
notify_event(#neuro_state{config = Config}, Event) ->
    notify_event_callback(Config, Event),
    maybe_publish_event(Config, Event).

%% @private
notify_event_callback(Config, Event) ->
    case Config#neuro_config.event_handler of
        undefined ->
            ok;
        {Module, InitArg} ->
            try
                Module:handle_event(Event, InitArg)
            catch
                Class:Reason ->
                    error_logger:error_msg(
                        "[neuroevolution_server] Event handler error: ~p:~p~n",
                        [Class, Reason]
                    )
            end
    end.

%% @private
maybe_publish_event(Config, Event) ->
    case Config#neuro_config.publish_events of
        true ->
            Realm = Config#neuro_config.realm,
            Topic = neuroevolution_events:events_topic(Realm),
            EventMap = event_to_map(Event, Realm),
            neuroevolution_events:publish(Topic, EventMap);
        false ->
            ok
    end.

%% @private
event_to_map({generation_started, Generation}, Realm) ->
    {generation_started, #{
        realm => Realm,
        source => neuroevolution_server,
        generation => Generation,
        timestamp => erlang:system_time(millisecond)
    }};
event_to_map({evaluation_progress, Generation, Completed, Total}, Realm) ->
    {evaluation_progress, #{
        realm => Realm,
        source => neuroevolution_server,
        generation => Generation,
        completed => Completed,
        total => Total,
        timestamp => erlang:system_time(millisecond)
    }};
event_to_map({generation_complete, Data}, Realm) when is_map(Data) ->
    GenStats = maps:get(generation_stats, Data),
    SiloState = maps:get(silo_state, Data, #{}),
    {generation_complete, #{
        realm => Realm,
        source => neuroevolution_server,
        %% PRIMARY: Evaluation-centric progress metrics
        total_evaluations => maps:get(total_evaluations, Data, 0),
        batch_evaluations => maps:get(batch_evaluations, Data, 0),
        %% Velocity-based stagnation metrics (from task_silo)
        improvement_velocity => maps:get(improvement_velocity, SiloState, 0.0),
        stagnation_severity => maps:get(stagnation_severity, SiloState, 0.0),
        %% SECONDARY: Cohort number (formerly "generation" - kept for backward compat)
        cohort => GenStats#generation_stats.generation,
        generation => GenStats#generation_stats.generation,  %% Deprecated alias
        %% Fitness metrics
        best_fitness => GenStats#generation_stats.best_fitness,
        avg_fitness => GenStats#generation_stats.avg_fitness,
        worst_fitness => GenStats#generation_stats.worst_fitness,
        best_individual_id => GenStats#generation_stats.best_individual_id,
        %% Include LC silo state for UI updates
        meta_params => maps:get(meta_params, Data, #{}),
        silo_state => SiloState,
        breeding_events => maps:get(breeding_events, Data, []),
        competitive_history => maps:get(competitive_history, Data, []),
        timestamp => erlang:system_time(millisecond)
    }};
event_to_map({training_started, _Config}, Realm) ->
    {training_started, #{
        realm => Realm,
        source => neuroevolution_server,
        timestamp => erlang:system_time(millisecond)
    }};
event_to_map({training_stopped, Generation}, Realm) ->
    {training_stopped, #{
        realm => Realm,
        source => neuroevolution_server,
        generation => Generation,
        timestamp => erlang:system_time(millisecond)
    }};
event_to_map({population_evaluated, Data}, Realm) when is_map(Data) ->
    {population_evaluated, Data#{
        realm => Realm,
        source => neuroevolution_server,
        timestamp => erlang:system_time(millisecond)
    }};
event_to_map({training_complete, Data}, Realm) when is_map(Data) ->
    {training_complete, Data#{
        realm => Realm,
        source => neuroevolution_server,
        timestamp => erlang:system_time(millisecond)
    }};
event_to_map(Event, Realm) ->
    {other, #{realm => Realm, source => neuroevolution_server, event => Event, timestamp => erlang:system_time(millisecond)}}.

%%% ============================================================================
%%% Internal Functions - Task Silo Integration (Liquid Conglomerate v2)
%%% ============================================================================

%% @private Query task_silo for parameter recommendations after each generation.
%% The task_silo provides L0 (safe defaults) + L1 (tactical adjustments) based
%% on generation performance metrics.
maybe_update_task_silo(GenStats, Config, State) ->
    try
        %% Build stats map for task_silo
        StatsMap = build_task_silo_stats(GenStats, State),

        %% Get recommendations from task_silo
        case task_silo:get_recommendations(whereis(task_silo), StatsMap) of
            Recommendations when is_map(Recommendations) ->
                %% Apply task_silo recommendations to config
                UpdatedConfig = apply_task_silo_params(Config, Recommendations),
                log_task_silo_changes(Config, UpdatedConfig, State#neuro_state.generation),
                {UpdatedConfig, State};
            _ ->
                {Config, State}
        end
    catch
        exit:{noproc, _} ->
            %% Task silo not started, continue with existing config
            {Config, State};
        Class:Reason:Stack ->
            error_logger:warning_msg(
                "[neuroevolution_server] Task silo query failed: ~p:~p~n  ~p~n",
                [Class, Reason, Stack]
            ),
            {Config, State}
    end.

%% @private Build stats map for task_silo from generation stats.
build_task_silo_stats(GenStats, State) ->
    %% Calculate improvement from last generation
    CurrentBest = GenStats#generation_stats.best_fitness,
    LastBest = State#neuro_state.last_gen_best,
    Improvement = case LastBest of
        undefined -> 0.0;
        _ -> CurrentBest - LastBest
    end,

    %% Calculate diversity index from fitness variance
    %% Higher variance = more diversity (normalized 0-1)
    AvgFitness = GenStats#generation_stats.avg_fitness,
    Population = State#neuro_state.population,
    DiversityIndex = compute_diversity_index(Population, AvgFitness),

    %% Species count ratio: actual species / expected species
    %% Expected = sqrt(population_size) as a heuristic
    SpeciesCount = length(State#neuro_state.species),
    PopSize = GenStats#generation_stats.population_size,
    ExpectedSpecies = max(1, round(math:sqrt(PopSize))),
    SpeciesCountRatio = min(1.0, SpeciesCount / ExpectedSpecies),

    %% Average network complexity from top individuals
    TopIndividuals = GenStats#generation_stats.top_individuals,
    AvgComplexity = compute_avg_complexity(TopIndividuals, PopSize),

    %% Resource pressure signal from resource_silo (0.0 = no pressure, 1.0 = high pressure)
    ResourceRecs = State#neuro_state.cached_resource_recommendations,
    ResourcePressure = maps:get(pressure_signal, ResourceRecs, 0.0),

    #{
        best_fitness => CurrentBest,
        avg_fitness => AvgFitness,
        improvement => Improvement,
        generation => GenStats#generation_stats.generation,
        population_size => PopSize,
        %% New metrics for task_silo sensors
        diversity_index => DiversityIndex,
        species_count_ratio => SpeciesCountRatio,
        avg_network_complexity => AvgComplexity,
        resource_pressure_signal => ResourcePressure,
        total_evaluations => State#neuro_state.total_evaluations
    }.

%% @private Compute diversity index from population fitness variance.
%% Returns normalized value 0.0-1.0 where higher = more diverse.
compute_diversity_index([], _AvgFitness) -> 0.5;
compute_diversity_index(Population, AvgFitness) ->
    Fitnesses = [I#individual.fitness || I <- Population],
    N = length(Fitnesses),
    case N of
        0 -> 0.5;
        1 -> 0.0;
        _ ->
            %% Compute variance
            SumSqDiff = lists:foldl(
                fun(F, Acc) -> Acc + math:pow(F - AvgFitness, 2) end,
                0.0,
                Fitnesses
            ),
            Variance = SumSqDiff / N,
            %% Normalize: assume max reasonable variance is 0.25 (for 0-1 fitness)
            min(1.0, math:sqrt(Variance) * 2.0)
    end.

%% @private Compute average network complexity from top individuals.
%% Returns normalized value 0.0-1.0.
compute_avg_complexity([], _PopSize) -> 0.5;
compute_avg_complexity(TopIndividuals, PopSize) ->
    Complexities = [maps:get(complexity, I, 0) || I <- TopIndividuals],
    case Complexities of
        [] -> 0.5;
        _ ->
            AvgComplexity = lists:sum(Complexities) / length(Complexities),
            %% Normalize: assume max reasonable complexity is 100 nodes
            %% Scale by population size (larger pops tend to have more complex networks)
            MaxExpected = 50 + PopSize,
            min(1.0, AvgComplexity / MaxExpected)
    end.

%% @private Apply task_silo recommendations to neuro_config.
apply_task_silo_params(Config, Recommendations) ->
    %% Only update params that task_silo provides
    MutationRate = maps:get(mutation_rate, Recommendations,
                            Config#neuro_config.mutation_rate),
    MutationStrength = maps:get(mutation_strength, Recommendations,
                                Config#neuro_config.mutation_strength),
    SelectionRatio = maps:get(selection_ratio, Recommendations,
                              Config#neuro_config.selection_ratio),

    Config#neuro_config{
        mutation_rate = MutationRate,
        mutation_strength = MutationStrength,
        selection_ratio = SelectionRatio
    }.

%% @private Log significant parameter changes from task_silo.
log_task_silo_changes(OldConfig, NewConfig, Generation) ->
    OldMR = OldConfig#neuro_config.mutation_rate,
    NewMR = NewConfig#neuro_config.mutation_rate,
    OldMS = OldConfig#neuro_config.mutation_strength,
    NewMS = NewConfig#neuro_config.mutation_strength,
    OldSR = OldConfig#neuro_config.selection_ratio,
    NewSR = NewConfig#neuro_config.selection_ratio,

    Threshold = 0.05,  % 5% change threshold
    Changes = [
        {mutation_rate, OldMR, NewMR} ||
        OldMR > 0, abs(NewMR - OldMR) / OldMR > Threshold
    ] ++ [
        {mutation_strength, OldMS, NewMS} ||
        OldMS > 0, abs(NewMS - OldMS) / OldMS > Threshold
    ] ++ [
        {selection_ratio, OldSR, NewSR} ||
        OldSR > 0, abs(NewSR - OldSR) / OldSR > Threshold
    ],

    case Changes of
        [] -> ok;
        _ ->
            error_logger:info_msg(
                "[neuroevolution_server] Cohort ~p: Task silo adjusted params: ~p~n",
                [Generation, Changes]
            )
    end.

%%% ============================================================================
%%% Internal Functions - Meta-Controller Integration (Legacy L2)
%%% ============================================================================

%% @private
maybe_start_meta_controller(Config) ->
    case Config#neuro_config.meta_controller_config of
        undefined ->
            undefined;
        MetaConfig ->
            case meta_controller:start_link(MetaConfig) of
                {ok, Pid} ->
                    error_logger:info_msg(
                        "[neuroevolution_server] Started meta-controller ~p~n",
                        [Pid]
                    ),
                    _ = meta_controller:start_training(Pid),
                    Pid;
                {error, Reason} ->
                    error_logger:error_msg(
                        "[neuroevolution_server] Failed to start meta-controller: ~p~n",
                        [Reason]
                    ),
                    undefined
            end
    end.

%% @private
maybe_update_meta_controller(GenStats, Config, State) ->
    case State#neuro_state.meta_controller of
        undefined ->
            {Config, State};
        MetaPid ->
            try
                NewParams = meta_controller:update(MetaPid, GenStats),
                UpdatedConfig = apply_config_params(Config, NewParams),
                log_param_changes(Config, UpdatedConfig, State#neuro_state.generation),
                {UpdatedConfig, State}
            catch
                Class:Reason ->
                    error_logger:error_msg(
                        "[neuroevolution_server] Meta-controller update failed: ~p:~p~n",
                        [Class, Reason]
                    ),
                    {Config, State}
            end
    end.

%%% ============================================================================
%%% LC Chain Integration (L2→L1→L0 Chained LTC Architecture)
%%% ============================================================================

%% @private Start the chained LTC controller if configured.
maybe_start_lc_chain(Config) ->
    case Config#neuro_config.lc_chain_config of
        undefined ->
            undefined;
        LcChainConfig ->
            case lc_chain:start_link(LcChainConfig) of
                {ok, Pid} ->
                    error_logger:info_msg(
                        "[neuroevolution_server] Started LC chain controller ~p~n",
                        [Pid]
                    ),
                    Pid;
                {error, Reason} ->
                    error_logger:error_msg(
                        "[neuroevolution_server] Failed to start LC chain: ~p~n",
                        [Reason]
                    ),
                    undefined
            end
    end.

%%% ============================================================================
%%% Self-Play Integration
%%% ============================================================================

%% @private Update hyperparameters via LC chain if active.
%%
%% The LC chain supersedes both task_silo and meta_controller when active.
%% Returns {UpdatedConfig, UpdatedState}.
maybe_update_lc_chain(GenStats, Config, State) ->
    case State#neuro_state.lc_chain of
        undefined ->
            {Config, State};
        LcChainPid ->
            try
                %% Build evolution metrics for L2
                EvoMetrics = build_evolution_metrics(GenStats, State),

                %% Build emergent metrics for L0
                EmergentMetrics = build_emergent_metrics(Config, State),

                %% Forward pass through L2→L1→L0 chain
                Hyperparams = lc_chain:forward(LcChainPid, EvoMetrics, EmergentMetrics),

                %% Apply hyperparameters to config
                UpdatedConfig = apply_lc_chain_params(Config, Hyperparams),
                log_lc_chain_changes(Config, UpdatedConfig, State#neuro_state.generation),

                {UpdatedConfig, State}
            catch
                Class:Reason:Stack ->
                    error_logger:error_msg(
                        "[neuroevolution_server] LC chain forward failed: ~p:~p~n~p~n",
                        [Class, Reason, Stack]
                    ),
                    {Config, State}
            end
    end.

%% @private Build evolution_metrics record for LC chain L2 input.
build_evolution_metrics(GenStats, State) ->
    %% Calculate improvement from last generation
    CurrentBest = GenStats#generation_stats.best_fitness,
    LastBest = State#neuro_state.last_gen_best,
    Improvement = case LastBest of
        V when V == 0.0 -> 0.0;
        _ -> (CurrentBest - LastBest) / max(abs(LastBest), 0.001)
    end,

    %% Calculate fitness variance
    AvgFitness = GenStats#generation_stats.avg_fitness,
    BestFitness = GenStats#generation_stats.best_fitness,
    WorstFitness = GenStats#generation_stats.worst_fitness,
    Variance = (BestFitness - WorstFitness) / max(BestFitness, 0.001),

    %% Calculate generation progress
    Config = State#neuro_state.config,
    MaxGens = Config#neuro_config.max_generations,
    CurrentGen = State#neuro_state.generation,
    Progress = case MaxGens of
        infinity -> 0.5;  % Indeterminate
        N when is_integer(N), N > 0 -> CurrentGen / N;
        _ -> 0.5
    end,

    %% Count stagnation (generations without improvement)
    StagnationCounter = count_stagnation(State#neuro_state.generation_history),

    %% Get population diversity and species count
    PopDiversity = calculate_population_diversity(State#neuro_state.population),
    SpeciesCount = length(State#neuro_state.species),

    #evolution_metrics{
        best_fitness = BestFitness,
        avg_fitness = AvgFitness,
        fitness_improvement = Improvement,
        fitness_variance = Variance,
        stagnation_counter = StagnationCounter,
        generation_progress = Progress,
        population_diversity = PopDiversity,
        species_count = max(1, SpeciesCount)
    }.

%% @private Build emergent_metrics record for LC chain L0 input.
build_emergent_metrics(Config, State) ->
    %% Get current hyperparameters
    CurrentMR = Config#neuro_config.mutation_rate,
    CurrentSR = Config#neuro_config.selection_ratio,

    %% Calculate rates from population dynamics
    Population = State#neuro_state.population,
    SurvivalRate = CurrentSR,  % Approximation
    OffspringRate = 1.0 - SurvivalRate,

    %% Calculate convergence rate from history
    ConvergenceRate = calculate_convergence_rate(State#neuro_state.generation_history),

    %% Calculate elite age (how long champion unchanged)
    EliteAge = calculate_elite_age(State#neuro_state.generation_history),

    %% Calculate complexity trend from network sizes in history
    ComplexityTrend = calculate_complexity_trend(State#neuro_state.generation_history),

    %% Calculate average network size from population
    AvgNetworkSize = calculate_avg_network_size(Population),

    #emergent_metrics{
        convergence_rate = ConvergenceRate,
        fitness_plateau_duration = EliteAge,
        current_mutation_rate = CurrentMR,
        current_selection_ratio = CurrentSR,
        survival_rate = SurvivalRate,
        offspring_rate = OffspringRate,
        elite_age = EliteAge,
        complexity_trend = ComplexityTrend,
        avg_network_size = AvgNetworkSize,
        species_extinction_rate = 0.0,
        species_creation_rate = 0.0,
        innovation_rate = 0.0,
        diversity_index = calculate_population_diversity(Population)
    }.

%% @private Calculate average network size (connection count) from population.
calculate_avg_network_size([]) -> 0.0;
calculate_avg_network_size(Population) ->
    Sizes = [estimate_network_size(I#individual.network) || I <- Population],
    lists:sum(Sizes) / length(Sizes).

%% @private Estimate network size from network structure.
%% Returns connection/weight count as a proxy for network complexity.
estimate_network_size(Network) when is_map(Network) ->
    case maps:get(weights, Network, []) of
        Weights when is_list(Weights) -> float(length(Weights));
        _ -> 0.0
    end;
estimate_network_size(_) ->
    0.0.

%% @private Apply LC chain hyperparameters to neuro_config.
apply_lc_chain_params(Config, Hyperparams) ->
    BaseConfig = Config#neuro_config{
        mutation_rate = Hyperparams#lc_hyperparams.mutation_rate,
        mutation_strength = Hyperparams#lc_hyperparams.mutation_strength,
        selection_ratio = Hyperparams#lc_hyperparams.selection_ratio
    },
    %% Apply topology mutation rates if topology_mutation_config is set
    apply_topology_rates(BaseConfig, Hyperparams).

%% @private Apply topology mutation rates to mutation_config if it exists.
apply_topology_rates(Config, Hyperparams) ->
    case Config#neuro_config.topology_mutation_config of
        undefined ->
            Config;
        MutConfig ->
            UpdatedMutConfig = MutConfig#mutation_config{
                add_node_rate = Hyperparams#lc_hyperparams.add_node_rate,
                add_connection_rate = Hyperparams#lc_hyperparams.add_connection_rate
            },
            Config#neuro_config{topology_mutation_config = UpdatedMutConfig}
    end.

%% @private Log significant changes from LC chain.
log_lc_chain_changes(OldConfig, NewConfig, Generation) ->
    OldMR = OldConfig#neuro_config.mutation_rate,
    NewMR = NewConfig#neuro_config.mutation_rate,

    %% Only log if significant change (>5%)
    case abs(NewMR - OldMR) / max(OldMR, 0.01) > 0.05 of
        true ->
            error_logger:info_msg(
                "[lc_chain] Cohort ~p: mutation_rate ~.3f → ~.3f~n",
                [Generation, OldMR, NewMR]
            );
        false ->
            ok
    end.

%% @private Count stagnation from cohort history.
count_stagnation([]) -> 0;
count_stagnation([{_, BestFitness, _} | Rest]) ->
    count_stagnation(Rest, BestFitness, 0).

count_stagnation([], _, Count) -> Count;
count_stagnation([{_, PrevBest, _} | Rest], CurrentBest, Count) ->
    case PrevBest >= CurrentBest of
        true -> count_stagnation(Rest, CurrentBest, Count + 1);
        false -> Count  % Improvement found, stop counting
    end.

%% @private Calculate convergence rate from history.
calculate_convergence_rate([]) -> 0.0;
calculate_convergence_rate([_]) -> 0.0;
calculate_convergence_rate([{_, Best1, _}, {_, Best2, _} | _]) ->
    case Best2 of
        V when V == 0.0 -> 0.0;
        _ -> (Best1 - Best2) / max(abs(Best2), 0.001)
    end.

%% @private Calculate elite age from history.
calculate_elite_age([]) -> 0;
calculate_elite_age([{_, BestFitness, _} | Rest]) ->
    calculate_elite_age(Rest, BestFitness, 0).

calculate_elite_age([], _, Age) -> Age;
calculate_elite_age([{_, PrevBest, _} | Rest], CurrentBest, Age) ->
    case abs(PrevBest - CurrentBest) < 0.0001 of
        true -> calculate_elite_age(Rest, CurrentBest, Age + 1);
        false -> Age
    end.

%% @private Calculate complexity trend from generation history.
%% Returns positive values when fitness is improving (networks getting more effective),
%% negative when fitness declining, and 0 when stable.
%% Uses fitness improvement as a proxy for complexity since we don't track network size.
calculate_complexity_trend([]) -> 0.0;
calculate_complexity_trend([_]) -> 0.0;
calculate_complexity_trend([{_, Best1, _}, {_, Best2, _} | _]) ->
    %% Simple linear trend: positive = improving, negative = declining
    Diff = Best1 - Best2,
    %% Clamp to [-1, 1] range
    clamp(Diff * 10.0, -1.0, 1.0).

%% @private Clamp value to range.
clamp(Value, Min, Max) ->
    max(Min, min(Max, Value)).

%% @private Calculate population diversity (0-1 scale).
calculate_population_diversity([]) -> 0.0;
calculate_population_diversity(Population) ->
    Fitnesses = [F || #{fitness := F} <- Population, is_number(F)],
    case Fitnesses of
        [] -> 0.0;
        [_] -> 0.0;
        _ ->
            Mean = lists:sum(Fitnesses) / length(Fitnesses),
            Variance = lists:sum([(F - Mean) * (F - Mean) || F <- Fitnesses]) / length(Fitnesses),
            StdDev = math:sqrt(Variance),
            %% Normalize to 0-1 (assuming fitness in 0-1 range)
            min(1.0, StdDev * 2)
    end.

%%% ============================================================================
%%% L0 Sensor Updates (Event-Driven)
%%% ============================================================================

%% @private Update task_l0_sensors with current evolution stats.
%%
%% This function pushes evolution statistics to the L0 sensor module,
%% which then publishes sensor events to Phoenix.PubSub via the EventBridge.
%% This enables real-time, event-driven updates to the dashboard UI
%% instead of polling-based state retrieval.
%%
%% The sensors module will throttle events (max 10Hz) and only publish
%% when values change significantly (>1% change).
update_task_l0_sensors(State, Sorted, CompetitiveEntry) ->
    case whereis(task_l0_sensors) of
        undefined ->
            %% task_l0_sensors not running, skip update
            ok;
        _Pid ->
            try
                %% Build stats map for task_l0_sensors:update_stats/1
                Stats = build_task_l0_sensor_stats(State, Sorted, CompetitiveEntry),
                task_l0_sensors:update_stats(Stats)
            catch
                Class:Reason ->
                    error_logger:warning_msg(
                        "[neuroevolution_server] Failed to update task_l0_sensors: ~p:~p~n",
                        [Class, Reason]
                    ),
                    ok
            end
    end.

%% @private Build stats map for task_l0_sensors.
%%
%% The map contains all the fields that task_l0_sensors expects:
%% - best_fitness, avg_fitness, fitness_variance (evolution state)
%% - species_count, avg_complexity, diversity_index (population metrics)
%% - total_evaluations, fitness_norm_factor (progress metrics)
build_task_l0_sensor_stats(State, Sorted, CompetitiveEntry) when is_list(Sorted) ->
    Config = State#neuro_state.config,

    %% Basic fitness stats
    BestFitness = maps:get(best_fitness, CompetitiveEntry, 0.0),
    AvgFitness = maps:get(avg_fitness, CompetitiveEntry, 0.0),
    FitnessVariance = maps:get(fitness_variance, CompetitiveEntry, 0.0),

    %% Species count (from strategy state if available, else 1)
    SpeciesCount = case State#neuro_state.species of
        Species when is_list(Species) -> length(Species);
        _ -> 1
    end,

    %% Average network complexity (use number of connections as proxy)
    AvgComplexity = calculate_avg_network_complexity(Sorted),

    %% Diversity index (from population fitness variance)
    DiversityIndex = calculate_population_diversity(State#neuro_state.population),

    %% Total evaluations
    TotalEvaluations = State#neuro_state.total_evaluations +
        (length(Sorted) * Config#neuro_config.evaluations_per_individual),

    %% Fitness normalization factor (use current best as reference)
    FitnessNormFactor = case BestFitness > 0 of
        true -> max(BestFitness * 2.0, 100.0);
        false -> 100.0
    end,

    #{
        best_fitness => BestFitness,
        avg_fitness => AvgFitness,
        fitness_variance => FitnessVariance,
        species_count => SpeciesCount,
        avg_complexity => AvgComplexity,
        diversity_index => DiversityIndex,
        total_evaluations => TotalEvaluations,
        fitness_norm_factor => FitnessNormFactor
    };
build_task_l0_sensor_stats(_State, _Sorted, _CompetitiveEntry) ->
    %% Empty or invalid sorted population
    #{}.

%% @private Calculate average network complexity from population.
calculate_avg_network_complexity([]) -> 0.0;
calculate_avg_network_complexity(Sorted) ->
    Complexities = lists:filtermap(
        fun(Ind) ->
            case Ind#individual.network of
                Network when is_map(Network) ->
                    %% Try to get connection count as complexity proxy
                    Conns = maps:get(connections, Network, []),
                    Nodes = maps:get(neurons, Network, maps:get(nodes, Network, [])),
                    Complexity = length(Conns) + length(Nodes),
                    {true, Complexity};
                _ ->
                    false
            end
        end,
        Sorted
    ),
    case Complexities of
        [] -> 0.0;
        _ -> lists:sum(Complexities) / length(Complexities)
    end.

%%% ============================================================================
%%% L0 Event-Driven Parameter Application
%%% ============================================================================

%% @private Apply L0 evolution params from event (reactive handler).
%%
%% Called when we receive a {neuro_event, <<"l0.evolution_params">>, Event}.
%% This is the event-driven way to update hyperparameters.
apply_l0_evolution_params(Params, State) ->
    Config = State#neuro_state.config,
    StrategyModule = State#neuro_state.strategy_module,
    StrategyState = State#neuro_state.strategy_state,

    %% Apply to config
    UpdatedConfig = apply_config_params(Config, Params),

    %% Apply to strategy state
    NewStrategyState = evolution_strategy:apply_meta_params(StrategyModule, Params, StrategyState),

    %% Log significant changes
    log_param_changes(Config, UpdatedConfig, State#neuro_state.generation),

    State#neuro_state{
        config = UpdatedConfig,
        strategy_state = NewStrategyState
    }.

%% @private Apply L0 resource params from event (reactive handler).
%%
%% Called when we receive a {neuro_event, <<"l0.resource_params">>, Event}.
%% Handles resource-related parameters like evaluation_timeout.
apply_l0_resource_params(Params, State) ->
    Config = State#neuro_state.config,

    %% Extract resource-specific params
    EvaluationTimeout = maps:get(evaluation_timeout, Params,
                                  Config#neuro_config.evaluation_timeout),

    UpdatedConfig = Config#neuro_config{
        evaluation_timeout = EvaluationTimeout
    },

    State#neuro_state{config = UpdatedConfig}.

%%% ============================================================================
%%% Legacy Parameter Helpers
%%% ============================================================================

%% @private Apply parameters to neuro_config.
apply_config_params(Config, NewParams) ->
    %% Fallback mutation rates
    MutationRate = maps:get(mutation_rate, NewParams, Config#neuro_config.mutation_rate),
    MutationStrength = maps:get(mutation_strength, NewParams, Config#neuro_config.mutation_strength),
    SelectionRatio = maps:get(selection_ratio, NewParams, Config#neuro_config.selection_ratio),
    EvaluationTimeout = maps:get(evaluation_timeout, NewParams, Config#neuro_config.evaluation_timeout),

    %% Layer-specific mutation rates (from L0 actuators)
    %% Only update if present in NewParams, otherwise keep current config value
    ReservoirMutationRate = maps:get(reservoir_mutation_rate, NewParams,
                                      Config#neuro_config.reservoir_mutation_rate),
    ReservoirMutationStrength = maps:get(reservoir_mutation_strength, NewParams,
                                          Config#neuro_config.reservoir_mutation_strength),
    ReadoutMutationRate = maps:get(readout_mutation_rate, NewParams,
                                    Config#neuro_config.readout_mutation_rate),
    ReadoutMutationStrength = maps:get(readout_mutation_strength, NewParams,
                                        Config#neuro_config.readout_mutation_strength),

    %% Update topology mutation config if structural rates are provided
    TopologyMutationConfig = apply_topology_params(
        Config#neuro_config.topology_mutation_config, NewParams),

    Config#neuro_config{
        mutation_rate = MutationRate,
        mutation_strength = MutationStrength,
        selection_ratio = SelectionRatio,
        evaluation_timeout = EvaluationTimeout,
        reservoir_mutation_rate = ReservoirMutationRate,
        reservoir_mutation_strength = ReservoirMutationStrength,
        readout_mutation_rate = ReadoutMutationRate,
        readout_mutation_strength = ReadoutMutationStrength,
        topology_mutation_config = TopologyMutationConfig
    }.

%% @private Apply topology mutation params from L0 actuators.
apply_topology_params(undefined, NewParams) ->
    %% No existing config - create one if structural rates are provided
    case has_topology_params(NewParams) of
        true ->
            #mutation_config{
                add_node_rate = maps:get(add_node_rate, NewParams, 0.03),
                add_connection_rate = maps:get(add_connection_rate, NewParams, 0.05),
                toggle_connection_rate = maps:get(delete_connection_rate, NewParams, 0.01),
                weight_perturb_strength = maps:get(mutation_strength, NewParams, 0.3)
            };
        false ->
            undefined
    end;
apply_topology_params(Config, NewParams) when is_record(Config, mutation_config) ->
    Config#mutation_config{
        add_node_rate = maps:get(add_node_rate, NewParams,
                                  Config#mutation_config.add_node_rate),
        add_connection_rate = maps:get(add_connection_rate, NewParams,
                                        Config#mutation_config.add_connection_rate),
        toggle_connection_rate = maps:get(delete_connection_rate, NewParams,
                                           Config#mutation_config.toggle_connection_rate),
        weight_perturb_strength = maps:get(mutation_strength, NewParams,
                                            Config#mutation_config.weight_perturb_strength)
    }.

%% @private Check if topology params are in the map.
has_topology_params(Params) ->
    maps:is_key(add_node_rate, Params) orelse
    maps:is_key(add_connection_rate, Params) orelse
    maps:is_key(delete_connection_rate, Params).

%% @private Build response map for update_config.
build_config_response(Config) ->
    #{
        mutation_rate => Config#neuro_config.mutation_rate,
        mutation_strength => Config#neuro_config.mutation_strength,
        selection_ratio => Config#neuro_config.selection_ratio,
        evaluation_timeout => Config#neuro_config.evaluation_timeout,
        %% Layer-specific mutation rates
        reservoir_mutation_rate => Config#neuro_config.reservoir_mutation_rate,
        reservoir_mutation_strength => Config#neuro_config.reservoir_mutation_strength,
        readout_mutation_rate => Config#neuro_config.readout_mutation_rate,
        readout_mutation_strength => Config#neuro_config.readout_mutation_strength,
        %% Topology rates (if configured)
        add_node_rate => get_topology_field(Config, add_node_rate),
        add_connection_rate => get_topology_field(Config, add_connection_rate),
        delete_connection_rate => get_topology_field(Config, toggle_connection_rate)
    }.

%% @private Get field from topology_mutation_config if it exists.
get_topology_field(Config, Field) ->
    case Config#neuro_config.topology_mutation_config of
        undefined -> undefined;
        MutConfig when is_record(MutConfig, mutation_config) ->
            case Field of
                add_node_rate -> MutConfig#mutation_config.add_node_rate;
                add_connection_rate -> MutConfig#mutation_config.add_connection_rate;
                toggle_connection_rate -> MutConfig#mutation_config.toggle_connection_rate;
                _ -> undefined
            end
    end.

%% @private
log_param_changes(OldConfig, NewConfig, Generation) ->
    OldMR = OldConfig#neuro_config.mutation_rate,
    NewMR = NewConfig#neuro_config.mutation_rate,
    OldMS = OldConfig#neuro_config.mutation_strength,
    NewMS = NewConfig#neuro_config.mutation_strength,
    OldSR = OldConfig#neuro_config.selection_ratio,
    NewSR = NewConfig#neuro_config.selection_ratio,

    Threshold = 0.10,
    Changes = [
        {mutation_rate, OldMR, NewMR} || abs(NewMR - OldMR) / max(0.01, OldMR) > Threshold
    ] ++ [
        {mutation_strength, OldMS, NewMS} || abs(NewMS - OldMS) / max(0.01, OldMS) > Threshold
    ] ++ [
        {selection_ratio, OldSR, NewSR} || abs(NewSR - OldSR) / max(0.01, OldSR) > Threshold
    ],

    case Changes of
        [] -> ok;
        _ ->
            error_logger:info_msg(
                "[neuroevolution_server] Cohort ~p: Meta-controller adjusted params: ~p~n",
                [Generation, Changes]
            )
    end.

%%% ============================================================================
%%% Internal Functions - Memory Protection
%%% ============================================================================

%% @private Proceed with cohort evaluation (called after memory check passes).
do_evaluate_generation(State) ->
    error_logger:info_msg(
        "[neuroevolution_server] Evaluating cohort ~p~n",
        [State#neuro_state.generation]
    ),

    %% Notify generation start
    notify_event(State, {generation_started, State#neuro_state.generation}),

    Config = State#neuro_state.config,
    Population = State#neuro_state.population,

    %% Choose evaluation mode
    NewState = case Config#neuro_config.evaluation_mode of
        mesh ->
            start_mesh_evaluation(Population, State);
        distributed ->
            start_distributed_evaluation(Population, State);
        direct ->
            start_direct_evaluation(Population, State)
    end,

    {noreply, NewState#neuro_state{evaluating = true}}.

%% @private Check memory pressure via resource_silo and return status.
%% Returns: critical (pause), high (throttle), or normal (continue).
%% Uses resource_silo if available, falls back to direct check if not.
check_memory_pressure() ->
    try
        case resource_silo:get_recommendations() of
            #{action := pause} -> critical;
            #{action := throttle} -> high;
            #{action := continue} -> normal;
            _ -> normal
        end
    catch
        exit:{noproc, _} ->
            %% Resource silo not started, use direct check (fallback)
            check_memory_pressure_direct();
        _:_ ->
            check_memory_pressure_direct()
    end.

%% @private Direct memory pressure check (fallback when resource_silo unavailable).
check_memory_pressure_direct() ->
    case resource_monitor:get_normalized_metrics() of
        #{memory_pressure := Pressure} when Pressure > 0.9 -> critical;
        #{memory_pressure := Pressure} when Pressure > 0.7 -> high;
        _ -> normal
    end.

%% @private Get combined state from all LC v2 silos for event publishing.
%% Returns a map with task_silo and resource_silo state for UI updates.
get_silo_state() ->
    TaskSiloState = try
        task_silo:get_state(whereis(task_silo))
    catch
        _:_ -> #{}
    end,

    ResourceSiloState = try
        resource_silo:get_state()
    catch
        _:_ -> #{}
    end,

    #{
        task_silo => TaskSiloState,
        resource_silo => ResourceSiloState
    }.

%% @private Get memory pressure as percentage.
get_memory_pressure_pct() ->
    Recommendations = try
        resource_silo:get_recommendations()
    catch
        _:_ -> #{}
    end,
    case Recommendations of
        #{memory_pressure := P} -> P * 100;
        _ ->
            case resource_monitor:get_normalized_metrics() of
                #{memory_pressure := P2} -> P2 * 100;
                _ -> 0.0
            end
    end.

%%% ============================================================================
%%% Internal Functions - NIF Memory Management
%%% ============================================================================

%% @private Strip compiled_ref from an individual's network to release NIF memory.
%%
%% CRITICAL: This must be called after evaluation and before storing individuals
%% in the population state. The compiled_ref holds a Rust ResourceArc that
%% keeps native memory alive. Without stripping, NIF memory accumulates
%% unboundedly across generations, causing memory leaks of ~30 GiB/minute.
%%
%% Networks are recompiled on-demand when needed for evaluation.
strip_compiled_ref_from_individual(#individual{network = Network} = Ind) ->
    CleanedNetwork = network_evaluator:strip_compiled_ref(Network),
    Ind#individual{network = CleanedNetwork};
strip_compiled_ref_from_individual(Other) ->
    %% Non-individual record - return as-is
    Other.

%%% ============================================================================
%%% Internal Functions - Network Checkpointing
%%% ============================================================================

%% @private Initialize checkpoint manager if checkpoint_config is set.
maybe_init_checkpoint_manager(Config) ->
    case Config#neuro_config.checkpoint_config of
        undefined ->
            ok;
        CheckpointConfig when is_map(CheckpointConfig) ->
            case checkpoint_manager:init(CheckpointConfig) of
                ok ->
                    error_logger:info_msg(
                        "[neuroevolution_server] Checkpoint manager initialized: ~s~n",
                        [checkpoint_manager:get_checkpoint_dir()]
                    );
                {error, Reason} ->
                    error_logger:warning_msg(
                        "[neuroevolution_server] Failed to init checkpoint manager: ~p~n",
                        [Reason]
                    )
            end
    end.

%% @private Save a checkpoint for the best individual.
%%
%% Called when a new fitness record is achieved or at generation intervals.
%% The checkpoint includes the full individual record with network.
maybe_save_checkpoint(Reason, Best, State) ->
    Config = State#neuro_state.config,
    case Config#neuro_config.checkpoint_config of
        undefined ->
            ok;
        CheckpointConfig when is_map(CheckpointConfig) ->
            ShouldSave = case Reason of
                fitness_record ->
                    maps:get(save_on_fitness_record, CheckpointConfig, true);
                generation_interval ->
                    GenInterval = maps:get(generation_interval, CheckpointConfig, 0),
                    GenInterval > 0 andalso
                        State#neuro_state.generation rem GenInterval =:= 0;
                training_complete ->
                    true;
                _ ->
                    false
            end,

            case ShouldSave of
                true ->
                    save_checkpoint_internal(Reason, Best, State, CheckpointConfig);
                false ->
                    ok
            end
    end.

%% @private Actually save the checkpoint.
save_checkpoint_internal(Reason, Best, State, CheckpointConfig) ->
    %% Strip compiled_ref to prevent serializing NIF resources
    CleanedBest = strip_compiled_ref_from_individual(Best),

    Metadata = #{
        reason => Reason,
        fitness => Best#individual.fitness,
        generation => State#neuro_state.generation,
        total_evaluations => State#neuro_state.total_evaluations,
        individual_id => Best#individual.id
    },

    case checkpoint_manager:save_checkpoint(CleanedBest, Metadata, CheckpointConfig) of
        ok ->
            %% Prune old checkpoints if configured
            MaxPerReason = maps:get(max_checkpoints_per_reason, CheckpointConfig, 20),
            checkpoint_manager:prune_checkpoints(CheckpointConfig#{max_per_reason => MaxPerReason});
        {error, Reason} ->
            error_logger:warning_msg(
                "[neuroevolution_server] Failed to save checkpoint: ~p~n",
                [Reason]
            )
    end.
