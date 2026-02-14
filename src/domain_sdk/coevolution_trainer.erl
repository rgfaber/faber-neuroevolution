%% @doc Coevolution Trainer - Evolves multiple species simultaneously.
%%
%% This module orchestrates competitive or cooperative coevolution of
%% multiple species in a shared environment.
%%
%% == Overview ==
%%
%% ```
%% ┌─────────────────────────────────────────────────────────────┐
%% │                  Coevolution Trainer                        │
%% ├─────────────────────────────────────────────────────────────┤
%% │                                                             │
%% │  Generation Loop:                                           │
%% │  ┌─────────────────────────────────────────────────────┐   │
%% │  │ 1. Spawn mixed population in shared environment     │   │
%% │  │ 2. Run episode (all species interact)               │   │
%% │  │ 3. Evaluate fitness per species                     │   │
%% │  │ 4. Update subspecies (behavioral clustering)        │   │
%% │  │ 5. Select survivors per species                     │   │
%% │  │ 6. Reproduce within species                         │   │
%% │  │ 7. Repeat until termination                         │   │
%% │  └─────────────────────────────────────────────────────┘   │
%% │                                                             │
%% └─────────────────────────────────────────────────────────────┘
%% '''
%%
%% == Evaluation Modes ==
%%
%% <ul>
%%   <li><b>competitive</b> - Species compete (predator/prey dynamics)</li>
%%   <li><b>cooperative</b> - Species benefit from collaboration</li>
%%   <li><b>mixed</b> - Intra-species competition, inter-species cooperation</li>
%% </ul>
%%
%% == Usage ==
%%
%% ```
%% {ok, Trainer} = coevolution_trainer:start(#{
%%     species => [forager_species, predator_species],
%%     environment => hex_arena_env,
%%     population_sizes => #{forager => 100, predator => 30},
%%     evaluation_mode => competitive,
%%     max_generations => 500
%% }).
%%
%% %% Monitor progress
%% Stats = coevolution_trainer:get_stats(Trainer).
%%
%% %% Get best networks per species
%% Champions = coevolution_trainer:get_champions(Trainer).
%% '''
%%
%% @author R.G. Lefever
%% @copyright 2024-2026 R.G. Lefever
%% @see species_registry
%% @see agent_species
-module(coevolution_trainer).
-behaviour(gen_server).

%%% ============================================================================
%%% Types
%%% ============================================================================

-type evaluation_mode() :: competitive | cooperative | mixed.

-type trainer_config() :: #{
    species := [module()],
    environment := module(),
    population_sizes => #{atom() => pos_integer()},
    evaluation_mode => evaluation_mode(),
    max_generations => pos_integer(),
    episodes_per_eval => pos_integer(),
    selection_ratio => float(),
    env_config => map(),
    subspeciation_enabled => boolean(),
    mutation_strength => float(),
    crossover_rate => float()
}.

-type trainer_state() :: #{
    registry := species_registry:registry(),
    config := trainer_config(),
    generation := non_neg_integer(),
    running := boolean(),
    stats := #{atom() => [generation_stats()]},
    champions := #{atom() => term()}
}.

-type generation_stats() :: #{
    generation := non_neg_integer(),
    best_fitness := float(),
    avg_fitness := float(),
    subspecies_count := non_neg_integer(),
    population_size := non_neg_integer()
}.

-export_type([
    evaluation_mode/0,
    trainer_config/0,
    trainer_state/0,
    generation_stats/0
]).

%%% ============================================================================
%%% API Functions
%%% ============================================================================

-export([
    start/1,
    start_link/1,
    stop/1,
    pause/1,
    resume/1,
    get_stats/1,
    get_champions/1,
    get_generation/1,
    get_registry/1,
    run_generation/1
]).

%% gen_server callbacks
-export([
    init/1,
    handle_call/3,
    handle_cast/2,
    handle_info/2,
    terminate/2
]).

%% @doc Starts the coevolution trainer.
-spec start(Config) -> {ok, pid()} | {error, term()} when
    Config :: trainer_config().
start(Config) ->
    gen_server:start(?MODULE, Config, []).

%% @doc Starts the coevolution trainer with link.
-spec start_link(Config) -> {ok, pid()} | {error, term()} when
    Config :: trainer_config().
start_link(Config) ->
    gen_server:start_link(?MODULE, Config, []).

%% @doc Stops the trainer.
-spec stop(Trainer) -> ok when
    Trainer :: pid().
stop(Trainer) ->
    gen_server:stop(Trainer).

%% @doc Pauses training.
-spec pause(Trainer) -> ok when
    Trainer :: pid().
pause(Trainer) ->
    gen_server:call(Trainer, pause).

%% @doc Resumes training.
-spec resume(Trainer) -> ok when
    Trainer :: pid().
resume(Trainer) ->
    gen_server:cast(Trainer, resume).

%% @doc Gets training statistics per species.
-spec get_stats(Trainer) -> #{atom() => [generation_stats()]} when
    Trainer :: pid().
get_stats(Trainer) ->
    gen_server:call(Trainer, get_stats).

%% @doc Gets champion networks per species.
-spec get_champions(Trainer) -> #{atom() => term()} when
    Trainer :: pid().
get_champions(Trainer) ->
    gen_server:call(Trainer, get_champions).

%% @doc Gets current generation number.
-spec get_generation(Trainer) -> non_neg_integer() when
    Trainer :: pid().
get_generation(Trainer) ->
    gen_server:call(Trainer, get_generation).

%% @doc Gets the species registry.
-spec get_registry(Trainer) -> species_registry:registry() when
    Trainer :: pid().
get_registry(Trainer) ->
    gen_server:call(Trainer, get_registry).

%% @doc Manually triggers one generation (for testing).
-spec run_generation(Trainer) -> ok when
    Trainer :: pid().
run_generation(Trainer) ->
    gen_server:cast(Trainer, run_generation).

%%% ============================================================================
%%% gen_server Callbacks
%%% ============================================================================

init(Config) ->
    %% Create species registry
    Species = maps:get(species, Config),
    Environment = maps:get(environment, Config),
    PopSizes = maps:get(population_sizes, Config, #{}),

    case species_registry:new(#{
        species => Species,
        environment => Environment,
        population_sizes => PopSizes
    }) of
        {ok, Registry} ->
            %% Initialize populations
            Registry1 = initialize_populations(Registry, Config),

            State = #{
                registry => Registry1,
                config => Config,
                generation => 0,
                running => true,
                stats => #{},
                champions => #{}
            },

            %% Start training loop
            self() ! run_generation,

            {ok, State};
        {error, Reason} ->
            {stop, Reason}
    end.

handle_call(pause, _From, State) ->
    {reply, ok, State#{running := false}};

handle_call(get_stats, _From, State) ->
    {reply, maps:get(stats, State), State};

handle_call(get_champions, _From, State) ->
    {reply, maps:get(champions, State), State};

handle_call(get_generation, _From, State) ->
    {reply, maps:get(generation, State), State};

handle_call(get_registry, _From, State) ->
    {reply, maps:get(registry, State), State};

handle_call(_Request, _From, State) ->
    {reply, {error, unknown_request}, State}.

handle_cast(resume, State) ->
    self() ! run_generation,
    {noreply, State#{running := true}};

handle_cast(run_generation, State) ->
    case maps:get(running, State) of
        true ->
            NewState = do_generation(State),
            {noreply, NewState};
        false ->
            {noreply, State}
    end;

handle_cast(_Msg, State) ->
    {noreply, State}.

handle_info(run_generation, State) ->
    case maps:get(running, State) of
        true ->
            NewState = do_generation(State),
            %% Check termination
            case should_terminate(NewState) of
                true ->
                    log_info("Coevolution complete after ~p generations",
                             [maps:get(generation, NewState)]),
                    {noreply, NewState#{running := false}};
                false ->
                    %% Continue training
                    erlang:send_after(10, self(), run_generation),
                    {noreply, NewState}
            end;
        false ->
            {noreply, State}
    end;

handle_info(_Info, State) ->
    {noreply, State}.

terminate(_Reason, _State) ->
    ok.

%%% ============================================================================
%%% Internal Functions
%%% ============================================================================

%% @private
initialize_populations(Registry, _Config) ->
    SpeciesIds = species_registry:list_species(Registry),
    lists:foldl(
        fun(SpeciesId, Reg) ->
            PopSize = species_registry:get_population_size(Reg, SpeciesId),
            {ok, SpeciesEntry} = species_registry:get_species(Reg, SpeciesId),
            SpeciesConfig = maps:get(config, SpeciesEntry),
            Topology = maps:get(topology, SpeciesConfig),

            %% Create initial random networks
            Networks = create_random_population(Topology, PopSize),
            species_registry:set_population(Reg, SpeciesId, Networks)
        end,
        Registry,
        SpeciesIds
    ).

%% @private
create_random_population(Topology, Size) ->
    {Inputs, Hidden, Outputs} = Topology,
    [network_evaluator:create_feedforward(Inputs, Hidden, Outputs, tanh)
     || _ <- lists:seq(1, Size)].

%% @private
do_generation(State) ->
    Generation = maps:get(generation, State) + 1,
    Registry = maps:get(registry, State),
    Config = maps:get(config, State),

    log_info("=== Generation ~p ===", [Generation]),

    %% Evaluate all species
    {FitnessResults, Registry1} = evaluate_all_species(Registry, Config),

    %% Update subspecies (behavioral clustering)
    Registry2 = case maps:get(subspeciation_enabled, Config, true) of
        true -> update_all_subspecies(Registry1, FitnessResults);
        false -> Registry1
    end,

    %% Select and reproduce per species
    Registry3 = evolve_all_species(Registry2, FitnessResults, Config),

    %% Update stats
    Stats = maps:get(stats, State),
    NewStats = update_stats(Stats, FitnessResults, Generation),

    %% Update champions
    Champions = update_champions(maps:get(champions, State), FitnessResults),

    State#{
        generation := Generation,
        registry := Registry3,
        stats := NewStats,
        champions := Champions
    }.

%% @private
evaluate_all_species(Registry, Config) ->
    SpeciesIds = species_registry:list_species(Registry),
    _Environment = maps:get(environment, Registry),
    EnvConfig = maps:get(env_config, Config, #{}),
    EpisodesPerEval = maps:get(episodes_per_eval, Config, 1),
    _EvalMode = maps:get(evaluation_mode, Config, competitive),

    %% Evaluate each species
    Results = lists:map(
        fun(SpeciesId) ->
            {ok, Bridge} = species_registry:get_bridge(Registry, SpeciesId),
            Population = species_registry:get_population(Registry, SpeciesId),

            %% Evaluate each network in the population
            Fitness = lists:map(
                fun(Network) ->
                    AvgFitness = evaluate_network(Bridge, Network, EnvConfig, EpisodesPerEval, SpeciesId),
                    {Network, AvgFitness}
                end,
                Population
            ),

            %% Sort by fitness (descending)
            SortedFitness = lists:reverse(lists:keysort(2, Fitness)),
            {SpeciesId, SortedFitness}
        end,
        SpeciesIds
    ),

    {maps:from_list(Results), Registry}.

%% @private
evaluate_network(Bridge, Network, EnvConfig, Episodes, SpeciesId) ->
    Fitnesses = [begin
        case agent_bridge:run_episode(Bridge, Network, EnvConfig, SpeciesId) of
            {ok, Fitness, _Metrics} -> Fitness;
            {ok, _Metrics} -> 0.0;
            {error, _} -> 0.0
        end
    end || _ <- lists:seq(1, Episodes)],
    lists:sum(Fitnesses) / length(Fitnesses).

%% @private
update_all_subspecies(Registry, FitnessResults) ->
    maps:fold(
        fun(SpeciesId, FitnessList, Reg) ->
            %% Get subspeciation threshold for this species
            {ok, Entry} = species_registry:get_species(Reg, SpeciesId),
            SpeciesConfig = maps:get(config, Entry),
            Threshold = maps:get(subspeciation_threshold, SpeciesConfig, infinity),

            case Threshold of
                infinity ->
                    Reg;
                T when is_float(T) ->
                    %% Create behavioral fingerprints (use fitness as simple fingerprint for now)
                    %% TODO: Use actual behavioral fingerprinting
                    Fingerprints = [{Net, [F]} || {Net, F} <- FitnessList],
                    SubspeciesMap = species_identifier:identify_species(Fingerprints, T, #{}),
                    species_registry:update_subspecies(Reg, SpeciesId, SubspeciesMap)
            end
        end,
        Registry,
        FitnessResults
    ).

%% @private
evolve_all_species(Registry, FitnessResults, Config) ->
    SelectionRatio = maps:get(selection_ratio, Config, 0.2),
    MutationStrength = maps:get(mutation_strength, Config, 0.3),
    CrossoverRate = maps:get(crossover_rate, Config, 0.7),

    maps:fold(
        fun(SpeciesId, FitnessList, Reg) ->
            %% Select top performers
            PopSize = length(FitnessList),
            SurvivorsCount = max(2, round(PopSize * SelectionRatio)),
            Survivors = lists:sublist(FitnessList, SurvivorsCount),

            %% Reproduce to fill population with mutation/crossover
            NewPopulation = reproduce(Survivors, PopSize, MutationStrength, CrossoverRate),
            species_registry:set_population(Reg, SpeciesId, NewPopulation)
        end,
        Registry,
        FitnessResults
    ).

%% @private
%% Reproduce population through selection, crossover, and mutation.
%% @param Survivors List of {Network, Fitness} tuples sorted by fitness
%% @param TargetSize Target population size
%% @param MutationStrength Standard deviation for weight perturbation
%% @param CrossoverRate Probability of using crossover vs mutation only
reproduce(Survivors, TargetSize, MutationStrength, CrossoverRate) ->
    SurvivorNetworks = [N || {N, _F} <- Survivors],
    NumSurvivors = length(SurvivorNetworks),

    %% Keep survivors + create offspring
    NumOffspring = TargetSize - NumSurvivors,

    Offspring = [begin
        case NumSurvivors of
            1 ->
                %% Only one parent - mutate directly
                Parent = hd(SurvivorNetworks),
                mutate_network(Parent, MutationStrength);
            _ ->
                %% Decide whether to use crossover
                case rand:uniform() < CrossoverRate of
                    true ->
                        %% Two parents - do crossover then mutate
                        Parent1Idx = rand:uniform(NumSurvivors),
                        Parent2Idx = case rand:uniform(NumSurvivors) of
                            Parent1Idx when NumSurvivors > 1 ->
                                %% Pick different parent
                                ((Parent1Idx) rem NumSurvivors) + 1;
                            Idx -> Idx
                        end,
                        Parent1 = lists:nth(Parent1Idx, SurvivorNetworks),
                        Parent2 = lists:nth(Parent2Idx, SurvivorNetworks),
                        %% Crossover then mutate
                        Child = crossover_networks(Parent1, Parent2),
                        mutate_network(Child, MutationStrength);
                    false ->
                        %% Single parent - just mutate
                        Parent = lists:nth(rand:uniform(NumSurvivors), SurvivorNetworks),
                        mutate_network(Parent, MutationStrength)
                end
        end
    end || _ <- lists:seq(1, NumOffspring)],

    SurvivorNetworks ++ Offspring.

%% @private
%% Crossover two networks using uniform crossover
crossover_networks(Parent1, Parent2) ->
    network_factory:crossover(Parent1, Parent2).

%% @private
%% Mutate network weights using gaussian perturbation
%% Uses network_factory:mutate/2 which applies gaussian noise
mutate_network(Network, MutationStrength) ->
    network_factory:mutate(Network, MutationStrength).

%% @private
update_stats(Stats, FitnessResults, Generation) ->
    maps:fold(
        fun(SpeciesId, FitnessList, Acc) ->
            Fitnesses = [F || {_N, F} <- FitnessList],
            GenStats = #{
                generation => Generation,
                best_fitness => case Fitnesses of [] -> 0.0; _ -> lists:max(Fitnesses) end,
                avg_fitness => case Fitnesses of [] -> 0.0; _ -> lists:sum(Fitnesses) / length(Fitnesses) end,
                subspecies_count => 1,  %% TODO: Get actual count
                population_size => length(FitnessList)
            },
            SpeciesStats = maps:get(SpeciesId, Acc, []),
            maps:put(SpeciesId, [GenStats | SpeciesStats], Acc)
        end,
        Stats,
        FitnessResults
    ).

%% @private
update_champions(Champions, FitnessResults) ->
    maps:fold(
        fun(SpeciesId, FitnessList, Acc) ->
            case FitnessList of
                [{BestNet, BestFitness} | _] ->
                    CurrentBest = maps:get(SpeciesId, Acc, {undefined, 0.0}),
                    case BestFitness > element(2, CurrentBest) of
                        true -> maps:put(SpeciesId, {BestNet, BestFitness}, Acc);
                        false -> Acc
                    end;
                [] -> Acc
            end
        end,
        Champions,
        FitnessResults
    ).

%% @private
should_terminate(State) ->
    Config = maps:get(config, State),
    Generation = maps:get(generation, State),
    MaxGenerations = maps:get(max_generations, Config, 1000),
    Generation >= MaxGenerations.

%% @private
log_info(Format, Args) ->
    io:format("[coevolution_trainer] " ++ Format ++ "~n", Args).
