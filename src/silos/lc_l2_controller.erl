%% @doc L2 Strategic Controller for Liquid Conglomerate Silos.
%%
%% Part of the Liquid Conglomerate v2 architecture. This module implements
%% the L2 strategic layer that learns to tune L1's hyperparameters based on
%% long-term L0 performance.
%%
%% == Architecture ==
%%
%% L2 is the slowest-evolving layer that:
%% - Observes L0 performance over many τ_L1 cycles
%% - Outputs adjustments to L1's hyperparameters
%% - Provides a stable strategic platform for the adaptation hierarchy
%%
%% == Learning Mechanism ==
%%
%% L2 learns through slow evolution or Bayesian optimization:
%% 1. Tiny population (3-5 L2 variants) or single individual with exploration
%% 2. Each L2 variant's settings tested over many L1 cycles
%% 3. Fitness = long-term cumulative L0 performance
%% 4. Very slow evolution (τ_L2 = 5 min for Resource, 10000 evals for Task)
%%
%% == Usage ==
%%
%% %% Create L2 controller for a silo
%% Config = #{
%%     silo_type => task,
%%     morphology_module => task_l0_morphology,
%%     tau_l2 => 300000,  % 5 minutes
%%     l1_hyperparameters => task_l0_morphology:l1_hyperparameters()
%% },
%% {ok, Pid} = lc_l2_controller:start_link(Config),
%%
%% %% Update with L1 performance (called every τ_L1)
%% lc_l2_controller:observe_l1_performance(Pid, L1Metrics),
%%
%% %% Get current L1 hyperparameter settings
%% L1Hyperparams = lc_l2_controller:get_l1_hyperparameters(Pid),
%%
%% @author R.G. Lefever
%% @copyright 2024-2026 R.G. Lefever
-module(lc_l2_controller).
-behaviour(gen_server).

%% API
-export([
    start_link/1,
    observe_l1_performance/2,
    get_l1_hyperparameters/1,
    get_performance_summary/1
]).

%% gen_server callbacks
-export([
    init/1,
    handle_call/3,
    handle_cast/2,
    handle_info/2,
    terminate/2
]).

-record(state, {
    %% Configuration
    silo_type :: atom(),
    morphology_module :: module(),
    tau_l2 :: pos_integer(),
    l1_hyperparameter_names :: [atom()],
    l1_bounds :: map(),

    %% Current L1 hyperparameters (what L2 outputs)
    current_l1_hyperparameters :: map(),

    %% Performance observation over long window
    long_term_history :: [map()],
    long_term_window_size :: pos_integer(),

    %% Exploration state (for simple exploration strategy)
    exploration_rate :: float(),
    exploration_direction :: map(),
    best_hyperparameters :: map(),
    best_cumulative_reward :: float(),

    %% Timing
    last_adjustment_time :: integer(),
    cycles_since_adjustment :: non_neg_integer()
}).

%%% ============================================================================
%%% API Functions
%%% ============================================================================

%% @doc Start L2 controller with configuration.
-spec start_link(map()) -> {ok, pid()} | ignore | {error, term()}.
start_link(Config) ->
    gen_server:start_link(?MODULE, Config, []).

%% @doc Observe L1 performance metrics.
%%
%% Called every τ_L1 with L1's performance summary
-spec observe_l1_performance(pid(), map()) -> ok.
observe_l1_performance(Pid, L1Metrics) ->
    gen_server:cast(Pid, {observe_l1, L1Metrics}).

%% @doc Get current L1 hyperparameter settings.
-spec get_l1_hyperparameters(pid()) -> map().
get_l1_hyperparameters(Pid) ->
    gen_server:call(Pid, get_l1_hyperparameters).

%% @doc Get performance summary for diagnostics.
-spec get_performance_summary(pid()) -> map().
get_performance_summary(Pid) ->
    gen_server:call(Pid, get_performance_summary).

%%% ============================================================================
%%% gen_server Callbacks
%%% ============================================================================

init(Config) ->
    SiloType = maps:get(silo_type, Config, unknown),
    MorphologyModule = maps:get(morphology_module, Config),
    TauL2 = maps:get(tau_l2, Config, 300000),  % 5 minutes default
    LongTermWindowSize = maps:get(long_term_window_size, Config, 50),
    ExplorationRate = maps:get(exploration_rate, Config, 0.1),

    %% Get L1 hyperparameter definitions
    L1HyperparamNames = MorphologyModule:l1_hyperparameters(),
    L1Defaults = MorphologyModule:get_l1_defaults(),
    L1Bounds = MorphologyModule:get_l1_bounds(),

    %% Initialize exploration direction randomly
    ExplorationDirection = initialize_exploration_direction(L1HyperparamNames),

    State = #state{
        silo_type = SiloType,
        morphology_module = MorphologyModule,
        tau_l2 = TauL2,
        l1_hyperparameter_names = L1HyperparamNames,
        l1_bounds = L1Bounds,
        current_l1_hyperparameters = L1Defaults,
        long_term_history = [],
        long_term_window_size = LongTermWindowSize,
        exploration_rate = ExplorationRate,
        exploration_direction = ExplorationDirection,
        best_hyperparameters = L1Defaults,
        best_cumulative_reward = -999999.0,
        last_adjustment_time = erlang:monotonic_time(millisecond),
        cycles_since_adjustment = 0
    },

    %% Subscribe to L1 metrics events (event-driven pattern)
    controller_events:subscribe_to_l1_metrics(SiloType),

    {ok, State}.

handle_call(get_l1_hyperparameters, _From, State) ->
    {reply, State#state.current_l1_hyperparameters, State};

handle_call(get_performance_summary, _From, State) ->
    Summary = #{
        silo_type => State#state.silo_type,
        cycles_observed => length(State#state.long_term_history),
        current_hyperparameters => State#state.current_l1_hyperparameters,
        best_hyperparameters => State#state.best_hyperparameters,
        best_cumulative_reward => State#state.best_cumulative_reward,
        exploration_rate => State#state.exploration_rate
    },
    {reply, Summary, State};

handle_call(_Request, _From, State) ->
    {reply, {error, unknown_request}, State}.

handle_cast({observe_l1, L1Metrics}, State) ->
    %% DEPRECATED: Use controller_events:publish_l1_metrics/2 instead
    logger:warning("[DEPRECATED] lc_l2_controller:observe_l1_performance/2 called. "
                   "Use controller_events:publish_l1_metrics/2 instead."),
    NewState = do_observe_l1(L1Metrics, State),
    {noreply, NewState};

handle_cast(_Msg, State) ->
    {noreply, State}.

%% Handle L1 metrics event (event-driven pattern)
handle_info({neuro_event, Topic, #{silo := Silo, metrics := L1Metrics}}, State) ->
    ExpectedTopic = controller_events:l1_metrics_topic(State#state.silo_type),
    case Topic =:= ExpectedTopic orelse Silo =:= State#state.silo_type of
        true ->
            NewState = do_observe_l1(L1Metrics, State),
            {noreply, NewState};
        false ->
            {noreply, State}
    end;

handle_info(_Info, State) ->
    {noreply, State}.

terminate(_Reason, _State) ->
    ok.

%%% ============================================================================
%%% Internal Functions - L1 Observation
%%% ============================================================================

%% @private Process L1 metrics observation (shared by cast and event handlers).
do_observe_l1(L1Metrics, State) ->
    %% Add to long-term history
    NewHistory = add_to_history(L1Metrics, State#state.long_term_history,
                                State#state.long_term_window_size),

    %% Check if it's time for an adjustment
    Now = erlang:monotonic_time(millisecond),
    TimeSinceAdjustment = Now - State#state.last_adjustment_time,
    NewCycleCount = State#state.cycles_since_adjustment + 1,

    case TimeSinceAdjustment >= State#state.tau_l2 of
        true ->
            %% Time to evaluate and potentially adjust
            evaluate_and_adjust(State#state{
                long_term_history = NewHistory,
                cycles_since_adjustment = 0,
                last_adjustment_time = Now
            });
        false ->
            State#state{
                long_term_history = NewHistory,
                cycles_since_adjustment = NewCycleCount
            }
    end.

%%% ============================================================================
%%% Internal Functions - L2 Adjustment Strategy
%%% ============================================================================

%% @private Evaluate performance and adjust L1 hyperparameters.
evaluate_and_adjust(State) ->
    %% Compute cumulative reward over observation window
    CumulativeReward = compute_cumulative_reward(State#state.long_term_history),

    %% Check if this is better than best found
    IsBetter = CumulativeReward > State#state.best_cumulative_reward,

    %% Update best if improved
    {NewBestHyperparams, NewBestReward} = case IsBetter of
        true ->
            {State#state.current_l1_hyperparameters, CumulativeReward};
        false ->
            {State#state.best_hyperparameters, State#state.best_cumulative_reward}
    end,

    %% Decide exploration strategy
    NewHyperparams = case IsBetter of
        true ->
            %% Continue in current direction with some probability
            continue_exploration(State);
        false ->
            %% Return to best and try new direction
            explore_from_best(NewBestHyperparams, State)
    end,

    %% Update exploration direction
    NewExplorationDirection = case IsBetter of
        true -> State#state.exploration_direction;
        false -> randomize_exploration_direction(State#state.l1_hyperparameter_names)
    end,

    NewState = State#state{
        current_l1_hyperparameters = NewHyperparams,
        best_hyperparameters = NewBestHyperparams,
        best_cumulative_reward = NewBestReward,
        exploration_direction = NewExplorationDirection
    },

    %% Publish L2 guidance (event-driven pattern)
    Guidance = #{
        l1_hyperparameters => NewHyperparams,
        exploration_rate => State#state.exploration_rate,
        adaptation_speed => case IsBetter of true -> 1.2; false -> 0.8 end
    },
    controller_events:publish_l2_guidance(State#state.silo_type, Guidance),

    NewState.

%% @private Compute cumulative reward from history.
compute_cumulative_reward([]) -> 0.0;
compute_cumulative_reward(History) ->
    %% Sum of L0 average rewards reported by L1
    lists:foldl(
        fun(Metrics, Acc) ->
            L0AvgReward = maps:get(l0_avg_reward, Metrics,
                                   maps:get(avg_reward, Metrics, 0.0)),
            Acc + L0AvgReward
        end,
        0.0,
        History
    ).

%% @private Continue exploration in current direction.
continue_exploration(State) ->
    ExplorationRate = State#state.exploration_rate,
    Direction = State#state.exploration_direction,
    CurrentHyperparams = State#state.current_l1_hyperparameters,
    Bounds = State#state.l1_bounds,

    %% Move in current direction
    maps:fold(
        fun(Name, Dir, Acc) ->
            case maps:get(Name, CurrentHyperparams, undefined) of
                undefined -> Acc;
                CurrentValue ->
                    %% Get bounds for this parameter
                    {Min, Max} = maps:get(Name, Bounds, {0.0, 1.0}),
                    Range = Max - Min,
                    %% Apply directional step
                    Step = Dir * ExplorationRate * Range * 0.1,
                    NewValue = max(Min, min(Max, CurrentValue + Step)),
                    maps:put(Name, NewValue, Acc)
            end
        end,
        CurrentHyperparams,
        Direction
    ).

%% @private Explore from best hyperparameters with new direction.
explore_from_best(BestHyperparams, State) ->
    ExplorationRate = State#state.exploration_rate,
    Direction = State#state.exploration_direction,
    Bounds = State#state.l1_bounds,

    %% Start from best and apply new direction
    maps:fold(
        fun(Name, Dir, Acc) ->
            case maps:get(Name, BestHyperparams, undefined) of
                undefined -> Acc;
                BestValue ->
                    {Min, Max} = maps:get(Name, Bounds, {0.0, 1.0}),
                    Range = Max - Min,
                    Step = Dir * ExplorationRate * Range * 0.05,  % Smaller initial step
                    NewValue = max(Min, min(Max, BestValue + Step)),
                    maps:put(Name, NewValue, Acc)
            end
        end,
        BestHyperparams,
        Direction
    ).

%%% ============================================================================
%%% Internal Functions - Utilities
%%% ============================================================================

%% @private Initialize exploration direction randomly.
initialize_exploration_direction(HyperparamNames) ->
    lists:foldl(
        fun(Name, Acc) ->
            %% Random direction: -1, 0, or +1
            Dir = case rand:uniform(3) of
                1 -> -1.0;
                2 -> 0.0;
                3 -> 1.0
            end,
            maps:put(Name, Dir, Acc)
        end,
        #{},
        HyperparamNames
    ).

%% @private Randomize exploration direction.
randomize_exploration_direction(HyperparamNames) ->
    initialize_exploration_direction(HyperparamNames).

%% @private Add observation to history with max size.
add_to_history(Observation, History, MaxSize) ->
    lists:sublist([Observation | History], MaxSize).
