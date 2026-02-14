%%% @doc Competitive Coevolution Manager.
%%%
%%% Manages the Red Team (champion archive) and coordinates competitive
%%% coevolution between Blue Team (evolving population) and Red Team.
%%%
%%% == Red Team vs Blue Team ==
%%%
%%% IMPORTANT NAMING CONVENTION:
%%% - Red Team = CHAMPIONS / Hall of Fame / Elite Archive
%%%   These are the "good guys" - networks that have proven themselves
%%%   and now serve as the benchmark that others must beat.
%%% - Blue Team = CHALLENGERS / Evolving Population
%%%   These are actively evolving, trying to beat the Red Team champions.
%%%
%%% The naming follows the "Red Queen" hypothesis from evolutionary biology,
%%% where the Red Queen (champion) sets the pace that others must match.
%%%
%%% This implements true competitive coevolution where:
%%% - Blue Team: Main evolving population (managed by neuroevolution_server)
%%% - Red Team: Elite champion archive that also evolves (managed here)
%%%
%%% Both teams evolve, creating an arms race dynamic where each team must
%%% continuously improve to beat the other.
%%%
%%% == Integration with Evaluation ==
%%%
%%% The domain evaluator calls this manager to get Red Team opponents:
%%%
%%% %% In domain evaluator:
%%% evaluate(Individual, Options) ->
%%%     ManagerPid = maps:get(coevolution_manager, Options),
%%%     BatchNetworks = maps:get(batch_networks, Options, []),
%%%     {ok, Opponent} = coevolution_manager:get_red_team_opponent(ManagerPid, BatchNetworks),
%%%     evaluate_vs_network(Individual, Opponent).
%%%
%%% == Arms Race Dynamics ==
%%%
%%% After Blue Team evaluation, results are reported to update Red Team:
%%% - Red Team members gain fitness when they beat Blue Team members
%%% - Immigration allows genetic material to flow between teams
%%%
%%% @end
-module(coevolution_manager).

-behaviour(gen_server).

%% API
-export([
    start_link/1,
    start_link/2,
    get_red_team_opponent/2,
    report_blue_team_result/2,
    report_red_team_fitness/3,
    add_to_red_team/2,
    immigrate_to_blue_team/2,
    get_stats/1,
    set_config/2,
    stop/1
]).

%% gen_server callbacks
-export([
    init/1,
    handle_call/3,
    handle_cast/2,
    handle_info/2,
    terminate/2
]).

-include_lib("kernel/include/logger.hrl").

%% Configuration
-record(config, {
    red_team_size :: pos_integer(),
    red_team_threshold :: float() | auto,
    min_fitness_percentile :: float(),
    immigration_rate :: float(),
    red_team_evolution_rate :: float()
}).

%% State
-record(state, {
    realm :: atom() | binary(),
    config :: #config{},
    red_team_archive_id :: red_team_archive:archive_id(),
    total_evaluations :: non_neg_integer(),
    red_team_members_added :: non_neg_integer(),
    blue_team_best_fitness :: float(),
    red_team_best_fitness :: float(),
    generation :: non_neg_integer()
}).

%% Default configuration
-define(DEFAULT_RED_TEAM_SIZE, 30).
-define(DEFAULT_MIN_FITNESS_PERCENTILE, 0.5).
-define(DEFAULT_IMMIGRATION_RATE, 0.05).
-define(DEFAULT_RED_TEAM_EVOLUTION_RATE, 0.5).

%%====================================================================
%% API
%%====================================================================

%% @doc Start the coevolution manager for a realm.
-spec start_link(Realm :: atom() | binary()) -> {ok, pid()} | {error, term()}.
start_link(Realm) ->
    start_link(Realm, #{}).

%% @doc Start with configuration options.
%% Options:
%%   red_team_size - Maximum members in Red Team (default: 30)
%%   red_team_threshold - Fitness threshold for Red Team entry (default: auto)
%%   min_fitness_percentile - Minimum percentile to enter Red Team (default: 0.5)
%%   immigration_rate - Fraction of individuals that immigrate per generation (default: 0.05)
%%   red_team_evolution_rate - How often Red Team evolves vs Blue Team (default: 0.5)
-spec start_link(Realm :: atom() | binary(), Options :: map()) -> {ok, pid()} | {error, term()}.
start_link(Realm, Options) ->
    gen_server:start_link(?MODULE, {Realm, Options}, []).

%% @doc Get a Red Team opponent for evaluating a Blue Team member.
%%
%% Returns a network from Red Team to compete against.
%% - If Red Team has members: sample from Red Team (fitness-weighted)
%% - If Red Team empty: sample from BatchNetworks (intra-batch pairing)
%%
%% BatchNetworks is a list of networks from the current Blue Team evaluation batch.
-spec get_red_team_opponent(pid(), [map()]) -> {ok, map()}.
get_red_team_opponent(Pid, BatchNetworks) ->
    gen_server:call(Pid, {get_red_team_opponent, BatchNetworks}).

%% @doc Report Blue Team evaluation result.
%% Called after each Blue Team evaluation to potentially update Red Team.
%% Result map should contain:
%%   individual - The evaluated Blue Team member (map with 'network' key)
%%   fitness - The fitness score achieved against Red Team
-spec report_blue_team_result(pid(), map()) -> ok.
report_blue_team_result(Pid, Result) ->
    gen_server:cast(Pid, {report_blue_team_result, Result}).

%% @doc Report fitness for a Red Team member.
%% Called when a Red Team opponent has finished competing against Blue Team.
%% RedTeamId identifies the Red Team member, Fitness is their performance.
-spec report_red_team_fitness(pid(), term(), float()) -> ok.
report_red_team_fitness(Pid, RedTeamId, Fitness) ->
    gen_server:cast(Pid, {report_red_team_fitness, RedTeamId, Fitness}).

%% @doc Add a champion directly to the Red Team.
%% Used when importing champions or promoting Blue Team members.
-spec add_to_red_team(pid(), map()) -> ok | rejected.
add_to_red_team(Pid, Champion) ->
    gen_server:call(Pid, {add_to_red_team, Champion}).

%% @doc Get immigrants from Red Team to join Blue Team.
%% Returns a list of networks to be injected into Blue Team population.
%% Count specifies how many immigrants to return.
-spec immigrate_to_blue_team(pid(), pos_integer()) -> {ok, [map()]}.
immigrate_to_blue_team(Pid, Count) ->
    gen_server:call(Pid, {immigrate_to_blue_team, Count}).

%% @doc Get current statistics.
-spec get_stats(pid()) -> map().
get_stats(Pid) ->
    gen_server:call(Pid, get_stats).

%% @doc Update configuration at runtime.
-spec set_config(pid(), map()) -> ok.
set_config(Pid, ConfigUpdates) ->
    gen_server:call(Pid, {set_config, ConfigUpdates}).

%% @doc Stop the manager.
-spec stop(pid()) -> ok.
stop(Pid) ->
    gen_server:stop(Pid).

%%====================================================================
%% gen_server callbacks
%%====================================================================

init({Realm, Options}) ->
    %% Parse configuration
    Config = #config{
        red_team_size = maps:get(red_team_size, Options, ?DEFAULT_RED_TEAM_SIZE),
        red_team_threshold = maps:get(red_team_threshold, Options, auto),
        min_fitness_percentile = maps:get(min_fitness_percentile, Options, ?DEFAULT_MIN_FITNESS_PERCENTILE),
        immigration_rate = maps:get(immigration_rate, Options, ?DEFAULT_IMMIGRATION_RATE),
        red_team_evolution_rate = maps:get(red_team_evolution_rate, Options, ?DEFAULT_RED_TEAM_EVOLUTION_RATE)
    },

    %% Start Red Team archive
    ArchiveId = {red_team, Realm},
    ArchiveConfig = #{
        max_size => Config#config.red_team_size,
        min_fitness_percentile => Config#config.min_fitness_percentile
    },
    {ok, _ArchivePid} = red_team_archive:start_link(ArchiveId, ArchiveConfig),

    State = #state{
        realm = Realm,
        config = Config,
        red_team_archive_id = ArchiveId,
        total_evaluations = 0,
        red_team_members_added = 0,
        blue_team_best_fitness = 0.0,
        red_team_best_fitness = 0.0,
        generation = 0
    },

    ?LOG_INFO("[coevolution_manager] Started for realm ~p (Red Team vs Blue Team mode)",
              [Realm]),

    {ok, State}.

handle_call({get_red_team_opponent, BatchNetworks}, _From, State) ->
    {Reply, NewState} = do_get_red_team_opponent(BatchNetworks, State),
    {reply, Reply, NewState};

handle_call({add_to_red_team, Champion}, _From, State) ->
    Reply = do_add_to_red_team(Champion, State),
    {reply, Reply, State};

handle_call({immigrate_to_blue_team, Count}, _From, State) ->
    Reply = do_immigrate_to_blue_team(Count, State),
    {reply, Reply, State};

handle_call(get_stats, _From, State) ->
    Stats = compute_stats(State),
    {reply, Stats, State};

handle_call({set_config, Updates}, _From, State) ->
    NewState = apply_config_updates(Updates, State),
    {reply, ok, NewState};

handle_call(_Request, _From, State) ->
    {reply, {error, unknown_request}, State}.

handle_cast({report_blue_team_result, Result}, State) ->
    NewState = do_report_blue_team_result(Result, State),
    {noreply, NewState};

handle_cast({report_red_team_fitness, RedTeamId, Fitness}, State) ->
    NewState = do_report_red_team_fitness(RedTeamId, Fitness, State),
    {noreply, NewState};

handle_cast(_Msg, State) ->
    {noreply, State}.

handle_info(_Info, State) ->
    {noreply, State}.

terminate(_Reason, #state{red_team_archive_id = ArchiveId}) ->
    %% Stop the Red Team archive
    catch red_team_archive:stop(ArchiveId),
    ok.

%%====================================================================
%% Internal functions
%%====================================================================

do_get_red_team_opponent(BatchNetworks, #state{
    red_team_archive_id = ArchiveId,
    total_evaluations = TotalEvals
} = State) ->
    %% Increment evaluation count
    NewTotalEvals = TotalEvals + 1,
    NewState = State#state{total_evaluations = NewTotalEvals},

    %% Try Red Team first, fall back to batch
    case red_team_archive:sample(ArchiveId) of
        {ok, Opponent} ->
            %% Got opponent from Red Team
            Network = maps:get(network, Opponent),
            {{ok, Network}, NewState};
        empty ->
            %% Red Team empty - use intra-batch pairing
            sample_from_batch(BatchNetworks, NewState)
    end.

%% @private Sample a random opponent from the batch.
sample_from_batch([], State) ->
    %% No batch networks provided - this shouldn't happen in normal operation
    %% Return a minimal "do nothing" network as last resort
    ?LOG_WARNING("[coevolution_manager] No batch networks and empty Red Team - returning minimal network"),
    MinimalNetwork = #{weights => [], topology => #{inputs => 0, outputs => 0}},
    {{ok, MinimalNetwork}, State};
sample_from_batch(BatchNetworks, State) ->
    %% Random selection from batch
    Index = rand:uniform(length(BatchNetworks)),
    Network = lists:nth(Index, BatchNetworks),
    {{ok, Network}, State}.

do_report_blue_team_result(Result, #state{
    config = Config,
    red_team_archive_id = ArchiveId,
    red_team_members_added = RedTeamAdded,
    blue_team_best_fitness = BlueBest
} = State) ->
    %% Extract result data
    Individual = maps:get(individual, Result, #{}),
    Fitness = maps:get(fitness, Result, 0.0),
    Network = maps:get(network, Individual, maps:get(network, Result, undefined)),
    Generation = maps:get(generation, Result, 0),

    %% Determine if Blue Team member should join Red Team (immigration)
    Threshold = get_red_team_threshold(Config, ArchiveId),
    ShouldAdd = (Network =/= undefined) andalso (Fitness >= Threshold),

    %% Update Blue Team best fitness tracking
    NewBlueBest = max(BlueBest, Fitness),

    case ShouldAdd of
        true ->
            Champion = #{
                network => Network,
                fitness => Fitness,
                generation => Generation,
                origin => blue_team
            },
            case red_team_archive:add(ArchiveId, Champion) of
                ok ->
                    ?LOG_DEBUG("[coevolution_manager] Blue Team champion (fitness ~p) promoted to Red Team", [Fitness]),
                    State#state{
                        red_team_members_added = RedTeamAdded + 1,
                        blue_team_best_fitness = NewBlueBest,
                        generation = Generation
                    };
                rejected ->
                    State#state{blue_team_best_fitness = NewBlueBest, generation = Generation}
            end;
        false ->
            State#state{blue_team_best_fitness = NewBlueBest, generation = Generation}
    end.

do_report_red_team_fitness(RedTeamId, Fitness, #state{
    red_team_archive_id = ArchiveId,
    red_team_best_fitness = RedBest
} = State) ->
    %% Update Red Team member's fitness
    red_team_archive:update_fitness(ArchiveId, RedTeamId, Fitness),
    NewRedBest = max(RedBest, Fitness),
    State#state{red_team_best_fitness = NewRedBest}.

get_red_team_threshold(#config{red_team_threshold = auto}, ArchiveId) ->
    %% Auto threshold: use 50% of average fitness
    case red_team_archive:stats(ArchiveId) of
        #{avg_fitness := Avg} when Avg > 0 -> Avg * 0.5;
        _ -> 0.0  % Accept anything when Red Team empty
    end;
get_red_team_threshold(#config{red_team_threshold = Fixed}, _ArchiveId) ->
    Fixed.

do_add_to_red_team(Champion, #state{red_team_archive_id = ArchiveId}) ->
    red_team_archive:add(ArchiveId, Champion).

do_immigrate_to_blue_team(Count, #state{red_team_archive_id = ArchiveId}) ->
    %% Get top performers from Red Team to immigrate to Blue Team
    case red_team_archive:get_top(ArchiveId, Count) of
        {ok, Immigrants} ->
            Networks = [maps:get(network, I) || I <- Immigrants],
            {ok, Networks};
        empty ->
            {ok, []}
    end.

compute_stats(#state{
    red_team_archive_id = ArchiveId,
    total_evaluations = TotalEvals,
    red_team_members_added = RedTeamAdded,
    blue_team_best_fitness = BlueBest,
    red_team_best_fitness = RedBest,
    generation = Generation,
    config = Config
}) ->
    ArchiveStats = red_team_archive:stats(ArchiveId),

    #{
        total_evaluations => TotalEvals,
        generation => Generation,
        red_team_members_added => RedTeamAdded,
        blue_team_best_fitness => BlueBest,
        red_team_best_fitness => RedBest,
        red_team_size => maps:get(count, ArchiveStats, 0),
        red_team_max_size => Config#config.red_team_size,
        red_team_avg_fitness => maps:get(avg_fitness, ArchiveStats, 0.0),
        red_team_max_fitness => maps:get(max_fitness, ArchiveStats, 0.0),
        immigration_rate => Config#config.immigration_rate,
        mode => competitive_coevolution
    }.

apply_config_updates(Updates, #state{config = Config} = State) ->
    NewConfig = Config#config{
        red_team_size = maps:get(red_team_size, Updates, Config#config.red_team_size),
        red_team_threshold = maps:get(red_team_threshold, Updates, Config#config.red_team_threshold),
        immigration_rate = maps:get(immigration_rate, Updates, Config#config.immigration_rate)
    },
    State#state{config = NewConfig}.
