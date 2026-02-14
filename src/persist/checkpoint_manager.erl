%%%-------------------------------------------------------------------
%%% @doc Network Checkpoint Manager.
%%%
%%% This module provides network checkpointing capabilities for
%%% saving and loading evolved networks at key milestones during training.
%%%
%%% == Checkpoint Triggers ==
%%%
%%% Networks are saved at configurable milestones:
%%%   fitness_record - When a new best fitness is achieved
%%%   generation_interval - Every N generations
%%%   evaluation_interval - Every N evaluations
%%%
%%% == Storage Format ==
%%%
%%% Each checkpoint is stored as an Erlang term file containing:
%%%   Network binary (from network_evaluator:to_binary/1)
%%%   Metadata (fitness, generation, timestamp, etc.)
%%%   Configuration used during training
%%%
%%% @author R.G. Lefever
%%% @copyright 2024-2026 R.G. Lefever
%%% @end
%%%-------------------------------------------------------------------
-module(checkpoint_manager).

-include("neuroevolution.hrl").

%% API
-export([
    %% Configuration
    init/1,
    get_checkpoint_dir/0,
    set_checkpoint_dir/1,

    %% Saving
    save_checkpoint/2,
    save_checkpoint/3,

    %% Loading
    load_checkpoint/1,
    load_latest/0,
    load_latest/1,
    load_best_fitness/0,
    load_best_fitness/1,

    %% Management
    list_checkpoints/0,
    list_checkpoints/1,
    delete_checkpoint/1,
    prune_checkpoints/1,

    %% Utilities
    checkpoint_filename/2,
    parse_checkpoint_filename/1
]).

%% Default checkpoint directory (relative to CWD)
-define(DEFAULT_CHECKPOINT_DIR, "_checkpoints").

%% File extension for checkpoints
-define(CHECKPOINT_EXT, ".checkpoint").

%% Maximum checkpoints to keep (per reason) before pruning old ones
-define(DEFAULT_MAX_CHECKPOINTS, 20).

%%==============================================================================
%% Types
%%==============================================================================

-type checkpoint_reason() ::
    fitness_record |          %% New best fitness achieved
    generation_interval |     %% Every N generations
    evaluation_interval |     %% Every N evaluations
    manual |                  %% User-triggered save
    pre_mutation |            %% Before topology mutation
    training_complete.        %% End of training

-type checkpoint_metadata() :: #{
    reason := checkpoint_reason(),
    fitness := float(),
    generation := non_neg_integer(),
    total_evaluations := non_neg_integer(),
    individual_id := term(),
    timestamp := non_neg_integer(),
    config => map()
}.

-type checkpoint() :: #{
    individual := individual(),
    metadata := checkpoint_metadata()
}.

-export_type([checkpoint_reason/0, checkpoint_metadata/0, checkpoint/0]).

%%==============================================================================
%% Configuration API
%%==============================================================================

%% @doc Initialize the checkpoint manager with a configuration.
%% Creates the checkpoint directory if it doesn't exist.
-spec init(map()) -> ok | {error, term()}.
init(Config) ->
    Dir = maps:get(checkpoint_dir, Config, ?DEFAULT_CHECKPOINT_DIR),
    case filelib:ensure_dir(filename:join(Dir, "dummy")) of
        ok ->
            persistent_term:put({?MODULE, checkpoint_dir}, Dir),
            ok;
        {error, Reason} ->
            {error, {create_dir_failed, Reason}}
    end.

%% @doc Get the current checkpoint directory.
-spec get_checkpoint_dir() -> file:filename().
get_checkpoint_dir() ->
    try
        persistent_term:get({?MODULE, checkpoint_dir})
    catch
        error:badarg ->
            ?DEFAULT_CHECKPOINT_DIR
    end.

%% @doc Set the checkpoint directory.
-spec set_checkpoint_dir(file:filename()) -> ok | {error, term()}.
set_checkpoint_dir(Dir) ->
    case filelib:ensure_dir(filename:join(Dir, "dummy")) of
        ok ->
            persistent_term:put({?MODULE, checkpoint_dir}, Dir),
            ok;
        {error, Reason} ->
            {error, {create_dir_failed, Reason}}
    end.

%%==============================================================================
%% Saving API
%%==============================================================================

%% @doc Save a checkpoint with the given individual and metadata.
-spec save_checkpoint(individual(), checkpoint_metadata()) -> ok | {error, term()}.
save_checkpoint(Individual, Metadata) ->
    save_checkpoint(Individual, Metadata, #{}).

%% @doc Save a checkpoint with additional options.
%% Options:
%%   checkpoint_dir - Override the default directory
%%   compress       - true (default) or false
-spec save_checkpoint(individual(), checkpoint_metadata(), map()) -> ok | {error, term()}.
save_checkpoint(Individual, Metadata, Options) ->
    Dir = maps:get(checkpoint_dir, Options, get_checkpoint_dir()),
    Compress = maps:get(compress, Options, true),

    %% Ensure metadata has required fields
    CompleteMetadata = complete_metadata(Metadata, Individual),

    %% Generate filename from metadata
    Filename = checkpoint_filename(CompleteMetadata, Dir),

    %% Build checkpoint data
    Checkpoint = #{
        version => 1,
        individual => Individual,
        metadata => CompleteMetadata
    },

    %% Serialize with optional compression
    Binary = case Compress of
        true -> term_to_binary(Checkpoint, [compressed]);
        false -> term_to_binary(Checkpoint)
    end,

    %% Write to file
    case file:write_file(Filename, Binary) of
        ok ->
            error_logger:info_msg("[checkpoint_manager] Saved checkpoint: ~s (~.2f KB)~n",
                                  [filename:basename(Filename), byte_size(Binary) / 1024]),
            ok;
        {error, Reason} ->
            error_logger:error_msg("[checkpoint_manager] Failed to save checkpoint: ~p~n", [Reason]),
            {error, {write_failed, Reason}}
    end.

%% @private Complete metadata with defaults and derived values.
-spec complete_metadata(checkpoint_metadata(), individual()) -> checkpoint_metadata().
complete_metadata(Metadata, Individual) ->
    Now = erlang:system_time(millisecond),
    Defaults = #{
        reason => manual,
        fitness => Individual#individual.fitness,
        generation => Individual#individual.generation_born,
        total_evaluations => 0,
        individual_id => Individual#individual.id,
        timestamp => Now
    },
    maps:merge(Defaults, Metadata).

%%==============================================================================
%% Loading API
%%==============================================================================

%% @doc Load a checkpoint from a file.
-spec load_checkpoint(file:filename()) -> {ok, individual(), checkpoint_metadata()} | {error, term()}.
load_checkpoint(Filename) ->
    case file:read_file(Filename) of
        {ok, Binary} ->
            try
                #{version := 1,
                  individual := Individual,
                  metadata := Metadata} = binary_to_term(Binary),
                {ok, Individual, Metadata}
            catch
                _:Reason ->
                    {error, {parse_failed, Reason}}
            end;
        {error, Reason} ->
            {error, {read_failed, Reason}}
    end.

%% @doc Load the most recent checkpoint.
-spec load_latest() -> {ok, individual(), checkpoint_metadata()} | {error, no_checkpoints | term()}.
load_latest() ->
    load_latest(#{}).

%% @doc Load the most recent checkpoint from a specific directory.
-spec load_latest(map()) -> {ok, individual(), checkpoint_metadata()} | {error, no_checkpoints | term()}.
load_latest(Options) ->
    case list_checkpoints(Options) of
        [] ->
            {error, no_checkpoints};
        Checkpoints ->
            %% Sort by timestamp (newest first) and take the first
            Sorted = lists:sort(fun(A, B) ->
                maps:get(timestamp, A, 0) > maps:get(timestamp, B, 0)
            end, Checkpoints),
            #{filename := Filename} = hd(Sorted),
            load_checkpoint(Filename)
    end.

%% @doc Load the checkpoint with the best fitness.
-spec load_best_fitness() -> {ok, individual(), checkpoint_metadata()} | {error, no_checkpoints | term()}.
load_best_fitness() ->
    load_best_fitness(#{}).

%% @doc Load the checkpoint with the best fitness from a specific directory.
-spec load_best_fitness(map()) -> {ok, individual(), checkpoint_metadata()} | {error, no_checkpoints | term()}.
load_best_fitness(Options) ->
    case list_checkpoints(Options) of
        [] ->
            {error, no_checkpoints};
        Checkpoints ->
            %% Sort by fitness (highest first) and take the first
            Sorted = lists:sort(fun(A, B) ->
                maps:get(fitness, A, 0) > maps:get(fitness, B, 0)
            end, Checkpoints),
            #{filename := Filename} = hd(Sorted),
            load_checkpoint(Filename)
    end.

%%==============================================================================
%% Management API
%%==============================================================================

%% @doc List all checkpoints in the default directory.
-spec list_checkpoints() -> [map()].
list_checkpoints() ->
    list_checkpoints(#{}).

%% @doc List all checkpoints in the specified directory.
%% Returns a list of maps with metadata and filename for each checkpoint.
-spec list_checkpoints(map()) -> [map()].
list_checkpoints(Options) ->
    Dir = maps:get(checkpoint_dir, Options, get_checkpoint_dir()),
    Pattern = filename:join(Dir, "*" ++ ?CHECKPOINT_EXT),
    Files = filelib:wildcard(Pattern),

    lists:filtermap(fun(Filename) ->
        case parse_checkpoint_filename(Filename) of
            {ok, Info} ->
                {true, Info#{filename => Filename}};
            {error, _} ->
                false
        end
    end, Files).

%% @doc Delete a specific checkpoint.
-spec delete_checkpoint(file:filename()) -> ok | {error, term()}.
delete_checkpoint(Filename) ->
    file:delete(Filename).

%% @doc Prune old checkpoints, keeping only the most recent N per reason.
%% Options:
%%   max_per_reason - Maximum checkpoints to keep per reason (default: 20)
%%   keep_best      - Always keep the best fitness checkpoint (default: true)
-spec prune_checkpoints(map()) -> {ok, non_neg_integer()} | {error, term()}.
prune_checkpoints(Options) ->
    MaxPerReason = maps:get(max_per_reason, Options, ?DEFAULT_MAX_CHECKPOINTS),
    KeepBest = maps:get(keep_best, Options, true),

    Checkpoints = list_checkpoints(Options),

    %% Group by reason
    ByReason = lists:foldl(fun(CP, Acc) ->
        Reason = maps:get(reason, CP, manual),
        Existing = maps:get(Reason, Acc, []),
        maps:put(Reason, [CP | Existing], Acc)
    end, #{}, Checkpoints),

    %% Find best fitness checkpoint to preserve
    BestCheckpoint = case KeepBest andalso length(Checkpoints) > 0 of
        true ->
            SortedByFitness = lists:sort(fun(A, B) ->
                maps:get(fitness, A, 0) > maps:get(fitness, B, 0)
            end, Checkpoints),
            hd(SortedByFitness);
        false ->
            undefined
    end,

    %% Prune each reason group
    Deleted = maps:fold(fun(_Reason, CPs, AccDeleted) ->
        %% Sort by timestamp (newest first)
        SortedByTime = lists:sort(fun(A, B) ->
            maps:get(timestamp, A, 0) > maps:get(timestamp, B, 0)
        end, CPs),

        %% Keep first MaxPerReason, delete the rest
        {_Keep, ToDelete} = lists:split(min(MaxPerReason, length(SortedByTime)), SortedByTime),

        lists:foldl(fun(CP, Acc) ->
            %% Don't delete the best fitness checkpoint
            case BestCheckpoint =/= undefined andalso
                 maps:get(filename, CP) =:= maps:get(filename, BestCheckpoint) of
                true ->
                    Acc;
                false ->
                    case delete_checkpoint(maps:get(filename, CP)) of
                        ok -> Acc + 1;
                        _ -> Acc
                    end
            end
        end, AccDeleted, ToDelete)
    end, 0, ByReason),

    {ok, Deleted}.

%%==============================================================================
%% Utility Functions
%%==============================================================================

%% @doc Generate a checkpoint filename from metadata.
%% Format: REASON-genN-fitF-TIMESTAMP.checkpoint
%% Uses dashes as separators since reason names may contain underscores.
-spec checkpoint_filename(checkpoint_metadata(), file:filename()) -> file:filename().
checkpoint_filename(Metadata, Dir) ->
    Reason = maps:get(reason, Metadata, manual),
    Gen = maps:get(generation, Metadata, 0),
    Fitness = maps:get(fitness, Metadata, 0.0),
    Timestamp = maps:get(timestamp, Metadata, erlang:system_time(millisecond)),

    %% Format fitness as string with 4 decimal places
    FitnessStr = io_lib:format("~.4f", [Fitness]),
    %% Remove any decimal points or negative signs for filename safety
    FitnessSafe = lists:flatten([C || C <- FitnessStr, C =/= $.]),

    Basename = io_lib:format("~s-gen~B-fit~s-~B~s",
                             [Reason, Gen, FitnessSafe, Timestamp, ?CHECKPOINT_EXT]),
    filename:join(Dir, lists:flatten(Basename)).

%% @doc Parse checkpoint information from a filename.
%% Extracts reason, generation, fitness, and timestamp from the filename.
%% Format: REASON-genN-fitF-TIMESTAMP.checkpoint
-spec parse_checkpoint_filename(file:filename()) -> {ok, map()} | {error, invalid_format}.
parse_checkpoint_filename(Filename) ->
    Basename = filename:basename(Filename, ?CHECKPOINT_EXT),
    case string:split(Basename, "-", all) of
        [ReasonStr, GenStr, FitStr, TimestampStr | _] ->
            try
                Reason = list_to_existing_atom(ReasonStr),
                Gen = parse_gen(GenStr),
                Fitness = parse_fitness(FitStr),
                Timestamp = list_to_integer(TimestampStr),
                {ok, #{
                    reason => Reason,
                    generation => Gen,
                    fitness => Fitness,
                    timestamp => Timestamp
                }}
            catch
                _:_ -> {error, invalid_format}
            end;
        _ ->
            {error, invalid_format}
    end.

%% @private Parse generation from "gen123" format.
parse_gen("gen" ++ Rest) ->
    list_to_integer(Rest);
parse_gen(Str) ->
    list_to_integer(Str).

%% @private Parse fitness from "fit12345" format (4 decimal places encoded).
parse_fitness("fit" ++ Rest) ->
    parse_fitness_value(Rest);
parse_fitness(Str) ->
    parse_fitness_value(Str).

parse_fitness_value(Str) ->
    %% Handle negative numbers (encoded as 'n' prefix)
    {Sign, NumStr} = case Str of
        [$n | Rest] -> {-1, Rest};
        _ -> {1, Str}
    end,
    %% Last 4 digits are decimal places
    Len = length(NumStr),
    case Len > 4 of
        true ->
            IntPart = list_to_integer(lists:sublist(NumStr, Len - 4)),
            DecPart = list_to_integer(lists:sublist(NumStr, Len - 3, 4)),
            Sign * (IntPart + DecPart / 10000);
        false ->
            %% All decimal
            Sign * list_to_integer(NumStr) / 10000
    end.
