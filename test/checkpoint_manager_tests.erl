%%%-------------------------------------------------------------------
%%% @doc Unit tests for checkpoint_manager.
%%%-------------------------------------------------------------------
-module(checkpoint_manager_tests).

-include_lib("eunit/include/eunit.hrl").
-include("neuroevolution.hrl").

%% Create a test individual
make_test_individual() ->
    make_test_individual(42, 0.85).

make_test_individual(Id, Fitness) ->
    #individual{
        id = Id,
        network = #{layers => [], activation => tanh},  %% Simplified test network
        genome = undefined,
        fitness = Fitness,
        generation_born = 1,
        metrics = #{}
    }.

%% Helper to get a unique test directory
get_test_dir() ->
    "/tmp/checkpoint_test_" ++ integer_to_list(erlang:unique_integer([positive])).

%% Helper to clean up a test directory
cleanup_test_dir(TestDir) ->
    Pattern = filename:join(TestDir, "*"),
    Files = filelib:wildcard(Pattern),
    [file:delete(F) || F <- Files],
    file:del_dir(TestDir).

%%==============================================================================
%% Test Cases
%%==============================================================================

%% Test initialization
init_test() ->
    TestDir = get_test_dir(),
    try
        ok = checkpoint_manager:init(#{checkpoint_dir => TestDir}),
        ?assertEqual(TestDir, checkpoint_manager:get_checkpoint_dir())
    after
        cleanup_test_dir(TestDir)
    end.

%% Test save and load basic checkpoint
save_load_test() ->
    TestDir = get_test_dir(),
    Opts = #{checkpoint_dir => TestDir},
    try
        ok = checkpoint_manager:init(Opts),

        Individual = make_test_individual(),
        Metadata = #{
            reason => fitness_record,
            fitness => 0.85,
            generation => 5,
            total_evaluations => 500
        },

        %% Save checkpoint
        ?assertEqual(ok, checkpoint_manager:save_checkpoint(Individual, Metadata, Opts)),

        %% Load latest should find it
        {ok, LoadedIndividual, LoadedMeta} = checkpoint_manager:load_latest(Opts),
        ?assertEqual(42, LoadedIndividual#individual.id),
        ?assertEqual(fitness_record, maps:get(reason, LoadedMeta)),

        %% Load best fitness should find it (only one checkpoint)
        {ok, BestIndividual, _} = checkpoint_manager:load_best_fitness(Opts),
        ?assertEqual(42, BestIndividual#individual.id)
    after
        cleanup_test_dir(TestDir)
    end.

%% Test multiple checkpoints with different fitness values
multiple_checkpoints_test() ->
    TestDir = get_test_dir(),
    Opts = #{checkpoint_dir => TestDir},
    try
        ok = checkpoint_manager:init(Opts),

        %% Save multiple checkpoints with different fitness
        Individual1 = make_test_individual(1, 0.50),
        Individual2 = make_test_individual(2, 0.90),  %% Best fitness
        Individual3 = make_test_individual(3, 0.75),

        Metadata1 = #{reason => generation_interval, generation => 10, total_evaluations => 100},
        Metadata2 = #{reason => fitness_record, generation => 20, total_evaluations => 200},
        Metadata3 = #{reason => generation_interval, generation => 30, total_evaluations => 300},

        ok = checkpoint_manager:save_checkpoint(Individual1, Metadata1, Opts),
        timer:sleep(10),
        ok = checkpoint_manager:save_checkpoint(Individual2, Metadata2, Opts),
        timer:sleep(10),
        ok = checkpoint_manager:save_checkpoint(Individual3, Metadata3, Opts),

        %% List should return 3 checkpoints
        Checkpoints = checkpoint_manager:list_checkpoints(Opts),
        ?assertEqual(3, length(Checkpoints)),

        %% Load latest should return Individual3 (most recent by timestamp)
        {ok, LatestIndividual, _} = checkpoint_manager:load_latest(Opts),
        ?assertEqual(3, LatestIndividual#individual.id),

        %% Load best fitness should return Individual2 (highest fitness)
        {ok, BestIndividual, _} = checkpoint_manager:load_best_fitness(Opts),
        ?assertEqual(2, BestIndividual#individual.id)
    after
        cleanup_test_dir(TestDir)
    end.

%% Test checkpoint reasons
checkpoint_reason_test() ->
    TestDir = get_test_dir(),
    Opts = #{checkpoint_dir => TestDir},
    try
        ok = checkpoint_manager:init(Opts),

        Individual = make_test_individual(),
        Reasons = [fitness_record, generation_interval, evaluation_interval, manual, training_complete],

        lists:foreach(fun(Reason) ->
            timer:sleep(5),
            Metadata = #{reason => Reason, generation => 1, total_evaluations => 100},
            ?assertEqual(ok, checkpoint_manager:save_checkpoint(Individual, Metadata, Opts))
        end, Reasons),

        %% Should have one checkpoint per reason
        Checkpoints = checkpoint_manager:list_checkpoints(Opts),
        ?assertEqual(length(Reasons), length(Checkpoints))
    after
        cleanup_test_dir(TestDir)
    end.

%% Test filename generation and parsing
filename_test() ->
    Metadata = #{
        reason => fitness_record,
        generation => 42,
        fitness => 0.8765,
        timestamp => 1703587200000
    },
    Filename = checkpoint_manager:checkpoint_filename(Metadata, "/tmp"),
    {ok, Parsed} = checkpoint_manager:parse_checkpoint_filename(Filename),

    ?assertEqual(fitness_record, maps:get(reason, Parsed)),
    ?assertEqual(42, maps:get(generation, Parsed)).

%% Test load with no checkpoints
empty_load_test() ->
    TestDir = get_test_dir(),
    Opts = #{checkpoint_dir => TestDir},
    try
        ok = checkpoint_manager:init(Opts),

        ?assertEqual({error, no_checkpoints}, checkpoint_manager:load_latest(Opts)),
        ?assertEqual({error, no_checkpoints}, checkpoint_manager:load_best_fitness(Opts))
    after
        cleanup_test_dir(TestDir)
    end.

%% Test set_checkpoint_dir
set_dir_test() ->
    TestDir1 = get_test_dir(),
    TestDir2 = get_test_dir(),
    try
        ok = checkpoint_manager:init(#{checkpoint_dir => TestDir1}),
        ?assertEqual(TestDir1, checkpoint_manager:get_checkpoint_dir()),

        ok = checkpoint_manager:set_checkpoint_dir(TestDir2),
        ?assertEqual(TestDir2, checkpoint_manager:get_checkpoint_dir())
    after
        cleanup_test_dir(TestDir1),
        cleanup_test_dir(TestDir2)
    end.

%% Test delete checkpoint
delete_test() ->
    TestDir = get_test_dir(),
    Opts = #{checkpoint_dir => TestDir},
    try
        ok = checkpoint_manager:init(Opts),

        Individual = make_test_individual(),
        Metadata = #{reason => manual, generation => 1, total_evaluations => 100},

        ok = checkpoint_manager:save_checkpoint(Individual, Metadata, Opts),
        ?assertEqual(1, length(checkpoint_manager:list_checkpoints(Opts))),

        [#{filename := Filename}] = checkpoint_manager:list_checkpoints(Opts),
        ok = checkpoint_manager:delete_checkpoint(Filename),

        ?assertEqual(0, length(checkpoint_manager:list_checkpoints(Opts)))
    after
        cleanup_test_dir(TestDir)
    end.

%% Test pruning
prune_test() ->
    TestDir = get_test_dir(),
    Opts = #{checkpoint_dir => TestDir},
    try
        ok = checkpoint_manager:init(Opts),

        %% Create 25 checkpoints
        lists:foreach(fun(I) ->
            Individual = make_test_individual(I, I / 100),
            Metadata = #{reason => generation_interval, generation => I, total_evaluations => I * 10},
            timer:sleep(5),  %% Ensure different timestamps
            checkpoint_manager:save_checkpoint(Individual, Metadata, Opts)
        end, lists:seq(1, 25)),

        %% Should have 25 checkpoints
        ?assertEqual(25, length(checkpoint_manager:list_checkpoints(Opts))),

        %% Prune to keep only 10
        {ok, Deleted} = checkpoint_manager:prune_checkpoints(Opts#{max_per_reason => 10}),
        ?assert(Deleted > 0),

        %% Should have at most 11 checkpoints now (10 + best preserved)
        Remaining = length(checkpoint_manager:list_checkpoints(Opts)),
        ?assert(Remaining =< 11)
    after
        cleanup_test_dir(TestDir)
    end.
