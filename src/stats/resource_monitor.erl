%% @doc System resource monitoring for adaptive neuroevolution.
%%
%% This module provides system resource metrics that can be used as inputs
%% to the LTC meta-controller, allowing it to adapt evolution hyperparameters
%% based on current memory and CPU pressure.
%%
%% == Metrics Provided ==
%%
%% Raw metrics:
%% - memory_total: Total VM memory usage (bytes)
%% - memory_processes: Memory used by processes (bytes)
%% - memory_binary: Binary memory (bytes) - often the culprit in leaks
%% - scheduler_utilization: Average CPU utilization (0.0 - 1.0)
%% - process_count: Number of processes in the VM
%% - message_queue_len: Total message queue length across monitored processes
%%
%% Normalized metrics (for LTC input):
%% - memory_pressure: 0.0 (plenty of memory) to 1.0 (critical)
%% - cpu_pressure: 0.0 (idle) to 1.0 (saturated)
%% - process_pressure: 0.0 (few processes) to 1.0 (at limit)
%%
%% == Usage ==
%%
%% %% Get raw metrics
%% #{memory_total := Mem} = resource_monitor:get_metrics(),
%%
%% %% Get normalized metrics for LTC input
%% #{memory_pressure := MemP, cpu_pressure := CpuP} = resource_monitor:get_normalized_metrics(),
%% LtcInputs = [MemP, CpuP, ...]
%%
%% @author R.G. Lefever
%% @copyright 2024-2026 R.G. Lefever
-module(resource_monitor).

-export([
    get_metrics/0,
    get_normalized_metrics/0,
    get_memory_limit/0,
    is_memory_critical/0,
    is_memory_high/0,
    check_health/0
]).

%%% ============================================================================
%%% API Functions
%%% ============================================================================

%% @doc Get current system resource metrics (raw values).
%%
%% Returns a map with various resource measurements.
-spec get_metrics() -> map().
get_metrics() ->
    Memory = erlang:memory(),
    #{
        memory_total => proplists:get_value(total, Memory, 0),
        memory_processes => proplists:get_value(processes, Memory, 0),
        memory_processes_used => proplists:get_value(processes_used, Memory, 0),
        memory_binary => proplists:get_value(binary, Memory, 0),
        memory_ets => proplists:get_value(ets, Memory, 0),
        memory_atom => proplists:get_value(atom, Memory, 0),
        scheduler_utilization => get_scheduler_utilization(),
        process_count => erlang:system_info(process_count),
        process_limit => erlang:system_info(process_limit),
        message_queue_len => get_sampled_message_queue_len(),
        run_queue => erlang:statistics(run_queue)
    }.

%% @doc Get normalized metrics for LTC input (0.0 to 1.0 range).
%%
%% These are suitable for direct use as neural network inputs.
-spec get_normalized_metrics() -> map().
get_normalized_metrics() ->
    Metrics = get_metrics(),
    MaxMemory = get_memory_limit(),
    ProcessLimit = maps:get(process_limit, Metrics, 262144),

    %% Use cgroup memory if available (containers), otherwise BEAM memory
    MemoryUsed = case get_cgroup_memory_usage() of
        {ok, CgroupMem} -> CgroupMem;
        error -> maps:get(memory_total, Metrics, 0)
    end,
    SchedulerUtil = maps:get(scheduler_utilization, Metrics, 0.0),
    ProcessCount = maps:get(process_count, Metrics, 0),
    RunQueue = maps:get(run_queue, Metrics, 0),
    MessageQueueLen = maps:get(message_queue_len, Metrics, 0),

    %% Memory pressure: ratio of used to available memory
    MemoryPressure = min(1.0, MemoryUsed / MaxMemory),

    %% CPU pressure: combination of scheduler utilization and run queue
    %% Run queue > 0 indicates backed-up work
    RunQueuePressure = min(1.0, RunQueue / max(1, erlang:system_info(schedulers))),
    CpuPressure = (SchedulerUtil + RunQueuePressure) / 2.0,

    %% Process pressure: ratio of active to max processes
    ProcessPressure = min(1.0, ProcessCount / ProcessLimit),

    %% Message queue pressure: high queue lengths indicate backpressure
    %% Normalize assuming 10000 total queued messages is "full"
    MessageQueuePressure = min(1.0, MessageQueueLen / 10000.0),

    #{
        memory_pressure => MemoryPressure,
        cpu_pressure => CpuPressure,
        process_pressure => ProcessPressure,
        message_queue_pressure => MessageQueuePressure,
        %% Composite pressure metric (for simple decisions)
        overall_pressure => (MemoryPressure + CpuPressure + ProcessPressure) / 3.0
    }.

%% @doc Get the memory limit for the system.
%%
%% Attempts to detect from:
%% 1. MACULA_MEMORY_LIMIT environment variable (bytes)
%% 2. Container cgroup limits
%% 3. System total memory
%% 4. Default fallback (8GB)
-spec get_memory_limit() -> pos_integer().
get_memory_limit() ->
    case os:getenv("MACULA_MEMORY_LIMIT") of
        false -> detect_memory_limit();
        EnvValue ->
            try list_to_integer(EnvValue)
            catch _:_ -> detect_memory_limit()
            end
    end.

%% @doc Check if memory usage is at critical level (>90%).
-spec is_memory_critical() -> boolean().
is_memory_critical() ->
    #{memory_pressure := Pressure} = get_normalized_metrics(),
    Pressure > 0.9.

%% @doc Check if memory usage is at high level (>70%).
-spec is_memory_high() -> boolean().
is_memory_high() ->
    #{memory_pressure := Pressure} = get_normalized_metrics(),
    Pressure > 0.7.

%% @doc Perform health check and return status.
%%
%% Returns a map with health status and any warnings/alerts.
-spec check_health() -> map().
check_health() ->
    Metrics = get_normalized_metrics(),
    MemoryPressure = maps:get(memory_pressure, Metrics),
    CpuPressure = maps:get(cpu_pressure, Metrics),
    MsgPressure = maps:get(message_queue_pressure, Metrics),

    Status = if
        MemoryPressure > 0.9 -> critical;
        MemoryPressure > 0.7 orelse CpuPressure > 0.9 -> warning;
        MsgPressure > 0.8 -> degraded;
        true -> healthy
    end,

    Warnings = lists:filtermap(
        fun({_Metric, Value, Threshold, Msg}) ->
            case Value > Threshold of
                true -> {true, Msg};
                false -> false
            end
        end,
        [
            {memory, MemoryPressure, 0.7, <<"Memory usage high">>},
            {memory, MemoryPressure, 0.9, <<"Memory usage critical">>},
            {cpu, CpuPressure, 0.9, <<"CPU saturated">>},
            {queue, MsgPressure, 0.8, <<"Message queues backed up">>}
        ]
    ),

    #{
        status => Status,
        warnings => Warnings,
        metrics => Metrics
    }.

%%% ============================================================================
%%% Internal Functions
%%% ============================================================================

%% @private Get current memory usage from cgroups (for containers).
%%
%% This is more accurate than erlang:memory(total) in containers because
%% it measures actual resident memory, not virtual allocations.
get_cgroup_memory_usage() ->
    %% Try cgroup v2 first
    case file:read_file("/sys/fs/cgroup/memory.current") of
        {ok, Bin} ->
            parse_cgroup_value(Bin);
        _ ->
            %% Try cgroup v1
            case file:read_file("/sys/fs/cgroup/memory/memory.usage_in_bytes") of
                {ok, Bin} ->
                    parse_cgroup_value(Bin);
                _ ->
                    error
            end
    end.

%% @private Parse a cgroup memory value.
parse_cgroup_value(Bin) ->
    Str = string:trim(binary_to_list(Bin)),
    try
        {ok, list_to_integer(Str)}
    catch
        _:_ -> error
    end.

%% @private Detect memory limit from system or cgroups.
detect_memory_limit() ->
    %% Try cgroup v2 first (container environments)
    case file:read_file("/sys/fs/cgroup/memory.max") of
        {ok, Bin} ->
            case parse_cgroup_limit(Bin) of
                max -> detect_system_memory();
                Limit -> Limit
            end;
        _ ->
            %% Try cgroup v1
            case file:read_file("/sys/fs/cgroup/memory/memory.limit_in_bytes") of
                {ok, Bin} ->
                    case parse_cgroup_limit(Bin) of
                        max -> detect_system_memory();
                        Limit -> Limit
                    end;
                _ ->
                    detect_system_memory()
            end
    end.

%% @private Parse cgroup limit value.
parse_cgroup_limit(Bin) ->
    Str = string:trim(binary_to_list(Bin)),
    case Str of
        "max" -> max;
        _ ->
            try list_to_integer(Str)
            catch _:_ -> max
            end
    end.

%% @private Detect system memory from /proc/meminfo.
detect_system_memory() ->
    case file:read_file("/proc/meminfo") of
        {ok, Bin} ->
            Lines = string:split(binary_to_list(Bin), "\n", all),
            parse_meminfo(Lines);
        _ ->
            %% Default fallback: 8GB
            8 * 1024 * 1024 * 1024
    end.

%% @private Parse MemTotal from /proc/meminfo.
parse_meminfo([]) ->
    8 * 1024 * 1024 * 1024;  % Default fallback
parse_meminfo(["MemTotal:" ++ Rest | _]) ->
    case string:tokens(string:trim(Rest), " \t") of
        [NumStr, "kB" | _] ->
            try list_to_integer(NumStr) * 1024
            catch _:_ -> 8 * 1024 * 1024 * 1024
            end;
        _ -> 8 * 1024 * 1024 * 1024
    end;
parse_meminfo([_ | Rest]) ->
    parse_meminfo(Rest).

%% @private Get scheduler utilization.
%%
%% Uses erlang:statistics(scheduler_wall_time) if available,
%% otherwise returns a simple approximation based on run queue.
get_scheduler_utilization() ->
    try
        %% scheduler_wall_time needs to be enabled first
        case erlang:statistics(scheduler_wall_time) of
            undefined ->
                %% Not enabled, try to enable and return approximation
                erlang:system_flag(scheduler_wall_time, true),
                estimate_cpu_from_run_queue();
            Times when is_list(Times) ->
                calculate_scheduler_utilization(Times)
        end
    catch
        _:_ ->
            estimate_cpu_from_run_queue()
    end.

%% @private Calculate utilization from wall time samples.
calculate_scheduler_utilization(Times) ->
    %% Times = [{SchedulerId, ActiveTime, TotalTime}]
    {TotalActive, TotalWall} = lists:foldl(
        fun({_Id, Active, Wall}, {AccActive, AccWall}) ->
            {AccActive + Active, AccWall + Wall}
        end,
        {0, 0},
        Times
    ),
    case TotalWall of
        0 -> 0.0;
        _ -> min(1.0, TotalActive / TotalWall)
    end.

%% @private Estimate CPU usage from run queue.
estimate_cpu_from_run_queue() ->
    RunQueue = erlang:statistics(run_queue),
    Schedulers = erlang:system_info(schedulers),
    %% If run queue is at scheduler count, we're at 100%
    min(1.0, RunQueue / Schedulers).

%% @private Get sampled message queue length.
%%
%% Samples a subset of processes to estimate total queue length.
%% Full enumeration would be too expensive.
get_sampled_message_queue_len() ->
    %% Sample up to 100 random processes
    AllProcs = erlang:processes(),
    SampleSize = min(100, length(AllProcs)),
    Sampled = sample_random(AllProcs, SampleSize),

    %% Sum message queue lengths
    Total = lists:foldl(
        fun(Pid, Acc) ->
            case erlang:process_info(Pid, message_queue_len) of
                {message_queue_len, Len} -> Acc + Len;
                undefined -> Acc
            end
        end,
        0,
        Sampled
    ),

    %% Extrapolate to full process count if we sampled
    case {SampleSize, length(AllProcs)} of
        {S, T} when S < T -> round(Total * T / S);
        _ -> Total
    end.

%% @private Sample N random elements from list.
sample_random(List, N) when N >= length(List) ->
    List;
sample_random(List, N) ->
    %% Simple reservoir sampling
    sample_random(List, N, [], 0).

sample_random([], _, Acc, _) ->
    Acc;
sample_random(_, N, Acc, _) when length(Acc) >= N ->
    Acc;
sample_random([H | T], N, Acc, Idx) ->
    %% Include with decreasing probability
    case rand:uniform() < (N / (Idx + 1)) of
        true when length(Acc) < N ->
            sample_random(T, N, [H | Acc], Idx + 1);
        true ->
            %% Replace random existing element
            ReplaceIdx = rand:uniform(length(Acc)),
            NewAcc = replace_nth(Acc, ReplaceIdx, H),
            sample_random(T, N, NewAcc, Idx + 1);
        false ->
            sample_random(T, N, Acc, Idx + 1)
    end.

%% @private Replace Nth element in list.
replace_nth([_ | T], 1, New) -> [New | T];
replace_nth([H | T], N, New) -> [H | replace_nth(T, N - 1, New)].
