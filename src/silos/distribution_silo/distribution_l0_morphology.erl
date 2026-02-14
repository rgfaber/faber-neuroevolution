%% @doc Distribution Silo L0 Morphology - TWEANN sensor/actuator definitions.
%%
%% Part of the Liquid Conglomerate v2 architecture. Defines the neural network
%% morphology for the Distribution Silo's L0 load balancing controller.
%%
%% == Architecture ==
%%
%% L0 is a TWEANN (Topology and Weight Evolving Artificial Neural Network) that:
%% - Takes 14 sensor inputs (normalized 0.0-1.0)
%% - Produces 10 actuator outputs (distribution decisions)
%% - Has 8 hyperparameters that L1 can tune
%% - Has 5 L1 hyperparameters that L2 can tune
%%
%% == Time Constant ==
%%
%% tau_L0 = 1 second (real-time routing decisions)
%%
%% Note: The Distribution Silo operates on wall-clock time for networking
%% decisions, not evaluation counts like the Task Silo.
%%
%% @author R.G. Lefever
%% @copyright 2024-2026 R.G. Lefever
-module(distribution_l0_morphology).

-export([
    %% Morphology definitions
    sensor_count/0,
    actuator_count/0,
    sensor_names/0,
    actuator_names/0,
    sensor_spec/1,
    actuator_spec/1,

    %% Hyperparameter definitions
    l0_hyperparameters/0,
    l0_hyperparameter_spec/1,
    l1_hyperparameters/0,
    l1_hyperparameter_spec/1,

    %% Bounds and defaults
    get_l0_defaults/0,
    get_l0_bounds/0,
    get_l1_defaults/0,
    get_l1_bounds/0,

    %% Time constants
    tau_l0/0,
    tau_l1/0,
    tau_l2/0
]).

%%% ============================================================================
%%% Time Constants (in milliseconds for real-time networking)
%%% ============================================================================

%% @doc L0 time constant - 1 second for real-time routing decisions.
-spec tau_l0() -> pos_integer().
tau_l0() -> 1000.  % milliseconds

%% @doc L1 time constant - 10 seconds for load balancing adaptation.
-spec tau_l1() -> pos_integer().
tau_l1() -> 10000.  % milliseconds

%% @doc L2 time constant - 1 minute for topology learning.
-spec tau_l2() -> pos_integer().
tau_l2() -> 60000.  % milliseconds

%%% ============================================================================
%%% Sensor Definitions (14 inputs)
%%% ============================================================================

%% @doc Number of sensors (neural network inputs).
-spec sensor_count() -> pos_integer().
sensor_count() -> 14.

%% @doc Ordered list of sensor names.
-spec sensor_names() -> [atom()].
sensor_names() ->
    [
        local_load,                    % 1. Current node pressure
        local_queue_depth,             % 2. Pending evaluations
        peer_count,                    % 3. Network connectivity
        avg_peer_load,                 % 4. Network health
        min_peer_load,                 % 5. Best offload target
        network_latency_avg,           % 6. Communication cost
        network_bandwidth_ratio,       % 7. Bandwidth headroom
        island_count,                  % 8. Island topology
        migration_success_rate,        % 9. Migration health
        species_distribution_entropy,  % 10. Diversity spread
        resource_silo_pressure,        % 11. Cross-silo: Resource constraint
        task_silo_exploration,         % 12. Cross-silo: Evolution aggressiveness
        evaluation_batch_pending,      % 13. Queued batches
        time_since_last_migration      % 14. Migration timing
    ].

%% @doc Get specification for a sensor.
-spec sensor_spec(atom()) -> map() | undefined.
sensor_spec(local_load) ->
    #{
        name => local_load,
        range => {0.0, 1.0},
        source => resource_monitor,
        description => <<"Local node CPU/memory pressure (combined)">>
    };
sensor_spec(local_queue_depth) ->
    #{
        name => local_queue_depth,
        range => {0.0, 1.0},
        source => internal,
        description => <<"Pending evaluations / max queue depth">>
    };
sensor_spec(peer_count) ->
    #{
        name => peer_count,
        range => {0.0, 1.0},
        source => mesh,
        description => <<"Connected peers / max peers">>
    };
sensor_spec(avg_peer_load) ->
    #{
        name => avg_peer_load,
        range => {0.0, 1.0},
        source => mesh,
        description => <<"Average peer load (from heartbeats)">>
    };
sensor_spec(min_peer_load) ->
    #{
        name => min_peer_load,
        range => {0.0, 1.0},
        source => mesh,
        description => <<"Load of least loaded peer">>
    };
sensor_spec(network_latency_avg) ->
    #{
        name => network_latency_avg,
        range => {0.0, 1.0},
        source => mesh,
        description => <<"Average RTT to peers (normalized)">>
    };
sensor_spec(network_bandwidth_ratio) ->
    #{
        name => network_bandwidth_ratio,
        range => {0.0, 1.0},
        source => mesh,
        description => <<"Available bandwidth / max bandwidth">>
    };
sensor_spec(island_count) ->
    #{
        name => island_count,
        range => {0.0, 1.0},
        source => evolution_stats,
        description => <<"Current islands / max islands">>
    };
sensor_spec(migration_success_rate) ->
    #{
        name => migration_success_rate,
        range => {0.0, 1.0},
        source => computed,
        description => <<"Successful migrations / attempted migrations">>
    };
sensor_spec(species_distribution_entropy) ->
    #{
        name => species_distribution_entropy,
        range => {0.0, 1.0},
        source => computed,
        description => <<"Species distribution entropy across islands">>
    };
sensor_spec(resource_silo_pressure) ->
    #{
        name => resource_silo_pressure,
        range => {0.0, 1.0},
        source => cross_silo,
        description => <<"Resource Silo's pressure signal">>
    };
sensor_spec(task_silo_exploration) ->
    #{
        name => task_silo_exploration,
        range => {0.0, 1.0},
        source => cross_silo,
        description => <<"Task Silo's exploration aggressiveness">>
    };
sensor_spec(evaluation_batch_pending) ->
    #{
        name => evaluation_batch_pending,
        range => {0.0, 1.0},
        source => internal,
        description => <<"Queued evaluation batches (normalized)">>
    };
sensor_spec(time_since_last_migration) ->
    #{
        name => time_since_last_migration,
        range => {0.0, 1.0},
        source => computed,
        description => <<"Time since last migration (normalized to cooldown)">>
    };
sensor_spec(_) ->
    undefined.

%%% ============================================================================
%%% Actuator Definitions (10 outputs)
%%% ============================================================================

%% @doc Number of actuators (neural network outputs).
-spec actuator_count() -> pos_integer().
actuator_count() -> 10.

%% @doc Ordered list of actuator names.
-spec actuator_names() -> [atom()].
actuator_names() ->
    [
        local_vs_remote_ratio,          % 1. Evaluation routing
        migration_rate,                 % 2. Population exchange rate
        migration_selection_pressure,   % 3. Which individuals to migrate
        target_island_selection,        % 4. Migration target strategy
        island_split_threshold,         % 5. When to split island
        island_merge_threshold,         % 6. When to merge islands
        load_balance_aggressiveness,    % 7. Work distribution
        peer_selection_strategy,        % 8. Which peer to use
        batch_size_for_remote,          % 9. Remote batch granularity
        topology_change_rate            % 10. Island graph evolution
    ].

%% @doc Get specification for an actuator.
-spec actuator_spec(atom()) -> map() | undefined.
actuator_spec(local_vs_remote_ratio) ->
    #{
        name => local_vs_remote_ratio,
        range => {0.0, 1.0},
        target => evaluation_routing,
        description => <<"0=all remote, 1=all local">>
    };
actuator_spec(migration_rate) ->
    #{
        name => migration_rate,
        range => {0.0, 0.2},
        target => population_exchange,
        description => <<"Fraction of population to migrate per tau cycle">>
    };
actuator_spec(migration_selection_pressure) ->
    #{
        name => migration_selection_pressure,
        range => {0.0, 1.0},
        target => migration_strategy,
        description => <<"0=random, 1=best only">>
    };
actuator_spec(target_island_selection) ->
    #{
        name => target_island_selection,
        range => {0.0, 1.0},
        target => migration_strategy,
        description => <<"0=random target, 1=most different target">>
    };
actuator_spec(island_split_threshold) ->
    #{
        name => island_split_threshold,
        range => {0.5, 0.95},
        target => island_topology,
        description => <<"Speciation threshold to trigger island split">>
    };
actuator_spec(island_merge_threshold) ->
    #{
        name => island_merge_threshold,
        range => {0.1, 0.5},
        target => island_topology,
        description => <<"Similarity threshold to trigger island merge">>
    };
actuator_spec(load_balance_aggressiveness) ->
    #{
        name => load_balance_aggressiveness,
        range => {0.0, 1.0},
        target => work_distribution,
        description => <<"How eagerly to offload work to peers">>
    };
actuator_spec(peer_selection_strategy) ->
    #{
        name => peer_selection_strategy,
        range => {0.0, 1.0},
        target => peer_selection,
        description => <<"0=nearest peer, 1=least loaded peer">>
    };
actuator_spec(batch_size_for_remote) ->
    #{
        name => batch_size_for_remote,
        range => {1, 20},
        target => network_efficiency,
        description => <<"Batch size for remote evaluations">>
    };
actuator_spec(topology_change_rate) ->
    #{
        name => topology_change_rate,
        range => {0.0, 0.1},
        target => island_graph,
        description => <<"Rate of island connection changes">>
    };
actuator_spec(_) ->
    undefined.

%%% ============================================================================
%%% L0 Hyperparameters (8 params, tuned by L1)
%%% ============================================================================

%% @doc List of L0 hyperparameter names.
-spec l0_hyperparameters() -> [atom()].
l0_hyperparameters() ->
    [
        max_islands,                 % 1. Maximum island count
        min_island_size,             % 2. Minimum individuals per island
        migration_cooldown_ms,       % 3. Between migrations
        load_imbalance_threshold,    % 4. When to rebalance
        network_timeout_ms,          % 5. Communication timeout
        local_preference_base,       % 6. Default local bias
        diversity_migration_boost,   % 7. Migration for diversity
        topology_stability_weight    % 8. Resist topology changes
    ].

%% @doc Get specification for an L0 hyperparameter.
-spec l0_hyperparameter_spec(atom()) -> map() | undefined.
l0_hyperparameter_spec(max_islands) ->
    #{
        name => max_islands,
        default => 8,
        range => {2, 32},
        description => <<"Maximum number of islands">>
    };
l0_hyperparameter_spec(min_island_size) ->
    #{
        name => min_island_size,
        default => 10,
        range => {5, 50},
        description => <<"Minimum individuals per island">>
    };
l0_hyperparameter_spec(migration_cooldown_ms) ->
    #{
        name => migration_cooldown_ms,
        default => 5000,
        range => {1000, 30000},
        description => <<"Minimum milliseconds between migrations">>
    };
l0_hyperparameter_spec(load_imbalance_threshold) ->
    #{
        name => load_imbalance_threshold,
        default => 0.3,
        range => {0.1, 0.6},
        description => <<"Load difference threshold for rebalancing">>
    };
l0_hyperparameter_spec(network_timeout_ms) ->
    #{
        name => network_timeout_ms,
        default => 1000,
        range => {100, 5000},
        description => <<"Timeout for network operations">>
    };
l0_hyperparameter_spec(local_preference_base) ->
    #{
        name => local_preference_base,
        default => 0.8,
        range => {0.5, 1.0},
        description => <<"Base preference for local evaluation">>
    };
l0_hyperparameter_spec(diversity_migration_boost) ->
    #{
        name => diversity_migration_boost,
        default => 1.5,
        range => {1.0, 3.0},
        description => <<"Migration rate boost for diversity">>
    };
l0_hyperparameter_spec(topology_stability_weight) ->
    #{
        name => topology_stability_weight,
        default => 0.5,
        range => {0.0, 1.0},
        description => <<"Resistance to topology changes">>
    };
l0_hyperparameter_spec(_) ->
    undefined.

%% @doc Get default values for L0 hyperparameters.
-spec get_l0_defaults() -> map().
get_l0_defaults() ->
    #{
        max_islands => 8,
        min_island_size => 10,
        migration_cooldown_ms => 5000,
        load_imbalance_threshold => 0.3,
        network_timeout_ms => 1000,
        local_preference_base => 0.8,
        diversity_migration_boost => 1.5,
        topology_stability_weight => 0.5
    }.

%% @doc Get bounds for L0 hyperparameters.
-spec get_l0_bounds() -> map().
get_l0_bounds() ->
    #{
        max_islands => {2, 32},
        min_island_size => {5, 50},
        migration_cooldown_ms => {1000, 30000},
        load_imbalance_threshold => {0.1, 0.6},
        network_timeout_ms => {100, 5000},
        local_preference_base => {0.5, 1.0},
        diversity_migration_boost => {1.0, 3.0},
        topology_stability_weight => {0.0, 1.0}
    }.

%%% ============================================================================
%%% L1 Hyperparameters (5 params, tuned by L2)
%%% ============================================================================

%% @doc List of L1 hyperparameter names.
%%
%% Note: These are called "meta-parameters" from L1's perspective,
%% but "hyperparameters" from L2's perspective.
-spec l1_hyperparameters() -> [atom()].
l1_hyperparameters() ->
    [
        load_sensitivity,                % 1. Response to load changes
        migration_adaptation_rate,       % 2. How fast to adjust migration
        topology_learning_rate,          % 3. Island structure adaptation
        cross_silo_responsiveness,       % 4. React to other silos
        exploration_diversity_coupling   % 5. Link exploration to distribution
    ].

%% @doc Get specification for an L1 hyperparameter.
-spec l1_hyperparameter_spec(atom()) -> map() | undefined.
l1_hyperparameter_spec(load_sensitivity) ->
    #{
        name => load_sensitivity,
        default => 1.0,
        range => {0.5, 2.0},
        description => <<"L1's response amplitude to load changes">>
    };
l1_hyperparameter_spec(migration_adaptation_rate) ->
    #{
        name => migration_adaptation_rate,
        default => 0.1,
        range => {0.01, 0.3},
        description => <<"How fast L1 adjusts migration parameters">>
    };
l1_hyperparameter_spec(topology_learning_rate) ->
    #{
        name => topology_learning_rate,
        default => 0.05,
        range => {0.01, 0.2},
        description => <<"L1's island structure adaptation speed">>
    };
l1_hyperparameter_spec(cross_silo_responsiveness) ->
    #{
        name => cross_silo_responsiveness,
        default => 0.5,
        range => {0.0, 1.0},
        description => <<"How much L1 reacts to other silos">>
    };
l1_hyperparameter_spec(exploration_diversity_coupling) ->
    #{
        name => exploration_diversity_coupling,
        default => 0.5,
        range => {0.0, 1.0},
        description => <<"L1's link between exploration and distribution">>
    };
l1_hyperparameter_spec(_) ->
    undefined.

%% @doc Get default values for L1 hyperparameters.
-spec get_l1_defaults() -> map().
get_l1_defaults() ->
    #{
        load_sensitivity => 1.0,
        migration_adaptation_rate => 0.1,
        topology_learning_rate => 0.05,
        cross_silo_responsiveness => 0.5,
        exploration_diversity_coupling => 0.5
    }.

%% @doc Get bounds for L1 hyperparameters.
-spec get_l1_bounds() -> map().
get_l1_bounds() ->
    #{
        load_sensitivity => {0.5, 2.0},
        migration_adaptation_rate => {0.01, 0.3},
        topology_learning_rate => {0.01, 0.2},
        cross_silo_responsiveness => {0.0, 1.0},
        exploration_diversity_coupling => {0.0, 1.0}
    }.
