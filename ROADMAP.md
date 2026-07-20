# faber-neuroevolution Roadmap

What this package intends to implement but does not yet.

`README.md` states what **is**. This file states what **will be**. A capability
moves from here into the README when it lands, accompanied by a test that
exercises it against real fitness rather than a mock.

Nothing here may be described as a feature in the README, the guides, EDoc
comments, or the hex package description until it moves.

---

## 1. Close the meta-learning loop

**Status:** the forward pass runs; nothing learns from it.

`meta_controller` is genuinely wired in and computes a real LTC/CfC forward
pass. But `ltc_weights` is assigned in `init` and in `reset` and **nowhere
else**. The weights are randomly initialised and then frozen for the lifetime
of the process, so it is a fixed random projection from evolution statistics to
hyperparameters, with a reward counter attached.

`process_generation/2` builds a `#meta_training_event{}` with
`gradients = #{}` hardcoded, passes it only to `maybe_log_progress/2`, and
discards the reward.

`meta_trainer` exports exactly the four functions that would close the loop,
`update_weights/4`, `compute_gradients/3`, `apply_gradients/3` and
`estimate_advantage/2`, and has **zero callers**.

Until this lands, the README must not describe the meta-controller as learning
or adapting. It selects hyperparameters; it does not improve at doing so.

## 2. Credit assignment in `lc_chain`

**Status:** present but not a learning rule.

`lc_chain:update_weight_specs/3` computes `Delta = LR * Reward * sign(W)`. The
update depends only on the sign of each weight, not on that weight's input, its
output, or its contribution to the outcome. On positive reward every weight in
the network moves away from zero by the same magnitude; on negative reward
every weight moves toward zero. It is a global gain knob that saturates at the
`clamp_weight/1` bound of ±10.

Intended: an update with actual credit assignment. Until then this is not
"learning" in any sense a reader would expect from the word.

## 3. Wire the silos, or remove them

**Status:** 13 silos implemented, none connected.

Every silo implements `apply_actuators/2` and `compute_reward/1`. Both have
**zero callers** across `src/` and `test/`. The 26 silo mutator functions
(`record_match`, `update_elo`, `record_innovation`, `register_niche`,
`record_income`, `add_epigenetic_mark`, `update_reputation`, `form_coalition`,
`send_message`, `update_episode`, and the rest) likewise have zero callers.

`collect_sensors/1` **is** called, but only by each silo reading its own state
and publishing to `lc_cross_silo`. Nothing closes the loop.

The sensors compute real arithmetic, but over state that nothing ever writes,
so they return constants. `temporal_silo:compute_reward/1` would return exactly
0.85 on every call, forever. This is a subtler failure than a stub and harder
to notice.

Ten of thirteen silos have zero non-self references anywhere in `src/`. Only
`task_silo` and `resource_silo` are consulted by `neuroevolution_server`, and
then only via `get_recommendations`/`get_state`, never a mutator.
`lc_supervisor:build_extension_silo_specs/1` defaults every `enable_*_silo` key
to `false`, so out of the box none of them start.

This is not a gap to fill mechanically. There is no book chapter and no
literature baseline for it, so there is no way to distinguish a correct
implementation from an incorrect one. It needs a falsifiable hypothesis before
any code, and it is gated behind a benchmarked foundation that can evaluate it.

Also unreferenced and unsupervised: `lc_controller` (16KB, a complete ES online
learner) and `lc_population` (24KB).

## 4. Transfer of meta-knowledge across domains

**Status:** not implemented.

The README claims training strategies transfer across domains. No mechanism
exists: nothing serialises a learned meta-policy, nothing loads one into a
different domain, and per item 1 there is no learned policy to transfer in the
first place.

Depends on item 1.

## 5. Portfolio manager for continual learning

**Status:** not implemented.

`guides/continual-learning.md` documents `portfolio_manager:init/1`. No
`portfolio_manager` module exists.

## 6. Verify the evolution strategies

**Status:** partially verified as of Phase 2. Three of five work; two are broken.

Running XOR against real fitness for the first time (see `test/xor_tests.erl`)
established:

| Strategy | State |
|---|---|
| `generational_strategy` | completes a run, returns an evaluable champion |
| `steady_state_strategy` | completes a run |
| `novelty_strategy` | completes a run |
| `island_strategy` | **broken.** `badarg` from `erlang:length/1` applied to the island map. Kills the calling process. |
| `map_elites_strategy` | **broken.** Fails to complete a run. |

`known_broken_strategies_test_/0` asserts the two failures so the suite stays
green while the breakage stays visible. Fixing either makes that test fail,
which is the signal to move it into the working list.

island_strategy matters most: it is the headline distribution feature and
Phase 9 of PLAN_FABER_FOUNDATION rests on it.

Original note, still true of the two broken ones:

`generational_strategy`, `steady_state_strategy`, `island_strategy`,
`novelty_strategy` and `map_elites_strategy` are all tested exclusively against
`mock_network_factory` and `mock_evaluator`. The mock mutation operator
**discards the parent's weights and regenerates them at random**, and the mock
evaluator returns `rand:uniform() * 100`.

So every strategy test verifies bookkeeping (population size, event emission,
record shape) against a system in which selection pressure is provably absent.
The island model in particular, which is the headline distribution feature, has
never been shown to work.

This is not a missing feature; it is a missing proof that an existing feature
works. Until a strategy has been run against real fitness on a problem with a
known answer, the README should not present these as verified capabilities.

## 7. Mesh distribution

**Status:** code present, dependency disabled.

`src/distribute/` contains `distributed_evaluator`, `evaluator_pool_registry`,
`macula_mesh` and `mesh_sup`, but the `macula` dependency is commented out in
`rebar.config`, and `macula_mesh.erl:293` reads
"This is a placeholder - actual macula integration would go here".

Intended scope is narrow: the mesh transfers **genomes only**. Evolution and
fitness evaluation are sharded at the node level, in-process. There is no
intent to use the mesh as a distributed compute fabric with per-step
communication, which would be latency-bound and far slower than local
evaluation.

Gated on a measurement showing that one machine is insufficient.

---

## Not planned

- **The mesh as a distributed evaluator.** Genomes cross the mesh; observations
  and actions do not.
- **Replacing `network_evaluator` with the process-per-neuron path.** Both are
  kept deliberately: the process path is the reference phenotype for
  correctness, plasticity and substrate work; the vectorised path is for fast
  fitness evaluation. They are held in agreement by a conformance test.
