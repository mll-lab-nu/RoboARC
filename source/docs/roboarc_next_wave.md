# RoboARC next wave (v2): what is on dev/robotangle_mujoco

State of 2026-10-15. The RoboARC release on main is 0.1.0: 94 bases, 188 task dirs, image `eai-roboarc:0.1.0`.

**Owner decision (2026-10-15):**
- All next-wave work lives on this branch, as one whole.
- There is no PR to main. Main takes only necessary fixes.
- New tasks, solutions and oracles go to main later, with **v2**, the next evaluation wave. The owner will add more
  tasks to that wave.
- **Nothing listed here goes to main before the v2 wave.**

**Review pending.** The texts and demos of the new tasks have not been reviewed by the owner yet:
- knot-untying-06-i01 and knot-untying-09-i01 (instruction, README, task.toml, and 09's region images);
- blanket-flatten-07-i01 and blanket-flatten-08-i01 (instructions).

No sweep runs on them before that review.

## What went in

| line | source (local, unpushed until now) | commits |
|---|---|---|
| speed cuts in topogym (host side, bit-identical) | `qineng-rt-mj-speed` | 8fb85d3b9d |
| 6 standard references within the joint-speed rule, release_rebuild keeps them | `dev/speedrule-refs` (the dev-side variant) | 7ff4bb983e, c99cde0261 |
| knot_untying rd as knot-untying-06-i01 / 09-i01 | `knot-dev` | 4f08f07d2e .. ca9b11abf8 (12 commits) |
| rope_threading rd 04-08 standard references | `rope_rd_speedlimit` | aa555f979c .. 3ede328cfd (19 commits), then c757f39d58 |
| buzz-wire-3d 03 / 04 references, 05 negative results | `dev/missing-refs` | f34f6fa192 .. 6c6a86d144 (7 commits) |
| blanket_flatten easy ported as blanket-flatten-07-i01 | `qineng-excluded-tiers` | 8966131959 .. a14b259961 (4 commits) |
| blanket_flatten medium ported as blanket-flatten-08-i01 (user 2026-10-15) | `flatten-medium` | 2cf580d63e and the commit after it |
| release_rebuild fixes found by this integration | this branch | e9f2a49414, a13dd0f854, cd872d8459 |

Three things were already in the base b903ab2530:
- the nyx_direct rope-birth fix, d0ad7be359 (the same patch as knot-dev's 01d384097e);
- the nyx_direct fixed-link fix, b903ab2530 (issue #210);
- knot-dev's copy of the rope-birth fix, which merged as a no-op.

`missing-refs-render` is a main-side render tree, built on dev/robotangle. Its buzz-03 / 04 release.json evidence
belongs to a v2 render on main, not to this branch. Its Harbor results are listed in the table below.

`dev/speedrule-refs-main` (31f7a28beb) is the main-side variant of the speed-rule references. It has two parts:
- the release.json standard evidence;
- the mk_tasks.py wording for a standard reference with "speed_rule".

It is not merged here (main side). The v2 render on main needs an equivalent change; see "Before the v2 build".

## Bases: new or changed

The budget is 3 h (10800 s) per trial. "Timing" means the following:
- the reference's replay time on a trial-class host, as its author measured it;
- "paired" means normalised against a control run in the same window.

| base | change | reference | timing (% of 3 h) | checks passed |
|---|---|---|---|---|
| knot-untying-06-i01 (rd very_easy) | **new base** | fv81az: 4809 rows, end e235713d44851ce5, sha256 dc88f5ff8f4c81f7 | paired 30.7 min = **17.1 %**; unpaired host C 41.4 min (23.0 %) | premade instance birth exact (b0c35b5a83a3f30c, two makes byte-identical); release-image replay success, no forcing, joint steps <= 0.039 rad; knot stress round clean, nop / -i01 / 5 breakers fail |
| knot-untying-09-i01 (rd medium) | **new base** | qp7: 10559 rows, end 120b884db510d9a5, sha256 c7e6d878de08ce63 | paired 49.4 min = **27.4 %** (knot: user-approved 30 % bar) | birth exact (343aa690921a6a1d); release-image replay success, no forcing, joint steps <= 0.039 rad |
| blanket-flatten-07-i01 (easy, ported on blanket_vbd_gpu) | **new base** | 1581 rows, end a6f78e4255ff5adf, sha256 dfb55159b9ae2794 | strict fresh replay on host A 838.9 s = **7.8 %** | birth probe exact (12fe5f2d5f7374a7); observer replay and nop CLEAN (v2.13b); max joint step 0.0275 rad; re-record byte-identical |
| blanket-flatten-08-i01 (medium, ported on blanket_vbd_gpu) | **new base** | 1480 rows, end d539d4d85d80ab0e, sha256 3f9d713fc87c9bc8 | strict fresh replay on host A 843.0 s = **7.8 %** | birth probe exact (49ee5e7e16239616); observer replay and nop CLEAN (v2.13b); max joint step 0.0303 rad; re-record byte-identical. Not in gate C's dry run (added after it) |
| buzz-wire-3d-03-i01 (easy) | released without a reference -> **reference** | 9501 rows, end 0ddfb562ca575b00, sha256 565180eae29f6f63 | fresh replay 296-300 s on host C = **2.8 %**; Harbor trial 1025 s (9.5 %) | 2 fresh replays, same hash; stress observer CLEAN; Harbor oracle 1 / nop 0, both modes, eai-roboarc:0.1.0 |
| buzz-wire-3d-04-i01 (medium) | released without a reference -> **reference** | 9219 rows, end bedce75e61d2be24, sha256 e092a8ab59fd604c | fresh replay 322-330 s on host C = **3.0 %**; Harbor trial 1105 s (10.2 %) | the same as 03 |
| rope-threading-04-i01 | standard reference re-recorded | 3888 rows, end 4c03879dec745ef3 | host C replay 33.0 min = **18.3 %** | live = fresh replay = service emulation; 0 rows over 0.04 rad; 0 outside the limits; 0 Session forcing; Harbor standard oracle on 0.1.0: reward 1, infra 0, clipped 0 |
| rope-threading-05-i01 | standard reference re-recorded | 4016 rows, end ccdf618be61ba534 | 33.4 min = **18.6 %** | the same |
| rope-threading-06-i01 | standard reference re-recorded | 5230 rows, end 3fbfb5c096cd678e | 46.2 min = **25.7 %** (privileged 53.5 min, same window) | the same |
| rope-threading-07-i01 | standard reference re-recorded | 4091 rows, end 390b6b10d175c98f | 36.2 min = **20.1 %** | the same |
| rope-threading-08-i01 | standard reference re-recorded | 5520 rows, end caaf4eb644bc4593 | 64.6 min at load ~150 = **35.9 %** (privileged 70.4 min, same window); **the user accepted 08's timing** | the same |
| fiber-bundling-07-i01 | standard reference within the speed rule | 6573 rows, end 68db8dfdc178b49c | Harbor standard trial 4169 s = **38.6 %** (old 4524 s) | 2 fresh replays in the release runtime; Harbor standard oracle on eai-roboarc:0.1.0: reward 1, infra 0, live end = replay end, clipped 0, every row one control step |
| fiber-bundling-08-i01 | the same | 6207 rows, end 9d3193cedbb68aed | 3078 s = **28.5 %** (old 4727 s) | the same |
| fiber-bundling-09-i01 | the same, plus one 2.4e-7 rad nudge | 14546 rows, end 747d33ff0f5ba7a8 | 9136 s = **84.6 %** (old 9748 s) | the same |
| fiber-bundling-10-i01 | the same | 6210 rows, end 34ecd704a1faff46 | 5161 s = **47.8 %** (old 5325 s) | the same |
| interlocked-claws-12-i01 | the same | 21120 rows, end 5671cc01e7914e63 | 4752 s = **44.0 %** (old 5834 s) | the same |
| buzz-wire-3d-09-i01 | 3 fast runs re-timed in place | 18024 rows, end a6b4b870da34c0f1 | 1569 s = **14.5 %** | the same (before: no success in standard mode) |

Two notes on the table:
- The fiber / claws / buzz-09 times are whole Harbor standard trials on a shared host C.
- The speed-rule references only replace the standard oracle. Each base's privileged reference and frozen instance
  are unchanged.

**Speed cuts (8fb85d3b9d):** these change no physics. The evidence:
- knot rd, 400 rows: the same end hash as the base;
- the full knot reference, 23897 rows: ends at the record's end;
- rope rd hard, 300 rows: the same hash;
- the 9 staged blanket references end at their staged hashes.

evaluate_state goes from 56.0 to 36.3 ms per control step. The cuts change the topogym source, so the v2 image has a
new source sha.

## Release tooling (tools/release/release_rebuild.py)

- `standard_reference()` (from dev/speedrule-refs) keeps a tree `standard.npz` whose meta has `"speed_rule"`; the
  rebuild no longer overwrites it with the privileged reference.
- **Birth guard (e9f2a49414).** The tree's cast-coil-04 / 05 `standard.npz` were recorded before the re-freeze, on
  births f11ed465 / 4babdba8. The released references use births 0a4dab04 / e0d93b1e. Without the guard, the rebuild
  would have shipped those stale files. A speed-rule file is now kept only on its reference's birth.
- **rope rd files moved and tagged (c757f39d58).** rope_rd_speedlimit wrote its five files to `scripts/roboarc/oracles/`,
  which the tool does not read. They now live in `scripts/roboarc/oracles/<base>/standard.npz` and carry
  `meta.speed_rule`. Their arrays are byte-identical.
- **Template texts from the head (a13dd0f854).** Main no longer has `tools/release/templates` (#187). So
  `--onto main` found no template texts for a new base, or for a tier ported now. They are now read from the head.
- **--onto checks (cd872d8459).** The changed tasks are now found by the `<bench>-` prefix. `harbor add --scan` runs
  only where `tasks/dataset.toml` still exists; main dropped it in #215.

## Gate results (2026-10-15)

**A. Release inputs of main's 94 bases.** Against b903ab2530, this branch changes only these:
- the standard references of the 11 bases above;
- the references of buzz-wire-3d-03 / 04;
- the new bases' oracle dirs and package_text dirs;
- mujoco_release.json: the knot new_bases, the blanket_flatten number 07, and the blanket_flatten tiers' overrides
  (blanket_vbd_gpu, settle up to 60 s; very_easy ships its premade instance unchanged).

No released base's package_text changed. The tooling (rcb_topogym.py, render_bridge.py, nyx_direct.py) is
byte-identical to main's `scripts/roboarc/shared/`.

**B. Regenerating main's 94 bases.** The test lays this branch's release-tool oracle output over origin/main
b62fb383e5, then renders with main's `scripts/roboarc/mk_tasks.py`. The oracle output is: privileged = the 0.1.0
reference; standard = `standard_reference()`.
- All 188 task dirs render.
- Only these 11 standard task dirs change, each in `solution/oracle_qpos.npy` and `DIGEST`:
  - buzz-wire-3d-09
  - fiber-bundling-07 / 08 / 09 / 10
  - interlocked-claws-12
  - rope-threading-04 / 05 / 06 / 07 / 08
- Nothing else changes.
- The six speed-rule `oracle_qpos.npy` are byte-identical to dev/speedrule-refs-main's render.
- The five rope `oracle_qpos.npy` are byte-identical to the dirs that passed the Harbor standard oracle on host A.
- Not rendered: the buzz-03 / 04 references, because main's release.json marks those bases `oracle none`. They need
  their evidence in release.json; missing-refs-render has it.

**C. New tasks build.** The test is a release_rebuild dry run on host C:
- `--only` the three new bases;
- their premade instances;
- `--onto origin/main --bench roboarc --no-harbor`;
- the two docker builds replaced by tags of `eai-roboarc:0.1.0` (same runtime);
- files under `<host work dir>`.

Result, run 2: **the 6 new task dirs build** (194 task dirs rendered). The run's checks:

| check | result |
|---|---|
| `mk_tasks.py --check`, all bases | 194 match a fresh render |
| `--assets <work> --check`, the instance bytes of the 3 new bases | match |
| `task_conventions --check` | pass |
| check-resources, check-cutoffs, check-seeds | pass |
| `ci_checks/run_invariants.sh` | all invariants hold |
| advisory checks | one finding for the new dirs (below) |

The 188 existing dirs differ from main only in the image tag and DIGEST.

Run 1 stopped on knot-untying-09-i01. Main's generator caps a trajectory at the rows that two verifier replays can fit:
- The cap uses the step time the instruction states. The knot texts state 1.1 s per step, inherited from the Genesis
  texts, which gives a cap of 13000 rows.
- qp7 has 10559 rows, and 1.5 x 10559 = 15838 rows, which is over the cap.

Run 2 therefore assumed `step_s` 0.3 for 09. That is the paired trial-host time, 49.4 min / 10559 rows = 0.28 s,
rounded up. It also used blanket-flatten-07's own `step_s`, 0.53. **The owner decides whether 09's texts state a
measured step time.** The knot agent left `step_s` out pending that OK.

The dry run also exposed three gaps in release_rebuild against today's main. All three are fixed on this branch:
- the template texts (a13dd0f854);
- the changed tasks were matched by the old `robotangle-` prefix;
- the tool read `tasks/dataset.toml`, which main dropped in #215 (fixed in cd872d8459).

The record (host C) is in `<host work dir>` (INDEX.md there):
- `nw_manifest2.json`, `nw_dry.py`;
- `logs/dry1_*`, `logs/dry2.log`, `logs/checks/`;
- `work/render`, the rendered tree.

The local `0.2.0-nwdry` tags were removed afterwards; the image `eai-roboarc:0.1.0` is untouched.

## Not done (next wave)

- **buzz-wire-3d-05-i01 (hard): no reference.** Two time boxes gave negative results; the details are in
  docs/rcb/families/buzz_wire_3d.md. The tree still holds an old in-tree `oracles/buzz-wire-3d-05-i01` that does not
  work. The v2 manifest must keep `"reference": null` for it.
- **fiber-bundling-05-i01: dropped** (user, 2026-10-14). It stays without a reference.
- **blanket-fold-05 and blanket-flatten-06:** their standard references do not work. Re-plan in the next wave. The
  findings are in the speed-rule agent's FOLD05_FINDINGS.md / FLATTEN06_FINDINGS.md (the work notes).
  V3: blanket-fold-05 has a new reference within the rule (below, V3 release build); blanket-flatten-06 is dropped
  (main's docs/easy_set.toml: "shortcut") and keeps its released reference.
- Not shippable this wave (over the 30 % bar):
  - the knot_untying rd easier and hard tiers: et1 ~39 %, hu1 ~42 %;
  - orbits-easier-i01: a breakdown in its stress round;
  - the blanket hard Genesis port: done in V3 as blanket-flatten-09-i01 (route hard5, docs/rcb/families/blanket_flatten.md);
    shipped in V3 (below).

## v2 finishing (2026-10-15)

The owner's decisions of 2026-10-15, done on this branch (work notes: workstation
the work notes, host B `<host work dir>`, INDEX.md in
both).

**Step time of knot-untying-09-i01 (and 06).** The texts now state the measured step time (`step_s` in
`records/freeze_manifests/freeze_manifest.v2.json`; main's mk_tasks.py states it and sizes the row cap from it). How the
released bases derive theirs: none of main's 94 has `step_s`; each states the sentence of the released Genesis text it
ports (knot 01-05: 1.1-1.3 s), and a template base states its template's. That Genesis 1.1 s gave 09 a cap of 13000
rows, under 1.5 x 10559. Measured the way the trials run (eai-roboarc:0.1.0, host B Ryzen 9 7950X, one physical core per
sim with `--cpus 1`, the trials' env, rendering off, host load 3.5-18), the two new knot references, run side by side:

| base | reference | rows | wall | per step | of the 3 h budget | stated | row cap |
|---|---|---|---|---|---|---|---|
| knot-untying-09-i01 | qp7 | 10559 | 3338 s | 0.316 s | 30.9 % | 0.32 s | 45000 |
| knot-untying-06-i01 | fv81az | 4809 | 2149 s | 0.447 s | 19.9 % | 0.45 s | 31000 |

Both ended at their recorded ends (120b884db510d9a5, e235713d44851ce5). 06 states its own time as well: the same rope
chain and wave, and 1.1 s was no more measured for it than for 09. blanket-flatten-07-i01 and -08-i01 state 0.53 s and
0.57 s (their strict fresh replays on host A: 838.9 s / 1581 rows, 843.0 s / 1480 rows). The released bases keep their
released sentence. **qp7 is at the knot bar, not under it:**
30.9 % unpaired here, 27.4 % by the paired timing. The same replay in the v2 noinst image, while this run's Harbor
trials and dry runs loaded host B (load 10-23), took 3586 s = 33.2 % (0.340 s per step; fv81az 0.494 s, 22.0 %); the
speed cuts did not show through that load. Whether qp7 stands or 09 is re-planned shorter is the owner's call.

**Speed-rule wording.** `tools/release/standard_speed_rule.json` holds, per speed-rule standard reference, the
sentence and the end state; release_rebuild puts both into the standard evidence (`speed_rule`, `end_state`) and lays
dev/speedrule-refs-main's mk_tasks.py wording over the --onto generator (its anchors asserted). Harbor's standard oracle
of such a base is compared with that end. Gate B again (main b62fb383e5, the tree's oracles, the 11 bases' standard
evidence on 0.1.0): `mk_tasks.py --check` 188 match; only the 11 standard dirs change, each in README.md,
solution/oracle_qpos.npy and DIGEST; the six of dev/speedrule-refs-main are byte-identical to its render.

**References.** The tree's `oracles/<base>/` of the 29 stale or missing released bases are main's files (the 0.1.0
references; buzz-wire-3d-05's non-working file removed); all 90 match 0.1.0's sha256, and `standard_reference()` differs
from main only for the 11. `freeze_manifest.v2.json` names every base of the wave (98, blanket-flatten-08-i01 of ecb19ba89e
included) with reference, sha256, expect, and step_s for the four new bases; buzz-05 and fiber-05
have `"reference": null`. A run passes `--instances`: the premade instances of every base.

**Knot instances re-made** in the v2 noinst image (eai-roboarc:0.2.0-v2fin-noinst = 9d0bd3a6996a, source c9171f79),
following ca9b11abf8's runbook: each made twice from its source's exact inputs (byte-identical), then the birth check:
06 b0c35b5a83a3f30c, 09 343aa690921a6a1d, equal to the frozen states and the sources', instance_exact and assets_exact.
The new instance.json records the region tier's task text (`text`); the state did not change.

**`[metadata].dynamics`.** The new bases' texts declare `dynamics = "static"`, the line of main's 188 dirs
(package_text/knot_untying and, for blanket-flatten-07 / 08, package_text/blanket_flatten/blanket-flatten-easy-i00-* /
-medium-i00-*/task.toml: the released task.toml with the line).

**Dry runs of the tool** (real builds on host B, local tags only, `--no-harbor`):
- knot-06 / 09 (`--only`, the re-made instances): PASS. Their texts state 0.32 s / 0.45 s, RCB_MAX_STEPS 45000 / 31000,
  dynamics declared; mk_tasks --check, the instance bytes, ci_checks/run_all.sh, task_conventions, check-resources and
  check-cutoffs all ok.
- The whole wave (every base of freeze_manifest.v2.json, premade instances: the 0.1.0 image's 94, the two re-made,
  blanket-07 / 08):
  the first run PASSed its checks but rewrote main's released texts from the head's RoboTangle packages (41 task.toml
  and 94 compose / test.sh / Dockerfile comments lost the 0.1.0 rename's doc paths, 16 instruction lines main's later
  fixes). Fixed in release_rebuild: a base the render tree already releases keeps its texts. The run again, with
  blanket-flatten-08 (98 bases, 196 task dirs): PASS, all births equal their references'; against main, the packages
  change only in the 8 new dirs (and every lock); the solution files change only in the 11 standard dirs and buzz-wire-3d-03 / 04 (their new references);
  everything else that differs is this run's own record (image tag, bindings, evidence wording, dates, DIGEST).

**Harbor, knot-06 / 09** (oracle and nop, both modes, rendering off, the v2 dry run's image eai-roboarc:0.2.0-v2fin with
the re-made instances): all pass, every oracle at its reference's end, the evidence claims hold.

| task | oracle | nop | oracle end state | oracle s | nop s |
|---|---|---|---|---|---|
| knot-untying-06-i01-privileged | 1 | 0 | e235713d44851ce5 | 4536 | 43 |
| knot-untying-06-i01-standard | 1 | 0 | e235713d44851ce5 | 4726 | 141 |
| knot-untying-09-i01-privileged | 1 | 0 | 120b884db510d9a5 | 7083 | 44 |
| knot-untying-09-i01-standard | 1 | 0 | 120b884db510d9a5 | 7369 | 154 |

Four trials ran at once on host B. The oracle times include the verifier's two replays (privileged), or the live episode
and the collect hook's fresh replay (standard).

## Before the v2 build

- Bring the new bases' texts in line with main's current conventions: per-task cutoffs (#221), Protocol v2.0
  invariants (#238), and the honour rule (#206). Do this after the owner's review.
- Harbor oracle and nop, both modes, on the v2 image, for every new or changed base. These records become the
  evidence in release.json.
- If the v2 image's simulator source changes again, re-make the knot instances in its noinst image (the births must
  stay 8f3bafca255df8b6 / 602470d6a7beb30f since the 2026-10-17 re-freeze, below; before it b0c35b5a83a3f30c /
  343aa690921a6a1d).
- Done 2026-10-17 (da1f100c4): write_release words a premade instance as main's 0.1.0 rename step did, "made before this
  image was built (TopoGym source <its own>; its bytes are bound into the release image ...)" (audit S7).

## V2 release build (2026-10-17)

The release build onto main (host B `<host work dir>`, INDEX.md there).
What the build found and changed on this branch:

- **release_rebuild.py.**
  - A premade instance is worded as main's 0.1.0 rename step did (audit S7, da1f100c4).
  - Evidence and nop records are in main's form (the image and the date, never a host or a job name). A mode whose
    reference main already releases keeps main's record; so does every released base's nop record.
  - A new or changed reference takes its record from `tools/release/release_evidence.json`: the Harbor trial that
    passed it. Those are the 11 speed-rule standard references and buzz-wire-3d-03 / 04, all passed on
    eai-roboarc:0.1.0 on 2026-10-14. Without a record a reference is `not_run`, which is the case for the four new bases
    until their Harbor runs on the 0.2.0 image.
  - The Change Log keeps a released base's first date and adds this release's entries: a reference where there was
    none, a new reference of a mode, the empty failure heading dropped (laid over main's generator, as the speed-rule
    wording is).
- **knot-06 / 09 re-frozen: provenance out of the instance.**
  - The leak probe of the V2 image found the lab start's provenance in instance.json (`task_parameters.redesign.source`):
    the oracle plan (kind, order, pin, a world-coordinate layout path), notes and a cluster path. A privileged agent reads
    instance.json in /app, and the Session holds it. `lab_start/scene_start.json` repeated the plan.
  - `make_premade.py` now drops `redesign.source`, `lab_start/`, `redesign_build.lab_start` and the birth manifest's path
    note; it records their hashes in `redesign_build.premade.dropped`. task_rd.py never reads them. It does read
    `frame` (the table-top check and the origin), the ropes, fixtures and regions, so those stay; knot-01..05 carry the
    same facts as object positions.
  - The task's `instance_id` hashes the whole task config, so the births changed, with the physics unchanged (the first
    tick records differ only in `task.instance_id`): knot-06 b0c35b5a83a3f30c -> 8f3bafca255df8b6, knot-09
    343aa690921a6a1d -> 602470d6a7beb30f. Each was made twice in eai-roboarc:0.2.0-noinst, byte-identical, and the
    births are exact (instance_exact, assets_exact). The references' `instance_sha` is the new birth; their actions are
    byte for byte the same.
  - The leak probe of the re-frozen bases (files and Session memory) finds no plan word.
  - HF eai-v2-internal/robotangle-assets cb43201c28: the two re-made dirs, and the six stale `lab_start/` files deleted.
- **Reference meta hygiene.** The meta of fiber-bundling-07..10's `standard.npz` and of knot-09's reference named the cluster
  paths. Each path is now its file name; every other member is byte for byte the same (the freeze manifest's sha256
  follows).
- **Harbor.** Only the four new bases run on the 0.2.0 image (oracle and nop, both modes). The 94 released bases rely on
  the bit-identical render and their existing evidence.

## V3 release build (2026-10-18): eai-roboarc 0.2.1, 101 bases

The release build onto dev/robotangle (host B `<host work dir>`, workstation
the work notes), `release_rebuild.py` with `records/freeze_manifests/freeze_manifest.v3.json`.
- **Image 0.2.1** = 0.2.0's runtime plus MuJoCo 3.14.0's EPA horizon-overflow fix, IKEA's 14-byte patch of the wheel's
  libmujoco.so (images/ikea-mujoco/mujoco-3.14.0-epa/; `images/robotangle/Dockerfile.mujoco`, the last step of the
  unsquashed stage, so every layer above it is the released one). The gate (the work notes):
  every one of the 98 released bases reaches its reference end state on the patched library (Zen 4; Intel stock =
  patched on the 85 CPU bases; the 13 blanket bases stock = patched on the same RTX 4090 hosts, their Intel replays on
  the cluster's GPUs gate the image push), stress A/B 1150 / 1150 episodes identical.
  The built image's filesystem equals that gated candidate (0.2.0 + the patched .so) file for file, apart from the three
  new instance dirs and directory mtimes.
- **New bases** (the user approved their reviews on 2026-10-18), premade instances (`--instances`), HF
  `eai-v2-internal/robotangle-assets` 9f34f894a454:
  - orbits-02-i01: orbits easier on a 0.3125 ms MuJoCo step (solver_substeps 8; docs/rcb/families/orbits.md);
  - knot-untying-07-i01: knot_untying rd easier, a re-timed route (docs/rcb/families/knot_untying.md);
  - blanket-flatten-09-i01: the Genesis blanket_flatten hard tier on blanket_vbd_gpu, route hard5
    (docs/rcb/families/blanket_flatten.md).
- **blanket-fold-05-i01: a new reference** in both modes: the released route re-recorded with the oracle rate limiter's
  guard-stop clamp (48409e39d6): 10195 rows (was 17189 with one row over the rule), end 4f1b6724fc0a6512, max arm joint
  step 0.0390 rad, standard rows = privileged rows. It ships under a timing exception of <= 30 % of the budget (user,
  2026-10-18). Its record: release_evidence.json (Harbor on 0.2.0, oracle 1 / nop 0 in both modes).
- **release_rebuild.py.** The image holds only the instances of the manifest's `only` bases, so the V3 manifest lists all
  101 (the released ones with their 0.2.0 references). `later_release()` now keeps a base's earlier Change Log entries
  and its first release date (it kept only this run's entries).
- The new bases' release.json evidence is `not_run`, as V2's new bases: their Harbor runs on 0.2.1 are the evidence of
  the PR.
- **Release commit** dev/robotangle acfcec108 (on 554787ed9b): 202 task dirs; `mk_tasks --check`, run_invariants,
  unit tests, check-video and `run_all --changed` pass (one advisory: orbits-02's package lacked `[metadata] dynamics`,
  now "static" in its package texts).
- **Bit identity on the built image** (41a17481d67a): its filesystem equals the gated candidate (0.2.0 + the patched
  .so) file for file, apart from the three new instance dirs and directory mtimes; spot replays (one base per family,
  both blanket families on the GPU, and the two hold-only bases): 16 / 16 at the gate's end hash.
- **Harbor on 0.2.1** (host B, rendering off, oracle and nop in both modes): orbits-02, knot-07, blanket-flatten-09 and
  blanket-fold-05: every oracle 1 at its reference's end (b9c68d6502105df4, 8ae062ea4c733dbc, 6a9c4796da46cf47,
  4f1b6724fc0a6512; privileged deterministic and instance_exact, standard live = replay), every nop 0.

## Easy set (owner's selection, 2026-10-16)

The benchmark owner put these RoboARC bases in the **easy set**: they stay in the release but are left out of the final
statistics (reported apart). The full cross-benchmark list is `docs/easy_set.toml` (the same file as main PR #265); how
main marks and reports it is the maintainer's decision. 31 RoboARC bases:

- `roboarc-blanket-flatten-01-i01` (shortcut: Luna P: passed_only_in_replay_hold_tolerance)
- `roboarc-blanket-flatten-05-i01`
- `roboarc-blanket-flatten-06-i01` (shortcut: Luna P: passed_only_in_replay_hold_tolerance)
- `roboarc-blanket-fold-02-i01`
- `roboarc-buzz-wire-3d-03-i01`
- `roboarc-buzz-wire-3d-05-i01`
- `roboarc-buzz-wire-3d-08-i01`
- `roboarc-cast-coil-01-i01`
- `roboarc-cast-coil-02-i01`
- `roboarc-cast-coil-03-i01`
- `roboarc-cast-coil-05-i01`
- `roboarc-cast-keyhole-01-i01`
- `roboarc-cast-keyhole-02-i01`
- `roboarc-cast-keyhole-06-i01`
- `roboarc-fiber-bundling-01-i01`
- `roboarc-fiber-bundling-02-i01`
- `roboarc-fiber-bundling-03-i01`
- `roboarc-fiber-bundling-05-i01`
- `roboarc-fiber-bundling-08-i01`
- `roboarc-hedgehog-in-cage-03-i01`
- `roboarc-hedgehog-in-cage-04-i01`
- `roboarc-hedgehog-in-cage-08-i01`
- `roboarc-horseshoe-ring-04-i01`
- `roboarc-interlocked-claws-04-i01`
- `roboarc-interlocked-claws-08-i01`
- `roboarc-interlocked-claws-10-i01`
- `roboarc-knot-untying-04-i01`
- `roboarc-lock-picking-02-i01`
- `roboarc-lock-picking-03-i01`
- `roboarc-rope-threading-01-i01`
- `roboarc-rope-threading-02-i01`

## Grasp release rules (T3) in V2: already stated in the released bases (checked 2026-10-16)

The owner asked that T3 (b2a9fc8233, Jianwen's local 1cbd0283d: `Session.grasp_rules()` lists the weld's slip force,
ALOHA 60 N / Robotiq 65 N over 20 ms, and the Robotiq 0.2 s contact loss) reach all 94 released bases, not only the new
ones. It already does, so no task text changes:
- T3 predates 0.1.0, and main's task-dir `rcb_topogym.py` (environment/, environment/sim/, tests/) is byte-identical to
  this branch's `scripts/roboarc/shared/rcb_topogym.py`: `/spec`'s `robot_spec()["grippers"][...]["command"]` and
  describe's `grasp_rules` state T3 at runtime.
- main's `scripts/roboarc/mk_tasks.py` rewrites each MuJoCo base's `Grasping:` / `- Grippers:` line from release.json
  `grasp_rules`, which release_rebuild reads in the release image, whatever the package texts say. So the 2-rule text
  left in the packages (`scripts/roboarc/packages/`, 32 of `tools/release/package_text/`) is never rendered, and
  the 10-15 keep-texts rule does not affect it.
- On main, all 156 weld instruction.md (78 bases) state T3: 66 Robotiq (four conditions), 90 ALOHA (three: ALOHA has no
  contact-loss rule, `WELD_CONTACT_LOST_S`). The other 32 are knot (`grasp_assist none`) and blanket (pbd_local_patch_v1),
  and their texts are unchanged.
- In the whole-wave dry run (host B `2026-10-15_roboarc_v2_finish/dry_all3/render`), the grasp lines of the 188 released dirs
  are byte-identical to main's. The new knot-06 / 09 dirs also state T3.
