# How to stress-test your tasks (a runbook)

For porters who have their tiers and references and now need step 2 (stress acceptance) and step 4's short stress check.
It covers the commands, the sizes, how to read the result and what to do when something breaks. The criteria and their
reasons are in [STRESS_VALIDATION.md](STRESS_VALIDATION.md) (criteria v2.9, frozen). Written 2026-10-11 by the
coordinator.

## 0. What you need

- **Code:** the head of the development branch. Pull it before every round, and run the whole round on one commit.
- **Python:** the RoboARC release image (built from `source/images/roboarc/Dockerfile` by
  `images/roboarc/build.sh`) with the checkout mounted, or any Python with `mujoco==3.14.0` and the image's packages.
  Run the commands below from the repository root.
  - The tools import `tools/stress/_env.py` first. It puts this checkout's `topogym` and
    `scripts/roboarc/shared` on `sys.path`, sets the numpy / OpenBLAS SIMD masking (so AVX2 and AVX-512 hosts give
    the same hashes), and pins every library to one thread.
- **CPU:** MuJoCo is single-threaded. Every worker is one process using one core. Size `--procs` to the free cores
  minus a margin, and run with `nice -n 10` on a shared host.

## 1. Lay out your inputs

The tools match an instance to its reference by the **name of the instance's directory** (the "base").

```
work/
  instances/<base>/instance.json      # one frozen instance per tier, made on the commit you test
  refs/<base>/privileged.npz          # that tier's reference action stream
```

- **Make each instance on the commit you test**, so the birth is from the current physics (`source/` first on the path):
  ```bash
  PYTHONPATH=source python scripts/roboarc/shared/rcb_topogym.py make-instance --task <task_id> --variation <variation> \
      --scene-seed 0 --out work/instances/<base>/instance.json
  ```
  For example, base `cast-keyhole-03-i01` holds the instance of `cast_keyhole` easy.
- **References:** copy or symlink your oracle's recording, `scripts/roboarc/oracles/<base>/privileged.npz`, to
  `work/refs/<base>/privileged.npz`.
  - It must succeed on that instance, and two fresh processes must end on the same state hash (step 1).
  - Record it with a hold tail, so the success window closes inside the stream.

## 2. Three commands

```bash
S=tools/stress
I="work/instances/*/instance.json"          # your tiers

# (a) one stream, with the observer: a reference, a nop, or a candidate someone sent you
python $S/replay.py --instance work/instances/<base>/instance.json --actions work/refs/<base>/privileged.npz \
    --out out/ref_<base>.json --hold 0 --trace out/ref_<base>.trace.json.gz
python $S/replay.py --instance work/instances/<base>/instance.json --actions nop:400 --out out/nop_<base>.json

# (b) self-check (minutes to ~1 h, mostly the reference replays): does the harness run, and are the references, nops and reproducers clean?
#     2,000 controls per worker. Never an acceptance.
nice -n 10 python $S/round.py run --self-check --instances $I --refs work/refs \
    --corpus $S/corpus/manifest.json --reproducers $S/corpus/reproducers --procs 8 --out out/selfcheck

# (c) the acceptance round (step 2): ~1,000 episodes per tier over all six strategies.
#     Pick a seed range of your own (e.g. 70000+) and never re-use it.
nice -n 10 python $S/round.py run --instances $I --refs work/refs \
    --corpus $S/corpus/manifest.json --reproducers $S/corpus/reproducers \
    --seed0 70000 --episodes 1000 --procs 24 --replay-workers 8 --out out/round_1
```

**Step 4's short stress check** on a new, redesigned tier is "the corpus + a 20k-control search". It is command (c)
with `--budget 3500` in place of `--episodes 1000`: six workers × 3,500 ≈ 21k controls.

**Long references:** the `prefix` strategy replays the reference up to a random row before it perturbs, so it is slow on
long routes. Name it several times to spread its share over more workers:
`--strategies random,directed,mutate,cem,periodic,prefix,prefix,prefix,prefix`.

**Families not in the corpus yet:** the corpus covers claws and fiber. `random`, `directed` and `cem` need no corpus;
`mutate` has nothing to mutate and stays small. That's fine. Keep passing `--corpus` and `--reproducers`: the reproducers
of shared mechanisms (G-, B-, C-, K-series) are matched by base and simply skipped for other families.

**Time:** a round takes roughly (episodes × average episode length) / procs. On ~24 cores, one 1,000-episode tier
usually takes 1-4 h. The references' fresh replays count too: each must stay under 180 min with the observer (the 3 h agent budget, user 2026-10-13), or under the base's own budget
where the owner excepted it (`scripts/task_conventions.py` CUTOFF_EXCEPTIONS; wire-splicing-01-i01: 360 min).

## 3. Read the result

`out/round_1/report.md` has one row per tier. Rebuild it any time with `python $S/round.py report out/round_1`.

| column | PASS needs |
|---|---|
| reference | success, CLEAN, the same hash in two fresh processes, ≤ 180 min (an excepted base: its budget) |
| nop | no success, CLEAN |
| corpus / reproducers | no BREAKDOWN, deterministic |
| confirmed BREAKDOWN **v2.12** | **0.** The search's candidates are replayed in fresh processes; only v2.12 decides |

Per-stream tables: `python $S/verdict.py out/round_1/results/confirm`.

- **BREAKDOWN:** the simulator broke. Examples: a fault or non-finite state; a contact deeper than 10 mm; energy created
  (X > 0.1 J); an uncaused speed jump; a stored-energy pop kicking a body by more than 3 m/s. One confirmed BREAKDOWN
  fails the tier.
- **VIOLENT:** violent but caused (thrown pieces, kN impacts, whips). Reported, not failed. Look at the trigger anyway.
- **CLEAN:** neither.

## 4. When a tier fails

1. **Shrink** the candidate to a short reproducer:
   `python $S/shrink.py --instance <instance.json> --actions <cand.npz> --match "<part of the fail reason>" --out repro.npz`
2. **Confirm** it twice with `replay.py`: the same verdict and the same state hash.
3. **Trace** it: run `replay.py --trace` and read the per-tick energy books and the contact pair categories. Find which
   contact pair, equality or limit did the work.
4. **Fix the mechanism, not the reading.** The criteria are frozen. Fixes we used, all in `topogym/world/mujoco/`:
   - **A thin or light body pushed into another:** give that pair its own stiffer contact with the profile field
     `pairs` (`scene.py`, see fiber's FB2 entry in `profiles.py`). Bodies that never meet in the references stay
     bit-identical.
   - **A degenerate contact between robot links that can't really touch:** exclude the pair. Examples: rule A, ALOHA
     pads vs the other finger's hull; the ALOHA forearm vs its own gripper base (landing).
   - **A rope or chain spun up until its connect rows pump energy:** the speed clamp (`knot_chain_v6`,
     `rope_chain_v2`, landing). It damps a link only above a threshold that no reference reaches, so references stay
     bit-identical.
   - **A robot-side force clamp does not help.** These events build within one 25 ms control step; see
     `MUJOCO_CORE.md`.
5. **Prove the fix:** the reproducer now reads non-BREAKDOWN in two fresh processes, every reference stays bit-identical
   (or is re-recorded), and the round re-runs. Add the reproducer to `corpus/reproducers/` (`.npz` plus a `.json` with
   base, family, mechanism, the commit it was found on and sha256), so every later round replays it.
6. If you are stuck, send the reproducer and its trace to the maintainers.

## 5. Optional: GPU screening

On a GPU you can screen many more episodes with an MJWarp port (a separate GPU-search branch, not in this repository) (exact diagonal
included). Its hits must be replayed on CPU with `replay.py`; only the CPU float64 result counts. We use this "plan B"
for the rope and cloth families: CPU 300 per tier, plus a GPU screen of ≥ 1,000 per tier with CPU confirmations.

## 6. Notes as of 2026-10-11

- **Robotiq 2F-85 grasp (changed 2026-10-11):** the spring links now keep the MJCF's springref 2.62 at every load
  (`robot.SPRING_REFS`, frozen scenes included; before: about 0, which curled the fingertips so the pads pinched with
  their tips and a weld carried buzz's wand with no finger contact for 41-65 % of its weld time), and a Robotiq grasp
  lets go once no finger has touched the welded body for 0.2 s (`grasp.WELD_CONTACT_LOST_S`, reason
  `finger_contact_lost`). Both apply to every Panda + Robotiq family (buzz, fiber, hedgehog; lock_picking once ported).
  - **Robotiq families:** once this is on the branch, rebuild your instances, re-validate or re-record your references
    on it, then run the acceptance round.
  - **ALOHA families:** unaffected; go ahead.
- **Physics in force:** exact constraint diagonal everywhere; rule A for ALOHA pads; claws welds at 2.5 ms (claws only).
  Rebuild instances and re-validate references on the current head before a round.
- Report results in your family's rows of the development repository's round tracker (not carried into this repository): round name, commit, seeds,
  episodes and the v2.12 verdict per tier.
