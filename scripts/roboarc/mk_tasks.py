#!/usr/bin/env python3
"""Render the RoboARC task dirs: tasks/roboarc-<family>-<NN>-i01-privileged and its standard twin -standard
(eai-standard/2.4), one pair per base in release.json. NN is the base's number in its family; the difficulty is
task.toml's `difficulty`.

usage: python3 scripts/roboarc/mk_tasks.py [--check] [--list] [--assets <dir>] [base ...]
       python3 scripts/roboarc/mk_tasks.py --dev <dev.json> --out <dir> base ...
  no base   every base in release.json
  --check   write nothing; exit 1 if a task would change, or if an identity check fails
  --list    print the bases with their evidence
  --assets  a local copy of the pinned HF dataset (the instances/<base>/ layout, scripts/roboarc/assets.py fetch): also
            check the frozen instances' bytes against the runtime bindings (without it, the pins in assets.json are
            compared)
  --dev     task dirs of a locally built image, written to --out (never tasks/; scripts/roboarc/harbor_check.py
            --instances): dev.json names the image ("image": reference, id; "runtime" and "uipc_builds" as its
            installed_runtime() reports them) and, per base, a local frozen instance directory ("instance") and the
            reference trajectory both modes replay ("reference")

A task dir is: the package's texts with their fields filled in, the pinned tooling and graders, the runtime binding,
the reference solution, then scripts/task_conventions.normalize (the protocol's shared sections) and the DIGEST.

Inputs (Python standard library only, plus scripts/task_conventions.normalize):
  packages/      per task dir, its texts as they ship (task.toml, README.md, instruction.md, the Dockerfiles and compose
                 files, tests/test.sh, environment/goal/*.png) before normalize. The values that come from another
                 source are fields, written {{NAME}} and filled by fields() below: the image (images/roboarc/VERSION), the
                 agent egress (scripts/set_agent_egress.py), the tooling version, the oracle level and every record of
                 release.json (reference solution, nop run, provenance, change log), the row cap and the standard
                 service's time per call. check_texts() holds the facts every text must state as the pinned tooling
                 and release.json have them. Privileged packages also hold the instance's lock
                 (runtime_binding.upstream.json).
  release.json   the release (simulator source, runtime, tooling) and, per base, where its frozen instance comes from
                 (ported_from: the Genesis-era base whose scene it ports; template: the base whose texts a new variation
                 starts from), its texts' facts (grasp_rules, physics_dt_s, step_s, chain) and the oracle runs that are
                 its evidence
  shared/        the pinned tooling rcb_topogym.py with render_bridge.py and nyx_direct.py, the graders, the sim service
                 adapter (server.py, replay.py, finalize.sh), the standard oracle player
  scripts/eai_standard/v2_4/ + scripts/eai_standard/   the eai-standard/2.4 layer, copied unchanged
  oracles/       the references (privileged.npz; standard.npz is the same file, or, where release.json's standard evidence
                 has "speed_rule", the reference re-timed to the standard service's joint-speed rule)
  assets.json    the frozen instances, not in git: the private HF dataset eai-v2-internal/robotangle-assets
                 (instances/<base>/), pinned by commit and sha256. The image carries them; a task keeps its
                 runtime_binding.json and stages its instance from the image (/opt/robotangle/stage_instance.py).
  roboarc_names.json   the Genesis-era base names -> RoboARC base, family, number, difficulty (a cross-check of the
                 packages' `difficulty`)
Nothing at run time imports repository code: the tooling and the sim layer are copies (PROTOCOL D9).
"""
from __future__ import annotations
import argparse, ast, hashlib, json, math, os, pathlib, re, shutil, sys, tempfile, tomllib, zipfile

BENCH = "roboarc"   # the benchmark's name: task prefix, image <BENCH>, scripts/<BENCH>/, docs/benchmarks/<BENCH>.md
HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
TASKS = ROOT / "tasks"
SHARED, PKG, ORC = HERE / "shared", HERE / "packages", HERE / "oracles"
sys.path.insert(0, str(ROOT / "scripts"))
import task_conventions as TC  # noqa: E402  the protocol's conventions: every generator ends with normalize(task_dir)

# The agent's time cutoff per mode (PROTOCOL §2.12): this benchmark's own number, written into every task's `[agent]
# timeout_sec`, the one source; normalize renders from it the instruction's closing `## Time limit`, the README row and
# the sim service's wall clock. To change it (an owner decision; the pull request states old -> new), edit it here and
# regenerate.
CUTOFF_SEC = 10800

REL = json.loads((HERE / "release.json").read_text())
ASSETS = json.loads((HERE / "assets.json").read_text())        # the pinned HF dataset (scripts/roboarc/assets.py)
NAMES = json.loads((HERE / "roboarc_names.json").read_text())   # owner's mapping: old base -> new base, family, number, difficulty
VERSION = (ROOT / "images" / BENCH / "VERSION").read_text().strip()
IMAGE = f"ghcr.io/embodied-first-exam/{BENCH}:{VERSION}"
EAI = ROOT / "scripts/eai_standard"
LAYER = {"environment/rcb_client.py": EAI / "v2_4/rcb_client.py", "environment/rcb1.py": EAI / "rcb1.py",
         "environment/client-requirements.txt": EAI / "client-requirements.txt",
         "environment/sim/rcb_limited.py": EAI / "v2_4/rcb_limited.py", "environment/sim/rcb1.py": EAI / "rcb1.py",
         "environment/sim/run.sh": EAI / "run.sh"}   # the eai-standard/2.4 layer, copied unchanged (check-tooling-identity.sh)
TEXTS = ("task.toml", "README.md", "instruction.md", "environment/Dockerfile", "environment/docker-compose.yaml",
         "tests/Dockerfile", "tests/docker-compose.yaml", "tests/test.sh")   # the package files rendered as text
GOAL_DIR = "environment/goal"   # a package's goal-state images (fold: the target shape is shown, never given as coordinates)
TIERS = ("very_easy", "easier", "easy", "medium", "hard")
MAX_STEPS = 200000        # rcb_topogym.MAX_STEPS (RCB_MAX_STEPS): the runaway guard on an episode's / a trajectory's length
MAX_CALL_WALL_S = 900     # v2_4/rcb_limited.MAX_CALL_WALL_S (RCB_MAX_CALL_WALL_S): no compose file overrides it
CHUNK_S, WATCH_S = 10, 30  # the sim service's RCB_MAX_CHUNK_S / RCB_MAX_WATCH_S defaults
BUILD_ALLOWANCE_S = 1800  # scene build, kernel compilation and video encoding inside the verifier's timeout
MUJOCO_ENV = (("NPY_DISABLE_CPU_FEATURES", "AVX512F,AVX512CD,AVX512_SKX,AVX512_CLX,AVX512_CNL,AVX512_ICL,AVX512_SPR"),
              ("OPENBLAS_CORETYPE", "Haswell"))   # every container that runs a Session (docs/simulator/MUJOCO_CORE.md)
sha = lambda b: hashlib.sha256(b).hexdigest()  # noqa: E731

# Agent egress (owner, 2026-10-10 / 10-11): the model APIs and the package registries, the two lines
# `scripts/set_agent_egress.py allowlist` writes, with that script's HOSTS (one list for every task:
# ci_checks/check-verifier-isolation.sh).
_HOSTS = re.search(r"(?ms)^HOSTS = \[(.*?)\]", (ROOT / "scripts/set_agent_egress.py").read_text()).group(1)
AGENT_EGRESS = ('network_mode = "allowlist"      # model APIs and package registries (scripts/set_agent_egress.py); PROTOCOL §2.11\n'
                "allowed_hosts = [" + ", ".join(f'"{h}"' for h in re.findall(r'"([^"]+)"', _HOSTS)) + "]")


def tooling_check() -> list[str]:
    """The tooling files in shared/ are the release's (release.json tooling_sha256, render_bridge_sha256,
    nyx_direct_sha256): a render or a check on any other copy is refused (2026-10-13: a rename had put an older
    rcb_topogym.py under the release, and the bindings were stamped with it)."""
    bad = []
    for key, name in (("tooling_sha256", "rcb_topogym.py"), ("render_bridge_sha256", "render_bridge.py"), ("nyx_direct_sha256", "nyx_direct.py")):
        f = SHARED / name
        got = sha(f.read_bytes()) if f.is_file() else "missing"
        if got != REL[key]:
            bad.append(f"scripts/{BENCH}/shared/{name}: sha256 {got[:16]}, release.json {key} {REL[key][:16]}")
    return bad


def tooling_constant(name: str):
    """A module-level constant of the shipped tooling (shared/rcb_topogym.py), read from its source: `NAME = <literal>`
    or `NAME = dict(k=<literal>, ...)`. The texts state what the tooling grades (owner, 2026-10-12: text and grader
    can never diverge)."""
    for node in ast.parse((SHARED / "rcb_topogym.py").read_text()).body:
        if isinstance(node, ast.Assign) and [getattr(t, "id", None) for t in node.targets] == [name]:
            v = node.value
            if isinstance(v, ast.Call) and getattr(v.func, "id", None) == "dict" and not v.args:
                return {k.arg: ast.literal_eval(k.value) for k in v.keywords}
            return ast.literal_eval(v)
    raise KeyError(f"shared/rcb_topogym.py has no constant {name}")


def direct_families() -> tuple[str, ...]:
    """The families whose tasks render with nyx_direct.py: the tooling's own settings (rcb_topogym.TASKS renderer
    "direct"), read from the copy this release ships. The tooling sets it in loops (`for _name in (...)`) and in single
    assignments (`TASKS["cast_coil"]["renderer"] = "direct"`); both forms are read, and the count of assignments in the
    file must equal the count understood here: a family missed ships without nyx_direct.py and its service dies at the
    first frame (2026-10-13: slotted_post_ring, orbits, then cast_coil and cast_keyhole)."""
    src = (SHARED / "rcb_topogym.py").read_text()
    loops = re.findall(r'(?ms)^for _name in \(([^)]*)\):\n\s+TASKS\[_name\]\["renderer"\] = "direct"', src)
    singles = re.findall(r'(?m)^TASKS\["([a-z0-9_]+)"\]\["renderer"\] = "direct"', src)
    total = len(re.findall(r'\["renderer"\] = "direct"', src))
    assert total == len(loops) + len(singles), f"rcb_topogym.py sets renderer \"direct\" {total} times, {len(loops) + len(singles)} understood"
    return tuple(n for loop in loops for n in re.findall(r'"([a-z0-9_]+)"', loop)) + tuple(singles)


CONTACT_RULE, BREAKDOWN_M, DIRECT = tooling_constant("CONTACT_RULE"), tooling_constant("BREAKDOWN_PENETRATION_M"), direct_families()
_SERVER = (SHARED / "standard" / "sim" / "server.py").read_text()   # the standard service's image budget per call
MAX_FRAMES, DEPTH_FRAME = (int(re.search(rf"^{k} = (\d+)", _SERVER, re.M).group(1)) for k in ("MAX_FRAMES", "DEPTH_FRAME"))


# ---------------------------------------------------------------- fields: the values a package takes from elsewhere
def step_time(instruction: str) -> float:
    return float(re.search(r"One control step takes about ([0-9.]+) s of wall clock", instruction).group(1))


def control_hz(base: str) -> int:
    return int(re.search(r"Control runs at (\d+) Hz", (PKG / f"{base}-privileged" / "instruction.md").read_text()).group(1))


def base_step(base: str) -> float:
    """The wall time per control step the base's texts state: the one measured on its reference's oracle run
    (release.json step_s), else its privileged instruction's."""
    rec = REL["bases"][base]
    if rec.get("step_s"):
        return float(rec["step_s"])
    return step_time((PKG / f"{base}-privileged" / "instruction.md").read_text())


def verifier_timeout(base: str) -> float:
    toml = (PKG / f"{base}-privileged" / "task.toml").read_text()
    return float(re.search(r"(?ms)^\[verifier\]\n.*?^timeout_sec = ([0-9.]+)", toml).group(1))


def row_cap(base: str) -> int:
    """The longest trajectory the privileged verifier accepts: its two replays must fit [verifier] timeout_sec at 1.5 x
    the step time the instruction states (the second replay also renders the video), after BUILD_ALLOWANCE_S; rounded
    down to 1000 rows, never above rcb_topogym's own guard. By the instructions' step times knot_untying (about 1.2 s per
    step) fits about 18000 rows twice into 45000 s, and a verifier that runs out of time leaves no reward (arithmetic
    from the stated step times, not a measured run)."""
    cap = min(MAX_STEPS, int((verifier_timeout(base) - BUILD_ALLOWANCE_S) / (3 * base_step(base))) // 1000 * 1000)
    traj = ORC / base / "privileged.npz"
    if traj.is_file():      # the reference trajectory must fit, with room
        with zipfile.ZipFile(traj) as z, z.open("actions.npy") as f:
            f.read(8); n = int.from_bytes(f.read(2), "little"); header = f.read(n).decode("latin1")
        rows = int(re.search(r"'shape': \((\d+),", header).group(1))
        assert 1.5 * rows <= cap, f"{base}: the reference trajectory has {rows} rows, the cap is {cap}"
    return cap


def time_per_call(step: float, hz: int) -> str:
    """The standard instruction's "Time per call" for eai-standard/2.4 (v2_4/rcb_limited.py call_begin / poll_stop / the
    operations endpoints, v2_4/rcb_client.py _post)."""
    chunk = CHUNK_S * hz * step
    fit = f"A `submit` of the maximum {CHUNK_S} s chunk takes about {chunk:.0f} s here"
    if 2 * chunk > MAX_CALL_WALL_S:
        fit += " (close to that limit on a slower host)"
    if 2 * WATCH_S * hz * step <= MAX_CALL_WALL_S:
        fit += f", and any watch up to {WATCH_S} s fits (both with a 2x margin)."
    else:
        n = int(MAX_CALL_WALL_S / (2 * step))
        fit += (f"; a `get_obs(seconds=...)` watch fits up to about {n} control steps ({n / hz:.2f} s, a 2x margin), and a "
                "longer one is cut short the same way.")
    return (
        "## Time per call\n"
        f"- One control step takes about {step:g} s of wall clock here. A request returns when all of its steps have run, or\n"
        f"  earlier: a call that has run for {MAX_CALL_WALL_S} s (`spec[\"max_call_wall_s\"]`) is stopped by the service at the next control\n"
        "  step and returns what it executed (`res.stopped` is `\"max_call_wall_s\"`; `res.meta[\"executed_rows\"]`, or\n"
        "  `res.meta[\"control_steps_watch\"]` for a watch, says how far it got; the robot holds its last target). Check\n"
        f"  `res.stopped` before you send the next rows. {fit}\n"
        "- The client waits longer than that (1800 s for `submit`, 1020 s for `get_obs`), so a client timeout or a connection\n"
        "  error means the connection or the service failed, not that the call is still on its way. Do NOT send the request\n"
        "  again blindly (it could run twice): `r.operation_status()` tells what became of the last one. Its `status` is\n"
        "  `completed` or `error` (with `rows_before` and `committed_rows`, the episode's control steps before and after it,\n"
        "  and `stopped`), `interrupted` (the service restarted during it and resumes the same episode in the state after the\n"
        "  last completed call: `spec[\"on_service_error\"]`, `r.wait_ready()`), or `admitted` (not finished); an HTTP 404 error\n"
        "  there means the service never admitted the request, so nothing ran. The service answers one request at a time, so\n"
        "  this query waits while a call is running: repeat it if it times out. If your client hangs up, the service stops\n"
        "  the call at the next control step. A request that arrives after its deadline is refused (408), and one whose\n"
        "  operation id was already admitted is refused (409); neither moves the robot. A control step that hangs for more\n"
        "  than 60 s past the call's limit ends the episode as an infrastructure error, not as your failure.\n")


def toml_str(s: str) -> str:
    return '"' + s.replace("\\", "\\\\").replace('"', '\\"') + '"'


def run_words(e: dict) -> str:
    """Where the evidence episode ran: the image build and the date, never a host or a job name (2026-10-13)."""
    return f"{e['image']} ({e['date']}; simulator source {e['source']}, tooling {e['tooling']})"


NO_REFERENCE = ("No oracle; human review in PR #N (PROTOCOL §2.7). No reference solution and no positive example yet: nothing "
                "is known to solve this task. Its physics was accepted on the family's acceptance rounds and a smoke stress "
                "round on this instance.")   # release.json oracle "none" (user 2026-10-13); main's wording for the first clause


def oracle_lines(base: str, rec: dict) -> str:
    """task.toml's oracle level: "full" with a passing oracle run in both modes (docs/benchmarks/roboarc.md), else "none"
    with its evidence (release.json oracle "none")."""
    if {"privileged", "standard"} <= set(rec["evidence"]):
        return 'oracle = "full"'
    assert rec.get("oracle") == "none", f"{base}: no oracle evidence in both modes and not oracle \"none\""
    return f'oracle = "none"\noracle_evidence = {toml_str(NO_REFERENCE)}'


NOT_CONFIRMED = {"speed_rule": "It is faster than the standard service's joint-speed rule",
                 "failed": "Its Harbor oracle trial did not score 1, and the cause is not established",
                 "not_run": "No Harbor oracle trial of it has been run yet",
                 "running": "Its Harbor oracle trial was still running when this release was rendered and is not confirmed"}


def ref_solution(mode: str, rec: dict) -> str:
    """README's Reference solution: what solution/ holds and the evidence run that scored it (release.json evidence)."""
    ev = rec["evidence"].get(mode)
    if not ev:
        assert rec.get("oracle") == "none", "every base has its evidence, or release.json oracle \"none\""
        return NO_REFERENCE + " `oracle = \"none\"`: there is no `solution/`, and Harbor checks the nop agent only."
    if ev.get("status", "passed") != "passed":     # a reference without a passing Harbor record
        why = NOT_CONFIRMED[ev["status"]]
        if mode == "privileged":
            return (f"`solution/solve.sh` copies `solution/trajectory.npz` (sha256 `{ev['trajectory_sha256'][:16]}…`) to "
                    f"`/app/output/`: the MuJoCo reference, which its author replayed to success with a reproducible end "
                    f"state outside Harbor. {why}, so no score is claimed here.")
        if ev["status"] == "speed_rule":       # 2026-10-13: the rope_threading rd references (888 to 1174 rows above the rule)
            return ("`solution/solve.sh` streams the recorded rows of the reference (`solution/oracle_qpos.npy`) through the "
                    "client, but **this is not a working oracle in standard mode**: the reference was recorded faster than the "
                    "standard service's joint-speed rule (0.04 rad per control step); replayed through the service "
                    "its rows are stretched over more control steps and the episode does not solve the task (the Harbor oracle "
                    "trial of rope-threading-rd-very-easy scored 0 for this reason). A re-timed reference follows. The "
                    "privileged twin's reference is the same rows and is graded without the speed rule.")
        over = (f" Standard oracle not finished; the reference has {ev['rows_over']} rows above the service's joint-speed rule (0.04 rad per "
                "control step) and may be stretched by the service." if ev.get("rows_over") else "")
        return ("`solution/solve.sh` streams the recorded rows of the reference (`solution/oracle_qpos.npy`, one "
                "`joint_position` row per control step, `hold`; the same rows as the privileged reference) through the "
                f"client (`solution/oracle_replay.py`). {why} on this standard service, so no score is claimed here.{over}")
    if mode == "privileged":
        route = ("this design's oracle, recorded on MuJoCo" if rec.get("template") or rec.get("design") else
                 "the released route re-recorded on MuJoCo with a held tail")
        return (f"`solution/solve.sh` copies `solution/trajectory.npz` (sha256 `{ev['trajectory_sha256'][:16]}…`) to "
                f"`/app/output/`: the MuJoCo reference ({route}), which the verifier scored 1 in two fresh-process "
                f"replays on {run_words(ev)}.")
    if ev.get("speed_rule"):     # 2026-10-14: a standard reference re-timed to the joint-speed rule (oracles/<base>/standard.npz)
        return ("`solution/solve.sh` streams the rows of the standard reference (`solution/oracle_qpos.npy`, one "
                "`joint_position` row per control step, `hold`) through the client (`solution/oracle_replay.py`): "
                f"{ev['speed_rule']} That episode scored 1 (live and fresh replay, end state `{ev['end_state']}`) on "
                f"{run_words(ev)}.")
    return ("`solution/solve.sh` streams the recorded rows of the standard-mode oracle episode "
            "(`solution/oracle_qpos.npy`, one `joint_position` row per 25 ms control step, `hold`; the same rows as the "
            "privileged reference) through the client (`solution/oracle_replay.py`). That episode scored 1 (live and "
            f"fresh replay) on {run_words(ev)}.")


def nop_words(base: str, mode: str, rec: dict) -> str:
    n = rec.get("nop")
    if n and n.get("status", "passed") != "passed":
        return f"`nop`: no record yet on this base (`harbor run -p tasks/{BENCH}-{base}-{mode} -a nop`)."
    if n:
        return f"`nop` scored 0 on this base in both modes ({n['run']}; {n['date']})."
    return f"`nop`: pending (`harbor run -p tasks/{BENCH}-{base}-{mode} -a nop`, docs/benchmarks/{BENCH}.md)."


def provenance(base: str, rec: dict) -> str:
    rows = [f"- Frozen instance: a MuJoCo instance (`simulator.physics_engine` \"mujoco\", its scene frozen under "
            f"`instance_assets/mujoco/`), built by TopoGym source `{rec['creation_source']}` ({rec['instance_package']}); "
            f"bound to this release's runtime (source `{REL['runtime']['source']['sha256'][:8]}`, tooling "
            f"`{REL['tooling_sha256'][:8]}`).",
            f"- Frozen instance files (`instance.json`, `instance_assets/`): not in this directory. They are in the private HF "
            f"dataset `{ASSETS['repo']}` at `{ASSETS['revision'][:12]}`, under `instances/{base}/` "
            "(scripts/roboarc/assets.json pins each file's sha256). The image carries them, and "
            "`/opt/robotangle/stage_instance.py` copies this task's instance next to its `runtime_binding.json`, "
            "which names the instance's canonical sha256 and every asset's sha256. The tooling checks them again at "
            "every session start.",
            f"- Release: RoboARC {VERSION}, image `{IMAGE}` (images/{BENCH}/); simulator source `{rec['release_commit'][:9]}` (PROVENANCE.md).",
            "- Name: RoboARC is the published name of this benchmark (2026-10-13). Code identifiers keep the simulator's name: the "
            "simulator package `topogym`, the tooling `rcb_topogym.py`, `TOPOGYM_*` variables."]
    return "\n".join(rows)


PORT_TASK = ("- {date}: new task of {bench} (RoboARC on MuJoCo 3.14.0, docs/benchmarks/{bench}.md): the scene, goal "
             "and rules of the Genesis 1.2.1 task `{old}` ({mode} mode), ported to MuJoCo: its own frozen "
             "instance and reference, the numpy / OpenBLAS SIMD masking in every container that runs a Session, and the "
             "render bridge next to the tooling (Nyx frames).")
NEW_TASK = ("- {date}: new task of {bench} (RoboARC on MuJoCo 3.14.0, docs/benchmarks/{bench}.md): a new "
            "design of {family}{design}. Its task files start from the Genesis 1.2.1 task `{template}` ({mode} mode), with this design's "
            "own task text, rules and goal, its own frozen instance and its own reference.")
# The privileged grading rule (2026-10-14, failure analysis item 12): success needs both grading replays to succeed AND at
# least one of them to reach the goal with the instance's own thresholds within the agent's own rows (shared/privileged/
# grade.py, `strict_replay_required` 1). The families with a grading hold state it in their instruction (HOLD_STRICT).
HOLD_STRICT = ("Success needs both replays to succeed AND at least one of them to reach the goal with the thresholds above "
               "within your own rows (not only in the hold")
FIX_STRICT_GOAL = ("- 2026-10-14: the grading rule (tests/grade.py; failure analysis 2026-10-14, item 12). Success needs both replays to "
                   "succeed AND at least one of them to reach the goal with the instance's own thresholds within the trajectory's rows "
                   "(`strict_replay_required` 1; `success_in_hold_only` 1 marks a trajectory whose replays succeeded only in the hold, "
                   "not credited). From 2026-10-05 (R-hold) to this change a replay could also confirm within the hold, with its "
                   "relaxed thresholds, and no strict goal was required: privileged mode has no graded live run, so the hold's "
                   "thresholds were the effective goal. The frozen instance, the reference and the replay are unchanged; results "
                   "graded before this change are re-graded from their recorded trajectories, nothing is re-run "
                   "(scripts/roboarc/regrade.py). Runs: docs/benchmarks/roboarc.md (Grading replays).")
FIX_STRICT_GOAL_NO_HOLD = ("- 2026-10-14: tests/grade.py, the same file in every privileged task, requires again that one grading replay "
                           "reach the goal with the instance's own thresholds within the trajectory's rows (failure analysis "
                           "2026-10-14, item 12; docs/benchmarks/roboarc.md, Grading replays). This family has no grading hold: a "
                           "replay succeeds only within the rows, so every verdict is unchanged. reward.json gains "
                           "`success_in_hold_only` (always 0 here), and `strict_replay_required` is 1.")


def change_log(base: str, mode: str, rec: dict) -> str:
    """README's Change Log (PROTOCOL §2.8): the entry that made the task, the 2026-10-14 grading fix (privileged: every
    tests/grade.py changed), then release.json's later changes."""
    date = rec.get("released") or rec["nop"]["date"]   # a base released before keeps its date
    if rec.get("template"):
        log = [NEW_TASK.format(date=date, bench=BENCH, family=rec["family"], template=rec["template"], mode=mode,
                               design=f": {rec['design']}" if rec.get("design") else "")]
    else:
        log = [PORT_TASK.format(date=date, bench=BENCH, old=rec["ported_from"], mode=mode)]
    if mode == "privileged":
        hold = HOLD_STRICT in (PKG / f"{base}-privileged" / "instruction.md").read_text()
        log.append(FIX_STRICT_GOAL if hold else FIX_STRICT_GOAL_NO_HOLD)
    log += [f"- {c['date']}: {c['text']}" for c in rec.get("changes", ()) if c["mode"] in (mode, "both")]
    return "\n".join(log)


def fields(base: str, mode: str, rec: dict) -> dict[str, str]:
    """The values of a package's {{NAME}} fields for one task dir."""
    f = {"IMAGE": IMAGE, "AGENT_EGRESS": AGENT_EGRESS, "TOOLING_VERSION": REL["tooling_version"],
         "ORACLE": oracle_lines(base, rec), "REFERENCE_SOLUTION": ref_solution(mode, rec), "NOP": nop_words(base, mode, rec),
         "PROVENANCE": provenance(base, rec), "CHANGE_LOG": change_log(base, mode, rec)}
    if mode == "privileged":
        cap = row_cap(base)
        f.update(ROW_CAP=str(cap), ROW_CAP_MIN=f"{cap / control_hz(base) / 60:.1f}", VERIFIER_BUDGET_S=f"{verifier_timeout(base):.0f}")
    else:
        f["TIME_PER_CALL"] = time_per_call(base_step(base), control_hz(base))
    return f


FIELD_RX = re.compile(r"\{\{([A-Z_]+)\}\}")


def fill(s: str, values: dict[str, str], where: str) -> str:
    unknown = sorted(set(FIELD_RX.findall(s)) - set(values))
    assert not unknown, f"{where}: unknown fields {unknown}"
    return FIELD_RX.sub(lambda m: values[m[1]], s)


# ---------------------------------------------------------------- checks: what every text must state
# Words that must never come back into a text: the table height and the robot base's position (user 2026-10-13: "in the
# wild nobody tells you how tall the table is"), dated decision notes, the R-hold rule (2026-10-05 .. 10-14: a replay
# could confirm within the hold, with its relaxed thresholds), Genesis-era GPU physics comments, a standard clock that
# starts with the service (the service's clock starts at the agent's first request, server.py touch()).
FORBIDDEN = {"README.md": ("robot base at", "table top at z", "(user 20"),
             "instruction.md": ("robot base at", "table top at z", "(user 20"),
             "docker-compose.yaml": ("CUDA physics", "TopoGym runs its physics")}
FORBIDDEN_PRIVILEGED = ("`strict_goal_reached`, not required", "each of them may complete its confirmation within that hold",
                        "every replay may confirm\nwithin that hold")
FORBIDDEN_STANDARD = ("service comes up", "since the service first started")
CONTACT_RX = re.compile(r"A success with deep interpenetration \(a task contact deeper than (?P<deep>[0-9.]+) mm, or deeper than "
                        r"(?P<sus>[0-9.]+) mm for at least (?P<win>[0-9.]+) s: (?P<ticks>\d+) consecutive physics ticks; for a "
                        r"finger contact with a held piece (?P<hdeep>[0-9.]+) mm, or (?P<hsus>[0-9.]+) mm for (?P<hwin>[0-9.]+) s\)")
BREAKDOWN_RX = re.compile(r"task contact deeper than ([0-9.]+) mm(?:\)|: the episode| is a simulator breakdown)")
GRASP_RX = re.compile(r"(?m)^(?:- )?(?:Grasping|Grippers): ((?:assisted grasping \(|physical grasping)[^\n]*)\.$")


def check_texts(base: str, mode: str, rec: dict, texts: dict[str, str]) -> None:
    """The rendered texts (before normalize) against the facts they state: the tooling's contact rule and breakdown
    depth, release.json's grasping rule and step time, the difficulty, the standard service's image budget, the family
    the grader and the sim service run, the files the tooling needs (nyx_direct.py for its families), one GPU without
    `runtime: nvidia`, and none of the FORBIDDEN words."""
    where = f"{BENCH}-{base}-{mode}"
    mm = lambda v: f"{v * 1000:g}"  # noqa: E731
    for name, s in texts.items():
        bad = [w for w in FORBIDDEN.get(name.split("/")[-1], ()) + (FORBIDDEN_PRIVILEGED if mode == "privileged" and name.endswith(".md")
               else FORBIDDEN_STANDARD if name == "instruction.md" else ()) if w in s]
        assert not bad, f"{where}/{name}: says {bad}"
        if name.endswith(".md"):
            for m in CONTACT_RX.finditer(s):     # owner decision 2026-10-12: the texts state what the tooling grades
                c = CONTACT_RULE
                assert (m["deep"], m["sus"], m["hdeep"], m["hsus"]) == tuple(mm(c[k]) for k in ("deep_m", "sustained_m", "held_deep_m", "held_sustained_m")) \
                    and float(m["win"]) == float(m["hwin"]) == c["window_s"], f"{where}/{name}: the contact rule is not the tooling's"
                if rec.get("physics_dt_s"):
                    assert int(m["ticks"]) == max(1, math.ceil(c["window_s"] / float(rec["physics_dt_s"]) - 1e-12)), f"{where}/{name}: contact ticks"
                assert all(d == mm(BREAKDOWN_M) for d in BREAKDOWN_RX.findall(s)), f"{where}/{name}: breakdown depth"
            if rec.get("grasp_rules"):
                assert all(g == rec["grasp_rules"] for g in GRASP_RX.findall(s)), f"{where}/{name}: not release.json's grasping rule"
    ins = texts["instruction.md"]
    if rec.get("step_s"):
        assert step_time(ins) == float(rec["step_s"]), f"{where}: the step time is not release.json's step_s"
    if mode == "standard":
        assert f"One call returns at most {MAX_FRAMES} images" in ins and f"`depth=\"frames\"` counts {DEPTH_FRAME}" in ins, \
            f"{where}: the image budget is not server.py's"
    # the task's own declarations, never taken from the committed task dir by normalize (a render depends on its inputs only)
    assert re.search(r'(?m)^dynamics = "(?:static|dynamic)"', texts["task.toml"]), f"{where}: task.toml declares no dynamics"
    d = re.search(r'(?m)^difficulty = "([^"]*)"', texts["task.toml"])
    assert d and d[1] in TIERS, f"{where}: task.toml has no difficulty of {TIERS}"
    named = [v["difficulty"] for v in NAMES.values() if v["new"] == f"{BENCH}-{base}"]
    assert named in ([], [d[1]]), f"{where}: difficulty {d[1]!r}, roboarc_names.json {named}"
    fam = rec["family"]
    assert re.findall(r"--family (\S+)", texts["tests/test.sh"]) in ([], [fam]), f"{where}: tests/test.sh grades another family"
    for name, s in texts.items():
        if name.endswith("docker-compose.yaml"):
            assert set(re.findall(r"RCB_TASK: (\S+)", s)) <= {fam}, f"{where}/{name}: the sim service runs another family"
            assert "count: 1" in s and "capabilities: [gpu]" in s and "NVIDIA_DRIVER_CAPABILITIES: all" in s \
                and not re.search(r"(?m)^\s*runtime: nvidia$", s), f"{where}/{name}: one GPU, no `runtime: nvidia`"
        if name in ("environment/Dockerfile", "tests/Dockerfile") and mode == "privileged":
            assert ("nyx_direct.py" in s) == (fam in DIRECT) and "render_bridge.py" in s, f"{where}/{name}: tooling files"
            assert "ENV " + " ".join(f"{k}={v}" for k, v in MUJOCO_ENV) + "\n" in s, f"{where}/{name}: no SIMD masking"
        if name == "environment/docker-compose.yaml" and mode == "standard":
            assert all(f"      {k}: {v}\n" in s for k, v in MUJOCO_ENV), f"{where}/{name}: the sim service has no SIMD masking"


# ---------------------------------------------------------------- the task dir
def binding(base: str, inst_dir: pathlib.Path | None, direct: bool) -> bytes:
    """The task's runtime binding: the upstream lock (instance and asset hashes) with this release's tooling and runtime,
    the render bridge next to the tooling (render_bridge_sha256) and, for a family that renders with nyx_direct.py, that
    file (nyx_direct_sha256): the tooling checks both at Session start. The frozen instance must be the pinned one in
    assets.json; with inst_dir (a local copy), its bytes are checked too."""
    lock = json.loads((PKG / f"{base}-privileged" / "runtime_binding.upstream.json").read_text())
    pinned = ASSETS["instances"].get(base, {}).get("instance_sha256")
    assert lock["instance_sha256"] == pinned, f"{base}: the bound instance is not the one pinned in assets.json"
    prefix = f"instances/{base}/"
    pinned_assets = {k[len(prefix):]: v for k, v in ASSETS["files"].items() if k.startswith(prefix + "instance_assets/")}
    # Frozen generated assets (image 0.2.22: knot_untying's rope volume meshes, rcb_topogym._install_frozen_assets), added to
    # the instance after the upstream lock: each is a generated file whose sha256 the instance's asset_identity records.
    frozen = {k: v for k, v in pinned_assets.items() if k.startswith("instance_assets/frozen_assets/")}
    lock["asset_files"] = {**lock["asset_files"], **frozen}
    assert lock["asset_files"] == pinned_assets, f"{base}: instance_assets pinned in assets.json differ from the binding"
    if inst_dir is not None and frozen:
        identity = json.loads((inst_dir / "instance.json").read_text()).get("asset_identity") or {}
        for rel, want in frozen.items():
            assert identity.get(pathlib.PurePosixPath(rel).name) == want, f"{base}: {rel} is not the generated asset it records"
    if inst_dir is not None:
        canon = json.dumps(json.loads((inst_dir / "instance.json").read_text()), sort_keys=True, separators=(",", ":"), allow_nan=False)
        assert lock["instance_sha256"] == sha(canon.encode()), f"{base}: instance.json is not the bound instance"
        assets = {p.relative_to(inst_dir).as_posix(): sha(p.read_bytes()) for p in sorted((inst_dir / "instance_assets").rglob("*")) if p.is_file()}
        assert lock["asset_files"] == assets, f"{base}: instance_assets differ from the binding"
    lock["tooling_sha256"] = sha((SHARED / "rcb_topogym.py").read_bytes())
    lock["runtime"] = REL["runtime"]
    # both libuipc builds the tooling chooses from by the GPU's compute capability (rcb_topogym.select_uipc_build), by their
    # uipc-build.json sha256; the runtime's own uipc_build_sha256 stays the audited build, the image's default
    lock["uipc_builds"] = {name: b["uipc_build_sha256"] for name, b in sorted(REL["uipc_builds"].items())}
    lock["render_bridge_sha256"] = sha((SHARED / "render_bridge.py").read_bytes())
    if direct:
        lock["nyx_direct_sha256"] = sha((SHARED / "nyx_direct.py").read_bytes())
    return (json.dumps(lock, indent=2) + "\n").encode()


def dev_binding(inst_dir: pathlib.Path, dev: dict, direct: bool) -> bytes:
    """--dev: the binding of a local frozen instance (canonical sha256, every asset's sha256) to the dev image's runtime
    and identity, with the same keys as binding()."""
    inst = json.loads((inst_dir / "instance.json").read_text())
    canon = json.dumps(inst, sort_keys=True, separators=(",", ":"), allow_nan=False)
    assets = {p.relative_to(inst_dir).as_posix(): sha(p.read_bytes()) for p in sorted((inst_dir / "instance_assets").rglob("*"))
              if p.is_file() and "__pycache__" not in p.parts}
    lock = dict(schema_version=1, instance_sha256=sha(canon.encode()), asset_files=assets,
                tooling_sha256=sha((SHARED / "rcb_topogym.py").read_bytes()), creation_source=inst["topogym_source"],
                runtime=dev["runtime"], image=dev["image"], uipc_builds=dev["uipc_builds"],
                render_bridge_sha256=sha((SHARED / "render_bridge.py").read_bytes()))
    if direct:
        lock["nyx_direct_sha256"] = sha((SHARED / "nyx_direct.py").read_bytes())
    return (json.dumps(lock, indent=2) + "\n").encode()


def put(dst: pathlib.Path, data: bytes, exe: bool = False) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_bytes(data)
    os.chmod(dst, 0o755 if exe else 0o644)


def goal_images(up: pathlib.Path, t: pathlib.Path, instruction: str, dockerfile: str) -> None:
    """A package's goal-state images: copied into the agent container at /app/goal/ (both modes); the instruction must
    name each one (an image nobody points to is a mistake)."""
    goal = up / GOAL_DIR
    if not goal.is_dir():
        assert "COPY goal/" not in dockerfile, f"{up.name}: environment/Dockerfile copies goal/ but there is none"
        return
    files = sorted(p for p in goal.iterdir() if p.is_file())
    assert files and dockerfile.count("COPY goal/ /app/goal/\n") == 1, f"{up.name}: {GOAL_DIR} and its COPY line"
    for p in files:
        assert f"/app/goal/{p.name}" in instruction, f"{up.name}: instruction.md does not name /app/goal/{p.name}"
        put(t / GOAL_DIR / p.name, p.read_bytes())


def render(base: str, out: pathlib.Path, inst_src: pathlib.Path | None = None, dev: dict | None = None) -> list[pathlib.Path]:
    rec = REL["bases"][base]
    assert rec.get("physics_engine") == "mujoco", f"{base}: every base of {BENCH} is a MuJoCo base (release.json physics_engine)"
    direct = rec["family"] in DIRECT
    lock = binding(base, inst_src, direct) if dev is None else dev_binding(inst_src, dev, direct)
    reference = pathlib.Path(dev["bases"][base]["reference"]) if dev else None
    tool = (SHARED / "rcb_topogym.py").read_bytes()
    extra = [("render_bridge.py", (SHARED / "render_bridge.py").read_bytes())] + \
            ([("nyx_direct.py", (SHARED / "nyx_direct.py").read_bytes())] if direct else [])
    dirs = []
    for mode in ("privileged", "standard"):
        t = out / f"{BENCH}-{base}-{mode}"; up = PKG / f"{base}-{mode}"; dirs.append(t)
        values = fields(base, mode, rec)
        texts = {n: fill((up / n).read_text(), values, f"packages/{up.name}/{n}") for n in TEXTS if (up / n).is_file()}
        texts["task.toml"] = TC.set_agent_timeout(texts["task.toml"], CUTOFF_SEC, base)
        check_texts(base, mode, rec, texts)
        for n, s in texts.items():
            put(t / n, s.encode(), exe=n.endswith(".sh"))
        goal_images(up, t, texts["instruction.md"], texts["environment/Dockerfile"])
        if mode == "privileged":
            for d in ("environment", "tests"):
                put(t / d / "runtime_binding.json", lock)
                put(t / d / "rcb_topogym.py", tool)
                for n, b in extra:
                    put(t / d / n, b)
            put(t / "tests/grade.py", (SHARED / "privileged/grade.py").read_bytes())
            traj = reference or ORC / base / "privileged.npz"
            if traj.is_file():
                put(t / "solution/trajectory.npz", traj.read_bytes())
                put(t / "solution/solve.sh", (
                    "#!/bin/bash\n# Reference solution (privileged): the trajectory the author's oracle produced and the verifier scored 1\n"
                    "# on the evidence run named in README.md (Reference solution). Uses nothing but the recorded file.\n"
                    "set -euo pipefail\nmkdir -p /app/output\ncp /solution/trajectory.npz /app/output/trajectory.npz\n").encode(), exe=True)
        else:
            for rel, src in LAYER.items():
                put(t / rel, src.read_bytes())
            for n in ("server.py", "replay.py", "finalize.sh"):
                put(t / "environment/sim" / n, (SHARED / "standard/sim" / n).read_bytes())
            put(t / "environment/sim/runtime_binding.json", lock)
            put(t / "environment/sim/rcb_topogym.py", tool)
            for n, b in extra:
                put(t / "environment/sim" / n, b)
            put(t / "tests/grade.py", (SHARED / "standard/grade.py").read_bytes())
            put(t / "tests/runtime_binding.json", lock)
            traj = reference or ORC / base / "standard.npz"
            if traj.is_file():
                with zipfile.ZipFile(traj) as z:
                    put(t / "solution/oracle_qpos.npy", z.read("actions.npy"))
                put(t / "solution/oracle_meta.json", (json.dumps({"K": 1, "interpolate": "hold", "rows": "the recorded "
                    "joint_position rows of the evidence episode (README.md, Reference solution)"}, indent=1) + "\n").encode())
                put(t / "solution/oracle_replay.py", (SHARED / "standard/solution/oracle_replay.py").read_bytes())
                put(t / "solution/solve.sh", (SHARED / "standard/solution/solve.sh").read_bytes(), exe=True)
    for t in dirs:
        TC.normalize(t)          # leaves the layer alone: scripts/eai_standard/ is a shared source (task_conventions)
        if main_cpus(t):
            TC.normalize(t)      # the texts rendered from [environment] (the resource paragraph) follow the applied cap
    return dirs


def main_cpus(task: pathlib.Path) -> bool:
    """`[environment].cpus` = `[metadata.resources.caps].main.cpus` (scripts/task_resources.toml: the task classes' own
    values, rigid / rope-chain / cloth). normalize sets `memory_mb` from the same cap; Harbor applies both to the agent's
    container, and check-resources wants the declared cap to be the applied one."""
    p = task / "task.toml"; s = p.read_text()
    main = ((tomllib.loads(s).get("metadata", {}).get("resources") or {}).get("caps") or {}).get("main") or {}
    if not main.get("cpus"):
        return False
    want = f"{float(main['cpus']):g}"
    new = re.sub(r"(?ms)(^\[environment\]\n.*?^cpus\s*=\s*)[0-9.]+", lambda m: m.group(1) + want, s, count=1)
    if new != s:
        p.write_text(new)
    return new != s


def snapshot(d: pathlib.Path) -> dict:
    return {p.relative_to(d).as_posix(): (p.read_bytes(), 0o755 if os.stat(p).st_mode & 0o100 else 0o644)   # exec bit only
            for p in sorted(d.rglob("*")) if p.is_file()}


def dev_main(a) -> int:
    """--dev: render the named bases from dev.json into --out (local only; never tasks/, nothing checked against
    release.json's instance pins). The texts name the dev image."""
    global IMAGE
    if a.out is None or not a.bases or a.check or a.assets:
        raise SystemExit("--dev needs --out and the bases, and takes neither --check nor --assets")
    dev = json.loads(a.dev.read_text())
    resolve = lambda v: (a.dev.parent / v).resolve()  # noqa: E731  (dev.json paths: absolute, or relative to dev.json)
    dev["bases"] = {b: {k: str(resolve(v)) for k, v in r.items()} for b, r in dev["bases"].items()}
    IMAGE = dev["image"]["reference"]
    a.out.mkdir(parents=True, exist_ok=True)
    for b in a.bases:
        for m in ("privileged", "standard"):
            shutil.rmtree(a.out / f"{BENCH}-{b}-{m}", ignore_errors=True)
        for t in render(b, a.out, pathlib.Path(dev["bases"][b]["instance"]), dev):
            print(f"dev task {t}")
    return 0


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true"); ap.add_argument("--list", action="store_true")
    ap.add_argument("--assets", type=pathlib.Path, help="a local copy of the pinned HF dataset (assets.py fetch)")
    ap.add_argument("--dev", type=pathlib.Path, help="dev.json: task dirs of a locally built image (see usage)")
    ap.add_argument("--out", type=pathlib.Path, help="with --dev: where the dev task dirs go")
    ap.add_argument("bases", nargs="*")
    a = ap.parse_args(argv)
    if a.dev is not None:
        return dev_main(a)
    bases = a.bases or sorted(REL["bases"])
    if a.list:
        for b in bases:
            r = REL["bases"][b]
            ev = "  ".join(f"{m}: {e.get('status', 'passed')}" + (f" ({e['date']}, source {e['source']})" if "date" in e else "")
                           for m, e in r["evidence"].items())
            print(f"{b:34s} instance {r['creation_source']}  " + (ev or "no oracle"))
        return 0
    bad = tooling_check()
    if bad and not a.check:
        print("\n".join(bad), file=sys.stderr); return 1
    with tempfile.TemporaryDirectory() as tmp:
        tmp = pathlib.Path(tmp)
        for b in bases:
            inst = a.assets / "instances" / b if a.assets else None
            if inst is not None and not (inst / "instance.json").is_file():
                bad.append(f"{b}: no frozen instance at {inst}"); continue
            for t in render(b, tmp / "out", inst):
                cur = TASKS / t.name
                if a.check:
                    if not cur.is_dir() or snapshot(cur) != snapshot(t):
                        bad.append(f"tasks/{t.name}: differs from a fresh render (run scripts/roboarc/mk_tasks.py)")
                else:
                    if cur.exists():
                        shutil.rmtree(cur)
                    shutil.copytree(t, cur, symlinks=False)
        known = {f"{BENCH}-{b}-{m}" for b in REL["bases"] for m in ("privileged", "standard")}
        stray = sorted(p.name for p in TASKS.glob(f"{BENCH}-*") if p.name not in known)
        bad += [f"tasks/{n}: not in scripts/roboarc/release.json" for n in stray]
        tools = {sha(p.read_bytes()) for p in TASKS.glob(f"{BENCH}-*/**/rcb_topogym.py")}
        if tools - {sha((SHARED / "rcb_topogym.py").read_bytes())}:
            bad.append("a task's rcb_topogym.py differs from scripts/roboarc/shared/rcb_topogym.py")
    for b in bad:
        print("FAIL", b)
    if not bad:
        print(f"ok {BENCH}: {2 * len(bases)} task dirs {'match a fresh render' if a.check else 'rendered'} "
              f"(release {REL['release_commit'][:9]}, image {IMAGE})")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
