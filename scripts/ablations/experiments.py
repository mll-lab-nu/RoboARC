#!/usr/bin/env python3
"""The ablation experiments E1..E5 (docs/ABLATIONS.md): tasks, conditions, modes, repeats, batch names.

    python3 scripts/ablations/experiments.py table              # the trials of every experiment, with the totals
    python3 scripts/ablations/experiments.py bases E1           # the runner's bases of E1 (<base>-<token>), one per line
    python3 scripts/ablations/experiments.py batches E1         # the batch names of E1's repeats, one per line
    python3 scripts/ablations/experiments.py field E1 <field>   # one setting: agent, model, effort, modes, run_id, repeats
    python3 scripts/ablations/experiments.py plan E1            # every trial: experiment, batch, repeat, base, condition, mode
    python3 scripts/ablations/experiments.py variants E1        # the variant dirs (tasks/<base>-<token>-<mode>)
    python3 scripts/ablations/experiments.py first E1           # the first task's base (the smoke's)
    python3 scripts/ablations/experiments.py dataset E1         # the Hugging Face dataset the results go to

Every trial runs at appearance seed 0. A repeat is a batch of its own (`ablation-e<n>-r<k>`), run with K=3 SEEDS=0 as
bash-runners/run_seed0.sh runs a batch: each task's one trial at seed 0, so the repeats of a task are three trials at
seed 0 under three batch names. Standard library only."""
from __future__ import annotations

import dataclasses
import json
import sys

# the 15 bases of the original information ablation (Standard failed where Privileged succeeded, 11-01 controls)
A15 = ("battery-replacement-01", "bed-bolt-01", "bed-bolt-02", "bed-bolt-04", "bed-bolt-05", "buzz-wire-3d-02",
       "buzz-wire-3d-06", "buzz-wire-3d-10", "caliper-sorting-01", "cast-coil-04", "cast-coil-07", "cast-coil-08",
       "cast-keyhole-05", "cast-keyhole-06", "hedgehog-in-cage-07")
E1_BASES = ("bed-bolt-03", "buzz-wire-3d-07", "cast-coil-06", "cast-keyhole-03", "cast-keyhole-04", "hedgehog-in-cage-06",
            "horseshoe-ring-02", "horseshoe-ring-03", "horseshoe-ring-04", "interlocked-claws-05", "interlocked-claws-11",
            "interlocked-claws-12", "interlocked-claws-13")
E3_BASES = ("blanket-flatten-06", "blanket-flatten-08", "blanket-fold-01", "blanket-fold-02", "blanket-fold-03",
            "blanket-fold-04", "blanket-fold-05", "fiber-bundling-02", "fiber-bundling-04", "fiber-bundling-05",
            "fiber-bundling-08", "fiber-bundling-09", "knot-untying-06", "knot-untying-09", "orbits-02", "rope-threading-06",
            "rope-threading-07", "rope-threading-08")
ASTRA = dict(agent="codex", model="gpt-6-astra", run_id="codex-gpt6_astra-high")
OPUS = dict(agent="claude-code", model="claude-opus-5.5", run_id="claude_code-opus_5_5-high")
TOTAL_TRIALS = 366
# where the ablations' results and variant checks are sent (bash-runners/ablations/send_results.sh, send_checks.sh): a
# public Hugging Face dataset of its own, <name>/<batch>/<bundle id>/; the main evaluation's stays RoboARC/RoboARC-outputs
RESULTS_DATASET = "RoboARC/RoboARC-ablation-outputs"


def full(short: str) -> str:
    return f"roboarc-{short}-i01"


def e4_bases() -> tuple[str, ...]:
    """E4's six puzzles, chosen by the E4 lane: scripts/ablations/cond_clearance.py BASES (full names)."""
    try:
        import cond_clearance
    except ModuleNotFoundError as exc:
        if exc.name == "cond_clearance":
            raise SystemExit("E4: scripts/ablations/cond_clearance.py is not on this branch yet") from None
        raise
    return tuple(cond_clearance.BASES)


@dataclasses.dataclass(frozen=True)
class Experiment:
    id: str
    purpose: str
    agent: str
    model: str                       # MODEL default (the value the runner logs); any spelling with the same run id works
    run_id: str                      # what scripts/harbor/run_agent_batch.sh derives from AGENT, MODEL and EFFORT
    conditions: tuple[str, ...]      # tokens of scripts/ablations/make_variants.py REGISTRY
    modes: tuple[str, ...]
    repeats: int
    effort: str = "high"

    def bases(self) -> tuple[str, ...]:
        if self.id == "E4":
            return e4_bases()
        return tuple(full(b) for b in {"E1": E1_BASES, "E2": A15, "E3": E3_BASES, "E5": A15}[self.id])

    def batches(self) -> list[str]:
        return [f"ablation-{self.id.lower()}-r{k}" for k in range(1, self.repeats + 1)]

    def runner_bases(self) -> list[str]:
        """The bases the batch runner is given: one per (task, condition), `<base>-<token>`."""
        return [f"{b}-{t}" for b in self.bases() for t in self.conditions]


EXPERIMENTS = {e.id: e for e in (
    Experiment("E1", "the information-access ablation on more puzzles", **ASTRA,
               conditions=("q1c00", "q1c10", "q1c0g", "q1c1g"), modes=("standard",), repeats=3),
    Experiment("E2", "the information-access ablation with a second agent", **OPUS,
               conditions=("q1c00", "q1c10", "q1c0g", "q1c1g"), modes=("standard",), repeats=2),
    Experiment("E3", "the information-access ablation on deformable tasks", **ASTRA,
               conditions=("q1c00", "cstate"), modes=("standard",), repeats=1),
    Experiment("E4", "relaxed clearances", **ASTRA,
               conditions=("clr2x", "clr4x"), modes=("privileged", "standard"), repeats=1),
    Experiment("E5", "observation realism", **ASTRA,
               conditions=("obsv1", "obsv2"), modes=("standard",), repeats=1),
)}


@dataclasses.dataclass(frozen=True)
class Variant:
    experiment: str
    base: str            # the released task's base, e.g. roboarc-bed-bolt-01-i01
    token: str           # the condition
    mode: str

    @property
    def task(self) -> str:
        return f"{self.base}-{self.token}-{self.mode}"


def variants(eid: str) -> list[Variant]:
    """Every variant dir experiment eid runs, in a fixed order (base, condition, mode)."""
    e = EXPERIMENTS[eid]
    return [Variant(eid, b, t, m) for b in e.bases() for t in e.conditions for m in e.modes]


def plan(eid: str) -> list[dict]:
    """Every trial of experiment eid: one per variant and repeat, all at seed 0."""
    e = EXPERIMENTS[eid]
    return [dict(experiment=eid, batch=batch, repeat=k, base=v.base, condition=v.token, mode=v.mode, task=v.task, seed=0)
            for k, batch in enumerate(e.batches(), 1) for v in variants(eid)]


def main(argv: list[str]) -> int:
    if argv[:1] == ["table"]:
        total = 0
        print("| experiment | agent | run id | tasks | conditions | modes | repeats | trials |\n|---|---|---|---|---|---|---|---|")
        for e in EXPERIMENTS.values():
            try:
                n_tasks, n = len(e.bases()), len(plan(e.id))
            except SystemExit as exc:
                print(f"| {e.id} | {e.agent} | {e.run_id} | ({exc}) | {', '.join(e.conditions)} | {', '.join(e.modes)} | {e.repeats} | - |")
                continue
            total += n
            print(f"| {e.id} | {e.agent} | {e.run_id} | {n_tasks} | {', '.join(e.conditions)} | {', '.join(e.modes)} | {e.repeats} | {n} |")
        print(f"\ntotal: {total} trials (planned: {TOTAL_TRIALS})")
        return 0
    if len(argv) >= 2 and argv[1] in EXPERIMENTS:
        e = EXPERIMENTS[argv[1]]
        if argv[0] == "bases":
            print("\n".join(e.runner_bases()))
            return 0
        if argv[0] == "dataset":
            print(RESULTS_DATASET)
            return 0
        if argv[0] == "first":
            print(e.bases()[0])
            return 0
        if argv[0] == "batches":
            print("\n".join(e.batches()))
            return 0
        if argv[0] == "variants":
            print("\n".join(v.task for v in variants(e.id)))
            return 0
        if argv[0] == "plan":
            for row in plan(e.id):
                print(json.dumps(row))
            return 0
        if argv[0] == "field" and len(argv) == 3:
            v = getattr(e, argv[2])
            print(" ".join(v) if isinstance(v, tuple) else v)
            return 0
    print(__doc__.split("\n\n")[1], file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
