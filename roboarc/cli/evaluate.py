"""roboarc eval: an evaluation batch with a coding agent. A wrapper of the batch runner,
scripts/harbor/run_agent_batch.sh: each option sets the runner's variable of the same meaning (`--usage-gateway` also
GATEWAY_RAW=0, the evaluation runs' value, when the caller has not set it), every other variable of the caller's
environment reaches the runner unchanged, and `--lanes` starts that many runners on the batch. Standard library only."""
from __future__ import annotations

import argparse
import os
import signal
import sys

from . import _common as C

RUNNER = "scripts/harbor/run_agent_batch.sh"
GATEWAY = "scripts/usage_gateway"
MODES = ("privileged", "standard", "standard-realtime")     # the runner's; RoboARC ships the first two
TRIALS = 3                                                  # the evaluation protocol: three trials per task and mode


def settings(a) -> dict:
    """The runner's variables this command sets: an option given wins, else the caller's environment stands (and is
    not repeated here), else the command's own default where it has one."""
    env = os.environ
    out: dict = {}
    for flag, name in (("agent", "AGENT"), ("model", "MODEL"), ("effort", "EFFORT")):
        value = getattr(a, flag)
        if value:
            out[name] = value
        elif not env.get(name):
            raise C.CliError(f"eval: --{flag} is required (or {name} in the environment)")
    if a.modes:
        modes = C.csv(a.modes)
        bad = [m for m in modes if m not in MODES]
        if bad or not modes:
            raise C.CliError(f"eval: --modes takes {', '.join(MODES[:2])} (comma-separated); got {a.modes}")
        out["MODES"] = " ".join(modes)
    if a.trials is not None:
        out["K"] = a.trials
    elif not env.get("K"):
        out["K"] = TRIALS
    if a.seeds:
        seeds = C.csv(a.seeds)
        if not all(s.isdigit() for s in seeds):
            raise C.CliError(f"eval: --seeds takes seeds, comma-separated, e.g. 0 or 0,1,2; got {a.seeds}")
        out["SEEDS"] = " ".join(seeds)
    if a.gpus:
        gpus = C.csv(a.gpus)
        if not all(g.isdigit() for g in gpus):
            raise C.CliError(f"eval: --gpus takes GPU indices, comma-separated, e.g. 0,1; got {a.gpus}")
        out["GPUS"] = ",".join(gpus)
    for flag, name in (("env_file", "ENV_FILE"), ("jobs", "JOBS"), ("results", "RESULTS")):
        if getattr(a, flag):
            out[name] = getattr(a, flag)
    if a.usage_gateway is True:
        out["USAGE_GATEWAY"] = str(C.repo() / GATEWAY)
        if not env.get("GATEWAY_RAW"):      # token counts only, as the evaluation runs; the runner alone keeps the bodies
            out["GATEWAY_RAW"] = "0"
    elif a.usage_gateway is False:
        out["USAGE_GATEWAY"] = ""
    return out


def bases(names: list[str]) -> list[str]:
    """The runner's task bases (a task directory's name without its mode): the tasks named, else every task."""
    from ..registry import list_tasks, resolve
    return [resolve(n).id for n in names] if names else sorted(t.id for t in list_tasks())


def _eval(a) -> int:
    changes = settings(a)
    cmd = C.command(RUNNER, a.batch, *bases(a.tasks))
    if a.print:
        print(C.quoted(cmd, changes))
        print(f"# started {a.lanes} time(s), one lane each; every other variable of your environment reaches the "
              "runner unchanged")
        return 0
    lanes = [C.spawn(cmd, changes) for _ in range(a.lanes)]
    if a.lanes > 1:
        print(f"roboarc eval: batch {a.batch}, {a.lanes} lanes started (PIDs {' '.join(str(p.pid) for p in lanes)})",
              file=sys.stderr, flush=True)

    def stop(signum, frame):            # a `kill` of this command ends its lanes too
        for p in lanes:
            if p.poll() is None:
                p.terminate()
    try:
        previous = signal.signal(signal.SIGTERM, stop)
    except ValueError:                  # not the main thread: no handler, the lanes are waited for all the same
        previous = None
    try:
        codes = [p.wait() for p in lanes]
    except KeyboardInterrupt:           # Ctrl-C reached the lanes already (the same terminal): wait for them to end
        codes = [p.wait() for p in lanes] + [130]
    finally:
        if previous is not None:
            signal.signal(signal.SIGTERM, previous)
    return max(abs(c) for c in codes)


def add_parsers(sub) -> None:
    p = sub.add_parser("eval", formatter_class=C.Formatter,
                       help="an evaluation batch with a coding agent (the batch runner)", description=f"""\
Run a coding agent over tasks, in both modes, unattended: one Harbor job per task, mode and trial, each trial from its
own appearance seed, placed on a GPU and admitted by the CPUs and memory its task declares.

This command is a wrapper of the batch runner, {RUNNER}:
  - each option below sets the runner's variable named in its help (--model sets MODEL, --trials sets K, ...), and
    --usage-gateway also GATEWAY_RAW=0 unless you set GATEWAY_RAW yourself;
  - every other variable of your environment reaches the runner unchanged, so its other settings stay available
    (GATEWAY_RAW, MEM_GUARD, AGENT_CONFIG, WEB_SEARCH, CODEX_FORCE_AUTH_JSON, ...: the runner's header lists them all,
    docs/RUNNING.md explains them), and a variable you already export is used when its option is not given;
  - --lanes N starts N runners on the batch. Each lane runs one trial at a time, and every job is claimed by exactly
    one lane. Starting the same command again adds lanes; after a reboot or a stopped lane it continues the batch,
    and finished trials are skipped (the runner's header says how to run an interrupted trial again).
The command stays in the foreground until its lanes end. For a long batch start it under nohup or tmux.

A batch is named by you; the name fixes the seeds of each task's trials, so use one name for one batch.
  progress        <jobs>/<batch>.log, or in a browser: python3 scripts/harbor/live_board.py --jobs <jobs>
  the result      <jobs>/<batch>.summary.md  (roboarc results summary <batch>)
  per trial       <jobs>/<batch>-<mode>-<task>[-t<k>]/ (Harbor's records), <results>/<batch>/<task>/<run id>/t<k>/
A single trial needs no batch: harbor run -p tasks -i <task>-<mode> -a <agent> -m <model> ...

Exit status: the highest of the lanes', each the runner's own: 0 when it ended, 2 when it refused a setting before
anything started.""",
                       epilog="""\
examples:
  # two tasks, both modes, three trials each, credentials from .env
  roboarc eval mybatch orbits-02 cast-coil-06 --agent codex --model openai/<model> --effort high --env-file .env

  # a first smoke run: one task, one trial per mode
  roboarc eval smoke interlocked-claws-08 --agent codex --model openai/<model> --effort high --trials 1 \\
      --env-file .env --jobs jobs/smoke

  # every task, seed 0 first (later the same command with --seeds 1, then --seeds 2), 8 lanes on two GPUs
  nohup roboarc eval full --agent codex --model openai/<model> --effort high --seeds 0 --lanes 8 --gpus 0,1 \\
      --env-file .env --results /data/roboarc_results --usage-gateway > full.out 2>&1 &

  # show the runner command instead of running it
  roboarc eval mybatch orbits-02 --agent codex --model openai/<model> --effort high --print""")
    p.add_argument("batch", metavar="BATCH", help="the batch's name: a word of your choice, the prefix of its job "
                   "names and the key of its seed assignment")
    p.add_argument("tasks", metavar="TASK", nargs="*", help="the tasks to run, by id or alias: roboarc-orbits-02-i01, "
                   "orbits-02, ikea-lack (default: every task of the checkout)")
    p.add_argument("--agent", metavar="AGENT", help="the agent harness (AGENT): codex, claude-code, dsh, antigravity or "
                   "kimi-code. Required, unless AGENT is set")
    p.add_argument("--model", metavar="MODEL", help="the agent's model as Harbor's -m takes it, <provider>/<model> (MODEL), "
                   "e.g. openai/<model>. Required, unless MODEL is set")
    p.add_argument("--effort", metavar="EFFORT", help="the reasoning effort passed to the agent (EFFORT), e.g. low, medium, "
                   "high, xhigh; `default` passes none and leaves the level to the harness. It is part of the run id. "
                   "Required, unless EFFORT is set")
    p.add_argument("--modes", metavar="MODE[,MODE]", help="the modes to run (MODES): privileged, standard (default: "
                   "privileged,standard)")
    p.add_argument("--trials", metavar="N", type=C.positive_int, help="trials per task and mode, an integer (K). The "
                   f"trials of a task get different seeds (default: {TRIALS}, the evaluation protocol's; the runner "
                   "script on its own defaults to 1)")
    p.add_argument("--seeds", metavar="SEED[,SEED...]", help="run only the trials at these seeds, a seed's trials before the "
                   "next seed's (SEEDS), e.g. 0, then later 1,2 with the same batch name. Needs as many trials as "
                   "a task has seeds, 3 (default: every trial, task by task)")
    p.add_argument("--lanes", metavar="N", type=C.positive_int, default=1, help="runners to start on the batch, an "
                   "integer: N trials at a time. The runner puts up to 4 trials of rigid tasks on a GPU and gives a "
                   "rope or cloth trial a GPU of its own, so up to 4 per GPU are useful (default: 1)")
    p.add_argument("--gpus", metavar="GPU[,GPU...]", help="the GPUs to use, by index (GPUS), e.g. 0,1 (default: every GPU "
                   "nvidia-smi lists)")
    p.add_argument("--env-file", metavar="FILE", help="a .env file with the agent's credentials, which Harbor loads "
                   "(ENV_FILE); .env.example lists the variables (default: none, the credentials come from your "
                   "environment)")
    p.add_argument("--jobs", metavar="DIR", help="Harbor's jobs directory (JOBS): the batch's log, plan, summary and "
                   "one directory per job (default: jobs/agents in the checkout, which git ignores)")
    p.add_argument("--results", metavar="DIR", help="where the per-trial results are written (RESULTS), a directory "
                   "outside the checkout (default: rcb_results next to the checkout)")
    p.add_argument("--usage-gateway", action=argparse.BooleanOptionalAction, default=None,
                   help=f"record token usage per request through the usage gateway of this checkout (USAGE_GATEWAY="
                   f"{GATEWAY}): the agent's model calls go through it, and it enforces the per-trial token budget. "
                   "The option also sets GATEWAY_RAW=0, as the evaluation runs do: the gateway keeps token counts "
                   "and no request or response body. A GATEWAY_RAW already set in your environment stands "
                   "(GATEWAY_RAW=1 keeps the bodies, which hold the whole conversation). antigravity and kimi-code "
                   "need the gateway. --no-usage-gateway runs without it even when USAGE_GATEWAY is set (default: "
                   "USAGE_GATEWAY as your environment has it, else off)")
    p.add_argument("--print", action="store_true", help="print the runner command this would start, with the variables "
                   "the options set, and start nothing")
    p.set_defaults(fn=_eval)
