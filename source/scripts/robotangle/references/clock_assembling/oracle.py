# Oracle code for `rcb_topogym.py run-oracle --task clock_assembling --code <this file>` on clock_assembling_gear_train
# (runs/watch_assembly_design_20261014, README_v3.md; the design's L3 oracle on topogym's control stack). run-oracle runs
# with the repository as the working directory; the oracle (clock_oracle.ClockOracle) sits next to this file. It plans
# every move from the measured state and commands only joint and gripper targets (one row per control step).
import os, sys, json, time
HERE = "scripts/robotangle/references/clock_assembling"
sys.path.insert(0, HERE)
from clock_oracle import ClockOracle


def _log(*a):
    print(*a, file=sys.stderr, flush=True)


t0 = time.time()
oracle = ClockOracle(session, log=_log)
oracle.run(os.environ.get("CLOCK_UPTO", "all"))
_log(json.dumps(dict(control_steps=session.n_steps, episode_end=session.episode_end, wall_s=round(time.time() - t0, 1),
                     stat=oracle.stat, tracking_mm=oracle.tracking(), mechanism=session.env._world.read_state()["mechanism"])))
