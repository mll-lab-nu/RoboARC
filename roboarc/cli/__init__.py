"""The `roboarc` command line (also `python -m roboarc`): one command with a subcommand for everything a user runs.

    roboarc doctor                       can this machine run RoboARC?                      doctor.py
    roboarc images pull|check|build      the pinned container images                        images.py, licences.py
    roboarc list | info | replay         the tasks, one task's facts, a trajectory's hash   tasks.py
    roboarc selftest                     through Harbor: the reference 1, doing nothing 0   selftest.py
    roboarc eval                         an evaluation batch with a coding agent            evaluate.py
    roboarc results summary|send|receive a batch's results                                  results.py
    roboarc vla run|stub|droid           VLA policies                                       vla.py

Every subcommand is a thin wrapper of code and scripts that work without it (the Python package, images/,
scripts/harbor/run_agent_batch.sh, scripts/send_results.sh, scripts/vla_droid.sh, Harbor): those keep their paths,
arguments and variables, and the wrappers add no behaviour of their own beyond the licence consent before an image is
pulled or built. Each subcommand's --help is its documentation. The host-side subcommands use the standard
library only, and nothing heavy is imported before a subcommand needs it."""
from __future__ import annotations

import argparse
import sys

from . import _common as C


def build_parser() -> argparse.ArgumentParser:
    import roboarc
    from . import doctor, evaluate, images, results, selftest, tasks, vla
    ap = argparse.ArgumentParser(prog="roboarc", formatter_class=C.Formatter, description=f"""\
RoboARC {roboarc.__version__}: physical puzzles for robotic manipulation. One command for everything a user runs.

A task is named by its id or any alias: roboarc-orbits-02-i01, orbits-02-i01, orbits-02 (IKEA: roboarc-ikea-lack-i00,
ikea-lack, lack). The modes are privileged and standard; the appearance seeds are 0, 1 and 2. Lists are
comma-separated: --modes privileged,standard --seeds 0,1,2.""", epilog="""\
Where they run. On the host, with Python 3.11 or newer and no simulator: doctor, images, list, info, selftest, eval,
results, vla droid. Inside an image, where the simulator is: info --sim, replay, vla run, and doctor for the graded
runtime. One trial of a coding agent needs no subcommand, it is a plain Harbor command:
  harbor run -p tasks -i <task>-<mode> -a <agent> -m <model> -o <jobs dir> --job-name <name>

first steps:
  roboarc doctor                                   # what this machine still needs
  roboarc images pull --accept-licences            # the four images, about 19 GB
  roboarc selftest interlocked-claws-08            # the reference scores 1, doing nothing scores 0
"roboarc COMMAND --help" has each command's arguments, defaults and examples.""")
    ap.add_argument("--version", action="version", version=f"roboarc {roboarc.__version__}")
    sub = ap.add_subparsers(dest="command", metavar="COMMAND", title="commands", required=True)
    doctor.add_parsers(sub)
    images.add_parsers(sub)
    tasks.add_parsers(sub)          # list, info, replay
    selftest.add_parsers(sub)
    evaluate.add_parsers(sub)
    results.add_parsers(sub)
    vla.add_parsers(sub)
    tasks.add_advanced_parsers(sub)     # patch-mujoco, and check (the graded-runtime part of doctor, not listed)
    return ap


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    try:
        if argv[:1] == ["vla"]:
            from . import vla
            handed = vla.forward(argv[1:])
            if handed is not None:
                return handed[0](handed[1])
        ap = build_parser()
        if not argv:
            ap.print_help(sys.stderr)
            return 2
        a = ap.parse_args(argv)
        return a.fn(a)
    except C.CliError as e:
        print(f"roboarc: {e}", file=sys.stderr)
        return e.code
    except KeyError as e:               # roboarc.resolve(): an unknown task, with the similar ids
        if "unknown RoboARC task" not in str(e):
            raise
        print(f"roboarc: {e.args[0]}", file=sys.stderr)
        return 2
