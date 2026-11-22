"""roboarc vla run / stub / droid: the VLA interface's three commands under one name. Each one hands its arguments on
as they are: `run` to roboarc.vla.run, `stub` to roboarc.vla.stub_server, `droid` to scripts/vla_droid.sh. Their
options, checks and help are those modules' and that script's own."""
from __future__ import annotations

from . import _common as C

DROID = "scripts/vla_droid.sh"


def _module(name: str, command: str):
    try:
        import importlib
        return importlib.import_module(f"roboarc.vla.{name}")
    except ImportError as e:
        raise C.CliError(f"vla {command}: {e}. It needs numpy and, for the policy connection, the packages of "
                         "`pip install -e '.[vla]'`; the images have numpy (docs/VLA.md, Setup)") from None


def run(argv: list[str]) -> int:
    return _module("run", "run").main(argv, prog="roboarc vla run")


def stub(argv: list[str]) -> int:
    return _module("stub_server", "stub").main(argv, prog="roboarc vla stub")


def droid(argv: list[str]) -> int:
    return C.run(C.command(DROID, *argv))


FORWARD = {"run": run, "stub": stub, "droid": droid}


def forward(argv: list[str]):
    """The handler and its arguments when `argv` (after `vla`) is one of the three commands, passed on untouched; None
    when the command line's own parser should answer (`roboarc vla --help`, `roboarc vla droid --help`, a typo)."""
    if not argv or argv[0] not in FORWARD:
        return None
    name, rest = argv[0], argv[1:]
    before = rest[:rest.index("--")] if "--" in rest else rest
    if name == "droid" and ("-h" in before or "--help" in before):
        return None                      # the script has no --help: the parser below prints its options
    return FORWARD[name], rest


def add_parsers(sub) -> None:
    top = sub.add_parser("vla", formatter_class=C.Formatter,
                         help="graded episodes with a vision-language-action policy server", description="""\
Vision-language-action policies on RoboARC (docs/VLA.md). A policy served the openpi way (openpi's websocket policy
server) gets the task's camera images, the robot's state and a one-sentence instruction, and returns action chunks;
each episode is graded by the task's own grader.

  run     graded episodes of one task against a running policy server; inside the task's image, with one GPU
  stub    a stand-in policy server without a model (hold still, or replay a file), to check a setup
  droid   a DROID-trained model on the one-arm puzzles in one command; on the host: it starts the model server and
          one container per episode

run and stub are the modules roboarc.vla.run and roboarc.vla.stub_server (`python -m roboarc.vla.run` is the same
command); droid is scripts/vla_droid.sh. All three take their arguments as those do.""", epilog="""\
example (inside the image):
  roboarc vla run --task interlocked-claws-08 --policy ws://POLICY_HOST:8000 --seed 0,1,2 --run-id my-policy \\
      --out /results
"roboarc vla SUBCOMMAND --help" has each subcommand's arguments.""")
    s = top.add_subparsers(dest="vla_cmd", metavar="SUBCOMMAND", title="subcommands", required=True)
    # run and stub never reach this parser (forward() hands them, --help included, to their modules); listed for the help
    s.add_parser("run", add_help=False, help="graded episodes of one task against a policy server (inside the image)")
    s.add_parser("stub", add_help=False, help="a stand-in policy server without a model")
    p = s.add_parser("droid", formatter_class=C.Formatter,
                     help="a DROID-trained model on the one-arm puzzles in one command (on the host)", description=f"""\
A DROID-trained policy (pi0.5-DROID, MolmoAct2-DROID, GR00T-N1.7-DROID, Cosmos3-Nano-Policy-DROID) on RoboARC's
one-arm puzzles in one command. Runs {DROID} with the same arguments: it starts one model server,
runs every (task, seed) episode in the puzzle image with DROID's camera poses, --jobs at a time against that server,
then stops the server. Results: <out>/<task>/<run id>/s<seed>/e1/.

Needs Docker with the GPU, the protocol packages in $VLA_LIB (default .vla-lib in the checkout) and the model's own
environment ($OPENPI, $MOLMOACT2, $GROOT, $COSMOS3): the script's header and docs/VLA.md, "DROID-trained policies",
say how to set them up and how much GPU memory each model takes. The variables IMAGE, VLA_LIB, OPENPI, MOLMOACT2,
GROOT and COSMOS3 reach the script from your environment.""", epilog="""\
examples:
  # no model: checks the setup
  roboarc vla droid --task hedgehog-in-cage-02 --model hold --out results -- --max-steps 400
  # pi0.5-DROID on two tasks, three seeds each, four episodes at a time
  roboarc vla droid --task lock-picking-03,buzz-wire-3d-06 --model jointpos --seed all --jobs 4 --out results""")
    p.add_argument("--task", metavar="TASK[,TASK...]", required=True, help="one-arm puzzle ids or aliases, comma-separated; "
                   "the one-arm families are buzz_wire_3d, fiber_bundling, hedgehog_in_cage and lock_picking (required)")
    p.add_argument("--model", metavar="NAME", help="jointpos, velocity, polaris (the three pi0.5-DROID checkpoints), "
                   "molmoact2, molmoact2-bf16, groot, cosmos3, or hold (no model: checks the setup); "
                   "roboarc/vla/droid_models.py describes each (default: jointpos)")
    p.add_argument("--seed", metavar="SEED[,SEED...]", help="the instance seeds: 0, 0,1,2 or all (default: 0)")
    p.add_argument("--jobs", metavar="N", help="episodes at a time, an integer; each takes about 1 GB of GPU memory "
                   "beside the model server (default: 1)")
    p.add_argument("--out", metavar="DIR", help="the results root (default: vla_results in the current directory)")
    p.add_argument("--run-id", metavar="ID", help="the policy's name in the results tree (default: named after the "
                   "model: pi05-droid-jointpos, molmoact2-droid, groot-n17-droid, cosmos3-nano-droid, ...)")
    p.add_argument("--policy", metavar="URL", help="ws://HOST:PORT of a DroidAdapter server of --model that already "
                   "runs; no server is started then (default: start one)")
    p.add_argument("--port", metavar="PORT", help="the port of the server the script starts, an integer (default: 8000)")
    p.add_argument("--gpu", metavar="GPU", help="one GPU for the model server and every episode: its nvidia-smi index "
                   "(0, 1, ...) or UUID. On a shared host with several GPUs, name a free one (default: every GPU is "
                   "visible to the containers, which is right for a host with one)")
    p.add_argument("run_args", metavar="-- RUN_ARGS", nargs="*", help="after `--`: more arguments for each episode's "
                   "`roboarc vla run`, e.g. --max-steps 400 (default: none)")
