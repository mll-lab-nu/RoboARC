"""roboarc results summary / send / receive: a batch's summary table as the batch runner wrote it, and the two scripts
that carry a batch's records to and from the results dataset (scripts/send_results.sh, scripts/receive_results.sh).
Each option sets the script's variable of the same meaning; the rest of the caller's environment reaches the script
unchanged. Standard library only."""
from __future__ import annotations

import os
import sys
from pathlib import Path

from . import _common as C

SEND, RECEIVE = "scripts/send_results.sh", "scripts/receive_results.sh"


def _given(a, pairs) -> dict:
    return {name: getattr(a, flag) for flag, name in pairs if getattr(a, flag)}


def _summary(a) -> int:
    jobs = Path(a.jobs or os.environ.get("JOBS") or C.repo() / "jobs" / "agents")
    path = jobs / f"{a.batch}.summary.md"
    if not path.is_file():
        raise C.CliError(f"results summary: no {path} (the batch runner writes it after each job; another jobs "
                         "directory: --jobs DIR)", code=1)
    sys.stdout.write(path.read_text())
    return 0


def _send(a) -> int:
    changes = _given(a, (("jobs", "JOBS"), ("results", "RESULTS"), ("env_file", "ENV_FILE"), ("notes", "NOTES")))
    changes.update({name: "1" for flag, name in (("create_pr", "CREATE_PR"), ("dry_run", "DRY_RUN")) if getattr(a, flag)})
    return C.run(C.command(SEND, a.batch, a.name), changes)


def _receive(a) -> int:
    if a.list and not a.delivery:
        raise C.CliError("results receive: --list needs a delivery, <name>/<batch>/<bundle id> (without one, every "
                         "delivery is listed)")
    if a.list and a.job:
        raise C.CliError("results receive: --list takes no job names")
    args = ([a.delivery] if a.delivery else []) + (["--jobs"] if a.list else a.job)
    return C.run(C.command(RECEIVE, *args), _given(a, (("dest", "DEST"), ("revision", "REVISION"))))


def add_parsers(sub) -> None:
    top = sub.add_parser("results", formatter_class=C.Formatter,
                         help="a batch's summary table; send results to, or fetch them from, the results dataset",
                         description="""\
The results of an evaluation batch (roboarc eval, or the batch runner started by hand).

  summary   print the batch's summary table, as the runner wrote it
  send      pack the batch's records, scan them for secrets and upload them to the results dataset
  receive   list, download, check and unpack deliveries from the results dataset

The dataset is the Hugging Face dataset RoboARC/RoboARC-outputs (HF_DATASET names another one).""", epilog="""\
example:
  roboarc results summary mybatch
"roboarc results SUBCOMMAND --help" has each subcommand's arguments.""")
    s = top.add_subparsers(dest="results_cmd", metavar="SUBCOMMAND", title="subcommands", required=True)

    p = s.add_parser("summary", formatter_class=C.Formatter, help="print a batch's summary table", description="""\
Print <jobs>/<batch>.summary.md, which the batch runner rewrites after every job: one table per comparable
configuration (mode, time limit, ...) with every trial's task, trial number, seed, success, result and token counts,
and the success rate over the trials that are results. A trial voided for infrastructure is listed and not counted.
Exit status: 0, or 1 when the batch has no summary yet.""", epilog="""\
example:
  roboarc results summary mybatch --jobs jobs/agents""")
    p.add_argument("batch", metavar="BATCH", help="the batch's name, as given to roboarc eval")
    p.add_argument("--jobs", metavar="DIR", help="the jobs directory the batch ran with (default: JOBS of your "
                   "environment, else jobs/agents in the checkout)")
    p.set_defaults(fn=_summary)

    p = s.add_parser("send", formatter_class=C.Formatter, help="upload a batch's records to the results dataset",
                     description=f"""\
Send one batch's records to the results dataset. Runs {SEND} <batch> <name>:
  1. a secret scan of the batch's jobs with the credentials the batch used (keys are redacted; nothing is sent while
     one is left);
  2. one archive per job, and one for the batch's log, summary, plan and per-trial results, with checksums;
  3. an index and a manifest (the checkout's commit, the host, the image ids, the runner's settings, no credentials);
  4. the upload into <dataset>/<name>/<batch>/<bundle id>/. A later delivery never overwrites an earlier one.
Needs the hf command (pip install -U huggingface_hub) and a write token of your Hugging Face account in
~/.roboarc_hf_token (HF_TOKEN_FILE names another file). HF_DATASET, OUT and the other variables of the script's header
reach it from your environment. The dataset is public: try --dry-run first.""", epilog="""\
examples:
  roboarc results send mybatch --name alice --dry-run
  roboarc results send mybatch --name alice --results /data/roboarc_results --notes "lane 3 restarted after a reboot"
""")
    p.add_argument("batch", metavar="BATCH", help="the batch's name, as given to roboarc eval")
    p.add_argument("--name", metavar="NAME", required=True, help="your name in the dataset's tree: lowercase letters, "
                   "digits, - and _ (required)")
    p.add_argument("--jobs", metavar="DIR", help="the jobs directory the batch ran with (JOBS; default: jobs/agents "
                   "in the checkout)")
    p.add_argument("--results", metavar="DIR", help="the per-trial results directory the batch ran with (RESULTS; "
                   "default: rcb_results next to the checkout)")
    p.add_argument("--env-file", metavar="FILE", help="the credentials file the batch used, whose values the secret "
                   "scan looks for (ENV_FILE; default: .env in the checkout)")
    p.add_argument("--notes", metavar="TEXT", help="one line for the manifest: crashes, outages, reruns (NOTES; "
                   "default: none)")
    p.add_argument("--create-pr", action="store_true", help="upload as a pull request to the dataset, for an account "
                   "without write access to it (CREATE_PR=1; default: a commit)")
    p.add_argument("--dry-run", action="store_true", help="do everything but the upload (DRY_RUN=1)")
    p.set_defaults(fn=_send)

    p = s.add_parser("receive", formatter_class=C.Formatter,
                     help="list, download and unpack deliveries from the results dataset", description=f"""\
Fetch deliveries made with `roboarc results send`. Runs {RECEIVE}:
  roboarc results receive                         every delivery, one line each: <name>/<batch>/<bundle id>, its
                                                  trial counts and its runner line
  roboarc results receive DELIVERY --list         that delivery's jobs: job, mode, trial, seed, reward, archive size
  roboarc results receive DELIVERY [JOB ...]      download that delivery (or only the jobs named), check every archive
                                                  and file against the delivery's checksums, unpack it, then print its
                                                  manifest and summary table
Needs the hf command (pip install -U huggingface_hub). HF_DATASET names another dataset.""", epilog="""\
examples:
  roboarc results receive
  roboarc results receive alice/mybatch/20261121T1830Z --list
  roboarc results receive alice/mybatch/20261121T1830Z --dest /data/roboarc_deliveries""")
    p.add_argument("delivery", metavar="DELIVERY", nargs="?", help="a delivery, <name>/<batch>/<bundle id>, as the "
                   "listing prints it (default: list every delivery)")
    p.add_argument("job", metavar="JOB", nargs="*", help="fetch only these jobs of the delivery, by the names --list "
                   "prints (default: every job)")
    p.add_argument("--list", action="store_true", help="list the delivery's jobs and download no archive")
    p.add_argument("--dest", metavar="DIR", help="where deliveries are unpacked, <DIR>/<name>/<batch>/<bundle id>/ "
                   "(DEST; default: ~/roboarc_helper_runs)")
    p.add_argument("--revision", metavar="REV", help="the dataset revision to read (REVISION), e.g. refs/pr/3 for a "
                   "delivery sent with --create-pr and not merged yet (default: main)")
    p.set_defaults(fn=_receive)
