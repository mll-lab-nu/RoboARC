"""The `roboarc` command line (roboarc/cli/): help, argument parsing, the consent gate, and that every subcommand
starts the right script with the right arguments and variables. Nothing here starts Docker, Harbor or a script of the
repository's runner: the two functions that start processes (roboarc.cli._common.run and spawn) are replaced, or,
in the end-to-end tests at the bottom, the runner, images/pull_all.sh and harbor are stand-ins that record how they
were started (the last one runs the repository's three small task-copy scripts for real, on a task under 1 MB)."""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

import roboarc
from roboarc import _runtime
from roboarc.cli import _common as C
from roboarc.cli import doctor, evaluate, images, licences, main, selftest, vla

REPO = Path(__file__).resolve().parents[2]
# every subcommand, as typed after `roboarc`
COMMANDS = [[], ["doctor"], ["images"], ["images", "pull"], ["images", "check"], ["images", "build"],
            ["list"], ["info"], ["replay"], ["selftest"], ["eval"], ["results"], ["results", "summary"],
            ["results", "send"], ["results", "receive"], ["vla"], ["vla", "droid"], ["patch-mujoco"], ["check"]]
BASH = "bash"


def script(rel: str) -> str:
    return str(REPO / rel)


@pytest.fixture(autouse=True)
def own_config(tmp_path, monkeypatch):
    """No test reads or writes the user's own consent file, and none inherits a runner variable."""
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    for name in ("AGENT", "MODEL", "EFFORT", "K", "SEEDS", "MODES", "GPUS", "JOBS", "RESULTS", "ENV_FILE", "USAGE_GATEWAY",
                 "GATEWAY_RAW", "ROBOARC_REPO", licences.VARIABLE):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def cli(capsys):
    """cli(*argv) -> (exit status, stdout, stderr) of `roboarc argv...`, run in this process."""
    def call(*argv):
        try:
            rc = main([str(a) for a in argv])
        except SystemExit as e:              # argparse: --help (0), a usage error (2)
            rc = e.code
        out = capsys.readouterr()
        return rc, out.out, out.err
    return call


@pytest.fixture
def started(monkeypatch):
    """Replace the starters: every command a subcommand would run is recorded as (argv, changes) instead."""
    calls = []

    def run(argv, changes=None, **kw):
        calls.append((list(argv), dict(changes or {})))
        return 0

    monkeypatch.setattr(C, "run", run)
    return calls


ACCEPTED = {"ROBOARC_ACCEPT_LICENCES": "1"}      # what an image script is started with once the user has accepted


def accept():
    licences.record("test")


# ------------------------------------------------------------------------------------------------ help and parsing

@pytest.mark.parametrize("command", COMMANDS, ids=lambda c: " ".join(c) or "roboarc")
def test_help_states_usage_arguments_and_an_example(cli, command):
    rc, out, err = cli(*command, "--help")
    assert rc == 0 and err == ""
    assert out.startswith("usage: " + " ".join(["roboarc", *command]))
    assert "example" in out
    assert "--set" not in out and "easy" not in out          # the easy set is retired: no option names it


def test_every_option_states_its_default_or_that_it_is_required(cli):
    """The help is the interface's documentation: each option that takes a value says what happens without it."""
    import argparse
    from roboarc.cli import build_parser

    def walk(parser, path):
        for action in parser._actions:
            if isinstance(action, argparse._SubParsersAction):
                for name, sub in action.choices.items():
                    yield from walk(sub, path + [name])
            elif action.option_strings and action.nargs != 0 and action.help != argparse.SUPPRESS:
                yield " ".join(path + [action.option_strings[0]]), action.help or ""

    missing = [name for name, text in walk(build_parser(), ["roboarc"])
               if "default" not in text and "equired" not in text]
    assert missing == []


def test_no_arguments_prints_the_help_and_fails(cli):
    rc, out, err = cli()
    assert rc == 2 and "commands:" in err and out == ""


def test_version(cli):
    rc, out, _ = cli("--version")
    assert rc == 0 and out.strip() == f"roboarc {roboarc.__version__}"


def test_the_nine_subcommands_are_listed(cli):
    _, out, _ = cli("--help")
    block = out.split("commands:\n")[1].split("\n\n")[0].splitlines()
    listed = [line.split()[0] for line in block if line[:4] == "    " and line[4:5] != " "]      # not a wrapped help line
    assert listed == ["doctor", "images", "list", "info", "replay", "selftest", "eval", "results", "vla", "patch-mujoco"]


def test_unknown_option_and_unknown_command_are_usage_errors(cli):
    assert cli("list", "--set", "easy")[0] == 2
    assert cli("eval", "b", "--agent", "codex", "--model", "m", "--effort", "high", "--set", "easy")[0] == 2
    assert cli("frobnicate")[0] == 2


def test_python_m_roboarc_is_the_same_command():
    from roboarc import __main__
    assert __main__.main is main
    p = subprocess.run([sys.executable, "-m", "roboarc", "--help"], capture_output=True, text=True, cwd=REPO, timeout=60)
    assert p.returncode == 0 and p.stdout.startswith("usage: roboarc ") and "commands:" in p.stdout


@pytest.mark.parametrize("argv", [["list", "--json"], ["info", "orbits-02"], ["selftest", "--print"], ["doctor", "--help"],
                                  ["eval", "b", "orbits-02", "--agent", "codex", "--model", "m", "--effort", "high", "--print"]],
                         ids=lambda a: a[0])
def test_host_side_subcommands_need_no_numpy(argv):
    code = "import sys; sys.modules['numpy'] = None; from roboarc.cli import main; sys.exit(main(sys.argv[1:]))"
    p = subprocess.run([sys.executable, "-c", code, *argv], capture_output=True, text=True, cwd=REPO, timeout=60)
    assert p.returncode == 0, p.stderr


# ------------------------------------------------------------------------------------------- list, info, check

def test_list(cli):
    rc, out, err = cli("list")
    assert rc == 0 and len(out.splitlines()) == 90 and err.strip() == "90 tasks"
    rc, out, _ = cli("list", "--suite", "ikea", "--json")
    assert rc == 0 and [t["suite"] for t in json.loads(out)] == ["ikea"] * 12


def test_list_family(cli):
    want = [t.id for t in roboarc.list_tasks(family="interlocked_claws")]
    for spelling in ("interlocked_claws", "interlocked-claws"):
        rc, out, _ = cli("list", "--family", spelling)
        assert rc == 0 and [line.split()[0] for line in out.splitlines()] == want
    rc, _, err = cli("list", "--family", "no-such-family")
    assert rc == 2 and "no family" in err and "interlocked_claws" in err


def test_info_and_an_unknown_task(cli):
    rc, out, _ = cli("info", "orbits-02")
    assert rc == 0 and json.loads(out)["task"]["id"] == "roboarc-orbits-02-i01"
    rc, out, err = cli("info", "no-such-task")
    assert rc == 2 and out == "" and err.startswith("roboarc: unknown RoboARC task 'no-such-task'")


def test_check_still_prints_the_graded_runtime_report(cli):
    """`python -m roboarc check` keeps its JSON and its exit status (docs/API.md); `doctor` is the new name."""
    rc, out, _ = cli("check")
    report = json.loads(out)
    assert set(report) == {"libmujoco_sha256", "libmujoco_epa_patched", "numpy_masking", "puzzle_image", "ikea_image",
                           "nyx_python (ikea -sim image)", "repo", "problems"}
    assert rc == (1 if report["problems"] else 0)


def test_patch_mujoco_runs_the_patch_script(cli, started, monkeypatch, tmp_path):
    import importlib.util
    spec = type("Spec", (), {"submodule_search_locations": [str(tmp_path)]})()
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: spec)
    assert cli("patch-mujoco")[0] == 0
    assert started == [([sys.executable, script("source/images/roboarc/mujoco-3.14.0-epa/bytepatch_libmujoco.py"),
                         str(tmp_path / "libmujoco.so.3.14.0")], {})]


# ---------------------------------------------------------------------------------------------- the environment

def test_child_env_is_the_callers_environment(monkeypatch):
    monkeypatch.setenv("GATEWAY_RAW", "0")
    monkeypatch.setenv("NPY_DISABLE_CPU_FEATURES", "set by importing roboarc")
    monkeypatch.setenv("OPENBLAS_CORETYPE", "the caller's own")
    monkeypatch.setattr(_runtime, "ENV_ADDED", {"NPY_DISABLE_CPU_FEATURES"})
    env = C.child_env({"K": 3, "USAGE_GATEWAY": "", "GATEWAY_RAW": None})
    assert env["K"] == "3" and env["USAGE_GATEWAY"] == "" and "GATEWAY_RAW" not in env
    assert "NPY_DISABLE_CPU_FEATURES" not in env and env["OPENBLAS_CORETYPE"] == "the caller's own"
    untouched = {k: v for k, v in os.environ.items() if k not in ("GATEWAY_RAW", "NPY_DISABLE_CPU_FEATURES")}
    assert {k: env[k] for k in untouched} == untouched


def test_run_and_spawn_start_the_command_with_that_environment(monkeypatch):
    monkeypatch.setenv("FROM_CALLER", "4")
    code = "import os, sys; sys.exit(int(os.environ['FROM_CALLER']) + int(os.environ['FROM_OPTION']))"
    assert C.run([sys.executable, "-c", code], {"FROM_OPTION": 3}) == 7
    assert C.spawn([sys.executable, "-c", code], {"FROM_OPTION": 1}).wait() == 5


def test_command_runs_shell_scripts_with_bash_and_python_scripts_with_this_python():
    assert C.command("images/pull_all.sh", "--list") == [BASH, script("images/pull_all.sh"), "--list"]
    assert C.command("scripts/task_seed.py", "list", 3) == [sys.executable, script("scripts/task_seed.py"), "list", "3"]
    with pytest.raises(C.CliError, match="no/such/script.sh is not in this checkout"):
        C.command("no/such/script.sh")


# ------------------------------------------------------------------------------------------- images and consent

@pytest.fixture
def image_scripts(monkeypatch, tmp_path):
    """A checkout that has every script `roboarc images` names."""
    fake = tmp_path / "repo"
    for rel in ("scripts/roboarc/shared/rcb_topogym.py", "LICENSES/THIRD_PARTY.md", *images.BUILDS.values(),
                *(s[0] for s in images.SCRIPTS.values())):
        (fake / rel).parent.mkdir(parents=True, exist_ok=True)
        (fake / rel).write_text("third-party licences\n")
    monkeypatch.setenv("ROBOARC_REPO", str(fake))
    return fake


def no_terminal(monkeypatch, answer=None):
    monkeypatch.setattr(sys.stdin, "isatty", lambda: answer is not None, raising=False)
    monkeypatch.setattr("builtins.input", lambda prompt="": answer)


@pytest.mark.parametrize("argv", [["images", "pull"], ["images", "build", "roboarc"]], ids=lambda a: a[1])
def test_without_consent_a_non_interactive_run_stops_with_one_sentence(cli, started, image_scripts, monkeypatch, argv):
    no_terminal(monkeypatch)
    rc, out, err = cli(*argv)
    assert rc == 2 and started == [] and out == ""
    assert len(err.strip().splitlines()) == 1 and err.count(". ") == 0          # one sentence
    assert "LICENSES/THIRD_PARTY.md" in err and "--accept-licences" in err
    assert not licences.consent_file().exists()


def test_the_flag_accepts_once_and_later_runs_do_not_ask(cli, started, image_scripts, monkeypatch):
    no_terminal(monkeypatch)
    rc, _, err = cli("images", "pull", "--accept-licences")
    assert rc == 0 and "accepted" in err
    record = json.loads(licences.consent_file().read_text())
    assert record["accepted"] is True and record["licences"] == "LICENSES/THIRD_PARTY.md" and record["how"] == "--accept-licences"
    assert len(record["sha256"]) == 64 and record["roboarc"] == roboarc.__version__
    assert licences.consent_file() == Path(os.environ["XDG_CONFIG_HOME"]) / "roboarc" / "licences-accepted.json"
    rc, _, err = cli("images", "pull")                       # no flag, no terminal: accepted before
    assert rc == 0 and err == ""
    pull = [BASH, str(image_scripts / "images/pull_all.sh")]
    assert started == [(pull, ACCEPTED), (pull, ACCEPTED)]        # the scripts are told through their own variable


def test_the_scripts_variable_accepts_like_the_flag(cli, started, image_scripts, monkeypatch):
    no_terminal(monkeypatch)
    monkeypatch.setenv(licences.VARIABLE, "0")
    assert cli("images", "pull")[0] == 2 and started == []
    monkeypatch.setenv(licences.VARIABLE, "1")
    assert cli("images", "pull")[0] == 0 and len(started) == 1
    assert json.loads(licences.consent_file().read_text())["how"] == "ROBOARC_ACCEPT_LICENCES=1"


def test_the_us_spelling_of_the_flag_works_too(cli, started, image_scripts, monkeypatch):
    no_terminal(monkeypatch)
    assert cli("images", "pull", "--accept-licenses")[0] == 0 and licences.accepted() and len(started) == 1


def test_on_a_terminal_the_user_is_asked(cli, started, image_scripts, monkeypatch):
    no_terminal(monkeypatch, answer="n")
    rc, _, err = cli("images", "pull")
    assert rc == 1 and started == [] and not licences.accepted()
    assert str(image_scripts / "LICENSES/THIRD_PARTY.md") in err and "not accepted" in err
    no_terminal(monkeypatch, answer="yes")
    rc, _, err = cli("images", "pull")
    assert rc == 0 and len(started) == 1 and licences.accepted()
    assert json.loads(licences.consent_file().read_text())["how"] == "prompt"


def test_an_unreadable_consent_file_is_no_consent(image_scripts):
    licences.consent_file().parent.mkdir(parents=True)
    licences.consent_file().write_text("not json")
    assert not licences.accepted()
    licences.consent_file().write_text('{"accepted": "yes"}')
    assert not licences.accepted()


def test_images_subcommands_run_their_scripts(cli, started, image_scripts, monkeypatch):
    no_terminal(monkeypatch)
    accept()
    at = lambda rel: str(image_scripts / rel)                # noqa: E731
    assert cli("images", "pull", "--pins", "/p/pins.json")[0] == 0
    assert cli("images", "check")[0] == 0
    assert cli("images", "check", "--pins", "/p/pins.json", "--json")[0] == 0
    assert cli("images", "check", "--archives", "/a")[0] == 0
    assert cli("images", "check", "--pins", "/p/pins.json", "--archives", "/a", "--json")[0] == 0
    assert cli("images", "build", "roboarc", "--", "--instances", "/i", "--noinst")[0] == 0
    assert cli("images", "build", "ikea")[0] == 0
    assert cli("images", "build", "monitor", "--", "--load")[0] == 0
    assert started == [
        ([BASH, at("images/pull_all.sh"), "--pins", "/p/pins.json"], ACCEPTED),
        ([BASH, at("images/pull_all.sh"), "--check"], {}),
        ([BASH, at("images/pull_all.sh"), "--check", "--json", "--pins", "/p/pins.json"], {}),
        ([sys.executable, at("images/pins.py"), "archives", "--archives", "/a"], {}),
        ([sys.executable, at("images/pins.py"), "archives", "--archives", "/a", "--json", "--pins", "/p/pins.json"], {}),
        ([BASH, at("images/roboarc/build.sh"), "--instances", "/i", "--noinst"], ACCEPTED),
        ([BASH, at("ikea/images/ikea-mujoco/build.sh")], ACCEPTED),
        ([BASH, at("images/monitor/build.sh"), "--load"], ACCEPTED)]


def test_images_check_asks_for_no_consent(cli, started, image_scripts, monkeypatch):
    no_terminal(monkeypatch)
    assert cli("images", "check")[0] == 0 and len(started) == 1 and not licences.accepted()


def test_images_usage_errors(cli, started, image_scripts):
    assert cli("images")[0] == 2
    assert cli("images", "build", "no-such-image")[0] == 2
    assert cli("images", "build")[0] == 2                    # which image
    assert started == []


def test_there_is_no_images_load(cli, started, image_scripts):
    """No script loads archives so that the tasks' pinned references resolve: images/README.md has the manual routes."""
    rc, _, err = cli("images", "load", "--archives", "/a", "--accept-licences")
    assert rc == 2 and "invalid choice: 'load'" in err and started == [] and not licences.accepted()
    assert "load" not in cli("images", "--help")[1]
    assert "load" not in cli("--help")[1]


def test_a_script_this_checkout_lacks_is_one_sentence(cli, monkeypatch):
    monkeypatch.setitem(images.SCRIPTS, "check", ("images/no_such_script.sh", "--check"))
    rc, _, err = cli("images", "check")
    assert rc == 2 and err.startswith("roboarc: images/no_such_script.sh is not in this checkout")
    monkeypatch.setitem(images.BUILDS, "monitor", "images/no_such_build.sh")
    rc, _, err = cli("images", "build", "monitor", "--accept-licences")
    assert rc == 2 and "images/no_such_build.sh is not in this checkout" in err
    assert not licences.accepted()                           # nothing is recorded for a command that cannot run


def test_the_scripts_and_the_licences_file_the_wrappers_name_exist():
    """The paths collaborators call keep their places (the wrappers call the same files)."""
    for rel in (images.SCRIPTS["pull"][0], *images.BUILDS.values(), evaluate.RUNNER, evaluate.GATEWAY + "/gateway.py",
                "scripts/send_results.sh", "scripts/receive_results.sh", vla.DROID, licences.LICENCES,
                "scripts/task_seed.py", "scripts/harbor/clamp_cpus.py", "scripts/harbor/pin_gpu.py",
                "scripts/harbor/host_info.py"):
        assert (REPO / rel).is_file(), rel


# --------------------------------------------------------------------------------------------------------- eval

class Lane:
    """What C.spawn returns, for a runner that ends at once with `code`."""
    pid = 4242

    def __init__(self, code=0):
        self.code = code

    def wait(self):
        return self.code

    def poll(self):
        return self.code


@pytest.fixture
def lanes(monkeypatch):
    calls, codes = [], []

    def spawn(argv, changes=None, **kw):
        calls.append((list(argv), dict(changes or {})))
        return Lane(codes.pop(0) if codes else 0)

    monkeypatch.setattr(C, "spawn", spawn)
    return calls, codes


RUNNER = [BASH, script("scripts/harbor/run_agent_batch.sh")]
REQUIRED = ["--agent", "codex", "--model", "openai/m", "--effort", "high"]


def test_eval_sets_the_runners_variables_and_starts_one_lane(cli, lanes):
    calls, _ = lanes
    rc, _, _ = cli("eval", "mybatch", "orbits-02", "roboarc-cast-coil-06-i01-standard", "lack", *REQUIRED)
    assert rc == 0
    assert calls == [(RUNNER + ["mybatch", "roboarc-orbits-02-i01", "roboarc-cast-coil-06-i01", "roboarc-ikea-lack-i00"],
                      {"AGENT": "codex", "MODEL": "openai/m", "EFFORT": "high", "K": 3})]


def test_eval_every_option(cli, lanes):
    calls, _ = lanes
    rc, _, err = cli("eval", "b", "orbits-02", *REQUIRED, "--modes", "standard,privileged", "--trials", "2", "--seeds", "0,1",
                     "--lanes", "3", "--gpus", "0,1", "--env-file", "my.env", "--jobs", "jobs/x", "--results", "/r",
                     "--usage-gateway")
    assert rc == 0 and "3 lanes started" in err
    want = {"AGENT": "codex", "MODEL": "openai/m", "EFFORT": "high", "MODES": "standard privileged", "K": 2, "SEEDS": "0 1",
            "GPUS": "0,1", "ENV_FILE": "my.env", "JOBS": "jobs/x", "RESULTS": "/r",
            "USAGE_GATEWAY": script("scripts/usage_gateway"), "GATEWAY_RAW": "0"}
    assert calls == [(RUNNER + ["b", "roboarc-orbits-02-i01"], want)] * 3


def test_eval_usage_gateway_keeps_token_counts_only_unless_the_caller_says_otherwise(cli, lanes, monkeypatch):
    """--usage-gateway sets GATEWAY_RAW=0, the evaluation runs' value; a GATEWAY_RAW of the caller's stands."""
    calls, _ = lanes
    gateway = script("scripts/usage_gateway")
    base = {"AGENT": "codex", "MODEL": "openai/m", "EFFORT": "high", "K": 3}
    argv = ("eval", "b", "orbits-02", *REQUIRED)
    assert cli(*argv, "--usage-gateway")[0] == 0
    assert cli(*argv)[0] == 0                                # no gateway asked for: GATEWAY_RAW is not touched
    assert cli(*argv, "--no-usage-gateway")[0] == 0
    for raw in ("1", "0"):
        monkeypatch.setenv("GATEWAY_RAW", raw)               # the caller's own choice reaches the runner as it is
        assert cli(*argv, "--usage-gateway")[0] == 0
        assert C.child_env(calls[-1][1])["GATEWAY_RAW"] == raw
    monkeypatch.setenv("GATEWAY_RAW", "")                    # empty is unset, for the runner too
    assert cli(*argv, "--usage-gateway")[0] == 0
    assert [changes for _, changes in calls] == [
        {**base, "USAGE_GATEWAY": gateway, "GATEWAY_RAW": "0"}, base, {**base, "USAGE_GATEWAY": ""},
        {**base, "USAGE_GATEWAY": gateway}, {**base, "USAGE_GATEWAY": gateway},
        {**base, "USAGE_GATEWAY": gateway, "GATEWAY_RAW": "0"}]
    monkeypatch.delenv("GATEWAY_RAW")
    rc, out, _ = cli(*argv, "--usage-gateway", "--print")
    assert rc == 0 and f"USAGE_GATEWAY={gateway} GATEWAY_RAW=0 " in out.splitlines()[0]
    assert "GATEWAY_RAW=0" in cli("eval", "--help")[1]


def test_eval_without_tasks_runs_every_task(cli, lanes):
    calls, _ = lanes
    assert cli("eval", "b", *REQUIRED)[0] == 0
    bases = calls[0][0][3:]
    assert bases == sorted(t.id for t in roboarc.list_tasks()) and len(bases) == 90
    assert bases == sorted(d.name[:-len("-privileged")] for d in (REPO / "tasks").glob("*-privileged"))


def test_eval_leaves_the_callers_variables_alone(cli, lanes, monkeypatch):
    """A variable the caller exports is not set again: it reaches the runner as it is (collaborators' invocations)."""
    calls, _ = lanes
    for name, value in (("AGENT", "claude-code"), ("MODEL", "anthropic/m"), ("EFFORT", "xhigh"), ("K", "1"),
                        ("USAGE_GATEWAY", "/my/gateway"), ("GATEWAY_RAW", "0")):
        monkeypatch.setenv(name, value)
    assert cli("eval", "b", "orbits-02")[0] == 0
    assert calls == [(RUNNER + ["b", "roboarc-orbits-02-i01"], {})]
    env = C.child_env(calls[0][1])
    assert (env["AGENT"], env["K"], env["USAGE_GATEWAY"], env["GATEWAY_RAW"]) == ("claude-code", "1", "/my/gateway", "0")
    assert cli("eval", "b", "orbits-02", "--trials", "3", "--no-usage-gateway", "--model", "openai/other")[0] == 0
    assert calls[1][1] == {"MODEL": "openai/other", "K": 3, "USAGE_GATEWAY": ""}


@pytest.mark.parametrize("missing", ["--agent", "--model", "--effort"])
def test_eval_needs_agent_model_and_effort(cli, lanes, missing):
    argv = list(REQUIRED)
    del argv[argv.index(missing):argv.index(missing) + 2]
    rc, _, err = cli("eval", "b", "orbits-02", *argv)
    assert rc == 2 and lanes[0] == [] and f"{missing} is required" in err and missing[2:].upper() in err


@pytest.mark.parametrize("bad", [["--modes", "easy"], ["--seeds", "zero"], ["--gpus", "all"], ["--trials", "0"],
                                 ["--lanes", "0"], ["--usage-gateway", "--no-such-option"]], ids=lambda b: b[0])
def test_eval_refuses_values_the_runner_would_misread(cli, lanes, bad):
    assert cli("eval", "b", "orbits-02", *REQUIRED, *bad)[0] == 2 and lanes[0] == []


def test_eval_unknown_task(cli, lanes):
    rc, _, err = cli("eval", "b", "no-such-task", *REQUIRED)
    assert rc == 2 and lanes[0] == [] and "unknown RoboARC task" in err


def test_eval_print_shows_the_command_and_starts_nothing(cli, lanes):
    rc, out, _ = cli("eval", "b", "orbits-02", *REQUIRED, "--seeds", "0,1", "--lanes", "2", "--print")
    assert rc == 0 and lanes[0] == []
    line = out.splitlines()[0]
    assert line == ("AGENT=codex MODEL=openai/m EFFORT=high K=3 SEEDS='0 1' " + " ".join(RUNNER) + " b roboarc-orbits-02-i01")
    assert "2 time(s)" in out.splitlines()[1]


def test_eval_exit_status_is_the_worst_lane(cli, lanes):
    calls, codes = lanes
    codes[:] = [0, 2, 0]
    assert cli("eval", "b", "orbits-02", *REQUIRED, "--lanes", "3")[0] == 2 and len(calls) == 3


# ----------------------------------------------------------------------------------------------------- selftest

SHORTEST = ["battery-replacement-01", "bed-bolt-05", "blanket-flatten-08", "blanket-fold-01", "buzz-wire-3d-04",
            "caliper-sorting-01", "cast-coil-06", "cast-keyhole-02", "clock-assembling-01", "fiber-bundling-02",
            "hedgehog-in-cage-02", "horseshoe-ring-02", "ikea-lack", "interlocked-claws-08", "knot-untying-02",
            "lock-picking-03", "orbits-01", "rope-threading-04", "slotted-post-ring-01", "wire-splicing-01"]


def test_reference_rows_reads_the_array_header():
    import numpy as np
    for alias in ("interlocked-claws-08", "rope-threading-04", "lack"):
        t = roboarc.resolve(alias)
        with np.load(t.reference) as z:
            rows = len(z["actions"] if "actions" in z.files else z["ctrl"])
        assert selftest.reference_rows(t.reference) == rows


def test_selftest_default_is_the_shortest_reference_of_each_family(cli):
    rc, out, _ = cli("selftest", "--print")
    assert rc == 0 and out.startswith("selftest: 80 trials over 20 task(s), 1 at a time")
    tasks = []
    for line in out.splitlines()[1:]:
        name = line.split(": ")[1].split(",")[0].rsplit("-", 1)[0]
        if name not in tasks:
            tasks.append(name)
    assert tasks == [roboarc.resolve(a).id for a in SHORTEST]


def test_selftest_plan(cli):
    rc, out, _ = cli("selftest", "--families", "orbits,ikea", "--modes", "standard", "--agents", "oracle", "--seeds", "0,2",
                     "--lanes", "2", "--gpus", "1,3", "--print")
    assert rc == 0
    assert out.splitlines() == [
        f"selftest: 4 trials over 2 task(s), 2 at a time, on GPU 1,3; jobs and logs in {REPO / 'jobs' / 'selftest'}",
        "  selftest-roboarc-orbits-01-i01-standard-oracle-s0: roboarc-orbits-01-i01-standard, agent oracle, seed 0, expected score 1",
        "  selftest-roboarc-orbits-01-i01-standard-oracle-s2: roboarc-orbits-01-i01-standard, agent oracle, seed 2, expected score 1",
        "  selftest-roboarc-ikea-lack-i00-standard-oracle-s0: roboarc-ikea-lack-i00-standard, agent oracle, seed 0, expected score 1",
        "  selftest-roboarc-ikea-lack-i00-standard-oracle-s2: roboarc-ikea-lack-i00-standard, agent oracle, seed 2, expected score 1"]


@pytest.mark.parametrize("bad", [["orbits-02", "--families", "orbits"], ["--families", "no-such-family"],
                                 ["--modes", "easy"], ["--agents", "codex"], ["--seeds", "x"], ["--gpus", "all"],
                                 ["--lanes", "0"], ["no-such-task"]], ids=lambda b: " ".join(b))
def test_selftest_refuses(cli, started, bad):
    assert cli("selftest", *bad, "--print")[0] == 2 and started == []


@pytest.fixture
def harbor(monkeypatch, tmp_path):
    """Replace the starter with a Harbor that scores every oracle trial 1 and every nop trial 0 (or `scores[agent]`),
    and a materialize that makes the copy's directory."""
    calls, scores = [], {"oracle": "1", "nop": "0"}

    def run(argv, changes=None, **kw):
        calls.append(list(argv))
        if argv[0] == "harbor":
            opt = {flag: argv[argv.index(flag) + 1] for flag in ("-o", "--job-name", "-a")}
            trial = Path(opt["-o"]) / opt["--job-name"] / "trial__abc" / "verifier"
            trial.mkdir(parents=True)
            if scores[opt["-a"]] is not None:
                (trial / "reward.txt").write_text(scores[opt["-a"]] + "\n")
        elif argv[1].endswith("task_seed.py"):
            (Path(argv[argv.index("--out") + 1]) / Path(argv[3]).name).mkdir(parents=True)
        return 0

    monkeypatch.setattr(C, "run", run)
    return calls, scores, tmp_path / "jobs"


def test_selftest_runs_each_trial_through_the_runner_steps(cli, harbor):
    calls, _, jobs = harbor
    rc, out, _ = cli("selftest", "interlocked-claws-08", "--modes", "privileged", "--jobs", jobs, "--gpus", "2")
    assert rc == 0 and "2 of 2 trials have the expected score" in out
    task = "roboarc-interlocked-claws-08-i01-privileged"
    job = f"selftest-{task}-oracle-s0"
    work = jobs / ".selftest" / job
    assert calls[:4] == [
        [sys.executable, script("scripts/task_seed.py"), "materialize", script(f"tasks/{task}"), "--seed", "0", "--out", str(work)],
        [sys.executable, script("scripts/harbor/clamp_cpus.py"), str(work / task)],
        [sys.executable, script("scripts/harbor/pin_gpu.py"), "--gpu", "2", str(work / task)],
        ["harbor", "run", "-p", str(work), "-i", task, "-a", "oracle", "-n", "1", "-k", "1", "-y", "-o", str(jobs),
         "--job-name", job]]
    assert [c[c.index("-a") + 1] for c in calls if c[0] == "harbor"] == ["oracle", "nop"]
    assert not (jobs / ".selftest" / job).exists() and (jobs / f"{job}.log").is_file()      # the copy is removed
    assert [line.split()[1] for line in out.splitlines() if line.startswith("[")] == ["ok", "ok"]


def test_selftest_without_gpus_pins_nothing(cli, harbor):
    calls, _, jobs = harbor
    assert cli("selftest", "lack", "--modes", "standard", "--agents", "nop", "--jobs", jobs)[0] == 0
    assert [Path(c[1]).name if c[0] != "harbor" else "harbor" for c in calls] == ["task_seed.py", "clamp_cpus.py", "harbor"]


def test_selftest_fails_on_an_unexpected_score_and_on_no_score(cli, harbor):
    calls, scores, jobs = harbor
    scores.update(oracle="0", nop=None)
    rc, out, _ = cli("selftest", "orbits-02", "--modes", "standard", "--jobs", jobs)
    assert rc == 1 and "0 of 2 trials have the expected score" in out
    lines = [line for line in out.splitlines() if line.startswith("[")]
    assert "FAIL" in lines[0] and "score 0 (expected 1)" in lines[0]
    assert "FAIL" in lines[1] and "score none (expected 0)" in lines[1] and "no verifier/reward.txt" in lines[1]


def test_selftest_continues_where_it_stopped(cli, harbor):
    calls, scores, jobs = harbor
    argv = ("selftest", "orbits-02", "--modes", "standard", "--jobs", jobs)
    scores["nop"] = None                                     # the nop trial is cut short: a job directory, no score
    assert cli(*argv)[0] == 1
    n = len(calls)
    scores["nop"] = "0"
    rc, out, _ = cli(*argv)
    assert rc == 1 and len(calls) == n                       # nothing runs: one is scored, one must be moved away first
    assert "scored by an earlier run" in out and "move it away" in out
    (jobs / "selftest-roboarc-orbits-02-i01-standard-nop-s0").rename(jobs / "moved-away")
    rc, out, _ = cli(*argv)
    assert rc == 0 and [c[0] for c in calls[n:]].count("harbor") == 1


def test_selftest_reports_a_missing_harbor(cli, monkeypatch, tmp_path):
    def run(argv, changes=None, **kw):
        if argv[0] == "harbor":
            raise FileNotFoundError("harbor")
        return 0
    monkeypatch.setattr(C, "run", run)
    rc, out, _ = cli("selftest", "orbits-02", "--modes", "standard", "--agents", "nop", "--jobs", tmp_path / "jobs")
    assert rc == 1 and "harbor not found (roboarc doctor)" in out


# ------------------------------------------------------------------------------------------------------ results

def test_results_summary_prints_the_runners_table(cli, tmp_path, monkeypatch):
    (tmp_path / "b.summary.md").write_text("# b\n\n| task |\n")
    assert cli("results", "summary", "b", "--jobs", tmp_path) == (0, "# b\n\n| task |\n", "")
    monkeypatch.setenv("JOBS", str(tmp_path))
    assert cli("results", "summary", "b")[1] == "# b\n\n| task |\n"
    rc, out, err = cli("results", "summary", "other")
    assert rc == 1 and out == "" and "other.summary.md" in err


def test_results_send(cli, started):
    assert cli("results", "send", "b", "--name", "alice")[0] == 0
    assert cli("results", "send", "b", "--name", "alice", "--jobs", "j", "--results", "/r", "--env-file", "e.env",
               "--notes", "a reboot", "--create-pr", "--dry-run")[0] == 0
    send = [BASH, script("scripts/send_results.sh"), "b", "alice"]
    assert started == [(send, {}), (send, {"JOBS": "j", "RESULTS": "/r", "ENV_FILE": "e.env", "NOTES": "a reboot",
                                           "CREATE_PR": "1", "DRY_RUN": "1"})]
    assert cli("results", "send", "b")[0] == 2               # --name is required


def test_results_receive(cli, started):
    receive = [BASH, script("scripts/receive_results.sh")]
    delivery = "alice/b/20261121T1830Z"
    assert cli("results", "receive")[0] == 0
    assert cli("results", "receive", delivery, "--list")[0] == 0
    assert cli("results", "receive", delivery, "job-1", "job-2", "--dest", "/d", "--revision", "refs/pr/3")[0] == 0
    assert started == [(receive, {}), (receive + [delivery, "--jobs"], {}),
                       (receive + [delivery, "job-1", "job-2"], {"DEST": "/d", "REVISION": "refs/pr/3"})]
    assert cli("results", "receive", "--list")[0] == 2
    assert cli("results", "receive", delivery, "job-1", "--list")[0] == 2
    assert len(started) == 3


# ---------------------------------------------------------------------------------------------------------- vla

def test_vla_droid_hands_its_arguments_to_the_script_untouched(cli, started):
    argv = ["--task", "lock-picking-03,buzz-wire-3d-06", "--model", "hold", "--gpu", "1", "--jobs", "4", "--",
            "--max-steps", "400", "--help"]
    assert cli("vla", "droid", *argv)[0] == 0
    assert started == [([BASH, script("scripts/vla_droid.sh"), *argv], {})]


@pytest.mark.parametrize("name,module,prog", [("run", "run", "roboarc vla run"), ("stub", "stub_server", "roboarc vla stub")])
def test_vla_run_and_stub_are_their_modules(cli, monkeypatch, name, module, prog):
    import importlib
    seen = []
    mod = importlib.import_module(f"roboarc.vla.{module}")
    monkeypatch.setattr(mod, "main", lambda argv, prog: seen.append((argv, prog)) or 0)
    assert cli("vla", name, "--task", "orbits-02", "--port", "8000")[0] == 0
    assert seen == [(["--task", "orbits-02", "--port", "8000"], prog)]


@pytest.mark.parametrize("name,old", [("run", "python -m roboarc.vla.run"), ("stub", "python -m roboarc.vla.stub_server")])
def test_vla_run_and_stub_help_is_their_modules_help_under_the_new_name(cli, name, old):
    rc, out, _ = cli("vla", name, "--help")
    assert rc == 0 and out.startswith(f"usage: roboarc vla {name} ") and old not in out and "--policy" in out


def test_the_module_commands_keep_their_names(capsys):
    from roboarc.vla import run, stub_server
    for mod, name in ((run, "python -m roboarc.vla.run"), (stub_server, "python -m roboarc.vla.stub_server")):
        with pytest.raises(SystemExit) as e:
            mod.main(["--help"])
        assert e.value.code == 0 and capsys.readouterr().out.startswith(f"usage: {name} ")


def test_vla_without_its_packages_is_one_sentence(cli, monkeypatch):
    monkeypatch.setitem(sys.modules, "roboarc.vla.run", None)
    rc, _, err = cli("vla", "run", "--task", "orbits-02")
    assert rc == 2 and err.startswith("roboarc: vla run: ") and "docs/VLA.md" in err


# ------------------------------------------------------------------------------------------------------- doctor

PINS = [p["ref"] for p in json.loads((REPO / "images/pins.json").read_text())["images"]]
HOST = {"cpu_model": "x", "cpus": 8, "memory_gb": 64, "docker": "Docker version 27.3.1", "harbor": "0.23.0",
        "gpus": [{"index": "0", "name": "NVIDIA L4", "driver": "580.95.05", "memory_mib": 23034}]}


@pytest.fixture
def machine(monkeypatch):
    """A host that passes every check; a test changes `answers` (keyed by the tool asked) to break one. It is a host
    wherever the tests run: inside a RoboARC image `roboarc doctor` would report the graded runtime instead (the last
    test of this section), so the two questions that tell an image are answered too."""
    answers = {"python3": (0, "3.12.9"), "host_info.py": (0, json.dumps(HOST)), "vulkaninfo": (0, " ".join(doctor.VULKAN)),
               "docker info --format {{.ServerVersion}}": (0, "27.3.1"), "docker compose": (0, "2.29.7"),
               "docker info --format {{json .Runtimes}}": (0, '{"nvidia":{"path":"nvidia-container-runtime"},"runc":{}}'),
               "nvidia-ctk": (None, "nvidia-ctk not found"), "pull_all.sh": (0, "\n".join(f"{r} sha256:{'0' * 64}" for r in PINS)),
               "docker image inspect": (0, "sha256:" + "0" * 64)}

    def capture(argv, timeout=60):
        line = " ".join(argv)
        # the longest key first: a script of the checkout is started with this interpreter, whose own path ends in
        # `python3` in a virtual environment and on the CI runner, and must still answer as that script
        for key in sorted(answers, key=len, reverse=True):
            if key in line:
                return answers[key]
        raise AssertionError(f"doctor ran an unexpected command: {line}")

    monkeypatch.setattr(C, "capture", capture)
    monkeypatch.setattr(doctor.shutil, "which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(_runtime, "in_puzzle_image", lambda: False)
    monkeypatch.setattr(_runtime, "in_ikea_image", lambda: False)
    return answers


def statuses(out: str) -> dict:
    return {c["name"]: c["status"] for c in json.loads(out)["checks"]}


def test_doctor_on_a_host_that_has_everything(cli, machine):
    rc, out, _ = cli("doctor")
    assert rc == 0 and out.splitlines()[0] == f"RoboARC {roboarc.__version__}, this host"
    assert out.splitlines()[-1] == "doctor: every check passed" and "FAIL" not in out
    rc, out, _ = cli("doctor", "--json")
    report = json.loads(out)
    assert rc == 0 and report["ok"] is True and report["where"] == "host" and report["roboarc"] == roboarc.__version__
    assert statuses(out) == dict.fromkeys(["python3", "repository", "GPU", "NVIDIA driver", "Vulkan ray query", "Docker",
                                           "Docker Compose", "NVIDIA Container Toolkit", "Harbor", "pinned images"], "ok")
    zero = "sha256:" + "0" * 64
    assert report["checks"][-1]["images"] == [{"ref": r, "pinned_id": zero, "local_id": zero} for r in PINS]


@pytest.mark.parametrize("tool,answer,check,fix", [
    ("python3", (0, "3.10.12"), "python3", "3.11 or newer"),
    ("host_info.py", (0, json.dumps({**HOST, "gpus": []})), "GPU", "NVIDIA driver"),
    ("host_info.py", (0, json.dumps({**HOST, "gpus": [{**HOST["gpus"][0], "driver": "570.133.20"}]})), "NVIDIA driver", "580 or newer"),
    ("host_info.py", (0, json.dumps({**HOST, "harbor": "0.22.4"})), "Harbor", "0.23 or newer"),
    ("host_info.py", (0, json.dumps({**HOST, "harbor": None})), "Harbor", "install Harbor"),
    ("vulkaninfo", (0, "VK_KHR_ray_query VK_EXT_extended_dynamic_state3"), "Vulkan ray query", "RT cores"),
    ("docker info --format {{.ServerVersion}}", (1, "permission denied while trying to connect"), "Docker", "group `docker`"),
    ("docker info --format {{.ServerVersion}}", (None, "docker not found"), "Docker", "install Docker"),
    ("docker compose", (1, "docker: 'compose' is not a docker command"), "Docker Compose", "Compose plugin"),
    ("docker info --format {{json .Runtimes}}", (0, '{"runc":{}}'), "NVIDIA Container Toolkit", "NVIDIA Container Toolkit"),
    ("docker image inspect", (1, "Error: No such image"), "pinned images", "roboarc images pull"),
], ids=lambda v: v if isinstance(v, str) and " " not in v and len(v) < 30 else None)
def test_doctor_names_what_fails_and_how_to_fix_it(cli, machine, tool, answer, check, fix):
    machine[tool] = answer
    rc, out, _ = cli("doctor", "--json")
    report = json.loads(out)
    failed = {c["name"]: c for c in report["checks"] if c["status"] == "FAIL"}
    assert rc == 1 and report["ok"] is False and list(failed) == [check] and fix in failed[check]["fix"]
    rc, out, _ = cli("doctor")
    assert rc == 1 and "1 failed" in out.splitlines()[-1] and f"-> {failed[check]['fix']}" in out


def test_doctor_does_not_fail_what_it_cannot_check(cli, machine, monkeypatch):
    monkeypatch.setattr(doctor.shutil, "which", lambda name: None)          # no vulkaninfo on this host
    rc, out, _ = cli("doctor")
    assert rc == 0 and out.splitlines()[-1] == "doctor: 1 could not be made" and "vulkan-tools" in out
    rc, out, _ = cli("doctor", "--json")
    assert statuses(out)["Vulkan ray query"] == "?" and json.loads(out)["ok"] is True


def test_doctor_without_docker_looks_for_no_image(cli, machine):
    machine["docker info --format {{.ServerVersion}}"] = (None, "docker not found")
    machine["docker image inspect"] = AssertionError         # must not be asked
    rc, out, _ = cli("doctor", "--json")
    assert rc == 1 and statuses(out)["pinned images"] == "?" and statuses(out)["NVIDIA Container Toolkit"] == "?"


def test_doctor_inside_an_image_reports_the_graded_runtime(cli, monkeypatch):
    from roboarc.cli import tasks
    report = {"libmujoco_sha256": _runtime.LIBMUJOCO_EPA_SHA256, "libmujoco_epa_patched": True, "numpy_masking": True,
              "puzzle_image": True, "ikea_image": False, "nyx_python (ikea -sim image)": False, "repo": "/repo", "problems": []}
    monkeypatch.setattr(_runtime, "in_puzzle_image", lambda: True)
    monkeypatch.setattr(tasks, "runtime_report", lambda: report)
    monkeypatch.setattr(C, "capture", lambda *a, **k: pytest.fail("no host check inside an image"))
    rc, out, _ = cli("doctor")
    assert rc == 0 and "inside a RoboARC image" in out and "graded runtime" in out and "FAIL" not in out
    rc, out, _ = cli("doctor", "--json")
    assert json.loads(out)["where"] == "image" and json.loads(out)["runtime"] == report
    report["problems"] = ["libmujoco.so is not MuJoCo 3.14.0 with the EPA patch"]
    rc, out, _ = cli("doctor")
    assert rc == 1 and "FAIL  graded runtime" in out and "EPA patch" in out


# ------------------------------------------------------------------------- end to end, with stand-in scripts

STAND_IN = """#!/usr/bin/env bash
exec "$FAKE_PYTHON" - "$@" <<'PY'
import json, os, sys
record = {"argv": sys.argv[1:], "env": {k: os.environ.get(k) for k in os.environ["FAKE_KEEP"].split()}}
with open(os.path.join(os.environ["FAKE_LOG"], f"{os.getpid()}.json"), "w") as f:
    json.dump(record, f)
sys.exit(int(os.environ.get("FAKE_EXIT", "0")))
PY
"""
KEEP = ["AGENT", "MODEL", "EFFORT", "K", "MODES", "SEEDS", "USAGE_GATEWAY", "GATEWAY_RAW", "CODEX_FORCE_AUTH_JSON",
        "NPY_DISABLE_CPU_FEATURES", "ROBOARC_ACCEPT_LICENCES"]


@pytest.fixture
def checkout(tmp_path, monkeypatch):
    """A checkout with one task in its catalogue and stand-ins for the batch runner and images/pull_all.sh; records()
    returns how each stand-in was started."""
    fake, log = tmp_path / "repo", tmp_path / "log"
    log.mkdir()
    for rel in ("scripts/roboarc/shared/rcb_topogym.py", "LICENSES/THIRD_PARTY.md"):
        (fake / rel).parent.mkdir(parents=True, exist_ok=True)
        (fake / rel).write_text("")
    for rel in (evaluate.RUNNER, images.SCRIPTS["pull"][0]):
        (fake / rel).parent.mkdir(parents=True, exist_ok=True)
        (fake / rel).write_text(STAND_IN)
    for mode in ("privileged", "standard"):                  # what the catalogue reads of a task
        for rel in ("task.toml", "tests/test.sh", "tests/Dockerfile", "environment/Dockerfile"):
            src = REPO / "tasks" / f"roboarc-orbits-02-i01-{mode}" / rel
            dst = fake / "tasks" / f"roboarc-orbits-02-i01-{mode}" / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_text(src.read_text())
    for name, value in (("ROBOARC_REPO", fake), ("FAKE_LOG", log), ("FAKE_PYTHON", sys.executable), ("FAKE_KEEP", " ".join(KEEP))):
        monkeypatch.setenv(name, str(value))
    return lambda: [json.loads(f.read_text()) for f in sorted(log.glob("*.json"))]


def test_eval_end_to_end_starts_the_runner_once_per_lane(cli, checkout, monkeypatch):
    monkeypatch.setenv("GATEWAY_RAW", "0")                   # two of the runner's variables, exported by the caller
    monkeypatch.setenv("CODEX_FORCE_AUTH_JSON", "1")
    rc, _, err = cli("eval", "mybatch", *REQUIRED, "--lanes", "2", "--modes", "standard", "--no-usage-gateway")
    assert rc == 0 and "2 lanes started" in err
    added = "NPY_DISABLE_CPU_FEATURES" in _runtime.ENV_ADDED        # set by importing roboarc, not by the caller
    want = {"AGENT": "codex", "MODEL": "openai/m", "EFFORT": "high", "K": "3", "MODES": "standard", "SEEDS": None,
            "USAGE_GATEWAY": "", "GATEWAY_RAW": "0", "CODEX_FORCE_AUTH_JSON": "1", "ROBOARC_ACCEPT_LICENCES": None,
            "NPY_DISABLE_CPU_FEATURES": None if added else os.environ.get("NPY_DISABLE_CPU_FEATURES")}
    assert checkout() == [{"argv": ["mybatch", "roboarc-orbits-02-i01"], "env": want}] * 2
    monkeypatch.setenv("FAKE_EXIT", "2")                     # the runner refuses a setting: its status is the command's
    assert cli("eval", "mybatch", "orbits-02", *REQUIRED)[0] == 2


def test_images_pull_end_to_end(cli, checkout, monkeypatch):
    no_terminal(monkeypatch)
    assert cli("images", "pull", "--pins", "/p/pins.json")[0] == 2 and checkout() == []          # no consent yet
    assert cli("images", "pull", "--pins", "/p/pins.json", "--accept-licences")[0] == 0
    assert [(r["argv"], r["env"]["ROBOARC_ACCEPT_LICENCES"]) for r in checkout()] == [(["--pins", "/p/pins.json"], "1")]
    monkeypatch.setenv("FAKE_EXIT", "1")                     # an image differs from its pin: the script's status
    assert cli("images", "pull")[0] == 1


STAND_IN_HARBOR = """#!{python}
import json, os, pathlib, sys
a = sys.argv[1:]
get = lambda flag: a[a.index(flag) + 1]
copy = pathlib.Path(get("-p")) / get("-i")
compose = (copy / "environment" / "docker-compose.yaml").read_text()
record = {{"argv": a, "task_toml": (copy / "task.toml").is_file(), "seed": (copy / "tests" / "seed").read_text(),
          "devices": [line.strip() for line in compose.splitlines() if "device_ids" in line or "count:" in line]}}
with open(os.path.join(os.environ["FAKE_LOG"], get("--job-name") + ".json"), "w") as f:
    json.dump(record, f)
verifier = pathlib.Path(get("-o")) / get("--job-name") / "trial__x" / "verifier"
verifier.mkdir(parents=True)
(verifier / "reward.txt").write_text("1" if get("-a") == "oracle" else "0")
"""


def test_selftest_end_to_end_with_the_repository_scripts_and_a_stand_in_harbor(cli, tmp_path, monkeypatch):
    """The real scripts/task_seed.py, clamp_cpus.py and pin_gpu.py take the arguments selftest gives them, and Harbor is
    handed a copy of the task at the seed, pinned to the GPU (the smallest standard task: under 1 MB)."""
    (tmp_path / "bin").mkdir()
    (tmp_path / "log").mkdir()
    harbor = tmp_path / "bin" / "harbor"
    harbor.write_text(STAND_IN_HARBOR.format(python=sys.executable))
    harbor.chmod(0o755)
    monkeypatch.setenv("PATH", f"{tmp_path / 'bin'}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setenv("FAKE_LOG", str(tmp_path / "log"))
    jobs = tmp_path / "jobs"
    rc, out, _ = cli("selftest", "slotted-post-ring-01", "--modes", "standard", "--seeds", "1", "--gpus", "3", "--jobs", jobs)
    assert rc == 0 and "2 of 2 trials have the expected score" in out, out
    task = "roboarc-slotted-post-ring-01-i01-standard"
    for agent in ("oracle", "nop"):
        job = f"selftest-{task}-{agent}-s1"
        seen = json.loads((tmp_path / "log" / f"{job}.json").read_text())
        assert seen["argv"] == ["run", "-p", str(jobs / ".selftest" / job), "-i", task, "-a", agent, "-n", "1", "-k", "1",
                                "-y", "-o", str(jobs), "--job-name", job]
        assert seen["task_toml"] and seen["seed"] == "1\n"
        assert seen["devices"] and set(seen["devices"]) == {'device_ids: ["3"]'}
    assert not (jobs / ".selftest").exists()                 # the copies are gone
    assert "pin_gpu.py" in (jobs / f"selftest-{task}-oracle-s1.log").read_text()
