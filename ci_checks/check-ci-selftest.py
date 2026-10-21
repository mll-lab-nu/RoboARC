#!/usr/bin/env python3
"""The checks check something: mutation self-test of the rules that guard the trust boundary and the resource caps.

A rule that reads compose files or task.toml by pattern can go blind without anyone noticing: the trailing comment that
scripts/rcb_monitor/inject.py writes after `volumes:` once hid every volume of the agent's container from
check-verifier-isolation.sh, and all checks stayed green on a task whose agent container mounted the simulator's trusted
volume (found in a review). This script copies a few real tasks into a temporary directory, breaks each copy in one
way a rule exists for, and requires that the rule's check fails with the rule's message -- and that the unbroken copy
passes. A mutation that passes is a blind rule: fix the check, not this list.

    python3 ci_checks/check-ci-selftest.py [--tier invariants|advisory|all]    # default: all. Standard library only.

Two tiers. `--tier invariants` (the required CI job `invariants`, ci_checks/run_invariants.sh): the
mutations of the rules that enforce docs/PROTOCOL.md Appendix H; each must FAIL its check. `--tier advisory` (the job
`advisories`): the mutations of the advisory rules; each must still be reported (a finding, not blocking).
"""
from __future__ import annotations

import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ISOLATION = ["bash", str(ROOT / "ci_checks/check-verifier-isolation.sh")]
RESOURCES = [sys.executable, str(ROOT / "ci_checks/check-resources.py")]
POINTERS = [sys.executable, str(ROOT / "ci_checks/check-agent-pointers.py")]
IDENTITY = ["bash", str(ROOT / "ci_checks/check-tooling-identity.sh")]   # repo-level: runs on a tasks directory (see main)
CUTOFFS = [sys.executable, str(ROOT / "ci_checks/check-cutoffs.py")]
SEEDS = [sys.executable, str(ROOT / "ci_checks/check-seeds.py")]
VIDEO = [sys.executable, str(ROOT / "ci_checks/check-video.py")]
TWINS = [sys.executable, str(ROOT / "ci_checks/check-twins.py")]              # repo-level-ish: reads the twin next to it
INSTANCE = ["bash", str(ROOT / "ci_checks/check-instance-copies.sh")]
STDIMG = ["bash", str(ROOT / "ci_checks/check-standard-agent-image.sh")]
MONITOR = [sys.executable, str(ROOT / "ci_checks/check-monitor.py"), "--tier", "invariants"]
DIGESTS = ["bash", str(ROOT / "ci_checks/check-dataset-manifest.sh")]
DOCKERFILE = ["bash", str(ROOT / "ci_checks/check-dockerfile.sh")]
PINS = [sys.executable, str(ROOT / "ci_checks/check-image-pins.py")]   # its own CI job (image-pins); advisory tier here
FIELDS = ["bash", str(ROOT / "ci_checks/check-task-fields.sh")]
# the checks that enforce Appendix H (run_invariants.sh); every other check here is advisory
INVARIANT_CMDS = (ISOLATION, POINTERS, TWINS, INSTANCE, STDIMG, MONITOR, DIGESTS)
WITH_NEIGHBOURS = (IDENTITY, TWINS)   # run on the broken copy among (links to) every other task

# the fixtures: real RoboARC tasks, copied and broken one way each
STD = "roboarc-orbits-02-i01-standard"         # a standard task: the agent container and the sim sidecar
PRIV = "roboarc-orbits-02-i01-privileged"      # its privileged twin
HEAVY = "roboarc-knot-untying-07-i01-standard"  # an allowlist; a GPU sidecar


def sub(pattern: str, repl: str, count: int = 1, flags: int = re.M):
    def f(s: str) -> str:
        out, n = re.subn(pattern, repl, s, count=count, flags=flags)
        if n == 0:
            raise AssertionError(f"mutation does not apply: /{pattern}/")
        return out
    return f


def in_service(service: str, lines: str, after: str = ""):
    """Insert `lines` into a compose service: right after its line matching `after` (default: the service line)."""
    def f(s: str) -> str:
        rows = s.splitlines(keepends=True)
        i = next(k for k, r in enumerate(rows) if re.match(rf"^  {service}:\s*(#.*)?$", r))
        j = next((k for k in range(i + 1, len(rows)) if re.match(r"^  \S|^\S", rows[k]) and not rows[k].startswith("  #")), len(rows))
        at = i if not after else next(k for k in range(i, j) if re.match(after, rows[k]))
        rows[at + 1:at + 1] = [x + "\n" for x in lines.split("\n")]
        return "".join(rows)
    return f


def append(text: str):
    return lambda s: s + text


def npz_text(member: str, text: str) -> bytes:
    """A numpy archive (no numpy needed) whose one member is a 0-d unicode array holding `text`, as np.savez writes it."""
    import io, zipfile
    head = "{'descr': '<U%d', 'fortran_order': False, 'shape': (), }" % len(text)
    head += " " * (-(10 + len(head) + 1) % 64) + "\n"
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr(member, b"\x93NUMPY\x01\x00" + len(head).to_bytes(2, "little") + head.encode("latin-1")
                   + text.encode("utf-32-le"))
    return buf.getvalue()


COMPOSE, TOML, TEST_COMPOSE = "environment/docker-compose.yaml", "task.toml", "tests/docker-compose.yaml"
TEST_SH, TEST_DF = "tests/test.sh", "tests/Dockerfile"
VOL = r"^    volumes:.*$"
VAR = "a variable in a volume source must be the whole source"
NO_TRAP = sub(r"^(trap .*EXIT.*\n)", r"\1trap - EXIT\n")       # the resource step's trap dropped again one line later
# (name, task, file, mutation, check, what the check must say)
MUTATIONS = [
    # ---- the agent's container and the trusted volumes (the M5 class: every way of writing the same mount)
    ("the agent container mounts the sim's trusted volume (short form, under the generated `volumes:   # rcb-monitor`)",
     STD, COMPOSE, in_service("main", "      - rcb_sim_cache_trusted:/cache", VOL), ISOLATION, "mounts trusted volume"),
    ("... long form (type: volume / source:)",
     STD, COMPOSE, in_service("main", "      - type: volume\n        source: rcb_sim_cache_trusted\n        target: /cache", VOL),
     ISOLATION, "mounts trusted volume"),
    ("... long form, source first",
     STD, COMPOSE, in_service("main", "      - source: rcb_sim_cache_trusted\n        target: /cache\n        type: volume", VOL),
     ISOLATION, "mounts trusted volume"),
    ("... quoted",
     STD, COMPOSE, in_service("main", '      - "rcb_sim_cache_trusted:/cache:ro"', VOL), ISOLATION, "mounts trusted volume"),
    ("... with the service line and the volumes line carrying other comments",
     STD, COMPOSE, lambda s: in_service("main", "      - rcb_sim_cache_trusted:/cache   # cache", r"^    volumes:.*$")(
         sub(r"^  main:\s*$", "  main:   # the agent's container")(s)), ISOLATION, "mounts trusted volume"),
    ("... through volumes_from",
     STD, COMPOSE, in_service("main", "    volumes_from: [sim]"), ISOLATION, "`volumes_from` is not allowed"),
    ("an inline volumes list (which the reader would have to guess at) with the trusted volume",
     STD, COMPOSE, lambda s: sub(r"^      - rcb_limits:/run/rcb-limits:ro.*\n", "")(
         sub(r"^  main:\n((?:    .*\n)*?)    volumes:.*$", r"  main:\n\1    volumes: [rcb_limits:/run/rcb-limits:ro, rcb_sim_cache_trusted:/cache]")(s)),
     ISOLATION, "mounts trusted volume"),
    ("a volume entry the reader does not understand is a failure, not skipped",
     STD, COMPOSE, in_service("main", "      - *shared_cache", VOL), ISOLATION, "not understood"),
    ("a sidecar mounts a volume that is not *_trusted",
     STD, COMPOSE, in_service("sim", "      - shared_scratch:/scratch", VOL), ISOLATION, "which is not a *_trusted volume"),
    # ---- the agent's container reaches the task's trusted files or the host
    ("the agent container bind-mounts the sidecar's code",
     STD, COMPOSE, in_service("main", "      - ./sim:/sim:ro", VOL), ISOLATION, "bind-mounts ./sim"),
    ("... the verifier's tests", STD, COMPOSE, in_service("main", "      - ../tests:/t:ro", VOL), ISOLATION, "bind-mounts ../tests"),
    ("... the reference solution", STD, COMPOSE, in_service("main", "      - ../solution:/s:ro", VOL), ISOLATION, "bind-mounts ../solution"),
    ("... the monitor's code", STD, COMPOSE, in_service("main", "      - ./rcb_monitor:/m", VOL), ISOLATION, "bind-mounts ./rcb_monitor"),
    ("... the docker socket",
     STD, COMPOSE, in_service("main", "      - /var/run/docker.sock:/var/run/docker.sock", VOL), ISOLATION, "bind-mounts /var/run/docker.sock"),
    ("the agent container is privileged", STD, COMPOSE, in_service("main", "    privileged: true"), ISOLATION, "`privileged` is not allowed"),
    ("the agent container shares the host's PID namespace", STD, COMPOSE, in_service("main", "    pid: host"), ISOLATION, "`pid` is not allowed"),
    ("the agent container mounts the beacon volume writable at another path",
     STD, COMPOSE, in_service("main", "      - rcb_limits:/mnt/beacon", VOL), ISOLATION, "mounts the beacon volume rcb_limits writable"),
    # ---- the sidecar is reachable only on localhost
    ("the sidecar publishes a port", STD, COMPOSE, in_service("sim", '    ports: ["8000:8000"]'), ISOLATION, "`ports` is not allowed"),
    ("the sidecar is on the host's network", STD, COMPOSE, in_service("sim", "    network_mode: host"), ISOLATION, "`network_mode` is not allowed"),
    # ---- the verifier's evidence comes from the sidecar
    ("the evidence is collected from the agent's container",
     STD, TOML, sub(r'\{ source = "(/(?:data|out)/final\.json)", service = "sim" \}', r'"\1"'), ISOLATION,
     "is collected from the agent's container"),
    ("the sidecar's collect hook is gone",
     STD, TOML, sub(r'(?ms)^\[\[verifier\.collect\]\]\nservice = "sim"\n.*?(?=^\[|\Z)', ""), ISOLATION, "no [[verifier.collect]] hook on a sidecar"),
    # ---- the verifier's own sandbox, the allowlist
    ("the verifier shares the agent's container", STD, TOML, sub(r'^environment_mode = "separate"', 'environment_mode = "shared"'),
     ISOLATION, 'environment_mode must be "separate"'),
    ("the allowlist lacks a model API's host",
     HEAVY, TOML, sub(r', "\*\.openai\.azure\.com"', ""), ISOLATION, "allowed_hosts differs from scripts/set_agent_egress.py"),
    ("the allowlist opens another host",
     HEAVY, TOML, sub(r'allowed_hosts = \["chatgpt\.com"', 'allowed_hosts = ["github.com", "chatgpt.com"'), ISOLATION,
     "allowed_hosts differs from scripts/set_agent_egress.py"),
    # ---- a service with its own network setting is outside Harbor's egress control (the allowlist)
    ("the agent container on a network of its own (`networks:`): the allowlist no longer applies to it",
     HEAVY, COMPOSE, in_service("main", "    networks: [default]"), ISOLATION, "service main: `networks` is not allowed"),
    ("... `network_mode: bridge`", HEAVY, COMPOSE, in_service("main", "    network_mode: bridge"), ISOLATION, "service main: `network_mode` is not allowed"),
    ("a sidecar on a network of its own", STD, COMPOSE, in_service("sim", "    networks: [default]"), ISOLATION, "service sim: `networks` is not allowed"),
    ("a top-level networks block", STD, COMPOSE, append("networks:\n  default:\n    driver: bridge\n"), ISOLATION, "top-level `networks` is not allowed"),
    ("a task drops its allowlist (PROTOCOL §2.11: one host list, in every task)",
     HEAVY, TOML, sub(r'^network_mode = "allowlist".*\nallowed_hosts = .*\n', ""), ISOLATION, '[agent].network_mode must be "allowlist"'),
    ("the instruction no longer states the external-material rule",
     STD, "instruction.md", sub(r"(?ms)^## External material\n.*?(?=^## )", ""), RESOURCES, "does not state `## External material`"),
    ("... in a string member of a numpy archive the agent reads (an instance archive's `meta`)",
     PRIV, "+environment/extra.npz", lambda s: npz_text("meta.npy", '{"source": "x", "demo": "water_plant_demo_10_2026"}'),
     POINTERS, "environment/extra.npz[meta.npy]:1: id `demo`"),
    ("a step with a network policy of its own (Harbor would let it override [agent])",
     STD, TOML, append('\n[[steps]]\nname = "s1"\n[steps.agent]\nnetwork_mode = "public"\n'), ISOLATION, "a step sets its own network policy"),
    ("the instruction links the upstream repository",
     STD, "instruction.md", append("\nThe benchmark's code: https://github.com/example-org/upstream-benchmark\n"), POINTERS,
     "URL `https://github.com/example-org/upstream-benchmark` locates upstream material"),
    ("the instruction states the external-material rule in other words",
     STD, "instruction.md", sub(r"This is an honour rule; trials are reviewed\.", "Do as you like."), RESOURCES,
     "does not state `## External material`"),
    # ---- the agent's time cutoff (PROTOCOL §2.12): the task's own number; task.toml, the instruction and the sim service agree
    ("task.toml's cutoff changed without normalizing (the instruction still states the old number)",
     STD, TOML, sub(r"(^\[agent\]\n(?:(?!\[).*\n)*?timeout_sec = )[0-9.]+", r"\g<1>1234.0"), CUTOFFS,
     "does not end with the `## Time limit` section rendered for its 1234 s"),
    ("a cutoff above the 14400 s cap",
     STD, TOML, sub(r"(^\[agent\]\n(?:(?!\[).*\n)*?timeout_sec = )[0-9.]+", r"\g<1>28800.0"), CUTOFFS, "is above the 14400 s cap"),
    ("a cutoff that is not a whole number of seconds",
     PRIV, TOML, sub(r"(^\[agent\]\n(?:(?!\[).*\n)*?timeout_sec = )[0-9.]+", r"\g<1>3600.5"), CUTOFFS,
     "is not a whole positive number of seconds"),
    ("a task.toml without the agent's cutoff",
     PRIV, TOML, sub(r"(^\[agent\]\n(?:(?!\[).*\n)*?)timeout_sec = [0-9.]+[^\n]*\n", r"\g<1>"), CUTOFFS,
     "has no `[agent] timeout_sec`"),
    ("the instruction's cutoff sentence with another number",
     STD, "instruction.md", sub(r"^You have \d+ seconds to complete this task\.", "You have 28800 seconds to complete this task."),
     CUTOFFS, "does not end with the `## Time limit` section"),
    ("the instruction without the cutoff sentence",
     PRIV, "instruction.md", sub(r"^You have \d+ seconds to complete this task\.[^\n]*\n\n", ""), CUTOFFS, "the sentence is missing"),
    ("the cutoff section is not the end of the instruction: something follows it",
     PRIV, "instruction.md", append("\nOne more remark after the time limit.\n"), CUTOFFS, "does not end with the `## Time limit` section"),
    ("the cutoff paragraphs under the title, where they stood until 2026-10-12, instead of at the end",
     PRIV, "instruction.md",
     lambda s: (lambda head, sep, block: head.split("\n", 1)[0] + "\n\n" + block.split("\n", 1)[1] + "\n" + head.split("\n", 1)[1])(*s.partition("\n## Time limit\n"))
     if "\n## Time limit\n" in s else (_ for _ in ()).throw(AssertionError("no `## Time limit` section")),
     CUTOFFS, "does not end with the `## Time limit` section"),
    ("a privileged task with the standard modes' second paragraph",
     PRIV, "instruction.md", sub(r"only what is in `/app/output/` at that moment is\s+graded", "the episode is graded as it stands"),
     CUTOFFS, "does not end with the `## Time limit` section"),
    ("the closing sections in another order (`## Time limit` before `## Resource limits`)",
     STD, "instruction.md",
     lambda s: (lambda a, b, c: a + "\n## Time limit\n" + c.rstrip("\n") + "\n\n## Resource limits\n" + b.rstrip("\n") + "\n")(
         s.split("\n## Resource limits\n")[0], *s.split("\n## Resource limits\n")[1].split("\n## Time limit\n")),
     CUTOFFS, "does not end with the `## Time limit` section"),
    ("... which the resources check refuses too (the limits paragraph is the last thing before `## Time limit`)",
     STD, "instruction.md",
     lambda s: (lambda a, b, c: a + "\n## Time limit\n" + c.rstrip("\n") + "\n\n## Resource limits\n" + b.rstrip("\n") + "\n")(
         s.split("\n## Resource limits\n")[0], *s.split("\n## Resource limits\n")[1].split("\n## Time limit\n")),
     RESOURCES, "does not end (before `## Time limit`) with the `## Resource limits` paragraph"),
    ("the instruction states the limit a second time, in a benchmark's old words",
     STD, "instruction.md", sub(r"^(## External material)$", r"Your wall-clock budget is 4 h (14400 s); there is no step budget.\n\n\1"),
     CUTOFFS, "states the agent's limit a second time"),
    ("... 'You have 60 minutes of wall-clock time'",
     PRIV, "instruction.md", sub(r"^(## External material)$", r"- You have 60 minutes of wall-clock time.\n\n\1"),
     CUTOFFS, "states the agent's limit a second time"),
    ("... in words nobody has used yet: 'You have one hour.'",
     STD, "instruction.md", sub(r"^(## External material)$", r"You have one hour.\n\n\1"), CUTOFFS, "states the agent's limit a second time"),
    ("... 'The limit is 3600 seconds.'",
     PRIV, "instruction.md", sub(r"^(## External material)$", r"The limit is 3600 seconds.\n\n\1"), CUTOFFS, "states the agent's limit a second time"),
    ("... 'Finish within 60 minutes.'",
     STD, "instruction.md", sub(r"^(## External material)$", r"Finish within 60 minutes.\n\n\1"), CUTOFFS, "states the agent's limit a second time"),
    ("the README's `Agent time budget` row with another number than the task's cutoff",
     STD, "README.md", sub(r"^(\| Agent time budget \| )\d+ s;", r"\g<1>28800 s;"), CUTOFFS, "README.md's `Agent time budget` row does not say"),
    ("the sim service is told another wall clock than the task's cutoff",
     STD, COMPOSE, sub(r'(RCB_WALL_CLOCK_S: ")\d+', r"\g<1>3600"), CUTOFFS, "RCB_WALL_CLOCK_S = 3600, the task's cutoff is"),
    # ---- a variable in a volume source, the top level of the file, files by another road
    ("the sim's trusted volume as the default of a variable",
     STD, COMPOSE, in_service("main", "      - ${RCB_CACHE_VOL:-rcb_sim_cache_trusted}:/cache", VOL), ISOLATION, VAR),
    ("the sidecar's code as the default of a variable", STD, COMPOSE, in_service("main", "      - ${RCB_SIM_DIR:-./sim}:/sim:ro", VOL), ISOLATION, VAR),
    ("... `${X-../tests}`", STD, COMPOSE, in_service("main", "      - ${X-../tests}:/t:ro", VOL), ISOLATION, VAR),
    ("a path after a variable", STD, COMPOSE, in_service("main", "      - ${PWD}/sim:/sim:ro", VOL), ISOLATION, VAR),
    ("a variable that is not the runner's (Harbor's own: the task's environment directory)",
     STD, COMPOSE, in_service("main", "      - ${CONTEXT_DIR}:/ctx:ro", VOL), ISOLATION, VAR),
    ("a variable inside a volume's name", STD, COMPOSE, in_service("main", "      - rcb_${B:-sim}_cache_trusted_${E:-}x:/cache", VOL), ISOLATION, VAR),
    ("... long form",
     STD, COMPOSE, in_service("main", "      - type: volume\n        source: ${V:-rcb_sim_cache_trusted}\n        target: /cache", VOL), ISOLATION, VAR),
    ("... inline list",
     STD, COMPOSE, lambda s: sub(r"^      - rcb_limits:/run/rcb-limits:ro.*\n", "")(
         sub(r"^  main:\n((?:    .*\n)*?)    volumes:.*$", r"  main:\n\1    volumes: [rcb_limits:/run/rcb-limits:ro, '${V:-rcb_sim_cache_trusted}:/cache']")(s)),
     ISOLATION, VAR),
    ("... `external:`", STD, COMPOSE, lambda s: in_service("main", "      - agent_scratch:/cache", VOL)(s) + "  agent_scratch:\n    external: true\n",
     ISOLATION, "is not a bare name"),
    ("a top-level volume that binds the sidecar's code (driver_opts)",
     STD, COMPOSE, lambda s: in_service("main", "      - agent_view:/sim", VOL)(s) + "  agent_view:\n    driver_opts:\n      type: none\n      o: bind\n      device: ${PWD}/sim\n",
     ISOLATION, "is not a bare name"),
    ("the reference solution as a secret of the agent container",
     STD, COMPOSE, lambda s: in_service("main", "    secrets: [sol]")(s) + "secrets:\n  sol:\n    file: ../solution/solve.sh\n", ISOLATION, "service main: `secrets` is not allowed"),
    ("the sidecar's code as a config of the agent container",
     STD, COMPOSE, lambda s: in_service("main", "    configs:\n      - source: simcode\n        target: /sim_server.py")(s) + "configs:\n  simcode:\n    file: ./sim/server.py\n",
     ISOLATION, "top-level `configs` is not allowed"),
    ("a task file read through env_file", STD, COMPOSE, in_service("main", "    env_file: ./sim/instance.json"), ISOLATION, "service main: `env_file` is not allowed"),
    ("the agent's image built from the task directory", STD, COMPOSE, in_service("main", "    build:\n      context: .."), ISOLATION, "service main: `build` is not allowed"),
    ("another compose file included", STD, COMPOSE, lambda s: "include:\n  - extra.yaml\n" + s, ISOLATION, "top-level `include` is not allowed"),
    ("a service whose keys are indented deeper than the reader looks",
     STD, COMPOSE, sub(r"(?s)^  main:\n.*?(?=^  sim:)", "  main:\n      networks: [default]\n      depends_on: [sim]\n"), ISOLATION, "at an unexpected indentation"),
    ("the kernel's OOM killer switched off for a container", STD, COMPOSE, in_service("sim", "    oom_kill_disable: true"), ISOLATION, "`oom_kill_disable` is not allowed"),
    # ---- pinned images in every stage and file; the verifier's log directory emptied, not a look-alike of it
    ("a second, unpinned stage in the verifier's Dockerfile", STD, TEST_DF, append("FROM python:latest\nCOPY --from=0 /tests /tests\n"), DOCKERFILE, "must be a pinned registry image"),
    # ---- every image by digest (images/pins.json)
    ("the simulator image by its tag only", PRIV, "environment/Dockerfile", sub(r"^(FROM \S+)@sha256:[0-9a-f]{64}", r"\1"),
     PINS, "not pinned by digest"),
    ("the sim sidecar's image by its tag only", STD, COMPOSE, sub(r"^(    image: \S+/roboarc:\S+)@sha256:[0-9a-f]{64}", r"\1"),
     PINS, "not pinned by digest"),
    ("RoboARC's image by another digest", PRIV, TEST_DF, sub(r"@sha256:[0-9a-f]{64}", "@sha256:" + "0" * 64),
     PINS, "not a pin of images/pins.json"),
    ("the client base from another repository", STD, TEST_DF, sub(r"^FROM docker\.io/library/python", "FROM docker.io/someone/python"),
     PINS, "outside ghcr.io/mll-lab-nu/"),
    ("RoboARC's image under its former registry name", PRIV, "environment/Dockerfile",
     sub(r"^FROM ghcr\.io/mll-lab-nu/", "FROM ghcr.io/embodied-first-exam/"), PINS, "outside ghcr.io/mll-lab-nu/"),
    ("tests/test.sh removes only a look-alike of the verifier's logs",
     STD, TEST_SH, sub(r"^rm -rf /logs/verifier/\*", "rm -rf /logs/verifier/*.bak"), ISOLATION, "must empty /logs/verifier"),
    # ---- shared sources (D9): one version of each shared file, the eai-standard/2.4 layer as in scripts/eai_standard/
    ("one standard task's client edited", STD, "environment/rcb_client.py", append("# drift\n"), IDENTITY,
     "differs from scripts/eai_standard/v2_4/rcb_client.py"),
    ("one privileged task's grader edited", PRIV, "tests/grade.py", append("# drift\n"), IDENTITY,
     "2 versions of privileged tests/grade.py"),
    ("one task's tooling module edited", PRIV, "tests/rcb_topogym.py", append("# drift\n"), IDENTITY,
     "2 versions of rcb_topogym.py"),
    # ---- caps and minimums (PROTOCOL §2.10)
    ("a declared cap that docker does not apply",
     STD, COMPOSE, sub(r"^    mem_limit: \d+m(   # rcb-monitor)$", r"    mem_limit: 1024m\1"), RESOURCES, "declare the cap that is applied"),
    # ---- Appendix H.7: twins share cutoff, seeds and frozen instance; H.3: no simulator in the agent's image; H.1: the
    # monitor's own isolation and the sanitize step; H.8: DIGEST
    ("a standard task whose cutoff is not its privileged twin's", STD, TOML,
     sub(r"(^\[agent\]\n(?:(?!\[).*\n)*?timeout_sec = )[0-9.]+", r"\g<1>1234.0"), TWINS, "the twins have the same cutoff"),
    ("a standard task whose allowed seeds are not its privileged twin's", STD, TOML,
     sub(r"^allowed = \[(.*)\]$", r"allowed = [\1, 12345]"), TWINS, "differ from its privileged twin's"),
    ("a standard task's agent image takes the sidecar's code", STD, "environment/Dockerfile",
     append("COPY sim/ /opt/sim/\n"), STDIMG, "name files outside sim/ only"),
    ("the monitor sidecar gets a network", STD, COMPOSE, sub(r"^    network_mode: none\n", ""), MONITOR,
     "missing `network_mode: none`"),
    ("tests/test.sh no longer sanitizes /app/output before the grader", STD, TEST_SH,
     sub(r"^.*rcb_limits\.py sanitize /app/output.*\n", ""), MONITOR, "no `rcb_limits.py sanitize /app/output` line"),
    ("a task file changed without its DIGEST", PRIV, "README.md", append("\nedited\n"), DIGESTS,
     "a DIGEST that is not the digest of the task's current files"),
]


def run(cmd: list[str], task: Path) -> tuple[int, str]:
    r = subprocess.run(cmd + [str(task)], capture_output=True, text=True)
    return r.returncode, r.stdout + r.stderr


def twin_cutoffs(tmp: Path) -> int:
    """The cutoff is each task's own (PROTOCOL §2.12): normalize renders a changed `[agent] timeout_sec` into the
    instruction's closing section, the README row and the sim service's wall clock, check-cutoffs.py accepts the task on
    its own, and check-twins.py refuses it next to its privileged twin, whose number differs (`same_cutoff_as`, §2.15). Returns the
    number of failures."""
    import tomllib
    bad = 0

    def expect(ok: bool, what: str, out: str = "") -> None:
        nonlocal bad
        if not ok:
            bad += 1
            print(f"FAIL check-ci-selftest [twin cutoffs: {what}]" + (f":\n{out[-600:]}" if out else ""))

    work = tmp / "twins"
    shutil.copytree(ROOT / "tasks" / STD, work / STD)   # alone first; its twin joins it below
    t = work / STD
    sec = int(tomllib.loads((t / "task.toml").read_text())["agent"]["timeout_sec"]) // 2   # a number that is not its twin's
    toml = (t / "task.toml").read_text()
    (t / "task.toml").write_text(sub(r"(^\[agent\]\n(?:(?!\[).*\n)*?timeout_sec = )[0-9.]+", rf"\g<1>{sec}.0")(toml))
    r = subprocess.run([sys.executable, str(ROOT / "scripts/task_conventions.py"), str(t)], capture_output=True, text=True)
    expect(r.returncode == 0, "normalize a task whose cutoff changed", r.stdout + r.stderr)
    expect(f"\nYou have {sec} seconds to complete this task." in (t / "instruction.md").read_text(),
           f"the instruction does not say {sec} seconds after normalize")
    expect(f"| Agent time budget | {sec} s;" in (t / "README.md").read_text(), f"the README row does not say {sec} s")
    expect(f'RCB_WALL_CLOCK_S: "{sec}"' in (t / "environment/docker-compose.yaml").read_text(),
           f"the sim service's wall clock is not {sec}")
    rc, out = run(CUTOFFS, t)
    expect(rc == 0, "check-cutoffs.py on the normalized task alone", out)
    shutil.copytree(ROOT / "tasks" / PRIV, work / PRIV)
    r = subprocess.run(TWINS + [str(t)], capture_output=True, text=True)
    out = r.stdout + r.stderr
    expect(r.returncode != 0 and "the twins have the same cutoff" in out, "check-twins.py must refuse twins with two cutoffs", out)
    if not bad:
        print(f"check-ci-selftest: a task's own cutoff ({sec} s) rendered by normalize and accepted alone, refused next to its "
              "privileged twin")
    return bad


def main(argv: list[str]) -> int:
    tier = argv[1] if argv[:1] == ["--tier"] and len(argv) > 1 else "all"
    if tier not in ("invariants", "advisory", "all"):
        raise SystemExit("--tier invariants|advisory|all")
    sel = [m for m in MUTATIONS if tier == "all" or (any(m[4] is c for c in INVARIANT_CMDS) == (tier == "invariants"))]
    bad = 0
    with tempfile.TemporaryDirectory(prefix="rcb-ci-selftest-") as tmp:
        clean = {}
        for name in sorted({m[1] for m in sel}):
            src = ROOT / "tasks" / name
            if not (src / "task.toml").is_file():
                print(f"FAIL check-ci-selftest: the task {name} this self-test mutates is gone (pick another one in MUTATIONS)")
                return 1
            clean[name] = Path(tmp) / "clean" / name
            shutil.copytree(src, clean[name])
            for cmd in sorted({id(m[4]): m[4] for m in sel if m[1] == name and m[4] not in WITH_NEIGHBOURS and m[4] is not DIGESTS}.values()):
                rc, out = run(cmd, clean[name])
                if rc != 0:
                    print(f"FAIL check-ci-selftest: the unmutated {name} does not pass {Path(cmd[-1]).name}:\n{out[-800:]}")
                    bad += 1
        for k, (what, name, rel, mutate, cmd, expect) in enumerate(sel):
            t = Path(tmp) / f"m{k}" / name                      # the directory keeps the task's name (its mode is in it)
            shutil.copytree(clean[name], t)
            try:
                for r, f in (mutate.items() if isinstance(mutate, dict) else [(rel, mutate)]):
                    if r.startswith("+"):                   # a file the mutation adds
                        r = r[1:]
                        (t / r).parent.mkdir(parents=True, exist_ok=True)
                        data = f("")
                        (t / r).write_bytes(data) if isinstance(data, bytes) else (t / r).write_text(data)
                        continue
                    (t / r).write_text(f((t / r).read_text()))
            except (AssertionError, StopIteration, OSError) as e:
                print(f"FAIL check-ci-selftest [{what}]: the mutation no longer applies to {name} ({e}): update MUTATIONS")
                bad += 1
                continue
            if any(cmd is c for c in WITH_NEIGHBOURS):   # the broken copy among every other task (links to the real ones)
                for other in (ROOT / "tasks").iterdir():
                    if other.name != name and (other / "task.toml").is_file():
                        (t.parent / other.name).symlink_to(other)
                rc, out = run(cmd, t.parent if cmd is IDENTITY else t)
            else:
                rc, out = run(cmd, t)
            if rc == 0 or expect not in out:
                bad += 1
                print(f"FAIL check-ci-selftest [{what}]: {Path(cmd[-1]).name} "
                      + ("PASSED the broken task: the rule is blind" if rc == 0 else f"failed, but not with `{expect}`:\n{out[-600:]}"))
        if tier != "advisory":
            bad += twin_cutoffs(Path(tmp))
    if not bad:
        print(f"check-ci-selftest ({tier}): {len(sel)} mutations of real tasks, each caught by its rule")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
