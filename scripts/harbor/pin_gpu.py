#!/usr/bin/env python3
"""Pin every NVIDIA device a task copy's containers reserve to one GPU (scripts/harbor/run_agent_batch.sh, GPU_SCHED=1).

    python3 scripts/harbor/pin_gpu.py --gpu N <task dir> ...            # rewrite the copies in place
    python3 scripts/harbor/pin_gpu.py --gpu N --check <task dir> ...    # write nothing; exit 0 if every device is GPU N

A task's compose files (environment/ and tests/) reserve their GPU as `count: 1`, and Docker maps a count to the first
GPUs of the host: every trial would run on GPU 0. In a copy, each `count: 1` under a `driver: nvidia` device becomes
`device_ids: ["N"]`, in every service (main or sim, rcb_monitor, the verifier) and both files, so a trial's containers
all see the same one GPU: rcb_monitor measures the GPU memory of the GPUs its own container sees, and Nyx needs
exactly one GPU visible (images/roboarc/README.md). A count other than 1 is refused. A compose file without an NVIDIA
device (IKEA privileged) is left as it is. Never run it on tasks/ itself: the task's files keep their values, a job's
copy is what Harbor runs. Standard library only."""
from __future__ import annotations

import re
import sys
from pathlib import Path

COMPOSE_FILES = ("docker-compose.yaml", "docker-compose.yml", "gpu.docker-compose.yaml", "gpu.docker-compose.yml")
DRIVER = re.compile(r"^(\s*)-\s*driver:\s*nvidia\s*$")
COUNT = re.compile(r"^(\s*)count:\s*(\S+)\s*$")
IDS = re.compile(r"^(\s*)device_ids:\s*(.*?)\s*$")


def files(task: Path) -> list[Path]:
    return [task / d / f for d in ("environment", "tests") for f in COMPOSE_FILES if (task / d / f).is_file()]


def devices(lines: list[str]):
    """-> for each `- driver: nvidia` list item, the indices of its count and device_ids lines (or None)."""
    out = []
    for i, line in enumerate(lines):
        m = DRIVER.match(line)
        if not m:
            continue
        indent = len(m.group(1))
        count = ids = None
        for j in range(i + 1, len(lines)):
            s = lines[j]
            if s.strip() == "":
                continue
            if len(s) - len(s.lstrip()) <= indent:     # the item (or its list) ended
                break
            if COUNT.match(s):
                count = j
            elif IDS.match(s):
                ids = j
        out.append((i, count, ids))
    return out


def pin(path: Path, gpu: str) -> tuple[str, int]:
    """-> (the file's new text, the number of devices pinned); raises ValueError on a device that cannot be pinned."""
    lines = path.read_text().split("\n")
    n = 0
    for i, count, ids in devices(lines):
        if count is not None:
            m = COUNT.match(lines[count])
            if m.group(2) != "1":
                raise ValueError(f"{path}:{count + 1}: count {m.group(2)}: a trial's container must see exactly one GPU")
            lines[count] = f'{m.group(1)}device_ids: ["{gpu}"]'
            if ids is not None:
                raise ValueError(f"{path}:{ids + 1}: both count and device_ids")
        elif ids is not None:
            m = IDS.match(lines[ids])
            lines[ids] = f'{m.group(1)}device_ids: ["{gpu}"]'
        else:
            raise ValueError(f"{path}:{i + 1}: an NVIDIA device with neither count nor device_ids")
        n += 1
    return "\n".join(lines), n


def check(path: Path, gpu: str) -> list[str]:
    """-> the problems: a device not pinned to exactly GPU gpu."""
    lines = path.read_text().split("\n")
    bad = []
    for i, count, ids in devices(lines):
        if count is not None:
            bad.append(f"{path}:{count + 1}: still a count")
        if ids is None or IDS.match(lines[ids]).group(2) != f'["{gpu}"]':
            bad.append(f"{path}:{i + 1}: not pinned to GPU {gpu}")
    return bad


def main(argv: list[str]) -> int:
    do_check = "--check" in argv
    args = [a for a in argv if a != "--check"]
    if "--gpu" not in args:
        sys.exit(__doc__)
    i = args.index("--gpu")
    try:
        gpu = args[i + 1]
    except IndexError:
        sys.exit("--gpu needs a GPU index")
    del args[i:i + 2]
    if not gpu.isdigit():
        sys.exit(f"--gpu {gpu}: a GPU index (0, 1, ...)")
    if not args:
        sys.exit(__doc__)
    repo_tasks = (Path(__file__).resolve().parents[2] / "tasks").resolve()
    problems = []
    for a in args:
        t = Path(a)
        if not (t / "task.toml").is_file():
            sys.exit(f"{a}: not a task directory (no task.toml)")
        if not do_check and t.resolve().parent == repo_tasks:
            sys.exit(f"{a}: a task of the repository itself; pin a copy (the task's files keep their values)")
        for p in files(t):
            if do_check:
                problems += check(p, gpu)
                continue
            try:
                text, n = pin(p, gpu)
            except ValueError as e:
                sys.exit(str(e))
            if n:
                p.write_text(text)
                print(f"{p}: {n} device(s) -> GPU {gpu}")
    for line in problems:
        print(line)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
