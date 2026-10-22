#!/usr/bin/env python3
"""Unit test of the Nyx render process's start-up retry (scripts/ikea_mujoco/standard/ikea_nyx.py, NyxRig._start), without
Nyx, MuJoCo or a GPU: RCB_NYX_PYTHON is a shell script that plays the render process, failing (an exit code, a signal, no
ready line) for the first attempts as each case says, then sending its ready line; or reporting a GPU Nyx cannot run on
(nyx_direct.UnsupportedDevice in its ready line), which is raised at once, never retried.
    python3 scripts/ikea_mujoco/tests/test_nyx_startup.py   (also run by test_grade.py)"""
import os, pathlib, sys, tempfile, types

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "standard"))
try:
    import numpy  # noqa: F401
except ImportError:                  # the start-up path does not use it: the standard-library CI tier runs this too
    sys.modules["numpy"] = types.ModuleType("numpy")
import ikea_nyx  # noqa: E402

# the render process: attempt k (1, 2, ...) does what word k of $PLAN says: exit3 / kill9 / hang / unsupported / ok
FAKE = r'''#!/bin/sh
folder="$4"; n=$(( $(cat "$folder/n" 2>/dev/null || echo 0) + 1 )); echo $n > "$folder/n"
what=$(echo "$PLAN" | cut -d' ' -f$n)
echo "fake render process: attempt $n: $what" >&2
case "$what" in
  exit3) exit 3 ;;
  kill9) kill -9 $$ ;;
  hang) exec sleep 60 ;;
  unsupported) echo '{"ready": false, "unsupported": "Nyx cannot render on this host: the GPU lacks VK_KHR_ray_query"}'; exit 1 ;;
  ok) echo '{"ready": true, "instances": 1, "meshes": 1, "build_s": 0.0}'; cat > /dev/null ;;
esac
'''


def start(plan: str):
    with tempfile.TemporaryDirectory() as td:
        td = pathlib.Path(td)
        (td / "fake").write_text(FAKE); (td / "fake").chmod(0o755)
        os.environ.update(RCB_NYX_PYTHON=str(td / "fake"), PLAN=plan)
        rig = ikea_nyx.NyxRig.__new__(ikea_nyx.NyxRig)
        rig.folder, rig.log_path = str(td), str(td / "nyx.log")
        try:
            rig._start()
            out = rig.startup
            rig.close()
        except ikea_nyx.RenderError as exc:
            out = exc
        return out, (td / "nyx.log").read_text()


def main() -> int:
    ikea_nyx.STARTUP_BACKOFF_S = (0.0, 0.0)
    ikea_nyx.READY_TIMEOUT_S = 2.0
    s, log = start("ok")
    assert s["attempts"] == 1 and s["failed_attempts"] == [] and "failed" not in log, (s, log)
    s, log = start("exit3 kill9 ok")              # died twice (an exit code, a signal), then started
    assert s["attempts"] == 3 and [f["exit"] for f in s["failed_attempts"]] == [3, -9], s
    assert log.count("failed: Nyx render process ended") == 2 and "exit code 3" in log and "exit code -9" in log, log
    assert "stderr tail:\nfake render process: attempt 1: exit3" in log, log
    s, log = start("hang ok")                     # no ready line in READY_TIMEOUT_S: killed (-9), started again
    assert s["attempts"] == 2 and s["failed_attempts"][0]["exit"] == -9 and "no ready line" in log, (s, log)
    e, log = start("exit3 exit3 kill9 ok")        # every attempt failed: RenderError after the last one, never a 4th
    assert isinstance(e, ikea_nyx.RenderError) and "3 attempts" in str(e) and "exit -9" in str(e), e
    assert log.count("start-up attempt") == 3 and "attempt 4" not in log, log
    e, log = start("unsupported ok")              # a GPU Nyx cannot run on: raised after the first attempt, not retried
    assert isinstance(e, ikea_nyx.UnsupportedGPU) and isinstance(e, ikea_nyx.RenderError) and "VK_KHR_ray_query" in str(e), e
    assert log.count("start-up attempt") == 1 and "not retried" in log and "attempt 2" not in log, log
    print("test_nyx_startup: ok (first start; two deaths then ready; a hang; three failures raise; an unsupported GPU is not retried)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
