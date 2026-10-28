#!/usr/bin/env bash
# Unit tests of the benchmark's tooling and the shared layer, in two tiers.
#  1. Standard library and a shell only (no numpy, no simulator, no image): the Harbor runner's helpers (retry rule,
#     account-failure rule, Codex config, the optional usage gateway switch and the trials' seeds and results with a fake
#     harbor, build stamps), the seed tool (scripts/task_seed.py: assign, materialize; ci_checks/check-seed-files.py), the
#     shipped usage gateway (scripts/usage_gateway/: its own tests, and the options and host lists the runner gives it),
#     the simulator source hash (tools/release/source_hash.py), and RoboARC's generator and asset pins
#     (scripts/roboarc/tests/: the packages' fields and checks, assets.py pin / check / check-instances on a fake
#     dataset). Run by run_all.sh and the CI job `unit-tests` (not required; the
#     test that enforces PROTOCOL Appendix H.5 also runs in the required job `invariants`, run_invariants.sh).
#  2. numpy + requests (no simulator, no image, no GPU): the eai-standard/2.4 service layer on a stub simulator (its
#     in-trial crash recovery, request checks), RoboARC's sim-service tests with stubs, Nyx's start-up retry
#     (tools/tests/test_nyx_startup.py) and its GPU check, both nyx_direct.py copies (tools/tests/test_nyx_device_check.py).
#     The cloth assist's exposure rule (tools/tests/test_exposure_rule.py) needs
#     numpy + scipy and is run when they import. Run when both modules can be
#     imported (the CI job `unit-tests` installs them), skipped with a note otherwise; `--all` makes a missing module a
#     failure (numpy, requests, scipy).
# Not here: tests that need the simulator image (the rest of tools/tests/, the IKEA image tests,
# ikea/ci/check.sh --image-tests). The resource monitor's own tests are run by check-monitor.py.
# Usage: check-unit-tests.sh [--all]   (repo-level; run_all.sh calls it once). The tests run concurrently
# (ci_checks/parallel.py, -j from RCB_CI_JOBS or the CPUs; the service tests on ports one at a time, see lane() below).
set -uo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"; rc=0
if [ "${1:-}" = "--one" ]; then   # --one <timeout s> <test path from the repository root> [arguments of the test]
  limit="$2" t="$3"; shift 3
  [ -f "$root/$t" ] || { echo "FAIL $t: listed in check-unit-tests.sh but missing"; exit 1; }
  if out=$(cd "${TMPDIR:-/tmp}" && PYTHONDONTWRITEBYTECODE=1 timeout "$limit" python3 "$root/$t" "$@" 2>&1); then
    echo "ok $t${*:+ $*}: ${out##*$'\n'}"; exit 0
  else echo "FAIL $t${*:+ $*}:"; echo "$out" | tail -n 40; exit 1; fi
fi
all=0; [ "${1:-}" = "--all" ] && { all=1; shift; }
self=ci_checks/check-unit-tests.sh; cmds=""   # the commands run from the repository root
# Tests that start a service on a port they pick (or may: sockets, subprocesses, servers) run one at a time, in the lane
# `port` of parallel.py: two of them at once can pick the same free port and one test then talks to the other's service
# (seen on 4-CPU runners: one self-test's client reached another's service). The rest run concurrently.
lane() { grep -qE 'socket|free_port|_PORT|Popen|serve|import test_' "$root/$1" 2>/dev/null && printf '@lane:port '; }
run() {   # run <timeout s> <test path from the repository root> [arguments of the test]: queue it
  cmds+="$(lane "$2")bash $self --one $*"$'\n'
}
# the standard-library tests
for t in scripts/harbor/tests/test_codex_tools_off.py scripts/harbor/tests/test_retry_policy.py \
         scripts/harbor/tests/test_usage_gateway_option.py scripts/harbor/tests/test_usage_gateway_shipped.py \
         scripts/usage_gateway/tests/test_rawlog.py scripts/usage_gateway/tests/test_paths.py \
         scripts/usage_gateway/tests/test_budget_stop.py scripts/tests/test_stamp_mtimes.py \
         scripts/tests/test_task_seed.py scripts/harbor/tests/test_runner_seeds.py scripts/harbor/tests/test_gpu_slots.py \
         scripts/harbor/tests/test_pin_gpu.py scripts/harbor/tests/test_dsh_acp.py tools/tests/test_stage_instance.py tools/tests/test_restamp.py \
         scripts/roboarc/tests/test_generator.py scripts/roboarc/tests/test_assets.py \
         scripts/roboarc/tests/test_generator_seeds.py tools/tests/test_release_seeds.py images/tests/test_pull_all.py \
         tools/tests/test_d4_tools.py scripts/roboarc/tests/test_cutoff_exceptions.py; do
  run 300 "$t"
done
run 120 scripts/harbor/account_failure_check.py --selftest   # the runner's lane stop and infra rules on synthetic trials
run 300 tools/release/source_hash.py --check   # the simulator source hashes to release.json's runtime.source (SOURCE.json)

if python3 -c "import numpy, requests" 2>/dev/null; then
  export PATH="$(dirname "$(command -v python3)"):$PATH"   # run.sh starts `python`: the same interpreter
  for t in scripts/eai_standard/v2_4/selftest.py scripts/roboarc/tests/test_server_requests.py; do
    run 900 "$t"
  done
  run 300 tools/tests/test_nyx_startup.py   # Nyx's start-up retry, with a stub build (numpy only)
  run 300 tools/tests/test_nyx_device_check.py   # Nyx's GPU check, mocked Vulkan extension lists (numpy only)
elif [ $all -eq 1 ]; then
  echo "FAIL check-unit-tests.sh --all: numpy and requests are needed for the second tier (pip install --no-deps --require-hashes -r ci_checks/requirements-ci.lock)"; rc=1
else
  echo "note check-unit-tests.sh: numpy / requests not installed, the service-layer tests were not run (CI job unit-tests runs them; locally: pip install numpy requests, then check-unit-tests.sh --all)"
fi
# the cloth assist's exposure rule on the simulator source (topogym.utils.grasp: numpy + scipy)
if python3 -c "import numpy, scipy" 2>/dev/null; then
  run 300 tools/tests/test_exposure_rule.py
  run 300 tools/tests/test_fiber_red_lay.py   # fiber 07-10's lay rule: either red fiber
  run 300 tools/tests/test_wire_splicing_criteria.py   # wire_splicing's around_both on a one-node strand
  run 300 tools/tests/test_active_parts_clearance.py   # the blanket families' release clearance policy
elif [ $all -eq 1 ]; then
  echo "FAIL check-unit-tests.sh --all: numpy and scipy are needed for tools/tests/test_exposure_rule.py (pip install --no-deps --require-hashes -r ci_checks/requirements-ci.lock)"; rc=1
else
  echo "note check-unit-tests.sh: numpy / scipy not installed, tools/tests/test_exposure_rule.py was not run"
fi
(cd "$root" && printf %s "$cmds" | python3 ci_checks/parallel.py) || rc=1
exit $rc
