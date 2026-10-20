#!/usr/bin/env bash
# Unit tests of the benchmark's tooling and the shared layer, in two tiers.
#  1. Standard library and a shell only (no numpy, no simulator, no image): the Harbor runner's helpers (retry rule,
#     account-failure rule, Codex config, build stamps). Run by run_all.sh and the CI job `unit-tests` (not required; the
#     test that enforces PROTOCOL Appendix H.5 also runs in the required job `invariants`, run_invariants.sh).
#  2. numpy + requests (no simulator, no image, no GPU): the eai-standard/2.4 service layer on a stub simulator (its
#     in-trial crash recovery, request checks) and RoboARC's sim-service tests with stubs. Run when both modules can be
#     imported (the CI job `unit-tests` installs them), skipped with a note otherwise; `--all` makes a missing module a
#     failure.
# Not here: tests that need the simulator image (scripts/roboarc/tests/test_nyx_startup.py and the IKEA image tests,
# ikea/ci/check.sh --image-tests). The resource monitor's own tests are run by check-monitor.py.
# Usage: check-unit-tests.sh [--all]   (repo-level; run_all.sh calls it once). The tests run concurrently
# (ci_checks/parallel.py, -j from RCB_CI_JOBS or the CPUs; the service tests on ports one at a time, see lane() below).
set -uo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"; rc=0
if [ "${1:-}" = "--one" ]; then   # --one <timeout s> <test path under scripts/> [arguments of the test]
  limit="$2" t="$3"; shift 3
  [ -f "$root/scripts/$t" ] || { echo "FAIL scripts/$t: listed in check-unit-tests.sh but missing"; exit 1; }
  if out=$(cd "${TMPDIR:-/tmp}" && PYTHONDONTWRITEBYTECODE=1 timeout "$limit" python3 "$root/scripts/$t" "$@" 2>&1); then
    echo "ok scripts/$t${*:+ $*}: ${out##*$'\n'}"; exit 0
  else echo "FAIL scripts/$t${*:+ $*}:"; echo "$out" | tail -n 40; exit 1; fi
fi
all=0; [ "${1:-}" = "--all" ] && { all=1; shift; }
self=ci_checks/check-unit-tests.sh; cmds=""   # the commands run from the repository root
# Tests that start a service on a port they pick (or may: sockets, subprocesses, servers) run one at a time, in the lane
# `port` of parallel.py: two of them at once can pick the same free port and one test then talks to the other's service
# (seen on 4-CPU runners: one self-test's client reached another's service). The rest run concurrently.
lane() { grep -qE 'socket|free_port|_PORT|Popen|serve|import test_' "$root/scripts/$1" 2>/dev/null && printf '@lane:port '; }
run() {   # run <timeout s> <test path under scripts/> [arguments of the test]: queue it
  cmds+="$(lane "$2")bash $self --one $*"$'\n'
}
# the standard-library tests
for t in harbor/tests/test_codex_tools_off.py harbor/tests/test_retry_policy.py tests/test_stamp_mtimes.py; do
  run 300 "$t"
done
run 120 harbor/account_failure_check.py --selftest   # the runner's lane stop and infra rules on synthetic trials

if python3 -c "import numpy, requests" 2>/dev/null; then
  export PATH="$(dirname "$(command -v python3)"):$PATH"   # run.sh starts `python`: the same interpreter
  for t in eai_standard/v2_4/selftest.py roboarc/tests/test_server_requests.py; do
    run 900 "$t"
  done
elif [ $all -eq 1 ]; then
  echo "FAIL check-unit-tests.sh --all: numpy and requests are needed for the second tier (pip install numpy requests)"; rc=1
else
  echo "note check-unit-tests.sh: numpy / requests not installed, the service-layer tests were not run (CI job unit-tests runs them; locally: pip install numpy requests, then check-unit-tests.sh --all)"
fi
(cd "$root" && printf %s "$cmds" | python3 ci_checks/parallel.py) || rc=1
exit $rc
