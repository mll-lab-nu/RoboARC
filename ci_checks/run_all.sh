#!/usr/bin/env bash
# Every check, in two tiers (docs/PROTOCOL.md: only Appendix H binds):
#   1. the invariants: ci_checks/run_invariants.sh (the required CI job `invariants`); a failure fails this run (exit 1);
#   2. the advisory checks (the CI job `advisories`): ci_checks/parallel.py --advisory, findings are warnings (exit 3).
# Exit 0: everything clean; 1: an invariant failed (or the invariant run was incomplete); 3: advisory findings only.
#   run_all.sh [--tier invariants|advisory] [task dirs]   one tier only (default: both)
#   run_all.sh [task dirs]       the per-task checks (check-task.sh, scripts/task_conventions.py --check) on the given
#                                tasks (default: all), and every repo-level check
# The generator's check mode (python3 scripts/roboarc/mk_tasks.py --check) is a CI job of its own (`generator`), and so
# is IKEA's (ikea/ci/check.sh); neither runs here.
# Everything runs concurrently (ci_checks/parallel.py; RCB_CI_JOBS=N caps it, default: the CPUs), each command's output
# printed in one piece when it ends. RCB_SKIP_UNIT_TESTS=1: leave out check-unit-tests.sh (the CI workflow runs it, with
# --all, in a job of its own).
set -uo pipefail
tier=all
if [ "${1:-}" = --tier ]; then tier="${2:-}"; shift 2; fi
case "$tier" in invariants|advisory|all) ;; *) echo "FAIL run_all.sh: --tier invariants|advisory"; exit 1 ;; esac
inv_rc=0
if [ "$tier" != advisory ]; then
  bash "$(dirname "$0")/run_invariants.sh" "$@" || inv_rc=1
  [ "$tier" = invariants ] && exit $inv_rc
fi
# The generator's check mode compares file modes with a fresh render, which depends on the caller's umask: with 0002 (a
# common default) every rendered file is group-writable and the checks fail on a clean checkout. The checks run with the
# umask the repository is written with; the checkout itself must be made with umask 022 too.
umask 022
root="$(cd "$(dirname "$0")/.." && pwd)"
# task dirs given relative to the caller's directory: made absolute first
args=(); for a in "$@"; do if [ -d "$a" ]; then args+=("$(cd "$a" && pwd)"); else args+=("$a"); fi; done; set -- ${args[@]+"${args[@]}"}
cd "$root" || exit 1
q() { printf %q "$1"; }
# the mode registry (PROTOCOL §2.16, scripts/task_conventions.py MODE_REGISTRY), read once for the per-task shell checks
RCB_MODES_TABLE="$(python3 "$root/scripts/task_conventions.py" modes)" || { echo "FAIL scripts/task_conventions.py modes"; exit 1; }
export RCB_MODES_TABLE

explicit=0
tasks=("$@"); [ ${#tasks[@]} -eq 0 ] && tasks=(tasks/*/) || explicit=1
real=(); for t in "${tasks[@]}"; do t="${t%/}"; [ -f "$t/task.toml" ] && real+=("$t"); done; tasks=(${real[@]+"${real[@]}"})
echo "per-task checks on ${#tasks[@]} task(s)"

cmds=""
add() { cmds+="$*"$'\n'; }
# the slow ones first
if [ $explicit -eq 0 ]; then
  add python3 ci_checks/check-ci-selftest.py --tier advisory   # the advisory rules still fire (the invariant ones: run_invariants.sh)
  [ "${RCB_SKIP_UNIT_TESTS:-0}" = 1 ] || add bash ci_checks/check-unit-tests.sh   # the tests that need only the standard library
fi
# the per-task checks, in chunks: check-task.sh (the shell checks) and the generator's normal form
# (scripts/task_conventions.py --check: README, task.toml, test.sh are what normalize writes)
chunk=24
for ((i = 0; i < ${#tasks[@]}; i += chunk)); do
  part=""; for t in "${tasks[@]:i:chunk}"; do part+=" $(q "$t")"; done
  add bash ci_checks/check-task.sh --tier advisory"$part"
  add python3 scripts/task_conventions.py --check"$part"
done
# per-task Python checks: the given tasks, else all
targs=""; if [ $explicit -eq 1 ]; then for t in "${tasks[@]}"; do targs+=" $(q "$t")"; done; fi
# no world coordinates in the standard tasks' texts (docs/benchmarks/roboarc.md, "Standard mode, world frame, network")
if [ ${#tasks[@]} -gt 0 ]; then
  hparts=""; for t in "${tasks[@]}"; do hparts+=" $(q "$t")"; done
  add bash ci_checks/check-world-frame-texts.sh --prefix roboarc-"$hparts"
fi
# repo-level, cheap: always on everything, or on the given task dirs
add bash ci_checks/check-tooling-identity.sh                # D9 shared sources; the eai-standard/2.4 layer = scripts/eai_standard/
add bash ci_checks/check-standard-client.sh                 # standard agent + verifier images: the pinned client base and pins
# PROTOCOL §2.9: [metadata.resources] (minimum compute) in every task.toml, consistent with the GPU reservations and caps,
# and the README's Compute resources section rendered from it
add python3 ci_checks/check-resources.py"$targs"
# PROTOCOL §2.12: the agent's time cutoff, each task's own `[agent] timeout_sec`: at most 14400 s, the closing
# `## Time limit` section of instruction.md, the README row and the standard mode's sim service say it, nothing else states
# the limit (the twins' equal number is an invariant: check-twins.py)
add python3 ci_checks/check-cutoffs.py"$targs"
add python3 ci_checks/check-seeds.py"$targs"
# PROTOCOL §4: every task renders a replay video and the runner can collect it; a declared video time bound is not
# shorter than the verdict replay's
add python3 ci_checks/check-video.py"$targs"
# The task protocol (docs/PROTOCOL.md): the protocol version per mode, oracle fields, the image registry, no results/, no personal paths
add python3 ci_checks/check-protocol.py"$targs"
# PROTOCOL §6: the resource monitor sidecar's copies and the verifier step (its isolation: run_invariants.sh)
add python3 ci_checks/check-monitor.py --tier advisory"$targs"
# every script that builds images builds from a stamped copy (scripts/stamp_mtimes.py; BuildKit and equal-size files)
add python3 ci_checks/check-build-stamps.py
# PROTOCOL §4: the easy set (docs/easy_set.toml) names existing task bases, with a known reason and their benchmark label
add python3 scripts/task_conventions.py easy-set --check

t0=$SECONDS
printf %s "$cmds" | python3 "$root/ci_checks/parallel.py" --advisory; rc=$?
echo "run_all.sh: advisories $(( SECONDS - t0 )) s, ${#tasks[@]} task(s)"
if [ $inv_rc -ne 0 ]; then echo "FAIL run_all.sh: an invariant failed (ci_checks/run_invariants.sh)"; exit 1; fi
case $rc in
  0) echo "all checks passed" ; exit 0 ;;
  3) echo "invariants hold; advisory findings above (not blocking)"; exit 3 ;;
  *) echo "FAIL run_all.sh: the advisory run itself failed (parallel.py exit $rc)"; exit 1 ;;
esac
