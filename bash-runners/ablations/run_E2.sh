#!/usr/bin/env bash
# Ablation experiment E2 (docs/ABLATIONS.md): the information-access ablation with a second agent; its trials are
# listed by `python3 scripts/ablations/experiments.py plan E2`. Starts only after bash-runners/ablations/check_variants.sh E2
# passed. Settings and environment: bash-runners/ablations/run_experiment.sh.
#   MODEL=claude-opus-5.5 RESULTS=/path/outside/the/repo bash-runners/ablations/run_E2.sh
exec "$(dirname "$0")/run_experiment.sh" E2 "$@"
