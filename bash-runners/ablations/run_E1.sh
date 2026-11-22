#!/usr/bin/env bash
# Ablation experiment E1 (docs/ABLATIONS.md): the information-access ablation on more puzzles; its trials are
# listed by `python3 scripts/ablations/experiments.py plan E1`. Starts only after bash-runners/ablations/check_variants.sh E1
# passed. Settings and environment: bash-runners/ablations/run_experiment.sh.
#   MODEL=gpt-6-astra RESULTS=/path/outside/the/repo bash-runners/ablations/run_E1.sh
exec "$(dirname "$0")/run_experiment.sh" E1 "$@"
