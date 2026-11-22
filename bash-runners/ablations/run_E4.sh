#!/usr/bin/env bash
# Ablation experiment E4 (docs/ABLATIONS.md): relaxed clearances; its trials are
# listed by `python3 scripts/ablations/experiments.py plan E4`. Starts only after bash-runners/ablations/check_variants.sh E4
# passed. Settings and environment: bash-runners/ablations/run_experiment.sh.
#   MODEL=gpt-6-astra RESULTS=/path/outside/the/repo bash-runners/ablations/run_E4.sh
exec "$(dirname "$0")/run_experiment.sh" E4 "$@"
