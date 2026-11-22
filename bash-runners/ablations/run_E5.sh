#!/usr/bin/env bash
# Ablation experiment E5 (docs/ABLATIONS.md): observation realism; its trials are
# listed by `python3 scripts/ablations/experiments.py plan E5`. Starts only after bash-runners/ablations/check_variants.sh E5
# passed. Settings and environment: bash-runners/ablations/run_experiment.sh.
#   MODEL=gpt-6-astra RESULTS=/path/outside/the/repo bash-runners/ablations/run_E5.sh
exec "$(dirname "$0")/run_experiment.sh" E5 "$@"
