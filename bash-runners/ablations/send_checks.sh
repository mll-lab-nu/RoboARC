#!/usr/bin/env bash
# Send the variant checks (bash-runners/ablations/check_variants.sh) to us before any agent trial
# (docs/ABLATIONS.md, step 1): jobs/ablations/checks/<E>.json and <E>.md of every experiment checked so far, the Harbor
# job folders of the trials that did not pass, and E4's rendered scenes (jobs/ablations/checks/e4_render/), into
# <name>/ablation-checks/<UTC time>/ of the ablations' results dataset (RoboARC/RoboARC-ablation-outputs, public), with
# the same token (~/.roboarc_hf_token) and secret scan as scripts/send_results.sh.
#   ENV_FILE=$PWD/.env CREATE_PR=1 bash-runners/ablations/send_checks.sh <your name, lowercase>
# Env (defaults): ENV_FILE (<repo>/.env), HF_TOKEN_FILE (~/.roboarc_hf_token), HF_DATASET (RoboARC/RoboARC-ablation-outputs),
# CREATE_PR=1 (a pull request), DRY_RUN=1 (everything but the upload; the bundle stays in <repo>/../send/).
set -euo pipefail
root="$(cd "$(dirname "$0")/../.." && pwd)"
[ $# -eq 1 ] || { sed -n '2,/^set -euo pipefail$/p' "$0" | sed '$d'; exit 2; }
H="$1"; [[ "$H" =~ ^[a-z0-9_-]+$ ]] || { echo "name: lowercase letters, digits, - and _ only" >&2; exit 2; }
C="$root/jobs/ablations/checks"
ls "$C"/E*.json >/dev/null 2>&1 || { echo "no reports in $C: run bash-runners/ablations/check_variants.sh first" >&2; exit 2; }
ENV_FILE="${ENV_FILE:-$root/.env}"; HF_TOKEN_FILE="${HF_TOKEN_FILE:-$HOME/.roboarc_hf_token}"
HF_DATASET="${HF_DATASET:-$(python3 "$root/scripts/ablations/experiments.py" dataset E1)}"; ID=$(date -u +%Y%m%dT%H%MZ)
OUT="$(dirname "$root")/send/ablation-checks__${H}__$ID"
[ ! -e "$OUT" ] || { echo "$OUT exists: run again in a minute" >&2; exit 2; }
mkdir -p "$OUT/failed_jobs"
cp "$C"/E*.json "$C"/E*.md "$OUT/"
for d in "$C"/*_render; do [ -d "$d" ] && cp -a "$d" "$OUT/"; done   # E4's rendered scenes (check_clearance_render.py), for us to look at
python3 - "$C" "$OUT/failed_jobs" <<'PY'
import json, pathlib, shutil, sys
checks, out = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
for rep in sorted(checks.glob("E*.json")):
    r = json.loads(rep.read_text())
    for v in r.get("variants", []):
        for a in ("oracle", "nop"):
            t = v.get(a) or {}
            src = checks / r["experiment"] / "jobs" / str(t.get("job"))
            if t and not t.get("pass") and src.is_dir():
                shutil.copytree(src, out / src.name, symlinks=True)
                log = checks / r["experiment"] / "logs" / f"{t['job']}.log"
                if log.is_file():
                    shutil.copyfile(log, out / f"{t['job']}.harbor.log")
PY
SCAN=(); [ -f "$ENV_FILE" ] && SCAN+=(--env-file "$ENV_FILE")
case "${CODEX_FORCE_AUTH_JSON:-}" in 1|true|TRUE|yes) SCAN+=(--codex-auth "${CODEX_AUTH_JSON_PATH:-$HOME/.codex/auth.json}") ;; esac
python3 "$root/scripts/harbor/scan_secrets.py" "$OUT" "${SCAN[@]}" --redact | tail -1
s=$(python3 "$root/scripts/harbor/scan_secrets.py" "$OUT" "${SCAN[@]}")
tail -1 <<<"$s" | grep -qE ': 0 contain one$' || { echo "$s" >&2; echo "a secret is left in $OUT: nothing is sent" >&2; exit 1; }
for f in "$OUT"/E*.md; do echo "--- $(basename "$f")"; sed -n '5p' "$f"; done
if [ "${DRY_RUN:-0}" = 1 ]; then echo "DRY_RUN=1: not uploaded; the bundle is in $OUT"; exit 0; fi
[ -s "$HF_TOKEN_FILE" ] || { echo "no token in $HF_TOKEN_FILE (docs/COLLABORATOR_SETUP.md, section 6)" >&2; exit 2; }
PR=(); case "${CREATE_PR:-0}" in 1|true|TRUE|yes) PR=(--create-pr) ;; esac
HF_TOKEN=$(cat "$HF_TOKEN_FILE") hf upload "$HF_DATASET" "$OUT" "$H/ablation-checks/$ID" --repo-type dataset --private "${PR[@]}" \
  --commit-message "RoboARC ablation variant checks: $H/$ID"
echo "Sent to $HF_DATASET: $H/ablation-checks/$ID. Tell us, with the pull request link above."
