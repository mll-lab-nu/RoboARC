#!/bin/bash
# Send one batch's records to the maintainers' Hugging Face dataset (docs/COLLABORATOR_SETUP.md, section 6).
#   scripts/send_results.sh <batch> <your name, lowercase>
# Steps:
#   1. secret scan (scripts/harbor/scan_secrets.py, the runner's own scanner, with the runner's credentials): the batch's jobs are
#      redacted, then scanned again and must show none; the log mirror is append-only and never redacted, so a mirror
#      that holds a secret, or that this user cannot read, is left out and MANIFEST.md says so;
#   2. one archive per job, jobs/<job>.tar.gz: jobs/agents/<job>/, .seeds/<job>.json and the mirror's <job>/; one
#      batch.tar.gz: <batch>.log, <batch>.summary.md, .plan/<batch>.json and .tsv, the per-trial results <RESULTS>/<batch>/
#      (as results/); INVENTORY.tsv (archive, sha256, path of every file in them) and SHA256SUMS (every archive). An
#      archive packed again from the same files is the same bytes (sorted, gzip -n);
#   3. index.json: per job its archive, size and sha256, its tasks with mode, trial and seed (.plan/<batch>.json), and per
#      trial directory its reward (verifier/reward.txt) and whether Harbor wrote an exception.txt;
#      MANIFEST.md: this checkout's commit, the host, GPUs, docker and harbor versions, every image of images/pins.json
#      with its local id and digests, the runner's settings line (agent, model, effort, run id, modes, usage gateway; no
#      credentials), trials planned / with a verdict, the seeds;
#   4. hf upload of the bundle (MANIFEST.md, index.json, SHA256SUMS, INVENTORY.tsv, batch.tar.gz, jobs/) into
#      <dataset>/<name>/<batch>/<bundle id>/, as one commit, or with CREATE_PR=1 as a pull request the maintainers merge
#      (for an account without write access to the dataset). A later delivery gets a new bundle id and never overwrites
#      an earlier one. scripts/receive_results.sh is the maintainers' side.
# Env (defaults): JOBS (<repo>/jobs/agents, as run_agent_batch.sh), RESULTS (<repo>/../rcb_results), ENV_FILE (<repo>/.env,
#   the file the batch used), CODEX_FORCE_AUTH_JSON / CODEX_AUTH_JSON_PATH (as for the batch), HF_TOKEN_FILE
#   (~/.roboarc_hf_token, a write token of your own account), HF_DATASET (RoboARC/RoboARC-outputs, public),
#   OUT (<repo>/../send), NOTES (one line for MANIFEST.md: crashes, OOM kills, outages, reruns), CREATE_PR=1 (upload as a
#   pull request), DRY_RUN=1 (everything but the upload).
set -euo pipefail
B=${1:?usage: send_results.sh <batch> <your name>}; H=${2:?usage: send_results.sh <batch> <your name>}
[[ "$H" =~ ^[a-z0-9_-]+$ ]] || { echo "name: lowercase letters, digits, - and _ only" >&2; exit 2; }
R=$(cd "$(dirname "$0")/.." && pwd)
JOBS=$(cd "${JOBS:-$R/jobs/agents}" && pwd); MIRROR=${JOBS}_mirror
RESULTS=${RESULTS:-$(dirname "$R")/rcb_results}; ENV_FILE=${ENV_FILE:-$R/.env}
HF_TOKEN_FILE=${HF_TOKEN_FILE:-$HOME/.roboarc_hf_token}; HF_DATASET=${HF_DATASET:-RoboARC/RoboARC-outputs}
ID=$(date -u +%Y%m%dT%H%MZ); NAME=${B}__${H}__${ID}
OUT=${OUT:-$(dirname "$R")/send}/$NAME
[ ! -e "$OUT" ] || { echo "$OUT exists (a run within the same minute): run again in a minute" >&2; exit 2; }
mkdir -p "$OUT"
ls -d "$JOBS/$B"-*/ >/dev/null 2>&1 || { echo "no jobs $JOBS/$B-*" >&2; exit 2; }
[ "${DRY_RUN:-0}" = 1 ] || [ -s "$HF_TOKEN_FILE" ] || { echo "no token in $HF_TOKEN_FILE" >&2; exit 2; }
[ "${DRY_RUN:-0}" = 1 ] || command -v hf >/dev/null || { echo "needs the hf CLI: pip install -U huggingface_hub" >&2; exit 2; }

SCAN=(); [ -f "$ENV_FILE" ] && SCAN+=(--env-file "$ENV_FILE")
case "${CODEX_FORCE_AUTH_JSON:-}" in 1|true|TRUE|yes) SCAN+=(--codex-auth "${CODEX_AUTH_JSON_PATH:-$HOME/.codex/auth.json}") ;; esac
clean() {   # clean <scan output>: the scanner's last line reads "... 0 contain one" with no damaged file
    tail -1 <<<"$1" | grep -qE ': 0 contain one$'
}
echo "== 1. secret scan"
probe=$(python3 "$R/scripts/harbor/scan_secrets.py" "$R/scripts/harbor" "${SCAN[@]}" 2>&1 || true)
grep -q "no secret loaded" <<<"$probe" && { echo "the scan has no credentials to look for: set ENV_FILE to the .env the batch used (now ${ENV_FILE})" >&2; exit 2; }
jobs=$(ls -d "$JOBS/$B"-*/ | sed 's#/$##')
for j in $jobs; do python3 "$R/scripts/harbor/scan_secrets.py" "$j" "${SCAN[@]}" --redact | tail -1; done
for j in $jobs; do
    s=$(python3 "$R/scripts/harbor/scan_secrets.py" "$j" "${SCAN[@]}")
    clean "$s" || { echo "$s" >&2; echo "a secret is left in $j (or a file could not be read): nothing is sent" >&2; exit 1; }
done
mirror_note="included"; mdirs=()
if ls -d "$MIRROR/$B"-*/ >/dev/null 2>&1; then
    for m in $(ls -d "$MIRROR/$B"-*/ | sed 's#/$##'); do
        if [ -n "$(find "$m" ! -readable -print -quit 2>/dev/null)" ]; then mirror_note="left out: $m is not readable by $(id -un)"; mdirs=(); break; fi
        s=$(python3 "$R/scripts/harbor/scan_secrets.py" "$m" "${SCAN[@]}")
        clean "$s" || { mirror_note="left out: the scan found a secret in $m"; mdirs=(); break; }
        mdirs+=("$m")
    done
else
    mirror_note="none ($MIRROR/$B-* does not exist)"
fi
echo "   jobs clean; mirror: $mirror_note"

echo "== 2. archives"
P=$(dirname "$JOBS"); base=$(basename "$JOBS"); mkdir -p "$OUT/jobs"
inventory() {   # inventory <archive> <dir> <path> ...: per file under <dir>/<path>: <archive> TAB <sha256> TAB <path>
    local a=$1 d=$2; shift 2
    ( cd "$d" && find "$@" -type f -print0 | sort -z | xargs -0 -r sha256sum | sed "s#^\([0-9a-f]*\)  #$a\t\1\t#" )
}
for j in $jobs; do
    n=$(basename "$j"); paths=("$base/$n")
    [ -f "$JOBS/.seeds/$n.json" ] && paths+=("$base/.seeds/$n.json")
    for m in "${mdirs[@]}"; do [ "$m" = "$MIRROR/$n" ] && paths+=("$(basename "$MIRROR")/$n"); done
    inventory "jobs/$n.tar.gz" "$P" "${paths[@]}" >> "$OUT/INVENTORY.tsv"
    tar --sort=name -C "$P" -cf - "${paths[@]}" | gzip -n > "$OUT/jobs/$n.tar.gz"
done
paths=()
for f in "$B.log" "$B.summary.md" ".plan/$B.json" ".plan/$B.tsv"; do [ -f "$JOBS/$f" ] && paths+=("$base/$f"); done
[ ${#paths[@]} -gt 0 ] && inventory batch.tar.gz "$P" "${paths[@]}" >> "$OUT/INVENTORY.tsv"
res_note="none ($RESULTS/$B does not exist)"
if [ -d "$RESULTS/$B" ]; then
    res_note="included as results/"
    inventory batch.tar.gz "$RESULTS" "$B" | sed "s#\t$B/#\tresults/$B/#" >> "$OUT/INVENTORY.tsv"
fi
tar --sort=name -C "$P" -cf - "${paths[@]}" \
    $([ -d "$RESULTS/$B" ] && echo "--transform=s#^$B#results/$B# -C $RESULTS $B") | gzip -n > "$OUT/batch.tar.gz"
( cd "$OUT" && sha256sum batch.tar.gz jobs/*.tar.gz > SHA256SUMS )

echo "== 3. index.json, MANIFEST.md"
python3 - "$OUT" "$JOBS" "$B" "$ID" "$H" <<'PY'
import json, pathlib, sys
out, jobs, batch, bundle, helper = sys.argv[1:]
out, jobs = pathlib.Path(out), pathlib.Path(jobs)
sums = {f: h for h, f in (l.split("  ", 1) for l in (out / "SHA256SUMS").read_text().splitlines())}
try:
    plan = json.loads((jobs / ".plan" / f"{batch}.json").read_text())
except (OSError, ValueError):
    plan = {}
tasks = {}
for t in plan.get("jobs", []):
    tasks.setdefault(t["job"], []).append({k: t.get(k) for k in ("task", "mode", "trial", "seed")})
rows = []
for a in sorted((out / "jobs").glob("*.tar.gz")):
    job = a.name[:-len(".tar.gz")]
    trials = []
    for d in sorted(p for p in (jobs / job).iterdir() if p.is_dir()):
        try:
            reward = float((d / "verifier" / "reward.txt").read_text().strip())
        except (OSError, ValueError):
            reward = None
        trials.append({"trial_dir": d.name, "reward": reward, "exception": (d / "exception.txt").exists()})
    rows.append({"job": job, "archive": f"jobs/{a.name}", "bytes": a.stat().st_size, "sha256": sums[f"jobs/{a.name}"],
                 "tasks": tasks.get(job, []), "trials": trials})
b = out / "batch.tar.gz"
(out / "index.json").write_text(json.dumps({
    "batch": batch, "bundle": bundle, "helper": helper, "run_id": plan.get("run_id"), "jobs_dir": jobs.name,
    "batch_archive": {"archive": b.name, "bytes": b.stat().st_size, "sha256": sums[b.name]}, "jobs": rows}, indent=1) + "\n")
PY
planned=$(python3 -c 'import json,sys; print(len(json.load(open(sys.argv[1]))["jobs"]))' "$JOBS/.plan/$B.json" 2>/dev/null || echo "?")
verdicts=$(ls "$JOBS/$B"-*/*/verifier/reward.json 2>/dev/null | wc -l)
trials=$(ls -d "$JOBS/$B"-*/*/ 2>/dev/null | wc -l)
img() {   # img <ref as images/pins.json names it, tag@digest>: the local image id and digests (by the ref, else by its tag)
    local v f='{{.Id}} {{range .RepoDigests}}{{.}} {{end}}'
    v=$(docker image inspect --format "$f" "$1" 2>/dev/null) && { echo "$1 $v"; return; }
    v=$(docker image inspect --format "$f" "${1%@*}" 2>/dev/null) && { echo "$1 (by its tag) $v"; return; }
    echo "$1 (not readable here: not pulled, or docker needs sg docker)"; }
{
echo "batch:      $B"
echo "bundle:     $ID"
echo "helper:     $H"
echo "repo:       $(git -C "$R" remote get-url origin 2>/dev/null) $(git -C "$R" rev-parse --abbrev-ref HEAD 2>/dev/null) $(git -C "$R" rev-parse HEAD 2>/dev/null)$( [ -n "$(git -C "$R" status --porcelain --untracked-files=no 2>/dev/null)" ] && echo ' + local changes')"
echo "host:       $(lscpu 2>/dev/null | sed -n 's/^Model name: *//p' | head -1), $(nproc) CPUs, $(free -g | awk '/^Mem:/{print $2}') GB RAM"
echo "GPUs:       $(nvidia-smi --query-gpu=index,name,driver_version --format=csv,noheader 2>/dev/null | paste -sd ';')"
echo "software:   $(docker --version 2>/dev/null); harbor $(harbor --version 2>/dev/null)"
python3 -c 'import json, sys; print("\n".join(i["ref"] for i in json.load(open(sys.argv[1]))["images"]))' "$R/images/pins.json" 2>/dev/null \
  | { first=1; while read -r ref; do [ $first = 1 ] && printf "images:     %s\n" "$(img "$ref")" || printf "            %s\n" "$(img "$ref")"; first=0; done; }
echo "runner:     $(grep -m1 -o 'runner [0-9]*: .*' "$JOBS/$B.log" 2>/dev/null || echo '(no batch log)')"
echo "trials:     planned jobs $planned, trial dirs $trials, with a verdict $verdicts"
echo "seeds:      $(python3 -c 'import json, sys; j = json.load(open(sys.argv[1]))["jobs"]; s = sorted({x["seed"] for x in j}); print("K = %d trials per task and mode; seeds used: %s (.plan/<batch>.json)" % (max(x["trials"] for x in j), s))' "$JOBS/.plan/$B.json" 2>/dev/null || echo "?")"
echo "mirror:     $mirror_note"
echo "results:    $res_note"
echo "archives:   $(ls "$OUT/jobs" | wc -l) in jobs/ (one per job) and batch.tar.gz, $(du -sm "$OUT" | cut -f1) MB in all (index.json)"
echo "SHA256SUMS: $(sha256sum "$OUT/SHA256SUMS" | cut -d' ' -f1)"
echo "notes:      ${NOTES:-none}"
} > "$OUT/MANIFEST.md"
cat "$OUT/MANIFEST.md"

if [ "${DRY_RUN:-0}" = 1 ]; then echo "== 4. DRY_RUN=1: not uploaded; the bundle is in $OUT"; exit 0; fi
PR=(); case "${CREATE_PR:-0}" in 1|true|TRUE|yes) PR=(--create-pr) ;; esac
echo "== 4. upload to $HF_DATASET $H/$B/$ID/${PR:+ as a pull request}"
# --private: hf upload creates a missing repo, public unless told otherwise (ignored when the dataset exists)
HF_TOKEN=$(cat "$HF_TOKEN_FILE") hf upload "$HF_DATASET" "$OUT" "$H/$B/$ID" --repo-type dataset --private "${PR[@]}" \
    --commit-message "RoboARC results: $H/$B/$ID" --commit-description "$(sed -n 's/^\(runner\|trials\|notes\): *//p' "$OUT/MANIFEST.md")"
echo "Sent. Tell the maintainers: batch $B, bundle $ID, SHA256SUMS sha256 $(sha256sum "$OUT/SHA256SUMS" | cut -d' ' -f1)${PR:+, and the pull request link above}"
