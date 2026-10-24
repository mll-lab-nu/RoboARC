#!/bin/bash
# The maintainers' side of scripts/send_results.sh: list, fetch, check and unpack the deliveries in the maintainers'
# Hugging Face dataset.
#   receive_results.sh                                   every delivery (only the MANIFEST.md files are downloaded), one
#                                                        line each: <name>/<batch>/<bundle id>, its trial counts and its
#                                                        runner line
#   receive_results.sh <name>/<batch>/<bundle> --jobs    that delivery's jobs (only its index.json is downloaded): job,
#                                                        mode, trial, seed, each trial's reward, archive size
#   receive_results.sh <name>/<batch>/<bundle> [<job> ...]
#                                                        that delivery, or only the jobs named (and batch.tar.gz):
#                                                        download, check each archive against SHA256SUMS, unpack into
#                                                        $DEST/<name>/<batch>/<bundle>/, check every unpacked file against
#                                                        INVENTORY.tsv, then print MANIFEST.md and the batch's summary
#                                                        table. Fetching more jobs later adds them to the same directory.
# Unpacked: agents/ (Harbor's records: agents/<batch>-<mode>-<task>[-t<k>]/<trial>/, .seeds/, <batch>.log,
# <batch>.summary.md, .plan/), agents_mirror/ (when the helper's mirror was clean), results/<batch>/ (<task>/<run id>/t<k>/
# per trial and the task's trials.json), MANIFEST.md, index.json.
# Needs the hf CLI (pip install -U huggingface_hub); a private dataset also needs a token that can read it (hf auth login,
# or HF_TOKEN).
# Env (defaults): HF_DATASET (RoboARC/RoboARC-outputs; deliveries at <name>/<batch>/<bundle id>/), REVISION
#   (main; refs/pr/<n> reads a delivery sent with CREATE_PR=1 before its pull request is merged), DEST
#   (~/roboarc_helper_runs), HF (hf)
set -euo pipefail
DS=${HF_DATASET:-RoboARC/RoboARC-outputs}; REV=${REVISION:-main}; DEST=${DEST:-$HOME/roboarc_helper_runs}
HF=${HF:-hf}
if [ $# -eq 0 ]; then
    "$HF" download "$DS" --repo-type dataset --revision "$REV" --include "*/*/*/MANIFEST.md" --local-dir "$DEST/.manifests" >/dev/null
    n=0
    for m in "$DEST/.manifests"/*/*/*/MANIFEST.md; do
        [ -f "$m" ] || continue
        d=${m%/MANIFEST.md}; n=$((n + 1))
        printf '%s | %s | %s\n' "${d#$DEST/.manifests/}" "$(sed -n 's/^trials: *//p' "$m")" "$(sed -n 's/^runner: *//p' "$m")"
    done
    [ $n -gt 0 ] || echo "no deliveries in $DS"
    exit 0
fi
P=${1%/}; shift
[[ "$P" =~ ^[a-z0-9_-]+/[^/]+/[0-9]{8}T[0-9]{4}Z$ ]] || { echo "usage: receive_results.sh [<name>/<batch>/<bundle id> [--jobs | <job> ...]]" >&2; exit 2; }
B=${P#*/}; B=${B%/*}; OUT=$DEST/$P; D=$DEST/.download/$P
"$HF" download "$DS" "$P/index.json" --repo-type dataset --revision "$REV" --local-dir "$DEST/.download" >/dev/null \
    || { echo "no index.json in $P of $DS at $REV (no such delivery, or one from before per-job archives)" >&2; exit 1; }
if [ "${1:-}" = --jobs ]; then
    python3 - "$D/index.json" <<'PY'
import json, sys
ix = json.load(open(sys.argv[1]))
for j in ix["jobs"]:
    t = ", ".join(f"{x['mode']} t{x['trial']} seed {x['seed']}" for x in j["tasks"]) or "(not in the plan)"
    r = " ".join("exc" if x["exception"] and x["reward"] is None else str(x["reward"]) for x in j["trials"]) or "-"
    print(f"{j['job']} | {t} | reward {r} | {j['bytes'] / 1e6:.1f} MB")
print(f"{len(ix['jobs'])} jobs; batch.tar.gz {ix['batch_archive']['bytes'] / 1e6:.1f} MB")
PY
    exit 0
fi
sel=(batch.tar.gz)
if [ $# -eq 0 ]; then
    sel+=($(python3 -c 'import json, sys; print(" ".join(j["archive"] for j in json.load(open(sys.argv[1]))["jobs"]))' "$D/index.json"))
else
    for j in "$@"; do
        python3 -c 'import json, sys; sys.exit(sys.argv[2] not in {j["job"] for j in json.load(open(sys.argv[1]))["jobs"]})' "$D/index.json" "$j" \
            || { echo "no job $j in $P (receive_results.sh $P --jobs lists them)" >&2; exit 2; }
        sel+=("jobs/$j.tar.gz")
    done
fi
"$HF" download "$DS" $(printf "$P/%s " MANIFEST.md SHA256SUMS INVENTORY.tsv "${sel[@]}") --repo-type dataset \
    --revision "$REV" --local-dir "$DEST/.download" >/dev/null
printf '%s\n' "${sel[@]}" > "$D/.selected"
( cd "$D" && awk 'NR == FNR {s[$0]; next} substr($0, 67) in s' .selected SHA256SUMS > .selected.sha256 \
  && [ "$(wc -l < .selected.sha256)" -eq ${#sel[@]} ] && sha256sum -c --quiet .selected.sha256 ) \
    || { echo "the archives in $D do not match SHA256SUMS: not unpacked" >&2; exit 1; }
mkdir -p "$OUT"
for a in "${sel[@]}"; do tar --skip-old-files -xzf "$D/$a" -C "$OUT"; done
cp "$D/MANIFEST.md" "$D/index.json" "$OUT/"
awk -F'\t' 'NR == FNR {s[$0]; next} $1 in s {print $2 "  " $3}' "$D/.selected" "$D/INVENTORY.tsv" > "$D/.selected.files"
( cd "$OUT" && sha256sum -c --quiet "$D/.selected.files" ) \
    || { echo "files in $OUT do not match INVENTORY.tsv" >&2; exit 1; }
cat "$OUT/MANIFEST.md"; echo
cat "$OUT"/*/"$B.summary.md" 2>/dev/null || echo "(no $B.summary.md in the archive)"
echo
echo "unpacked $((${#sel[@]} - 1)) job(s) and batch.tar.gz, $(wc -l < "$D/.selected.files") files checked: $OUT"
