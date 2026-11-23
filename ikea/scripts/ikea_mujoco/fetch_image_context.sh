#!/usr/bin/env bash
# Prepare the roboarc-ikea build context images/ikea-mujoco/context/:
#   - the scenes' meshes, <task>/assets/<file>, from the Hugging Face dataset RoboARC/RoboARC-ikea-assets
#     (https://huggingface.co/datasets/RoboARC/RoboARC-ikea-assets): PUBLISHED below maps the revision pinned in
#     scripts/ikea_mujoco/assets.json ("repo", "revision") to the public dataset commit with exactly those files.
#     ROBOARC_IKEA_ASSETS_SOURCE=<repo>@<revision> names another copy instead. Every file's sha256 is checked against
#     the same file's per-task "files" entries, whatever the source;
#   - the scene code from git, images/ikea-mujoco/scenes/<task>/;
#   - then scripts/ikea_mujoco/stage_context.py on the two together, which checks every staged file again and writes the
#     context (scenes/, assets.json, bytepatch_libmujoco.py, requirements.lock).
# The dataset is public and read without a token (a token in the environment is never printed). Needs huggingface_hub
# (pip install huggingface_hub).
# usage: scripts/ikea_mujoco/fetch_image_context.sh [<download dir>]   (default: $RCB_IKEA_ASSETS or ~/.cache/ikea-mujoco-assets)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"; HERE="$ROOT/scripts/ikea_mujoco"
DL="${1:-${RCB_IKEA_ASSETS:-$HOME/.cache/ikea-mujoco-assets}}"
python3 - "$HERE/assets.json" "$DL" <<'PY'
import hashlib, json, os, pathlib, re, sys
pin, dl = json.load(open(sys.argv[1])), pathlib.Path(sys.argv[2])
if not re.fullmatch(r"[0-9a-f]{40}", pin.get("revision", "")):
    sys.exit(f"scripts/ikea_mujoco/assets.json: revision {pin.get('revision')!r} is not a commit of {pin['repo']}")
# assets.json "revision" -> (repo, revision) of the public dataset commit with exactly the pinned asset files
PUBLISHED = {"610b424b3d4152d8990e226eae60b9170ac7dbdf": ("RoboARC/RoboARC-ikea-assets",
                                                         "471bcc0753985d1778fb7ba938156f3448204e4f")}
src = os.environ.get("ROBOARC_IKEA_ASSETS_SOURCE", "")
repo, at, rev = src.partition("@")
if src and not (repo and at and rev):
    sys.exit(f"ROBOARC_IKEA_ASSETS_SOURCE must be <repo>@<revision>, not {src!r}")
repo, rev = (repo, rev) if src else PUBLISHED.get(pin["revision"], (pin["repo"], pin["revision"]))
want = {f"{t}/{f}": h for t, rec in pin["tasks"].items() for f, h in rec["files"].items() if f.startswith("assets/")}
try:
    from huggingface_hub import snapshot_download
except ImportError:
    sys.exit("needs huggingface_hub: pip install huggingface_hub")
dl.mkdir(parents=True, exist_ok=True)
try:    # huggingface_hub sends a token when it has one and reads a public dataset without
    snapshot_download(repo_id=repo, repo_type=pin["repo_type"], revision=rev, local_dir=str(dl),
                      allow_patterns=[f"{t}/assets/**" for t in pin["tasks"]])
except Exception as e:  # noqa: BLE001  the message names the repository, never the token
    sys.exit(f"download of {repo}@{rev[:12]} failed ({type(e).__name__}). The dataset is read without a token, and a "
             "token the Hub rejects stops the download: unset HF_TOKEN or run `hf auth logout`. "
             "ROBOARC_IKEA_ASSETS_SOURCE=<repo>@<revision> downloads from another copy")


def sha(p):
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


bad = [rel for rel, h in sorted(want.items()) if not (dl / rel).is_file() or sha(dl / rel) != h]
extra = sorted(str(p.relative_to(dl)) for t in pin["tasks"] for p in (dl / t / "assets").rglob("*")
               if p.is_file() and str(p.relative_to(dl)) not in want)
if bad or extra:
    sys.exit("FAIL meshes differ from scripts/ikea_mujoco/assets.json: " + ", ".join((bad + extra)[:10])
             + (f" ... ({len(bad) + len(extra)} files)" if len(bad) + len(extra) > 10 else ""))
print(f"ok {repo}@{rev[:12]}: {len(want)} mesh files match scripts/ikea_mujoco/assets.json ({dl})")
PY
TREE="$(mktemp -d)"; trap 'rm -rf "$TREE"' EXIT
for d in "$ROOT"/images/ikea-mujoco/scenes/*/; do
  t="$(basename "$d")"; mkdir -p "$TREE/scenes/$t"
  cp -r "$d." "$TREE/scenes/$t/"; cp -r "$DL/$t/assets" "$TREE/scenes/$t/assets"
done
python3 "$HERE/stage_context.py" "$TREE"
