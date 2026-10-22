#!/usr/bin/env bash
# Pull every image the tasks pin, then check each one's image id (images/README.md).
#
#   bash images/pull_all.sh [--pins <pins.json>] [--list]
#
# images/pins.json lists the images in pull order (a base before the image built FROM it), each with the reference the
# tasks pin (<repo>:<tag>@sha256:<manifest digest>) and its image id (the config digest). For each: `docker pull <ref>`,
# then `docker image inspect <ref>` must report the pinned id. It uses the Docker login you already have
# (`docker login ghcr.io` once, with a token that can read the packages); it never asks for, reads or prints a token.
# --list: print "<ref> <id>" per image and pull nothing. Exit 0 only when every image was pulled and has its pinned id.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PINS="$ROOT/images/pins.json" LIST=0
while [ $# -gt 0 ]; do
  case "$1" in
    --pins) PINS="$2"; shift 2 ;;
    --list) LIST=1; shift ;;
    *) echo "usage: bash images/pull_all.sh [--pins <pins.json>] [--list]" >&2; exit 2 ;;
  esac
done
# "<ref> <id>" per pinned image, in the file's order; a malformed entry stops everything before any pull
rows="$(python3 - "$PINS" <<'PY'
import json, re, sys
REF = re.compile(r"^[a-z0-9.-]+(?::[0-9]+)?/[a-z0-9._/-]+:[A-Za-z0-9._-]+@sha256:[0-9a-f]{64}$")
ID = re.compile(r"^sha256:[0-9a-f]{64}$")
try:
    images = json.load(open(sys.argv[1]))["images"]
except (OSError, ValueError, KeyError, TypeError) as e:
    sys.exit(f"pull_all: cannot read the pins in {sys.argv[1]}: {e}")
if not isinstance(images, list) or not images:
    sys.exit(f"pull_all: {sys.argv[1]} pins no image")
for p in images:
    ref, iid = (p or {}).get("ref", ""), (p or {}).get("id", "")
    if not REF.match(ref) or not ID.match(iid):
        sys.exit(f"pull_all: {p.get('name', '?')}: not a pinned <repo>:<tag>@sha256 reference with an image id")
    print(ref, iid)
PY
)" || exit 2
if [ "$LIST" = 1 ]; then printf '%s\n' "$rows"; exit 0; fi
bad=0
while read -r ref id; do
  echo "pull $ref"
  if ! docker pull -q "$ref" >/dev/null; then
    echo "FAIL $ref: docker pull failed (logged in to the registry? \`docker login ghcr.io\`)" >&2; bad=1; continue
  fi
  got="$(docker image inspect --format '{{.Id}}' "$ref" 2>/dev/null)"
  if [ "$got" != "$id" ]; then
    echo "FAIL $ref: image id ${got:-none}, pinned $id" >&2; bad=1
  else
    echo "ok   $ref = ${id:7:12}"
  fi
done <<< "$rows"
[ "$bad" = 0 ] && echo "pull_all: every pinned image is local with its pinned id" || echo "pull_all: some images are missing or differ (above)" >&2
exit $bad
