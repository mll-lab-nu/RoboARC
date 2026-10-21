#!/usr/bin/env bash
# Push RoboARC's images from this host's local store to their registry, credential-free.
#
#   <token source> | GHCR_USER=<github user> images/roboarc/push.sh [--log <dir>] <ref>=<expected image id> ...
#
# e.g. ghcr.io/embodied-first-exam/roboarc:1.0.0=sha256:18a821a1...    (the RoboARC image)
# (Since 1.1.0 the IKEA images are pushed from their reproducible OCI archives with images/monitor/push.sh, which keeps
# their manifest digests: ikea/images/ikea-mujoco/README.md, Registry. A docker push would change those digests.)
# - The registry token (write:packages) is read from STDIN only. It lives in a temporary DOCKER_CONFIG directory that
#   is logged out and removed on exit; nothing is written to ~/.docker. The token is never printed.
# - Each local tag must be its expected image id (the verified build); a tag that is not is skipped and fails the run.
# - After each push: the pushed digest must equal the remote manifest's and the remote config digest the image id. An
#   anonymous pull of the digest is reported (new packages start private).
# Log: <dir>/push.log (default: the current directory). Exit 0 only when every image was pushed and checked.
set -u
L=./push.log
if [ "${1:-}" = "--log" ]; then L="$2/push.log"; shift 2; fi
[ $# -gt 0 ] || { echo "usage: push.sh [--log <dir>] <ref>=<image id> ..." >&2; exit 2; }
: "${GHCR_USER:?set GHCR_USER to the GitHub user that owns the token}"
for spec in "$@"; do   # check every tag before logging in
  ref=${spec%%=*}; want=${spec#*=}; case "$want" in sha256:*) ;; *) want="sha256:$want" ;; esac
  have=$(docker image inspect --format '{{.Id}}' "$ref" 2>/dev/null || true)
  case "$have" in "$want"*) ;; *) echo "$(date -Is) $ref local id '$have' is not $want: nothing pushed" | tee -a "$L"; cat > /dev/null; exit 1 ;; esac
done
d=$(umask 077; mktemp -d "$HOME/.dcfg.XXXXXX")
trap 'docker --config "$d" logout ghcr.io >/dev/null 2>&1; rm -rf "$d"' EXIT
docker --config "$d" login ghcr.io -u "$GHCR_USER" --password-stdin >/dev/null 2>&1 \
  || { echo "$(date -Is) login failed" | tee -a "$L"; exit 1; }
rc=0
for spec in "$@"; do
  ref=${spec%%=*}; have=$(docker image inspect --format '{{.Id}}' "$ref")
  echo "$(date -Is) push $ref (id $have)" >> "$L"
  pushed=$(docker --config "$d" push "$ref" 2>&1 | tee -a "${L%.log}_raw.log" | grep -o 'digest: sha256:[0-9a-f]*' | tail -1 | cut -d' ' -f2)
  raw=$(docker --config "$d" buildx imagetools inspect --raw "$ref")
  remote=$(printf '%s' "$raw" | sha256sum | awk '{print "sha256:"$1}')
  cfg=$(printf '%s' "$raw" | python3 -c 'import json,sys;print(json.load(sys.stdin)["config"]["digest"])')
  echo "  pushed_digest=$pushed remote_manifest=$remote remote_config=$cfg" >> "$L"
  if [ -n "$pushed" ] && [ "$pushed" = "$remote" ] && [ "$cfg" = "$have" ]; then echo "  OK $ref" | tee -a "$L"
  else echo "  MISMATCH $ref" | tee -a "$L"; rc=1; continue; fi
  anon=$(mktemp -d)
  if DOCKER_CONFIG="$anon" docker buildx imagetools inspect "${ref%:*}@$pushed" >/dev/null 2>&1; then
    echo "  anonymous pull: yes (public)" >> "$L"; else echo "  anonymous pull: no (the package is private)" >> "$L"; fi
  rm -rf "$anon"
done
echo "$(date -Is) done rc=$rc" >> "$L"
exit $rc
