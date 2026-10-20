#!/usr/bin/env bash
# Push the monitor image built by images/monitor/build.sh to its registry, byte for byte, credential-free.
#
#   <token source> | GHCR_USER=<github user> images/monitor/push.sh [--log <dir>] <archive> [<repo>:<tag>@<digest>]
#
# <archive> is build.sh's OCI archive; the target defaults to the reference every task pins (scripts/rcb_monitor/
# inject.py `IMAGE`), e.g. ghcr.io/embodied-first-exam/roboarc-monitor:0.1.0@sha256:...
# - Before logging in: the archive's manifest digest must be the target's digest; otherwise nothing is pushed.
# - The copy runs in a skopeo container (pinned by digest) with --preserve-digests: the registry stores the archive's
#   manifest bytes unchanged, so the pushed digest is the pinned one. The registry token (write:packages) is read from
#   STDIN only and passed to that container's stdin; its login file lives in the container's tmpfs and is gone when the
#   container exits. Nothing is written to ~/.docker; the token is never printed.
# - After the copy: the registry's manifest for <repo>@<digest> is fetched and hashed; it must be the target's digest.
#   An anonymous fetch of the digest is reported (new packages start private; the tasks need it public).
# A registry on localhost / 127.0.0.1 is used without TLS (a local test registry).
# Log: <dir>/push.log (default: the current directory). Exit 0 only when the image was pushed and checked.
set -u
SKOPEO="quay.io/skopeo/stable:v1.20.0@sha256:47853bb9fb24202af9110531ebd6e43c5f97701254ca290596640290d17942f4"
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
L=./push.log
if [ "${1:-}" = "--log" ]; then L="$2/push.log"; shift 2; fi
[ $# -ge 1 ] && [ $# -le 2 ] || { echo "usage: push.sh [--log <dir>] <archive> [<repo>:<tag>@<digest>]" >&2; exit 2; }
: "${GHCR_USER:?set GHCR_USER to the GitHub user that owns the token}"
ARCHIVE="$(realpath "$1")"
TARGET="${2:-$(python3 -c 'import re,sys; print(re.search(r"^IMAGE = \"([^\"]+)\"", open(sys.argv[1]).read(), re.M).group(1))' \
  "$ROOT/scripts/rcb_monitor/inject.py")}"
WANT="${TARGET##*@}"; NAME="${TARGET%@*}"; REPO="${NAME%:*}"; REG="${REPO%%/*}"
case "$WANT" in sha256:*) ;; *) echo "the target must be <repo>:<tag>@sha256:<digest>: $TARGET" >&2; cat > /dev/null; exit 2 ;; esac
HAVE="$(python3 - "$ARCHIVE" <<'PY' 2>/dev/null
import json, sys, tarfile
(m,) = json.load(tarfile.open(sys.argv[1]).extractfile("index.json"))["manifests"]
print(m["digest"])
PY
)"
if [ "$HAVE" != "$WANT" ]; then
  echo "$(date -Is) $ARCHIVE manifest '$HAVE' is not $WANT: nothing pushed" | tee -a "$L"; cat > /dev/null; exit 1
fi
TLS=""; case "$REG" in localhost:*|127.0.0.1:*) TLS="--tls-verify=false" ;; esac
echo "$(date -Is) push $ARCHIVE -> $TARGET" >> "$L"
# In the container: log in (password from stdin), copy, hash the registry's manifest, then an anonymous fetch.
out=$(docker run --rm -i --network host --memory 1g --label roboarc-push=monitor --tmpfs /auth:mode=0700 \
  -e "U=$GHCR_USER" -e "REG=$REG" -e "REPO=$REPO" -e "NAME=$NAME" -e "WANT=$WANT" -e "TLS=$TLS" \
  -v "$ARCHIVE:/in/image.oci.tar:ro" --entrypoint sh "$SKOPEO" -c '
    set -u; A=/auth/auth.json
    skopeo login $TLS --authfile "$A" -u "$U" --password-stdin "$REG" >/dev/null 2>&1 || { echo "result=login-failed"; exit 1; }
    skopeo copy $TLS --preserve-digests --authfile "$A" oci-archive:/in/image.oci.tar "docker://$NAME" >&2 \
      || { echo "result=copy-failed"; skopeo logout --authfile "$A" "$REG" >/dev/null 2>&1; exit 1; }
    echo "remote=sha256:$(skopeo inspect $TLS --raw --authfile "$A" "docker://$REPO@$WANT" | sha256sum | cut -d" " -f1)"
    echo "remote_tag=sha256:$(skopeo inspect $TLS --raw --authfile "$A" "docker://$NAME" | sha256sum | cut -d" " -f1)"
    skopeo logout --authfile "$A" "$REG" >/dev/null 2>&1
    if skopeo inspect $TLS --raw --no-creds "docker://$REPO@$WANT" >/dev/null 2>&1; then echo "anonymous=yes"; else echo "anonymous=no"; fi
  ' 2>>"${L%.log}_raw.log")
echo "$out" | sed 's/^/  /' >> "$L"
remote=$(echo "$out" | sed -n 's/^remote=//p'); tag=$(echo "$out" | sed -n 's/^remote_tag=//p')
if [ "$remote" = "$WANT" ] && [ "$tag" = "$WANT" ]; then
  echo "  OK $TARGET" | tee -a "$L"
  case "$out" in *anonymous=yes*) echo "  anonymous pull: yes (public)" >> "$L" ;;
                 *) echo "  anonymous pull: no (the package is private; the tasks need it public)" | tee -a "$L" ;; esac
  echo "$(date -Is) done rc=0" >> "$L"; exit 0
fi
echo "  MISMATCH or failure $TARGET (remote '$remote', tag '$tag')" | tee -a "$L"; echo "$(date -Is) done rc=1" >> "$L"; exit 1
