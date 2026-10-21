# roboarc-monitor (the resource-monitor sidecar's image)

`ghcr.io/embodied-first-exam/roboarc-monitor:<VERSION>` (this directory's `VERSION`) runs the `rcb_monitor` service
of every task (docs/PROTOCOL.md §2.10). It is Python and nothing else: the monitor's code is not in the image, each
task mounts its own copy read-only (`environment/rcb_monitor/`, part of the task's DIGEST). Every task pins the image
by its manifest digest, `scripts/rcb_monitor/inject.py` `IMAGE`:

    ghcr.io/embodied-first-exam/roboarc-monitor:0.1.0@sha256:ea1b0d4b859998d6a47e2cb7a72c3a9ae85870b32c75de1bc1448e387da07093

0.1.0 has the same filesystem as the monitor image RoboARC 1.0.0 pinned (`ghcr.io/mll-lab-nu/efe:monitor-0.1.0@sha256:f2c910f5…`):
the same five layer diff ids (Python 3.12.15 on Debian bookworm, and `/var/rcb-monitor`), the same environment,
working directory and command (`python3`). Its config adds three OCI labels (title, version, source).

## Build

```bash
images/monitor/build.sh [--out <file>] [--load]
```

builds `Dockerfile` reproducibly into an OCI archive (default `./roboarc-monitor-<VERSION>.oci.tar`): a throwaway
BuildKit builder of a pinned version (`docker-container` driver), the base pinned by digest, no provenance or SBOM
attestations, every timestamp set to a fixed `SOURCE_DATE_EPOCH`. Two builds give the same archive byte for byte, so
its manifest digest is known before anything is published; `build.sh` prints it and exits 1 when it is not the digest
the tasks pin. `--load` also loads it into the local Docker image store as `ghcr.io/embodied-first-exam/roboarc-monitor:<VERSION>`.
A local tag is not the pinned reference: Docker resolves `<name>@sha256:<digest>` only for an image pulled from a
registry, so a task runs from a local build only after its `image:` line is changed to the tag (testing only).

## Push

`images/push_all.sh` pushes the archive byte for byte with RoboARC's other images (skopeo `--preserve-digests`, in a
container pinned by digest), so the registry's digest is the pinned one; images/README.md. The pin is also in
`images/pins.json` (`ci_checks/check-image-pins.py` checks that it is `inject.py`'s `IMAGE`).

## Changing the image

Change `Dockerfile`, bump `VERSION`, run `build.sh`, put the printed reference into `inject.py` `IMAGE` and
`images/pins.json`, re-render every task (both generators; docs/DEVELOPING.md), and push the archive before the tasks
are published.
