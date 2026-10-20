# records/

Inputs and archives of earlier releases, kept for provenance. Nothing here is read by the task generator, the image
build or the graders.

| path | what it is |
|---|---|
| `freeze_manifests/freeze_manifest.v2.json` | the inputs of the 0.2.0 freeze (`tools/release/release_rebuild.py --manifest`): 98 bases, references, expected end states |
| `freeze_manifests/freeze_manifest.v3.json`, `freeze_manifest.v3_knot.json` | the inputs of the 0.2.1 freeze (V3), which release 1.0.0 carries forward file for file: 101 bases |
| `genesis_release/` | the Genesis RoboTangle release generator (archive; see its README) |
| `mujoco-3.14.0-epa-source-build/` | the source build of MuJoCo 3.14.0 with the upstream EPA fix that was tried and not used (not bit-identical to the wheel; [docs/simulator/MUJOCO_EPA_PATCH.md](../docs/simulator/MUJOCO_EPA_PATCH.md)) |

In the freeze manifests, each base's `reference` is relative to the manifest and points at `scripts/roboarc/oracles/`;
its `sha256` is the file the freeze used. All of v3's match the files in this repository; in v2, blanket-fold-05-i01's
reference is the one V3 replaced (V3 re-recorded the route with the oracle rate limiter's guard-stop clamp).
