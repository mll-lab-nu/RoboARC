# The Genesis RoboTangle release generator (archive)

RoboARC's tasks were first released as RoboTangle on Genesis physics (task bases `<family>-<tier>-i00`, image
`eai-robotangle:0.2.22`). This folder keeps that release's generator and records:

| file | what it was |
|---|---|
| `mk_tasks.py` | the RoboTangle task generator (Genesis bases; also MuJoCo dev task dirs with `--dev`) |
| `release.json` | the RoboTangle release record: release commit, simulator source, runtime, uipc builds and per-base evidence |
| `assets.json`, `assets.py` | the pins of the private HF dataset that held its instances and image build context, and the fetch/check tool |
| `nop_records.json`, `import_evidence.py` | the nop records and the one-off import of the oracle evidence |
| `check_context.py`, `fetch_image_context.sh` | the Genesis base image's build context check and fetch |

It is not runnable here: it reads `scripts/robotangle/` paths, the grader copies it rendered with were older than the
graded ones and have been removed, and its task dirs are not in this repository. The current generator is
`scripts/roboarc/mk_tasks.py`; the Genesis base image the release image builds on is pinned by digest in
`source/images/robotangle/Dockerfile.mujoco`, its recipe is in `images/roboarc/`. The Genesis-era task texts that new
bases still start from are `tools/release/templates/`, and the Genesis references `tools/references/oracles_genesis/`.
