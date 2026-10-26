# Premade MuJoCo instances: orbits-03-i01, orbits-04-i01

The two Orbits chain-tier instances were frozen in the TopoGym development tree. `release_rebuild.py` cannot make
them itself (`MUJOCO_CHAIN_SETTLE_S` has no chain entry, by design). Their `topogym_source` is the ported source
64330fb8: a rebuild under it gives the same `scene.xml` and meshes byte for byte.

Use them with `tools/release/release_rebuild.sh --instances tools/release/premade ...`, then upload them to the HF
dataset with the run's other instances (the run's REPORT.md prints the command). This folder is only for that
hand-over: remove it once the instances are on HF.
