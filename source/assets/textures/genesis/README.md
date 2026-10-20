# The ground plane texture

`textures/checker.png` (1902 bytes, sha256 d41a3c09dd7cd2f6b87e4dfdf479688c8e15f0e6adba52f0d598ad1f8396e6aa) is
genesis-world 1.2.1's `genesis/assets/textures/checker.png` (Genesis, Apache-2.0), byte for byte: the checker texture of
the ground plane the Nyx frames show. Until 1.0.x the renderer (`scripts/roboarc/shared/nyx_direct.py`) read it from the
installed genesis package; since 1.1.0 the image has no Genesis and installs this copy at
`/opt/topogym-visuals/genesis/textures/checker.png` (`source/images/roboarc/Dockerfile`), where nyx_direct.py finds it.
