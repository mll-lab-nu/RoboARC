eket SIGSEGV in MuJoCo 3.14.0 (2026-10-15, stress test)

Cause: EPA in engine_collision_gjk.c keeps the polytope horizon in two fixed 24-int buffers (horizon.indices / edges,
mjc_ccdSize and mjc_ccd) whatever ccd_iterations is; addEdge writes past them. The IKEA scenes' ccd_iterations 500 +
ccd_tolerance 1e-10 let EPA grow polytopes of 50-60 vertices whose horizon reaches 25+ edges (a dowel cylinder pressed
into a thin convex mesh piece), and the overflow corrupts the collision buffer: SIGSEGV in mj_narrowphase. MuJoCo 3.15+
sizes the buffers 24 + max(0, iterations - 35) and bounds-checks them (warning, no contact) - backported here.

- eket_dowel4_side_a_c059.xml, eket_dowel3_side_b_c064.xml: standalone two-body models (compiled mesh vertices / faces,
  the geoms' poses at the crashing step, the scenes' <option>): `python check.py <xml>` gives rc -11 on MuJoCo 3.14.0
  (pip), no crash on the patched build or on 3.16.0.
- segv_extract.py: builds such a model from a scene's pre-crash state (crashgeoms-style two-geom isolation first).
- variants.sh: the option matrix on a standalone model (3.14.0 vs 3.16.0).
- mujoco-3.14.0-epa-horizon-backport.patch (superseded by images/ikea-mujoco/mujoco-3.14.0-epa/; not used): an early backport (the horizon sizing, the bounds check in addEdge and its
  propagation through horizonRec / horizon, the "out of memory for horizon edges" warning path); nothing else.
- Dockerfile.mjpatch (not used: not bit-identical; see images/ikea-mujoco/mujoco-3.14.0-epa/): builds stock and patched libmujoco.so.3.14.0 from the 3.14.0 tag with clang/lld 20.1.8 (the
  toolchain the PyPI wheel's .so names), upstream's default Release options (-O3 -flto=thin -mavx), FROM python:3.12-slim.
- The fix in use: images/ikea-mujoco/mujoco-3.14.0-epa/bytepatch_libmujoco.py (a 14-byte patch of the wheel's own .so; README there).
