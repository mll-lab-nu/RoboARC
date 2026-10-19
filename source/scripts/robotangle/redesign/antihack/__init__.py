"""antihack: shared inverse-generation / anti-hack tooling of the soft-body redesign (rope_threading, knot_untying,
blanket_flatten on the MuJoCo port). See ../README.md.

Modules (import what you need; the heavy ones import mujoco / topogym lazily):
  antihack.aloha       ALOHA constants and a stateless TCP forward kinematics on a frozen MuJoCo scene (no Session)
  antihack.trajectory  trajectory complexity from a row stream (+FK) or from a planned waypoint / phase list
  antihack.objects     object pose change start -> goal (rope chain, cloth particles), crossings, ring passages
  antihack.report      the antihack_report.json schema (one for all families), old-vs-new comparison and verdict
  antihack.record      replay rows open-loop in a MuJoCo Session and record states (CLI; used by the harness)
  antihack.harness     old-oracle replay on a new scene: as-is and retargeted (CLI)
  antihack.breakers    heuristic breaker policies (kinematic / object-level first; Session versions)
"""
SCHEMA = "robotangle-redesign-antihack/1"
__all__ = ["SCHEMA"]
