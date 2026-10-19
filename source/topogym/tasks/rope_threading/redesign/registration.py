"""Product registration of the redesigned rope_threading tiers (rope agent, 2026-10-11): the variation records and the
MuJoCo profile entries the grader's build needs, in one place. session_build.register uses the same functions
in-process; patches/rope_threading_redesign_registration.patch wires them into topogym/config/variation_records.py and
topogym/world/mujoco/profiles.py.

Profile entry per tier = the family's profile (rope_chain_v3) with
- the tier's ring names split into their exact convex tube segments (as R1-R3 are), and
- the family's ring rule (static_contact.keep, ccd4ba7fa: the rings, stands and bases keep the task's nearly hard contact
  against the robot) applied to the tier's own ring entities: the ring, and its _stand / _base (a ring on a post) or its
  _pedestal (a low ring). Keep keys must name existing entities (the build refuses unknown ones).
"""
from __future__ import annotations

TIERS = ("very_easy", "easier", "easy", "medium", "hard")
IMPLEMENTATION = "topogym.tasks.rope_threading.redesign.task:{cls}"


def variation_id(tier: str) -> str:
    return f"rope_threading_rd_{tier}"


def ring_entities(tier: str) -> dict:
    """{entity name: reason} of the tier's rings and their mounts."""
    from . import frozen as S
    keep = {}
    for r in S.load(tier).rings:
        keep[r["name"]] = "redesign ring: the family's ring rule (as R[0-9]*)"
        for part in {"post": ("stand", "base"), "pedestal": ("pedestal",)}.get(r["mount"], ()):
            keep[f"{r['name']}_{part}"] = f"redesign ring {part}: the family's ring rule (as R[0-9]*)"
    return keep


def profile_entry(tier: str, base: dict) -> dict:
    """The tier's MuJoCo profile entry from the family's base entry."""
    from . import frozen as S
    names = [r["name"] for r in S.load(tier).rings]
    entry = dict(base, name=base.get("name", "rope_chain") + "_rd",
                 split={n: "topogym.assets.geometry.rope_threading:ring_hulls" for n in names})
    sc = base.get("static_contact")
    if sc is not None and "keep" in sc:
        entry["static_contact"] = dict(sc, keep=ring_entities(tier))
    return entry


def profile_entries(base: dict) -> dict:
    """{("rope_threading", variation id): entry} for every tier."""
    return {("rope_threading", variation_id(t)): profile_entry(t, base) for t in TIERS}


def variation_records() -> list:
    """The variation records (topogym.config.variation_records format) of the five tiers."""
    from .task import CLASSES
    out = []
    for t in TIERS:
        cls = CLASSES[t]
        out.append(dict(task_id="rope_threading", variation_id=variation_id(t), difficulty=t, family_id="bulkhead_lacing",
                        model_revision=cls.model_revision, implementation=IMPLEMENTATION.format(cls=cls.__name__),
                        aliases=[], status="integrated", reason=None, validation="redesign_dev", pool="redesign"))
    return out
