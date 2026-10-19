"""The rope_threading redesign tiers' hand plans per ring (which hand feeds and which receives; the regrasps):
authoring and oracle data, not part of the simulator image. legs_for(design) gives a TierDesign's plan
(topogym.planning.rope_threading.redesign.scenes builds the tiers with legs = [])."""


def _legs(rings, plan):
    out = []
    for r in rings:
        f, rc = plan[r.name]
        out.append(dict(ring=r.name, feeder=f, receiver=rc if rc is not None else f, last=r is rings[-1]))
    return out


PLANS = {
    'Bulkhead lacing, x and y rings': ('list', [dict(ring="G1", feeder="right", receiver="left"), dict(ring="A", feeder="left", receiver="left", keeper="right"), dict(ring="G2", feeder="left", receiver="right"), dict(ring="B", feeder="right", receiver="right")]),
    'Bulkhead lacing with a downward eyelet': ('list', [dict(ring="G1", feeder="right", receiver="left"), dict(ring="V", feeder="left", receiver="left", keeper="right"), dict(ring="G2", feeder="left", receiver="right"), dict(ring="B", feeder="right", receiver="right")]),
    'Bulkhead lacing under a shelf': ('list', [dict(ring="G1", feeder="right", receiver="left"), dict(ring="V", feeder="left", receiver="left", keeper="right"), dict(ring="G2", feeder="left", receiver="right"), dict(ring="B", feeder="right", receiver="right")]),
    'Bulkhead lacing, tilted ring under a shelf': ('list', [dict(ring="G1", feeder="right", receiver="left"), dict(ring="V", feeder="left", receiver="left", keeper="right"), dict(ring="G2", feeder="left", receiver="right"), dict(ring="T", feeder="right", receiver="right")]),
    'Bulkhead lacing, five passages': ('list', [dict(ring="G1", feeder="right", receiver="left"), dict(ring="V", feeder="left", receiver="left", keeper="right"), dict(ring="G2", feeder="left", receiver="right"), dict(ring="T", feeder="right", receiver="right"), dict(ring="G3", feeder="right", receiver="left")]),
    'Bulkhead lacing v1: two grommets, two low rings': ('plan', {"G1": ("right", "left"), "A": ("left", "left"), "G2": ("left", "right"), "B": ("right", None)}),
    'Bulkhead lacing v1: a downward eyelet last': ('plan', {"G1": ("right", "left"), "A": ("left", "left"), "G2": ("left", "right"), "V": ("right", None)}),
    'Bulkhead lacing v1: three wall passages': ('plan', {"G1": ("right", "left"), "A": ("left", "left"), "G2": ("left", "right"), "B": ("right", "right"), "G3": ("right", None)}),
    'Bulkhead lacing v1: a ring turned off both arms': ('plan', {"G1": ("right", "left"), "A": ("left", "left"), "G2": ("left", "right"), "T": ("right", "right"), "G3": ("right", None)}),
    'Bulkhead lacing v1: a ring turned off both arms (alt)': ('plan', {"G1": ("right", "left"), "A": ("left", "left"), "G2": ("left", "right"), "T": ("right", "right"), "G3": ("right", None)}),
    'Bulkhead lacing v1: six rings, three wall passages': ('plan', {"G1": ("right", "left"), "A": ("left", "left"), "G2": ("left", "right"), "T": ("right", "right"), "G3": ("right", "left"), "C": ("left", None)}),
}


def legs_for(design):
    """The design's per-ring hand plan (design.legs when the design carries one)."""
    if getattr(design, "legs", None):
        return design.legs
    kind, plan = PLANS[design.title]
    return [dict(x) for x in plan] if kind == "list" else _legs(list(design.rings), plan)
