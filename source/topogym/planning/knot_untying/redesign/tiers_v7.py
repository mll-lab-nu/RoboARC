"""knot_untying redesign v7 (coordinator 2026-10-10 19:10; main approved 18:57): the v6 tiers with the oracle's new
moves, plus a looser easy.

easy v7: easy's stacked overhands authored with 30 mm loops (v5/v6: 20 mm). The 1 N oracle on easy_v6 slipped on 33 % of
its strokes, and a single finger dragging the tail cinched the overhands to 18.6 N, so they cinch readily. The two
trefoils move apart (centres 0.20 m apart instead of 0.16 m) and down 1.5 cm so the bigger lobes keep their gaps; the
clutter posts move out with them. Everything else is easy_v5's: anchored standing end behind the curb, the region,
the end path through the front-right, the oracle order (k2 on the free-end side first)."""
from __future__ import annotations

from . import compose as C
from topogym.tasks.knot_untying.redesign import geometry as G
from . import tiers_v5
from .tiers import _finish, block, post, rect, serpentine
from .tiers_v5 import _birth, check  # noqa: F401  (check re-exported for the scene scripts)


def easy(tighten=None, a=.030, dx=.10, y=.035):
    """Stacked overhands (two trefoils in series, 6 crossings) with 30 mm loops, anchored, clutter, region."""
    k1 = G.place(G.trefoil_core(a)[::-1], (-dx, y))
    k2 = G.place(G.trefoil_core(a)[::-1], (dx, y))
    # the approach to k1 from below-left and the link below the cores keep the smallest bend radius at 10.4 mm (v5:
    # 10.2 mm; the first v7 draft's straight approach bent to 6 mm at k1's entry)
    rope = C.rope_through(start_points=[(-.38, .215), (-.31, .13), (-.20, -.08)], cores=[k1, k2],
                          links=[[(0., -.08)]],
                          end_points=[(.20, -.09), (.26, -.17), (.27, -.27), (.16, -.32), (.04, -.30)])
    fixtures = [post("post_away", (.235, .03)), post("post_front", (-.10, -.10)),
                block("curb", (-.05, .135), (.40, .02, .03), role="curb")]
    regions = {"rope": rect(-.43, .32, .155, .31)}
    oracle = dict(kind="anchored_withdrawal_two_knots_then_layout", rope="rope", order=["k2 (free-end side)", "k1"],
                  pin="over_strand_next_to_crossing", stroke_max_m=.16, layout_region=True, layout_lift_over=["curb"],
                  layout_path={"rope": serpentine((-.38, .215), [(.27, .18), (-.40, .24), (.30, .295)], .03)},
                  moves="v7: clear the tail after each deletion")
    return _finish("easy", [rope], anchors=[dict(name="rope", anchor=dict(end="start", links=2))],
                   fixtures=fixtures, regions=regions, birth=_birth(tighten={"rope": tighten} if tighten else None),
                   oracle=oracle, notes=f"v7: two loose overhands (loops a {a * 1e3:.0f} mm) {2 * dx * 100:.0f} cm apart in "
                                        "the bimanual lens; clamp in the back region behind the curb")


TIERS = dict(tiers_v5.TIERS, easy=easy)


def build(tier, **kw):
    return TIERS[tier](**kw)
