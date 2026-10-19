"""Material routes for the frozen Blanket representatives.

A route is a sequence of stages. Each stage names one or two material grasp points
(rest UV, metres) and piecewise-linear patch targets; a segment without `to`
holds the hands still. Targets are world XY and height above the table top of
the held patch centre. The hands release at the end of every stage; the next
stage regrasps on the live cloth. The virtual-hand witness and the robot oracle
read the same route; the robot adds its TCP offset above the pinched patch.

The robot runner passes PATCH to the assist's SoftGraspTarget, so the witness
pins what the real assist would hold, not an idealised whole edge.

`settle_s` bounds how long the robot waits, hands withdrawn, for the Task's own
settled predicate after the last release (10 s unless a route needs longer).
`max_strain_ratio` is the assist's material-edge guard for the route. The
default 1.3 is kept wherever a route stays below it. A route that must carry
the cloth's weight on two pinches declares a higher value explicitly; see the
proof README for the trade-off.
"""

PATCH = dict(spatial_radius_m=.018, material_radius_m=.018)
# Routes target the current frozen states (gathered_archive_v2). The accepted proofs ran on gathered_scene_v1, where
# the cloth differs by up to 19 mm; their routes are in each run's source_snapshot. On v2 three pinches were covered
# by another layer, so each moved to the nearest exposed point and, for absolute targets, its hand's targets moved
# by the same offset.
DEFAULT_STRAIN_GUARD = 1.3

# easy / gathered_scene_v1 / seed 0: pleats run along X and gather the width
# from 0.48 m to about 0.31 m. Pinch both corners of the far long edge
# (v = +0.24 side, 2 cm inset), lift 10 cm and drag outward so that the near
# edge stays anchored by table friction while tension opens the pleats; stop
# short of a taut sheet, then finish the spread while lowering. A full hang
# (0.5 m lift) also flattens but carries the whole weight on two patches
# (peak edge ratio about 1.5 near the pinch, above the assist's 1.3 guard).
# (2026-10-14, blanket_vbd_gpu, the -i01 birth settled on the GPU step) the far-edge drag leaves one pleat across the
# whole width near the near edge (rest angles set by the plastic bending: material v -0.13..-0.06 gathered into ~2 cm,
# 20 mm high; coverage 0.922, height 30.4 mm). A second stage pinches the near edge at u +-0.15 (the corners are out of
# top-down reach) and pulls it 4 cm toward the robot, 5 cm up, then lowers it 3 cm above the table: coverage 0.987 at the
# first settle, 0.9998 and 5 mm at success (1581 rows). The fold layout and the 40 s settle are MEDIUM's.
EASY = dict(
    max_strain_ratio=DEFAULT_STRAIN_GUARD,
    settle_s=40.,
    fold_layout=dict(settle_s=10., deviation_m=.03, corner_radius_m=.15, min_particles=25, max_stages=3, creases=True,
                     edge_band_m=.12, lay="drop", beyond_m=0.),
    stages=(
        dict(
            grasp_uv=dict(left=(-.27, .234), right=(.28, .22)),   # v1: left (-.28, .22)
            segments=(
                dict(name="lift", duration_s=1.5, to=dict(left=(-.395, .304, .10), right=(.135, .28, .10))),
                dict(name="drag", duration_s=6., to=dict(left=(-.405, .404, .10), right=(.145, .39, .10))),
                dict(name="lower", duration_s=3., to=dict(left=(-.405, .454, .015), right=(.145, .44, .015))),
            ),
        ),
        dict(
            grasp_uv=dict(left=(-.15, -.22), right=(.15, -.22)),
            segments=(
                dict(name="lift", duration_s=1.5, by=dict(left=(0., 0., .05), right=(0., 0., .05))),
                dict(name="pull", duration_s=4., by=dict(left=(0., -.04, .05), right=(0., -.04, .05))),
                dict(name="lower", duration_s=2., by=dict(left=(0., -.04, .03), right=(0., -.04, .03))),
            ),
        ),
    ),
)

# medium / seed 0: the partly relaxed twist leaves the far edge on the far side
# along its length, but the last ~8 cm of the far-left edge is tucked under
# (only its fold crest, v = +0.16, passes the assist's exposure test) and the
# far-right corner is folded. Folds left on the table keep jittering under PBD
# self-contact, so the Task's settled gate never holds; the route must lay
# every edge out. Pinch the left crest and the exposed far-right point, drag
# outward past the flat position at 10 cm so the tucked strip hangs below the
# left hand, then lower while moving that hand back: the hanging strip lands
# outward instead of under. Two end-of-lowering spikes at the left pinch were
# removed: the V-shaped crest patch is released 3 cm above the table (pressing
# its lower layer into the table drove the edge ratio to ~1.7), and the hand
# moves back only 3 cm (a longer move stretches the strip once friction holds
# it on the table: 1.51 in a robot run). The laid-out edge keeps relaxing for
# ~20 s after release before the sheet comes to rest. Over-pulling the far
# edge still loads the pinch slightly past the default guard (about 1.31),
# hence the declared 1.5.
MEDIUM = dict(
    max_strain_ratio=1.5,
    settle_s=40.,
    # Up to three measured fold layouts after the route, as hard and easier declare, creases included (2026-10-10,
    # MuJoCo); a crease is laid with the drop-then-lay profile, in reach (beyond_m 0). A route that already covers adds
    # only the layout's settle.
    fold_layout=dict(settle_s=10., deviation_m=.03, corner_radius_m=.15, min_particles=25, max_stages=3, creases=True,
                     edge_band_m=.12, lay="drop", beyond_m=0.),
    stages=(
        dict(
            grasp_uv=dict(left=(-.27, .156), right=(.26, .20)),   # v1: left (-.27, .16)
            segments=(
                dict(name="lift", duration_s=1.5, to=dict(left=(-.407, .251, .10), right=(.115, .249, .10))),
                dict(name="drag", duration_s=6., to=dict(left=(-.407, .476, .10), right=(.115, .46, .10))),
                dict(name="lower", duration_s=3., to=dict(left=(-.407, .446, .03), right=(.115, .50, .03))),
            ),
        ),
        # (2026-10-15, blanket_vbd_gpu, the -i01 birth 49ee5e7e16239616 settled on the GPU step) the birth carries a
        # standing pleat across the whole width (material v -0.11..-0.01: ~14 cm of cloth in 3-6 cm of y, 2.5 cm tall, an S
        # of three stacked layers set by the plastic bending), and the far-edge drag carries it along: 7.4 cm of slack in
        # every u column, coverage 0.78. Pulls on the table do not open it: the near-edge pull of easy (10 cm) and the
        # measured slide move the whole sheet with the pleat, and a pin-and-pull (one hand holding the sheet just beyond
        # the pleat, the other pulling the near part 9 cm) opened it by 0.7 cm. Hung in the air it opens: both hands pinch
        # the near edge at u +-0.15 (the corners are out of top-down reach), lift it 12 cm, rise to 20 cm while moving 3 cm
        # toward the robot, so the pleat hangs off the table under the near part's weight, hold 2 s (the creases'
        # rest angles follow), then lay the near part down 9 cm toward the robot, 3 cm above the table: slack 0 in every
        # column but u +0.30 (1.9 cm), coverage 0.954, success at the retreat (1480 rows; the measured stages below are
        # not reached). 25 cm / 10 cm stretched the cloth to 1.20 at the release.
        dict(
            grasp_uv=dict(left=(-.15, -.22), right=(.15, -.22)),
            segments=(
                dict(name="lift", duration_s=1.5, by=dict(left=(0., 0., .12), right=(0., 0., .12))),
                dict(name="rise", duration_s=2., by=dict(left=(0., -.03, .20), right=(0., -.03, .20))),
                dict(name="hold", duration_s=2., by=dict(left=(0., -.03, .20), right=(0., -.03, .20))),
                dict(name="lay", duration_s=4., by=dict(left=(0., -.09, .03), right=(0., -.09, .03))),
            ),
        ),
        # (2026-10-10, MuJoCo) The Genesis cloth finished this route by creeping (coverage 0.882 -> 0.975 in the 15 s
        # after the release, nothing touching it). The MuJoCo cloth stays as the hands leave it: coverage 0.87 with
        # the far edge's strip tucked under along its length at y 0.45-0.52, beyond the arms' top-down reach. Measured:
        # both hands slide the sheet toward the robot from just inside its near edge until its far side is within
        # y 0.40 (11.6 cm on the -i01 birth); the tucked strip rolls out from under the sliding sheet (turned
        # triangles 0.5 %, coverage 0.964).
        dict(measured="slide", far_y_m=.40),
        # what the slide leaves tucked along the far edge, laid out with the drop-then-lay profile (none on -i01)
        dict(measured="crease", edge=(1, 1.), min_particles=25, edge_band_m=.12, lay="drop", beyond_m=0.),
    ),
)

# hard / seed 0: a band with about one full (+360 deg) twist; both ends lie face
# down, the middle face up. Wrist rotation is limited, so the twist is removed
# without turning a wrist: stage A turns each end over like a page about the
# long axis (left end +180 deg: its far-side edge is lifted over toward the
# robot; right end -180 deg: its near-side edge is lifted over away from it),
# with a low 9 cm arc to keep the pinch load down. The corners themselves are
# under other layers, so each pinch sits on the lifted edge 4 cm from its
# corner, where the patch is exposed, flat and complete. The untwisted
# sheet then lies diagonally with its right end still gathered; stage B pulls
# the far-left side outward and the near-right side toward the robot (a
# rotation couple, grasp-relative) to open both ends, and the sheet relaxes
# flat over ~40 s after release. As in medium, patches are released 3 cm
# above the table so that pinned particles are never pressed into it. The
# left page turn lands short of the mirrored position (y -0.04, not -0.12):
# carrying the flipped end further pulls it taut over the hinge and drove the
# edge ratio at that pinch to ~1.48.
# oraclefix 11 (lab, 2026-10-04): on lab's runs stage B leaves the far-left corner folded; its crest jitters under PBD
# self-contact (E24: spikes up to 0.22 m/s in 5.6 % of 2.5 ms ticks 20 s after the last action, all on that crest), so
# the settled window never completes and the coverage stays at 0.942. After stage B and a `fold_layout` settle, the
# robot measures the sheet and lays out the largest folded corner with one hand (robot_oracle.fold_stage); the relay
# before the measurement ends as soon as the sheet is quiet (cap fold_layout.settle_s), the final settle at success.
HARD = dict(
    max_strain_ratio=1.5,
    settle_s=60.,
    # (2026-10-10, MuJoCo) up to three layouts, creases included: on the coarse MuJoCo flex the page turns leave both
    # long edges' strips tucked under (a tri-fold, a third of the cloth turned over) on a sheet no rigid fit places;
    # crease_stage lays each strip out with both hands, measured locally.
    fold_layout=dict(settle_s=25., deviation_m=.03, corner_radius_m=.15, min_particles=25, max_stages=3, creases=True,
                     edge_band_m=.12),
    # oraclefix 11b: the laid-out sheet must stay clear of the arm bases' covers (robot_oracle.base_clearance_stage)
    base_clearance=dict(max_stages=2, settle_s=10., margin_m=.01, extra_m=.01, edge_band_m=(.025, .05), lift_m=.03,
                        speed_m_s=.02),
    stages=(
        dict(
            grasp_uv=dict(left=(-.29, -.20), right=(.28, .20)),
            segments=(
                dict(name="lift", duration_s=1.5, to=dict(left=(-.335, .248, .054), right=(.066, .073, .054))),
                dict(name="over", duration_s=3.5, to=dict(left=(-.35, .06, .09), right=(.075, .21, .09))),
                dict(name="down", duration_s=3.5, to=dict(left=(-.35, -.04, .03), right=(.08, .35, .03))),
            ),
        ),
        dict(
            grasp_uv=dict(left=(-.24, .10), right=(.228, -.108)),   # v1: right (.24, -.12)
            segments=(
                dict(name="lift", duration_s=1.5, by=dict(left=(0., 0., .08), right=(0., 0., .08))),
                dict(name="pull", duration_s=6., by=dict(left=(-.0155, .0264, .08), right=(.0357, -.0609, .08))),
                dict(name="lower", duration_s=3., by=dict(left=(-.0182, .0311, .03), right=(.0420, -.0716, .03))),
            ),
        ),
    ),
)

ROUTES = dict(easy=EASY, medium=MEDIUM, hard=HARD)

from .below_easy_routes import VERY_EASY, EASIER
ROUTES.update(very_easy=VERY_EASY, easier=EASIER)


def route_for(variation):
    """The route of `variation`: a declared ROUTES entry, else a redesigned tier's (redesign_routes: the redesigned
    blanket_flatten tiers by their lab names, the blanket_fold tiers as blanket_fold_<difficulty>). A redesign route
    carries a planner that holds one episode's state, so it is built fresh for every call."""
    if variation in ROUTES:
        return ROUTES[variation]
    if variation == "hard5":            # the Genesis hard tier ported to blanket_vbd_gpu (hard_port.py, 2026-10-17)
        from .hard_port import route
        return route()
    from .redesign_routes import redesign_routes
    routes = redesign_routes()
    if variation not in routes:
        raise KeyError(f"no blanket route {variation!r} (declared: {sorted(ROUTES)}; redesign: {sorted(routes)})")
    return routes[variation]
