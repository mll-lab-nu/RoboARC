"""blanket_fold: fold the blanket into a target shape.

The cloth, its birth and the robot are blanket_flatten's (this Task subclasses BlanketFlattenTask: same declaration,
binding and frozen start). The goal is not coverage but a target state made by inverse generation: rigid page turns
of the flat sheet, settled in the same cloth physics (topogym/tasks/blanket_fold/redesign/tiers.py; the target
particles are frozen per instance, `fold_target_path` + `fold_target_sha256`). Success: the sheet is settled, intact
and above the table (blanket_flatten's gates) and the fold predicate holds against the target (named points, stacking
order by panel, fitted outline, flatness, placement; topogym/tasks/blanket_fold/redesign/predicate.py) with the tier's
`fold_tolerances`.
"""
from pathlib import Path

import numpy as np

from topogym.tasks.blanket_flatten.common import BlanketFlattenTask

# ladder B's tolerances
FOLD_TOLERANCES = dict(named_mm=55., stack_all=.88, stack_covered=.80, iou=.85, dz_mm=35., over_mm=30., offset_mm=40.,
                       angle_deg=5.)


# a variation's ladder-B lab tier (topogym/tasks/blanket_fold/redesign/tiers.py), for its public goal words
LAB_TIER = dict(very_easy="b_very_easy", easier="b_easier", easy="b_easy", medium="b_medium", hard="b_hard")


def _mm(x):
    return f"{1e3 * float(x):.0f} mm"


class BlanketFoldTask(BlanketFlattenTask):
    task_id = "blanket_fold"
    defaults = {**BlanketFlattenTask.defaults, **dict(
        fold_target_path=None, fold_target_sha256=None, fold_creases=[], fold_tolerances=dict(FOLD_TOLERANCES),
        # the frozen archive's acceptance, as the flatten tier it is born from applies it (a gathered tier's archive,
        # hard's, is scene_layout-qualified)
        initial_state_acceptance="released_stable")}

    def identity_config(self):
        config = super().identity_config()
        config.pop("fold_target_path", None)        # the archive is pinned by its hash, as the initial state
        return config

    def fold_target(self):
        from topogym.utils.io import sha256_file
        path = self.config["fold_target_path"]
        if not path:
            raise ValueError("blanket_fold requires a frozen fold target (fold_target_path)")
        path = Path(path).expanduser().resolve()
        if sha256_file(path) != self.config["fold_target_sha256"]:
            raise ValueError("blanket_fold requires the matching fold-target SHA-256")
        with np.load(path, allow_pickle=False) as z:
            return np.asarray(z["particles"], dtype=float)

    def fold_goal_words(self):
        """The public goal (goal_description.describe_goal): the target fold in words, with its sizes, and the fold
        predicate's terms with this instance's tolerances. No world coordinates and no frame definitions (user
        2026-10-13): where the fold lies is shown by the target images (the finished fold rendered in this scene, robot
        at home, from the agent's own cameras), which the task ships beside the instruction. Directions are as seen in
        the overhead camera cam_high's image (near = its lower side). Returns (summary, success conditions)."""
        from topogym.tasks.blanket_fold.redesign import tiers as FT
        from topogym.tasks.blanket_flatten.redesign.tiers import GRIP_IN as grip
        cfg = self.config
        tier = FT.TIERS[LAB_TIER[self.difficulty]]
        L, W = float(cfg["length_m"]), float(cfg["width_m"])
        tol = cfg["fold_tolerances"]
        names = [f["name"] for f in tier["target"]]
        dog = "dog_ear_top" in names
        start = tier.get("start")
        if start is None:
            begin = f"The blanket ({_mm(L)} x {_mm(W)}) starts lying flat on the table."
        else:
            begin = f"The blanket ({_mm(L)} x {_mm(W)}) starts folded over on itself."
        half = (f"Fold it in half: turn the far half (the {_mm(W / 2)} of material along the far long edge) over onto the "
                f"near half about the middle line parallel to the long edges, so that the far edge lies on the near edge and "
                f"the far corners on the near corners. The folded stack is {_mm(L)} x {_mm(W / 2)}, two layers, the far half "
                f"upside down on top.")
        text = [begin, half]
        if dog:
            a = np.subtract(FT.DOG2_T, FT.C_TOP_NOW)
            text.append(
                f"Then dog-ear the top layer: turn the top layer's near-left corner (the blanket's original far-left corner) "
                f"over onto the top layer, the top layer alone, so that its corner point ({_mm(grip)} in from both edges) "
                f"lands {_mm(a[0])} to the right and {_mm(a[1])} farther than where it lay after the half fold; the crease "
                f"runs at about 45 degrees across the corner.")
        for ob in cfg.get("obstacles") or ():
            hx, hy, hz = (2 * float(h) for h in ob["half_size_m"])
            text.append(f"A fixed block ({_mm(hx)} long, {_mm(hy)} deep, {_mm(hz)} high) stands on the table beyond the "
                        f"blanket's far side; it does not move, and the blanket must end on the table, not on the block.")
        summary = ("Fold the blanket into the target shape on the table, then release it and withdraw. " + " ".join(text)
                   + " The target images show the required final state: this is how the folded cloth must look and where "
                   "it must lie; fold it like this. Near, far, left and right are as seen in the overhead camera cam_high "
                   "(near = the lower side of its image).")
        panels = "the middle line" + (" and the dog-ear crease" if dog else "")
        on_block = " (cloth resting on the block does not count)" if cfg.get("obstacles") else ""
        rules = [
            f"Corners: each of the blanket's four corner points (the material point {_mm(grip)} in from both edges at each "
            f"corner) must lie within {tol['named_mm']:g} mm, in the top view, of its place in the target fold.",
            f"Layer order: the blanket's panels are its regions between the target's creases ({panels}). For each "
            f"point of the blanket, the panel directly beneath it (material within 8 mm horizontally and 1-20 mm below) "
            f"must be the same as in the target fold for at least {100 * tol['stack_all']:g}% of all points, and for at "
            f"least {100 * tol['stack_covered']:g}% of the points that have material beneath them in the target.",
            f"Outline: after the best rigid top-view fit of the blanket onto the target, the {_mm(.02)} grid cells it "
            f"covers must overlap the target's with an intersection over union of at least {tol['iou']:g}.",
            f"Flat: the folded cloth must lie flat on the tabletop, its bottom layer fully in contact with the table "
            f"surface{on_block}: every point of the blanket within {tol['dz_mm']:g} mm of its height in the target fold, and "
            f"no point more than {tol['over_mm']:g} mm above the target's highest point.",
            f"Placement: placed where the target images show it, within {tol['offset_mm']:g} mm (the blanket's mean top-view "
            f"position against the target's) and {tol['angle_deg']:g} degrees (the fitted rotation).",
            f"Keep every cloth point no more than {_mm(cfg['particle_size_m'])} below the tabletop. No material edge may "
            f"stretch beyond {cfg['maximum_stretch_ratio']:g} times its original length.",
        ]
        return summary, rules

    def bind(self, entities):
        super().bind(entities)
        uv = self.material_uv
        self.fold_uv = uv - (uv.min(0) + uv.max(0)) / 2          # the predicate's centred material coordinates
        self.fold_target_particles = self.fold_target()
        if self.fold_target_particles.shape != (len(uv), 3):
            raise ValueError("blanket_fold target particles do not match the cloth")
        self.fold_creases = [tuple(map(tuple, c)) for c in self.config["fold_creases"]]

    def evaluate_state(self, state):
        from topogym.tasks.blanket_fold.redesign import predicate
        particle_state = state["particles"]["blanket"]
        positions = np.asarray(particle_state["position_m"])
        velocity = np.asarray(particle_state["velocity_m_s"])
        if not np.isfinite(positions).all() or not np.isfinite(velocity).all():
            return False, "Non-finite cloth particle state", {}
        heights = positions[:, 2] - self.table_height_m
        if heights.min() < -0.1:
            return False, "Cloth fell below the table", {}
        cfg = self.config
        stretch = np.linalg.norm(positions[self.particle_edges[:, 1]] - positions[self.particle_edges[:, 0]],
                                 axis=1) / self.rest_lengths
        speed = float(np.linalg.norm(velocity, axis=1).max())
        cheap = dict(above_table=float(heights.min()) >= -self.layout.particle_size_m,
                     material_intact=float(stretch.max()) <= cfg["maximum_stretch_ratio"],
                     settled=speed <= cfg["maximum_speed_m_s"])
        fold = None
        # the fold predicate (KD-tree stacking, layer maps) decides the goal only when the cheap gates hold, so it is
        # computed only then and at birth (physics step 0: the frozen instances' birth state hashes include it); a tick
        # whose cheap gates fail reports fold None (not computed), the control step's last tick included (user
        # 2026-10-12, as blanket_flatten's coverage; until then the last tick always computed it, complete_metrics)
        if all(cheap.values()) or int(state.get("physics_step", -1)) == 0:
            fold = predicate.evaluate(self.fold_uv, self.fold_target_particles, positions, self.fold_creases,
                                      th=cfg["fold_tolerances"])
        gates = dict(folded=fold is not None and fold["ok"], **cheap)
        return all(gates.values()), None, dict(
            gates=gates, fold=None if fold is None else {k: v for k, v in fold.items() if k != "thresholds"},
            maximum_height_m=float(heights.max()), maximum_speed_m_s=speed, maximum_stretch_ratio=float(stretch.max()))



def tier_defaults(lab_tier):
    """A ladder-B tier's Task parameters: the creases of its target (the predicate's panels) and the tolerances. The
    frozen start (initial_state_*) and target (fold_target_*) are set per instance."""
    from topogym.tasks.blanket_fold.redesign.tiers import TIERS
    creases = [[list(map(float, a)), list(map(float, b))] for a, b in TIERS[lab_tier]["creases"]]
    return {**BlanketFoldTask.defaults, "fold_creases": creases, "initialization_mode": "frozen"}
