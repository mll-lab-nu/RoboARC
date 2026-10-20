"""contact_weld_v1 on MuJoCo: the GraspManager's rigid backend protocol (topogym.world.physics.grasp_rigid) on the
scene's inactive welds, one per (hand, grasp target link), declared by scene.SceneBuilder.add_grasp_welds.

Acquisition activates the pair's weld with the current relative pose, anchored at the hand link's origin (as Genesis'
add_weld_constraint anchors at link_a; L12's source-hand anchor): eq_data = [hand origin in the target frame,
0 0 0, target-in-hand quaternion, torquescale]. Release deactivates it. The anchors are model data: the world's
checkpoints carry model.eq_data, and restore re-checks every binding against the active welds. The checkpoints also
carry the welds' slip windows (sample(), after every substep).

Two opt-in rules (topogym/utils/grasp.py, 2026-10-14) run only when the configuration declares them: a
ForceClosureGraspTarget is a candidate only on a force-closure pinch (_force_closure), and MultiRopeGraspAssistConfig
lets one hand weld several ropes as one binding (_rope_bundle, and the "parts" branches below). A third (2026-10-18):
a WeakHoldGraspTarget's hold that does not close force at its firm coefficient is lost once its link has moved
weak_hold_travel_m from where it was taken (_weak, check_health).

A task mechanism (MujocoWorld.set_mechanism) may refuse targets: when it has grasp_allowed(entity, link), a target it
refuses is no candidate, and a binding on it is lost (reason "target_not_graspable"; bed_bolt: the bolt while its
thread is engaged). Without a mechanism, or with one that has no grasp_allowed, every target is allowed, as before."""
from __future__ import annotations

import math
from collections import deque

import numpy as np

from topogym.utils.grasp import bilateral_contact_rows, closed_fraction, relative_error, relative_pose, rotation
from topogym.utils.grasp import ForceClosureGraspTarget, WeakHoldGraspTarget, force_closure_pair


# The grasp slips (clamp, never fail), per robot: a weld that has to pull or push its piece with more than this force
# (the weld constraint's translational force, averaged over the last WELD_SLIP_WINDOW_S) releases, as any lost grasp
# does (reason "grasp_slipped"; the hand grips again after opening and closing). A real two-finger grasp holds about the
# friction its grip allows, 2 * mu * grip; each value also stays at least twice the largest weld force its references
# need (2026-10-09), so no reference slips while an arm can no longer drive a welded piece into another with hundreds of N.
#   aloha          60 N: the model's finger drive presses with 35 N, mu 1 on both fingers (70 N); the claws references
#                  need 29 N at most.
#   panda_robotiq  65 N: the Robotiq 2F-85 grips with 20-235 N (datasheet); 2 * mu 0.65 (its pads, 0.7 / 0.6) * 50 N,
#                  in the lower part of that range. The Panda references need 31.1 N at most as a 20 ms mean: fiber's,
#                  with the cable as wire (very_easy 29.9 N, easier 31.1, easy 11.3, medium 10.4); hedgehog 16.7 N. 40 N until 2026-10-09, 1.3x fiber's need.
# A robot without an entry never slips.
# The window: the force after every substep, 20 ms of them, zeros before the grasp; the grasp slips once their mean
# exceeds the slip force (checked after every substep, released at the tick's end). A real grip rides out a brief impact
# and slips under a sustained excess: a force held above the slip force for 20 ms always slips, a short impact only when
# its impulse over the window exceeds slip force * 20 ms (1.2 N s on the ALOHA). The single-substep rule it replaces
# dropped a 1.07 kg assembly once: its weld went from 11.7 N to 25-57 N for ~30 ms, one 2.5 ms tick at 70.3 N; its mean over any 20 ms stays under
# ~40 N, and with the window the hold survives.
WELD_SLIP_FORCE_N = {"aloha": 60., "panda_robotiq": 65.}
WELD_SLIP_WINDOW_S = .020
# The grasp lets go once no finger has touched the welded body for this long (seconds), per robot (clamp, never fail):
# the weld releases as a lost grasp (reason "finger_contact_lost"; the hand grips again after opening and closing), so
# a weld never carries a body its fingers have left. The time counts MuJoCo steps whose contacts include no row between
# the hand's finger links (with their child bodies) and the welded link; any such row restarts it. Robotiq (2026-10-11):
# with its spring links at the MJCF's springref 2.62 (robot.SPRING_REFS) every Panda reference keeps a finger on its
# held body except fiber's 55-70 ms right at acquisition; 0.2 s is 2.9x that. With the springs about 0 (the import before)
# the fingertips curled and buzz's references carried the wand with no finger on it for 41-65 % of their weld time.
# A robot without an entry never lets go this way (the ALOHA: not measured).
WELD_CONTACT_LOST_S = {"panda_robotiq": .2}


class MujocoGraspBackend:
    def __init__(self, world, robot, config):
        import mujoco
        self.world, self.robot, self.config = world, robot, config
        m = world.model
        self.hands = {}
        for name, group in robot.groups.items():
            if group.grasp is None:
                raise ValueError(f"No grasp geometry declared for {name}")
            self.hands[name] = (robot.entity.get_link(group.hand_link),
                                tuple(robot.entity.get_link(n).idx for n in group.grasp.finger_links))
        self.targets = {}
        for spec in config.targets:
            entity = world.entities[spec.entity]
            if spec.entity == "robot":
                raise ValueError("Grasp target must be a task rigid link")
            link = entity.get_link(spec.link)
            if link.is_fixed or entity.n_dofs == 0:
                raise ValueError("Fixed or robot targets cannot use rigid grasp")
            self.targets[spec.key] = (spec, link)
        self.welds = {}
        for w in world.manifest["welds"]:
            eid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_EQUALITY, w["name"])
            if eid < 0:
                raise ValueError(f"scene lacks grasp weld {w['name']}")
            self.welds[w["hand"], tuple(w["target"])] = eid
        missing = [(h, k) for h in self.hands for k in self.targets if (h, k) not in self.welds]
        if missing:
            raise ValueError(f"scene lacks grasp welds for {missing}")
        self._evidence = {}
        self._distance = {}
        self._reach = None    # candidates' prefilter: the targets' link ids and centre offsets
        self.slip_force = WELD_SLIP_FORCE_N.get(robot.description.robot_id)
        self.slip_samples = max(1, round(WELD_SLIP_WINDOW_S / float(m.opt.timestep)))
        self._window = {}     # active weld id -> its slip window (the last slip_samples substeps' forces, N)
        self._slipped = set()  # active welds whose window mean exceeded the slip force
        self.contact_lost_s = WELD_CONTACT_LOST_S.get(robot.description.robot_id)
        self._untouched = {}  # active weld id -> seconds since a finger last touched its welded link (contact_lost_s)
        children = {}
        for b in range(1, m.nbody):
            children.setdefault(int(m.body_parentid[b]), []).append(b)

        def subtree(b):
            out, stack = set(), [b]
            while stack:
                x = stack.pop()
                out.add(x)
                stack.extend(children.get(x, ()))
            return out
        self._finger_bodies = {name: np.asarray(sorted(set().union(*(subtree(f) for f in fingers))))
                               for name, (hand_link, fingers) in self.hands.items()}
        # Opt-in rules (topogym/utils/grasp.py); a configuration without them leaves both empty and skips their code.
        # ForceClosureGraspTarget: target key -> friction coefficient; such a target is a candidate only on a
        # force-closure pinch (_force_closure).
        self._closure = {key: spec.friction_coefficient for key, (spec, _) in self.targets.items()
                         if isinstance(spec, ForceClosureGraspTarget)}
        # WeakHoldGraspTarget: target key -> (firm friction coefficient, weak-hold travel m); acquisition records whether
        # the pinch is firm and, for a weak one, the link's origin; check_health releases it beyond the travel.
        self._weak = {key: (spec.firm_friction_coefficient, spec.weak_hold_travel_m) for key, (spec, _) in self.targets.items()
                      if isinstance(spec, WeakHoldGraspTarget)}
        # MultiRopeGraspAssistConfig.weld_each_rope: a rope pinch welds one link of each touched rope to the hand, as one
        # binding whose "parts" are the plain bindings of those welds (_rope_bundle, _bundle_health).
        self._each_rope = frozenset(getattr(config, "weld_each_rope", ()))
        if self._each_rope:
            self.conflicts = self._bundle_conflicts     # GraspManager's ownership test, for bundle keys
            self._rope_links = {link.idx: key for key, (_, link) in self.targets.items() if key[0] in self._each_rope}
            limit = config.rope_max_rotation_error_rad
            self._rope_rotation = config.max_rotation_error_rad if limit is None else limit

    def _grasp_allowed(self):
        """The task mechanism's grasp_allowed(entity, link), or None (every target allowed)."""
        return getattr(getattr(self.world, "_mechanism", None), "grasp_allowed", None)

    def fraction(self, hand):
        group = self.robot.groups[hand]
        value = self.robot.capture_state().targets[hand][group.gripper_field]
        return closed_fraction(value, group.grasp.open_position, group.grasp.closed_position)

    def relative_pose(self, hand, target):
        parent = self.hands[hand][0]
        child = self.targets[tuple(target)][1]
        return relative_pose(parent.get_pos(), parent.get_quat(), child.get_pos(), child.get_quat())

    def finger_fractions(self, hand):
        group = self.robot.groups[hand]
        q = self.robot.entity.get_qpos()
        values = [q[self.robot.entity.get_joint(n).qs_idx_local[0]] for n in group.finger_joints]
        return [closed_fraction(v, group.grasp.open_position, group.grasp.closed_position) for v in values]

    def health_reason(self, binding):
        actual = np.asarray(self.finger_fractions(binding["hand"]))
        captured = np.asarray(binding["finger_fractions"])
        if np.max(captured - actual) > self.config.max_finger_opening_fraction:
            return "physical_fingers_opened"
        return None

    def check_health(self, binding):
        if "parts" in binding:      # opt-in rope bundle
            return self._bundle_health(binding)
        allowed = self._grasp_allowed()
        if allowed is not None and not allowed(*binding["target"]):
            return "target_not_graspable"       # the task's mechanism refuses the target now
        reason = self.health_reason(binding)
        translation, angle = relative_error(binding["relative_pose"], self.relative_pose(binding["hand"], binding["target"]))
        if translation > self.config.max_translation_error_m or angle > self.config.max_rotation_error_rad:
            reason = "relative_pose_error"
        eid = self.welds[binding["hand"], tuple(binding["target"])]
        if reason is None and eid in self._slipped:
            reason = "grasp_slipped"
        if reason is None and self.contact_lost_s is not None and self._untouched.get(eid, 0.) >= self.contact_lost_s - 1e-12:
            reason = "finger_contact_lost"
        if reason is None and binding.get("weak_hold_origin_m") is not None:     # opt-in WeakHoldGraspTarget
            travel = self._weak[tuple(binding["target"])][1]
            moved = np.linalg.norm(self.targets[tuple(binding["target"])][1].get_pos() - np.asarray(binding["weak_hold_origin_m"]))
            if moved > travel:
                reason = "weak_hold_travel"
        return reason

    def sample(self):
        """After every substep: each active grasp weld's translational constraint force enters its slip window; a window
        whose mean exceeds the robot's slip force marks the grasp slipped. MuJoCo puts the equality constraints' rows
        first (d.ne of them, by equality id), a weld's 3 translational rows before its 3 rotational ones."""
        if self._untouched:
            self._sample_touch()
        if not self._window:
            return
        d = self.world.data
        ids, force = d.efc_id[:d.ne], d.efc_force
        for eid, window in self._window.items():
            rows = np.flatnonzero(ids == eid)
            window.append(math.hypot(*force[rows[0]:rows[0] + 3]) if len(rows) else 0.)
            if sum(window) / self.slip_samples > self.slip_force:
                self._slipped.add(eid)

    def _sample_touch(self):
        """After every substep: each active weld's time without a finger contact row on its welded link grows by the
        step, or restarts at zero."""
        m, d = self.world.model, self.world.data
        n = int(d.ncon)
        geom = np.asarray(d.contact.geom[:n])
        bodies = np.where(geom >= 0, np.asarray(m.geom_bodyid)[np.maximum(geom, 0)], -1)
        dt = float(m.opt.timestep)
        for (hand, target), eid in self.welds.items():
            if eid not in self._untouched:
                continue
            link = self.targets[target][1].idx
            fingers = self._finger_bodies[hand]
            touch = ((bodies[:, 0] == link) & np.isin(bodies[:, 1], fingers)) | \
                    ((bodies[:, 1] == link) & np.isin(bodies[:, 0], fingers))
            self._untouched[eid] = 0. if touch.any() else self._untouched[eid] + dt

    def capture_slip(self):
        return dict(windows={eid: tuple(window) for eid, window in self._window.items()}, slipped=sorted(self._slipped),
                    untouched=dict(self._untouched))

    def restore_slip(self, state):
        self._window = {eid: deque(window, maxlen=self.slip_samples) for eid, window in state["windows"].items()}
        self._slipped = set(state["slipped"])
        self._untouched = dict(state.get("untouched", {}))

    def candidates(self, hand):
        contacts = self.world.contacts()
        parent, fingers = self.hands[hand]
        group = self.robot.groups[hand]
        tcp = parent.get_pos() + rotation(parent.get_quat()).apply(group.tcp_position_in_hand_m)
        result = []
        # A target whose link origin lies further from the TCP than the distance limit plus its centre's offset cannot
        # pass the exact test below (|centre - tcp| >= |origin - tcp| - |offset|), so only the others are evaluated
        # (1e-9 m covers rounding; a non-finite distance goes to the exact test)
        if self._reach is None:
            self._reach = (np.array([link.idx for _, link in self.targets.values()], dtype=np.int64),
                           np.array([np.linalg.norm(spec.center_m) for spec, _ in self.targets.values()]))
        idx, offset = self._reach
        far = np.linalg.norm(np.asarray(self.world.data.xpos)[idx] - tcp, axis=1) - offset > \
            self.config.max_tcp_distance_m + 1e-9
        allowed = self._grasp_allowed()
        for (key, (spec, link)), skip in zip(self.targets.items(), far):
            if skip or (allowed is not None and not allowed(*key)):
                continue
            center = link.get_pos() + rotation(link.get_quat()).apply(spec.center_m)
            if np.linalg.norm(center - tcp) > self.config.max_tcp_distance_m:
                continue
            rows = bilateral_contact_rows(contacts, fingers, link.idx, center_m=center, radius_m=spec.radius_m,
                                          max_penetration_m=self.config.max_penetration_m)
            if rows and key in self._closure and not self._force_closure(contacts, fingers, rows, self._closure[key]):
                continue        # opt-in ForceClosureGraspTarget: a pinch outside the friction cones is no candidate
            firm = None
            if rows and key in self._weak:      # opt-in WeakHoldGraspTarget: firm when it closes at the firm coefficient
                firm = bool(self._force_closure(contacts, fingers, rows, self._weak[key][0]))
            if rows:
                result.append(key)
                self._distance[hand, key] = float(np.linalg.norm(center - tcp))
                self._evidence[hand, key] = dict(
                    physics_step=self.world.physics_step, source="mujoco_last_substep", rows=[dict(
                        link_a=int(contacts["link_a"][i]), link_b=int(contacts["link_b"][i]),
                        position_m=contacts["position"][i].tolist(), penetration_m=float(contacts["penetration"][i]),
                        force_n=contacts["force"][i].tolist()) for i in rows])
                if firm is not None:
                    self._evidence[hand, key]["firm_hold"] = firm
        ropes = getattr(self.world, "_ropes", None) or {}
        if ropes:
            # A rope chain's fingers straddle several links: the grasp is the link nearest the TCP (as the Genesis soft
            # band binds the rope at the TCP); other entities keep every candidate (ambiguity rejects, as before).
            keep = [k for k in result if k[0] not in ropes]
            for rope in {k[0] for k in result if k[0] in ropes}:
                keep.append(min((k for k in result if k[0] == rope), key=lambda k: (self._distance[hand, k], k)))
            result = keep
        if self._each_rope:     # opt-in MultiRopeGraspAssistConfig: several ropes on one hand
            return self._rope_bundle(hand, result, contacts, fingers, tcp)
        return sorted(result)

    def _force_closure(self, contacts, fingers, rows, mu):
        """ForceClosureGraspTarget (opt-in): whether a contact row of one finger and a row of the other (among `rows`,
        the target's bilateral rows) close force (force_closure_pair). The normals are those of MuJoCo's contacts of the
        tick's last substep, data.contact, which the readout's rows index after the excluded contacts drop
        (MujocoWorld._read_contacts; nothing recomputes them before after_tick)."""
        c = self.world.data.contact
        n = int(self.world.data.ncon)
        keep = np.flatnonzero(np.asarray(c.exclude[:n]) == 0)
        if len(keep) != len(contacts["geom_a"]) or np.any(np.asarray(c.geom1[:n])[keep] != contacts["geom_a"]) or \
                np.any(np.asarray(c.geom2[:n])[keep] != contacts["geom_b"]):
            raise RuntimeError("force closure: the contact readout is not MuJoCo's current contact list")
        normal = np.asarray(c.frame[:n]).reshape(-1, 9)[keep, :3]      # contact normal, from geom1 toward geom2
        a, b, position = contacts["link_a"], contacts["link_b"], contacts["position"]
        sides = [[(position[i], normal[i] if a[i] == f else -normal[i]) for i in rows if f in (a[i], b[i])]
                 for f in fingers]
        return any(force_closure_pair(p, n_p, q, n_q, mu) for p, n_p in sides[0] for q, n_q in sides[1])

    def _rope_bundle(self, hand, result, contacts, fingers, tcp):
        """MultiRopeGraspAssistConfig (opt-in): the hand's candidates with its named ropes. A rope pinch (the config's
        docstring) gives one bundle key, the sorted tuple of each touched rope's touched link nearest the TCP (acquire
        welds each), in place of those ropes' plain candidates; without one, each named rope keeps its bilaterally
        touched link nearest the TCP. Other candidates stay (next to a bundle the GraspManager finds them ambiguous)."""
        a, b = np.asarray(contacts["link_a"]), np.asarray(contacts["link_b"])
        depth, position = np.asarray(contacts["penetration"]), np.asarray(contacts["position"])
        rope_ids = np.fromiter(self._rope_links, dtype=np.int64)
        touched = [{}, {}]          # per finger: rope -> {link key: contact rows}
        for side, finger in enumerate(fingers):
            for i in np.flatnonzero(((a == finger) & np.isin(b, rope_ids)) | ((b == finger) & np.isin(a, rope_ids))):
                key = self._rope_links[int(b[i] if a[i] == finger else a[i])]
                spec, link = self.targets[key]
                center = link.get_pos() + rotation(link.get_quat()).apply(spec.center_m)
                if (np.isfinite(depth[i]) and 0 <= depth[i] <= self.config.max_penetration_m
                        and np.isfinite(position[i]).all() and np.linalg.norm(position[i] - center) <= spec.radius_m
                        and np.linalg.norm(center - tcp) <= self.config.max_tcp_distance_m):
                    touched[side].setdefault(key[0], {}).setdefault(key, []).append(int(i))
        ropes = set(touched[0]) | set(touched[1])
        pinch = bool(touched[0]) and bool(touched[1]) and any(x != y for x in touched[0] for y in touched[1])
        if pinch:
            # every rope held by both fingers, or pressed against another touched rope inside a touched link's region
            regions = [(link.get_pos() + rotation(link.get_quat()).apply(spec.center_m), spec.radius_m)
                       for side in touched for keys in side.values() for spec, link in map(self.targets.get, keys)]
            owner = {link.idx: rope for rope in ropes for link in self.world.entities[rope].links}
            pressed = set()
            for i in np.flatnonzero(np.isin(a, list(owner)) & np.isin(b, list(owner)) & (depth >= 0)):
                x, y = owner[int(a[i])], owner[int(b[i])]
                if x != y and any(np.linalg.norm(position[i] - c) <= r for c, r in regions):
                    pressed |= {x, y}
            pinch = all(rope in touched[0] and rope in touched[1] or rope in pressed for rope in ropes)
        if not pinch:
            nearest = {}
            for k in sorted((k for k in result if k[0] in self._each_rope), key=lambda k: (self._distance[hand, k], k)):
                nearest.setdefault(k[0], k)
            return sorted([k for k in result if k[0] not in self._each_rope] + list(nearest.values()))
        bundle = []
        for rope in sorted(ropes):
            rows = {}
            for side in touched:
                for key, found in side.get(rope, {}).items():
                    rows.setdefault(key, []).extend(found)
            distance = {}
            for key in rows:
                spec, link = self.targets[key]
                distance[key] = float(np.linalg.norm(link.get_pos() + rotation(link.get_quat()).apply(spec.center_m) - tcp))
            key = min(rows, key=lambda k: (distance[k], k))
            self._distance[hand, key] = distance[key]
            self._evidence[hand, key] = dict(
                physics_step=self.world.physics_step, source="mujoco_last_substep", rope_pinch=True, rows=[dict(
                    link_a=int(a[i]), link_b=int(b[i]), position_m=position[i].tolist(), penetration_m=float(depth[i]),
                    force_n=np.asarray(contacts["force"][i]).tolist()) for i in sorted(set(rows[key]))])
            bundle.append(key)
        return sorted(k for k in result if k[0] not in self._each_rope) + [tuple(bundle)]

    def _bundle_health(self, binding):
        """MultiRopeGraspAssistConfig (opt-in): a rope bundle's grasp is lost on its first failing weld, each checked
        as check_health checks a plain weld except the rotation, against rope_max_rotation_error_rad."""
        allowed = self._grasp_allowed()
        for part in binding["parts"]:
            if allowed is not None and not allowed(*part["target"]):
                return "target_not_graspable"
            reason = self.health_reason(part)
            translation, angle = relative_error(part["relative_pose"], self.relative_pose(part["hand"], part["target"]))
            if translation > self.config.max_translation_error_m or angle > self._rope_rotation:
                reason = "relative_pose_error"
            eid = self.welds[part["hand"], tuple(part["target"])]
            if reason is None and eid in self._slipped:
                reason = "grasp_slipped"
            if reason is None and self.contact_lost_s is not None and \
                    self._untouched.get(eid, 0.) >= self.contact_lost_s - 1e-12:
                reason = "finger_contact_lost"
            if reason is not None:
                return reason
        return None

    def _bundle_conflicts(self, first, second):
        """MultiRopeGraspAssistConfig (opt-in): two hands' keys conflict when they share a link key, inside a bundle
        key too (for two plain keys: equality, the GraspManager's default)."""
        def parts(key):
            return set(key) if isinstance(key[0], tuple) else {tuple(key)}
        return bool(parts(first) & parts(second))

    def _active(self):
        d = self.world.data
        return {key: eid for key, eid in self.welds.items() if d.eq_active[eid]}

    def acquire(self, hand, target):
        if self._each_rope and isinstance(target[0], tuple):    # opt-in rope bundle: one weld per rope
            parts = [self.acquire(hand, key) for key in target]
            return dict(hand=hand, target=tuple(tuple(k) for k in target), parts=parts,
                        links=(self.hands[hand][0].idx, *(p["links"][1] for p in parts)))
        m, d = self.world.model, self.world.data
        key = (hand, tuple(target))
        eid = self.welds[key]
        if d.eq_active[eid]:
            raise RuntimeError("Grasp pair already constrained")
        reference = self.relative_pose(hand, target)
        # weld eq_data: [anchor in body2 (target), anchor in body1 (hand), body1->body2 quaternion, torquescale]
        q_rel = np.asarray(reference["quaternion_wxyz"], dtype=np.float64)
        p_rel = np.asarray(reference["position_m"], dtype=np.float64)
        m.eq_data[eid, 0:3] = -rotation(q_rel).inv().apply(p_rel)     # the hand origin, in the target's frame
        m.eq_data[eid, 3:6] = 0.
        m.eq_data[eid, 6:10] = q_rel
        d.eq_active[eid] = 1
        if self.slip_force is not None:       # a new grasp: zeros before it
            self._window[eid] = deque([0.] * self.slip_samples, maxlen=self.slip_samples)
        self._slipped.discard(eid)
        if self.contact_lost_s is not None:   # acquired on a bilateral contact: touched now
            self._untouched[eid] = 0.
        links = (self.hands[hand][0].idx, self.targets[tuple(target)][1].idx)
        binding = dict(hand=hand, target=tuple(target), links=links, relative_pose=reference,
                       finger_fractions=self.finger_fractions(hand), weld=int(eid),
                       acquisition_evidence=self._evidence[hand, tuple(target)])
        if tuple(target) in self._weak:         # opt-in WeakHoldGraspTarget: a weak hold remembers where it was taken
            firm = bool(binding["acquisition_evidence"].get("firm_hold"))
            binding.update(firm_hold=firm, weak_hold_origin_m=None if firm else
                           self.targets[tuple(target)][1].get_pos().tolist())
        return binding

    def exists(self, binding):
        if "parts" in binding:      # opt-in rope bundle
            return all(self.exists(p) for p in binding["parts"])
        return bool(self.world.data.eq_active[self.welds[binding["hand"], tuple(binding["target"])]])

    def release(self, binding):
        if "parts" in binding:      # opt-in rope bundle
            for part in binding["parts"]:
                self.release(part)
            return
        eid = self.welds[binding["hand"], tuple(binding["target"])]
        if not self.world.data.eq_active[eid]:
            raise RuntimeError("Grasp release of an inactive weld")
        self.world.data.eq_active[eid] = 0
        self._window.pop(eid, None)
        self._slipped.discard(eid)
        self._untouched.pop(eid, None)

    def verify_bindings(self, bindings):
        self.verify_current_bindings(bindings)

    def verify_current_bindings(self, bindings):
        if self._each_rope:         # opt-in rope bundles: their welds
            bindings = [p for b in bindings for p in b.get("parts", (b,))]
        expected = sorted((b["hand"], tuple(b["target"])) for b in bindings)
        if sorted(self._active()) != expected:
            raise ValueError("Restored grasp ownership differs from the active welds")
