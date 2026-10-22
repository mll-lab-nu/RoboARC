"""The caliper's locking screw (the user, 2026-10-17): turning its knob clockwise (seen from the scale side) 20 deg or
more off the loose stop locks the slider where it stands; turning it back to 10 deg or less frees it. The 10 deg between
the two is a hysteresis, so a knob resting at one threshold does not switch back and forth.

The lock is the joint equality caliper::slider_lock (slider = polycoef[0]; declared with the caliper's URDF,
topogym.assets.geometry.caliper_sorting.caliper_lock_equality), nearly hard. On locking the mechanism writes the slider's
position into polycoef[0] (model.eq_data) and switches the equality on (data.eq_active); on freeing it switches it off.
It writes nothing else: no qpos, no other model field (docs/rcb/MUJOCO_CORE.md, "Task mechanisms"). The caliper is born
locked (the equality authored active at the closed jaws, the knob on its tight stop)."""
import math

from topogym.assets.geometry import caliper_sorting as geo

SEP = "::"


class CaliperLock:
    def __init__(self, world):
        import mujoco
        m, d = world.model, world.data
        self.world, self.m = world, m
        self.eq = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_EQUALITY, f"caliper{SEP}slider_lock")
        js = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"caliper{SEP}screw")
        jl = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"caliper{SEP}slider")
        if min(self.eq, js, jl) < 0:
            raise ValueError("caliper_sorting: the scene has no slider lock, screw joint or slider joint")
        self.q_screw, self.q_slider = int(m.jnt_qposadr[js]), int(m.jnt_qposadr[jl])
        self.lock_rad, self.release_rad = math.radians(geo.LOCK_DEG), math.radians(geo.RELEASE_DEG)
        self.switches = 0                       # locks and frees so far (a count for read())

    def locked(self, d=None) -> bool:
        d = self.world.data if d is None else d
        return bool(d.eq_active[self.eq])

    def tick(self, m, d):
        angle = float(d.qpos[self.q_screw])
        if not d.eq_active[self.eq] and angle >= self.lock_rad:
            m.eq_data[self.eq, 0] = d.qpos[self.q_slider]
            d.eq_active[self.eq] = 1
            self.switches += 1
        elif d.eq_active[self.eq] and angle <= self.release_rad:
            d.eq_active[self.eq] = 0
            self.switches += 1

    def capture(self) -> dict:
        """Its own count, the equality's switch and the lock position it wrote (the world's checkpoint also holds them:
        data.eq_active in MuJoCo's integration state, model.eq_data whole)."""
        return dict(switches=self.switches, locked=self.locked(), lock_position_m=float(self.m.eq_data[self.eq, 0]))

    def restore(self, snap: dict) -> None:
        self.switches = int(snap["switches"])
        self.m.eq_data[self.eq, 0] = float(snap["lock_position_m"])
        self.world.data.eq_active[self.eq] = int(bool(snap["locked"]))

    def read(self) -> dict:
        """The screw's angle off the loose stop, the lock's switch and position (state_sha hashes them) and the count."""
        d = self.world.data
        return dict(screw_angle_deg=math.degrees(float(d.qpos[self.q_screw])), locked=self.locked(d),
                    lock_position_m=float(self.m.eq_data[self.eq, 0]), switches=self.switches)
