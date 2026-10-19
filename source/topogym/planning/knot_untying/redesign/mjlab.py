"""Object-level MuJoCo lab for the redesigned knot_untying scenes: rope chains, anchors, clutter, virtual hands.

The rope model is the family's MuJoCo port (topogym/world/mujoco/rope_chain.py, profile knot_chain_v2), rebuilt here
from a centre line instead of a frozen IPC birth (new scenes have none): free capsule links of ~4.5 mm joined by nearly
hard connect equalities, the discrete elastic rod (E 5e4 Pa, nu .3, straight rest) applied as exact couples before every
step, the inertia floor and rod damping rules, bend stops at 0.9 rad (neighbour spheres), self-contact excluded within 3
links, contacts acting from 0.2 mm (geom margin), the rope's friction 0.3 on the table / itself / fixtures, the profile's
solver (implicitfast, Newton 100, elliptic cone, impratio 10, converged CCD) and 0.625 ms steps. Static bodies (table,
clutter) carry the profile's soft static class with solmix 0, so rope contacts keep the rope's own parameters.

Anchors: a rope end's first `links` links welded to the world at their pose when the anchor is applied (a clamp; the
scene's staple fixture covers it). Virtual hands: kinematic mocap bodies; a grasp welds the hand to one rope link at the
current relative pose (anchored at the hand origin, as the robot's contact_weld_v1 does) and is lost when the 20 ms mean
of the weld's translational force exceeds the hand's limit (default 8 N: the rope agent's measured hold of a real rope
grasp on MuJoCo; 1 N emulates the pure friction pinch). A grasp is only placed where a parallel gripper fits: the
pinched link must have no other rope material and no fixture within `pinch_clearance_m` (default 10 mm: the family's
grasp_site rule, half opening >= 6 mm with 4 mm pad margin).
"""
from __future__ import annotations

import json
import math
import xml.etree.ElementTree as ET
from collections import deque
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

TABLE_TOP = 0.9196429
TABLE_POS = (-0.1356786, 0., 0.45982145)
TABLE_HALF = (0.75, 0.7, 0.45982145)
ORIGIN_XY = (-0.1356786, 0.)          # redesigned scenes: task-local (0, 0) at the midpoint between the ALOHA bases

PROFILE = dict(
    dt=.000625, link_m=.0045, E=5e4, nu=.3, rho=650., friction=.3, table_friction=.3,
    inertia_rule=4., damping_rule=.3, exclude_links=3, bend_stop_rad=.9, margin=.0002,
    contact_solref=(.00125, 2.), contact_solimp=(.9, .99, .001, .5, 2.),
    static_solref=(.005, 2.), static_solimp=(.9, .95, .001, .5, 2.),
    joint_solref=(.00125, 1.), joint_solimp=(.9999, .9999, .001, .5, 2.),
    weld_solref=(.00125, 1.), weld_solimp=(.9, .95, .001, .5, 2.),
    anchor_solref=(.00125, 1.), anchor_solimp=(.9999, .9999, .001, .5, 2.),
    torsional=.005, rolling=.0001,
    hand_mass=.3, hand_drive_solref=(.03, 1.), hand_drive_solimp=(.9, .95, .001, .5, 2.))


def fmt(v):
    return " ".join(format(float(x), ".12g") for x in np.atleast_1d(v))


def parallel_frames(nodes):
    """Per link: x along the link, y / z parallel-transported (no twist)."""
    d = np.diff(nodes, axis=0)
    x = d / np.linalg.norm(d, axis=1, keepdims=True)
    R = np.zeros((len(x), 3, 3))
    up = np.array([0., 0., 1.])
    y = np.cross(up, x[0])
    if np.linalg.norm(y) < 1e-9:
        y = np.cross([1., 0., 0.], x[0])
    y /= np.linalg.norm(y)
    for k in range(len(x)):
        if k:
            a, b = x[k - 1], x[k]
            v = np.cross(a, b)
            c = float(np.dot(a, b))
            if np.linalg.norm(v) > 1e-15:
                vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
                y = (np.eye(3) + vx + vx @ vx * (1 / (1 + c))) @ y
            y -= np.dot(y, b) * b
            y /= np.linalg.norm(y)
        R[k] = np.column_stack([x[k], y, np.cross(x[k], y)])
    return R


def chain_nodes(centerline, link_m):
    p = np.asarray(centerline, float)
    s = np.r_[0., np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))]
    n = max(2, int(round(s[-1] / link_m)))
    sn = np.linspace(0, s[-1], n + 1)
    return sn, np.column_stack([np.interp(sn, s, p[:, k]) for k in range(3)])


def to_world(p_local):
    p = np.array(p_local, float)
    p[..., 0] += ORIGIN_XY[0]
    p[..., 1] += ORIGIN_XY[1]
    p[..., 2] += TABLE_TOP
    return p


def to_local(p_world):
    p = np.array(p_world, float)
    p[..., 0] -= ORIGIN_XY[0]
    p[..., 1] -= ORIGIN_XY[1]
    p[..., 2] -= TABLE_TOP
    return p


def fixture_geom(parent, item, name, attrs):
    """Static fixture (task-local pose: pos = centre, z above the table top) as an MJCF geom."""
    kind = item["kind"]
    pos = to_world(np.asarray(item.get("pos", (0, 0, 0)), float))
    q = Rotation.from_euler('z', float(item.get("yaw_deg", 0.)), degrees=True).as_quat()[[3, 0, 1, 2]]
    g = dict(name=name, rgba=item.get("rgba", ".42 .45 .52 1"), **attrs)
    if kind == "box":
        g.update(type="box", pos=fmt(pos), quat=fmt(q), size=fmt(np.asarray(item["size"], float) / 2))
    elif kind == "cylinder":
        g.update(type="cylinder", pos=fmt(pos), quat=fmt(q), size=fmt([item["radius"], item["height"] / 2]))
    elif kind == "capsule":
        a, b = to_world(np.asarray(item["fromto"][:3], float)), to_world(np.asarray(item["fromto"][3:], float))
        g.update(type="capsule", fromto=fmt(np.r_[a, b]), size=fmt([item["radius"]]))
    else:
        raise ValueError(kind)
    return ET.SubElement(parent, "geom", **g)


def build_xml(scene, profile=PROFILE, hands=("left", "right"), qpos=None, finger=False):
    """MJCF of a lab scene: dict(ropes=[dict(name, centerline (local, m), radius)], fixtures=[...]). Returns (xml, meta)."""
    P = profile
    root = ET.Element("mujoco", model="knot_lab")
    ET.SubElement(root, "compiler", angle="radian", autolimits="true", inertiafromgeom="false")
    ET.SubElement(root, "option", timestep=fmt(P["dt"]), gravity="0 0 -9.81", integrator="implicitfast", solver="Newton",
                  iterations="100", ls_iterations="50", tolerance="1e-8", cone="elliptic", impratio="10",
                  ccd_iterations="500", ccd_tolerance="1e-10")
    ET.SubElement(root, "size", memory="1G")
    vis = ET.SubElement(root, "visual")
    ET.SubElement(vis, "global", offwidth="1280", offheight="960")
    asset = ET.SubElement(root, "asset")
    ET.SubElement(asset, "texture", name="grid", type="2d", builtin="checker", rgb1=".82 .83 .85", rgb2=".76 .77 .80",
                  width="512", height="512")
    ET.SubElement(asset, "material", name="table", texture="grid", texrepeat="12 12", reflectance="0")
    world = ET.SubElement(root, "worldbody")
    ET.SubElement(world, "light", pos="0 -0.3 2.4", dir="0 0.1 -1", diffuse=".8 .8 .8", specular=".1 .1 .1")
    ET.SubElement(world, "light", pos="-0.6 0.6 2.0", dir="0.3 -0.3 -1", diffuse=".35 .35 .35", directional="true")
    static = dict(solref=fmt(P["static_solref"]), solimp=fmt(P["static_solimp"]), solmix="0",
                  friction=fmt([P["table_friction"], P["torsional"], P["rolling"]]), condim="3")
    tb = ET.SubElement(world, "body", name="table", pos=fmt(TABLE_POS))
    ET.SubElement(tb, "geom", name="table", type="box", size=fmt(TABLE_HALF), material="table", **static)
    fb = ET.SubElement(world, "body", name="fixtures")
    for k, item in enumerate(scene.get("fixtures", [])):
        attrs = dict(static) if item.get("collide", True) else dict(contype="0", conaffinity="0")
        fixture_geom(fb, item, item.get("name", f"fixture{k:02d}"), attrs)
    contact = ET.SubElement(root, "contact")
    equality = ET.SubElement(root, "equality")
    task = dict(solref=fmt(P["contact_solref"]), solimp=fmt(P["contact_solimp"]), condim="3",
                friction=fmt([P["friction"], P["torsional"], P["rolling"]]), margin=fmt([P["margin"]]))
    meta = dict(ropes={}, hands={})
    dt = P["dt"]
    for spec in scene["ropes"]:
        name, r = spec["name"], float(spec.get("radius", .004))
        s_nodes, nodes = chain_nodes(to_world(spec["centerline"]), P["link_m"])
        R = parallel_frames(nodes)
        n = len(nodes) - 1
        lengths = np.linalg.norm(np.diff(nodes, axis=0), axis=1)
        mat_len = np.diff(s_nodes)
        E, nu, rho = P["E"], P["nu"], P["rho"]
        G = E / (2 * (1 + nu))
        Ia = np.pi * r ** 4 / 4
        mass = rho * np.pi * r ** 2 * mat_len
        k_tw, k_be = G * 2 * Ia / lengths, E * Ia / lengths
        I_ax = np.maximum(mass * r ** 2 / 2, P["inertia_rule"] * k_tw * dt ** 2)
        I_pe = np.maximum(mass * (3 * r ** 2 + lengths ** 2) / 12, P["inertia_rule"] * k_be * dt ** 2)
        node_l = .5 * (mat_len[:-1] + mat_len[1:])
        K = np.stack([G * 2 * Ia / node_l, E * Ia / node_l, E * Ia / node_l], axis=1)
        rule = P["damping_rule"]
        C = np.stack([rule * np.minimum(I_ax[:-1], I_ax[1:]) / dt, rule * np.minimum(I_pe[:-1], I_pe[1:]) / dt,
                      rule * np.minimum(I_pe[:-1], I_pe[1:]) / dt], axis=1)
        r_stop = .5 * float(np.mean(lengths)) * math.cos(.5 * P["bend_stop_rad"])
        rgba = spec.get("rgba", ".86 .53 .19 1")
        for k in range(n):
            bn = f"{name}::{k:03d}"
            q = Rotation.from_matrix(R[k]).as_quat()[[3, 0, 1, 2]]
            body = ET.SubElement(world, "body", name=bn, pos=fmt(nodes[k]), quat=fmt(q))
            ET.SubElement(body, "freejoint", name=f"{bn}::root")
            ET.SubElement(body, "inertial", pos=fmt([lengths[k] / 2, 0, 0]), mass=fmt([mass[k]]),
                          diaginertia=fmt([I_ax[k], I_pe[k], I_pe[k]]))
            x0 = r if k == 0 else 0.
            x1 = lengths[k] - r if k == n - 1 else lengths[k]
            col = rgba
            if k < 6:
                col = spec.get("rgba_start", ".17 .45 .76 1")
            elif k >= n - 6:
                col = spec.get("rgba_end", ".8 .24 .25 1")
            if x1 - x0 < 1e-6:
                ET.SubElement(body, "geom", name=f"{bn}::g", type="sphere", pos=fmt([.5 * (x0 + x1), 0, 0]),
                              size=fmt([r]), rgba=col, **task)
            else:
                ET.SubElement(body, "geom", name=f"{bn}::g", type="capsule", fromto=fmt([x0, 0, 0, x1, 0, 0]),
                              size=fmt([r]), rgba=col, **task)
            ET.SubElement(body, "geom", name=f"{bn}::stop", type="sphere", size=fmt([r_stop]),
                          pos=fmt([lengths[k] / 2, 0, 0]), contype="0", conaffinity="0", group="3", rgba="0 0 0 0")
        for k in range(n - 1):
            ET.SubElement(equality, "connect", name=f"{name}::node{k + 1:03d}", body1=f"{name}::{k:03d}",
                          body2=f"{name}::{k + 1:03d}", anchor=fmt([lengths[k], 0, 0]),
                          solref=fmt(P["joint_solref"]), solimp=fmt(P["joint_solimp"]))
        for i in range(n):
            for j in range(i + 1, min(n, i + 1 + P["exclude_links"])):
                ET.SubElement(contact, "exclude", body1=f"{name}::{i:03d}", body2=f"{name}::{j:03d}")
        for k in range(n - 1):
            ET.SubElement(contact, "pair", geom1=f"{name}::{k:03d}::stop", geom2=f"{name}::{k + 1:03d}::stop",
                          condim="1", solref=task["solref"], solimp=task["solimp"])
        # anchors: inactive welds to the world on the end links (activated by Lab.anchor)
        for end in ("start", "end"):
            for j in range(int(spec.get("anchor_links", 4))):
                k = j if end == "start" else n - 1 - j
                ET.SubElement(equality, "weld", name=f"{name}::anchor_{end}{j}", body1=f"{name}::{k:03d}",
                              active="false", solref=fmt(P["anchor_solref"]), solimp=fmt(P["anchor_solimp"]))
        meta["ropes"][name] = dict(n=n, s_nodes=s_nodes, lengths=lengths, K=K, C=C, radius=r, r_stop=r_stop)
    for k, h in enumerate(hands):
        # the arm: a kinematic target (mocap) and a gravity-compensated hand body that follows it through a soft weld
        # (P["hand_drive_solref"]: ~330 N/m with the 0.3 kg hand, critically damped), so a jammed pull builds force
        # gradually as on the PD-driven ALOHA arm instead of through an infinitely stiff kinematic hand
        home = f"{-.3 + .6 * k} -0.35 1.15"
        tb_ = ET.SubElement(world, "body", name=f"target_{h}", mocap="true", pos=home)
        ET.SubElement(tb_, "geom", type="sphere", size=".004", contype="0", conaffinity="0", rgba=".9 .9 .2 .4", group="2")
        hb = ET.SubElement(world, "body", name=f"hand_{h}", pos=home, gravcomp="1")
        ET.SubElement(hb, "freejoint", name=f"hand_{h}::root")
        ET.SubElement(hb, "inertial", pos="0 0 0", mass=fmt([P["hand_mass"]]), diaginertia=fmt([1e-3, 1e-3, 1e-3]))
        ET.SubElement(hb, "geom", type="sphere", size=".006", contype="0", conaffinity="0", rgba=".2 .7 .3 .5", group="1")
        ET.SubElement(equality, "weld", name=f"drive_{h}", body1=f"target_{h}", body2=f"hand_{h}",
                      solref=fmt(P["hand_drive_solref"]), solimp=fmt(P["hand_drive_solimp"]))
        first = f"{scene['ropes'][0]['name']}::000"
        ET.SubElement(equality, "weld", name=f"grasp_{h}", body1=f"hand_{h}", body2=first, active="false",
                      solref=fmt(P["weld_solref"]), solimp=fmt(P["weld_solimp"]))
    for k in range(int(finger)):
        # a collidable kinematic finger (a vertical capsule, r 6 mm like an ALOHA finger tip): a hook with no grasp
        # limit, driven by its mocap pose (the arm's torque is not modelled: it pushes as hard as the contact needs);
        # finger=2 adds a second one ("finger2") for two-hook tests (coordinator 10-10, R3 breakers)
        nm = "finger" if k == 0 else f"finger{k + 1}"
        fb_ = ET.SubElement(world, "body", name=nm, mocap="true", pos=f"{.1 * k} 0.5 1.3")
        ET.SubElement(fb_, "geom", name=nm, type="capsule", fromto="0 0 0 0 0 .04", size=".006",
                      rgba=".9 .3 .3 1", **static)
    return ET.tostring(root, encoding="unicode"), meta


def _log_so3(D):
    tr = np.clip((np.trace(D, axis1=1, axis2=2) - 1) / 2, -1., 1.)
    th = np.arccos(tr)
    v = np.stack([D[:, 2, 1] - D[:, 1, 2], D[:, 0, 2] - D[:, 2, 0], D[:, 1, 0] - D[:, 0, 1]], axis=1)
    small = th < 1e-6
    f = np.where(small, .5 + th * th / 12, th / (2 * np.where(small, 1., np.sin(th))))
    return v * f[:, None]


def _jr_inv_T(theta, g):
    th = np.linalg.norm(theta, axis=1)
    small = th < 1e-4
    ths = np.where(small, 1., th)
    c = np.where(small, 1 / 12 + th * th / 720, 1 / ths ** 2 - (1 + np.cos(ths)) / (2 * ths * np.sin(ths)))
    tg = np.cross(theta, g)
    return g - .5 * tg + c[:, None] * np.cross(theta, tg)


class Rod:
    """The family's discrete elastic rod on one chain (rope_chain.RopeChain.apply, straight rest)."""

    def __init__(self, m, name, info):
        import mujoco
        self.m, self.name = m, name
        self.n = info["n"]
        self.K, self.C, self.lengths = info["K"], info["C"], info["lengths"]
        self.s_nodes = info["s_nodes"]
        self.radius = info["radius"]
        self.bodies = np.array([mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"{name}::{k:03d}") for k in range(self.n)])
        jnt = np.array([m.body_jntadr[b] for b in self.bodies])
        self.dof = np.array([m.jnt_dofadr[j] for j in jnt])
        self.qadr = np.array([m.jnt_qposadr[j] for j in jnt])
        self.rot_dofs = self.dof[:, None] + np.arange(3, 6)[None]
        self.energy = 0.

    def _links(self, R, tau):
        tw = np.einsum("nij,nj->ni", R[1:], tau)
        tot = np.zeros((self.n, 3))
        tot[1:] += tw
        tot[:-1] -= tw
        return np.einsum("nji,nj->ni", R, tot)

    def apply(self, d):
        R = np.asarray(d.xmat)[self.bodies].reshape(-1, 3, 3)
        Q = np.einsum("nji,njk->nik", R[:-1], R[1:])
        theta = _log_so3(Q)
        g = self.K * theta
        tau = -_jr_inv_T(theta, g)
        w = np.asarray(d.qvel)[self.rot_dofs]
        ww = np.einsum("nij,nj->ni", R, w)
        rel = np.einsum("nji,nj->ni", R[1:], ww[1:] - ww[:-1])
        np.asarray(d.qfrc_applied)[self.rot_dofs] = self._links(R, tau - self.C * rel)
        self.energy = float(.5 * np.sum(theta * g))

    def nodes(self, d):
        P = np.asarray(d.xpos)[self.bodies]
        end = P[-1] + np.asarray(d.xmat)[self.bodies[-1]].reshape(3, 3)[:, 0] * self.lengths[-1]
        return np.vstack([P, end])

    def centres(self, d):
        """Link centres (n, 3), the material points s_nodes[k] + l_k / 2."""
        P = np.asarray(d.xpos)[self.bodies]
        X = np.asarray(d.xmat)[self.bodies].reshape(-1, 3, 3)[:, :, 0]
        return P + X * (self.lengths[:, None] / 2)

    def speeds(self, d):
        import mujoco
        out = np.zeros(self.n)
        v = np.zeros(6)
        for k, b in enumerate(self.bodies):
            mujoco.mj_objectVelocity(self.m, d, mujoco.mjtObj.mjOBJ_BODY, b, v, 0)
            out[k] = np.linalg.norm(v[3:])
        return out


class Lab:
    def __init__(self, scene, *, profile=PROFILE, hands=("left", "right"), hand_limit_n=8., pinch_clearance_m=.010,
                 log_every=40, finger=False):
        import mujoco
        import os
        # coordinator 10-10: the coming shared physics, opt-in by environment (scripts that pass their own profile or
        # set the flag themselves are unaffected): RD_KNOT_INERTIA=8 -> knot_chain_v3's rotational inertia floor
        # (rope agent 78e0c03a9); RD_DIAGEXACT=1 -> MuJoCo's exact constraint diagonal (shared head b1ec43997)
        if profile is PROFILE and (os.environ.get("RD_KNOT_INERTIA") or os.environ.get("RD_KNOT_DAMPING")):
            # knot_chain_v4 (rope agent 47941d360): RD_KNOT_INERTIA=12 RD_KNOT_DAMPING=0.15
            profile = dict(PROFILE)
            if os.environ.get("RD_KNOT_INERTIA"):
                profile["inertia_rule"] = float(os.environ["RD_KNOT_INERTIA"])
            if os.environ.get("RD_KNOT_DAMPING"):
                profile["damping_rule"] = float(os.environ["RD_KNOT_DAMPING"])
        self.scene, self.P = scene, profile
        self.xml, self.meta = build_xml(scene, profile, hands, finger=finger)
        self.m = mujoco.MjModel.from_xml_string(self.xml)
        self.diag_exact = os.environ.get("RD_DIAGEXACT", "") == "1"
        if self.diag_exact:
            self.m.opt.enableflags |= int(mujoco.mjtEnableBit.mjENBL_DIAGEXACT)
        self.d = mujoco.MjData(self.m)
        mujoco.mj_forward(self.m, self.d)
        self.rods = {name: Rod(self.m, name, info) for name, info in self.meta["ropes"].items()}
        self.hands = {}
        for h in hands:
            bid = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, f"hand_{h}")
            tid = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, f"target_{h}")
            jid = int(self.m.body_jntadr[bid])
            self.hands[h] = dict(body=bid, target=tid, mocap=int(self.m.body_mocapid[tid]),
                                 qadr=int(self.m.jnt_qposadr[jid]), dadr=int(self.m.jnt_dofadr[jid]),
                                 eq=mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_EQUALITY, f"grasp_{h}"),
                                 grasp=None, window=deque(maxlen=max(1, round(.02 / profile["dt"]))), limit=hand_limit_n)
        self.pinch_clearance = pinch_clearance_m
        self.t = 0.
        self.events = []
        self.trace = []
        self.log_every = log_every
        self.steps = 0
        self.max_force = {h: 0. for h in hands}
        self.hand_path = {h: [] for h in hands}
        self.hand_t = []

    # -------------------------------------------------------------- state
    def save_state(self, path, **extra):
        import mujoco
        np.savez_compressed(path, qpos=self.d.qpos.copy(), qvel=self.d.qvel.copy(), eq_active=self.d.eq_active.copy(),
                            mocap_pos=self.d.mocap_pos.copy(), mocap_quat=self.d.mocap_quat.copy(), t=self.t,
                            eq_data=self.m.eq_data.copy(), eq_obj2id=self.m.eq_obj2id.copy(),
                            extra=json.dumps(extra))

    def load_state(self, path, velocities=True):
        import mujoco
        z = np.load(path, allow_pickle=False)
        self.d.qpos[:] = z["qpos"]
        self.d.qvel[:] = z["qvel"] if velocities else 0.
        self.d.eq_active[:] = z["eq_active"]
        self.m.eq_data[:] = z["eq_data"]
        self.m.eq_obj2id[:] = z["eq_obj2id"]
        k = min(len(self.d.mocap_pos), len(z["mocap_pos"]))      # extra mocaps (a finger) come last
        self.d.mocap_pos[:k] = z["mocap_pos"][:k]
        self.d.mocap_quat[:k] = z["mocap_quat"][:k]
        for h in self.hands.values():
            h["grasp"] = None
            if self.d.eq_active[h["eq"]]:
                self.d.eq_active[h["eq"]] = 0
        self._anchor_ids = [e for e in range(self.m.neq)
                            if "::anchor_" in (mujoco.mj_id2name(self.m, mujoco.mjtObj.mjOBJ_EQUALITY, e) or "")
                            and self.d.eq_active[e]]
        mujoco.mj_forward(self.m, self.d)

    def clamp_slip(self, rope, end, slip_n):
        """Make the active clamp of (rope, end) yield above slip_n (20 ms mean of its weld forces)."""
        import mujoco
        ids = [e for e in self._anchor_ids
               if (mujoco.mj_id2name(self.m, mujoco.mjtObj.mjOBJ_EQUALITY, e) or "").startswith(f"{rope}::anchor_{end}")]
        self._anchor_slip = getattr(self, "_anchor_slip", {})
        self._anchor_slip[(rope, end)] = dict(limit=float(slip_n), ids=ids,
                                              window=deque(maxlen=max(1, round(.02 / self.P["dt"]))))

    # -------------------------------------------------------------- readout
    def centerline(self, rope):
        return self.rods[rope].nodes(self.d)

    def centerlines_local(self):
        return {k: to_local(r.nodes(self.d)) for k, r in self.rods.items()}

    def link_at(self, rope, s):
        rod = self.rods[rope]
        centres = .5 * (rod.s_nodes[:-1] + rod.s_nodes[1:])
        return int(np.argmin(np.abs(centres - s)))

    def point(self, rope, s):
        rod = self.rods[rope]
        nodes = rod.nodes(self.d)
        return np.array([np.interp(s, rod.s_nodes, nodes[:, k]) for k in range(3)])

    def tangent(self, rope, s, h=.006):
        L = self.rods[rope].s_nodes[-1]
        t = self.point(rope, min(L, s + h)) - self.point(rope, max(0., s - h))
        return t / np.linalg.norm(t)

    def length(self, rope):
        return float(self.rods[rope].s_nodes[-1])

    def pinch_clearance_at(self, rope, s, exclude_m=.03):
        """3D distance from the rope centre at s to other rope material (> exclude_m away along the same rope, or any
        other rope) and to fixtures (surface), minus the rope radius: the room a finger has next to the pinched strand."""
        import mujoco
        p = self.point(rope, s)
        best = np.inf
        for name, rod in self.rods.items():
            c = rod.centres(self.d)
            sc = .5 * (rod.s_nodes[:-1] + rod.s_nodes[1:])
            keep = np.ones(len(c), bool) if name != rope else np.abs(sc - s) > exclude_m
            if keep.any():
                best = min(best, float(np.min(np.linalg.norm(c[keep] - p, axis=1))) - rod.radius)
        fid = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, "fixtures")
        for g in range(self.m.ngeom):
            if self.m.geom_bodyid[g] != fid or self.m.geom_contype[g] == 0:
                continue
            best = min(best, _point_geom_distance(self.m, self.d, g, p))
        return best

    def crossings(self):
        from topogym.tasks.knot_untying.redesign.geometry import projected_crossings
        ropes = [self.rods[k].nodes(self.d) for k in self.rods]
        return projected_crossings(ropes)

    def goal_metrics(self, region=None):
        """The family predicate's parts on the chain's node centre line (material stations s_nodes): planar clearance at
        2 r (every rope, and between ropes), on table, speed; plus the redesign's region (if given)."""
        from topogym.utils.geometry.passage import planar_clearance
        out = dict(ropes={})
        all_ok = True
        names = list(self.rods)
        for name in names:
            rod = self.rods[name]
            p = to_local(rod.nodes(self.d))
            clear, geo = planar_clearance(p, rod.s_nodes, 2 * rod.radius)
            z = p[:, 2]
            on_table = bool(z.min() >= -.001 and z.max() <= 2 * rod.radius + .003)
            v = float(rod.speeds(self.d).max())
            rec = dict(planar_clear=bool(clear), min_planar_distance_m=geo.get("minimum_nonlocal_distance_m"),
                       on_table=on_table, z_max_m=float(z.max()), max_speed_m_s=v)
            if region is not None and name in region:
                poly = np.asarray(region[name], float)
                inside = _inside(p[:, :2], poly)
                rec["in_region"] = bool(inside.all())
                rec["outside_fraction"] = float(1 - inside.mean())
            out["ropes"][name] = rec
            all_ok &= rec["planar_clear"] and on_table and rec.get("in_region", True)
        if len(names) > 1:
            from topogym.utils.geometry.passage import segment_distance_2d
            dmin = np.inf
            for i in range(len(names)):
                for j in range(i + 1, len(names)):
                    a = to_local(self.rods[names[i]].nodes(self.d))[:, :2]
                    b = to_local(self.rods[names[j]].nodes(self.d))[:, :2]
                    from scipy.spatial import cKDTree
                    dd, _ = cKDTree(b).query(a)
                    dmin = min(dmin, float(dd.min()))
            out["mutual_min_planar_distance_m"] = dmin
            all_ok &= dmin >= 2 * self.rods[names[0]].radius
        out["crossings"] = len(self.crossings())
        out["static_ok"] = bool(all_ok)
        return out

    # -------------------------------------------------------------- actuation
    def anchor_force(self):
        """{eq id: translational force (N)} of the active anchor welds (last step)."""
        d = self.d
        out = {}
        ids = np.asarray(d.efc_id[:d.ne])
        for e in getattr(self, "_anchor_ids", []):
            if d.eq_active[e]:
                rows = np.flatnonzero(ids == e)
                out[e] = float(np.linalg.norm(d.efc_force[rows[0]:rows[0] + 3])) if len(rows) else 0.
        return out

    def anchor(self, rope, end="start", links=4, on=True, slip_n=None):
        """Weld the end links to the world (the clamp). slip_n: the clamp yields (all its welds release) when the
        20 ms mean of its summed weld force exceeds slip_n (None: never)."""
        import mujoco
        self._anchor_ids = list(getattr(self, "_anchor_ids", []))
        self._anchor_slip = getattr(self, "_anchor_slip", {})
        for j in range(links):
            e = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_EQUALITY, f"{rope}::anchor_{end}{j}")
            if on:
                b = self.m.eq_obj1id[e]
                # weld body1 (the link) to body2 (the world) at its current pose, in grasp.py's eq_data convention:
                # [the point in body2's frame (world), the same point in body1's frame, q_body1^-1 q_body2, torquescale]
                q = self.d.xquat[b]
                self.m.eq_data[e, 0:3] = self.d.xpos[b]
                self.m.eq_data[e, 3:6] = 0.
                self.m.eq_data[e, 6:10] = [q[0], -q[1], -q[2], -q[3]]
                self.m.eq_data[e, 10] = 1.
            self.d.eq_active[e] = 1 if on else 0
            if on and e not in self._anchor_ids:
                self._anchor_ids.append(e)
        if slip_n is not None:
            self._anchor_slip[(rope, end)] = dict(limit=float(slip_n), window=deque(maxlen=max(1, round(.02 / self.P["dt"]))),
                                                  ids=[e for e in self._anchor_ids])
        self.events.append(dict(t=self.t, kind="anchor", rope=rope, end=end, links=links, on=on, slip_n=slip_n))

    def hand_pos(self, h):
        """The hand's commanded (target) position."""
        return self.d.mocap_pos[self.hands[h]["mocap"]].copy()

    def hand_actual(self, h):
        return self.d.xpos[self.hands[h]["body"]].copy()

    def set_hand(self, h, pos, quat=None):
        hd = self.hands[h]
        self.d.mocap_pos[hd["mocap"]] = pos
        if quat is not None:
            self.d.mocap_quat[hd["mocap"]] = quat

    def place_hand(self, h, pos, quat):
        """Teleport target and hand body (a pinch that has arrived: the planner moved the hand there)."""
        hd = self.hands[h]
        self.set_hand(h, pos, quat)
        self.d.qpos[hd["qadr"]:hd["qadr"] + 3] = pos
        self.d.qpos[hd["qadr"] + 3:hd["qadr"] + 7] = quat
        self.d.qvel[hd["dadr"]:hd["dadr"] + 6] = 0.

    def grasp(self, h, rope, s, *, check_pinch=True, lift=0.):
        """Weld hand h to the rope link at material s (the hand jumps to the link centre: a pinch closing on it)."""
        import mujoco
        clearance = self.pinch_clearance_at(rope, s)
        ok = clearance >= self.pinch_clearance
        rec = dict(t=self.t, kind="grasp", hand=h, rope=rope, s=float(s), pinch_clearance_m=float(clearance),
                   pinch_ok=bool(ok), point_world=self.point(rope, s).round(5).tolist(),
                   tangent_world=self.tangent(rope, s).round(4).tolist())
        if check_pinch and not ok:
            rec["refused"] = "no room for a parallel gripper at this strand"
            self.events.append(rec)
            return False
        hd = self.hands[h]
        k = self.link_at(rope, s)
        b = self.rods[rope].bodies[k]
        c = self.rods[rope].centres(self.d)[k]
        self.place_hand(h, c + [0, 0, lift], self.d.xquat[b].copy())
        mujoco.mj_kinematics(self.m, self.d)
        e = hd["eq"]
        self.m.eq_obj2id[e] = b
        # as grasp.MujocoGraspBackend.acquire: anchor at the hand origin, expressed in the target's frame
        hp, hq = self.d.xpos[hd["body"]].copy(), self.d.xquat[hd["body"]].copy()
        Rh = Rotation.from_quat(np.r_[hq[1:], hq[0]])
        Rl = Rotation.from_quat(np.r_[self.d.xquat[b][1:], self.d.xquat[b][0]])
        p_rel = Rh.inv().apply(self.d.xpos[b] - hp)
        q_rel = (Rh.inv() * Rl).as_quat()[[3, 0, 1, 2]]
        self.m.eq_data[e, 0:3] = -Rotation.from_quat(np.r_[q_rel[1:], q_rel[0]]).inv().apply(p_rel)
        self.m.eq_data[e, 3:6] = 0.
        self.m.eq_data[e, 6:10] = q_rel
        self.m.eq_data[e, 10] = 1.
        self.d.eq_active[e] = 1
        hd["grasp"] = dict(rope=rope, s=float(s), link=k)
        hd["window"].clear()
        hd["window"].extend([0.] * hd["window"].maxlen)
        self.events.append(rec)
        return True

    def release(self, h, reason="release"):
        hd = self.hands[h]
        if hd["grasp"] is not None:
            self.d.eq_active[hd["eq"]] = 0
            self.events.append(dict(t=self.t, kind=reason, hand=h, **hd["grasp"]))
            hd["grasp"] = None

    def chain_tension(self):
        """{rope: max force (N) over its link-link connect constraints} at the last step: the chain's internal tension
        (coordinator 10-10, R3: compared with the fault tension T_crit)."""
        import mujoco
        d = self.d
        ids = np.asarray(d.efc_id[:d.nefc])
        types = np.asarray(d.efc_type[:d.nefc])
        if not hasattr(self, "_connect_ids"):
            self._connect_ids = {}
            for name in self.rods:
                self._connect_ids[name] = {e for e in range(self.m.neq)
                                           if (mujoco.mj_id2name(self.m, mujoco.mjtObj.mjOBJ_EQUALITY, e) or "").startswith(f"{name}::node")}
        out = {}
        eq_rows = np.flatnonzero(types == mujoco.mjtConstraint.mjCNSTR_EQUALITY)
        for name, eids in self._connect_ids.items():
            best = 0.
            for r0 in eq_rows:
                e = int(ids[r0])
                if e in eids and (r0 == 0 or int(ids[r0 - 1]) != e or types[r0 - 1] != types[r0]):
                    best = max(best, float(np.linalg.norm(d.efc_force[r0:r0 + 3])))
            out[name] = best
        return out

    def weld_force(self, h):
        d = self.d
        e = self.hands[h]["eq"]
        if not d.eq_active[e]:
            return 0.
        rows = np.flatnonzero(np.asarray(d.efc_id[:d.ne]) == e)
        if not len(rows):
            return 0.
        return float(np.linalg.norm(d.efc_force[rows[0]:rows[0] + 3]))

    def step(self, n=1):
        import mujoco
        for _ in range(n):
            for rod in self.rods.values():
                rod.apply(self.d)
            mujoco.mj_step(self.m, self.d)
            self.t += self.P["dt"]
            self.steps += 1
            for h, hd in self.hands.items():
                if hd["grasp"] is None:
                    continue
                f = self.weld_force(h)
                hd["window"].append(f)
                mean = sum(hd["window"]) / len(hd["window"])
                self.max_force[h] = max(self.max_force[h], mean)
                if mean > hd["limit"] or f > 3 * hd["limit"]:
                    self.release(h, reason="slip")
            for key, rec in list(getattr(self, "_anchor_slip", {}).items()):
                f = sum(self.anchor_force().get(e, 0.) for e in rec["ids"])
                rec["window"].append(f)
                rec["max"] = max(rec.get("max", 0.), sum(rec["window"]) / len(rec["window"]))
                if sum(rec["window"]) / len(rec["window"]) > rec["limit"]:
                    for e in rec["ids"]:
                        self.d.eq_active[e] = 0
                    self.events.append(dict(t=self.t, kind="anchor_slip", rope=key[0], end=key[1], limit=rec["limit"]))
                    del self._anchor_slip[key]
            if self.steps % self.log_every == 0:
                self.trace.append(dict(t=round(self.t, 4), hands={h: (self.hand_pos(h).round(4).tolist(),
                                                                         round(sum(hd["window"]) / max(1, len(hd["window"])), 3)
                                                                         if hd["grasp"] else None)
                                                                    for h, hd in self.hands.items()}))
                for h in self.hands:
                    self.hand_path[h].append(self.hand_actual(h))
                self.hand_t.append(self.t)
            if self.d.warning[mujoco.mjtWarning.mjWARN_BADQACC].number:
                forces = {h: round(float(sum(hd["window"]) / max(1, len(hd["window"]))), 2) for h, hd in self.hands.items()}
                raise FloatingPointError(f"MuJoCo: bad qacc at t={self.t:.4f} (hand forces {forces})")

    def wait(self, seconds):
        self.step(int(round(seconds / self.P["dt"])))

    def move(self, moves, speed=.05, min_time=.2):
        """moves: {hand: target (world, m)}; all hands go together on smooth (min-jerk) paths at <= speed."""
        starts = {h: self.hand_pos(h) for h in moves}
        dist = max(float(np.linalg.norm(np.asarray(moves[h]) - starts[h])) for h in moves)
        T = max(min_time, 1.875 * dist / speed)
        n = max(1, int(round(T / self.P["dt"])))
        for i in range(1, n + 1):
            u = i / n
            sm = 10 * u ** 3 - 15 * u ** 4 + 6 * u ** 5
            for h in moves:
                self.set_hand(h, starts[h] + (np.asarray(moves[h]) - starts[h]) * sm)
            self.step(1)
        return T

    def path(self, h, waypoints, speed=.05):
        for w in waypoints:
            self.move({h: np.asarray(w, float)}, speed)

    def quiet(self, limit=.001):
        return max(float(r.speeds(self.d).max()) for r in self.rods.values()) <= limit

    def settle(self, max_s=3., limit=.001, hold=.5, check_every=.05):
        """Step until every link is slower than limit for `hold` seconds (or max_s)."""
        t0, quiet_since = self.t, None
        while self.t - t0 < max_s:
            self.wait(check_every)
            if self.quiet(limit):
                quiet_since = self.t if quiet_since is None else quiet_since
                if self.t - quiet_since >= hold - 1e-9:
                    return True
            else:
                quiet_since = None
        return False


def _inside(xy, poly):
    """Points inside a polygon (even-odd rule)."""
    x, y = xy[:, 0], xy[:, 1]
    inside = np.zeros(len(xy), bool)
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        cond = ((y1 > y) != (y2 > y)) & (x < (x2 - x1) * (y - y1) / np.where(y2 != y1, y2 - y1, 1e-18) + x1)
        inside ^= cond
    return inside


def _point_geom_distance(m, d, g, p):
    """Distance from point p to static geom g's surface (box / cylinder / capsule; others: bounding sphere)."""
    import mujoco
    pos, R = d.geom_xpos[g], d.geom_xmat[g].reshape(3, 3)
    q = R.T @ (p - pos)
    t = m.geom_type[g]
    size = m.geom_size[g]
    if t == mujoco.mjtGeom.mjGEOM_BOX:
        out = np.abs(q) - size
        return float(np.linalg.norm(np.maximum(out, 0)) + min(out.max(), 0))
    if t == mujoco.mjtGeom.mjGEOM_CYLINDER:
        dr = math.hypot(q[0], q[1]) - size[0]
        dz = abs(q[2]) - size[1]
        return float(math.hypot(max(dr, 0), max(dz, 0)) + min(max(dr, dz), 0))
    if t == mujoco.mjtGeom.mjGEOM_CAPSULE:
        z = np.clip(q[2], -size[1], size[1])
        return float(np.linalg.norm(q - [0, 0, z]) - size[0])
    return float(np.linalg.norm(p - pos) - m.geom_rbound[g])
