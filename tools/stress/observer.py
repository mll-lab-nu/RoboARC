"""The stress observer for a RoboTangle MuJoCo Session: read-only (it never writes the model or the data); a run with it
ends on the same state hash as a run without it.

After EVERY MuJoCo step (0.625 ms), not only at the tick's end as the Session's own readout does:
  - non-finite qpos / qvel / qacc, MuJoCo warnings of any kind, solver iterations at the cap;
  - every contact's depth and force (the full vector, decoded from efc_force as mj_contactForce does), by pair category:
    task-robot, task-task, task-static, robot-static, robot-robot (task = a movable non-robot link);
  - task-link speeds, and the work of every non-conservative force (drives, gravity compensation, damping, fluid,
    applied, with implicitfast's qDeriv term or the discrete integrator's effective metric) and of the constraint forces;
    each contact pair's work; with flexes under the discrete integrator, its own loss on their elasticity (L).
After every World tick (2.5 ms):
  - task-link speeds and the causal check of every speed jump; each task link's spin and its surface speed omega * r_link
    (criteria v2.7's angular trigger); tunnelling and the workspace; grasp losses by reason;
  - the energy balance: E = kinetic + potential on a private MjData, W = the work above, X = E - E0 - W (- min L). A
    passive scene has X <= 0; X equals the constraint work to 0.1 J in every run (the books balance). A rope chain's rod
    is stored energy: its elastic energy is in E and its elastic couples (applied forces) are left out of W.
  - S = sum max(0, -1/2 f pos) over the active constraint rows (an estimate of the soft constraints' stored energy) and
    the rise of X + S above its minimum; X's rise within 0.1 s; each contact pair's net work over its contact episode.

The causal check (docs/simulator/STRESS_VALIDATION.md, criteria v2.6): a task link whose centre-of-mass speed rises by more than
JUMP_M_S within one tick needs a cause:
  - a contact or weld partner moving fast enough: |v_after| <= max(sqrt(|v_before|^2 + w.Iw/m), its own peak speed over
    the last 50 ms) + 2 * v_partner + 0.1 m/s;
  - or a chain: the non-task partner (robot link, fixture, weld) of a task link connected to this one by task-task
    contacts in this tick or the previous one, with the same bound (an impulse passing through several links in one
    MuJoCo step);
  - or, for a link of an articulated entity (a cable), a hit on the entity by another entity within WHIP_WINDOW_S at point
    speed u: |v_after| <= WHIP_GAIN * (2u + 0.1) (a whip; reported as violent but causal when the link ends above
    3 m/s, criteria v2.8; every whip's end speed is the peak whip_speed);
  - or a stored-energy release (a "pop"): the link's kinetic energy gain in the tick is at most the drop of the stored
    energy acting on it (a rope link's includes its rod joints' elastic energy); a pop whose kick |v_after - v_before|
    exceeds POP_SPEED_MAX is a breakdown (criteria v2.9; up to v2.8 its after-speed);
  - or the robot paid for it (criteria v2.4): in that tick the static and task partners did <= 0 work on the body, its
    kinetic-energy gain is at most the robot partners' work (contacts + weld), and every robot contact that did positive
    work is passive itself (its pair net over the contact episode <= PAIR_PASSIVE_J). A wedge ejection: an arm squeezes
    a body out of a fixture faster than twice its own speed, and pays for it.
  - or, for a link of an articulated entity that no active equality ties to another body, redistribution within the
    chain (criteria v2.5): in that tick the link took
    no positive work from outside its entity, the entity's kinetic energy rose by at most the work done on it from outside
    (contacts with other entities, the robot and statics) plus gravity's (+ ENTITY_EPS_J), and every outside contact that
    did positive work is passive. A pinned-rod whip: a flung bundle stopped by a still link, its strand tips 10-16 %
    faster while the bundle's kinetic energy drops (reported as a whip, violent but causal).
  - criteria v2.6, checked after all of the above so that a report can still read v2.5 exactly: gravity's own speed
    change within one tick, |g| dt_tick, in the first two bounds (jump_causal_gravity; jump_causal_gravity_pop for a
    v2.5 fast pop); and redistribution within a chain that an equality ties to another body, with each tie's work on
    the entity's dofs booked as outside work (the tie passive over its active episode) and the drop of the entity's
    exact stored potentials (joint springs, a rope's rod) in the allowance; the link a tie holds is not excused this
    way (whip_redistribution_v26).
  - criteria v2.9 (2026-10-10), after all of the above: the chain's balance of rule 6 also over the last
    CHAIN_WINDOW_TICKS ticks (an impulse that crosses a tick boundary inside the chain; whip_redistribution_v29).
  - criteria v2.10 (2026-10-11), after all of the above: rule 6 (its v2.5, v2.6 and v2.9-window forms) with a numerical
    tolerance on the link clause: the link's positive work from outside its entity up to LINK_OUT_EPS_J is contact
    round-off, not outside work (a rope link lying on the table takes ~1e-8 J from it while the chain kicks it;
    whip_redistribution_v210).
  - criteria v2.10 (2026-10-11), the pop guard's scope: a fast pop (above 3 m/s by v2.8's after-speed or v2.9's kick) whose
    kinetic-energy gain the drop of the link's rod elastic energy alone covers (a rope's bend / twist potential, a
    modelled physical energy, released as a wave toward the free end) is no breakdown: pop_rod_v210_<fast|kick|after>,
    the suffix naming the class it had up to v2.9. The guard keeps every release of energy stored in contacts,
    equalities (welds, connects) or limits.
  - criteria v2.11: a fast pop that the rod does not pay for is a breakdown only if the constraint-stored
    energy released into that link in the tick (contacts, equalities, limits; the rod's elastic storage excluded) could
    alone give it the guard's speed, v_c = sqrt(2 E_c / m_link) > POP_SPEED_MAX (pop_share_v211_<fast|kick|after>).
Anything else is an uncaused jump: a breakdown.
Criteria v2.9 also books the equality and limit rows' work per constraint (constraint_net_work_j: with the contact pairs,
the local energy rule's culprit), and records no peaks or events after the first mjWARN_BADQACC, MuJoCo's auto-reset to
qpos0 (the summary's post_reset)."""
from __future__ import annotations

import math
import os
from collections import defaultdict

import numpy as np

JUMP_M_S = .5              # a speed jump within one tick that needs a causal partner
REBOUND_TICKS = 20         # 50 ms: a body may rebound up to its own peak speed within this window (stored contact energy)
WHIP_WINDOW_S = .25        # a cable whip needs an external hit within this window before
WHIP_GAIN = 6.             # ~sqrt(links set moving): the speed gain a uniform chain allows from a hit to its tip
TRIGGERS = dict(speed=3., angular=60., force=2000., depth=.010, review_depth=.003)
# the degenerate-row signature (claws W4 + rule A, CL8t): a contact row with A_ii (efc_diagA) below DEGENERATE_A that
# carries more than DEGENERATE_F_N, whatever its bodies (BL1 e30 before rule A: 40.7 kN at A_ii 2.8e-4). Recorded as
# the event degenerate_row and the peak degenerate_force; a measurement, never a verdict.
DEGENERATE_A, DEGENERATE_F_N = .1, 2000.
WHIP_KINDS = ("whip_causal", "whip_redistribution", "whip_redistribution_v26", "whip_redistribution_v29",
              "whip_redistribution_v210")
POP_SPEED_MAX = 3.         # a stored-energy pop faster than this is a breakdown (criteria v2)
WORK_CLAUSE = True         # criteria v2.4 (2026-10-10): a robot-powered jump is causal (see the docstring)
PAIR_PASSIVE_J = .01       # a robot contact counts as passive when its pair net over the contact episode is at most this
REDISTRIBUTION = True      # criteria v2.5 (2026-10-10): a chain's internal redistribution is causal (see the docstring)
ENTITY_EPS_J = .001        # its numerical tolerance on the entity's energy balance per tick
TIES_BOOKED = True         # criteria v2.6: a tying equality's work is booked, so a tied chain can use rule 6
STORED_DROP = True         # criteria v2.6: rule 6 counts the drop of the entity's exact stored potentials
GRAVITY_DT = True          # criteria v2.6: rules 1-2 allow gravity's speed change within one tick, |g| dt_tick
CONSTRAINT_BOOKS = True    # criteria v2.9: equality / limit work books (+ forensic S variants)
S_WINDOW = 8               # criteria v2.9: ticks (20 ms) over which S's spikes are filtered
CHAIN_WINDOW_TICKS = 3     # criteria v2.9 (c): rule 6's chain balance also over the last 3 ticks (7.5 ms)
LINK_OUT_EPS_J = 1e-6      # criteria v2.10: rule 6's link clause tolerates this much positive outside work (round-off)
ROD_RELEASE = True         # criteria v2.10: a fast pop that the rod's own elastic energy pays for is no breakdown
ENERGY_SHARE = True        # criteria v2.11: a fast pop fails only if its constraint-stored release alone could
                           # produce the guard's speed: v_c = sqrt(2 E_constraint / m_link) > POP_SPEED_MAX
POP_KEEP = 2000            # pop speeds kept per episode for the distribution
# criteria v2.13b (adopted by the user 2026-10-13; STRESS_INTEGRATOR_BOOKS=0 restores the v2.12 measurements): the
# measurements the rules read, booked as MuJoCo's integrator applies them. Every threshold is unchanged.
#  - contact work: each contact's force at the step's start state times the velocity of its point on each body taken
#    with the start state's kinematics and the step's mean generalized velocity, (v0 + v1) / 2 (= J(q0)' f . v_mean, the
#    work the integrator does with that force). v2.12 took the mean of the point velocity at the start state and at the
#    END state's kinematics, which drifts on a long hold of a turning body (orbits easier prefix14_e1: a finger x cord
#    pair booked +22.7 mJ net over a 19,000-tick hold that the integrator books at +1.1 mJ, so rule 6 refused a passive
#    contact);
#  - a task link's kinetic energy includes its joints' armature (rotor) energy, 1/2 sum armature v^2 over the dofs of
#    the link's own joints: MuJoCo's mass matrix, the global energy E and the integrator carry it, and a chain whose
#    stored bend energy is released through rotors otherwise shows link kinetic energy "from nowhere".
INTEGRATOR_BOOKS = os.environ.get("STRESS_INTEGRATOR_BOOKS", "1") != "0"
CATS = ("task_robot", "task_task", "task_static", "robot_static", "robot_robot")


class _SpeedBook:
    """Per body, the fastest partner point speed within a tick and that partner (a dict-like .get(body) -> (speed, name)).
    Updated per substep with numpy: each body's first maximum in contact order, kept only if strictly faster than the
    book (so earlier substeps win ties, as a sequential pass would)."""
    __slots__ = ("speed", "other", "names")

    def __init__(self, nb, names):
        self.speed = np.full(nb, -1.)
        self.other = np.full(nb, -1, np.int64)
        self.names = names

    def get(self, body):
        sp = self.speed[body]
        return None if sp < 0 else (float(sp), self.names[int(self.other[body])])

    def update(self, body, other, sp):
        if not len(body):
            return
        order = np.lexsort((-sp, body))             # by body, fastest first; stable: the earliest contact on ties
        bs = body[order]
        first = np.ones(len(order), bool)
        first[1:] = bs[1:] != bs[:-1]
        i = order[first]
        bu, su, ou = body[i], sp[i], other[i]
        better = su > self.speed[bu]
        self.speed[bu[better]] = su[better]
        self.other[bu[better]] = ou[better]


class Observer:
    def __init__(self, session, *, trace=False):
        import mujoco
        self.mujoco = mujoco
        self.s, self.w = session, session.env._world
        w = self.w
        self.m, self.d = w.model, w.data
        m = self.m
        self.dt_sub = float(m.opt.timestep)
        self.g_dt = float(np.linalg.norm(np.asarray(m.opt.gravity))) * float(w.physics_dt_s)   # |g| dt_tick (v2.6)
        self.elliptic = int(m.opt.cone) == int(mujoco.mjtCone.mjCONE_ELLIPTIC)
        self.discrete = int(m.opt.integrator) == int(mujoco.mjtIntegrator.mjINT_DISCRETE)
        self.iter_cap = int(m.opt.iterations)
        nb = m.nbody
        self.cat = np.zeros(nb, np.int8)            # 0 static/world, 1 robot (movable), 2 task (movable)
        self.entity_of = np.full(nb, -1, np.int64)  # entity index of each body
        self.articulated = set()                    # entity indices with more than one movable link
        self.entity_names = list(w.entities)
        self.body_name = [m.body(i).name for i in range(nb)]
        for ei, (name, e) in enumerate(w.entities.items()):
            movable = [l for l in e.links if not l.is_fixed]
            for l in e.links:
                self.entity_of[l.idx] = ei
            for l in movable:
                self.cat[l.idx] = 1 if name == "robot" else 2
            if name != "robot" and len(movable) > 1:
                self.articulated.add(ei)
        # a flex's vertex bodies (a cloth) belong to no entity's links: each flex becomes one articulated task entity, so
        # its speeds, jumps, whips, redistribution (with flex_energy as its exact potential) and tunnelling are checked
        self.flex_entity = {}                       # flex id -> entity index
        for f in range(int(m.nflex)):
            adr, num = int(m.flex_vertadr[f]), int(m.flex_vertnum[f])
            vb = sorted({int(b) for b in np.asarray(m.flex_vertbodyid[adr:adr + num]) if b > 0})
            free = [b for b in vb if self.entity_of[b] < 0]
            if not free:
                continue
            ei = len(self.entity_names)
            self.entity_names.append(f"flex:{mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_FLEX, f) or f}")
            self.entity_of[free] = ei
            self.cat[free] = 2
            self.flex_entity[f] = ei
            if len(free) > 1:
                self.articulated.add(ei)
        # bodies without dofs (welded to world) are static even if declared movable
        dof_tree = np.zeros(nb, bool)
        for b in range(nb):
            p, moving = b, False
            while p > 0:
                if m.body_dofnum[p] > 0:
                    moving = True
                    break
                p = m.body_parentid[p]
            dof_tree[b] = moving
        self.cat[~dof_tree] = 0
        self.task = np.flatnonzero(self.cat == 2)
        self.robot = np.flatnonzero(self.cat == 1)
        self.task_index = np.full(nb, -1, np.int64)  # body -> index in self.task (or -1)
        self.task_index[self.task] = np.arange(len(self.task))
        # each task link's size r_link: the farthest point of its geoms from its centre of mass (|geom pos - COM| +
        # bounding radius); omega * r_link is its surface speed, for the angular trigger (criteria v2.7)
        gb = np.asarray(m.geom_bodyid)
        self.r_task = np.zeros(len(self.task))
        for k, b in enumerate(self.task.tolist()):
            gs = np.flatnonzero(gb == b)
            if len(gs):
                off = np.linalg.norm(np.asarray(m.geom_pos)[gs] - np.asarray(m.body_ipos)[b], axis=1)
                self.r_task[k] = float(np.max(off + np.asarray(m.geom_rbound)[gs]))
        self.free_dofs = {}                         # task body with a free joint -> its 6 dof addresses
        for b in self.task.tolist():
            j = int(m.body_jntadr[b])
            if j >= 0 and int(m.body_jntnum[b]) == 1 and int(m.jnt_type[j]) == int(mujoco.mjtJoint.mjJNT_FREE):
                a = int(m.jnt_dofadr[j])
                self.free_dofs[b] = np.arange(a, a + 6)
        # each articulated task entity's own dofs (its tree) and joint springs (exact potentials), for rule 6 (v2.6)
        self.ent_dofs, self.ent_springs = {}, {}
        for e in self.articulated:
            bodies = [int(b) for b in np.flatnonzero(self.entity_of == e)]
            dofs = [int(m.body_dofadr[b]) + k for b in bodies for k in range(int(m.body_dofnum[b]))]
            self.ent_dofs[e] = np.array(sorted(set(dofs)), dtype=np.int64)
            self.ent_springs[e] = [(j, int(m.jnt_type[j]), int(m.jnt_qposadr[j]), float(m.jnt_stiffness[j]))
                                   for j in range(m.njnt) if int(m.jnt_bodyid[j]) in bodies and float(m.jnt_stiffness[j]) > 0]
        C = mujoco.mjtConstraint
        self._ctype_contact = np.array([int(C.mjCNSTR_CONTACT_FRICTIONLESS), int(C.mjCNSTR_CONTACT_PYRAMIDAL),
                                        int(C.mjCNSTR_CONTACT_ELLIPTIC)])
        self._ctype_eq, self._ctype_limit = int(C.mjCNSTR_EQUALITY), int(C.mjCNSTR_LIMIT_JOINT)
        O = mujoco.mjtObj
        self._obj_body, self._obj_xbody, self._obj_site, self._obj_joint = (int(O.mjOBJ_BODY), int(O.mjOBJ_XBODY),
                                                                            int(O.mjOBJ_SITE), int(O.mjOBJ_JOINT))
        self.rootid = np.asarray(m.body_rootid)
        # the robot's finger links (bodies named *finger*): finger-finger contacts and every constraint row on a finger
        # (contacts, the finger joints' limits, equalities on a finger joint or body) with its A_ii (efc_diagA, the
        # constraint diagonal; a degenerate row has A_ii near 0 and takes kN at ~1 mm, claws W4 + rule A, CL8t)
        self.finger = np.zeros(nb, bool)
        for b in self.robot.tolist():
            self.finger[b] = "finger" in self.body_name[b]
        fj = [j for j in range(m.njnt) if self.finger[int(m.jnt_bodyid[j])]]
        self._finger_jnt = np.array(fj, dtype=np.int64)
        eq_t = np.asarray(m.eq_type)
        o1, o2 = np.asarray(m.eq_obj1id), np.asarray(m.eq_obj2id)
        J, B_ = int(mujoco.mjtEq.mjEQ_JOINT), (int(mujoco.mjtEq.mjEQ_CONNECT), int(mujoco.mjtEq.mjEQ_WELD))
        fe = [e for e in range(m.neq)
              if (int(eq_t[e]) == J and (int(o1[e]) in fj or int(o2[e]) in fj))
              or (int(eq_t[e]) in B_ and int(m.eq_objtype[e]) in (int(O.mjOBJ_BODY), int(O.mjOBJ_XBODY))
                  and ((0 <= int(o1[e]) < nb and self.finger[int(o1[e])]) or (0 <= int(o2[e]) < nb and self.finger[int(o2[e])])))]
        self._finger_eq = np.array(fe, dtype=np.int64)
        self._has_diagA = hasattr(self.d, "efc_diagA")
        self.mass = np.asarray(m.body_mass)
        task_obj = session.env._task
        self.table_z = float(getattr(task_obj, "table_height_m", np.nan))
        t = w.env_cfg.world.table
        c, size = np.asarray(t.pos[:2], float), np.asarray(t.size[:2], float)
        self.table_lo, self.table_hi = c - size / 2, c + size / 2
        self.trace = [] if trace else None
        self.ropes = getattr(w, "_ropes", None) or {}   # rope chains (topogym/world/mujoco/rope_chain.py): the rod's books
        # VBD cloths (topogym/world/mujoco/cloth_vbd.py, stepped outside MuJoCo): one energy book each, kept from the
        # cloth's step hook (vbd_books.VBDBook); their pins' reactions on the hands (xfrc_applied) are booked in W here
        self.vbd = {}
        for name, cloth in (getattr(w, "_cloths", None) or {}).items():
            if getattr(cloth, "steps_outside", False) and hasattr(cloth, "step_hooks"):
                from vbd_books import VBDBook
                book = VBDBook(cloth)

                def hook(c, dt, book=book):
                    try:
                        book.hook(c, dt)
                    except Exception:               # the observer's own fault must not read as the simulator's
                        self._note_error()
                cloth.step_hooks.append(hook)
                self.vbd[name] = book
        self.rope_link_energy = {}                  # rod energy stored at each rope link's joints, the tick's end (body -> J)
        self._clamp0 = self._rope_clamp()           # the ropes' speed-clamp counts at the start (counted per episode)
        self._Mv = np.zeros(m.nv)
        self.d2 = mujoco.MjData(m)                  # private: energy at the tick's end state (never the session's data)
        # v2.13 books: a private MjData for the contact points' velocities at the step's start kinematics and mean
        # generalized velocity; each task link's armature dofs (the dofs of its own joints)
        self._d4 = mujoco.MjData(m) if INTEGRATOR_BOOKS else None
        self.res_hist = []
        self.qpos_before = None
        dof_b = np.asarray(m.dof_bodyid)
        tb = self.task_index[dof_b] if len(self.task) else np.full(m.nv, -1)
        self._arm_dofs = np.flatnonzero((tb >= 0) & (np.asarray(m.dof_armature) > 0)) if INTEGRATOR_BOOKS else np.zeros(0, int)
        self._arm_task = tb[self._arm_dofs]
        self._arm_val = np.asarray(m.dof_armature)[self._arm_dofs]
        self._D_rows = np.repeat(np.arange(m.nv), np.asarray(m.D_rownnz))
        self._D_cols = np.asarray(m.D_colind)
        # flexes under the discrete integrator (cloth): their vertex dofs and a private MjData for the integrator's loss
        self._flex_dofs = self._flex_vertex_dofs() if m.nflex and self.discrete else None
        self._d3 = mujoco.MjData(m) if self._flex_dofs is not None else None
        self.Wc = 0.                                # work of all constraint forces (contacts, welds, limits, equalities)
        self._pairs_reset()                         # per body pair: cumulative contact work J, running min, last substep
        T = mujoco.mjtConstraint                    # v2.9: the row types the equality / limit books cover
        self._cb_types = np.array([int(T.mjCNSTR_EQUALITY), int(T.mjCNSTR_LIMIT_JOINT), int(T.mjCNSTR_LIMIT_TENDON)])
        self._eq_type = int(T.mjCNSTR_EQUALITY)
        self.pending = None                         # last substep's contacts awaiting their end-of-step velocities
        self.reset_run()
        self._install()

    PRESS_WINDOW_S = 0.25     # v2.12 pressed pile: a cloth-floor guard stop within this time marks the cloth step pressed

    def _press_flag(self):
        """Mark the VBD books' next cloth steps as pressed while a gripper was stopped on its cloth floor (the Session's
        table guard, floor "cloth") within PRESS_WINDOW_S: the guard counts its stops per control step."""
        mon = getattr(self.s, "monitor", None)
        n = int(mon.episode.get("guard", 0)) if mon is not None else 0
        if n > self._guard_n:
            ev = getattr(self.s, "last_guard", None) or {}
            if any(isinstance(v, dict) and v.get("floor") == "cloth" for v in ev.values()):
                self._press_tick = self.ticks
        self._guard_n = n
        pressed = self._press_tick is not None and (self.ticks - self._press_tick) * float(self.w.physics_dt_s) <= self.PRESS_WINDOW_S
        for book in self.vbd.values():
            book.pressed = pressed

    def reset_run(self):
        """Clear every per-episode measurement (call after Session.reset; the search runs many episodes per process)."""
        mon = getattr(getattr(self, "s", None), "monitor", None)
        self._guard_n = int(mon.episode.get("guard", 0)) if mon is not None else 0
        self._press_tick = None
        for book in (getattr(self, "vbd", None) or {}).values():   # the VBD cloths' books are per episode too
            book.reset()
        self._warn0 = self._warnings()
        self.trace = [] if self.trace is not None else None
        self.peaks = {}
        self.events = defaultdict(list)            # kind -> [records] (first few kept per kind)
        self.counts = defaultdict(int)
        self.nsub = 0
        self.ticks = 0
        self.v_prev = None                          # task-link COM speeds at the previous tick end
        self.lin_prev = None                        # their COM velocity vectors (v2.9 (d): a pop's kick |v_after - v_before|)
        self.vrot_prev = None                       # their rotational-energy equivalent speeds (sqrt(w.Iw/m))
        self.v_hist = []                            # task-link speeds of the last REBOUND_TICKS ticks
        self.KE_prev = None                         # task-link kinetic energies (translation + rotation) at the last tick
        self.com_prev = None                        # task-link COM positions at the last tick (gravity's work, v2.5)
        self.tie_work_tick = defaultdict(float)     # (entity, equality) -> the tie's work on the entity's dofs this tick
        self.eq_net = {}                            # tying equality -> its total work (both sides) over its active episode
        self.U_prev = {}                            # articulated entity -> its exact stored potentials at the last tick
        self.S_prev = None                          # stored constraint energy acting on each task link at the last tick
        self.R_prev = None                          # v2.10: the rod elastic part of it (rope_link_energy) at the last tick
        self.pop_speeds = []                        # speeds after every stored-energy pop (distribution)
        nb_ = self.m.nbody
        self.partner_tick = _SpeedBook(nb_, self.body_name)   # task body -> (max partner point speed, partner) this tick
        self.partner_prev = _SpeedBook(nb_, self.body_name)
        self.tt_tick, self.tt_prev = set(), set()   # task-task contact pairs (different entities) this / last tick
        self._bw_parts, self._bw_cache = [], None   # this tick's contact work on task bodies, per (body, partner), lazily
        self.body_cwork_tick = defaultdict(float)   # task free body -> all constraint work on its dofs this tick
        self.ext_tick, self.ext_prev = _SpeedBook(nb_, self.body_name), _SpeedBook(nb_, self.body_name)   # non-task partners
        self.ext_hits = []                          # (tick, entity, partner speed) external hits on articulated entities
        self.qvel_before = None
        self.W = 0.
        self.E0 = None
        self.excess_hist = []
        self.solver_capped = 0
        self.max_niter = 0
        self.nonfinite = None
        self.warn_new = {}
        self.observer_error, self.observer_errors = None, 0   # an exception in the observer itself (never the physics)
        self.Wc = 0.
        self.L = self.L_min = 0.                    # the integrator's loss on flex elasticity (_flex_loss)
        self._U_end = None
        self._pairs_reset()
        self.pending = None
        self.Y_min = None
        self._cbooks = {}                           # v2.9: (row type, id) -> [episode net work, running min, last substep]
        self.Y2_min = None                          # v2.9: the running minimum of X + the springs' stored energy
        self.S_hist, self.Yw_min = [], None         # v2.9: the last S_WINDOW ticks of S; the windowed Y's floor
        self.ent_hist = {}                          # v2.9 (c): articulated entity -> its last CHAIN_WINDOW_TICKS balances
        self.post_reset = None                      # v2.9 (e): the first mjWARN_BADQACC (MuJoCo's auto-reset), if any

    # ---------------------------------------------------------------- helpers
    def _warnings(self):
        return np.array([int(self.d.warning[i].number) for i in range(len(self.d.warning))], dtype=np.int64)

    def _peak(self, key, value, where=None):
        if not math.isfinite(value) or self.post_reset is not None:   # v2.9 (e): nothing after MuJoCo's auto-reset
            return
        old = self.peaks.get(key)
        if old is None or value > old[0]:
            self.peaks[key] = (float(value), int(self.w.physics_step), self.nsub, where)

    def _event(self, kind, rec, keep=12):
        if self.post_reset is not None and kind not in ("warning", "nonfinite"):
            return                  # v2.9 (e): after MuJoCo's auto-reset (mjWARN_BADQACC) the state is qpos0, not physics
        self.counts[kind] += 1
        if len(self.events[kind]) < keep:
            self.events[kind].append(rec)
        if kind in WHIP_KINDS:       # criteria v2.8: a whip is a VIOLENT trigger only when its link ends above 3 m/s
            after = float(rec.get("after") or 0.)
            self._peak("whip_speed", after, rec.get("body"))
            if after > TRIGGERS["speed"]:
                self.counts[kind + "_fast"] += 1

    def _finger_contacts(self, b1, b2, fn, pen, adr, dim, act):
        """Finger-finger contacts' peak force and depth (force_finger_finger, depth_finger_finger) and the smallest A_ii
        (efc_diagA) of any constraint row on a finger, as 1 / A_ii so that the peak is the smallest A_ii: the
        finger-finger contacts' rows (ff_row_inv_A, the degenerate-row mechanism), every finger contact's rows (with the
        table, a piece or another link: finger_row_inv_A), and its joints' limit and equality rows (finger_eqlim_inv_A).
        The contact peaks' `where` is (body, body, the contact's force N, its depth mm) at that substep."""
        f1, f2 = self.finger[b1], self.finger[b2]
        pair = lambda i: (self.body_name[b1[i]], self.body_name[b2[i]])
        ff = np.flatnonzero(f1 & f2)
        if len(ff):
            a, b = int(ff[np.argmax(fn[ff])]), int(ff[np.argmax(pen[ff])])
            self._peak("force_finger_finger", fn[a], pair(a))
            self._peak("depth_finger_finger", pen[b], pair(b))
        if not self._has_diagA:
            return
        dA = np.asarray(self.d.efc_diagA)
        rows = np.flatnonzero((f1 | f2) & act)
        if len(rows):
            amin = self._rows_min_A(rows, adr, dim)
            ctx = lambda i: pair(i) + (round(float(fn[i]), 1), round(1000 * float(pen[i]), 3))   # + its force N, depth mm
            j = int(np.argmin(amin))
            if amin[j] > 0:
                self._peak("finger_row_inv_A", 1. / amin[j], ctx(rows[j]))
            both = np.flatnonzero(f1[rows] & f2[rows])
            if len(both):
                j = int(both[np.argmin(amin[both])])
                if amin[j] > 0:
                    self._peak("ff_row_inv_A", 1. / amin[j], ctx(rows[j]))
        if len(self._finger_jnt) or len(self._finger_eq):
            n = int(self.d.nefc)
            et, eid = np.asarray(self.d.efc_type[:n]), np.asarray(self.d.efc_id[:n])
            sel = np.flatnonzero(((et == self._ctype_limit) & np.isin(eid, self._finger_jnt))
                                 | ((et == self._ctype_eq) & np.isin(eid, self._finger_eq)))
            if len(sel):
                j = int(sel[np.argmin(dA[sel])])
                if dA[j] > 0:
                    what = ("limit " if int(et[j]) == self._ctype_limit else "equality ") + str(int(eid[j]))
                    self._peak("finger_eqlim_inv_A", 1. / float(dA[j]), what)

    def _rows_min_A(self, idx, adr, dim):
        """The smallest A_ii (efc_diagA) over each listed contact's constraint rows (elliptic: dim rows; pyramidal:
        2 (dim - 1), or 1 when frictionless)."""
        dA = np.asarray(self.d.efc_diagA)
        nrow = dim[idx] if self.elliptic else np.where(dim[idx] > 1, 2 * (dim[idx] - 1), 1)
        amin = np.full(len(idx), np.inf)
        for r in np.unique(nrow):
            k = np.flatnonzero(nrow == r)
            amin[k] = dA[adr[idx[k]][:, None] + np.arange(int(r))[None, :]].min(axis=1)
        return amin

    def _com_vel(self, bodies):
        """(linear COM velocity, angular velocity) of bodies, world frame, from cvel / xipos / subtree_com."""
        d = self.d
        cv = np.asarray(d.cvel)[bodies]
        ang = cv[:, :3]
        lin = cv[:, 3:] + np.cross(ang, np.asarray(d.xipos)[bodies] - np.asarray(d.subtree_com)[self.rootid[bodies]])
        return lin, ang

    def _point_vel(self, bodies, points):
        d = self.d
        cv = np.asarray(d.cvel)[bodies]
        return cv[:, 3:] + np.cross(cv[:, :3], points - np.asarray(d.subtree_com)[self.rootid[bodies]])

    def _tip_peaks(self, an):
        """The angular trigger's readings (criteria v2.7): the fastest surface speed omega * r_link (task_tip_speed), and
        the fastest spin among the links whose surface moves faster than the speed trigger (task_angular_fast_surface)."""
        tip = an * self.r_task
        k = int(np.argmax(tip))
        self._peak("task_tip_speed", tip[k], self.body_name[self.task[k]])
        big = tip > TRIGGERS["speed"]
        if big.any():
            k = int(np.flatnonzero(big)[np.argmax(an[big])])
            self._peak("task_angular_fast_surface", an[k], self.body_name[self.task[k]])

    def _contact_bodies(self, ci):
        """The two bodies of the contacts ci (indices into d.contact). A geom side maps through geom_bodyid; a flex side
        (geom -1) maps to its vertex body (con.vert), or, for an element contact, to the element's vertex nearest the
        contact point (geom_bodyid[-1] would be the model's last geom's body)."""
        m, d = self.m, self.d
        con = d.contact
        g = np.asarray(con.geom)[ci].reshape(-1, 2)
        b = np.asarray(m.geom_bodyid)[np.maximum(g, 0)]
        neg = np.argwhere(g < 0)
        if len(neg):
            fl = np.asarray(con.flex)[ci].reshape(-1, 2)
            ve = np.asarray(con.vert)[ci].reshape(-1, 2)
            el = np.asarray(con.elem)[ci].reshape(-1, 2)
            pos = np.asarray(con.pos)[ci].reshape(-1, 3)
            vbody = np.asarray(m.flex_vertbodyid)
            for r, side in neg.tolist():
                f = int(fl[r, side])
                if f < 0:
                    b[r, side] = 0
                    continue
                vadr = int(m.flex_vertadr[f])
                if int(ve[r, side]) >= 0:
                    b[r, side] = int(vbody[vadr + int(ve[r, side])])
                elif int(el[r, side]) >= 0:
                    dim = int(m.flex_dim[f])
                    a0 = int(m.flex_elemdataadr[f]) + int(el[r, side]) * (dim + 1)
                    loc = np.asarray(m.flex_elem[a0:a0 + dim + 1])
                    bodies = vbody[vadr + loc]
                    x = np.asarray(d.xpos)[bodies]
                    b[r, side] = int(bodies[int(np.argmin(np.linalg.norm(x - pos[r], axis=1)))])
                else:
                    b[r, side] = 0
        return b[:, 0], b[:, 1]

    def _stored_by_task_body(self, e_rows):
        """Soft-constraint energy stored in the constraint rows acting on each task link: contacts (both bodies), equality
        constraints (welds / connects between bodies or sites; joint equalities), joint limits. A row shared by two task
        links counts for both."""
        out = np.zeros(len(self.task))
        n = len(e_rows)
        if not n or not len(self.task):
            return out
        m, d = self.m, self.d
        types, ids = np.asarray(d.efc_type[:n]), np.asarray(d.efc_id[:n])
        ba, bb = np.full(n, -1, np.int64), np.full(n, -1, np.int64)
        isc = np.isin(types, self._ctype_contact)
        if isc.any():
            ci = ids[isc]
            ba[isc], bb[isc] = self._contact_bodies(ci)
        ise = types == self._ctype_eq
        if ise.any():
            for q in np.unique(ids[ise]).tolist():         # one pass per active equality constraint
                rows = ise & (ids == q)
                ot = int(m.eq_objtype[q])
                for dst, obj in ((ba, int(m.eq_obj1id[q])), (bb, int(m.eq_obj2id[q]))):
                    if obj < 0:
                        continue
                    if ot in (self._obj_body, self._obj_xbody):
                        dst[rows] = obj
                    elif ot == self._obj_site:
                        dst[rows] = int(m.site_bodyid[obj])
                    elif ot == self._obj_joint:
                        dst[rows] = int(m.jnt_bodyid[obj])
        isl = types == self._ctype_limit
        if isl.any():
            ba[isl] = np.asarray(m.jnt_bodyid)[ids[isl]]
        for arr in (ba, bb):
            ok = np.flatnonzero(arr >= 0)
            k = self.task_index[arr[ok]]
            good = k >= 0
            np.add.at(out, k[good], e_rows[ok][good])
        return out

    def _chain_partner(self, body, welds, com, k):
        """The fastest external partner (pre-impulse point speed) of the task links connected to `body` through task-task
        contacts of this tick or the previous one (body excluded): (speed, 'partner via link')."""
        pairs = self.tt_tick | self.tt_prev
        if not pairs:
            return 0., None
        adj = defaultdict(set)
        for a, b in pairs:
            adj[a].add(b)
            adj[b].add(a)
        seen, todo = {body}, [body]
        while todo:
            x = todo.pop()
            for y in adj[x]:
                if y not in seen:
                    seen.add(y)
                    todo.append(y)
        best = (0., None)
        for mbody in seen - {body}:
            # only non-task partners can be a chain's root: every task link it touches is inside the chain, and its
            # sampled speed may already carry this tick's impulse
            for c in (self.ext_tick.get(mbody), self.ext_prev.get(mbody)):
                if c and c[1] is not None and c[0] > best[0]:
                    best = (c[0], f"{c[1]} via {self.body_name[mbody]}")
            if mbody in welds:
                j = int(self.task_index[mbody])
                hv = float(np.linalg.norm(self._point_vel(np.array([welds[mbody]]), com[j:j + 1])[0]))
                if hv > best[0]:
                    best = (hv, f"{self.body_name[welds[mbody]]} (weld) via {self.body_name[mbody]}")
        return best

    # ---------------------------------------------------------------- hooks
    def _install(self):
        mujoco = self.mujoco
        original_step = mujoco.mj_step
        obs = self

        def mj_step(m, d, *args, **kwargs):
            if d is obs.d:
                obs.qvel_before = np.array(d.qvel)
                if obs._d4 is not None:
                    obs.qpos_before = np.array(d.qpos)
                original_step(m, d, *args, **kwargs)
                try:
                    obs._after_substep()
                except Exception:                       # the observer's own fault must not read as the simulator's
                    obs._note_error()
            else:
                original_step(m, d, *args, **kwargs)
        self._original_step = original_step
        mujoco.mj_step = mj_step
        w = self.w
        original_advance = w.advance

        def advance():
            if obs.vbd:                                 # v2.12 pressed pile: a gripper held on its cloth floor lately
                try:
                    obs._press_flag()
                except Exception:
                    obs._note_error()
            obs.partner_prev, obs.partner_tick = obs.partner_tick, _SpeedBook(obs.m.nbody, obs.body_name)
            obs.tt_prev, obs.tt_tick = obs.tt_tick, set()
            obs._bw_parts, obs._bw_cache, obs.body_cwork_tick = [], None, defaultdict(float)
            obs.tie_work_tick = defaultdict(float)
            obs.ext_prev, obs.ext_tick = obs.ext_tick, _SpeedBook(obs.m.nbody, obs.body_name)
            original_advance()
            try:
                obs._after_tick()
            except Exception:
                obs._note_error()
        self._original_advance = original_advance
        w.advance = advance

    def _note_error(self):
        """Record an exception raised by the observer's own code (reported as observer_error, never as a world fault)."""
        import traceback
        self.observer_errors += 1
        if self.observer_error is None:
            self.observer_error = traceback.format_exc()[-1500:]

    def uninstall(self):
        self.mujoco.mj_step = self._original_step
        self.w.advance = self._original_advance

    # ---------------------------------------------------------------- per substep
    def _discrete_drives(self, qacc):
        """MuJoCo 3.14's discrete integrator applies the drives and the dof damping implicitly, in its effective metric
        M + H (qDeriv stays zero). Over a step it applies f(v0) + efm_ca - (h D) qacc - sum_a efm_as[a] m_a (m_a . qacc):
        efm_ca is the actuators' position-stiffness shift h gp m (m . v0); efm_diag = h D + h^2 K with efm_ck = h k;
        efm_as = h^2 gp + h gv on the actuator's moment m_a. The stiffness terms of joint springs and flexes (efm_c,
        h^2 K, the flex's K) stay out of W: those forces are conservative and their potential is in E. Exact against the
        step's own velocity change, M(q0) qacc = qfrc_smooth + qfrc_constraint + efm_c + efm_ca - H qacc (2e-9 N)."""
        d, h = self.d, self.dt_sub
        if not int(d.efm_active):
            return 0.
        if int(d.nefmT) or int(d.nefmcon) or self.m.opt.density or self.m.opt.viscosity:
            raise NotImplementedError("stress books: tendon, contact or fluid terms in the discrete metric are not booked")
        corr = np.asarray(d.efm_ca) - (np.asarray(d.efm_diag) - h * np.asarray(d.efm_ck)) * qacc
        for j in range(int(d.nefmA)):
            i = int(d.efm_aid[j])
            adr, nnz = int(d.moment_rowadr[i]), int(d.moment_rownnz[i])
            cols, vals = np.asarray(d.moment_colind[adr:adr + nnz]), np.asarray(d.actuator_moment[adr:adr + nnz])
            corr[cols] -= float(d.efm_as[j]) * vals * float(vals @ qacc[cols])
        return corr

    def _flex_vertex_dofs(self):
        """The dofs of the flexes' vertex bodies (each a point mass on three slides, as cloth.py builds them)."""
        m, slide = self.m, int(self.mujoco.mjtJoint.mjJNT_SLIDE)
        dofs = []
        for b in np.unique(np.asarray(m.flex_vertbodyid)).tolist():
            if b <= 0:
                continue
            j = int(m.body_jntadr[b])
            if int(m.body_jntnum[b]) != 3 or any(int(m.jnt_type[j + k]) != slide for k in range(3)):
                raise NotImplementedError("stress books: a flex vertex body that is not a point mass on three slides")
            dofs.extend(range(int(m.body_dofadr[b]), int(m.body_dofadr[b]) + 3))
        return np.asarray(dofs, dtype=np.int64)

    def _flex_loss(self, qacc, va):
        """The discrete integrator's own loss on flex elasticity. Per step: the work of the elastic force the step applied
        to the flex vertices plus the change of the flexes' elastic potential (flex_energy at the step's end, on a private
        MjData). The applied elastic force is the explicit one (qfrc_spring) with its effective-metric terms: the shift
        efm_c (h K v0) and -K qacc (efm_K); it holds no contact, constraint or drive force (exact against the step's own
        velocity change: 1e-11 N with converged solves). Booked as a loss only (the running minimum of L enters X); a rise
        of L above its minimum would be numerical creation and stays in X."""
        mujoco, m, d, d3, k = self.mujoco, self.m, self.d, self._d3, self._flex_dofs
        from topogym.world.mujoco.energy import flex_energy
        n, nnz = np.asarray(d.efm_K_rownnz), int(d.nefmK)
        rows = np.repeat(np.arange(m.nv), n)
        cols = np.asarray(d.efm_K_colind)[:nnz]
        if len(rows) != nnz or not np.array_equal(np.asarray(d.efm_K_rowadr), np.cumsum(n) - n):
            raise NotImplementedError("stress books: a non-contiguous effective-stiffness matrix")
        Kq = np.bincount(rows, weights=np.asarray(d.efm_K_val)[:nnz] * qacc[cols], minlength=m.nv)
        applied = (np.asarray(d.qfrc_spring) + np.asarray(d.efm_c) - Kq)[k]
        d3.qpos[:] = d.qpos
        mujoco.mj_kinematics(m, d3)
        mujoco.mj_flex(m, d3)
        U = sum(flex_energy(m, d3))
        if self._U_end is not None:
            self.L += float(va[k] @ applied) * self.dt_sub + (U - self._U_end)
            self.L_min = min(self.L_min, self.L)
        self._U_end = U

    def _after_substep(self):
        m, d = self.m, self.d
        self.nsub += 1
        qpos, qvel, qacc = np.asarray(d.qpos), np.asarray(d.qvel), np.asarray(d.qacc)
        if self.nonfinite is None and not (np.isfinite(qpos).all() and np.isfinite(qvel).all() and np.isfinite(qacc).all()):
            self.nonfinite = dict(tick=int(self.w.physics_step) + 1, substep=self.nsub)
            self._event("nonfinite", self.nonfinite)
        warn = self._warnings()
        if (warn != self._warn0).any():
            import mujoco
            names = [n for n in dir(mujoco.mjtWarning) if n.startswith("mjWARN")]
            for i in np.flatnonzero(warn != self._warn0):
                name = next((n for n in names if int(getattr(mujoco.mjtWarning, n)) == int(i)), str(i))
                self.warn_new[name] = self.warn_new.get(name, 0) + int(warn[i] - self._warn0[i])
                self._event("warning", dict(tick=int(self.w.physics_step) + 1, substep=self.nsub, warning=name))
                if name == "mjWARN_BADQACC" and self.post_reset is None:
                    # v2.9 (e): MuJoCo resets the state to qpos0 on a bad acceleration, so this substep's state
                    # and everything after it describe the reset (overlapping arms, deep robot contacts), not the event:
                    # stop the peaks and events here; the warning itself fails the run
                    self.post_reset = dict(tick=int(self.w.physics_step) + 1, substep=self.nsub)
            self._warn0 = warn
        niter = int(np.max(np.asarray(d.solver_niter)[:max(1, int(d.nisland))]))
        self.max_niter = max(self.max_niter, niter)
        if niter >= self.iter_cap:
            self.solver_capped += 1
        # the work of every non-conservative smooth force over this substep (drives, gravity compensation, damping,
        # fluid, applied) as implicitfast applies it: f(v0) + qDeriv (v1 - v0), or as discrete applies it (its effective
        # metric, _discrete_drives); and the constraint forces' work
        if self.qvel_before is not None and np.isfinite(qvel).all():
            v0, v1 = self.qvel_before, qvel
            dv = v1 - v0
            f = (np.asarray(d.qfrc_actuator) + np.asarray(d.qfrc_gravcomp) + np.asarray(d.qfrc_damper)
                 + np.asarray(d.qfrc_fluid) + np.asarray(d.qfrc_applied))
            if self.ropes:                          # the rods' elastic couples are internal: their potential is in E
                from topogym.world.mujoco.rope_chain import rope_elastic_qfrc
                f = f - rope_elastic_qfrc(self.w)
            f = f + np.bincount(self._D_rows, weights=np.asarray(d.qDeriv) * dv[self._D_cols], minlength=m.nv)
            if self.vbd:                            # the VBD pins' reactions on the hands (body wrenches at the COM)
                f = f + self._xfrc_qfrc()
            drives = self._discrete_drives(dv / self.dt_sub) if self.discrete else 0.
            f = f + drives
            va = 0.5 * (v0 + v1)
            self.W += float(f @ va) * self.dt_sub
            self.Wc += float(np.asarray(d.qfrc_constraint) @ va) * self.dt_sub
            if CONSTRAINT_BOOKS and int(d.nefc):
                self._constraint_books(va)
            if self._flex_dofs is not None:
                self._flex_loss(dv / self.dt_sub, va)
            if WORK_CLAUSE and self.free_dofs:
                qc = np.asarray(d.qfrc_constraint)
                for b, dofs in self.free_dofs.items():
                    self.body_cwork_tick[b] += float(qc[dofs] @ va[dofs]) * self.dt_sub
            if TIES_BOOKED and self.ent_dofs:
                self._tie_work(va)
        self._finish_pending()
        # task-link speeds (start of this substep's state)
        if len(self.task):
            lin, ang = self._com_vel(self.task)
            sp, an = np.linalg.norm(lin, axis=1), np.linalg.norm(ang, axis=1)
            i, j = int(np.argmax(sp)), int(np.argmax(an))
            self._peak("task_speed", sp[i], self.body_name[self.task[i]])
            self._peak("task_angular", an[j], self.body_name[self.task[j]])
            self._tip_peaks(an)
        if len(self.robot):
            lin, _ = self._com_vel(self.robot)
            sp = np.linalg.norm(lin, axis=1)
            i = int(np.argmax(sp))
            self._peak("robot_speed", sp[i], self.body_name[self.robot[i]])
        n = int(d.ncon)
        if not n:
            return
        con = d.contact
        keep = np.flatnonzero(np.asarray(con.exclude[:n]) == 0)
        if not len(keep):
            return
        b1, b2 = self._contact_bodies(keep)
        pen = -np.asarray(con.dist[:n])[keep]
        adr = np.asarray(con.efc_address[:n])[keep]
        dim = np.asarray(con.dim[:n])[keep]
        frame = np.asarray(con.frame[:n]).reshape(-1, 3, 3)[keep]
        pos = np.asarray(con.pos[:n])[keep]
        efc = np.asarray(d.efc_force)
        fl = np.zeros((len(keep), 3))
        act = adr >= 0
        if self.elliptic:
            for k in range(3):
                ok = act & (dim > k)
                fl[ok, k] = efc[adr[ok] + k]
        else:
            mu = np.asarray(con.friction[:n])[keep]
            for dm in np.unique(dim[act]):
                rows = np.flatnonzero(act & (dim == dm))
                if dm == 1:
                    fl[rows, 0] = efc[adr[rows]]
                    continue
                edges = efc[adr[rows][:, None] + np.arange(2 * (dm - 1))[None, :]]
                fl[rows, 0] = edges.sum(axis=1)
                for k in range(min(2, dm - 1)):
                    fl[rows, k + 1] = (edges[:, 2 * k] - edges[:, 2 * k + 1]) * mu[rows, k]
        fw = np.einsum("nij,ni->nj", frame, fl)
        fn = np.linalg.norm(fw, axis=1)
        c1, c2 = self.cat[b1], self.cat[b2]
        moving = (c1 > 0) | (c2 > 0)
        if moving.any():
            k = np.flatnonzero(moving & (fn > 0))
            if len(k):
                if self._d4 is not None and self.qpos_before is not None and self.qvel_before is not None:
                    self._book_contacts_integrator(b1[k], b2[k], pos[k], fw[k], qvel)
                else:
                    v1s, v2s = self._point_vel(b1[k], pos[k]), self._point_vel(b2[k], pos[k])
                    self.pending = (b1[k], b2[k], pos[k], fw[k], v2s - v1s, v1s, v2s)
        lo, hi = np.minimum(c1, c2), np.maximum(c1, c2)
        cat = np.where(hi == 2, np.where(lo == 2, 1, np.where(lo == 1, 0, 2)), np.where(hi == 1, np.where(lo == 1, 4, 3), -1))
        for ci, name in enumerate(CATS):
            k = np.flatnonzero(cat == ci)
            if len(k):
                a, b = int(k[np.argmax(fn[k])]), int(k[np.argmax(pen[k])])
                pair = lambda i: (self.body_name[b1[i]], self.body_name[b2[i]])
                self._peak(f"force_{name}", fn[a], pair(a))
                self._peak(f"depth_{name}", pen[b], pair(b))
                if pen[b] > TRIGGERS["depth"]:
                    self._event("deep_task_contact" if ci <= 2 else "deep_robot_contact", dict(tick=int(self.w.physics_step) + 1, substep=self.nsub,
                                                          depth_mm=round(1000 * float(pen[b]), 3), pair=pair(b),
                                                          force_n=round(float(fn[b]), 1)))
        if self.finger.any():
            self._finger_contacts(b1, b2, fn, pen, adr, dim, act)
        strong = np.flatnonzero(act & (fn > DEGENERATE_F_N))
        if len(strong) and self._has_diagA:
            amin = self._rows_min_A(strong, adr, dim)
            for i, a_ii in zip(strong[amin < DEGENERATE_A].tolist(), amin[amin < DEGENERATE_A].tolist()):
                where = (self.body_name[b1[i]], self.body_name[b2[i]])
                self._peak("degenerate_force", fn[i], where)
                self._event("degenerate_row", dict(tick=int(self.w.physics_step) + 1, substep=self.nsub, pair=where,
                                                   force_n=round(float(fn[i]), 1), depth_mm=round(1000 * float(pen[i]), 3),
                                                   A_ii=float(a_ii)))
        # partner point speeds for task bodies (this tick)
        t1, t2 = self.cat[b1] == 2, self.cat[b2] == 2
        rows = np.flatnonzero(t1 | t2)
        if len(rows):
            v1 = self._point_vel(b1[rows], pos[rows])
            v2 = self._point_vel(b2[rows], pos[rows])
            s1, s2 = np.linalg.norm(v1, axis=1), np.linalg.norm(v2, axis=1)
            tick = int(self.w.physics_step) + 1
            body = np.stack([b1[rows], b2[rows]], axis=1).reshape(-1)     # per contact: side 1 then side 2
            other = np.stack([b2[rows], b1[rows]], axis=1).reshape(-1)
            spd = np.stack([s2, s1], axis=1).reshape(-1)
            # a link touching its own entity (cable or cloth self-contact) is not an outside cause
            ok = (self.cat[body] == 2) & (self.entity_of[body] != self.entity_of[other])
            body, other, spd = body[ok], other[ok], spd[ok]
            if len(body):
                tt = self.cat[other] == 2
                for a_, b_ in zip(body[tt].tolist(), other[tt].tolist()):
                    self.tt_tick.add((min(a_, b_), max(a_, b_)))
                self.ext_tick.update(body[~tt], other[~tt], spd[~tt])
                self.partner_tick.update(body, other, spd)
                if self.articulated:
                    ents = self.entity_of[body]
                    hit = np.isin(ents, list(self.articulated)) & (self.cat[other] != 0)
                    for e_, sp_ in zip(ents[hit].tolist(), spd[hit].tolist()):
                        self.ext_hits.append((tick, e_, sp_))

    def _ties(self):
        """Active equalities that tie an articulated task entity to another body: [(equality, type, entity)]; a type whose
        work is not booked gives entity None (v2.6)."""
        m, d, mj = self.m, self.d, self.mujoco
        out = []
        for i in np.flatnonzero(np.asarray(d.eq_active)):
            t, o1, o2 = int(m.eq_type[i]), int(m.eq_obj1id[i]), int(m.eq_obj2id[i])
            if t in (int(mj.mjtEq.mjEQ_CONNECT), int(mj.mjtEq.mjEQ_WELD)):
                if int(m.eq_objtype[i]) == int(mj.mjtObj.mjOBJ_SITE):
                    o1, o2 = int(m.site_bodyid[o1]), int(m.site_bodyid[o2])
            elif t == int(mj.mjtEq.mjEQ_JOINT):
                o1 = int(m.jnt_bodyid[o1])
                o2 = int(m.jnt_bodyid[o2]) if o2 >= 0 else 0
            else:
                out.append((int(i), t, None))
                continue
            e1, e2 = int(self.entity_of[o1]), int(self.entity_of[o2])
            if e1 != e2:
                for e in (e1, e2):
                    if e in self.ent_dofs:
                        out.append((int(i), t, e))
        return out

    def _tie_bodies(self, i):
        """The two bodies an equality ties (connect / weld by body or site; joint: the joints' bodies)."""
        m, mj = self.m, self.mujoco
        t, o1, o2 = int(m.eq_type[i]), int(m.eq_obj1id[i]), int(m.eq_obj2id[i])
        if t in (int(mj.mjtEq.mjEQ_CONNECT), int(mj.mjtEq.mjEQ_WELD)):
            if int(m.eq_objtype[i]) == int(mj.mjtObj.mjOBJ_SITE):
                return {int(m.site_bodyid[o1]), int(m.site_bodyid[o2])}
            return {o1, o2}
        return {int(m.jnt_bodyid[o1]), int(m.jnt_bodyid[o2]) if o2 >= 0 else 0}

    def _constraint_books(self, va):
        """Criteria v2.9: this substep's work of the equality and limit rows, f . (J va) dt, booked per
        equality and per limited joint / tendon over its active episode (a new episode after 25 ms inactive), as the
        contact pairs are. The largest episode net is the local rule's culprit signal next to the contact pairs'
        (constraint_net_work_j): a closed loop (a four-bar's connects) or a limit can then name the gain too."""
        m, d, mj = self.m, self.d, self.mujoco
        n = int(d.nefc)
        ty = np.asarray(d.efc_type[:n])
        sel = np.flatnonzero(np.isin(ty, self._cb_types))
        if not len(sel):
            return
        jv = np.zeros(n)
        mj.mj_mulJacVec(m, d, jv, va)
        w = np.asarray(d.efc_force[:n])[sel] * jv[sel] * self.dt_sub
        acc = defaultdict(float)
        for t_, i_, w_ in zip(ty[sel].tolist(), np.asarray(d.efc_id[:n])[sel].tolist(), w.tolist()):
            acc[(t_, i_)] += w_
        best = None
        for key, wk in acc.items():
            rec = self._cbooks.get(key)
            if rec is None or self.nsub - rec[2] > 40:
                rec = [0., 0., self.nsub]
            rec[0] += wk
            rec[1] = min(rec[1], rec[0])
            rec[2] = self.nsub
            self._cbooks[key] = rec
            if best is None or rec[0] > best[1]:
                best = (key, rec[0])
        if best is not None:
            (t_, i_), net = best
            if t_ == self._eq_type:
                name = "eq:" + (mj.mj_id2name(m, mj.mjtObj.mjOBJ_EQUALITY, i_) or str(i_))
            elif t_ == int(mj.mjtConstraint.mjCNSTR_LIMIT_JOINT):
                name = "limit:" + (mj.mj_id2name(m, mj.mjtObj.mjOBJ_JOINT, i_) or str(i_))
            else:
                name = "limit:" + (mj.mj_id2name(m, mj.mjtObj.mjOBJ_TENDON, i_) or str(i_))
            self._peak("constraint_net_work_j", net, name)

    def _spring_energy(self, n):
        """Criteria v2.9: the soft constraints' stored energy from their own springs, 1/2 k r^2 per row with
        k = K imp / (A + R) (the stiffness part of MuJoCo's reference acceleration, -K imp r, over the row's effective
        inverse mass A + R) and r = pos - margin, counted only when violated for contacts and limits. Unlike
        -1/2 f.pos it leaves out the damping and impulsive part of f, so it does not overshoot at an impact."""
        if not n:
            return 0.
        d = self.d
        kbip = np.asarray(d.efc_KBIP[:n])
        den = np.asarray(d.efc_diagA[:n]) + np.asarray(d.efc_R[:n])
        r = np.asarray(d.efc_pos[:n]) - np.asarray(d.efc_margin[:n])
        one_sided = np.asarray(d.efc_type[:n]) != self._eq_type
        r = np.where(one_sided & (r > 0), 0., r)
        ok = den > 0
        return float(np.sum(0.5 * kbip[ok, 0] * kbip[ok, 2] * r[ok] ** 2 / den[ok]))

    def _tie_work(self, va):
        """Each tying equality's work on its entity's dofs this substep (J_i^T f_i . v), and its total work on both sides
        over its active episode: the equality's own net, which must stay passive for its work to count (v2.6)."""
        m, d, mj = self.m, self.d, self.mujoco
        ties = self._ties()
        active = {i for i, _, _ in ties}
        for i in list(self.eq_net):
            if i not in active:
                del self.eq_net[i]                      # released: a new activation is a new episode
        if not ties:
            return
        etype, eid, f = np.asarray(d.efc_type), np.asarray(d.efc_id), np.asarray(d.efc_force)
        res = np.zeros(m.nv)
        for i in active:
            rows = np.flatnonzero((etype == int(mj.mjtConstraint.mjCNSTR_EQUALITY)) & (eid == i))
            if not len(rows):
                continue
            vec = np.zeros(int(d.nefc))
            vec[rows] = f[rows]
            mj.mj_mulJacTVec(m, d, res, vec)
            self.eq_net[i] = self.eq_net.get(i, 0.) + float(res @ va) * self.dt_sub
            for j, t, e in ties:
                if j == i and e is not None:
                    dofs = self.ent_dofs[e]
                    self.tie_work_tick[(e, i)] += float(res[dofs] @ va[dofs]) * self.dt_sub

    def _entity_potential(self, e):
        """An articulated entity's exact stored potentials at this tick's end: its joint springs and, for a rope chain,
        the rod's elastic energy (RopeChain.joint_energy, where the tree has it), on the private MjData (v2.6)."""
        m, d2, mj = self.m, self.d2, self.mujoco
        u = 0.
        for j, t, qa, k in self.ent_springs.get(e, []):
            if t in (int(mj.mjtJoint.mjJNT_HINGE), int(mj.mjtJoint.mjJNT_SLIDE)):
                u += .5 * k * float(d2.qpos[qa] - m.qpos_spring[qa]) ** 2
            elif t == int(mj.mjtJoint.mjJNT_BALL):
                dif = np.zeros(3)
                mj.mju_subQuat(dif, np.asarray(d2.qpos[qa:qa + 4]), np.asarray(m.qpos_spring[qa:qa + 4]))
                u += .5 * k * float(dif @ dif)
        for r in (getattr(self.w, "_ropes", None) or {}).values():
            if hasattr(r, "joint_energy") and len(r.bodies) and int(self.entity_of[int(r.bodies[0])]) == e:
                u += float(np.sum(r.joint_energy(d2)))
        if e in self.flex_entity.values() and len(self.flex_entity) == 1 and int(m.nflex) == 1:
            from topogym.world.mujoco.energy import flex_energy   # the cloth's membrane + bending energy (exact)
            mj.mj_flex(m, d2)
            u += float(sum(flex_energy(m, d2)))
        return u

    def _attached_entities(self, welds):
        """Entities tied to another entity, the robot or a static body by an active equality (a grasp weld, a connect,
        a joint coupling) in this tick. Its work is not in the contact books, so the redistribution rule (v2.5) does not
        apply to them. Computed once per tick."""
        tick = int(self.w.physics_step)
        if getattr(self, "_att_tick", None) == tick:
            return self._att
        m, d, mj = self.m, self.d, self.mujoco
        att = {int(self.entity_of[b]) for b in welds}
        for i in np.flatnonzero(np.asarray(d.eq_active)):
            t, o1, o2 = int(m.eq_type[i]), int(m.eq_obj1id[i]), int(m.eq_obj2id[i])
            if t in (int(mj.mjtEq.mjEQ_CONNECT), int(mj.mjtEq.mjEQ_WELD)):
                if int(m.eq_objtype[i]) == int(mj.mjtObj.mjOBJ_SITE):
                    o1, o2 = int(m.site_bodyid[o1]), int(m.site_bodyid[o2])
            elif t == int(mj.mjtEq.mjEQ_JOINT):
                o1 = int(m.jnt_bodyid[o1])
                o2 = int(m.jnt_bodyid[o2]) if o2 >= 0 else 0
            else:
                continue
            e1, e2 = int(self.entity_of[o1]), int(self.entity_of[o2])
            if e1 != e2:
                att |= {e1, e2}
        self._att_tick, self._att = tick, att
        return att

    def _body_work(self):
        """This tick's contact work on each task body by each partner, {(task body, partner): J}, in first-contact order
        (summed in contact order, as a sequential pass would)."""
        if self._bw_cache is None:
            if not self._bw_parts:
                self._bw_cache = {}
            else:
                keys = np.concatenate([k for k, _ in self._bw_parts])
                vals = np.concatenate([v for _, v in self._bw_parts])
                uk, first, inv = np.unique(keys, return_index=True, return_inverse=True)
                sums = np.bincount(inv.reshape(-1), weights=vals, minlength=len(uk))
                nb = self.m.nbody
                self._bw_cache = {(int(uk[j] // nb), int(uk[j] % nb)): float(sums[j]) for j in np.argsort(first, kind="stable")}
        return self._bw_cache

    def _pairs_reset(self):
        self._pk = np.zeros(0, np.int64)            # sorted pair keys (body a * nbody + body b, a < b)
        self._pnet = np.zeros(0)                    # the contact episode's cumulative work
        self._pmin = np.zeros(0)                    # its running minimum
        self._plast = np.zeros(0, np.int64)         # the last substep it was in contact

    def _pair_rec(self, a, b):
        """[net work over the contact episode, its running minimum, last substep] of a body pair, or None."""
        key = min(a, b) * self.m.nbody + max(a, b)
        i = int(np.searchsorted(self._pk, key))
        if i < len(self._pk) and self._pk[i] == key:
            return [float(self._pnet[i]), float(self._pmin[i]), int(self._plast[i])]
        return None

    def _pairs_add(self, keys, w):
        """One substep's contact work per body pair: a new contact episode after 25 ms apart; the peaks of the episode
        net and of its rise above the running minimum, in first-contact order (as a sequential pass would)."""
        if not len(keys):
            return
        uk, first, inv = np.unique(keys, return_index=True, return_inverse=True)
        acc = np.bincount(inv.reshape(-1), weights=w, minlength=len(uk))
        order = np.argsort(first, kind="stable")
        uk, acc = uk[order], acc[order]
        pos = np.searchsorted(self._pk, uk)
        found = pos < len(self._pk)
        found[found] = self._pk[pos[found]] == uk[found]
        base_net, base_min = np.zeros(len(uk)), np.zeros(len(uk))
        live = found.copy()
        live[found] = self.nsub - self._plast[pos[found]] <= 40
        base_net[live] = self._pnet[pos[live]]
        base_min[live] = self._pmin[pos[live]]
        net = base_net + acc
        mn = np.minimum(base_min, net)
        self._pnet[pos[found]] = net[found]
        self._pmin[pos[found]] = mn[found]
        self._plast[pos[found]] = self.nsub
        new = ~found
        if new.any():
            ins = np.searchsorted(self._pk, uk[new])
            self._pk = np.insert(self._pk, ins, uk[new])
            self._pnet = np.insert(self._pnet, ins, net[new])
            self._pmin = np.insert(self._pmin, ins, mn[new])
            self._plast = np.insert(self._plast, ins, np.full(int(new.sum()), self.nsub, np.int64))
        nb = self.m.nbody
        j = int(np.argmax(net))                     # the first maximum in contact order, as a sequential pass keeps it
        names = (self.body_name[int(uk[j] // nb)], self.body_name[int(uk[j] % nb)])
        # net work of this contact episode: a passive contact never gives back more than it took (F5)
        self._peak("contact_net_work_j", net[j], names)
        rise = net - mn
        j = int(np.argmax(rise))
        names = (self.body_name[int(uk[j] // nb)], self.body_name[int(uk[j] % nb)])
        # energy given back after the episode's minimum (release of stored energy: forensic only)
        self._peak("contact_work_rise_j", rise[j], names)

    def _book_contacts_integrator(self, b1, b2, pos, fw, qvel):
        """v2.13: this step's contact work as the integrator does it, booked at once: the point velocities at the step's
        start kinematics (qpos_before) and mean generalized velocity ((qvel_before + qvel) / 2), on a private MjData."""
        mujoco, m, d4 = self.mujoco, self.m, self._d4
        d4.qpos[:] = self.qpos_before
        d4.qvel[:] = 0.5 * (self.qvel_before + qvel)
        mujoco.mj_kinematics(m, d4)
        mujoco.mj_comPos(m, d4)
        mujoco.mj_comVel(m, d4)
        cv = np.asarray(d4.cvel)
        sub = np.asarray(d4.subtree_com)

        def pv(b):
            return cv[b, 3:] + np.cross(cv[b, :3], pos - sub[self.rootid[b]])
        v1a, v2a = pv(b1), pv(b2)
        if WORK_CLAUSE:
            w2 = np.einsum("ij,ij->i", fw, v2a) * self.dt_sub
            w1 = -np.einsum("ij,ij->i", fw, v1a) * self.dt_sub
            body = np.stack([b1, b2], axis=1).reshape(-1)
            other = np.stack([b2, b1], axis=1).reshape(-1)
            wk = np.stack([w1, w2], axis=1).reshape(-1)
            t = self.cat[body] == 2
            if t.any():
                self._bw_parts.append((body[t] * self.m.nbody + other[t], wk[t]))
                self._bw_cache = None
        p = np.einsum("ij,ij->i", fw, v2a - v1a) * self.dt_sub
        lo_, hi_ = np.minimum(b1, b2).astype(np.int64), np.maximum(b1, b2).astype(np.int64)
        self._pairs_add(lo_ * self.m.nbody + hi_, p)

    def _finish_pending(self):
        """The previous substep's contact work, with the relative point velocity averaged over its step (start: stored;
        end: now). Per body pair: the cumulative work, and its largest rise above the running minimum (energy the
        contact gave back; a passive contact gives back at most what it stored)."""
        if self.pending is None:
            return
        b1, b2, pos, fw, vr0, v1s, v2s = self.pending
        self.pending = None
        v1e, v2e = self._point_vel(b1, pos), self._point_vel(b2, pos)
        vr1 = v2e - v1e
        if WORK_CLAUSE:                             # the work each contact did on each of its task bodies
            w2 = np.einsum("ij,ij->i", fw, .5 * (v2s + v2e)) * self.dt_sub
            w1 = -np.einsum("ij,ij->i", fw, .5 * (v1s + v1e)) * self.dt_sub
            body = np.stack([b1, b2], axis=1).reshape(-1)          # per contact: side 1 then side 2
            other = np.stack([b2, b1], axis=1).reshape(-1)
            wk = np.stack([w1, w2], axis=1).reshape(-1)
            t = self.cat[body] == 2
            if t.any():
                self._bw_parts.append((body[t] * self.m.nbody + other[t], wk[t]))
                self._bw_cache = None
        p = np.einsum("ij,ij->i", fw, 0.5 * (vr0 + vr1)) * self.dt_sub
        lo_, hi_ = np.minimum(b1, b2).astype(np.int64), np.maximum(b1, b2).astype(np.int64)
        self._pairs_add(lo_ * self.m.nbody + hi_, p)

    def _xfrc_qfrc(self):
        """xfrc_applied as a generalized force (each body's wrench at its centre of mass), at this step's kinematics."""
        mujoco, m, d = self.mujoco, self.m, self.d
        X = np.asarray(d.xfrc_applied)
        out = np.zeros(m.nv)
        for b in np.flatnonzero(np.any(X != 0, axis=1)):
            q = np.zeros(m.nv)
            mujoco.mj_applyFT(m, d, X[b, :3].copy(), X[b, 3:].copy(), np.asarray(d.xipos[b]).copy(), int(b), q)
            out += q
        return out

    # ---------------------------------------------------------------- per tick
    def _energy(self):
        """Kinetic + potential energy of the tick's end state, on a private MjData (exact mass matrix of this state). The
        potential includes the 2D flexes' membrane and bending energy, which mj_energyPos leaves out (flex_energy), and the
        rope chains' rod elastic energy, 1/2 theta^T K theta, which it leaves out too (rope_energy)."""
        mujoco, m, d2 = self.mujoco, self.m, self.d2
        d2.qpos[:] = self.d.qpos
        d2.qvel[:] = self.d.qvel
        mujoco.mj_kinematics(m, d2)
        mujoco.mj_comPos(m, d2)
        mujoco.mj_crb(m, d2)
        mujoco.mj_energyPos(m, d2)
        mujoco.mj_mulM(m, d2, self._Mv, d2.qvel)
        energy = float(d2.energy[0]) + 0.5 * float(np.asarray(d2.qvel) @ self._Mv)
        if m.nflex:
            from topogym.world.mujoco.energy import flex_energy
            mujoco.mj_flex(m, d2)
            energy += sum(flex_energy(m, d2))
        if self.ropes:
            from topogym.world.mujoco.rope_chain import rope_energy, rope_link_energy
            energy += rope_energy(self.w, d2)
            self.rope_link_energy = rope_link_energy(self.w, d2)
        return energy

    def _after_tick(self):
        w, d = self.w, self.d
        self.ticks += 1
        tick = int(w.physics_step)
        self._finish_pending()                      # end-of-tick kinematics are the last substep's end state
        E = self._energy()
        if self.E0 is None:
            self.E0, self.W, self.Wc = E, 0., 0.
            self.L = self.L_min = 0.
        excess = (E - self.E0) - self.W - self.L_min
        if self.vbd:                                # the VBD cloths' own balances (their energy is not in E)
            excess += sum(book.X for book in self.vbd.values())
            self._peak("vbd_cloth_excess_rise_j", max(book.X_max_rise for book in self.vbd.values()))
            if any(getattr(book.cloth, "coupling", None) is not None for book in self.vbd.values()):
                self._peak("vbd_coupling_rise_j", max(book.C_max_rise for book in self.vbd.values()))
        if self._flex_dofs is not None:
            self._peak("integrator_flex_loss_j", -self.L_min)
            self._peak("integrator_flex_gain_j", self.L - self.L_min)
        self.excess_hist.append(excess)
        self._peak("energy_excess_j", excess)
        # passivity with the soft constraints' stored energy: Y = E + S - W never rises (S ~ sum of max(0, -f.pos / 2)
        # over the active constraint rows: contact penetration springs, weld / equality and limit violations)
        n = int(d.nefc)
        e_rows = np.maximum(0., -.5 * np.asarray(d.efc_force[:n]) * np.asarray(d.efc_pos[:n])) if n else np.zeros(0)
        S = float(e_rows.sum())
        Y = excess + S
        self.Y_min = Y if self.Y_min is None else min(self.Y_min, Y)
        self._peak("energy_created_j", Y - self.Y_min)
        self._peak("stored_constraint_j", S)
        if CONSTRAINT_BOOKS:      # v2.9, forensic only: X + S with S from the springs, and with S windowed
            S2 = self._spring_energy(n)
            Y2 = excess + S2
            self.Y2_min = Y2 if self.Y2_min is None else min(self.Y2_min, Y2)
            self._peak("energy_created_spring_j", Y2 - self.Y2_min)
            self._peak("stored_spring_j", S2)
            # windowed S (v2.9): a spike of -1/2 f.pos (a limit's or contact's force chattering under an
            # impact) lasts a few ticks, stored energy lasts. The current value uses S's minimum over the last
            # S_WINDOW ticks (no upward spikes), the running floor uses its maximum (no dips while S catches up).
            self.S_hist.append(S)
            if len(self.S_hist) > S_WINDOW:
                self.S_hist.pop(0)
            Yw = excess + min(self.S_hist)
            floor = excess + max(self.S_hist)
            self.Yw_min = floor if self.Yw_min is None else min(self.Yw_min, floor)
            self._peak("energy_created_win_j", Yw - self.Yw_min)
        # windowed rise of the excess (40 ticks = 0.1 s): a sudden creation of energy
        if len(self.excess_hist) > 40:
            rise = excess - min(self.excess_hist[-41:-1])
            self._peak("energy_rise_0p1s_j", rise)
        if INTEGRATOR_BOOKS:
            # v2.13: the integrator's own energy residual, cumulated: the excess less the constraint forces' work
            # (Wc, J' f . v_mean, every contact, equality and limit row). It rises when a step creates energy that no
            # constraint row did (the mass matrix changing within a step, e.g. a light chain's landing); its rise over
            # the same 0.1 s window is a culprit signal of the local rule beside the pairs' and rows' episode nets.
            self.res_hist.append(excess - self.Wc)
            if len(self.res_hist) > 41:
                self.res_hist.pop(0)
            if len(self.res_hist) > 1:
                r_rise = self.res_hist[-1] - min(self.res_hist[:-1])
                self._peak("step_residual_rise_j", r_rise, "integrator step")
                if len(self.excess_hist) > 40:
                    # one local event: the excess's 0.1 s rise and the residual's over the same window, at this tick;
                    # their minimum passes the local threshold only if both do (energy_created(integrator))
                    self._peak("integrator_created_j", min(rise, r_rise), "integrator step")
        if not len(self.task):
            return
        lin, ang = self._com_vel(self.task)
        sp, an = np.linalg.norm(lin, axis=1), np.linalg.norm(ang, axis=1)
        i, j = int(np.argmax(sp)), int(np.argmax(an))
        self._peak("task_speed", sp[i], self.body_name[self.task[i]])
        self._peak("task_angular", an[j], self.body_name[self.task[j]])
        self._tip_peaks(an)
        # weld partners (assisted grasp); grasp losses this tick by reason (a slip above the weld force limit, a pose
        # error, fingers opened), counted for the record only
        welds = {}
        g = w._grasp_manager
        for ev in (getattr(g, "events", None) or []) if g is not None else []:
            if ev.get("event") == "lost":
                self._event(f"grasp_lost:{ev.get('reason')}", dict(tick=tick, hand=ev.get("hand")), keep=5)
        if g is not None:
            for hand, st in g.hands.items():
                b = st.get("binding")
                if b is not None and "links" in b:
                    welds[int(b["links"][1])] = int(b["links"][0])
                elif b is not None:                     # a cloth patch grasp: each pin ties a vertex body to the hand
                    for e_, _ in (b.get("pins") or []):
                        welds[int(self.m.eq_obj2id[e_])] = int(self.m.eq_obj1id[e_])
        com = np.asarray(d.xipos)[self.task]
        # kinetic energy (translation + rotation) and stored constraint energy of every task link, this tick
        R = np.asarray(d.ximat)[self.task].reshape(-1, 3, 3)
        w_local = np.einsum("nji,nj->ni", R, ang)                  # world angular velocity in the inertial frame
        inertia = np.asarray(self.m.body_inertia)[self.task]
        mass = np.maximum(np.asarray(self.m.body_mass)[self.task], 1e-9)
        e_rot = .5 * np.einsum("ni,ni->n", inertia, w_local ** 2)
        KE = .5 * mass * sp ** 2 + e_rot
        if len(self._arm_dofs):                   # v2.13: the links' own joints' armature (rotor) energy
            qv = np.asarray(d.qvel)[self._arm_dofs]
            KE = KE + np.bincount(self._arm_task, weights=.5 * self._arm_val * qv * qv, minlength=len(self.task))
        S_body = self._stored_by_task_body(e_rows)
        R_body = np.zeros(len(self.task))          # v2.10: the rod elastic part alone
        for b, ej in self.rope_link_energy.items():  # a rope link's stored energy includes its rod joints' (a released bend)
            if self.task_index[b] >= 0:
                S_body[self.task_index[b]] += ej
                R_body[self.task_index[b]] += ej
        ent_bal = {}                                # each articulated entity's energy balance this tick (v2.5), lazily
        ties_now = self._ties() if TIES_BOOKED else []
        unbooked_tie = any(e_ is None for _, _, e_ in ties_now)
        U_now = {e_: self._entity_potential(e_) for e_ in self.ent_dofs} if STORED_DROP else {}

        def own_out_ok(body, e):
            return sum(wk for (b_, other), wk in self._body_work().items()
                       if b_ == body and int(self.entity_of[other]) != e) <= 0

        def entity_books(e, ties_e):
            """The articulated entity's energy balance this tick (v2.5, with v2.6's ties and stored drop)."""
            ks = np.flatnonzero(self.entity_of[self.task] == e)
            bodies = set(int(b) for b in self.task[ks])
            g = np.asarray(self.m.opt.gravity)
            dke = float(np.sum(KE[ks] - self.KE_prev[ks]))
            wg = float(np.sum(mass[ks] * ((com[ks] - self.com_prev[ks]) @ g)))
            wo, passive_out, pos_pairs = 0., True, []
            for (b_, other), wk in self._body_work().items():
                if b_ in bodies and int(self.entity_of[other]) != e:
                    wo += wk
                    if wk > 0:
                        pr = self._pair_rec(b_, other)
                        net = pr[0] if pr else 0.
                        passive_out &= net <= PAIR_PASSIVE_J
                        pos_pairs.append((self.body_name[other], round(net, 5)))
            passive25 = passive_out                 # the contacts alone (v2.5)
            wt_, tie_rec = 0., []
            for i, t in ties_e:                     # v2.6: the ties' work, each tie passive over its episode
                wk = self.tie_work_tick.get((e, i), 0.)
                wt_ += wk
                if wk > 0:
                    passive_out &= self.eq_net.get(i, 0.) <= PAIR_PASSIVE_J
                tie_rec.append((int(i), round(wk, 6), round(self.eq_net.get(i, 0.), 5)))
            du = (U_now.get(e, 0.) - self.U_prev.get(e, U_now.get(e, 0.))) if STORED_DROP else 0.
            return dict(dke=dke, wg=wg, wo=wo + wt_, passive=passive_out, pos_pairs=pos_pairs[:4],
                        ties=tie_rec, du=du, wo25=wo, passive25=passive25)

        # v2.9 (c): every articulated entity's balance every tick, kept for CHAIN_WINDOW_TICKS ticks, so that
        # rule 6 can also be read over a short window (a one-tick wobble of a whipping chain's energy nets out)
        if CHAIN_WINDOW_TICKS and REDISTRIBUTION and self.KE_prev is not None and self.com_prev is not None:
            for e_ in self.articulated:
                ties_e_ = [(i, t) for i, t, ee in ties_now if ee == e_] if TIES_BOOKED else []
                ent_bal[e_] = entity_books(e_, ties_e_)
                h = self.ent_hist.setdefault(e_, [])
                b_ = ent_bal[e_]
                h.append((b_["dke"], b_["wg"], b_["wo"], b_["du"], b_["passive"]))
                if len(h) > CHAIN_WINDOW_TICKS:
                    h.pop(0)
        if self.v_prev is not None:
            for k in np.flatnonzero(sp - self.v_prev > JUMP_M_S):
                body = int(self.task[k])
                cands = [self.partner_tick.get(body), self.partner_prev.get(body)]
                partner = max([c for c in cands if c], default=(0., None))
                if body in welds:
                    hv = float(np.linalg.norm(self._point_vel(np.array([welds[body]]), com[k:k + 1])[0]))
                    if hv > partner[0]:
                        partner = (hv, self.body_name[welds[body]] + " (weld)")
                e = int(self.entity_of[body])
                rec = dict(tick=tick, body=self.body_name[body], before=round(float(self.v_prev[k]), 4),
                           after=round(float(sp[k]), 4), partner=partner[1], partner_speed=round(partner[0], 4))
                # the body's own rotational energy can turn into translation at a contact (a tumbling piece hitting
                # something): |v_after| <= sqrt(|v_before|^2 + 2 E_rot_before / m) + 2 * partner + 0.1
                v_rot = self.vrot_prev[k] if self.vrot_prev is not None else 0.
                # a rebound returns at most the body's own recent kinetic energy (stored in the contact it compressed):
                # its peak speed over the last REBOUND_TICKS ticks also bounds the new speed
                recent = max((float(h[k]) for h in self.v_hist), default=0.)
                own = max(math.sqrt(float(self.v_prev[k]) ** 2 + v_rot ** 2), recent)
                gdt = self.g_dt if GRAVITY_DT else 0.      # gravity's own speed change within one tick (v2.6)
                bound = own + 2 * partner[0] + .1
                if sp[k] <= bound:
                    self._event("jump_causal", rec)
                    continue
                # a chain: the external partners of the task links in contact with this one (v2.1)
                chain = self._chain_partner(body, welds, com, k)
                if chain[1] is not None and sp[k] <= own + 2 * chain[0] + .1:
                    rec.update(chain_partner=chain[1], chain_partner_speed=round(chain[0], 4))
                    self._event("jump_causal", rec)
                    continue
                # v2.6: gravity's own speed change within one tick in rules 1-2 (checked last, so that every v2.5 rule
                # keeps its own verdict and a report can still read v2.5 exactly)
                grav_ok = sp[k] <= bound + gdt or (chain[1] is not None and sp[k] <= own + 2 * chain[0] + .1 + gdt)
                if e in self.articulated:
                    lo = tick - int(round(WHIP_WINDOW_S / w.physics_dt_s))
                    hits = [h[2] for h in self.ext_hits if h[1] == e and h[0] >= lo]
                    hit = max(hits, default=0.)
                    rec.update(window_hit_speed=round(hit, 4))
                    if sp[k] <= WHIP_GAIN * (2 * hit + .1):
                        self._event("whip_causal", rec)
                        continue
                rec.update(bound=round(bound, 4))
                # stored-energy release (a pop): the kinetic energy gained this tick is at most the drop of the
                # constraint energy stored on this body (the robot's squeeze or weld being released)
                if self.KE_prev is not None and self.S_prev is not None:
                    ke_gain, s_drop = float(KE[k] - self.KE_prev[k]), float(self.S_prev[k] - S_body[k])
                    rec.update(ke_gain_j=round(ke_gain, 5), stored_before_j=round(float(self.S_prev[k]), 5),
                               stored_drop_j=round(s_drop, 5))
                    if ke_gain <= s_drop:
                        if len(self.pop_speeds) < POP_KEEP:
                            self.pop_speeds.append(round(float(sp[k]), 4))
                        self._peak("pop_speed", float(sp[k]), self.body_name[body])
                        # v2.9 (d): the guard bounds the release's kick, the vector change of the link's velocity
                        # |v_after - v_before| (for a body at rest that is its after-speed, as before); up to v2.8 it bounded
                        # the after-speed. Both are recorded so that every reading stays exact.
                        kick = (float(np.linalg.norm(lin[k] - self.lin_prev[k])) if self.lin_prev is not None
                                and len(self.lin_prev) == len(lin) else float(sp[k]))
                        rec.update(kick_m_s=round(kick, 4))
                        self._peak("pop_kick", kick, self.body_name[body])
                        fast28, fast29 = sp[k] > POP_SPEED_MAX, kick > POP_SPEED_MAX
                        if not fast28 and not fast29:
                            self._event("pop_stored_release", rec)
                        elif grav_ok and fast28:        # v2.6: gravity explains it, so it is no pop (v2.5: pop_fast)
                            self._event("jump_causal_gravity_pop", rec)
                        elif grav_ok:                   # only the kick is fast, and gravity explains it
                            self._event("pop_stored_release", rec)
                        else:
                            # v2.11: the constraint-stored part of the release into this link (contacts,
                            # equalities, limits; the rod's elastic storage excluded) and the speed it alone could give
                            rod_d = (float(self.R_prev[k] - R_body[k]) if self.R_prev is not None
                                     and len(self.R_prev) == len(R_body) else 0.)
                            con_d = max(0., s_drop - rod_d)
                            vc = math.sqrt(2. * con_d / float(mass[k]))
                            rec.update(constraint_drop_j=round(con_d, 8), vc_m_s=round(vc, 4))
                        cls_ = "fast" if fast28 and fast29 else "after" if fast28 else "kick"
                        if not fast28 and not fast29 or grav_ok:
                            pass
                        elif (ROD_RELEASE and self.R_prev is not None and len(self.R_prev) == len(R_body)
                              and ke_gain <= float(self.R_prev[k] - R_body[k]) + 1e-9):
                            # v2.10: the rod's own elastic energy pays for the whole gain (a physical whip release); the
                            # suffix keeps the class it had up to v2.9, so that every earlier reading stays exact
                            rec.update(rod_drop_j=round(float(self.R_prev[k] - R_body[k]), 7))
                            self._event("pop_rod_v210_" + cls_, rec)
                        elif ENERGY_SHARE and vc <= POP_SPEED_MAX:
                            # v2.11: the constraint-stored energy released into this link could not by itself
                            # move it at the guard's speed; the suffix keeps its class up to v2.10
                            self._event("pop_share_v211_" + cls_, rec)
                        elif fast28 and fast29:
                            self._event("pop_fast", rec)            # a breakdown: the brute-force launch guard
                        elif fast28:
                            self._event("pop_fast_v28", rec)        # fast after-speed, small kick: a breakdown up to v2.8
                        else:
                            self._event("pop_kick_v29", rec)        # small after-speed, fast kick: a breakdown from v2.9
                        continue
                if WORK_CLAUSE and self.KE_prev is not None:
                    # v2.4 "work": the robot paid for it. In this tick the static and task partners did <= 0
                    # work on the body, and its kinetic-energy gain is at most the robot partners' work (contacts +
                    # the weld: the body's constraint work beyond its contacts). F5 still judges the energy.
                    ws = wt = wr = 0.
                    robot_pairs = []
                    for (b_, other), wk in self._body_work().items():
                        if b_ != body:
                            continue
                        c_ = int(self.cat[other])
                        if c_ == 0:
                            ws += wk
                        elif c_ == 2:
                            wt += wk
                        else:
                            wr += wk
                            if wk > 0:
                                pr = self._pair_rec(body, other)
                                robot_pairs.append((self.body_name[other], round(pr[0], 5) if pr else 0.))
                    if body in self.free_dofs and body in welds:
                        wr += self.body_cwork_tick.get(body, 0.) - (ws + wt + wr)
                    ke_gain = float(KE[k] - self.KE_prev[k])
                    # the robot's contacts must be passive themselves (their pair net over the contact episode <= a
                    # numerical 0.01 J), or the "robot work" may have come from a contact that creates energy
                    passive = all(net <= PAIR_PASSIVE_J for _, net in robot_pairs)
                    rec.update(work_static_j=round(ws, 5), work_task_j=round(wt, 5), work_robot_j=round(wr, 5),
                               ke_gain_j=round(ke_gain, 5), robot_pairs_net_j=robot_pairs)
                    if ws <= 0 and wt <= 0 and ke_gain <= wr and passive:
                        self._event("jump_work_causal", rec)
                        continue
                ties_e = [(i, t) for i, t, ee in ties_now if ee == e] if TIES_BOOKED else []
                attached25 = e in self._attached_entities(welds)   # v2.5: any active tie keeps the entity out
                if not TIES_BOOKED or unbooked_tie:
                    attached = attached25
                else:                                   # v2.6: tied entities take part; a tied link itself does not
                    attached = any(body in self._tie_bodies(i) for i, _ in ties_e)
                    if not ties_e and any(int(self.entity_of[b]) == e for b in welds):
                        attached = True                 # a grasp binding that is not a MuJoCo equality: not booked
                if (REDISTRIBUTION and e in self.articulated and self.KE_prev is not None and self.com_prev is not None
                        and not attached):
                    # v2.5 "redistribution": the speed came from the chain itself. In this tick the link took no
                    # positive work from outside its entity, the entity's kinetic energy rose by at most the outside work
                    # on it (contacts with other entities, the robot and statics) plus gravity's, and every outside
                    # contact that did positive work is passive (pair net over its contact episode <= PAIR_PASSIVE_J)
                    eb = ent_bal.get(e)
                    if eb is None:
                        eb = ent_bal[e] = entity_books(e, ties_e)
                    own_out = sum(wk for (b_, other), wk in self._body_work().items()
                                  if b_ == body and int(self.entity_of[other]) != e)
                    rec.update(entity_ke_gain_j=round(eb["dke"], 6), entity_out_work_j=round(eb["wo"], 6),
                               entity_gravity_work_j=round(eb["wg"], 6), link_out_work_j=round(own_out, 6),
                               entity_pos_pairs=eb["pos_pairs"], entity_ties=eb["ties"],
                               entity_potential_change_j=round(eb["du"], 6))
                    ok25 = (not attached25 and own_out <= 0 and eb["dke"] <= eb["wo25"] + eb["wg"] + ENTITY_EPS_J
                            and eb["passive25"])
                    ok26 = own_out <= 0 and eb["dke"] <= eb["wo"] + eb["wg"] - eb["du"] + ENTITY_EPS_J and eb["passive"]
                    if ok25:
                        self._event("whip_redistribution", rec)
                        continue
                else:
                    ok26 = False
                # v2.6 extras, after every v2.5 rule: gravity in rules 1-2, then rule 6 with the ties and stored potentials
                if grav_ok:
                    self._event("jump_causal_gravity", rec)
                    continue
                if ok26:
                    self._event("whip_redistribution_v26", rec)
                    continue
                # v2.9 (c), after everything else: rule 6 over the last CHAIN_WINDOW_TICKS ticks. The chain's
                # kinetic gain over the window must stay within the outside work, gravity's work and the stored drop over
                # the same window (+ ENTITY_EPS_J), every tick's outside contacts passive. It looks back only: a creation
                # followed by dissipation is still flagged when it happens; a dissipation followed by a gain is excused only
                # when the window's net is nil.
                if (CHAIN_WINDOW_TICKS and e in self.articulated and e in self.ent_hist and own_out_ok(body, e)
                        and not attached):
                    h = self.ent_hist[e]
                    sk, sg, so, su = (sum(x[i] for x in h) for i in range(4))
                    rec.update(window_ticks=len(h), window_ke_gain_j=round(sk, 6), window_out_work_j=round(so, 6),
                               window_gravity_work_j=round(sg, 6), window_potential_change_j=round(su, 6))
                    if sk <= so + sg - su + ENTITY_EPS_J and all(x[4] for x in h):
                        self._event("whip_redistribution_v29", rec)
                        continue
                # v2.10, after everything else: rule 6 (v2.5 / v2.6 / v2.9-window forms) with the link clause's
                # numerical tolerance: the link's positive work from outside its entity up to LINK_OUT_EPS_J is contact
                # round-off (traced: a rope link on the table took 4e-9 / 2e-8 J while the chain kicked it). Its own
                # event, so that every earlier reading stays exact.
                if (LINK_OUT_EPS_J and REDISTRIBUTION and e in self.articulated and self.KE_prev is not None
                        and self.com_prev is not None and not attached):
                    own_v = sum(wk for (b_, other), wk in self._body_work().items()
                                if b_ == body and int(self.entity_of[other]) != e)
                    if 0 < own_v <= LINK_OUT_EPS_J:
                        eb = ent_bal.get(e)
                        if eb is None:
                            eb = ent_bal[e] = entity_books(e, ties_e)
                        ok_t = ((not attached25 and eb["dke"] <= eb["wo25"] + eb["wg"] + ENTITY_EPS_J and eb["passive25"])
                                or (eb["dke"] <= eb["wo"] + eb["wg"] - eb["du"] + ENTITY_EPS_J and eb["passive"]))
                        if not ok_t and CHAIN_WINDOW_TICKS and e in self.ent_hist:
                            h = self.ent_hist[e]
                            sk, sg, so, su = (sum(x[i] for x in h) for i in range(4))
                            ok_t = sk <= so + sg - su + ENTITY_EPS_J and all(x[4] for x in h)
                        if ok_t:
                            rec.update(link_out_work_exact_j=float(own_v))
                            self._event("whip_redistribution_v210", rec)
                            continue
                self._peak("uncaused_jump_excess_m_s", float(sp[k]) - bound, self.body_name[body])
                self._event("jump_uncaused", rec)
        self.v_prev = sp
        self.lin_prev = lin.copy()
        self.v_hist.append(sp.copy())
        if len(self.v_hist) > REBOUND_TICKS:
            self.v_hist.pop(0)
        # equivalent linear speed of each task link's rotational kinetic energy, sqrt(w . I w / m), for the next tick
        self.vrot_prev = np.sqrt(2 * e_rot / mass)
        self.KE_prev, self.S_prev, self.R_prev = KE, S_body, R_body
        self.com_prev = com.copy()
        if STORED_DROP:
            self.U_prev = U_now
        # prune old external hits
        if self.ext_hits and len(self.ext_hits) > 4000:
            lo = tick - int(round(WHIP_WINDOW_S / w.physics_dt_s))
            self.ext_hits = [h for h in self.ext_hits if h[0] >= lo]
        # tunnelling / workspace (COM)
        xy, z = com[:, :2], com[:, 2] - self.table_z
        inside = np.all((xy > self.table_lo + .02) & (xy < self.table_hi - .02), axis=1)
        for k in np.flatnonzero(inside & (z < -.01)):
            self._event("tunnel", dict(tick=tick, body=self.body_name[int(self.task[k])], z_mm=round(1000 * float(z[k]), 2),
                                       xy=[round(float(v), 4) for v in xy[k]]))
        out = (~np.all((xy > self.table_lo - .1) & (xy < self.table_hi + .1), axis=1)) | (z > 1.0) | (z < -.1)
        for k in np.flatnonzero(out):
            self._event("left_workspace", dict(tick=tick, body=self.body_name[int(self.task[k])],
                                               com=[round(float(v), 4) for v in com[k]]), keep=3)
        self._peak("min_task_z_over_table_mm", -1000 * float(np.min(np.where(inside, z, np.inf), initial=np.inf)))
        if self.trace is not None:
            self.trace.append(dict(tick=tick, speed=round(float(sp[i]), 5), angular=round(float(an[j]), 4),
                                   excess=round(excess, 6), E=round(E, 6), W=round(self.W, 6), Wc=round(self.Wc, 6)))

    # ---------------------------------------------------------------- summary
    def _rope_clamp(self):
        """The ropes' speed-clamp counts so far (rope_chain.rope_clamp_stats); {} without a clamped rope."""
        if not self.ropes:
            return {}
        try:                                        # a runtime from before the clamp (knot_chain <= v5) has no counts
            from topogym.world.mujoco.rope_chain import rope_clamp_stats
        except ImportError:
            return {}
        return rope_clamp_stats(self.w)

    def rope_clamp(self):
        """Per clamped rope, its speed-clamp activations since this observer started: substeps with any link above a
        threshold, link-substeps (all / by speed / by spin), the first and last tick (world ticks), the largest speed and
        spin it met, and the energy it removed (J; also in W as dissipation); and its tension guard's: link-substeps
        with a raised inertia, the largest scale and connect tension, the first tick, the kinetic energy it removed (J;
        in E, not in W)."""
        out = {}
        for name, st in self._rope_clamp().items():
            s0 = self._clamp0.get(name) or {}
            d = {k: st[k] - s0.get(k, 0) for k in ("substeps", "link_substeps", "speed_link_substeps", "spin_link_substeps")}
            d["loss_j"] = round(st["loss_j"] - s0.get("loss_j", 0.), 9)
            fresh = d["substeps"] > 0
            d.update(first_tick=st["first_tick"] if fresh else None,      # (the world's first, if it ran before)
                     last_tick=st["last_tick"] if fresh else None,
                     max_speed_m_s=round(st["max_speed_m_s"], 4) if fresh else None,
                     max_spin_rad_s=round(st["max_spin_rad_s"], 3) if fresh else None)
            if "tg_link_substeps" in st:              # the tension guard (rope_chain.RopeChain._tension_guard)
                tg = st["tg_link_substeps"] - s0.get("tg_link_substeps", 0)
                d.update(tension_guard_link_substeps=tg,
                         tension_guard_loss_j=round(st["tg_loss_j"] - s0.get("tg_loss_j", 0.), 9),
                         tension_guard_max_scale=round(st["tg_max_scale"], 4) if tg else None,
                         tension_guard_max_tension_n=round(st["tg_max_tension_n"], 3) if tg else None,
                         tension_guard_first_tick=st["tg_first_tick"] if tg else None)
            out[name] = d
        return out

    def summary(self):
        p = {k: dict(value=v[0], tick=v[1], substep=v[2], where=v[3]) for k, v in self.peaks.items()}
        return dict(ticks=self.ticks, substeps=self.nsub, peaks=p, counts=dict(self.counts), post_reset=self.post_reset,
                    rope_clamp=self.rope_clamp(),
                    vbd={name: book.summary() for name, book in self.vbd.items()},
                    events={k: v for k, v in self.events.items()}, nonfinite=self.nonfinite, warnings=self.warn_new,
                    solver_capped_substeps=self.solver_capped, max_solver_iterations=self.max_niter, pop_speeds=self.pop_speeds,
                    observer_errors=self.observer_errors, observer_error=self.observer_error,
                    iteration_cap=self.iter_cap, energy=dict(E0=self.E0, W=self.W, Wc=self.Wc,
                                                             **(dict(L=self.L, L_min=self.L_min) if self._flex_dofs is not None
                                                                else {}),
                                                             final_excess=self.excess_hist[-1] if self.excess_hist else None))
