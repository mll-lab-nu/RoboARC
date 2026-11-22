#!/usr/bin/env python
"""topogym/world/mujoco/rope_chain.py rope_energy against the work of the rope chain's own rod couples.

A 12-link rope chain (free capsules, no gravity, no contact) with the rod as RopeChain applies it: a shaped rest (each
joint's rest a random rotation of up to 0.2 rad) and different twist and bend stiffnesses per joint. Three random smooth
bends and twists of up to 0.8 rad per link are applied quasi-statically in 400 equal steps from the rest. The energy the
couples store is minus their work along that path: the trapezoid of the couples . dq, with dq from mj_differentiatePos,
the free joints' body-frame rotations. rope_energy at the end state must match it within 1e-5 relative; the measured
residual is 1.4e-8 .. 4.7e-8 at 400 steps and 16x less at 1600: the trapezoid's own O(h^2) error.
Also checked:
  - at rest the rod stores nothing;
  - at zero velocity the couples RopeChain.apply writes into qfrc_applied are exactly their elastic part
    (rope_elastic_qfrc);
  - rope_link_energy counts each joint for both its links, so it sums to twice rope_energy;
  - mujoco's data.energy books none of the rod's energy.

  python tools/tests/test_rope_energy.py
"""
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "source"))

N, L, RADIUS = 12, .0045, .004


def chain_model():
    import mujoco
    bodies = "".join(f"""
<body name="rope::{k:03d}" pos="{k * L} 0 0"><freejoint name="rope::{k:03d}::root"/>
<inertial pos="{L / 2} 0 0" mass="2e-4" diaginertia="2e-9 3e-9 3e-9"/>
<geom type="capsule" fromto="0 0 0 {L} 0 0" size="{RADIUS}"/></body>""" for k in range(N))
    return mujoco.MjModel.from_xml_string(f"""
<mujoco><option gravity="0 0 0"><flag energy="enable" contact="disable"/></option>
<worldbody>{bodies}</worldbody></mujoco>""")


def expm(v):
    from scipy.spatial.transform import Rotation
    return Rotation.from_rotvec(v).as_matrix()


def quat(R):
    from scipy.spatial.transform import Rotation
    return Rotation.from_matrix(R).as_quat()[[3, 0, 1, 2]]


def build(rng, folder):
    """The model, a world stand-in with the chain as RopeChain reads it from a built scene, and the rest orientations."""
    import mujoco
    from topogym.world.mujoco.rope_chain import RopeChain
    m = chain_model()
    d = mujoco.MjData(m)
    rest_rel = np.stack([expm(rng.uniform(-1, 1, 3) * .2 / np.sqrt(3)) for _ in range(N - 1)])
    K = np.column_stack([rng.uniform(.5, 1.5, N - 1) * 1e-3, *(2 * [rng.uniform(.5, 1.5, N - 1) * 2e-3])])
    np.savez(folder / "rope_rope.npz", s_nodes=np.arange(N + 1) * L, lengths=np.full(N, L), rest_rel=rest_rel, K=K,
             C=np.full((N - 1, 3), 1e-6), birth=np.zeros((1, 3)), elements=np.zeros((1, 4), np.int64),
             material_s=np.zeros(1), radius=RADIUS)
    world = SimpleNamespace(model=m, data=d, _ropes={})
    record = dict(rope=dict(file="rope_rope.npz"), links=[dict(body=f"rope::{k:03d}") for k in range(N)])
    world._ropes["rope"] = RopeChain(world, "rope", record, folder)
    R0 = [np.eye(3)]
    for k in range(N - 1):
        R0.append(R0[-1] @ rest_rel[k])
    return m, d, world, np.stack(R0)


def pose(m, d, R):
    for k in range(N):
        a = int(m.jnt_qposadr[k])
        d.qpos[a:a + 3] = [k * L, 0., 0.]
        d.qpos[a + 3:a + 7] = quat(R[k])


def stored_by_path(m, d, world, R0, omega, steps):
    """-(work of the elastic couples) along R_k(s) = R0_k exp(s omega_k), s: 0 -> 1 (trapezoid), at zero velocity."""
    import mujoco
    from topogym.world.mujoco.rope_chain import rope_elastic_qfrc
    chain = world._ropes["rope"]
    prev, stored = None, 0.
    dq = np.zeros(m.nv)
    for i in range(steps + 1):
        pose(m, d, np.einsum("nij,njk->nik", R0, np.stack([expm(i / steps * w) for w in omega])))
        d.qvel[:] = 0.
        d.qfrc_applied[:] = 0.
        chain.apply(d)                      # refreshes the kinematics, writes the couples
        f = np.array(d.qfrc_applied)
        assert np.array_equal(f, rope_elastic_qfrc(world)), "at rest velocity the couples are their elastic part"
        q = np.array(d.qpos)
        if prev is not None:
            mujoco.mj_differentiatePos(m, dq, 1., prev[0], q)
            stored -= .5 * float((f + prev[1]) @ dq)
        prev = (q, f)
    return stored


def main():
    import mujoco
    from topogym.world.mujoco.rope_chain import rope_energy, rope_link_energy
    rng = np.random.default_rng(3)
    with tempfile.TemporaryDirectory() as tmp:
        m, d, world, R0 = build(rng, Path(tmp))
        pose(m, d, R0)
        mujoco.mj_forward(m, d)
        assert rope_energy(world) < 1e-25, "the rest stores no energy"
        worst = 0.
        for trial in range(3):
            omega = rng.uniform(-1, 1, (N, 3)) * .8 / np.sqrt(3)
            res = {}
            for steps in (400, 1600):
                stored = stored_by_path(m, d, world, R0, omega, steps)
                mujoco.mj_forward(m, d)
                U = rope_energy(world, d)
                res[steps] = abs(U - stored) / stored
            links = rope_link_energy(world, d)
            assert abs(sum(links.values()) - 2 * U) <= 1e-12 * U, "each joint counts for both its links"
            assert float(d.energy[0]) == 0., "mujoco books none of the rod's energy (no gravity, no springs)"
            assert res[400] < 1e-5, (trial, res)
            assert res[1600] < res[400] / 8, ("the residual is the trapezoid's O(h^2) error", trial, res)
            worst = max(worst, res[400])
            print(f"trial {trial}: rod energy {U:.6e} J; relative residual {res[400]:.1e} (400 steps), "
                  f"{res[1600]:.1e} (1600)")
    print(f"PASS (worst relative residual at 400 steps: {worst:.1e})")


if __name__ == "__main__":
    main()
