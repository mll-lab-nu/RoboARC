#!/usr/bin/env python
"""topogym/world/mujoco/energy.py flex_energy against the work of MuJoCo's own flex elastic forces.

A 5x5 2D flex grid (12 mm, no gravity, no contact) for each elastic2d mode ("stretch", "bend", "both"); three smooth
deformations per mode (a random affine stretch / shear, a bend onto a cylinder arc, plus 0.2 mm noise), applied
quasi-statically in 400 equal steps from the flat rest. The energy the elastic forces store is minus their work along
that path (the trapezoid of qfrc_spring . dq); flex_energy at the end state must match it:
  membrane within 1e-5 relative (measured ~3e-7: the trapezoid's own error), bending within 1e-9 (measured ~1e-13).
Also: the elastic forces are in qfrc_spring (not qfrc_damper), mujoco's data.energy[0] does not book them, and
data.energy[1] is 1/2 v^T M v of every dof (the flex vertices included).

  python tools/tests/test_flex_energy.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "source"))


def grid(elastic2d):
    import mujoco
    return mujoco.MjModel.from_xml_string(f"""
<mujoco><option gravity="0 0 0" integrator="discrete" solver="CG"><flag energy="enable" contact="disable"/></option>
<worldbody><flexcomp name="f" type="grid" count="5 5 1" spacing=".012 .012 .012" dim="2" radius=".004" mass=".0138">
<elasticity young="3e7" poisson=".3" thickness="1.1e-4" elastic2d="{elastic2d}"/>
<contact contype="0" conaffinity="0"/></flexcomp></worldbody></mujoco>""")


def vertex_qpos(model):
    adr, n = int(model.flex_vertadr[0]), int(model.flex_vertnum[0])
    out = []
    for b in np.asarray(model.flex_vertbodyid[adr:adr + n]):
        j0, nj = int(model.body_jntadr[b]), int(model.body_jntnum[b])
        out.append([int(model.jnt_qposadr[j0 + k]) for k in range(nj)] if nj == 3 else None)
    return out


def stored_by_path(model, disp, steps=400):
    """-(work of qfrc_spring) along the straight path from rest to rest + disp (trapezoid), and the end state."""
    import mujoco
    d = mujoco.MjData(model)
    qa = vertex_qpos(model)
    prev, stored = None, 0.
    for k in range(steps + 1):
        for i, q in enumerate(qa):
            if q is not None:
                d.qpos[q] = k / steps * disp[i]
        mujoco.mj_forward(model, d)
        f = np.array(d.qfrc_spring)
        assert not np.any(d.qfrc_damper), "flex elastic forces must not be in qfrc_damper"
        if prev is not None:
            stored -= .5 * float((f + prev[1]) @ (np.array(d.qpos) - prev[0]))
        prev = (np.array(d.qpos), f)
    return stored, d


def main():
    import mujoco
    from topogym.world.mujoco.energy import flex_energy
    rng = np.random.default_rng(1)
    worst = dict(membrane=0., bending=0.)
    for mode in ("stretch", "bend", "both"):
        m = grid(mode)
        d0 = mujoco.MjData(m)
        mujoco.mj_forward(m, d0)
        assert max(map(abs, flex_energy(m, d0))) < 1e-15, "the flat rest stores no energy"
        adr, n = int(m.flex_vertadr[0]), int(m.flex_vertnum[0])
        x0 = np.array(d0.flexvert_xpos[adr:adr + n])
        c = x0.mean(axis=0)
        for trial in range(3):
            A = np.eye(3) + (rng.normal(scale=.03, size=(3, 3)) if mode != "bend" else 0.)
            R = .04 + .02 * trial
            u = x0[:, 0] - c[0]
            bent = np.c_[c[0] + R * np.sin(u / R), x0[:, 1], c[2] + R * (1 - np.cos(u / R))] if mode != "stretch" else x0
            target = c + (bent - c) @ A.T + rng.normal(scale=2e-4, size=x0.shape)
            stored, d = stored_by_path(m, target - x0)
            em, eb = flex_energy(m, d)
            assert float(d.energy[0]) == 0., "mujoco 3.14 books no flex elasticity (if it now does, drop flex_energy)"
            rel = abs(em + eb - stored) / stored
            key = "bending" if mode == "bend" else "membrane"
            worst[key] = max(worst[key], rel)
            assert rel < (1e-9 if mode == "bend" else 1e-5), (mode, trial, em, eb, stored)
            print(f"{mode:7s} trial {trial}: stored {stored:.6e} J, membrane {em:.6e}, bending {eb:.6e}, relative {rel:.1e}")
    m = grid("both")
    d = mujoco.MjData(m)
    d.qvel[:] = np.random.default_rng(0).normal(size=m.nv) * .1
    mujoco.mj_forward(m, d)
    M = np.zeros((m.nv, m.nv))
    mujoco.mj_fullM(m, d, M)
    ke = .5 * float(d.qvel @ M @ d.qvel)
    assert abs(float(d.energy[1]) - ke) <= 1e-12 * ke
    print(f"kinetic: data.energy[1] {float(d.energy[1]):.12e} = 1/2 v^T M v {ke:.12e}")
    solid = mujoco.MjModel.from_xml_string("""
<mujoco><option gravity="0 0 0" integrator="discrete" solver="CG"/>
<worldbody><flexcomp name="s" type="grid" count="3 3 3" spacing=".012 .012 .012" dim="3" radius=".001" mass=".01">
<elasticity young="1e5" poisson=".3"/><contact contype="0" conaffinity="0"/></flexcomp></worldbody></mujoco>""")
    try:
        flex_energy(solid, mujoco.MjData(solid))
        raise AssertionError("a 3D flex with elasticity must be refused")
    except NotImplementedError as e:
        print(f"refused: {e}")
    print(f"PASS (worst relative: membrane {worst['membrane']:.1e}, bending {worst['bending']:.1e})")


if __name__ == "__main__":
    main()
