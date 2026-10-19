#!/usr/bin/env python3
"""Static audit of the cylinder -> prism rule (ikea_runtime.cylinder_prisms) on one IKEA scene, run in its own process.

usage: python audit.py <scene_dir_new> [<scene_dir_base>]   -> one JSON line
  - the rule's selection (converted geoms) and every cylinder it leaves alone, with its collision bits;
  - the robot the rule saw (body trees with an actuated joint) against rcb_ikea.robot_bodies on the compiled model;
  - with the base scene dir (the runtime before the rule): initial-state hash both ways, and whether the compiled
    models are byte-identical (expected only when nothing is converted), plus build times.
"""
import hashlib, importlib, json, os, sys, time
import numpy as np
import mujoco

new = os.path.abspath(sys.argv[1])
base = os.path.abspath(sys.argv[2]) if len(sys.argv) > 2 else None
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "shared"))
import rcb_ikea  # noqa: E402


def model_bytes(m):
    n = mujoco.mj_sizeModel(m)
    buf = np.zeros(n, dtype=np.uint8)
    mujoco.mj_saveModel(m, None, buf)
    return hashlib.sha256(buf.tobytes()).hexdigest()


def init_sha(m):
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, m.key("home").id)
    mujoco.mj_forward(m, d)
    return rcb_ikea.state_sha(m, d)


sys.path.insert(0, new)
rt = importlib.import_module("ikea_runtime")
xml = os.path.join(new, "scene.xml")
spec = mujoco.MjSpec.from_file(xml)
cyl_before = {g.name: dict(body=g.parent.name, contype=int(g.contype), conaffinity=int(g.conaffinity),
                           r=float(g.size[0])) for g in spec.geoms if g.type == mujoco.mjtGeom.mjGEOM_CYLINDER}
t0 = time.time()
audit = rt.cylinder_prisms(spec)
m_new = spec.compile()
t_new = time.time() - t0
t0 = time.time()
m_xml = mujoco.MjModel.from_xml_path(xml)
t_xml = time.time() - t0
robot = rcb_ikea.robot_bodies(m_xml)
robot_bits = sorted({(int(m_xml.geom_contype[g]), int(m_xml.geom_conaffinity[g])) for g in range(m_xml.ngeom)
                     if m_xml.geom_bodyid[g] in set(robot.tolist())})
conv = {a["geom"] for a in audit}
out = dict(scene=os.path.basename(new), converted=audit, n_converted=len(audit),
           left_as_cylinders={k: v for k, v in cyl_before.items() if k not in conv},
           robot_roots=sorted({m_xml.body(int(m_xml.body_rootid[b])).name for b in robot}),
           robot_bits=robot_bits, build_s=dict(spec_rule=round(t_new, 3), from_xml=round(t_xml, 3)),
           init_sha_new=init_sha(m_new), init_sha_xml=init_sha(m_xml),
           model_sha_new=model_bytes(m_new), model_sha_xml=model_bytes(m_xml))
# The rule with nothing selected must compile the same model as from_xml_path.
spec0 = mujoco.MjSpec.from_file(xml)
out["model_sha_spec_plain"] = model_bytes(spec0.compile())
# The Scene object as the task builds it
S = importlib.import_module("scene").load()
out["scene_attr_converted"] = [a["geom"] for a in S.cylinder_prisms]
out["scene_init_sha"] = rcb_ikea.state_sha(S.model, S.data)
if base:
    # the base runtime's Scene, in a clean module namespace
    for k in [k for k in sys.modules if k in ("scene", "ikea_runtime", "success") or k.endswith("_ops") or
              k.endswith("_mechanism")]:
        del sys.modules[k]
    sys.path.remove(new)
    sys.path.insert(0, base)
    S0 = importlib.import_module("scene").load()
    out["base_scene_init_sha"] = rcb_ikea.state_sha(S0.model, S0.data)
    out["base_model_sha"] = model_bytes(S0.model)
print(json.dumps(out))
