# Porting a RoboTangle family to MuJoCo: guide for collaborators and their agents

Read this together with [MUJOCO_CORE.md](MUJOCO_CORE.md) ("Porting a family"), the acceptance bar in [MUJOCO_MIGRATION.md](MUJOCO_MIGRATION.md), and the plan and claims in [ROBOTANGLE_TRACKER.md](ROBOTANGLE_TRACKER.md). Integrator: Qineng.

## Shared infrastructure today (2026-10-09 night)

**On `dev/robotangle_mujoco` now:**
- The MuJoCo core: World/Robot behind TopoGym's interface, the scene builder and the Session integration.
- The Nyx render bridge, which keeps frames identical to today's.
- Contact policy v2:
  - converged convex-collision CCD;
  - robot hull caps;
  - overdamped near-hard contacts for claws.
- Datasheet joint-speed clamps.
- Cross-CPU SIMD masking.
- The acceptance bar and the tracker.

**On the branch since 18f641258:** the MuJoCo IK/FK for the families' closed-loop oracles (bypassing `GenesisIKSolver` / `GenesisFKSolver`), the `capsules` profile option, contact-role geom names and the release geometry.

**In review, landing in the next push:**
- Real servo torque caps for the ALOHA, plus a grasp slip that releases the grasp and never fails the episode. The slip threshold will be set per robot.
- Profile tools for exact collision geometry:
  - `capsules`: a cylinder plus its end caps becomes one exact capsule;
  - `prism_partition`: thin slab prisms become fat exact prisms;
  - `split`: a family-supplied exact convex decomposition of a non-convex mesh, refused unless it matches the mesh.
- URDF collision geoms named by contact role.
- MuJoCo release geometry for the final settling.

**In the repo now:** the stress-validation harness, `scripts/robotangle/stress/` with [STRESS_VALIDATION.md](STRESS_VALIDATION.md). It holds the per-substep observer with an energy balance, the self-play search, the shrinker, the acceptance-round runner and criteria v2.3. Run its self-check for step 2 before asking for the formal round.

## How to start

1. **Claim the family:** write your name in its Owner cell in `ROBOTANGLE_TRACKER.md` and commit that to `dev/robotangle_mujoco`.
2. **Branch** from `dev/robotangle_mujoco`.
   - Family code goes in family files and your profile entry.
   - Shared changes go in small separate commits titled `shared: …`. Tell Qineng; they are integrated in batches after review.
   - Never force-push.
3. **Runtime:**
   - mujoco **3.14.0**, the version in the RoboTangle image.
   - The RoboTangle image (Genesis 1.2.1 + gs-nyx) is still needed, to read the frozen instances and to render Nyx frames.
   - MuJoCo physics runs on the CPU, one thread per process, so parallelise across processes. Nyx frames need an NVIDIA GPU with RT cores.
4. **Do steps 1 and 2 of the plan for every tier.** The definitions of done are in the tracker. Then ask for the formal stress acceptance round.

## What porting has taught us so far (traps)

1. **Open-loop replay of the released references often fails on MuJoCo.**
   - Most references are recordings of closed-loop policies, with sub-millimetre route clearances.
   - Re-run the family's own feedback policy inside a MuJoCo Session; this needs the shared IK/FK.
   - Keep the frozen instance and the task unchanged.
2. **Never collide a non-convex mesh as one hull.** MuJoCo collides every mesh as its convex hull, which seals windows and slots. Use an exact convex decomposition (`split`) or primitives.
3. **Avoid sliver hulls,** i.e. pieces thinner than about 0.5 mm.
   - The narrow phase can return bogus depths: a just-touching sliver reported 21 mm and kicked the body away.
   - Use `prism_partition`, primitives, or merge the slivers.
4. **Thin hulls flip their contact normal once penetration nears half their thickness.** We saw 187 kN and 486 J created. Use thick hulls or primitives for anything that can be pushed deep.
5. **Converged CCD is required.** The default 35 iterations gave fake 130 mm depths. The shared profile sets it; don't override it.
6. **Too-soft contacts let bodies sink deep,** and that causes more normal flips. Stiffer, overdamped contacts behaved better:
   - claws: 1.25 ms, dampratio 2, solimp dmax .99;
   - fiber: 2.5 ms, dampratio 1.
7. **Represent thin round things with capsules:** wires, rods and cable links (`capsules`). They are exact and fast.
8. **Respect the robot limits when you record references.** Report the max joint torque and weld force your references need.
9. **Check stress with the per-substep observer and its energy balance.** The Session's tick-end readout misses short events.
10. **Robot-internal constraints need the same care as contacts.** The ALOHA's MJCF couples its second finger to the driven one with an equality at MuJoCo's default 20 ms. Under a fast gripper oscillation (~20 Hz square wave) that coupling created energy (E1, found by the GPU search). Every ALOHA family sets `robot.coupling = ALOHA_COUPLING` (1.25 ms, dampratio 2; [MUJOCO_CORE.md](MUJOCO_CORE.md)), then re-records its references.
11. **Know which time constants sit at MuJoCo's floor, 2·dt.** On the exact constraint diagonal every constraint
    follows its nominal solref, so 2·dt is as stiff as MuJoCo allows (with `refsafe`, the default, a smaller value acts as
    2·dt under implicitfast; the discrete integrator, blanket's, skips `refsafe`, so there a smaller value acts as written:
    check efc_KBIP). Two such constraints side by side can ratchet load. In claws the grasp welds at 2·dt, beside 2·dt
    contacts and finger coupling, jammed (22 -> 149 N in 20 ms) and fired a piece through a finger; at 4·dt (claws_v6)
    that is clean.
    The same change broke other families' references (buzz hard, keyhole easy, blanket's pins), so choose per family and
    re-run its references and stress set. Scenes on the shared head (2026-10-10; MuJoCo timestep 0.625 ms, blanket 2.5 ms):

    | family | at 2·dt | above |
    |---|---|---|
    | claws | task contacts (1.25 ms, dr 2, .9 / .99); ALOHA finger coupling (1.25 ms, dr 2) | grasp welds 2.5 ms; robot-robot and robot-table 5 ms |
    | buzz | every contact class (1.25 ms, dr 2); grasp weld (1.25 ms, .99 / .999) | Robotiq four-bar closures and finger mimic 5 ms; pads 4 ms |
    | fiber | the cable's own contacts (1.25 ms, dr 2); grasp weld (1.25 ms) | robot class 2.5 ms; robot-robot and statics 5 ms; Robotiq closures 5 ms |
    | hedgehog | every contact class (1.25 ms); grasp weld (1.25 ms) | Robotiq closures and finger mimic 5 ms |
    | keyhole | task and static contacts (1.25 ms); grasp welds (1.25 ms) | ALOHA finger coupling at MuJoCo's default 20 ms (not yet `ALOHA_COUPLING`) |
    | knot | rope node connects (1.25 ms, dr 1, .9999); rope contacts; ALOHA coupling | |
    | rope_threading | rope node connects; rope contacts; grasp welds (one per hand and link); ALOHA coupling | statics 5 ms |
    | blanket | every contact (5 ms); the grasp pins (connects, 5 ms); below it: the ALOHA coupling at its nominal 1.25 ms (0.5·dt, no `refsafe` under the discrete integrator), quiet: a 20 Hz gripper square wave only dissipates (-0.004 J over 60 controls, 0.03 mm violation) | joint limits 20 ms |

    Robot joint limits are at 5-20 ms everywhere. The list is a scan of each family's frozen scene: its equalities,
    joint limits and geom contact classes, in units of dt.

## The physics bar

- We do **not** need to match Genesis. Choose what MuJoCo does well, as long as the task's own needs hold: tolerances, grasps, what counts as touching.
- Learn from mature models such as MuJoCo Menagerie.
- Clamp, never fail: limits and recovery, no new episode-ending rules.
- Protocol first: don't change task goals, rules or texts beyond what the port needs.
