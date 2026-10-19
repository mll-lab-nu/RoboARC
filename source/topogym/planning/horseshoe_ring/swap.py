"""The hand swap that lets the right hand leave the ring on the table (hard, MuJoCo port).

Scene (HorseshoePolicy.drop_ring_by_hand_swap): after the fold, the tilt and the left hand's release, the right hand
holds the right shoe by one leg with its plane upright; the left shoe hangs from the chain links; the ring hangs
around the right hand's two fingers, free of the puzzle. The right hand cannot leave the ring without opening, and
opening drops the shoes (docs/rcb/families/horseshoe_ring.md). So the left hand takes the same shoe by its other leg,
the right hand opens and carries the ring on its open fingers over the goal region, a few centimetres above the table,
and pitches them down; the ring is then in the goal (on the fingers or on the table under them), the remainder in the
left hand. Holding is allowed by the task's rule ("remainder behind separator; holding allowed"), so the shoes are
never let go. These are the read-only plans; execution is closed loop in the policy.
"""
import numpy as np
from scipy.spatial.transform import Rotation

from topogym.planning.rigid_reference.feedback import body_pose, tcp_pose
from topogym.planning.rigid_reference.geometry import interpolate, pose_dict
from topogym.utils.io import write_json_atomic

STRUCTURAL=('table','inter_arm','same_arm_collision')


def plan_takeover_grasp(planner, observation, *, angles=tuple(range(40,125,5)), approach_m=.08):
    """A left-hand grasp of the right shoe while the right hand holds it.

    The band is pinched across its 8 mm thickness, the gripper coming in the shoe's plane from outside the U (the
    left arm's side), TCP 13.8 mm short of the band's centreline as in the shoe grasps. Candidates: points of the
    band's centreline by ellipse angle, both jaw senses. Checked: native IK of the left arm, the approach's start and
    end and 20 samples between under the native contact limits (every piece at its measured pose), and a joint transit
    from the arm's present pose (direct, else the seeded search).
    """
    from topogym.assets.geometry.horseshoe_ring import band_centerline
    from .joint_transit import plan_transit
    hands=observation['privileged']['world']['grasp']['hands']
    if hands['left']['binding'] is not None or not hands['right']['binding'] or hands['right']['binding']['target'][0]!='horseshoe_right':
        raise ValueError('The takeover requires the left hand free and the right shoe held by the right hand')
    task,checker=planner.task,planner.checker
    bodies,seed=planner.observed(observation)
    p,r=bodies['horseshoe_right'];h,unit=task.assembly.template['horseshoe'],task.assembly.unit
    centre=((task.assembly.template['chain']['eye_to_link_center']+h['eye_offset_x'])*unit
            +band_centerline(h,unit,np.array([np.deg2rad(h['mouth_half_angle_deg'])]))[0,0])
    turned=Rotation.from_euler('z',np.pi);fingers=planner.robot._q_indices['left'][6:]
    left_base=np.asarray(planner.robot.entity.get_link(planner.robot.groups['left'].hand_link.split('/')[0]+'/base_link').get_pos()).reshape(-1)
    rows=[];attempts=[]
    for angle in angles:
        a=np.deg2rad(angle)
        point=band_centerline(h,unit,np.array([a]))[0]
        tangent=band_centerline(h,unit,np.array([a+1e-5]))[0]-band_centerline(h,unit,np.array([a-1e-5]))[0]
        radial=np.array([tangent[1],-tangent[0],0.]);radial/=np.linalg.norm(radial)
        local=turned.apply(point)+[centre,0,0]-task.body_origins['horseshoe_right'];outward=turned.apply(radial)
        if outward@local<0:outward=-outward
        world=p+r.apply(local);inward=-r.apply(outward);normal=r.apply([0.,0.,1.])
        rows.append((float(np.linalg.norm(world[:2]-left_base[:2])-.5*world[2]),angle,world,inward,normal))
    # nearest to the left arm's base first, higher on the shoe preferred (clear of the hanging shoe)
    for _,angle,world,inward,normal in sorted(rows,key=lambda row:row[0]):
        for sense,jaw in ((1,normal),(-1,-normal)):
            turn=Rotation.from_matrix(np.column_stack([inward,jaw,np.cross(inward,jaw)]))
            target=(world+turn.apply([-.01381,0.,0.]),turn);pre=(target[0]-approach_m*inward,turn)
            record=dict(angle_deg=angle,jaw_sense=sense)
            try:
                q=seed.copy();q[fingers]=.027
                pre_q=checker.solve('left',pre,q);samples=[pre_q.copy()];current=pre_q
                for u in np.linspace(0,1,21)[1:]:
                    current=checker.solve('left',interpolate(pre,target,u),current);samples.append(current.copy())
                for index,sample in enumerate(samples):
                    checked=checker.check(sample,body_poses=bodies)
                    if not checked['passed']:raise ValueError(f'approach sample {index}: {checked["issues"]}')
                first=seed.copy();first[fingers]=.027
                transit=plan_transit(planner,first,pre_q,bodies,side='left',seconds=20.)
            except ValueError as error:
                attempts.append(dict(record,error=str(error)[:200]));continue
            result=dict(passed=True,**record,point_world_m=world.tolist(),pre_tcp=pose_dict(pre),target_tcp=pose_dict(target),
                transit=transit['waypoints'],transit_method=transit['method'],attempts=attempts,
                scope='sampled left-arm IK and native contacts at measured poses; bilateral acquisition requires execution')
            write_json_atomic(planner.directory/'takeover_grasp_plan.json',result)
            return result
    write_json_atomic(planner.directory/'takeover_grasp_plan.json',dict(passed=False,attempts=attempts))
    raise ValueError('No checked left-hand grasp of the held right shoe')


def plan_ring_drop(planner, observation, segments):
    """The right hand's TCP path with the ring on its open fingers, from its measured pose.

    segments: dicts with move_m [x, y, z] and / or pitch_deg (the tool pitches down about the horizontal axis
    perpendicular to its starting direction, through the TCP). Checked at 1 cm / 1 deg steps: continuous right-arm
    IK, joint limits, same-arm and arm-arm collision (the left arm at its measured pose, holding the shoe), the table.
    The ring is not screened: it hangs on the fingers and is meant to reach the table. The right arm must not touch
    any other piece beyond the task's ordinary contact limit (native contact classification).
    """
    if not isinstance(segments,(list,tuple)) or not 1<=len(segments)<=6:raise ValueError('Invalid ring drop path')
    bodies,q=planner.observed(observation)
    tcp=tcp_pose(observation['robot']['right']);tool=tcp[1].apply([1.,0.,0.]);axis=np.cross([0.,0.,1.],tool)
    if np.linalg.norm(axis)<.5:raise ValueError('The ring drop expects a roughly horizontal gripper')
    axis/=np.linalg.norm(axis);position,rotation=tcp[0].copy(),tcp[1];rows=[]
    for number,segment in enumerate(segments):
        if not isinstance(segment,dict) or set(segment)-{'move_m','pitch_deg','seconds'}:raise ValueError(f'Invalid ring drop segment {number}')
        move=np.asarray(segment.get('move_m',(0.,0.,0.)),dtype=float);pitch=float(segment.get('pitch_deg',0.))
        if move.shape!=(3,) or not np.isfinite(move).all() or np.linalg.norm(move)>.4 or not np.isfinite(pitch) or abs(pitch)>60:
            raise ValueError(f'Invalid ring drop segment {number}')
        for u in np.linspace(0,1,max(2,int(np.ceil(max(np.linalg.norm(move)/.01,abs(pitch))))+1))[1:]:
            pose=(position+u*move,Rotation.from_rotvec(axis*np.deg2rad(u*pitch))*rotation);previous=q.copy()
            q=planner.checker.solve('right',pose,q);checked=planner.checker.check(q)
            issues=[x for x in checked['issues'] if x in STRUCTURAL or x.endswith('/joint_limits')]
            issues+=[f"{v['pair'][0]}|{v['pair'][1]}" for v in checked.get('contact_violations',())
                     if v['pair'][1]!='target_ring' and v['pair'][0].startswith('right')]
            if issues or np.abs(q-previous).max()>.15:
                raise ValueError(f'Ring drop robot issue or joint branch change in segment {number}: {issues}')
        position,rotation=pose
        rows.append(dict(tcp=pose_dict(pose),seconds=float(segment.get('seconds') or max(2.,np.linalg.norm(move)/.02,abs(pitch)/5.))))
    result=dict(passed=True,start_tcp=pose_dict(tcp),segments=rows,observed_state=observation['state'],
        scope='sampled right-arm IK and robot structure; the ring on the fingers is unscreened; execution required')
    write_json_atomic(planner.directory/'ring_drop_plan.json',result)
    return result


def plan_ring_into_goal(planner, observation, *, gap_m=.06, pitch_deg=35., margin_m=.02):
    """The right hand's path from its measured pose that brings the ring on its fingers over the goal region.

    The ring hanging on the open fingers is carried as if rigidly attached. Candidate translations move its measured
    box a fraction (1, 0.8, ..., 0) of the way to the region's centre along x and along y, and bring its lowest point
    gap_m (or gap_m + 3 cm) above the table; then the fingers pitch down by pitch_deg. Candidates whose box stays
    margin_m inside the region's border are tried in order of decreasing margin; the first that plan_ring_drop
    accepts is taken. (Hard: the full centring is out of the right arm's reach or changes its IK branch.)
    """
    task=planner.task;region=task.goal_region
    rp,rr=body_pose(observation['privileged']['world']['bodies']['target_ring'])
    ring=rr.apply(task.placement_vertices['target_ring'])+rp
    lo,hi=ring.min(0),ring.max(0)
    centre=np.asarray(region.center_xy_m,dtype=float);half=np.asarray(region.half_size_m,dtype=float)
    full=centre-.5*(lo[:2]+hi[:2]);fractions=(1.,.8,.6,.4,.2,0.);candidates=[]
    for fx in fractions:
        for fy in fractions:
            shift=full*[fx,fy]
            margin=float(min(np.min(lo[:2]+shift-(centre-half)),np.min(centre+half-(hi[:2]+shift))))
            if margin<margin_m:continue
            for gap in (gap_m,gap_m+.03):
                candidates.append((margin,gap==gap_m,[float(shift[0]),float(shift[1]),float(planner.checker.table_height+gap-lo[2])]))
    attempts=[]
    for margin,_,move in sorted(candidates,key=lambda row:(-round(row[0],3),not row[1])):
        try:plan=plan_ring_drop(planner,observation,[dict(move_m=move),dict(pitch_deg=pitch_deg)])
        except ValueError as error:
            attempts.append(dict(move_m=move,margin_m=margin,error=str(error)[:120]));continue
        plan.update(ring_move_m=move,ring_box_margin_m=margin,attempts=attempts)
        write_json_atomic(planner.directory/'ring_into_goal_plan.json',plan)
        return plan
    write_json_atomic(planner.directory/'ring_into_goal_plan.json',dict(passed=False,attempts=attempts))
    raise ValueError(f'No checked carry of the ring over the goal region ({len(attempts)} tried)')


def plan_held_shift(planner, observation, side, shift_m):
    """Translate the body held by one hand by shift_m (1 cm steps): IK of that arm, robot structure, joint limits."""
    shift=np.asarray(shift_m,dtype=float)
    if shift.shape!=(3,) or not np.isfinite(shift).all() or not 0<np.linalg.norm(shift)<=.15:
        raise ValueError('A held-body shift is a finite vector within 15 cm')
    bodies,q=planner.observed(observation);tcp=tcp_pose(observation['robot'][side]);rows=[]
    for u in np.linspace(0,1,max(2,int(np.ceil(np.linalg.norm(shift)/.01))+1))[1:]:
        previous=q.copy();q=planner.checker.solve(side,(tcp[0]+u*shift,tcp[1]),q)
        checked=planner.checker.check(q)
        issues=[x for x in checked['issues'] if x in STRUCTURAL or x.endswith('/joint_limits')]
        if issues or np.abs(q-previous).max()>.15:raise ValueError(f'Held shift robot issue at {u:.2f}: {issues}')
        rows.append(q.tolist())
    return dict(passed=True,side=side,shift_m=shift.tolist(),samples=len(rows))
