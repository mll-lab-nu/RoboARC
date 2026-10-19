"""The ring pulled over the hanging shoe by the left hand (MuJoCo port; the below-easy mechanism, hand-assisted).

Scene (stages' hang_release with pull_over): the right hand holds the right shoe 0.3 m above the table, its plane
horizontal; the two chain links hang from its eyes and the left shoe hangs from the links, bend lowest. The ring is
around both links and hangs from them beside the left shoe, its plane close to the shoe's. It is free once the hanging
shoe has passed through it: seen along the hanging axis the shoe is a bar (its width by the eyes' thickness), and the
ring's bore holds that bar when its long axis lies along the shoe's width (tasks/horseshoe_ring/below_easy.py).

The left hand grasps the ring and carries it along three poses measured from the hanging shoe: (1) level, centred on
the shoe's axis, just above the shoe's top (the links then pass through its middle); (2) the same attitude below the
shoe's bend; the swing from the hanging pose to (1) is one interpolated edge. The hanging shoe and the links are
contact participants: they are free and give way, so the plan does not screen the ring against them; the held right
shoe, the table and the robot (native contact limits) are screened at every 0.75 mm of ring motion, with continuous
left-arm IK. The physical passage is checked by the execution's per-tick contact rule and by the measured separation.
"""
import numpy as np
from scipy.spatial.transform import Rotation

from topogym.planning.rigid_reference.feedback import relative_pose
from topogym.planning.rigid_reference.geometry import pose_dict
from topogym.utils.io import write_json_atomic
from .ring_route import HeldRingPath

# The links pass through the ring all along and are not screened; the hanging shoe is (a path that swings the ring
# across it jams: easier, 2026-10-12, the left hand stalled with 5 cm of feedback bias two thirds into the swing).
PASSIVE=('chain_upper_1','chain_lower_1')


def pull_over_targets(planner, bodies, *, above_m=.005, below_m=.02):
    """Ring poses (level above the hanging shoe's top, level below its bend) for each admissible attitude."""
    task=planner.task;loop=task.assembly.loops['target_ring']
    centre_local=loop.center-task.body_origins['target_ring']
    rp,rr=bodies['target_ring'];frame=rr*loop.rotation
    hp,hr=bodies['horseshoe_left']
    shoe=hr.apply(task.placement_vertices['horseshoe_left'])+hp
    axis=hr.apply([1.,0.,0.])
    if axis[2]>0:axis=-axis
    if axis[2]>-.9:raise ValueError('The left shoe does not hang along gravity')
    width=hr.apply([0.,1.,0.]);face=np.cross(axis,width)
    mid=lambda values:.5*(values.min()+values.max())
    line=width*mid(shoe@width)+face*mid(shoe@face)
    top,bottom=float((shoe@axis).min()),float((shoe@axis).max())
    radii=np.asarray(loop.radii,dtype=float);round_bore=abs(radii[0]-radii[1])<1e-6
    frames=[]
    for sign in (1.,-1.):
        normal=sign*axis
        if round_bore:
            # any phase passes: the shortest turn of the ring's normal onto the hanging axis
            current=frame.apply([0.,0.,1.]);turn=np.cross(current,normal);s=np.linalg.norm(turn)
            delta=(Rotation.identity() if s<1e-9 else Rotation.from_rotvec(turn/s*np.arctan2(s,current@normal)))
            frames.append(delta*frame)
        else:
            for side in (1.,-1.):
                long=side*width;other=np.cross(normal,long)
                columns=[long,other,normal] if radii[0]>radii[1] else [-other,long,normal]
                frames.append(Rotation.from_matrix(np.column_stack(columns)))
    # The bore holds the shoe's section over a band of turns about the hanging axis (below_easy.py: easy at 53 of 90
    # phases), so turns of 15 and 30 deg about it are candidates too; the descent's audit keeps those that pass.
    frames=[Rotation.from_rotvec(axis*np.deg2rad(yaw))*f for f in frames for yaw in (0.,15.,-15.,30.,-30.)]
    frames.sort(key=lambda f:(f*frame.inv()).magnitude())
    rows=[]
    for target in frames:
        rotation=target*loop.rotation.inv()
        rows.append([(line+axis*level-rotation.apply(centre_local),rotation)
                     for level in (top-above_m,bottom+below_m+float(loop.wire_radius))])
    return rows,dict(hanging_axis_world=axis.tolist(),shoe_top_m=top,shoe_bottom_m=bottom,round_bore=bool(round_bore))


def screw_poses(start, end, step_deg=5.):
    """Poses along the one screw motion from start to end: a turn about a fixed line with a slide along it.

    The ring's centre then moves on an arc instead of a chord. Hard: the ring lies around the held right shoe, tilted;
    the turn onto the hanging axis that carries its upper arc over the shoe's eye end is such an arc about a line
    under the eye end (the chord would cut through the shoe).
    """
    delta=end[1]*start[1].inv();vector=delta.as_rotvec();angle=float(np.linalg.norm(vector))
    if angle<1e-3:return [start,end]
    u=vector/angle;slide=float(u@(end[0]-start[0]))
    fixed=np.linalg.pinv(np.eye(3)-delta.as_matrix())@(end[0]-delta.apply(start[0])-slide*u)
    rows=[]
    for part in np.linspace(0,1,max(2,int(np.ceil(np.rad2deg(angle)/step_deg))+1)):
        turn=Rotation.from_rotvec(u*angle*part)
        rows.append((fixed+turn.apply(start[0]-fixed)+part*slide*u,turn*start[1]))
    return rows


def plan_ring_pull_over(planner, observation, *, grasp_tcp=None, seed=None, write=True, strict=False):
    """Checked path of the held ring over the hanging shoe.

    For each admissible level attitude: the way from the measured ring pose to the descent's start is tried as the
    interpolated swing and as the screw motion (screw_poses), then the descent along the hanging axis follows. Ring
    contact with the screened bodies is admitted up to 0.5 mm of sampled penetration, half of the task's 1 mm limit
    (the ring starts touching the links' eyes). With grasp_tcp / seed a grasp that is not yet made is screened, and
    only the descent is required (its start reachable from that grasp, then the whole descent): the way to the
    descent's start is planned from the measured grasp afterwards (ManipulationPlanner.ring_grasp's screen; the
    complete audit of every candidate grasp took hours). strict: the screen also asks for a direct way (swing or
    screw) to one of the two descent starts nearest in turn (easy: from the grasp the descent-only screen chose, no
    joint route was found within the search's budget).
    """
    hands=observation['privileged']['world']['grasp']['hands']
    if not hands['right']['binding'] or hands['right']['binding']['target'][0]!='horseshoe_right':
        raise ValueError('The pull-over requires the right shoe held')
    checker=HeldRingPath(planner,observation,passive_contact_names=PASSIVE)
    checker.contact_limit_m=.0005
    if grasp_tcp is not None:checker.offset=relative_pose(checker.start,grasp_tcp)
    if seed is not None:checker.seed=np.asarray(seed,dtype=float).copy()
    targets,geometry=pull_over_targets(planner,checker.bodies)
    attempts=[];first=checker.seed.copy()
    def done(result):
        if write:write_json_atomic(planner.directory/'ring_pull_over_plan.json',result)
        return result
    usable=[]
    for above,below in targets:
        if grasp_tcp is not None:
            reach=checker.check(above,first)
            if not reach['passed']:
                attempts.append(dict(above=pose_dict(above),descent_start=reach.get('reason') or reach.get('issues')));continue
            checker.seed=np.asarray(reach['qpos']);audit=checker.ordered_audit([above,below]);checker.seed=first.copy()
            if audit['passed'] and not strict:
                return done(dict(passed=True,descent_only=True,ordered_audit=audit,attempts=attempts,**geometry,
                    scope='grasp screen: the descent is reachable; the way to its start is planned after the grasp'))
            if audit['passed']:usable.append((above,below))
            else:attempts.append(dict(above=pose_dict(above),descent=audit))
            if len(usable)<2:continue
            break
        for way,approach in (('swing',[checker.start,above]),('screw',screw_poses(checker.start,above))):
            audit=checker.ordered_audit(approach+[below])
            if not audit['passed']:
                attempts.append(dict(way=way,above=pose_dict(above),audit=audit));continue
            return done(dict(passed=True,way=way,poses=[pose_dict(p) for p in approach+[below]],ordered_audit=audit,
                attempts=attempts,**geometry,
                scope='sampled held-ring path with left-arm IK; the links are unscreened contact participants'))
    if grasp_tcp is not None:   # strict screen: a direct way to one of the first two usable descent starts
        for above,below in usable:
            for way,approach in (('swing',[checker.start,above]),('screw',screw_poses(checker.start,above))):
                audit=checker.ordered_audit(approach)
                if audit['passed']:
                    return done(dict(passed=True,descent_only=False,way=way,ordered_audit=audit,attempts=attempts,**geometry,
                        scope='strict grasp screen: a direct way to a descent start and the descent are reachable'))
                attempts.append(dict(way=way,above=pose_dict(above),audit=audit))
        return done(dict(passed=False,attempts=attempts,**geometry))
    # Neither direct way: a joint-space search of the held ring (the links' linkage with the ring kept) to any descent
    # start whose descent passes from there, then the checked descent.
    from .joint_route import plan_free_ring_transport
    from topogym.planning.rigid_reference.feedback import body_pose
    usable=[]
    for above,below in targets:
        reach=checker.check(above,first)
        if not reach['passed']:continue
        checker.seed=np.asarray(reach['qpos']);audit=checker.ordered_audit([above,below]);checker.seed=first.copy()
        if audit['passed']:usable.append((above,below))
    if not usable:return done(dict(passed=False,attempts=attempts,reason='no descent start passes',**geometry))
    try:route=plan_free_ring_transport(planner,observation,[above for above,_ in usable],passive_contact_names=PASSIVE)
    except (ValueError,TimeoutError) as error:
        attempts.append(dict(way='joint search',goals=len(usable),error=str(error)[:200]))
        return done(dict(passed=False,attempts=attempts,**geometry))
    end=body_pose(route['poses'][-1])
    above,below=min(usable,key=lambda t:np.linalg.norm(t[0][0]-end[0])+.085*(t[0][1]*end[1].inv()).magnitude())
    checker.seed=np.asarray(route['joint_knots'][-1]);audit=checker.ordered_audit([end,below]);checker.seed=first.copy()
    if not audit['passed']:
        attempts.append(dict(way='joint search',goals=len(usable),descent=audit))
        return done(dict(passed=False,attempts=attempts,**geometry))
    return done(dict(passed=True,way='joint search',poses=route['poses']+[pose_dict(below)],ordered_audit=audit,
        attempts=attempts,transport_queries=route['queries'],goals=len(usable),**geometry,
        scope='joint search and Cartesian replay of the held ring, then the sampled descent; the links unscreened'))
