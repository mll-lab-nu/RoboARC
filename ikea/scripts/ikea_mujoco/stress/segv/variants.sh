# variants.sh <dir>: the option matrix on <dir>/segv_min.xml, MuJoCo 3.14.0 (scene venv) and 3.16.0, rc per run
cd $1; . /workspace/qn_ikea/stress/env.sh; P16=/workspace/qn_ikea/stress/venv_m316/bin/python
CHECK="import mujoco,sys; m=mujoco.MjModel.from_xml_path(sys.argv[1]); d=mujoco.MjData(m); mujoco.mj_forward(m,d)"
v() { name=$1; shift; cp segv_min.xml v_$name.xml; for e in "$@"; do sed -i "$e" v_$name.xml; done
  $PY -c "$CHECK" v_$name.xml >/dev/null 2>&1; a=$?; $P16 -c "$CHECK" v_$name.xml >/dev/null 2>&1; b=$?; echo "$name: 3.14.0 rc=$a 3.16.0 rc=$b"; }
v as_is
v nativeccd_off 's/nativeccd="enable"/nativeccd="disable"/'
v multiccd_off 's/multiccd="enable"/multiccd="disable"/'
v tol1e-6 's/ccd_tolerance="1e-10"/ccd_tolerance="1e-06"/'
v tol1e-8 's/ccd_tolerance="1e-10"/ccd_tolerance="1e-08"/'
v tol1e-9 's/ccd_tolerance="1e-10"/ccd_tolerance="1e-09"/'
v tol1e-12 's/ccd_tolerance="1e-10"/ccd_tolerance="1e-12"/'
v iter35 's/ccd_iterations="500"/ccd_iterations="35"/'
v iter50 's/ccd_iterations="500"/ccd_iterations="50"/'
v iter100 's/ccd_iterations="500"/ccd_iterations="100"/'
v iter5000 's/ccd_iterations="500"/ccd_iterations="5000"/'
v defaults 's/ccd_iterations="500" ccd_tolerance="1e-10"//' 's/<flag [^>]*>//'
