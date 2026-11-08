#!/usr/bin/env bash
# A DROID-trained policy (pi0.5-DROID, MolmoAct2-DROID, GR00T-N1.7-DROID) on RoboARC's one-arm puzzles in one
# command (docs/VLA.md, "DROID-trained policies"): starts one DroidAdapter model server, runs every (task, seed) episode
# in the puzzle image with DROID's camera poses, --jobs at a time against that server, then stops the server. Results:
# <out>/<task>/<run id>/s<seed>/e1/ (roboarc.vla.run). Every model gets two images, as the official DROID simulations
# send them: the exterior camera on the robot's left (agentview) and the wrist; MolmoAct2 with agentview aimed as in
# AI2's simulation (--camera-poses molmoact2), the others as in sim-evals (--camera-poses droid). Cosmos3 also gets
# the exterior camera on the robot's right (front_oblique), as RoboLab's Cosmos3 client sends three.
#
#   scripts/vla_droid.sh --task lock-picking-03,buzz-wire-3d-06 --model jointpos --seed all --jobs 4 --out DIR
#
#   --task    one-arm puzzle ids, comma-separated (buzz-wire-3d, fiber-bundling, hedgehog-in-cage, lock-picking)
#   --model   jointpos (default) | velocity | polaris | molmoact2 | molmoact2-bf16 | groot | cosmos3 | hold (no model:
#             checks the setup) (roboarc/vla/droid_models.py)
#   --seed    0 (default) | 0,1,2 | all              --jobs    episodes at a time (default 1)
#   --out     results root (default ./vla_results)   --run-id  default pi05-droid-<model>, <model>-droid (the others)
#   --policy  ws://HOST:PORT of a running DroidAdapter server of --model (then none is started)
#   --port    the started server's port (default 8000);   after --: more roboarc.vla.run arguments
#
# Needs docker with the GPU, the protocol packages in $VLA_LIB (default .vla-lib; docs/VLA.md, Setup) and the model's
# environment set up with `uv sync`: openpi at 215abfb in $OPENPI (default ~/openpi) for pi0.5, allenai/molmoact2 at
# 66b87e6 in $MOLMOACT2 (default ~/molmoact2) for MolmoAct2, NVIDIA/Isaac-GR00T at 1a1837f in $GROOT (default
# ~/Isaac-GR00T) for GR00T, NVIDIA/cosmos-framework at bbd4a67 in $COSMOS3 (default ~/cosmos-framework) for Cosmos3.
# The server takes the GPU memory it uses (pi0.5-DROID 8.6 GB; MolmoAct2-DROID 11 GB in bfloat16, about 24 GB in
# float32; GR00T-N1.7-DROID 6.4 GB; Cosmos3-Nano-Policy-DROID 33 GB), each episode about 1 GB (docs/VLA.md).
set -euo pipefail
REPO=$(cd "$(dirname "$0")/.." && pwd)
IMAGE=${IMAGE:-ghcr.io/mll-lab-nu/roboarc:2.2.0@sha256:96a0a895c0e4934adca39f7a370e67bf82bfb4db7ef049131a3ac0dc9d0f9057}
VLA_LIB=${VLA_LIB:-$REPO/.vla-lib}
OPENPI=${OPENPI:-$HOME/openpi}
MOLMOACT2=${MOLMOACT2:-$HOME/molmoact2}
GROOT=${GROOT:-$HOME/Isaac-GR00T}
COSMOS3=${COSMOS3:-$HOME/cosmos-framework}
TASKS="" MODEL=jointpos SEEDS=0 JOBS=1 OUT=$PWD/vla_results RUN_ID="" POLICY="" PORT=8000
while [ $# -gt 0 ]; do
  case $1 in
    --task) TASKS=$2;; --model) MODEL=$2;; --seed) SEEDS=$2;; --jobs) JOBS=$2;; --out) OUT=$2;;
    --run-id) RUN_ID=$2;; --policy) POLICY=$2;; --port) PORT=$2;;
    --) shift; break;;
    *) echo "unknown option $1 (see the top of $0)" >&2; exit 2;;
  esac
  shift 2
done
[ -n "$TASKS" ] || { echo "--task is needed (see the top of $0)" >&2; exit 2; }
[ -d "$VLA_LIB/websockets" ] || { echo "no protocol packages in $VLA_LIB (docs/VLA.md, Setup)" >&2; exit 2; }
CAMERAS=observation/exterior_image_1_left=agentview,observation/wrist_image_left=wrist
# per model: its environment, convention, actions executed per query (each model's own DROID client), image size and
# camera poses
ENV_DIR=$OPENPI CONV=droid HORIZON=8 SIZE=224 NAME=pi05-droid-$MODEL POSES=droid
case $MODEL in
  velocity) CONV=droid-velocity;;
  molmoact2*) ENV_DIR=$MOLMOACT2 HORIZON=15 SIZE=320 NAME=$MODEL-droid POSES=molmoact2;;
  groot) ENV_DIR=$GROOT HORIZON=15 SIZE=320 NAME=groot-n17-droid;;
  cosmos3) ENV_DIR=$COSMOS3 HORIZON=32 SIZE=640 NAME=cosmos3-nano-droid
    CAMERAS=$CAMERAS,observation/exterior_image_2_left=front_oblique
    # its server fetches the checkpoint with `uv run ... hf download`
    command -v uv > /dev/null || { echo "cosmos3 needs uv on PATH (docs/VLA.md)" >&2; exit 2; };;
esac
RUN_ID=${RUN_ID:-$NAME}
[ "$SEEDS" = all ] && SEEDS=0,1,2
mkdir -p "$OUT"
OUT=$(cd "$OUT" && pwd)
DOCKER=(docker run --rm --gpus all -e NVIDIA_DRIVER_CAPABILITIES=all --network host -u "$(id -u):$(id -g)" -e HOME=/tmp
        -v "$REPO:/repo:ro" -v "$VLA_LIB:/vla-lib:ro" -w /repo -e PYTHONPATH=/repo:/opt/topogym:/vla-lib)

server=""
if [ -z "$POLICY" ]; then
  POLICY=ws://127.0.0.1:$PORT
  log=$OUT/server-$MODEL-$PORT.log
  if [ "$MODEL" = hold ]; then
    "${DOCKER[@]}" --name "vla-server-$PORT" "$IMAGE" python -m roboarc.vla.droid_adapter --model hold --port "$PORT" \
      > "$log" 2>&1 &
  else
    [ -x "$ENV_DIR/.venv/bin/python" ] || { echo "no $MODEL environment in $ENV_DIR (docs/VLA.md, DROID-trained policies)" >&2; exit 2; }
    XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONPATH=$REPO \
      "$ENV_DIR/.venv/bin/python" -m roboarc.vla.droid_adapter --model "$MODEL" --port "$PORT" > "$log" 2>&1 &
  fi
  server=$!
  trap 'kill $server 2>/dev/null; docker rm -f vla-server-$PORT >/dev/null 2>&1 || true' EXIT
  until curl -sf "http://127.0.0.1:$PORT/healthz" > /dev/null; do
    kill -0 "$server" 2>/dev/null || { echo "the model server stopped (log $log):" >&2; tail -5 "$log" >&2; exit 1; }
    sleep 5
  done
  echo "model server ready: $MODEL on port $PORT (log $log)"
fi

episode() {   # <task> <seed> [runner args]: one runner in its own container; prints the runner's line for the episode
  local task=$1 seed=$2
  shift 2
  "${DOCKER[@]}" --name "vla-$RUN_ID-$task-s$seed-$$" -v "$OUT:/results" "$IMAGE" python -m roboarc.vla.run \
    --task "$task" --policy "$POLICY" --convention "$CONV" --policy-hz 15 --horizon "$HORIZON" --image-size "$SIZE" \
    --camera-poses "$POSES" --cameras "$CAMERAS" --run-id "$RUN_ID" --seed "$seed" --out /results "$@" 2>&1 | tail -1 | sed "s/^/$task seed $seed: /"
}

pids=()
for task in ${TASKS//,/ }; do
  for seed in ${SEEDS//,/ }; do
    while [ "$(jobs -rp | grep -cvx "${server:-none}")" -ge "$JOBS" ]; do sleep 2; done
    episode "$task" "$seed" "$@" &
    pids+=($!)
  done
done
wait "${pids[@]}"
echo "results: $OUT/<task>/$RUN_ID/"
