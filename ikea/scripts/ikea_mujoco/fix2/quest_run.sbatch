#!/bin/bash -l
# Run a file of python command lines (one process each, $SLURM_CPUS_PER_TASK in parallel) in the IKEA SIF with the EPA
# byte-patched libmujoco bound over the wheel's (images/ikea-mujoco/mujoco-3.14.0-epa/), SIMD-masked; cwd = $F.
#   sbatch -A p32476 -p short -c 24 --mem=64G -t 04:00:00 --export=ALL,F=<work dir>,CMDS=<file> quest_run.sbatch
#SBATCH --job-name=ikea-fix2
#SBATCH --output=<cluster work dir>/ikea_fix2_20261015/logs/%x_%j.out
S=<cluster work dir>/ikea_stress_20261015
SIF=${SIF:-$S/sif/eai-ikea-mujoco_0.2.0_6b40176.sif}
LIB=${LIB:-$F/lib/libmujoco.so.3.14.0}
module load singularityce/4.3.1-gcc-8.5.0
cd "$F"
echo "start $(date -Is) $(hostname) $(lscpu | grep 'Model name' | sed 's/  */ /g') cpus=$SLURM_CPUS_PER_TASK"
one() {
  singularity exec --cleanenv --containall --no-home --pwd "$F" -B $F:$F -B $S:$S \
    -B $LIB:/usr/local/lib/python3.12/site-packages/mujoco/libmujoco.so.3.14.0 $SIF env \
    NPY_DISABLE_CPU_FEATURES=AVX512F,AVX512CD,AVX512_SKX,AVX512_CLX,AVX512_CNL,AVX512_ICL,AVX512_SPR \
    OPENBLAS_CORETYPE=Haswell OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 HOME=/tmp \
    sh -c "python -X faulthandler $1; echo rc=\$?"
}
export -f one; export F S SIF LIB
grep -v '^\s*$' "$CMDS" | xargs -P $SLURM_CPUS_PER_TASK -d '\n' -I{} bash -c 'one "$@"' _ {}
echo "end $(date -Is)"
