#!/bin/bash
#SBATCH --job-name=zmorph_bd_gpu
#SBATCH --account=elbert                  # Hyak account allocation
#SBATCH --partition=gpu-a40               # A40 partition
#SBATCH --gres=gpu:1                      # one GPU (use gpu:a40:1 if your partition needs the type)
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=13                # also sets the CPU baseline's thread count
#SBATCH --mem=64G
#SBATCH --time=0-03:00:00
#SBATCH --output=logs/zmorph_bd_gpu_%j.out
#SBATCH --error=logs/zmorph_bd_gpu_%j.err

# Time the GPU Brownian dynamics against the CPU reference and cross-check them.
# Submit from the repo root:  sbatch slurm/bench_bd_gpu.sbatch
# Results land in results/gpu_bench/bench_<jobid>.txt (and the logs/ files).

set -eo pipefail
mkdir -p logs results/gpu_bench

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMBA_NUM_THREADS="${SLURM_CPUS_PER_TASK:-13}"
export MPLBACKEND=Agg
export PYTHONUNBUFFERED=1

CONDA_ROOT=/gscratch/elbert/miniconda3
ENV_DIR="${ZMORPH_ENV:-$CONDA_ROOT/envs/rr_env2}"      # needs numpy, scipy, scikit-image, numba with CUDA support
REPO="${ZMORPH_REPO:-/gscratch/elbert/git/zmorph}"

if [ -f "$CONDA_ROOT/etc/profile.d/conda.sh" ]; then
    source "$CONDA_ROOT/etc/profile.d/conda.sh"
else
    source ~/.bashrc
fi
export LD_LIBRARY_PATH="$ENV_DIR/lib:${LD_LIBRARY_PATH:-}"
conda activate "$ENV_DIR"
# module load cuda    # only if numba cannot find libnvvm / libcudart from the environment

cd "$REPO"
echo "node: $(hostname)  job: ${SLURM_JOB_ID:-local}  threads: $NUMBA_NUM_THREADS"
nvidia-smi
conda list 2>/dev/null | grep -i -E "^(numba|cuda|nvidia|pynvjitlink|nvjitlink|cudatoolkit|numba-cuda)" || true

# The driver's PTX JIT only accepts PTX up to the version of the driver's CUDA release. If numba's NVVM is newer than
# the driver (error CUDA_ERROR_UNSUPPORTED_PTX_VERSION), link with nvJitLink instead (bundled ptxas -> cubin):
#   pip install pynvjitlink-cu12   (or -cu13, matching the toolkit in the environment), then submit with ZMORPH_NVJITLINK=1
if [ "${ZMORPH_NVJITLINK:-0}" = "1" ]; then export NUMBA_CUDA_ENABLE_PYNVJITLINK=1; fi

python - <<'PY'
import sys
from numba import cuda
if not cuda.is_available():
    sys.exit("numba cannot see a CUDA device. Install numba CUDA support in the environment, e.g. "
             "conda install -c conda-forge numba cuda-nvcc cuda-cudart  (then resubmit).")
print("numba CUDA device:", cuda.get_current_device().name)
PY

# 1. the geometry cache (git-ignored); needs the committed data/tile_*.npz
if [ ! -f results/synapse_source/geom_170398_60741.npz ]; then
    python examples/make_geom_cache.py
fi

OUT="results/gpu_bench/bench_${SLURM_JOB_ID:-local}.txt"

# 2. kernel logic tests on the real device
python -m pytest tests/test_bd_gpu.py -q 2>&1 | tee "$OUT" || echo "GPU kernel tests FAILED (see above); continuing to the benchmark" | tee -a "$OUT"

# 3. throughput on CPU and GPU, and the sink-problem cross-check
python examples/bench_bd_gpu.py --ns 1000 10000 100000 1000000 --steps 2000 --check-n 100000 2>&1 | tee -a "$OUT"
