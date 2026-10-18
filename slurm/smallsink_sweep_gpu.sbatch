#!/bin/bash
#SBATCH --job-name=zmorph_smallsink_sweep
#SBATCH --account=elbert
#SBATCH --partition=gpu-a40
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=0-03:00:00
#SBATCH --output=logs/zmorph_smallsink_sweep_%j.out
#SBATCH --error=logs/zmorph_smallsink_sweep_%j.err

# Static-gap sweep on the GPU: 2 sites x R = 100, 200, 300 nm x w = 12, 20, 30, 40 nm, 20000 tracers x 2 seeds each
# (about 80 s per run, 48 runs). Per-seed final captures are in the json (F_final_seeds) so the w effect can be tested
# against seed scatter. Breathing is left out (already resolved on the GPU: no effect within a point).
# Submit from the repo root:  sbatch slurm/smallsink_sweep_gpu.sbatch
set -eo pipefail
mkdir -p logs
export OMP_NUM_THREADS=1 MPLBACKEND=Agg PYTHONUNBUFFERED=1
CONDA_ROOT=/gscratch/elbert/miniconda3
source "$CONDA_ROOT/etc/profile.d/conda.sh"
export LD_LIBRARY_PATH="$CONDA_ROOT/envs/rr_env2/lib:${LD_LIBRARY_PATH:-}"
conda activate "$CONDA_ROOT/envs/rr_env2"
cd "${ZMORPH_REPO:-/gscratch/elbert/git/zmorph}"
[ -f results/synapse_source/geom_170398_60741.npz ] || python examples/make_geom_cache.py
python examples/run_synapse_smallsink.py --gpu --modes static --sites 2 --rs 100 200 300 --ws 12 20 30 40 \
    --n 20000 --seeds 2 --tmax 1e6 --tag _gpusweep
