#!/bin/bash
#SBATCH --job-name=zmorph_smallsink_gpu
#SBATCH --account=elbert
#SBATCH --partition=gpu-a40
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=0-01:00:00
#SBATCH --output=logs/zmorph_smallsink_gpu_%j.out
#SBATCH --error=logs/zmorph_smallsink_gpu_%j.err

# Cross-check the GPU against the CPU small-sink results (site 0: R=100 nm -> 0.32-0.36, R=200 nm -> 0.63-0.65 for static
# w = 12-40 nm; breathing 0.325 and 0.632), with 20000 tracers x 2 seeds. Submit from the repo root.
set -eo pipefail
mkdir -p logs
export OMP_NUM_THREADS=1 MPLBACKEND=Agg PYTHONUNBUFFERED=1
CONDA_ROOT=/gscratch/elbert/miniconda3
source "$CONDA_ROOT/etc/profile.d/conda.sh"
export LD_LIBRARY_PATH="$CONDA_ROOT/envs/rr_env2/lib:${LD_LIBRARY_PATH:-}"
conda activate "$CONDA_ROOT/envs/rr_env2"
cd "${ZMORPH_REPO:-/gscratch/elbert/git/zmorph}"
[ -f results/synapse_source/geom_170398_60741.npz ] || python examples/make_geom_cache.py
python examples/run_synapse_smallsink.py --gpu --sites 1 --rs 100 200 --ws 20 --n 20000 --seeds 2 --phases 3 --tmax 1e6 --tag _gpu
