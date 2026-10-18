#!/bin/bash
#SBATCH --job-name=zmorph_smallsink_cpu
#SBATCH --account=elbert
#SBATCH --partition=compute               # change to your CPU partition if different
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=13
#SBATCH --mem=32G
#SBATCH --time=0-03:00:00
#SBATCH --output=logs/zmorph_smallsink_cpu_%j.out
#SBATCH --error=logs/zmorph_smallsink_cpu_%j.err

# CPU reference for the GPU check: site 0, R = 100 nm, static w = 20 nm with 20000 tracers x 2 seeds (the CPU runs with
# 800 tracers gave 0.319, the GPU 0.342). Compare with results/synapse_source/smallsink_gpu_*.md.
# Submit from the repo root:  sbatch slurm/smallsink_check_cpu.sbatch
set -eo pipefail
mkdir -p logs
export OMP_NUM_THREADS=1 MPLBACKEND=Agg PYTHONUNBUFFERED=1
export NUMBA_NUM_THREADS="${SLURM_CPUS_PER_TASK:-13}"
CONDA_ROOT=/gscratch/elbert/miniconda3
source "$CONDA_ROOT/etc/profile.d/conda.sh"
conda activate "$CONDA_ROOT/envs/rr_env2"
cd "${ZMORPH_REPO:-/gscratch/elbert/git/zmorph}"
echo "threads: $NUMBA_NUM_THREADS"
[ -f results/synapse_source/geom_170398_60741.npz ] || python examples/make_geom_cache.py
python examples/run_synapse_smallsink.py --modes static --sites 1 --rs 100 --ws 20 --n 20000 --seeds 2 --tmax 1e6 --tag _cpu20k
