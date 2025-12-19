#!/bin/bash
#SBATCH --job-name=fullsurvey
#SBATCH --output=main_%j.log
#SBATCH --error=main_%j.log
#SBATCH --mail-user=antoine.marchal@cnrs.fr --mail-type=ALL

#SBATCH -M gpu
#SBATCH -A ens
#SBATCH -p ens
#SBATCH --qos=gpu_ens_long

#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 24
#SBATCH --gres=gpu:1
#SBATCH --time=2-00:00:00

set -euo pipefail
export PYTHONUNBUFFERED=1

cd "$SLURM_SUBMIT_DIR"

source /totoro/anmarchal/library/miniforge3/etc/profile.d/conda.sh
conda activate ivis

echo "Starting on $(hostname)"
echo "PWD=$(pwd)"
echo "CPUs: ${SLURM_CPUS_PER_TASK:-?}  GPUs: ${SLURM_GPUS:-?}"
which python3
python3 -V
nvidia-smi
date

srun python3 -u main_fullsurvey.py --nproc "${SLURM_CPUS_PER_TASK}"

echo "Finished at $(date)"
