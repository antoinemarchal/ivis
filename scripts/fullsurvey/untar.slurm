#!/bin/bash
#SBATCH --job-name=untar
#SBATCH --output=untar_%j.out
#SBATCH --error=untar_%j.err
#SBATCH --clusters=totoro
#SBATCH --partition=def
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --time=08:00:00

# Go to the directory where the job was submitted
cd "$SLURM_SUBMIT_DIR"

# --- Conda setup ---
source /totoro/anmarchal/library/miniforge3/etc/profile.d/conda.sh

# Activate the IVIS environment
conda activate ivis

echo "Starting untar job on $(hostname)"
echo "Using ${SLURM_CPUS_PER_TASK} cores"
date

# Run your Python untar script (parallelism = Slurm CPUS)
python3 untar.py --nproc "$SLURM_CPUS_PER_TASK" > untar_${SLURM_JOB_ID}.log 2>&1

echo "Finished at $(date)"
date
