#!/bin/bash
#SBATCH --job-name=fullsurvey
#SBATCH --output=main_%j.out
#SBATCH --error=main_%j.err
#SBATCH --clusters=totoro
#SBATCH --partition=def
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=08:00:00

# Go to the directory where the job was submitted
cd "$SLURM_SUBMIT_DIR"

# --- Conda setup ---
source /totoro/anmarchal/library/miniforge3/etc/profile.d/conda.sh

# Activate the IVIS environment
conda activate ivis

echo "Starting main IViS job on $(hostname)"
echo "Using ${SLURM_CPUS_PER_TASK} cores"
date

# Run your Python untar script (parallelism = Slurm CPUS)
python3 main_fullsurvey.py --nproc "$SLURM_CPUS_PER_TASK" > main_${SLURM_JOB_ID}.log 2>&1

echo "Finished at $(date)"
date
