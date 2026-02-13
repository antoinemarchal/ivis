#!/bin/bash
#SBATCH --job-name=fullsurvey
#SBATCH --output=main_%j.log
#SBATCH --error=main_%j.log
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
echo "Activating conda env..."
time conda activate ivis
echo "Conda env: ${CONDA_DEFAULT_ENV:-unset}"

which python3
python3 -V

echo "Starting main LINEAR job on $(hostname)"
echo "Using ${SLURM_CPUS_PER_TASK} cores"
date

# Run your Python untar script (parallelism = Slurm CPUS)
python3 linear_mosaic_parallel.py --nproc "$SLURM_CPUS_PER_TASK" > main_${SLURM_JOB_ID}.log 2>&1

echo "Finished at $(date)"
date
