#!/bin/bash
#SBATCH --job-name=fullsurvey
#SBATCH --output=main_%j.log
#SBATCH --error=main_%j.log
#SBATCH --mail-user=antoine.marchal@cnrs.fr
#SBATCH --mail-type=ALL

#SBATCH -M gpu
#SBATCH -A ens
#SBATCH -p ens
#SBATCH --qos=gpu_ens_long

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:1
#SBATCH --time=2-00:00:00

#SBATCH --exclude=node0298

set -euo pipefail
export PYTHONUNBUFFERED=1

cd "$SLURM_SUBMIT_DIR"

echo "== JOB START =="
echo "JobID=$SLURM_JOB_ID"
echo "Starting on $(hostname)"
echo "PWD=$(pwd)"
echo "CPUs per task: ${SLURM_CPUS_PER_TASK:-?}"
echo "SLURM_JOB_GPUS: ${SLURM_JOB_GPUS:-unset}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-unset}"
date

source /totoro/anmarchal/library/miniforge3/etc/profile.d/conda.sh
echo "Activating conda env..."
time conda activate ivis
echo "Conda env: ${CONDA_DEFAULT_ENV:-unset}"

which python3
python3 -V

echo "== nvidia-smi =="
nvidia-smi || true

echo "== running =="
# Option 1 (simplest): run directly (recommended for single-task jobs)
python3 -u main_fullsurvey.py --nproc "${SLURM_CPUS_PER_TASK:-12}"

# Option 2 (if you want Slurm binding/enforcement), comment Option 1 and use:
# srun --cpu-bind=cores --accel-bind=gn python3 -u main_fullsurvey.py --nproc "${SLURM_CPUS_PER_TASK:-12}"

echo "Finished at $(date)"

# #!/bin/bash
# #SBATCH --job-name=fullsurvey
# #SBATCH --output=main_%j.log
# #SBATCH --error=main_%j.log
# #SBATCH --mail-user=antoine.marchal@cnrs.fr --mail-type=ALL

# #SBATCH -M gpu
# #SBATCH -A ens
# #SBATCH -p ens
# #SBATCH --qos=gpu_ens_long

# #SBATCH -N 1
# #SBATCH -n 1
# #SBATCH -c 12
# #SBATCH --gres=gpu:1
# #SBATCH --time=2-00:00:00

# #SBATCH --exclusive
# #SBATCH --nodelist=node0257

# set -euo pipefail
# export PYTHONUNBUFFERED=1

# cd "$SLURM_SUBMIT_DIR"

# source /totoro/anmarchal/library/miniforge3/etc/profile.d/conda.sh
# conda activate ivis

# echo "Starting on $(hostname)"
# echo "PWD=$(pwd)"
# echo "CPUs: ${SLURM_CPUS_PER_TASK:-?}  GPUs: ${SLURM_GPUS:-?}"
# which python3
# python3 -V
# nvidia-smi
# date

# srun python3 -u main_fullsurvey.py --nproc "${SLURM_CPUS_PER_TASK}"

# echo "Finished at $(date)"
