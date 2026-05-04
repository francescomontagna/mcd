#!/bin/bash

#----------------------------------
# MCD paper experiments (CPU)
# Array: one task per n-envs value
#   task 0 -> n-envs 3
#   task 1 -> n-envs 5
#   task 2 -> n-envs 7
#----------------------------------

#SBATCH --job-name=mcd_paper
#SBATCH --output=/nfs/scistore19/locatgrp/fmontagn/slurm_outputs/%A_%a.out
#SBATCH --array=0-2

#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --time=8:00:00
#SBATCH --mem=10G

#SBATCH --mail-user=fmontagn@ist.ac.at
#SBATCH --mail-type=END,FAIL

#SBATCH --export=NONE
unset SLURM_EXPORT_ENV

#-----------------------------
# Environment setup
#-----------------------------

PROJECT_DIR=/nfs/scistore19/locatgrp/fmontagn/mcd
cd "$PROJECT_DIR"

source ./venv/bin/activate

export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"

echo "Node: $(hostname)"
echo "CPUs per task: $SLURM_CPUS_PER_TASK"
echo "Array task ID: $SLURM_ARRAY_TASK_ID"
echo "Using Python: $(which python3)"
python3 --version

#-----------------------------
# Map array index -> n-envs
#-----------------------------

N_ENVS_VALUES=(3 5 7)
N_ENVS=${N_ENVS_VALUES[$SLURM_ARRAY_TASK_ID]}
ALGO_NAME="mcd-contrast-hessian"

echo "Running with --n-envs $N_ENVS"

#-----------------------------
# Run experiment
#-----------------------------

python3 scripts/run_experiment.py \
    --algorithms "$ALGO_NAME" \
    --n-reps 20 \
    --d-list 3,5,10,20 \
    --n-samples 500,1500,3000 \
    --n-envs "$N_ENVS" \
    --density 0.8,0.5,0.25,0.25 \
    --krr-reg 1e-3 \
    --eta-stein 1e-3 \
    --outdir "outputs/paper/$ALGO_NAME/nenvs${N_ENVS}" \
    --mechanism obs_id \
    --shift-all-eta \
    --min-env-diff 1.5 \

