#!/bin/bash

#----------------------------------
# MCD paper experiments (CPU)
# Array: one task per (algorithm, mechanism, n-envs) triple
#   task_id = algo_idx * (n_mechanisms * n_n_envs)
#           + mechanism_idx * n_n_envs
#           + n_envs_idx
#
# Submit with:  bash run_paper_experiments.sh
# (the script computes --array automatically and re-submits itself)
#----------------------------------

#SBATCH --job-name=mcd_paper
#SBATCH --output=/nfs/scistore19/locatgrp/fmontagn/slurm_outputs/%A_%a.out

#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --time=8:00:00
#SBATCH --mem=10G

#SBATCH --mail-user=fmontagn@ist.ac.at
#SBATCH --mail-type=END,FAIL

#SBATCH --export=NONE
unset SLURM_EXPORT_ENV

#-----------------------------
# Sweep parameters
#-----------------------------

ALGORITHMS=("mcd-hsic")
MECHANISMS=("linear" "gp")
N_ENVS_VALUES=(3)

#-----------------------------
# Self-submit: if not inside SLURM, compute array range and resubmit
#-----------------------------

if [[ -z "$SLURM_ARRAY_TASK_ID" ]]; then
    N_TOTAL=$(( ${#ALGORITHMS[@]} * ${#MECHANISMS[@]} * ${#N_ENVS_VALUES[@]} ))
    exec sbatch --array=0-$(( N_TOTAL - 1 )) "$0" "$@"
fi

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
# Map array index -> (algorithm, mechanism, n-envs)
#-----------------------------

N_ALGORITHMS=${#ALGORITHMS[@]}
N_MECHANISMS=${#MECHANISMS[@]}
N_N_ENVS=${#N_ENVS_VALUES[@]}

ALGO_IDX=$(( SLURM_ARRAY_TASK_ID / (N_MECHANISMS * N_N_ENVS) ))
MECHANISM_IDX=$(( (SLURM_ARRAY_TASK_ID / N_N_ENVS) % N_MECHANISMS ))
N_ENVS_IDX=$(( SLURM_ARRAY_TASK_ID % N_N_ENVS ))

ALGO_NAME=${ALGORITHMS[$ALGO_IDX]}
MECHANISM=${MECHANISMS[$MECHANISM_IDX]}
N_ENVS=${N_ENVS_VALUES[$N_ENVS_IDX]}

echo "Running with --algorithms $ALGO_NAME --mechanism $MECHANISM --n-envs $N_ENVS"

#-----------------------------
# Run experiment
#-----------------------------

python3 scripts/run_experiment.py \
    --algorithms "$ALGO_NAME" \
    --n-reps 20 \
    --d-list 3,5,10,20 \
    --n-samples 1000,2000,3000 \
    --n-envs "$N_ENVS" \
    --density 0.8,0.5,0.25,0.25 \
    --krr-reg 1e-3 \
    --eta-stein 1e-3 \
    --outdir "outputs/paper/new_gp/$ALGO_NAME/nenvs${N_ENVS}" \
    --mechanism "$MECHANISM" \
    --shift-all-eta \

