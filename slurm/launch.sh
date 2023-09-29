#!/bin/bash

export PROJECT_NAME="squared-circuits"
export PYTHONPATH=${PYTHONPATH:-src}

# These flags need to be updated accordingly:
# SCRATCH_DIR: a directory within the local filesystem of a node
# EXPS_ID: some identifier for the experiments
# VENV_PATH: the path containing the pip virtual environment
export SCRATCH_DIR=${SCRATCH_DIR:-/disk/scratch_big/$USER}
export EXPS_ID=${EXPS_ID:-exps}
export VENV_PATH=${VENV_PATH:-venv}

# The Slurm partition to use, e.g.,
#PARTITION=PGR-Standard
PARTITION=${PARTITION:-}
# An optional list of Slurm node to exclude, e.g.,
#EXCL_NODES=${EXCL_NODES:-busynode[01-07]}
EXCL_NODES=${EXCL_NODES:-}
# An optional list of Slurm node to allow
LIST_NODES=${LIST_NODES:-}
# The maximum number of parallel jobs to dispatch
MAX_PARALLEL_JOBS=20

# Resources and maximum execution time
NUM_CPUS=3
NUM_GPUS=1
TIME=48:00:00

JOB_NAME="$PROJECT_NAME-$EXPS_ID"
OUTPUT="slurm/logs/$JOB_NAME-%j.out"
EXPS_FILE="$1"
NUM_EXPS=`cat ${EXPS_FILE} | wc -l`

sbatch --job-name $JOB_NAME --output "$OUTPUT" --partition "$PARTITION" \
  --nodes 1 --ntasks 1 --cpus-per-task $NUM_CPUS --gres=gpu:$NUM_GPUS \
  --time $TIME --exclude "$EXCL_NODES" \
  --array=1-${NUM_EXPS}%${MAX_PARALLEL_JOBS} \
  slurm/run.sh "$EXPS_FILE"
