#!/bin/bash

RESULTS_PATH="$SCRATCH_DIR/$SLURM_JOB_ID"
DESTINATION_PATH="$HOME/$PROJECT_NAME"
TBOARD_DIR="$RESULTS_PATH/tboard-runs/$EXPS_ID"
CHECKPOINT_DIR="$RESULTS_PATH/checkpoints/$EXPS_ID"

# Create local directories where to save model checkpoints and tensorboard logs
mkdir -p "$DESTINATION_PATH/tboard-runs" || exit 1
mkdir -p "$DESTINATION_PATH/checkpoints" || exit 1
mkdir -p "$RESULTS_PATH" || exit 1

# Get the command to run
COMMAND="`sed \"${SLURM_ARRAY_TASK_ID}q;d\" $1`"

# Activate the virtual environment, and launch the command.
# Then, rsync is used to copy the model checkpoints and tensorboard logs from the
# nodes local disk to the shared disk.
source "$VENV_PATH/bin/activate" && \
  $COMMAND --device cuda --tboard-path "$TBOARD_DIR" --checkpoint-path "$CHECKPOINT_DIR" && \
  deactivate && \
  rsync -r -a --verbose --ignore-existing "$TBOARD_DIR" "$DESTINATION_PATH/tboard-runs/" && \
  rsync -r -a --verbose --ignore-existing "$CHECKPOINT_DIR" "$DESTINATION_PATH/checkpoints/"

# Cleanup before exiting
rm -rf "$RESULTS_PATH"
rmdir "$SCRATCH_DIR"
