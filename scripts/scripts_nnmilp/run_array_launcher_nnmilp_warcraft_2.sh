#!/bin/bash

# --- Get the directory where this script is located ---
SCRIPT_DIR=$(cd -- "$(dirname -- "$0")" &>/dev/null && pwd)

# --- Define the project root relative to this script's location ---
# Assumes the parent directory (../) is the project root
PROJECT_ROOT=$(cd -- "$SCRIPT_DIR/../" &>/dev/null && pwd)


# --- Define a shared timestamp and save destination for this entire launch ---
LAUNCH_TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
# --- Change the save destination to an absolute path from the project root ---
SAVEDIR="$PROJECT_ROOT/results_warcraft_2_nnmilp_${LAUNCH_TIMESTAMP}"

# --- Create the destination directory ---
mkdir -p "$SAVEDIR"

# --- Copy a snapshot of this launcher itself ---
cp "$SCRIPT_DIR/$(basename "$0")" "$SAVEDIR/launcher_nnmilp.sh_snapshot"

# --- List the epoch values you want to try, separated by spaces ---
EPOCHS_VALUES=(100 300 1000 5000 10000 25000)

# --- Manage delay in minutes ---
DELAY_MINUTES=0
DELAY_INTERVAL_MINUTES=1 # Time interval between each job submission (minutes)

echo "Project Root: $PROJECT_ROOT"
echo "Submitting jobs with a ${DELAY_INTERVAL_MINUTES}-minute delay between each parameter set..."

for epochs in "${EPOCHS_VALUES[@]}"; do
  # Calculate a future time
  FUTURE_TIME="now + $DELAY_MINUTES minutes"
  
  # 1. Generate the time for sbatch's --begin option
  BEGIN_TIME=$(date -d "$FUTURE_TIME" +%Y-%m-%dT%H:%M:%S)
  
  # 2. Generate the shared timestamp to use in the directory name, for the same future time
  SHARED_TIMESTAMP=$(date -d "$FUTURE_TIME" +%Y%m%d_%H%M%S)
  
  echo "Submitting jobs for epochs = $epochs to start at $BEGIN_TIME (File Timestamp: $SHARED_TIMESTAMP)"
  
  # Submit the experiment script with sbatch
  # --chdir fixes the job's execution location to the project root, and the script path is specified as an absolute path
  # Arguments: <epochs_value> <shared_timestamp> <save_dir>
  sbatch --begin="$BEGIN_TIME" \
         --chdir="$PROJECT_ROOT" \
         "$PROJECT_ROOT/scripts/scripts_nnmilp/run_array_nnmilp_warcraft_2.sh" "$epochs" "$SHARED_TIMESTAMP" "$SAVEDIR"
  
  # Increase the delay for the next job
  DELAY_MINUTES=$((DELAY_MINUTES + DELAY_INTERVAL_MINUTES))
done

echo "All jobs have been submitted."