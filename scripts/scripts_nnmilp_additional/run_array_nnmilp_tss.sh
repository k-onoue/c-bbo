#!/bin/bash -l
#SBATCH --job-name=nnmilp_tss    # Job name
#SBATCH --partition=cluster_short     # Queue
#SBATCH --array=0-9                   # Array 0–9 for seeds

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=01:00:00

#SBATCH --output=logs/nnmilp_tss%A_%a.out
#SBATCH --error=logs/nnmilp_tss%A_%a.err

# --- Check for arguments ---
if [ -z "$1" ] || [ -z "$2" ] || [ -z "$3" ]; then
  echo "Error: Missing arguments."
  echo "Usage: sbatch $0 <epochs_value> <shared_timestamp> <save_dir>"
  exit 1
fi
EPOCHS_VAL=$1
TIMESTAMP=$2
SAVEDIR=$3

### --- Prepare environment --- ###
# source /path/to/your/venv/bin/activate

### --- Seed and directories --- ###
SEED=$SLURM_ARRAY_TASK_ID
TIMESTAMP_DIR="$SAVEDIR/${TIMESTAMP}"
mkdir -p logs "$TIMESTAMP_DIR"

# スクリプトのスナップショットを保存 (アレイジョブの最初のタスクのみ実行)
if [ "$SLURM_ARRAY_TASK_ID" -eq 0 ]; then
    cp "$0" "$TIMESTAMP_DIR/run_array_nnmilp_tss.sh_snapshot"
fi

echo "Starting TSS NN+MILP experiment for seed $SEED at $TIMESTAMP with epochs = $EPOCHS_VAL..."

### --- Build argument array --- ###
MAP_OPTION=2
N_TRIALS=500
HIDDEN_DIM=16
N_STARTUP=50
FUNCTION="tss"  # Change this to the desired function

declare -a FLAGS=(
  --timestamp         "$TIMESTAMP"
  --seed              "$SEED"
  --n_trials          "$N_TRIALS"
  --function          "$FUNCTION"  # ここは実験する関数に応じて変更
  --map_option        "$MAP_OPTION"
  --base_dir          "$SAVEDIR"
  --constrain         # 制約ありバージョンを実行
  --epochs            "$EPOCHS_VAL"
  --hidden_dim        "$HIDDEN_DIM"
  --n_startup_trials  "$N_STARTUP"
)

### --- Run experiment --- ###
# Pythonスクリプトのパスとファイル名を指定
python experiments/nnmilp_additional.py "${FLAGS[@]}"

echo "Experiment finished for seed $SEED with epochs = $EPOCHS_VAL."