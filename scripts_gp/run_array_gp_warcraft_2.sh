#!/bin/bash -l
#SBATCH --job-name=gp_warcraft_2        # Job name
#SBATCH --partition=cluster_short       # Queue
#SBATCH --array=0-9                     # Array 0–9 for seeds

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=01:00:00

#SBATCH --output=logs/gp_warcraft_2_%A_%a.out
#SBATCH --error=logs/gp_warcraft_2_%A_%a.err

# --- Check for arguments ---
if [ -z "$1" ] || [ -z "$2" ] || [ -z "$3" ]; then
  echo "Error: Missing arguments."
  echo "Usage: sbatch $0 <n_init_paths> <shared_timestamp> <save_dir>"
  exit 1
fi
N_PATHS_VAL=$1
TIMESTAMP=$2
SAVEDIR=$3

### --- Prepare environment --- ###
# source /path/to/your/venv/bin/activate

### --- Seed and directories --- ###
SEED=$SLURM_ARRAY_TASK_ID
TIMESTAMP_DIR="$SAVEDIR/${TIMESTAMP}"
mkdir -p logs "$TIMESTAMP_DIR/plots"

# 配列ジョブの最初のタスク（ID=0）が代表してスナップショットをコピー
if [ "$SLURM_ARRAY_TASK_ID" -eq 0 ]; then
    cp "$0" "$TIMESTAMP_DIR/run_array_gp_warcraft_2.sh_snapshot"
fi

echo "Starting Warcraft 2 GP experiment for n_paths=$N_PATHS_VAL, seed=$SEED at $TIMESTAMP..."

### --- Build argument array --- ###
# Pythonスクリプトに渡す引数
declare -a FLAGS=(
  --timestamp              "$TIMESTAMP"
  --seed                   "$SEED"
  --base_dir               "$SAVEDIR"
  --plot_save_dir          "$TIMESTAMP_DIR/plots"

  # 実験設定
  --function               "warcraft"
  --map_option             2          # Warcraft 2 を指定
  --constrain              # warcraftは制約付き問題のため指定
  --sampler                "gp"
  --iter_bo                500
  --n_startup_trials       1
  
  # ランチャーから渡された可変パラメータ
  --n_init_violation_paths "$N_PATHS_VAL"
)

### --- Run experiment --- ###
# Pythonスクリプトのパスを指定 (experiments/benchmark.py)
python experiments/benchmark.py "${FLAGS[@]}"

echo "Experiment finished for n_paths=$N_PATHS_VAL, seed=$SEED."```
