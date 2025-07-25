#!/bin/bash -l
#SBATCH --job-name=ring_2_warcraft_2   # Job name
#SBATCH --partition=cluster_short     # Queue
#SBATCH --array=0-9                   # Array 0–9 for seeds

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=01:00:00

#SBATCH --output=logs/ring_2_warcraft_2_%A_%a.out
#SBATCH --error=logs/ring_2_warcraft_2_%A_%a.err

# --- Check for arguments ---
if [ -z "$1" ] || [ -z "$2" ] || [ -z "$3" ]; then
  echo "Error: Missing arguments."
  echo "Usage: sbatch $0 <lambda_value> <shared_timestamp> <save_dir>"
  exit 1
fi
LAMBDA_VAL=$1
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
    cp "$0" "$TIMESTAMP_DIR/run_array_ring_3_warcraft_2.sh_snapshot"
fi

echo "Starting Warcraft 2 Ring(Rank=3) experiment for seed $SEED at $TIMESTAMP with lambda $LAMBDA_VAL..."

### --- Build argument array --- ###
declare -a FLAGS=(
  # --- General parameters ---
  --timestamp        "$TIMESTAMP"
  --seed             "$SEED"
  --base_dir          "$SAVEDIR"
  --plot_save_dir     "$TIMESTAMP_DIR/plots"
  --function         "warcraft"
  --map_option       2
  --constrain               # 制約ありバージョンを実行
  --iter_bo          500
  --n_startup_trials  1

  # --- TF-specific arguments ---
  --tf_constraint_lambda "$LAMBDA_VAL" # ランチャーから渡された可変パラメータ
  --tf_method         "ring"
  --tf_rank           3               # ランクを2に固定
  --tf_lr             0.01
  --tf_max_iter       100000
  --tf_tol          1e-6
  --tf_reg_lambda     0

  # --- Sampler parameters ---
  --decomp_iter_num   10
  --mask_ratio        1.0
  --acqf_dist         "n"
)

### --- Run experiment --- ###
python experiments/tf_continual.py "${FLAGS[@]}"

echo "Experiment finished for seed $SEED with lambda $LAMBDA_VAL."