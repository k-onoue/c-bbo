#!/bin/bash -l
#SBATCH --job-name=bo_array_exp     # Job name
#SBATCH --partition=cluster_short   # Queue
#SBATCH --array=0-9                 # Array 0–9

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=01:00:00

#SBATCH --output=logs/experiment_%A_%a.out
#SBATCH --error=logs/experiment_%A_%a.err

# --- Check for arguments ---
if [ -z "$1" ] || [ -z "$2" ] || [ -z "$3" ]; then
  echo "Error: Missing arguments."
  echo "Usage: sbatch $0 <lambda_value> <shared_timestamp>"
  exit 1
fi
LAMBDA_VAL=$1
TIMESTAMP=$2    # 2番目の引数として共有タイムスタンプを受け取る
SAVEDIR=$3    # 3番目の引数として保存ディレクトリを受け取る example: "results_v0"

### --- Prepare environment --- ###
# source /path/to/venv/bin/activate   # if you need a virtualenv

### --- Seed and timestamp --- ###
SEED=$SLURM_ARRAY_TASK_ID
# ↓↓↓この行を削除↓↓↓
# TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# Use lambda value in the directory structure for clarity
TIMESTAMP_DIR="$SAVEDIR/${TIMESTAMP}"
mkdir -p logs "$TIMESTAMP_DIR/plots"

# cpコマンドはファイル名を指定してコピーするように修正
cp "$0" "$TIMESTAMP_DIR/run_array.sh_snapshot"


echo "Starting experiment for seed $SEED at $TIMESTAMP with lambda $LAMBDA_VAL..."

### --- Build argument array --- ###
declare -a FLAGS=(
  # --- General parameters ---
  --timestamp        "$TIMESTAMP"  # ここは変更なし (受け取ったTIMESTAMP変数を使う)
  --seed             "$SEED"
  --iter_bo          500
  --function         "ackley"
  --dimension        2
  # ↓ 必要時にアンコメント
  #--map_option       1       # For 'warcraft'
  --constraint               # Use constraint
  #--direction                # Maximize objective

  # --- TF-specific arguments ---
  --tf_method         "train"
  --tf_rank           6
  --tf_lr             0.01
  --tf_max_iter       1000
  --tf_tol            1e-5
  --tf_reg_lambda     1e-3
  --tf_constraint_lambda "$LAMBDA_VAL"

  # --- Sampler parameters ---
  --decomp_iter_num   10
  --mask_ratio        1.0
  #--include_observed_points  # Uncomment to activate
  #--unique_sampling          # Uncomment to activate
  --n_startup_trials  1
  --acqf_dist         "n"

  # --- Save directory ---
  --base_dir          "$SAVEDIR"
  --plot_save_dir     "$TIMESTAMP_DIR/plots"
)

### --- Run experiment --- ###
python experiments/tf_continual.py "${FLAGS[@]}"

echo "Experiment finished for seed $SEED."