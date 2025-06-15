#!/bin/bash -l
#SBATCH --job-name=bo_benchmark       # Job name
#SBATCH --partition=cluster_short     # Queue
#SBATCH --array=0-9                   # Array 0–9 for seeds

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=01:00:00

#SBATCH --output=logs/benchmark_%A_%a.out
#SBATCH --error=logs/benchmark_%A_%a.err

# --- Check for arguments ---
if [ -z "$1" ] || [ -z "$2" ] || [ -z "$3" ]; then
  echo "Error: Missing arguments."
  echo "Usage: sbatch $0 <n_init_paths> <shared_timestamp> <save_dir>"
  exit 1
fi
N_PATHS_VAL=$1
TIMESTAMP=$2
SAVEDIR=$3      # 3番目の引数として親ディレクトリを受け取る

### --- Prepare environment --- ###
# source /path/to/venv/bin/activate   # if you need a virtualenv

### --- Seed and directories --- ###
SEED=$SLURM_ARRAY_TASK_ID

# ↓↓↓保存先ディレクトリの構造を見本に合わせて修正↓↓↓
# 親ディレクトリはlauncherから受け取ったSAVEDIRを使い、
# その下にタイムスタンプ（パラメータグループごと）のディレクトリを作成
TIMESTAMP_DIR="$SAVEDIR/${TIMESTAMP}"
mkdir -p logs "$TIMESTAMP_DIR/plots"

# この実行スクリプト自身のスナップショットをコピー
cp "$0" "$TIMESTAMP_DIR/run_array_gp.sh.snapshot"


echo "Starting experiment for n_paths=$N_PATHS_VAL, seed=$SEED at $TIMESTAMP"

### --- Build argument array for Python script --- ###
declare -a FLAGS=(
  --timestamp        "$TIMESTAMP"
  --seed             "$SEED"
  --iter_bo          500
  --function         "ackley"
  --dimension        2
  --sampler          "gp"
  --n_startup_trials 1
  # --acq_maximize
  
  # launcherから渡された可変パラメータ
  --n_init_violation_paths "$N_PATHS_VAL"
  
  # 結果保存用（Pythonスクリプト側でbase_dirを正しく参照しているか確認してください）
  --base_dir          "$SAVEDIR"
  --plot_save_dir     "$TIMESTAMP_DIR/plots"
)

### --- Run experiment --- ###
# Pythonスクリプトのパスを環境に合わせて修正してください
python experiments/benchmark-constrained.py "${FLAGS[@]}"

echo "Experiment finished for n_paths=$N_PATHS_VAL, seed=$SEED."