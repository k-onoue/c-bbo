# create_experiment_scripts.py

import os
import argparse

# --- Template for the LAUNCHER script ---
LAUNCHER_TEMPLATE = """#!/bin/bash

# --- このスクリプト自身のディレクトリを取得 ---
SCRIPT_DIR=$(cd -- "$(dirname -- "$0")" &>/dev/null && pwd)

# --- プロジェクトルートを、このスクリプトの場所から相対的に定義する ---
# このスクリプトの親ディレクトリ (../) がプロジェクトルートになる
PROJECT_ROOT=$(cd -- "$SCRIPT_DIR/../" &>/dev/null && pwd)

# --- このランチャー実行全体で共有するタイムスタンプと保存先を定義 ---
LAUNCH_TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
# --- 保存先のパスをプロジェクトルートからの絶対パスに変更 ---
SAVEDIR="$PROJECT_ROOT/results_{function}_train_{rank}_${{LAUNCH_TIMESTAMP}}"

# --- 保存先ディレクトリを作成 ---
mkdir -p "$SAVEDIR"

# --- このランチャー自身のスナップショットをコピー ---
# --- $0は実行時のパスになるため、$SCRIPT_DIRを使って自身のフルパスを指定 ---
cp "$SCRIPT_DIR/$(basename "$0")" "$SAVEDIR/run_array_launcher_train_{rank}_{function}.sh_snapshot"

# --- ここに試したいラムダの値をスペース区切りで記述 ---
LAMBDA_VALUES=(1)

# --- 遅延を分単位で管理 ---
DELAY_MINUTES=0
DELAY_INTERVAL_MINUTES=1 # 各ジョブ投入の時間間隔（分）

echo "Project Root: $PROJECT_ROOT"
echo "Saving results in: $SAVEDIR"
echo "Submitting jobs with a ${{DELAY_INTERVAL_MINUTES}}-minute delay between each parameter set..."

for lambda in "${{LAMBDA_VALUES[@]}}"; do
  # 未来の時刻を計算
  FUTURE_TIME="now + $DELAY_MINUTES minutes"
  
  # sbatchの--beginオプション用の時刻を生成
  BEGIN_TIME=$(date -d "$FUTURE_TIME" +%Y-%m-%dT%H:%M:%S)
  
  # ディレクトリ名で使う共有タイムスタンプを、同じ未来の時刻で生成
  SHARED_TIMESTAMP=$(date -d "$FUTURE_TIME" +%Y%m%d_%H%M%S)
  
  echo "Submitting jobs for lambda = $lambda to start at $BEGIN_TIME (File Timestamp: $SHARED_TIMESTAMP)"
  
  # sbatchで実験スクリプトを投入
  # 実行スクリプトは scripts_train/ ディレクトリにあると仮定
  sbatch --begin="$BEGIN_TIME" \\
         --chdir="$PROJECT_ROOT" \\
         "$PROJECT_ROOT/scripts_train/run_array_train_{rank}_{function}.sh" "$lambda" "$SHARED_TIMESTAMP" "$SAVEDIR"
  
  # 次のジョブの遅延を増やす
  DELAY_MINUTES=$((DELAY_MINUTES + DELAY_INTERVAL_MINUTES))
done

echo "All jobs have been submitted."
"""

# --- Template for the SBATCH script ---
SBATCH_TEMPLATE = """#!/bin/bash -l
#SBATCH --job-name=train_{rank}_{function}     # Job name
#SBATCH --partition=cluster_short     # Queue
#SBATCH --array=0-9                   # Array 0–9 for seeds

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=02:00:00

#SBATCH --output=logs/train_{rank}_{function}_%A_%a.out
#SBATCH --error=logs/train_{rank}_{function}_%A_%a.err

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
TIMESTAMP_DIR="$SAVEDIR/${{TIMESTAMP}}"
mkdir -p logs "$TIMESTAMP_DIR/plots"

# 配列ジョブの最初のタスク（ID=0）が代表してスナップショットをコピー
if [ "$SLURM_ARRAY_TASK_ID" -eq 0 ]; then
    cp "$0" "$TIMESTAMP_DIR/run_array_train_{rank}_{function}.sh_snapshot"
fi

echo "Starting {function_title} Train(Rank={rank}) experiment for seed $SEED at $TIMESTAMP with lambda $LAMBDA_VAL..."

### --- Build argument array --- ###
declare -a FLAGS=(
  # --- General parameters ---
  --timestamp        "$TIMESTAMP"
  --seed             "$SEED"
  --base_dir          "$SAVEDIR"
  --plot_save_dir     "$TIMESTAMP_DIR/plots"
  --function         "{function}"
  --constrain               # 制約ありバージョンを実行
  --iter_bo          500
  --n_startup_trials  1

  # --- TF-specific arguments ---
  --tf_constraint_lambda "$LAMBDA_VAL" # ランチャーから渡された可変パラメータ
  --tf_method         "train"
  --tf_rank           {rank}               
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
python experiments/tf_continual.py "${{FLAGS[@]}}"

echo "Experiment finished for seed $SEED with lambda $LAMBDA_VAL."
"""

def create_scripts(rank, function, output_dir):
    """Generates the launcher and sbatch scripts for a given rank and function."""
    
    # --- Prepare placeholders ---
    placeholders = {
        "rank": str(rank),
        "function": function,
        "function_title": function.replace('_', ' ').title() # For echo statements
    }
    
    # --- Create Launcher Script ---
    launcher_filename = f"run_array_launcher_train_{rank}_{function}.sh"
    launcher_content = LAUNCHER_TEMPLATE.format(**placeholders)
    launcher_path = os.path.join(output_dir, launcher_filename)
    
    with open(launcher_path, 'w') as f:
        f.write(launcher_content)
    os.chmod(launcher_path, 0o755) # Make it executable
    
    print(f"✅ Created Launcher script: {launcher_path}")

    # --- Create Sbatch Script ---
    sbatch_filename = f"run_array_train_{rank}_{function}.sh"
    sbatch_content = SBATCH_TEMPLATE.format(**placeholders)
    sbatch_path = os.path.join(output_dir, sbatch_filename)

    with open(sbatch_path, 'w') as f:
        f.write(sbatch_content)
    os.chmod(sbatch_path, 0o755) # Make it executable
        
    print(f"✅ Created Sbatch script:   {sbatch_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate experiment scripts for a specific rank and function.")
    parser.add_argument("rank", type=int, help="The rank for the experiment (e.g., 2, 3, 4).")
    parser.add_argument("function", type=str, default="diabetes", help="The objective function name (e.g., 'diabetes', 'ackley').")
    parser.add_argument("--output_dir", type=str, default="./", 
                        help="The directory to save the generated scripts.")
    
    args = parser.parse_args()
    
    # --- Create output directory if it doesn't exist ---
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        print(f"📂 Created output directory: {args.output_dir}")
        
    create_scripts(args.rank, args.function, args.output_dir)
    
    print("\n🎉 Script generation complete!")
