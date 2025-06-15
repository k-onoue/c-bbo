#!/bin/bash

# --- このランチャー実行全体で共有するタイムスタンプと保存先を定義 ---
LAUNCH_TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
SAVEDIR="results_benchmark_${LAUNCH_TIMESTAMP}"

# --- 保存先ディレクトリを作成し、このランチャー自身のスナップショットをコピー ---
mkdir -p "$SAVEDIR"
cp "$0" "$SAVEDIR/launcher.sh_snapshot"


# --- ここに試したい n_init_violation_paths の値を記述 ---
N_INIT_PATHS_VALUES=(0 50 100 200 300 500 1000 2000 3000 3500)
# --- 実行するスクリプトのファイル名を指定 ---
SCRIPT_TO_RUN="run_array_gp.sh"
# --- 各ジョブ投入の時間間隔（分） ---
DELAY_INTERVAL_MINUTES=1

# --- 遅延を分単位で管理 ---
DELAY_MINUTES=0

echo "Submitting benchmark jobs. Results will be saved in: $SAVEDIR"
echo "Submitting with a ${DELAY_INTERVAL_MINUTES}-minute delay between each..."

for n_paths in "${N_INIT_PATHS_VALUES[@]}"; do
  # ↓↓↓タイムスタンプの同期処理を修正↓↓↓
  # 基準となる未来の時刻を定義
  FUTURE_TIME="now + $DELAY_MINUTES minutes"
  
  # 1. sbatchの--beginオプション用の時刻を生成
  BEGIN_TIME=$(date -d "$FUTURE_TIME" +%Y-%m-%dT%H:%M:%S)
  
  # 2. ディレクトリ名やファイル名で使う共有タイムスタンプを、同じ未来の時刻で生成
  SHARED_TIMESTAMP=$(date -d "$FUTURE_TIME" +%Y%m%d_%H%M%S)
  
  # ↑↑↑タイムスタンプの同期処理を修正↑↑↑

  echo "Submitting job for n_paths = $n_paths (timestamp: $SHARED_TIMESTAMP) to start at $BEGIN_TIME"
  
  # sbatchで実験スクリプトを投入。引数に「ユニークなSAVEDIR」も渡す
  sbatch --begin="$BEGIN_TIME" "$SCRIPT_TO_RUN" "$n_paths" "$SHARED_TIMESTAMP" "$SAVEDIR"
  
  # 次のジョブの遅延を増やす
  DELAY_MINUTES=$((DELAY_MINUTES + DELAY_INTERVAL_MINUTES))
done

echo "All jobs have been submitted."