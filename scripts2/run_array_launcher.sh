#!/bin/bash

# --- このランチャー実行全体で共有するタイムスタンプと保存先を定義 ---
LAUNCH_TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
SAVEDIR="results_${LAUNCH_TIMESTAMP}"

# --- 保存先ディレクトリを作成 ---
mkdir -p "$SAVEDIR"

# --- このランチャー自身のスナップショットをコピー ---
cp "$0" "$SAVEDIR/launcher.sh_snapshot"

# --- ここに試したいラムダの値をスペース区切りで記述 ---
LAMBDA_VALUES=(10 5 1 0.5 0.1 0.01 0.001 0.0001)

# --- 遅延を分単位で管理 ---
DELAY_MINUTES=0
# --- 各ジョブ投入の時間間隔（分） ---
DELAY_INTERVAL_MINUTES=1 # ユーザーのスクリプトに合わせて1分に変更

echo "Submitting jobs with ${DELAY_INTERVAL_MINUTES}-minute delay..."

for lambda in "${LAMBDA_VALUES[@]}"; do
  # ↓↓↓ここから変更↓↓↓
  
  # 基準となる未来の時刻を定義
  FUTURE_TIME="now + $DELAY_MINUTES minutes"
  
  # 1. sbatchの--beginオプション用の時刻を生成 (例: 2025-06-14T14:30:00)
  BEGIN_TIME=$(date -d "$FUTURE_TIME" +%Y-%m-%dT%H:%M:%S)
  
  # 2. ディレクトリ名やファイル名で使う共有タイムスタンプを、同じ未来の時刻で生成 (例: 20250614_143000)
  SHARED_TIMESTAMP=$(date -d "$FUTURE_TIME" +%Y%m%d_%H%M%S)
  
  # ↑↑↑ここまで変更↑↑↑

  echo "Submitting job for lambda = $lambda to start at $BEGIN_TIME (Timestamp for files: $SHARED_TIMESTAMP)"
  
  # sbatchで実験スクリプトを投入。--beginで開始時刻を、引数でラムダの値と「同期された」タイムスタンプを渡す
  sbatch --begin="$BEGIN_TIME" run_array.sh "$lambda" "$SHARED_TIMESTAMP" "$SAVEDIR"
  
  # 次のジョブの遅延を増やす
  DELAY_MINUTES=$((DELAY_MINUTES + DELAY_INTERVAL_MINUTES))
done

echo "All jobs have been submitted."