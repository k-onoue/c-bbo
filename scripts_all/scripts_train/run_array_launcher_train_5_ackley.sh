#!/bin/bash

# --- このスクリプト自身のディレクトリを取得 ---
SCRIPT_DIR=$(cd -- "$(dirname -- "$0")" &>/dev/null && pwd)

# --- プロジェクトルートを、このスクリプトの場所から相対的に定義する ---
# このスクリプトの親ディレクトリ (../) がプロジェクトルートになる
PROJECT_ROOT=$(cd -- "$SCRIPT_DIR/../" &>/dev/null && pwd)

# --- このランチャー実行全体で共有するタイムスタンプと保存先を定義 ---
LAUNCH_TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
# --- 保存先のパスをプロジェクトルートからの絶対パスに変更 ---
SAVEDIR="$PROJECT_ROOT/results_ackley_train_5_${LAUNCH_TIMESTAMP}"

# --- 保存先ディレクトリを作成 ---
mkdir -p "$SAVEDIR"

# --- このランチャー自身のスナップショットをコピー ---
# --- $0は実行時のパスになるため、$SCRIPT_DIRを使って自身のフルパスを指定 ---
cp "$SCRIPT_DIR/$(basename "$0")" "$SAVEDIR/run_array_launcher_train_5_ackley.sh_snapshot"

# --- ここに試したいラムダの値をスペース区切りで記述 ---
LAMBDA_VALUES=(1 0)

# --- 遅延を分単位で管理 ---
DELAY_MINUTES=0
DELAY_INTERVAL_MINUTES=1 # 各ジョブ投入の時間間隔（分）

echo "Project Root: $PROJECT_ROOT"
echo "Saving results in: $SAVEDIR"
echo "Submitting jobs with a ${DELAY_INTERVAL_MINUTES}-minute delay between each parameter set..."

for lambda in "${LAMBDA_VALUES[@]}"; do
  # 未来の時刻を計算
  FUTURE_TIME="now + $DELAY_MINUTES minutes"
  
  # sbatchの--beginオプション用の時刻を生成
  BEGIN_TIME=$(date -d "$FUTURE_TIME" +%Y-%m-%dT%H:%M:%S)
  
  # ディレクトリ名で使う共有タイムスタンプを、同じ未来の時刻で生成
  SHARED_TIMESTAMP=$(date -d "$FUTURE_TIME" +%Y%m%d_%H%M%S)
  
  echo "Submitting jobs for lambda = $lambda to start at $BEGIN_TIME (File Timestamp: $SHARED_TIMESTAMP)"
  
  # sbatchで実験スクリプトを投入
  # 実行スクリプトは scripts_train/ ディレクトリにあると仮定
  sbatch --begin="$BEGIN_TIME" \
         --chdir="$PROJECT_ROOT" \
         "$PROJECT_ROOT/scripts_train/run_array_train_5_ackley.sh" "$lambda" "$SHARED_TIMESTAMP" "$SAVEDIR"
  
  # 次のジョブの遅延を増やす
  DELAY_MINUTES=$((DELAY_MINUTES + DELAY_INTERVAL_MINUTES))
done

echo "All jobs have been submitted."