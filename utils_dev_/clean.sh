#!/bin/bash

# 現在の時間を「年-月-日_時-分」の形式で取得
CURRENT_TIME=$(date +"%Y-%m-%d_%H-%M")

# アーカイブディレクトリのパスを設定
ARCHIVE_DIR="experiments/archive"

# アーカイブディレクトリが存在しない場合は作成
mkdir -p $ARCHIVE_DIR

# ルートディレクトリのrun_*.shファイルをアーカイブディレクトリに移動
for file in run_*.sh; do
    if [ -f "$file" ]; then
        # ファイル名に現在の日時を追加して移動
        mv "$file" "$ARCHIVE_DIR/${file%.sh}_$CURRENT_TIME.sh"
    fi
done

# 処理完了メッセージ
echo "All run_*.sh files have been moved to $ARCHIVE_DIR/ with the date and time (YYYY-MM-DD_HH-MM) appended."