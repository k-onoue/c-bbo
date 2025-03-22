#!/bin/bash

# logs ディレクトリのパス
LOG_DIR="logs"

# 移動するファイルの一覧を取得
log_files=($LOG_DIR/*.log)

# ファイルがない場合は終了
if [ ${#log_files[@]} -eq 0 ]; then
  echo "ログファイルが見つかりません。"
  exit 1
fi

# ファイル名から日付を抽出し、ソート
dates=($(for file in "${log_files[@]}"; do
  basename "$file" | cut -d'_' -f1 | tr -d '-'
done | sort))

# 最も古い日付と最も新しい日付を取得
oldest_date=${dates[0]}
newest_date=${dates[${#dates[@]}-1]}  # 最後の要素を取得

oldest_date_formatted=$(echo $oldest_date | sed -E 's/(.{4})(.{2})(.{2})/\1-\2-\3/')
newest_date_formatted=$(echo $newest_date | sed -E 's/(.{4})(.{2})(.{2})/\1-\2-\3/')

# 新しいディレクトリの作成
new_dir="${LOG_DIR}/${oldest_date_formatted}_${newest_date_formatted}"
mkdir -p "$new_dir"

# ログファイルを新しいディレクトリに移動
for file in "${log_files[@]}"; do
  mv "$file" "$new_dir/"
done

echo "All done!"
