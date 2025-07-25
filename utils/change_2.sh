#!/bin/bash

# --- 設定 ---
# 変更対象のディレクトリ
TARGET_DIRS=("scripts_cp_ablation" "scripts_ring_ablation" "scripts_train_ablation")

# 変更前の行の完全な文字列
OLD_LINE="LAMBDA_VALUES=(1 0)"

# 変更後の行の完全な文字列
NEW_LINE="LAMBDA_VALUES=(10 5 1 0.5 0.1 0.01 0.001 0.0001)"

echo "アブレーション用スクリプトの LAMBDA_VALUES を更新します。"
echo "--------------------------------------------------"

# 各対象ディレクトリに対してループ処理
for dir in "${TARGET_DIRS[@]}"; do
  # ディレクトリが存在するかチェック
  if [ ! -d "$dir" ]; then
    echo "注意: ディレクトリ '$dir' が見つかりません。スキップします。"
    continue
  fi
  
  echo "処理中: $dir"
  
  # ディレクトリ内の全シェルスクリプトを検索
  find "$dir" -type f -name "*.sh" -print0 | while IFS= read -r -d $'\0' file; do
    # ファイル内に変更対象の行が存在するかを固定文字列として(-F)確認
    if grep -qF -- "$OLD_LINE" "$file"; then
      echo "  -> 更新: $file"
      
      # sedで特定の行を完全に置換する
      # デリミタを'|'にすることで、スラッシュを含む文字列にも対応しやすくなります
      sed -i "s|${OLD_LINE}|${NEW_LINE}|" "$file"
    fi
  done
done

echo "--------------------------------------------------"
echo "処理が完了しました。"
