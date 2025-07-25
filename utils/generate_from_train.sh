#!/bin/bash

# --- 設定 ---
# 変換元のディレクトリ名
SOURCE_DIR="scripts_train"
# 変換先のディレクトリ名と言葉のリスト
TARGETS=("ring" "cp")

# --- メイン処理 ---
# スクリプトがあるディレクトリに移動
cd "$(dirname "$0")" || exit

# 変換元ディレクトリが存在するかチェック
if [ ! -d "$SOURCE_DIR" ]; then
  echo "エラー: 変換元のディレクトリ '$SOURCE_DIR' が見つかりません。"
  exit 1
fi

for target in "${TARGETS[@]}"; do
  TARGET_DIR="scripts_${target}"
  echo "--- '$target' バージョンの生成を開始します ---"

  # 1. ディレクトリを複製
  echo "ディレクトリを複製中: $SOURCE_DIR -> $TARGET_DIR"
  rm -rf "$TARGET_DIR"
  cp -r "$SOURCE_DIR" "$TARGET_DIR"

  # 2. ファイル名を一括で置換
  echo "ファイル名を置換中..."
  pushd "$TARGET_DIR" > /dev/null
  for file in *train*; do
    # bashの文字列置換機能を利用
    mv -- "$file" "${file//train/$target}"
  done
  popd > /dev/null

  # 3. ファイルの中身を置換
  echo "ファイルの内容を置換中..."
  find "$TARGET_DIR" -type f -name "*.sh" -print0 | while IFS= read -r -d $'\0' file; do
    # const"train" を避けるため、文脈を限定して置換する
    sed -i "s/scripts_train/scripts_${target}/g" "$file"
    sed -i "s/_train_/_${target}_/g" "$file"
    
    # SBATCHのジョブ名やログファイル名
    sed -i "s/--job-name=train/--job-name=${target}/g" "$file"
    sed -i "s/logs\/train_/logs\/${target}_/g" "$file"
    
    # 実行スクリプト内のコメントやパス
    if [ "$target" == "cp" ]; then
      # CPの場合は大文字にする
      sed -i "s/Train(Rank=/CP(Rank=/g" "$file"
    else
      # Ringの場合は先頭大文字
      sed -i "s/Train(Rank=/Ring(Rank=/g" "$file"
    fi

    # 最も重要なメソッド名の置換
    sed -i "s/--tf_method         \"train\"/--tf_method         \"${target}\"/" "$file"
  done

  echo "--- '$target' バージョンの生成が完了しました ---"
  echo
done

# run_all.sh と generate_rank3_scripts.sh について注意喚起
echo "注意: 'run_all.sh' と 'generate_rank3_scripts.sh' は"
echo "内容に応じて手動での修正が必要な場合があります。"