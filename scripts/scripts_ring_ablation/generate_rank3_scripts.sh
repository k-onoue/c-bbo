# #!/bin/bash

# # --- 設定 ---
# # スクリプトが置かれているディレクトリ
# TARGET_DIR="scripts_ring"
# # 元となるランク
# SRC_RANK=2
# # 作成するランク
# DEST_RANK=3

# # --- スクリプトのメイン処理 ---
# echo "Starting script generation for Rank ${DEST_RANK}..."
# echo "Source Directory: ${TARGET_DIR}"
# echo "----------------------------------------"

# # ターゲットディレクトリが存在するか確認
# if [ ! -d "$TARGET_DIR" ]; then
#   echo "Error: Directory '$TARGET_DIR' not found."
#   exit 1
# fi

# # カウンターを初期化
# created_count=0

# # ランク2のスクリプトをループ処理
# # "run_" で始まるファイルのみを対象とし、"run_all.sh" などを除外
# for src_file in "$TARGET_DIR"/run_*_ring_${SRC_RANK}_*.sh; do

#   # マッチするファイルが一つもなかった場合のglobの挙動に対応
#   if [ ! -e "$src_file" ]; then
#     echo "No Rank ${SRC_RANK} scripts found to process."
#     break
#   fi

#   # 1. 新しいファイル名を生成
#   dest_file="${src_file/_ring_${SRC_RANK}_/_ring_${DEST_RANK}_}"

#   # 既にファイルが存在する場合はスキップ
#   if [ -e "$dest_file" ]; then
#     echo "Skipping: '$dest_file' already exists."
#     continue
#   fi

#   # ファイルをコピー
#   cp "$src_file" "$dest_file"
#   echo "Created: $dest_file"

#   # 2. ファイル内の文字列を置換 (sed -i で直接ファイルを編集)
#   sed -i \
#     -e "s/_ring_${SRC_RANK}_/_ring_${DEST_RANK}_/g" \
#     -e "s/--tf_rank           ${SRC_RANK}/--tf_rank           ${DEST_RANK}/" \
#     -e "s/Ring(Rank=${SRC_RANK})/Ring(Rank=${DEST_RANK})/" \
#     "$dest_file"

#   echo "Modified content for Rank ${DEST_RANK}."
#   created_count=$((created_count + 1))
# done

# echo "----------------------------------------"
# if [ "$created_count" -gt 0 ]; then
#   echo "Successfully created and modified ${created_count} file pairs for Rank ${DEST_RANK}."
# else
#   echo "No new files were created."
# fi
# echo "Script generation finished."


#!/bin/bash

# --- 設定 ---
# スクリプトが置かれているディレクトリ
TARGET_DIR="scripts_ring"
# 元となるランク (テンプレートとして使用)
SRC_RANK=2
# 作成したいランクのリスト (スペース区切り)
DEST_RANKS=(4 5 6)

# --- スクリプトのメイン処理 ---
echo "Starting script generation for multiple ranks..."
echo "Source Directory: ${TARGET_DIR}"
echo "Template Rank: ${SRC_RANK}"
echo "----------------------------------------"

# ターゲットディレクトリが存在するか確認
if [ ! -d "$TARGET_DIR" ]; then
  echo "Error: Directory '$TARGET_DIR' not found."
  exit 1
fi

# 生成するランクのリストをループ
for dest_rank in "${DEST_RANKS[@]}"; do
  echo ">>> Generating scripts for Rank ${dest_rank}..."
  
  # カウンターを初期化
  created_count=0

  # ランク2のスクリプトをループ処理
  # "run_" で始まるファイルのみを対象とする
  for src_file in "$TARGET_DIR"/run_*_ring_${SRC_RANK}_*.sh; do

    # マッチするファイルが一つもなかった場合のglobの挙動に対応
    if [ ! -e "$src_file" ]; then
      echo "No Rank ${SRC_RANK} scripts found to process. Aborting for this rank."
      break
    fi

    # 1. 新しいファイル名を生成
    # まず、ファイル名内のランク部分を置換
    dest_file_basename=$(basename "$src_file")
    new_basename="${dest_file_basename/_ring_${SRC_RANK}_/_ring_${dest_rank}_}"
    dest_file="$TARGET_DIR/$new_basename"

    # 既にファイルが存在する場合はスキップ
    if [ -e "$dest_file" ]; then
      echo "Skipping: '$dest_file' already exists."
      continue
    fi

    # ファイルをコピー
    cp "$src_file" "$dest_file"
    echo "  Created: $dest_file"

    # 2. ファイル内の文字列を置換 (sed -i で直接ファイルを編集)
    # 複数の置換ルールを適用
    sed -i \
      -e "s/_ring_${SRC_RANK}_/_ring_${dest_rank}_/g" \
      -e "s/--tf_rank           ${SRC_RANK}/--tf_rank           ${dest_rank}/" \
      -e "s/Ring(Rank=${SRC_RANK})/Ring(Rank=${dest_rank})/" \
      "$dest_file"

    echo "  Modified content for Rank ${dest_rank}."
    created_count=$((created_count + 1))
  done

  if [ "$created_count" -gt 0 ]; then
    echo "  Successfully created and modified ${created_count} files for Rank ${dest_rank}."
  else
    echo "  No new files were created for Rank ${dest_rank}."
  fi
done

echo "----------------------------------------"
echo "Script generation finished for all specified ranks."