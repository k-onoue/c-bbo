#!/bin/bash

# --- 設定 ---
# 削除対象の引数（この文字列が含まれる行が削除されます）
ARG_TO_DELETE="--dimension"
# 対象のディレクトリ名のパターン
TARGET_DIR_PATTERN="scripts_*"

echo "シェルスクリプトから '${ARG_TO_DELETE}' を含む行を削除します。"
echo "対象ディレクトリ: ${TARGET_DIR_PATTERN}"
echo "--------------------------------------------------"

# findコマンドで対象ディレクトリ内の全シェルスクリプトを検索
find . -type f -path "./${TARGET_DIR_PATTERN}/*.sh" -print0 | while IFS= read -r -d $'\0' file; do
    
    # --- ここからが修正点 ---
    # grepに -e オプションを付け、次の引数が検索パターンであることを明示する
    if grep -q -e "$ARG_TO_DELETE" "$file"; then
        echo "-> 変更中: $file"
        
        # sed を使って '--dimension' を含む行全体を削除する
        sed -i "/${ARG_TO_DELETE}/d" "$file"
    fi
    # --- 修正点ここまで ---

done

echo "--------------------------------------------------"
echo "処理が完了しました。"