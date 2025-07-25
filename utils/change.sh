#!/bin/bash

# --- 設定 ---
# 変更前の値
OLD_VALUE="5"
# 変更後の値
NEW_VALUE="500"
# 対象のディレクトリ名のパターン
TARGET_DIR_PATTERN="scripts_*"

echo "スクリプト内のパラメータ値を '${OLD_VALUE}' から '${NEW_VALUE}' に変更します。"
echo "対象: --iter_bo, N_TRIALS"
echo "対象ディレクトリ: ${TARGET_DIR_PATTERN}"
echo "--------------------------------------------------"

# findコマンドで対象ディレクトリ内の全シェルスクリプトを検索
find . -type f -path "./${TARGET_DIR_PATTERN}/*.sh" -print0 | while IFS= read -r -d $'\0' file; do
    
    # ファイル内に変更対象のパターンが存在するかを事前にチェック
    # grepに複数の -e を渡すことで、OR条件で検索できる
    # これにより、変更の必要がないファイルはメッセージが表示されなくなる
    if grep -q -E -e "--iter_bo\s+${OLD_VALUE}" -e "^N_TRIALS\s*=\s*\"?${OLD_VALUE}\"?" "$file"; then
        echo "-> 変更中: $file"
        
        # 1. --iter_bo の値を置換 (例: --iter_bo          500 -> --iter_bo          5)
        sed -i -E "s/(--iter_bo\s+)${OLD_VALUE}/\1${NEW_VALUE}/" "$file"
        
        # 2. N_TRIALS の値を置換 (例: N_TRIALS=500 -> N_TRIALS=5)
        #    行頭(^)にある変数定義に限定し、スペースやクォーテーションの有無に対応
        sed -i -E "s/(^N_TRIALS\s*=\s*\"?)${OLD_VALUE}(\"?)/\1${NEW_VALUE}\2/" "$file"
    fi
done

echo "--------------------------------------------------"
echo "処理が完了しました。"