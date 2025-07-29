#!/bin/bash

#
# このスクリプトは、すべてのランチャースクリプト内のsbatchコマンドが呼び出す
# 実行スクリプトのパスを修正します。
# 具体的には、"$PROJECT_ROOT/scripts_..." を "$PROJECT_ROOT/scripts/scripts_..." に置換します。
#
# 実行前に、プロジェクトのルートディレクトリ（'c-bbo'など）にいることを確認してください。
#

# --- カレントディレクトリのチェック ---
if [ ! -d "scripts" ]; then
  echo "エラー: 'scripts'ディレクトリが見つかりません。" >&2
  echo "このスクリプトはプロジェクトのルートディレクトリから実行してください。" >&2
  exit 1
fi

echo "プロジェクトルートで実行されていることを確認しました。"
echo "ランチャースクリプトの検索を開始します..."
echo ""

# --- 対象となるランチャースクリプトを検索 ---
# findコマンドで 'scripts/scripts_*' ディレクトリ内にある 'run_array_launcher_*.sh' という名前のファイルを探す
files_to_modify=$(find scripts/scripts_* -type f -name "run_array_launcher_*.sh")

if [ -z "$files_to_modify" ]; then
  echo "変更対象のランチャースクリプトが見つかりませんでした。"
  exit 0
fi

echo "以下のファイルのパスを修正します:"
echo "$files_to_modify" | sed 's/^/  - /'
echo ""
read -p "これらのファイルを変更してもよろしいですか？ (y/n): " -n 1 -r
echo ""
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "処理を中断しました。"
    exit 1
fi
echo ""

# --- 各ファイルに対して置換処理を実行 ---
successful_count=0
failed_count=0

for file in $files_to_modify; do
  # sed -i を使ってファイルを直接編集します。
  # '$PROJECT_ROOT' がシェルによって展開されるのを防ぐため、sedの式全体をシングルクォートで囲みます。
  # これにより、"$PROJECT_ROOT/scripts_" という文字列そのものを検索・置換します。
  sed -i 's|\"$PROJECT_ROOT/scripts_|\"$PROJECT_ROOT/scripts/scripts_|' "$file"

  # 変更が成功したかどうかの確認（簡易的）
  if grep -q '"$PROJECT_ROOT/scripts/scripts_' "$file"; then
    echo "✅ 修正完了: $file"
    successful_count=$((successful_count + 1))
  else
    # 置換に失敗した場合、元のパスがまだ残っているか確認
    if grep -q '"$PROJECT_ROOT/scripts_' "$file"; then
       echo "⚠️  未修正またはエラー: $file"
       failed_count=$((failed_count + 1))
    else
       # 元のパスも新しいパスも見つからない場合、既に修正済みか、対象外のファイル
       echo "ℹ️  既に修正済みか対象外: $file"
    fi
  fi
done

echo ""
echo "--- 処理結果 ---"
echo "すべての処理が完了しました。"
echo "成功: $successful_count ファイル"
echo "失敗/未修正: $failed_count ファイル"

