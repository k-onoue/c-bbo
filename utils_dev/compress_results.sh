#!/bin/bash

# 引数が指定されていない場合は使用法を表示して終了
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 target_dir_name"
    exit 1
fi

# コマンドライン引数でディレクトリ名を取得
target_dir_name=$1

# 圧縮コマンド
tar -czvf "${target_dir_name}.tar.gz" "${target_dir_name}"