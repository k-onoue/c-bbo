#!/bin/bash

# 指定されたディレクトリを圧縮する関数
compress_directory() {
    local target_dir=$1
    if [ -d "$target_dir" ]; then
        tar -czvf "${target_dir}.tar.gz" "${target_dir}"
        echo "Compressed ${target_dir} to ${target_dir}.tar.gz"
    else
        echo "Directory ${target_dir} not found"
    fi
}

# 圧縮対象のディレクトリリスト
directories=(
    "results_acqf_dist"
    "results_benchmark"
    "results_tf_method_2"
    "results_tf_method_3"
    "results_tf_method_4"
    "results_tf_method_5"
    "results_tf_method_6"
)

# 各ディレクトリを圧縮
for dir in "${directories[@]}"; do
    compress_directory "$dir"
done