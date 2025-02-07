#!/bin/bash

# 圧縮対象のディレクトリリスト
directories=(
    # "results_acqf_dist"
    # # "results_benchmark"
    # "results_tf_method_2"
    # "results_tf_method_2_additional"
    # "results_tf_method_3"
    # "results_tf_method_4"
    # "results_tf_method_4_additional"
    # "results_tf_method_5"
    # "results_tf_method_5_additional"
    # "results_tf_method_6"
    # "results_tf_method_6_additional"
    "results_new"
)

# 圧縮するターゲットファイル名
output_archive="results_new_all.tar.gz"

# ディレクトリが存在するか確認し、圧縮する
valid_dirs=()
for dir in "${directories[@]}"; do
    if [ -d "$dir" ]; then
        valid_dirs+=("$dir")
    else
        echo "Warning: Directory ${dir} not found, skipping."
    fi
done

# 圧縮処理（存在するディレクトリのみ）
if [ ${#valid_dirs[@]} -gt 0 ]; then
    tar -czvf "$output_archive" "${valid_dirs[@]}"
    echo "Compressed directories into $output_archive"
else
    echo "No valid directories found for compression."
fi
