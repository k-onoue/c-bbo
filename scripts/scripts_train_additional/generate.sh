#!/bin/bash

# --- 設定 ---
# 変換元となるテンプレートファイル名（ackleyのランク2を使用）
LAUNCHER_SRC_TEMPLATE="run_array_launcher_train_2_ackley.sh"
WORKER_SRC_TEMPLATE="run_array_train_2_ackley.sh"

# 作成したいタスク名のリスト
TASKS=("gap_a" "gap_b" "ising_a" "ising_b" "tss" "sss")

# 作成したいランクのリスト
RANKS=(2 3 4 5 6)

# --- テンプレートファイルの存在チェック ---
if [ ! -f "$LAUNCHER_SRC_TEMPLATE" ] || [ ! -f "$WORKER_SRC_TEMPLATE" ]; then
    echo "エラー: テンプレートとなるスクリプトファイルが見つかりません。"
    echo "  - $LAUNCHER_SRC_TEMPLATE"
    echo "  - $WORKER_SRC_TEMPLATE"
    echo "このスクリプトをackley用のスクリプトと同じディレクトリで実行してください。"
    exit 1
fi

# --- 各タスクとランクの組み合わせでスクリプトを生成 ---
for task in "${TASKS[@]}"; do
    for rank in "${RANKS[@]}"; do
        echo "--- タスク '$task' (Rank=$rank) のスクリプトを生成中... ---"

        # --- 新しい識別子とファイル名を定義 ---
        new_job_id="train_${rank}_${task}"
        new_result_id="${task}_train_${rank}"
        new_launcher="run_array_launcher_${new_job_id}.sh"
        new_worker="run_array_${new_job_id}.sh"

        # --- テンプレートからファイルをコピー ---
        cp "$LAUNCHER_SRC_TEMPLATE" "$new_launcher"
        cp "$WORKER_SRC_TEMPLATE" "$new_worker"

        # --- ランチャースクリプトの内容を置換 (sed) ---
        sed -i \
            -e "s/results_ackley_train_2_/results_${new_result_id}_/g" \
            -e "s/run_array_launcher_train_2_ackley.sh/${new_launcher}/g" \
            -e "s/run_array_train_2_ackley.sh/${new_worker}/g" \
            "$new_launcher"

        # --- ワーカースクリプトの内容を置換 (sed) ---
        # 表示用のタスク名 (例: gap_a -> GAP-A)
        task_display_name=$(echo "$task" | tr 'a-z_' 'A-Z-')

        sed -i \
            -e "s/train_2_ackley/${new_job_id}/g" \
            -e "s/run_array_train_2_ackley.sh_snapshot/${new_worker}_snapshot/g" \
            -e "s/--function         \"ackley\"/--function         \"${task}\"/" \
            -e "s/--tf_rank           2               # ランクを2に固定/--tf_rank           ${rank}               # ランクを${rank}に設定/" \
            -e "s/Ackley Train(Rank=2)/${task_display_name} Train(Rank=${rank})/" \
            "$new_worker"

        echo "  - 生成完了: $new_launcher"
        echo "  - 生成完了: $new_worker"

    done
    echo "" # タスクごとに改行
done

echo "すべてのタスクとランクのスクリプト生成が完了しました。"