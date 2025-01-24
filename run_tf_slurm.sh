#!/bin/bash -l
EXE_FILE="experiments/tf_continual.py"

run_experiment() {
    local seed=$1
    local timestamp=$2
    local plot_save_dir="results_2/$timestamp/plots"
    mkdir -p "$plot_save_dir"

    # 共通の引数を関数内で定義
    local COMMON_ARGS=(
        --function "warcraft"
        --tf_lr 0.01
        --tf_tol 1e-6
        --tf_reg_lambda 0
        --tf_constraint_lambda 1.0
        --decomp_iter_num 10
        --tf_max_iter 10000
        --mask_ratio 1
        --n_startup_trials 1
        --iter_bo 500
        --map_option 1
        --tf_method train
        --tf_rank 3
        --acqf_dist t1 # n, t1, t2
    )

    # オプションパラメータの追加
    local constraint=true
    local direction=false
    [ "$constraint" = true ] && COMMON_ARGS+=(--constraint)
    [ "$direction" = true ] && COMMON_ARGS+=(--direction)

    # sbatch でジョブをサブミット
    sbatch --job-name="tf_experiment_seed_${seed}" <<EOF
#!/bin/bash -l
#SBATCH --partition=cluster_short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --time=4:00:00
#SBATCH --output=%x_%j.log

python3 $EXE_FILE \
    --timestamp "$timestamp" \
    --seed "$seed" \
    --plot_save_dir "$plot_save_dir" \
    ${COMMON_ARGS[*]}
EOF
}

# 初期化
timestamp=$(date +"%Y-%m-%d_%H-%M-%S")
results_dir="results/$timestamp"
mkdir -p "$results_dir"
cp "$0" "$results_dir"

# シードリスト
seed_list=(0 1 2 3 4 5 6 7 8 9)

# 実験をジョブとしてサブミット
for seed in "${seed_list[@]}"; do
    run_experiment "$seed" "$timestamp"
done

echo "All experiments submitted!"
