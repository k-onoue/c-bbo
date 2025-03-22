#!/bin/bash -l

# 共通設定
tf_rank=3
base_dir="results_tf_${tf_rank}_diabetes"
acqf_dist=n
partition_name="cluster_short"
duration="2:00:00"
EXE_FILE="experiments/tf_continual_diabetes.py"

# 共通引数
BASE_ARGS=(
    --tf_lr 0.01
    --tf_tol 1e-9
    --tf_reg_lambda 0
    --tf_constraint_lambda 1.0
    --decomp_iter_num 10
    --tf_max_iter 10000
    --mask_ratio 1
    --n_startup_trials 1
    --iter_bo 500
    --tf_rank $tf_rank
    --acqf_dist $acqf_dist
)

# シードリスト
seed_list=(0 1 2 3 4 5 6 7 8 9)

# 実験実行関数
run_experiment() {
    local seed=$1
    local timestamp=$2
    local method=$3
    local constraint=$4
    
    local plot_save_dir="$base_dir/$timestamp/plots"
    mkdir -p "$plot_save_dir"

    # 実験固有の引数
    local COMMON_ARGS=(${BASE_ARGS[@]})
    COMMON_ARGS+=(--tf_method $method)
    
    # 制約パラメータ追加 (制約ありの場合は制約違反パスを初期サンプルに追加)
    if [ "$constraint" = true ]; then
        COMMON_ARGS+=(--constraint)
    fi
    
    # ジョブ名作成
    local job_name="tf_diabetes_${method}_${constraint}_seed_${seed}"
    
    # sbatch でジョブをサブミット
    sbatch --job-name=$job_name <<EOF
#!/bin/bash -l
#SBATCH --partition=$partition_name
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --time=$duration
#SBATCH --output=%x_%j.log

python3 $EXE_FILE \
    --timestamp "$timestamp" \
    --seed "$seed" \
    --plot_save_dir "$plot_save_dir" \
    --base_dir "$base_dir" \
    ${COMMON_ARGS[@]}
EOF

    echo "Submitted job: $job_name"
}

実験パラメータの配列
methods=("cp" "train" "ring")
constraints=(false)


# 全パラメータ組み合わせで実験実行
for constraint in "${constraints[@]}"; do
    for method in "${methods[@]}"; do
        # 初期化
        timestamp=$(date +"%Y-%m-%d_%H-%M-%S")
        results_dir="$base_dir/$timestamp"
        mkdir -p "$results_dir"
        cp "$0" "$results_dir"
        
        echo "Starting experiments: method=$method, constraint=$constraint"
        
        # 各シードで実行
        for seed in "${seed_list[@]}"; do
            run_experiment "$seed" "$timestamp" "$method" "$constraint"
        done
        
        # 次の実験セットの前に少し待機
        sleep 1
    done
done

echo "All experiments submitted!"