#!/bin/bash -l

# Re-run specific missing experiments for all tensor ranks
for tf_rank in 2; do
    base_dir="results_tf_${tf_rank}_ackley_v3"
    acqf_dist=n
    partition_name="cluster_short"
    duration="01:00:00"
    EXE_FILE="experiments/tf_continual.py"

    # 共通引数
    BASE_ARGS=(
        --tf_lr 0.01
        --tf_tol 1e-9
        --tf_reg_lambda 0
        --tf_constraint_lambda 1.0
        --decomp_iter_num 1
        --tf_max_iter 10000
        --mask_ratio 1
        --n_startup_trials 1
        --tf_rank $tf_rank
        --acqf_dist $acqf_dist
        --iter_bo 25 # 9 25 49
        --function "ackley"
    )

    # 実験実行関数
    run_experiment() {
        local seed=$1
        local method=$2
        local constraint=$3
        local timestamp=$4
        
        # タイムスタンプ作成
        # local timestamp=$(date +"%Y-%m-%d_%H-%M-%S")
        local plot_save_dir="$base_dir/$timestamp/plots"
        mkdir -p "$plot_save_dir"
        
        # 結果ディレクトリ作成
        local results_dir="$base_dir/$timestamp"
        mkdir -p "$results_dir"
        cp "$0" "$results_dir"

        # 実験固有の引数
        local COMMON_ARGS=(${BASE_ARGS[@]})
        COMMON_ARGS+=(--tf_method $method)
        
        # 制約パラメータ追加
        if [ "$constraint" = true ]; then
            COMMON_ARGS+=(--constraint)
        fi
        
        # ジョブ名作成
        local job_name="tf_ackley_${method}_${constraint}_seed_${seed}_rank_${tf_rank}"
        
        # sbatch でジョブをサブミット
        sbatch --job-name=$job_name <<EOF
#!/bin/bash -l
#SBATCH --partition=$partition_name
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=$duration
#SBATCH --output=%x_%j.log

# Initialize conda before activating environment
# source /work/keisuke-o/anaconda3/etc/profile.d/conda.sh
# conda activate bo-env_v3

python3 $EXE_FILE \\
    --timestamp "$timestamp" \\
    --seed "$seed" \\
    --plot_save_dir "$plot_save_dir" \\
    --base_dir "$base_dir" \\
    ${COMMON_ARGS[@]}
EOF

        echo "Submitted job: $job_name"
    }

    echo "===================================================================="
    echo "RUNNING MISSING EXPERIMENTS FOR TENSOR RANK $tf_rank"
    echo "===================================================================="

    # Run experiments based on which tensor rank we're processing
    if [ $tf_rank -eq 2 ]; then
        timestamp=$(date +"%Y-%m-%d_%H-%M-%S")
        # run_experiment 0 "train" false $timestamp
        # run_experiment 1 "train" false $timestamp
        # run_experiment 2 "train" false $timestamp
        # run_experiment 3 "train" false $timestamp
        run_experiment 4 "train" false $timestamp
        run_experiment 5 "train" false $timestamp
        # run_experiment 6 "train" false $timestamp
        # run_experiment 7 "train" false $timestamp
        # run_experiment 8 "train" false $timestamp
        # run_experiment 9 "train" false $timestamp
    fi

    sleep 1

    # Run experiments based on which tensor rank we're processing
    if [ $tf_rank -eq 2 ]; then
        timestamp=$(date +"%Y-%m-%d_%H-%M-%S")
        run_experiment 0 "train" true $timestamp
        run_experiment 1 "train" true $timestamp
        # run_experiment 2 "train" true $timestamp
        # run_experiment 3 "train" true $timestamp
        # run_experiment 4 "train" true $timestamp
        # run_experiment 5 "train" true $timestamp
        # run_experiment 6 "train" true $timestamp
        # run_experiment 7 "train" true $timestamp
        # run_experiment 8 "train" true $timestamp
        # run_experiment 9 "train" true $timestamp
    fi
done

echo "All missing experiments submitted!"