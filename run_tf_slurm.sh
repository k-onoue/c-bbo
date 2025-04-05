#!/bin/bash -l

# Re-run specific missing experiments for all tensor ranks
for tf_rank in 2 3 4 5 6; do
    base_dir="results_tf_${tf_rank}_diabetes_complement_3"
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

    # 実験実行関数
    run_experiment() {
        local seed=$1
        local method=$2
        local constraint=$3
        
        # タイムスタンプ作成
        local timestamp=$(date +"%Y-%m-%d_%H-%M-%S")
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
        local job_name="tf_diabetes_${method}_${constraint}_seed_${seed}_rank_${tf_rank}"
        
        # sbatch でジョブをサブミット
        sbatch --job-name=$job_name <<EOF
#!/bin/bash -l
#SBATCH --partition=$partition_name
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
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
        sleep 2
    }

    echo "===================================================================="
    echo "RUNNING MISSING EXPERIMENTS FOR TENSOR RANK $tf_rank"
    echo "===================================================================="

    # Run experiments based on which tensor rank we're processing
    if [ $tf_rank -eq 2 ]; then
        # For rank 2 (diabetes_2.csv) - Only run missing experiments
        echo "Running CP seed 9 (non-constrained)"
        run_experiment 9 "cp" false
    fi
        
    # elif [ $tf_rank -eq 3 ]; then
    #     # For rank 3 (diabetes_3.csv) - all experiments are complete
    #     echo "All experiments for tensor rank 3 are already completed."
        
    # elif [ $tf_rank -eq 4 ]; then
    #     # For rank 4 (diabetes_4.csv) - Only run missing experiments
    #     echo "Running CP seed 6 (non-constrained)"
    #     run_experiment 6 "cp" false
    #     echo "Running CP seed 7 (non-constrained)"
    #     run_experiment 7 "cp" false
        
    # elif [ $tf_rank -eq 5 ]; then
    #     # For rank 5 (diabetes_5.csv) - all experiments are complete
    #     echo "All experiments for tensor rank 5 are already completed."
        
    # elif [ $tf_rank -eq 6 ]; then
    #     # For rank 6 (diabetes_6.csv) - Only run missing experiments
    #     echo "Running Train seed 5 (constrained)"
    #     run_experiment 5 "train" true
    #     echo "Running Train seed 5 (non-constrained)"
    #     run_experiment 5 "train" false
    # fi
    
    # echo "Waiting before proceeding to next rank..."
    # sleep 2s
done

echo "All missing experiments submitted!"