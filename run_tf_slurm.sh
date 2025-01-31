###########################################################################################
#!/bin/bash -l
EXE_FILE="experiments/benchmark.py"

run_experiment() {
    local seed=$1
    local timestamp=$2

    # map_option を直接ここで定義
    local map_option="1" # "1", "2", "3", or "" (空の場合はフラグなし)

    local plot_save_dir="results/$timestamp/plots"
    mkdir -p "$plot_save_dir"

    # 共通の引数リストを作成
    local COMMON_ARGS=(
        --function "warcraft"
        --sampler "tpe" # or "gp"
        --iter_bo 500
        --seed "$seed"
        --n_startup_trials 1
        --timestamp "$timestamp"
        --plot_save_dir "$plot_save_dir"
    )

    # map_option が空でない場合のみ追加
    if [ -n "$map_option" ]; then
        COMMON_ARGS+=(--map_option "$map_option")
    fi

    # sbatch のジョブスクリプトを生成してサブミット
    sbatch --job-name="unconstrained_seed_${seed}_map_${map_option:-none}" <<EOF
#!/bin/bash -l
#SBATCH --partition=cluster_long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --time=4:00:00
#SBATCH --output=log_unconstrained_seed_${seed}_map_${map_option:-none}.log

python3 $EXE_FILE ${COMMON_ARGS[*]}
EOF
}

# 初期化
timestamp=$(date +"%Y-%m-%d_%H-%M-%S")
results_dir="results/$timestamp"
mkdir -p "$results_dir"
cp "$0" "$results_dir"

# シードリスト
seed_list=(0 1 2 3 4 5 6 7 8 9)

# Submit jobs
for seed in "${seed_list[@]}"; do
    run_experiment "$seed" "$timestamp"
done

sleep 1

###########################################################################################
#!/bin/bash -l
EXE_FILE="experiments/benchmark.py"

run_experiment() {
    local seed=$1
    local timestamp=$2

    # map_option を直接ここで定義
    local map_option="2" # "1", "2", "3", or "" (空の場合はフラグなし)

    local plot_save_dir="results/$timestamp/plots"
    mkdir -p "$plot_save_dir"

    # 共通の引数リストを作成
    local COMMON_ARGS=(
        --function "warcraft"
        --sampler "tpe" # or "gp"
        --iter_bo 2000
        --seed "$seed"
        --n_startup_trials 1
        --timestamp "$timestamp"
        --plot_save_dir "$plot_save_dir"
    )

    # map_option が空でない場合のみ追加
    if [ -n "$map_option" ]; then
        COMMON_ARGS+=(--map_option "$map_option")
    fi

    # sbatch のジョブスクリプトを生成してサブミット
    sbatch --job-name="unconstrained_seed_${seed}_map_${map_option:-none}" <<EOF
#!/bin/bash -l
#SBATCH --partition=cluster_long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --time=4:00:00
#SBATCH --output=log_unconstrained_seed_${seed}_map_${map_option:-none}.log

python3 $EXE_FILE ${COMMON_ARGS[*]}
EOF
}

# 初期化
timestamp=$(date +"%Y-%m-%d_%H-%M-%S")
results_dir="results/$timestamp"
mkdir -p "$results_dir"
cp "$0" "$results_dir"

# シードリスト
seed_list=(0 1 2 3 4 5 6 7 8 9)

# Submit jobs
for seed in "${seed_list[@]}"; do
    run_experiment "$seed" "$timestamp"
done

sleep 1

###########################################################################################
#!/bin/bash -l
EXE_FILE="experiments/benchmark.py"

run_experiment() {
    local seed=$1
    local timestamp=$2

    # map_option を直接ここで定義
    local map_option="3" # "1", "2", "3", or "" (空の場合はフラグなし)

    local plot_save_dir="results/$timestamp/plots"
    mkdir -p "$plot_save_dir"

    # 共通の引数リストを作成
    local COMMON_ARGS=(
        --function "warcraft"
        --sampler "tpe" # or "gp"
        --iter_bo 2000
        --seed "$seed"
        --n_startup_trials 1
        --timestamp "$timestamp"
        --plot_save_dir "$plot_save_dir"
    )

    # map_option が空でない場合のみ追加
    if [ -n "$map_option" ]; then
        COMMON_ARGS+=(--map_option "$map_option")
    fi

    # sbatch のジョブスクリプトを生成してサブミット
    sbatch --job-name="unconstrained_seed_${seed}_map_${map_option:-none}" <<EOF
#!/bin/bash -l
#SBATCH --partition=cluster_long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --time=4:00:00
#SBATCH --output=log_unconstrained_seed_${seed}_map_${map_option:-none}.log

python3 $EXE_FILE ${COMMON_ARGS[*]}
EOF
}

# 初期化
timestamp=$(date +"%Y-%m-%d_%H-%M-%S")
results_dir="results/$timestamp"
mkdir -p "$results_dir"
cp "$0" "$results_dir"

# シードリスト
seed_list=(0 1 2 3 4 5 6 7 8 9)

# Submit jobs
for seed in "${seed_list[@]}"; do
    run_experiment "$seed" "$timestamp"
done

sleep 1

###########################################################################################
#!/bin/bash -l
EXE_FILE="experiments/benchmark.py"

run_experiment() {
    local seed=$1
    local timestamp=$2

    # map_option を直接ここで定義
    local map_option="" # "1", "2", "3", or "" (空の場合はフラグなし)

    local plot_save_dir="results/$timestamp/plots"
    mkdir -p "$plot_save_dir"

    # 共通の引数リストを作成
    local COMMON_ARGS=(
        --function "ackley"
        --sampler "tpe" # or "gp"
        --iter_bo 500
        --seed "$seed"
        --n_startup_trials 1
        --timestamp "$timestamp"
        --plot_save_dir "$plot_save_dir"
    )

    # map_option が空でない場合のみ追加
    if [ -n "$map_option" ]; then
        COMMON_ARGS+=(--map_option "$map_option")
    fi

    # sbatch のジョブスクリプトを生成してサブミット
    sbatch --job-name="unconstrained_seed_${seed}_map_${map_option:-none}" <<EOF
#!/bin/bash -l
#SBATCH --partition=cluster_long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --time=4:00:00
#SBATCH --output=log_unconstrained_seed_${seed}_map_${map_option:-none}.log

python3 $EXE_FILE ${COMMON_ARGS[*]}
EOF
}

# 初期化
timestamp=$(date +"%Y-%m-%d_%H-%M-%S")
results_dir="results/$timestamp"
mkdir -p "$results_dir"
cp "$0" "$results_dir"

# シードリスト
seed_list=(0 1 2 3 4 5 6 7 8 9)

# Submit jobs
for seed in "${seed_list[@]}"; do
    run_experiment "$seed" "$timestamp"
done

sleep 1


echo "All unconstrained experiments submitted!"
