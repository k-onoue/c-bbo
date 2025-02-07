#!/bin/bash -l

base_dir="results_tf_method_5"
acqf_dist=n
tf_rank=5

partition_name="cluster_short"
duration="2:00:00"

# =========================================================================================
constraint=true

# ###########################################################################################

# function="warcraft"
# map_option=1 # "1", "2", "3", or "" (空の場合はフラグなし)

# # -----------------------------------------------------------------------------------------
# EXE_FILE="experiments/tf_continual.py"

# run_experiment() {
#     local seed=$1
#     local timestamp=$2
#     local plot_save_dir="$base_dir/$timestamp/plots"
#     mkdir -p "$plot_save_dir"

#     # 共通の引数を関数内で定義
#     local COMMON_ARGS=(
#         --function "$function"
#         --tf_lr 0.01
#         --tf_tol 1e-6
#         --tf_reg_lambda 0
#         --tf_constraint_lambda 1.0
#         --decomp_iter_num 10
#         --tf_max_iter 10000
#         --mask_ratio 1
#         --n_startup_trials 1
#         --iter_bo 500
#         --tf_method cp
#         --tf_rank $tf_rank
#         --acqf_dist $acqf_dist # n, t1, t2
#     )

#     # map_option が空でない場合のみ追加
#     if [ -n "$map_option" ]; then
#         COMMON_ARGS+=(--map_option "$map_option")
#     fi

#     # オプションパラメータの追加
#     local constraint=$constraint
#     local direction=false
#     [ "$constraint" = true ] && COMMON_ARGS+=(--constraint)
#     [ "$direction" = true ] && COMMON_ARGS+=(--direction)

#     # sbatch でジョブをサブミット
#     sbatch --job-name="tf_experiment_seed_${seed}" <<EOF
# #!/bin/bash -l
# #SBATCH --partition=$partition_name
# #SBATCH --nodes=1
# #SBATCH --ntasks=1
# #SBATCH --cpus-per-task=20
# #SBATCH --time=$duration
# #SBATCH --output=%x_%j.log

# python3 $EXE_FILE \
#     --timestamp "$timestamp" \
#     --seed "$seed" \
#     --plot_save_dir "$plot_save_dir" \
#     --base_dir "$base_dir" \
#     ${COMMON_ARGS[*]}
# EOF
# }

# # 初期化
# timestamp=$(date +"%Y-%m-%d_%H-%M-%S")
# results_dir="$base_dir/$timestamp"
# mkdir -p "$results_dir"
# cp "$0" "$results_dir"

# # シードリスト
# seed_list=(0 1 2 3 4 5 6 7 8 9)

# # 実験をジョブとしてサブミット
# for seed in "${seed_list[@]}"; do
#     run_experiment "$seed" "$timestamp"
# done

# sleep 1

# # -----------------------------------------------------------------------------------------
# EXE_FILE="experiments/tf_continual.py"

# run_experiment() {
#     local seed=$1
#     local timestamp=$2
#     local plot_save_dir="$base_dir/$timestamp/plots"
#     mkdir -p "$plot_save_dir"

#     # 共通の引数を関数内で定義
#     local COMMON_ARGS=(
#         --function "$function"
#         --tf_lr 0.01
#         --tf_tol 1e-6
#         --tf_reg_lambda 0
#         --tf_constraint_lambda 1.0
#         --decomp_iter_num 10
#         --tf_max_iter 10000
#         --mask_ratio 1
#         --n_startup_trials 1
#         --iter_bo 500
#         --tf_method train
#         --tf_rank $tf_rank
#         --acqf_dist $acqf_dist # n, t1, t2
#     )

#     # map_option が空でない場合のみ追加
#     if [ -n "$map_option" ]; then
#         COMMON_ARGS+=(--map_option "$map_option")
#     fi

#     # オプションパラメータの追加
#     local constraint=$constraint
#     local direction=false
#     [ "$constraint" = true ] && COMMON_ARGS+=(--constraint)
#     [ "$direction" = true ] && COMMON_ARGS+=(--direction)

#     # sbatch でジョブをサブミット
#     sbatch --job-name="tf_experiment_seed_${seed}" <<EOF
# #!/bin/bash -l
# #SBATCH --partition=$partition_name
# #SBATCH --nodes=1
# #SBATCH --ntasks=1
# #SBATCH --cpus-per-task=20
# #SBATCH --time=$duration
# #SBATCH --output=%x_%j.log

# python3 $EXE_FILE \
#     --timestamp "$timestamp" \
#     --seed "$seed" \
#     --plot_save_dir "$plot_save_dir" \
#     --base_dir "$base_dir" \
#     ${COMMON_ARGS[*]}
# EOF
# }

# # 初期化
# timestamp=$(date +"%Y-%m-%d_%H-%M-%S")
# results_dir="$base_dir/$timestamp"
# mkdir -p "$results_dir"
# cp "$0" "$results_dir"

# # シードリスト
# seed_list=(0 1 2 3 4 5 6 7 8 9)

# # 実験をジョブとしてサブミット
# for seed in "${seed_list[@]}"; do
#     run_experiment "$seed" "$timestamp"
# done

# sleep 1

# # -----------------------------------------------------------------------------------------
# EXE_FILE="experiments/tf_continual.py"

# run_experiment() {
#     local seed=$1
#     local timestamp=$2
#     local plot_save_dir="$base_dir/$timestamp/plots"
#     mkdir -p "$plot_save_dir"

#     # 共通の引数を関数内で定義
#     local COMMON_ARGS=(
#         --function "$function"
#         --tf_lr 0.01
#         --tf_tol 1e-6
#         --tf_reg_lambda 0
#         --tf_constraint_lambda 1.0
#         --decomp_iter_num 10
#         --tf_max_iter 10000
#         --mask_ratio 1
#         --n_startup_trials 1
#         --iter_bo 500
#         --tf_method ring
#         --tf_rank $tf_rank
#         --acqf_dist $acqf_dist # n, t1, t2
#     )

#     # map_option が空でない場合のみ追加
#     if [ -n "$map_option" ]; then
#         COMMON_ARGS+=(--map_option "$map_option")
#     fi

#     # オプションパラメータの追加
#     local constraint=$constraint
#     local direction=false
#     [ "$constraint" = true ] && COMMON_ARGS+=(--constraint)
#     [ "$direction" = true ] && COMMON_ARGS+=(--direction)

#     # sbatch でジョブをサブミット
#     sbatch --job-name="tf_experiment_seed_${seed}" <<EOF
# #!/bin/bash -l
# #SBATCH --partition=$partition_name
# #SBATCH --nodes=1
# #SBATCH --ntasks=1
# #SBATCH --cpus-per-task=20
# #SBATCH --time=$duration
# #SBATCH --output=%x_%j.log

# python3 $EXE_FILE \
#     --timestamp "$timestamp" \
#     --seed "$seed" \
#     --plot_save_dir "$plot_save_dir" \
#     --base_dir "$base_dir" \
#     ${COMMON_ARGS[*]}
# EOF
# }

# # 初期化
# timestamp=$(date +"%Y-%m-%d_%H-%M-%S")
# results_dir="$base_dir/$timestamp"
# mkdir -p "$results_dir"
# cp "$0" "$results_dir"

# # シードリスト
# seed_list=(0 1 2 3 4 5 6 7 8 9)

# # 実験をジョブとしてサブミット
# for seed in "${seed_list[@]}"; do
#     run_experiment "$seed" "$timestamp"
# done

# sleep 1

# ###########################################################################################

# function="warcraft"
# map_option=2

# # -----------------------------------------------------------------------------------------
# EXE_FILE="experiments/tf_continual.py"

# run_experiment() {
#     local seed=$1
#     local timestamp=$2
#     local plot_save_dir="$base_dir/$timestamp/plots"
#     mkdir -p "$plot_save_dir"

#     # 共通の引数を関数内で定義
#     local COMMON_ARGS=(
#         --function "$function"
#         --tf_lr 0.01
#         --tf_tol 1e-6
#         --tf_reg_lambda 0
#         --tf_constraint_lambda 1.0
#         --decomp_iter_num 10
#         --tf_max_iter 10000
#         --mask_ratio 1
#         --n_startup_trials 1
#         --iter_bo 500
#         --tf_method cp
#         --tf_rank $tf_rank
#         --acqf_dist $acqf_dist # n, t1, t2
#     )

#     # map_option が空でない場合のみ追加
#     if [ -n "$map_option" ]; then
#         COMMON_ARGS+=(--map_option "$map_option")
#     fi

#     # オプションパラメータの追加
#     local constraint=$constraint
#     local direction=false
#     [ "$constraint" = true ] && COMMON_ARGS+=(--constraint)
#     [ "$direction" = true ] && COMMON_ARGS+=(--direction)

#     # sbatch でジョブをサブミット
#     sbatch --job-name="tf_experiment_seed_${seed}" <<EOF
# #!/bin/bash -l
# #SBATCH --partition=$partition_name
# #SBATCH --nodes=1
# #SBATCH --ntasks=1
# #SBATCH --cpus-per-task=20
# #SBATCH --time=$duration
# #SBATCH --output=%x_%j.log

# python3 $EXE_FILE \
#     --timestamp "$timestamp" \
#     --seed "$seed" \
#     --plot_save_dir "$plot_save_dir" \
#     --base_dir "$base_dir" \
#     ${COMMON_ARGS[*]}
# EOF
# }

# # 初期化
# timestamp=$(date +"%Y-%m-%d_%H-%M-%S")
# results_dir="$base_dir/$timestamp"
# mkdir -p "$results_dir"
# cp "$0" "$results_dir"

# # シードリスト
# seed_list=(0 1 2 3 4 5 6 7 8 9)

# # 実験をジョブとしてサブミット
# for seed in "${seed_list[@]}"; do
#     run_experiment "$seed" "$timestamp"
# done

# sleep 1

# # -----------------------------------------------------------------------------------------
# EXE_FILE="experiments/tf_continual.py"

# run_experiment() {
#     local seed=$1
#     local timestamp=$2
#     local plot_save_dir="$base_dir/$timestamp/plots"
#     mkdir -p "$plot_save_dir"

#     # 共通の引数を関数内で定義
#     local COMMON_ARGS=(
#         --function "$function"
#         --tf_lr 0.01
#         --tf_tol 1e-6
#         --tf_reg_lambda 0
#         --tf_constraint_lambda 1.0
#         --decomp_iter_num 10
#         --tf_max_iter 10000
#         --mask_ratio 1
#         --n_startup_trials 1
#         --iter_bo 500
#         --tf_method train
#         --tf_rank $tf_rank
#         --acqf_dist $acqf_dist # n, t1, t2
#     )

#     # map_option が空でない場合のみ追加
#     if [ -n "$map_option" ]; then
#         COMMON_ARGS+=(--map_option "$map_option")
#     fi

#     # オプションパラメータの追加
#     local constraint=$constraint
#     local direction=false
#     [ "$constraint" = true ] && COMMON_ARGS+=(--constraint)
#     [ "$direction" = true ] && COMMON_ARGS+=(--direction)

#     # sbatch でジョブをサブミット
#     sbatch --job-name="tf_experiment_seed_${seed}" <<EOF
# #!/bin/bash -l
# #SBATCH --partition=$partition_name
# #SBATCH --nodes=1
# #SBATCH --ntasks=1
# #SBATCH --cpus-per-task=20
# #SBATCH --time=$duration
# #SBATCH --output=%x_%j.log

# python3 $EXE_FILE \
#     --timestamp "$timestamp" \
#     --seed "$seed" \
#     --plot_save_dir "$plot_save_dir" \
#     --base_dir "$base_dir" \
#     ${COMMON_ARGS[*]}
# EOF
# }

# # 初期化
# timestamp=$(date +"%Y-%m-%d_%H-%M-%S")
# results_dir="$base_dir/$timestamp"
# mkdir -p "$results_dir"
# cp "$0" "$results_dir"

# # シードリスト
# seed_list=(0 1 2 3 4 5 6 7 8 9)

# # 実験をジョブとしてサブミット
# for seed in "${seed_list[@]}"; do
#     run_experiment "$seed" "$timestamp"
# done

# sleep 1

# # -----------------------------------------------------------------------------------------
# EXE_FILE="experiments/tf_continual.py"

# run_experiment() {
#     local seed=$1
#     local timestamp=$2
#     local plot_save_dir="$base_dir/$timestamp/plots"
#     mkdir -p "$plot_save_dir"

#     # 共通の引数を関数内で定義
#     local COMMON_ARGS=(
#         --function "$function"
#         --tf_lr 0.01
#         --tf_tol 1e-6
#         --tf_reg_lambda 0
#         --tf_constraint_lambda 1.0
#         --decomp_iter_num 10
#         --tf_max_iter 10000
#         --mask_ratio 1
#         --n_startup_trials 1
#         --iter_bo 500
#         --tf_method ring
#         --tf_rank $tf_rank
#         --acqf_dist $acqf_dist # n, t1, t2
#     )

#     # map_option が空でない場合のみ追加
#     if [ -n "$map_option" ]; then
#         COMMON_ARGS+=(--map_option "$map_option")
#     fi

#     # オプションパラメータの追加
#     local constraint=$constraint
#     local direction=false
#     [ "$constraint" = true ] && COMMON_ARGS+=(--constraint)
#     [ "$direction" = true ] && COMMON_ARGS+=(--direction)

#     # sbatch でジョブをサブミット
#     sbatch --job-name="tf_experiment_seed_${seed}" <<EOF
# #!/bin/bash -l
# #SBATCH --partition=$partition_name
# #SBATCH --nodes=1
# #SBATCH --ntasks=1
# #SBATCH --cpus-per-task=20
# #SBATCH --time=$duration
# #SBATCH --output=%x_%j.log

# python3 $EXE_FILE \
#     --timestamp "$timestamp" \
#     --seed "$seed" \
#     --plot_save_dir "$plot_save_dir" \
#     --base_dir "$base_dir" \
#     ${COMMON_ARGS[*]}
# EOF
# }

# # 初期化
# timestamp=$(date +"%Y-%m-%d_%H-%M-%S")
# results_dir="$base_dir/$timestamp"
# mkdir -p "$results_dir"
# cp "$0" "$results_dir"

# # シードリスト
# seed_list=(0 1 2 3 4 5 6 7 8 9)

# # 実験をジョブとしてサブミット
# for seed in "${seed_list[@]}"; do
#     run_experiment "$seed" "$timestamp"
# done

# sleep 1

###########################################################################################

function="ackley"
map_option=""

# -----------------------------------------------------------------------------------------
EXE_FILE="experiments/tf_continual.py"

run_experiment() {
    local seed=$1
    local timestamp=$2
    local plot_save_dir="$base_dir/$timestamp/plots"
    mkdir -p "$plot_save_dir"

    # 共通の引数を関数内で定義
    local COMMON_ARGS=(
        --function "$function"
        --tf_lr 0.01
        --tf_tol 1e-6
        --tf_reg_lambda 0
        --tf_constraint_lambda 1.0
        --decomp_iter_num 10
        --tf_max_iter 10000
        --mask_ratio 1
        --n_startup_trials 1
        --iter_bo 500
        --tf_method cp
        --tf_rank $tf_rank
        --acqf_dist $acqf_dist # n, t1, t2
    )

    # map_option が空でない場合のみ追加
    if [ -n "$map_option" ]; then
        COMMON_ARGS+=(--map_option "$map_option")
    fi

    # オプションパラメータの追加
    local constraint=$constraint
    local direction=false
    [ "$constraint" = true ] && COMMON_ARGS+=(--constraint)
    [ "$direction" = true ] && COMMON_ARGS+=(--direction)

    # sbatch でジョブをサブミット
    sbatch --job-name="tf_experiment_seed_${seed}" <<EOF
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
    ${COMMON_ARGS[*]}
EOF
}

# 初期化
timestamp=$(date +"%Y-%m-%d_%H-%M-%S")
results_dir="$base_dir/$timestamp"
mkdir -p "$results_dir"
cp "$0" "$results_dir"

# シードリスト
# seed_list=(0 1 2 3 4 5 6 7 8 9)
seed_list=(7)

# 実験をジョブとしてサブミット
for seed in "${seed_list[@]}"; do
    run_experiment "$seed" "$timestamp"
done

sleep 1

# # -----------------------------------------------------------------------------------------
# EXE_FILE="experiments/tf_continual.py"

# run_experiment() {
#     local seed=$1
#     local timestamp=$2
#     local plot_save_dir="$base_dir/$timestamp/plots"
#     mkdir -p "$plot_save_dir"

#     # 共通の引数を関数内で定義
#     local COMMON_ARGS=(
#         --function "$function"
#         --tf_lr 0.01
#         --tf_tol 1e-6
#         --tf_reg_lambda 0
#         --tf_constraint_lambda 1.0
#         --decomp_iter_num 10
#         --tf_max_iter 10000
#         --mask_ratio 1
#         --n_startup_trials 1
#         --iter_bo 500
#         --tf_method train
#         --tf_rank $tf_rank
#         --acqf_dist $acqf_dist # n, t1, t2
#     )

#     # map_option が空でない場合のみ追加
#     if [ -n "$map_option" ]; then
#         COMMON_ARGS+=(--map_option "$map_option")
#     fi

#     # オプションパラメータの追加
#     local constraint=$constraint
#     local direction=false
#     [ "$constraint" = true ] && COMMON_ARGS+=(--constraint)
#     [ "$direction" = true ] && COMMON_ARGS+=(--direction)

#     # sbatch でジョブをサブミット
#     sbatch --job-name="tf_experiment_seed_${seed}" <<EOF
# #!/bin/bash -l
# #SBATCH --partition=$partition_name
# #SBATCH --nodes=1
# #SBATCH --ntasks=1
# #SBATCH --cpus-per-task=20
# #SBATCH --time=$duration
# #SBATCH --output=%x_%j.log

# python3 $EXE_FILE \
#     --timestamp "$timestamp" \
#     --seed "$seed" \
#     --plot_save_dir "$plot_save_dir" \
#     --base_dir "$base_dir" \
#     ${COMMON_ARGS[*]}
# EOF
# }

# # 初期化
# timestamp=$(date +"%Y-%m-%d_%H-%M-%S")
# results_dir="$base_dir/$timestamp"
# mkdir -p "$results_dir"
# cp "$0" "$results_dir"

# # シードリスト
# seed_list=(0 1 2 3 4 5 6 7 8 9)

# # 実験をジョブとしてサブミット
# for seed in "${seed_list[@]}"; do
#     run_experiment "$seed" "$timestamp"
# done

# sleep 1

# # -----------------------------------------------------------------------------------------
# EXE_FILE="experiments/tf_continual.py"

# run_experiment() {
#     local seed=$1
#     local timestamp=$2
#     local plot_save_dir="$base_dir/$timestamp/plots"
#     mkdir -p "$plot_save_dir"

#     # 共通の引数を関数内で定義
#     local COMMON_ARGS=(
#         --function "$function"
#         --tf_lr 0.01
#         --tf_tol 1e-6
#         --tf_reg_lambda 0
#         --tf_constraint_lambda 1.0
#         --decomp_iter_num 10
#         --tf_max_iter 10000
#         --mask_ratio 1
#         --n_startup_trials 1
#         --iter_bo 500
#         --tf_method ring
#         --tf_rank $tf_rank
#         --acqf_dist $acqf_dist # n, t1, t2
#     )

#     # map_option が空でない場合のみ追加
#     if [ -n "$map_option" ]; then
#         COMMON_ARGS+=(--map_option "$map_option")
#     fi

#     # オプションパラメータの追加
#     local constraint=$constraint
#     local direction=false
#     [ "$constraint" = true ] && COMMON_ARGS+=(--constraint)
#     [ "$direction" = true ] && COMMON_ARGS+=(--direction)

#     # sbatch でジョブをサブミット
#     sbatch --job-name="tf_experiment_seed_${seed}" <<EOF
# #!/bin/bash -l
# #SBATCH --partition=$partition_name
# #SBATCH --nodes=1
# #SBATCH --ntasks=1
# #SBATCH --cpus-per-task=20
# #SBATCH --time=$duration
# #SBATCH --output=%x_%j.log

# python3 $EXE_FILE \
#     --timestamp "$timestamp" \
#     --seed "$seed" \
#     --plot_save_dir "$plot_save_dir" \
#     --base_dir "$base_dir" \
#     ${COMMON_ARGS[*]}
# EOF
# }

# # 初期化
# timestamp=$(date +"%Y-%m-%d_%H-%M-%S")
# results_dir="$base_dir/$timestamp"
# mkdir -p "$results_dir"
# cp "$0" "$results_dir"

# # シードリスト
# seed_list=(0 1 2 3 4 5 6 7 8 9)

# # 実験をジョブとしてサブミット
# for seed in "${seed_list[@]}"; do
#     run_experiment "$seed" "$timestamp"
# done

# sleep 1





# =========================================================================================
constraint=false

# ###########################################################################################

# function="warcraft"
# map_option=1 # "1", "2", "3", or "" (空の場合はフラグなし)

# # -----------------------------------------------------------------------------------------
# EXE_FILE="experiments/tf_continual.py"

# run_experiment() {
#     local seed=$1
#     local timestamp=$2
#     local plot_save_dir="$base_dir/$timestamp/plots"
#     mkdir -p "$plot_save_dir"

#     # 共通の引数を関数内で定義
#     local COMMON_ARGS=(
#         --function "$function"
#         --tf_lr 0.01
#         --tf_tol 1e-6
#         --tf_reg_lambda 0
#         --tf_constraint_lambda 1.0
#         --decomp_iter_num 10
#         --tf_max_iter 10000
#         --mask_ratio 1
#         --n_startup_trials 1
#         --iter_bo 500
#         --tf_method cp
#         --tf_rank $tf_rank
#         --acqf_dist $acqf_dist # n, t1, t2
#     )

#     # map_option が空でない場合のみ追加
#     if [ -n "$map_option" ]; then
#         COMMON_ARGS+=(--map_option "$map_option")
#     fi

#     # オプションパラメータの追加
#     local constraint=$constraint
#     local direction=false
#     [ "$constraint" = true ] && COMMON_ARGS+=(--constraint)
#     [ "$direction" = true ] && COMMON_ARGS+=(--direction)

#     # sbatch でジョブをサブミット
#     sbatch --job-name="tf_experiment_seed_${seed}" <<EOF
# #!/bin/bash -l
# #SBATCH --partition=$partition_name
# #SBATCH --nodes=1
# #SBATCH --ntasks=1
# #SBATCH --cpus-per-task=20
# #SBATCH --time=$duration
# #SBATCH --output=%x_%j.log

# python3 $EXE_FILE \
#     --timestamp "$timestamp" \
#     --seed "$seed" \
#     --plot_save_dir "$plot_save_dir" \
#     --base_dir "$base_dir" \
#     ${COMMON_ARGS[*]}
# EOF
# }

# # 初期化
# timestamp=$(date +"%Y-%m-%d_%H-%M-%S")
# results_dir="$base_dir/$timestamp"
# mkdir -p "$results_dir"
# cp "$0" "$results_dir"

# # シードリスト
# seed_list=(0 1 2 3 4 5 6 7 8 9)

# # 実験をジョブとしてサブミット
# for seed in "${seed_list[@]}"; do
#     run_experiment "$seed" "$timestamp"
# done

# sleep 1

# # -----------------------------------------------------------------------------------------
# EXE_FILE="experiments/tf_continual.py"

# run_experiment() {
#     local seed=$1
#     local timestamp=$2
#     local plot_save_dir="$base_dir/$timestamp/plots"
#     mkdir -p "$plot_save_dir"

#     # 共通の引数を関数内で定義
#     local COMMON_ARGS=(
#         --function "$function"
#         --tf_lr 0.01
#         --tf_tol 1e-6
#         --tf_reg_lambda 0
#         --tf_constraint_lambda 1.0
#         --decomp_iter_num 10
#         --tf_max_iter 10000
#         --mask_ratio 1
#         --n_startup_trials 1
#         --iter_bo 500
#         --tf_method train
#         --tf_rank $tf_rank
#         --acqf_dist $acqf_dist # n, t1, t2
#     )

#     # map_option が空でない場合のみ追加
#     if [ -n "$map_option" ]; then
#         COMMON_ARGS+=(--map_option "$map_option")
#     fi

#     # オプションパラメータの追加
#     local constraint=$constraint
#     local direction=false
#     [ "$constraint" = true ] && COMMON_ARGS+=(--constraint)
#     [ "$direction" = true ] && COMMON_ARGS+=(--direction)

#     # sbatch でジョブをサブミット
#     sbatch --job-name="tf_experiment_seed_${seed}" <<EOF
# #!/bin/bash -l
# #SBATCH --partition=$partition_name
# #SBATCH --nodes=1
# #SBATCH --ntasks=1
# #SBATCH --cpus-per-task=20
# #SBATCH --time=$duration
# #SBATCH --output=%x_%j.log

# python3 $EXE_FILE \
#     --timestamp "$timestamp" \
#     --seed "$seed" \
#     --plot_save_dir "$plot_save_dir" \
#     --base_dir "$base_dir" \
#     ${COMMON_ARGS[*]}
# EOF
# }

# # 初期化
# timestamp=$(date +"%Y-%m-%d_%H-%M-%S")
# results_dir="$base_dir/$timestamp"
# mkdir -p "$results_dir"
# cp "$0" "$results_dir"

# # シードリスト
# seed_list=(0 1 2 3 4 5 6 7 8 9)

# # 実験をジョブとしてサブミット
# for seed in "${seed_list[@]}"; do
#     run_experiment "$seed" "$timestamp"
# done

# sleep 1

# # -----------------------------------------------------------------------------------------
# EXE_FILE="experiments/tf_continual.py"

# run_experiment() {
#     local seed=$1
#     local timestamp=$2
#     local plot_save_dir="$base_dir/$timestamp/plots"
#     mkdir -p "$plot_save_dir"

#     # 共通の引数を関数内で定義
#     local COMMON_ARGS=(
#         --function "$function"
#         --tf_lr 0.01
#         --tf_tol 1e-6
#         --tf_reg_lambda 0
#         --tf_constraint_lambda 1.0
#         --decomp_iter_num 10
#         --tf_max_iter 10000
#         --mask_ratio 1
#         --n_startup_trials 1
#         --iter_bo 500
#         --tf_method ring
#         --tf_rank $tf_rank
#         --acqf_dist $acqf_dist # n, t1, t2
#     )

#     # map_option が空でない場合のみ追加
#     if [ -n "$map_option" ]; then
#         COMMON_ARGS+=(--map_option "$map_option")
#     fi

#     # オプションパラメータの追加
#     local constraint=$constraint
#     local direction=false
#     [ "$constraint" = true ] && COMMON_ARGS+=(--constraint)
#     [ "$direction" = true ] && COMMON_ARGS+=(--direction)

#     # sbatch でジョブをサブミット
#     sbatch --job-name="tf_experiment_seed_${seed}" <<EOF
# #!/bin/bash -l
# #SBATCH --partition=$partition_name
# #SBATCH --nodes=1
# #SBATCH --ntasks=1
# #SBATCH --cpus-per-task=20
# #SBATCH --time=$duration
# #SBATCH --output=%x_%j.log

# python3 $EXE_FILE \
#     --timestamp "$timestamp" \
#     --seed "$seed" \
#     --plot_save_dir "$plot_save_dir" \
#     --base_dir "$base_dir" \
#     ${COMMON_ARGS[*]}
# EOF
# }

# # 初期化
# timestamp=$(date +"%Y-%m-%d_%H-%M-%S")
# results_dir="$base_dir/$timestamp"
# mkdir -p "$results_dir"
# cp "$0" "$results_dir"

# # シードリスト
# seed_list=(0 1 2 3 4 5 6 7 8 9)

# # 実験をジョブとしてサブミット
# for seed in "${seed_list[@]}"; do
#     run_experiment "$seed" "$timestamp"
# done

# sleep 1

# ###########################################################################################

# function="warcraft"
# map_option=2

# # -----------------------------------------------------------------------------------------
# EXE_FILE="experiments/tf_continual.py"

# run_experiment() {
#     local seed=$1
#     local timestamp=$2
#     local plot_save_dir="$base_dir/$timestamp/plots"
#     mkdir -p "$plot_save_dir"

#     # 共通の引数を関数内で定義
#     local COMMON_ARGS=(
#         --function "$function"
#         --tf_lr 0.01
#         --tf_tol 1e-6
#         --tf_reg_lambda 0
#         --tf_constraint_lambda 1.0
#         --decomp_iter_num 10
#         --tf_max_iter 10000
#         --mask_ratio 1
#         --n_startup_trials 1
#         --iter_bo 500
#         --tf_method cp
#         --tf_rank $tf_rank
#         --acqf_dist $acqf_dist # n, t1, t2
#     )

#     # map_option が空でない場合のみ追加
#     if [ -n "$map_option" ]; then
#         COMMON_ARGS+=(--map_option "$map_option")
#     fi

#     # オプションパラメータの追加
#     local constraint=$constraint
#     local direction=false
#     [ "$constraint" = true ] && COMMON_ARGS+=(--constraint)
#     [ "$direction" = true ] && COMMON_ARGS+=(--direction)

#     # sbatch でジョブをサブミット
#     sbatch --job-name="tf_experiment_seed_${seed}" <<EOF
# #!/bin/bash -l
# #SBATCH --partition=$partition_name
# #SBATCH --nodes=1
# #SBATCH --ntasks=1
# #SBATCH --cpus-per-task=20
# #SBATCH --time=$duration
# #SBATCH --output=%x_%j.log

# python3 $EXE_FILE \
#     --timestamp "$timestamp" \
#     --seed "$seed" \
#     --plot_save_dir "$plot_save_dir" \
#     --base_dir "$base_dir" \
#     ${COMMON_ARGS[*]}
# EOF
# }

# # 初期化
# timestamp=$(date +"%Y-%m-%d_%H-%M-%S")
# results_dir="$base_dir/$timestamp"
# mkdir -p "$results_dir"
# cp "$0" "$results_dir"

# # シードリスト
# seed_list=(0 1 2 3 4 5 6 7 8 9)

# # 実験をジョブとしてサブミット
# for seed in "${seed_list[@]}"; do
#     run_experiment "$seed" "$timestamp"
# done

# sleep 1

# # -----------------------------------------------------------------------------------------
# EXE_FILE="experiments/tf_continual.py"

# run_experiment() {
#     local seed=$1
#     local timestamp=$2
#     local plot_save_dir="$base_dir/$timestamp/plots"
#     mkdir -p "$plot_save_dir"

#     # 共通の引数を関数内で定義
#     local COMMON_ARGS=(
#         --function "$function"
#         --tf_lr 0.01
#         --tf_tol 1e-6
#         --tf_reg_lambda 0
#         --tf_constraint_lambda 1.0
#         --decomp_iter_num 10
#         --tf_max_iter 10000
#         --mask_ratio 1
#         --n_startup_trials 1
#         --iter_bo 500
#         --tf_method train
#         --tf_rank $tf_rank
#         --acqf_dist $acqf_dist # n, t1, t2
#     )

#     # map_option が空でない場合のみ追加
#     if [ -n "$map_option" ]; then
#         COMMON_ARGS+=(--map_option "$map_option")
#     fi

#     # オプションパラメータの追加
#     local constraint=$constraint
#     local direction=false
#     [ "$constraint" = true ] && COMMON_ARGS+=(--constraint)
#     [ "$direction" = true ] && COMMON_ARGS+=(--direction)

#     # sbatch でジョブをサブミット
#     sbatch --job-name="tf_experiment_seed_${seed}" <<EOF
# #!/bin/bash -l
# #SBATCH --partition=$partition_name
# #SBATCH --nodes=1
# #SBATCH --ntasks=1
# #SBATCH --cpus-per-task=20
# #SBATCH --time=$duration
# #SBATCH --output=%x_%j.log

# python3 $EXE_FILE \
#     --timestamp "$timestamp" \
#     --seed "$seed" \
#     --plot_save_dir "$plot_save_dir" \
#     --base_dir "$base_dir" \
#     ${COMMON_ARGS[*]}
# EOF
# }

# # 初期化
# timestamp=$(date +"%Y-%m-%d_%H-%M-%S")
# results_dir="$base_dir/$timestamp"
# mkdir -p "$results_dir"
# cp "$0" "$results_dir"

# # シードリスト
# seed_list=(0 1 2 3 4 5 6 7 8 9)

# # 実験をジョブとしてサブミット
# for seed in "${seed_list[@]}"; do
#     run_experiment "$seed" "$timestamp"
# done

# sleep 1

# # -----------------------------------------------------------------------------------------
# EXE_FILE="experiments/tf_continual.py"

# run_experiment() {
#     local seed=$1
#     local timestamp=$2
#     local plot_save_dir="$base_dir/$timestamp/plots"
#     mkdir -p "$plot_save_dir"

#     # 共通の引数を関数内で定義
#     local COMMON_ARGS=(
#         --function "$function"
#         --tf_lr 0.01
#         --tf_tol 1e-6
#         --tf_reg_lambda 0
#         --tf_constraint_lambda 1.0
#         --decomp_iter_num 10
#         --tf_max_iter 10000
#         --mask_ratio 1
#         --n_startup_trials 1
#         --iter_bo 500
#         --tf_method ring
#         --tf_rank $tf_rank
#         --acqf_dist $acqf_dist # n, t1, t2
#     )

#     # map_option が空でない場合のみ追加
#     if [ -n "$map_option" ]; then
#         COMMON_ARGS+=(--map_option "$map_option")
#     fi

#     # オプションパラメータの追加
#     local constraint=$constraint
#     local direction=false
#     [ "$constraint" = true ] && COMMON_ARGS+=(--constraint)
#     [ "$direction" = true ] && COMMON_ARGS+=(--direction)

#     # sbatch でジョブをサブミット
#     sbatch --job-name="tf_experiment_seed_${seed}" <<EOF
# #!/bin/bash -l
# #SBATCH --partition=$partition_name
# #SBATCH --nodes=1
# #SBATCH --ntasks=1
# #SBATCH --cpus-per-task=20
# #SBATCH --time=$duration
# #SBATCH --output=%x_%j.log

# python3 $EXE_FILE \
#     --timestamp "$timestamp" \
#     --seed "$seed" \
#     --plot_save_dir "$plot_save_dir" \
#     --base_dir "$base_dir" \
#     ${COMMON_ARGS[*]}
# EOF
# }

# # 初期化
# timestamp=$(date +"%Y-%m-%d_%H-%M-%S")
# results_dir="$base_dir/$timestamp"
# mkdir -p "$results_dir"
# cp "$0" "$results_dir"

# # シードリスト
# seed_list=(0 1 2 3 4 5 6 7 8 9)

# # 実験をジョブとしてサブミット
# for seed in "${seed_list[@]}"; do
#     run_experiment "$seed" "$timestamp"
# done

# sleep 1

# ###########################################################################################

function="ackley"
map_option=""

# -----------------------------------------------------------------------------------------
EXE_FILE="experiments/tf_continual.py"

run_experiment() {
    local seed=$1
    local timestamp=$2
    local plot_save_dir="$base_dir/$timestamp/plots"
    mkdir -p "$plot_save_dir"

    # 共通の引数を関数内で定義
    local COMMON_ARGS=(
        --function "$function"
        --tf_lr 0.01
        --tf_tol 1e-6
        --tf_reg_lambda 0
        --tf_constraint_lambda 1.0
        --decomp_iter_num 10
        --tf_max_iter 10000
        --mask_ratio 1
        --n_startup_trials 1
        --iter_bo 500
        --tf_method cp
        --tf_rank $tf_rank
        --acqf_dist $acqf_dist # n, t1, t2
    )

    # map_option が空でない場合のみ追加
    if [ -n "$map_option" ]; then
        COMMON_ARGS+=(--map_option "$map_option")
    fi

    # オプションパラメータの追加
    local constraint=$constraint
    local direction=false
    [ "$constraint" = true ] && COMMON_ARGS+=(--constraint)
    [ "$direction" = true ] && COMMON_ARGS+=(--direction)

    # sbatch でジョブをサブミット
    sbatch --job-name="tf_experiment_seed_${seed}" <<EOF
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
    ${COMMON_ARGS[*]}
EOF
}

# 初期化
timestamp=$(date +"%Y-%m-%d_%H-%M-%S")
results_dir="$base_dir/$timestamp"
mkdir -p "$results_dir"
cp "$0" "$results_dir"

# シードリスト
# seed_list=(0 1 2 3 4 5 6 7 8 9)
seed_list=(0)

# 実験をジョブとしてサブミット
for seed in "${seed_list[@]}"; do
    run_experiment "$seed" "$timestamp"
done

sleep 1

# # -----------------------------------------------------------------------------------------
# EXE_FILE="experiments/tf_continual.py"

# run_experiment() {
#     local seed=$1
#     local timestamp=$2
#     local plot_save_dir="$base_dir/$timestamp/plots"
#     mkdir -p "$plot_save_dir"

#     # 共通の引数を関数内で定義
#     local COMMON_ARGS=(
#         --function "$function"
#         --tf_lr 0.01
#         --tf_tol 1e-6
#         --tf_reg_lambda 0
#         --tf_constraint_lambda 1.0
#         --decomp_iter_num 10
#         --tf_max_iter 10000
#         --mask_ratio 1
#         --n_startup_trials 1
#         --iter_bo 500
#         --tf_method train
#         --tf_rank $tf_rank
#         --acqf_dist $acqf_dist # n, t1, t2
#     )

#     # map_option が空でない場合のみ追加
#     if [ -n "$map_option" ]; then
#         COMMON_ARGS+=(--map_option "$map_option")
#     fi

#     # オプションパラメータの追加
#     local constraint=$constraint
#     local direction=false
#     [ "$constraint" = true ] && COMMON_ARGS+=(--constraint)
#     [ "$direction" = true ] && COMMON_ARGS+=(--direction)

#     # sbatch でジョブをサブミット
#     sbatch --job-name="tf_experiment_seed_${seed}" <<EOF
# #!/bin/bash -l
# #SBATCH --partition=$partition_name
# #SBATCH --nodes=1
# #SBATCH --ntasks=1
# #SBATCH --cpus-per-task=20
# #SBATCH --time=$duration
# #SBATCH --output=%x_%j.log

# python3 $EXE_FILE \
#     --timestamp "$timestamp" \
#     --seed "$seed" \
#     --plot_save_dir "$plot_save_dir" \
#     --base_dir "$base_dir" \
#     ${COMMON_ARGS[*]}
# EOF
# }

# # 初期化
# timestamp=$(date +"%Y-%m-%d_%H-%M-%S")
# results_dir="$base_dir/$timestamp"
# mkdir -p "$results_dir"
# cp "$0" "$results_dir"

# # シードリスト
# seed_list=(0 1 2 3 4 5 6 7 8 9)

# # 実験をジョブとしてサブミット
# for seed in "${seed_list[@]}"; do
#     run_experiment "$seed" "$timestamp"
# done

# sleep 1

# # -----------------------------------------------------------------------------------------
# EXE_FILE="experiments/tf_continual.py"

# run_experiment() {
#     local seed=$1
#     local timestamp=$2
#     local plot_save_dir="$base_dir/$timestamp/plots"
#     mkdir -p "$plot_save_dir"

#     # 共通の引数を関数内で定義
#     local COMMON_ARGS=(
#         --function "$function"
#         --tf_lr 0.01
#         --tf_tol 1e-6
#         --tf_reg_lambda 0
#         --tf_constraint_lambda 1.0
#         --decomp_iter_num 10
#         --tf_max_iter 10000
#         --mask_ratio 1
#         --n_startup_trials 1
#         --iter_bo 500
#         --tf_method ring
#         --tf_rank $tf_rank
#         --acqf_dist $acqf_dist # n, t1, t2
#     )

#     # map_option が空でない場合のみ追加
#     if [ -n "$map_option" ]; then
#         COMMON_ARGS+=(--map_option "$map_option")
#     fi

#     # オプションパラメータの追加
#     local constraint=$constraint
#     local direction=false
#     [ "$constraint" = true ] && COMMON_ARGS+=(--constraint)
#     [ "$direction" = true ] && COMMON_ARGS+=(--direction)

#     # sbatch でジョブをサブミット
#     sbatch --job-name="tf_experiment_seed_${seed}" <<EOF
# #!/bin/bash -l
# #SBATCH --partition=$partition_name
# #SBATCH --nodes=1
# #SBATCH --ntasks=1
# #SBATCH --cpus-per-task=20
# #SBATCH --time=$duration
# #SBATCH --output=%x_%j.log

# python3 $EXE_FILE \
#     --timestamp "$timestamp" \
#     --seed "$seed" \
#     --plot_save_dir "$plot_save_dir" \
#     --base_dir "$base_dir" \
#     ${COMMON_ARGS[*]}
# EOF
# }

# # 初期化
# timestamp=$(date +"%Y-%m-%d_%H-%M-%S")
# results_dir="$base_dir/$timestamp"
# mkdir -p "$results_dir"
# cp "$0" "$results_dir"

# # シードリスト
# seed_list=(0 1 2 3 4 5 6 7 8 9)

# # 実験をジョブとしてサブミット
# for seed in "${seed_list[@]}"; do
#     run_experiment "$seed" "$timestamp"
# done

# sleep 1



echo "All experiments submitted!"
