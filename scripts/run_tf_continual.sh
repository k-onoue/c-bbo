#!/bin/bash

EXE_FILE="experiments/tf_continual.py"

run_experiment() {
    local timestamp=$1
    local seed=$2
    local map_option=$3
    local tf_method=$4
    local tf_rank=$5
    local constraint=$6
    local direction=$7
    local plot_save_dir=$8
    local tf_max_iter=$9
    local n_startup_trials=${10}
    local iter_bo=${11}
    local acquisition_function=${12}
    local mask_ratio=${13}

    local cmd=(
        python3 "$EXE_FILE"
        --timestamp "$timestamp"
        # --function "warcraft"
        --function "ackley"
        # --function "eggholder"
        --seed "$seed"
        --map_option "$map_option"
        # TF parameters
        --tf_method "$tf_method"
        --tf_rank "$tf_rank"
        --tf_lr 0.01
        --tf_tol 1e-6
        --tf_reg_lambda 0
        --tf_constraint_lambda 1.0
        # Sampler parameters
        --decomp_iter_num 10
        --mask_ratio "$mask_ratio"
        --n_startup_trials "$n_startup_trials"
        # Acquisition function parameters
        # Other parameters
        --iter_bo "$iter_bo"
        --plot_save_dir "$plot_save_dir"
        --acqf_dist n
    )

    [ "$tf_max_iter" != "None" ] && cmd+=(--tf_max_iter "$tf_max_iter")
    [ "$constraint" = true ] && cmd+=(--constraint)
    [ "$direction" = true ] && cmd+=(--direction)

    echo "Running: ${cmd[*]}"
    "${cmd[@]}"
}

# Initialize timestamp and directories
timestamp=$(date +"%Y-%m-%d_%H-%M-%S")
results_dir="results/$timestamp"
plot_save_dir="$results_dir/plots"
mkdir -p "$results_dir" "$plot_save_dir"
cp "$0" "$results_dir"

# Experiment configurations
map_options=(1)
seed_list=(0 1 2 3 4 5 6 7 8 9) 

# Algorithm parameters
tf_methods=("train")
tf_ranks=(3)
acquisition_function="ei"
mask_ratio=1
tf_max_iter=10000

# Flags and other settings
constraint=false
direction=false
n_startup_trials=1
iter_bo=500

# Run experiments
for map_option in "${map_options[@]}"; do
    for tf_method in "${tf_methods[@]}"; do
        for tf_rank in "${tf_ranks[@]}"; do
            for seed in "${seed_list[@]}"; do
                run_experiment "$timestamp" "$seed" "$map_option" \
                    "$tf_method" "$tf_rank" \
                    "$constraint" "$direction" "$plot_save_dir" \
                    "$tf_max_iter" "$n_startup_trials" "$iter_bo" \
                    "$acquisition_function" "$mask_ratio" &
                
            done
            wait
        done
    done
done


#############################################################################################################
#!/bin/bash

EXE_FILE="experiments/tf_continual.py"

run_experiment() {
    local timestamp=$1
    local seed=$2
    local map_option=$3
    local tf_method=$4
    local tf_rank=$5
    local constraint=$6
    local direction=$7
    local plot_save_dir=$8
    local tf_max_iter=$9
    local n_startup_trials=${10}
    local iter_bo=${11}
    local acquisition_function=${12}
    local mask_ratio=${13}

    local cmd=(
        python3 "$EXE_FILE"
        --timestamp "$timestamp"
        # --function "warcraft"
        --function "ackley"
        # --function "eggholder"
        --seed "$seed"
        --map_option "$map_option"
        # TF parameters
        --tf_method "$tf_method"
        --tf_rank "$tf_rank"
        --tf_lr 0.01
        --tf_tol 1e-6
        --tf_reg_lambda 0
        --tf_constraint_lambda 1.0
        # Sampler parameters
        --decomp_iter_num 10
        --mask_ratio "$mask_ratio"
        --n_startup_trials "$n_startup_trials"
        # Acquisition function parameters
        # Other parameters
        --iter_bo "$iter_bo"
        --plot_save_dir "$plot_save_dir"
        --acqf_dist t1
    )

    [ "$tf_max_iter" != "None" ] && cmd+=(--tf_max_iter "$tf_max_iter")
    [ "$constraint" = true ] && cmd+=(--constraint)
    [ "$direction" = true ] && cmd+=(--direction)

    echo "Running: ${cmd[*]}"
    "${cmd[@]}"
}

# Initialize timestamp and directories
timestamp=$(date +"%Y-%m-%d_%H-%M-%S")
results_dir="results/$timestamp"
plot_save_dir="$results_dir/plots"
mkdir -p "$results_dir" "$plot_save_dir"
cp "$0" "$results_dir"

# Experiment configurations
map_options=(1)
seed_list=(0 1 2 3 4 5 6 7 8 9) 

# Algorithm parameters
tf_methods=("train")
tf_ranks=(3)
acquisition_function="ei"
mask_ratio=1
tf_max_iter=10000

# Flags and other settings
constraint=false
direction=false
n_startup_trials=1
iter_bo=500

# Run experiments
for map_option in "${map_options[@]}"; do
    for tf_method in "${tf_methods[@]}"; do
        for tf_rank in "${tf_ranks[@]}"; do
            for seed in "${seed_list[@]}"; do
                run_experiment "$timestamp" "$seed" "$map_option" \
                    "$tf_method" "$tf_rank" \
                    "$constraint" "$direction" "$plot_save_dir" \
                    "$tf_max_iter" "$n_startup_trials" "$iter_bo" \
                    "$acquisition_function" "$mask_ratio" &
                
            done
            wait
        done
    done
done

#############################################################################################################
#!/bin/bash

EXE_FILE="experiments/tf_continual.py"

run_experiment() {
    local timestamp=$1
    local seed=$2
    local map_option=$3
    local tf_method=$4
    local tf_rank=$5
    local constraint=$6
    local direction=$7
    local plot_save_dir=$8
    local tf_max_iter=$9
    local n_startup_trials=${10}
    local iter_bo=${11}
    local acquisition_function=${12}
    local mask_ratio=${13}

    local cmd=(
        python3 "$EXE_FILE"
        --timestamp "$timestamp"
        # --function "warcraft"
        --function "ackley"
        # --function "eggholder"
        --seed "$seed"
        --map_option "$map_option"
        # TF parameters
        --tf_method "$tf_method"
        --tf_rank "$tf_rank"
        --tf_lr 0.01
        --tf_tol 1e-6
        --tf_reg_lambda 0
        --tf_constraint_lambda 1.0
        # Sampler parameters
        --decomp_iter_num 10
        --mask_ratio "$mask_ratio"
        --n_startup_trials "$n_startup_trials"
        # Acquisition function parameters
        # Other parameters
        --iter_bo "$iter_bo"
        --plot_save_dir "$plot_save_dir"
        --acqf_dist t2
    )

    [ "$tf_max_iter" != "None" ] && cmd+=(--tf_max_iter "$tf_max_iter")
    [ "$constraint" = true ] && cmd+=(--constraint)
    [ "$direction" = true ] && cmd+=(--direction)

    echo "Running: ${cmd[*]}"
    "${cmd[@]}"
}

# Initialize timestamp and directories
timestamp=$(date +"%Y-%m-%d_%H-%M-%S")
results_dir="results/$timestamp"
plot_save_dir="$results_dir/plots"
mkdir -p "$results_dir" "$plot_save_dir"
cp "$0" "$results_dir"

# Experiment configurations
map_options=(1)
seed_list=(0 1 2 3 4 5 6 7 8 9) 

# Algorithm parameters
tf_methods=("train")
tf_ranks=(3)
acquisition_function="ei"
mask_ratio=1
tf_max_iter=10000

# Flags and other settings
constraint=false
direction=false
n_startup_trials=1
iter_bo=500

# Run experiments
for map_option in "${map_options[@]}"; do
    for tf_method in "${tf_methods[@]}"; do
        for tf_rank in "${tf_ranks[@]}"; do
            for seed in "${seed_list[@]}"; do
                run_experiment "$timestamp" "$seed" "$map_option" \
                    "$tf_method" "$tf_rank" \
                    "$constraint" "$direction" "$plot_save_dir" \
                    "$tf_max_iter" "$n_startup_trials" "$iter_bo" \
                    "$acquisition_function" "$mask_ratio" &
                
            done
            wait
        done
    done
done

#############################################################################################################
#!/bin/bash

EXE_FILE="experiments/tf_continual.py"

run_experiment() {
    local timestamp=$1
    local seed=$2
    local map_option=$3
    local tf_method=$4
    local tf_rank=$5
    local constraint=$6
    local direction=$7
    local plot_save_dir=$8
    local tf_max_iter=$9
    local n_startup_trials=${10}
    local iter_bo=${11}
    local acquisition_function=${12}
    local mask_ratio=${13}

    local cmd=(
        python3 "$EXE_FILE"
        --timestamp "$timestamp"
        # --function "warcraft"
        --function "ackley"
        # --function "eggholder"
        --seed "$seed"
        --map_option "$map_option"
        # TF parameters
        --tf_method "$tf_method"
        --tf_rank "$tf_rank"
        --tf_lr 0.01
        --tf_tol 1e-6
        --tf_reg_lambda 0
        --tf_constraint_lambda 1.0
        # Sampler parameters
        --decomp_iter_num 10
        --mask_ratio "$mask_ratio"
        --n_startup_trials "$n_startup_trials"
        # Acquisition function parameters
        # Other parameters
        --iter_bo "$iter_bo"
        --plot_save_dir "$plot_save_dir"
        --acqf_dist n
    )

    [ "$tf_max_iter" != "None" ] && cmd+=(--tf_max_iter "$tf_max_iter")
    [ "$constraint" = true ] && cmd+=(--constraint)
    [ "$direction" = true ] && cmd+=(--direction)

    echo "Running: ${cmd[*]}"
    "${cmd[@]}"
}

# Initialize timestamp and directories
timestamp=$(date +"%Y-%m-%d_%H-%M-%S")
results_dir="results/$timestamp"
plot_save_dir="$results_dir/plots"
mkdir -p "$results_dir" "$plot_save_dir"
cp "$0" "$results_dir"

# Experiment configurations
map_options=(1)
seed_list=(0 1 2 3 4 5 6 7 8 9) 

# Algorithm parameters
tf_methods=("train")
tf_ranks=(3)
acquisition_function="ei"
mask_ratio=1
tf_max_iter=10000

# Flags and other settings
constraint=true
direction=false
n_startup_trials=1
iter_bo=500

# Run experiments
for map_option in "${map_options[@]}"; do
    for tf_method in "${tf_methods[@]}"; do
        for tf_rank in "${tf_ranks[@]}"; do
            for seed in "${seed_list[@]}"; do
                run_experiment "$timestamp" "$seed" "$map_option" \
                    "$tf_method" "$tf_rank" \
                    "$constraint" "$direction" "$plot_save_dir" \
                    "$tf_max_iter" "$n_startup_trials" "$iter_bo" \
                    "$acquisition_function" "$mask_ratio" &
                
            done
            wait
        done
    done
done


#############################################################################################################
#!/bin/bash

EXE_FILE="experiments/tf_continual.py"

run_experiment() {
    local timestamp=$1
    local seed=$2
    local map_option=$3
    local tf_method=$4
    local tf_rank=$5
    local constraint=$6
    local direction=$7
    local plot_save_dir=$8
    local tf_max_iter=$9
    local n_startup_trials=${10}
    local iter_bo=${11}
    local acquisition_function=${12}
    local mask_ratio=${13}

    local cmd=(
        python3 "$EXE_FILE"
        --timestamp "$timestamp"
        # --function "warcraft"
        --function "ackley"
        # --function "eggholder"
        --seed "$seed"
        --map_option "$map_option"
        # TF parameters
        --tf_method "$tf_method"
        --tf_rank "$tf_rank"
        --tf_lr 0.01
        --tf_tol 1e-6
        --tf_reg_lambda 0
        --tf_constraint_lambda 1.0
        # Sampler parameters
        --decomp_iter_num 10
        --mask_ratio "$mask_ratio"
        --n_startup_trials "$n_startup_trials"
        # Acquisition function parameters
        # Other parameters
        --iter_bo "$iter_bo"
        --plot_save_dir "$plot_save_dir"
        --acqf_dist t1
    )

    [ "$tf_max_iter" != "None" ] && cmd+=(--tf_max_iter "$tf_max_iter")
    [ "$constraint" = true ] && cmd+=(--constraint)
    [ "$direction" = true ] && cmd+=(--direction)

    echo "Running: ${cmd[*]}"
    "${cmd[@]}"
}

# Initialize timestamp and directories
timestamp=$(date +"%Y-%m-%d_%H-%M-%S")
results_dir="results/$timestamp"
plot_save_dir="$results_dir/plots"
mkdir -p "$results_dir" "$plot_save_dir"
cp "$0" "$results_dir"

# Experiment configurations
map_options=(1)
seed_list=(0 1 2 3 4 5 6 7 8 9) 

# Algorithm parameters
tf_methods=("train")
tf_ranks=(3)
acquisition_function="ei"
mask_ratio=1
tf_max_iter=10000

# Flags and other settings
constraint=true
direction=false
n_startup_trials=1
iter_bo=500

# Run experiments
for map_option in "${map_options[@]}"; do
    for tf_method in "${tf_methods[@]}"; do
        for tf_rank in "${tf_ranks[@]}"; do
            for seed in "${seed_list[@]}"; do
                run_experiment "$timestamp" "$seed" "$map_option" \
                    "$tf_method" "$tf_rank" \
                    "$constraint" "$direction" "$plot_save_dir" \
                    "$tf_max_iter" "$n_startup_trials" "$iter_bo" \
                    "$acquisition_function" "$mask_ratio" &
                
            done
            wait
        done
    done
done

#############################################################################################################
#!/bin/bash

EXE_FILE="experiments/tf_continual.py"

run_experiment() {
    local timestamp=$1
    local seed=$2
    local map_option=$3
    local tf_method=$4
    local tf_rank=$5
    local constraint=$6
    local direction=$7
    local plot_save_dir=$8
    local tf_max_iter=$9
    local n_startup_trials=${10}
    local iter_bo=${11}
    local acquisition_function=${12}
    local mask_ratio=${13}

    local cmd=(
        python3 "$EXE_FILE"
        --timestamp "$timestamp"
        # --function "warcraft"
        --function "ackley"
        # --function "eggholder"
        --seed "$seed"
        --map_option "$map_option"
        # TF parameters
        --tf_method "$tf_method"
        --tf_rank "$tf_rank"
        --tf_lr 0.01
        --tf_tol 1e-6
        --tf_reg_lambda 0
        --tf_constraint_lambda 1.0
        # Sampler parameters
        --decomp_iter_num 10
        --mask_ratio "$mask_ratio"
        --n_startup_trials "$n_startup_trials"
        # Acquisition function parameters
        # Other parameters
        --iter_bo "$iter_bo"
        --plot_save_dir "$plot_save_dir"
        --acqf_dist t2
    )

    [ "$tf_max_iter" != "None" ] && cmd+=(--tf_max_iter "$tf_max_iter")
    [ "$constraint" = true ] && cmd+=(--constraint)
    [ "$direction" = true ] && cmd+=(--direction)

    echo "Running: ${cmd[*]}"
    "${cmd[@]}"
}

# Initialize timestamp and directories
timestamp=$(date +"%Y-%m-%d_%H-%M-%S")
results_dir="results/$timestamp"
plot_save_dir="$results_dir/plots"
mkdir -p "$results_dir" "$plot_save_dir"
cp "$0" "$results_dir"

# Experiment configurations
map_options=(1)
seed_list=(0 1 2 3 4 5 6 7 8 9) 

# Algorithm parameters
tf_methods=("train")
tf_ranks=(3)
acquisition_function="ei"
mask_ratio=1
tf_max_iter=10000

# Flags and other settings
constraint=true
direction=false
n_startup_trials=1
iter_bo=500

# Run experiments
for map_option in "${map_options[@]}"; do
    for tf_method in "${tf_methods[@]}"; do
        for tf_rank in "${tf_ranks[@]}"; do
            for seed in "${seed_list[@]}"; do
                run_experiment "$timestamp" "$seed" "$map_option" \
                    "$tf_method" "$tf_rank" \
                    "$constraint" "$direction" "$plot_save_dir" \
                    "$tf_max_iter" "$n_startup_trials" "$iter_bo" \
                    "$acquisition_function" "$mask_ratio" &
                
            done
            wait
        done
    done
done

echo "All experiments completed!"