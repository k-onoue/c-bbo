#!/bin/bash

EXE_FILE="experiments/benchmark.py"
EXE_FILE_C="experiments/benchmark-constrained.py"

# Experiment configurations
iter_bo=3
n_startup_trials=1
n_init_violation_paths=20
sampler_list=("tpe" "gp")
functions=("warcraft" "ackley")
map_options=(1 2)
# seed_list=(0 1 2 3 4 5 6 7 8 9)
seed_list=(0 1)
constraint=false

# Initialize timestamp and directories
timestamp=$(date +"%Y-%m-%d_%H-%M-%S")
results_dir="results/$timestamp"
plot_save_dir="$results_dir/plots"
mkdir -p "$results_dir" "$plot_save_dir"
cp "$0" "$results_dir"

# Run all jobs
for function in "${functions[@]}"; do
    for sampler in "${sampler_list[@]}"; do
        for seed in "${seed_list[@]}"; do
            if [ "$function" = "warcraft" ]; then
                for map_option in "${map_options[@]}"; do
                    log_file="log-$function-$sampler-$seed-$map_option.log"

                    echo "Running: $function $sampler seed=$seed map_option=$map_option"

                    if [ "$constraint" = true ]; then
                        python3 "$EXE_FILE_C" \
                            --timestamp "$timestamp" \
                            --function "$function" \
                            --sampler "$sampler" \
                            --iter_bo "$iter_bo" \
                            --n_init_violation_paths "$n_init_violation_paths" \
                            --seed "$seed" \
                            --map_option "$map_option" \
                            --n_startup_trials "$n_startup_trials" \
                            --plot_save_dir "$plot_save_dir" \
                            &> "$log_file" &
                    else
                        python3 "$EXE_FILE" \
                            --timestamp "$timestamp" \
                            --function "$function" \
                            --sampler "$sampler" \
                            --iter_bo "$iter_bo" \
                            --seed "$seed" \
                            --map_option "$map_option" \
                            --n_startup_trials "$n_startup_trials" \
                            --plot_save_dir "$plot_save_dir" \
                            &> "$log_file" &
                    fi
                done
            else
                log_file="log-$function-$sampler-$seed.log"

                echo "Running: $function $sampler seed=$seed"

                if [ "$constraint" = true ]; then
                    python3 "$EXE_FILE_C" \
                        --timestamp "$timestamp" \
                        --function "$function" \
                        --sampler "$sampler" \
                        --iter_bo "$iter_bo" \
                        --n_init_violation_paths "$n_init_violation_paths" \
                        --seed "$seed" \
                        --n_startup_trials "$n_startup_trials" \
                        --plot_save_dir "$plot_save_dir" \
                        &> "$log_file" &
                else
                    python3 "$EXE_FILE" \
                        --timestamp "$timestamp" \
                        --function "$function" \
                        --sampler "$sampler" \
                        --iter_bo "$iter_bo" \
                        --seed "$seed" \
                        --n_startup_trials "$n_startup_trials" \
                        --plot_save_dir "$plot_save_dir" \
                        &> "$log_file" &
                fi
            fi
        done
    done
done

# Wait for all background jobs to finish
wait

echo "All jobs completed."
