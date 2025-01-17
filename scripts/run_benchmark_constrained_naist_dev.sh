#!/bin/bash -l
#SBATCH --job-name=benchmark
#SBATCH --partition=cluster_short
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --time=4:00:00

EXE_FILE="experiments/benchmark-constrained.py"

# Experiment configurations
iter_bo=500
n_startup_trials=1
n_init_violation_paths=200
sampler_list=("gp")
functions=("warcraft")
dimensions=(4)
map_options=(1 2)
seed_list=(0 1 2 3 4 5 6 7 8 9)

# Calculate total number of tasks
function_count=${#functions[@]}
seed_count=${#seed_list[@]}
map_option_count=${#map_options[@]}
total_tasks=$((function_count * seed_count * map_option_count))

# Flattened task index
if (( SLURM_ARRAY_TASK_ID >= total_tasks )); then
    echo "Task ID $SLURM_ARRAY_TASK_ID exceeds total task count $total_tasks. Exiting."
    exit 1
fi

# Extract parameters based on task ID
function_index=$((SLURM_ARRAY_TASK_ID / (seed_count * map_option_count)))
remaining=$((SLURM_ARRAY_TASK_ID % (seed_count * map_option_count)))
map_option_index=$((remaining / seed_count))
seed_index=$((remaining % seed_count))

function=${functions[function_index]}
seed=${seed_list[seed_index]}
map_option=${map_options[map_option_index]}

# Restrict iter_bo for gp sampler
sampler="gp"
if [ "$sampler" == "gp" ] && [ "$iter_bo" -gt 500 ]; then
    iter_bo=500
fi

# Initialize timestamp and results directory
# timestamp=$(date +"%Y-%m-%d_%H-%M-%S")
timestamp="2025-01-17_18-00-00"
results_dir="results/$timestamp"
mkdir -p "$results_dir"

# Run the benchmark
cmd=(
    python3 "$EXE_FILE"
    --timestamp "$timestamp"
    --function "$function"
    --sampler "$sampler"
    --dimension 2
    --iter_bo "$iter_bo"
    --n_init_violation_paths "$n_init_violation_paths"
    --seed "$seed"
    --map_option "$map_option"
    --n_startup_trials "$n_startup_trials"
)

echo "Running: ${cmd[*]}"
"${cmd[@]}"
