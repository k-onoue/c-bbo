# --- Configuration ---
EXE_FILE="experiments/benchmark-constrained.py"
BASE_DIR="results_ackley_constrained_gp" # Fixed base directory name
FUNCTION_NAME="ackley"
ITER_BO=500
N_STARTUP_TRIALS=1
N_INIT_VIOLATION_PATHS=200
DIMENSION=2
SAMPLER_TO_RUN="gp"
SEEDS=(0 1 0 1 0 1) # Example seeds - adjust as needed
PAUSE_AFTER_N_SEEDS=12
PAUSE_DURATION_SECONDS=$((2 * 3600 + 5 * 60)) # 2 hours and 5 minutes

# --- SLURM Settings for individual jobs ---
JOB_PARTITION_NAME="cluster_short" # Partition for the actual compute jobs
JOB_DURATION="04:00:00"           # Time limit *per job*
JOB_CPUS=20                        # CPUs *per job*
JOB_MEM="16G"                      # Memory *per job*

# --- Base Arguments for the Python Script (Common across seeds) ---
BASE_ARGS=(
    --function "$FUNCTION_NAME"
    --iter_bo "$ITER_BO"
    --n_startup_trials "$N_STARTUP_TRIALS"
    --n_init_violation_paths "$N_INIT_VIOLATION_PATHS"
    --dimension "$DIMENSION"
    --sampler "$SAMPLER_TO_RUN"
    # --acq_trade_off_param 3.0 # Example overrides
    # --acq_batch_size 10
)


timestamp=$(date +"%Y-%m-%d_%H-%M-%S")
# --- Experiment Execution Function ---
run_experiment() {
    local seed=$1
    local sampler=$2 # Fixed to "gp"
    local base_dir=$3 # Pass base directory (e.g., results_ackley_constrained_gp)

    # Define directories based on the timestamp
    local results_dir="$base_dir/$timestamp" # Main directory for this specific run
    local plot_save_dir="$results_dir/plots"  # Subdirectory for plots

    mkdir -p "$plot_save_dir" # This also creates results_dir if needed
    cp "$0" "$results_dir/"   # Save this submission script for reproducibility

    # Experiment-specific arguments for Python script
    local SCRIPT_ARGS=("${BASE_ARGS[@]}")
    SCRIPT_ARGS+=(--seed "$seed")
    SCRIPT_ARGS+=(--timestamp "$timestamp") # Pass this timestamp to Python
    SCRIPT_ARGS+=(--plot_save_dir "$plot_save_dir")
    # SCRIPT_ARGS+=(--base_dir "$base_dir") # Add if your python script needs this

    # Job name for SLURM
    # Note: Python script uses timestamp + function + sampler + seed for Optuna study name
    local job_name="ackley_con_${sampler}_s${seed}_ts${timestamp}" # Make job name unique

    echo "Submitting job: $job_name"
    echo "  Seed: $seed, Sampler: $sampler"
    echo "  Timestamp: $timestamp"
    echo "  Results Dir: $results_dir"
    echo "  Plot Dir: $plot_save_dir"

    # Submit job using sbatch
    # Note: SLURM log file will be placed inside the results dir
    sbatch --job-name="$job_name" \
           --output="${results_dir}/slurm_%x_%j.log" \
           --partition="$JOB_PARTITION_NAME" \
           --nodes=1 \
           --ntasks=1 \
           --cpus-per-task="$JOB_CPUS" \
           --time="$JOB_DURATION" \
           --mem="$JOB_MEM" <<EOF
#!/bin/bash -l
#SBATCH --job-name=$job_name
# Ensure SLURM log path is accessible inside job script
#SBATCH --output=${results_dir}/slurm_%x_%j.log

echo "Starting job $job_name on node \$(hostname)"
echo "Timestamp: $timestamp"
echo "Seed: $seed"
echo "Sampler: $sampler"
echo "Python script: $EXE_FILE"
echo "Results Dir: $results_dir" # For logging purposes inside the job

# --- Activate Conda Environment (adjust path if needed) ---
# echo "Activating Conda environment..."
# source /path/to/your/anaconda3/etc/profile.d/conda.sh # Adjust this path
# conda activate bo-env_v3 # Use your actual environment name
# echo "Conda environment activated."

# --- Run Python Script ---
# The Python script should handle creating the .db file inside results_dir
# based on the timestamp and other args.
echo "Running Python script..."
python3 "$EXE_FILE" ${SCRIPT_ARGS[@]}

echo "Job $job_name finished."
EOF

    # Check sbatch exit status (optional but good practice)
    local sbatch_exit_status=$?
    if [ $sbatch_exit_status -ne 0 ]; then
        echo "Error submitting job $job_name (exit status: $sbatch_exit_status)" >&2
        # Consider whether to exit the script or just continue
        # exit 1 # Uncomment to stop submission on error
    fi


    # sleep 1 # Small sleep to avoid overwhelming scheduler immediately
}

# --- Main Loop ---
# Create the main base directory *once*
mkdir -p "$BASE_DIR"
echo "Main results directory: $BASE_DIR"
echo "Running constrained Ackley with GP sampler."
echo "Pausing for ${PAUSE_DURATION_SECONDS}s after every ${PAUSE_AFTER_N_SEEDS} seeds."

job_count=0
total_seeds=${#SEEDS[@]}

for seed in "${SEEDS[@]}"; do
    # Pass BASE_DIR to the function
    run_experiment "$seed" "$SAMPLER_TO_RUN" "$BASE_DIR"
    job_count=$((job_count + 1))

    # Pause after every N jobs, but not after the very last one
    if (( job_count % PAUSE_AFTER_N_SEEDS == 0 && job_count < total_seeds )); then
        echo "-----------------------------------------------------"
        echo "Submitted ${job_count} jobs. Pausing for ${PAUSE_DURATION_SECONDS} seconds..."
        echo "Next submission will start around: $(date -d "+${PAUSE_DURATION_SECONDS} seconds")"
        echo "-----------------------------------------------------"
        sleep "$PAUSE_DURATION_SECONDS"
        echo "-----------------------------------------------------"
        echo "Resuming submissions."
        echo "-----------------------------------------------------"
    fi
done

echo "All constrained Ackley GP jobs submitted!"