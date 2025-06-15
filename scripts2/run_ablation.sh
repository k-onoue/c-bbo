#!/bin/bash -l
#SBATCH --job-name=bo_array_exp     # Job name
#SBATCH --partition=cluster_short   # Queue
#SBATCH --array=0-9                 # Array 0–9

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=01:00:00

#SBATCH --output=logs/experiment_%A_%a.out
#SBATCH --error=logs/experiment_%A_%a.err

### --- Prepare environment --- ###
# source /path/to/venv/bin/activate   # if you need a virtualenv

### --- Seed and timestamp --- ###
SEED=$SLURM_ARRAY_TASK_ID
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

SAVEDIR="results_v2"
mkdir -p logs $SAVEDIR
mkdir -p "$SAVEDIR/$TIMESTAMP/plots" 
cp "$0" "$SAVEDIR/$TIMESTAMP/"


echo "Starting experiment for seed $SEED at $TIMESTAMP..."

### --- Build argument array --- ###
declare -a FLAGS=(
  # --- General parameters ---
  --timestamp        "$TIMESTAMP"
  --seed             "$SEED"
  --iter_bo          500
  --function         "ackley"
  --dimension        2
  # ↓ 必要時にアンコメント
  #--map_option       1       # For 'warcraft'
  --constraint               # Use constraint
  #--direction                # Maximize objective

  # --- TF-specific arguments ---
  --tf_method         "train"
  --tf_rank           3
  --tf_lr             0.01
  --tf_max_iter       1000
  --tf_tol            1e-5
  --tf_reg_lambda     1e-3
  --tf_constraint_lambda 1

  # --- Sampler parameters ---
  --decomp_iter_num   10
  --mask_ratio        1.0
  #--include_observed_points  # Uncomment to activate
  #--unique_sampling          # Uncomment to activate
  --n_startup_trials  1
  --acqf_dist         "n"

  # --- Save directory ---
  --base_dir          "$SAVEDIR"
  --plot_save_dir     "$SAVEDIR/$TIMESTAMP/plots"
)

### --- Run experiment --- ###
python experiments/tf_continual.py "${FLAGS[@]}"

echo "Experiment finished for seed $SEED."
