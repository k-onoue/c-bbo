#!/bin/bash -l
#SBATCH --job-name=benchmark
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#SBATCH --partition=cluster_short
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=4:00:00

EXE_FILE="experiments/benchmark-constrained.py"

# Experiment configurations
iter_bo=500
n_startup_trials=1
n_init_violation_paths=200
sampler_list=("tpe")
functions=("warcraft")
dimensions=(4)
map_options=(1 2)
seed_list=(0 1 2 3 4 5 6 7 8 9)

# Initialize timestamp and results directory
timestamp=$(date +"%Y-%m-%d_%H-%M-%S")
results_dir="results/$timestamp"
mkdir -p "$results_dir"

# Submit all jobs
for function in "${functions[@]}"; do
    for dimension in "${dimensions[@]}"; do
        for sampler in "${sampler_list[@]}"; do
            for seed in "${seed_list[@]}"; do
                for map_option in "${map_options[@]}"; do
                    sbatch --export=ALL,EXE_FILE="$EXE_FILE",timestamp="$timestamp",function="$function",sampler="$sampler",dimension="$dimension",iter_bo="$iter_bo",n_init_violation_paths="$n_init_violation_paths",seed="$seed",map_option="$map_option",n_startup_trials="$n_startup_trials" <<EOT
#!/bin/bash -l
#SBATCH --job-name=benchmark-$function-$sampler-$seed
#SBATCH --output=$results_dir/slurm-$function-$sampler-$seed-%j.out
#SBATCH --error=$results_dir/slurm-$function-$sampler-$seed-%j.err
#SBATCH --partition=cluster_short
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=4:00:00

python3 "$EXE_FILE" \
    --timestamp "$timestamp" \
    --function "$function" \
    --sampler "$sampler" \
    --dimension "$dimension" \
    --iter_bo "$iter_bo" \
    --n_init_violation_paths "$n_init_violation_paths" \
    --seed "$seed" \
    --map_option "$map_option" \
    --n_startup_trials "$n_startup_trials"
EOT
                done
            done
        done
    done
done
