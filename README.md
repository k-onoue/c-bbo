# Constraint-Aware Discrete Black-Box Optimization Using Tensor Decomposition


## 1. Installation

### 1.1 Prerequisites

*   **OS:** Ubuntu 22.04.5 LTS
*   **Python:** 3.12.2

### 1.2. Installation Steps

Clone the repository and set up the virtual environment by executing the following commands. The `requirements.txt` file includes all necessary libraries such as PyTorch, NumPy, cvxpy, and OR-Tools.

```sh
git clone https://github.com/k-onoue/c-bbo.git
cd c-bbo
pyenv local 3.12
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# This command writes the absolute path of the project to the config file.
sed -i "s|^project_dir = .*|project_dir = $(pwd)|" config.ini
```

## 2. Directory Structure

```
.
├── src/                # Python source code
├── experiments/        # Scripts and definitions for reproducing experiments
├── data/               # Datasets
├── scripts/            # Scripts to run experiments
├── config.ini          # Configuration file for paths
├── run_main.sh         # Shell script to run the main experiments
├── run_ablation.sh     # Shell script to run the ablation studies
└── requirements.txt    # List of dependencies
```

## 3. Usage

We use `optuna` for the framework to execute optimization problems.

The pseudocode below demonstrates how our proposed method can be used.

For more detailed information, see `./experiments/tf_continual.py` directly.

```python
// Minimal pseudocode for running a constrained TFContinualSampler.

// ## 1. Setup
// First, define the objective function and its associated constraints.

function objective_function(trial):
  x1 = trial.suggest_categorical("x1", [list_of_categories_for_x1])
  x2 = trial.suggest_categorical("x2", [list_of_categories_for_x2])
  x = [x1, x2]
  return calculate_value(x)

// Obtain an numpy array representing the constraints with binary (0 or 1) valued entries. 
tensor_constraint = get_constraint_tensor_for_the_problem()

// Define optimization parameters.
n_trials = 100 // Number of optimization iterations
seed = 42      // Random seed for reproducibility

// ## 2. Initialize TFContinualSampler
// Next, initialize the TFContinualSampler. The crucial step is to pass
// the `tensor_constraint` to it. 

sampler = new TFContinualSampler(
    seed = seed,
    method = "train", // Tensor factorization method (e.g., 'cp', 'ring')
    tensor_constraint = tensor_constraint, // ★ Pass the constraint tensor here
    // ... other sampler-specific settings
)

// ## 3. Configure Optuna Study
study = optuna.create_study(
    sampler = sampler,
    direction = "minimize"
)

// ## 4. Run Optimization
study.optimize(objective_function, n_trials = n_trials)

// ## 5. Check Results
print("Best value found: ", study.best_value)
print("Best parameters: ", study.best_params)

```

## 4. Experiments

### 4.1 Hardware & Runtime Environment

All experiments in the paper were conducted under the following specific hardware and runtime conditions:

- CPU: Intel Xeon Gold 6230R (4 cores allocated per run)
- Memory: 8 GB
- Timeout: 3600 seconds (1 hour)
- Job Manager: Slurm 23.02.3

### 4.2 Running Experiments

The experiments reported in the paper can be started by running the shell scripts located in the root directory of the repository.

__Running the Main Experiments__
To reproduce the main results from the paper, run the following command:

```sh
bash run_main.sh
```

__Running the Ablation Studies__
To run the ablation studies that evaluate the importance of each component of our proposed method, execute:

```sh
bash run_ablation.sh
```

## 5. Citation

Citation information is pending. We will add the BibTeX entry here once the paper is publicly available.
