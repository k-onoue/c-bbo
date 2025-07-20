import argparse
import logging
import os
import random
from typing import Dict, List, Any

import numpy as np
import optuna
import torch
import torch.nn as nn
from optuna.samplers import BaseSampler
from ortools.linear_solver import pywraplp
# optunaの可視化機能を使うためにplotlyが必要
# import plotly

# ==========================================================================
# ユーティリティ関数
# ==========================================================================
def set_logger(log_filename_base: str, results_dir: str):
    log_filepath = os.path.join(results_dir, f"{log_filename_base}.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(log_filepath), logging.StreamHandler()]
    )

# ==========================================================================
# Ackley 問題定義
# ==========================================================================
class AckleyBenchmark:
    def __init__(self, constrain=False):
        self.bounds = [-32, 32]
        self.n_dim = 2
        self.fmin = 0
        self.fmax = 22.3497
        self._tensor_constraint = None
        if constrain:
            X, Y = np.meshgrid(
                np.arange(self.bounds[0], self.bounds[1] + 1),
                np.arange(self.bounds[0], self.bounds[1] + 1)
            )
            R_squared = X**2 + Y**2
            self._tensor_constraint = (R_squared < 10**2).astype(int)

    def evaluate(self, x: List[int]) -> float:
        # この評価関数は制約を考慮しない。制約はソルバーが担当するため。
        a, b, c = 20, 0.2, 2 * np.pi
        d = 2
        sum1 = -a * np.exp(-b * np.sqrt((x[0]**2 + x[1]**2) / d))
        sum2 = -np.exp((np.cos(c * x[0]) + np.cos(c * x[1])) / d)
        return sum1 + sum2 + a + np.exp(1)

# ==========================================================================
# NN+MILP Sampler
# ==========================================================================
class SimpleReLU_NN(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 16):
        super(SimpleReLU_NN, self).__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.output_layer = nn.Linear(hidden_dim, 1)
    def forward(self, x):
        return self.output_layer(self.relu(self.layer1(x)))

class NNMILPSampler(BaseSampler):
    def __init__(self, search_space_info: Dict, explicit_constraint_tensor: np.ndarray = None,
                 seed: int = None, sampler_settings: Dict = None):
        self._rng = np.random.RandomState(seed)
        torch.manual_seed(seed if seed is not None else random.randint(0, 2**32 - 1))
        if seed is not None: random.seed(seed)
        
        self._independent_sampler = optuna.samplers.RandomSampler(seed=seed)
        
        self.param_names = search_space_info['param_names']
        self.categories = search_space_info['categories']
        self.num_vars = len(self.param_names)
        
        self.input_dim = len(self.categories) * self.num_vars
        self.one_hot_lookup = self._create_one_hot_lookup()
        
        self.constraint_tensor = explicit_constraint_tensor

        settings = sampler_settings if sampler_settings is not None else {}
        self.n_startup_trials = settings.get("n_startup_trials", 10)
        self.hidden_dim = settings.get("hidden_dim", 16)
        self.epochs = settings.get("epochs", 300)

        self.surrogate_model = SimpleReLU_NN(self.input_dim, self.hidden_dim)
        self.optimizer = torch.optim.Adam(self.surrogate_model.parameters(), lr=0.01)
        self.loss_fn = nn.MSELoss()

    def _create_one_hot_lookup(self) -> List[List[int]]:
        lookup, start_idx = [], 0
        num_cats_per_var = len(self.categories)
        for _ in range(self.num_vars):
            lookup.append(list(range(start_idx, start_idx + num_cats_per_var)))
            start_idx += num_cats_per_var
        return lookup

    def _params_to_x_cat(self, params: Dict[str, Any]) -> List[int]:
        return [self.categories.index(params[name]) for name in self.param_names]
    
    def _x_cat_to_params(self, x_cat: List[int]) -> Dict[str, Any]:
        return {name: self.categories[x_cat[i]] for i, name in enumerate(self.param_names)}
        
    def _encode_one_hot(self, x_cat: List[int]) -> List[int]:
        one_hot = [0] * self.input_dim
        for i, val in enumerate(x_cat):
            one_hot[self.one_hot_lookup[i][val]] = 1
        return one_hot

    def _decode_one_hot(self, one_hot_solution: List[float]) -> List[int]:
        x_cat = []
        for indices in self.one_hot_lookup:
            found = False
            for i, idx in enumerate(indices):
                if one_hot_solution[idx] > 0.5:
                    x_cat.append(i); found = True; break
            if not found: x_cat.append(0)
        return x_cat

    def infer_relative_search_space(self, study: optuna.Study, trial: optuna.Trial) -> Dict:
        return optuna.search_space.intersection_search_space(study.get_trials(deepcopy=False))

    def sample_independent(self, study: optuna.Study, trial: optuna.Trial, param_name: str, param_distribution: optuna.distributions.BaseDistribution) -> Any:
        return self._independent_sampler.sample_independent(study, trial, param_name, param_distribution)

    def sample_relative(self, study: optuna.Study, trial: optuna.Trial, search_space: Dict) -> Dict:
        completed_trials = study.get_trials(deepcopy=False, states=[optuna.trial.TrialState.COMPLETE])
        if len(completed_trials) < self.n_startup_trials: return {}

        dataset = [{'x_cat': self._params_to_x_cat(t.params), 'y': t.value} for t in completed_trials]
        self._train_surrogate(dataset)
        past_x_cats = [d['x_cat'] for d in dataset]
        next_x_cat = self._build_and_solve_milp(past_x_cats)
        
        return self._x_cat_to_params(next_x_cat)
        
    def _train_surrogate(self, dataset: List[Dict]):
        logging.info("  Training surrogate model...")
        x_encoded = [self._encode_one_hot(d['x_cat']) for d in dataset]
        x_train = torch.tensor(x_encoded, dtype=torch.float32)
        y_train = torch.tensor([d['y'] for d in dataset], dtype=torch.float32).view(-1, 1)
        y_mean, y_std = y_train.mean(), y_train.std()
        y_train_norm = (y_train - y_mean) / (y_std + 1e-8)
        for _ in range(self.epochs):
            self.optimizer.zero_grad()
            predictions = self.surrogate_model(x_train)
            loss = self.loss_fn(predictions, y_train_norm)
            loss.backward()
            self.optimizer.step()
        logging.info(f"  Training complete. Final Loss: {loss.item():.4f}")

    def _build_and_solve_milp(self, past_x_cats: List[List[int]]) -> List[int]:
        logging.info("  Building and solving MILP problem...")
        solver = pywraplp.Solver.CreateSolver('SCIP')
        if not solver: return [self._rng.randint(0, len(self.categories)) for _ in range(self.num_vars)]
            
        x = [solver.BoolVar(f'x_{i}') for i in range(self.input_dim)]
        h = [solver.NumVar(0, solver.infinity(), f'h_{j}') for j in range(self.hidden_dim)]
        a = [solver.BoolVar(f'a_{j}') for j in range(self.hidden_dim)]
        output_var = solver.NumVar(-solver.infinity(), solver.infinity(), 'output')
        
        w1, b1 = self.surrogate_model.layer1.weight.data.numpy().T, self.surrogate_model.layer1.bias.data.numpy()
        w_out, b_out = self.surrogate_model.output_layer.weight.data.numpy().T, self.surrogate_model.output_layer.bias.data.numpy()
        M = 1000
        for j in range(self.hidden_dim):
            pre_act = solver.Sum([float(w1[i, j]) * x[i] for i in range(self.input_dim)]) + float(b1[j])
            solver.Add(h[j] >= pre_act)
            solver.Add(h[j] <= pre_act + M * (1 - a[j]))
            solver.Add(h[j] <= M * a[j])

        output_expr = solver.Sum([float(w_out[j, 0]) * h[j] for j in range(self.hidden_dim)]) + float(b_out[0])
        solver.Add(output_var == output_expr)
        
        for indices in self.one_hot_lookup: solver.Add(solver.Sum(x[idx] for idx in indices) == 1)
        for x_cat in past_x_cats:
            past_one_hot = self._encode_one_hot(x_cat)
            solver.Add(solver.Sum(x[i] for i, bit in enumerate(past_one_hot) if bit == 0) + \
                       solver.Sum(1 - x[i] for i, bit in enumerate(past_one_hot) if bit == 1) >= 1)
        
        if self.constraint_tensor is not None:
            logging.info(f"  Adding {np.sum(self.constraint_tensor == 0)} explicit constraints...")
            invalid_indices = np.argwhere(self.constraint_tensor == 0)
            for y_idx, x_idx in invalid_indices:
                x_var_idx = self.one_hot_lookup[0][x_idx]
                y_var_idx = self.one_hot_lookup[1][y_idx]
                solver.Add(x[x_var_idx] + x[y_var_idx] <= 1)
        
        solver.Maximize(output_var)
        status = solver.Solve()
        if status == pywraplp.Solver.OPTIMAL or status == pywraplp.Solver.FEASIBLE:
            return self._decode_one_hot([var.solution_value() for var in x])
        else:
            return [self._rng.randint(0, len(self.categories)) for _ in range(self.num_vars)]

# ==========================================================================
# 実験実行関数
# ==========================================================================
def run_bo(settings: Dict):
    random.seed(settings['seed'])
    np.random.seed(settings['seed'])
    torch.manual_seed(settings['seed'])

    ackley_objective = AckleyBenchmark(constrain=settings['constrain'])
    CATEGORIES = list(range(ackley_objective.bounds[0], ackley_objective.bounds[1] + 1))
    PARAM_NAMES = [f'x_{i}' for i in range(ackley_objective.n_dim)]

    def objective(trial: optuna.Trial) -> float:
        x = [trial.suggest_categorical(name, CATEGORIES) for name in PARAM_NAMES]
        return ackley_objective.evaluate(x)

    search_space_info = {'param_names': PARAM_NAMES, 'categories': CATEGORIES}
    
    sampler = NNMILPSampler(
        search_space_info=search_space_info,
        explicit_constraint_tensor=ackley_objective._tensor_constraint if settings['constrain'] else None,
        seed=settings['seed'],
        sampler_settings=settings['sampler_settings']
    )
    
    study = optuna.create_study(
        study_name=settings['name'],
        storage=settings['storage'],
        sampler=sampler,
        direction="minimize",
        load_if_exists=True
    )
    
    study.optimize(objective, n_trials=settings['n_trials'])
    
    logging.info(f"\n{'='*15} Optimization Complete {'='*15}")
    logging.info(f"Best score: {study.best_value:.4f}")
    logging.info(f"Best params: {study.best_params}")

    if settings.get("plot_save_dir"):
        try:
            fig = optuna.visualization.plot_optimization_history(study)
            plot_path = os.path.join(settings["plot_save_dir"], f"{settings['name']}_history.png")
            fig.write_image(plot_path)
            logging.info(f"Optimization history plot saved to {plot_path}")
        except Exception as e:
            logging.error(f"Failed to generate plot: {e}")

# ==========================================================================
# Main execution block
# ==========================================================================
def parse_args():
    parser = argparse.ArgumentParser(description="Run BO with NN+MILP Sampler for Ackley")
    parser.add_argument("--timestamp", type=str, required=True, help="Shared timestamp for the experiment run")
    parser.add_argument("--seed", type=int, required=True, help="Random seed")
    parser.add_argument("--n_trials", type=int, default=200, help="Number of BO trials")
    parser.add_argument("--base_dir", type=str, required=True, help="Base directory to save all results")
    parser.add_argument("--constrain", action="store_true", help="Use constrained version of Ackley")
    
    parser.add_argument("--epochs", type=int, default=300, help="Epochs for surrogate NN training")
    parser.add_argument("--hidden_dim", type=int, default=16, help="Hidden dimension of surrogate NN")
    parser.add_argument("--n_startup_trials", type=int, default=20, help="Number of initial random trials")
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    
    results_dir = os.path.join(args.base_dir, args.timestamp)
    plot_dir = os.path.join(results_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)
    
    constraint_str = "con" if args.constrain else "uncon"
    log_filename_base = f"ackley_{constraint_str}_nnmilp_epochs{args.epochs}_seed{args.seed}"
    exp_name = f"{args.timestamp}_{log_filename_base}"
    
    set_logger(log_filename_base, results_dir)

    storage_path = os.path.join(results_dir, f"{log_filename_base}.db")
    storage_url = f"sqlite:///{storage_path}"

    settings = {
        "name": exp_name,
        "seed": args.seed,
        "n_trials": args.n_trials,
        "constrain": args.constrain,
        "storage": storage_url,
        "plot_save_dir": plot_dir,
        "sampler_settings": {
            "epochs": args.epochs,
            "hidden_dim": args.hidden_dim,
            "n_startup_trials": args.n_startup_trials,
        }
    }
    
    logging.info(f"Starting experiment with settings: {settings}")
    run_bo(settings)
    logging.info("Experiment finished.")