import argparse
import logging
import os
import random
from typing import Dict, List, Any
from itertools import product

import numpy as np
import pandas as pd
import optuna
import torch
import torch.nn as nn
from optuna.samplers import BaseSampler
from ortools.linear_solver import pywraplp

from _src import set_logger# from sklearn.datasets import fetch_openml

# ==========================================================================
# データと制約の準備
# ==========================================================================
def prepare_diabetes_data_and_constraints():
    """Diabetesデータセットをロードし、制約情報を生成する"""
    data_path = 'data/diabetes.csv'
    tensors_path = 'data/diabetes_analysis_tensors.npz'

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"{data_path} not found. Please prepare the dataset.")
    if not os.path.exists(tensors_path):
         raise FileNotFoundError(f"{tensors_path} not found. Please prepare the pre-analyzed tensor file.")

    df = pd.read_csv(data_path)
    df_features = df.drop(columns='Outcome')
    features = df_features.columns.tolist()
    n_features = len(features)
    n_bins = 5

    binned_data = {c: pd.qcut(df_features[c], q=n_bins, labels=False, duplicates='drop') for c in features}
    binned_df = pd.DataFrame(binned_data)

    zero_entries_dict = {}
    for i in range(n_features):
        for j in range(i + 1, n_features):
            f1, f2 = features[i], features[j]
            existing_bins = set(map(tuple, binned_df[[f1, f2]].drop_duplicates().values))
            all_bins = set(product(range(n_bins), repeat=2))
            missing_bins = all_bins - existing_bins
            if missing_bins:
                zero_entries_dict[tuple(sorted((f1, f2)))] = list(missing_bins)
    
    return df, zero_entries_dict

# ==========================================================================
# Diabetes 目的関数クラス
# ==========================================================================
class DiabetesObjective:
    def __init__(self, df, start_point_index=None, seed=42):
        np.random.seed(seed)
        self.df = df
        self.df_features = self.df.drop(columns='Outcome')
        self.features = self.df_features.columns.tolist()
        
        loaded_tensors = np.load('data/diabetes_analysis_tensors.npz')
        self._tensor_predicted = loaded_tensors['predicted_tensor']
        
        if start_point_index is None:
            positive_samples = self.df[self.df['Outcome'] == 1]
            if positive_samples.empty: raise ValueError("No positive samples for start point.")
            start_row = positive_samples.sample(1, random_state=seed)
            
            # ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼ 修正箇所 ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼
            # qcutの結果がNaNになる場合に対処するロジックを追加
            binned_start = []
            for f in self.features:
                # 単一の値を持つSeriesに対してqcutを実行
                binned_value_series = pd.qcut(start_row[f], q=5, labels=False, duplicates='drop')
                binned_value = binned_value_series.iloc[0]
                
                # 結果がNaNかどうかを判定し、NaNならデフォルト値(0)を設定
                if pd.isna(binned_value):
                    binned_start.append(0)
                else:
                    binned_start.append(int(binned_value))
            
            self._x_start = np.array(binned_start, dtype=int)
            # ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲ 修正箇所 ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲

        else:
            self._x_start = np.array(start_point_index, dtype=int)
        
        logging.info(f"Start point x': {self._x_start}")
        logging.info(f"Predicted value at start point: {self._tensor_predicted[tuple(self._x_start)]:.4f}")

    def __call__(self, x: np.ndarray) -> float:
        f_x = self._tensor_predicted[tuple(x)]
        max_distance = np.linalg.norm(np.array([4] * len(x)))
        distance_penalty = np.linalg.norm(x - self._x_start) / max_distance
        # 最小化問題に変換するため、最大化したい値の符号を反転
        return -(f_x + distance_penalty)


# ==========================================================================
# NN+MILP Sampler
# ==========================================================================
class SimpleReLU_NN(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 16):
        super(SimpleReLU_NN, self).__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.output_layer = nn.Linear(hidden_dim, 1)
    def forward(self, x): return self.output_layer(self.relu(self.layer1(x)))

class NNMILPSampler(BaseSampler):
    def __init__(self, search_space_info: Dict, zero_entries_dict: Dict = None, seed: int = None, sampler_settings: Dict = None):
        self._rng = np.random.RandomState(seed)
        torch.manual_seed(seed if seed is not None else random.randint(0, 2**32 - 1))
        if seed is not None: random.seed(seed)
        
        self._independent_sampler = optuna.samplers.RandomSampler(seed=seed)
        
        self.param_names = search_space_info['param_names']
        self.categories_per_param = search_space_info['categories_per_param']
        self.num_vars = len(self.param_names)
        
        self.input_dim = sum(len(cats) for cats in self.categories_per_param.values())
        self.one_hot_lookup = self._create_one_hot_lookup()
        self.zero_entries_dict = zero_entries_dict if zero_entries_dict else {}

        settings = sampler_settings if sampler_settings is not None else {}
        self.n_startup_trials = settings.get("n_startup_trials", 10)
        self.hidden_dim = settings.get("hidden_dim", 16)
        self.epochs = settings.get("epochs", 300)

        self.surrogate_model = SimpleReLU_NN(self.input_dim, self.hidden_dim)
        self.optimizer = torch.optim.Adam(self.surrogate_model.parameters(), lr=0.01)
        self.loss_fn = nn.MSELoss()

    def _create_one_hot_lookup(self) -> Dict[str, List[int]]:
        lookup, start_idx = {}, 0
        for name in self.param_names:
            num_cats = len(self.categories_per_param[name])
            lookup[name] = list(range(start_idx, start_idx + num_cats))
            start_idx += num_cats
        return lookup

    def _params_to_x_cat_dict(self, params: Dict[str, Any]) -> Dict[str, int]:
        return {name: self.categories_per_param[name].index(val) for name, val in params.items()}
        
    def _encode_one_hot(self, x_cat_dict: Dict[str, int]) -> List[int]:
        one_hot = [0] * self.input_dim
        for name, cat_idx in x_cat_dict.items():
            one_hot[self.one_hot_lookup[name][cat_idx]] = 1
        return one_hot

    def _decode_one_hot(self, one_hot_solution: List[float]) -> Dict[str, int]:
        x_cat_dict = {}
        for name, indices in self.one_hot_lookup.items():
            found = False
            for i, idx in enumerate(indices):
                if one_hot_solution[idx] > 0.5: x_cat_dict[name] = i; found = True; break
            if not found: x_cat_dict[name] = 0
        return x_cat_dict
    
    def _x_cat_to_params(self, x_cat_dict: Dict[str, int]) -> Dict[str, Any]:
        return {name: self.categories_per_param[name][idx] for name, idx in x_cat_dict.items()}

    def infer_relative_search_space(self, study: optuna.Study, trial: optuna.Trial) -> Dict:
        return optuna.search_space.intersection_search_space(study.get_trials(deepcopy=False))

    def sample_independent(self, study: optuna.Study, trial: optuna.Trial, param_name: str, param_distribution: optuna.distributions.BaseDistribution) -> Any:
        return self._independent_sampler.sample_independent(study, trial, param_name, param_distribution)

    def sample_relative(self, study: optuna.Study, trial: optuna.Trial, search_space: Dict) -> Dict:
        completed_trials = study.get_trials(deepcopy=False, states=[optuna.trial.TrialState.COMPLETE])
        if len(completed_trials) < self.n_startup_trials: return {}

        dataset = [{'x_cat': self._params_to_x_cat_dict(t.params), 'y': t.value} for t in completed_trials]
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

    def _build_and_solve_milp(self, past_x_cats: List[Dict[str, int]]) -> Dict[str, int]:
        logging.info("  Building and solving MILP problem...")
        solver = pywraplp.Solver.CreateSolver('SCIP')
        if not solver: return {name: self._rng.randint(len(cats)) for name, cats in self.categories_per_param.items()}
            
        x_vars = [solver.BoolVar(f'x_{i}') for i in range(self.input_dim)]
        h_vars = [solver.NumVar(0, solver.infinity(), f'h_{j}') for j in range(self.hidden_dim)]
        a_vars = [solver.BoolVar(f'a_{j}') for j in range(self.hidden_dim)]
        output_var = solver.NumVar(-solver.infinity(), solver.infinity(), 'output')
        
        w1, b1 = self.surrogate_model.layer1.weight.data.numpy().T, self.surrogate_model.layer1.bias.data.numpy()
        w_out, b_out = self.surrogate_model.output_layer.weight.data.numpy().T, self.surrogate_model.output_layer.bias.data.numpy()
        M=1000
        for j in range(self.hidden_dim):
            pre_act = solver.Sum([float(w1[i, j]) * x_vars[i] for i in range(self.input_dim)]) + float(b1[j])
            solver.Add(h_vars[j] >= pre_act)
            solver.Add(h_vars[j] <= pre_act + M * (1-a_vars[j]))
            solver.Add(h_vars[j] <= M * a_vars[j])
        
        output_expr = solver.Sum([float(w_out[j, 0]) * h_vars[j] for j in range(self.hidden_dim)]) + float(b_out[0])
        solver.Add(output_var == output_expr)
        
        for name, indices in self.one_hot_lookup.items(): solver.Add(solver.Sum(x_vars[idx] for idx in indices) == 1)
        for x_cat_dict in past_x_cats:
            past_one_hot = self._encode_one_hot(x_cat_dict)
            solver.Add(solver.Sum(x_vars[i] for i, bit in enumerate(past_one_hot) if bit == 0) + \
                       solver.Sum(1 - x_vars[i] for i, bit in enumerate(past_one_hot) if bit == 1) >= 1)
        
        if self.zero_entries_dict:
            logging.info(f"  Adding {sum(len(v) for v in self.zero_entries_dict.values())} pairwise constraints...")
            for (f1_name, f2_name), bin_combinations in self.zero_entries_dict.items():
                for b1, b2 in bin_combinations:
                    var1_indices = self.one_hot_lookup[f1_name]
                    var2_indices = self.one_hot_lookup[f2_name]
                    solver.Add(x_vars[var1_indices[b1]] + x_vars[var2_indices[b2]] <= 1)

        solver.Maximize(output_var)
        status = solver.Solve()

        if status in [pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE]:
            return self._decode_one_hot([var.solution_value() for var in x_vars])
        else:
            return {name: self._rng.randint(len(cats)) for name, cats in self.categories_per_param.items()}

# ==========================================================================
# 実験実行関数
# ==========================================================================
def run_bo(settings: Dict, df: pd.DataFrame, zero_entries: Dict):
    random.seed(settings['seed'])
    np.random.seed(settings['seed'])
    torch.manual_seed(settings['seed'])

    features = df.drop(columns='Outcome').columns.tolist()
    diabetes_obj = DiabetesObjective(df=df, seed=settings['seed'])

    def objective_wrapper(trial: optuna.Trial) -> float:
        params = {name: trial.suggest_int(name, 0, 4) for name in features}
        x = np.array([params[name] for name in features])
        return diabetes_obj(x)

    search_space_info = {
        'param_names': features,
        'categories_per_param': {f: list(range(5)) for f in features}
    }
    
    sampler = NNMILPSampler(
        search_space_info=search_space_info,
        zero_entries_dict=zero_entries if settings['constrain'] else {},
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
    
    study.optimize(objective_wrapper, n_trials=settings['n_trials'])
    
    logging.info(f"\n{'='*15} Optimization Complete {'='*15}")
    logging.info(f"Best score (minimized objective): {study.best_value:.4f}")
    logging.info(f"Best params: {study.best_params}")

# ==========================================================================
# Main execution block
# ==========================================================================
def parse_args():
    parser = argparse.ArgumentParser(description="Run BO with NN+MILP Sampler for Diabetes")
    parser.add_argument("--timestamp", type=str, required=True, help="Shared timestamp")
    parser.add_argument("--seed", type=int, required=True, help="Random seed")
    parser.add_argument("--n_trials", type=int, default=200, help="Number of BO trials")
    parser.add_argument("--base_dir", type=str, required=True, help="Base directory to save all results")
    parser.add_argument("--constrain", action="store_true", help="Use pairwise zero-entry constraints")
    
    parser.add_argument("--epochs", type=int, default=300, help="Epochs for NN training")
    parser.add_argument("--hidden_dim", type=int, default=16, help="Hidden dimension of NN")
    parser.add_argument("--n_startup_trials", type=int, default=20, help="Number of initial random trials")
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    
    df, zero_entries = prepare_diabetes_data_and_constraints()
    
    results_dir = os.path.join(args.base_dir, args.timestamp)
    os.makedirs(results_dir, exist_ok=True)
    
    constraint_str = "con" if args.constrain else "uncon"
    log_filename_base = f"diabetes_{constraint_str}_nnmilp_epochs{args.epochs}_seed{args.seed}"
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
        "plot_save_dir": results_dir,
        "sampler_settings": {
            "epochs": args.epochs,
            "hidden_dim": args.hidden_dim,
            "n_startup_trials": args.n_startup_trials,
        }
    }
    
    logging.info(f"Starting experiment with settings: {settings}")
    run_bo(settings, df, zero_entries)
    logging.info("Experiment finished.")