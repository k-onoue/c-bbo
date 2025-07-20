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

def get_map(map_option: int) -> np.ndarray:
    if map_option == 1:
        return np.array([[1, 4], [2, 1]])
    elif map_option == 2:
        return np.array([[1, 4, 1], [2, 1, 1]])
    elif map_option == 3:
        return np.array([[1, 4, 1], [2, 1, 3], [5, 2, 1]])
    else:
        raise ValueError(f"Invalid map option: {map_option}")

# ==========================================================================
# Warcraft 問題定義 (詳細バージョン)
# ==========================================================================
def get_opposite(direction: str) -> str:
    pair_dict = {"a": "c", "c": "a", "b": "d", "d": "b"}
    return pair_dict.get(direction, "")

def judge_continuity(d_from: str, current_direction: str) -> bool:
    d_opposite = get_opposite(d_from)
    return d_opposite in current_direction

def get_next_coordinate(d_to: str, current_coordinate: tuple[int, int]) -> tuple[int, int]:
    update_dict = {"a": (-1, 0), "b": (0, -1), "c": (0, 1), "d": (1, 0)}
    delta = update_dict.get(d_to, (0, 0))
    return (current_coordinate[0] + delta[0], current_coordinate[1] + delta[1])

def judge_location_validity(current: tuple[int, int], shape: tuple[int, int]) -> bool:
    return 0 <= current[0] < shape[0] and 0 <= current[1] < shape[1]

def get_d_to(d_from: str, current_direction: str) -> str:
    return current_direction[1] if current_direction[0] == d_from else current_direction[0]

def navigate_through_matrix(direction_matrix, start, goal):
    history = []
    current = start
    shape = direction_matrix.shape
    current_direction = direction_matrix[current]
    if "a" in current_direction or "b" in current_direction:
        d_to = get_d_to("a", current_direction) if "a" in current_direction else get_d_to("b", current_direction)
    else:
        return history
    if current_direction == "ab":
        history.append(current)
        return history
    history.append(current)
    next_pos = get_next_coordinate(d_to, current)
    while judge_location_validity(next_pos, shape) and current != goal:
        if direction_matrix[next_pos] == "oo": break
        if not judge_continuity(d_to, direction_matrix[next_pos]): break
        current = next_pos
        history.append(current)
        if current == goal: break
        direction = direction_matrix[current]
        d_from = get_opposite(d_to)
        d_to = get_d_to(d_from, direction)
        next_pos = get_next_coordinate(d_to, current)
    return history

def manhattan_distance(coord1: tuple[int, int], coord2: tuple[int, int]) -> int:
    return abs(coord1[0] - coord2[0]) + abs(coord1[1] - coord2[1])

class WarcraftObjectiveTF:
    def __init__(self, weight_matrix: np.ndarray):
        self.weight_matrix = weight_matrix / np.sum(weight_matrix)
        self.shape = weight_matrix.shape
        self._val_mask_dict = {
            "oo": np.zeros((3, 3)), "ab": np.array([[0, 1, 0], [1, 1, 0], [0, 0, 0]]),
            "ac": np.array([[0, 0, 0], [1, 1, 1], [0, 0, 0]]), "ad": np.array([[0, 0, 0], [1, 1, 0], [0, 1, 0]]),
            "bc": np.array([[0, 1, 0], [0, 1, 1], [0, 0, 0]]), "bd": np.array([[0, 1, 0], [0, 1, 0], [0, 1, 0]]),
            "cd": np.array([[0, 0, 0], [0, 1, 1], [0, 1, 0]]),
        }

    def calculate_penalty_type2(self, idx, val, map_shape):
        arr_expanded = np.zeros((map_shape[0] * 2 + 1, map_shape[1] * 2 + 1))
        x_s, y_s = idx[0] * 2, idx[1] * 2
        arr_expanded[x_s : x_s + 3, y_s : y_s + 3] = self._val_mask_dict.get(val, np.zeros((3, 3)))
        ones_indices = np.argwhere(arr_expanded == 1)
        row, col = arr_expanded.shape[0] - 1, arr_expanded.shape[1] - 1
        max_distance = manhattan_distance((0, 0), (row, col - 1))
        min_distance = max_distance
        index_goal_list = [(row, col - 1), (row - 1, col)]
        for one_idx in ones_indices:
            for target_idx in index_goal_list:
                dist = manhattan_distance(one_idx, target_idx)
                if dist < min_distance: min_distance = dist
        return min_distance / max_distance

    def __call__(self, direction_matrix: np.ndarray) -> float:
        mask = np.where(direction_matrix == "oo", 0, 1)
        penalty_1 = np.sum(self.weight_matrix * mask)
        start, goal = (0, 0), (self.shape[0] - 1, self.shape[1] - 1)
        history = navigate_through_matrix(direction_matrix, start, goal)
        penalty_3 = self.calculate_penalty_type2(history[-1], direction_matrix[history[-1]], self.shape) if history else 1
        return penalty_1 + penalty_3
        
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
    def __init__(self, search_space_info: Dict, constraints: Dict, seed: int = None, sampler_settings: Dict = None):
        self._rng = np.random.RandomState(seed)
        torch.manual_seed(seed if seed is not None else random.randint(0, 2**32 - 1))
        if seed is not None: random.seed(seed)
        
        self._independent_sampler = optuna.samplers.RandomSampler(seed=seed)
        
        self.search_space_info = search_space_info
        self.constraints = constraints # 空辞書か制約情報を持つ辞書
        self.map_shape = self.search_space_info['map_shape']
        self.directions = self.search_space_info['directions']
        self.num_vars = self.map_shape[0] * self.map_shape[1]
        self.input_dim = len(self.directions) * self.num_vars
        self.one_hot_lookup = self._create_one_hot_lookup()

        settings = sampler_settings if sampler_settings is not None else {}
        self.n_startup_trials = settings.get("n_startup_trials", 10)
        self.hidden_dim = settings.get("hidden_dim", 16)
        self.epochs = settings.get("epochs", 300)
        self.time_limit_sec = settings.get("time_limit_sec", None)

        self.surrogate_model = SimpleReLU_NN(self.input_dim, self.hidden_dim)
        self.optimizer = torch.optim.Adam(self.surrogate_model.parameters(), lr=0.01)
        self.loss_fn = nn.MSELoss()

    def _create_one_hot_lookup(self) -> List[List[int]]:
        lookup, start_idx = [], 0
        num_cats_per_var = len(self.directions)
        for _ in range(self.num_vars):
            lookup.append(list(range(start_idx, start_idx + num_cats_per_var)))
            start_idx += num_cats_per_var
        return lookup

    def _params_to_x_cat(self, params: Dict[str, Any]) -> List[int]:
        x_cat = [0] * self.num_vars
        for r in range(self.map_shape[0]):
            for c in range(self.map_shape[1]):
                param_name = f"x_{r}_{c}"
                direction_str = params[param_name]
                cat_index = self.directions.index(direction_str)
                flat_index = r * self.map_shape[1] + c
                x_cat[flat_index] = cat_index
        return x_cat
    
    def _x_cat_to_params(self, x_cat: List[int]) -> Dict[str, Any]:
        params = {}
        for r in range(self.map_shape[0]):
            for c in range(self.map_shape[1]):
                param_name = f"x_{r}_{c}"
                flat_index = r * self.map_shape[1] + c
                cat_index = x_cat[flat_index]
                params[param_name] = self.directions[cat_index]
        return params
        
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
                if one_hot_solution[idx] > 0.5: x_cat.append(i); found = True; break
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
        x_encoded = [self._encode_one_hot(d['x_cat']) for d in dataset]
        x_train = torch.tensor(x_encoded, dtype=torch.float32)
        y_train = torch.tensor([d['y'] for d in dataset], dtype=torch.float32).view(-1, 1)
        y_mean, y_std = y_train.mean(), y_train.std()
        y_train_norm = (y_train - y_mean) / (y_std + 1e-8)
        for _ in range(self.epochs):
            self.optimizer.zero_grad(); predictions = self.surrogate_model(x_train)
            loss = self.loss_fn(predictions, y_train_norm); loss.backward(); self.optimizer.step()

    def _build_and_solve_milp(self, past_x_cats: List[List[int]]) -> List[int]:
        solver = pywraplp.Solver.CreateSolver('SCIP')
        if not solver: return [self._rng.randint(len(self.directions)) for _ in range(self.num_vars)]
        if self.time_limit_sec: solver.SetTimeLimit(self.time_limit_sec * 1000)
            
        x,h,a,output = ([solver.BoolVar(f'x_{i}') for i in range(self.input_dim)],
                        [solver.NumVar(0,solver.infinity(),f'h_{j}') for j in range(self.hidden_dim)],
                        [solver.BoolVar(f'a_{j}') for j in range(self.hidden_dim)],
                        solver.NumVar(-solver.infinity(),solver.infinity(),'output'))
        w1,b1,w_out,b_out=(d.data.numpy() for d in [self.surrogate_model.layer1.weight,self.surrogate_model.layer1.bias,self.surrogate_model.output_layer.weight,self.surrogate_model.output_layer.bias])
        w1=w1.T; M=1000
        for j in range(self.hidden_dim):
            pre_act = solver.Sum([float(w1[i,j])*x[i] for i in range(self.input_dim)])+float(b1[j])
            solver.Add(h[j]>=pre_act); solver.Add(h[j]<=pre_act+M*(1-a[j])); solver.Add(h[j]<=M*a[j])
        solver.Add(output==solver.Sum([float(w_out[0,j])*h[j] for j in range(self.hidden_dim)])+float(b_out[0]))
        for indices in self.one_hot_lookup: solver.Add(solver.Sum(x[idx] for idx in indices)==1)
        for x_cat in past_x_cats:
            past_one_hot = self._encode_one_hot(x_cat)
            solver.Add(solver.Sum(x[i] for i,b in enumerate(past_one_hot) if b==0)+solver.Sum(1-x[i] for i,b in enumerate(past_one_hot) if b==1)>=1)
        
        if self.constraints:
            logging.info("  Adding Warcraft-specific constraints...")
            start_indices = self.one_hot_lookup[0]
            for direction in self.constraints['start_forbidden']:
                solver.Add(x[start_indices[self.directions.index(direction)]] == 0)
            goal_indices = self.one_hot_lookup[-1]
            for direction in self.constraints['goal_forbidden']:
                solver.Add(x[goal_indices[self.directions.index(direction)]] == 0)
            gain_expr = solver.Sum(self.constraints['gains'][d_idx]*x[self.one_hot_lookup[c_idx][d_idx]] for c_idx in range(self.num_vars) for d_idx in range(len(self.directions)))
            solver.Add(gain_expr == self.constraints['ideal_gain'])

        solver.Maximize(output)
        status = solver.Solve()
        if status in [pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE]:
            return self._decode_one_hot([var.solution_value() for var in x])
        else:
            return [self._rng.randint(len(self.directions)) for _ in range(self.num_vars)]

# ==========================================================================
# 実験実行関数
# ==========================================================================
def run_bo(settings: Dict):
    random.seed(settings['seed']); np.random.seed(settings['seed']); torch.manual_seed(settings['seed'])

    DIRECTIONS = ["oo", "ab", "ac", "ad", "bc", "bd", "cd"]
    weight_matrix = get_map(settings['map_option'])
    # ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼ 修正箇所 ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼
    warcraft_objective = WarcraftObjectiveTF(weight_matrix=weight_matrix)
    # ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲ 修正箇所 ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲
    map_shape = weight_matrix.shape

    def objective(trial: optuna.Trial) -> float:
        params = {f"x_{r}_{c}": trial.suggest_categorical(f"x_{r}_{c}", DIRECTIONS) for r in range(map_shape[0]) for c in range(map_shape[1])}
        direction_matrix = np.array(list(params.values())).reshape(map_shape)
        return warcraft_objective(direction_matrix)

    search_space_info = {'map_shape': map_shape, 'directions': DIRECTIONS}
    warcraft_constraints = {"start_forbidden":["oo","ab"],"goal_forbidden":["oo","cd"],"gains":[0,2,2,2,2,2,2],"ideal_gain":(map_shape[0]+map_shape[1]-1)*2}

    sampler = NNMILPSampler(
        search_space_info=search_space_info,
        constraints=warcraft_constraints if settings['constrain'] else {}, # flagに応じて切り替え
        seed=settings['seed'],
        sampler_settings=settings['sampler_settings']
    )
    
    study = optuna.create_study(study_name=settings['name'], storage=settings['storage'], sampler=sampler, direction="minimize", load_if_exists=True)
    study.optimize(objective, n_trials=settings['n_trials'])
    
    logging.info(f"\n{'='*15} Optimization Complete {'='*15}")
    logging.info(f"Best score: {study.best_value:.4f}, Best params: {study.best_params}")

# ==========================================================================
# Main execution block
# ==========================================================================
def parse_args():
    parser = argparse.ArgumentParser(description="Run BO with NN+MILP Sampler for Warcraft")
    parser.add_argument("--timestamp", type=str, required=True, help="Shared timestamp")
    parser.add_argument("--seed", type=int, required=True, help="Random seed")
    parser.add_argument("--n_trials", type=int, default=100, help="Number of BO trials")
    parser.add_argument("--map_option", type=int, default=1, choices=[1, 2, 3], help="Warcraft map option")
    parser.add_argument("--base_dir", type=str, required=True, help="Base directory to save results")
    parser.add_argument("--constrain", action="store_true", help="Use Warcraft constraints")
    parser.add_argument("--epochs", type=int, default=300, help="Epochs for NN training")
    parser.add_argument("--hidden_dim", type=int, default=16, help="Hidden dimension of NN")
    parser.add_argument("--n_startup_trials", type=int, default=50, help="Number of initial random trials")
    parser.add_argument("--time_limit_sec", type=int, default=None, help="Time limit for MILP solver in seconds")
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    results_dir = os.path.join(args.base_dir, args.timestamp)
    os.makedirs(results_dir, exist_ok=True)
    
    constraint_str = "con" if args.constrain else "uncon"
    log_filename_base = f"warcraft_map{args.map_option}_nnmilp_epochs{args.epochs}_{constraint_str}_seed{args.seed}"
    exp_name = f"{args.timestamp}_{log_filename_base}"
    set_logger(log_filename_base, results_dir)
    storage_url = f"sqlite:///{os.path.join(results_dir, f'{log_filename_base}.db')}"

    settings = {
        "name": exp_name, "seed": args.seed, "n_trials": args.n_trials,
        "map_option": args.map_option, "constrain": args.constrain, "storage": storage_url,
        "sampler_settings": {
            "epochs": args.epochs, "hidden_dim": args.hidden_dim,
            "n_startup_trials": args.n_startup_trials, "time_limit_sec": args.time_limit_sec,
        }
    }
    logging.info(f"Starting experiment with settings: {settings}")
    run_bo(settings)
    logging.info("Experiment finished.")