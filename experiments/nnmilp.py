
# import argparse
# import logging
# import os
# import random
# from itertools import product
# from typing import Any, Dict, List, Tuple

# import numpy as np
# import optuna
# import pandas as pd
# import torch
# import torch.nn as nn
# from optuna.samplers import BaseSampler
# from ortools.linear_solver import pywraplp

# # Optional: For saving plots. Requires 'plotly' and 'kaleido'.
# # try:
# #     import plotly
# # except ImportError:
# #     pass

# # ==========================================================================
# # ユーティリティ関数 (Common Utility Functions)
# # ==========================================================================

# def set_logger(log_filename_base: str, results_dir: str):
#     """Initializes the logger to write to both file and console."""
#     log_filepath = os.path.join(results_dir, f"{log_filename_base}.log")
#     logging.basicConfig(
#         level=logging.INFO,
#         format="%(asctime)s [%(levelname)s] %(message)s",
#         handlers=[logging.FileHandler(log_filepath), logging.StreamHandler()]
#     )

# # ==========================================================================
# # 問題定義: Ackley (Problem Definition: Ackley)
# # ==========================================================================

# class AckleyBenchmark:
#     """Defines the Ackley benchmark function."""
#     def __init__(self, constrain: bool = False):
#         self.bounds = [-32, 32]
#         self.n_dim = 2
#         self._tensor_constraint = None
#         # if constrain:

#         X, Y = np.meshgrid(
#             np.arange(self.bounds[0], self.bounds[1] + 1),
#             np.arange(self.bounds[0], self.bounds[1] + 1)
#         )
#         R_squared = X**2 + Y**2
#         self._tensor_constraint = (R_squared < 10**2).astype(int)

#     def evaluate(self, x: List[int]) -> float:

#         # check if x satisfies the constraint


#         a, b, c = 20, 0.2, 2 * np.pi
#         d = 2
#         term1 = -a * np.exp(-b * np.sqrt((x[0]**2 + x[1]**2) / d))
#         term2 = -np.exp((np.cos(c * x[0]) + np.cos(c * x[1])) / d)
#         return term1 + term2 + a + np.exp(1)

# # ==========================================================================
# # 問題定義: Diabetes (Problem Definition: Diabetes)
# # ==========================================================================

# def prepare_diabetes_data_and_constraints() -> Tuple[pd.DataFrame, Dict]:
#     """Loads Diabetes dataset and generates pairwise feature constraints."""
#     data_path = 'data/diabetes.csv'
#     if not os.path.exists(data_path):
#         raise FileNotFoundError(f"{data_path} not found. Please prepare the dataset.")

#     df = pd.read_csv(data_path)
#     df_features = df.drop(columns='Outcome')
#     features = df_features.columns.tolist()
#     n_features = len(features)
#     n_bins = 5

#     binned_df = pd.DataFrame({
#         c: pd.qcut(df_features[c], q=n_bins, labels=False, duplicates='drop') for c in features
#     })

#     zero_entries_dict = {}
#     for i in range(n_features):
#         for j in range(i + 1, n_features):
#             f1, f2 = features[i], features[j]
#             existing_bins = set(map(tuple, binned_df[[f1, f2]].drop_duplicates().values))
#             all_bins = set(product(range(n_bins), repeat=2))
#             missing_bins = all_bins - existing_bins
#             if missing_bins:
#                 zero_entries_dict[tuple(sorted((f1, f2)))] = list(missing_bins)
    
#     return df, zero_entries_dict

# class DiabetesObjective:
#     """Defines the objective function for the Diabetes counterfactual task."""
#     def __init__(self, df: pd.DataFrame, seed: int = 42):
#         np.random.seed(seed)
#         self.df = df
#         self.df_features = self.df.drop(columns='Outcome')
#         self.features = self.df_features.columns.tolist()
        
#         tensors_path = 'data/diabetes_analysis_tensors.npz'
#         if not os.path.exists(tensors_path):
#             raise FileNotFoundError(f"{tensors_path} not found. Please prepare the pre-analyzed tensor file.")
        
#         loaded_tensors = np.load(tensors_path)
#         self._tensor_predicted = loaded_tensors['predicted_tensor']
        
#         positive_samples = self.df[self.df['Outcome'] == 1]
#         if positive_samples.empty:
#             raise ValueError("No positive samples (Outcome=1) to select a start point.")
#         start_row = positive_samples.sample(1, random_state=seed)
        
#         binned_start = []
#         for f in self.features:
#             binned_value_series = pd.qcut(start_row[f], q=5, labels=False, duplicates='drop')
#             binned_value = binned_value_series.iloc[0]
#             binned_start.append(0 if pd.isna(binned_value) else int(binned_value))
        
#         self._x_start = np.array(binned_start, dtype=int)
        
#         logging.info(f"Start point x': {self._x_start}")
#         logging.info(f"Predicted value at start point: {self._tensor_predicted[tuple(self._x_start)]:.4f}")

#     def __call__(self, x: np.ndarray) -> float:
#         """Calculates the objective value for the counterfactual task."""
#         f_x = self._tensor_predicted[tuple(x)]
#         max_distance = np.linalg.norm(np.array([4] * len(x)))
#         distance_penalty = np.linalg.norm(x - self._x_start) / max_distance
#         return f_x + distance_penalty

# # ==========================================================================
# # 問題定義: Pressure Vessel (Problem Definition: Pressure Vessel)
# # ==========================================================================

# class PressureVesselObjective:
#     """Pressure Vessel design optimization problem."""
#     def __init__(self, start_point=None, seed=42, n_bins=10):
#         self.seed = seed
#         np.random.seed(seed)
        
#         self.features = ['Ts', 'Th', 'R', 'L']

#         # Define variable ranges
#         ts_min, ts_max = 0.0625, 6.1875
#         th_min, th_max = 0.0625, 6.1875
#         r_min, r_max = 10.0, 200.0
#         l_min, l_max = 10.0, 200.0
        
#         # Discretize each variable and get representative midpoints
#         self.mid_points = []
#         for min_val, max_val in [(ts_min, ts_max), (th_min, th_max), (r_min, r_max), (l_min, l_max)]:
#             edges = np.linspace(min_val, max_val, n_bins + 1)
#             mids = np.array([(edges[i] + edges[i+1]) / 2 for i in range(n_bins)])
#             self.mid_points.append(mids)
        
#         # Pre-compute objective values and constraint satisfaction for all combinations
#         shape = tuple(len(m) for m in self.mid_points)
#         self._tensor_objective = np.empty(shape)
#         self._tensor_constraint = np.empty(shape, dtype=bool)

#         for i, ts in enumerate(self.mid_points[0]):
#             for j, th in enumerate(self.mid_points[1]):
#                 for k, r in enumerate(self.mid_points[2]):
#                     for l, l_val in enumerate(self.mid_points[3]):
#                         x = np.array([ts, th, r, l_val])
#                         self._tensor_objective[i, j, k, l] = self._objective_formula(x)
#                         g_vals = self._constraints_formula(x)
#                         self._tensor_constraint[i, j, k, l] = np.all(g_vals <= 0)

#         self._feasible_indices = np.argwhere(self._tensor_constraint == True)
#         if len(self._feasible_indices) == 0:
#             raise RuntimeError("No feasible points found in the discretized search space. Try increasing n_bins.")
            
#         logging.info(f"Problem space discretized into {shape} grid. Found {len(self._feasible_indices)} feasible points.")

#     def _objective_formula(self, x: np.ndarray) -> float:
#         """Cost function formula."""
#         ts, th, r, l_val = x[0], x[1], x[2], x[3]
#         return (0.6224 * ts * r * l_val + 1.7781 * th * r**2 + 
#                 3.1661 * ts**2 * l_val + 19.84 * ts**2 * r)

#     def _constraints_formula(self, x: np.ndarray) -> np.ndarray:
#         """Constraint functions g(x) <= 0."""
#         ts, th, r, l_val = x[0], x[1], x[2], x[3]
#         g1 = -ts + 0.0193 * r
#         g2 = -th + 0.00954 * r
#         g3 = -np.pi * r**2 * l_val - (4.0/3.0) * np.pi * r**3 + 1_296_000
#         g4 = l_val - 240.0
#         return np.array([g1, g2, g3, g4])

#     def __call__(self, x: np.ndarray, is_constrained: bool) -> float:
#         """Returns the objective value for a point given by its indices."""
#         x_tuple = tuple(x)
#         # If constraints are active and the point is infeasible, return a large penalty
#         if not self._tensor_constraint[x_tuple]:
#             # The penalty is the max value from the feasible region, ensuring it's always worse.
#             return np.max(self._tensor_objective[self._tensor_constraint]) 
        
#         return self._tensor_objective[x_tuple]

# # ==========================================================================
# # 問題定義: Warcraft (Problem Definition: Warcraft)
# # ==========================================================================

# def get_map(map_option: int) -> np.ndarray:
#     """Returns a predefined map layout based on the option."""
#     if map_option == 1:
#         return np.array([[1, 4], [2, 1]])
#     if map_option == 2:
#         return np.array([[1, 4, 1], [2, 1, 1]])
#     if map_option == 3:
#         return np.array([[1, 4, 1], [2, 1, 3], [5, 2, 1]])
#     raise ValueError(f"Invalid map option: {map_option}")

# def navigate_through_matrix(direction_matrix, start, goal):
#     """Helper function to trace a path through the Warcraft map."""
#     def get_opposite(d): return {"a":"c","c":"a","b":"d","d":"b"}.get(d,"")
#     def judge_continuity(d_from, current_dir): return get_opposite(d_from) in current_dir
#     def get_d_to(d_from, current_dir): return current_dir[1] if current_dir[0] == d_from else current_dir[0]
#     def get_next_coord(d_to, current):
#         delta = {"a":(-1,0),"b":(0,-1),"c":(0,1),"d":(1,0)}.get(d_to,(0,0))
#         return (current[0]+delta[0], current[1]+delta[1])
#     def is_valid(coord, shape): return 0<=coord[0]<shape[0] and 0<=coord[1]<shape[1]

#     history, current = [], start
#     shape = direction_matrix.shape
#     current_direction = direction_matrix[current]
#     if "a" in current_direction or "b" in current_direction:
#         d_to = get_d_to("a", current_direction) if "a" in current_direction else get_d_to("b", current_direction)
#     else: return history
#     if current_direction == "ab": history.append(current); return history
    
#     history.append(current)
#     next_pos = get_next_coord(d_to, current)
#     while is_valid(next_pos, shape) and current != goal:
#         if direction_matrix[next_pos] == "oo": break
#         if not judge_continuity(d_to, direction_matrix[next_pos]): break
#         current = next_pos
#         history.append(current)
#         if current == goal: break
#         d_from = get_opposite(d_to)
#         d_to = get_d_to(d_from, direction_matrix[current])
#         next_pos = get_next_coord(d_to, current)
#     return history

# class WarcraftObjective:
#     """Defines the objective function for the Warcraft pathfinding problem."""
#     def __init__(self, weight_matrix: np.ndarray):
#         self.weight_matrix = weight_matrix / np.sum(weight_matrix)
#         self.shape = weight_matrix.shape
#         self._val_mask_dict = {
#             "oo":np.zeros((3,3)), "ab":np.array([[0,1,0],[1,1,0],[0,0,0]]), "ac":np.array([[0,0,0],[1,1,1],[0,0,0]]),
#             "ad":np.array([[0,0,0],[1,1,0],[0,1,0]]), "bc":np.array([[0,1,0],[0,1,1],[0,0,0]]),
#             "bd":np.array([[0,1,0],[0,1,0],[0,1,0]]), "cd":np.array([[0,0,0],[0,1,1],[0,1,0]]),
#         }

#     def _calculate_penalty_type2(self, idx, val, map_shape):
#         def manhattan_dist(c1, c2): return abs(c1[0]-c2[0]) + abs(c1[1]-c2[1])
#         arr_exp = np.zeros((map_shape[0]*2+1, map_shape[1]*2+1))
#         x_s, y_s = idx[0]*2, idx[1]*2
#         arr_exp[x_s:x_s+3, y_s:y_s+3] = self._val_mask_dict.get(val, np.zeros((3,3)))
#         ones = np.argwhere(arr_exp == 1)
#         row, col = arr_exp.shape[0]-1, arr_exp.shape[1]-1
#         max_dist = manhattan_dist((0,0), (row,col-1))
#         min_dist = max_dist
#         goals = [(row, col-1), (row-1, col)]
#         for one_idx in ones:
#             for target_idx in goals:
#                 dist = manhattan_dist(one_idx, target_idx)
#                 if dist < min_dist: min_dist = dist
#         return min_dist / max_dist if max_dist > 0 else 1

#     def __call__(self, direction_matrix: np.ndarray) -> float:
#         mask = np.where(direction_matrix == "oo", 0, 1)
#         penalty_1 = np.sum(self.weight_matrix * mask)
#         start, goal = (0, 0), (self.shape[0] - 1, self.shape[1] - 1)
#         history = navigate_through_matrix(direction_matrix, start, goal)
#         penalty_3 = self._calculate_penalty_type2(history[-1], direction_matrix[history[-1]], self.shape) if history else 1
#         return penalty_1 + penalty_3

# # ==========================================================================
# # 代理モデル (Surrogate Model)
# # ==========================================================================
# class SimpleReLU_NN(nn.Module):
#     """A simple fully-connected neural network with one hidden ReLU layer."""
#     def __init__(self, input_dim: int, hidden_dim: int):
#         super(SimpleReLU_NN, self).__init__()
#         self.layer1 = nn.Linear(input_dim, hidden_dim)
#         self.relu = nn.ReLU()
#         self.output_layer = nn.Linear(hidden_dim, 1)

#     def forward(self, x):
#         return self.output_layer(self.relu(self.layer1(x)))

# # ==========================================================================
# # 統合 NN+MILP Sampler (Unified NN+MILP Sampler)
# # ==========================================================================
# class NNMILPSampler(BaseSampler):
#     """A unified Optuna sampler using a Neural Network surrogate and a MILP solver."""
#     def __init__(self, search_space_info: Dict, problem_name: str, constraints: Dict,
#                  seed: int = None, sampler_settings: Dict = None):
#         self._rng = np.random.RandomState(seed)
#         torch.manual_seed(seed if seed is not None else random.randint(0, 2**32 - 1))
#         if seed is not None: random.seed(seed)
#         self._independent_sampler = optuna.samplers.RandomSampler(seed=seed)
#         self.problem_name = problem_name
#         self.param_names = search_space_info['param_names']
#         self.categories_per_param = search_space_info['categories_per_param']
#         self.constraints = constraints if constraints else {}
#         self.num_vars = len(self.param_names)
#         self.input_dim = sum(len(cats) for cats in self.categories_per_param.values())
#         self.one_hot_lookup = self._create_one_hot_lookup()
#         settings = sampler_settings if sampler_settings is not None else {}
#         self.n_startup_trials = settings.get("n_startup_trials", 20)
#         self.hidden_dim = settings.get("hidden_dim", 16)
#         self.epochs = settings.get("epochs", 300)
#         self.time_limit_sec = settings.get("time_limit_sec", None)
#         self.surrogate_model = SimpleReLU_NN(self.input_dim, self.hidden_dim)
#         self.optimizer = torch.optim.Adam(self.surrogate_model.parameters(), lr=0.01)
#         self.loss_fn = nn.MSELoss()

#     def _create_one_hot_lookup(self) -> Dict[str, List[int]]:
#         lookup, start_idx = {}, 0
#         for name in self.param_names:
#             num_cats = len(self.categories_per_param[name])
#             lookup[name] = list(range(start_idx, start_idx + num_cats))
#             start_idx += num_cats
#         return lookup

#     def _params_to_x_cat_dict(self, params: Dict[str, Any]) -> Dict[str, int]:
#         return {name: self.categories_per_param[name].index(val) for name, val in params.items()}

#     def _encode_one_hot(self, x_cat_dict: Dict[str, int]) -> List[int]:
#         one_hot = [0] * self.input_dim
#         for name, cat_idx in x_cat_dict.items():
#             one_hot[self.one_hot_lookup[name][cat_idx]] = 1
#         return one_hot

#     def _decode_one_hot(self, one_hot_solution: List[float]) -> Dict[str, int]:
#         x_cat_dict = {}
#         for name, indices in self.one_hot_lookup.items():
#             found = False
#             for i, idx in enumerate(indices):
#                 if one_hot_solution[idx] > 0.5:
#                     x_cat_dict[name] = i
#                     found = True
#                     break
#             if not found: x_cat_dict[name] = 0
#         return x_cat_dict
    
#     def _x_cat_to_params(self, x_cat_dict: Dict[str, int]) -> Dict[str, Any]:
#         return {name: self.categories_per_param[name][idx] for name, idx in x_cat_dict.items()}

#     def infer_relative_search_space(self, study: optuna.Study, trial: optuna.Trial) -> Dict:
#         return optuna.search_space.intersection_search_space(study.get_trials(deepcopy=False))

#     def sample_independent(self, study: optuna.Study, trial: optuna.Trial, param_name: str,
#                            param_distribution: optuna.distributions.BaseDistribution) -> Any:
#         return self._independent_sampler.sample_independent(study, trial, param_name, param_distribution)

#     def sample_relative(self, study: optuna.Study, trial: optuna.Trial, search_space: Dict) -> Dict:
#         completed_trials = study.get_trials(deepcopy=False, states=[optuna.trial.TrialState.COMPLETE])
#         if len(completed_trials) < self.n_startup_trials: return {}
#         dataset = [{'x_cat': self._params_to_x_cat_dict(t.params), 'y': t.value} for t in completed_trials]
#         self._train_surrogate(dataset)
#         past_x_cats = [d['x_cat'] for d in dataset]
#         next_x_cat = self._build_and_solve_milp(past_x_cats)
#         return self._x_cat_to_params(next_x_cat)

#     def _train_surrogate(self, dataset: List[Dict]):
#         logging.info("  Training surrogate model...")
#         x_encoded = [self._encode_one_hot(d['x_cat']) for d in dataset]
#         x_train = torch.tensor(x_encoded, dtype=torch.float32)
#         y_train = torch.tensor([d['y'] for d in dataset], dtype=torch.float32).view(-1, 1)
#         y_mean, y_std = y_train.mean(), y_train.std()
#         # y_train_norm = (y_train - y_mean) / (y_std + 1e-8)
#         # ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼ 修正箇所 ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼
#         # y_stdが0またはnanの場合のハンドリング
#         # y_trainの要素数が1の場合、y_stdはnanになる
#         if len(y_train) <= 1 or torch.isnan(y_std) or y_std < 1e-8:
#             # 目的値にばらつきがない場合、正規化は不要。0で埋めるか、y_meanを引くだけにする。
#             y_train_norm = y_train - y_mean 
#         else:
#             y_train_norm = (y_train - y_mean) / y_std
#         # ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲ 修正箇所 ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲
#         for _ in range(self.epochs):
#             self.optimizer.zero_grad()
#             predictions = self.surrogate_model(x_train)
#             loss = self.loss_fn(predictions, y_train_norm)
#             loss.backward()
#             self.optimizer.step()
#         logging.info(f"  Training complete. Final Loss: {loss.item():.4f}")

#     def _build_and_solve_milp(self, past_x_cats: List[Dict[str, int]]) -> Dict[str, int]:
#         logging.info("  Building and solving MILP problem...")
#         solver = pywraplp.Solver.CreateSolver('SCIP')
#         if not solver: return {name: self._rng.randint(len(cats)) for name, cats in self.categories_per_param.items()}
#         if self.time_limit_sec: solver.SetTimeLimit(self.time_limit_sec * 1000)

#         x = [solver.BoolVar(f'x_{i}') for i in range(self.input_dim)]
#         h = [solver.NumVar(0, solver.infinity(), f'h_{j}') for j in range(self.hidden_dim)]
#         a = [solver.BoolVar(f'a_{j}') for j in range(self.hidden_dim)]
#         output_var = solver.NumVar(-solver.infinity(), solver.infinity(), 'output')
        
#         w1, b1 = self.surrogate_model.layer1.weight.data.numpy().T, self.surrogate_model.layer1.bias.data.numpy()
#         w_out, b_out = self.surrogate_model.output_layer.weight.data.numpy().T, self.surrogate_model.output_layer.bias.data.numpy()
#         M = 1000
#         for j in range(self.hidden_dim):
#             pre_act = solver.Sum([float(w1[i, j]) * x[i] for i in range(self.input_dim)]) + float(b1[j])
#             solver.Add(h[j] >= pre_act)
#             solver.Add(h[j] <= pre_act + M * (1 - a[j]))
#             solver.Add(h[j] <= M * a[j])
#         output_expr = solver.Sum([float(w_out[j, 0]) * h[j] for j in range(self.hidden_dim)]) + float(b_out[0])
#         solver.Add(output_var == output_expr)
        
#         for name, indices in self.one_hot_lookup.items():
#             solver.Add(solver.Sum(x[idx] for idx in indices) == 1)
        
#         for x_cat_dict in past_x_cats:
#             past_one_hot = self._encode_one_hot(x_cat_dict)
#             solver.Add(solver.Sum(x[i] for i, bit in enumerate(past_one_hot) if bit == 0) +
#                        solver.Sum(1 - x[i] for i, bit in enumerate(past_one_hot) if bit == 1) >= 1)
        
#         self._add_problem_specific_constraints(solver, x)
        
#         solver.Maximize(output_var)
#         status = solver.Solve()

#         if status in [pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE]:
#             solution = [var.solution_value() for var in x]
#             logging.info("  MILP solved. Found a new point.")
#             return self._decode_one_hot(solution)
#         else:
#             logging.warning("  MILP solver could not find a feasible solution. Returning a random point.")
#             return {name: self._rng.randint(len(cats)) for name, cats in self.categories_per_param.items()}

#     def _add_problem_specific_constraints(self, solver, x_vars):
#         """Adds constraints to the MILP based on the problem type."""
#         # Generalized handler for tensor-based constraints (works for Ackley and PressureVessel)
#         if self.problem_name in ['ackley', 'pressure'] and self.constraints.get('tensor') is not None:
#             tensor = self.constraints['tensor']
#             logging.info(f"  Adding {np.sum(tensor == 0)} tensor-based constraints for {self.problem_name}...")
#             param_names = self.param_names 
#             infeasible_coords = np.argwhere(tensor == 0)
#             for coord in infeasible_coords:
#                 # Forbid the specific combination of choices.
#                 # Ex: x_p0_c0 + x_p1_c1 + ... <= n_dims - 1
#                 constraint_vars = []
#                 for i, param_idx in enumerate(coord):
#                     param_name = param_names[i]
#                     var_index = self.one_hot_lookup[param_name][param_idx]
#                     constraint_vars.append(x_vars[var_index])
#                 solver.Add(solver.Sum(constraint_vars) <= len(constraint_vars) - 1)
        
#         elif self.problem_name == 'diabetes' and self.constraints.get('zero_entries') is not None:
#             zero_entries = self.constraints['zero_entries']
#             num_pairwise = sum(len(v) for v in zero_entries.values())
#             logging.info(f"  Adding {num_pairwise} pairwise constraints for Diabetes...")
#             for (f1, f2), combinations in zero_entries.items():
#                 indices_f1, indices_f2 = self.one_hot_lookup[f1], self.one_hot_lookup[f2]
#                 for b1, b2 in combinations:
#                     solver.Add(x_vars[indices_f1[b1]] + x_vars[indices_f2[b2]] <= 1)

#         elif self.problem_name == 'warcraft' and self.constraints:
#             logging.info("  Adding Warcraft-specific constraints...")
#             directions = self.categories_per_param[self.param_names[0]]
#             map_shape = self.constraints['map_shape']
#             start_param_name, goal_param_name = f"x_0_0", f"x_{map_shape[0]-1}_{map_shape[1]-1}"
            
#             for direction in self.constraints.get('start_forbidden', []):
#                 solver.Add(x_vars[self.one_hot_lookup[start_param_name][directions.index(direction)]] == 0)
#             for direction in self.constraints.get('goal_forbidden', []):
#                 solver.Add(x_vars[self.one_hot_lookup[goal_param_name][directions.index(direction)]] == 0)
            
#             gains, ideal_gain = self.constraints.get('gains'), self.constraints.get('ideal_gain')
#             if gains and ideal_gain is not None:
#                 gain_expr = solver.Sum(
#                     gains[d_idx] * x_vars[self.one_hot_lookup[p_name][d_idx]]
#                     for p_name in self.param_names for d_idx in range(len(directions))
#                 )
#                 solver.Add(gain_expr == ideal_gain)

# # ==========================================================================
# # 統合実験実行関数 (Unified Experiment Runner)
# # ==========================================================================

# def run_bo(settings: Dict):
#     """Main function to set up and run a Bayesian Optimization experiment."""
#     problem_name = settings['problem_name']
#     logging.info(f"Setting up experiment for problem: '{problem_name}'")

#     random.seed(settings['seed']); np.random.seed(settings['seed']); torch.manual_seed(settings['seed'])
    
#     if problem_name == 'ackley':
#         ackley_obj = AckleyBenchmark(constrain=settings['constrain'])
#         CATEGORIES = list(range(ackley_obj.bounds[0], ackley_obj.bounds[1] + 1))
#         PARAM_NAMES = [f'x_{i}' for i in range(ackley_obj.n_dim)]
#         def objective_wrapper(trial: optuna.Trial) -> float:
#             x = [trial.suggest_categorical(name, CATEGORIES) for name in PARAM_NAMES]
#             return ackley_obj.evaluate(x)
#         search_space_info = {'param_names': PARAM_NAMES, 'categories_per_param': {name: CATEGORIES for name in PARAM_NAMES}}
#         constraints = {'tensor': ackley_obj._tensor_constraint} if settings['constrain'] else {}

#     elif problem_name == 'diabetes':
#         df, zero_entries = prepare_diabetes_data_and_constraints()
#         diabetes_obj = DiabetesObjective(df=df, seed=settings['seed'])
#         features = df.drop(columns='Outcome').columns.tolist()
#         def objective_wrapper(trial: optuna.Trial) -> float:
#             params = {name: trial.suggest_int(name, 0, 4) for name in features}
#             x = np.array([params[name] for name in features])
#             return diabetes_obj(x)
#         search_space_info = {'param_names': features, 'categories_per_param': {f: list(range(5)) for f in features}}
#         constraints = {'zero_entries': zero_entries} if settings['constrain'] else {}
    
#     elif problem_name == 'pressure':
#         pressure_obj = PressureVesselObjective(seed=settings['seed'], n_bins=settings['n_bins'])
#         PARAM_NAMES = pressure_obj.features
#         CATEGORIES_PER_PARAM = {name: pressure_obj.mid_points[i].tolist() for i, name in enumerate(PARAM_NAMES)}
#         def objective_wrapper(trial: optuna.Trial) -> float:
#             x_indices = []
#             for i, name in enumerate(PARAM_NAMES):
#                 category_values = CATEGORIES_PER_PARAM[name]
#                 suggested_val = trial.suggest_categorical(name, category_values)
#                 x_indices.append(category_values.index(suggested_val))
#             return pressure_obj(np.array(x_indices), is_constrained=settings['constrain'])
#         search_space_info = {'param_names': PARAM_NAMES, 'categories_per_param': CATEGORIES_PER_PARAM}
#         constraints = {'tensor': pressure_obj._tensor_constraint} if settings['constrain'] else {}

#     elif problem_name == 'warcraft':
#         DIRECTIONS = ["oo", "ab", "ac", "ad", "bc", "bd", "cd"]
#         weight_matrix = get_map(settings['map_option'])
#         warcraft_obj = WarcraftObjective(weight_matrix=weight_matrix)
#         map_shape = weight_matrix.shape
#         PARAM_NAMES = [f"x_{r}_{c}" for r in range(map_shape[0]) for c in range(map_shape[1])]
#         def objective_wrapper(trial: optuna.Trial) -> float:
#             params = {name: trial.suggest_categorical(name, DIRECTIONS) for name in PARAM_NAMES}
#             direction_matrix = np.array(list(params.values())).reshape(map_shape)
#             return warcraft_obj(direction_matrix)
#         search_space_info = {'param_names': PARAM_NAMES, 'categories_per_param': {name: DIRECTIONS for name in PARAM_NAMES}}
#         constraints = {
#             "map_shape": map_shape, "start_forbidden": ["oo", "ab"], "goal_forbidden": ["oo", "cd"],
#             "gains": [0, 2, 2, 2, 2, 2, 2], "ideal_gain": (map_shape[0] + map_shape[1] - 1) * 2
#         } if settings['constrain'] else {}

#     else:
#         raise ValueError(f"Unknown problem name: {problem_name}")
    
#     sampler = NNMILPSampler(
#         problem_name=problem_name, search_space_info=search_space_info,
#         constraints=constraints, seed=settings['seed'], sampler_settings=settings['sampler_settings']
#     )
    
#     study = optuna.create_study(
#         study_name=settings['name'], storage=settings['storage'],
#         sampler=sampler, direction="minimize", load_if_exists=True
#     )
    
#     study.optimize(objective_wrapper, n_trials=settings['n_trials'])
    
#     logging.info(f"\n{'='*15} Optimization Complete for '{problem_name}' {'='*15}")
#     logging.info(f"Best score: {study.best_value:.4f}")
#     logging.info(f"Best params: {study.best_params}")

#     if settings.get("plot_save_dir"):
#         try:
#             fig = optuna.visualization.plot_optimization_history(study)
#             plot_path = os.path.join(settings["plot_save_dir"], f"{settings['name']}_history.png")
#             fig.write_image(plot_path)
#             logging.info(f"Optimization history plot saved to {plot_path}")
#         except Exception as e:
#             logging.error(f"Failed to generate or save plot. Is 'plotly' installed? Error: {e}")

# # ==========================================================================
# # メイン実行ブロック (Main Execution Block)
# # ==========================================================================

# if __name__ == '__main__':
#     parser = argparse.ArgumentParser(description="Run BO with NN+MILP Sampler for various problems.")
    
#     parser.add_argument("--function", type=str, required=True,
#                         choices=['ackley', 'diabetes', 'pressure', 'warcraft'],
#                         help="Select the problem to run.")
    
#     # General experiment arguments
#     parser.add_argument("--timestamp", type=str, required=True, help="Shared timestamp for the experiment run")
#     parser.add_argument("--base_dir", type=str, default="results", help="Base directory to save all results")
    
#     # Sampler and optimization arguments
#     parser.add_argument("--seed", type=int, required=True, help="Random seed")
#     parser.add_argument("--n_trials", type=int, default=100, help="Number of BO trials")
#     parser.add_argument("--constrain", action="store_true", help="Enable problem-specific constraints")
#     parser.add_argument("--epochs", type=int, default=300, help="Epochs for surrogate NN training")
#     parser.add_argument("--hidden_dim", type=int, default=16, help="Hidden dimension of surrogate NN")
#     parser.add_argument("--n_startup_trials", type=int, default=20, help="Number of initial random trials")

#     # Problem-specific arguments
#     parser.add_argument("--n_bins", type=int, default=10, help="Number of bins for PressureVessel discretization")
#     parser.add_argument("--map_option", type=int, default=1, choices=[1, 2, 3], help="Warcraft map option")
#     parser.add_argument("--time_limit_sec", type=int, default=None, help="Time limit for MILP solver in seconds")

#     args = parser.parse_args()
    
#     results_dir = os.path.join(args.base_dir, args.timestamp)
#     plot_dir = os.path.join(results_dir, "plots")
#     os.makedirs(plot_dir, exist_ok=True)
    
#     constraint_str = "con" if args.constrain else "uncon"
#     log_filename_base = f"{args.function}_{constraint_str}_seed{args.seed}"
#     exp_name = f"{args.timestamp}_{log_filename_base}"
    
#     set_logger(log_filename_base, results_dir)

#     storage_path = os.path.join(results_dir, f"{log_filename_base}.db")
#     storage_url = f"sqlite:///{storage_path}"

#     settings = {
#         "problem_name": args.function, "name": exp_name, "seed": args.seed,
#         "n_trials": args.n_trials, "constrain": args.constrain, "storage": storage_url,
#         "plot_save_dir": plot_dir,
#         "sampler_settings": {
#             "epochs": args.epochs, "hidden_dim": args.hidden_dim,
#             "n_startup_trials": args.n_startup_trials,
#         }
#     }
    
#     if args.function == 'pressure':
#         settings['n_bins'] = args.n_bins
#     if args.function == 'warcraft':
#         settings['map_option'] = args.map_option
#         settings['sampler_settings']['time_limit_sec'] = args.time_limit_sec

#     logging.info(f"Starting experiment with settings: {settings}")
#     try:
#         run_bo(settings)
#         logging.info("Experiment finished successfully.")
#     except Exception as e:
#         logging.error(f"An error occurred during the experiment: {e}", exc_info=True)


import argparse
import logging
import os
import random
from itertools import product
from typing import Any, Dict, List, Tuple

import numpy as np
import optuna
import pandas as pd
import torch
import torch.nn as nn
from optuna.samplers import BaseSampler
from ortools.linear_solver import pywraplp

# Optional: For saving plots. Requires 'plotly' and 'kaleido'.
# try:
#     import plotly
# except ImportError:
#     pass

# ==========================================================================
# ユーティリティ関数 (Common Utility Functions)
# ==========================================================================

def set_logger(log_filename_base: str, results_dir: str):
    """Initializes the logger to write to both file and console."""
    log_filepath = os.path.join(results_dir, f"{log_filename_base}.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(log_filepath), logging.StreamHandler()]
    )

# ==========================================================================
# 問題定義: Ackley (Problem Definition: Ackley)
# ==========================================================================

class AckleyBenchmark:
    """Defines the Ackley benchmark function."""
    ### ▼▼▼ 修正箇所 ▼▼▼ ###
    # __init__の引数 `constrain` は外部とのAPI互換性のために残すが、
    # 内部の制約テンソルは常に構築する。
    def __init__(self, constrain: bool = False):
        self.bounds = [-32, 32]
        self.n_dim = 2
        self.fmax = 22.3497
        
        # 制約テンソルを無条件で構築
        X, Y = np.meshgrid(
            np.arange(self.bounds[0], self.bounds[1] + 1),
            np.arange(self.bounds[0], self.bounds[1] + 1)
        )
        R_squared = X**2 + Y**2
        self._tensor_constraint = (R_squared < 10**2).astype(int)

    def _coord_to_index(self, x: List[int]) -> List[int]:
        return [int(xi - self.bounds[0]) for xi in x]

    def evaluate(self, x: List[int]) -> float:
        # 評価時には常に制約をチェックする
        idx = self._coord_to_index(x)
        if not self._tensor_constraint[idx[1], idx[0]]:
            return self.fmax # 制約違反時はfmaxを返す

        a, b, c = 20, 0.2, 2 * np.pi
        d = 2
        term1 = -a * np.exp(-b * np.sqrt((x[0]**2 + x[1]**2) / d))
        term2 = -np.exp((np.cos(c * x[0]) + np.cos(c * x[1])) / d)
        return term1 + term2 + a + np.exp(1)
    ### ▲▲▲ 修正箇所 ▲▲▲ ###

# ==========================================================================
# 問題定義: Diabetes (Problem Definition: Diabetes)
# ==========================================================================

def prepare_diabetes_data_and_constraints() -> Tuple[pd.DataFrame, Dict]:
    # (この関数は変更なし)
    data_path = 'data/diabetes.csv'
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"{data_path} not found. Please prepare the dataset.")
    df = pd.read_csv(data_path)
    df_features = df.drop(columns='Outcome')
    features = df_features.columns.tolist()
    n_features = len(features)
    n_bins = 5
    binned_df = pd.DataFrame({
        c: pd.qcut(df_features[c], q=n_bins, labels=False, duplicates='drop') for c in features
    })
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

class DiabetesObjective:
    """Defines the objective function for the Diabetes counterfactual task."""
    ### ▼▼▼ 修正箇所 ▼▼▼ ###
    # __init__で制約情報を無条件に受け取り、内部に保持する
    def __init__(self, df: pd.DataFrame, zero_entries: Dict, seed: int = 42):
        np.random.seed(seed)
        self.df = df
        self.df_features = self.df.drop(columns='Outcome')
        self.features = self.df_features.columns.tolist()
        # ペアワイズ制約情報をクラス内に保持
        self.zero_entries = zero_entries
        # 制約違反の組み合わせを高速にチェックできるようセットに変換
        self.zero_entries_set = {k: set(v) for k, v in self.zero_entries.items()}
        
        tensors_path = 'data/diabetes_analysis_tensors.npz'
        if not os.path.exists(tensors_path):
            raise FileNotFoundError(f"{tensors_path} not found. Please prepare the pre-analyzed tensor file.")
        loaded_tensors = np.load(tensors_path)
        self._tensor_predicted = loaded_tensors['predicted_tensor']
        
        positive_samples = self.df[self.df['Outcome'] == 1]
        if positive_samples.empty: raise ValueError("No positive samples (Outcome=1) to select a start point.")
        start_row = positive_samples.sample(1, random_state=seed)
        
        binned_start = []
        for f in self.features:
            binned_value_series = pd.qcut(start_row[f], q=5, labels=False, duplicates='drop')
            binned_value = binned_value_series.iloc[0]
            binned_start.append(0 if pd.isna(binned_value) else int(binned_value))
        self._x_start = np.array(binned_start, dtype=int)
        
        logging.info(f"Start point x': {self._x_start}")
        logging.info(f"Predicted value at start point: {self._tensor_predicted[tuple(self._x_start)]:.4f}")

    def __call__(self, x: np.ndarray) -> float:
        # 評価時には常に制約をチェックする
        for (f1, f2), combinations_set in self.zero_entries_set.items():
            f1_idx = self.features.index(f1)
            f2_idx = self.features.index(f2)
            current_bin_pair = (x[f1_idx], x[f2_idx])
            if current_bin_pair in combinations_set:
                return 2.0 # 見本コードに従いペナルティは 2.0 (1+1)
        
        # 制約を満たす場合のみ目的関数を計算
        f_x = self._tensor_predicted[tuple(x)]
        max_distance = np.linalg.norm(np.array([4] * len(x)))
        distance_penalty = np.linalg.norm(x - self._x_start) / max_distance
        return f_x + distance_penalty
    ### ▲▲▲ 修正箇所 ▲▲▲ ###

# ==========================================================================
# 問題定義: Pressure Vessel (Problem Definition: Pressure Vessel)
# ==========================================================================

# (このクラスは元から要件を満たしていたため、変更なし)
class PressureVesselObjective:
    def __init__(self, start_point=None, seed=42, n_bins=10):
        self.seed = seed; np.random.seed(seed)
        self.features = ['Ts', 'Th', 'R', 'L']
        ts_min, ts_max = 0.0625, 6.1875
        th_min, th_max = 0.0625, 6.1875
        r_min, r_max = 10.0, 200.0
        l_min, l_max = 10.0, 200.0
        self.mid_points = []
        for min_val, max_val in [(ts_min, ts_max), (th_min, th_max), (r_min, r_max), (l_min, l_max)]:
            edges = np.linspace(min_val, max_val, n_bins + 1)
            self.mid_points.append([(edges[i] + edges[i+1]) / 2 for i in range(n_bins)])
        shape = tuple(len(m) for m in self.mid_points)
        self._tensor_objective = np.empty(shape)
        self._tensor_constraint = np.empty(shape, dtype=bool)
        for i, ts in enumerate(self.mid_points[0]):
            for j, th in enumerate(self.mid_points[1]):
                for k, r in enumerate(self.mid_points[2]):
                    for l, l_val in enumerate(self.mid_points[3]):
                        x = np.array([ts, th, r, l_val])
                        self._tensor_objective[i, j, k, l] = self._objective_formula(x)
                        self._tensor_constraint[i, j, k, l] = np.all(self._constraints_formula(x) <= 0)
        self._feasible_indices = np.argwhere(self._tensor_constraint == True)
        if len(self._feasible_indices) == 0:
            raise RuntimeError("No feasible points found.")
        logging.info(f"Problem space discretized. Found {len(self._feasible_indices)} feasible points.")
    def _objective_formula(self, x: np.ndarray) -> float:
        ts, th, r, l_val = x[0], x[1], x[2], x[3]
        return (0.6224*ts*r*l_val + 1.7781*th*r**2 + 3.1661*ts**2*l_val + 19.84*ts**2*r)
    def _constraints_formula(self, x: np.ndarray) -> np.ndarray:
        ts, th, r, l_val = x[0], x[1], x[2], x[3]
        g1 = -ts + 0.0193*r; g2 = -th + 0.00954*r
        g3 = -np.pi*r**2*l_val - (4/3)*np.pi*r**3 + 1_296_000; g4 = l_val - 240
        return np.array([g1, g2, g3, g4])
    def __call__(self, x: np.ndarray) -> float:
        x_tuple = tuple(x)
        if not self._tensor_constraint[x_tuple]:
            return np.max(self._tensor_objective[self._tensor_constraint])
        return self._tensor_objective[x_tuple]

# ==========================================================================
# 問題定義: Warcraft (Problem Definition: Warcraft)
# ==========================================================================

def get_map(map_option: int) -> np.ndarray:
    if map_option == 1: return np.array([[1, 4], [2, 1]])
    if map_option == 2: return np.array([[1, 4, 1], [2, 1, 1]])
    if map_option == 3: return np.array([[1, 4, 1], [2, 1, 3], [5, 2, 1]])
    raise ValueError(f"Invalid map option: {map_option}")
def navigate_through_matrix(direction_matrix, start, goal):
    def get_opposite(d): return {"a":"c","c":"a","b":"d","d":"b"}.get(d,"")
    def judge_continuity(d_from, current_dir): return get_opposite(d_from) in current_dir
    def get_d_to(d_from, current_dir): return current_dir[1] if current_dir[0] == d_from else current_dir[0]
    def get_next_coord(d_to, current):
        delta = {"a":(-1,0),"b":(0,-1),"c":(0,1),"d":(1,0)}.get(d_to,(0,0))
        return (current[0]+delta[0], current[1]+delta[1])
    def is_valid(coord, shape): return 0<=coord[0]<shape[0] and 0<=coord[1]<shape[1]
    history, current = [], start; shape = direction_matrix.shape
    current_direction = direction_matrix[current]
    if "a" in current_direction or "b" in current_direction:
        d_to = get_d_to("a", current_direction) if "a" in current_direction else get_d_to("b", current_direction)
    else: return history
    if current_direction == "ab": history.append(current); return history
    history.append(current)
    next_pos = get_next_coord(d_to, current)
    while is_valid(next_pos, shape) and current != goal:
        if direction_matrix[next_pos] == "oo": break
        if not judge_continuity(d_to, direction_matrix[next_pos]): break
        current = next_pos; history.append(current)
        if current == goal: break
        d_from = get_opposite(d_to); d_to = get_d_to(d_from, direction_matrix[current])
        next_pos = get_next_coord(d_to, current)
    return history

class WarcraftObjective:
    """Defines the objective function for the Warcraft pathfinding problem."""
    ### ▼▼▼ 修正箇所 ▼▼▼ ###
    # constrain引数を残しつつ、制約ルールはクラス内部で無条件に定義する
    def __init__(self, weight_matrix: np.ndarray, constrain: bool = False):
        self.weight_matrix = weight_matrix / np.sum(weight_matrix)
        self.shape = weight_matrix.shape
        # 制約ルールを内部で定義
        self.DIRECTIONS = ["oo", "ab", "ac", "ad", "bc", "bd", "cd"]
        self.start_forbidden = {"oo", "ab"}
        self.goal_forbidden = {"oo", "cd"}
        self.gains = [0, 2, 2, 2, 2, 2, 2]
        self.ideal_gain = (self.shape[0] + self.shape[1] - 1) * 2
        
        self._val_mask_dict = {"oo":np.zeros((3,3)),"ab":np.array([[0,1,0],[1,1,0],[0,0,0]]),"ac":np.array([[0,0,0],[1,1,1],[0,0,0]]),"ad":np.array([[0,0,0],[1,1,0],[0,1,0]]),"bc":np.array([[0,1,0],[0,1,1],[0,0,0]]),"bd":np.array([[0,1,0],[0,1,0],[0,1,0]]),"cd":np.array([[0,0,0],[0,1,1],[0,1,0]])}
    ### ▲▲▲ 修正箇所 ▲▲▲ ###
    
    def _calculate_penalty_type2(self, idx, val, map_shape):
        def manhattan_dist(c1,c2): return abs(c1[0]-c2[0])+abs(c1[1]-c2[1])
        arr_exp = np.zeros((map_shape[0]*2+1, map_shape[1]*2+1))
        x_s, y_s = idx[0]*2, idx[1]*2
        arr_exp[x_s:x_s+3, y_s:y_s+3] = self._val_mask_dict.get(val, np.zeros((3,3)))
        ones = np.argwhere(arr_exp == 1)
        row, col = arr_exp.shape[0]-1, arr_exp.shape[1]-1
        max_dist = manhattan_dist((0,0), (row,col-1)); min_dist = max_dist
        goals = [(row, col-1), (row-1, col)]
        for one_idx in ones:
            for target_idx in goals:
                dist = manhattan_dist(one_idx, target_idx)
                if dist < min_dist: min_dist = dist
        return min_dist / max_dist if max_dist > 0 else 1

    def __call__(self, direction_matrix: np.ndarray) -> float:
        ### ▼▼▼ 修正箇所 ▼▼▼ ###
        # 評価時には常に制約をチェックする
        start_dir = direction_matrix[0, 0]
        goal_dir = direction_matrix[self.shape[0] - 1, self.shape[1] - 1]

        if start_dir in self.start_forbidden: return 2.0
        if goal_dir in self.goal_forbidden: return 2.0
        
        current_gain = sum(self.gains[self.DIRECTIONS.index(d)] for d in direction_matrix.flatten())
        if current_gain != self.ideal_gain: return 2.0
        ### ▲▲▲ 修正箇所 ▲▲▲ ###
        
        # 制約を満たす場合のみ目的関数を計算
        mask = np.where(direction_matrix == "oo", 0, 1)
        penalty_1 = np.sum(self.weight_matrix * mask)
        start, goal = (0, 0), (self.shape[0] - 1, self.shape[1] - 1)
        history = navigate_through_matrix(direction_matrix, start, goal)
        penalty_3 = self._calculate_penalty_type2(history[-1], direction_matrix[history[-1]], self.shape) if history else 1
        return penalty_1 + penalty_3

# ==========================================================================
# 代理モデル と Sampler (Surrogate Model and Sampler are unchanged)
# ==========================================================================
class SimpleReLU_NN(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super(SimpleReLU_NN, self).__init__(); self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU(); self.output_layer = nn.Linear(hidden_dim, 1)
    def forward(self, x): return self.output_layer(self.relu(self.layer1(x)))

class NNMILPSampler(BaseSampler):
    def __init__(self, search_space_info: Dict, problem_name: str, constraints: Dict,
                 seed: int = None, sampler_settings: Dict = None):
        self._rng = np.random.RandomState(seed); torch.manual_seed(seed if seed is not None else random.randint(0, 2**32-1))
        if seed is not None: random.seed(seed)
        self._independent_sampler = optuna.samplers.RandomSampler(seed=seed)
        self.problem_name = problem_name; self.param_names = search_space_info['param_names']
        self.categories_per_param = search_space_info['categories_per_param']
        self.constraints = constraints if constraints else {}; self.num_vars = len(self.param_names)
        self.input_dim = sum(len(cats) for cats in self.categories_per_param.values())
        self.one_hot_lookup = self._create_one_hot_lookup()
        settings = sampler_settings if sampler_settings is not None else {}
        self.n_startup_trials = settings.get("n_startup_trials", 20)
        self.hidden_dim = settings.get("hidden_dim", 16); self.epochs = settings.get("epochs", 300)
        self.time_limit_sec = settings.get("time_limit_sec", None)
        self.surrogate_model = SimpleReLU_NN(self.input_dim, self.hidden_dim)
        self.optimizer = torch.optim.Adam(self.surrogate_model.parameters(), lr=0.01)
        self.loss_fn = nn.MSELoss()
    def _create_one_hot_lookup(self) -> Dict[str, List[int]]:
        lookup, start_idx = {}, 0
        for name in self.param_names:
            num_cats = len(self.categories_per_param[name])
            lookup[name] = list(range(start_idx, start_idx + num_cats)); start_idx += num_cats
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
    def sample_independent(self, study: optuna.Study, trial: optuna.Trial, param_name: str,
                           param_distribution: optuna.distributions.BaseDistribution) -> Any:
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
        if len(y_train) <= 1 or torch.isnan(y_std) or y_std < 1e-8:
            y_train_norm = y_train - y_mean
        else:
            y_train_norm = (y_train - y_mean) / y_std
        for _ in range(self.epochs):
            self.optimizer.zero_grad(); predictions = self.surrogate_model(x_train)
            loss = self.loss_fn(predictions, y_train_norm); loss.backward(); self.optimizer.step()
        logging.info(f"  Training complete. Final Loss: {loss.item():.4f}")
    def _build_and_solve_milp(self, past_x_cats: List[Dict[str, int]]) -> Dict[str, int]:
        logging.info("  Building and solving MILP problem...")
        solver = pywraplp.Solver.CreateSolver('SCIP')
        if not solver: return {name: self._rng.randint(len(cats)) for name, cats in self.categories_per_param.items()}
        if self.time_limit_sec: solver.SetTimeLimit(self.time_limit_sec * 1000)
        x = [solver.BoolVar(f'x_{i}') for i in range(self.input_dim)]
        h = [solver.NumVar(0, solver.infinity(), f'h_{j}') for j in range(self.hidden_dim)]
        a = [solver.BoolVar(f'a_{j}') for j in range(self.hidden_dim)]
        output_var = solver.NumVar(-solver.infinity(), solver.infinity(), 'output')
        w1, b1 = self.surrogate_model.layer1.weight.data.numpy().T, self.surrogate_model.layer1.bias.data.numpy()
        w_out,b_out = self.surrogate_model.output_layer.weight.data.numpy().T, self.surrogate_model.output_layer.bias.data.numpy()
        M = 1000
        for j in range(self.hidden_dim):
            pre_act = solver.Sum([float(w1[i,j])*x[i] for i in range(self.input_dim)]) + float(b1[j])
            solver.Add(h[j] >= pre_act); solver.Add(h[j] <= pre_act + M*(1-a[j])); solver.Add(h[j] <= M*a[j])
        output_expr = solver.Sum([float(w_out[j,0])*h[j] for j in range(self.hidden_dim)]) + float(b_out[0])
        solver.Add(output_var == output_expr)
        for name, indices in self.one_hot_lookup.items():
            solver.Add(solver.Sum(x[idx] for idx in indices) == 1)
        for x_cat_dict in past_x_cats:
            past_one_hot = self._encode_one_hot(x_cat_dict)
            solver.Add(solver.Sum(x[i] for i,bit in enumerate(past_one_hot) if bit==0) +
                       solver.Sum(1-x[i] for i,bit in enumerate(past_one_hot) if bit==1) >= 1)
        self._add_problem_specific_constraints(solver, x)
        solver.Minimize(output_var) # Changed to Minimize as per standard BO
        status = solver.Solve()
        if status in [pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE]:
            solution = [var.solution_value() for var in x]
            logging.info("  MILP solved. Found a new point.")
            return self._decode_one_hot(solution)
        else:
            logging.warning("  MILP solver could not find a feasible solution. Returning a random point.")
            return {name: self._rng.randint(len(cats)) for name, cats in self.categories_per_param.items()}
    def _add_problem_specific_constraints(self, solver, x_vars):
        if self.problem_name in ['ackley', 'pressure'] and self.constraints.get('tensor') is not None:
            tensor = self.constraints['tensor']; logging.info(f"  Adding {np.sum(tensor == 0)} tensor constraints...")
            param_names = self.param_names; infeasible_coords = np.argwhere(tensor == 0)
            for coord in infeasible_coords:
                constraint_vars = []
                for i, param_idx in enumerate(coord):
                    param_name = param_names[i]
                    var_index = self.one_hot_lookup[param_name][param_idx]
                    constraint_vars.append(x_vars[var_index])
                solver.Add(solver.Sum(constraint_vars) <= len(constraint_vars) - 1)
        elif self.problem_name == 'diabetes' and self.constraints.get('zero_entries') is not None:
            zero_entries = self.constraints['zero_entries']; num_pairwise = sum(len(v) for v in zero_entries.values())
            logging.info(f"  Adding {num_pairwise} pairwise constraints for Diabetes...")
            for (f1, f2), combinations in zero_entries.items():
                indices_f1, indices_f2 = self.one_hot_lookup[f1], self.one_hot_lookup[f2]
                for b1, b2 in combinations:
                    solver.Add(x_vars[indices_f1[b1]] + x_vars[indices_f2[b2]] <= 1)
        elif self.problem_name == 'warcraft' and self.constraints:
            logging.info("  Adding Warcraft-specific constraints..."); directions = self.categories_per_param[self.param_names[0]]
            map_shape = self.constraints['map_shape']
            start_param_name, goal_param_name = f"x_0_0", f"x_{map_shape[0]-1}_{map_shape[1]-1}"
            for direction in self.constraints.get('start_forbidden', []):
                solver.Add(x_vars[self.one_hot_lookup[start_param_name][directions.index(direction)]] == 0)
            for direction in self.constraints.get('goal_forbidden', []):
                solver.Add(x_vars[self.one_hot_lookup[goal_param_name][directions.index(direction)]] == 0)
            gains, ideal_gain = self.constraints.get('gains'), self.constraints.get('ideal_gain')
            if gains and ideal_gain is not None:
                gain_expr = solver.Sum(gains[d_idx] * x_vars[self.one_hot_lookup[p_name][d_idx]] for p_name in self.param_names for d_idx in range(len(directions)))
                solver.Add(gain_expr == ideal_gain)

# ==========================================================================
# 統合実験実行関数 (Unified Experiment Runner)
# ==========================================================================

def run_bo(settings: Dict):
    problem_name = settings['problem_name']
    logging.info(f"Setting up experiment for problem: '{problem_name}'")
    random.seed(settings['seed']); np.random.seed(settings['seed']); torch.manual_seed(settings['seed'])

    if problem_name == 'ackley':
        # `constrain` フラグは渡すが、クラス内部では無視される
        ackley_obj = AckleyBenchmark(constrain=settings['constrain'])
        CATEGORIES = list(range(ackley_obj.bounds[0], ackley_obj.bounds[1] + 1))
        PARAM_NAMES = [f'x_{i}' for i in range(ackley_obj.n_dim)]
        def objective_wrapper(trial: optuna.Trial) -> float:
            x = [trial.suggest_categorical(name, CATEGORIES) for name in PARAM_NAMES]
            return ackley_obj.evaluate(x)
        search_space_info = {'param_names': PARAM_NAMES, 'categories_per_param': {name: CATEGORIES for name in PARAM_NAMES}}
        # サンプラーに渡す制約は `constrain` フラグで制御
        constraints = {'tensor': ackley_obj._tensor_constraint} if settings['constrain'] else {}

    elif problem_name == 'diabetes':
        ### ▼▼▼ 修正箇所 ▼▼▼ ###
        # 制約データを常に準備し、目的関数オブジェクトに渡す
        df, zero_entries = prepare_diabetes_data_and_constraints()
        diabetes_obj = DiabetesObjective(df=df, zero_entries=zero_entries, seed=settings['seed'])
        features = diabetes_obj.features
        def objective_wrapper(trial: optuna.Trial) -> float:
            params = {name: trial.suggest_int(name, 0, 4) for name in features}
            x = np.array([params[name] for name in sorted(features)]) # 順序を保証
            return diabetes_obj(x)
        search_space_info = {'param_names': sorted(features), 'categories_per_param': {f: list(range(5)) for f in sorted(features)}}
        # サンプラーに渡す制約は `constrain` フラグで制御
        constraints = {'zero_entries': zero_entries} if settings['constrain'] else {}
        ### ▲▲▲ 修正箇所 ▲▲▲ ###
    
    elif problem_name == 'pressure':
        pressure_obj = PressureVesselObjective(seed=settings['seed'], n_bins=settings['n_bins'])
        PARAM_NAMES = pressure_obj.features
        CATEGORIES_PER_PARAM = {name: pressure_obj.mid_points[i] for i, name in enumerate(PARAM_NAMES)}
        def objective_wrapper(trial: optuna.Trial) -> float:
            x_indices = []
            for i, name in enumerate(PARAM_NAMES):
                suggested_val = trial.suggest_categorical(name, CATEGORIES_PER_PARAM[name])
                x_indices.append(CATEGORIES_PER_PARAM[name].index(suggested_val))
            return pressure_obj(np.array(x_indices))
        search_space_info = {'param_names': PARAM_NAMES, 'categories_per_param': CATEGORIES_PER_PARAM}
        constraints = {'tensor': pressure_obj._tensor_constraint} if settings['constrain'] else {}

    elif problem_name == 'warcraft':
        ### ▼▼▼ 修正箇所 ▼▼▼ ###
        # 目的関数オブジェクトは常に制約を内包して初期化
        weight_matrix = get_map(settings['map_option'])
        warcraft_obj = WarcraftObjective(weight_matrix=weight_matrix, constrain=settings['constrain'])
        map_shape = weight_matrix.shape
        PARAM_NAMES = [f"x_{r}_{c}" for r in range(map_shape[0]) for c in range(map_shape[1])]
        def objective_wrapper(trial: optuna.Trial) -> float:
            params = {name: trial.suggest_categorical(name, warcraft_obj.DIRECTIONS) for name in PARAM_NAMES}
            direction_matrix = np.array([params[name] for name in PARAM_NAMES]).reshape(map_shape)
            return warcraft_obj(direction_matrix)
        search_space_info = {'param_names': PARAM_NAMES, 'categories_per_param': {name: warcraft_obj.DIRECTIONS for name in PARAM_NAMES}}
        # サンプラーに渡す制約は `constrain` フラグで制御
        constraints = {
            "map_shape": map_shape, "start_forbidden": list(warcraft_obj.start_forbidden),
            "goal_forbidden": list(warcraft_obj.goal_forbidden), "gains": warcraft_obj.gains,
            "ideal_gain": warcraft_obj.ideal_gain
        } if settings['constrain'] else {}
        ### ▲▲▲ 修正箇所 ▲▲▲ ###

    else:
        raise ValueError(f"Unknown problem name: {problem_name}")
    
    sampler = NNMILPSampler(
        problem_name=problem_name, search_space_info=search_space_info,
        constraints=constraints, seed=settings['seed'], sampler_settings=settings['sampler_settings']
    )
    
    study = optuna.create_study(
        study_name=settings['name'], storage=settings['storage'],
        sampler=sampler, direction="minimize", load_if_exists=True
    )
    study.optimize(objective_wrapper, n_trials=settings['n_trials'])
    
    logging.info(f"\n{'='*15} Optimization Complete for '{problem_name}' {'='*15}")
    logging.info(f"Best score: {study.best_value:.4f}")
    logging.info(f"Best params: {study.best_params}")

    if settings.get("plot_save_dir"):
        try:
            fig = optuna.visualization.plot_optimization_history(study)
            plot_path = os.path.join(settings["plot_save_dir"], f"{settings['name']}_history.png")
            fig.write_image(plot_path)
            logging.info(f"Optimization history plot saved to {plot_path}")
        except Exception as e:
            logging.error(f"Failed to generate or save plot. Is 'plotly' installed? Error: {e}")

# ==========================================================================
# メイン実行ブロック (Main Execution Block is unchanged)
# ==========================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run BO with NN+MILP Sampler.")
    parser.add_argument("--function", type=str, required=True, choices=['ackley','diabetes','pressure','warcraft'])
    parser.add_argument("--timestamp", type=str, required=True)
    parser.add_argument("--base_dir", type=str, default="results")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--n_trials", type=int, default=100)
    parser.add_argument("--constrain", action="store_true")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--hidden_dim", type=int, default=16)
    parser.add_argument("--n_startup_trials", type=int, default=20)
    parser.add_argument("--n_bins", type=int, default=10)
    parser.add_argument("--map_option", type=int, default=1, choices=[1,2,3])
    parser.add_argument("--time_limit_sec", type=int, default=None)
    args = parser.parse_args()
    results_dir = os.path.join(args.base_dir, args.timestamp)
    plot_dir = os.path.join(results_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)
    constraint_str = "con" if args.constrain else "uncon"
    log_filename_base = f"{args.function}_{constraint_str}_seed{args.seed}"
    exp_name = f"{args.timestamp}_{log_filename_base}"
    set_logger(log_filename_base, results_dir)
    storage_path = os.path.join(results_dir, f"{log_filename_base}.db")
    storage_url = f"sqlite:///{storage_path}"
    settings = {
        "problem_name": args.function, "name": exp_name, "seed": args.seed,
        "n_trials": args.n_trials, "constrain": args.constrain, "storage": storage_url,
        "plot_save_dir": plot_dir,
        "sampler_settings": {
            "epochs": args.epochs, "hidden_dim": args.hidden_dim,
            "n_startup_trials": args.n_startup_trials,
            "time_limit_sec": args.time_limit_sec,
        }
    }
    if args.function == 'pressure': settings['n_bins'] = args.n_bins
    if args.function == 'warcraft': settings['map_option'] = args.map_option
    logging.info(f"Starting experiment with settings: {settings}")
    try:
        run_bo(settings)
        logging.info("Experiment finished successfully.")
    except Exception as e:
        logging.error(f"An error occurred during the experiment: {e}", exc_info=True)