import logging
import random
from typing import Literal, Optional

import numpy as np
import optuna
import torch
from optuna.samplers import BaseSampler
from optuna.study import StudyDirection
from optuna.trial import TrialState
from scipy.stats import norm
from scipy.stats import t


from ..tensor_factorization_continual import TensorFactorization


class TFContinualSampler(BaseSampler):
    def __init__(
        self,
        seed: Optional[int] = None,
        method: Literal["cp", "tucker", "train", "ring"] = "cp",
        acquisition_function: Literal["ucb", "ei", "ts"] = "ucb",
        sampler_params: dict = {},
        tf_params: dict = {},
        acqf_params: dict = {},
        torch_device: Optional[torch.device] = None,
        torch_dtype: Optional[torch.dtype] = None,
        tensor_constraint=None,
        acqf_dist: Literal["n", "t1", "t2"] = "n",
    ):
        # Random seed
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        random.seed(seed)
        torch.manual_seed(seed)

        # Device and dtype
        if torch_device is None:
            torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.torch_device = torch_device
        if torch_dtype is None:
            torch_dtype = torch.float64
        self.torch_dtype = torch_dtype

        # Essential parameters
        self.method = method
        self.acquisition_function = acquisition_function
        self.independent_sampler = optuna.samplers.RandomSampler(seed=seed)

        # Sampler parameters
        self.n_startup_trials = sampler_params.get("n_startup_trials", 1)
        self.decomp_iter_num = sampler_params.get("decomp_iter_num", 10) if self.acquisition_function != "ts" else 1
        self.mask_ratio = sampler_params.get("mask_ratio", 0.9)
        self.include_observed_points = sampler_params.get("include_observed_points", False)
        self.unique_sampling = sampler_params.get("unique_sampling", False)

        # Acquisition function parameters
        self.trade_off_param = acqf_params.get("trade_off_param", 1.0)
        self.batch_size = acqf_params.get("batch_size", 1)  # Fixed to 1
        self.acqf_dist = acqf_dist

        # TF optim parameters
        self.rank = tf_params.get("rank", 3)
        self.lr = tf_params.get("lr", 0.01)
        self.max_iter = tf_params.get("max_iter", None)
        self.tol = tf_params.get("tol", 1e-6)
        self.reg_lambda = tf_params.get("reg_lambda", 1e-3)
        self.constraint_lambda = tf_params.get("constraint_lambda", 1.0)

        # Internal storage
        self._param_names = None
        self._category_maps = None
        self._shape = None
        self._tensor_eval = None
        self._tensor_eval_bool = None
        self._tensor_constraint = tensor_constraint
        self._maximize = None
        self._model_states = [None for _ in range(self.decomp_iter_num)]

        # Debugging
        self.mean_tensor = None
        self.std_tensor = None
        self.save_dir = None

        # Loss tracking
        self.loss_history = {
            "trial": [],
            "tf_index": [],
            "epoch": [],
            "total": [],
            "mse": [],
            "constraint": [],
            "l2": [],
        }

    def infer_relative_search_space(self, study, trial):
        search_space = optuna.search_space.intersection_search_space(
            study.get_trials(deepcopy=False)
        )
        relevant_search_space = {}
        for name, distribution in search_space.items():
            if isinstance(
                distribution,
                (
                    optuna.distributions.IntDistribution,
                    optuna.distributions.CategoricalDistribution,
                ),
            ):
                relevant_search_space[name] = distribution
        return relevant_search_space

    def sample_relative(self, study, trial, search_space):
        if not search_space:
            return {}
        
        states = (TrialState.COMPLETE,)
        trials = study._get_trials(deepcopy=False, states=states, use_cache=True)
        
        if len(trials) < self.n_startup_trials:
            return {}

        if self._param_names is None:
            self._initialize_internal_structure(search_space, study)

        # Build tensor from past trials
        self._update_tensor(study)

        # Perform decomposition and suggest next parameter set
        mean_tensor, std_tensor = self._fit(
            self._tensor_eval,
            self._tensor_eval_bool
        )

        # Update Loss History
        prev_len = len(self.loss_history["trial"])
        current_len = len(self.loss_history["epoch"])
        self.loss_history["trial"].extend([trial.number] * (current_len - prev_len))

        # Suggest next indices based on the selected acquisition function
        if self.acquisition_function == "ucb":
            next_indices = self._suggest_ucb_candidates(
                mean_tensor=mean_tensor,
                std_tensor=std_tensor,
                trade_off_param=self.trade_off_param,
                batch_size=self.batch_size,
                maximize=self._maximize,
            )
        elif self.acquisition_function == "ei":
            next_indices = self._suggest_ei_candidates(
                mean_tensor=mean_tensor,
                std_tensor=std_tensor,
                batch_size=self.batch_size,
                maximize=self._maximize,
            )
        elif self.acquisition_function == "ts":
            next_indices = self._suggest_ts_candidates(
                mean_tensor=mean_tensor,
                std_tensor=std_tensor,
                batch_size=self.batch_size,
                maximize=self._maximize
            )
        else:
            raise ValueError("acquisition_function must be 'ucb', 'ei', or 'ts'.")

        next_index = next_indices[0]

        # Convert indices back to parameter values
        params = {}
        for i, param_name in enumerate(self._param_names):
            category_index = next_index[i]
            category = self._category_maps[param_name][category_index]
            params[param_name] = category

        return params

    def sample_independent(self, study, trial, param_name, param_distribution):
        logging.info(f"Using sample_independent for sampling with {self.independent_sampler} sampler.")
        return self.independent_sampler.sample_independent(study, trial, param_name, param_distribution)

    def _initialize_internal_structure(self, search_space, study):
        self._param_names = sorted(search_space.keys())
        self._category_maps = {}
        self._shape = []
        for param_name in self._param_names:
            distribution = search_space[param_name]
            if isinstance(distribution, optuna.distributions.CategoricalDistribution):
                categories = distribution.choices
            elif isinstance(distribution, optuna.distributions.IntDistribution):
                categories = list(
                    range(distribution.low, distribution.high + 1, distribution.step)
                )
            else:
                continue
            self._category_maps[param_name] = categories
            self._shape.append(len(categories))
        self._shape = tuple(self._shape)
        self._tensor_eval = np.full(self._shape, np.nan)
        self._tensor_eval_bool = np.zeros(self._shape, dtype=bool)
        self._evaluated_indices = []
        self._maximize = study.direction == StudyDirection.MAXIMIZE

    def _update_tensor(self, study):
        trials = study.get_trials(deepcopy=False)
        for trial in trials:
            if trial.state != optuna.trial.TrialState.COMPLETE:
                continue
            index = []
            for param_name in self._param_names:
                if param_name not in trial.params:
                    break
                category = trial.params[param_name]
                try:
                    category_index = self._category_maps[param_name].index(category)
                except ValueError:
                    break
                index.append(category_index)
            else:
                index = tuple(index)
                if index not in self._evaluated_indices:
                    value = trial.value
                    self._tensor_eval[index] = value
                    self._tensor_eval_bool[index] = True
                    self._evaluated_indices.append(index)

    def _fit(
        self,
        tensor_eval: np.ndarray,
        tensor_eval_bool: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        The main routine that repeatedly decomposes the tensor (decomp_iter_num times) and
        aggregates the results into mean_tensor and std_tensor.
        """
        eval_mean, eval_std = self._calculate_eval_stats(tensor_eval)

        random_tensor = np.random.uniform(0, 1, size=tensor_eval.shape)

        # We'll build up multiple reconstructions:
        tensors_list = []

        for tf_index in range(self.decomp_iter_num):

            # Perform the decomposition for iteration i
            decomposed_tensor = self._decompose_with_optional_mask(
                tensor_eval=tensor_eval,
                tensor_eval_bool=tensor_eval_bool,
                eval_mean=eval_mean,
                eval_std=eval_std,
                maximize=self._maximize,
                tf_index=tf_index,
                random_tensor=random_tensor,
            )

            tensors_list.append(decomposed_tensor)

        # After collecting all reconstructions, compute final mean/std
        return self._calculate_mean_std_tensors(
            tensors_list,
            tensor_eval,
            tensor_eval_bool
        )

    def _calculate_eval_stats(
        self, tensor_eval: np.ndarray
    ) -> tuple[float, float]:
        eval_copy = np.copy(tensor_eval)
        # Filter with constraint if available
        if self._tensor_constraint is not None:
            eval_copy[self._tensor_constraint == 0] = np.nan

        finite_values = eval_copy[np.isfinite(eval_copy)]
        mean_ = np.nanmean(finite_values)
        std_ = np.nanstd(finite_values)

        return (mean_, std_)

    def _select_mask_indices(
        self, tensor_shape: tuple, tensor_eval_bool: np.ndarray
    ) -> np.ndarray:
        # If we have a tensor_constraint, only consider entries where constraint == 1
        if self._tensor_constraint is not None:
            constrained_indices = np.argwhere(self._tensor_constraint == 1)

            if self.include_observed_points:
                cand_indices = np.indices(tensor_shape).reshape(len(tensor_shape), -1).T
            else:
                cand_indices = np.argwhere(tensor_eval_bool == False)

            # Intersect the constrained indices with the candidate indices
            constrained_indices_set = set(map(tuple, constrained_indices))
            cand_indices = np.array([
                idx for idx in cand_indices if tuple(idx) in constrained_indices_set
            ])
        else:
            if self.include_observed_points:
                cand_indices = np.indices(tensor_shape).reshape(len(tensor_shape), -1).T
            else:
                cand_indices = np.argwhere(tensor_eval_bool == False)

        # Determine the mask size
        # mask_size = max(1, int(len(cand_indices) * self.mask_ratio))
        mask_size = max(0, int(len(cand_indices) * self.mask_ratio))

        # Select mask indices randomly
        selected_indices = self.rng.choice(len(cand_indices), mask_size, replace=False)
        return cand_indices[selected_indices]

    def _create_mask_tensor(
        self, tensor_shape: tuple, mask_indices: np.ndarray
    ) -> np.ndarray:
        mask_tensor = np.ones(tensor_shape, dtype=bool)
        for mask_index in mask_indices:
            mask_tensor[tuple(mask_index)] = False
        return mask_tensor

    def _decompose_with_optional_mask(
        self,
        tensor_eval: np.ndarray,
        tensor_eval_bool: np.ndarray,
        eval_mean: float,
        eval_std: float,
        maximize: bool,
        tf_index: int = None,
        random_tensor = None,
    ) -> np.ndarray:
        """
        Create a masked tensor (if needed), initialize or load from prev_state,
        do factorization, return the reconstructed tensor as a NumPy array.

        If you want to store factor parameters for continual learning, you can:
          - Modify TensorFactorization to return its factors in e.g. a get_state() method,
          - Return (reconstructed_tensor, factor_params) here,
          - Save factor_params to self._model_states[tf_index].
        """
        # # Standardize
        # standardized_tensor_eval = (tensor_eval - eval_mean) / (eval_std + 1e-8)
        # standardized_tensor_eval[~tensor_eval_bool] = np.nan  # unobserved

        # Calculate nanmin and nanmax once and store them in variables
        tensor_min = np.nanmin(tensor_eval)
        tensor_max = np.nanmax(tensor_eval)

        # Normalize tensor_eval with a fallback to 0.5 when all values are the same
        if tensor_min == tensor_max:
            normalized_tensor_eval = np.full_like(tensor_eval, 0.5)
            threshold = 0.5
        else:
            normalized_tensor_eval = (tensor_eval - tensor_min) / (tensor_max - tensor_min)
            
            if maximize:
                threshold = 0
            else:
                threshold = 1
        
        normalized_tensor_eval[~tensor_eval_bool] = np.nan  # unobserved

        # Create mask if needed
        if self.mask_ratio != 0:
            mask_indices = self._select_mask_indices(tensor_eval.shape, tensor_eval_bool)
            mask_tensor = self._create_mask_tensor(tensor_eval.shape, mask_indices)
        else:
            mask_tensor = self._create_mask_tensor(tensor_eval.shape, [])

        mask_tensor = np.logical_or(mask_tensor, self._tensor_constraint == 0)

        # Initialize values
        # init_tensor_eval = self.rng.normal(0, 1, tensor_eval.shape)
        init_tensor_eval = self.rng.uniform(0, 1, tensor_eval.shape)

        if self._tensor_constraint is not None:
            condition = np.logical_and(tensor_eval_bool, self._tensor_constraint)
        else:
            condition = tensor_eval_bool

        # init_tensor_eval[condition] = standardized_tensor_eval[condition]
        init_tensor_eval[condition] = normalized_tensor_eval[condition]

        # Convert to Torch
        constraint = None
        if self._tensor_constraint is not None:
            constraint = torch.tensor(self._tensor_constraint, dtype=self.torch_dtype)
        
        prev_state = self._model_states[tf_index]

        tf = TensorFactorization(
            tensor=torch.tensor(init_tensor_eval, dtype=self.torch_dtype),
            rank=self.rank,
            method=self.method,
            mask=torch.tensor(mask_tensor, dtype=self.torch_dtype),
            constraint=constraint,
            is_maximize_c=maximize,
            device=self.torch_device,
            prev_state=prev_state,  # pass the previously saved factors
        )

        tf.optimize(
            lr=self.lr,
            max_iter=self.max_iter,
            tol=self.tol,
            mse_tol=1e-2,
            const_tol=1e-2,
            reg_lambda=self.reg_lambda,
            constraint_lambda=self.constraint_lambda,
            thr=threshold
        )

        _epoch = tf.loss_history["epoch"]
        _total = tf.loss_history["total"]
        _mse = tf.loss_history["mse"]
        _constraint = tf.loss_history["constraint"]
        _l2 = tf.loss_history["l2"]
        
        self.loss_history["tf_index"].extend([tf_index] * len(_epoch))
        self.loss_history["epoch"].extend(_epoch)
        self.loss_history["total"].extend(_total)
        self.loss_history["mse"].extend(_mse)
        self.loss_history["constraint"].extend(_constraint)
        self.loss_history["l2"].extend(_l2)
     
        if self.method == "tucker":
            self._model_states[tf_index] = (tf.core, tf.factors)
        else:
            self._model_states[tf_index] = tf.factors

        reconstructed_tensor = tf.reconstruct()
        return reconstructed_tensor.detach().cpu().numpy()

    def _calculate_mean_std_tensors(
        self,
        tensors_list: list[np.ndarray],
        tensor_eval: np.ndarray,
        tensor_eval_bool: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Combine all decomposed tensors in tensors_list into mean/std arrays,
        then re-insert known values for observed points, and handle constraints.
        """
        tensors_stack = np.stack(tensors_list)
        mean_tensor = np.mean(tensors_stack, axis=0)
        std_tensor = np.std(tensors_stack, axis=0)

        mean_tensor[tensor_eval_bool] = tensor_eval[tensor_eval_bool]
        std_tensor[tensor_eval_bool] = 1e-8  # small value for observed points

        # Handle constraints (if any)
        if self._tensor_constraint is not None:
            
            std_tensor[self._tensor_constraint == 0] = 1e-8

        # Save for debugging
        self.mean_tensor = mean_tensor
        self.std_tensor = std_tensor

        return mean_tensor, std_tensor

    def _suggest_ucb_candidates(
        self,
        mean_tensor: np.ndarray,
        std_tensor: np.ndarray,
        trade_off_param: float,
        batch_size: int,
        maximize: bool,
    ) -> list[tuple[int, ...]]:
        def _ucb(mean_tensor, std_tensor, trade_off_param, maximize=True) -> np.ndarray:
            if maximize:
                return mean_tensor + trade_off_param * std_tensor
            else:
                return -mean_tensor + trade_off_param * std_tensor

        ucb_values = _ucb(mean_tensor, std_tensor, trade_off_param, maximize)

        if self.unique_sampling:
            ucb_values[self._tensor_eval_bool == True] = -np.inf if maximize else np.inf

        # Get indices of top UCB values
        flat_indices = np.argsort(ucb_values.flatten())[::-1]  # descending
        top_indices = np.unravel_index(flat_indices[:batch_size], ucb_values.shape)
        top_indices = list(zip(*top_indices))
        return top_indices

    def _suggest_ei_candidates(
        self,
        mean_tensor: np.ndarray,
        std_tensor: np.ndarray,
        batch_size: int,
        maximize: bool,
    ) -> list[tuple[int, ...]]:

        def _ei(mean_tensor, std_tensor, f_best, maximize=True) -> np.ndarray:
            std_tensor = np.clip(std_tensor, 1e-9, None)
            if maximize:
                z = (mean_tensor - f_best) / std_tensor
            else:
                z = (f_best - mean_tensor) / std_tensor

            if self.acqf_dist == "n":
                ei_values = std_tensor * (z * norm.cdf(z) + norm.pdf(z))
            elif self.acqf_dist == "t1":
                n = np.sum(self._tensor_eval_bool)
                n = n if n > 1 else 2
                ei_values = std_tensor * (z * t.cdf(z, df=n-1) + t.pdf(z, df=n-1))
            elif self.acqf_dist == "t2":
                n = 10 # decomp iter num
                ei_values = std_tensor * (z * t.cdf(z, df=n-1) + t.pdf(z, df=n-1))
            else:
                raise ValueError("acqf_dist must be 'n', 't1', or 't2'.")

            return ei_values

        tensor_min = np.nanmin(self._tensor_eval)
        tensor_max = np.nanmax(self._tensor_eval)

        if tensor_min == tensor_max:
            f_best = 0.5
        else:
            if maximize:
                f_best = 1
            else:
                f_best = 0

        ei_values = _ei(mean_tensor, std_tensor, f_best, maximize)

        if self.unique_sampling:
            ei_values[self._tensor_eval_bool == True] = -np.inf if maximize else np.inf

        flat_indices = np.argsort(ei_values.flatten())[::-1]  # descending
        top_indices = np.unravel_index(flat_indices[:batch_size], ei_values.shape)
        top_indices = list(zip(*top_indices))
        return top_indices

    def _suggest_ts_candidates(
        self,
        mean_tensor: np.ndarray,
        std_tensor: np.ndarray,
        batch_size: int,
        maximize: bool,
    ) -> list[tuple[int, ...]]:
        """
        Thompson Sampling approach can be more involved, but here is a naive version
        using mean + std (for maximize) or -(mean - std) (for minimize).
        """
        def _ts(mean_tensor, std_tensor, maximize=True) -> np.ndarray:
            if maximize:
                return mean_tensor + std_tensor
            else:
                return -mean_tensor + std_tensor

        ts_values = _ts(mean_tensor, std_tensor, maximize)

        if self.unique_sampling:
            ts_values[self._tensor_eval_bool == True] = -np.inf if maximize else np.inf

        flat_indices = np.argsort(ts_values.flatten())[::-1]  # descending
        top_indices = np.unravel_index(flat_indices[:batch_size], ts_values.shape)
        top_indices = list(zip(*top_indices))
        return top_indices