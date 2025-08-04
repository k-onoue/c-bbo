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
        if seed is not None:
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

        self._update_tensor(study)

        mean_tensor, std_tensor = self._fit(
            self._tensor_eval,
            self._tensor_eval_bool
        )

        # # Logging for debugging (operates on the [0, 1] normalized scale)
        # mean_max = np.max(mean_tensor)
        # mean_max_idx = tuple(int(i) for i in np.unravel_index(np.argmax(mean_tensor), mean_tensor.shape))
        # mean_min = np.min(mean_tensor)
        # mean_min_idx = tuple(int(i) for i in np.unravel_index(np.argmin(mean_tensor), mean_tensor.shape))
        # std_max = np.max(std_tensor)
        # std_idx = tuple(int(i) for i in np.unravel_index(np.argmax(std_tensor), std_tensor.shape))

        # logging.info(f"Normalized mean max: {mean_max:.4f}, index: {mean_max_idx}, min: {mean_min:.4f}, index: {mean_min_idx}")
        # logging.info(f"Normalized std max: {std_max:.4f}, index: {std_idx}")


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

        eval_mean, eval_std = self._calculate_eval_stats(tensor_eval)
        tensors_list = []

        for tf_index in range(self.decomp_iter_num):
            decomposed_tensor = self._decompose_with_optional_mask(
                tensor_eval=tensor_eval,
                tensor_eval_bool=tensor_eval_bool,
                eval_mean=eval_mean,
                eval_std=eval_std,
                maximize=self._maximize,
                tf_index=tf_index,
            )
            tensors_list.append(decomposed_tensor)

        return self._calculate_mean_std_tensors(
            tensors_list,
            tensor_eval,
            tensor_eval_bool
        )

    def _calculate_eval_stats(
        self, tensor_eval: np.ndarray
    ) -> tuple[float, float]:
        """
        [修正済み]
        観測された全ての有限な値から統計量を計算する。
        これにより、初期に実行不可能な点のみが観測されても学習が破綻しない。
        """
        finite_values = tensor_eval[np.isfinite(tensor_eval)]

        if len(finite_values) == 0:
            return 0.0, 1.0  # 観測値がなければデフォルト値を返す

        mean_ = np.nanmean(finite_values)
        std_ = np.nanstd(finite_values)

        if std_ < 1e-8:
            std_ = 1.0  # 観測値が全て同じ場合、標準偏差を1にしてゼロ除算を防ぐ

        return (mean_, std_)

    def _select_mask_indices(
        self, tensor_shape: tuple, tensor_eval_bool: np.ndarray
    ) -> np.ndarray:
        # If we have a tensor_constraint, only consider entries where constraint == 1
        if self._tensor_constraint is not None:
            # マスク候補は、制約を満たす点のみ
            constrained_indices = np.argwhere(self._tensor_constraint == 1)

            if self.include_observed_points:
                cand_indices_base = np.indices(tensor_shape).reshape(len(tensor_shape), -1).T
            else:
                cand_indices_base = np.argwhere(tensor_eval_bool == False)

            # 候補点と制約を満たす点の積集合を取る
            constrained_indices_set = set(map(tuple, constrained_indices))
            cand_indices = np.array([
                idx for idx in cand_indices_base if tuple(idx) in constrained_indices_set
            ])
        else:
            if self.include_observed_points:
                cand_indices = np.indices(tensor_shape).reshape(len(tensor_shape), -1).T
            else:
                cand_indices = np.argwhere(tensor_eval_bool == False)

        if len(cand_indices) == 0:
            return np.array([], dtype=int)

        mask_size = max(0, int(len(cand_indices) * self.mask_ratio))
        selected_indices_pos = self.rng.choice(len(cand_indices), mask_size, replace=False)
        return cand_indices[selected_indices_pos]

    def _create_mask_tensor(
        self, tensor_shape: tuple, mask_indices: np.ndarray
    ) -> np.ndarray:
        mask_tensor = np.ones(tensor_shape, dtype=bool)
        if mask_indices.size > 0:
            # NumPyの高度なインデックス付けを使用して高速化
            mask_indices_tuple = tuple(mask_indices.T)
            mask_tensor[mask_indices_tuple] = False
        return mask_tensor

    def _decompose_with_optional_mask(
        self,
        tensor_eval: np.ndarray,
        tensor_eval_bool: np.ndarray,
        eval_mean: float,
        eval_std: float,
        maximize: bool,
        tf_index: int = None,
    ) -> np.ndarray:
        
        tensor_min = np.nanmin(tensor_eval)
        tensor_max = np.nanmax(tensor_eval)

        if tensor_min == tensor_max or np.isnan(tensor_min):
            normalized_tensor_eval = np.full_like(tensor_eval, 0.5)
        else:
            normalized_tensor_eval = (tensor_eval - tensor_min) / (tensor_max - tensor_min)
        
        normalized_tensor_eval[~tensor_eval_bool] = np.nan

        # この設計思想の核心部分
        # 最小化問題: threshold=1 (悪い値)
        # 最大化問題: threshold=0 (悪い値)
        # 制約違反の点は、この「悪い値」に近づくように学習される
        threshold = 1.0 if not maximize else 0.0
        if tensor_min == tensor_max:
            threshold = 0.5

        # Create mask for unobserved points within the feasible region
        mask_indices = self._select_mask_indices(tensor_eval.shape, tensor_eval_bool)
        mask_tensor = self._create_mask_tensor(tensor_eval.shape, mask_indices)

        # Intentionally UN-MASK infeasible points.
        # This forces the model to learn their values, guided by the constraint loss.
        if self._tensor_constraint is not None:
            mask_tensor = np.logical_or(mask_tensor, self._tensor_constraint == 0)

        init_tensor_eval = self.rng.uniform(0, 1, tensor_eval.shape)

        # Fill known values
        init_tensor_eval[tensor_eval_bool] = normalized_tensor_eval[tensor_eval_bool]

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
            prev_state=prev_state,
        )

        tf.optimize(
            lr=self.lr,
            max_iter=self.max_iter,
            tol=self.tol,
            mse_tol=1e-2,
            const_tol=1e-1,
            reg_lambda=self.reg_lambda,
            constraint_lambda=self.constraint_lambda,
            thr=threshold,
            severe_conv_control=True,
        )

        # Record loss history
        _epoch = tf.loss_history["epoch"]
        self.loss_history["tf_index"].extend([tf_index] * len(_epoch))
        self.loss_history["epoch"].extend(_epoch)
        self.loss_history["total"].extend(tf.loss_history["total"])
        self.loss_history["mse"].extend(tf.loss_history["mse"])
        self.loss_history["constraint"].extend(tf.loss_history["constraint"])
        self.loss_history["l2"].extend(tf.loss_history["l2"])
     
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
        [修正済み]
        全ての計算を一貫して正規化後の[0, 1]スケールで行う。
        観測済みの点も正規化された値で上書きすることで、スケールの不整合を防ぐ。
        """
        tensors_stack = np.stack(tensors_list)
        mean_tensor = np.mean(tensors_stack, axis=0)
        std_tensor = np.std(tensors_stack, axis=0, ddof=1)

        tensor_min = np.nanmin(tensor_eval)
        tensor_max = np.nanmax(tensor_eval)
        
        if tensor_min != tensor_max and not np.isnan(tensor_min):
            normalized_eval = (tensor_eval - tensor_min) / (tensor_max - tensor_min)
            mean_tensor[tensor_eval_bool] = normalized_eval[tensor_eval_bool]
        else:
            mean_tensor[tensor_eval_bool] = 0.5
        
        std_tensor[tensor_eval_bool] = 1e-8

        if self._tensor_constraint is not None:
            std_tensor[self._tensor_constraint == 0] = 0.0

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
        if maximize:
            ucb_values = mean_tensor + trade_off_param * std_tensor
        else:
            ucb_values = -mean_tensor + trade_off_param * std_tensor

        if self.unique_sampling:
            ucb_values[self._tensor_eval_bool] = -np.inf

        flat_indices = np.argsort(ucb_values.flatten())[::-1]
        top_indices = np.unravel_index(flat_indices[:batch_size], ucb_values.shape)
        return list(zip(*top_indices))

    def _suggest_ei_candidates(
        self,
        mean_tensor: np.ndarray,
        std_tensor: np.ndarray,
        batch_size: int,
        maximize: bool,
    ) -> list[tuple[int, ...]]:
        """
        [修正済み]
        f_best（最良の目的関数値）を正規化後の[0, 1]スケールで定義する。
        """
        std_tensor_clipped = np.clip(std_tensor, 1e-9, None)

        # Define f_best in the normalized [0, 1] scale
        if maximize:
            f_best = 1.0  # Best possible value for maximization is 1
        else:
            f_best = 0.0  # Best possible value for minimization is 0

        # Edge case: if no valid observations yet, or all are the same.
        tensor_min = np.nanmin(self._tensor_eval)
        tensor_max = np.nanmax(self._tensor_eval)
        if np.isnan(tensor_min) or tensor_min == tensor_max:
            f_best = 0.5 # If uncertain, aim for the middle

        if maximize:
            z = (mean_tensor - f_best) / std_tensor_clipped
        else:
            z = (f_best - mean_tensor) / std_tensor_clipped

        # Select distribution for EI calculation
        if self.acqf_dist == "n":
            dist = norm
            ei_values = std_tensor_clipped * (z * dist.cdf(z) + dist.pdf(z))
        else:
            if self.acqf_dist == "t1":
                n = np.sum(self._tensor_eval_bool)
                df = max(1, n - 1)
            elif self.acqf_dist == "t2":
                df = max(1, self.decomp_iter_num - 1)
            else:
                raise ValueError("acqf_dist must be 'n', 't1', or 't2'.")
            dist = t(df=df)
            ei_values = std_tensor_clipped * (z * dist.cdf(z) + dist.pdf(z))

        if self.unique_sampling:
            ei_values[self._tensor_eval_bool] = -np.inf

        flat_indices = np.argsort(ei_values.flatten())[::-1]
        top_indices = np.unravel_index(flat_indices[:batch_size], ei_values.shape)
        return list(zip(*top_indices))

    def _suggest_ts_candidates(
        self,
        mean_tensor: np.ndarray,
        std_tensor: np.ndarray,
        batch_size: int,
        maximize: bool,
    ) -> list[tuple[int, ...]]:
        
        # Sample from the posterior distribution
        if self.acqf_dist == "n":
            sampled_values = self.rng.normal(mean_tensor, std_tensor)
        else: # t-distribution
            if self.acqf_dist == "t1":
                n = np.sum(self._tensor_eval_bool)
                df = max(1, n - 1)
            else: # t2
                df = max(1, self.decomp_iter_num - 1)
            # scale parameter for t-distribution is std
            sampled_values = self.rng.standard_t(df, size=mean_tensor.shape) * std_tensor + mean_tensor

        if self.unique_sampling:
            sampled_values[self._tensor_eval_bool] = -np.inf if maximize else np.inf

        if maximize:
            flat_indices = np.argsort(sampled_values.flatten())[::-1]
        else:
            flat_indices = np.argsort(sampled_values.flatten())

        top_indices = np.unravel_index(flat_indices[:batch_size], sampled_values.shape)
        return list(zip(*top_indices))