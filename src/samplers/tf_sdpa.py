import logging
import random
import itertools
import copy
from typing import Literal, Optional, List, Tuple, Any, Dict, Union

import numpy as np
import optuna
from optuna.samplers import BaseSampler
from optuna.study import StudyDirection
from optuna.trial import TrialState
import sympy as sp
from ncpol2sdpa import generate_variables

from ..tensor_factorization_sdpa import TensorTrain

# モジュールレベルでロガーを設定
_logger = logging.getLogger("TFSdpaSampler")
if not _logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    _logger.addHandler(handler)
    _logger.setLevel(logging.INFO)


class TFSdpaSampler(BaseSampler):
    def __init__(
        self,
        seed: Optional[int] = None,
        acquisition_function: Literal["ucb", "ei", "ts"] = "ucb",  # API互換用
        sampler_params: dict = {},
        tf_params: dict = {},
        acqf_params: dict = {},  # API互換用
        tensor_constraint: Optional[np.ndarray] = None,
    ):
        """
        テンソル分解を用いたSDPベースのサンプラー
        
        Parameters:
        -----------
        seed : int, optional
            乱数シード
        acquisition_function : str, optional
            API互換用（実際には使用されません）
        sampler_params : dict, optional
            mask_ratio : float
                未観測点からフィッティングに含める点の比率 [0-1]
            n_startup_trials : int
                ランダムサンプリングを使用する最初のN回の試行
            include_observed_points : bool
                観測済みの点を候補に含めるかどうか
            unique_sampling : bool
                再評価を避けるためのフラグ
        tf_params : dict, optional
            rank : int
                テンソル分解のランク
            sdp_level : int
                SDPの緩和レベル
            target_value : float
                テンソルフィッティングのターゲット値
        tensor_constraint : np.ndarray, optional
            探索空間の制約を表すバイナリマスク（1=有効、0=無効）
        """
        # 乱数シード
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        random.seed(seed)

        # 必須パラメータ
        self.acquisition_function = acquisition_function  # API互換用
        self.independent_sampler = optuna.samplers.RandomSampler(seed=seed)

        # サンプラーパラメータ
        self.n_startup_trials = sampler_params.get("n_startup_trials", 1)
        self.mask_ratio = sampler_params.get("mask_ratio", 0.1)  # 未観測点からフィッティングに含める比率
        self.include_observed_points = sampler_params.get("include_observed_points", False)
        self.unique_sampling = sampler_params.get("unique_sampling", False)

        # テンソルファクターのパラメータ
        self.tt_rank = tf_params.get("rank", 2)
        self.sdp_level = tf_params.get("sdp_level", 2)
        self.target_value = tf_params.get("target_value", 0.5)  # 制約の値

        # 内部変数
        self._param_names = None
        self._category_maps = {}
        self._shape = None
        self._tensor_eval = None
        self._tensor_eval_bool = None
        self._tensor_constraint = tensor_constraint  # 探索空間の制約（バイナリマスク）
        self._maximize = None
        self._evaluated_indices = []
        
        # 途中経過データ
        self._last_reconstructed_tensor = None
        self._last_mask_tensor = None
        
        # モジュールレベルのロガーを使用
        self._logger = _logger

    def __deepcopy__(self, memo):
        """deepcopy実装 - ロガーの複製を防ぐ"""
        obj = self.__class__.__new__(self.__class__)
        memo[id(self)] = obj
        
        for key, value in self.__dict__.items():
            if key == '_logger':
                # ロガーはコピーせず参照を維持
                obj.__dict__[key] = value
            else:
                # その他の属性は通常通りコピー
                obj.__dict__[key] = copy.deepcopy(value, memo)
        
        return obj

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
            self._logger.info(f"Using independent sampler (startup phase: {len(trials)}/{self.n_startup_trials})")
            return {}

        if self._param_names is None:
            self._initialize_internal_structure(search_space, study)

        # 過去の試行からテンソルを構築
        self._update_tensor(study)

        # テンソル分解を実行し、次のポイントを提案
        try:
            reconstructed_tensor = self._fit_and_reconstruct(
                self._tensor_eval,
                self._tensor_eval_bool
            )
            self._last_reconstructed_tensor = reconstructed_tensor
        except Exception as e:
            self._logger.error(f"Error in tensor decomposition: {e}")
            self._logger.info("Using independent sampler due to decomposition error")
            return {}

        # 再構成されたテンソルから最良の点を見つける
        next_index = self._suggest_next_point(reconstructed_tensor)

        # インデックスをパラメータ値に変換
        params = {}
        for i, param_name in enumerate(self._param_names):
            category_index = next_index[i]
            category = self._category_maps[param_name][category_index]
            params[param_name] = category
            
        self._logger.info(f"Suggesting next point: {params}")
        return params

    def sample_independent(self, study, trial, param_name, param_distribution):
        self._logger.debug(f"Using sample_independent for {param_name}")
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
        self._logger.info(f"Initialized internal structure with shape {self._shape}, direction: {'maximize' if self._maximize else 'minimize'}")

    def _update_tensor(self, study):
        trials = study.get_trials(deepcopy=False)
        n_added = 0
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
                    n_added += 1
        
        if n_added > 0:
            self._logger.info(f"Added {n_added} new evaluations to tensor")

    def _calculate_eval_stats(self, tensor_eval: np.ndarray) -> Tuple[float, float]:
        """評価テンソルの統計情報を計算"""
        eval_copy = np.copy(tensor_eval)
        # 制約がある場合はフィルタリング
        if self._tensor_constraint is not None:
            # 制約テンソルが0の位置はNaNに設定
            constraint_mask = self._tensor_constraint == 0
            eval_copy[constraint_mask] = np.nan

        finite_values = eval_copy[np.isfinite(eval_copy)]
        if len(finite_values) == 0:
            return 0.0, 1.0
            
        mean_ = np.nanmean(finite_values)
        std_ = np.nanstd(finite_values)
        if std_ == 0:
            std_ = 1.0

        return mean_, std_

    def _create_mask_tensor(self, tensor_eval_bool: np.ndarray) -> np.ndarray:
        """
        フィッティング用のマスクテンソルを作成
        
        マスクテンソルの値が1の位置は、テンソル分解のフィッティングに使用される点
        (観測済みの点 + ランダムに選ばれた未観測の点の一部)
        """
        # 観測済みの点をすべてマスクに含める
        mask_tensor = tensor_eval_bool.copy().astype(int)
        
        # 追加のランダムマスクポイントを追加 (未観測の点からmask_ratio分を選択)
        if self.mask_ratio > 0:
            # 未観測点の座標を取得
            unobserved = np.where(~tensor_eval_bool)
            unobserved_count = len(unobserved[0])
            
            if unobserved_count > 0:
                # マスク比率に基づいて未観測点からランダムに選択
                additional_mask_count = int(unobserved_count * self.mask_ratio)
                if additional_mask_count > 0:
                    indices = self.rng.choice(unobserved_count, additional_mask_count, replace=False)
                    for i in indices:
                        idx = tuple(dim[i] for dim in unobserved)
                        mask_tensor[idx] = 1
        
        # 制約がある場合は、制約に基づいてフィルタリング
        if self._tensor_constraint is not None:
            # 制約テンソルが0の場所はマスクからも除外
            mask_tensor = mask_tensor & self._tensor_constraint
        
        n_masked = np.sum(mask_tensor)
        total_points = mask_tensor.size
        valid_points = np.sum(self._tensor_constraint) if self._tensor_constraint is not None else total_points
        
        self._logger.debug(f"Created mask with {n_masked}/{valid_points} points (total: {total_points})")
        
        # マスクを保存（デバッグ用）
        self._last_mask_tensor = mask_tensor
        
        return mask_tensor

    def _fit_and_reconstruct(
        self,
        tensor_eval: np.ndarray,
        tensor_eval_bool: np.ndarray
    ) -> np.ndarray:
        """
        TensorTrainを使用してテンソル分解を実行し、再構成したテンソルを返す
        """
        self._logger.info("Starting tensor decomposition with TensorTrain")
        eval_mean, eval_std = self._calculate_eval_stats(tensor_eval)
        self._logger.debug(f"Tensor stats: mean={eval_mean}, std={eval_std}")

        # 最適化のためにテンソルを正規化
        normalized_tensor = np.copy(tensor_eval)
        tensor_min = np.nanmin(tensor_eval) if not np.isnan(tensor_eval).all() else 0
        tensor_max = np.nanmax(tensor_eval) if not np.isnan(tensor_eval).all() else 1
        
        if tensor_min == tensor_max:
            normalized_tensor = np.full_like(tensor_eval, 0.5)
        else:
            # [0, 1]にスケール
            normalized_tensor = (tensor_eval - tensor_min) / (tensor_max - tensor_min)
        
        normalized_tensor[np.isnan(normalized_tensor)] = 0.5  # 不明な値は中点を使用
        normalized_tensor = np.array(normalized_tensor, dtype=object)  # SymPyと互換性をもたせる

        # フィッティング用のマスクを作成 (観測点と一部の未観測点を含む)
        mask_tensor = self._create_mask_tensor(tensor_eval_bool)
        
        # 制約テンソル - 探索空間の有効な部分を示す (1=有効、0=無効)
        constraint_tensor = self._tensor_constraint
        if constraint_tensor is None:
            # 制約がない場合は全領域が有効
            constraint_tensor = np.ones(self._shape, dtype=int)

        # TensorTrainのランク設定
        dimensions = self._shape
        ranks = [1]
        for _ in range(len(dimensions)-1):
            ranks.append(self.tt_rank)
        ranks.append(1)

        # 変数の数を計算
        num_variables = sum([ranks[i] * dimensions[i] * ranks[i+1] for i in range(len(dimensions))])
        self._logger.debug(f"Using TT-ranks: {ranks}, total variables: {num_variables}")

        # SDP用の変数を生成
        x = generate_variables('x', num_variables, commutative=True)
        
        # TensorTrainインスタンスを作成
        tt = TensorTrain(ranks, dimensions, variables=x)
        
        # SDPを使用して解く
        self._logger.info(f"Solving SDP with level {self.sdp_level}")
        sdp, reconstructed_tensor = tt.solve_sdp(
            original_tensor=normalized_tensor,
            mask_tensor=mask_tensor,
            constraint_tensor=constraint_tensor,
            target_value=self.target_value,
            level=self.sdp_level
        )
        
        if reconstructed_tensor is None:
            # 解が見つからない場合はゼロを返す
            self._logger.warning("SDP solver returned None, using zero tensor")
            reconstructed_tensor = np.zeros(dimensions)
        else:
            self._logger.info(f"SDP solved successfully with objective value: {sdp.primal if hasattr(sdp, 'primal') else 'N/A'}")
        
        # 制約外の領域を除外
        if constraint_tensor is not None:
            # 制約が0の場所は最低/最高値に設定して選ばれないようにする
            invalid_points = constraint_tensor == 0
            reconstructed_tensor[invalid_points] = float('-inf') if self._maximize else float('inf')
        
        # 観測済みの値を再構成テンソルにマーク (再評価を避ける)
        if not self.include_observed_points:
            n_observed = np.sum(tensor_eval_bool)
            self._logger.debug(f"Excluding {n_observed} observed points from suggestion")
            for idx in zip(*np.where(tensor_eval_bool)):
                reconstructed_tensor[idx] = float('-inf') if self._maximize else float('inf')
            
        return reconstructed_tensor

    def _suggest_next_point(self, reconstructed_tensor: np.ndarray) -> Tuple:
        """最適化の方向に基づいて再構成テンソルの最良の点を見つける"""
        if self._maximize:
            flat_idx = np.argmax(reconstructed_tensor)
            best_value = np.max(reconstructed_tensor)
        else:
            flat_idx = np.argmin(reconstructed_tensor)
            best_value = np.min(reconstructed_tensor)
            
        next_index = np.unravel_index(flat_idx, reconstructed_tensor.shape)
        self._logger.debug(f"Selected point at index {next_index} with predicted value {best_value}")
        return next_index

    def get_reconstructed_tensor(self) -> Optional[np.ndarray]:
        """最後に再構成したテンソルを取得（デバッグ用）"""
        return self._last_reconstructed_tensor
    
    def get_mask_tensor(self) -> Optional[np.ndarray]:
        """最後に使用したマスクテンソルを取得（デバッグ用）"""
        return self._last_mask_tensor
