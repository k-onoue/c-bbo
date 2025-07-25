import numpy as np
import optuna


class PressureVesselObjective:
    """
    圧力容器設計問題の目的関数クラス
    """
    def __init__(self, start_point=None, seed=42):
        """
        PressureVesselObjectiveオブジェクトの初期化
        
        Parameters:
        -----------
        start_point : array-like, optional
            最適化の開始点（カテゴリのインデックス）
        is_constrained : bool, optional
            制約を考慮するかどうか
        seed : int, optional
            乱数シード
        n_bins : int, optional
            各連続変数を離散化する際のカテゴリ数
        """
        self.is_constrained = True
        self.seed = seed
        np.random.seed(seed)
        
        # 変数名を定義
        self.features = ['Ts', 'Th', 'R', 'L']

        self.valid_best = 12408.3421
        self.valid_worst = 663935.9375
        
        # 1. 各変数の範囲を定義
        # Ts (x1), Th (x2) は 0.0625 の倍数
        # R (x3), L (x4) は連続値
        ts_min, ts_max = 0.0625, 6.1875  # 1*0.0625 to 99*0.0625
        th_min, th_max = 0.0625, 6.1875  # 1*0.0625 to 99*0.0625
        r_min, r_max = 10.0, 200.0
        l_min, l_max = 10.0, 200.0

        n_bins = 10  # 各連続変数を離散化する際のカテゴリ数
        
        # 2. 各変数をカテゴリ化し、代表値（中央値）を計算
        self.mid_points = []
        for min_val, max_val in [(ts_min, ts_max), (th_min, th_max), (r_min, r_max), (l_min, l_max)]:
            # np.linspaceで範囲を n_bins+1 個の点で区切り、n_bins 個の区間を作成
            edges = np.linspace(min_val, max_val, n_bins + 1)
            # 各区間の中央値を代表値とする
            mids = np.array([(edges[i] + edges[i+1]) / 2 for i in range(n_bins)])
            self.mid_points.append(mids)
        
        # 3. 全組み合わせの目的関数値と制約違反を事前に計算
        shape = tuple(len(m) for m in self.mid_points)
        self._tensor_objective = np.empty(shape)
        self._tensor_constraint = np.empty(shape, dtype=bool)

        # ネストループで全組み合わせを評価
        for i, ts in enumerate(self.mid_points[0]):
            for j, th in enumerate(self.mid_points[1]):
                for k, r in enumerate(self.mid_points[2]):
                    for l, l_val in enumerate(self.mid_points[3]):
                        x = np.array([ts, th, r, l_val])
                        self._tensor_objective[i, j, k, l] = self._objective_formula(x)
                        g_vals = self._constraints_formula(x)
                        self._tensor_constraint[i, j, k, l] = np.all(g_vals <= 0)

        # 4. 制約を満たす点／満たさない点のインデックスを保存
        self._feasible_indices = np.argwhere(self._tensor_constraint == True)
        self._infeasible_indices = np.argwhere(self._tensor_constraint == False)

        # 5. 開始点 x' を設定
        if start_point is not None:
            self._x_start = np.array(start_point)
        elif len(self._feasible_indices) > 0:
            # 制約を満たす点からランダムに選択
            rand_idx = np.random.randint(len(self._feasible_indices))
            self._x_start = self._feasible_indices[rand_idx]
        else:
            raise ValueError("実行可能なサンプルが見つかりませんでした。")
        
        print(f"開始点 x': {self._x_start}")
        print(f"開始点の目的関数値: {self._tensor_objective[tuple(self._x_start)]:.4f}")

    def _objective_formula(self, x: np.ndarray) -> float:
        """目的関数の計算式"""
        ts, th, r, l_val = x[0], x[1], x[2], x[3]
        return (0.6224 * ts * r * l_val + 
                1.7781 * th * r**2 + 
                3.1661 * ts**2 * l_val + 
                19.84 * ts**2 * r)

    def _constraints_formula(self, x: np.ndarray) -> np.ndarray:
        """制約 g_i(x) <= 0 の計算式"""
        ts, th, r, l_val = x[0], x[1], x[2], x[3]
        g1 = -ts + 0.0193 * r
        g2 = -th + 0.00954 * r
        g3 = -np.pi * r**2 * l_val - (4.0/3.0) * np.pi * r**3 + 1_296_000
        g4 = l_val - 240.0
        return np.array([g1, g2, g3, g4])

    def sample_feasible_indices(self, n_samples=1):
        """制約を満たす点のインデックスをランダムにサンプリングする"""
        if n_samples > len(self._feasible_indices):
            raise ValueError("要求されたサンプル数が、利用可能な実行可能サンプル数を超えています。")
        indices = np.random.choice(len(self._feasible_indices), n_samples, replace=False)
        return self._feasible_indices[indices]

    def sample_infeasible_indices(self, n_samples=1):
        """制約に違反する点のインデックスをランダムにサンプリングする"""
        if n_samples > len(self._infeasible_indices):
            raise ValueError("要求されたサンプル数が、利用可能な制約違反サンプル数を超えています。")
        indices = np.random.choice(len(self._infeasible_indices), n_samples, replace=False)
        return self._infeasible_indices[indices]
    
    def __call__(self, x: np.ndarray) -> float:
        """
        目的関数: f(x)
        
        Parameters:
        -----------
        x : numpy.ndarray
            4次元の整数ベクトル（各特徴量のカテゴリインデックス）
        
        Returns:
        --------
        float
            目的関数の値（低いほど良い）
        """
        x_tuple = tuple(x)
        # # 制約チェック - is_constrainedがTrueの場合、制約違反には高いペナルティを科す
        # if self.is_constrained and not self._tensor_constraint[x_tuple]:
        #     # ペナルティとして、実行可能領域の最大値よりも大きな値を返す
        #     return np.max(self._tensor_objective[self._tensor_constraint]) 

        if not self._tensor_constraint[x_tuple]:
            # ペナルティとして、実行可能領域の最大値よりも大きな値を返す
            return np.max(self._tensor_objective[self._tensor_constraint]) 
        
        # 事前計算したテンソルから目的関数の値を返す
        return self._tensor_objective[x_tuple]

    def sample_violation_indices(self, num_samples: int) -> np.ndarray:
        if self._tensor_constraint is None:
            raise ValueError("Constraint not initialized")
        
        indices = np.array(np.where(self._tensor_constraint == 0)).T
        if num_samples > len(indices):
            raise ValueError("num_samples is too large")
            
        return indices[np.random.choice(len(indices), size=num_samples, replace=False)]

    def sample_violation_path(self, num_samples: int = 200) -> list[tuple[int, int]]:
        random_indices = self.sample_violation_indices(num_samples)
        return [tuple(self._index_to_coord(idx)) for idx in random_indices]