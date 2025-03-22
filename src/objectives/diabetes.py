import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier


class DiabetesObjective:
    """
    糖尿病データセットの目的関数クラス
    
    属性:
    -----
    data_path : str
        糖尿病データセットのパス
    constraint_path : str
        制約テンソルのパス
    distribution_path : str
        分布テンソルのパス
    """
    data_path = 'data/diabetes.csv'
    constraint_path = 'data/diabetes_constraint_count_zero.npz'
    distribution_path = 'data/diabetes_analysis_tensors.npz'

    def __init__(self, start_point=None, is_constrained=False, seed=42):
        """
        Diabetesオブジェクトの初期化
        
        Parameters:
        -----------
        start_point : array-like, optional
            開始点（指定がない場合はランダムに選択）
        """

        self.is_constrained = is_constrained
        self.seed = seed
        np.random.seed(seed)

        # データの読み込み
        self.df = pd.read_csv(self.data_path)
        self.df_features = self.df.drop(columns='Outcome')
        self.df_target = self.df[['Outcome']]

        # 制約テンソルの読み込み
        loaded_constraint = np.load(self.constraint_path)
        self._tensor_constraint = loaded_constraint['tensor']
        self.features = [f.decode('utf-8') if isinstance(f, bytes) else f for f in loaded_constraint['features']]
        
        # 分布テンソルと予測テンソルの読み込み
        loaded_distribution = np.load(self.distribution_path)
        self._tensor_positive = loaded_distribution['positive_tensor']
        self._tensor_negative = loaded_distribution['negative_tensor']
        self._tensor_predicted = loaded_distribution['predicted_tensor']
        
        # positive_tensorからnanでないインデックスを取得
        self._positive_indices = []
        it = np.nditer(self._tensor_positive, flags=['multi_index'])
        while not it.finished:
            idx = it.multi_index
            if not np.isnan(self._tensor_positive[idx]):
                self._positive_indices.append(idx)
            it.iternext()
        
        # 開始点x'を設定
        if start_point is None and self._positive_indices:
            # ランダムに陽性例から選択
            self._x_start = np.array(self._positive_indices[np.random.randint(len(self._positive_indices))])
        elif start_point is not None:
            # 指定された開始点を使用
            self._x_start = np.array(start_point)
        else:
            raise ValueError("No positive samples found in the tensor and no start_point provided")
        
        print(f"開始点 x': {self._x_start}")
        print(f"開始点の予測値: {self._tensor_predicted[tuple(self._x_start)]:.4f}")
        
        # Random Forestモデルの初期化と学習（オプション）
        self._train_model()
    
    def sample_positive_indices(self, n_samples=1):
        """
        陽性例のインデックスをサンプリングする
        
        Parameters:
        -----------
        n_samples : int, optional
            サンプリングする陽性例の数
        
        Returns:
        --------
        numpy.ndarray
            サンプリングされた陽性例のインデックス
        """
        if n_samples <= len(self._positive_indices):
            sampled = np.random.choice(len(self._positive_indices), n_samples, replace=False)
            return np.array([self._positive_indices[i] for i in sampled])
        else:
            raise ValueError("Number of samples exceeds the number of positive indices")
        
    def sample_violation_indices(self, n_samples=1):
        """
        制約違反のインデックスをサンプリングする
        
        Parameters:
        -----------
        n_samples : int, optional
            サンプリングする制約違反の数
        
        Returns:
        --------
        numpy.ndarray
            サンプリングされた制約違反のインデックス
        """
        if n_samples <= np.sum(self._tensor_constraint == 0):
            violation_indices = np.argwhere(self._tensor_constraint == 0)
            sampled = np.random.choice(len(violation_indices), n_samples, replace=False)
            return np.array([violation_indices[i] for i in sampled])
        else:
            raise ValueError("Number of samples exceeds the number of constraint violations")
    
    def _train_model(self):
        """RandomForestモデルを学習する（必要に応じて）"""
        # 特徴量をビン分割してモデルを学習
        X_binned = np.zeros((len(self.df), len(self.features)))
        
        for i, feature in enumerate(self.features):
            X_binned[:, i] = pd.cut(self.df_features[feature], bins=5, labels=False)
            
        y = self.df_target['Outcome'].values
        
        # モデルの学習 - 常に同じものを使用するため，random_stateを42に固定している
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model.fit(X_binned, y)
    
    def __call__(self, x):
        """
        目的関数: f(x) + L2_norm(x - x') / max_distance
        
        Parameters:
        -----------
        x : numpy.ndarray
            8次元の整数ベクトル（各特徴量のビン値）
        
        Returns:
        --------
        float
            目的関数の値（低いほど良い）
        """
        # 制約チェック - 無効な組み合わせは高いペナルティを返す
        if self.is_constrained and self._tensor_constraint[tuple(x)] == 0:
            return 1 + 1
        
        # f(x)を計算: predicted_tensorから予測値を取得
        f_x = self._tensor_predicted[tuple(x)]
        
        # sqrt(x - x')^2を計算: 開始点とユークリッド距離
        max_distance = np.linalg.norm(np.array(np.ones_like(x) * 4))
        distance_penalty = np.linalg.norm(x - self._x_start) / max_distance # 0-1の範囲に正規化
        
        # 目的関数: 予測値(高いほど糖尿病リスクが高い) + 距離ペナルティ
        objective_value = f_x + distance_penalty
        
        return objective_value

def diabetes_objective(trial, diabetes_instance):
    """
    Optuna用の目的関数ラッパー
    
    Parameters:
    -----------
    trial : optuna.Trial
        Optunaのトライアルオブジェクト
    diabetes_instance : Diabetes
        Diabetesクラスのインスタンス
    
    Returns:
    --------
    float
        最適化する目的関数の値
    """
    _base = diabetes_instance
    categories = _base.features
    x = np.array([trial.suggest_int(f"x_{category}", 0, 4) for category in categories])
    return _base(x)

if __name__ == "__main__":
    import optuna
    from functools import partial
    
    # Diabetesインスタンスの作成
    obj = DiabetesObjective()
    objective_with_args = partial(diabetes_objective, diabetes_instance=obj)

    print(obj.sample_positive_indices(5))

    
    # Optunaによる最適化
    study = optuna.create_study(direction="minimize")
    study.optimize(objective_with_args, n_trials=5)
    
    # 最適値の表示
    best_params = study.best_params
    best_x = np.array([best_params[f"x_{feature}"] for feature in obj.features])
    
    print("\n最適化結果:")
    print(f"開始点: {obj._x_start}")
    print(f"最適点: {best_x}")
    print(f"最適点の予測値: {obj._tensor_predicted[tuple(best_x)]:.4f}")
    print(f"開始点からの変化: {best_x - obj._x_start}")
    print(f"目的関数値: {study.best_value:.4f}")

    # optuna.visualization.plot_optimization_history(study).show()