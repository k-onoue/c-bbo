import numpy as np
import optuna
from functools import partial
import itertools

# ===================================================================
# 問題1-A: 制約付きIsingモデル (Optuna版)
# ===================================================================
class Ising_A_Objective:
    """
    問題1-A（14アイテム, 2制約）の目的関数クラス
    """
    def __init__(self):
        self.n_items = 14
        self.group_a_inds = list(range(7))
        self.group_b_inds = list(range(7, 14))
        self.features = [f'x_{i}' for i in range(self.n_items)]
        
        self.ising_potentials = np.array([
            [ 1.76405235,  0.40015721,  0.97873798,  2.2408932,   1.86755799, -0.97727788,
            0.95008842, -0.15135721, -0.10321885,  0.4105985,   0.14404357,  1.45427351,
            0.76103773,  0.12167502],
            [ 0.44386323,  0.33367433,  1.49407907, -0.20515826,  0.3130677,  -0.85409574,
            -2.55298982,  0.6536186,   0.8644362,  -0.74216502,  2.26975462, -1.45436567,
            0.04575852, -0.18718385],
            [ 1.53277921,  1.46935877,  0.15494743,  0.37816252, -0.88778575, -1.98079647,
            -0.34791215,  0.15634897,  1.23029068,  1.20237985, -0.38732682, -0.30230275,
            -1.04855297, -1.42001794],
            [-1.70627019,  1.9507754,  -0.50965218, -0.4380743,  -1.25279536,  0.77749036,
            -1.61389785, -0.21274028, -0.89546656,  0.3869025,  -0.51080514, -1.18063218,
            -0.02818223,  0.42833187],
            [ 0.06651722,  0.3024719,  -0.63432209, -0.36274117, -0.67246045, -0.35955316,
            -0.81314628, -1.7262826,   0.17742614, -0.40178094, -1.63019835,  0.46278226,
            -0.90729836,  0.0519454 ],
            [ 0.72909056,  0.12898291,  1.13940068, -1.23482582,  0.40234164, -0.68481009,
            -0.87079715, -0.57884966, -0.31155253,  0.05616534, -1.16514984,  0.90082649,
            0.46566244, -1.53624369],
            [ 1.48825219,  1.89588918,  1.17877957, -0.17992484, -1.07075262,  1.05445173,
            -0.40317695,  1.22244507,  0.20827498,  0.97663904,  0.3563664,   0.70657317,
            0.01050002,  1.78587049],
            [ 0.12691209,  0.40198936,  1.8831507,  -1.34775906, -1.270485,    0.96939671,
            -1.17312341,  1.94362119, -0.41361898, -0.74745481,  1.92294203,  1.48051479,
            1.86755896,  0.90604466],
            [-0.86122569,  1.91006495, -0.26800337,  0.8024564,   0.94725197, -0.15501009,
            0.61407937,  0.92220667,  0.37642553, -1.09940079,  0.29823817,  1.3263859,
            -0.69456786, -0.14963454],
            [-0.43515355,  1.84926373,  0.67229476,  0.40746184, -0.76991607,  0.53924919,
            -0.67433266,  0.03183056, -0.63584608,  0.67643329,  0.57659082, -0.20829876,
            0.39600671, -1.09306151],
            [-1.49125759,  0.4393917,   0.1666735,   0.63503144,  2.38314477,  0.94447949,
            -0.91282223,  1.11701629, -1.31590741, -0.4615846,  -0.06824161,  1.71334272,
            -0.74475482, -0.82643854],
            [-0.09845252, -0.66347829,  1.12663592, -1.07993151, -1.14746865, -0.43782004,
            -0.49803245,  1.92953205,  0.94942081,  0.08755124, -1.22543552,  0.84436298,
            -1.00021535, -1.5447711 ],
            [ 1.18802979,  0.31694261,  0.92085882,  0.31872765,  0.85683061, -0.65102559,
            -1.03424284,  0.68159452, -0.80340966, -0.68954978, -0.4555325,   0.01747916,
            -0.35399391, -1.37495129],
            [-0.6436184,  -2.22340315,  0.62523145, -1.60205766, -1.10438334,  0.05216508,
            -0.739563,    1.5430146,  -1.29285691,  0.26705087, -0.03928282, -1.1680935,
            0.52327666, -0.17154633]
        ])
        
        # 【変更点】ペナルティ値を設定（実行可能なエネルギーの最大値より十分に大きい値）
        self.penalty = 15 

        self._tensor_constraint = self.create_tensor_constraint()
        
        print("--- 問題1-A: 制約付きIsingモデル (最小化問題) ---")
        print(f"アイテム数: {self.n_items}")
        print("制約1: グループAとBの選択数が等しい")
        print("制約2: 全体の選択数がちょうど4個")

    def __call__(self, x_assignment):
        """
        目的関数: Isingエネルギーを計算（最小化が目的）
        """
        # --- 制約チェック ---
        count_a = sum(x_assignment[i] for i in self.group_a_inds)
        count_b = sum(x_assignment[i] for i in self.group_b_inds)
        total_count = sum(x_assignment)
        
        is_rule1_valid = (count_a == count_b)
        is_rule2_valid = (total_count == 4)

        if not (is_rule1_valid and is_rule2_valid):
            return self.penalty

        # --- エネルギー計算 (実行可能な解のみ) ---
        energy = 0
        for i in range(self.n_items):
            for j in range(i + 1, self.n_items):
                # 選択されている(x=1)アイテム間の相互作用のみを合計
                if x_assignment[i] == 1 and x_assignment[j] == 1:
                    energy += self.ising_potentials[i, j]
        return energy

    def create_tensor_constraint(self):
        """
        全探索を行い、制約を満たす選択パターンの位置が1のテンソルを作成する。
        """
        print("Creating tensor constraint for Ising-1B...")
        shape = tuple([2] * self.n_items)
        tensor_constraint = np.zeros(shape, dtype=np.int8)

        # 全ての可能な選択パターンを生成 (2^15 = 32,768通り)
        all_assignments = itertools.product([0, 1], repeat=self.n_items)

        for assignment_tuple in all_assignments:
            # 制約チェック
            count_a = sum(assignment_tuple[i] for i in self.group_a_inds)
            count_b = sum(assignment_tuple[i] for i in self.group_b_inds)
            count_c = sum(assignment_tuple[i] for i in self.group_c_inds)

            is_rule1_valid = (count_a == count_b)
            is_rule2_valid = (count_c == 1)
            
            if is_rule1_valid and is_rule2_valid:
                tensor_constraint[assignment_tuple] = 1

        print(f"Constraint tensor created. Feasible points: {np.sum(tensor_constraint)} / {tensor_constraint.size}")
        return tensor_constraint

# ===================================================================
# 問題1-B: 制約付きIsingモデル (Optuna版)
# ===================================================================
class Ising_B_Objective:
    """
    問題1-B（15アイテム, 2制約）の目的関数クラス
    """
    def __init__(self):
        self.n_items = 15
        self.group_a_inds = list(range(5))
        self.group_b_inds = list(range(5, 10))
        self.group_c_inds = list(range(10, 15))
        self.features = [f'x_{i}' for i in range(self.n_items)]
        
        self.ising_potentials = np.array([
            [ 1.76405235,  0.40015721,  0.97873798,  2.2408932,   1.86755799, -0.97727788,
            0.95008842, -0.15135721, -0.10321885,  0.4105985,   0.14404357,  1.45427351,
            0.76103773,  0.12167502,  0.44386323],
            [ 0.33367433,  1.49407907, -0.20515826,  0.3130677,  -0.85409574, -2.55298982,
            0.6536186,   0.8644362,  -0.74216502,  2.26975462, -1.45436567,  0.04575852,
            -0.18718385,  1.53277921,  1.46935877],
            [ 0.15494743,  0.37816252, -0.88778575, -1.98079647, -0.34791215,  0.15634897,
            1.23029068,  1.20237985, -0.38732682, -0.30230275, -1.04855297, -1.42001794,
            -1.70627019,  1.9507754,  -0.50965218],
            [-0.4380743,  -1.25279536,  0.77749036, -1.61389785, -0.21274028, -0.89546656,
            0.3869025,  -0.51080514, -1.18063218, -0.02818223,  0.42833187,  0.06651722,
            0.3024719,  -0.63432209, -0.36274117],
            [-0.67246045, -0.35955316, -0.81314628, -1.7262826,   0.17742614, -0.40178094,
            -1.63019835,  0.46278226, -0.90729836,  0.0519454,   0.72909056,  0.12898291,
            1.13940068, -1.23482582,  0.40234164],
            [-0.68481009, -0.87079715, -0.57884966, -0.31155253,  0.05616534, -1.16514984,
            0.90082649,  0.46566244, -1.53624369,  1.48825219,  1.89588918,  1.17877957,
            -0.17992484, -1.07075262,  1.05445173],
            [-0.40317695,  1.22244507,  0.20827498,  0.97663904,  0.3563664,   0.70657317,
            0.01050002,  1.78587049,  0.12691209,  0.40198936,  1.8831507,  -1.34775906,
            -1.270485,    0.96939671, -1.17312341],
            [ 1.94362119, -0.41361898, -0.74745481,  1.92294203,  1.48051479,  1.86755896,
            0.90604466, -0.86122569,  1.91006495, -0.26800337,  0.8024564,   0.94725197,
            -0.15501009,  0.61407937,  0.92220667],
            [ 0.37642553, -1.09940079,  0.29823817,  1.3263859,  -0.69456786, -0.14963454,
            -0.43515355,  1.84926373,  0.67229476,  0.40746184, -0.76991607,  0.53924919,
            -0.67433266,  0.03183056, -0.63584608],
            [ 0.67643329,  0.57659082, -0.20829876,  0.39600671, -1.09306151, -1.49125759,
            0.4393917,   0.1666735,   0.63503144,  2.38314477,  0.94447949, -0.91282223,
            1.11701629, -1.31590741, -0.4615846 ],
            [-0.06824161,  1.71334272, -0.74475482, -0.82643854, -0.09845252, -0.66347829,
            1.12663592, -1.07993151, -1.14746865, -0.43782004, -0.49803245,  1.92953205,
            0.94942081,  0.08755124, -1.22543552],
            [ 0.84436298, -1.00021535, -1.5447711,   1.18802979,  0.31694261,  0.92085882,
            0.31872765,  0.85683061, -0.65102559, -1.03424284,  0.68159452, -0.80340966,
            -0.68954978, -0.4555325,   0.01747916],
            [-0.35399391, -1.37495129, -0.6436184,  -2.22340315,  0.62523145, -1.60205766,
            -1.10438334,  0.05216508, -0.739563,    1.5430146,  -1.29285691,  0.26705087,
            -0.03928282, -1.1680935,   0.52327666],
            [-0.17154633,  0.77179055,  0.82350415,  2.16323595,  1.33652795, -0.36918184,
            -0.23937918,  1.0996596,   0.65526373,  0.64013153, -1.61695604, -0.02432612,
            -0.73803091,  0.2799246,  -0.09815039],
            [ 0.91017891,  0.31721822,  0.78632796, -0.4664191,  -0.94444626, -0.41004969,
            -0.01702041,  0.37915174,  2.25930895, -0.04225715, -0.955945,   -0.34598178,
            -0.46359597,  0.48148147, -1.54079701]
        ])
        
        # 【変更点】ペナルティ値を設定
        self.penalty = 20

        self._tensor_constraint = self.create_tensor_constraint()

        print("\n--- 問題1-B: 制約付きIsingモデル (最小化問題) ---")
        print(f"アイテム数: {self.n_items}")
        print("制約1: グループAとBの選択数が等しい")
        print("制約2: グループCの選択数がちょうど1個")
        
    def __call__(self, x_assignment):
        """
        目的関数: Isingエネルギーを計算（最小化が目的）
        """
        # --- 制約チェック ---
        count_a = sum(x_assignment[i] for i in self.group_a_inds)
        count_b = sum(x_assignment[i] for i in self.group_b_inds)
        count_c = sum(x_assignment[i] for i in self.group_c_inds)

        is_rule1_valid = (count_a == count_b)
        is_rule2_valid = (count_c == 1)

        if not (is_rule1_valid and is_rule2_valid):
            return self.penalty

        # --- エネルギー計算 (実行可能な解のみ) ---
        energy = 0
        for i in range(self.n_items):
            for j in range(i + 1, self.n_items):
                if x_assignment[i] == 1 and x_assignment[j] == 1:
                    energy += self.ising_potentials[i, j]
        return energy

    def create_tensor_constraint(self):
        """
        全探索を行い、制約を満たす選択パターンの位置が1のテンソルを作成する。
        """
        print("Creating tensor constraint for Ising-1B...")
        shape = tuple([2] * self.n_items)
        tensor_constraint = np.zeros(shape, dtype=np.int8)

        # 全ての可能な選択パターンを生成 (2^15 = 32,768通り)
        all_assignments = itertools.product([0, 1], repeat=self.n_items)

        for assignment_tuple in all_assignments:
            # 制約チェック
            count_a = sum(assignment_tuple[i] for i in self.group_a_inds)
            count_b = sum(assignment_tuple[i] for i in self.group_b_inds)
            count_c = sum(assignment_tuple[i] for i in self.group_c_inds)

            is_rule1_valid = (count_a == count_b)
            is_rule2_valid = (count_c == 1)
            
            if is_rule1_valid and is_rule2_valid:
                tensor_constraint[assignment_tuple] = 1

        print(f"Constraint tensor created. Feasible points: {np.sum(tensor_constraint)} / {tensor_constraint.size}")
        return tensor_constraint

# ===================================================================
# Optuna用の統一目的関数ラッパー
# ===================================================================
def objective(trial, function_name=None, problem_instance=None):
    if function_name in ["ising_1a", "ising_1b"]:
        # 各アイテムを選択するか(1)しないか(0)をOptunaに提案させる
        x_list = [trial.suggest_categorical(name, [0, 1]) for name in problem_instance.features]
        x = np.array(x_list)
        return problem_instance(x)

# ===================================================================
# メインの実行部分
# ===================================================================
if __name__ == "__main__":
    
    # --- 問題1-Aの最適化 ---
    problem_1a = Ising_A_Objective()
    objective_1a = partial(objective, function_name="ising_1a", problem_instance=problem_1a)
    study_1a = optuna.create_study(direction="minimize")
    # 全探索空間(16,384)に近い試行回数で実行
    study_1a.optimize(objective_1a, n_trials=8000)
    
    print("\n" + "="*20 + " 問題1-A 最適化結果 " + "="*20)
    print(f"目的関数の最小エネルギー: {study_1a.best_value:.4f}")
    # best_paramsは辞書なので、キーでソートして値を取得し、選択されたアイテムのインデックスを表示
    best_assignment_1a = [i for i, val in sorted(study_1a.best_params.items(), key=lambda item: int(item[0].split('_')[1])) if val == 1]
    print(f"最適な選択（インデックス）: {best_assignment_1a}")


    # --- 問題1-Bの最適化 ---
    problem_1b = Ising_B_Objective()
    objective_1b = partial(objective, function_name="ising_1b", problem_instance=problem_1b)
    study_1b = optuna.create_study(direction="minimize")
    # 全探索空間(32,768)に近い試行回数で実行
    study_1b.optimize(objective_1b, n_trials=10000)

    print("\n" + "="*20 + " 問題1-B 最適化結果 " + "="*20)
    print(f"目的関数の最小エネルギー: {study_1b.best_value:.4f}")
    best_assignment_1b = [i for i, val in sorted(study_1b.best_params.items(), key=lambda item: int(item[0].split('_')[1])) if val == 1]
    print(f"最適な選択（インデックス）: {best_assignment_1b}")