import numpy as np
import optuna
from functools import partial
import itertools 


# ===================================================================
# 問題2-A: 容量制約のあるGAP
# ===================================================================
class GAP_A_Objective:
    """
    問題2-A（9アイテム, 3箱, 容量制約）の目的関数クラス
    """
    def __init__(self):
        self.n_items = 9
        self.n_bins = 3
        self.features = [f'item_{i}' for i in range(self.n_items)]
        self.item_weights = [1] * self.n_items
        self.bin_capacities = [2, 3, 4]
        
        self.assignment_values = np.array([
            [0.417022, 0.720324, 0.000114], [0.302333, 0.146756, 0.092339],
            [0.186260, 0.345561, 0.396767], [0.538817, 0.419195, 0.685220],
            [0.204452, 0.878117, 0.027388], [0.670468, 0.417305, 0.558690],
            [0.140387, 0.198101, 0.800745], [0.968262, 0.313424, 0.692323],
            [0.876389, 0.894607, 0.085044]
        ])
        
        self.penalty = 0.0
        self._tensor_constraint = self.create_tensor_constraint()
        
        print("--- 問題2-A: 容量制約のあるGAP (最小化問題) ---")

    def __call__(self, x_assignment):
        """目的関数"""
        bin_loads = [0] * self.n_bins
        for item_idx, bin_idx in enumerate(x_assignment):
            bin_loads[bin_idx] += self.item_weights[item_idx]
        
        if bin_loads != self.bin_capacities:
            return self.penalty

        total_value = np.sum([self.assignment_values[i, x_assignment[i]] for i in range(self.n_items)])
        return -total_value

    # 【新規追加】制約テンソルを生成するメソッド
    def create_tensor_constraint(self):
        """
        全探索を行い、制約を満たす割り当ての位置が1、それ以外が0のテンソルを作成する。
        """
        print("Creating tensor constraint for GAP-2A...")
        # 探索空間の形状は (箱の数, 箱の数, ..., 箱の数) となる (アイテムの数だけ次元がある)
        shape = tuple([self.n_bins] * self.n_items)
        tensor_constraint = np.zeros(shape, dtype=np.int8)

        # 全ての可能な割り当てパターンを生成 (3^9 = 19,683通り)
        all_assignments = itertools.product(range(self.n_bins), repeat=self.n_items)

        for assignment_tuple in all_assignments:
            # 制約チェック
            bin_loads = [0] * self.n_bins
            for item_idx, bin_idx in enumerate(assignment_tuple):
                bin_loads[bin_idx] += self.item_weights[item_idx]
            
            # 制約を満たす場合、テンソルの対応する位置を1にする
            if bin_loads == self.bin_capacities:
                tensor_constraint[assignment_tuple] = 1
        
        print(f"Constraint tensor created. Feasible points: {np.sum(tensor_constraint)} / {tensor_constraint.size}")
        return tensor_constraint

# ===================================================================
# 問題2-B: 論理制約のあるGAP
# ===================================================================
class GAP_B_Objective:
    """
    問題2-B（7アイテム, 4箱, 論理制約）の目的関数クラス
    """
    def __init__(self, is_constrained=False):
        is_constrained = is_constrained # 既にあるコードとの互換性を保つための引数
        self.n_items = 7
        self.n_bins = 4
        self.features = [f'item_{i}' for i in range(self.n_items)]
        
        self.assignment_values = np.array([
            [0.417022, 0.720324, 0.000114, 0.302333], [0.146756, 0.092339, 0.186260, 0.345561],
            [0.396767, 0.538817, 0.419195, 0.685220], [0.204452, 0.878117, 0.027388, 0.670468],
            [0.417305, 0.558690, 0.140387, 0.198101], [0.800745, 0.968262, 0.313424, 0.692323],
            [0.876389, 0.894607, 0.085044, 0.039055]
        ])

        self.penalty = 0.0
        
        self._tensor_constraint = self.create_tensor_constraint()

        print("\n--- 問題2-B: 論理制約のあるGAP (最小化問題) ---")

    def __call__(self, x_assignment):
        """目的関数"""
        is_rule1_valid = list(x_assignment).count(0) <= 1
        is_rule2_valid = list(x_assignment).count(1) <= 1
        
        if not (is_rule1_valid and is_rule2_valid):
            return self.penalty

        total_value = np.sum([self.assignment_values[i, x_assignment[i]] for i in range(self.n_items)])
        return -total_value

    # 【新規追加】制約テンソルを生成するメソッド
    def create_tensor_constraint(self):
        """
        全探索を行い、制約を満たす割り当ての位置が1、それ以外が0のテンソルを作成する。
        """
        print("Creating tensor constraint for GAP-2B...")
        shape = tuple([self.n_bins] * self.n_items)
        tensor_constraint = np.zeros(shape, dtype=np.int8)

        # 全ての可能な割り当てパターンを生成 (4^7 = 16,384通り)
        all_assignments = itertools.product(range(self.n_bins), repeat=self.n_items)

        for assignment_tuple in all_assignments:
            # 制約チェック
            is_rule1_valid = list(assignment_tuple).count(0) <= 1
            is_rule2_valid = list(assignment_tuple).count(1) <= 1
            
            if is_rule1_valid and is_rule2_valid:
                tensor_constraint[assignment_tuple] = 1
        
        print(f"Constraint tensor created. Feasible points: {np.sum(tensor_constraint)} / {tensor_constraint.size}")
        return tensor_constraint

# ===================================================================
# Optuna用の目的関数ラッパー
# ===================================================================
def objective(trial, problem_instance=None):
    x_list = [trial.suggest_categorical(name, list(range(problem_instance.n_bins))) for name in problem_instance.features]
    x = np.array(x_list)
    return problem_instance(x)