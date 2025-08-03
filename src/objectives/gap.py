import numpy as np
import optuna
from functools import partial
import itertools 


class GAP_A_Objective:
    def __init__(self):
        self.n_items = 9
        self.n_bins = 3
        self.features = [f'item_{i}' for i in range(self.n_items)]
        self.item_weights = [1] * self.n_items
        self.bin_capacities = [2, 3, 4]
        
        self.assignment_values = np.load('data/gap_a.npz')['assignment']
        
        self.penalty = 0.0
        self._tensor_constraint = self.create_tensor_constraint()
        
    def __call__(self, x_assignment):
        """目的関数"""
        bin_loads = [0] * self.n_bins
        for item_idx, bin_idx in enumerate(x_assignment):
            bin_loads[bin_idx] += self.item_weights[item_idx]
        
        if bin_loads != self.bin_capacities:
            return self.penalty

        total_value = np.sum([self.assignment_values[i, x_assignment[i]] for i in range(self.n_items)])
        return -total_value

    def create_tensor_constraint(self):
        shape = tuple([self.n_bins] * self.n_items)
        tensor_constraint = np.zeros(shape, dtype=np.int8)

        all_assignments = itertools.product(range(self.n_bins), repeat=self.n_items)

        for assignment_tuple in all_assignments:
            bin_loads = [0] * self.n_bins
            for item_idx, bin_idx in enumerate(assignment_tuple):
                bin_loads[bin_idx] += self.item_weights[item_idx]
            
            if bin_loads == self.bin_capacities:
                tensor_constraint[assignment_tuple] = 1
        
        print(f"Constraint tensor created. Feasible points: {np.sum(tensor_constraint)} / {tensor_constraint.size}")
        return tensor_constraint


class GAP_B_Objective:
    def __init__(self, is_constrained=False):
        is_constrained = is_constrained 
        self.n_items = 7
        self.n_bins = 4
        self.features = [f'item_{i}' for i in range(self.n_items)]
        
        self.assignment_values = np.load('data/gap_b.npz')['assignment']

        self.penalty = 0.0
        
        self._tensor_constraint = self.create_tensor_constraint()

        print("\n--- 問題2-B: 論理制約のあるGAP (最小化問題) ---")

    def __call__(self, x_assignment):
        is_rule1_valid = list(x_assignment).count(0) <= 1
        is_rule2_valid = list(x_assignment).count(1) <= 1
        
        if not (is_rule1_valid and is_rule2_valid):
            return self.penalty

        total_value = np.sum([self.assignment_values[i, x_assignment[i]] for i in range(self.n_items)])
        return -total_value

    def create_tensor_constraint(self):
        print("Creating tensor constraint for GAP-2B...")
        shape = tuple([self.n_bins] * self.n_items)
        tensor_constraint = np.zeros(shape, dtype=np.int8)

        all_assignments = itertools.product(range(self.n_bins), repeat=self.n_items)

        for assignment_tuple in all_assignments:
            is_rule1_valid = list(assignment_tuple).count(0) <= 1
            is_rule2_valid = list(assignment_tuple).count(1) <= 1
            
            if is_rule1_valid and is_rule2_valid:
                tensor_constraint[assignment_tuple] = 1
        
        print(f"Constraint tensor created. Feasible points: {np.sum(tensor_constraint)} / {tensor_constraint.size}")
        return tensor_constraint

# # ===================================================================
# # Optuna用の目的関数ラッパー
# # ===================================================================
# def objective(trial, problem_instance=None):
#     x_list = [trial.suggest_categorical(name, list(range(problem_instance.n_bins))) for name in problem_instance.features]
#     x = np.array(x_list)
#     return problem_instance(x)


# if __name__ == "__main__":
#     # テスト用のサンプルコード
#     objective = GAP_A_Objective()
    
#     print("\n--- 問題2-A: 容量制約のあるGAP (最小化問題) ---")
#     tensor_constraint = objective._tensor_constraint
#     print("制約を満たす点の数:", np.sum(tensor_constraint))
#     print("制約に違反する点の数:", np.sum(~tensor_constraint))
#     print("充足率:", np.sum(tensor_constraint) / tensor_constraint.size)
#     print(tensor_constraint.shape)

#     print()
#     objective = GAP_B_Objective()
    
#     print("\n--- 問題2-B: 論理制約のあるGAP (最小化問題) ---")
#     tensor_constraint = objective._tensor_constraint
#     print("制約を満たす点の数:", np.sum(tensor_constraint))
#     print("制約に違反する点の数:", np.sum(~tensor_constraint))
#     print("充足率:", np.sum(tensor_constraint) / tensor_constraint.size)
#     print(tensor_constraint.shape)


def run_full_search(problem_instance):
    """
    与えられた問題インスタンスに対して全探索を実行し、結果を表示する
    """
    # 問題のパラメータを取得
    n_items = problem_instance.n_items
    n_bins = problem_instance.n_bins
    assignment_values = problem_instance.assignment_values
    problem_name = problem_instance.__class__.__name__
    
    # 結果を格納する変数を初期化
    min_constrained_value = float('inf')
    best_constrained_assignment = None
    min_unconstrained_value = float('inf')
    best_unconstrained_assignment = None

    # 全組み合わせの数を計算
    n_combinations = n_bins ** n_items
    print(f"\n--- [{problem_name}] 全探索を開始します ({n_combinations:,} 通り) ---")

    # 全ての割り当てパターンを生成
    all_assignments = itertools.product(range(n_bins), repeat=n_items)

    for i, assignment_tuple in enumerate(all_assignments):
        assignment = np.array(assignment_tuple)
        
        # 1. 制約ありの最小値探索
        constrained_value = problem_instance(assignment)
        # 目的関数は制約違反で0、有効解で負の値を返すため、より小さい値を探す
        if constrained_value < min_constrained_value:
            min_constrained_value = constrained_value
            best_constrained_assignment = assignment
            
        # 2. 制約なしの最小値探索
        unconstrained_total_value = np.sum([assignment_values[item_idx, bin_idx] for item_idx, bin_idx in enumerate(assignment)])
        unconstrained_objective_value = -unconstrained_total_value
        if unconstrained_objective_value < min_unconstrained_value:
            min_unconstrained_value = unconstrained_objective_value
            best_unconstrained_assignment = assignment
    
    print(f"--- [{problem_name}] 全探索完了 ---")
    
    # --- 結果の表示 ---
    print("\n" + "="*50)
    print(f"       ✨ 全探索結果: {problem_name} ✨")
    print("="*50)
    
    # 制約ありの結果
    print("\n## 📦 制約あり (Constrained)")
    if best_constrained_assignment is not None:
        print(f"最小目的関数値: {min_constrained_value:.2f} (元の価値: {-min_constrained_value:.2f})")
        print(f"最適な割り当て: {best_constrained_assignment}")
        # 検算
        if problem_name == 'GAP_A_Objective':
            final_loads = [list(best_constrained_assignment).count(b) for b in range(n_bins)]
            print(f"  ➡️ 検算: ビン占有量 = {final_loads} (目標: {problem_instance.bin_capacities})")
        elif problem_name == 'GAP_B_Objective':
            count_0 = list(best_constrained_assignment).count(0)
            count_1 = list(best_constrained_assignment).count(1)
            print(f"  ➡️ 検算: ビン0の数={count_0} (<=1), ビン1の数={count_1} (<=1)")
    else:
        print("制約を満たす解が見つかりませんでした。")

    # 制約なしの結果
    print("\n## 🌐 制約なし (Unconstrained)")
    if best_unconstrained_assignment is not None:
        print(f"最小目的関数値: {min_unconstrained_value:.2f} (元の価値: {-min_unconstrained_value:.2f})")
        print(f"最適な割り当て: {best_unconstrained_assignment}")
    else:
        print("解が見つかりませんでした。")
    print("\n" + "="*50 + "\n")


# --- メインの実行ブロック ---

if __name__ == "__main__":
    # 問題Aの全探索を実行
    objective_a = GAP_A_Objective()
    run_full_search(objective_a)
    
    # 問題Bの全探索を実行
    objective_b = GAP_B_Objective()
    run_full_search(objective_b)
