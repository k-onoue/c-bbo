import numpy as np
import itertools
import logging
import os




# --- ユーザー提供のクラス ---
class GAP_A_Objective:
    """
    一般化割り当て問題（GAP）の目的関数クラス
    - 制約：各ビンの容量が厳密に一致する必要がある
    """
    def __init__(self):
        self.n_items, self.n_bins = 9, 3
        self.param_names = [f'item_{i}' for i in range(self.n_items)]
        # 各アイテムの重さはすべて1
        self.item_weights = [1] * self.n_items
        # ビンの容量 [ビン0, ビン1, ビン2]
        self.bin_capacities = [2, 3, 4] # 合計9アイテム
        # アイテムを各ビンに割り当てた際の価値をファイルからロード
        self.assignment_values = np.load('data/gap_a.npz')['assignment']
        # 制約を満たさない場合のペナルティ（ここでは0）
        self.penalty = 0.0
        # self._tensor_constraint = self._create_tensor_constraint() # 全探索では不要のためコメントアウト
        self.constraints_info = {
            # 'tensor_constraint': {'type': 'tensor', 'tensor': self._tensor_constraint},
            'rule_constraint': {
                'type': 'gap_a_rules',
                'n_items': self.n_items,
                'n_bins': self.n_bins,
                'item_weights': self.item_weights,
                'bin_capacities': self.bin_capacities,
            }
        }
        logging.info("--- Loaded Problem: GAP-A (Capacity Constraints) ---")
        logging.info(f"Bin capacities: {self.bin_capacities}")

    def __call__(self, x_assignment: np.ndarray) -> float:
        """
        与えられた割り当て(x_assignment)に対する目的関数の値を計算する。
        制約（ビン容量）を満たさない場合はペナルティ値を返す。
        """
        bin_loads = [0] * self.n_bins
        for item_idx, bin_idx in enumerate(x_assignment):
            bin_loads[bin_idx] += self.item_weights[item_idx]

        # 制約チェック：計算されたビン占有量が指定の容量と完全に一致するか
        if list(bin_loads) != self.bin_capacities:
            return self.penalty

        # 制約を満たす場合、価値の合計を計算して負の値として返す（最小化問題にするため）
        total_value = np.sum([self.assignment_values[i, x_assignment[i]] for i in range(self.n_items)])
        return -total_value

    # _create_tensor_constraintメソッドは全探索では使用しないため、定義は省略

# --- 全探索の実行 ---
if __name__ == "__main__":
    # 目的関数クラスをインスタンス化
    objective = GAP_A_Objective()

    # 結果を格納する変数を初期化
    min_constrained_value = float('inf')
    best_constrained_assignment = None
    min_unconstrained_value = float('inf')
    best_unconstrained_assignment = None

    # 考えられるすべての割り当てを生成
    # itertools.product(range(3), repeat=9) は (0,0,0,0,0,0,0,0,0) から (2,2,2,2,2,2,2,2,2) までの全組み合わせを生成
    n_combinations = objective.n_bins ** objective.n_items
    logging.info(f"Starting brute-force search for {n_combinations} combinations...")

    all_assignments = itertools.product(range(objective.n_bins), repeat=objective.n_items)

    for i, assignment_tuple in enumerate(all_assignments):
        assignment = np.array(assignment_tuple)
        
        # --- 1. 制約ありの最小値を探索 ---
        # objective.__call__ は制約を満たさない場合に 0.0 を返す。
        # 価値は負の値で返されるため、0より小さい値が有効な解。
        constrained_value = objective(assignment)
        if constrained_value < min_constrained_value:
            min_constrained_value = constrained_value
            best_constrained_assignment = assignment
            
        # --- 2. 制約なしの最小値を探索 ---
        # ビン容量の制約を無視して、純粋に価値の合計だけを計算する
        unconstrained_total_value = np.sum([objective.assignment_values[i, assignment[i]] for i in range(objective.n_items)])
        unconstrained_objective_value = -unconstrained_total_value

        if unconstrained_objective_value < min_unconstrained_value:
            min_unconstrained_value = unconstrained_objective_value
            best_unconstrained_assignment = assignment
            
        if (i + 1) % 5000 == 0:
            logging.info(f"Processed {i+1}/{n_combinations} combinations...")

    logging.info("Brute-force search finished.")

    # --- 結果の表示 ---
    print("\n" + "="*50)
    print("           全探索（ブルートフォース）結果")
    print("="*50)
    
    print("\n## 📦 制約あり (Constrained)")
    if best_constrained_assignment is not None:
        print(f"最小目的関数値: {min_constrained_value:.2f} (元の価値: {-min_constrained_value:.2f})")
        print(f"最適な割り当て: {best_constrained_assignment}")
        # 検算
        final_loads = [0] * objective.n_bins
        for item, bin_idx in enumerate(best_constrained_assignment):
            final_loads[bin_idx] += 1
        print(f"この割り当てでのビン占有量: {final_loads} (目標: {objective.bin_capacities})")
    else:
        print("制約を満たす解が見つかりませんでした。")

    print("\n" + "-"*50)

    print("\n## 🌐 制約なし (Unconstrained)")
    if best_unconstrained_assignment is not None:
        print(f"最小目的関数値: {min_unconstrained_value:.2f} (元の価値: {-min_unconstrained_value:.2f})")
        print(f"最適な割り当て: {best_unconstrained_assignment}")
        # 参考：この割り当てでのビン占有量
        final_loads_unconstrained = [0] * objective.n_bins
        for item, bin_idx in enumerate(best_unconstrained_assignment):
            final_loads_unconstrained[bin_idx] += 1
        print(f"この割り当てでのビン占有量: {final_loads_unconstrained} (制約は無視)")
    else:
        print("解が見つかりませんでした。")
    print("\n" + "="*50)