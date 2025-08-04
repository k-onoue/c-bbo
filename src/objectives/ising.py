import numpy as np
import optuna
from functools import partial
import itertools


class Ising_A_Objective:
    def __init__(self):
        self.n_items = 14
        self.group_a_inds = list(range(7))
        self.group_b_inds = list(range(7, 14))
        self.features = [f'x_{i}' for i in range(self.n_items)]
        
        self.ising_potentials = np.load('data/ising_potentials_a.npz')['ising_potentials']
        
        self.penalty = 15 

        self._tensor_constraint = self.create_tensor_constraint()
        
        print("--- 問題1-A: 制約付きIsingモデル (最小化問題) ---")
        print(f"アイテム数: {self.n_items}")
        print("制約1: グループAとBの選択数が等しい")
        print("制約2: 全体の選択数がちょうど4個")

    def __call__(self, x_assignment):
        count_a = sum(x_assignment[i] for i in self.group_a_inds)
        count_b = sum(x_assignment[i] for i in self.group_b_inds)
        total_count = sum(x_assignment)
        
        is_rule1_valid = (count_a == count_b)
        is_rule2_valid = (total_count == 4)

        if not (is_rule1_valid and is_rule2_valid):
            return self.penalty

        energy = 0
        for i in range(self.n_items):
            for j in range(i + 1, self.n_items):
                if x_assignment[i] == 1 and x_assignment[j] == 1:
                    energy += self.ising_potentials[i, j]
        return energy

    def create_tensor_constraint(self):
        print("Creating tensor constraint for Ising-1B...")
        shape = tuple([2] * self.n_items)
        tensor_constraint = np.zeros(shape, dtype=np.int8)

        all_assignments = itertools.product([0, 1], repeat=self.n_items)

        for assignment_tuple in all_assignments:
            count_a = sum(assignment_tuple[i] for i in self.group_a_inds)
            count_b = sum(assignment_tuple[i] for i in self.group_b_inds)
            total_count = sum(assignment_tuple)

            is_rule1_valid = (count_a == count_b)
            is_rule2_valid = (total_count == 4)
            
            if is_rule1_valid and is_rule2_valid:
                tensor_constraint[assignment_tuple] = 1

        print(f"Constraint tensor created. Feasible points: {np.sum(tensor_constraint)} / {tensor_constraint.size}")
        return tensor_constraint


class Ising_B_Objective:
    def __init__(self):
        self.n_items = 15
        self.group_a_inds = list(range(5))
        self.group_b_inds = list(range(5, 10))
        self.group_c_inds = list(range(10, 15))
        self.features = [f'x_{i}' for i in range(self.n_items)]
        
        self.ising_potentials = np.load('data/ising_potentials_b.npz')['ising_potentials']
        
        self.penalty = 20

        self._tensor_constraint = self.create_tensor_constraint()

        print("\n--- 問題1-B: 制約付きIsingモデル (最小化問題) ---")
        print(f"アイテム数: {self.n_items}")
        print("制約1: グループAとBの選択数が等しい")
        print("制約2: グループCの選択数がちょうど1個")
        
    def __call__(self, x_assignment):
        count_a = sum(x_assignment[i] for i in self.group_a_inds)
        count_b = sum(x_assignment[i] for i in self.group_b_inds)
        count_c = sum(x_assignment[i] for i in self.group_c_inds)

        is_rule1_valid = (count_a == count_b)
        is_rule2_valid = (count_c == 1)

        if not (is_rule1_valid and is_rule2_valid):
            return self.penalty

        energy = 0
        for i in range(self.n_items):
            for j in range(i + 1, self.n_items):
                if x_assignment[i] == 1 and x_assignment[j] == 1:
                    energy += self.ising_potentials[i, j]
        return energy

    def create_tensor_constraint(self):
        print("Creating tensor constraint for Ising-1B...")
        shape = tuple([2] * self.n_items)
        tensor_constraint = np.zeros(shape, dtype=np.int8)

        all_assignments = itertools.product([0, 1], repeat=self.n_items)

        for assignment_tuple in all_assignments:
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

# # ===================================================================
# # メインの実行部分
# # ===================================================================
# if __name__ == "__main__":
    
#     # --- 問題1-Aの最適化 ---
#     problem_1a = Ising_A_Objective()
#     objective_1a = partial(objective, function_name="ising_1a", problem_instance=problem_1a)
#     study_1a = optuna.create_study(direction="minimize")
#     # 全探索空間(16,384)に近い試行回数で実行
#     study_1a.optimize(objective_1a, n_trials=8000)
    
#     print("\n" + "="*20 + " 問題1-A 最適化結果 " + "="*20)
#     print(f"目的関数の最小エネルギー: {study_1a.best_value:.4f}")
#     # best_paramsは辞書なので、キーでソートして値を取得し、選択されたアイテムのインデックスを表示
#     best_assignment_1a = [i for i, val in sorted(study_1a.best_params.items(), key=lambda item: int(item[0].split('_')[1])) if val == 1]
#     print(f"最適な選択（インデックス）: {best_assignment_1a}")


#     # --- 問題1-Bの最適化 ---
#     problem_1b = Ising_B_Objective()
#     objective_1b = partial(objective, function_name="ising_1b", problem_instance=problem_1b)
#     study_1b = optuna.create_study(direction="minimize")
#     # 全探索空間(32,768)に近い試行回数で実行
#     study_1b.optimize(objective_1b, n_trials=10000)

#     print("\n" + "="*20 + " 問題1-B 最適化結果 " + "="*20)
#     print(f"目的関数の最小エネルギー: {study_1b.best_value:.4f}")
#     best_assignment_1b = [i for i, val in sorted(study_1b.best_params.items(), key=lambda item: int(item[0].split('_')[1])) if val == 1]
#     print(f"最適な選択（インデックス）: {best_assignment_1b}")


# if __name__ == "__main__":
#     # テスト用のサンプルコード
#     objective = Ising_A_Objective()
    
#     print("\n--- 問題2-A:")
#     tensor_constraint = objective._tensor_constraint
#     print("制約を満たす点の数:", np.sum(tensor_constraint))
#     print("制約に違反する点の数:", np.sum(~tensor_constraint))
#     print("充足率:", np.sum(tensor_constraint) / tensor_constraint.size)
#     print(tensor_constraint.shape)

#     print()
#     objective = Ising_B_Objective()
    
#     print("\n--- 問題2-B:")
#     tensor_constraint = objective._tensor_constraint
#     print("制約を満たす点の数:", np.sum(tensor_constraint))
#     print("制約に違反する点の数:", np.sum(~tensor_constraint))
#     print("充足率:", np.sum(tensor_constraint) / tensor_constraint.size)
#     print(tensor_constraint.shape)


if __name__ == "__main__":

    # ----------------------------------------------------
    # ここからが検証用のコード
    # ----------------------------------------------------

    # 1. クラスのインスタンスを作成
    ising_problem = Ising_A_Objective()

    # 2. `_tensor_constraint` で値が 1 のインデックスを取得
    # np.argwhere は条件を満たす要素のインデックスを行列として返す
    feasible_indices = np.argwhere(ising_problem._tensor_constraint == 1)

    print(f"\n--- 検証開始: {len(feasible_indices)}個の実行可能点をチェックします ---")

    all_constraints_satisfied = True
    failed_assignments = []

    # 3. 取得した各インデックス（割り当て）をループで検証
    for i, assignment_array in enumerate(feasible_indices):
        # __call__ メソッドはタプルを受け取るので変換する
        assignment_tuple = tuple(assignment_array)
        
        # __call__ メソッドを呼び出して結果を取得
        result = ising_problem(assignment_tuple)
        
        # 制約を満たしているか（ペナルティ値が返ってこないか）をチェック
        is_satisfied = (result != ising_problem.penalty)
        
        # もし制約違反が見つかった場合
        if not is_satisfied:
            all_constraints_satisfied = False
            failed_assignments.append(assignment_tuple)
            print(f"❌ 失敗: 割り当て {assignment_tuple} は制約を満たしませんでした。")


    # 4. 最終結果の表示
    print("\n--- 検証終了 ---")
    if all_constraints_satisfied:
        print(f"✅ **成功**: `_tensor_constraint`で1とマークされた{len(feasible_indices)}個の割り当ては、すべて`__call__`メソッドの制約を満たすことが確認できました。")
    else:
        print(f"❌ **失敗**: `_tensor_constraint`と`__call__`のロジックに矛盾があります。")
        print(f"以下の{len(failed_assignments)}個の割り当てが制約を満たしませんでした:")
        for failed in failed_assignments:
            print(f"  - {failed}")