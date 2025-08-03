import nats_bench
import numpy as np
import itertools # 全探索のためにitertoolsをインポート


class TSSObjective:
    def __init__(self, is_constrained=False):
        # self.DATA_PATH = "../data/NATS-tss-v1_0-3ffb9-simple" 
        self.DATA_PATH = "data/NATS-tss-v1_0-3ffb9-simple" 

        self.is_constrained = is_constrained
        self.penalty_value = 0
        self.operations = ['none', 'skip_connect', 'avg_pool_3x3', 'nor_conv_1x1', 'nor_conv_3x3']
        self.op_to_idx = {op: i for i, op in enumerate(self.operations)}

        self._tensor_constraint = self.create_tensor_constraint()
        
        try:
            self.api = nats_bench.create(self.DATA_PATH, 'tss', fast_mode=True, verbose=False)
            print("NATS-Bench API loaded successfully (in silent mode).")
        except (FileNotFoundError, TypeError, ValueError) as e:
            print(f"エラー: データファイルの読み込みに失敗しました。")
            print(f"'{self.DATA_PATH}' のパスが正しいか確認してください。")
            print(f"詳細: {e}")
            exit()

    def __call__(self, arch_ops):
        arch_indices = tuple(self.op_to_idx[op] for op in arch_ops)
        if self._tensor_constraint[arch_indices] == 0:
            return self.penalty_value

        arch_str = f"|{arch_ops[0]}~0|+|{arch_ops[1]}~0|{arch_ops[2]}~1|+|{arch_ops[3]}~0|{arch_ops[4]}~1|{arch_ops[5]}~2|"
        
        arch_index = self.api.query_index_by_arch(arch_str)
        info = self.api.get_more_info(arch_index, 'cifar10', hp='200')
        accuracy = info['test-accuracy']
        return -accuracy # 最小化問題に変換

    # 【新規追加】制約テンソルを生成するメソッド
    def create_tensor_constraint(self):
        print("Creating constraint tensor (feasible=1, infeasible=0)...")
        shape = tuple([len(self.operations)] * 6)
        tensor_constraint = np.zeros(shape, dtype=np.int8)

        all_arch_ops = itertools.product(self.operations, repeat=6)

        for arch_ops_tuple in all_arch_ops:
            skip_count = arch_ops_tuple.count('skip_connect')
            conv3x3_count = arch_ops_tuple.count('nor_conv_3x3')
            is_rule1_valid = skip_count >= 3
            is_rule2_valid = conv3x3_count <= 2
            
            if is_rule1_valid and is_rule2_valid:
                arch_indices = tuple(self.op_to_idx[op] for op in arch_ops_tuple)
                tensor_constraint[arch_indices] = 1
        
        print(f"Tensor created. Feasible points: {np.sum(tensor_constraint)} / {tensor_constraint.size}")
        return tensor_constraint


class SSSObjective:
    def __init__(self, is_constrained=False): 
        self.DATA_PATH = "data/NATS-sss-v1_0-50262-simple"

        self.is_constrained = is_constrained
        self.penalty_value = 0 
        
        self.features = ['C1', 'S1_C', 'S1_R', 'S2_C', 'S2_R']
        self.channel_options = [8, 16, 24, 32, 40, 48, 56, 64]
        self.channel_to_idx = {ch: i for i, ch in enumerate(self.channel_options)}
        
        self._tensor_constraint = self.create_tensor_constraint()

        try:
            self.api = nats_bench.create(self.DATA_PATH, 'sss', fast_mode=True, verbose=False)
            print("NATS-Bench (SSS) API loaded successfully (in silent mode).")
        except (FileNotFoundError, TypeError, ValueError) as e:
            print(f"エラー: NATS-Benchデータファイルの読み込みに失敗しました。")
            print(f"'{self.DATA_PATH}' のパスが正しいか確認してください。")
            print(f"詳細: {e}")
            exit()

    def __call__(self, arch_channels):
        arch_indices = tuple(self.channel_to_idx[ch] for ch in arch_channels)
        if self._tensor_constraint[arch_indices] == 0:
            return self.penalty_value

        arch_str = ":".join(map(str, arch_channels))
        
        arch_index = self.api.query_index_by_arch(arch_str)
        info = self.api.get_more_info(arch_index, 'cifar10', hp='90')
        accuracy = info['test-accuracy']
        return -accuracy # 最小化問題に変換

    def create_tensor_constraint(self):
        print("Creating constraint tensor for NATS-Bench (SSS)...")
        shape = tuple([len(self.channel_options)] * len(self.features))

        tensor_constraint = np.zeros(shape, dtype=np.int8)

        all_channel_combos = itertools.product(self.channel_options, repeat=len(self.features))

        for arch_channels_tuple in all_channel_combos:
            is_rule1_valid = sum(arch_channels_tuple) <= 160
            is_rule2_valid = arch_channels_tuple[3] >= arch_channels_tuple[1]

            if is_rule1_valid and is_rule2_valid:
                arch_indices = tuple(self.channel_to_idx[ch] for ch in arch_channels_tuple)
                tensor_constraint[arch_indices] = 1

        print(f"Tensor created. Feasible points: {np.sum(tensor_constraint)} / {tensor_constraint.size}")
        return tensor_constraint



if __name__ == "__main__":
    # テスト用のサンプルコード
    objective = TSSObjective()
    
    print("\n--- 問題2-A:")
    tensor_constraint = objective._tensor_constraint
    print("制約を満たす点の数:", np.sum(tensor_constraint))
    print("制約に違反する点の数:", np.sum(~tensor_constraint))
    print("充足率:", np.sum(tensor_constraint) / tensor_constraint.size)
    print(tensor_constraint.shape)

    print()
    objective = SSSObjective()
    
    print("\n--- 問題2-B:")
    tensor_constraint = objective._tensor_constraint
    print("制約を満たす点の数:", np.sum(tensor_constraint))
    print("制約に違反する点の数:", np.sum(~tensor_constraint))
    print("充足率:", np.sum(tensor_constraint) / tensor_constraint.size)
    print(tensor_constraint.shape)

