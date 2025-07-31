import nats_bench
import numpy as np
import itertools # 全探索のためにitertoolsをインポート


class TSSObjective:
    """
    NATS-Bench (tss) のための目的関数クラス
    """
    def __init__(self, is_constrained=False):
        self.DATA_PATH = "../data/NATS-tss-v1_0-3ffb9-simple" 

        self.is_constrained = is_constrained
        self.penalty_value = 0
        self.operations = ['none', 'skip_connect', 'avg_pool_3x3', 'nor_conv_1x1', 'nor_conv_3x3']
        # オペレーション名をインデックスに変換するための辞書
        self.op_to_idx = {op: i for i, op in enumerate(self.operations)}

        # 【変更点】インスタンス化の際に制約テンソルを自動で生成・保持する
        self._tensor_constraint = self.create_tensor_constraint()
        
        # NATS-Bench APIの初期化
        try:
            # 【変更点】verbose=Falseを追加し、APIのログ出力を抑制
            self.api = nats_bench.create(self.DATA_PATH, 'tss', fast_mode=True, verbose=False)
            print("NATS-Bench API loaded successfully (in silent mode).")
        except (FileNotFoundError, TypeError, ValueError) as e:
            print(f"エラー: データファイルの読み込みに失敗しました。")
            print(f"'{self.DATA_PATH}' のパスが正しいか確認してください。")
            print(f"詳細: {e}")
            exit()

    def __call__(self, arch_ops):
        """
        目的関数:
        - 制約を満たす場合: -1 * test_accuracy
        - 制約違反の場合: penalty_value
        """
        # 【変更点】事前に生成したテンソルを使って高速に制約をチェック
        arch_indices = tuple(self.op_to_idx[op] for op in arch_ops)
        if self._tensor_constraint[arch_indices] == 0:
            return self.penalty_value

        # --- 評価値の計算 (実行可能な解のみ) ---
        arch_str = f"|{arch_ops[0]}~0|+|{arch_ops[1]}~0|{arch_ops[2]}~1|+|{arch_ops[3]}~0|{arch_ops[4]}~1|{arch_ops[5]}~2|"
        
        try:
            arch_index = self.api.query_index_by_arch(arch_str)
            info = self.api.get_more_info(arch_index, 'cifar10', hp='200')
            accuracy = info['test-accuracy']
            return -accuracy # 最小化問題に変換
        except (KeyError, IndexError):
            return self.penalty_value

    # 【新規追加】制約テンソルを生成するメソッド
    def create_tensor_constraint(self):
        """
        全探索を行い、制約を満たす（実行可能な）アーキテクチャの位置が1、
        制約に違反する（実行不可能な）位置が0のテンソルを作成する。
        """
        print("Creating constraint tensor (feasible=1, infeasible=0)...")
        # 探索空間の形状は (5, 5, 5, 5, 5, 5)
        shape = tuple([len(self.operations)] * 6)
        tensor_constraint = np.zeros(shape, dtype=np.int8)

        # 5^6 = 15,625通りのすべての組み合わせを生成
        all_arch_ops = itertools.product(self.operations, repeat=6)

        for arch_ops_tuple in all_arch_ops:
            # --- 制約チェック ---
            skip_count = arch_ops_tuple.count('skip_connect')
            conv3x3_count = arch_ops_tuple.count('nor_conv_3x3')
            is_rule1_valid = skip_count >= 3
            is_rule2_valid = conv3x3_count <= 2
            
            # 制約を満たす場合、テンソルの対応する位置を1にする
            if is_rule1_valid and is_rule2_valid:
                # オペレーション名のタプルをインデックスのタプルに変換して位置を指定
                arch_indices = tuple(self.op_to_idx[op] for op in arch_ops_tuple)
                tensor_constraint[arch_indices] = 1
        
        print(f"Tensor created. Feasible points: {np.sum(tensor_constraint)} / {tensor_constraint.size}")
        return tensor_constraint


class SSSObjective:
    """
    NATS-Bench (sss) のための目的関数クラス (最小化問題に変換)
    """
    def __init__(self, is_constrained=True): # is_constrainedをデフォルトでTrueに変更
        # 【重要】このパスが 'sss' 用のデータパスであることを確認
        self.DATA_PATH = "../data/NATS-sss-v1_0-50262-simple"

        self.is_constrained = is_constrained
        self.penalty_value = 0.0 # 最小化問題なのでペナルティは最も大きい値(ここでは0)
        
        self.features = ['C1', 'S1_C', 'S1_R', 'S2_C', 'S2_R']
        self.channel_options = [8, 16, 24, 32, 40, 48, 56, 64]
        # 【追加】チャネル値とインデックスを相互変換するための辞書
        self.channel_to_idx = {ch: i for i, ch in enumerate(self.channel_options)}
        
        # 【変更点】インスタンス化の際に制約テンソルを自動生成
        self._tensor_constraint = self.create_tensor_constraint()

        # NATS-Bench APIの初期化
        try:
            self.api = nats_bench.create(self.DATA_PATH, 'sss', fast_mode=True, verbose=False)
            print("NATS-Bench (SSS) API loaded successfully (in silent mode).")
        except (FileNotFoundError, TypeError, ValueError) as e:
            print(f"エラー: NATS-Benchデータファイルの読み込みに失敗しました。")
            print(f"'{self.DATA_PATH}' のパスが正しいか確認してください。")
            print(f"詳細: {e}")
            exit()

    def __call__(self, arch_channels):
        """
        目的関数:
        - 制約を満たす場合: -1 * test_accuracy
        - 制約違反の場合: penalty_value
        """
        # 【変更点】事前に生成したテンソルを使って高速に制約をチェック
        arch_indices = tuple(self.channel_to_idx[ch] for ch in arch_channels)
        if self._tensor_constraint[arch_indices] == 0:
            return self.penalty_value

        # --- 評価値の計算 (実行可能な解のみ) ---
        arch_str = ":".join(map(str, arch_channels))
        
        try:
            arch_index = self.api.query_index_by_arch(arch_str)
            info = self.api.get_more_info(arch_index, 'cifar10', hp='90')
            accuracy = info['test-accuracy']
            return -accuracy # 最小化問題に変換
        except (KeyError, IndexError, ValueError):
            return self.penalty_value

    # 【新規追加】制約テンソルを生成するメソッド
    def create_tensor_constraint(self):
        """
        全探索を行い、制約を満たす（実行可能な）アーキテクチャの位置が1、
        制約に違反する（実行不可能な）位置が0のテンソルを作成する。
        """
        print("Creating constraint tensor for NATS-Bench (SSS)...")
        # 探索空間の形状は (8, 8, 8, 8, 8)
        shape = tuple([len(self.channel_options)] * len(self.features))

        # is_constrainedがFalseなら、すべてが1のテンソルを返す
        if not self.is_constrained:
            print("Constraint is OFF. All architectures are considered feasible.")
            return np.ones(shape, dtype=np.int8)

        tensor_constraint = np.zeros(shape, dtype=np.int8)

        # 8^5 = 32,768通りのすべてのチャネル数の組み合わせを生成
        all_channel_combos = itertools.product(self.channel_options, repeat=len(self.features))

        for arch_channels_tuple in all_channel_combos:
            # --- 制約チェック ---
            is_rule1_valid = sum(arch_channels_tuple) <= 160
            is_rule2_valid = arch_channels_tuple[3] >= arch_channels_tuple[1] # S2_C >= S1_C

            # 制約を満たす場合、テンソルの対応する位置を1にする
            if is_rule1_valid and is_rule2_valid:
                # チャネル値のタプルをインデックスのタプルに変換して位置を指定
                arch_indices = tuple(self.channel_to_idx[ch] for ch in arch_channels_tuple)
                tensor_constraint[arch_indices] = 1

        print(f"Tensor created. Feasible points: {np.sum(tensor_constraint)} / {tensor_constraint.size}")
        return tensor_constraint

