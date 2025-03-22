import argparse
import os
import numpy as np
import pandas as pd


DIRECTIONS_LIST = ["oo", "ab", "ac", "ad", "bc", "bd", "cd"]

def constraint_func_ackley(row):
    """制約条件: x_0^2 + x_1^2 <= 10^2"""
    return row['params_x_0']**2 + row['params_x_1']**2 <= 10**2

def constraint_func_warcraft_map1(row, tensor_constraint):
    """WarcraftMap1の制約条件をチェック"""
    path = [
        row['params_x_0_0'], row['params_x_0_1'],
        row['params_x_1_0'], row['params_x_1_1']
    ]
    path_indices = [DIRECTIONS_LIST.index(direction) for direction in path]
    return bool(tensor_constraint[tuple(path_indices)])

def constraint_func_warcraft_map2(row, tensor_constraint):
    """WarcraftMap2の制約条件をチェック"""
    path = [
        row['params_x_0_0'], row['params_x_0_1'], row['params_x_0_2'],
        row['params_x_1_0'], row['params_x_1_1'], row['params_x_1_2']
    ]
    path_indices = [DIRECTIONS_LIST.index(direction) for direction in path]
    return bool(tensor_constraint[tuple(path_indices)])

def constraint_func_warcraft_map3(row, tensor_constraint):
    """WarcraftMap3の制約条件をチェック"""
    path = [
        row['params_x_0_0'], row['params_x_0_1'], row['params_x_0_2'],
        row['params_x_1_0'], row['params_x_1_1'], row['params_x_1_2'],
        row['params_x_2_0'], row['params_x_2_1'], row['params_x_2_2']
    ]
    path_indices = [DIRECTIONS_LIST.index(direction) for direction in path]
    return bool(tensor_constraint[tuple(path_indices)])

def add_constraint_column(results_dir):
    """CSVファイルに制約条件の列を追加"""
    # Load smaller tensors at startup
    constraints_dir = "data/warcraft_constraints"
    tensor_map1 = np.load(os.path.join(constraints_dir, "map1.npy"))
    tensor_map2 = np.load(os.path.join(constraints_dir, "map2.npy"))

    for subdir in os.listdir(results_dir):
        subdir_path = os.path.join(results_dir, subdir)
        if not os.path.isdir(subdir_path):
            continue

        for file in os.listdir(subdir_path):
            # loss_history ファイルをスキップ
            if 'loss_history' in file:
                continue
            
            if not file.endswith(".csv"):
                continue

            csv_path = os.path.join(subdir_path, file)
            
            try:
                # CSVファイル読み込み時のエラーを詳細に捕捉
                try:
                    df = pd.read_csv(csv_path)
                except UnicodeDecodeError as ude:
                    print(f"UnicodeDecodeError in {csv_path}")
                    print(f"Error details: {str(ude)}")
                    # ファイルの問題箇所を特定
                    with open(csv_path, 'rb') as f:
                        lines = f.readlines()
                        for i, line in enumerate(lines, 1):
                            try:
                                line.decode('utf-8')
                            except UnicodeDecodeError as e:
                                print(f"Decoding error at line {i}: {str(e)}")
                                print(f"Problematic bytes: {line}")
                    continue
                except Exception as e:
                    print(f"Error reading {csv_path}")
                    print(f"Error type: {type(e).__name__}")
                    print(f"Error details: {str(e)}")
                    continue

                if "ackley_dim2" in file:
                    df['constraint'] = df.apply(constraint_func_ackley, axis=1)
                elif "warcraft_map1" in file:
                    df['constraint'] = df.apply(lambda row: constraint_func_warcraft_map1(row, tensor_map1), axis=1)
                elif "warcraft_map2" in file:
                    df['constraint'] = df.apply(lambda row: constraint_func_warcraft_map2(row, tensor_map2), axis=1)
                elif "warcraft_map3" in file:
                    tensor_map3 = np.load(os.path.join(constraints_dir, "map3.npy"))
                    df['constraint'] = df.apply(lambda row: constraint_func_warcraft_map3(row, tensor_map3), axis=1)
                    del tensor_map3
                else:
                    continue

                df.to_csv(csv_path, index=False)
                print(f"Successfully added constraint column to {csv_path}")
            
            except Exception as e:
                print(f"Error processing {csv_path}")
                print(f"Error type: {type(e).__name__}")
                print(f"Error details: {str(e)}")
                if hasattr(e, '__traceback__'):
                    import traceback
                    print("Traceback:")
                    traceback.print_tb(e.__traceback__)
                continue



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Add constraint columns to CSV files.")
    parser.add_argument("results_dir", type=str, help="Path to the results directory")
    args = parser.parse_args()
    add_constraint_column(args.results_dir)