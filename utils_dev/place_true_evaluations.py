import os
import argparse
import numpy as np
import pandas as pd
from _src import AckleyTF, WarcraftObjectiveTF, get_map, DiabetesObjective


def compute_ackley_value(row, objective):
    """Ackley関数の真の評価値を計算"""
    x = [row['params_x_0'], row['params_x_1']]
    return objective.evaluate(x)

def compute_warcraft_value(row, objective, map_shape):
    """Warcraft問題の真の評価値を計算"""
    if map_shape == (2, 2):
        path = [
            [row['params_x_0_0'], row['params_x_0_1']],
            [row['params_x_1_0'], row['params_x_1_1']]
        ]
    elif map_shape == (2, 3):
        path = [
            [row['params_x_0_0'], row['params_x_0_1'], row['params_x_0_2']],
            [row['params_x_1_0'], row['params_x_1_1'], row['params_x_1_2']]
        ]
    else:  # (3, 3)
        path = [
            [row['params_x_0_0'], row['params_x_0_1'], row['params_x_0_2']],
            [row['params_x_1_0'], row['params_x_1_1'], row['params_x_1_2']],
            [row['params_x_2_0'], row['params_x_2_1'], row['params_x_2_2']]
        ]
    return objective(np.array(path))

def compute_diabetes_value(row, objective):
    path = [
        row['params_x_age'], row['params_x_insu'], row['params_x_mass'], row['params_x_pedi'], 
        row['params_x_plas'], row['params_x_preg'], row['params_x_pres'], row['params_x_skin']
    ]
    return objective(np.array(path))


def add_true_values(results_dir):
    """CSVファイルの評価値を真の値に置き換え"""
    for subdir in os.listdir(results_dir):
        subdir_path = os.path.join(results_dir, subdir)
        if not os.path.isdir(subdir_path):
            continue

        for file in os.listdir(subdir_path):
            if 'loss_history' in file or not file.endswith('.csv'):
                continue

            csv_path = os.path.join(subdir_path, file)
            try:
                df = pd.read_csv(csv_path)
                df = df[df['state'] == 'COMPLETE'].copy()

                if "ackley_dim2" in file:
                    objective = AckleyTF()
                    df['value'] = df.apply(lambda row: compute_ackley_value(row, objective), axis=1)
                
                elif "warcraft_map1" in file:
                    objective = WarcraftObjectiveTF(get_map(1))
                    df['value'] = df.apply(lambda row: compute_warcraft_value(row, objective, (2, 2)), axis=1)
                
                elif "warcraft_map2" in file:
                    objective = WarcraftObjectiveTF(get_map(2))
                    df['value'] = df.apply(lambda row: compute_warcraft_value(row, objective, (2, 3)), axis=1)
                
                elif "warcraft_map3" in file:
                    objective = WarcraftObjectiveTF(get_map(3))
                    df['value'] = df.apply(lambda row: compute_warcraft_value(row, objective, (3, 3)), axis=1)

                elif "diabetes" in file:
                    # Extract seed from filename
                    import re
                    seed_match = re.search(r'seed(\d+)', file)
                    seed = int(seed_match.group(1)) if seed_match else 0
                    
                    objective = DiabetesObjective(
                        start_point=np.array([2, 3, 2, 1, 2, 2, 0, 2]),
                        is_constrained=False,
                        seed=seed
                    )
                    df['value'] = df.apply(lambda row: compute_diabetes_value(row, objective), axis=1)

                df.to_csv(csv_path, index=False)
                print(f"Successfully updated values in {csv_path}")

            except Exception as e:
                print(f"Error processing {csv_path}: {str(e)}")
                continue


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Replace values with true evaluations")
    parser.add_argument("results_dir", type=str, help="Path to the results directory")
    args = parser.parse_args()
    add_true_values(args.results_dir)