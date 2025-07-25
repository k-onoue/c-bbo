import argparse
import logging
import os
import random
from functools import partial

import numpy as np
import pandas as pd
import optuna
from optuna.samplers import TPESampler

# _src内のモジュールが直接インポート可能であると仮定
from _src import (
    set_logger, get_map, WarcraftObjectiveBenchmark, EggholderBenchmark,
    AckleyBenchmark, GPSampler, DiabetesObjective, ConstraintWarcraft,
    PressureVesselObjective
)


def objective_general(trial, function_name, map_shape=None, objective_instance=None):
    """
    汎用的な目的関数
    """
    if function_name in ["eggholder", "ackley"]:
        categories = list(range(-100, 100)) if function_name == "eggholder" else list(range(-32, 33))
        x = np.array([trial.suggest_categorical(f"x_{i}", categories) for i in range(2)])
        return objective_instance.evaluate(x)
        
    elif function_name == "warcraft":
        directions = ["oo", "ab", "ac", "ad", "bc", "bd", "cd"]
        x = np.empty(map_shape, dtype=object)
        for i in range(map_shape[0]):
            for j in range(map_shape[1]):
                x[i, j] = trial.suggest_categorical(f"x_{i}_{j}", directions)
        return objective_instance(x)
        
    elif function_name == "diabetes":
        categories = objective_instance.features
        x = np.array([trial.suggest_categorical(f"x_{i}_{category}", [0, 1 ,2, 3, 4]) for i, category in enumerate(categories)])
        return objective_instance(x)

    elif function_name == "pressure":
        categories = objective_instance.features
        x = np.array([trial.suggest_categorical(f"x_{i}_{category}", [0, 1 ,2, 3, 4, 5, 6, 7, 8, 9]) for i, category in enumerate(categories)])
        return objective_instance(x)
        
    else:
        raise ValueError(f"Unsupported function: {function_name}")


def run_bo(settings):
    """
    指定された設定でベイズ最適化を実行
    """
    random.seed(settings['seed'])
    np.random.seed(settings['seed'])

    function_name = settings["function"]
    is_constrained = settings.get("constraint", False)
    
    direction = "maximize" if settings["acqf_settings"]["maximize"] else "minimize"
    study = optuna.create_study(
        study_name=settings["name"],
        sampler=settings["sampler"],
        direction=direction,
        storage=settings["storage"],
    )
    logging.info(f"Created new study '{settings['name']}' in {settings['storage']}")

    # --- 関数に応じた初期設定 ---
    if function_name in ["eggholder", "ackley", "warcraft"]:
        if function_name == "warcraft":
            map_targeted = get_map(settings["map_option"])
            map_shape = map_targeted.shape
            if is_constrained:
                constraint_builder = ConstraintWarcraft(map_shape)
                tensor_constraint = constraint_builder.tensor_constraint
                init_violation_paths = constraint_builder.sample_violation_path(settings["n_init_violation_paths"])
                objective_instance = WarcraftObjectiveBenchmark(map_targeted, tensor_constraint=tensor_constraint)
            else:
                objective_instance = WarcraftObjectiveBenchmark(map_targeted, tensor_constraint=None)
            objective_with_args = partial(objective_general, function_name=function_name, map_shape=map_shape, objective_instance=objective_instance)
        else:
            ObjectiveClass = EggholderBenchmark if function_name == "eggholder" else AckleyBenchmark
            objective_instance = ObjectiveClass(constrain=is_constrained)
            if is_constrained:
                init_violation_paths = objective_instance.sample_violation_path(settings["n_init_violation_paths"])
            objective_with_args = partial(objective_general, function_name=function_name, objective_instance=objective_instance)

        if is_constrained:
            logging.info(f"Adding {len(init_violation_paths)} initial trials from constraint violations")
            for violation_path in init_violation_paths:
                if function_name == "warcraft":
                    params = {f"x_{i}_{j}": violation_path[i * map_shape[1] + j] for i in range(map_shape[0]) for j in range(map_shape[1])}
                    distributions = {f"x_{i}_{j}": optuna.distributions.CategoricalDistribution(["oo", "ab", "ac", "ad", "bc", "bd", "cd"]) for i in range(map_shape[0]) for j in range(map_shape[1])}
                    value = objective_instance(np.array(violation_path).reshape(map_shape))
                else:
                    params = {f"x_{i}": violation_path[i] for i in range(2)}
                    categories = list(range(-100, 100)) if function_name == "eggholder" else list(range(-32, 33))
                    distributions = {f"x_{i}": optuna.distributions.CategoricalDistribution(categories) for i in range(2)}
                    value = objective_instance.evaluate(violation_path)
                trial = optuna.trial.create_trial(params=params, distributions=distributions, value=value)
                study.add_trial(trial)

    elif function_name == "diabetes":
        objective_instance = DiabetesObjective(is_constrained=is_constrained, seed=settings["seed"])
        objective_with_args = partial(objective_general, function_name=function_name, objective_instance=objective_instance)
        
        if is_constrained:
            init_violation_paths = objective_instance.sample_violation_indices(settings["n_init_violation_paths"])
            logging.info(f"Adding {len(init_violation_paths)} initial trials from constraint violations")
            for violation_path in init_violation_paths:
                params = {f"x_{feature}": int(violation_path[i]) for i, feature in enumerate(objective_instance.features)}
                distributions = {f"x_{feature}": optuna.distributions.IntDistribution(0, 4) for feature in objective_instance.features}
                value = objective_instance(violation_path)
                trial = optuna.trial.create_trial(params=params, distributions=distributions, value=value)
                study.add_trial(trial)

    elif function_name == "pressure":
        objective_instance = PressureVesselObjective(seed=settings["seed"])
        objective_with_args = partial(objective_general, function_name=function_name, objective_instance=objective_instance)
        
        if is_constrained:
            init_violation_paths = objective_instance.sample_violation_indices(settings["n_init_violation_paths"])
            logging.info(f"Adding {len(init_violation_paths)} initial trials from constraint violations")
            for violation_path in init_violation_paths:
                params = {f"x_{feature}": int(violation_path[i]) for i, feature in enumerate(objective_instance.features)}
                distributions = {f"x_{feature}": optuna.distributions.IntDistribution(0, 9) for feature in objective_instance.features}
                value = objective_instance(violation_path)
                trial = optuna.trial.create_trial(params=params, distributions=distributions, value=value)
                study.add_trial(trial)

    # --- 最適化の実行 ---
    study.optimize(objective_with_args, n_trials=settings["iter_bo"])

    # --- 結果のログ記録 ---
    logging.info(f"Best value: {study.best_value}")
    logging.info(f"Best params: {study.best_params}")

    if function_name == "warcraft":
        map_shape = get_map(settings["map_option"]).shape
        best_x = np.empty(map_shape, dtype=object)
        for i in range(map_shape[0]):
            for j in range(map_shape[1]):
                best_x[i, j] = study.best_params[f"x_{i}_{j}"]
        logging.info(f"Best Direction Matrix:\n{best_x}")
    elif function_name == "diabetes":
        # 再度インスタンスを生成しないように、上で生成したものを利用
        best_x = np.array([study.best_params[f"x_{feature}"] for feature in objective_instance.features])
        logging.info(f"Starting point: {objective_instance._x_start}")
        logging.info(f"Best point: {best_x}")
        logging.info(f"Predicted value at best point: {objective_instance._tensor_predicted[tuple(best_x)]:.4f}")
        logging.info(f"Change from starting point: {best_x - objective_instance._x_start}")
    elif function_name == "pressure":
        best_x = np.array([study.best_params[f"x_{feature}"] for feature in objective_instance.features])
        best_params_values = np.array([
            objective_instance.mid_points[i][idx]
            for i, idx in enumerate(best_x)
        ])
        logging.info(f"Best point (indices): {best_x}")
        logging.info(f"Best point (values): [Ts={best_params_values[0]:.4f}, Th={best_params_values[1]:.4f}, R={best_params_values[2]:.2f}, L={best_params_values[3]:.2f}]")

    # --- 結果プロットの保存 ---
    if settings.get("plot_save_dir"):
        fig = optuna.visualization.plot_optimization_history(study)
        plot_path = os.path.join(settings["plot_save_dir"], f"{settings['name']}_optimization_history.png")
        os.makedirs(os.path.dirname(plot_path), exist_ok=True)
        fig.write_image(plot_path)
        logging.info(f"Saved optimization history plot to {plot_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Unified Bayesian Optimization Benchmark Experiment")
    # General arguments
    parser.add_argument("--function", type=str, required=True, choices=["warcraft", "ackley", "eggholder", "diabetes", "pressure"], help="Objective function to optimize.")
    parser.add_argument("--timestamp", type=str, help="Timestamp for the experiment. If not provided, current time is used.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility.")
    parser.add_argument("--iter_bo", type=int, default=300, help="Number of iterations for Bayesian optimization.")
    parser.add_argument("--sampler", type=str, choices=["random", "tpe", "gp"], default="tpe", help="Sampler for the optimization process.")
    parser.add_argument("--n_startup_trials", type=int, default=10, help="Number of initial trials for TPE/GP sampler.")
    parser.add_argument("--constraint", action="store_true", help="Whether to apply constraints to the optimization.")
    parser.add_argument("--plot_save_dir", type=str, help="Directory to save the result plots.")
    parser.add_argument("--base_dir", type=str, default="results", help="Base directory to save results.")
    
    # Acquisition function settings
    parser.add_argument("--acq_maximize", action="store_true", help="Whether to maximize the objective function.")
    
    # Function-specific arguments
    parser.add_argument("--map_option", type=int, choices=[1, 2, 3], default=1, help="Map configuration for Warcraft (1: 2x2, 2: 3x2, 3: 3x3).")
    parser.add_argument("--n_init_violation_paths", type=int, default=10, help="Number of initial trials from constraint violating paths.")
    
    return parser.parse_args()


def get_sampler(sampler_type, seed, n_startup_trials):
    if sampler_type == "random":
        return optuna.samplers.RandomSampler(seed=seed)
    elif sampler_type == "tpe":
        return TPESampler(seed=seed, n_startup_trials=n_startup_trials)
    elif sampler_type == "gp":
        return GPSampler(seed=seed, n_startup_trials=n_startup_trials)
    else:
        raise ValueError(f"Unsupported sampler type: {sampler_type}")


if __name__ == "__main__":
    args = parse_args()

    # --- ディレクトリとロギングの設定 ---
    timestamp = args.timestamp if args.timestamp else pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
    results_dir = os.path.join(args.base_dir, timestamp)
    os.makedirs(results_dir, exist_ok=True)

    log_filename_base = f"{args.function}_{args.sampler}_seed{args.seed}"
    if args.function == "warcraft":
        log_filename_base += f"_map{args.map_option}"
    if args.constraint:
        log_filename_base += "_constrained"
    
    set_logger(log_filename_base, results_dir)

    # --- Optunaストレージの設定 ---
    storage_filename = f"{log_filename_base}.db"
    storage_path = os.path.join(results_dir, storage_filename)
    storage_url = f"sqlite:///{storage_path}"

    # --- サンプラーの取得 ---
    sampler = get_sampler(args.sampler, args.seed, args.n_startup_trials)

    # --- 実験設定の辞書を作成 ---
    settings = {
        "name": f"{timestamp}_{log_filename_base}",
        "seed": args.seed,
        "function": args.function,
        "iter_bo": args.iter_bo,
        "constraint": args.constraint,
        "storage": storage_url,
        "results_dir": results_dir,
        "plot_save_dir": args.plot_save_dir,
        "sampler": sampler,
        "acqf_settings": {
            "maximize": args.acq_maximize,
        },
        "n_init_violation_paths": args.n_init_violation_paths,
    }

    if args.function == "warcraft":
        settings["map_option"] = args.map_option

    logging.info(f"Experiment settings: {settings}")
    run_bo(settings)