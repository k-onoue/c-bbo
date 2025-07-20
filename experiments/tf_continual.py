import argparse
import logging
import os
import random
import csv
from functools import partial

import numpy as np
import pandas as pd
import optuna

# Import all necessary modules from the _src library
from _src import (
    TFContinualSampler, 
    set_logger, 
    get_map,
    DiabetesObjective,
    PressureVesselObjective, # Added
    ConstraintWarcraft, 
    WarcraftObjectiveTF as WarcraftObjective, 
    EggholderTF as Eggholder, 
    AckleyTF as Ackley
)


def objective(trial, function=None, map_shape=None, objective_function=None):
    """
    Unified objective function for Optuna optimization.
    """
    if function == "diabetes":
        categories = objective_function.features
        x = np.array([trial.suggest_int(f"x_{category}", 0, 4) for category in categories])
        return objective_function(x)

    elif function == "pressure": # Added
        x = np.array([trial.suggest_int(f"x_{feature}", 0, 9) for feature in objective_function.features])
        return objective_function(x)

    elif function == "eggholder":
        categories = list(range(-100, 100))
        x = np.array([trial.suggest_categorical(f"x_{i}", categories) for i in range(2)])
        return objective_function.evaluate(x)
        
    elif function == "ackley":
        categories = list(range(-32, 33))
        x = np.array([trial.suggest_categorical(f"x_{i}", categories) for i in range(2)])
        return objective_function.evaluate(x)
        
    elif function == "warcraft":
        directions = ["oo", "ab", "ac", "ad", "bc", "bd", "cd"]
        x = np.empty(map_shape, dtype=object)
        for i in range(map_shape[0]):
            for j in range(map_shape[1]):
                x[i, j] = trial.suggest_categorical(f"x_{i}_{j}", directions)
        return objective_function(x)
        
    else:
        raise ValueError(f"Unsupported function type: {function}")
    

def run_bo(settings):
    """
    Run Bayesian optimization with the given settings.
    """
    random.seed(settings['seed'])
    np.random.seed(settings['seed'])

    function = settings["function"]
    
    # --- Objective Function Setup ---
    if function == "diabetes":
        objective_function = DiabetesObjective(seed=settings["seed"])
        tensor_constraint = objective_function._tensor_constraint if settings.get("constraint") else None
        objective_with_args = partial(objective, function=function, objective_function=objective_function)
    
    elif function == "pressure": # Added
        objective_function = PressureVesselObjective(seed=settings["seed"])
        tensor_constraint = objective_function._tensor_constraint if settings.get("constraint") else None
        objective_with_args = partial(objective, function=function, objective_function=objective_function)

    elif function == "warcraft":
        map_targeted = settings["map"]
        map_shape = map_targeted.shape
        if settings["constraint"]:
            constraint_builder = ConstraintWarcraft(map_shape)
            tensor_constraint = constraint_builder.tensor_constraint 
            objective_function = WarcraftObjective(map_targeted, tensor_constraint)
        else:
            tensor_constraint = None
            objective_function = WarcraftObjective(map_targeted)
        objective_with_args = partial(objective, map_shape=map_shape, objective_function=objective_function, function=function)

    elif function in ["eggholder", "ackley"]:
        ObjectiveClass = Eggholder if function == "eggholder" else Ackley
        objective_function = ObjectiveClass(constrain=settings["constraint"])
        tensor_constraint = objective_function._tensor_constraint if settings["constraint"] else None
        objective_with_args = partial(objective, function=function, objective_function=objective_function)
        
    else:
        raise ValueError(f"Unsupported function type: {function}")

    # --- Sampler Setup ---
    sampler = TFContinualSampler(
        seed=settings["seed"],
        method=settings["tf_settings"]["method"],
        acquisition_function="ei",
        sampler_params=settings["sampler_settings"],
        tf_params=settings["tf_settings"],
        tensor_constraint=tensor_constraint,
        acqf_dist=settings["sampler_settings"]["acqf_dist"],
    )

    # --- Study Setup ---
    direction = "maximize" if settings["direction"] else "minimize"
    study = optuna.create_study(
        study_name=settings["name"],
        sampler=sampler,
        direction=direction,
        storage=settings["storage"],
        load_if_exists=True,
    )

    # --- Optimization ---
    study.optimize(objective_with_args, n_trials=settings["iter_bo"])

    # --- Save Results ---
    history_dict = sampler.loss_history
    if history_dict:
        rows = [dict(zip(history_dict.keys(), row_data))
                for row_data in zip(*history_dict.values())]
        filepath = os.path.join(settings["results_dir"], f"{settings['name']}_loss_history.csv")
        with open(filepath, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=history_dict.keys())
            writer.writeheader()
            writer.writerows(rows)
    
    logging.info(f"Best objective value: {study.best_value:.4f}")
    if function == "diabetes":
        best_x = np.array([study.best_params[f"x_{feature}"] for feature in objective_function.features])
        logging.info(f"Starting point: {objective_function._x_start}")
        logging.info(f"Best point: {best_x}")
        logging.info(f"Predicted value at best point: {objective_function._tensor_predicted[tuple(best_x)]:.4f}")
        logging.info(f"Change from starting point: {best_x - objective_function._x_start}")

    elif function == "pressure": # Added
        best_x = np.array([study.best_params[f"x_{feature}"] for feature in objective_function.features])
        best_params_values = np.array([
            objective_function.mid_points[i][idx]
            for i, idx in enumerate(best_x)
        ])
        logging.info(f"Best point (indices): {best_x}")
        logging.info(f"Best point (values): [Ts={best_params_values[0]:.4f}, Th={best_params_values[1]:.4f}, R={best_params_values[2]:.2f}, L={best_params_values[3]:.2f}]")

    elif function == "warcraft":
        best_x = np.empty(map_shape, dtype=object)
        for i in range(map_shape[0]):
            for j in range(map_shape[1]):
                best_x[i, j] = study.best_params[f"x_{i}_{j}"]
        logging.info(f"Best Direction Matrix:\n{best_x}")
    else: # ackley, eggholder
        logging.info(f"Best params: {study.best_params}")


    if settings.get("plot_save_dir"):
        fig = optuna.visualization.plot_optimization_history(study)
        plot_path = os.path.join(settings["plot_save_dir"], f"{settings['name']}_optimization_history.png")
        os.makedirs(os.path.dirname(plot_path), exist_ok=True)
        fig.write_image(plot_path)
        logging.info(f"Saved optimization history plot to {plot_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Unified Bayesian Optimization with Tensor Factorization")
    # Basic parameters
    parser.add_argument("--timestamp", type=str, help="Timestamp for the experiment")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--iter_bo", type=int, default=300, help="Number of BO iterations")
    parser.add_argument("--function", type=str, required=True, choices=["diabetes", "pressure", "warcraft", "eggholder", "ackley"], help="Objective function to run.")
    parser.add_argument("--constraint", action="store_true", help="Use constraint in the objective function")
    parser.add_argument("--direction", action="store_true", help="Maximize the objective function")

    if "--function" in os.sys.argv and "warcraft" in os.sys.argv:
         parser.add_argument("--map_option", type=int, choices=[1, 2, 3], default=1, help="Map option for Warcraft")

    # TF-specific arguments
    parser.add_argument("--tf_method", type=str, choices=["cp", "tucker", "train", "ring"], default="cp")
    parser.add_argument("--tf_rank", type=int, default=3, help="Tensor rank")
    parser.add_argument("--tf_lr", type=float, default=0.01, help="Learning rate")
    parser.add_argument("--tf_max_iter", type=int, default=None, help="Max iterations")
    parser.add_argument("--tf_tol", type=float, default=1e-5, help="Convergence tolerance")
    parser.add_argument("--tf_reg_lambda", type=float, default=1e-3, help="Regularization strength")
    parser.add_argument("--tf_constraint_lambda", type=float, default=1.0, help="Constraint penalty")
    
    # Sampler parameters
    parser.add_argument("--decomp_iter_num", type=int, default=10)
    parser.add_argument("--mask_ratio", type=float, default=1)
    parser.add_argument("--include_observed_points", action="store_true")
    parser.add_argument("--unique_sampling", action="store_true")
    parser.add_argument("--n_startup_trials", type=int, default=1)
    parser.add_argument("--acqf_dist", type=str, choices=["n", "t1", "t2"], default="n")

    # Save directory
    parser.add_argument("--base_dir", type=str, default="results")
    parser.add_argument("--plot_save_dir", type=str, help="Directory to save the results")
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    timestamp = args.timestamp if args.timestamp else f"{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}"
    results_dir = os.path.join(args.base_dir, timestamp)
    os.makedirs(results_dir, exist_ok=True)

    log_filename_base = f"{args.function}_{args.tf_method}_{args.acqf_dist}_seed{args.seed}"
    if args.function == "warcraft":
        log_filename_base = f"{args.function}_map{args.map_option}_{args.tf_method}_{args.acqf_dist}_seed{args.seed}"
    if args.constraint:
        log_filename_base += "_constrained"

    log_filepath = set_logger(log_filename_base, results_dir)

    storage_filename = f"{log_filename_base}.db"
    storage_path = os.path.join(results_dir, storage_filename)
    storage_url = f"sqlite:///{storage_path}"

    settings = {
        "name": f"{timestamp}_{log_filename_base}",
        "seed": args.seed,
        "function": args.function,
        "constraint": args.constraint,
        "direction": args.direction,
        "iter_bo": args.iter_bo,
        "storage": storage_url,
        "results_dir": results_dir,
        "plot_save_dir": args.plot_save_dir,
        "tf_settings": {
            "method": args.tf_method,
            "rank": args.tf_rank,
            "lr": args.tf_lr,
            "max_iter": args.tf_max_iter,
            "tol": args.tf_tol,
            "reg_lambda": args.tf_reg_lambda,
            "constraint_lambda": args.tf_constraint_lambda,
        },
        "sampler_settings": {
            "decomp_iter_num": args.decomp_iter_num,
            "mask_ratio": args.mask_ratio,
            "include_observed_points": args.include_observed_points,
            "unique_sampling": args.unique_sampling,
            "n_startup_trials": args.n_startup_trials,
            "acqf_dist": args.acqf_dist
        },
    }
    
    if args.function == "warcraft":
        settings["map"] = get_map(args.map_option)

    logging.info(f"Experiment settings: {settings}")
    run_bo(settings)