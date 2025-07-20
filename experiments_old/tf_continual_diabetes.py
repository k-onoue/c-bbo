import argparse
import logging
import os
from functools import partial

import random
import numpy as np
import pandas as pd
import optuna
from _src import TFContinualSampler, set_logger
from _src import DiabetesObjective


def diabetes_objective(trial, diabetes_instance):
    """
    Objective function wrapper for Optuna
    
    Parameters:
    -----------
    trial : optuna.Trial
        Optuna trial object
    diabetes_instance : DiabetesObjective
        DiabetesObjective class instance
    
    Returns:
    --------
    float
        Objective function value to optimize
    """
    _base = diabetes_instance
    categories = _base.features
    x = np.array([trial.suggest_int(f"x_{category}", 0, 4) for category in categories])
    return _base(x)


def run_bo(settings):
    """
    Run Bayesian optimization with the given settings
    """
    random.seed(settings['seed'])
    np.random.seed(settings['seed'])
    
    # Create diabetes objective instance
    objective_function = DiabetesObjective(
        start_point=np.array([2, 3, 2, 1, 2, 2, 0, 2]),
        is_constrained=False,
        seed=settings["seed"]
    )
    
    # Get constraint tensor if using constraints
    tensor_constraint = objective_function._tensor_constraint if settings.get("constraint", False) else None

    # Setup TFContinualSampler
    sampler = TFContinualSampler(
        seed=settings["seed"],
        method=settings["tf_settings"]["method"],
        acquisition_function="ei",
        sampler_params={
            "n_startup_trials": settings["sampler_settings"].get("n_startup_trials", 1),
            "decomp_iter_num": settings["sampler_settings"].get("decomp_iter_num", 10),
            "mask_ratio": settings["sampler_settings"].get("mask_ratio", 1),
            "include_observed_points": settings["sampler_settings"].get("include_observed_points", False),
            "unique_sampling": settings["sampler_settings"].get("unique_sampling", False),
        },
        tf_params={
            "rank": settings["tf_settings"]["rank"],
            "lr": settings["tf_settings"]["optim_params"]["lr"],
            "max_iter": settings["tf_settings"]["optim_params"]["max_iter"],
            "tol": settings["tf_settings"]["optim_params"]["tol"],
            "reg_lambda": settings["tf_settings"]["optim_params"]["reg_lambda"],
            "constraint_lambda": settings["tf_settings"]["optim_params"]["constraint_lambda"],
        },
        tensor_constraint=tensor_constraint,
        acqf_dist=settings["sampler_settings"]["acqf_dist"],
    )

    # Prepare the objective function with its instance
    objective_with_args = partial(diabetes_objective, diabetes_instance=objective_function)

    # Setup optimization direction
    direction = "maximize" if settings["direction"] else "minimize"
    
    # Create and setup the study
    study = optuna.create_study(
        study_name=settings["name"],
        sampler=sampler,
        direction=direction,
        storage=settings["storage"],
        load_if_exists=True,
    )
    
    # Run optimization
    study.optimize(objective_with_args, n_trials=settings["iter_bo"])
    
    # Save loss history
    import csv
    history_dict = sampler.loss_history
    if history_dict:
        rows = [dict(zip(history_dict.keys(), row_data))
                for row_data in zip(*history_dict.values())]
        filepath = os.path.join(settings["results_dir"], f"{settings['name']}_loss_history.csv")
        with open(filepath, "w") as f:
            writer = csv.DictWriter(f, fieldnames=history_dict.keys())
            writer.writeheader()
            writer.writerows(rows)
    
    # Get best parameters
    best_x = np.array([study.best_params[f"x_{feature}"] for feature in objective_function.features])
    logging.info(f"Starting point: {objective_function._x_start}")
    logging.info(f"Best point: {best_x}")
    logging.info(f"Predicted value at best point: {objective_function._tensor_predicted[tuple(best_x)]:.4f}")
    logging.info(f"Change from starting point: {best_x - objective_function._x_start}")
    logging.info(f"Best objective value: {study.best_value:.4f}")
    
    # Save optimization history plot if save_dir is provided
    if settings.get("plot_save_dir"):
        fig = optuna.visualization.plot_optimization_history(study)
        plot_path = os.path.join(
            settings["plot_save_dir"], 
            f"{settings['name']}_optimization_history.png"
        )
        os.makedirs(os.path.dirname(plot_path), exist_ok=True)
        fig.write_image(plot_path)
        logging.info(f"Saved optimization history plot to {plot_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Diabetes Optimization with Tensor Factorization")
    # Basic parameters
    parser.add_argument("--timestamp", type=str, help="Timestamp for the experiment")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--iter_bo", type=int, default=300, help="Number of BO iterations")
    parser.add_argument("--constraint", action="store_true", help="Use constraint in the objective function")
    parser.add_argument("--direction", action="store_true", help="Maximize the objective function")
    parser.add_argument("--function", type=str, default="diabetes", help="Objective function to optimize")

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

    # Set timestamp if not provided
    timestamp = args.timestamp if args.timestamp else f"{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}"
    results_dir = os.path.join(args.base_dir, timestamp)
    os.makedirs(results_dir, exist_ok=True)

    # Set up logging filename
    log_filename_base = f"diabetes_{args.tf_method}_{args.acqf_dist}_seed{args.seed}"
    if args.constraint:
        log_filename_base += "_constrained"

    log_filepath = set_logger(log_filename_base, results_dir)

    # Set up storage for optuna
    storage_filename = f"{log_filename_base}.db"
    storage_path = os.path.join(results_dir, storage_filename)
    storage_url = f"sqlite:///{storage_path}"

    # Experiment settings
    settings = {
        "name": f"{timestamp}_{log_filename_base}",
        "seed": args.seed,
        "function": "diabetes",
        "constraint": args.constraint,
        "direction": args.direction,
        "iter_bo": args.iter_bo,
        "storage": storage_url,
        "results_dir": results_dir,
        "plot_save_dir": args.plot_save_dir,
        "tf_settings": {
            "method": args.tf_method,
            "rank": args.tf_rank,
            "optim_params": {
                "lr": args.tf_lr,
                "max_iter": args.tf_max_iter,
                "tol": args.tf_tol,
                "reg_lambda": args.tf_reg_lambda,
                "constraint_lambda": args.tf_constraint_lambda,
            }
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

    logging.info(f"Experiment settings: {settings}")
    run_bo(settings)