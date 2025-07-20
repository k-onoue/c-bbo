import argparse
import logging
import os
from functools import partial

import random 

import numpy as np
import pandas as pd
import optuna
from optuna.samplers import TPESampler

from _src import set_logger
from _src import DiabetesObjective
from _src import GPSampler


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
        is_constrained=settings.get("constraint", False), 
        seed=settings["seed"]
    )
    objective_with_args = partial(diabetes_objective, diabetes_instance=objective_function)
    

    print(f"settings.get('constraint'): {settings.get('constraint')}")

    if settings.get("constraint"):
        init_violation_paths = objective_function.sample_violation_indices(settings["n_init_violation_paths"])

    sampler = settings["sampler"]
    direction = "maximize" if settings["acqf_settings"]["maximize"] else "minimize"
    
    study = optuna.create_study(
        study_name=settings["name"],
        sampler=sampler,
        direction=direction,
        storage=settings["storage"],
    )
    
    # Add initial trials from violation paths
    if settings.get("constraint"):
        logging.info(f"Adding {len(init_violation_paths)} initial trials from constraint violations")
        for violation_path in init_violation_paths:
            params = {f"x_{feature}": int(violation_path[i]) for i, feature in enumerate(objective_function.features)}
            distributions = {f"x_{feature}": optuna.distributions.IntDistribution(0, 4) for feature in objective_function.features}
            value = objective_function(violation_path)

            trial = optuna.trial.create_trial(
                params=params,
                distributions=distributions,
                value=value,
            )
            study.add_trial(trial)
    
    logging.info(f"Created new study '{settings['name']}' in {settings['storage']}")
    
    study.optimize(objective_with_args, n_trials=settings["iter_bo"])
    
    logging.info(f"Best value: {study.best_value}")
    logging.info(f"Best params: {study.best_params}")
    
    # Get best parameters 
    best_x = np.array([study.best_params[f"x_{feature}"] for feature in objective_function.features])
    logging.info(f"Starting point: {objective_function._x_start}")
    logging.info(f"Best point: {best_x}")
    logging.info(f"Predicted value at best point: {objective_function._tensor_predicted[tuple(best_x)]:.4f}")
    logging.info(f"Change from starting point: {best_x - objective_function._x_start}")
    
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
    parser = argparse.ArgumentParser(description="Diabetes Optimization Benchmark Experiment")
    parser.add_argument("--timestamp", type=str, help="Timestamp for the experiment.")
    parser.add_argument("--function", type=str, default="diabetes", help="Function to optimize.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility.")
    parser.add_argument("--iter_bo", type=int, default=500, help="Number of iterations for Bayesian optimization.")
    parser.add_argument("--acq_trade_off_param", type=float, default=1.0, help="Trade-off parameter for the acquisition function.")
    parser.add_argument("--acq_batch_size", type=int, default=1, help="Batch size for optimization.")
    parser.add_argument("--acq_maximize", action="store_true", help="Whether to maximize the acquisition function.")
    parser.add_argument("--sampler", type=str, choices=["random", "tpe", "gp"], default="tpe", help="Sampler for the optimization process.")
    parser.add_argument("--n_startup_trials", type=int, default=1, help="Number of initial trials for the optimization process.")
    parser.add_argument("--constraint", action="store_true", help="Whether to apply constraints to the optimization.")
    parser.add_argument("--n_init_violation_paths", type=int, default=10, help="Number of initial violation paths for Warcraft.")
    parser.add_argument("--plot_save_dir", type=str, help="Directory to save the result plots")
    parser.add_argument("--base_dir", type=str, default="results")
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

    # Set timestamp if not provided
    timestamp = args.timestamp if args.timestamp else f"{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}"
    results_dir = os.path.join(args.base_dir, timestamp)
    os.makedirs(results_dir, exist_ok=True)

    # Set up logging filename
    log_filename_base = f"diabetes_{args.sampler}_seed{args.seed}"
    if args.constraint:
        log_filename_base += "_constrained"

    log_filepath = set_logger(log_filename_base, results_dir)

    # Set up storage for optuna
    storage_filename = f"{log_filename_base}.db"
    storage_path = os.path.join(results_dir, storage_filename)
    storage_url = f"sqlite:///{storage_path}"

    # Get sampler
    sampler = get_sampler(args.sampler, args.seed, args.n_startup_trials)

    # Experiment settings
    settings = {
        "name": f"{timestamp}_{log_filename_base}",
        "seed": args.seed,
        "function": "diabetes",
        "iter_bo": args.iter_bo,
        "storage": storage_url,
        "results_dir": results_dir,
        "plot_save_dir": args.plot_save_dir,
        "sampler": sampler,
        "acqf_settings": {
            "trade_off_param": args.acq_trade_off_param,
            "batch_size": args.acq_batch_size,
            "maximize": args.acq_maximize,
        },
        "constraint": True if args.constraint else False,
        "n_init_violation_paths": args.n_init_violation_paths,
    }

    logging.info(f"Experiment settings: {settings}")
    run_bo(settings)