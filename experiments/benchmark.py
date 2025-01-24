import argparse
import logging
import os
from functools import partial

import random 

import numpy as np
import optuna
from optuna.samplers import TPESampler

from _src import set_logger
from _src import get_map
from _src import WarcraftObjectiveBenchmark as WarcraftObjective
from _src import EggholderBenchmark as Eggholder
from _src import AckleyBenchmark as Ackley
from _src import GPSampler


def objective(trial, function=None, map_shape=None, objective_function=None):
    """
    Objective function for Bayesian optimization.
    """
    if function == "eggholder":
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
    random.seed(settings['seed'])

    function = settings["function"]
    
    if function == "eggholder":
        objective_function = Eggholder(constrain=settings["constraint"])
        objective_with_args = partial(objective, function=function, objective_function=objective_function)
    elif function == "ackley":
        objective_function = Ackley(constrain=settings["constraint"])
        objective_with_args = partial(objective, function=function, objective_function=objective_function)
    elif function == "warcraft":
        map_targeted = settings["map"]
        map_shape = map_targeted.shape

        objective_function = WarcraftObjective(map_targeted, tensor_constraint=settings["constraint"])
        objective_with_args = partial(objective, map_shape=map_shape, objective_function=objective_function, function=function)
    else:
        raise ValueError(f"Unsupported function type: {function}") 

    sampler = settings["sampler"]
    direction = "maximize" if settings["acqf_settings"]["maximize"] else "minimize"

    study = optuna.create_study(
        study_name=settings["name"],
        sampler=sampler,
        direction=direction,
        storage=settings["storage"],
    )

    logging.info(f"Created new study '{settings['name']}' in {settings['storage']}")

    study.optimize(objective_with_args, n_trials=settings["iter_bo"])

    logging.info(f"Best value: {study.best_value}")
    logging.info(f"Best params: {study.best_params}")

    if function == "warcraft":
        best_x = np.empty(map_shape, dtype=object)
        for i in range(map_shape[0]):
            for j in range(map_shape[1]):
                best_x[i, j] = study.best_params[f"x_{i}_{j}"]
        logging.info(f"Best Direction Matrix:\n{best_x}")

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
    parser = argparse.ArgumentParser(description="Bayesian Optimization Benchmark Experiment")
    parser.add_argument("--timestamp", type=str, help="Timestamp for the experiment.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility.")
    parser.add_argument("--iter_bo", type=int, default=300, help="Number of iterations for Bayesian optimization.")
    parser.add_argument("--acq_trade_off_param", type=float, default=3.0, help="Trade-off parameter for the acquisition function.")
    parser.add_argument("--acq_batch_size", type=int, default=10, help="Batch size for optimization.")
    parser.add_argument("--acq_maximize", action="store_true", help="Whether to maximize the acquisition function.")
    parser.add_argument("--sampler", type=str, choices=["random", "tpe", "gp", "bruteforce"], default="random", help="Sampler for the optimization process.")
    parser.add_argument("--n_startup_trials", type=int, default=10, help="Number of initial trials for the optimization process.")
    parser.add_argument("--dimension", type=int, default=2, help="Number of dimensions for the function.")
    parser.add_argument("--function", type=str, choices=["sphere", "ackley", "warcraft"], default="sphere", help="Objective function to optimize.")
    parser.add_argument("--map_option", type=int, choices=[1, 2, 3], default=1, help="Select the map configuration: 1 for 2x2, 2 for 3x2, 3 for 3x3 (only for Warcraft).")
    parser.add_argument("--plot_save_dir", type=str, help="Directory to save the results")
    return parser.parse_args()


def get_sampler(sampler_type: str, seed: int):
    if sampler_type == "tpe":
        return TPESampler(seed=seed, n_startup_trials=args.n_startup_trials)
    elif sampler_type == "gp":
        return GPSampler(seed=seed, n_startup_trials=args.n_startup_trials)
    else:
        raise ValueError(f"Unsupported sampler type: {sampler_type}")


if __name__ == "__main__":
    base_script_name = os.path.splitext(__file__.split("/")[-1])[0]
    args = parse_args()

    timestamp = args.timestamp
    results_dir = os.path.join("results", timestamp)
    os.makedirs(results_dir, exist_ok=True)

    if args.function == "warcraft":
        map_targeted = get_map(args.map_option)
        log_filename_base = f"{args.function}_map{args.map_option}_{args.sampler}_seed{args.seed}"
    else:
        map_targeted = None
        log_filename_base = f"{args.function}_dim{args.dimension}_{args.sampler}_seed{args.seed}"

    log_filepath = set_logger(log_filename_base, results_dir)

    storage_filename = f"{log_filename_base}.db"
    storage_path = os.path.join(results_dir, storage_filename)
    storage_url = f"sqlite:///{storage_path}"

    sampler = get_sampler(args.sampler, args.seed)

    settings = {
        "name": f"{timestamp}_{log_filename_base}",
        "seed": args.seed,
        "dimension": args.dimension,
        "function": args.function,
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
        "constraint": False,
    }

    if args.function == "warcraft":
        settings["map"] = map_targeted

    logging.info(f"Experiment settings: {settings}")
    run_bo(settings)