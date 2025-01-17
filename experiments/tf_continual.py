import argparse
import logging
import os
from functools import partial

import random
import numpy as np
import optuna
from _src import TFContinualSampler, set_logger
from _src import WarcraftObjective, ConstraintWarcraft, get_map
from _src import sphere, ackley, Eggholder


def objective(trial, dimension=None, function=None, map_shape=None, objective_function=None):
    """
    Objective function for Bayesian optimization.
    """
    if function in ["sphere", "ackley"]:
        categories = list(range(-5, 6))

        x = np.array([trial.suggest_categorical(f"x_{i}", categories) for i in range(dimension)])
        if function == "sphere":
            return sphere(x)
        elif function == "ackley":
            return ackley(x)
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
    
    if function in ["sphere", "ackley"]:
        if settings["constraint"]:
            raise ValueError("Constraint not supported for this function")
        dimension = settings["dimension"]
        objective_with_args = partial(objective, dimension=dimension, function=function)
    elif function == "warcraft":
        map_targeted = settings["map"]
        map_shape = map_targeted.shape

        if settings["constraint"]:
            constriant_builder = ConstraintWarcraft(map_shape)
            tensor_constraint = constriant_builder.tensor_constraint 
            objective_function = WarcraftObjective(map_targeted, tensor_constraint)
        else:
            tensor_constraint = None
            objective_function = WarcraftObjective(map_targeted)

        objective_with_args = partial(objective, map_shape=map_shape, objective_function=objective_function, function=function)
    else:
        raise ValueError(f"Unsupported function type: {function}")
    
    acqf = settings["acqf_settings"]["acquisition_function"]

    sampler = TFContinualSampler(
        seed=settings["seed"],
        method=settings["tf_settings"]["method"],
        acquisition_function=acqf,
        sampler_params={
            "n_startup_trials": settings["sampler_settings"].get("n_startup_trials", 1),
            "decomp_iter_num": settings["sampler_settings"].get("decomp_iter_num", 10) if acqf != "ts" else 1,
            "mask_ratio": settings["sampler_settings"].get("mask_ratio", 0.9),
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
        acqf_params={
            "trade_off_param": settings["acqf_settings"]["trade_off_param"],
        },
        tensor_constraint=tensor_constraint
    )

    direction = "maximize" if settings["direction"] else "minimize"

    study = optuna.create_study(
        study_name=settings["name"],
        sampler=sampler,
        direction=direction,
        storage=settings["storage"],
        load_if_exists=True,
    )

    study.optimize(objective_with_args, n_trials=settings["iter_bo"])

    # Save loss history
    import csv
    history_dict = sampler.loss_history
    rows = [dict(zip(history_dict.keys(), row_data))
            for row_data in zip(*history_dict.values())]
    filepath = os.path.join(settings["results_dir"], f"{settings['name']}_loss_history.csv")
    with open(filepath, "w") as f:
        writer = csv.DictWriter(f, fieldnames=history_dict.keys())
        writer.writeheader()
        writer.writerows(rows)
    
    if function == "warcraft":
        best_x = np.empty(map_shape, dtype=object)
        for i in range(map_shape[0]):
            for j in range(map_shape[1]):
                best_x[i, j] = study.best_params[f"x_{i}_{j}"]
        print(f"Best Direction Matrix:\n{best_x}")

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
    parser = argparse.ArgumentParser(description="Bayesian Optimization with Tensor Factorization")
    # Basic parameters
    parser.add_argument("--timestamp", type=str, help="Timestamp for the experiment")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--iter_bo", type=int, default=300, help="Number of BO iterations")
    parser.add_argument("--dimension", type=int, default=2, help="Problem dimension")
    parser.add_argument("--function", type=str, choices=["sphere", "ackley", "warcraft"], default="sphere")
    parser.add_argument("--map_option", type=int, choices=[1, 2, 3], default=1)
    parser.add_argument("--constraint", action="store_true", help="Use constraint in the objective function")
    parser.add_argument("--direction", action="store_true", help="Maximize the objective function")

    # TF-specific arguments
    parser.add_argument("--tf_method", type=str, choices=["cp", "tucker", "train", "ring"], default="cp")
    parser.add_argument("--tf_rank", type=int, default=3, help="Tensor rank")
    parser.add_argument("--tf_lr", type=float, default=0.01, help="Learning rate")
    parser.add_argument("--tf_max_iter", type=int, default=None, help="Max iterations")
    parser.add_argument("--tf_tol", type=float, default=1e-5, help="Convergence tolerance")
    parser.add_argument("--tf_reg_lambda", type=float, default=1e-3, help="Regularization strength")
    parser.add_argument("--tf_constraint_lambda", type=float, default=1.0, help="Constraint penalty")
    # parser.add_argument("--tf_fill_method", type=str, choices=["zero", "normal", "minmax"], default="zero")
    
    # Sampler parameters
    parser.add_argument("--decomp_iter_num", type=int, default=10)
    parser.add_argument("--mask_ratio", type=float, default=0.9)
    parser.add_argument("--include_observed_points", action="store_true")
    parser.add_argument("--unique_sampling", action="store_true")
    parser.add_argument("--n_startup_trials", type=int, default=1)

    # Acquisition function arguments
    parser.add_argument("--acquisition_function", type=str, choices=["ucb", "ei", "ts"], default="ucb")
    parser.add_argument("--acq_trade_off_param", type=float, default=1.0)

    # Save directory
    parser.add_argument("--plot_save_dir", type=str, help="Directory to save the results")
    
    return parser.parse_args()



if __name__ == "__main__":
    base_script_name = os.path.splitext(__file__.split("/")[-1])[0]
    args = parse_args()

    timestamp = args.timestamp
    results_dir = os.path.join("results", timestamp)
    os.makedirs(results_dir, exist_ok=True)

    if args.function == "warcraft":
        map_targeted = get_map(args.map_option)
        log_filename_base = f"{args.function}_map{args.map_option}_{args.tf_method}_seed{args.seed}"
    else:
        map_targeted = None
        log_filename_base = f"{args.function}_dim{args.dimension}_{args.tf_method}_seed{args.seed}"

    log_filepath = set_logger(log_filename_base, results_dir)

    storage_filename = f"{log_filename_base}.db"
    storage_path = os.path.join(results_dir, storage_filename)
    storage_url = f"sqlite:///{storage_path}"

    settings = {
        "name": f"{timestamp}_{log_filename_base}",
        "seed": args.seed,
        "dimension": args.dimension,
        "function": args.function,
        "constraint": args.constraint,
        "direction": args.direction,
        "map": map_targeted,
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
        "acqf_settings": {
            "acquisition_function": args.acquisition_function,
            "trade_off_param": args.acq_trade_off_param,
        },
        "sampler_settings": {
            "decomp_iter_num": args.decomp_iter_num,
            "mask_ratio": args.mask_ratio,
            "include_observed_points": args.include_observed_points,
            "unique_sampling": args.unique_sampling,
            "n_startup_trials": args.n_startup_trials,
        },
    }

    logging.info(f"Experiment settings: {settings}")
    run_bo(settings)