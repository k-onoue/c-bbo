from functools import partial

import torch
from botorch.acquisition.analytic import LogExpectedImprovement
from botorch.fit import fit_gpytorch_model
from botorch.models.gp_regression import SingleTaskGP
from botorch.models.transforms.outcome import Standardize
from botorch.optim import optimize_acqf
from botorch.utils.transforms import normalize, unnormalize
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.mlls import ExactMarginalLogLikelihood

from optuna.integration.botorch import BoTorchSampler


def matern52_logei_candidates_func(
    train_x: torch.Tensor,
    train_obj: torch.Tensor,
    train_con: torch.Tensor | None,
    bounds: torch.Tensor,
    pending_x: torch.Tensor | None,
) -> torch.Tensor:
    """
    Custom candidate generation function for Optuna BoTorchSampler
    with Matern52 kernel and Log Expected Improvement (LogEI).
    """

    # Normalize the input
    train_x = normalize(train_x, bounds=bounds)

    # Change the model to use Matern52 kernel
    model = SingleTaskGP(
        train_x,
        train_obj,
        covar_module=ScaleKernel(
            MaternKernel(nu=2.5, ard_num_dims=train_x.size(-1))  # Matern52 with ν=2.5
        ),
        outcome_transform=Standardize(m=train_obj.size(-1)),
    )
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    # Train the model
    model.train()
    mll.train()
    fit_gpytorch_model(mll)  # Fit the model

    # Set up LogEI
    best_f = train_obj.max()  # Get the current best objective value
    acqf = LogExpectedImprovement(
        model=model,
        best_f=best_f,
    )

    # Standardized bounds of the search space
    standard_bounds = torch.zeros_like(bounds)
    standard_bounds[1] = 1

    # Optimize the acquisition function
    candidates, _ = optimize_acqf(
        acq_function=acqf,
        bounds=standard_bounds,
        q=1,  # Single candidate optimization
        num_restarts=10,
        raw_samples=512,
        options={"batch_limit": 5, "maxiter": 200},  # Default settings
        sequential=True,
    )

    # Restore the original scale
    return unnormalize(candidates.detach(), bounds=bounds)


class GPSampler(BoTorchSampler):
    """
    A wrapper for BoTorchSampler that automatically applies a custom candidates_func.
    """

    def __init__(self, candidates_func=None, **kwargs):
        # Use the custom candidates_func if provided, otherwise default to matern52_logei_candidates_func
        if candidates_func is None:
            candidates_func = matern52_logei_candidates_func

        # Apply the candidates_func automatically
        super().__init__(candidates_func=candidates_func, **kwargs)
