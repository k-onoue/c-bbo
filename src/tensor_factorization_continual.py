import logging

import torch
import torch.optim as optim


class TensorFactorization:
    def __init__(
        self, 
        tensor, 
        rank, 
        method="cp", 
        mask=None, 
        constraint=None,  
        is_maximize_c=True,
        device=None,
        prev_state=None,   # Added for continual learning
        verbose=False
    ):
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

        # Move tensors to the device
        tensor = tensor.to(self.device)
        if mask is None:
            mask = torch.ones_like(tensor, device=self.device)
        else:
            mask = mask.to(self.device)
        if constraint is None:
            constraint = torch.ones_like(tensor, device=self.device)
        else:
            constraint = constraint.to(self.device)

        assert tensor.shape == mask.shape == constraint.shape, \
            "Tensor, mask, and constraint must have the same shape."

        self.tensor = tensor
        self.mask = mask
        self.constraint = constraint
        self.is_maximize_c = is_maximize_c

        self.method = method.lower()
        self.total_params = 0  # Initialize total_params

        if self.method == "cp":
            self.rank = rank
            self.dims = tensor.shape
            # Initialize or create factor parameters
            self.factors = [torch.randn(dim, rank, requires_grad=True, device=self.device) 
                            for dim in self.dims]
            self.total_params = sum(factor.numel() for factor in self.factors)

        elif self.method == "tucker":
            self.rank = rank if isinstance(rank, tuple) else (rank,) * len(tensor.shape)
            self.core = torch.randn(*self.rank, requires_grad=True, device=self.device)
            self.factors = [torch.randn(dim, r, requires_grad=True, device=self.device) 
                            for dim, r in zip(tensor.shape, self.rank)]
            self.total_params = self.core.numel() + sum(factor.numel() for factor in self.factors)

        elif self.method == "train":
            # Automatically expand rank to [1, rank, ..., rank, 1] if rank is int
            if isinstance(rank, int):
                rank = [1] + [rank] * (len(tensor.shape) - 1) + [1]

            self.ranks = rank
            assert self.ranks[0] == self.ranks[-1] == 1, "Tensor Train ranks must start and end with 1."
            assert len(self.ranks) == len(tensor.shape) + 1, \
                "Ranks length must be equal to tensor dimensions + 1."
            
            self.factors = [
                torch.randn(self.ranks[i], tensor.shape[i], self.ranks[i + 1], 
                            requires_grad=True, device=self.device)
                for i in range(len(tensor.shape))
            ]
            self.total_params = sum(factor.numel() for factor in self.factors)


        elif self.method == "ring":
            self.rank = rank
            self.factors = [
                torch.randn(rank, tensor.shape[i], rank, requires_grad=True, device=self.device)
                for i in range(len(tensor.shape))
            ]
            self.total_params = sum(factor.numel() for factor in self.factors)
        else:
            raise ValueError(f"Unsupported method: {method}. Choose from 'cp', 'tucker', 'train', or 'ring'.")

        # Attempt to load previous state if provided
        if prev_state is not None:
            self._load_state(prev_state)
            # print("Loaded from prev state!")

        # For logging
        self.loss = None
        self.mse_loss = None
        self.constraint_loss = None
        self.l2_loss = None
        self.iter_end = None

        self.loss_history = {
            "epoch": [],
            "total": [],
            "mse": [],
            "constraint": [],
            "l2": [],
        }

        # Verbosity
        self.verbose = verbose
        if self.verbose:
            logging.info(f"Initialized {method} decomposition with rank {rank} on device {self.device}.")
            logging.info(f"Total parameters: {self.total_params}")

    def _load_state(self, prev_state):
        """
        Simple demonstration of loading factor parameters from prev_state.
        Modify according to how you save the states.
        prev_state could be a list of Tensors or any structure you define.
        """
        if self.method == "cp":
            # Expecting prev_state to be a list of factor Tensors, same shape as self.factors
            if len(prev_state) == len(self.factors):
                for factor, saved_factor in zip(self.factors, prev_state):
                    factor.data.copy_(saved_factor.data)
        elif self.method == "tucker":
            # Suppose prev_state = (core, [factor1, factor2, ...])
            core, prev_factors = prev_state
            self.core.data.copy_(core.data)
            for f1, f2 in zip(self.factors, prev_factors):
                f1.data.copy_(f2.data)
        elif self.method == "train":
            # Suppose prev_state is a list of TT-cores
            for factor, saved_factor in zip(self.factors, prev_state):
                factor.data.copy_(saved_factor.data)
        elif self.method == "ring":
            # Suppose prev_state is a list of ring factors
            for factor, saved_factor in zip(self.factors, prev_state):
                factor.data.copy_(saved_factor.data)

    def get_state(self):
        """
        Return the current factor parameters (and core if tucker, etc.).
        This can be used for continual optimization in TFSampler.
        """
        if self.method == "cp":
            return [factor.clone().detach() for factor in self.factors]
        elif self.method == "tucker":
            return (
                self.core.clone().detach(),
                [factor.clone().detach() for factor in self.factors]
            )
        elif self.method == "train":
            return [factor.clone().detach() for factor in self.factors]
        elif self.method == "ring":
            return [factor.clone().detach() for factor in self.factors]

    def reconstruct(self):
        """
        Reconstruct the tensor based on the decomposition method.
        """
        if self.method == "cp":
            R = self.rank
            recon = torch.zeros_like(self.tensor, device=self.device)
            for r in range(R):
                # Outer product across all modes
                component = self.factors[0][:, r]
                for mode in range(1, len(self.dims)):
                    component = torch.ger(component, self.factors[mode][:, r]).flatten()
                # Reshape it back to self.dims
                recon += component.view(*self.dims)
            return recon

        elif self.method == "tucker":
            # Start with core
            recon = self.core
            # Repeatedly tensordot with factor matrices
            for i, factor in enumerate(self.factors):
                recon = torch.tensordot(recon, factor, dims=[[0], [1]])
            return recon

        elif self.method == "train":
            # TT decomposition reconstruction with einsum
            recon = self.factors[0]
            for factor in self.factors[1:]:
                recon = torch.einsum("...i,ijk->...jk", recon, factor)
            return recon.squeeze()

        elif self.method == "ring":
            # Very rough ring decomposition reconstruction
            n_modes = len(self.factors)
            result = self.factors[0]
            for i in range(1, n_modes - 1):
                result = torch.einsum('ijk,klm->ijlm', result, self.factors[i])
                s1, s2, s3, s4 = result.shape
                result = result.reshape(s1, s2 * s3, s4)
            result = torch.einsum('ijk,klm->jl', result, self.factors[-1])
            result = result.reshape(self.tensor.shape)
            return result

    def optimize(
        self, 
        lr=0.01, 
        max_iter=None, 
        tol=1e-6, 
        mse_tol=1e-1, 
        const_tol=1e-1, 
        reg_lambda=0.0, 
        constraint_lambda=1,
        thr=0.5,
        severe_conv_control=True
    ):
        """
        Perform optimization for the specified decomposition method.

        Args:
          - lr: float, learning rate
          - max_iter: int or None, maximum number of iterations (if None, stop based on tol)
          - tol: float, tolerance for total loss change
          - mse_tol: float, tolerance for MSE loss
          - const_tol: float, tolerance for constraint loss
          - reg_lambda: float, L2 regularization coefficient
          - constraint_lambda: float, penalty coefficient for constraint violations

        Returns:
          - factors: (Optional) Possibly return the updated factors for reuse
        """

        if max_iter is None:
            severe_conv_control = True
        if max_iter is None and not severe_conv_control:
            max_iter = 1000
    
        params = []
        if self.method == "tucker":
            params = [self.core] + self.factors
        else:
            params = self.factors

        optimizer = optim.Adam(params, lr=lr)
        prev_loss = float('inf')
        iteration = 0

        min_iter = 10

        if self.is_maximize_c:
            sign = 1
            # thr = torch.min(self.tensor)
        else:
            sign = -1
            # thr = torch.max(self.tensor)

        while True:
            optimizer.zero_grad()
            reconstruction = self.reconstruct()

            def loss_fn():
                # Count of observed entries
                n_se = torch.sum(self.mask)
                # Count of constraint-violating entries
                n_c = torch.sum(1 - self.constraint)
                n_c = n_c if n_c > 0 else 1
                
                error_term = self.constraint * self.mask * (self.tensor - reconstruction)
                mse_loss = torch.norm(error_term) ** 2 / n_se if n_se > 0 else 0

                violation_term = torch.clamp(
                    (1 - self.constraint) * sign * (reconstruction - thr),
                    min=0
                )
                constraint_loss = constraint_lambda * torch.sum(violation_term) / n_c

                # L2 regularization
                l2_loss = torch.tensor(0., device=self.device, dtype=mse_loss.dtype)
                for p in params:
                    l2_loss += torch.norm(p) ** 2 / p.numel()
                l2_loss *= reg_lambda

                total_loss = mse_loss + constraint_loss + l2_loss
                return total_loss, mse_loss, constraint_loss, l2_loss

            loss, mse_loss, c_loss, l2_loss = loss_fn()
            loss.backward()
            optimizer.step()

            # Logging
            self.loss = loss
            self.mse_loss = mse_loss
            self.constraint_loss = c_loss
            self.l2_loss = l2_loss

            self.loss_history["epoch"].append(iteration+1)
            self.loss_history["total"].append(loss.item())
            self.loss_history["mse"].append(mse_loss.item())
            self.loss_history["constraint"].append(c_loss.item())
            self.loss_history["l2"].append(l2_loss.item())

            if self.verbose:
                logging.info(f"Iter: {iteration}, Loss: {loss.item()}")
                logging.info(f"MSE: {mse_loss.item()}, CONST: {c_loss.item()}, L2: {l2_loss.item()}")

            if severe_conv_control:
                # Check for severe convergence control
                if mse_loss < mse_tol and c_loss < const_tol and iteration > min_iter:
                    if self.verbose:
                        logging.info("Converged based on severe convergence control.")
                    break

            # Check for total loss difference
            if abs(prev_loss - loss.item()) < tol and iteration > min_iter:
                if self.verbose:
                    logging.info("Converged based on total loss tolerance.")
                break

            if max_iter is not None and iteration >= max_iter - 1 and iteration > min_iter:
                if self.verbose:
                    logging.info("Reached max iteration limit.")
                break

            prev_loss = loss.item()
            iteration += 1

        self.iter_end = iteration

        return [p.detach() for p in params]