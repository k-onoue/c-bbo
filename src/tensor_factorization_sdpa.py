import itertools
import time
import sympy as sp
import numpy as np
from ncpol2sdpa import generate_variables, SdpRelaxation


class TensorTrain:
    def __init__(self, ranks, dimensions, variables=None):
        self.ranks = ranks
        self.dimensions = dimensions
        if variables is None:
            self.variables = self._create_tt_variables()
        else:
            self.variables = variables
        self.cores = self._variables_to_cores()

    def _create_tt_variables(self):
        variables = []
        total_elements = 0
        for i, dim in enumerate(self.dimensions):
            r1 = self.ranks[i]
            r2 = self.ranks[i + 1]
            for a in range(r1):
                for b in range(dim):
                    for c in range(r2):
                        variables.append(sp.symbols(f'y_{total_elements}'))
                        total_elements += 1
        return variables

    def _variables_to_cores(self):
        cores = []
        var_index = 0
        for i, dim in enumerate(self.dimensions):
            r1 = self.ranks[i]
            r2 = self.ranks[i + 1]
            core = np.zeros((r1, dim, r2), dtype=object)
            for a in range(r1):
                for b in range(dim):
                    for c in range(r2):
                        core[a, b, c] = self.variables[var_index]
                        var_index += 1
            cores.append(core)
        return cores

    def get_tensor_element(self, indices):
        if len(indices) != len(self.dimensions):
            raise ValueError("Indices length must match tensor order")
        result = np.array([[1]], dtype=object)
        for i, idx in enumerate(indices):
            core = self.cores[i]
            slice = core[:, idx, :]
            result = np.matmul(result, slice)
        return result[0, 0]

    def reconstruct_tensor(self):
        tensor = np.zeros(self.dimensions, dtype=object)
        for idx in itertools.product(*[range(d) for d in self.dimensions]):
            tensor[idx] = self.get_tensor_element(list(idx))
        return tensor

    def frobenius_norm_squared_with_mask(self, original_tensor, mask_tensor=None):
        if mask_tensor is None:
            mask_tensor = np.ones(self.dimensions)
        recon = self.reconstruct_tensor()
        squared_diff = (original_tensor - recon) ** 2
        masked_squared_diff = squared_diff * mask_tensor
        total = sum(masked_squared_diff.flat)
        return sp.expand(total)

    def solve_sdp(self, original_tensor, mask_tensor=None, constraint_tensor=None,
                  target_value=0.5, level=2, solver="sdpa"):
        objective = self.frobenius_norm_squared_with_mask(original_tensor, mask_tensor)

        equality_constraints = []
        if constraint_tensor is not None:
            for idx in np.ndindex(*self.dimensions):
                if constraint_tensor[idx] == 1:
                    poly = self.get_tensor_element(list(idx))
                    constraint = sp.Eq(poly, target_value)
                    equality_constraints.append(constraint)

        sdp = SdpRelaxation(self.variables)
        sdp.get_relaxation(level=level, objective=objective,
                           equalities=equality_constraints if equality_constraints else None)

        try:
            sdp.solve(solver=solver)
            # print(f"SDP solved with objective value: {sdp.primal}")
        except Exception as e:
            print(f"SDP solver error: {e}")
            return sdp, None

        reconstructed_tensor = np.zeros(self.dimensions)
        try:
            # print("Entering solution extraction block...")
            solution_dict = {}

            if hasattr(sdp, 'primal_solution') and sdp.primal_solution is not None:
                # print("Using sdp.primal_solution")
                for i, var in enumerate(self.variables):
                    if i < len(sdp.primal_solution):
                        solution_dict[var] = float(sdp.primal_solution[i])
                    else:
                        solution_dict[var] = target_value

            elif hasattr(sdp, 'x_mat'):
                # print("Using sdp.x_mat")
                if sdp.x_mat is not None:
                    if isinstance(sdp.x_mat, list):
                        for i, var in enumerate(self.variables):
                            raw_val = sdp.x_mat[0][i + 1]
                            # print(f"x_mat[0][{i+1}] = {raw_val} (type: {type(raw_val)})")
                            try:
                                if isinstance(raw_val, (np.ndarray, list)):
                                    solution_dict[var] = float(np.array(raw_val).flatten()[0])
                                else:
                                    solution_dict[var] = float(raw_val)
                            except Exception:
                                solution_dict[var] = target_value
                    else:
                        for i, var in enumerate(self.variables):
                            raw_val = sdp.x_mat[0, i + 1]
                            try:
                                if isinstance(raw_val, (np.ndarray, list)):
                                    solution_dict[var] = float(np.array(raw_val).flatten()[0])
                                else:
                                    solution_dict[var] = float(raw_val)
                            except Exception:
                                solution_dict[var] = target_value
                else:
                    for var in self.variables:
                        solution_dict[var] = target_value
            else:
                for var in self.variables:
                    solution_dict[var] = target_value

            # print("Finished building solution_dict. Starting reconstruction...")
            for idx in np.ndindex(*self.dimensions):
                tt_value = self.get_tensor_element(list(idx))
                try:
                    subs_expr = tt_value.subs(solution_dict)
                    # print(f"Subs expression at {idx}: {repr(subs_expr)} (type: {type(subs_expr)})")
                    if isinstance(subs_expr, np.ndarray):
                        subs_value = float(subs_expr.item())
                    elif isinstance(subs_expr, sp.Expr):
                        subs_value = float(subs_expr.evalf())
                    else:
                        subs_value = float(subs_expr)
                    reconstructed_tensor[idx] = subs_value
                except Exception:
                    reconstructed_tensor[idx] = target_value

        except Exception as e:
            print(f"Error extracting solutions: {e}")
            reconstructed_tensor = np.full(self.dimensions, target_value)
            if mask_tensor is not None:
                for idx in np.ndindex(*self.dimensions):
                    if mask_tensor[idx]:
                        reconstructed_tensor[idx] = original_tensor[idx]

        return sdp, reconstructed_tensor


def run_tensor_train_example(ranks=[1, 2, 1], dimensions=[2, 2],
                             mask_ratio=0.1, constraint_ratio=0.9):
    s0 = time.time()

    num_variables = sum([ranks[i] * dimensions[i] * ranks[i + 1] for i in range(len(dimensions))])
    x = generate_variables('x', num_variables, commutative=True)
    tt = TensorTrain(ranks, dimensions, variables=x)

    original_tensor_np = np.random.rand(*dimensions)
    original_tensor = np.array(original_tensor_np, dtype=object)
    mask_tensor = np.random.choice([0, 1], size=dimensions, p=[1 - mask_ratio, mask_ratio])
    constraint_tensor = np.random.choice([0, 1], size=dimensions, p=[1 - constraint_ratio, constraint_ratio])

    sdp, reconstructed = tt.solve_sdp(
        original_tensor=original_tensor,
        mask_tensor=mask_tensor,
        constraint_tensor=constraint_tensor,
        level=2
    )

    e0 = time.time()
    print(f"Objective value (primal): {sdp.primal}")
    print(f"Time taken: {e0 - s0} sec")

    return sdp, reconstructed


if __name__ == "__main__":
    sdp, reconstructed = run_tensor_train_example()
