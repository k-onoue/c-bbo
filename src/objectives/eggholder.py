import numpy as np


class EggholderBenchmark:
    def __init__(
        self,
        constrain=False,
    ):
        self.bounds = [-512, 512]
        self.n_dim = 2
        self.min = [446, 449]
        self.fmin = -958.5266506438448
        self.max = [499, 0]
        self.fmax = 1049.131623504493
        
        self.grid_size = self.bounds[1] - self.bounds[0] + 1
        self._tensor_constraint = None
        if constrain:
            self._build_constraint()

    def _build_constraint(self) -> None:
        X, Y = np.meshgrid(
            np.arange(self.bounds[0], self.bounds[1] + 1),
            np.arange(self.bounds[0], self.bounds[1] + 1)
        )
        R_squared = X**2 + Y**2
        self._tensor_constraint = (R_squared > 500**2).astype(int)

    def _coord_to_index(self, x):
        return [int(xi - self.bounds[0]) for xi in x]

    def _index_to_coord(self, idx):
        return [int(i + self.bounds[0]) for i in idx]

    def function(self, x, y):
        return -(y + 47) * np.sin(np.sqrt(abs(x / 2 + (y + 47)))) - x * np.sin(np.sqrt(abs(x - (y + 47))))

    def evaluate(self, x):
        if not self.is_dimensionality_valid(x) or not self.is_in_bounds(x):
            return self.fmax
            
        if self._tensor_constraint is not None:
            idx_y, idx_x = self._coord_to_index(x)
            if not self._tensor_constraint[idx_y, idx_x]:
                return self.fmax
        
        return self.function(x[0], x[1])

    def sample_violation_indices(self, num_samples: int) -> np.ndarray:
        if self._tensor_constraint is None:
            raise ValueError("Constraint not initialized")
        
        indices = np.array(np.where(self._tensor_constraint == 0)).T
        if num_samples > len(indices):
            raise ValueError("num_samples is too large")
            
        return indices[np.random.choice(len(indices), size=num_samples, replace=False)]

    def sample_violation_path(self, num_samples: int = 200) -> list[tuple[int, int]]:
        random_indices = self.sample_violation_indices(num_samples)
        return [tuple(self._index_to_coord(idx)) for idx in random_indices]

    def is_dimensionality_valid(self, x):
        return len(x) == 2

    def is_in_bounds(self, x):
        return all(self.bounds[0] <= xi <= self.bounds[1] for xi in x)
    

class EggholderTF:
    def __init__(
        self,
        constrain=False,
    ):
        self.bounds = [-512, 512]
        self.n_dim = 2
        self.min = [446, 449]
        self.fmin = -958.5266506438448
        self.max = [499, 0]
        self.fmax = 1049.131623504493
        
        self.grid_size = self.bounds[1] - self.bounds[0] + 1
        self._tensor_constraint = None
        if constrain:
            self._build_constraint()

    def _build_constraint(self) -> None:
        X, Y = np.meshgrid(
            np.arange(self.bounds[0], self.bounds[1] + 1),
            np.arange(self.bounds[0], self.bounds[1] + 1)
        )
        R_squared = X**2 + Y**2
        self._tensor_constraint = (R_squared > 500**2).astype(int)

    def _coord_to_index(self, x):
        return [int(xi - self.bounds[0]) for xi in x]

    def _index_to_coord(self, idx):
        return [int(i + self.bounds[0]) for i in idx]

    def function(self, x, y):
        return -(y + 47) * np.sin(np.sqrt(abs(x / 2 + (y + 47)))) - x * np.sin(np.sqrt(abs(x - (y + 47))))

    def evaluate(self, x):
        if not self.is_dimensionality_valid(x) or not self.is_in_bounds(x):
            return self.fmax
        
        return self.function(x[0], x[1])

    def sample_violation_indices(self, num_samples: int) -> np.ndarray:
        if self._tensor_constraint is None:
            raise ValueError("Constraint not initialized")
        
        indices = np.array(np.where(self._tensor_constraint == 0)).T
        if num_samples > len(indices):
            raise ValueError("num_samples is too large")
            
        return indices[np.random.choice(len(indices), size=num_samples, replace=False)]

    def sample_violation_path(self, num_samples: int = 200) -> list[tuple[int, int]]:
        random_indices = self.sample_violation_indices(num_samples)
        return [tuple(self._index_to_coord(idx)) for idx in random_indices]

    def is_dimensionality_valid(self, x):
        return len(x) == 2

    def is_in_bounds(self, x):
        return all(self.bounds[0] <= xi <= self.bounds[1] for xi in x)