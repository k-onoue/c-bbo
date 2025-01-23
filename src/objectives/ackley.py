import numpy as np


class AckleyBenchmark:
    def __init__(
        self,
        constrain=False,
    ):
        self.bounds = [-32, 32]  # Standard bounds for Ackley
        self.n_dim = 2
        self.min = [0, 0]  # Global minimum at origin
        self.fmin = 0
        self.max = [32, 32]
        self.fmax = 22.3497  # Approximate max value
        
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
        self._tensor_constraint = (R_squared < 10**2).astype(int)

    def _coord_to_index(self, x):
        return [int(xi - self.bounds[0]) for xi in x]

    def _index_to_coord(self, idx):
        return [int(i + self.bounds[0]) for i in idx]

    def function(self, x, y):
        a, b, c = 20, 0.2, 2*np.pi
        d = 2  # 2D case
        xy = np.array([x, y])
        sum1 = -a * np.exp(-b * np.sqrt(np.sum(xy ** 2) / d))
        sum2 = -np.exp(np.sum(np.cos(c * xy)) / d)
        return sum1 + sum2 + a + np.exp(1)

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


class AckleyTF:
    def __init__(
        self,
        constrain=False,
    ):
        self.bounds = [-32, 32]
        self.n_dim = 2
        self.min = [0, 0]
        self.fmin = 0
        self.max = [32, 32]
        self.fmax = 22.3497
        
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
        self._tensor_constraint = (R_squared < 10**2).astype(int)

    def _coord_to_index(self, x):
        return [int(xi - self.bounds[0]) for xi in x]

    def _index_to_coord(self, idx):
        return [int(i + self.bounds[0]) for i in idx]

    def function(self, x, y):
        a, b, c = 20, 0.2, 2*np.pi
        d = 2
        xy = np.array([x, y])
        sum1 = -a * np.exp(-b * np.sqrt(np.sum(xy ** 2) / d))
        sum2 = -np.exp(np.sum(np.cos(c * xy)) / d)
        return sum1 + sum2 + a + np.exp(1)

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