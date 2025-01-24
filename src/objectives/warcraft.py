import numpy as np


def get_opposite(direction: str) -> str:
    pair_dict = {"a": "c", "c": "a", "b": "d", "d": "b"}
    return pair_dict.get(direction, "")


def judge_continuity(d_from: str, current_direction: str) -> bool:
    d_opposite = get_opposite(d_from)
    return d_opposite in current_direction


def get_next_coordinate(
    d_to: str, current_coordinate: tuple[int, int]
) -> tuple[int, int]:
    update_dict = {"a": (-1, 0), "b": (0, -1), "c": (0, 1), "d": (1, 0)}
    delta = update_dict.get(d_to, (0, 0))
    return (current_coordinate[0] + delta[0], current_coordinate[1] + delta[1])


def judge_location_validity(current: tuple[int, int], shape: tuple[int, int]) -> bool:
    return 0 <= current[0] < shape[0] and 0 <= current[1] < shape[1]


def get_d_to(d_from: str, current_direction: str) -> str:
    return (
        current_direction[1] if current_direction[0] == d_from else current_direction[0]
    )


def navigate_through_matrix(direction_matrix, start, goal):
    history = []
    current = start
    shape = direction_matrix.shape

    # Determine the initial direction based on the current cell
    current_direction = direction_matrix[current]

    # If "a" or "b" is present in the current direction, set d_to to the other direction
    if "a" in current_direction or "b" in current_direction:
        d_to = (
            get_d_to("a", current_direction)
            if "a" in current_direction
            else get_d_to("b", current_direction)
        )
    else:
        return history  # If neither "a" nor "b" is present, no movement is possible, so return the history

    # If the current direction is exactly "ab", append the current position and return the history
    if current_direction == "ab":
        history.append(current)
        return history

    # Append the initial position to the history
    history.append(current)
    next_pos = get_next_coordinate(d_to, current)

    # Continue navigating through the matrix while the position is valid and hasn't reached the goal
    while judge_location_validity(next_pos, shape) and current != goal:
        # If the next position is "oo", navigation cannot continue
        if direction_matrix[next_pos] == "oo":
            break

        # If continuity is broken, stop navigation
        if not judge_continuity(d_to, direction_matrix[next_pos]):
            break

        # Move to the next position and append it to the history
        current = next_pos
        history.append(current)
        if current == goal:
            break

        # Determine the new direction based on the current position
        direction = direction_matrix[current]
        d_from = get_opposite(d_to)
        d_to = get_d_to(d_from, direction)
        next_pos = get_next_coordinate(d_to, current)

    return history


def manhattan_distance(coord1: tuple[int, int], coord2: tuple[int, int]) -> int:
    return abs(coord1[0] - coord2[0]) + abs(coord1[1] - coord2[1])


def convert_path_index_to_arr(path, map_shape):
    path_gen_reversed = iter(reversed(path))
    map = np.zeros(map_shape)

    for i in range(map_shape[0]):
        for j in range(map_shape[1]):
            map[i, j] = next(path_gen_reversed)

    return map


class WarcraftObjectiveTF:
    def __init__(
        self,
        weight_matrix: np.ndarray,
        tensor_constraint: np.ndarray = None,
    ) -> None:
        self.weight_matrix = weight_matrix / np.sum(weight_matrix)  # normalize
        self.shape = weight_matrix.shape

        self._val_mask_dict = {
            "oo": np.zeros((3, 3)),
            "ab": np.array([[0, 1, 0], [1, 1, 0], [0, 0, 0]]),
            "ac": np.array([[0, 0, 0], [1, 1, 1], [0, 0, 0]]),
            "ad": np.array([[0, 0, 0], [1, 1, 0], [0, 1, 0]]),
            "bc": np.array([[0, 1, 0], [0, 1, 1], [0, 0, 0]]),
            "bd": np.array([[0, 1, 0], [0, 1, 0], [0, 1, 0]]),
            "cd": np.array([[0, 0, 0], [0, 1, 1], [0, 1, 0]]),
        }

        self._tensor_constraint = tensor_constraint

    def calculate_penalty_type2(self, idx, val, map_shape):
        # Define the mask dictionary within the function
        val_mask_dict = self._val_mask_dict

        # Initialize the expanded array with zeros, sized (2*map_shape[0] + 1, 2*map_shape[1] + 1)
        arr_expanded = np.zeros((map_shape[0] * 2 + 1, map_shape[1] * 2 + 1))

        # Calculate the starting positions for the mask application
        x_s, y_s = idx[0] * 2, idx[1] * 2

        # Apply the corresponding mask from the dictionary based on 'val'
        arr_expanded[x_s : x_s + 3, y_s : y_s + 3] = val_mask_dict.get(
            val, np.zeros((3, 3))
        )

        # Get the indices of the ones in the expanded array
        ones_indices = np.argwhere(arr_expanded == 1)

        # Initialize a variable to store the minimum distance
        row = arr_expanded.shape[0] - 1
        col = arr_expanded.shape[1] - 1

        max_distance = manhattan_distance((0, 0), (row, col - 1))
        min_distance = max_distance

        index_goal_list = [(row, col - 1), (row - 1, col)]

        # Iterate through all pairs of (1 indices, target indices)
        for one_idx in ones_indices:
            for target_idx in index_goal_list:
                dist = manhattan_distance(one_idx, target_idx)
                if dist < min_distance:
                    min_distance = dist

        penalty = min_distance / max_distance
        return penalty

    def __call__(self, x):
        """Calculate the objective function."""
        if isinstance(x, np.ndarray):
            direction_matrix = x
        else:
            direction_matrix = np.array(x)

        # Create a mask where "oo" is 0 and other directions are 1
        mask = np.where(direction_matrix == "oo", 0, 1)
        penalty_1 = np.sum(self.weight_matrix * mask)

        start = (0, 0)
        goal = (self.shape[0] - 1, self.shape[1] - 1)

        history = navigate_through_matrix(direction_matrix, start, goal)

        if history:
            penalty_3 = self.calculate_penalty_type2(
                history[-1], direction_matrix[history[-1]], self.shape
            )
        else:
            penalty_3 = 1

        score = penalty_1 + penalty_3
        return score
    
    def check_penalty_type2(self, x):
        penalty_3 = self.get_penalty_type2(x)
        return penalty_3 == 0
    
    def get_penalty_type2(self, x):
        if isinstance(x, np.ndarray):
            direction_matrix = x
        else:
            direction_matrix = np.array(x)

        start = (0, 0)
        goal = (self.shape[0] - 1, self.shape[1] - 1)

        history = navigate_through_matrix(direction_matrix, start, goal)

        if history:
            penalty_3 = self.calculate_penalty_type2(
                history[-1], direction_matrix[history[-1]], self.shape
            )
        else:
            penalty_3 = 1

        return penalty_3

    def visualize(self, x):
        """Visualize the direction matrix."""
        direction_matrix = x
        print(direction_matrix)


class WarcraftObjectiveBenchmark:
    def __init__(
        self,
        weight_matrix: np.ndarray,
        tensor_constraint: np.ndarray = None,
    ) -> None:
        self.weight_matrix = weight_matrix / np.sum(weight_matrix)  # normalize
        self.shape = weight_matrix.shape

        self._val_mask_dict = {
            "oo": np.zeros((3, 3)),
            "ab": np.array([[0, 1, 0], [1, 1, 0], [0, 0, 0]]),
            "ac": np.array([[0, 0, 0], [1, 1, 1], [0, 0, 0]]),
            "ad": np.array([[0, 0, 0], [1, 1, 0], [0, 1, 0]]),
            "bc": np.array([[0, 1, 0], [0, 1, 1], [0, 0, 0]]),
            "bd": np.array([[0, 1, 0], [0, 1, 0], [0, 1, 0]]),
            "cd": np.array([[0, 0, 0], [0, 1, 1], [0, 1, 0]]),
        }

        self._tensor_constraint = tensor_constraint

    def calculate_penalty_type2(self, idx, val, map_shape):
        # Define the mask dictionary within the function
        val_mask_dict = self._val_mask_dict

        # Initialize the expanded array with zeros, sized (2*map_shape[0] + 1, 2*map_shape[1] + 1)
        arr_expanded = np.zeros((map_shape[0] * 2 + 1, map_shape[1] * 2 + 1))

        # Calculate the starting positions for the mask application
        x_s, y_s = idx[0] * 2, idx[1] * 2

        # Apply the corresponding mask from the dictionary based on 'val'
        arr_expanded[x_s : x_s + 3, y_s : y_s + 3] = val_mask_dict.get(
            val, np.zeros((3, 3))
        )

        # Get the indices of the ones in the expanded array
        ones_indices = np.argwhere(arr_expanded == 1)

        # Initialize a variable to store the minimum distance
        row = arr_expanded.shape[0] - 1
        col = arr_expanded.shape[1] - 1

        max_distance = manhattan_distance((0, 0), (row, col - 1))
        min_distance = max_distance

        index_goal_list = [(row, col - 1), (row - 1, col)]

        # Iterate through all pairs of (1 indices, target indices)
        for one_idx in ones_indices:
            for target_idx in index_goal_list:
                dist = manhattan_distance(one_idx, target_idx)
                if dist < min_distance:
                    min_distance = dist

        penalty = min_distance / max_distance
        return penalty

    def __call__(self, x):
        """Calculate the objective function."""
        if isinstance(x, np.ndarray):
            direction_matrix = x
        else:
            direction_matrix = np.array(x)

        ############################################################
        ############################################################
        ############################################################
        ############################################################
        ############################################################
        ############################################################
        ############################################################
        ############################################################
        ############################################################
        if self._tensor_constraint:
            directions_list = list(self._val_mask_dict.keys())
            sequence = tuple(directions_list.index(direction) for direction in direction_matrix.flatten())
            if not self._tensor_constraint[sequence]:
                return 2

        # Create a mask where "oo" is 0 and other directions are 1
        mask = np.where(direction_matrix == "oo", 0, 1)
        penalty_1 = np.sum(self.weight_matrix * mask)

        start = (0, 0)
        goal = (self.shape[0] - 1, self.shape[1] - 1)

        history = navigate_through_matrix(direction_matrix, start, goal)

        if history:
            penalty_3 = self.calculate_penalty_type2(
                history[-1], direction_matrix[history[-1]], self.shape
            )
        else:
            penalty_3 = 1

        score = penalty_1 + penalty_3
        return score
    
    def check_penalty_type2(self, x):
        penalty_3 = self.get_penalty_type2(x)
        return penalty_3 == 0
    
    def get_penalty_type2(self, x):
        if isinstance(x, np.ndarray):
            direction_matrix = x
        else:
            direction_matrix = np.array(x)

        start = (0, 0)
        goal = (self.shape[0] - 1, self.shape[1] - 1)

        history = navigate_through_matrix(direction_matrix, start, goal)

        if history:
            penalty_3 = self.calculate_penalty_type2(
                history[-1], direction_matrix[history[-1]], self.shape
            )
        else:
            penalty_3 = 1

        return penalty_3

    def visualize(self, x):
        """Visualize the direction matrix."""
        direction_matrix = x
        print(direction_matrix)


def build_constraint_warcraft(map_shape: tuple[int, int]) -> np.ndarray:
    # Directions dictionary
    directions_dict = {
        "oo": np.array([0, 0]),
        "ab": np.array([1, 1]),
        "ac": np.array([0, 2]),
        "ad": np.array([1, 1]),
        "bc": np.array([1, 1]),
        "bd": np.array([2, 0]),
        "cd": np.array([1, 1]),
    }

    directions_list = list(directions_dict.keys())

    # Map parameters
    map_length = map_shape[0] * map_shape[1]
    ideal_gain = (map_shape[0] + map_shape[1] - 1) * 2

    # Initialize constraints as NumPy arrays
    tensor_constraint_1 = np.zeros((len(directions_list),) * map_length)
    tensor_constraint_2 = np.zeros((len(directions_list),) * map_length)
    tensor_constraint_3 = np.zeros((len(directions_list),) * map_length)

    # Constraint 1: (0, 0) != "oo", "ab"
    for direction in directions_list:
        if direction not in ["oo", "ab"]:
            # tensor_constraint_1[..., directions_list.index(direction)] = 1
            tensor_constraint_1[directions_list.index(direction), ...] = 1

    # Constraint 2: (map_shape[0] - 1, map_shape[1] - 1) != "oo", "cd"
    for direction in directions_list:
        if direction not in ["oo", "cd"]:
            # tensor_constraint_2[directions_list.index(direction), ...] = 1
            tensor_constraint_2[..., directions_list.index(direction)] = 1

    # Constraint 3: len[path] == map_shape[0] * map_shape[1]
    for index, _ in np.ndenumerate(tensor_constraint_3):
        gain = np.sum([directions_dict[directions_list[idx]].sum() for idx in index])
        if gain == ideal_gain:
            tensor_constraint_3[index] = 1

    # Combine constraints with logical AND
    tensor_constraint = np.logical_and(
        tensor_constraint_1,
        np.logical_and(tensor_constraint_2, tensor_constraint_3)
    )

    return tensor_constraint


class ConstraintWarcraft:
    def __init__(self, map_shape: tuple[int, int]) -> None:
        self.map_shape = map_shape
        self.directions_dict = {
            "oo": np.array([0, 0]),
            "ab": np.array([1, 1]),
            "ac": np.array([0, 2]),
            "ad": np.array([1, 1]),
            "bc": np.array([1, 1]),
            "bd": np.array([2, 0]),
            "cd": np.array([1, 1]),
        }
        self.directions_list = list(self.directions_dict.keys())

        self.tensor_constraint = self._build()

    def _build(self) -> np.ndarray:
        directions_dict = self.directions_dict
        directions_list = self.directions_list

        # Map parameters
        map_length = self.map_shape[0] * self.map_shape[1]
        ideal_gain = (self.map_shape[0] + self.map_shape[1] - 1) * 2

        # Initialize constraints as NumPy arrays
        tensor_constraint_1 = np.zeros((len(directions_list),) * map_length)
        tensor_constraint_2 = np.zeros((len(directions_list),) * map_length)
        tensor_constraint_3 = np.zeros((len(directions_list),) * map_length)

        # Constraint 1: (0, 0) != "oo", "ab"
        for direction in directions_list:
            if direction not in ["oo", "ab"]:
                tensor_constraint_1[directions_list.index(direction), ...] = 1

        # Constraint 2: (map_shape[0] - 1, map_shape[1] - 1) != "oo", "cd"
        for direction in directions_list:
            if direction not in ["oo", "cd"]:
                tensor_constraint_2[..., directions_list.index(direction)] = 1

        # Constraint 3: len[path] == map_shape[0] * map_shape[1]
        for index, _ in np.ndenumerate(tensor_constraint_3):
            gain = np.sum([directions_dict[directions_list[idx]].sum() for idx in index])
            if gain == ideal_gain:
                tensor_constraint_3[index] = 1

        # Combine constraints with logical AND
        tensor_constraint = np.logical_and(
            tensor_constraint_1,
            np.logical_and(tensor_constraint_2, tensor_constraint_3)
        )

        return tensor_constraint  

    def sample_violation_indices(self, num_samples: int) -> np.ndarray:
        indices = np.array(np.where(self.tensor_constraint == 0)).T

        if num_samples > len(indices):
            raise ValueError("num_samples is too large")

        random_indices = indices[np.random.choice(len(indices), size=num_samples, replace=False)]

        return random_indices

    def sample_violation_path(self, num_samples: int = 200) -> list[list[str]]:
        random_indices = self.sample_violation_indices(num_samples)
        random_paths = [list(self.directions_list[idx] for idx in index) for index in random_indices]

        return random_paths


def get_map(map_option: int):
    if map_option == 1:
        map_targeted = np.array([[1, 4], [2, 1]])
    elif map_option == 2:
        map_targeted = np.array([[1, 4, 1], [2, 1, 1]])
    elif map_option == 3:
        map_targeted = np.array([[1, 4, 1], [2, 1, 3], [5, 2, 1]])
    else:
        raise ValueError(f"Invalid map option: {map_option}")
    
    # print()
    # print()
    # print(f"map_option: {map_option}")
    # print(f"map_targeted: {map_targeted}")
    # print()
    # print()

    return map_targeted / map_targeted.sum()