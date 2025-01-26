import os 

import numpy as np


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
    

if __name__ == "__main__":
    dir_path = "data/warcraft_constraints"

    if not os.path.exists(dir_path):
        os.makedirs(dir_path, exist_ok=True)
    
    tensor_constraint_map1 = ConstraintWarcraft((2, 2)).tensor_constraint
    np.save(os.path.join(dir_path, "map1.npy"), tensor_constraint_map1)

    tensor_constraint_map2 = ConstraintWarcraft((2, 3)).tensor_constraint
    np.save(os.path.join(dir_path, "map2.npy"), tensor_constraint_map2)

    tensor_constraint_map3 = ConstraintWarcraft((3, 3)).tensor_constraint
    np.save(os.path.join(dir_path, "map3.npy"), tensor_constraint_map3)