import os
import numpy as np
from _src import AckleyTF
from _src import WarcraftObjectiveTF, get_map

def create_ackley_tensor():
    """AckleyTFの評価値テンソル作成"""
    ackley = AckleyTF()
    bounds = ackley.bounds
    size = bounds[1] - bounds[0] + 1
    values = np.zeros((size, size))
    
    for x in range(bounds[0], bounds[1] + 1):
        for y in range(bounds[0], bounds[1] + 1):
            values[y - bounds[0], x - bounds[0]] = ackley.evaluate([x, y])
    
    return values

def create_warcraft_tensor(map_shape):
    """WarcraftTFの評価値テンソル作成"""
    map_option = 1 if map_shape == (2, 2) else (2 if map_shape == (2, 3) else 3)
    weight_matrix = get_map(map_option)
    warcraft = WarcraftObjectiveTF(weight_matrix)
    
    directions = ["oo", "ab", "ac", "ad", "bc", "bd", "cd"]
    map_length = map_shape[0] * map_shape[1]
    values = np.zeros([len(directions)] * map_length)
    
    for idx in np.ndindex(*([len(directions)] * map_length)):
        path = [directions[i] for i in idx]
        direction_matrix = np.array(path).reshape(map_shape)
        values[idx] = warcraft(direction_matrix)

        print(f"Index: {idx}")
    
    return values

if __name__ == "__main__":

    # Save Ackley tensor
    dir_path = "data/ackley_evaluations"
    os.makedirs(dir_path, exist_ok=True)
    
    ackley_values = create_ackley_tensor()
    np.save(os.path.join(dir_path, "dim2.npy"), ackley_values)
    print("Saved Ackley evaluation tensor")
    

    # Save Warcraft tensors
    dir_path = "data/warcraft_evaluations"
    os.makedirs(dir_path, exist_ok=True)

    for map_shape, name in [((2, 2), "map1"), ((2, 3), "map2"), ((3, 3), "map3")]:
        warcraft_values = create_warcraft_tensor(map_shape)
        np.save(os.path.join(dir_path, f"{name}.npy"), warcraft_values)
        print(f"Saved Warcraft {name} evaluation tensor")