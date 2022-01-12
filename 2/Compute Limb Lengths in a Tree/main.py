from typing import Dict, List, Tuple
import numpy as np
import re


def read_input() -> Tuple[int, int, np.ndarray]:
    num_leaves = int(input())
    leaf = int(input())
    distance_matrix = np.zeros((num_leaves, num_leaves), dtype=int)
    for i in range(num_leaves):
        distance_matrix[i, :] = np.array(
            list(map(int, re.split(r'\s+', input()))))
    return num_leaves, leaf, distance_matrix


def get_limb_length(num_leaves: int, leaf: int, distance_matrix: np.ndarray) -> int:
    limb_length = float('inf')
    for i in range(num_leaves):
        if i != leaf:
            for k in range(i+1, num_leaves):
                if k != leaf:
                    d = (
                        distance_matrix[leaf, i]+distance_matrix[leaf, k]-distance_matrix[i, k])/2
                    limb_length = min(limb_length, d)
    return int(limb_length)


num_leaves, leaf, distance_matrix = read_input()
limb_length = get_limb_length(num_leaves, leaf, distance_matrix)
print(limb_length)
