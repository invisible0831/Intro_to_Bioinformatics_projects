from typing import Dict, List, Tuple, Union
import numpy as np
import re
from pathlib import Path


def read_input() -> np.ndarray:
    path = Path() / input()
    with open(path, 'r') as input_file:
        num_leaves = int(input_file.readline())
        distance_matrix = np.zeros((num_leaves, num_leaves), dtype=float)
        for i in range(num_leaves):
            row = re.split(r'\s+', input_file.readline())
            if row[-1] == '':
                row.pop()
            distance_matrix[i, :] = np.array(list(map(int, row)))
    return distance_matrix, num_leaves


def write_output(adjacency_list: Dict[int, List[Tuple[int, int]]]):
    path = Path() / 'result.txt'
    s = ''
    for node in range(len(adjacency_list)):
        for neighbor, weight in adjacency_list[node]:
            s += f'{node}->{neighbor}:{weight:.3f}\n'
    path.write_text(s[:-1])


def UPGMA(distance_matrix: np.ndarray, num_nodes: int, node_number_row: Dict[int, int], age: Dict[int, float], cluster_size: Dict[int, int]) -> Dict[int, List[Tuple[int, int]]]:
    adjacency_list = {}
    while len(distance_matrix) > 1:
        np.fill_diagonal(distance_matrix, float('inf'))
        i, j = np.unravel_index(
            np.argmin(distance_matrix, axis=None), distance_matrix.shape)
        np.fill_diagonal(distance_matrix, 0.)
        internal_node = num_nodes
        num_nodes += 1
        age[internal_node] = distance_matrix[i, j] / 2

        # add i j to the tree
        node_num_i, node_num_j = node_number_row[i], node_number_row[j]
        weight_i, weight_j = age[internal_node] - \
            age[node_num_i], age[internal_node]-age[node_num_j]
        add_to_tree(internal_node, node_num_i, node_num_j,
                    adjacency_list, weight_i, weight_j)

        # constructing new distance matrix by removing i and j rows and columns from distance matrix and add new node row and column to it
        distance_matrix = get_distance_matrix(
            distance_matrix.copy(), i, j, node_number_row, cluster_size)
        node_number_row = get_node_number_row(
            internal_node, i, j, node_number_row.copy())
        cluster_size[internal_node] = cluster_size[node_num_i] + \
            cluster_size[node_num_j]

    return adjacency_list


def add_to_tree(internal_node: int, i: int, j: int, adjacency_list: Dict[int, List[Tuple[int, int]]], weight_i: float, weight_j: float):
    adjacency_list[internal_node] = [(i, weight_i), (j, weight_j)]
    if i not in adjacency_list:
        adjacency_list[i] = []
    if j not in adjacency_list:
        adjacency_list[j] = []
    adjacency_list[i].append((internal_node, weight_i))
    adjacency_list[j].append((internal_node, weight_j))


def get_distance_matrix(distance_matrix: np.ndarray, i: int, j: int, node_number_row: Dict[int, int], cluster_size: Dict[int, int]) -> np.ndarray:
    n = len(distance_matrix)
    dist_internal_node = np.zeros((n, ))
    node_num_i, node_num_j = node_number_row[i], node_number_row[j]
    for k in range(n):
        dist_internal_node[k] = (cluster_size[node_num_i] * distance_matrix[i, k] + cluster_size[node_num_j]
                                 * distance_matrix[j, k]) / (cluster_size[node_num_i] + cluster_size[node_num_j])
    new_distance_matrix = np.zeros((n+1, n+1))
    new_distance_matrix[:-1, :-1] = distance_matrix
    new_distance_matrix[n, :-1] = dist_internal_node
    new_distance_matrix[:-1, n] = dist_internal_node
    new_distance_matrix = np.delete(new_distance_matrix, [i, j], axis=0)
    return np.delete(new_distance_matrix, [i, j], axis=1)


def get_node_number_row(internal_node: int, i: int, j: int, node_number_row: Dict[int, int]) -> Dict[int, int]:
    n = len(node_number_row)
    new_node_number_row = {}
    for k in range(n):
        if k < min(i, j):
            new_node_number_row[k] = node_number_row[k]
        elif min(i, j) < k < max(i, j):
            new_node_number_row[k-1] = node_number_row[k]
        elif k > max(i, j):
            new_node_number_row[k-2] = node_number_row[k]
    new_node_number_row[n-2] = internal_node
    return new_node_number_row


distance_matrix, num_leaves = read_input()
node_number_row = {i: i for i in range(num_leaves)}
age = {i: 0 for i in range(num_leaves)}
cluster_size = {i: 1 for i in range(num_leaves)}
adjacency_list = UPGMA(distance_matrix, num_leaves,
                       node_number_row, age, cluster_size)
write_output(adjacency_list)
