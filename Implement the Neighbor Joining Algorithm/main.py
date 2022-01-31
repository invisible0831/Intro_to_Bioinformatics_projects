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
            distance_matrix[i, :] = np.array(
                list(map(int, re.split(r'\s+', input_file.readline())[:-1])))
    return distance_matrix, num_leaves


def write_output(adjacency_list: Dict[int, List[Tuple[int, int]]]):
    path = Path() / 'result.txt'
    s = ''
    for node in range(len(adjacency_list)):
        for neighbor, weight in adjacency_list[node]:
            s += f'{node}->{neighbor}:{weight:.3f}\n'
    path.write_text(s[:-1])


def neighbor_joining(distance_matrix: np.ndarray, num_nodes: int, node_number_row: Dict[int, int]) -> Dict[int, List[Tuple[int, int]]]:
    if len(distance_matrix) == 2:
        adjacency_list = {}
        node1, node2 = node_number_row[0], node_number_row[1]
        adjacency_list[node1] = [(node2, distance_matrix[0, 1])]
        adjacency_list[node2] = [(node1, distance_matrix[1, 0])]
        return adjacency_list

    # constructing neighbor joining matrix and find i j (neighbors)
    total_distance = np.sum(distance_matrix, axis=1)
    neighbor_joining_matrix = get_neighbor_joining_matrix(
        distance_matrix, total_distance)
    i, j = np.unravel_index(
        np.argmin(neighbor_joining_matrix, axis=None), neighbor_joining_matrix.shape)
    delta = (total_distance[i] - total_distance[j]) / \
        (len(distance_matrix) - 2)
    limb_length_i, limb_length_j = (
        distance_matrix[i, j] + delta) / 2, (distance_matrix[i, j] - delta) / 2
    # constructing new distance matrix by removing i and j rows and columns from distance matrix and add new node row and column to it
    internal_node = num_nodes
    node_num_i, node_num_j = node_number_row[i], node_number_row[j]
    new_distance_matrix = get_new_distance_matrix(distance_matrix, i, j)
    new_node_number_row = get_new_node_number_row(
        internal_node, i, j, node_number_row)

    adjacency_list = neighbor_joining(
        new_distance_matrix, num_nodes+1, new_node_number_row)

    # add i j to the tree
    add_to_tree(internal_node, node_num_i, node_num_j,
                adjacency_list, limb_length_i, limb_length_j)

    return adjacency_list


def get_neighbor_joining_matrix(distance_matrix: np.ndarray, total_distance: np.ndarray) -> np.ndarray:
    n = len(distance_matrix)
    neighbor_joining_matrix = np.zeros((n, n), dtype=float)
    for i in range(n):
        x = np.full((n, ), total_distance[i])
        neighbor_joining_matrix[i, :] -= x
        neighbor_joining_matrix[:, i] -= x
        neighbor_joining_matrix[i, i] = float('inf')
    neighbor_joining_matrix += (n-2) * distance_matrix
    return neighbor_joining_matrix


def get_new_distance_matrix(distance_matrix: np.ndarray, i: int, j: int) -> np.ndarray:
    n = len(distance_matrix)
    dist_internal_node = np.zeros((n, ))
    for k in range(n):
        dist_internal_node[k] = (
            distance_matrix[k, i] + distance_matrix[k, j] - distance_matrix[i, j]) / 2
    new_distance_matrix = np.zeros((n+1, n+1))
    new_distance_matrix[:-1, :-1] = distance_matrix
    new_distance_matrix[n, :-1] = dist_internal_node
    new_distance_matrix[:-1, n] = dist_internal_node
    new_distance_matrix = np.delete(new_distance_matrix, [i, j], axis=0)
    return np.delete(new_distance_matrix, [i, j], axis=1)


def get_new_node_number_row(internal_node: int, i: int, j: int, node_number_row: Dict[int, int]) -> Dict[int, int]:
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


def add_to_tree(internal_node: int, i: int, j: int, adjacency_list: Dict[int, List[Tuple[int, int]]], limb_length_i: int, limb_length_j: int):
    adjacency_list[internal_node].append((i, limb_length_i))
    adjacency_list[internal_node].append((j, limb_length_j))
    if i not in adjacency_list:
        adjacency_list[i] = []
    if j not in adjacency_list:
        adjacency_list[j] = []
    adjacency_list[i].append((internal_node, limb_length_i))
    adjacency_list[j].append((internal_node, limb_length_j))


distance_matrix, num_leaves = read_input()
node_number_row = {i: i for i in range(num_leaves)}
adjacency_list = neighbor_joining(distance_matrix, num_leaves, node_number_row)
write_output(adjacency_list)
