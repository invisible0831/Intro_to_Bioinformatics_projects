from typing import Dict, List, Tuple, Union
import numpy as np
import re


def read_input() -> np.ndarray:
    num_leaves = int(input())
    distance_matrix = np.zeros((num_leaves, num_leaves), dtype=int)
    for i in range(num_leaves):
        distance_matrix[i, :] = np.array(
            list(map(int, re.split(r'\s+', input()))))
    return distance_matrix, num_leaves


def get_limb_length(j: int, distance_matrix: np.ndarray) -> Tuple[int, int, int]:
    limb_length = float('inf')
    for i in range(len(distance_matrix)):
        if i != j:
            for k in range(i+1, len(distance_matrix)):
                if k != j:
                    d = (
                        distance_matrix[j, i]+distance_matrix[j, k]-distance_matrix[i, k])/2  # dij + djk - dik  i -> j -> k
                    if d < limb_length:
                        l = i
                        m = k
                        limb_length = d
    return l, m, int(limb_length)


def additive_phylogeny(distance_matrix: np.ndarray, num_leaves: int) -> Tuple[Dict[int, List[Tuple[int, int]]], int]:
    if len(distance_matrix) == 2:
        adjacency_list = {}
        adjacency_list[0] = [(1, distance_matrix[0, 1])]
        adjacency_list[1] = [(0, distance_matrix[1, 0])]
        return adjacency_list, num_leaves
    j = len(distance_matrix) - 1
    i, k, limb_length = get_limb_length(j, distance_matrix)  # i -> j -> k
    distance_internal_node_from_i = distance_matrix[i, j] - limb_length
    adjacency_list, internal_node = additive_phylogeny(
        distance_matrix[:-1, :-1], num_leaves)
    add_internal_node(
        internal_node, distance_internal_node_from_i, i, k, adjacency_list)
    adjacency_list[internal_node].append((j, limb_length))
    adjacency_list[j] = [(internal_node, limb_length)]
    return adjacency_list, internal_node+1


def add_internal_node(internal_node: int, distance: int, i: int, k: int, adjacency_list: Dict[int, List[Tuple[int, int]]]):
    visited = {node: False for node in adjacency_list}
    node1, node2, weight, distance_node1_from_i = DFS(
        distance, i, k, adjacency_list, visited, 0)
    adjacency_list[node1].remove((node2, weight))
    adjacency_list[node2].remove((node1, weight))
    weight1 = distance - distance_node1_from_i
    weight2 = weight - weight1
    adjacency_list[node1].append((internal_node, weight1))
    adjacency_list[node2].append((internal_node, weight2))
    adjacency_list[internal_node] = [(node1, weight1), (node2, weight2)]


def DFS(distance: int, node: int, k: int, adjacency_list: Dict[int, List[Tuple[int, int]]], visited: List[bool], distance_node_from_i: int) -> Union[bool, Tuple[int, int, int, int]]:
    visited[node] = True
    if node == k:
        return True
    for neighbor, weight in adjacency_list[node]:
        if not visited[neighbor]:
            distance_neighbor_from_i = distance_node_from_i + weight
            result = DFS(distance, neighbor, k, adjacency_list,
                         visited, distance_neighbor_from_i)
            if isinstance(result, bool) and result:
                if distance_node_from_i <= distance <= distance_neighbor_from_i:
                    return node, neighbor, weight, distance_node_from_i
                return True
            elif isinstance(result, tuple):
                return result
    return False


distance_matrix, num_leaves = read_input()
adjacency_list, x = additive_phylogeny(distance_matrix, num_leaves)
for node in range(len(adjacency_list)):
    for neighbor, weight in adjacency_list[node]:
        print(f'{node}->{neighbor}:{weight}')
