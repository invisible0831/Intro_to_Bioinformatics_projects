from typing import Dict, List, Tuple, Union
import numpy as np
import re


def read_input() -> Tuple[int, Dict[int, List[Tuple[int, int]]]]:
    num_leaves = int(input())
    adjacency_list = {}
    while True:
        line = input()
        if not line:
            break
        i, j, weight = map(int, re.split(r'->|:', line))
        if i not in adjacency_list:
            adjacency_list[i] = []
        adjacency_list[i].append((j, weight))
    return num_leaves, adjacency_list


def get_distance_matrix(num_leaves: int, adjacency_list: Dict[int, List[Tuple[int, int]]]) -> np.ndarray:
    distance_matrix = np.zeros((num_leaves, num_leaves), dtype=int)
    visited = [False for i in range(len(adjacency_list))]
    for leaf in range(num_leaves):
        visited[leaf] = True
        DFS(num_leaves, leaf, leaf, 0, adjacency_list,
            distance_matrix, visited.copy())
    return distance_matrix


def DFS(num_leaves: int, leaf: int, node: int, distance: int, adjacency_list: Dict[int, List[Tuple[int, int]]], distance_matrix: np.ndarray, visited: List[bool]):
    if node < num_leaves:
        distance_matrix[leaf, node] = distance
        distance_matrix[node, leaf] = distance
    for neighbor, weight in adjacency_list[node]:
        if not visited[neighbor]:
            neighbor_distance = distance + weight
            visited[node] = True
            DFS(num_leaves, leaf, neighbor, neighbor_distance,
                adjacency_list, distance_matrix, visited)


num_leaves, adjacency_list = read_input()
distance_matrix = get_distance_matrix(num_leaves, adjacency_list)
for row in distance_matrix:
    print(*row)
