from dis import dis
import re
from typing import Dict, List, Tuple, Union
from pathlib import Path


def read_input() -> List[str]:
    path = Path() / input()
    return [s for s in path.read_text().split('\n')][:-1]


def write_output(result: str):
    path = Path() / 'result.txt'
    path.write_text(result)


def get_result(adjacency_list: Dict[str, List[Tuple[str, str]]]) -> str:
    node = next(iter(adjacency_list))
    edges = []
    for _ in range(len(adjacency_list)):
        neighbor, edge = adjacency_list[node][0]
        edges.append(edge[0])
        node = neighbor
    return ''.join([edge[0] for edge in edges])


def get_adjacency_list(k_mers: List[str]) -> Dict[str, List[Tuple[str, str]]]:
    adjacency_list = {}
    for k_mer in k_mers:
        adjacency_list[k_mer[:-1]] = []
        adjacency_list[k_mer[1:]] = []

    for node in adjacency_list:
        for k_mer in k_mers:
            if node in k_mer[:-1]:
                adjacency_list[node].append((k_mer[1:], k_mer))
    return adjacency_list


k_mers = read_input()
adjacency_list = get_adjacency_list(k_mers)
result = get_result(adjacency_list)
write_output(result)
