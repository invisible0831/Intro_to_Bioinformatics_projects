from dis import dis
from typing import Dict, List, Tuple, Union
from pathlib import Path
import numpy as np


def read_input() -> Tuple[int, int, int, np.ndarray]:
    path = Path() / input()
    inputs = [s for s in path.read_text().split('\n') if len(s) > 0]
    k = int(inputs[0])
    n = len(inputs[1])
    t = len(inputs) - 1
    DNA_matrix = np.zeros((t, n))
    nucleotide_dict = {'A': 1, 'C': 2, 'G': 3, 'T': 4}
    for i in range(t):
        for j in range(n):
            DNA_matrix[i, j] = nucleotide_dict[inputs[i+1][j]]
    return k, t, n, DNA_matrix


def write_output(result: str):
    path = Path() / 'result.txt'
    path.write_text(result)


def branch_and_bound_median_search(k: int, t: int, n: int, DNA_matrix: int) -> str:
    k_mer = np.full(k, 1)
    median_string = k_mer.copy()
    best_distance = float('inf')
    depth = 1
    while depth > 0:
        if depth < k:
            prefix = k_mer[:depth]
            optimistic_distance = total_distance(prefix, DNA_matrix)
            if optimistic_distance > best_distance:
                k_mer, depth = bypass(k_mer, depth, k, 4)
            else:
                k_mer, depth = next_vertex(k_mer, depth, k, 4)
        else:
            if best_distance > total_distance(k_mer, DNA_matrix):
                best_distance = total_distance(k_mer, DNA_matrix)
                median_string = k_mer.copy()
            k_mer, depth = next_vertex(k_mer, depth, k, 4)
    return get_nucleotide_median_string(median_string)


def next_vertex(k_mer: np.ndarray, depth: int, total_depth: int, num_children: int) -> Tuple[np.ndarray, int]:
    if depth < total_depth:
        k_mer[depth] = 1
        return k_mer, depth+1
    for i in range(total_depth, 0, -1):
        if k_mer[i-1] < num_children:
            k_mer[i-1] += 1
            return k_mer, i
    return k_mer, 0


def bypass(k_mer: np.ndarray, depth: int, total_depth: int, num_children: int) -> Tuple[np.ndarray, int]:
    for i in range(depth, 0, -1):
        if k_mer[i-1] < num_children:
            k_mer[i-1] += 1
            return k_mer, i
    return k_mer, 0


def total_distance(s: np.ndarray, DNA_matrix: np.ndarray) -> int:
    distance = 0
    for DNA in DNA_matrix:
        distance += minimum_hamming_distance(s, DNA)
    return distance


def minimum_hamming_distance(s: np.ndarray, DNA: np.ndarray) -> int:
    l = len(s)
    distance = float('inf')
    for i in range(len(DNA)-l+1):
        distance = min(distance, hamming_distance(s, DNA[i:i+l]))
    return distance


def hamming_distance(s1: np.ndarray, s2: np.ndarray) -> int:
    return len(s1) - np.sum(s1 == s2)


def get_nucleotide_median_string(median_string: np.ndarray) -> str:
    num_dict = {1: 'A', 2: 'C', 3: 'G', 4: 'T'}
    nucleotide_median_string = ''
    for i in median_string:
        nucleotide_median_string += num_dict[i]
    return nucleotide_median_string


k, t, n, DNA_matrix = read_input()
median_string = branch_and_bound_median_search(k, t, n, DNA_matrix)
write_output(median_string)
