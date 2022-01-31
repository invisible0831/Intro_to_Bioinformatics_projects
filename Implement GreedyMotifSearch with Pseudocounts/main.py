from typing import Dict, List, Tuple, Union
from pathlib import Path
import numpy as np


def read_input() -> Tuple[int, int, int, np.ndarray]:
    path = Path() / input()
    inputs = [s for s in path.read_text().split('\n') if len(s) > 0]
    k, t = map(int, inputs[0].split(' '))
    n = len(inputs[1])
    DNA_matrix = np.zeros((t, n), dtype=int)
    nucleotide_dict = {'A': 1, 'C': 2, 'G': 3, 'T': 4}
    for i in range(t):
        for j in range(n):
            DNA_matrix[i, j] = nucleotide_dict[inputs[i+1][j]]
    return k, t, n, DNA_matrix


def write_output(result: str):
    path = Path() / 'result.txt'
    path.write_text(result)


def greedy_motif_search(k: int, t: int, n: int, DNA_matrix: np.ndarray) -> np.ndarray:
    best_motifs = DNA_matrix[:, :k].copy()
    motifs = np.zeros((t, k), dtype=int)
    for i in range(n-k+1):
        motifs[0] = DNA_matrix[0, i:i+k].copy()
        for j in range(1, t):
            count_matrix = get_count_matrix(j, motifs, k)
            count_matrix += np.full((4, k), 1)
            profile_matrix = count_matrix.copy() / (j+4)
            motifs[j] = get_most_probable_k_mer(
                DNA_matrix[j], profile_matrix, k, n)
        if score(motifs, t, k) < score(best_motifs, t, k):
            best_motifs = motifs.copy()
    return best_motifs


def get_count_matrix(num_motifs: int, motifs: np.ndarray, k: int) -> np.ndarray:
    count_matrix = np.zeros((4, k), dtype=int)
    for i, motif in enumerate(motifs):
        if i == num_motifs:
            break
        for j, base in enumerate(motif):
            count_matrix[base-1, j] += 1
    return count_matrix


def get_most_probable_k_mer(DNA: np.ndarray, profile_matrix: np.ndarray, k: int, n: int) -> np.ndarray:
    most_probable_k_mer = DNA[:k].copy()
    max_probability = get_probability(0, DNA, profile_matrix, k)
    for i in range(1, n-k+1):
        probability = get_probability(i, DNA, profile_matrix, k)
        if probability > max_probability:
            max_probability = probability
            most_probable_k_mer = DNA[i:i+k].copy()
    return most_probable_k_mer


def get_probability(start: int, DNA: np.ndarray, profile_matrix: np.ndarray, k: int) -> int:
    probability = 1
    for i in range(k):
        probability *= profile_matrix[DNA[start+i]-1, i]
    return probability


def score(motifs: np.ndarray, t: int, k: int) -> int:
    count_matrix = get_count_matrix(t, motifs, k)
    return np.sum(np.full(k, t) - np.max(count_matrix, axis=0))


def get_nucleotide_best_motifs(best_motifs: np.ndarray) -> str:
    num_dict = {1: 'A', 2: 'C', 3: 'G', 4: 'T'}
    nucleotide_best_motifs = ''
    for motif in best_motifs:
        for i in motif:
            nucleotide_best_motifs += num_dict[i]
        nucleotide_best_motifs += '\n'
    return nucleotide_best_motifs[:-1]


k, t, n, DNA_matrix = read_input()
best_motifs = greedy_motif_search(k, t, n, DNA_matrix)
write_output(get_nucleotide_best_motifs(best_motifs))
