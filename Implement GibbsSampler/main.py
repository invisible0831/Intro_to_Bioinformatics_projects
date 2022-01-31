from typing import Dict, List, Tuple, Union
from pathlib import Path
import numpy as np
import random


def read_input() -> Tuple[int, int, int, int, np.ndarray]:
    path = Path() / input()
    inputs = [s for s in path.read_text().split('\n') if len(s) > 0]
    k, t, N = map(int, inputs[0].split(' '))
    n = len(inputs[1])
    DNA_matrix = np.zeros((t, n), dtype=int)
    nucleotide_dict = {'A': 1, 'C': 2, 'G': 3, 'T': 4}
    for i in range(t):
        for j in range(n):
            DNA_matrix[i, j] = nucleotide_dict[inputs[i+1][j]]
    return k, t, n, N, DNA_matrix


def write_output(result: str):
    path = Path() / 'result.txt'
    path.write_text(result)


def iterative_gibbs_sampler(run_times: int, k: int, t: int, n: int, N: int, DNA_matrix: np.ndarray) -> np.ndarray:
    best_motifs = gibbs_sampler(k, t, n, N, DNA_matrix)
    for _ in range(1, run_times):
        motifs = gibbs_sampler(k, t, n, N, DNA_matrix)
        if score(motifs, t, k) < score(best_motifs, t, k):
            best_motifs = motifs
    return best_motifs


def gibbs_sampler(k: int, t: int, n: int, N: int, DNA_matrix: np.ndarray) -> np.ndarray:
    motifs = get_random_motifs(k, t, n, DNA_matrix)
    best_motifs = motifs.copy()
    for _ in range(N):
        i = random.randint(0, t-1)
        count_matrix = get_count_matrix(i, motifs, k)
        count_matrix += np.full((4, k), 1)
        profile_matrix = count_matrix.copy() / (t-1+4)
        k_mer_probabilities = get_k_mer_probabilities(
            DNA_matrix[i], profile_matrix, k, n)
        motifs[i] = get_random_k_mer(
            DNA_matrix[i], k_mer_probabilities, k, n)
        if score(motifs, t, k) < score(best_motifs, t, k):
            best_motifs = motifs
    return motifs


def get_random_motifs(k: int, t: int, n: int, DNA_matrix: np.ndarray) -> np.ndarray:
    motifs = np.zeros((t, k), dtype=int)
    for i in range(t):
        idx = random.randint(0, n-k)
        motifs[i] = DNA_matrix[i, idx:idx+k].copy()
    return motifs


def get_count_matrix(except_num: int, motifs: np.ndarray, k: int) -> np.ndarray:
    count_matrix = np.zeros((4, k), dtype=int)
    for idx, motif in enumerate(motifs):
        if idx == except_num:
            continue
        for j, base in enumerate(motif):
            count_matrix[base-1, j] += 1
    return count_matrix


def get_k_mer_probabilities(DNA: np.ndarray, profile_matrix: np.ndarray, k: int, n: int) -> np.ndarray:
    k_mer_probabilities = np.array(
        [get_probability(i, DNA, profile_matrix, k) for i in range(n-k+1)])
    return k_mer_probabilities / np.sum(k_mer_probabilities)


def get_probability(start: int, DNA: np.ndarray, profile_matrix: np.ndarray, k: int) -> int:
    probability = 1
    for i in range(k):
        probability *= profile_matrix[DNA[start+i]-1, i]
    return probability


def get_random_k_mer(DNA_matrix: np.ndarray, k_mer_probabilities: np.ndarray, k: int, n: int) -> np.ndarray:
    random_num = random.random()
    for idx in range(n-k+1):
        if random_num < np.sum(k_mer_probabilities[:idx+1]):
            break
    return DNA_matrix[idx:idx+k]


def score(motifs: np.ndarray, t: int, k: int) -> int:
    count_matrix = get_count_matrix(t+1, motifs, k)
    return np.sum(np.full(k, t) - np.max(count_matrix, axis=0))


def get_nucleotide_best_motifs(best_motifs: np.ndarray) -> str:
    num_dict = {1: 'A', 2: 'C', 3: 'G', 4: 'T'}
    nucleotide_best_motifs = ''
    for motif in best_motifs:
        for i in motif:
            nucleotide_best_motifs += num_dict[i]
        nucleotide_best_motifs += '\n'
    return nucleotide_best_motifs[:-1]


k, t, n, N, DNA_matrix = read_input()
run_times = 20
best_motifs = iterative_gibbs_sampler(run_times, k, t, n, N, DNA_matrix)
write_output(get_nucleotide_best_motifs(best_motifs))
