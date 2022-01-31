from typing import List, Tuple
from enum import Enum


def get_path(s: str, t: str, source: Tuple[int, int], sink: Tuple[int, int], sigma: int) -> List[Tuple[int, int]]:
    # return vertexes from source to sink in the longest path except source
    path = []
    if source[0] == sink[0]:
        # when source and sink are in the same row
        for j in range(source[1]+1, sink[1]+1):
            path.append((source[0], j))
        return path
    if source[1] == sink[1]:
        # when source and sink are in the same column
        for i in range(source[0]+1, sink[0]+1):
            path.append((i, source[1]))
        return path
    middle_vertex = get_middle_vertex(s, t, source, sink, sigma)
    if source == middle_vertex:
        '''because there is no vertex in the mid_column and longest path downer than the source and, sink and
        source are in consecutive columns then the next edge is diagonal'''
        path.append((source[0]+1, source[1]+1))
        path = path + get_path(s, t, (source[0]+1, source[1]+1), sink, sigma)
        return path
    prefix_path = get_path(s, t, source, middle_vertex, sigma)
    suffix_path = get_path(s, t, middle_vertex, sink, sigma)
    path = prefix_path + suffix_path
    return path


def get_middle_vertex(s: str, t: str, source: Tuple[int, int], sink: Tuple[int, int], sigma: int) -> Tuple[int, int]:
    # downest vertex in the mid_column that is in the longest path
    mid_column = int((sink[1] + source[1]) / 2)
    prefix_score = compute_score(
        s[source[1]:mid_column], t[source[0]:sink[0]], sigma)
    suffix_score = compute_score(
        s[mid_column:sink[1]][::-1], t[source[0]:sink[0]][::-1], sigma)
    suffix_score.reverse()
    max_length = -float('inf')
    mid_row = 0
    for i in range(source[0], sink[0]+1):
        score_index = i - source[0]
        if max_length <= prefix_score[score_index] + suffix_score[score_index]:
            max_length = prefix_score[score_index] + suffix_score[score_index]
            mid_row = i
    return (mid_row, mid_column)


def compute_score(s: str, t: str, sigma: int) -> List[int]:
    n = len(s)
    m = len(t)
    score = [-float('inf') for _ in range(m+1)]
    score_past = [-float('inf') for _ in range(m+1)]
    for j in range(n + 1):
        for i in range(m + 1):
            if j == 0:
                score[i] = -i * sigma
            elif i == 0 and j != 0:
                score_past[0], score[0] = score[0], -j * sigma
            else:
                score_past[i] = score[i]
                score[i] = max(
                    score_past[i-1] + get_score(s[j-1], t[i-1]), score_past[i] - sigma, score[i-1] - sigma)
    return score


def compute_augmented_strings(s: str, t: str, sigma: int) -> Tuple[int, str, str]:
    path = get_path(s, t, (0, 0), (len(t), len(s)), sigma)
    i, j = 0, 0
    score = 0
    augmented_s = ''
    augmented_t = ''
    while True:
        next_vertex = path.pop(0)
        if next_vertex[0] == i+1 and next_vertex[1] == j+1:
            i, j = next_vertex
            score += get_score(s[j-1], t[i-1])
            augmented_s += s[j-1]
            augmented_t += t[i-1]
        elif next_vertex[0] == i+1 and next_vertex[1] == j:
            i, j = next_vertex
            score -= sigma
            augmented_s += '-'
            augmented_t += t[i-1]
        elif next_vertex[0] == i and next_vertex[1] == j+1:
            i, j = next_vertex
            score -= sigma
            augmented_s += s[j-1]
            augmented_t += '-'
        if i == len(t) and j == len(s):
            break
    return score, augmented_s, augmented_t


def read_input() -> Tuple[str, str]:
    s = ''
    t = ''
    line = input()
    while True:
        line = input()
        if line[0] != '>':
            s = s + line
        else:
            break
    while True:
        line = input()
        if not line:
            break
        t = t + line
    return s, t


class Alphabet(Enum):
    A = 0
    C = 1
    D = 2
    E = 3
    F = 4
    G = 5
    H = 6
    I = 7
    K = 8
    L = 9
    M = 10
    N = 11
    P = 12
    Q = 13
    R = 14
    S = 15
    T = 16
    V = 17
    W = 18
    Y = 19


def init_BLOSUM62_scoring_matrix(scoring_matrix: List[List[int]]) -> None:
    scoring_matrix.append(
        [4, 0, -2, -1, -2, 0, -2, -1, -1, -1, -1, -2, -1, -1, -1, 1, 0, 0, -3, -2])
    scoring_matrix.append(
        [0, 9, -3, -4, -2, -3, -3, -1, -3, -1, -1, -3, -3, -3, -3, -1, -1, -1, -2, -2])
    scoring_matrix.append(
        [-2, -3, 6, 2, -3, -1, -1, -3, -1, -4, -3, 1, -1, 0, -2, 0, -1, -3, -4, -3])
    scoring_matrix.append(
        [-1, -4, 2, 5, -3, -2, 0, -3, 1, -3, -2, 0, -1, 2, 0, 0, -1, -2, -3, -2])
    scoring_matrix.append(
        [-2, -2, -3, -3, 6, -3, -1, 0, -3, 0, 0, -3, -4, -3, -3, -2, -2, -1, 1, 3])
    scoring_matrix.append(
        [0, -3, -1, -2, -3, 6, -2, -4, -2, -4, -3, 0, -2, -2, -2, 0, -2, -3, -2, -3])
    scoring_matrix.append(
        [-2, -3, -1, 0, -1, -2, 8, -3, -1, -3, -2, 1, -2, 0, 0, -1, -2, -3, -2, 2])
    scoring_matrix.append(
        [-1, -1, -3, -3, 0, -4, -3, 4, -3, 2, 1, -3, -3, -3, -3, -2, -1, 3, -3, -1])
    scoring_matrix.append(
        [-1, -3, -1, 1, -3, -2, -1, -3, 5, -2, -1, 0, -1, 1, 2, 0, -1, -2, -3, -2])
    scoring_matrix.append(
        [-1, -1, -4, -3, 0, -4, -3, 2, -2, 4, 2, -3, -3, -2, -2, -2, -1, 1, -2, -1])
    scoring_matrix.append(
        [-1, -1, -3, -2, 0, -3, -2, 1, -1, 2, 5, -2, -2, 0, -1, -1, -1, 1, -1, -1])
    scoring_matrix.append(
        [-2, -3, 1, 0, -3, 0, 1, -3, 0, -3, -2, 6, -2, 0, 0, 1, 0, -3, -4, -2])
    scoring_matrix.append(
        [-1, -3, -1, -1, -4, -2, -2, -3, -1, -3, -2, -2, 7, -1, -2, -1, -1, -2, -4, -3])
    scoring_matrix.append(
        [-1, -3, 0, 2, -3, -2, 0, -3, 1, -2, 0, 0, -1, 5, 1, 0, -1, -2, -2, -1])
    scoring_matrix.append(
        [-1, -3, -2, 0, -3, -2, 0, -3, 2, -2, -1, 0, -2, 1, 5, -1, -1, -3, -3, -2])
    scoring_matrix.append(
        [1, -1, 0, 0, -2, 0, -1, -2, 0, -2, -1, 1, -1, 0, -1, 4, 1, -2, -3, -2])
    scoring_matrix.append(
        [0, -1, -1, -1, -2, -2, -2, -1, -1, -1, -1, 0, -1, -1, -1, 1, 5, 0, -2, -2])
    scoring_matrix.append(
        [0, -1, -3, -2, -1, -3, -3, 3, -2, 1, 1, -3, -2, -2, -3, -2, 0, 4, -3, -1])
    scoring_matrix.append(
        [-3, -2, -4, -3, 1, -2, -2, -3, -3, -2, -1, -4, -4, -2, -3, -3, -2, -3, 11, 2])
    scoring_matrix.append(
        [-2, -2, -3, -2, 3, -3, 2, -1, -2, -1, -1, -2, -3, -1, -2, -2, -2, -1, 2, 7])


def get_score(v: str, w: str) -> float:
    return BLOSUM62_scoring_matrix[Alphabet[v].value][Alphabet[w].value]


BLOSUM62_scoring_matrix = []
init_BLOSUM62_scoring_matrix(BLOSUM62_scoring_matrix)


sigma = 5  # indel penalty
s, t = read_input()
score, augmented_s, augmented_t = compute_augmented_strings(s, t, sigma)
print(score)
print(augmented_s)
print(augmented_t)
