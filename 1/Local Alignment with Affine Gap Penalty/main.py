from typing import List, Tuple, Union
from enum import Enum


def local_alignment(s: str, t: str, sigma: int, epsilon: int) -> Tuple[int, str, str]:
    lower, middle, upper = get_lower_middle_upper(s, t, sigma, epsilon)
    end_point_s, end_point_t = get_end_point(middle, len(t), len(s))
    i = end_point_t
    j = end_point_s
    augmented_s = []
    augmented_t = []
    action = 'M'
    while True:
        if i == 0 and j == 0:
            break
        # backward middle
        elif action == 'M':
            if middle[i][j][1] == 'O':
                break
            elif middle[i][j][1] == 'L':
                action = 'L'
            elif middle[i][j][1] == 'U':
                action = 'U'
            elif middle[i][j][1] == 'M':
                augmented_s.append(s[j-1])
                augmented_t.append(t[i-1])
                j -= 1
                i -= 1
        # backward lower
        elif action == 'L':
            if lower[i][j][1] == 'L':
                action = 'L'
            elif lower[i][j][1] == 'M':
                action = 'M'
            augmented_s.append('-')
            augmented_t.append(t[i-1])
            i -= 1
        # backward upper
        elif action == 'U':
            if upper[i][j][1] == 'U':
                action = 'U'
            elif upper[i][j][1] == 'M':
                action = 'M'
            augmented_s.append(s[j-1])
            augmented_t.append('-')
            j -= 1
    augmented_s.reverse()
    augmented_t.reverse()
    return middle[end_point_t][end_point_s][0], ''.join(augmented_s), ''.join(augmented_t)


def get_end_point(middle: List[List[List[Union[float, str]]]], m: int, n: int) -> Tuple[int, int]:
    max_score = 0
    end_point_s = 0
    end_point_t = 0
    for i in range(m+1):
        for j in range(n+1):
            if max_score < middle[i][j][0]:
                max_score = middle[i][j][0]
                end_point_s = j
                end_point_t = i
    return end_point_s, end_point_t


def get_lower_middle_upper(s: str, t: str, sigma: int, epsilon: int) -> Tuple[List[List[List[Union[float, str]]]]]:
    n = len(s)
    m = len(t)
    lower = [[[-float('inf'), ''] for _ in range(n+1)] for _ in range(m+1)]
    middle = [[[-float('inf'), ''] for _ in range(n+1)] for _ in range(m+1)]
    upper = [[[-float('inf'), ''] for _ in range(n+1)] for _ in range(m+1)]
    for i in range(m + 1):
        for j in range(n + 1):
            if i == 0 and j == 0:
                lower[0][0] = [0, '']
                upper[0][0] = [0, '']
                middle[0][0] = [0, '']
            elif i == 0 and j != 0:
                upper[0][j] = [-sigma, 'M']
                middle[0][j] = [0, 'O']
            elif i != 0 and j == 0:
                lower[i][0] = [-sigma, 'M']
                middle[i][0] = [0, 'O']
            else:
                # lower
                if lower[i][j][0] < lower[i-1][j][0]-epsilon:
                    lower[i][j] = [lower[i-1][j][0]-epsilon, 'L']
                if lower[i][j][0] < middle[i-1][j][0]-sigma:
                    lower[i][j] = [middle[i-1][j][0]-sigma, 'M']
                # upper
                if upper[i][j][0] < upper[i][j-1][0]-epsilon:
                    upper[i][j] = [upper[i][j-1][0]-epsilon, 'U']
                if upper[i][j][0] < middle[i][j-1][0]-sigma:
                    upper[i][j] = [middle[i][j-1][0]-sigma, 'M']
                # middle
                middle[i][j] = [0, 'O']
                if middle[i][j][0] < middle[i-1][j-1][0] + get_score(s[j-1], t[i-1]):
                    middle[i][j] = [middle[i-1][j-1][0] +
                                    get_score(s[j-1], t[i-1]), 'M']
                if middle[i][j][0] < lower[i][j][0]:
                    middle[i][j] = [lower[i][j][0], 'L']
                if middle[i][j][0] < upper[i][j][0]:
                    middle[i][j] = [upper[i][j][0], 'U']
    return lower, middle, upper


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

s, t = read_input()
sigma = 11  # gap opening penalty
epsilon = 1  # gap extension penalty
# get_lower_middle_upper(s, t, sigma, epsilon)
score, augmented_s, augmented_t = local_alignment(s, t, sigma, epsilon)
# score, augmented_s, augmented_t = local_alignment(
#     'PLEASANTLY', 'MEANLY', sigma, epsilon)
print(score)
print(augmented_s)
print(augmented_t)
