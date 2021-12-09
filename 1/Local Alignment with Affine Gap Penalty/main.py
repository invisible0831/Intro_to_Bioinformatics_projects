from typing import List, Tuple, Union
from enum import Enum


def local_alignment(s: str, t: str, sigma: int, epsilon: int) -> Tuple[int, str, str]:
    lower_action, middle_action, upper_action, max_score, end_point_s, end_point_t = get_lower_middle_upper(
        s, t, sigma, epsilon)
    i = end_point_t
    j = end_point_s
    action = 'M'
    while True:
        if i == 0 and j == 0:
            break
        # backward middle
        elif action == 'M':
            if middle_action[i][j] == 'O':
                break
            elif middle_action[i][j] == 'L':
                action = 'L'
            elif middle_action[i][j] == 'U':
                action = 'U'
            elif middle_action[i][j] == 'M':
                j -= 1
                i -= 1
        # backward lower
        elif action == 'L':
            if lower_action[i][j] == 'L':
                action = 'L'
            elif lower_action[i][j] == 'M':
                action = 'M'
            i -= 1
        # backward upper
        elif action == 'U':
            if upper_action[i][j] == 'U':
                action = 'U'
            elif upper_action[i][j] == 'M':
                action = 'M'
            j -= 1
    return max_score, s[j:end_point_s], t[i:end_point_t]


def get_lower_middle_upper(s: str, t: str, sigma: int, epsilon: int) -> Tuple[List[List[List[Union[float, str]]]]]:
    n = len(s)
    m = len(t)
    lower_score = [[] for _ in range(m+1)]
    lower_action = [[] for _ in range(m+1)]
    middle_score = [[] for _ in range(m+1)]
    middle_action = [[] for _ in range(m+1)]
    upper_score = [[] for _ in range(m+1)]
    upper_action = [[] for _ in range(m+1)]
    max_score = 0
    end_point_s = 0
    end_point_t = 0
    for i in range(m + 1):
        for j in range(n + 1):
            if i == 0 and j == 0:
                lower_score[0].append(0)
                lower_action[0].append('')
                upper_score[0].append(0)
                upper_action[0].append('')
                middle_score[0].append(0)
                middle_action[0].append('')
            elif i == 0 and j != 0:
                lower_score[0].append(-float('inf'))
                lower_action[0].append('')
                upper_score[0].append(-sigma)
                upper_action[0].append('M')
                middle_score[0].append(0)
                middle_action[0].append('O')
            elif i != 0 and j == 0:
                lower_score[i].append(-sigma)
                lower_action[i].append('M')
                upper_score[i].append(-float('inf'))
                upper_action[i].append('')
                middle_score[i].append(0)
                middle_action[i].append('O')
            else:
                # lower
                if middle_score[i-1][j]-sigma < lower_score[i-1][j]-epsilon:
                    lower_score[i].append(lower_score[i-1][j]-epsilon)
                    lower_action[i].append('L')
                else:
                    lower_score[i].append(middle_score[i-1][j]-sigma)
                    lower_action[i].append('M')
                # upper
                if middle_score[i][j-1]-sigma < upper_score[i][j-1]-epsilon:
                    upper_score[i].append(upper_score[i][j-1]-epsilon)
                    upper_action[i].append('U')
                else:
                    upper_score[i].append(middle_score[i][j-1]-sigma)
                    upper_action[i].append('M')
                # middle
                middle_max_score, action = 0, 'O'
                if middle_max_score < middle_score[i-1][j-1] + get_score(s[j-1], t[i-1]):
                    middle_max_score, action = middle_score[i-1][j -
                                                                 1] + get_score(s[j-1], t[i-1]), 'M'
                if middle_max_score < lower_score[i][j]:
                    middle_max_score, action = lower_score[i][j], 'L'
                if middle_max_score < upper_score[i][j]:
                    middle_max_score, action = upper_score[i][j], 'U'
                middle_score[i].append(middle_max_score)
                middle_action[i].append(action)
                if max_score < middle_max_score:
                    max_score = middle_max_score
                    end_point_s = j
                    end_point_t = i
    return lower_action, middle_action, upper_action, max_score, end_point_s, end_point_t


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
score, substring_s, substring_t = local_alignment(s, t, sigma, epsilon)
print(score)
print(substring_s)
print(substring_t)
