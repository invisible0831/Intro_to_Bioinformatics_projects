from typing import List, Tuple, Union


def fitting_alignment(s: str, t: str) -> Tuple[int, str, str]:
    score, end_point_s = compute_score(s, t)
    i, j = len(t), end_point_s
    augmented_s = []
    augmented_t = []
    while True:
        if (i == 0 and j == 0) or score[i][j][1] == 'O':
            break
        if score[i][j][1] == 'L':
            augmented_s.append(s[j-1])
            augmented_t.append('-')
            j -= 1
        elif score[i][j][1] == 'U':
            augmented_s.append('-')
            augmented_t.append(t[i-1])
            i -= 1
        elif score[i][j][1] == 'UL':
            augmented_s.append(s[j-1])
            augmented_t.append(t[i-1])
            i -= 1
            j -= 1
    augmented_s.reverse()
    augmented_t.reverse()
    return score[len(t)][end_point_s][0], ''.join(augmented_s), ''.join(augmented_t)


def compute_score(s: str, t: str) -> List[List[List[Union[float, str]]]]:
    n = len(s)
    m = len(t)
    score = [[[-float('inf'), ''] for _ in range(n+1)] for _ in range(m+1)]
    max_score = -float('inf')
    end_point_s = 0
    for i in range(m + 1):
        for j in range(n + 1):
            if i == 0 and j == 0:
                score[0][0] = [0, '']
            elif i == 0 and j != 0:
                score[0][j] = [0, 'O']
            elif i != 0 and j == 0:
                score[i][0] = [-i, 'U']
            else:
                if s[j-1] == t[i-1]:
                    score[i][j] = [score[i-1][j-1][0]+1, 'UL']
                else:
                    if score[i][j][0] < score[i-1][j-1][0]-1:
                        score[i][j] = [score[i-1][j-1][0]-1, 'UL']
                    if score[i][j][0] < score[i-1][j][0]-1:
                        score[i][j] = [score[i-1][j][0]-1, 'U']
                    if score[i][j][0] < score[i][j-1][0]-1:
                        score[i][j] = [score[i][j-1][0]-1, 'L']
            if i == m and max_score < score[m][j][0]:
                max_score = score[m][j][0]
                end_point_s = j
    return score, end_point_s


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


s, t = read_input()
distance, augmented_substring_s, augmented_substring_t = fitting_alignment(
    s, t)
print(distance)
print(augmented_substring_s)
print(augmented_substring_t)
