from typing import List, Tuple, Union


def edit_distance(s: str, t: str) -> Tuple[int, str, str]:
    distance = compute_distance(s, t)
    i, j = len(t), len(s)
    augmented_s = []
    augmented_t = []
    while True:
        if i == 0 and j == 0:
            break
        if distance[i][j][1] == 'L':
            augmented_s.append(s[j-1])
            augmented_t.append('-')
            j -= 1
        elif distance[i][j][1] == 'U':
            augmented_s.append('-')
            augmented_t.append(t[i-1])
            i -= 1
        elif distance[i][j][1] == 'UL':
            augmented_s.append(s[j-1])
            augmented_t.append(t[i-1])
            i -= 1
            j -= 1
    augmented_s.reverse()
    augmented_t.reverse()
    return distance[len(t)][len(s)][0], ''.join(augmented_s), ''.join(augmented_t)


def compute_distance(s: str, t: str) -> List[List[List[Union[float, str]]]]:
    n = len(s)
    m = len(t)
    distance = [[[float('inf'), ''] for _ in range(n+1)] for _ in range(m+1)]
    for i in range(m + 1):
        for j in range(n + 1):
            if i == 0 and j == 0:
                distance[0][0] = [0, '']
            elif i == 0 and j != 0:
                distance[0][j] = [j, 'L']
            elif i != 0 and j == 0:
                distance[i][0] = [i, 'U']
            else:
                if s[j-1] == t[i-1]:
                    distance[i][j] = [distance[i-1][j-1][0], 'UL']
                else:
                    if distance[i][j][0] > distance[i-1][j-1][0]+1:
                        distance[i][j] = [distance[i-1][j-1][0]+1, 'UL']
                    if distance[i][j][0] > distance[i-1][j][0]+1:
                        distance[i][j] = [distance[i-1][j][0]+1, 'U']
                    if distance[i][j][0] > distance[i][j-1][0]+1:
                        distance[i][j] = [distance[i][j-1][0]+1, 'L']
    return distance


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
distance, augmented_s, augmented_t = edit_distance(s, t)
print(distance)
print(augmented_s)
print(augmented_t)
