from typing import Dict, List, Tuple, Union
from pathlib import Path
import numpy as np
import re


def read_input() -> Tuple[str, List[str], List[str], Dict[str, int], Dict[str, int]]:
    path = Path() / input()
    inputs = [s for s in path.read_text().split('\n') if len(s)
              > 0 and s[0] != '-']
    observations = inputs[0]
    alphabet = re.split('\s+', inputs[1])
    states = {idx: state for idx, state in enumerate(
        re.split('\s+', inputs[2]))}
    transition = {}
    emission = {}
    i = 4
    while True:
        if inputs[i][0] == '\t':
            break
        for idx, probability in enumerate(re.split('\s+', inputs[i])[1:]):
            transition[states[idx]+inputs[i][0]] = float(probability)
        i += 1
    i += 1
    while True:
        if i == len(inputs):
            break
        for idx, probability in enumerate(re.split('\s+', inputs[i])[1:]):
            emission[alphabet[idx]+inputs[i][0]] = float(probability)
        i += 1
    return observations, alphabet, states, transition, emission


def write_output(result: str):
    path = Path() / 'result.txt'
    path.write_text(result)


def viterbi(observations: str, states: Dict[int, str], transition: Dict[str, int], emission: Dict[str, int]) -> str:
    m = np.zeros((len(observations), len(states)))
    action = np.zeros((len(observations), len(states)), dtype=int)
    for idx, state in states.items():
        m[0, idx] = emission[observations[0]+state] / len(states)
    for i in range(1, (len(observations))):
        for idx, state in states.items():
            m[i, idx] = 0
            for pre_idx, pre_state in states.items():
                if emission[observations[i]+state] * transition[state+pre_state] * m[i-1, pre_idx] > m[i, idx]:
                    m[i, idx] = emission[observations[i]+state] * \
                        transition[state+pre_state] * m[i-1, pre_idx]
                    action[i-1, idx] = pre_idx

    idx = int(np.argmax(m[len(observations)-1]))
    path = states[idx]
    for i in range(len(observations)-2, -1, -1):
        idx = action[i, idx]
        path = states[idx] + path
    return path


observations, alphabet, states, transition, emission = read_input()
write_output(viterbi(observations, states, transition, emission))
