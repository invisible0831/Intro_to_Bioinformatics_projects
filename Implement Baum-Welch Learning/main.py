from turtle import backward
from typing import Dict, List, Tuple, Union
from pathlib import Path
import numpy as np
import re


def read_input() -> Tuple[int, str, List[str], List[str], Dict[str, int], Dict[str, int]]:
    path = Path() / input()
    inputs = [s for s in path.read_text().split('\n') if len(s)
              > 0 and s[0] != '-']
    run_times = int(inputs[0])
    observations = inputs[1]
    alphabet = re.split('\s+', inputs[2])
    states = {idx: state for idx, state in enumerate(
        re.split('\s+', inputs[3]))}
    transition = {}
    emission = {}
    i = 5
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
    return run_times, observations, alphabet, states, transition, emission


def write_output(result: str):
    path = Path() / 'result.txt'
    path.write_text(result)


def get_forward(observations: str, states: Dict[int, str], transition: Dict[str, int], emission: Dict[str, int]) -> np.ndarray:
    forward = np.zeros((len(observations), len(states)))
    for idx, state in states.items():
        forward[0, idx] = emission[observations[0]+state] / len(states)
    for i in range(1, (len(observations))):
        for idx, state in states.items():
            forward[i, idx] = emission[observations[i]+state]
            t = 0
            for pre_idx, pre_state in states.items():
                t += transition[state+pre_state] * forward[i-1, pre_idx]
            forward[i, idx] *= t
    return forward


def get_backward(observations: str, states: Dict[int, str], transition: Dict[str, int], emission: Dict[str, int]) -> np.ndarray:
    backward = np.zeros((len(observations), len(states)))
    backward[len(observations)-1] = np.full(len(states), 1)
    for i in range(len(observations)-2, -1, -1):
        for pre_idx, pre_state in states.items():
            backward[i, pre_idx] = 0
            for idx, state in states.items():
                backward[i, pre_idx] += emission[observations[i+1]+state] * \
                    backward[i+1, idx] * transition[state+pre_state]
    return backward


def get_node_responsibility(forward: np.ndarray, backward: np.ndarray, observations: str) -> np.ndarray:
    t = forward * backward
    normalization_matrix = np.sum(t, axis=1).reshape((len(observations), 1))
    node_responsibility = t / normalization_matrix
    return node_responsibility


def get_edge_responsibility(forward: np.ndarray, backward: np.ndarray, observations: str, states: Dict[int, str], transition: Dict[str, int], emission: Dict[str, int]) -> np.ndarray:
    edge_responsibility = np.zeros(
        (len(observations)-1, len(states), len(states)))
    for i in range(len(observations)-1):
        for pre_idx, pre_state in states.items():
            for idx, state in states.items():
                weight = transition[state+pre_state] * \
                    emission[observations[i+1]+state]
                edge_responsibility[i, pre_idx, idx] = forward[i,
                                                               pre_idx] * weight * backward[i+1, idx]
    normalization_amount = np.sum(edge_responsibility, axis=(1, 2))
    for i in range(len(observations)-1):
        edge_responsibility[i] /= normalization_amount[i]
    return edge_responsibility


def get_emission(node_responsibility: np.ndarray, observations: str, alphabet: List[str], states: Dict[int, str]) -> Dict[str, int]:
    emission = {e+state: 0 for e in alphabet for state in states.values()}
    for idx, state in states.items():
        sum = 0
        for e in alphabet:
            for i in range(len(observations)):
                emission[e+state] += node_responsibility[i,
                                                         idx] if observations[i] == e else 0
            sum += emission[e+state]
        for e in alphabet:
            emission[e+state] /= sum
    return emission


def get_transition(edge_responsibility: np.ndarray, observations: str, alphabet: List[str], states: Dict[int, str]) -> Dict[str, int]:
    transition = {state+pre_state: 0 for state in states.values()
                  for pre_state in states.values()}
    for pre_idx, pre_state in states.items():
        sum = 0
        for idx, state in states.items():
            for i in range(len(observations)-1):
                transition[state +
                           pre_state] += edge_responsibility[i, pre_idx, idx]
            sum += transition[state+pre_state]
        for idx, state in states.items():
            transition[state+pre_state] /= sum
    return transition


def get_formatted_output(transition: Dict[str, int], emission: Dict[str, int], alphabet: List[str], states: Dict[int, str]) -> str:
    s = ''
    transition_str = [''] * len(states)
    for state in states.values():
        s += state + '\t'
    for i, pre_state in states.items():
        transition_str[i] += pre_state + '\t'
        for state in states.values():
            transition_str[i] += '{:.3f}\t'.format(transition[state+pre_state])
        s += '\n' + transition_str[i]
    s += '\n--------\n\t'
    emission_str = [''] * len(states)
    for e in alphabet:
        s += e + '\t'
    for i, state in states.items():
        emission_str[i] += state + '\t'
        for e in alphabet:
            emission_str[i] += '{:.3f}\t'.format(emission[e+state])
        s += '\n' + emission_str[i]
    return s


run_times, observations, alphabet, states, transition, emission = read_input()
for _ in range(run_times):
    forward = get_forward(observations, states, transition, emission)
    backward = get_backward(observations, states, transition, emission)
    node_responsibility = get_node_responsibility(
        forward, backward, observations)
    edge_responsibility = get_edge_responsibility(forward, backward, observations,
                                                  states, transition, emission)
    emission = get_emission(node_responsibility,
                            observations, alphabet, states)
    transition = get_transition(
        edge_responsibility, observations, alphabet, states)
write_output(get_formatted_output(transition, emission, alphabet, states))
