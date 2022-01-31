from typing import Dict, List, Tuple, Union
import numpy as np
from pathlib import Path


def read_input() -> np.ndarray:
    path = Path() / input()
    DNA_strings = []
    for s in path.read_text().split('\n')[:-1]:
        if s not in DNA_strings:
            DNA_strings.append(s)
        s_complement = get_reverse_compliment(s)
        if s_complement not in DNA_strings:
            DNA_strings.append(s_complement)
    return DNA_strings


def write_output(result: str):
    path = Path() / 'result.txt'
    path.write_text(result)


def get_reverse_compliment(DNA_string: str) -> str:
    base_pair_dict = {'T': 'A', 'A': 'T', 'C': 'G', 'G': 'C'}
    return ''.join([base_pair_dict[base] for base in reversed(DNA_string)])


DNA_strings = read_input()
result = [f'({DNA_string[:-1]}, {DNA_string[1:]})' for DNA_string in DNA_strings]
write_output('\n'.join(result))
