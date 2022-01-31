from typing import Dict, List, Tuple, Union
import numpy as np
from pathlib import Path


def read_input() -> np.ndarray:
    path = Path() / input()
    lengths = []
    for s in path.read_text().split('\n'):
        lengths += [len(s)] * len(s)
    return np.array(lengths, dtype=int)


def write_output(result: str):
    path = Path() / 'result.txt'
    path.write_text(result)


lengths = read_input()
s1 = int(np.median(lengths))
s2 = int(np.quantile(lengths, 0.25))
write_output(f'{s1} {s2}')
