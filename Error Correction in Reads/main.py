from dis import dis
from typing import Dict, List, Tuple, Union
from pathlib import Path


def read_input() -> List[str]:
    path = Path() / input()
    return [s for s in path.read_text().split('\n') if len(s) > 0 and s[0] != '>']


def write_output(result: str):
    path = Path() / 'result.txt'
    path.write_text(result)


def get_reverse_compliment(DNA_string: str) -> str:
    base_pair_dict = {'T': 'A', 'A': 'T', 'C': 'G', 'G': 'C'}
    return ''.join([base_pair_dict[base] for base in reversed(DNA_string)])


def split_reads(reads: List[str]) -> Tuple[List[str], List[str]]:
    correct_reads, incorrect_reads = [], []
    for read in set(reads):
        read_reverse = get_reverse_compliment(read)
        count = reads.count(read) + reads.count(read_reverse)
        if count >= 2:
            correct_reads.append(read)
        else:
            incorrect_reads.append(read)

    return correct_reads, incorrect_reads


def get_distance(s1: str, s2: str) -> int:
    distance = 0
    for i in range(len(s1)):
        if s1[i] != s2[i]:
            distance += 1
    return distance


def get_new_read(old_read: str, correct_reads: List[str]) -> str:
    for read in correct_reads:
        read_reverse = get_reverse_compliment(read)
        if get_distance(old_read, read) == 1:
            return read
        elif read_reverse not in correct_reads and get_distance(old_read, read_reverse) == 1:
            return read_reverse


reads = read_input()
correct_reads, incorrect_reads = split_reads(reads)
corrections = [
    f'{old_read}->{get_new_read(old_read, correct_reads)}' for old_read in incorrect_reads]
write_output('\n'.join(corrections))
