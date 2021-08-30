import numpy as np


def process(board):
    entries = []
    for i, e in enumerate(board):
        if int(e) != 0:
            entries.append((i // 9, i % 9, int(e) - 1))
    return entries


def process_solution(solu):
    sol = np.zeros((9, 9))
    for i, e in enumerate(solu):
        sol[i // 9, i % 9] = int(e)
    return sol
