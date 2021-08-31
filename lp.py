from cvxopt import solvers, matrix
import time
import cvxpy as cp
import numpy as np
import pandas
from util import process, process_solution


solvers.options['show_progress'] = False


def sudoku_constraint(x, entries):
    """
    x: a 9x9x9 numpy array of variables that tell them the linear constraints.
    """
    const = []
    for v in range(9):
        const.append(cp.sum(x[v][:, :], axis=1) == 1)
        const.append(cp.sum(x[v][:, :], axis=0) == 1)

        for k in range(9):
            row, column = k // 3, k % 3
            const.append(cp.sum(x[v][3*row:3*row+3, 3*column:3*column+3]) == 1)

    for i in range(x[0].shape[0]):
        for j in range(x[0].shape[1]):
            total = 0
            for v in range(9):
                total += x[v][i, j]
            const.append(total == 1)

    for entry in entries:
        for v in range(9):
            if entry[2] == v:
                const.append(x[v][entry[0], entry[1]] == 1)
            else:
                const.append(x[v][entry[0], entry[1]] == 0)
    return const


def mismatch(x):
    counter = 0
    for v in range(9):
        counter += np.sum(np.sum(x[:, :] == v) - 1)
        counter += np.sum(np.sum(x[:, :] == v) - 1)

        for k in range(9):
            row, column = k // 3, k % 3
            counter += np.sum(x[row:row+3, column:column+3] == v) - 1
    return counter


def post_processing(board):
    processed = np.zeros((9, 9))
    for v in range(9):
        for i in range(board[v].shape[0]):
            for j in range(board[v].shape[1]):
                if board[v][i, j] > 0:
                    processed[i, j] = v + 1
    return processed


def solve(entries, integer=True):
    """
    Accepts a list of entries at various positions
    """
    x = []
    for i in range(9):
        if integer:
            x.append(cp.Variable((9, 9), name=f'where are {i}', integer=True))
        else:
            x.append(cp.Variable((9, 9), name=f'where are {i}'))
    const = sudoku_constraint(x, entries)
    for v in x:
        const.append(v >= 0)
        if integer:
            const.append(v <= 1)
    obj = cp.Minimize(sum(cp.sum(r) for r in x))
    prob = cp.Problem(obj, const)
    err = prob.solve()
    if err >= 10e5:
        return np.zeros((9, 9))
    return post_processing([v.value for v in x])


if __name__ == "__main__":

    # We test the following algoritm on small data set.
    data = pandas.read_csv("./data/small1.csv")

    corr_cnt = 0
    start = time.time()
    for round in range(len(data)):
        quiz = data["quizzes"][round]
        solu = data["solutions"][round]
        entries = process(quiz)
        t = solve(entries)
        if not np.all(t == process_solution(solu)):
            pass
        else:
            corr_cnt += 1

        if (round+1) % 5 == 0:
            end = time.time()
            print("Aver Time: {t:6.2f} secs. Success rate: {corr} / {all} ".format(
                t=(end-start)/(round+1), corr=corr_cnt, all=round+1))

    end = time.time()
    print("Aver Time: {t:6.2f} secs. Success rate: {corr} / {all} ".format(
        t=(end-start)/(round+1), corr=corr_cnt, all=round+1))
