

from lp import solve
import numpy as np
from util import process, process_solution
import sys
import os
import pandas
import time

sys.path.append(os.path.dirname(os.path.realpath(__file__)))


def solve_single(input):
    entries = process(input)
    t = solve(entries)
    return ''.join([''.join([str(int(c)) for c in r]) for r in t])


def solver(file_name: str):
    data = pandas.read_csv(file_name)
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
    return corr_cnt / all
