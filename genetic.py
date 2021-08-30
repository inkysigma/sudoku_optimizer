import numpy as np
import time
import pandas
import pygad
from numba import jit

PENALTY_DOUBLE_COUNT = -1


@jit(nopython=True)
def mismatch(x, verbose=False):
    counter = 0
    for v in range(9):
        counter += np.sum(np.abs(np.sum(x[:, :] == v, axis=0) - 1))
        counter += np.sum(np.abs(np.sum(x[:, :] == v, axis=1) - 1))
        for k in range(9):
            row, column = k // 3, k % 3
            counter += np.abs(np.sum(x[3 * row:3 * row+3, 3 *
                              column: 3 * column+3] == v) - 1)

    return counter * PENALTY_DOUBLE_COUNT


def process(board):
    entries = []
    for i, e in enumerate(board):
        if int(e) != 0:
            entries.append((i // 9, i % 9, int(e) - 1))
    return entries


def to_grid(processed):
    board = np.zeros((9, 9)) - 1
    for p in processed:
        board[p[0], p[1]] = p[2]
    return board


def fill_in(x, filled_in):
    ans = np.copy(filled_in)
    c = 0
    for i in range(ans.shape[0]):
        for j in range(ans.shape[1]):
            if ans[i, j] == -1:
                ans[i, j] = x[c]
                c += 1
    return ans


def process_solution(solu):
    sol = np.zeros((9, 9))
    for i, e in enumerate(solu):
        sol[i // 9, i % 9] = int(e)
    return sol


# it's really slow...
if __name__ == "__main__":
    data = pandas.read_csv("./data/small1.csv")

    corr_cnt = 0
    start = time.time()
    for i in range(len(data)):
        quiz = data["quizzes"][i]
        solu = data["solutions"][i]
        entries = process(quiz)
        curr = to_grid(entries)

        def genetic(x, sol_idx):
            return mismatch(fill_in(x, curr))
        fitness_function = genetic

        num_generations = 2000
        num_parents_mating = 5

        sol_per_pop = 700
        num_genes = 81 - len(entries)

        parent_selection_type = "rank"
        keep_parents = 5

        crossover_type = "single_point"

        mutation_type = "random"
        mutation_percent_genes = 3
        instance = pygad.GA(num_generations=num_generations,
                            num_parents_mating=num_parents_mating,
                            fitness_func=fitness_function,
                            sol_per_pop=sol_per_pop,
                            num_genes=num_genes,
                            gene_type=int,
                            gene_space=[0, 1, 2, 3, 4, 5, 6, 7, 8],
                            parent_selection_type=parent_selection_type,
                            keep_parents=keep_parents,
                            crossover_type=crossover_type,
                            mutation_type=mutation_type,
                            mutation_percent_genes=mutation_percent_genes)
        instance.run()
        solution, solution_fitness, solution_idx = instance.best_solution()
        print(fill_in(solution, curr) + 1)
        print(solution_fitness)
        print(mismatch(fill_in(solution, curr), True))
        quit()
