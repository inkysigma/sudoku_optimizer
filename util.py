def process_solution(solu):
    sol = np.zeros((9, 9))
    for i, e in enumerate(solu):
        sol[i // 9, i % 9] = int(e)
    return sol
