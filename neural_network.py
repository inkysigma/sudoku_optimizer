import pickle as pkl
import os
import pandas
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split

EXTRA_LOSS = True
ERROR_WEIGHT = 1 if EXTRA_LOSS else 0
NETWORK_NAME = f"neural_network_{'with' if EXTRA_LOSS else 'no'}_extra_loss.pkl"


def create_network():
    return torch.nn.Sequential(
        nn.Conv2d(10, 64, 3, padding=2),
        nn.ReLU(),
        nn.Conv2d(64, 32, 3, padding=2),
        nn.ReLU(),
        nn.Conv2d(32, 16, 3, padding=2),
        nn.ReLU(),
        nn.Conv2d(16, 8, 3, padding=2),
        nn.ReLU(),
        nn.ConvTranspose2d(8, 16, 3, padding=2),
        nn.ReLU(),
        nn.ConvTranspose2d(16, 32, 3, padding=2),
        nn.ReLU(),
        nn.ConvTranspose2d(32, 64, 3, padding=2),
        nn.ReLU(),
        nn.ConvTranspose2d(64, 9, 3, padding=2)
    )


def penalty(x: torch.Tensor):
    error = 0
    x = nn.Softmax()(x)
    for v in range(9):
        error += torch.sum(torch.abs(torch.sum(x[0, v, :, :], dim=1) - 1))
        error += torch.sum(torch.abs(torch.sum(x[0, v, :, :], dim=0) - 1))

        for k in range(9):
            row, column = k // 3, k % 3
            error += torch.sum(torch.abs(torch.sum(x[0, v, 3 *
                               row:3*row+3, 3*column:3*column+3]) - 1))

    error += torch.sum(torch.abs(torch.sum(x, dim=[2, 3]) - 1))
    return error


def convert_to_readable(x: torch.Tensor):
    return torch.argmax(x, dim=1).flatten()


def convert_tensor(s):
    if "0" in s:
        t = torch.zeros(size=(1, 10, 9, 9))
        for i, e in enumerate(s):
            if int(e) != 0:
                t[0, int(e) - 1, i // 9, i % 9] = 1
                t[0, 9, i // 9, i % 9] = 1
        return t
    else:
        t = torch.zeros(size=(1, 9, 9, 9))
        for i, e in enumerate(s):
            t[0, int(e) - 1, i // 9, i % 9] = 1
        return t


def train(network, data):
    quiz = data["quizzes"]
    solu = data["solutions"]

    optimizer = torch.optim.Adam(network.parameters())
    loss_fn = torch.nn.CrossEntropyLoss()
    for e in range(30):
        for i, (q, s) in enumerate(zip(quiz, solu)):
            optimizer.zero_grad()
            actual = convert_tensor(s)
            inp = convert_tensor(q)

            output = network(inp)
            loss = loss_fn(output, torch.argmax(actual, dim=1)) + ERROR_WEIGHT * penalty(output)
            loss.backward()
            optimizer.step()
            if i % 100 == 0:
                print(f"{i}: loss is {loss.detach()}")



if __name__ == "__main__":
    if not os.path.exists(NETWORK_NAME):
        m = create_network()
    else:
        with open(NETWORK_NAME, 'rb') as f:
            m = pkl.load(f)

    complete_data = pandas.read_csv("data/small2.csv")
    training, test = train_test_split(complete_data)
    train(m, training)
    with open(NETWORK_NAME, 'wb+') as f:
        pkl.dump(m, f)

    for q, s in zip(test['quizzes'], test['solutions']):
        test_q = convert_tensor(q)
        test_s = convert_tensor(s)
        print(convert_to_readable(m(test_q)), torch.argmax(test_s, dim=1))
        break
    for q, s in zip(training['quizzes'], training['solutions']):
        test_q = convert_tensor(q)
        test_s = convert_tensor(s)
        print(convert_to_readable(m(test_q)), torch.argmax(test_s, dim=1))
        quit()
