
import numpy as np
import pandas as pd
import copy

from Environment import Env

env = Env()
interval = 0.01
status = np.arange(-3, 3, interval)
actions = np.arange(-5, 5, interval)

# Q = {i: {} for i in status}

Q = np.zeros([len(status), len(actions)])

gamma = 0.99
coefs = [9.54151441, 6.8100, 2.76072571]


def transf(state, action):
    s = int((state - min(status)) / interval)
    a = int((action - min(actions)) / interval)
    return s, a

def Qval(state, action, step):
    _, reward, done, _ = env.step(np.array([action]))
    qval = reward

    s, a = transf(state, action)
    if not np.isnan(Q[s, a]):
        return Q[s, a]

    if done or step >= 20:
        Q[s][a] = qval
        return qval

    tmp = []
    for ac in actions:
        tmp.append(Qval(env.status[0], ac, step + 1))
        env.reset(status=state)

    qval += gamma * max(tmp)
    Q[s][a] = qval
    return qval


def write(path):
    for state in status:
        env.reset(status=state)
        for action in actions:
            Qval(state, action, 0)

    a = pd.DataFrame(Q)
    a.to_csv('realQ.csv', encoding='utf8')

def foo(X):
    return coefs[0] * np.power(X, 2) + coefs[1] * X + coefs[2]


GAMMA = 0.99
R = copy.copy(Q)

for i, s in enumerate(status):
    for j, a in enumerate(actions):
        R[i, j] = foo(s) - foo(s + a)


def getMaxQ(state):
    return max(Q[state, :])

def QLearning(state):
    for action in range(len(actions)):
        s, a = transf(state, action)
        Q[s,a] = R[s, a] + GAMMA * getMaxQ(s + a)


for count in range(1000):
    for i in range(len(status)):
        QLearning(i)

Q = pd.DataFrame(Q)
Q.to_csv('realQ.csv')




