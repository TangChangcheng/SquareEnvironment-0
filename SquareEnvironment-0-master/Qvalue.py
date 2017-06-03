
import numpy as np
import pandas as pd
import copy


interval = 0.05
status = np.arange(-3, 3, interval)
actions = np.arange(-6, 6, interval)

Q = np.zeros([len(status), len(actions)])

coefs = [9.54151441, 6.8100, 2.76072571]


def transf(state, action):
    s = int(np.around((state - min(status)) / interval))
    a = int(np.around((action - min(actions)) / interval))
    return s, a

def foo(X):
    return coefs[0] * np.power(X, 2) + coefs[1] * X + coefs[2]


GAMMA = 0.5
R = copy.copy(Q)
for i, s in enumerate(status):
    for j, a in enumerate(actions):
        R[i, j] = foo(s) - foo(s + a)


def getMaxQ(state):
    if state < 0  or state >= len(status):
        raise Exception('state is out of index')
    return max(Q[state, :])

def QLearning(state):
    for action in actions:
        s, a = transf(state, action)
        try:
            Q[s,a] = R[s, a] + GAMMA * getMaxQ(int(np.around((state + action - min(status)) / interval)))
        except Exception:
            # print(state, action, state + action, s, a, s + a)
            pass

if __name__ == '__main__':
    cur = 100
    prev = 0

    #np.random.shuffle(status)
    for count in range(5000):
        for i in status:
            QLearning(i)
        prev = cur
        cur =  (np.max(Q, axis=1).mean())
        print(cur)
        if abs(prev - cur) < 0.01 and count > 10:
            break

    Q = pd.DataFrame(Q)
    Q.to_csv('realQ.csv')
