
import numpy as np

from Environment import Env

env = Env()
interval = 0.01
status = np.arange(-3, 3, interval)
actions = np.arange(-5, 5, interval)

# Q = {i: {} for i in status}

Q = np.zeros([len(status), len(actions)]) + np.nan

gamma = 0.99

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

    print(qval)

    return qval


with open('realQ', 'w') as fw:
    import pandas as pd
    for state in status:
        env.reset(status=state)
        for action in actions:
            Qval(state, action, 0)

    a = pd.DataFrame(Q)
    a.to_csv('realQ.csv', encoding='utf8')

