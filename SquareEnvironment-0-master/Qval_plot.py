import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import  matplotlib
import json

matplotlib.style.use('ggplot')

from Qvalue import status, actions
from Environment import Env

# ###Q value
X, Y = np.meshgrid(actions, status)
# with open('dqn_qvalue.json', 'r') as fr:
#     Z = np.array(json.load(fr)['q_value'])

Z = pd.read_csv('dqn_qvalue.csv', index_col=0)

fig = plt.figure(1)
ax = Axes3D(fig)
ax.plot_surface(Y, X, Z)

# fig = plt.figure(2)
# ax = fig.add_subplot(111)
# ax.plot(a.iloc[:10, :].transpose())


# with open('actor.json', 'r') as fr:
#     ac = np.array(json.load(fr)['action'])
#     fig = plt.figure(2)
#     ax = fig.add_subplot(111)
#     ax.plot(status, ac)
#
#     env = Env()
#     loss = env.foo(status)
#     ax.plot(status, loss)


plt.show()