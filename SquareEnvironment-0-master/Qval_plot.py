import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import  matplotlib
import json

matplotlib.style.use('ggplot')

from Qvalue import status, actions

X, Y = np.meshgrid(status, actions)


with open('dqn_qvalue.json', 'r') as fr:
    Z = np.array(json.load(fr)['q_value']).transpose()

# a = pd.read_csv('realQ.csv', index_col=0)
# Z = a.transpose()

fig = plt.figure(1)
ax = Axes3D(fig)
ax.plot_surface(Y, X, Z)

# fig = plt.figure(2)
# ax = fig.add_subplot(111)
# ax.plot(a.iloc[:10, :].transpose())

plt.show()