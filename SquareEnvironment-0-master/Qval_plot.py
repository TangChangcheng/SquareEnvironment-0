import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import  matplotlib

matplotlib.style.use('ggplot')

interval = 0.01
status = np.arange(-3, 3, interval)
actions = np.arange(-5, 5, interval)

X, Y = np.meshgrid(status, actions)


a = pd.read_csv('realQ.csv', index_col=0)
fig = plt.figure(1)
ax = Axes3D(fig)
ax.plot_surface(Y, X, a.transpose())

plt.show()