import numpy as np

from multiprocessing import Pool
from mpl_toolkits.mplot3d import Axes3D
# Axes3D import has side effects, it enables using projection='3d' in add_subplot
import matplotlib.pyplot as plt
from matplotlib import cm

from core import arx_trainer


if __name__ == '__main__':

    train_engine = arx_trainer.ARXTrainer()

    orders = np.arange(start=2, stop=10, step=1, dtype=np.int)
    contexts = np.arange(start=2, stop=10, step=1, dtype=np.int)

    X, Y = np.meshgrid(orders, contexts)

    mean_CCC = []

    for order, context in zip(np.ravel(X), np.ravel(Y)):
        mean_CCC.append(train_engine.train_model(order, context))

    Z = np.array(mean_CCC).reshape(X.shape)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=0, antialiased=False)

    ax.set_xlabel('y order')
    ax.set_ylabel('x context')
    ax.set_zlabel('mean CCC value')

    fig.savefig('arx_val.png')
