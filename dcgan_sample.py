
from __future__ import division, print_function, absolute_import

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from dcgan_model import DCGAN
from dcgan_model import normalize

def create_plot_example_3d():
    fig, ax = plt.subplots()
    ax = fig.gca(projection='3d')
    def plot(heightmap):
        ax.cla()
        X = np.arange(0, heightmap.shape[0], 1)
        Y = np.arange(0, heightmap.shape[1], 1)
        X, Y = np.meshgrid(X, Y)
        surf = ax.plot_surface(X, Y, heightmap, rstride=1, cstride=1,
                               linewidth=0, antialiased=False)
        ax.set_zlim(0.0, 1.0)
        # fig.colorbar(surf, shrink=0.5, aspect=5)

        ax.zaxis.set_major_locator(LinearLocator(10))
        ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    return plot


with tf.Session() as sess:
    model = DCGAN(sess)

    model.load('model/Jul07_0022/save-3900')

    plot_example_3d = create_plot_example_3d()

    sample_z = model.get_data_batch_z(1)
    res = sess.run([model.gen], feed_dict={ model.Z: sample_z })
    plot_example_3d(np.reshape(res[0][0, 0:, 0:, 0], (model.image_size, model.image_size)))

    plt.waitforbuttonpress()
