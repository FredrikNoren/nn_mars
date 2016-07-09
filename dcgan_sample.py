
from __future__ import division, print_function, absolute_import

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from dcgan_model import DCGAN
from dcgan_model import normalize
import random
from array import array

def write_dhm_file(filename, heightmap):
    output_file = open(filename, 'wb')
    int_array = array('i', [heightmap.shape[0], heightmap.shape[1]])
    int_array.tofile(output_file)

    floats = []
    for y in range(heightmap.shape[0]):
        for x in range(heightmap.shape[1]):
            floats.append(heightmap[x, y])

    float_array = array('f', floats)
    float_array.tofile(output_file)
    output_file.close()


def create_plot_example_3d():
    fig, ax = plt.subplots()
    ax = fig.gca(projection='3d')
    def plot(heightmap):
        ax.cla()
        X = np.arange(0, heightmap.shape[0], 1)
        Y = np.arange(0, heightmap.shape[1], 1)
        X, Y = np.meshgrid(X, Y)
        surf = ax.plot_surface(X, Y, heightmap, rstride=1, cstride=1,
                               linewidth=0, antialiased=False, shade=True)
        ax.set_zlim(0.0, 1.0)
        # fig.colorbar(surf, shrink=0.5, aspect=5)

        ax.zaxis.set_major_locator(LinearLocator(10))
        ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    return plot

def create_heightmaps_plot(n_width, n_height):
    fig, ax = plt.subplots(n_width, n_height, figsize=(n_width, n_height))
    fig.show()
    def update(x, y, heightmap):
        ax[x][y].imshow(heightmap)
    return update

def create_samples_plot(model, n_width, n_height):
    update_hm = create_heightmaps_plot(n_width, n_height)
    def update(sample_z):
        res = model.sess.run([model.gen], feed_dict={ model.Z: sample_z })
        print(np.min(res[0][0]), np.max(res[0][0]))
        for x in range(n_width):
            for y in range(n_height):
                update_hm(x, y, np.reshape(res[0][x + y * n_width, 0:, 0:, 0], (model.image_size, model.image_size)))
    return update

def create_diff_samples_plot(model, n_width, n_height):
    fig, sample_plot_a = plt.subplots(n_height, n_width, figsize=(n_height, n_width))
    fig.show()
    sample_z = model.get_data_batch_z(n_height*n_width)
    z_i = random.randint(0, n_height)
    for y in range(n_height):
        for x in range(n_width):
            z = np.copy(sample_z[y*n_width])
            z[z_i] = 2 * x / (n_width - 1) - 1
            sample_z[y*n_width + x] = z
    def update():
        res = model.sess.run([model.gen], feed_dict={ model.Z: sample_z })
        print(np.min(res[0][0]), np.max(res[0][0]))
        for x in range(n_width):
            for y in range(n_height):
                sample_plot_a[y][x].imshow(np.reshape(res[0][x + y * n_width, 0:, 0:, 0], (model.image_size, model.image_size)))
    return update


with tf.Session() as sess:
    model = DCGAN(sess)

    # model.load('model/Jul07_0100/save-35300')

    #sample_z = model.get_data_batch_z(1)
    #
    # plot_example_3d = create_plot_example_3d()
    # res = sess.run([model.gen], feed_dict={ model.Z: sample_z })
    # plot_example_3d(np.reshape(res[0][0, 0:, 0:, 0], (model.image_size, model.image_size)))

    # update_samples_plot = create_samples_plot(model, 5, 5)
    # update_samples_plot()

    # create_diff_samples_plot(model, 15, 4)()

    #update_hm = create_heightmaps_plot(2, 2)

    n_write = 50
    # sample_z = model.get_data_batch_z(n_write)
    # res = model.sess.run([model.gen], feed_dict={ model.Z: sample_z })
    # for i in range(n_write):
    #     heightmap = np.reshape(res[0][i, 0:, 0:, 0], (model.image_size, model.image_size))
    #     write_dhm_file('heightmaps/gen/heightmap-' + str(i) + '.dhm', heightmap)

    model.prepare_train()
    real_data = model.get_real_data_batch_x(50)
    for i in range(n_write):
        print(np.var(real_data[i, 0:, 0:]))
        heightmap = np.reshape(real_data[i, 0:, 0:], (model.image_size, model.image_size))
        write_dhm_file('heightmaps/real2/heightmap-' + str(i) + '.dhm', heightmap)

    # update_hm(0, 0, heightmap)

    plt.show()
