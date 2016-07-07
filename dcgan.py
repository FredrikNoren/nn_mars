
from __future__ import division, print_function, absolute_import

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from dcgan_model import DCGAN
from dcgan_model import normalize

model = DCGAN()
n_samples_w = 5
n_samples_h = 5

model.prepare_train()

costs = []

plt.ion()

print("Creating plots")
sample_plot_f, sample_plot_a = plt.subplots(n_samples_w, n_samples_h, figsize=(n_samples_w, n_samples_h))
sample_plot_f.show()
cost_plot_f, cost_plot_a = plt.subplots()
cost_plot_f.show()
real_plot_f, real_plot_a = plt.subplots(n_samples_w, n_samples_h, figsize=(n_samples_w, n_samples_h))
real_plot_f.show()
filters_out_plot_f, filters_out_plot_a = plt.subplots(2, 2)
filters_out_plot_f.show()
filters_plot_f, filters_plot_a = plt.subplots(1, 2)
filters_plot_f.show()
filters_plot_a[0].set_label("descriminator")
filters_plot_a[1].set_label("generator")
variance_filters_plot_f, variance_filters_plot_a = plt.subplots(2, 1)
variance_filters_plot_f.show()
print("Creating plots done")

viz_filters_batch_x = model.get_real_data_batch_x(1)
viz_filters_batch_z = model.get_data_batch_z(1)
filters_out_plot_a[0][0].imshow(viz_filters_batch_x[0, 0:, 0:, 0])
def pack_filters(filters_array, width=256):
    print("pack", len(filters_array))
    x_pos = 0
    y_pos = 0
    positions = []
    height = 0
    max_row_height = 0
    for i in range(len(filters_array)):
        filtr = filters_array[i]
        height = max(height, y_pos + filtr.shape[0])
        if x_pos + filtr.shape[1] > width:
            x_pos = 0
            y_pos += max_row_height
            max_row_height = 0
        positions.append((x_pos, y_pos))
        x_pos += filtr.shape[0]
        max_row_height = max(max_row_height, filtr.shape[1])
    packed_image = np.zeros((height, width))
    for i in range(len(filters_array)):
        filtr = filters_array[i]
        x_pos, y_pos = positions[i]
        packed_image[y_pos:(y_pos + filtr.shape[1]), x_pos:(x_pos + filtr.shape[0])] = filtr
    return packed_image
def filters_out_to_array(filters_out):
    filters_array = []
    for y in range(len(filters_out)):
        filters = filters_out[y]
        for x in range(filters.shape[-1]):
            img = normalize(np.reshape(filters[0, 0:, 0:, x], (filters.shape[1], filters.shape[2])))
            filters_array.append(normalize(img))
    return filters_array
def visualize_filters_out():
    res = model.sess.run(model.real_desc_filters_out + model.gen_desc_filters_out + [model.gen], feed_dict={ model.X: viz_filters_batch_x, model.Z: viz_filters_batch_z })
    a = len(model.real_desc_filters_out)
    b = a + len(model.gen_desc_filters_out)
    real_filters_out = pack_filters(filters_out_to_array(res[0:a]))
    gen_filters_out = pack_filters(filters_out_to_array(res[a:b]))
    generated_image = np.reshape(res[b], (model.image_size, model.image_size))
    filters_out_plot_a[1][0].imshow(real_filters_out, interpolation="nearest")
    filters_out_plot_a[0][1].imshow(generated_image)
    filters_out_plot_a[1][1].imshow(gen_filters_out, interpolation="nearest")

def filter_vars_to_array(filter_vars):
    filters_array = []
    for i in range(len(filter_vars)):
        filter_var = filter_vars[i]
        print(filter_var.name)
        filter_data = filter_var.eval()
        print(filter_data.shape)
        for x in range(filter_data.shape[2]):
            for y in range(filter_data.shape[3]):
                filters_array.append(normalize(filter_data[0:, 0:, x, y]))
    filters_array.sort(lambda a, b: -1 if np.mean(np.power(a, 2)) < np.mean(np.power(b, 2)) else 1)
    return filters_array
def filters_array_variation(filters_array):
    v = 0.0
    for i in range(len(filters_array) - 1):
        v += np.mean(np.power((filters_array[i] - filters_array[i + 1]), 2))
    return v
def visualize_filters():
    desc_filters_array = filter_vars_to_array(model.desc_filter_vars)
    gen_filters_array = filter_vars_to_array(model.gen_filter_vars)
    filters_plot_a[0].imshow(pack_filters(desc_filters_array, width=512), interpolation="nearest")
    filters_plot_a[1].imshow(pack_filters(gen_filters_array, width=512), interpolation="nearest")
    return filters_array_variation(desc_filters_array), filters_array_variation(gen_filters_array)

heightmap3d_plot_f, heightmap3d_plot_a = plt.subplots()
heightmap3d_plot_a = heightmap3d_plot_f.gca(projection='3d')
def plot_example_3d(heightmap):
    heightmap3d_plot_a.cla()
    X = np.arange(0, heightmap.shape[0], 1)
    Y = np.arange(0, heightmap.shape[1], 1)
    X, Y = np.meshgrid(X, Y)
    surf = heightmap3d_plot_a.plot_surface(X, Y, heightmap, rstride=1, cstride=1,
                           linewidth=0, antialiased=False)
    heightmap3d_plot_a.set_zlim(-0.1, 1.1)
    # heightmap3d_plot_f.colorbar(surf, shrink=0.5, aspect=5)

    heightmap3d_plot_a.zaxis.set_major_locator(LinearLocator(10))
    heightmap3d_plot_a.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

# plot_example_3d(normalize(get_subimage()))
# plt.waitforbuttonpress()
# exit()

print("Plotting real data")
real_data = model.get_real_data_batch_x(n_samples_w * n_samples_h)
for x in range(n_samples_w):
    for y in range(n_samples_h):
        real_plot_a[x][y].imshow(np.reshape(real_data[x + y * n_samples_w, 0:, 0:, 0], (model.image_size, model.image_size)))
print("Plotting real data done")

sample_z = model.get_data_batch_z(n_samples_w*n_samples_h)

print("Creating session")
variance_desc_filters_points = []
variance_gen_filters_points = []
def epoch_callback(epoch, epoch_costs):
    costs.append(epoch_costs)
    print(epoch, epoch_costs)
    if epoch % 100 != 0:
        return
    visualize_filters_out()
    variance_desc_filters, variance_gen_filters = visualize_filters()
    variance_desc_filters_points.append(variance_desc_filters)
    variance_gen_filters_points.append(variance_gen_filters)
    variance_filters_plot_a[0].plot(variance_desc_filters_points)
    variance_filters_plot_a[1].plot(variance_gen_filters_points)
    print("variance_desc_filters", variance_desc_filters, "variance_gen_filters", variance_gen_filters)

    cost_plot_a.plot(costs)

    # res1 = sess.run([real_desc], feed_dict={ X: batch_x })
    # print(res1[0])

    res2 = model.sess.run([model.gen], feed_dict={ model.Z: sample_z })
    print(np.min(res2[0][0]), np.max(res2[0][0]))
    for x in range(n_samples_w):
        for y in range(n_samples_h):
            sample_plot_a[x][y].imshow(np.reshape(res2[0][x + y * n_samples_w, 0:, 0:, 0], (model.image_size, model.image_size)))
    plot_example_3d(np.reshape(res2[0][0, 0:, 0:, 0], (model.image_size, model.image_size)))
    plt.draw()

    plt.pause(0.05)
model.train(epoch_callback)

plt.waitforbuttonpress()
