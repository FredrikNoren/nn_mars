
from __future__ import division, print_function, absolute_import

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import random
from skimage.measure import block_reduce
from noise import pnoise2


def draw_mars_data(data):
    u = np.unique(data.flatten())
    u.sort()
    print(u[0], u[1], u[20])
    #print np.min(data), np.max(data)
    data = np.clip(data, u[20], np.max(data))
    imgplot = plt.imshow(data, interpolation="nearest")
    plt.waitforbuttonpress()

print("Loading mars data")
mars_data = np.load("hirise_20.npy")
# for i in range(len(mars_data)):
#     draw_mars_data(mars_data[i])
# exit()
print("Loading mars data done.")

batch_size = 16
z_dim = 100
learning_rate_desc = 0.0001
learning_rate_gen  = 0.0001
image_size = 64
n_samples_w = 5
n_samples_h = 5
pnoise_scale = 0.02

def normalize(x):
    return (x - np.min(x)) / (np.max(x) - np.min(x))

def lrelu(x, leak=0.2, name="lrelu"):
    with tf.variable_scope(name):
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)
        return f1 * x + f2 * abs(x)

def conv2d(input_, output_dim, name="conv2d", stride=2, reuse=False, filter_size=5):
    with tf.variable_scope(name) as scope:
        if reuse:
            scope.reuse_variables()
        w = tf.get_variable("w", [filter_size, filter_size, input_.get_shape()[-1], output_dim], initializer=tf.truncated_normal_initializer(stddev=0.002))
        b = tf.get_variable("b", [output_dim], initializer=tf.constant_initializer(0.0))
        conv = tf.nn.conv2d(input_, w, strides=[1, stride, stride, 1], padding='SAME')
        return tf.nn.bias_add(conv, b)

def deconv2d(input_, output_shape, name="deconv2d", stride=2, reuse=False, filter_size=5):
    with tf.variable_scope(name) as scope:
        if reuse:
            scope.reuse_variables()
        w = tf.get_variable("w", [filter_size, filter_size, output_shape[-1], input_.get_shape()[-1]], initializer=tf.random_normal_initializer(stddev=0.002))
        b = tf.get_variable("b", [output_shape[-1]], initializer=tf.constant_initializer(0.0))
        deconv = tf.nn.conv2d_transpose(input_, w, output_shape=output_shape, strides=[1, stride, stride, 1], padding='SAME')
        return tf.nn.bias_add(deconv, b)

def linear(input_, output_size, name="linear"):
    with tf.variable_scope(name):
        w = tf.get_variable("w", [input_.get_shape()[1], output_size], initializer=tf.random_normal_initializer(stddev=0.002))
        b = tf.get_variable("b", [output_size], initializer=tf.constant_initializer(0.0))
        return tf.matmul(input_, w) + b

def generator(z):
    with tf.variable_scope("generator"):
        batch_size = tf.shape(z)[0]
        l1 = lrelu(linear(z, 4 * 4 * 64))
        l1 = tf.reshape(l1, [-1, 4, 4, 64])
        l2 = lrelu(deconv2d(l1, [batch_size, 8, 8, 64], name="deconv_l1"))
        l3 = lrelu(deconv2d(l2, [batch_size, 16, 16, 64], name="deconv_l2"))
        l4 = lrelu(deconv2d(l3, [batch_size, 32, 32, 64], name="deconv_l3"))
        l5 = tf.sigmoid(deconv2d(l4, [batch_size, 64, 64, 1], name="deconv_l4"))
        return l5

def descriminator(x, reuse=False):
    with tf.variable_scope("descriminator") as scope:
        if reuse:
            scope.reuse_variables()
        l1 = lrelu(conv2d(x, 64, name="conv_l1"))
        l2 = lrelu(conv2d(l1, 64, name="conv_l2"))
        l3 = lrelu(conv2d(l2, 64, name="conv_l3"))
        l4 = lrelu(conv2d(l3, 64, name="conv_l4"))
        l5 = linear(tf.reshape(l4, [-1, 4 * 4 * 64]), 1)
        return tf.sigmoid(l5), l5, [l1, l2, l3, l4]


def get_subimage():
    i = random.randint(0, len(mars_data) - 1)
    xpos = random.randint(0, mars_data[i].shape[0])
    ypos = random.randint(0, mars_data[i].shape[1])
    subdata = mars_data[i][xpos:(xpos + image_size), ypos:(ypos + image_size)]
    if subdata.shape[0] != image_size or subdata.shape[1] != image_size or np.min(subdata) < -5000:
        return get_subimage()
    else:
        return subdata

def get_real_data_batch_x(batch_size):
    batch_xs = np.zeros((batch_size, image_size, image_size, 1))
    for i in range(batch_size):
        x = normalize(get_subimage())
        x = np.rot90(x, random.randint(0, 3))
        if random.randint(0, 1) == 0:
            x = np.fliplr(x)
        if random.randint(0, 1) == 0:
            x = np.flipud(x)
        batch_xs[i, 0:image_size, 0:image_size, 0] = x
    return batch_xs

def get_perlin_data_batch_x(batch_size):
    batch_xs = np.zeros((batch_size, image_size, image_size, 1))
    for i in range(batch_size):
        pnoise_offset = random.random()*100000
        img = np.zeros((image_size, image_size))
        for x in range(image_size):
            for y in range(image_size):
                img[y, x] = pnoise2(x * pnoise_scale + pnoise_offset, y * pnoise_scale + pnoise_offset, 1)
        batch_xs[i, 0:image_size, 0:image_size, 0] = normalize(img)
    return batch_xs

# def get_real_data_batch_x(batch_size):
#     data = np.zeros((batch_size, 16, 16, 1))
#     data[0:batch_size, 0:16, 0:8, 0] = 1
#     return data

def get_data_batch_z(batch_size):
    return np.random.uniform(-1, 1, [batch_size, z_dim]).astype(np.float32)

Z = tf.placeholder("float", [None, z_dim])
gen = generator(Z)

X = tf.placeholder("float", [None, image_size, image_size, 1])
real_desc, real_desc_logits, real_desc_filters_out = descriminator(X)
X2 = tf.placeholder("float", [None, image_size, image_size, 1])
real_desc2, real_desc_logits2, real_desc_filters_out2 = descriminator(X2, reuse=True)

gen_desc, gen_desc_logits, gen_desc_filters_out = descriminator(gen, reuse=True)

real_desc_cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(real_desc_logits, tf.ones_like(real_desc_logits)))
real_desc_cost2 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(real_desc_logits2, tf.zeros_like(real_desc_logits)))
gen_desc_cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(gen_desc_logits, tf.zeros_like(gen_desc_logits)))
gen_cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(gen_desc_logits, tf.ones_like(gen_desc_logits)))

# real_desc_cost = tf.reduce_mean(tf.pow(tf.ones_like(real_desc) - real_desc, 2))
# gen_desc_cost = tf.reduce_mean(tf.pow(tf.zeros_like(gen_desc) - gen_desc, 2))
# gen_cost = tf.reduce_mean(tf.pow(tf.ones_like(gen_desc) - gen_desc, 2))

desc_cost = real_desc_cost + gen_desc_cost
desc_cost2 = real_desc_cost + real_desc_cost2

t_vars = tf.trainable_variables()
desc_vars = [var for var in t_vars if 'descriminator' in var.name]
gen_vars = [var for var in t_vars if 'generator' in var.name]
desc_filter_vars = [var for var in desc_vars if 'conv_' in var.name and '/w' in var.name]
gen_filter_vars = [var for var in gen_vars if 'conv_' in var.name and '/w' in var.name]

desc_optim = tf.train.AdamOptimizer(learning_rate_desc, beta1=0.5).minimize(desc_cost, var_list=desc_vars)
desc_optim2 = tf.train.AdamOptimizer(0.0001).minimize(desc_cost2, var_list=desc_vars)
gen_optim = tf.train.AdamOptimizer(learning_rate_gen, beta1=0.5).minimize(gen_cost, var_list=gen_vars)

init = tf.initialize_all_variables()
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

viz_filters_batch_x = get_real_data_batch_x(1)
viz_filters_batch_z = get_data_batch_z(1)
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
    res = sess.run(real_desc_filters_out + gen_desc_filters_out + [gen], feed_dict={ X: viz_filters_batch_x, Z: viz_filters_batch_z })
    a = len(real_desc_filters_out)
    b = a + len(gen_desc_filters_out)
    real_filters_out = pack_filters(filters_out_to_array(res[0:a]))
    gen_filters_out = pack_filters(filters_out_to_array(res[a:b]))
    generated_image = np.reshape(res[b], (image_size, image_size))
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
    desc_filters_array = filter_vars_to_array(desc_filter_vars)
    gen_filters_array = filter_vars_to_array(gen_filter_vars)
    filters_plot_a[0].imshow(pack_filters(desc_filters_array, width=512), interpolation="nearest")
    filters_plot_a[1].imshow(pack_filters(gen_filters_array, width=512), interpolation="nearest")
    return filters_array_variation(desc_filters_array), filters_array_variation(gen_filters_array)

print("Plotting real data")
real_data = get_real_data_batch_x(n_samples_w * n_samples_h)
for x in range(n_samples_w):
    for y in range(n_samples_h):
        real_plot_a[x][y].imshow(np.reshape(real_data[x + y * n_samples_w, 0:image_size, 0:image_size, 0], (image_size, image_size)))
print("Plotting real data done")

sample_z = get_data_batch_z(n_samples_w*n_samples_h)

print("Creating session")
variance_desc_filters_points = []
variance_gen_filters_points = []
with tf.Session() as sess:
    sess.run(init)


    # for i in range(1000):
    #     batch_x1 = get_real_data_batch_x(batch_size)
    #     batch_x2 = get_perlin_data_batch_x(batch_size)
    #     _, c1a, c1b = sess.run([desc_optim2, real_desc_cost, real_desc_cost2], feed_dict={ X: batch_x1, X2: batch_x2 })
    #     print("pretrain", "i", i, "cost_real_desc", c1a, "cost_real_desc2", c1b)

    for epoch in range(1000):
        visualize_filters_out()
        variance_desc_filters, variance_gen_filters = visualize_filters()
        variance_desc_filters_points.append(variance_desc_filters)
        variance_gen_filters_points.append(variance_gen_filters)
        variance_filters_plot_a[0].plot(variance_desc_filters_points)
        variance_filters_plot_a[1].plot(variance_gen_filters_points)
        print("variance_desc_filters", variance_desc_filters, "variance_gen_filters", variance_gen_filters)
        plt.pause(0.05)

        for i in range(100):
            batch_x = get_real_data_batch_x(batch_size)
            batch_z = get_data_batch_z(batch_size)
            _, c1a, c1b = sess.run([desc_optim, real_desc_cost, gen_desc_cost], feed_dict={ X: batch_x, Z: batch_z })
            batch_z = get_data_batch_z(batch_size)
            _, c2 = sess.run([gen_optim, gen_cost], feed_dict={ Z: batch_z })
            costs.append((c1a, c1b, c2))
            print("epoch", epoch, "i", i, "cost_real_desc", c1a, "cost_gen_desc", c1b, "cost_gen", c2)
        cost_plot_a.plot(costs)

        # res1 = sess.run([real_desc], feed_dict={ X: batch_x })
        # print(res1[0])

        res2 = sess.run([gen], feed_dict={ Z: sample_z })
        print(np.min(res2[0][0]), np.max(res2[0][0]))
        for x in range(n_samples_w):
            for y in range(n_samples_h):
                sample_plot_a[x][y].imshow(np.reshape(res2[0][x + y * n_samples_w, 0:image_size, 0:image_size, 0], (image_size, image_size)))
        plt.draw()

        plt.pause(0.05)


print("Optimization Finished!")

plt.waitforbuttonpress()
