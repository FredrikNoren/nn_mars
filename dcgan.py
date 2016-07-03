
from __future__ import division, print_function, absolute_import

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import gdal
import gdalconst
import random

mars = [
    gdal.Open("DTEEC_036307_1665_035951_1665_A01.IMG", gdalconst.GA_ReadOnly),
    gdal.Open("DTEED_032193_1990_031837_1990_A01.IMG", gdalconst.GA_ReadOnly),
    gdal.Open("DTEEC_034109_2085_033608_2085_A01.IMG", gdalconst.GA_ReadOnly),
    gdal.Open("DTEED_022962_1570_022672_1570_A01.IMG", gdalconst.GA_ReadOnly),
    gdal.Open("DTEEC_028149_2085_028215_2085_A01.IMG", gdalconst.GA_ReadOnly),
    ]
mars_data = []
for v in mars: mars_data.append(v.ReadAsArray())

filter_stride_x = 2
filter_stride_y = 2
filter_width = 5
filter_height = 5
batch_size = 60
z_dim = 40
learning_rate = 0.001
image_size = 16
n_samples_w = 5
n_samples_h = 5

def lrelu(x, leak=0.2, name="lrelu"):
    with tf.variable_scope(name):
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)
        return f1 * x + f2 * abs(x)

def conv2d(input_, output_dim, name="conv2d"):
    with tf.variable_scope(name):
        w = tf.get_variable("w", [filter_width, filter_height, input_.get_shape()[-1], output_dim], initializer=tf.truncated_normal_initializer(stddev=0.02))
        b = tf.get_variable("b", [output_dim], initializer=tf.constant_initializer(0.0))
        conv = tf.nn.conv2d(input_, w, strides=[1, filter_stride_x, filter_stride_y, 1], padding='SAME')
        return tf.nn.bias_add(conv, b)

def deconv2d(input_, output_shape, name="deconv2d"):
    with tf.variable_scope(name):
        w = tf.get_variable("w", [filter_width, filter_height, output_shape[-1], input_.get_shape()[-1]], initializer=tf.random_normal_initializer(stddev=0.02))
        b = tf.get_variable("b", [output_shape[-1]], initializer=tf.constant_initializer(0.0))
        deconv = tf.nn.conv2d_transpose(input_, w, output_shape=output_shape, strides=[1, filter_stride_x, filter_stride_y, 1], padding='SAME')
        return tf.nn.bias_add(deconv, b)

def linear(input_, output_size, name="linear"):
    with tf.variable_scope(name):
        w = tf.get_variable("w", [input_.get_shape()[1], output_size], initializer=tf.random_normal_initializer(stddev=0.02))
        b = tf.get_variable("b", [output_size], initializer=tf.constant_initializer(0.0))
        return tf.matmul(input_, w) + b

def generator(z):
    with tf.variable_scope("generator"):
        batch_size = tf.shape(z)[0]
        l1 = tf.tanh(linear(z, 4*4*16))
        l1 = tf.reshape(l1, [-1, 4, 4, 16])
        l2 = lrelu(deconv2d(l1, [batch_size, 8, 8, 8], name="l2"))
        l3 = lrelu(deconv2d(l2, [batch_size, 16, 16, 1], name="l3"))
        return tf.tanh(l3)

def descriminator(x, reuse=False):
    with tf.variable_scope("descriminator") as scope:
        if reuse:
            scope.reuse_variables()
        l1 = lrelu(conv2d(x, 8, name="l1"))
        l2 = lrelu(conv2d(l1, 16, name="l2"))
        l2 = tf.reshape(l2, [-1, 4*4*16])
        l3 = linear(l2, 1)
        return tf.tanh(l3), l3


def get_subimage():
    i = random.randint(0, len(mars_data) - 1)
    xpos = random.randint(0, mars_data[i].shape[0])
    ypos = random.randint(0, mars_data[i].shape[1])
    subdata = mars_data[i][xpos:(xpos + image_size), ypos:(ypos + image_size)]
    if subdata.shape[0] != image_size or subdata.shape[1] != image_size or np.min(subdata) < -5000:
        return get_subimage()
    else:
        return subdata

def get_real_data_batch():
    batch_xs = np.zeros((batch_size, 16, 16, 1))
    for i in range(batch_size):
        x = get_subimage()
        x = (x - np.min(x)) / (np.max(x) - np.min(x))
        batch_xs[i, 0:image_size, 0:image_size, 0] = x
    return batch_xs

# def get_real_data_batch():
#     data = np.zeros((batch_size, 16, 16, 1))
#     data[0:batch_size, 0:16, 0:8, 0] = 1
#     return data

Z = tf.placeholder("float", [None, z_dim])
gen = generator(Z)

X = tf.placeholder("float", [None, image_size, image_size, 1])
real_desc, real_desc_logits = descriminator(X)

gen_desc, gen_desc_logits = descriminator(gen, reuse=True)

real_desc_cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(real_desc_logits, tf.ones_like(real_desc_logits)))
gen_desc_cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(gen_desc_logits, tf.zeros_like(gen_desc_logits)))
gen_cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(gen_desc_logits, tf.ones_like(gen_desc_logits)))

# real_desc_cost = tf.reduce_mean(tf.pow(tf.ones_like(real_desc) - real_desc, 2))
# gen_desc_cost = tf.reduce_mean(tf.pow(tf.zeros_like(gen_desc) - gen_desc, 2))
# gen_cost = tf.reduce_mean(tf.pow(tf.ones_like(gen_desc) - gen_desc, 2))

desc_cost = real_desc_cost + gen_desc_cost

t_vars = tf.trainable_variables()
desc_vars = [var for var in t_vars if 'descriminator' in var.name]
gen_vars = [var for var in t_vars if 'generator' in var.name]

desc_optim = tf.train.AdamOptimizer(learning_rate).minimize(desc_cost, var_list=desc_vars)
gen_optim = tf.train.AdamOptimizer(learning_rate).minimize(gen_cost, var_list=gen_vars)

init = tf.initialize_all_variables()
costs = []

plt.ion()

sample_plot_f, sample_plot_a = plt.subplots(n_samples_w, n_samples_h, figsize=(n_samples_w, n_samples_h))
sample_plot_f.show()
cost_plot_f, cost_plot_a = plt.subplots()
cost_plot_f.show()
real_plot_f, real_plot_a = plt.subplots(n_samples_w, n_samples_h, figsize=(n_samples_w, n_samples_h))
real_plot_f.show()

real_data = get_real_data_batch()
for x in range(n_samples_w):
    for y in range(n_samples_h):
        real_plot_a[x][y].imshow(np.reshape(real_data[x + y * n_samples_w, 0:image_size, 0:image_size, 0], (image_size, image_size)), interpolation="nearest")

sample_z = np.random.uniform(-1, 1, [n_samples_w*n_samples_h, z_dim]).astype(np.float32)

with tf.Session() as sess:
    sess.run(init)

    for epoch in range(100):
        for i in range(100):
            batch_x = get_real_data_batch()
            batch_z = np.random.uniform(-1, 1, [batch_size, z_dim]).astype(np.float32)
            _, c1 = sess.run([desc_optim, desc_cost], feed_dict={ X: batch_x, Z: batch_z })
            _, c2 = sess.run([gen_optim, gen_cost], feed_dict={ Z: batch_z })
            costs.append((c1, c2))
            print("epoch", epoch, "i", i, "cost_desc", c1, "cost_gen", c2)
        cost_plot_a.plot(costs)

        # res1 = sess.run([real_desc], feed_dict={ X: batch_x })
        # print(res1[0])

        res2 = sess.run([gen], feed_dict={ Z: sample_z })
        for x in range(n_samples_w):
            for y in range(n_samples_h):
                sample_plot_a[x][y].imshow(np.reshape(res2[0][x + y * n_samples_w, 0:image_size, 0:image_size, 0], (image_size, image_size)), interpolation="nearest")
        plt.draw()

        plt.pause(0.05)


print("Optimization Finished!")

plt.waitforbuttonpress()
