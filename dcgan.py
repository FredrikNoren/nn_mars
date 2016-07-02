
from __future__ import division, print_function, absolute_import

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

filter_stride_x = 2
filter_stride_y = 2
filter_width = 5
filter_height = 5
batch_size = 100
z_dim = 2
learning_rate = 0.01

def lrelu(x, leak=0.2, name="lrelu"):
    with tf.variable_scope(name):
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)
        return f1 * x + f2 * abs(x)

def conv2d(input_, output_dim, name="conv2d"):
    with tf.variable_scope(name):
        w = tf.get_variable("w", [filter_width, filter_height, input_.get_shape()[-1], output_dim], initializer=tf.random_normal_initializer(stddev=0.1))
        b = tf.get_variable("b", [output_dim], initializer=tf.constant_initializer(0.0))
        conv = tf.nn.conv2d(input_, w, strides=[1, filter_stride_x, filter_stride_y, 1], padding='SAME')
        return tf.nn.bias_add(conv, b)

def deconv2d(input_, output_shape, name="deconv2d"):
    with tf.variable_scope(name):
        w = tf.get_variable("w", [filter_width, filter_height, output_shape[-1], input_.get_shape()[-1]], initializer=tf.random_normal_initializer(stddev=0.1))
        b = tf.get_variable("b", [output_shape[-1]], initializer=tf.constant_initializer(0.0))
        deconv = tf.nn.conv2d_transpose(input_, w, output_shape=output_shape, strides=[1, filter_stride_x, filter_stride_y, 1], padding='SAME')
        return tf.nn.bias_add(deconv, b)

def linear(input_, output_size, name="linear"):
    with tf.variable_scope(name):
        w = tf.get_variable("w", [input_.get_shape()[1], output_size], initializer=tf.random_normal_initializer(stddev=0.1))
        b = tf.get_variable("b", [output_size], initializer=tf.constant_initializer(0.0))
        return tf.matmul(input_, w) + b

def generator(z):
    with tf.variable_scope("generator"):
        l1 = linear(z, 4*4*16)
        l1 = tf.reshape(l1, [-1, 4, 4, 16])
        l2 = lrelu(deconv2d(l1, [batch_size, 8, 8, 8], name="l2"))
        l3 = lrelu(deconv2d(l2, [batch_size, 16, 16, 1], name="l3"))
        return l3

def descriminator(x, reuse=False):
    with tf.variable_scope("descriminator") as scope:
        if reuse:
            scope.reuse_variables()
        l1 = lrelu(conv2d(x, 8, name="l1"))
        print("l1", l1.get_shape())
        l2 = lrelu(conv2d(l1, 16, name="l2"))
        print("l2", l2.get_shape())
        l2 = tf.reshape(l2, [-1, 4*4*16])
        l3 = tf.sigmoid(linear(l2, 1))
        return l3

def gen_data_batch():
    data = np.zeros((batch_size, 16, 16, 1))
    data[0:batch_size, 0:16, 0:8, 0] = 1
    return data

Z = tf.placeholder("float", [None, z_dim])
gen = generator(Z)

X = tf.placeholder("float", [None, 16, 16, 1])
real_desc = descriminator(X)

gen_desc = descriminator(gen, reuse=True)

real_desc_cost = tf.reduce_mean(tf.pow(tf.ones_like(real_desc) - real_desc, 2))
gen_desc_cost = tf.reduce_mean(tf.pow(tf.zeros_like(gen_desc) - gen_desc, 2))
desc_cost = real_desc_cost + gen_desc_cost*0.05

gen_cost = tf.reduce_mean(tf.pow(tf.ones_like(gen_desc) - gen_desc, 2))

t_vars = tf.trainable_variables()
desc_vars = [var for var in t_vars if 'descriminator' in var.name]
gen_vars = [var for var in t_vars if 'generator' in var.name]

desc_optim = tf.train.AdamOptimizer(learning_rate).minimize(desc_cost, var_list=desc_vars)
gen_optim = tf.train.AdamOptimizer(learning_rate).minimize(gen_cost, var_list=gen_vars)

init = tf.initialize_all_variables()
costs = []

plt.ion()

sample_plot_f, sample_plot_a = plt.subplots()
sample_plot_f.show()
cost_plot_f, cost_plot_a = plt.subplots()
cost_plot_f.show()

with tf.Session() as sess:
    sess.run(init)

    for epoch in range(1000):
        batch_x = gen_data_batch()
        batch_z = np.random.uniform(-1, 1, [batch_size, z_dim]).astype(np.float32)
        _, c1 = sess.run([desc_optim, desc_cost], feed_dict={ X: batch_x, Z: batch_z })
        _, c2 = sess.run([gen_optim, gen_cost], feed_dict={ Z: batch_z })
        costs.append((c1, c2))
        cost_plot_a.plot(costs)
        print("cost", c1, c2)


        # res1 = sess.run([real_desc], feed_dict={ X: batch_x })
        # print(res1[0])

        res2 = sess.run([gen], feed_dict={ Z: batch_z })
        sample_plot_a.imshow(np.reshape(res2[0][0, 0:16, 0:16, 0], (16, 16)), interpolation="nearest")
        plt.draw()

        plt.pause(0.05)
