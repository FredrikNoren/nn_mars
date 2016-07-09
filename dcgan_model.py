
from __future__ import division, print_function, absolute_import

import tensorflow as tf
import numpy as np
import random
import datetime
import os
import errno


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
        l1 = lrelu(linear(z, 4 * 4 * 256))
        l1 = tf.reshape(l1, [-1, 4, 4, 256])
        l2 = lrelu(deconv2d(l1, [batch_size, 8, 8, 128], name="deconv_l1"))
        l3 = lrelu(deconv2d(l2, [batch_size, 16, 16, 64], name="deconv_l2"))
        l4 = lrelu(deconv2d(l3, [batch_size, 32, 32, 32], name="deconv_l3"))
        l5 = tf.sigmoid(deconv2d(l4, [batch_size, 64, 64, 1], name="deconv_l4"))
        return l5

def descriminator(x, reuse=False):
    with tf.variable_scope("descriminator") as scope:
        if reuse:
            scope.reuse_variables()
        l1 = lrelu(conv2d(x, 32, name="conv_l1"))
        l2 = lrelu(conv2d(l1, 64, name="conv_l2"))
        l3 = lrelu(conv2d(l2, 128, name="conv_l3"))
        l4 = lrelu(conv2d(l3, 256, name="conv_l4"))
        l5 = linear(tf.reshape(l4, [-1, 4 * 4 * 256]), 1)
        return tf.sigmoid(l5), l5, [l1, l2, l3, l4]


class DCGAN(object):
    def __init__(self, sess):
        self.sess = sess
        self.batch_size = 16
        self.z_dim = 100
        self.learning_rate_desc = 0.0001
        self.learning_rate_gen  = 0.0001
        self.image_size = 64

        self.Z = tf.placeholder("float", [None, self.z_dim])
        self.gen = generator(self.Z)

        self.X = tf.placeholder("float", [None, self.image_size, self.image_size, 1])
        self.real_desc, self.real_desc_logits, self.real_desc_filters_out = descriminator(self.X)

        self.gen_desc, self.gen_desc_logits, self.gen_desc_filters_out = descriminator(self.gen, reuse=True)

    def load(self, path):
        saver = tf.train.Saver()
        saver.restore(self.sess, path)

    def prepare_train(self):
        print("Loading mars data")
        self.mars_data = np.load("hirise_20.npy")
        print("Loading mars data done.")

    def train(self, epoch_callback=None):

        saver = tf.train.Saver()
        save_path = datetime.datetime.now().strftime('model/%b%d_%H%M')
        mkdir_p(save_path)
        save_filename = save_path + '/save'

        real_desc_cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(self.real_desc_logits, tf.ones_like(self.real_desc_logits)))
        gen_desc_cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(self.gen_desc_logits, tf.zeros_like(self.gen_desc_logits)))
        gen_cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(self.gen_desc_logits, tf.ones_like(self.gen_desc_logits)))

        # real_desc_cost = tf.reduce_mean(tf.pow(tf.ones_like(real_desc) - real_desc, 2))
        # gen_desc_cost = tf.reduce_mean(tf.pow(tf.zeros_like(gen_desc) - gen_desc, 2))
        # gen_cost = tf.reduce_mean(tf.pow(tf.ones_like(gen_desc) - gen_desc, 2))

        desc_cost = real_desc_cost + gen_desc_cost

        t_vars = tf.trainable_variables()
        desc_vars = [var for var in t_vars if 'descriminator' in var.name]
        gen_vars = [var for var in t_vars if 'generator' in var.name]
        self.desc_filter_vars = [var for var in desc_vars if 'conv_' in var.name and '/w' in var.name]
        self.gen_filter_vars = [var for var in gen_vars if 'conv_' in var.name and '/w' in var.name]

        desc_optim = tf.train.AdamOptimizer(self.learning_rate_desc, beta1=0.5).minimize(desc_cost, var_list=desc_vars)
        gen_optim = tf.train.AdamOptimizer(self.learning_rate_gen, beta1=0.5).minimize(gen_cost, var_list=gen_vars)

        init = tf.initialize_all_variables()

        self.sess.run(init)

        for epoch in range(100000):
            batch_x = self.get_real_data_batch_x(self.batch_size)
            batch_z = self.get_data_batch_z(self.batch_size)
            _, real_desc_cost_val, gen_desc_cost_val = self.sess.run([desc_optim, real_desc_cost, gen_desc_cost], feed_dict={ self.X: batch_x, self.Z: batch_z })
            batch_z = self.get_data_batch_z(self.batch_size)
            _, gen_cost_val = self.sess.run([gen_optim, gen_cost], feed_dict={ self.Z: batch_z })
            if epoch % 100 == 0:
                p = saver.save(self.sess, save_filename, global_step=epoch)
                print('PATH', p)
            if epoch_callback:
                epoch_callback(epoch, (real_desc_cost_val, gen_desc_cost_val, gen_cost_val))

        print("Optimization Finished!")

    def get_subimage(self):
        i = random.randint(0, len(self.mars_data) - 1)
        xpos = random.randint(0, self.mars_data[i].shape[0])
        ypos = random.randint(0, self.mars_data[i].shape[1])
        subdata = self.mars_data[i][xpos:(xpos + self.image_size), ypos:(ypos + self.image_size)]
        if subdata.shape[0] != self.image_size or subdata.shape[1] != self.image_size or np.min(subdata) < -5000 or np.var(subdata) < 1000:
            return self.get_subimage()
        else:
            return subdata

    def get_real_data_batch_x(self, batch_size):
        batch_xs = np.zeros((batch_size, self.image_size, self.image_size, 1))
        max_subimage_diff = 1226.27258301 # Sampled 50000 subimages to see what the biggest diff was
        for i in range(batch_size):
            x = self.get_subimage()
            x = (x - np.min(x)) / max_subimage_diff
            x = np.rot90(x, random.randint(0, 3))
            if random.randint(0, 1) == 0:
                x = np.fliplr(x)
            if random.randint(0, 1) == 0:
                x = np.flipud(x)
            batch_xs[i, 0:, 0:, 0] = x
        return batch_xs

    def get_data_batch_z(self, batch_size):
        return np.random.uniform(-1, 1, [batch_size, self.z_dim]).astype(np.float32)


def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise
