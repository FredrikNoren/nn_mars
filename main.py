
from __future__ import division, print_function, absolute_import

import tensorflow as tf
import numpy as np

import gdal
import gdalconst
import matplotlib.pyplot as plt
import random
from skimage.measure import block_reduce
import math

def draw_mars_data(data):
    u = np.unique(data.flatten())
    u.sort()
    print(u[0], u[1])
    #print np.min(data), np.max(data)
    data = np.clip(data, u[1], np.max(data))
    imgplot = plt.imshow(data, interpolation="nearest")
    plt.waitforbuttonpress()
#
# #gtif = gdal.Open("DTEEC_036307_1665_035951_1665_A01.IMG", gdalconst.GA_ReadOnly)
# mars_global = gdal.Open("MOLA/megt90n000cb.lbl", gdalconst.GA_ReadOnly)
# print(mars_global.GetGeoTransform())
# #print(mars_global.ReadAsArray().shape)
# draw_mars_data(mars_global.ReadAsArray())
# exit()

# Complete MOLA data: http://astrogeology.usgs.gov/search/map/Mars/GlobalSurveyor/MOLA/Mars_MGS_MOLA_DEM_mosaic_global_463m
# Source data http://www.uahirise.org
mars = [
    gdal.Open("DTEEC_036307_1665_035951_1665_A01.IMG", gdalconst.GA_ReadOnly),
    gdal.Open("DTEED_032193_1990_031837_1990_A01.IMG", gdalconst.GA_ReadOnly),
    gdal.Open("DTEEC_034109_2085_033608_2085_A01.IMG", gdalconst.GA_ReadOnly),
    gdal.Open("DTEED_022962_1570_022672_1570_A01.IMG", gdalconst.GA_ReadOnly),
    gdal.Open("DTEEC_028149_2085_028215_2085_A01.IMG", gdalconst.GA_ReadOnly),
    ]
mars_data = []
for v in mars: mars_data.append(v.ReadAsArray())


# Parameters
learning_rate = 0.01
training_epochs = 200
display_step = 1
batch_size = 256
image_size = 16
input_image_size = 16
show_test_count = 7

# Network Parameters
n_filters = 16
filter_size = 3
n_input = input_image_size * input_image_size
n_output = image_size * image_size
n_hidden = 32 # 1st layer num features
n_hidden_encoding = 32 # 2nd layer num features



def downsample_block_mean(image, size):
    return block_reduce(image, block_size=(size, size), func=np.mean)

# draw_mars_data(mars_data[4])
# exit()

def get_subimage():
    i = random.randint(0, len(mars_data) - 1)
    xpos = random.randint(0, mars_data[i].shape[0])
    ypos = random.randint(0, mars_data[i].shape[1])
    subdata = mars_data[i][xpos:(xpos + image_size), ypos:(ypos + image_size)]
    # print(xpos, ypos, subdata.shape)
    #print subdata.shape[0],subdata.shape[1]
    if subdata.shape[0] != image_size or subdata.shape[1] != image_size or np.min(subdata) < -5000:
        return get_subimage()
    else:
        return subdata

def get_data_batch(count):
    batch_xs = []
    batch_ys = []
    batch_full = []
    for i in range(count):
        full = get_subimage()
        full = (full - np.min(full)) / (np.max(full) - np.min(full))
        #print(np.min(y), np.max(y))
        x = downsample_block_mean(full, int(image_size / input_image_size))
        #x = full[0:image_size:int(image_size / input_image_size), 0:image_size:int(image_size / input_image_size)]
        y = full
        # print(y.shape)
        # print(x.shape)
        # imgplot = plt.imshow(x, interpolation="nearest")
        # plt.waitforbuttonpress()
        # imgplot = plt.imshow(y, interpolation="nearest")
        # plt.waitforbuttonpress()
        # exit()
        x = np.reshape(x, (input_image_size * input_image_size))
        y = np.reshape(y, (image_size * image_size))
        batch_xs.append(x)
        batch_ys.append(y)
        batch_full.append(full)
    return batch_xs, batch_ys, batch_full

conv_h = tf.Variable(tf.truncated_normal([filter_size, filter_size, 1, n_filters], stddev=0.1))
conv_b = tf.Variable(tf.constant(0.0, shape=[n_filters]))
def encoder(x):
    x_image = tf.reshape(x, [-1,image_size,image_size,1])
    conv_l = tf.nn.relu(tf.nn.conv2d(x_image, conv_h, strides=[1, 1, 1, 1], padding='SAME') + conv_b)
    conv_l_reshape = tf.reshape(conv_l, [-1, image_size * image_size * n_filters])
    enc_l = tf.nn.tanh(tf.add(tf.matmul(conv_l_reshape, tf.Variable(tf.truncated_normal([image_size * image_size * n_filters, n_hidden_encoding], stddev=0.1))),
                                     tf.Variable(tf.constant(0.1, shape=[n_hidden_encoding]))))
    return enc_l, conv_l

dec_h_1 = tf.Variable(tf.truncated_normal([n_hidden_encoding, image_size * image_size * n_filters], stddev=0.1))
dec_b_1 = tf.Variable(tf.constant(0.1, shape=[image_size * image_size * n_filters]))
deconv_b = tf.Variable(tf.constant(0.0, shape=[image_size * image_size]))
#
# dec_h_2 = tf.Variable(tf.truncated_normal([image_size * image_size * 32, n_output], stddev=0.1))
# dec_b_2 = tf.Variable(tf.constant(0.1, shape=[n_output]))
def decoder(x):
    print(x.get_shape())
    l_0 = tf.nn.relu(tf.add(tf.matmul(x, dec_h_1), dec_b_1))
    l_0_reshape = tf.reshape(l_0, [-1, image_size, image_size, n_filters])
    print(l_0_reshape.get_shape())
    # shape = (?/256, 16, 16, 32)
    deconv_l = tf.nn.conv2d_transpose(l_0_reshape, conv_h, tf.pack([batch_size, image_size, image_size, 1]), strides=[1, 1, 1, 1], padding='SAME')
    print(deconv_l.get_shape())
    deconv_l_reshape = tf.reshape(deconv_l, [-1, image_size * image_size])
    print(deconv_l_reshape.get_shape())
    l_1 = tf.nn.relu(deconv_l_reshape + deconv_b)
    print(l_1.get_shape())

    return l_1

test_set_xs, test_set_ys, test_set_full = get_data_batch(batch_size)


disp_encoding_placeholder = tf.placeholder("float", [None, n_hidden_encoding])
disp_decoder = decoder(disp_encoding_placeholder)
disp_rand_encoding = []
tmp_rand = np.random.rand(n_hidden_encoding)
for i in range(batch_size):
    r = np.copy(tmp_rand)
    r[0] = i / show_test_count
    disp_rand_encoding.append(r)
def show_test_set(sess, epoch):
    encode_decode, conv_l_out = sess.run([y_pred, conv_l], feed_dict={X: test_set_xs, Y: test_set_ys})
    decoder_random_res = sess.run(disp_decoder, feed_dict={disp_encoding_placeholder: disp_rand_encoding })
    show_filters(conv_l_out, epoch)

    for i in range(show_test_count):
        test_viz_a[0][i].imshow(np.reshape(test_set_xs[i], (input_image_size, input_image_size)), interpolation="nearest")
        test_viz_a[1][i].imshow(np.reshape(test_set_ys[i], (image_size, image_size)), interpolation="nearest")
        test_viz_a[2][i].imshow(np.reshape(encode_decode[i], (image_size, image_size)), interpolation="nearest")
        test_viz_a[3][i].imshow(np.reshape(encode_decode[i] - test_set_ys[i] , (image_size, image_size)), interpolation="nearest")
        test_viz_a[4][i].imshow(np.reshape(decoder_random_res[i], (image_size, image_size)), interpolation="nearest")
        # x = np.reshape(test_set_xs[i], (int(image_size/2), image_size))
        # y = np.reshape(encode_decode[i], (int(image_size/2), image_size))
        # a[1][i].imshow(np.concatenate((x, y)))
    plt.draw()

n_filters_sq = int(math.ceil(math.sqrt(n_filters)))
filters_plot_f, filters_plot_a = plt.subplots(n_filters_sq, n_filters_sq, figsize=(n_filters_sq, n_filters_sq))
filters_plot_f.show()
def show_filters(data, epoch):
    print("filters", data.shape)
    for y in range(n_filters_sq):
        for x in range(n_filters_sq):
            i = y*n_filters_sq + x
            if i < n_filters:
                filters_plot_a[y][x].imshow(data[0, 0:image_size, 0:image_size, i], interpolation="nearest")

# tf Graph input (only pictures)
X = tf.placeholder("float", [None, n_input])
Y = tf.placeholder("float", [None, n_output])

# Construct model
encoder_op, conv_l = encoder(X)
decoder_op = decoder(encoder_op)

# Prediction
y_pred = decoder_op
# Targets (Labels) are the input data.
y_true = Y

# Define loss and optimizer, minimize the squared error
cost = tf.reduce_mean(tf.pow(y_true - y_pred, 2))
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)
# AdamOptimizer best so far, 0.006802175 after 90 epochs

# Initializing the variables
init = tf.initialize_all_variables()

# Launch the graph
# Using InteractiveSession (more convenient while using Notebooks)
sess = tf.InteractiveSession()
sess.run(init)

# plt.axis([0, training_epochs, 0, 1])
plt.ion()

test_viz_f, test_viz_a = plt.subplots(5, show_test_count, figsize=(show_test_count, 5))
test_viz_f.show()

test_viz_a[0][0].set_ylabel("X")
test_viz_a[1][0].set_ylabel("Y")
test_viz_a[2][0].set_ylabel("Y_pred")
test_viz_a[3][0].set_ylabel("Error")
test_viz_a[4][0].set_ylabel("Sampled")

learn_viz_f, learn_viz_a = plt.subplots()
learn_viz_f.show()
learn_viz_a.axis([0, training_epochs, 0, 0.02])

total_batch = 10
plot_prev = 1
# Training cycle
for epoch in range(training_epochs):
    # Loop over all batches
    for i in range(total_batch):
        batch_xs, batch_ys, _ = get_data_batch(batch_size)
        # Run optimization op (backprop) and cost op (to get loss value)
        _, c = sess.run([optimizer, cost], feed_dict={X: batch_xs, Y: batch_ys})
    # Display logs per epoch step
    if epoch % display_step == 0:
        learn_viz_a.plot([epoch - 1, epoch], [plot_prev, c])
        plot_prev = c
        show_test_set(sess, epoch)
        #show_filters(conv_h.eval())
        plt.pause(0.05)
        print("Epoch:", '%04d' % (epoch+1),
              "cost=", "{:.9f}".format(c))

print("Optimization Finished!")

# show_test_set(sess)
plt.waitforbuttonpress()
