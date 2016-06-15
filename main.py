
from __future__ import division, print_function, absolute_import

import tensorflow as tf
import numpy as np

import gdal
import gdalconst
import matplotlib.pyplot as plt
import random

# Source data http://www.uahirise.org
mars = [
    gdal.Open("DTEEC_036307_1665_035951_1665_A01.IMG", gdalconst.GA_ReadOnly),
    gdal.Open("DTEED_032193_1990_031837_1990_A01.IMG", gdalconst.GA_ReadOnly),
    gdal.Open("DTEEC_034109_2085_033608_2085_A01.IMG", gdalconst.GA_ReadOnly),
    ]
mars_data = []
for v in mars: mars_data.append(v.ReadAsArray())


# Parameters
learning_rate = 0.01
training_epochs = 50
display_step = 1
batch_size = 256
image_size = 22

# Network Parameters
n_hidden_1 = 32 # 1st layer num features
n_hidden_2 = 16 # 2nd layer num features
n_input = image_size * int(image_size/2)
n_output = image_size * int(image_size/2)


def draw_mars_data(data):
    u = np.unique(data.flatten())
    u.sort()
    print(u[0], u[1])
    #print np.min(data), np.max(data)
    data = np.clip(data, u[1], np.max(data))
    imgplot = plt.imshow(data, interpolation="nearest")
    plt.waitforbuttonpress()

# draw_mars_data(mars_data[2])
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
        x = full[0:int(image_size/2)]
        y = full[int(image_size/2):image_size]
        # print(x.shape)
        # imgplot = plt.imshow(y, interpolation="nearest")
        # plt.waitforbuttonpress()
        # exit()
        x = np.reshape(x, (image_size * int(image_size/2)))
        y = np.reshape(y, (image_size * int(image_size/2)))
        batch_xs.append(x)
        batch_ys.append(y)
        batch_full.append(full)
    return batch_xs, batch_ys, batch_full


# tf Graph input (only pictures)
X = tf.placeholder("float", [None, n_input])
Y = tf.placeholder("float", [None, n_output])

weights = {
    'encoder_h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'encoder_h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'decoder_h1': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_1])),
    'decoder_h2': tf.Variable(tf.random_normal([n_hidden_1, n_output])),
}
biases = {
    'encoder_b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'encoder_b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'decoder_b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'decoder_b2': tf.Variable(tf.random_normal([n_output])),
}

# Building the encoder
def encoder(x):
    # Encoder Hidden layer with sigmoid activation #1
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['encoder_h1']),
                                   biases['encoder_b1']))
    # Decoder Hidden layer with sigmoid activation #2
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['encoder_h2']),
                                   biases['encoder_b2']))
    return layer_2


# Building the decoder
def decoder(x):
    # Encoder Hidden layer with sigmoid activation #1
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['decoder_h1']),
                                   biases['decoder_b1']))
    # Decoder Hidden layer with sigmoid activation #2
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['decoder_h2']),
                                   biases['decoder_b2']))
    return layer_2

# Construct model
encoder_op = encoder(X)
decoder_op = decoder(encoder_op)

# Prediction
y_pred = decoder_op
# Targets (Labels) are the input data.
y_true = Y

# Define loss and optimizer, minimize the squared error
cost = tf.reduce_mean(tf.pow(y_true - y_pred, 2))
optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(cost)

# Initializing the variables
init = tf.initialize_all_variables()

# Launch the graph
# Using InteractiveSession (more convenient while using Notebooks)
sess = tf.InteractiveSession()
sess.run(init)

plt.axis([0, training_epochs, 0, 1])
plt.ion()

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
        plt.plot([epoch - 1, epoch], [plot_prev, c])
        plot_prev = c
        plt.pause(0.05)
        print("Epoch:", '%04d' % (epoch+1),
              "cost=", "{:.9f}".format(c))

print("Optimization Finished!")

# def show_generated_landscape(width, height):
#     test_set_xs, test_set_ys, test_set_full = get_data_batch(1)
#     landscape = np.zeros((width, height))
#     landscape[]

def show_test_set():
    # Applying encode and decode over test set
    test_count = 7
    test_set_xs, test_set_ys, test_set_full = get_data_batch(test_count)
    encode_decode = sess.run(y_pred, feed_dict={X: test_set_xs, Y: test_set_ys})

    # Compare original images with their reconstructions
    f, a = plt.subplots(2, test_count, figsize=(test_count, 2))
    for i in range(test_count):
        a[0][i].imshow(np.reshape(test_set_full[i], (image_size, image_size)))
        x = np.reshape(test_set_xs[i], (int(image_size/2), image_size))
        y = np.reshape(encode_decode[i], (int(image_size/2), image_size))
        a[1][i].imshow(np.concatenate((x, y)))
    f.show()
    plt.draw()
    plt.waitforbuttonpress()

show_test_set()
