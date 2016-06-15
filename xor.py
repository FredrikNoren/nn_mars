import tensorflow as tf
import numpy

train_X = [numpy.matrix([0, 0]), numpy.matrix([0, 1]), numpy.matrix([1, 0]), numpy.matrix([1, 1])]
train_Y = [numpy.matrix([1]), numpy.matrix([0]), numpy.matrix([0]), numpy.matrix([1])]

x = tf.placeholder("float", [1, 2])
y = tf.placeholder("float", [1, 1])

h1 = tf.Variable(tf.random_normal([2, 3]))
h2 = tf.Variable(tf.random_normal([3, 1]))
b1 = tf.Variable(tf.random_normal([3]))
b2 = tf.Variable(tf.random_normal([1]))

layer_0 = tf.add(tf.matmul(x, h1), b1)
y_ = tf.add(tf.matmul(tf.tanh(layer_0), h2), b2)

cost = tf.reduce_mean(tf.square(y - y_)) # tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_, y))
optimizer = tf.train.AdamOptimizer(learning_rate=0.1).minimize(cost)

init = tf.initialize_all_variables()

with tf.Session() as sess:
    sess.run(init)

    for i in range(1000):
        _, c = sess.run([optimizer, cost], feed_dict={x: train_X[i % 4],
                                                      y: train_Y[i % 4]})

        print c, train_X[i % 4], train_Y[i % 4]

    print train_X[0], y_.eval({ x: train_X[0] })
    print train_X[1], y_.eval({ x: train_X[1] })
    print train_X[2], y_.eval({ x: train_X[2] })
    print train_X[3], y_.eval({ x: train_X[3] })
