
import numpy
import tensorflow as tf 
import matplotlib.pyplot as plt 
from __future__ import print_function

rng = numpy.random

# parameters ------------------------------------------------------------------
learning_rate = 0.01
training_epochs = 10000
display_step = 50

# training data ---------------------------------------------------------------
X_train = numpy.asarray([3.3,4.4,5.5,6.71,6.93,4.168,9.779,6.182,7.59,2.167,
                         7.042,10.791,5.313,7.997,5.654,9.27,3.1])
Y_train = numpy.asarray([1.7,2.76,2.09,3.19,1.694,1.573,3.366,2.596,2.53,1.221,
                         2.827,3.465,1.65,2.904,2.42,2.94,1.3])

n_samples = X_train.shape[0]

X = tf.placeholder("float")
Y = tf.placeholder("float")

W = tf.Variable(rng.randn(), name = "weight")
b = tf.Variable(rng.randn(), name = "bias")

# construct a linear model
linear_model = tf.add(tf.multiply(X, W), b)
# mean squared error
loss = tf.reduce_sum(tf.pow(linear_model - Y, 2))/(2 * n_samples)

# gradient descent
optimizer = tf.train.GradientDescentOptimizer(learning_rate)
train = optimizer.minimize(loss)

# initialize the variables
init = tf.global_variables_initializer()

# start training --------------------------------------------------------------
with tf.Session() as sess:
	sess.run(init)
	for epoch in range(training_epochs):
		for (x, y) in zip(X_train, Y_train):
			sess.run(train, feed_dict = {X: x, Y: y})
		if (epoch + 1) % display_step == 0:
			c = sess.run(loss, feed_dict = {X: X_train, Y: Y_train})
			print("Epoch:", "%04d" % (epoch + 1), "cost = ", "{:.9f}".format(c), \
				"W = ", sess.run(W), "b = ", sess.run(b))
	print("Optimization Finished.")
	training_cost = sess.run(loss, feed_dict = {X: X_train, Y: Y_train})
	print("Training cost = ", training_cost, "W = ", sess.run(W), "b = ", sess.run(b), "\n")



