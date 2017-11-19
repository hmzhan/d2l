

import tensorflow as tf

# node and session ------------------------------------------------------------
node1 = tf.constant(3.0, dtype = tf.float32)
node2 = tf.constant(4.0)
print(node1, node2)

sess = tf.Session()
print(sess.run([node1, node2]))

from __future__ import print_function
node3 = tf.add(node1, node2)
print("node3:", node3)
print("sess.run(node3):", sess.run(node3))


# placeholder -----------------------------------------------------------------
a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)
adder_node = a + b
print(sess.run(adder_node, {a: 3, b: 4.5}))
print(sess.run(adder_node, {a: [1, 3], b: [2, 4]}))

add_and_triple = adder_node * 3
print(sess.run(add_and_triple, {a: 3, b: 4.5}))


# variable --------------------------------------------------------------------
W = tf.Variable([.3], dtype = tf.float32)
b = tf.Variable([-.3], dtype = tf.float32)
x = tf.placeholder(tf.float32)
linear_model = W*x + b

init = tf.initialize_all_variables()  # global_variables_initializer does not work
sess.run(init)
print(sess.run(linear_model, {x: [1, 2, 3, 4]}))


# model evaluation ------------------------------------------------------------
y = tf.placeholder(tf.float32)
squared_deltas = tf.square(linear_model - y)
loss = tf.reduce_sum(squared_deltas)
print(sess.run(loss, {x: [1,2,3,4], y:[0,-1,-2,-3]}))

fixW = tf.assign(W, [-1.])
fixb = tf.assign(b, [1.])
sess.run([fixW, fixb])
print(sess.run(loss, {x: [1,2,3,4], y:[0,-1,-2,-3]}))


# train API -------------------------------------------------------------------
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)

sess.run(init)
for i in range(1000):
	sess.run(train, {x: [1,2,3,4], y:[0,-1,-2,-3]})

print(sess.run([W, b]))


# complete program ------------------------------------------------------------
import tensorflow as tf
# model parameters
W = tf.Variable([.3], dtype = tf.float32)
b = tf.Variable([-.3], dtype = tf.float32)
# model input and output
x = tf.placeholder(tf.float32)
linear_model = W*x + b
y = tf.placeholder(tf.float32)
# loss 
loss = tf.reduce_sum(tf.square(linear_model - y))
# optimizer
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)
# training data
x_train = [1,2,3,4]
y_train = [0,-1,-2,-3]
# training loop
init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)
for i in range(1000):
	sess.run(train, {x: x_train, y: y_train})
# evaluate training accuracy
curr_W, curr_b, curr_loss = sess.run([W, b, loss], {x: x_train, y:y_train})
print("W: %s b: %s loss: %s" % (curr_W, curr_b, curr_loss))


# estimator -------------------------------------------------------------------
import numpy as np 
import tensorflow as tf  

feature_columns = [tf.feature_column.numeric_column("x", shape = [1])]
estimator = tf.estimator.LinearRegressor(feature_columns = feature_columns)

x_train = np.array([1., 2., 3., 4.])
y_train = np.array([0., -1., -2., -3.])
x_eval = np.array([2., 5., 8., 1.])
y_eval = np.array([-1.01, -4.1, -7, 0.])

input_fn = tf.estimator.inputs.numpy_input_fn(
	{"x": x_train}, y_train, batch_size = 4, num_epochs = None, shuffle = True)
train_input_fn = tf.estimator.inputs.numpy_input_fn(
	{"x": x_train}, y_train, batch_size = 4, num_epochs = 1000, shuffle = False)
eval_input_fn = tf.estimator.inputs.numpy_input_fn(
	{"x": x_eval}, y_eval, batch_size = 4, num_epochs = 1000, shuffle = False)

estimator.train(input_fn = input_fn, steps = 1000)
train_metrics = estimator.evaluate(input_fn = train_input_fn)
eval_metrics = estimator.evaluate(input_fn = eval_input_fn)
print("train metrics: %r" % train_metrics)
print("eval metrics: %r" % eval_metrics)
