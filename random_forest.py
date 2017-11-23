
# random forest

import os
import tensorflow as tf 
from tensorflow.contrib.tensor_forest.python import tensor_forest
from tensorflow.examples.tutorials.mnist import input_data
from __future__ import print_function

os.environ["CUDA_VISIBLE_DEVICES"] = ""

# import data
mnist = input_data.read_data_sets("MNIST_data/", one_hot = False)

# parameters
num_steps = 500
batch_size = 1024
num_classes = 10
num_features = 784
num_trees = 10
max_nodes = 1000

# input and target data
X = tf.placeholder(tf.float32, shape = [None, num_features])
Y = tf.placeholder(tf.int32, shape = [None])

# random forest parameters
hparams = tensor_forest.ForestHParams(
	num_classes = num_classes,
	num_features = num_features,
	num_trees = num_trees,
	max_nodes = max_nodes
	).fill()

# build the random forest
forest_graph = tensor_forest.RandomForestGraphs(hparams)
train_op = forest_graph.training_graph(X, Y)
loss_op = forest_graph.training_loss(X, Y)

# measure the accuracy
infer_op = forest_graph.inference_graph(X)

























