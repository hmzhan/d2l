
import os
import sys
import time
import argparse

import tensorflow as tf 
from six.moves import xrange
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.examples.tutorials.mnist import mnist

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

FLAGS = None

def placeholder_inputs(batch_size):
	images_placeholder = tf.placeholder(
		tf.float32,
		shape = (batch_size, mnist.IMAGE_PIXELS)
	)
	labels_placeholder = tf.placeholder(
		tf.int32,
		shape = (batch_size)
	)
	return images_placeholder, labels_placeholder


def fill_feed_dict(data_set, images_pl, labels_pl):
	images_feed, labels_feed = data_set.next_batch(
		FLAGS.batch_size,
		FLAGS.fake_data
	)
	feed_dict = {
	    images_pl: images_feed,
	    labels_pl: labels_feed
	}
	return feed_dict


def do_eval(sess, 
	        eval_correct, 
	        images_placeholder, 
	        labels_placeholder, 
	        data_set):
	true_count = 0
	steps_per_epoch = data_set.num_examples // FLAGS.batch_size
	num_examples = steps_per_epoch * FLAGS.batch_size
	for step in xrange(steps_per_epoch):
		feed_dict = fill_feed_dict(
			data_set,
			images_placeholder,
			labels_placeholder
		)
		true_count += sess.run(eval_correct, feed_dict = feed_dict)
	precision = float(true_count) / num_examples
	print(' Num examples: %d Num correct: %d Precision @ 1: %0.04f' %
		(num_examples, true_count, precision))




















