import sys
import tensorflow as tf
import numpy as np
import math
import time
from tensorflow.examples.tutorials.mnist import input_data
from mnist_models import *

# NN Model
# lr = 8e-3
# total_epochs = 50
# batch_size = 256
# X, y, yhat = mnist_nn()

# Conv net Model
# 98.66% 
# lr = 3e-3
# total_epochs = 20
# batch_size = 48
# X, y, yhat = mnist_conv_net()

# Conv net
lr = 1e-4
dropout = 0.5
total_epochs = 20
batch_size = 48
X, y, dropout_prob, yhat = mnist_conv_net()

# Load data
mnist = input_data.read_data_sets("data/mnist", one_hot=True)

# Loss function
loss_func = tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=yhat)
avg_ce_loss_func = tf.reduce_mean(loss_func)

# Optimizer
optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(avg_ce_loss_func)

# Validation
dev_data, dev_labels = mnist.validation.next_batch(5000)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    n_batches = int(mnist.train.num_examples / batch_size)
    
    for epoch in range(total_epochs):
        loss_per_epoch = 0

        start = time.time()

        for batch in range(n_batches):
            train, labels = mnist.train.next_batch(batch_size)
            avg_ce_loss_train, logits_train, loss, _ = sess.run([avg_ce_loss_func, yhat, loss_func, optimizer], 
                                                                feed_dict={X: train, y: labels, dropout_prob: dropout})
            loss_per_epoch += avg_ce_loss_train

        loss_per_epoch /= n_batches

        avg_ce_loss_dev, logits_dev = sess.run([avg_ce_loss_func, yhat], feed_dict={X: dev_data, y: dev_labels, dropout_prob: 1.0})

        argmax_logits_dev = np.argmax(logits_dev, axis=1)
        argmax_dev_labels = np.argmax(dev_labels, axis=1)
        acc_dev = argmax_logits_dev == argmax_dev_labels
        acc_dev = acc_dev.sum() / len(acc_dev)

        end = time.time()

        print("Epoch {0}: loss={1}, devloss={2}, devacc={3}, time={4}secs".format(epoch, loss_per_epoch, avg_ce_loss_dev, acc_dev, end-start))

    test_data, test_labels = mnist.test.next_batch(10000)
    avg_ce_loss_test, logits_test = sess.run([avg_ce_loss_func, yhat], feed_dict={X: test_data, y: test_labels, dropout_prob: 1.0})
    acc = np.argmax(test_labels, axis=1) == np.argmax(logits_test, axis=1)
    acc = acc.sum() / len(acc)
    print("{0} {1}".format(avg_ce_loss_test, acc))

