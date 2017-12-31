import tensorflow as tf
import os
import numpy as np
from six import iteritems
import pickle
from tensorflow.contrib.tensorboard.plugins import projector

def pickle_it(d, filename):
    with open(filename, "wb") as f:
        pickle.dump(d, f)

def text2matrix(filename, window_size=4):
    with open(filename, "r") as f:
        words = f.read().split(' ')

    word2id = {}
    id2word = {}

    for i, word in enumerate(set(words)):
        word2id[word] = i
        id2word[i] = word

    matrix = np.zeros((len(word2id), len(word2id)))
    for index, word in enumerate(words):
        center = word
        context = words[max(0, index - window_size) : index] + words[index + 1 : min(index + 1 + window_size, len(words))]
        for context_word in context:
            matrix[word2id[center], word2id[context_word]] += 1.

    return word2id, id2word, matrix

def save_vocab(id2word, path):
    with open(path, "w") as f:
        for i, word in sorted(iteritems(id2word), key=lambda x: x[0]):
            f.write(word + "\n")

def svd_model(matrix, embedding_size=256):
    with tf.Graph().as_default() as g:
        X = tf.placeholder(dtype=tf.float32, shape=matrix.shape, name="X")
        d, u, v = tf.svd(X, name="SVD") 
        truncated_d = d[:embedding_size]
        truncated_u = u[:, :embedding_size]

    return g, X, truncated_d, truncated_u, v

def main():
    logdir = "./model"
    word2id, id2word, matrix = text2matrix("text8")
    g, X, d, u, v = svd_model(matrix)

    with tf.Session(graph=g) as sess:
        # Summary writer
        writer = tf.summary.FileWriter(logdir, graph=sess.graph)

        # Projector
        config = projector.ProjectorConfig()
        embedding_config = config.embeddings.add()
        embedding_config.tensor_name = u.name
        embedding_config.metadata_path = os.path.join(logdir, 'labels_256.tsv')
        projector.visualize_embeddings(writer, config)

        # Eval
        sess.run(tf.global_variables_initializer())
        d, u, v = sess.run([d, u, v], feed_dict={X: matrix})
        singular_values = tf.Variable(d, "singular")
        embedding = tf.Variable(u, "embedding")
        sess.run([singular_values.initializer, embedding.initializer])

        # Saver, Save model
        saver = tf.train.Saver([singular_values, embedding])
        saver.save(sess, logdir)
        save_vocab(id2word, os.path.join(logdir, 'labels_256.tsv'))

        #np.save("d", _d)
        #np.save("u", _u)
        #np.save("v", _v)

if __name__ == '__main__':
    main()

