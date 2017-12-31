import tensorflow as tf
import os
import re
import numpy as np
from six import iteritems, iterkeys
import pickle
from tensorflow.contrib.tensorboard.plugins import projector

def pickle_it(d, filename):
    with open(filename, "wb") as f:
        pickle.dump(d, f)

def text2matrix(filename, window_size=4):
    # Read in preprocessed text
    space_regex = re.compile(r'\s+')
    with open(filename, "r") as f:
        words = space_regex.split(f.read().strip())

    # Counts
    print("Bulding count dictionary", flush=True)
    counts = {}
    for word in words:
        c = counts.get(word, 0) + 1
        counts[word] = c

    # Trim by frequency
    print("Trimming by frequency", flush=True)
    keep_size = 9999
    counts = dict(sorted(iteritems(counts), key=lambda x: x[1])[-keep_size:])

    # Build vocab
    print("Building vocab", flush=True)
    word2id = {}
    id2word = {}
    i = 0
    for word in iterkeys(counts):
        word2id[word] = i
        id2word[i] = word
        i += 1
    word2id['<UNK>'] = i
    id2word[i] = '<UNK>'

    # Build co-occurence matrix
    print("Building cooccurrence matrix", flush=True)
    matrix = np.zeros((len(word2id), len(word2id)))
    for index, word in enumerate(words):
        context = words[max(0, index - window_size) : index] + words[index + 1 : min(index + 1 + window_size, len(words))]

        if word not in word2id:
            word = '<UNK>'

        for context_word in context:
            if context_word not in word2id:
                context_word = '<UNK>'

            matrix[word2id[word], word2id[context_word]] += 1.

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

        # Eval
        sess.run(tf.global_variables_initializer())
        d, u, v = sess.run([d, u, v], feed_dict={X: matrix})
        singular_values = tf.Variable(d, "singular")
        embedding = tf.Variable(u, "embedding")
        sess.run([singular_values.initializer, embedding.initializer])

        # Projector
        config = projector.ProjectorConfig()
        embedding_config = config.embeddings.add()
        embedding_config.tensor_name = embedding.name
        embedding_config.metadata_path = 'labels_256.tsv'
        projector.visualize_embeddings(writer, config)

        # Saver, Save model
        saver = tf.train.Saver([singular_values, embedding])
        saver.save(sess, os.path.join(logdir, 'model.ckpt'))
        save_vocab(id2word, os.path.join(logdir, 'labels_256.tsv'))

        print("Finished!", flush=True)
        #np.save("d", _d)
        #np.save("u", _u)
        #np.save("v", _v)

if __name__ == '__main__':
    main()

