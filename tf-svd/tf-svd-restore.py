import numpy as np
import tensorflow as tf
import pickle

def pickle_load(filename):
    with open(filename, "rb") as f:
        return pickle.load(f)

def cos_similarity(v, m):
    return m.dot(v.T) / (np.sqrt(np.sum(v**2)) * np.sqrt(np.sum(m**2, axis=1)))

def main():
    id2word = pickle_load("id2word.p")
    word2id = pickle_load("word2id.p")
    d = np.load("d.bin.npy")[:256]
    u = np.load("u.bin.npy")[:, :256]
    v = np.load("v.bin.npy")[:, :256]

    print(d.shape)
    print(u.shape)
    print(v.shape)


if __name__ == '__main__':
    main()
