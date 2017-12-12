import tensorflow as tf

def mnist_linear():
    with tf.name_scope("linear") as scope:
        X = tf.placeholder(dtype=tf.float32, shape=(None, 784), name="X")
        y = tf.placeholder(dtype=tf.int32, shape=(None, 10), name="Y")

        W1 = tf.Variable(tf.random_normal((784, 10)), name="W1")
        b1 = tf.Variable(tf.zeros((1, 10)), name="b1")

        yhat = tf.matmul(X, W1) + b1

        return X, y, yhat

def mnist_nn():
    with tf.name_scope("nn") as scope:
        X = tf.placeholder(dtype=tf.float32, shape=(None, 784), name="X")
        y = tf.placeholder(dtype=tf.int32, shape=(None, 10), name="Y")

        W1 = tf.Variable(tf.random_normal((784, 256)), name="W1")
        b1 = tf.Variable(tf.zeros((1, 256)), name="b1")

        W2 = tf.Variable(tf.random_normal((256,10)), name="W2")
        b2 = tf.Variable(tf.zeros((1, 10)), name="b2")

        Z1 = tf.matmul(X, W1) + b1
        A1 = tf.nn.relu(Z1)
        logits = tf.matmul(A1, W2) + b2

        return X, y, logits

def mnist_conv_net():
    with tf.name_scope("conv_net") as scope:
        X = tf.placeholder(dtype=tf.float32, shape=(None, 784), name="X")
        X_img = tf.reshape(tensor=X, shape=(-1, 28, 28, 1), name="X_img")
        y = tf.placeholder(dtype=tf.float32, shape=(None, 10), name="y")
        dropout_prob = tf.placeholder(dtype=tf.float32, name="dropout_prob")

        W_conv1 = tf.Variable(tf.random_normal((5, 5, 1, 32)), name="W_conv1")
        b_conv1 = tf.Variable(tf.zeros((1, 32)), name="b_conv1")

        W_conv2 = tf.Variable(tf.random_normal((5, 5, 32, 64)), name="W_conv2")
        b_conv2 = tf.Variable(tf.zeros((1, 64)), name="b_conv2")

        W_fc1 = tf.Variable(tf.random_normal((3136, 1024)), name="W_fc1")
        b_fc1 = tf.Variable(tf.zeros((1, 1024)), name="b_fc1")

        W_fc2 = tf.Variable(tf.random_normal((1024, 10)), name="W_fc2")
        b_fc2 = tf.Variable(tf.zeros((1, 10)), name="b_fc2")

        conv1 = tf.nn.conv2d(input=X_img, filter=W_conv1, strides=(1,1,1,1), padding="SAME") + b_conv1
        relu1 = tf.nn.relu(conv1)
        pool1 = tf.nn.max_pool(value=relu1, ksize=(1,2,2,1), strides=(1,2,2,1), padding="SAME")

        conv2 = tf.nn.conv2d(input=pool1, filter=W_conv2, strides=(1,1,1,1), padding="SAME") + b_conv2
        relu2 = tf.nn.relu(conv2)
        pool2 = tf.nn.max_pool(value=relu2, ksize=(1,2,2,1), strides=(1,2,2,1), padding="SAME")

        pool2_flat = tf.reshape(pool2, shape=(-1, 3136))
        fc1 = tf.matmul(pool2_flat, W_fc1) + b_fc1
        relu_fc1 = tf.nn.relu(fc1)
        dropout_fc1 = tf.nn.dropout(relu_fc1, dropout_prob)

        logits = tf.matmul(dropout_fc1, W_fc2) + b_fc2

        return X, y, dropout_prob, logits

