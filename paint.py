import tensorflow as tf
import numpy as np
from skimage.transform import resize
from skimage import data
from scipy.misc import imresize
import os
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import resize
from skimage import data
from scipy.misc import imresize
import gif


def load_image(path):
    img = plt.imread(path)
    img = imresize(img, (200, 200))
    return img

def image_as_feature_xy(img):
    xs = []
    ys = []

    m, n = img.shape[:-1]
    for i in range(m):
        for j in range(n):
            xs.append([i, j])
            ys.append(img[i, j, :])

    xs = np.array(xs)
    ys = np.array(ys)

    return xs, ys

def normalize_X(X):
    return X - np.mean(X) / (np.std(X))

def normalize_Y(Y):
    return Y / 255

def train(xs, ys, cost, optimizer, Y_pred):
    sess.run(tf.global_variables_initializer())

    imgs = []
    costs = []
    gif_step = 10#n_iterations // 10
    step_i = 0

    print ("start training")
    for it_i in range(n_iterations):
        print ("iteration " + str(it_i))

        idxs = np.random.permutation(range(len(xs)))
        # The number of batches we have to iterate over
        n_batches = len(idxs) // batch_size
        # Now iterate over our stochastic minibatches:
        for batch_i in range(n_batches):
            # Get just minibatch amount of data
            idxs_i = idxs[batch_i * batch_size: (batch_i + 1) * batch_size]
            training_cost = sess.run(
                [cost, optimizer],
                feed_dict={X: xs[idxs_i], Y: ys[idxs_i]})[0]

        if (it_i + 1) % gif_step == 0:
            print ("append " + str(len(costs)))
            costs.append(training_cost / n_batches)
            ys_pred = Y_pred.eval(feed_dict={X: xs}, session=sess)
            imgs.append(ys_pred.reshape(img.shape))

    return imgs, costs

def linear(x, n_output, name=None, activation=None, reuse=None):
    """Fully connected layer

    Parameters
    ----------
    x : tf.Tensor
        Input tensor to connect
    n_output : int
        Number of output neurons
    name : None, optional
        Scope to apply

    Returns
    -------
    op : tf.Tensor
        Output of fully connected layer.
    """
    if len(x.get_shape()) != 2:
        x = flatten(x, reuse=reuse)

    n_input = x.get_shape().as_list()[1]

    with tf.variable_scope(name or "fc", reuse=reuse):
        W = tf.get_variable(
            name='W',
            shape=[n_input, n_output],
            dtype=tf.float32,
            initializer=tf.contrib.layers.xavier_initializer())

        b = tf.get_variable(
            name='b',
            shape=[n_output],
            dtype=tf.float32,
            initializer=tf.constant_initializer(0.0))

        h = tf.nn.bias_add(
            name='h',
            value=tf.matmul(x, W),
            bias=b)

        if activation:
            h = activation(h)

        return h, W

if __name__ == "__main__":
    img = load_image("/Users/jeremiesimon/Desktop/17522073_1468704699826464_2119989739_o.jpg")
    print("loaded image")
    xs, ys = image_as_feature_xy(img)
    xs = normalize_X(xs)
    ys = normalize_Y(ys)
    print("normalized x and y")

    tf.reset_default_graph()
    X = tf.placeholder(name="X", shape=(None, 2), dtype=tf.float32)
    Y = tf.placeholder(name='Y', dtype=tf.float32, shape=(None, 3))

    n_neurons = 10

    h1, W1 = linear(x=X, n_output=n_neurons, name='layer1', activation=tf.nn.relu)
    h2, W2 = linear(x=h1, n_output=n_neurons, name='layer2', activation=tf.nn.relu)
    h3, W3 = linear(x=h2, n_output=n_neurons, name='layer3', activation=tf.nn.relu)
    h4, W4 = linear(x=h3, n_output=n_neurons, name='layer4', activation=tf.nn.relu)
    h5, W5 = linear(x=h4, n_output=n_neurons, name='layer5', activation=tf.nn.relu)
    h6, W6 = linear(x=h5, n_output=n_neurons, name='layer6', activation=tf.nn.relu)
    h7, W7 = linear(x=h6, n_output=n_neurons, name='layer7', activation=tf.nn.relu)
    h8, W8 = linear(x=h7, n_output=n_neurons, name='layer8', activation=tf.nn.relu)
    h9, W9 = linear(x=h8, n_output=n_neurons, name='layer9', activation=tf.nn.relu)
    h10, W10 = linear(x=h9, n_output=n_neurons, name='layer10', activation=tf.nn.relu)
    h11, W11 = linear(x=h10, n_output=n_neurons, name='layer11', activation=tf.nn.relu)
    h12, W12 = linear(x=h11, n_output=n_neurons, name='layer12', activation=tf.nn.relu)
    h13, W13 = linear(x=h12, n_output=n_neurons, name='layer13', activation=tf.nn.relu)
    h14, W14 = linear(x=h13, n_output=n_neurons, name='layer14', activation=tf.nn.relu)
    h15, W15 = linear(x=h14, n_output=n_neurons, name='layer15', activation=tf.nn.relu)

    # Now, make one last layer to make sure your network has 3 outputs:
    Y_pred, W15 = linear(h15, 3, activation=None, name='pred')

    l1_error = tf.abs(Y - Y_pred)
    l2_error = tf.pow((Y - Y_pred), 2)
    sum_error = tf.reduce_sum(l1_error, 1)
    cost = tf.reduce_sum(sum_error)

    optimizer = tf.train.AdamOptimizer(0.001).minimize(cost)

    # Create parameters for the number of iterations to run for (< 100)
    n_iterations = 100

    # And how much data is in each minibatch (< 500)
    batch_size = 100

    # Then create a session
    sess = tf.Session()
    imgs, costs = train(xs, ys, cost, optimizer, Y_pred)
    print("training done")
    print(costs)
    _ = gif.build_gif(imgs, saveto='/Users/jeremiesimon/Desktop/lilico.gif', show_gif=False)
    imgs_as_np = np.array(imgs)
    plt.imsave("/Users/jeremiesimon/Desktop/ya.jpg", np.reshape(imgs_as_np, [imgs_as_np.shape[0] * imgs_as_np.shape[1], imgs_as_np.shape[2], imgs_as_np.shape[3]]))
