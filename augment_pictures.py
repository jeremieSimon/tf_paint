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

def build_model(name,
                input_width,
                output_width,
                n_neurons=30,
                n_layers=10,
                activation_fn=tf.nn.relu,
                final_activation_fn=tf.nn.tanh,
                cost_type='l2_norm'):

    X = tf.placeholder(name='X', shape=[None, input_width],
                       dtype=tf.float32)
    Y = tf.placeholder(name='Y', shape=[None, output_width],
                       dtype=tf.float32)

    current_input = X
    for layer_i in range(n_layers):
        current_input = linear(
            current_input, n_neurons,
            activation=activation_fn,
            name='{}_layer{}'.format(name, layer_i))[0]

    Y_pred = linear(
        current_input, output_width,
        activation=final_activation_fn,
        name='{}_pred'.format(name))[0]

    if cost_type == 'l1_norm':
        cost = tf.reduce_mean(tf.reduce_sum(
                tf.abs(Y - Y_pred), 1))
    elif cost_type == 'l2_norm':
        cost = tf.reduce_mean(tf.reduce_sum(
                tf.squared_difference(Y, Y_pred), 1))
    else:
        raise ValueError(
            'Unknown cost_type: {}.  '.format(
            cost_type) + 'Use only "l1_norm" or "l2_norm"')

    return {'X': X, 'Y': Y, 'Y_pred': Y_pred, 'cost': cost}

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

def image_as_feature_xy(img):
    xs = []
    ys = []

    m, n = img.shape[:-1] # remove channel dim
    for i in range(m):
        for j in range(n):
            xs.append([i, j]) # coords of the image
            ys.append(img[i, j, :]) # value for the given coords

    xs = np.array(xs)
    ys = np.array(ys)

    return xs, ys

def train_model(img,
          model,
          sess,
          learning_rate=0.0001,
          batch_size=200,
          n_iterations=10):

    def img_as_xys(img):
        xs, ys = image_as_feature_xy(img)
        xs = np.array(xs)
        xs = (xs - np.mean(xs, 0)) / np.std(xs, 0)
        ys = np.array(ys)
        ys = ys / 127.5 - 1
        return (xs, ys)

    xs, ys = img_as_xys(img1)

    optimizer = tf.train.AdamOptimizer(
        learning_rate=learning_rate).minimize(model['cost'])

    sess.run(tf.global_variables_initializer())
    gifs = []
    costs = []
    step_i = 0
    for it_i in range(n_iterations):
        # Get a random sampling of the dataset
        idxs = np.random.permutation(range(len(xs)))

        # The number of batches we have to iterate over
        n_batches = len(idxs) // batch_size
        training_cost = 0

        # Now iterate over our stochastic minibatches:
        for batch_i in range(n_batches):

            # Get just minibatch amount of data
            idxs_i = idxs[batch_i * batch_size:
                          (batch_i + 1) * batch_size]

            # And optimize, also returning the cost so we can monitor
            # how our optimization is doing.
            cost = sess.run(
                [model['cost'], optimizer],
                feed_dict={model['X']: xs[idxs_i],
                           model['Y']: ys[idxs_i]})[0]
            training_cost += cost
        gifs.append(model['Y_pred'].eval(feed_dict={model['X']: xs}, session=sess))
        print('iteration {}/{}: cost {}'.format(
                it_i + 1, n_iterations, training_cost / n_batches))

    xs, ys = img_as_xys(img2)
    for it_i in range(n_iterations):
        # Get a random sampling of the dataset
        idxs = np.random.permutation(range(len(xs)))

        # The number of batches we have to iterate over
        n_batches = len(idxs) // batch_size
        training_cost = 0

        # Now iterate over our stochastic minibatches:
        for batch_i in range(n_batches):

            # Get just minibatch amount of data
            idxs_i = idxs[batch_i * batch_size:
                          (batch_i + 1) * batch_size]

            # And optimize, also returning the cost so we can monitor
            # how our optimization is doing.
            cost = sess.run(
                [model['cost'], optimizer],
                feed_dict={model['X']: xs[idxs_i],
                           model['Y']: ys[idxs_i]})[0]
            training_cost += cost
        gifs.append(model['Y_pred'].eval(feed_dict={model['X']: xs}, session=sess))
        print('iteration {}/{}: cost {}'.format(
                it_i + 1, n_iterations, training_cost / n_batches))
    return gifs

img = imresize(plt.imread('/Users/jeremiesimon/Desktop/top100/IMG_0135.jpg'), (200, 200))
xs, ys = image_as_feature_xy(img)
xs_ = xs
xs = np.array(xs)
xs = (xs - np.mean(xs, 0)) / np.std(xs, 0) # normalize

sess = tf.Session()

mountain_model = build_model("mountain", 2, 3)
(trained_model, tings) = train_model(ting, mountain_model, sess,  n_iterations=30)

y_ = trained_model['Y_pred'].eval(feed_dict={model['X']: xs + coeff}, session=sess)
plt.imshow(y_.reshape(200, 200, 3) * 127.5 + 127.5, cmap='gray')
plt.show()
#*127.5 + 127.5
