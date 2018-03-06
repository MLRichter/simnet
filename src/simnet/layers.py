import tensorflow as tf

"""
Some helper functions to build networks
"""

def feed_forward_layer(x, target_size, normalize=False, activation_function=None):
    """
    Simple fully-connected network.
    :param x: input
    :param target_size: number of neurons
    :param normalize: should batch norm be used
    :param activation_function: used activation function
    :return: the tensor for fully-connected layer
    """
    print("Forward-Layer:" + str(x.shape))

    fan_in = int(x.shape[-1])

    if activation_function == tf.nn.relu:
        var_init = tf.random_normal_initializer(stddev=2 / fan_in)
    else:
        var_init = tf.random_normal_initializer(stddev=fan_in ** (-1 / 2))
    weights = tf.get_variable("weights", [x.shape[1], target_size], tf.float32, var_init)

    var_init = tf.constant_initializer(0.0)
    biases = tf.get_variable("biases", [target_size], tf.float32, var_init)

    activation = tf.matmul(x, weights) + biases

    if normalize:
        activation = batch_norm(activation, [0])

    return activation_function(activation) if callable(activation_function) else activation


def conv_layer(x, kernel_quantity, kernel_size, stride_size, normalize=False, activation_function=False):
    """
    Builds a convolutional layer.
    :param x: input
    :param kernel_quantity: number of filters
    :param kernel_size: kernel size
    :param stride_size: stride size
    :param normalize: should batch norm be applied
    :param activation_function: the activation function, that should be used. if false, no acivation function would be used
    :return: a conv layer
    """
    print("Conv-Layer:" + str(x.shape))
    depth = x.shape[-1]
    fan_in = int(x.shape[1] * x.shape[2])

    if activation_function == tf.nn.relu or activation_function == tf.nn.leaky_relu:
        var_init = tf.random_normal_initializer(stddev=2 / fan_in)
    else:
        var_init = tf.random_normal_initializer(stddev=fan_in ** (-1 / 2))
    kernels = tf.get_variable("kernels", [kernel_size, kernel_size, depth, kernel_quantity], tf.float32, var_init)

    var_init = tf.constant_initializer(0.0)
    biases = tf.get_variable("biases", [kernel_quantity], initializer=var_init)

    activation = tf.nn.conv2d(x, kernels, strides=[1, stride_size, stride_size, 1], padding="SAME") + biases

    if normalize:
        activation = batch_norm(activation, [0, 1, 2])

    return activation_function(activation) if callable(activation_function) else activation

def batch_norm(x, axes):
    """
    Batch norm builder.
    :param x: input
    :param axes:
    :return:
    """
    depth = x.shape[-1]
    mean, var = tf.nn.moments(x, axes=axes)

    var_init = tf.constant_initializer(0.0)
    offset = tf.get_variable("offset", [depth], tf.float32, var_init)
    var_init = tf.constant_initializer(1.0)
    scale = tf.get_variable("scale", [depth], tf.float32, var_init)

    pop_mean = tf.get_variable("pop_mean", [depth], initializer=tf.zeros_initializer(), trainable=False)
    pop_var = tf.get_variable("pop_var", [depth], initializer=tf.ones_initializer(), trainable=False)

    return tf.cond(
        is_training,
        lambda: _batch_norm(x, pop_mean, pop_var, mean, var, offset, scale),
        lambda: _pop_batch_norm(x, pop_mean, pop_var, offset, scale)
    )


def _pop_batch_norm(x, pop_mean, pop_var, offset, scale):
    return tf.nn.batch_normalization(x, pop_mean, pop_var, offset, scale, 1e-6)


def _batch_norm(x, pop_mean, pop_var, mean, var, offset, scale):
    decay = 0.99

    dependency_1 = tf.assign(pop_mean, pop_mean * decay + mean * (1 - decay))
    dependency_2 = tf.assign(pop_var, pop_var * decay + var * (1 - decay))

    with tf.control_dependencies([dependency_1, dependency_2]):
        return tf.nn.batch_normalization(x, mean, var, offset, scale, 1e-6)



