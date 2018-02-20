from simnet.model import AbstractModel
from simnet.layers import *


class Simnet(AbstractModel):
    def __init__(self):
        super().__init__()
        self._init_network()

    def _get_train_step(self):
        return self._train_step

    def _get_loss(self):
        return self._loss

    def _build_feed_dict(self, samples, labels):
        return {self.X1: samples[0], self.X2: samples[1], self.Y: labels}

    def _init_network(self):
        # Define the Network Graph
        with tf.variable_scope('Input'):
            self.X1 = tf.placeholder(tf.float32, shape=[None, 28, 28, 1], name='Img1')
            self.X2 = tf.placeholder(tf.float32, shape=[None, 28, 28, 1], name='Img2')
            self.Y = tf.placeholder(tf.float32, shape=[None, 1], name='Labels')

        with tf.variable_scope('Siamese_Segment') as scope:
            left_net = self._get_siamnese(self.X1)
            scope.reuse_variables()
            right_net = self._get_siamnese(self.X2)

        with tf.variable_scope('Dense_Segment'):
            concatenated = tf.concat([left_net, right_net], axis=1, name='concatenated_out')
            d1 = feed_forward_layer(concatenated, 1024, activation_function=tf.nn.relu)

        # logits have their own scope to avoid variable dublication
        with tf.variable_scope('Logit'):
            logits = feed_forward_layer(d1, 1)

        with tf.variable_scope('Loss_Metrics_and_Training'):
            # use weighted loss, since images of two different classes statically outnumber the matches by 10:1
            self._loss = tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(self.Y, logits, 0.1, 'Loss'))

            # calculate the accuracy
            self._predicted_class = tf.greater(tf.nn.sigmoid(logits), 0.5)
            sigmoidal_out = tf.nn.sigmoid(logits)
            correct = tf.equal(self._predicted_class, tf.equal(self.Y, 1.0))
            self._accuracy = tf.reduce_mean(tf.cast(correct, 'float'))

            # use GradientDecent to train, interestingly ADAM results in a collapsing model. Standard SGD performed reliably better
            self._train_step = tf.train.GradientDescentOptimizer(0.01).minimize(self._loss)

    def _get_siamnese(self, X):
        """
        creates the siamnese stem for the neural network. This is the part both images are send through sequentially
        :param X: the input tensor
        :return:
        """
        with tf.variable_scope('Layer1'):
            conv1 = conv_layer(X, 16, 5, 5, activation_function=tf.nn.relu)

        with tf.variable_scope('Layer2'):
            conv2 = conv_layer(conv1, 32, 3, 3, activation_function=tf.nn.relu)

        with tf.variable_scope('Layer4'):
            conv3 = conv_layer(conv2, 128, 2, 2, activation_function=tf.nn.relu)

        flat = tf.layers.flatten(conv3)
        return flat
