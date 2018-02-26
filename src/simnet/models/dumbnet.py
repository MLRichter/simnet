from simnet.model import AbstractModel
from simnet.layers import *


class Dumbnet(AbstractModel):
    def __init__(self, num_classes=10):
        super().__init__()
        self._num_classes = num_classes
        self._init_network()

    def _get_train_step(self):
        return self._train_step

    def _get_loss(self):
        return self._loss

    def _build_feed_dict(self, samples, labels):
        return {self.X1: samples, self.Y: labels}

    def _init_network(self):
        # Define the Network Graph
        with tf.variable_scope('Input'):
            self.X1 = tf.placeholder(tf.float32, shape=[None, 28, 28, 1], name='Img1')
            self.Y = tf.placeholder(tf.float32, shape=[None, self._num_classes], name='Labels')

        with tf.variable_scope('Siamese_Segment') as scope:
            left_net = self._get_siamnese(self.X1)

        with tf.variable_scope('Dense_Segment'):
            concatenated = left_net
            d1 = feed_forward_layer(concatenated, 1024, activation_function=tf.nn.relu)

        # logits have their own scope to avoid variable dublication
        with tf.variable_scope('Logit'):
            logits = feed_forward_layer(d1, self._num_classes)

        with tf.variable_scope('Loss_Metrics_and_Training'):
            # use weighted loss, since images of two different classes statically outnumber the matches by 10:1
            self._loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.Y, logits=logits))

            # calculate the accuracy
            self._predicted_class = tf.argmax(logits, 1)
            ground_truth = tf.argmax(self.Y, 1)
            correct = tf.equal(self._predicted_class, ground_truth)
            self._accuracy = tf.reduce_mean(tf.cast(correct, 'float'))

            true_positive = tf.reduce_sum(tf.cast(tf.equal(tf.equal(self._predicted_class, 1), tf.equal(ground_truth, 1)), 'float'))
            false_positive = tf.reduce_sum(tf.cast(tf.not_equal(tf.equal(self._predicted_class, 1), tf.equal(ground_truth, 1)), 'float'))

            self._precision = true_positive / (true_positive + false_positive + 1e-6)

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

    def _get_metrics(self):
        return {'accuracy': self._accuracy, 'precision': self._precision}
