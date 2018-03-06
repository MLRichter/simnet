from src.simnet.model import AbstractModel
from src.simnet.layers import *
from src.simnet.util import get_class_weights
from sklearn.metrics.classification import accuracy_score
import numpy as np
import time

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
            self.sigmoidal_out = tf.nn.sigmoid(logits)
            correct = tf.equal(self._predicted_class, tf.equal(self.Y, 1.0))

            true_positive = tf.reduce_sum(tf.cast(tf.equal(self._predicted_class, tf.equal(self.Y, 1.0)), 'float'))
            false_positive = tf.reduce_sum(tf.cast(tf.not_equal(self._predicted_class, tf.equal(self.Y, 1.0)), 'float'))

            self._accuracy = tf.reduce_mean(tf.cast(correct, 'float'))

            self._precision = true_positive / (true_positive + false_positive + 1e-6)

            # use GradientDecent to train, interestingly ADAM results in a collapsing model. Standard SGD performed reliably better
            self._train_step = tf.train.GradientDescentOptimizer(0.01).minimize(self._loss)

        with tf.variable_scope("summaries"):
            tf.summary.scalar('loss', self._loss)
            tf.summary.scalar('acc', self._accuracy)
            self._merged = tf.summary.merge_all()

    def evaluate_special(self, session: tf.Session, val_generator, batch_size: int, classification_samples, size, emnist=True):
        test_acc = []
        samples_per_shot = 100
        total_data_processed = 0.0
        correct = 0.0
        correct_avg = 0.0

        # stuff for the weighted accuracy
        predictions = []
        sample_weights = []
        ground_truth = []
        class_weights = get_class_weights()

        for data, labels in val_generator(batch_size):
            data = data.reshape((data.shape[0], 28, 28, 1))
            print('[INFO] processing', total_data_processed, 'of', size)

            # calssify a single sample
            for i in range(len(data)):
                if emnist:
                    x1, y1 = classification_samples(samples_per_shot // 62)
                else:
                    x1, y1 = classification_samples(samples_per_shot // 10)
                x2 = np.asarray([list(data[i])] * len(y1))

                pc = session.run([self.sigmoidal_out], feed_dict={self.X1: x1, self.X2: x2})
                prediction = y1[np.argmin(pc)]
                prediction_avg = self._get_mean_prediction(np.squeeze(pc), y1)

                if prediction == labels[i]:
                    correct += 1.0
                if prediction_avg == labels[i]:
                    correct_avg += 1.0
                predictions.append(prediction_avg)
                sample_weights.append(class_weights)
                ground_truth.append(labels[i])

                total_data_processed += 1.0
                # keep track of loss and accuracy
        accuracy = correct / total_data_processed
        avg_acc = correct_avg / total_data_processed
        try:
            weighted_acc = accuracy_score(ground_truth,predictions,True,sample_weights)
        except:
            weighted_acc = None
        return accuracy, avg_acc, weighted_acc

    def _get_metrics(self):
        return {'accuracy': self._accuracy, 'precision': self._precision}


    def _get_mean_prediction(self, predictions, y):
        score_map = []
        for i in range(10):
            c_pred = predictions[y == i]
            m = np.mean(c_pred)
            score_map.append(m)
        return np.argmin(score_map)

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

    def _get_summary(self):
        return self._merged


