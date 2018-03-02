from simnet.helper import MNIST, EMNIST
from simnet.models.dumbnet import Dumbnet

from simnet.generator import SimnetGenerator, SimpleGenerator
from simnet.models.simnet import Simnet
import tensorflow as tf
import numpy as np


class SimnetMNISTController:

    def run(self):
        # Define generators
        data = MNIST('../data/mnist/')
        train = SimnetGenerator(data.get_training_batch, data.get_sizes()[0])
        val = SimnetGenerator(data.get_validation_batch, data.get_sizes()[1])
        test = SimnetGenerator(data.get_test_batch, data.get_sizes()[2])

        with tf.Session() as sess:
            # Define models
            simnet = Simnet()

            # Do fitting
            train_history, val_history = simnet.fit(sess, train, val, 30, 32, 100)

            acc, avg_acc = simnet.evaluate_special(sess, data.get_test_batch, 128, data.get_classification_samples, data.get_sizes()[2])

            print("Test ACC: ", acc, " TEST AVG ACC: ", avg_acc)

class SimnetEMNISTController:
    def run(self):
        # Define generators
        data = EMNIST('../data/emnist/')
        train = SimnetGenerator(data.get_training_batch, data.get_sizes()[0])
        val = SimnetGenerator(data.get_validation_batch, data.get_sizes()[1])
        test = SimnetGenerator(data.get_test_batch, data.get_sizes()[2])

        with tf.Session() as sess:

            # Define models
            simnet = Simnet()

            # Do fitting
            train_history, val_history = simnet.fit(sess, train, val, 30, 32, 100)

            acc, avg_acc = simnet.evaluate_special(sess, data.get_test_batch, 128, data.get_classification_samples, data.get_sizes()[2])

            print("Test ACC: ", acc, " TEST AVG ACC: ", avg_acc)

class DumbnetMNISTController:
    def run(self):
        # Define generators
        data = MNIST('../data/mnist/')
        train = SimpleGenerator(data.get_training_batch, data.get_sizes()[0])
        val = SimpleGenerator(data.get_validation_batch, data.get_sizes()[1])
        test = SimpleGenerator(data.get_test_batch, data.get_sizes()[2])

        with tf.Session() as sess:
            # Define models
            dumbnet = Dumbnet()

            # Do fitting
            train_history, val_history = dumbnet.fit(sess, train, val, 10, 32, 100)

            test_history = dumbnet.evaluate(sess, test, 128)

            print("TEST ACC: ", np.mean(test_history['accuracy'][0]))

class DumbnetEMNISTController:
    def run(self):
        # Define generators
        data = EMNIST('../data/emnist/')
        train = SimpleGenerator(data.get_training_batch, data.get_sizes()[0], 62)
        val = SimpleGenerator(data.get_validation_batch, data.get_sizes()[1], 62)
        test = SimnetGenerator(data.get_test_batch, data.get_sizes()[2])

        with tf.Session() as sess:
            # Define models
            dumbnet = Dumbnet(num_classes=62)

            # Do fitting
            train_history, val_history = dumbnet.fit(sess, train, val, 10, 32, 100)

            test_history = dumbnet.evaluate(sess, test, 128)

            print("TEST ACC: ", np.mean(test_history['accuracy'][0]))

if __name__ == '__main__':
    controller = DumbnetMNISTController()
    controller.run()
