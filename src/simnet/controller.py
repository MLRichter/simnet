from src.simnet.helper import MNIST, EMNIST
from src.simnet.models.dumbnet import Dumbnet

from src.simnet.generator import SimnetGenerator, SimpleGenerator
from src.simnet.models.simnet import Simnet
from sklearn.utils import compute_class_weight
import tensorflow as tf
import numpy as np

"""
Control the experiments
"""

class SimnetMNISTController:

    def run(self):
        # Define generators
        data = MNIST('../../data/mnist/')
        train = SimnetGenerator(data.get_training_batch, data.get_sizes()[0])
        val = SimnetGenerator(data.get_validation_batch, data.get_sizes()[1])

        #class weights
        class_weights = compute_class_weight('balanced', np.unique(data._test_labels),
                                             data._test_labels)

        with tf.Session() as sess:
            # Define models
            simnet = Simnet()

            # Do fitting
            train_history, val_history = simnet.fit(sess, train, val, 100, 256, 100)

            acc, avg_acc, weighted_acc = simnet.evaluate_special(sess, data.get_test_batch, 256, data.get_classification_samples, data.get_sizes()[2],class_weights=class_weights, emnist=False)

            print("Test ACC: ", acc, " TEST AVG ACC: ", avg_acc)

class SimnetEMNISTController:
    def run(self):

        # Define generators
        data = EMNIST('../data/emnist/')
        train = SimnetGenerator(data.get_training_batch, data.get_sizes()[0])
        val = SimnetGenerator(data.get_validation_batch, data.get_sizes()[1])

        #class weights
        class_weights = compute_class_weight('balanced', np.unique(data._test_labels),
                                             data._test_labels)

        with tf.Session() as sess:

            # Define models
            simnet = Simnet()

            # Do fitting
            train_history, val_history = simnet.fit(sess, train, val, 100, 1024, 100)

            acc, avg_acc, weighted_acc = simnet.evaluate_special(sess, data.get_test_batch, 1024, data.get_classification_samples, data.get_sizes()[2],class_weights=class_weights)

            print("Test ACC: ", acc, " TEST AVG ACC: ", avg_acc, "Weightes AVG ACC:", weighted_acc)

class DumbnetMNISTController:
    def run(self):
        # Define generators
        data = MNIST('../../data/mnist/')
        train = SimpleGenerator(data.get_training_batch, data.get_sizes()[0])
        val = SimpleGenerator(data.get_validation_batch, data.get_sizes()[1])
        test = SimpleGenerator(data.get_test_batch, data.get_sizes()[2])

        with tf.Session() as sess:
            # Define models
            dumbnet = Dumbnet()

            # Do fitting
            train_history, val_history = dumbnet.fit(sess, train, val, 10, 256, 100)

            test_history = dumbnet.evaluate(sess, test, 256)

            print("TEST ACC: ", np.mean(test_history['accuracy'][-1]),"TEST PRC:", np.mean(test_history['precision']))
            print("TEST WACC: ", np.mean(test_history['weighted_acc'][-1]),"TEST PRC:", np.mean(test_history['precision']))

            #print("TRAIN ACC: ", np.mean(train_history['accuracy'][0]),"TRAIN PRC:", np.mean(train_history['precision']))

class DumbnetEMNISTController:
    def run(self):
        # Define generators
        data = EMNIST('../../data/emnist/')
        train = SimpleGenerator(data.get_training_batch, data.get_sizes()[0], 62)
        val = SimpleGenerator(data.get_validation_batch, data.get_sizes()[1], 62)
        test = SimpleGenerator(data.get_test_batch, data.get_sizes()[2],62)

        with tf.Session() as sess:
            # Define models
            dumbnet = Dumbnet(num_classes=62)

            # Do fitting
            train_history, val_history = dumbnet.fit(sess, train, val, 10, 256, 100)

            test_history = dumbnet.evaluate(sess, test, 256)

            print("TEST ACC: ", np.mean(test_history['accuracy'][-1]),"TEST PRC:", np.mean(test_history['precision']))
            print("TEST WACC: ", np.mean(test_history['weighted_acc'][-1]),"TEST PRC:", np.mean(test_history['precision']))

if __name__ == '__main__':
    controller = SimnetEMNISTController()
    controller.run()
