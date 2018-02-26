from simnet.helper import MNIST, EMNIST
from simnet.models.dumbnet import Dumbnet

from simnet.generator import SimnetGenerator, SimpleGenerator
from simnet.models.simnet import Simnet


class SimnetMNISTController:

    def run(self):
        # Define generators
        data = MNIST('../data/mnist/')
        train = SimnetGenerator(data.get_training_batch, data.get_sizes()[0])
        val = SimnetGenerator(data.get_validation_batch, data.get_sizes()[1])

        # Define models
        simnet = Simnet()

        # Do fitting
        train_history, val_history = simnet.fit(train, val, 30, 32, 100)

class SimnetEMNISTController:
    def run(self):
        # Define generators
        data = EMNIST('../data/emnist/')
        train = SimnetGenerator(data.get_training_batch, data.get_sizes()[0])
        val = SimnetGenerator(data.get_validation_batch, data.get_sizes()[1])

        # Define models
        simnet = Simnet()

        # Do fitting
        train_history, val_history = simnet.fit(train, val, 30, 32, 100)

class DumbnetMNISTController:
    def run(self):
        # Define generators
        data = MNIST('../data/mnist/')
        train = SimpleGenerator(data.get_training_batch, data.get_sizes()[0])
        val = SimpleGenerator(data.get_validation_batch, data.get_sizes()[1])

        # Define models
        dumbnet = Dumbnet()

        # Do fitting
        train_history, val_history = dumbnet.fit(train, val, 10, 32, 100)

class DumbnetEMNISTController:
    def run(self):
        # Define generators
        data = EMNIST('../data/emnist/')
        train = SimpleGenerator(data.get_training_batch, data.get_sizes()[0], 62)
        val = SimpleGenerator(data.get_validation_batch, data.get_sizes()[1], 62)

        # Define models
        dumbnet = Dumbnet(num_classes=62)

        # Do fitting
        train_history, val_history = dumbnet.fit(train, val, 10, 32, 100)

if __name__ == '__main__':
    controller = DumbnetEMNISTController()
    controller.run()
