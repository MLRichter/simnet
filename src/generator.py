import numpy as np


class SimnetGenerator():
    def __init__(self, helper):
        self._helper = helper

    def get_batch(self, batch_size):
        half_batch = batch_size // 2
        for samples, labels in self._helper(batch_size):
            samples = samples.reshape((samples.shape[0], 28, 28, 1))
            x1 = samples[:half_batch]
            x2 = samples[half_batch:]

            y1 = labels[:half_batch]
            y2 = labels[half_batch:]

            y = np.zeros((half_batch, 1))

            # calculate binary labels for the siamnese network
            y[y1 == y2] = 0.0
            y[y1 != y2] = 1.0

            yield [x1, x2], y


class SimpleGenerator():

    def __init__(self, helper):
        self._helper = helper

    def get_batch(self, batch_size):
        for samples, labels in self._helper(batch_size):
            samples = samples.reshape((samples.shape[0], 28, 28, 1))
            yield samples, labels
