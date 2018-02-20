
import logging

log = logging.getLogger(__name__)

class Callback:
    def __init__(self):
        self.model = None

    def train_start(self, logs):
        pass

    def train_end(self, logs):
        pass

    def epoch_start(self, logs):
        pass

    def epoch_end(self, logs):
        pass

    def batch_start(self, logs):
        pass

    def batch_end(self, logs):
        pass

    def validation_start(self, logs):
        pass

    def validation_end(self, logs):
        pass

    def set_model(self, model):
        self.model = model


class Monitor(Callback):

    def __init__(self):
        super().__init__()
        self.train_history = {}
        self.validation_history = {}

    def epoch_start(self, logs):
        print("Start epoch {}.".format(logs['epoch']))

    def epoch_end(self, logs):
        print("Finish epoch {}.".format(logs['epoch']))

    def batch_end(self, logs):
        if 'loss' in logs:
            self._append_to_train_history('loss', logs['loss'])

    def validation_end(self, logs):
        if 'loss' in logs:
            self._append_to_train_history('loss', logs['loss'])

    def _append_to_train_history(self, key, value):
        self._append(self.train_history, key, value)

    def _append_to_validation_history(self, key, value):
        self._append(self.validation_history, key, value)

    def _append(self, history, key, value):
        if key not in history:
            history[key] = []

        history[key].append(value)
