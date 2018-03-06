import logging
from abc import ABC, abstractmethod

import tensorflow as tf

from src.simnet.callbacks import Monitor
from src.simnet.util import Progbar
from src.simnet.util import get_class_weights
from sklearn.metrics.classification import accuracy_score

logging.basicConfig()
log = logging.getLogger(__name__)
import numpy as np
import time

class Model(ABC):
    @abstractmethod
    def fit(self, session: tf.Session, train_generator, val_generator, epochs: int, batch_size: int, validation_step: int, callbacks=[]):
        pass

    @abstractmethod
    def evaluate(self, session: tf.Session, val_generator, batch_size):
        pass


class AbstractModel(Model):

    def __init__(self):
        self.timestamp = str(time.time())
        self.train_writer = tf.summary.FileWriter('logs/{}/train'.format(self.timestamp))
        self.val_writer = tf.summary.FileWriter('logs/{}/val'.format(self.timestamp))
        self._global_step = 0

    def fit(self, session: tf.Session, train_generator, val_generator, epochs: int, batch_size: int, validation_step: int, callbacks=[]):
        monitor = Monitor()
        callbacks.append(monitor)

        self._do_fit(session, train_generator, val_generator, epochs, batch_size, validation_step, callbacks)

        return monitor.train_history, monitor.validation_history

    def evaluate(self, session: tf.Session, val_generator, batch_size: int):
        monitor = Monitor()
        callbacks = []
        callbacks.append(monitor)

        self._do_evaluation(session, val_generator, batch_size, 1, callbacks, 0)

        return monitor.validation_history

    def _do_fit(self, sess: tf.Session, train_generator, val_generator, epochs: int, validation_step: int, batch_size: int, callbacks):
        self._init_session(sess)
        for callback in callbacks:
            callback.train_start({})

        for epoch in range(epochs):
            self._do_epoch(sess, epoch, train_generator, val_generator, validation_step, batch_size, callbacks)

        for callback in callbacks:
            callback.train_end({})

    def _do_epoch(self, sess: tf.Session, epoch: int, train_generator, val_generator, validation_step: int,
                  batch_size: int, callbacks):
        for callback in callbacks:
            callback.epoch_start({'epoch': epoch})

        step = 0

        sample_weight_dict = get_class_weights()
        stateful_metrics = ['val_loss']
        stateful_metrics.extend([ 'val_' + key for key in self._get_metrics().keys()])
        progbar = Progbar(train_generator.steps(batch_size), stateful_metrics=stateful_metrics)
        for data, labels in train_generator.get_batch(batch_size):
            for callback in callbacks:
                callback.batch_start({})

            targets = [self._get_train_step(), self._get_loss()]
            targets.extend(self._get_metrics().values())
            if self._get_summary() is not None:
                targets.append(self._get_summary())

            # run training step
            _results = sess.run(targets, feed_dict=self._build_feed_dict(data, labels))

            class_weights = get_class_weights()

            if self._get_summary() is not None:
                self.train_writer.add_summary(_results[-1], self._global_step)

            _loss = _results[1]

            metric_values = {'loss': _loss}

            for i, metric_name in enumerate(self._get_metrics().keys()):
                metric_values[metric_name] = _results[i + 2]

            progbar.update(step + 1, [(key, value) for key, value in metric_values.items()])

            if step % validation_step == 0:
                self._do_evaluation(sess, val_generator, batch_size, validation_step, callbacks, step, progbar)

            step += 1
            self._global_step += 1

            for callback in callbacks:
                callback.batch_end(metric_values)

        for callback in callbacks:
            callback.epoch_end({'epoch': epoch})

    def _get_train_step(self):
        pass

    def _get_loss(self):
        pass

    def _get_summary(self):
        return None

    def _build_feed_dict(self, samples, labels):
        pass

    def _get_metrics(self):
        return {}

    def _init_session(self, session: tf.Session):
        session.run(tf.global_variables_initializer())

    def _do_evaluation(self, session: tf.Session, val_generator, batch_size, validation_step, callbacks, step, progbar = None):
        for callback in callbacks:
            callback.validation_start({})

        metrics = {}

        # test model on validation data
        for data, labels in val_generator.get_batch(batch_size):

            targets = [self._get_loss()]
            targets.extend(self._get_metrics().values())
            if self._get_summary() is not None:
                targets.append(self._get_summary())

            _results = session.run(targets, feed_dict=self._build_feed_dict(data, labels))

            if self._get_summary() is not None:
                self.val_writer.add_summary(_results[-1], self._global_step)

            _loss = _results[0]

            metric_values = {'loss': _loss}

            for i, metric_name in enumerate(self._get_metrics().keys()):
                metric_values[metric_name] = _results[i + 1]

            for key, value in metric_values.items():
                if key not in metrics:
                    metrics[key] = []

                metrics[key].append(value)

        progbar.update(step + 1, [('val_' + key, np.mean(value)) for key, value in metrics.items()])

        for callback in callbacks:
            callback.validation_end(logs={'metrics': metrics, 'validation_step': validation_step})
