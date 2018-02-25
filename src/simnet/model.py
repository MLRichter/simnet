import logging
from abc import ABC, abstractmethod

import tensorflow as tf

from simnet.callbacks import Monitor
from simnet.util import Progbar

logging.basicConfig()
log = logging.getLogger(__name__)


class Model(ABC):
    @abstractmethod
    def fit(self, train_generator, val_generator, epochs: int, batch_size: int, validation_step: int, callbacks=[]):
        pass


class AbstractModel(Model):
    def fit(self, train_generator, val_generator, epochs: int, batch_size: int, validation_step: int, callbacks=[]):
        monitor = Monitor()
        callbacks.append(monitor)

        self._do_fit(train_generator, val_generator, epochs, batch_size, validation_step, callbacks)

        return monitor.train_history, monitor.validation_history

    def _do_fit(self, train_generator, val_generator, epochs: int, validation_step: int, batch_size: int, callbacks):
        with tf.Session() as sess:
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

        progbar = Progbar(train_generator.steps(batch_size))
        for data, labels in train_generator.get_batch(batch_size):
            for callback in callbacks:
                callback.batch_start({})

            targets = [self._get_train_step(), self._get_loss()]
            targets.extend(self._get_metrics().values())

            # run training step
            _results = sess.run(targets, feed_dict=self._build_feed_dict(data, labels))

            _loss = _results[1]

            metric_values = {'loss': _loss}

            for i, metric_name in enumerate(self._get_metrics().keys()):
                metric_values[metric_name] = _results[i + 2]

            progbar.update(step + 1, [(key, value) for key, value in metric_values.items()])

            if step % validation_step == 0:
                self._do_evaluation(sess, val_generator, batch_size, validation_step, callbacks)

            step += 1

            for callback in callbacks:
                callback.batch_end(metric_values)

        for callback in callbacks:
            callback.epoch_end({'epoch': epoch})

    def _get_train_step(self):
        pass

    def _get_loss(self):
        pass

    def _build_feed_dict(self, samples, labels):
        pass

    def _get_metrics(self):
        return {}

    def _init_session(self, session: tf.Session):
        session.run(tf.global_variables_initializer())

    def _do_evaluation(self, session: tf.Session, val_generator, batch_size, validation_step, callbacks):
        for callback in callbacks:
            callback.validation_start({})

        metrics = {}

        # test model on validation data
        for data, labels in val_generator.get_batch(batch_size):

            targets = [self._get_loss()]
            targets.extend(self._get_metrics().values())

            _results = session.run(targets, feed_dict=self._build_feed_dict(data, labels))

            _loss = _results[0]

            metric_values = {'loss': _loss}

            for i, metric_name in enumerate(self._get_metrics().keys()):
                metric_values[metric_name] = _results[i + 1]

            for key, value in metric_values.items():
                if key not in metrics:
                    metrics[key] = []

                metrics[key].append(value)

        for callback in callbacks:
            callback.validation_end(logs={'metrics': metrics, 'validation_step': validation_step})
