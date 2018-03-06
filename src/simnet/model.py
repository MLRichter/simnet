import logging
from abc import ABC, abstractmethod

import tensorflow as tf

from src.simnet.callbacks import Monitor
from src.simnet.util import Progbar
from src.simnet.util import get_class_weights
from sklearn.metrics.classification import accuracy_score
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

logging.basicConfig()
log = logging.getLogger(__name__)


class Model(ABC):
    @abstractmethod
    def fit(self, session: tf.Session, train_generator, val_generator, epochs: int, batch_size: int, validation_step: int, callbacks=[]):
        pass

    @abstractmethod
    def evaluate(self, session: tf.Session, val_generator, batch_size):
        pass


class AbstractModel(Model):
    def fit(self, session: tf.Session, train_generator, val_generator, epochs: int, batch_size: int, validation_step: int, callbacks=[]):
        monitor = Monitor()
        callbacks.append(monitor)

        self._do_fit(session, train_generator, val_generator, epochs, batch_size, validation_step, callbacks)

        return monitor.train_history, monitor.validation_history

    def evaluate(self, session: tf.Session, val_generator, batch_size: int):
        monitor = Monitor()
        callbacks = []
        callbacks.append(monitor)

        self._do_evaluation(session, val_generator, batch_size, 1, callbacks)

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

        progbar = Progbar(train_generator.steps(batch_size))
        for data, labels in train_generator.get_batch(batch_size):
            for callback in callbacks:
                callback.batch_start({})

            targets = [self._get_train_step(), self._get_loss()]
            targets.extend(self._get_metrics().values())

            # run training step
            _results = sess.run(targets, feed_dict=self._build_feed_dict(data, labels))

            class_weights = get_class_weights()

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

        t_y = None
        t_preds = None
        # test model on validation data
        for data, labels in val_generator.get_batch(batch_size):

            targets = [self._get_loss()]
            targets.extend(self._get_metrics().values())

            _results = session.run(targets, feed_dict=self._build_feed_dict(data, labels))
            preds = session.run(self._predicted_class, feed_dict=self._build_feed_dict(data, labels))

            if type(t_y) == type(None):
                t_y = np.argmax(labels,axis=1)
                t_preds = preds
            else:
                t_y = np.hstack((t_y,np.argmax(labels,axis=1)))
                t_preds = np.hstack((t_preds,preds))




            _loss = _results[0]

            metric_values = {'loss': _loss}

            for i, metric_name in enumerate(self._get_metrics().keys()):
                metric_values[metric_name] = _results[i + 1]

            for key, value in metric_values.items():
                if key not in metrics:
                    metrics[key] = []

                metrics[key].append(value)

        class_weight = compute_class_weight('balanced',np.unique(t_y),t_y)
        sample_weights = np.zeros(t_y.shape[0])
        for i,weight in enumerate(class_weight):
            sample_weights[t_y == i] = weight
        try:
            weighted_acc = accuracy_score(t_y,t_preds,True,sample_weights)
        except:
            weighted_acc = None
        metrics['weighted_acc'] = weighted_acc

        for callback in callbacks:
            callback.validation_end(logs={'metrics': metrics, 'validation_step': validation_step})
