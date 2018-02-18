from abc import ABC, abstractmethod
import tensorflow as tf

from .callbacks import Monitor


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
                self._do_epoch(train_generator, val_generator, val_generator, validation_step, batch_size, callbacks)

            for callback in callbacks:
                callback.train_end({})

    def _do_epoch(self, sess: tf.Session, train_generator, val_generator, validation_step: int, batch_size: int, callbacks):
        for callback in callbacks:
            callback.epoch_start({})

        step = 0
        for data, labels in train_generator.get_batch(batch_size):
            for callback in callbacks:
                callback.batch_start({})

            # run training step
            _, _loss = sess.run([self._get_train_step(), self._get_loss()], feed_dict=self._build_feed_dict(data, labels))

            if step % validation_step == 0:
                self._do_evaluation(sess, val_generator, batch_size, validation_step, callbacks)

            step += 1

            for callback in callbacks:
                callback.batch_end({'loss': _loss})

        for callback in callbacks:
            callback.epoch_start({})

    def _get_train_step(self):
        pass

    def _get_loss(self):
        pass

    def _build_feed_dict(self, samples, labels):
        pass

    def _init_session(self, session: tf.Session):
        session.run(tf.global_variables_initializer())

    def _do_evaluation(self, session: tf.Session, val_generator, batch_size, validation_step, callbacks):
        for callback in callbacks:
            callback.validation_start({})

        v_loss = []
        v_acc = []
        # test model on validation data
        for data, labels in val_generator.get_batch(batch_size):

            _loss = session.run([self._get_loss()], feed_dict=self._build_feed_dict(data, labels))

            # keep track of loss and accuracy
            v_loss.append(_loss)

        for callback in callbacks:
            callback.validation_end(logs={
                'loss': v_loss * validation_step,
                'accuracy': v_acc * validation_step
            })
