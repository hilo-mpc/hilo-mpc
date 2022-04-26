#   
#   This file is part of HILO-MPC
#
#   HILO-MPC is a toolbox for easy, flexible and fast development of machine-learning-supported
#   optimal control and estimation problems
#
#   Copyright (c) 2021 Johannes Pohlodek, Bruno Morabito, Rolf Findeisen
#                      All rights reserved
#
#   HILO-MPC is free software: you can redistribute it and/or modify
#   it under the terms of the GNU Lesser General Public License as
#   published by the Free Software Foundation, either version 3
#   of the License, or (at your option) any later version.
#
#   HILO-MPC is distributed in the hope that it will be useful,
#   but WITHOUT ANY WARRANTY; without even the implied warranty of
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
#   GNU Lesser General Public License for more details.
#
#   You should have received a copy of the GNU Lesser General Public License
#   along with HILO-MPC. If not, see <http://www.gnu.org/licenses/>.
#

import copy
from distutils.version import StrictVersion
from typing import Optional
import warnings

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Dropout, Input
import tensorflow.keras.losses as tfkl
from tensorflow.keras.optimizers import Adadelta, Adagrad, Adam, Adamax, Ftrl, Nadam, RMSprop, SGD
import tensorflow.keras.metrics as tfkm
from tensorflow.keras.callbacks import CallbackList, EarlyStopping, ReduceLROnPlateau
if StrictVersion(tf.__version__) >= StrictVersion('2.5.0'):
    from tensorflow.python.keras.utils.tf_utils import sync_to_numpy_or_python_type
else:
    from tensorflow.python.keras.utils.tf_utils import to_numpy_or_python_type as sync_to_numpy_or_python_type

from ...util.machine_learning import net_to_casadi_graph


def _mlp(n_features, n_labels, layers):
    """

    :param n_features:
    :param n_labels:
    :param layers:
    :return:
    """
    hidden = _process_layers(layers)
    inputs = Input(shape=(n_features, ))  # add name?
    x = inputs
    for layer in hidden:
        x = layer(x)
    # NOTE: Right now the output layer is assumed to be linear and cannot be changed
    outputs = Dense(n_labels)(x)

    return Model(inputs=inputs, outputs=outputs)


LOSS = {
    'mse': tfkl.MeanSquaredError,
    'mae': tfkl.MeanAbsoluteError,
    'mape': tfkl.MeanAbsolutePercentageError,
    'msle': tfkl.MeanSquaredLogarithmicError,
    'cs': tfkl.CosineSimilarity,
    'huber': tfkl.Huber,
    'log-cosh': tfkl.LogCosh
}
OPTIMIZER = {
    'adadelta': Adadelta,
    'adagrad': Adagrad,
    'adam': Adam,
    # 'adamw': NotImplemented,
    # 'sparse_adam': NotImplemented,
    'adamax': Adamax,
    # 'asgd': NotImplemented,
    'ftrl': Ftrl,
    # 'lbfgs': NotImplemented,
    'nadam': Nadam,
    'rmsprop': RMSprop,
    # 'rprop': NotImplemented,
    'sgd': SGD
}
METRIC = {
    'mse': tfkm.MeanSquaredError,
    'rmse': tfkm.RootMeanSquaredError,
    'mae': tfkm.MeanAbsoluteError,
    'mape': tfkm.MeanAbsolutePercentageError,
    'msle': tfkm.MeanSquaredLogarithmicError,
    'cs': tfkm.CosineSimilarity,
    'log-cosh': tfkm.LogCoshError
}


class _TensorFlowWrapper:
    """"""
    def __init__(self, module, **kwargs):
        """Constructor method"""
        self._module = module

        self._seed = kwargs.get('seed')

        learning_rate = kwargs.get('learning_rate')
        if learning_rate is None:
            learning_rate = .001

        loss = kwargs.get('loss')
        if loss is None:
            loss = 'mse'
        loss = LOSS[loss.lower()]()
        self._loss = loss

        optimizer = kwargs.get('optimizer')
        if optimizer is None:
            optimizer = 'adam'
        optimizer = OPTIMIZER[optimizer.lower()](learning_rate=learning_rate)
        self._optimizer = optimizer

        metric = kwargs.get('metric')
        if metric is None:
            metric = []
        self._metrics = [METRIC[k.lower()]() for k in metric]
        self._metric_names = metric

        # self._module.compile(optimizer=optimizer, loss=loss)

        self._tensorboard = kwargs.get('tensorboard')
        self._tensorboard_comment = {
            'batch_size': 0,
            'learning_rate': learning_rate,
            'n_layers': len(self._module.layers) - 2,
            'nodes': tuple(layer.units for k, layer in enumerate(self._module.layers) if hasattr(layer, 'units'))[:-1],
            'seed': self._seed
        }
        self._browser = kwargs.get('browser')

        self._train_loader = None
        self._test_loader = None
        self._early_stopping = None
        self._scheduler = None
        self._callbacks = None
        self._writer = None

    def _get_data_loader(self, data, batch_size, shuffle):
        """

        :param data:
        :param batch_size:
        :return:
        """
        data_loader = tf.data.Dataset.from_tensor_slices(data)
        if shuffle:
            data_loader = data_loader.shuffle(len(data_loader), seed=self._seed,
                                              reshuffle_each_iteration=True).batch(batch_size)
        else:
            data_loader = data_loader.batch(batch_size)
        return data_loader

    def _preprocessing(
            self,
            data,
            validation_data,
            batch_size,
            verbose,
            patience,
            shuffle
    ):
        """

        :param data:
        :param validation_data:
        :param batch_size:
        :param verbose:
        :param patience:
        :param shuffle:
        :return:
        """
        self._train_loader = self._get_data_loader(data, batch_size, shuffle)
        # TODO: Does it make sense to shuffle validation data
        self._test_loader = self._get_data_loader(validation_data, batch_size, False)

        if patience is not None and len(self._test_loader) > 0:
            # TODO: Check and implement behavior shown in https://www.tensorflow.org/guide/keras/custom_callback
            self._early_stopping = EarlyStopping(monitor='val_loss', patience=patience, verbose=verbose)
            self._scheduler = ReduceLROnPlateau(monitor='val_loss', patience=int(patience / 2), verbose=verbose)
            self._module.optimizer = self._optimizer
        elif patience is not None and len(self._test_loader) == 0:
            warnings.warn("Early stopping is only available when validation data is supplied for the training of the "
                          "neural network")

        if self._tensorboard is not None:
            self._tensorboard_comment['batch_size'] = batch_size
            self._writer = self._tensorboard.get_callback(**self._tensorboard_comment)
            if self._browser is not None:
                self.show_tensorboard()

    def _train(self, verbose):
        """

        :param verbose:
        :return:
        """
        running_loss = 0.

        # TODO: Need a better variable name
        batches = len(self._train_loader)

        logs = None
        for idx, (x, y) in enumerate(self._train_loader):
            self._callbacks.on_train_batch_begin(idx)
            loss = self._train_step(x, y)
            running_loss += float(loss)
            logs = {'loss': running_loss / (idx + 1)}
            logs.update({metric.name: float(metric.result()) for metric in self._metrics})
            self._callbacks.on_train_batch_end(idx + 1, logs=logs)
            if verbose == 1:
                _progress_bar(idx + 1, batches, logs, 30, "=", "\r")
            if self._module.stop_training:
                break
        if verbose == 1:
            print()
        elif verbose == 2:
            _one_line_per_epoch(batches, logs)
        logs = sync_to_numpy_or_python_type(logs)
        return logs

    @tf.function
    def _train_step(self, x, y):
        """

        :param x:
        :param y:
        :return:
        """
        with tf.GradientTape() as tape:
            pred = self._module(x, training=True)
            loss = self._loss(y, pred)
        grads = tape.gradient(loss, self._module.trainable_weights)
        self._optimizer.apply_gradients(zip(grads, self._module.trainable_weights))
        for metric in self._metrics:
            metric.update_state(y, pred)
        return loss

    def _test(self, data_loader, callbacks, verbose):
        """

        :param data_loader:
        :param callbacks:
        :param verbose:
        :return:
        """
        running_loss = 0.

        # TODO: Need a better variable name
        batches = len(data_loader)

        logs = None
        for idx, (x, y) in enumerate(data_loader):
            callbacks.on_test_batch_begin(idx)
            loss = self._test_step(x, y)
            running_loss += float(loss)
            logs = {'loss': running_loss / (idx + 1)}
            logs.update({metric.name: float(metric.result()) for metric in self._metrics})
            callbacks.on_test_batch_end(idx + 1, logs=logs)
            if verbose == 1:
                _progress_bar(idx + 1, batches, logs, 30, "=", "\r")
        if verbose == 1:
            print()
        elif verbose == 2:
            _one_line_per_epoch(batches, logs)
        logs = sync_to_numpy_or_python_type(logs)
        return logs

    @tf.function
    def _test_step(self, x, y):
        """

        :param x:
        :param y:
        :return:
        """
        pred = self._module(x, training=False)
        loss = self._loss(y, pred)
        for metric in self._metrics:
            metric.update_state(y, pred)
        return loss

    def train(
            self,
            data,
            validation_data,
            batch_size,
            epochs,
            verbose=1,
            patience=None,
            shuffle=True
    ):
        """

        :param data:
        :param validation_data:
        :param batch_size:
        :param epochs:
        :param verbose:
        :param patience:
        :param shuffle:
        :return:
        """
        self._preprocessing(data, validation_data, batch_size, verbose, patience, shuffle)

        callbacks = [self._early_stopping, self._scheduler, self._writer]
        callbacks = [callback for callback in callbacks if callback is not None]
        self._callbacks = CallbackList(callbacks=callbacks, model=self._module, verbose=verbose, epochs=epochs,
                                       steps=len(self._train_loader))
        self._callbacks.on_train_begin()

        self._module.stop_training = False
        training_logs = None
        for epoch in range(epochs):
            if verbose > 0:
                print(f"Epoch {epoch + 1}/{epochs}")
            self._callbacks.on_epoch_begin(epoch)

            logs = self._train(verbose)
            if logs is None:
                raise ValueError("Dataset is empty")
            for metric in self._metrics:
                metric.reset_states()
            epoch_logs = copy.copy(logs)

            if len(self._test_loader) > 1:
                if verbose > 0:
                    print("Evaluate on validation data")
                val_logs = self.evaluate(callbacks=self._callbacks, verbose=verbose)
                for metric in self._metrics:
                    metric.reset_states()
                val_logs = {'val_' + name: val for name, val in val_logs.items()}
                epoch_logs.update(val_logs)

            self._callbacks.on_epoch_end(epoch, logs=epoch_logs)
            training_logs = epoch_logs
            if self._module.stop_training:
                break

        self._callbacks.on_train_end(logs=training_logs)

    def evaluate(
            self,
            data=None,
            batch_size=None,
            verbose=1,
            callbacks=None
    ):
        """

        :param data:
        :param batch_size:
        :param verbose:
        :param callbacks:
        :return:
        """
        if data is not None:
            data_loader = self._get_data_loader(data, batch_size, False)
        else:
            data_loader = self._test_loader

        if not isinstance(callbacks, CallbackList):
            callbacks = CallbackList(callbacks=callbacks, model=self._module, verbose=verbose, epochs=1,
                                     steps=len(data_loader))

        callbacks.on_test_begin()
        val_logs = self._test(data_loader, callbacks, verbose)
        callbacks.on_test_end(logs=val_logs)

        return val_logs

    def build_graph(self, x, layers, input_scaling=None, output_scaling=None):
        """

        :param x:
        :param layers:
        :param input_scaling:
        :param output_scaling:
        :return:
        """
        return net_to_casadi_graph(self, x, layers, input_scaling=input_scaling, output_scaling=output_scaling)

    def get_weights_and_bias(self):
        """

        :return:
        """
        weights = []
        bias = []

        for val in self._module.get_weights():
            if val.ndim == 2:
                weights.append(val.T)
            elif val.ndim == 1:
                bias.append(val)

        return weights, bias

    def show_tensorboard(self, browser: Optional[str] = None) -> None:
        """

        :param browser:
        :return:
        """
        if browser is None:
            browser = self._browser
        browser = browser.lower()

        if self._tensorboard is not None:
            self._tensorboard.show(browser)
        else:
            warnings.warn("No TensorBoard was saved. If you want to see a TensorBoard set the corresponding flag "
                          "during setup and train again.")

    def close_tensorboard(self):
        """

        :return:
        """
        if self._tensorboard is not None:
            self._tensorboard.close()


LAYER = {
    'dense': Dense,
    'dropout': Dropout
}


def _process_layers(layers):
    """

    :param layers:
    :return:
    """
    hidden = []
    for layer in layers:
        type_ = layer.type.lower()
        if type_ == 'dropout':
            hidden.append(LAYER[type_](layer.rate))
        else:
            hidden.append(LAYER[type_](layer.nodes, activation=layer.activation.lower()))
    return hidden


def _progress_bar(iteration, total, logs, length, fill, print_end):
    """

    :param iteration:
    :param total:
    :param logs:
    :param length:
    :param fill:
    :param print_end:
    :return:
    """
    # TODO: Migrate together with _progress_bar from PyTorch wrapper
    loss_metrics_string = ""
    for key, val in logs.items():
        loss_metrics_string += f" - {key}: {val:.4f}"
    filled_length = int(length * iteration // total)
    unfilled_length = length - filled_length
    bar = fill * filled_length + ">" * bool(unfilled_length) + "." * (unfilled_length - 1)
    print(f"\r{iteration}/{total} [{bar}] {loss_metrics_string}", end=print_end)


def _one_line_per_epoch(total, logs):
    """

    :param total:
    :param logs:
    :return:
    """
    loss_metrics_string = ""
    for key, val in logs.items():
        loss_metrics_string += f" - {key}: {val:.4f}"
    print(f"{total}/{total} {loss_metrics_string}")


MODULES = {
    'mlp': _mlp
}


def get_wrapper(kind, *args, **kwargs):
    """

    :param kind:
    :param args:
    :param kwargs:
    :return:
    """
    # TODO: Support for device and seed
    seed = kwargs.get('seed')
    tf.random.set_seed(seed)

    module = MODULES[kind.lower()](*args)

    return _TensorFlowWrapper(module, **kwargs)


__all__ = [
    'get_wrapper'
]
