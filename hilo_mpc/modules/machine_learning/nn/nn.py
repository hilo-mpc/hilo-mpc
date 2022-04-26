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

from typing import Any, Callable, Optional, TypeVar, Union
import warnings

import casadi as ca
import numpy as np

from ..base import LearningBase
from ....plugins.plugins import LearningManager, LearningVisualizationManager, check_version
from ....util.data import DataSet
from ....util.machine_learning import net_to_casadi_graph
from ....util.util import is_list_like


ML = TypeVar('ML', bound=LearningBase)


class ArtificialNeuralNetwork(LearningBase):
    """Artificial neural network class"""
    def __init__(self, features, labels, id=None, name=None, **kwargs):
        """Constructor method"""
        super().__init__(features, labels, id=id, name=name)

        self._layers = []
        self._weights = None
        self._bias = None
        self._dropouts = None
        self._hidden = None

        self._seed = kwargs.get('seed')

        learning_rate = kwargs.get('learning_rate')
        if learning_rate is None:
            learning_rate = .001
        self._learning_rate = learning_rate

        loss = kwargs.get('loss')
        if loss is None:
            loss = 'mse'
        self._loss = loss

        optimizer = kwargs.get('optimizer')
        if optimizer is None:
            optimizer = 'adam'
        self._optimizer = optimizer

        metric = kwargs.get('metric')
        if metric is None:
            metric = []
        self._metric = metric

        self._data_sets = []

        train_backend = kwargs.get('backend')
        if train_backend is None:
            train_backend = 'pytorch'
        self._set_backend(train_backend)
        self._net = None

        self._check_data_sets()

        self._scaler_x = None
        self._scaler_y = None
        self._train_data = (None, None)
        self._validate_data = (None, None)
        self._test_data = (None, None)
        self._pandas_version_checked = False

    @staticmethod
    def _set_scaling(scaler: Union[str, Callable], backend: Optional[str] = None) -> Any:
        """

        :param scaler:
        :param backend:
        :return:
        """
        # TODO: Can we use something other than Any for typing the returned value?
        if isinstance(scaler, str):
            if backend is not None:
                scaler = LearningManager(backend).setup(scaler)
            else:
                warnings.warn("Scaling was selected, but no scaler backend was supplied. Please select a scaler backend"
                              " or supply your own scaler object. No scaling applied.")
                return None
        if callable(scaler):
            scaler = scaler()
        has_fit = hasattr(scaler, 'fit') and callable(scaler.fit)
        if not has_fit:
            raise RuntimeError("Supplied scaler is missing the method 'fit'")  # TODO: Return a little more info
        has_transform = hasattr(scaler, 'transform') and callable(scaler.transform)
        if not has_transform:
            raise RuntimeError("Supplied scaler is missing the method 'transform'")  # TODO: Return a little more info

        return scaler

    @staticmethod
    def _check_scaler(scaler: Any) -> None:
        """

        :param scaler:
        :return:
        """
        # TODO: See self._set_scaling (regarding Any)
        if not hasattr(scaler, 'mean_'):
            # TODO: Return a little more info
            raise RuntimeError("Supplied scaler is missing the attribute 'mean_'")
        if not hasattr(scaler, 'scale_'):
            # TODO: Return a little more info
            raise RuntimeError("Supplied scaler is missing the attribute 'scale_'")

    def _check_data_sets(self, data_sets=None):
        """

        :param data_sets:
        :return:
        """
        if data_sets is None:
            data_sets = self._data_sets

        if not is_list_like(data_sets):
            data_sets = [data_sets]

        for data_set in data_sets:
            if isinstance(data_set, DataSet):
                for feature in self._features:
                    if feature not in data_set.features:
                        raise ValueError(f"Feature {feature} does not exist in the supplied data set")
                for label in self._labels:
                    if label not in data_set.labels:
                        raise ValueError(f"Label {label} does not exist in the supplied data set")
            else:
                # NOTE: Right now only pandas dataframes are supported
                if not self._pandas_version_checked:
                    check_version('pandas')
                    self._pandas_version_checked = True
                for feature in self._features:
                    if feature not in data_set.columns:
                        raise ValueError(f"Feature {feature} does not exist in the supplied data set")
                for label in self._labels:
                    if label not in data_set.columns:
                        raise ValueError(f"Label {label} does not exist in the supplied data set")

    @LearningBase.features.setter
    def features(self, arg):
        self._features = arg
        self._n_features = len(arg)
        self._check_data_sets()

    @LearningBase.labels.setter
    def labels(self, arg):
        self._labels = arg
        self._n_labels = len(arg)
        self._check_data_sets()

    @property
    def seed(self):
        """

        :return:
        """
        return self._seed

    @seed.setter
    def seed(self, seed):
        self._seed = seed

    @property
    def learning_rate(self):
        """

        :return:
        """
        return self._learning_rate

    @learning_rate.setter
    def learning_rate(self, lr):
        self._learning_rate = lr

    @property
    def loss(self):
        """

        :return:
        """
        return self._loss

    @loss.setter
    def loss(self, loss):
        self._loss = loss

    @property
    def optimizer(self):
        """

        :return:
        """
        return self._optimizer

    @optimizer.setter
    def optimizer(self, opti):
        self._optimizer = opti

    @property
    def metric(self):
        """

        :return:
        """
        return self._metric

    @metric.setter
    def metric(self, metric):
        self._metric = metric

    @property
    def depth(self):
        """

        :return:
        """
        return len([layer for layer in self._layers if layer.type.lower() != 'dropout'])

    @property
    def shape(self):
        """

        :return:
        """
        return (self._n_features,) + tuple(layer.nodes for layer in self._layers if layer.type.lower() != 'dropout') + (
            self._n_labels,)

    @property
    def backend(self):
        """

        :return:
        """
        return self._backend.backend

    @backend.setter
    def backend(self, backend):
        self._set_backend(backend)

    def add_layers(self, layers):
        """

        :param layers:
        :return:
        """
        if isinstance(layers, (list, tuple, set)):
            for layer in layers:
                self.add_layers(layer)
        else:
            self._layers.append(layers)
            layers.parent = self

    def add_data_set(self, data_set):
        """

        :param data_set:
        :return:
        """
        self._check_data_sets(data_set)
        self._data_sets.append(data_set)

    def build_graph(self, weights=None, bias=None):
        """

        :param weights:
        :param bias:
        :return:
        """
        x = ca.SX.sym('x', self._n_features)
        if weights is None and bias is None:
            self._function = self._net.build_graph(x, self._layers, input_scaling=self._scaler_x,
                                                   output_scaling=self._scaler_y)
        else:
            self._function = net_to_casadi_graph({'weights': weights, 'bias': bias}, x, self._layers,
                                                 input_scaling=self._scaler_x, output_scaling=self._scaler_y)

    def prepare_data_set(
            self,
            train_split: float = 1.,
            validation_split: float = 0.,
            scale_data: bool = False,
            scaler: Optional[str] = None,
            scaler_backend: Optional[str] = None,
            shuffle: bool = True
    ) -> None:
        """

        :param train_split:
        :param validation_split:
        :param scale_data:
        :param scaler:
        :param scaler_backend:
        :param shuffle:
        :return:
        """
        data = self._data_sets[0].append(self._data_sets[1:], ignore_index=True, sort=False)
        if isinstance(data, DataSet):
            x_data, y_data = data.raw_data
            # TODO: Make this the default in the future, so we don't have to transpose (therefore transpose for pandas
            #  objects)
            x_data = x_data.T
            y_data = y_data.T
            # NOTE: I don't think at the moment that a length check is necessary here
        else:
            # NOTE: pandas is assumed here for now
            x_data = data[self._features].values
            y_data = data[self._labels].values

            if len(x_data) != len(y_data):
                raise ValueError(f"Dimension mismatch. Features have {len(x_data)} entries and labels have "
                                 f"{len(y_data)} entries.")

        if scale_data or scaler is not None or scaler_backend is not None:
            if scaler is None:
                scaler = 'StandardScaler'
            self.set_scaling(scaler, backend=scaler_backend)

        if self._scaler_x is not None:
            self._scaler_x.fit(x_data)
            self._check_scaler(self._scaler_x)
            x_data = self._scaler_x.transform(x_data)
        if self._scaler_y is not None:
            self._scaler_y.fit(y_data)
            self._check_scaler(self._scaler_y)
            y_data = self._scaler_y.transform(y_data)

        # TODO: Add support for data split directly in DataSet class
        data_set_size = len(data)
        indices = list(range(data_set_size))
        if shuffle:
            np.random.seed(self._seed)
            np.random.shuffle(indices)
        # NOTE: For odd number of data sets this can lead to different sizes although same split percentage was supplied
        train, validate, test = np.split(indices, [int(train_split * data_set_size),
                                                   int((train_split + validation_split) * data_set_size)])

        x_train = x_data[train, :]
        y_train = y_data[train, :]
        self._train_data = (x_train, y_train)

        x_validate = x_data[validate, :]
        y_validate = y_data[validate, :]
        self._validate_data = (x_validate, y_validate)

        x_test = x_data[test, :]
        y_test = y_data[test, :]
        self._test_data = (x_test, y_test)

    def set_input_scaling(self, scaler: Union[str, Callable], backend: Optional[str] = None) -> None:
        """

        :param scaler:
        :param backend:
        :return:
        """
        self._scaler_x = self._set_scaling(scaler, backend=backend)

    def set_output_scaling(self, scaler: str, backend: Optional[str] = None) -> None:
        """

        :param scaler:
        :param backend:
        :return:
        """
        self._scaler_y = self._set_scaling(scaler, backend=backend)

    def set_scaling(self, scaler: Union[str, Callable], backend: Optional[str] = None) -> None:
        """

        :param scaler:
        :param backend:
        :return:
        """
        if isinstance(scaler, str) and backend is not None:
            scaler = LearningManager(backend).setup(scaler)
        self.set_input_scaling(scaler)
        self.set_output_scaling(scaler)

    def setup(self, **kwargs):
        """

        :param kwargs:
        :return:
        """
        loss = kwargs.get('loss')
        if loss is None:
            loss = self._loss
        else:
            self._loss = loss

        optimizer = kwargs.get('optimizer')
        if optimizer is None:
            optimizer = self._optimizer
        else:
            self._optimizer = optimizer

        metric = kwargs.get('metric')
        if metric is None:
            metric = self._metric
        else:
            self._metric = metric

        show_tensorboard = kwargs.get('show_tensorboard')
        if show_tensorboard is None:
            show_tensorboard = False
            save_tensorboard = kwargs.get('save_tensorboard')
            if save_tensorboard is None:
                save_tensorboard = False
        else:
            if show_tensorboard:
                save_tensorboard = True
            else:
                save_tensorboard = kwargs.get('save_tensorboard')
                if save_tensorboard is None:
                    save_tensorboard = False
        tensorboard_log_dir = kwargs.get('tensorboard_log_dir')
        if save_tensorboard:
            tensorboard = LearningVisualizationManager('tensorboard').setup(log_dir=tensorboard_log_dir)
        else:
            tensorboard = None
        if show_tensorboard:
            browser = kwargs.get('browser')
            if browser is None:
                browser = 'chrome'
        else:
            browser = None

        device = kwargs.get('device')

        properties = [self._n_features, self._n_labels, self._layers]
        options = {
            'seed': self._seed,
            'learning_rate': self._learning_rate,
            'loss': loss,
            'optimizer': optimizer,
            'metric': metric,
            'tensorboard': tensorboard,
            'browser': browser,
            'device': device
        }

        self._net = self._backend.setup('MLP', *properties, **options)

    def is_setup(self):
        """

        :return:
        """
        if self._net is not None:
            return True
        else:
            return False

    def train(
            self,
            batch_size: int,
            epochs: int,
            verbose: int = 1,
            validation_split: float = 0.,
            test_split: float = 0.,
            scale_data: bool = False,
            scaler: Optional[str] = None,
            scaler_backend: Optional[str] = None,
            shuffle: bool = True,
            patience: Optional[int] = None
    ) -> None:
        """

        :param batch_size:
        :param epochs:
        :param verbose:
        :param validation_split:
        :param test_split:
        :param scale_data:
        :param scaler:
        :param scaler_backend:
        :param shuffle:
        :param patience:
        :return:
        """
        if not self._data_sets:
            raise RuntimeError("No data set to train on was supplied. Please add a data set by using the 'add_data_set'"
                               " method.")

        if validation_split > 1. or validation_split < 0.:
            raise ValueError("Validation split has to be between 0 and 1")
        if test_split > 1. or test_split < 0.:
            raise ValueError("Test split has to be between 0 and 1")
        train_split = 1. - validation_split - test_split
        if train_split <= 0.:
            raise ValueError("Train split is not big enough. Reduce validation split or test split.")

        self.prepare_data_set(train_split=train_split, validation_split=validation_split, scale_data=scale_data,
                              scaler=scaler, scaler_backend=scaler_backend, shuffle=shuffle)

        self._net.train(self._train_data, self._validate_data, batch_size, epochs, verbose, patience, shuffle)
        if all(data.size > 0 for data in self._test_data):
            print("Evaluate on test data")
            self._net.evaluate(data=self._test_data, batch_size=batch_size, verbose=verbose)

        self.build_graph()

    def is_trained(self):
        """

        :return:
        """
        return super().is_setup()

    def predict(self, X_query):
        """

        :param X_query:
        :return:
        """
        # TODO: We could also add a predict method to the wrapper and just link here, but this would not work with
        #  CasADi variables like this
        return self._function(X_query)[0]

    def show_tensorboard(self, browser=None):
        """

        :param browser:
        :return:
        """
        if self.is_trained():
            if browser is None:
                browser = 'chrome'
            self._net.show_tensorboard(browser.lower())
        else:
            warnings.warn("The artificial neural network has not been trained yet. Aborting...")

    def close_tensorboard(self):
        """

        :return:
        """
        if self.is_trained():
            self._net.close_tensorboard()


__all__ = [
    'ArtificialNeuralNetwork'
]
