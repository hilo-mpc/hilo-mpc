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

from .layer import Probabilistic
from ..base import LearningBase
from ....plugins.plugins import LearningManager, LearningVisualizationManager, check_version
from ....util.data import DataSet
from ....util.machine_learning import Activation, Hyperparameter, net_to_casadi_graph
from ....util.probability import Gaussian, GaussianPrior
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
        # TODO: Check whether migration to LearningBase class makes sense, since _PBPApproximation also needs this
        #  method
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

    def _parse_options(self, **kwargs) -> dict:
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

        return {
            'seed': self._seed,
            'learning_rate': self._learning_rate,
            'loss': loss,
            'optimizer': optimizer,
            'metric': metric,
            'tensorboard': tensorboard,
            'browser': browser,
            'device': device
        }

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
        properties = [self._n_features, self._n_labels, self._layers]
        options = self._parse_options(**kwargs)

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


class BayesianNeuralNetwork:
    """Bayesian neural network class"""
    def __new__(cls, *args, **kwargs):
        """Creator method"""
        if args:
            approximation = args[0]
            args = args[1:]
        else:
            if kwargs:
                approximation = kwargs.pop('approximation', None)
            else:
                approximation = None
        if approximation is None:
            raise ValueError("No approximation method for the Bayesian neural network was specified. "
                             "Choose either 'laplace' or 'pbp'!")
        approximation_lower = approximation.lower()
        if approximation_lower not in ['laplace', 'pbp']:
            raise ValueError(
                f"Approximation method '{approximation}' for Bayesian neural networks not recognized or not implemented"
            )
        elif approximation_lower == 'laplace':
            kwargs['backend'] = 'laplace-torch'
            return _LaplaceApproximation(*args, **kwargs)
        elif approximation_lower == 'pbp':
            return _PBPApproximation(*args, **kwargs)


class _LaplaceApproximation(ArtificialNeuralNetwork):
    """Bayesian neural network with Laplace approximation"""
    def __init__(self, features, labels, id=None, name=None, **kwargs):
        """Constructor method"""
        super().__init__(features, labels, id=id, name=name, **kwargs)

        likelihood = kwargs.get('likelihood')
        if likelihood is None:
            likelihood = 'regression'
        self._likelihood = likelihood

        subset_of_weights = kwargs.get('subset_of_weights')
        if subset_of_weights is None:
            subset_of_weights = 'all'
        self._subset_of_weights = subset_of_weights

        hessian_structure = kwargs.get('hessian_structure')
        if hessian_structure is None:
            hessian_structure = 'full'
        self._hessian_structure = hessian_structure

        hyperparameter_learning_rate = kwargs.get('hyperparameter_learning_rate')
        if hyperparameter_learning_rate is None:
            hyperparameter_learning_rate = .1
        self._hyperparameter_learning_rate = hyperparameter_learning_rate

        hyperparameter_steps = kwargs.get('hyperparameter_steps')
        if hyperparameter_steps is None:
            hyperparameter_steps = 100
        self._hyperparameter_steps = hyperparameter_steps

    def _parse_options(self, **kwargs) -> dict:
        """

        :param kwargs:
        :return:
        """
        options = super()._parse_options(**kwargs)

        subset_of_weights = kwargs.get('subset_of_weights')
        if subset_of_weights is None:
            subset_of_weights = self._subset_of_weights
        else:
            self._subset_of_weights = subset_of_weights

        hessian_structure = kwargs.get('hessian_structure')
        if hessian_structure is None:
            hessian_structure = self._hessian_structure
        else:
            self._hessian_structure = hessian_structure

        options['likelihood'] = self._likelihood
        options['subset_of_weights'] = subset_of_weights
        options['hessian_structure'] = hessian_structure
        options['hyperparameter_learning_rate'] = self._hyperparameter_learning_rate
        options['hyperparameter_steps'] = self._hyperparameter_steps

        return options

    def build_graph(self, weights=None, bias=None):
        """

        :param weights:
        :param bias:
        :return:
        """
        x = ca.SX.sym('x', self._n_features)
        weights, bias = self._net.get_weights_and_bias()

        weights_sym = [ca.SX.sym(f'weights_{k}', *w.shape) for k, w in enumerate(weights)]
        bias_sym = [ca.SX.sym(f'bias_{k}', *b.shape) for k, b in enumerate(bias)]

        parameters = [ca.horzcat(ca.reshape(w, 1, -1), ca.reshape(b, 1, -1)) for (w, b) in zip(weights_sym, bias_sym)]
        parameters = ca.horzcat(*parameters)

        f_mu = net_to_casadi_graph({'weights': weights_sym, 'bias': bias_sym, 'symbolic': True}, x, self._layers,
                                   input_scaling=self._scaler_x)
        J_sym = ca.Function('J', [x, *weights_sym, *bias_sym],
                            [ca.gradient(f_mu(x, *weights_sym, *bias_sym)[0], parameters)])
        J = J_sym(x, *weights, *bias)

        mean = f_mu(x, *weights, *bias)[0]
        var = J @ self._net.module.posterior_covariance.detach().cpu().numpy() @ J.T
        if self._scaler_y is not None:
            mean *= self._scaler_y.scale_
            mean += self._scaler_y.mean_
            var *= self._scaler_y.var_

        self._function = ca.Function('neural_network', [x], [mean, var], ['features'], ['label_mean', 'label_variance'])

    def predict(self, X_query):
        """

        :param X_query:
        :return:
        """
        mean, var = self._function(X_query)
        if self._scaler_y is not None:
            var += self._net.module.sigma_noise.detach().cpu().numpy() ** 2 * self._scaler_y.var_
        else:
            var += self._net.module.sigma_noise.detach().cpu().numpy() ** 2
        return mean, var


class _PBPApproximation(LearningBase):
    """"""
    def __init__(self, features, labels, hyperprior=None, id=None, name=None, **kwargs):
        """Constructor method"""
        super().__init__(features, labels, id=id, name=name)

        self._layers = []

        if hyperprior is None:
            hyperprior = 'gamma'
        hyperprior_parameters = kwargs.get('hyperprior_parameters')

        if is_list_like(hyperprior):
            hyper_kwargs = {'prior': hyperprior[0]}
        else:
            hyper_kwargs = {'prior': hyperprior}
        if hyperprior_parameters is not None:
            hyper_kwargs['prior_parameters'] = hyperprior_parameters.get('noise_variance')
        self.noise_variance = Hyperparameter('PBP.noise_variance', **hyper_kwargs)

        if is_list_like(hyperprior):
            hyper_kwargs = {'prior': hyperprior[1]}
        if hyperprior_parameters is not None:
            hyper_kwargs['prior_parameters'] = hyperprior_parameters.get('weights')
        # TODO: See test_means.py
        # else:
        #     hyper_kwargs['prior_parameters'] = {'shape': 6., 'rate': 6.}
        self._weights = Hyperparameter('PBP.weights', **hyper_kwargs)
        if hyperprior_parameters is None and hyper_kwargs.get('prior_parameters') is None:
            self._weights.prior.shape = 6.
            self._weights.prior.rate = 6.

        self._data_sets = []

        self._net = None

        self._check_data_sets()

        self._scaler_x = None
        self._scaler_y = None
        self._train_data = (None, None)
        self._validate_data = (None, None)
        self._test_data = (None, None)
        self._pandas_version_checked = False

    _check_data_sets = ArtificialNeuralNetwork._check_data_sets

    depth = ArtificialNeuralNetwork.depth

    shape = ArtificialNeuralNetwork.shape

    add_layers = ArtificialNeuralNetwork.add_layers

    add_data_set = ArtificialNeuralNetwork.add_data_set

    prepare_data_set = ArtificialNeuralNetwork.prepare_data_set

    def setup(self, **kwargs):
        """

        :param kwargs:
        :return:
        """
        properties = [self._n_features, self._n_labels, self._layers, self.noise_variance.prior, self._weights.prior]

        self._net = _ProbabilisticMLP(*properties)

    def train(self, epochs: int, verbose: int = 1):
        """"""
        if not self._data_sets:
            raise RuntimeError("No data set to train on was supplied. Please add a data set by using the 'add_data_set'"
                               " method.")

        self.prepare_data_set(shuffle=False)

        self._net.train(self._train_data, epochs, verbose)

    def predict(self, X_query):
        """"""
        mean, var = self._function(X_query)
        var += self.noise_variance.prior.rate / (self.noise_variance.prior.shape - 1.)
        return mean, var


class _DataSet:
    """"""
    def __init__(self, x_data, y_data):
        """Constructor method"""
        self.features = x_data
        self.labels = y_data

    def __len__(self):
        """Length method"""
        return len(self.features)

    def __getitem__(self, item):
        """Item getter method"""
        return self.features[item, :], self.labels[item, :]


class _ProbabilisticMLP:
    """"""
    def __init__(self, n_features, n_labels, layers, noise_variance_prior, weight_prior):
        """Constructor method"""
        hidden, activation = _process_probabilistic_layers(n_features, n_labels, layers, weight_prior)
        self._hidden = hidden
        self._activation = activation
        self._n_layers = len(layers)
        self._nodes = (n_features, ) + tuple(layer.nodes for layer in layers) + (n_labels, )

        self._initialize_weights(symbolic=True)
        self._generate_normalizer_and_gradients(n_features, n_labels)

        self._train_loader = None
        self._permutations = None
        self._noise_variance_prior = noise_variance_prior

    @staticmethod
    def _get_data_loader(data):
        """"""
        data_loader = _DataSet(data[0], data[1])
        return data_loader

    @staticmethod
    def _remove_invalid_updates(new_mean, old_mean, new_var, old_var):
        """

        :param new_mean:
        :param old_mean:
        :param new_var:
        :param old_var:
        :return:
        """
        idx1 = np.where(new_var <= 1e-100)
        idx2 = np.where(np.logical_or(np.isnan(new_mean), np.isnan(new_var)))
        idx = (np.concatenate((idx1[0], idx2[0])), np.concatenate((idx1[1], idx2[1])))

        if idx[0].size > 0:
            new_mean[idx] = old_mean[idx]
            new_var[idx] = old_var[idx]

        return new_mean, new_var

    def _assumed_density_filtering(self, x, y):
        """"""
        alpha = self._noise_variance_prior.shape
        beta = self._noise_variance_prior.rate

        Z = self._normalizer(x, y, alpha, beta, *[w for weight in self._weights for w in weight])
        logZ, logZ1, logZ2 = Z[:3]

        self._noise_variance_prior.shape = float(1. / (np.exp(logZ + logZ2 - 2. * logZ1) * (alpha + 1.) / alpha - 1.))
        self._noise_variance_prior.rate = float(
            1. / (np.exp(logZ2 - logZ1) * (alpha + 1.) / beta - np.exp(logZ1 - logZ) * alpha / beta))

        new_weights = []
        for k, weight in enumerate(self._weights):
            mean = weight[0]
            var = weight[1]
            mean += var * Z[2 * k + 3]
            var -= var ** 2. * (Z[2 * k + 3] ** 2. - 2. * Z[2 * k + 4])
            mean, var = self._remove_invalid_updates(mean.full(), weight[0], var.full(), weight[1])
            new_weights.append((mean, var))

        self._weights = new_weights

    def _initialize_weights(self, symbolic=False):
        """

        :param symbolic:
        :return:
        """
        if symbolic:
            self._weights = [(ca.SX.sym(f'w_mean_{k}', self._nodes[k + 1], self._nodes[k] + 1),
                              ca.SX.sym(f'w_var_{k}', self._nodes[k + 1], self._nodes[k] + 1)) for k in
                             range(self._n_layers + 1)]
        else:
            self._weights = [(GaussianPrior(mean=0., variance=1. / (self._nodes[k] + 1)).sample(
                (self._nodes[k + 1], self._nodes[k] + 1)),
                              self._noise_variance_prior.rate / (self._noise_variance_prior.shape - 1.) * np.ones(
                                  (self._nodes[k + 1], self._nodes[k] + 1))) for k in range(self._n_layers + 1)]

    def _generate_normalizer_and_gradients(self, n_inputs, n_outputs):
        """

        :param n_inputs:
        :param n_outputs:
        :return:
        """
        x = ca.SX.sym('x', n_inputs)
        y = ca.SX.sym('y', n_outputs)

        mean, var = self.forward(x)
        alpha = ca.SX.sym('alpha')
        beta = ca.SX.sym('beta')

        Z = Gaussian()
        logZ = Z.pdf(y, mean, var + beta / (alpha - 1), log=True)
        logZ1 = Z.pdf(y, mean, var + beta / alpha, log=True)
        logZ2 = Z.pdf(y, mean, var + beta / (alpha + 1), log=True)

        normalizer_in = [x, y, alpha, beta]
        normalizer_out = [logZ, logZ1, logZ2]
        for weight in self._weights:
            normalizer_in.append(weight[0])  # mean
            normalizer_in.append(weight[1])  # variance
            normalizer_out.append(ca.gradient(logZ, weight[0]))  # gradient w.r.t. mean
            normalizer_out.append(ca.gradient(logZ, weight[1]))  # gradient w.r.t. variance

        self._normalizer = ca.Function('normalizer', normalizer_in, normalizer_out)

    def _preprocessing(self, data, epochs):
        """"""
        self._train_loader = self._get_data_loader(data)
        self._initialize_weights()
        batches = len(self._train_loader)
        self._permutations = (np.random.choice(range(batches), batches, replace=False) for _ in range(epochs))

    def _train(self, permutation, verbose):
        """"""
        batches = permutation.size
        for idx, k in np.ndenumerate(permutation):
            self._assumed_density_filtering(*self._train_loader[k])
            if verbose == 1:
                _progress_bar(idx[0] + 1, batches, 30, "=", "\r")

        if verbose == 1:
            print()

        if verbose == 2:
            _one_line_per_epoch(batches)

        self._expectation_propagation_updates()

    def forward(self, mean):
        """

        :param mean:
        :return:
        """
        var = ca.SX.zeros(mean.shape)
        for k in range(self._n_layers + 1):
            mean, var = self._hidden[k](mean, var, *self._weights[k])
            if self._activation[k] is not None:
                mean, var = self._activation[k](mean, var)

        return mean, var

    def train(self, data, epochs, verbose):
        """"""
        self._preprocessing(data, epochs)

        for epoch, permutation in enumerate(self._permutations):
            if verbose > 0:
                print(f"Epoch {epoch + 1}/{epochs}")

            self._train(permutation, verbose)


def _process_probabilistic_layers(n_features, n_labels, layers, prior):
    """

    :param n_features:
    :param n_labels:
    :param layers:
    :param prior:
    :return:
    """
    n_inputs = n_features
    hidden = []
    activation = []
    output_layer = Probabilistic(n_labels)
    output_layer.activation = None
    x_mean = ca.SX.sym('x_mean', n_inputs)
    x_var = ca.SX.sym('x_var', n_inputs)
    for k, layer in enumerate(layers + [output_layer]):
        layer.initializer = prior
        w_mean = ca.SX.sym('w_mean', layer.nodes, n_inputs)
        w_var = ca.SX.sym('w_var', layer.nodes, n_inputs)
        hidden.append(layer.forward(x_mean, x_var, w_mean=w_mean, w_var=w_var))
        n_inputs = layer.nodes
        x_mean = ca.SX.sym('x_mean', n_inputs)
        x_var = ca.SX.sym('x_var', n_inputs)
        if layer.activation is not None:
            activation.append(
                ca.Function(layer.activation, [x_mean, x_var], Activation(layer.activation)(x_mean, x_var)))
        else:
            activation.append(None)
    return hidden, activation


def _progress_bar(iteration, total, length, fill, print_end):
    """

    :param iteration:
    :param total:
    :param length:
    :param fill:
    :param print_end:
    :return:
    """
    filled_length = int(length * iteration // total)
    unfilled_length = length - filled_length
    bar = fill * filled_length + ">" * bool(unfilled_length) + "." * (unfilled_length - 1)
    print(f"\r{iteration}/{total} [{bar}]", end=print_end)


def _one_line_per_epoch(total):
    """

    :param total:
    :return:
    """
    print(f"{total}/{total}")


__all__ = [
    'ArtificialNeuralNetwork',
    'BayesianNeuralNetwork'
]
