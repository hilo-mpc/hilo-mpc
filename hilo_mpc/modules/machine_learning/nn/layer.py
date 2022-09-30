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

from itertools import product
import warnings

import casadi as ca
import numpy as np

from ....util.probability import Gaussian, Prior
from ....util.util import is_list_like


class Layer:
    """Base layer"""
    def __init__(self, nodes, activation=None, initializer=None, parent=None, **kwargs):
        """Constructor method"""
        self._nodes = nodes
        if activation is None:
            activation = 'sigmoid'
        activation = activation.lower().replace(' ', '_')
        self._activation = activation
        self._set_initializer(initializer, **kwargs)
        self._parent = parent
        self._type = None

    def __len__(self):
        """Length method"""
        return self._nodes

    def _set_initializer(self, initializer, **kwargs):
        """

        :param initializer:
        :param kwargs:
        :return:
        """
        if initializer is not None:
            initializer = initializer.lower().replace(' ', '_')
            if initializer == 'uniform':
                min_val = kwargs.get('min_val')
                max_val = kwargs.get('max_val')
                if min_val is None:
                    min_val = -.05
                if max_val is None:
                    max_val = .05
                initializer = {'uniform': {'a': min_val, 'b': max_val}}
            elif initializer == 'normal':
                mean = kwargs.get('mean')
                std = kwargs.get('std')
                if mean is None:
                    mean = 0.
                if std is None:
                    std = .05
                initializer = {'normal': {'mean': mean, 'std': std}}
            elif initializer == 'constant':
                val = kwargs.get('val')
                if val is None:
                    val = 0
                initializer = {'constant': val}
            elif initializer == 'ones':
                pass
            elif initializer == 'zeros':
                pass
            elif initializer == 'eye':
                pass
            # Dirac initializer will be skipped, since it's only applicable to {3,4,5}-dimensional input tensors
            elif initializer in ['glorot_uniform', 'xavier_uniform']:
                initializer = 'xavier_uniform'
            elif initializer in ['glorot_normal', 'xavier_normal']:
                initializer = 'xavier_normal'
            elif initializer in ['kaiming_uniform', 'he_uniform']:
                if self._activation in ['relu', 'leaky_relu']:
                    initializer = {'kaiming_uniform': {}}

                    if self._activation == 'leaky_relu':
                        a = kwargs.get('a')
                        if a is None:
                            a = 0
                        initializer['kaiming_uniform']['a'] = a

                    mode = kwargs.get('mode')
                    if mode is None:
                        mode = 'fan_in'
                    initializer['kaiming_uniform']['mode'] = mode

                    initializer['kaiming_uniform']['nonlinearity'] = self._activation
                else:
                    initializer = None
                    warnings.warn("It is recommended to use Kaiming uniform weight initialization only with ReLU and "
                                  "leaky ReLU")
            elif initializer in ['kaiming_normal', 'he_normal']:
                if self._activation in ['relu', 'leaky_relu']:
                    initializer = {'kaiming_normal': {}}

                    if self._activation == 'leaky_relu':
                        a = kwargs.get('a')
                        if a is None:
                            a = 0
                        initializer['kaiming_normal']['a'] = a

                    mode = kwargs.get('mode')
                    if mode is None:
                        mode = 'fan_in'
                    initializer['kaiming_normal']['mode'] = mode

                    initializer['kaiming_normal']['nonlinearity'] = self._activation
                else:
                    initializer = None
            elif initializer == 'orthogonal':
                gain = kwargs.get('gain')
                if gain is None:
                    gain = 1
                initializer = {'orthogonal': {'gain': gain}}
            elif initializer == 'sparse':
                sparsity = kwargs.get('sparsity')
                if sparsity is None:
                    sparsity = .1
                std = kwargs.get('std')
                if std is None:
                    std = .01
                initializer = {'sparse': {'sparsity': sparsity, 'std': std}}
            else:
                raise ValueError("Initializer not recognized")
        self._initializer = initializer

    @property
    def activation(self):
        """

        :return:
        """
        return self._activation

    @activation.setter
    def activation(self, arg):
        self._activation = arg

    @property
    def parent(self):
        """

        :return:
        """
        return self._parent

    @parent.setter
    def parent(self, arg):
        self._parent = arg

    @property
    def type(self):
        """

        :return:
        """
        return self._type

    @property
    def initializer(self):
        """

        :return:
        """
        if isinstance(self._initializer, str):
            return self._initializer
        elif isinstance(self._initializer, dict):
            return list(self._initializer)[0]
        return None

    # @initializer.setter
    def set_initializer(self, initializer, **kwargs):
        """

        :param initializer:
        :param kwargs:
        :return:
        """
        self._set_initializer(initializer, **kwargs)

    @staticmethod
    def dense(nodes, activation='linear', initializer=None, dropout=None, where=None, parent=None, **kwargs):
        """

        :param nodes:
        :param activation:
        :param initializer:
        :param dropout:
        :param where:
        :param parent:
        :param kwargs:
        :return:
        """
        if is_list_like(nodes):
            n_layers = len(nodes)

            if is_list_like(activation):
                if len(activation) != n_layers:
                    raise ValueError(f"Dimension mismatch between supplied nodes list of length {n_layers} "
                                     f"and supplied activation function list of length {len(activation)}")
            else:
                activation = n_layers * [activation]

            if is_list_like(initializer):
                if len(initializer) != n_layers:
                    raise ValueError(f"Dimension mismatch between supplied nodes list of length {n_layers} "
                                     f"and supplied initializer list of length {len(initializer)}")
            else:
                initializer = n_layers * [initializer]

            if dropout is not None:
                if where is None:
                    where = 'after'
                else:
                    where = where.lower()

                if is_list_like(dropout):
                    if where in ['before', 'after']:
                        if len(dropout) != n_layers:
                            raise ValueError(f"Dimension mismatch between supplied nodes list of length {n_layers} "
                                             f"and supplied dropout list of length {len(dropout)}")
                    else:
                        raise KeyError(f"Wrong keyword argument '{where}' for keyword 'where'")
                else:
                    dropout = n_layers * [dropout]

                layers = []
                for k in range(n_layers):
                    if where == 'after':
                        layers.append(Dense(nodes[k], activation=activation[k], initializer=initializer[k],
                                            parent=parent, **kwargs))
                        layers.append(Dropout(dropout[k], parent=parent))
                    else:
                        layers.append(Dropout(dropout[k], parent=parent))
                        layers.append(Dense(nodes[k], activation=activation[k], initializer=initializer[k],
                                            parent=parent, **kwargs))
            else:
                layers = [Dense(nodes[k], activation=activation[k], initializer=initializer[k], parent=parent, **kwargs)
                          for k in range(n_layers)]

            return layers
        else:
            if is_list_like(activation):
                raise ValueError(f"Dimension mismatch between supplied nodes and supplied activation function list of "
                                 f"length {len(activation)}")

            if is_list_like(initializer):
                raise ValueError(f"Dimension mismatch between supplied nodes and supplied initializer list of length "
                                 f"{len(initializer)}")

            if dropout is not None:
                if where is None:
                    where = 'after'
                else:
                    where = where.lower()

                if is_list_like(dropout):
                    raise ValueError(f"Dimension mismatch between supplied nodes and supplied activation function list"
                                     f" of length {len(dropout)}")

                if where == 'after':
                    return [Dense(nodes, activation=activation, initializer=initializer, parent=parent, **kwargs),
                            Dropout(dropout, parent=parent)]
                elif where == 'before':
                    return [Dropout(dropout, parent=parent),
                            Dense(nodes, activation=activation, initializer=initializer, parent=parent, **kwargs)]
                else:
                    raise KeyError(f"Wrong keyword argument '{where}' for keyword 'where'")
            else:
                return Dense(nodes, activation=activation, initializer=initializer, parent=parent, **kwargs)

    @staticmethod
    def dropout(rate, parent=None):
        """

        :param rate:
        :param parent:
        :return:
        """
        return Dropout(rate, parent=parent)

    @staticmethod
    def probabilistic(nodes, activation='probabilistic_relu', initializer=None, parent=None, **kwargs):
        """

        :param nodes:
        :param activation:
        :param initializer:
        :param parent:
        :param kwargs:
        :return:
        """
        return Probabilistic(nodes, activation=activation, initializer=initializer, parent=parent, **kwargs)


class Dense(Layer):
    """Dense layer"""
    def __init__(self, nodes, activation='linear', initializer=None, parent=None, **kwargs):
        """Constructor method"""
        super().__init__(nodes, activation=activation, initializer=initializer, parent=parent, **kwargs)

        self._type = 'dense'

    @property
    def nodes(self):
        """

        :return:
        """
        return self._nodes

    @nodes.setter
    def nodes(self, arg):
        self._nodes = arg


class Dropout(Layer):
    """Dropout layer

    :param rate: Dropout rate
    :type rate: float
    """
    def __init__(self, rate, parent=None):
        """Constructor method"""
        super().__init__(0, parent=parent)
        self._activation = None
        self._type = 'dropout'
        self._rate = rate

    @property
    def rate(self):
        """

        :return:
        """
        return self._rate

    @rate.setter
    def rate(self, arg):
        self._rate = arg

    def set_initializer(self, initializer, **kwargs):
        """

        :param initializer:
        :param kwargs:
        :return:
        """
        print("Initialization is not required for Dropout layers")


class Probabilistic(Dense):
    """Probabilistic layer --- to be used in Bayesian neural networks where the approximation is set to probabilistic
        backpropagation (pbp)
    """
    def __init__(self, nodes, activation='probabilistic_relu', prior=None, initializer=None, parent=None, **kwargs):
        """Constructor method"""
        if not activation.startswith('probabilistic_'):
            activation = 'probabilistic_' + activation
        super().__init__(nodes, activation=activation, initializer=initializer, parent=parent, **kwargs)

        self._type = 'probabilistic'
        self._prior = prior
        self._mean_tilde_over_var_tilde = None
        self._var_tilde_inverted = None
        self._alpha_tilde = None
        self._beta_tilde = None

    def _set_initializer(self, initializer, **kwargs):
        """

        :param initializer:
        :param kwargs:
        :return:
        """
        if initializer is not None:
            initializer = initializer.lower().replace(' ', '_')
            if self._prior is not None and self._prior.shape is not None and self._prior.rate is not None:
                if initializer == 'default':
                    def _initializer(n_inputs):
                        """

                        :param n_inputs:
                        :return:
                        """
                        mean = Prior.gaussian(mean=0., variance=1. / (n_inputs + 1)).sample((self._nodes, n_inputs + 1))
                        var = self._prior.rate / (self._prior.shape - 1.) * np.ones((self._nodes, n_inputs + 1))

                        self._mean_tilde_over_var_tilde = np.zeros((self._nodes, n_inputs + 1))
                        self._var_tilde_inverted = 1. / var
                        self._alpha_tilde = np.zeros((self._nodes, n_inputs + 1))
                        self._beta_tilde = np.zeros((self._nodes, n_inputs + 1))

                        return mean, var

                    initializer = _initializer
                else:
                    raise ValueError("Initializer not recognized")
            else:
                # TODO: Maybe differentiate between the 2 possibilities here and show different warnings
                warnings.warn(
                    "Prior of the weights of the probabilistic layer hasn't been set yet or was not set up correctly. "
                    "No weight initializer was set."
                )
        self._initializer = initializer

    @property
    def initializer(self):
        """

        :return:
        """
        if callable(self._initializer):
            return self._initializer
        return None

    @property
    def prior(self):
        """

        :return:
        """
        return self._prior

    @prior.setter
    def prior(self, arg):
        if isinstance(arg, Prior):
            self._prior = arg
            self.set_initializer('default')
        else:
            raise ValueError("Argument not recognized. Argument must be an object of the Prior classes.")

    def forward(self, x_mean, x_var, w_mean=None, w_var=None):
        """

        :param x_mean:
        :param x_var:
        :param w_mean:
        :param w_var:
        :return:
        """
        n_inputs = x_mean.size1()
        x_mean_plus_bias = ca.vertcat(x_mean, ca.SX.ones(1))  # equation 17
        x_var_plus_bias = ca.vertcat(x_var, ca.SX.zeros(1))  # equation 17

        if w_mean is None:
            pass
        else:
            # Add weights for bias term
            w_mean = ca.horzcat(w_mean, ca.SX.sym('b_mean', self._nodes))
        if w_var is None:
            pass
        else:
            # Add weights for bias term
            w_var = ca.horzcat(w_var, ca.SX.sym('b_var', self._nodes))

        # Linear propagation of Gaussian distribution (equations 13 and 14)
        mean = w_mean @ x_mean_plus_bias / ca.sqrt(n_inputs)
        var = (w_var @ x_var_plus_bias + w_mean ** 2 @ x_var_plus_bias + w_var @ x_mean_plus_bias ** 2) / n_inputs

        return ca.Function('layer', [x_mean, x_var, w_mean, w_var], [mean, var])

    def initialize_weights(self, n_inputs):
        """

        :param n_inputs:
        :return:
        """
        if self._initializer is None:
            raise RuntimeError("Initializer not set. Weights could not be initialized.")
        return self._initializer(n_inputs)

    def refine_prior(self, mean, var):
        """

        :param mean:
        :param var:
        :return:
        """
        n_outputs, n_inputs = mean.shape
        Z = Gaussian()
        for i, j in product(range(n_outputs), range(n_inputs)):
            var_inverted = 1. / var[i, j]
            mean_over_var = mean[i, j] / var[i, j]
            alpha = self._prior.shape
            beta = self._prior.rate

            var_cavity_inverted = var_inverted - self._var_tilde_inverted[i, j]
            mean_cavity_over_var_cavity = mean_over_var - self._mean_tilde_over_var_tilde[i, j]
            var_cavity = 1. / var_cavity_inverted
            mean_cavity = mean_cavity_over_var_cavity * var_cavity

            alpha_cavity = alpha - self._alpha_tilde[i, j] + 1.
            beta_cavity = beta - self._beta_tilde[i, j]

            if 0. < var_cavity < 1e6 and alpha_cavity > 1. and beta_cavity > 0.:
                logZ = Z.pdf(0., mean_cavity, var_cavity + beta_cavity / (alpha_cavity - 1.), log=True)
                logZ1 = Z.pdf(0., mean_cavity, var_cavity + beta_cavity / alpha_cavity, log=True)
                logZ2 = Z.pdf(0., mean_cavity, var_cavity + beta_cavity / (alpha_cavity + 1.), log=True)
                dlogZdm = -mean_cavity / (var_cavity + beta_cavity / (alpha_cavity - 1.))
                dlogZdv = .5 * mean_cavity ** 2. / var_cavity ** 2. - np.pi / var_cavity

                mean[i, j] = mean_cavity + var_cavity * dlogZdm
                var[i, j] = var_cavity - var_cavity ** 2. * (dlogZdm ** 2. - 2. * dlogZdv)

                alpha = float(1. / (np.exp(logZ + logZ2 - 2. * logZ1) * (alpha_cavity + 1.) / alpha_cavity - 1.))
                beta = float(1. / (np.exp(logZ2 - logZ1) * (alpha_cavity + 1.) / beta_cavity - np.exp(
                    logZ1 - logZ) * alpha_cavity / beta_cavity))
                self._prior.shape = alpha
                self._prior.rate = beta

                self._var_tilde_inverted[i, j] = var_inverted - var_cavity_inverted
                self._mean_tilde_over_var_tilde[i, j] = mean_over_var - mean_cavity_over_var_cavity

                self._alpha_tilde[i, j] = alpha - alpha_cavity + 1.
                self._beta_tilde[i, j] = beta - beta_cavity

        return mean, var


__all__ = [
    'Layer',
    'Dense',
    'Dropout',
    'Probabilistic'
]
