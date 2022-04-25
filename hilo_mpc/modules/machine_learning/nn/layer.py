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

import warnings

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


__all__ = [
    'Layer',
    'Dense',
    'Dropout'
]
