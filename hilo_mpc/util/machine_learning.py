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

import casadi as ca


class Activation:
    """"""
    def __new__(cls, *args, **kwargs):
        activation = super().__new__(cls)
        activation.__init__(*args)
        return activation()

    def __init__(self, name):
        self._name = name

    def __call__(self, *args, **kwargs):
        return {
            'sigmoid': self.sigmoid,
            'tanh': ca.tanh,
            'relu': self.rectifier,
            'softplus': self.soft_plus,
            'softmax': self.soft_max,
            'linear': self.linear,
            'scale': self.scale
        }[self._name]

    @staticmethod
    def sigmoid(x):
        """

        :param x:
        :return:
        """
        return 1 / (1 + ca.exp(-x))

    @staticmethod
    def rectifier(x):
        """

        :param x:
        :return:
        """
        return ca.fmax(0, x)

    @staticmethod
    def soft_plus(x):
        """

        :param x:
        :return:
        """
        return ca.log(1 + ca.exp(x))

    @staticmethod
    def soft_max(x):
        """

        :param x:
        :return:
        """
        x_exp = ca.exp(x)
        return x_exp / ca.sum1(x_exp)

    @staticmethod
    def linear(x):
        """

        :param x:
        :return:
        """
        return x

    @staticmethod
    def scale(x):
        """

        :param x:
        :return:
        """
        return 1 / (1 - x)


def net_to_casadi_graph(net, x, layers, **kwargs):
    """

    :param net:
    :param x:
    :param layers:
    :param kwargs:
    :return:
    """
    input_scaling = kwargs.get('input_scaling')
    output_scaling = kwargs.get('output_scaling')
    h = x
    graph = []
    # TODO: What if bias was deactivated?
    if isinstance(net, dict):
        weights = net['weights']
        bias = net['bias']
    else:
        weights, bias = net.get_weights_and_bias()

    if input_scaling is not None:
        h -= input_scaling.mean_
        h /= input_scaling.scale_

    # NOTE: We could also use a counter (initialized with 0) here, that increases by 1 every time we come across a
    #  Dropout layer and then subtract the counter from 'k' in every index.
    layers = [layer for layer in layers if layer.type.lower() != 'dropout']
    for k, layer in enumerate(layers):
        if layer.type.lower() == 'dense':
            if not graph:
                y = weights[k] @ h + bias[k]
            else:
                y = weights[k] @ graph[-1] + bias[k]
            graph.append(Activation(layer.activation)(y))
    # NOTE: Right now the output layer is assumed to be linear and cannot be changed
    if not graph:
        y = weights[-1] @ h + bias[-1]
    else:
        y = weights[-1] @ graph[-1] + bias[-1]

    if output_scaling is not None:
        y *= output_scaling.scale_
        y += output_scaling.mean_

    graph = [y] + graph

    hidden = []
    ct = 1
    for layer in layers:
        if layer.type.lower() != 'dropout':
            hidden.append("layer_" + str(ct))
            ct += 1

    return ca.Function('neural_network',
                       [x],
                       graph,
                       ["features"],
                       ["labels"] + hidden)
