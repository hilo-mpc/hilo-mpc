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

from __future__ import annotations

import functools
from typing import Any, Callable, Optional, Sequence, TypeVar, Union, cast

import casadi as ca
import numpy as np

from ..modules.object import Object
from ..modules.machine_learning.base import LearningBase
from .probability import Prior
from .util import convert, is_list_like


ML = TypeVar('ML', bound=LearningBase)
Pr = TypeVar('Pr', bound=Prior)
Func = TypeVar('Func', bound=Callable[..., Any])
Numeric = Union[int, float]


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


def register_hyperparameters(obj: ML, ids: Sequence[str]):
    """

    :param obj:
    :param ids:
    :return:
    """
    for i in ids:
        HyperparameterManager.register(i, obj)


class HyperparameterManager:
    """
    :note: This class providing a decorator method is probably overkill. We could just as easily let
        register_hyperparameters() populate a global dictionary that is then used by the corresponding properties,
        where the update method of the values is called.
    """
    _mapping = {}

    @classmethod
    def register(cls, key: str, value: ML) -> None:
        """

        :param key:
        :param value:
        :return:
        """
        cls._mapping[key] = value

    @classmethod
    def observe(cls) -> Func:
        """

        :return:
        """
        def decorator(func: Func) -> Func:
            """

            :param func:
            :return:
            """
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                """

                :param args:
                :param kwargs:
                :return:
                """
                name = args[0].name  # First argument is always the Hyperparameter object
                idx = args[0].id
                if idx in cls._mapping:
                    cls._mapping[idx].update(name, *args[1:])
                return func(*args, **kwargs)
            return cast(Func, wrapper)
        return decorator


class Parameter(Object):
    """
    Base class for all parameters

    This class implements the hyperparameters and their behaviour, including the correct setting and getting of their
    values and bounds. A hyperparameter can be flagged as 'fixed' and thereby excluded from any changes during the
    optimization routines in the gaussian process regression.

    :param name: Name of the hyperparameter that serves as identifier in the kernel CasADi function and during updates
    :param value: Single values must always lie inside the bounds. Can contain a single item or multiple, e.g. a vector
        for length scales. Defaults to 1.
    :param prior:
    :param kwargs:
    """
    # TODO: Hyperparameter prior?
    def __init__(
            self,
            name: str,
            value: Union[Numeric, Sequence[Numeric]] = 1.,
            prior: Optional[str, list[str]] = None,
            **kwargs
    ) -> None:
        """Constructor method"""
        super().__init__(name=name)
        self._create_id()

        self._value = convert(value, ca.DM)
        self._shape = self._value.shape

        if prior is not None:
            if not is_list_like(prior):
                prior = [prior]

            n_val = self._value.numel()
            if len(prior) == 1 and n_val != 1:
                prior *= n_val

            if len(prior) > 1:
                raise NotImplementedError
            else:
                params = kwargs.get('prior_parameters')
                prior = self._check_prior(prior[0], params)
        self._prior = prior

        if '.' in name:
            _, name = name.split('.')
        self._SX = ca.SX.sym(name, *self._shape)
        self._MX = ca.MX.sym(name, *self._shape)

        fixed = kwargs.get('fixed')
        if fixed is None:
            fixed = False
        if fixed:
            if prior is not None and prior.name != 'Delta':
                raise RuntimeError("Keyword 'fixed' was set to True and a prior other than 'Delta' was supplied. "
                                   "Please specify one or the other, not both.")
            elif prior is None:
                self._prior = self._check_prior('Delta')

    def __repr__(self) -> str:
        """Representation method"""
        message = "Parameter("
        message += f"name='{self.name}', "
        message += f"value={self._value}, "
        message += f"fixed={self.fixed})"
        return message

    def _values_in_bounds(self) -> bool:  # pragma: no cover
        """

        :return:
        """
        if ca.mmin(ca.fmin(self._value, self._bounds[0])) < self._bounds[0]:
            return False
        if ca.mmax(ca.fmax(self._value, self._bounds[1])) > self._bounds[1]:
            return False
        return True

    @staticmethod
    def _check_prior(name: str, parameters: Optional[dict[str, Union[Numeric, Sequence[Numeric]]]] = None) -> Pr:
        """

        :param name:
        :param parameters:
        :return:
        """
        if parameters is None:
            parameters = {}

        id_ = name.replace("'", "").replace(' ', '_').lower()
        if id_ == 'gaussian':
            prior = Prior.gaussian(**parameters)
        elif id_ == 'laplace':
            prior = Prior.laplace(**parameters)
        elif id_ == 'students_t':
            prior = Prior.students_t(**parameters)
        elif id_ == 'delta':
            prior = Prior.delta()
        else:
            raise ValueError(f"Prior '{name}' not recognized")

        return prior

    @HyperparameterManager.observe()
    def _set_properties(self, value, bounds):
        """

        :param value:
        :param bounds:
        :return:
        """
        # TODO: Check whether it could happen that both value and bounds are not None
        if value is not None:
            # NOTE: This is somewhat of a hack, but since we use floats the optimization may return values just so
            #  slightly out of bounds
            # TODO: Use epsilon and absolute of difference?
            value = convert(value, ca.DM)

            bounds_supplied = True
            if bounds is None:
                bounds = (-ca.inf, ca.inf)
                bounds_supplied = False

            if bool(ca.logic_all(value >= bounds[0])) and bool(ca.logic_all(value <= bounds[1])):
                self._value = value
                new_shape = value.shape
                if new_shape != self._shape:
                    self._shape = new_shape
                    name = self.name
                    if '.' in name:
                        _, name = name.split('.')
                    self._SX = ca.SX.sym(name, *self._shape)
                    self._MX = ca.MX.sym(name, *self._shape)
            else:
                raise ValueError(f"Given {self.name} value is not in between bounds!")

            if not bounds_supplied:
                bounds = None

        if bounds is not None:  # pragma: no cover
            if bounds == 'fixed':
                self._fixed = True
                self._bounds = (0, ca.inf)
            elif isinstance(bounds, (int, float)):
                self._bounds = (bounds, ca.inf)
                self._fixed = False
            elif len(bounds) == 2:
                assert np.less(bounds[0], bounds[1]), "Lower bound not smaller than upper bound!"
                self._bounds = tuple(bounds)
                self._fixed = False
            elif len(bounds) == 1:
                self._bounds = (*bounds, ca.inf)
                self._fixed = False
            else:
                raise ValueError(f"Unsupported type {type(bounds).__name__} for bounds attribute")

    @property
    def value(self) -> ca.DM:
        """
        Single values must always lie inside the bounds. Can contain a single item or multiple, e.g. a vector for
        length scales

        :return:
        """
        return self._value

    @value.setter
    def value(self, value: list[Numeric]) -> None:
        self._set_properties(value, None)

    @property
    def prior(self) -> Pr:
        """

        :return:
        """
        return self._prior

    @prior.setter
    def prior(self, name: str) -> None:
        self._prior = self._check_prior(name)

    @property
    def bounds(self) -> (Numeric, Numeric):  # pragma: no cover
        """
        First item represents the lower bound, the second item the upper bound. No lower/upper bound is defined by
        an inf object.

        :return:
        """
        return self._bounds

    @bounds.setter
    def bounds(self, bounds: Union[str, Numeric, tuple[Numeric], tuple[Numeric, Numeric]]) -> None:  # pragma: no cover
        self._set_properties(self._log, bounds)

    @property
    def lower_bound(self) -> Numeric:  # pragma: no cover
        """
        Lower bound of the hyperparameter value

        :return:
        """
        return self.bounds[0]

    @lower_bound.setter
    def lower_bound(self, lower_bound: Numeric) -> None:  # pragma: no cover
        self.bounds = (lower_bound, self.upper_bound)

    @property
    def upper_bound(self) -> Numeric:  # pragma: no cover
        """
        Upper bound of the hyperparameter value

        :return:
        """
        return self.bounds[1]

    @upper_bound.setter
    def upper_bound(self, upper_bound: Numeric) -> None:  # pragma: no cover
        self.bounds = (self.lower_bound, upper_bound)

    @property
    def SX(self) -> ca.SX:
        """

        :return:
        """
        return self._SX

    @property
    def MX(self) -> ca.MX:  # pragma: no cover
        """

        :return:
        """
        return self._MX

    @property
    def fixed(self) -> bool:
        """

        :return:
        """
        if self._prior is not None:
            return self._prior.name == 'Delta'
        return False

    @fixed.setter
    def fixed(self, value: bool) -> None:
        if value:
            if self._prior is None:
                self._prior = Prior.delta()
            elif self._prior.name != 'Delta':
                raise RuntimeError("Hyperparameter cannot be set to fixed, since a hyperprior already exists")
        else:
            if self._prior is not None:
                if self._prior.name != 'Delta':
                    raise RuntimeError("Hyperparameter cannot be set to variable, since a hyperprior already exists")
                else:
                    self._prior = None

    def is_scalar(self) -> bool:
        """

        :return:
        """
        return self.SX.is_scalar()


class PositiveParameter(Parameter):
    """"""
    def __init__(
            self,
            name: str,
            value: Union[Numeric, Sequence[Numeric]] = 1.,
            prior: Optional[str, list[str]] = None,
            prior_mean: Optional[Numeric, list[Numeric]] = None,
            prior_variance: Optional[Numeric, list[Numeric]] = None,
            bounds: tuple[float, float] = (0., ca.inf),
            fixed: bool = False
    ) -> None:
        """Constructor method"""
        if bounds[0] < 0. or bounds[1] < 0.:  # pragma: no cover
            raise ValueError("Bounds need to be positive")
        super().__init__(name, value=value, prior=prior, prior_mean=prior_mean, prior_variance=prior_variance,
                         bounds=bounds, fixed=fixed)

        if ca.mmin(self._value) <= 0.:
            raise ValueError("Hyperparameters can only take positive values")

        self._log = ca.log(self._value)

    def _set_properties(self, value, bounds):
        """

        :param value:
        :param bounds:
        :return:
        """
        super()._set_properties(value, bounds)

        if value is not None:
            self._log = ca.log(value)

        if bounds is not None:  # pragma: no cover
            self._log_bounds = (ca.log(self._bounds[0]), ca.log(self._bounds[1]))

    @property
    def log(self) -> ca.DM:
        """

        :return:
        """
        return self._log

    @property
    def log_bounds(self) -> (Numeric, Numeric):  # pragma: no cover
        """

        :return:
        """
        return self._log_bounds


Param = TypeVar('Param', bound=Parameter)


class Hyperparameter:
    """"""
    def __new__(cls, *args, **kwargs) -> Param:
        """Creator method"""
        positive = kwargs.pop('positive', None)
        if positive is None:
            positive = True

        if positive:
            return PositiveParameter(*args, **kwargs)
        else:
            return Parameter(*args, **kwargs)


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
