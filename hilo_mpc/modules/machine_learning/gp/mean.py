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

from abc import ABCMeta, abstractmethod
import copy
from typing import Optional, Sequence, TypeVar, Union

import casadi as ca
import numpy as np

from ....util.machine_learning import Parameter, Hyperparameter
from ....util.util import is_list_like


IntArray = Union[int, Sequence[int]]
Numeric = Union[int, float]
Coeff = Union[Numeric, Sequence[Numeric], np.ndarray]
Mu = TypeVar('Mu', bound='Mean')
Array = TypeVar('Array', ca.SX, ca.MX, np.ndarray)
Param = TypeVar('Param', bound=Parameter)


class Mean(metaclass=ABCMeta):
    """
    Mean function base class

    :param active_dims:
    :type: active_dims: int, list of int, optional
    """
    def __init__(self, active_dims: Optional[IntArray] = None) -> None:
        """Constructor method"""
        self.active_dims = active_dims

    def __add__(self, other: Mu) -> 'Sum':
        """Addition method"""
        return Sum(self, other)

    def __mul__(self, other: Union[Mu, Numeric]) -> Union['Product', 'Scale']:
        """Multiplication method"""
        if isinstance(other, (int, float)):
            return Scale(self, other)
        else:
            return Product(self, other)

    def __pow__(self, power: Numeric, modulo: Optional[int] = None) -> 'Power':
        """Power method"""
        return Power(self, power)

    def __radd__(self, other: Mu) -> 'Sum':
        """Addition method (from the right)"""
        return Sum(other, self)

    def __rmul__(self, other: Union[Mu, Numeric]) -> Union['Product', 'Scale']:
        """Multiplication method (from the right)"""
        if isinstance(other, (int, float)):
            return Scale(self, other)
        else:
            return Product(other, self)

    def __str__(self) -> str:
        """String representation method"""
        message = "Mean with \n"
        for attribute, value in self.__dict__.items():
            if attribute[0] != '_':
                message += f"\t {attribute}: {value} \n"
        return message

    def __call__(self, X: Array) -> Array:
        """Calling method"""
        is_symbolic = False
        if isinstance(X, (ca.SX, ca.MX)):
            is_symbolic = True

        dimension_input_space = X.shape[0]
        if self.active_dims is None:
            active_dims = np.arange(dimension_input_space, dtype=np.int_)
        else:
            active_dims = np.atleast_1d(np.asarray(self.active_dims, dtype=np.int_))
        x = ca.SX.sym('x', dimension_input_space, 1)

        mean_function = self.get_mean_function(x, active_dims)

        if is_symbolic:
            hyperparameters = {parameter.name: parameter.SX for parameter in self.hyperparameters}
        else:
            hyperparameters = {parameter.name: parameter.value for parameter in self.hyperparameters}

        mean = mean_function(x=X, **hyperparameters)['mean']

        if is_symbolic:
            return mean
        else:
            return mean.full()

    @property
    def hyperparameters(self) -> list[Param]:
        """

        :return:
        """
        hyperparameters = []
        for attribute in self.__dict__.values():
            if isinstance(attribute, Parameter):
                hyperparameters.append(attribute)
        return hyperparameters

    @property
    def hyperparameter_names(self) -> list[str]:
        """

        :return:
        """
        names = []
        for attribute in self.__dict__.values():
            if isinstance(attribute, Parameter):
                names.append(attribute.name)
        return names

    @abstractmethod
    def get_mean_function(self, x: ca.SX, active_dims: np.ndarray) -> ca.Function:
        """

        :param x:
        :param active_dims:
        :return:
        """
        pass

    def is_bounded(self) -> bool:  # pragma: no cover
        """

        :return:
        """
        for parameter in self.hyperparameters:
            bounds = parameter.bounds
            if bounds[0] != -ca.inf:
                return True
            if bounds[1] != ca.inf:
                return True
        return False

    def update(self, *args) -> None:
        """

        :param args:
        :return:
        """
        self.parent.update(*args)

    @staticmethod
    def constant(
            bias: Numeric = 1.,
            hyperprior: Optional[str] = None,
            **kwargs
    ) -> Mu:
        """

        :param bias:
        :param hyperprior:
        :param kwargs:
        :return:
        """
        return ConstantMean(bias=bias, hyperprior=hyperprior, **kwargs)

    @staticmethod
    def zero() -> Mu:
        """

        :return:
        """
        return ZeroMean()

    @staticmethod
    def one() -> Mu:
        """

        :return:
        """
        return OneMean()

    @staticmethod
    def polynomial(
            degree: int,
            active_dims: Optional[IntArray] = None,
            coefficient: Coeff = 1.,
            offset: Numeric = 1.,
            hyperprior: Optional[Union[str, dict[str, Union[str, list[str]]]]] = None,
            **kwargs
    ) -> Mu:
        """

        :param degree:
        :param active_dims:
        :param coefficient:
        :param offset:
        :param hyperprior:
        :param kwargs:
        :return:
        """
        return PolynomialMean(degree, active_dims=active_dims, coefficient=coefficient, offset=offset,
                              hyperprior=hyperprior, **kwargs)

    @staticmethod
    def linear(
            active_dims: Optional[IntArray] = None,
            coefficient: Coeff = 1.,
            hyperprior: Optional[Union[str, dict[str, Union[str, list[str]]]]] = None,
            **kwargs
    ) -> Mu:
        """

        :param active_dims:
        :param coefficient:
        :param hyperprior:
        :param kwargs:
        :return:
        """
        return LinearMean(active_dims=active_dims, coefficient=coefficient, hyperprior=hyperprior, **kwargs)


class ConstantMean(Mean):
    """
    Constant mean function

    :param bias:
    :type bias:
    :param hyperprior:
    :type hyperprior:
    :param kwargs:
    """
    acronym = "Const"

    def __init__(
            self,
            bias: Numeric = 1.,
            hyperprior: Optional[str] = None,
            **kwargs
    ) -> None:
        """Constructor method"""
        super().__init__()

        hyper_kwargs = {}
        if hyperprior is not None:
            hyper_kwargs['prior'] = hyperprior
        hyperprior_parameters = kwargs.get('hyperprior_parameters')
        if hyperprior_parameters is not None:
            hyper_kwargs['prior_parameters'] = hyperprior_parameters
        bounds = kwargs.get('bounds')
        if bounds is not None:  # pragma: no cover
            bias_bounds = bounds.get('bias')
            if bias_bounds is not None:
                if bias_bounds == 'fixed':
                    hyper_kwargs['fixed'] = True
                else:
                    hyper_kwargs['bounds'] = bias_bounds
        self.bias = Hyperparameter(f'{self.acronym}.bias', positive=False, value=bias, **hyper_kwargs)

    def get_mean_function(self, x: ca.SX, active_dims: np.ndarray) -> ca.Function:
        """

        :param x:
        :param active_dims:
        :return:
        """
        bias = self.bias
        if isinstance(bias, Parameter):
            bias_arg_name = [bias.name]
            bias = bias.SX
            bias_arg = [bias]
        else:
            bias_arg = []
            bias_arg_name = []

        mean_function = ca.Function(
            'mean',
            [x] + bias_arg,
            [bias],
            ['x'] + bias_arg_name,
            ['mean']
        )

        return mean_function


class ZeroMean(ConstantMean):
    """Zero mean function"""
    acronym = "Zero"

    def __init__(self):
        """Constructor method"""
        super().__init__()

        self.bias = 0.


class OneMean(ConstantMean):
    """One mean function"""
    acronym = "One"

    def __init__(self):
        """Constructor method"""
        super().__init__()

        self.bias = 1.


class PolynomialMean(Mean):
    """
    Polynomial mean function

    :param degree:
    :type degree:
    :param active_dims:
    :type active_dims:
    :param coefficient:
    :type coefficient:
    :param offset:
    :type offset:
    :param hyperprior:
    :type hyperprior:
    :param kwargs:
    """
    acronym = "Poly"

    def __init__(
            self,
            degree: int,
            active_dims: Optional[IntArray] = None,
            coefficient: Coeff = 1.,
            offset: Numeric = 1.,
            hyperprior: Optional[Union[str, dict[str, Union[str, list[str]]]]] = None,
            **kwargs
    ) -> None:
        super().__init__(active_dims=active_dims)

        bounds = kwargs.get('bounds')
        if bounds is not None:  # pragma: no cover
            coefficient_bounds = bounds.get('coefficient')
            offset_bounds = bounds.get('offset')
        else:  # pragma: no cover
            coefficient_bounds, offset_bounds = None, None

        if active_dims is not None and is_list_like(coefficient):
            if len(active_dims) != len(coefficient):
                raise ValueError(f"Dimension mismatch between 'active_dims' ({len(active_dims)}) and the number of "
                                 f"coefficients ({len(coefficient)})")

        if hyperprior is not None:
            if not isinstance(hyperprior, dict):
                if isinstance(hyperprior, str):
                    hyperprior = {'coefficient': hyperprior, 'offset': hyperprior}
                else:
                    raise TypeError(f"Wrong type '{type(hyperprior).__name__}' for keyword argument 'hyperprior'")
        else:
            hyperprior = {'coefficient': None, 'offset': None}

        hyperprior_parameters = kwargs.get('hyperprior_parameters')
        if hyperprior_parameters is None:
            hyperprior_parameters = {}

        hyper_kwargs = {}
        if 'coefficient' in hyperprior:
            hyper_kwargs['prior'] = hyperprior.get('coefficient')
        if 'coefficient' in hyperprior_parameters:
            hyper_kwargs['prior_parameters'] = hyperprior_parameters.get('coefficient')
        if coefficient_bounds is not None:  # pragma: no cover
            if coefficient_bounds == 'fixed':
                hyper_kwargs['fixed'] = True
            else:
                hyper_kwargs['bounds'] = coefficient_bounds
        self.coefficient = Hyperparameter(f'{self.acronym}.coefficient', positive=False, value=coefficient,
                                          **hyper_kwargs)

        hyper_kwargs = {}
        if 'offset' in hyperprior:
            hyper_kwargs['prior'] = hyperprior.get('offset')
        if 'offset' in hyperprior_parameters:
            hyper_kwargs['prior_parameters'] = hyperprior_parameters.get('offset')
        if offset_bounds is not None:  # pragma: no cover
            if offset_bounds == 'fixed':
                hyper_kwargs['fixed'] = True
            else:
                hyper_kwargs['bounds'] = offset_bounds
        self.offset = Hyperparameter(f'{self.acronym}.offset', positive=False, value=offset, **hyper_kwargs)

        self._p = degree

    @property
    def degree(self) -> int:
        """

        :return:
        """
        return self._p

    @degree.setter
    def degree(self, value: int):
        self._p = value

    def get_mean_function(self, x: ca.SX, active_dims: np.ndarray) -> ca.Function:
        """

        :param x:
        :param active_dims:
        :return:
        """
        p = self._p

        coefficient = self.coefficient.SX
        M = self.get_parameterized_coefficients(active_dims.size, coefficient)

        offset = self.offset
        if isinstance(offset, Parameter):
            offset_arg_name = [offset.name]
            offset = offset.SX
            offset_arg = [offset]
        else:
            offset_arg = []
            offset_arg_name = []

        mean_function = ca.Function(
            'mean',
            [x, coefficient] + offset_arg,
            [(M(coefficient).T @ x[active_dims] + offset) ** p],
            ['x', self.coefficient.name] + offset_arg_name,
            ['mean']
        )

        return mean_function

    @staticmethod
    def get_parameterized_coefficients(dimension_input_space: int, coefficient: ca.SX) -> ca.Function:
        """

        :param dimension_input_space:
        :param coefficient:
        :return:
        """
        if coefficient.is_scalar():
            M = ca.Function('M',
                            [coefficient],
                            [coefficient * ca.SX.ones(dimension_input_space)])
        elif coefficient.numel() == dimension_input_space:
            M = ca.Function('M', [coefficient], [coefficient])
        else:
            raise ValueError("Coefficient vector dimension does not equal input space dimension.")
        return M


class LinearMean(PolynomialMean):
    """
    Linear mean function

    :param active_dims:
    :type active_dims:
    :param coefficient:
    :type coefficient:
    :param hyperprior:
    :type hyperprior:
    :param kwargs:
    """
    acronym = "Lin"

    def __init__(
            self,
            active_dims: Optional[IntArray] = None,
            coefficient: Coeff = 1.,
            hyperprior: Optional[Union[str, dict[str, Union[str, list[str]]]]] = None,
            **kwargs
    ) -> None:
        """Constructor method"""
        super().__init__(1, active_dims=active_dims, coefficient=coefficient, hyperprior=hyperprior, **kwargs)

        self.offset = 0.


class MeanOperator(Mean, metaclass=ABCMeta):
    """
    Mean function operator base class

    :param mean_1:
    :type mean_1:
    :param mean_2:
    :type mean_2:
    """
    def __init__(self, mean_1: Mu, mean_2: Optional[Mu] = None) -> None:
        """Constructor method"""
        super().__init__()

        self.mean_1 = copy.deepcopy(mean_1)
        if mean_2 is not None:
            self.mean_2 = copy.deepcopy(mean_2)
        else:
            self.mean_2 = mean_2
        self.disambiguate_hyperparameter_names()

    @property
    def hyperparameters(self) -> list[Param]:
        """

        :return:
        """
        if self.mean_2 is not None:
            return self.mean_1.hyperparameters + self.mean_2.hyperparameters
        else:
            return self.mean_1.hyperparameters

    @property
    def hyperparameter_names(self) -> list[str]:
        """

        :return:
        """
        if self.mean_2 is not None:
            return self.mean_1.hyperparameter_names + self.mean_2.hyperparameter_names
        else:
            return self.mean_1.hyperparameter_names

    def disambiguate_hyperparameter_names(self) -> None:
        """

        :return:
        """
        if self.mean_2 is not None:
            mean_1_has_acronym = hasattr(self.mean_1, 'acronym')
            mean_2_has_acronym = hasattr(self.mean_2, 'acronym')
            if mean_1_has_acronym and mean_2_has_acronym:
                if self.mean_1.acronym == self.mean_2.acronym:
                    for parameter in self.mean_1.hyperparameters:
                        old_name = parameter.name
                        if '.' in old_name:
                            new_name = self.mean_1.acronym + '_1.' + old_name.split('.')[1]
                        else:
                            new_name = self.mean_1.acronym + '_1.' + old_name
                        parameter.name = new_name
                    for parameter in self.mean_2.hyperparameters:
                        old_name = parameter.name
                        if '.' in old_name:
                            new_name = self.mean_2.acronym + '_2.' + old_name.split('.')[1]
                        else:
                            new_name = self.mean_2.acronym + '_2.' + old_name
                        parameter.name = new_name
            elif mean_1_has_acronym:
                mean_2_acronyms = dict.fromkeys([name.split('.')[0] for name in self.mean_2.hyperparameter_names])
                has_mean_1_acronym = [self.mean_1.acronym in name for name in mean_2_acronyms]
                ct = 1
                for k, val in enumerate(mean_2_acronyms.keys()):
                    if has_mean_1_acronym[k]:
                        ct += 1
                        mean_2_acronyms[val] = ct
                for parameter in self.mean_1.hyperparameters:
                    old_name = parameter.name
                    if '.' in old_name:
                        new_name = self.mean_1.acronym + '_1.' + old_name.split('.')[1]
                    else:
                        new_name = self.mean_1.acronym + '_1.' + old_name
                    parameter.name = new_name
                for parameter in self.mean_2.hyperparameters:
                    args = parameter.name.split('.')
                    old_acronym = args[0]
                    val = mean_2_acronyms.get(old_acronym)
                    if val is not None:
                        new_acronym = old_acronym.split('_')[0] + '_' + str(val)
                        parameter.name = new_acronym + '.' + args[1]
            elif mean_2_has_acronym:
                mean_1_acronyms = set([name.split('.')[0] for name in self.mean_1.hyperparameter_names])
                has_mean_2_acronym = [self.mean_2.acronym in name for name in mean_1_acronyms]
                new_index = has_mean_2_acronym.count(True) + 1
                for parameter in self.mean_2.hyperparameters:
                    old_name = parameter.name
                    if '.' in old_name:
                        new_name = self.mean_2.acronym + '_' + str(new_index) + '.' + old_name.split('.')[1]
                    else:
                        new_name = self.mean_2.acronym + '_' + str(new_index) + '.' + old_name
                    parameter.name = new_name
            else:
                mean_1_acronyms = set([name.split('.')[0] for name in self.mean_1.hyperparameter_names])
                mean_1_acronyms = [name.split('_')[0] for name in mean_1_acronyms]
                mean_2_acronyms = dict.fromkeys([name.split('.')[0] for name in self.mean_2.hyperparameter_names])
                mean_2_counter = {}
                for k, val in enumerate(mean_2_acronyms.keys()):
                    args = val.split('_')
                    acronym = args[0]
                    count = mean_1_acronyms.count(acronym)
                    mean_2_count = mean_2_counter.get(acronym)
                    if mean_2_count is not None:
                        count += mean_2_count
                        mean_2_counter[acronym] += 1
                    else:
                        mean_2_counter[acronym] = 1
                    if count != 0:
                        mean_2_acronyms[val] = count + 1
                for parameter in self.mean_2.hyperparameters:
                    args = parameter.name.split('.')
                    old_acronym = args[0]
                    val = mean_2_acronyms.get(old_acronym)
                    if val is not None:
                        new_acronym = old_acronym.split('_')[0] + '_' + str(val)
                        parameter.name = new_acronym + '.' + args[1]


class Scale(MeanOperator):
    """
    Scale operator for mean functions

    :param mean:
    :type mean:
    :param scale:
    :type scale:
    """
    def __init__(self, mean: Mu, scale: Numeric) -> None:
        """Constructor method"""
        super().__init__(mean, mean_2=None)

        self.scale = scale

    def get_mean_function(self, x: ca.SX, active_dims: np.ndarray) -> ca.Function:
        """

        :param x:
        :param active_dims:
        :return:
        """
        mean = self.mean_1(x)
        scale = self.scale

        hyperparameters = [parameter.SX for parameter in self.hyperparameters]
        hyperparameter_names = [parameter.name for parameter in self.hyperparameters]

        mean_scale = ca.Function(
            'mean',
            [x, *hyperparameters],
            [scale * mean],
            ['x', *hyperparameter_names],
            ['mean']
        )

        return mean_scale


class Sum(MeanOperator):
    """
    Sum operator for mean functions

    :param mean_1:
    :type mean_1:
    :param mean_2:
    :type mean_2:
    """
    def get_mean_function(self, x: ca.SX, active_dims: np.ndarray) -> ca.Function:
        """

        :param x:
        :param active_dims:
        :return:
        """
        mean_1 = self.mean_1(x)
        mean_2 = self.mean_2(x)

        hyperparameters = [parameter.SX for parameter in self.hyperparameters]
        hyperparameter_names = [parameter.name for parameter in self.hyperparameters]

        mean_sum = ca.Function(
            'mean',
            [x, *hyperparameters],
            [mean_1 + mean_2],
            ['x', *hyperparameter_names],
            ['mean']
        )

        return mean_sum


class Product(MeanOperator):
    """
    Product operator for mean functions

    :param mean_1:
    :type mean_1:
    :param mean_2:
    :type mean_2:
    """
    def get_mean_function(self, x: ca.SX, active_dims: np.ndarray) -> ca.Function:
        """

        :param x:
        :param active_dims:
        :return:
        """
        mean_1 = self.mean_1(x)
        mean_2 = self.mean_2(x)

        hyperparameters = [parameter.SX for parameter in self.hyperparameters]
        hyperparameter_names = [parameter.name for parameter in self.hyperparameters]

        mean_prod = ca.Function(
            'mean',
            [x, *hyperparameters],
            [mean_1 * mean_2],
            ['x', *hyperparameter_names],
            ['mean']
        )

        return mean_prod


class Power(MeanOperator):
    """
    Power operator for mean functions

    :param mean:
    :type mean:
    :param power:
    :type power:
    """
    def __init__(self, mean: Mu, power: Numeric) -> None:
        """Constructor method"""
        super().__init__(mean, mean_2=None)

        self.power = power

    def get_mean_function(self, x: ca.SX, active_dims: np.ndarray) -> ca.Function:
        """

        :param x:
        :param active_dims:
        :return:
        """
        mean = self.mean_1(x)
        power = self.power

        hyperparameters = [parameter.SX for parameter in self.hyperparameters]
        hyperparameter_names = [parameter.name for parameter in self.hyperparameters]

        mean_pow = ca.Function(
            'mean',
            [x, *hyperparameters],
            [mean ** power],
            ['x', *hyperparameter_names],
            ['mean']
        )

        return mean_pow


class Warp(MeanOperator):
    """"""


__all__ = [
    'Mean',
    'ConstantMean',
    'ZeroMean',
    'OneMean',
    'PolynomialMean',
    'LinearMean'
]
