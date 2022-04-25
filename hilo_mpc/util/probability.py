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

from abc import ABCMeta, abstractmethod
from typing import Any, Optional, Sequence, TypeVar, Union

import casadi as ca
import numpy as np
from scipy.special import gammaln


Numeric = Union[int, float]
Pr = TypeVar('Pr', bound='Prior')


class Distribution(metaclass=ABCMeta):
    """"""
    def __init__(self):
        """Constructor method"""
        self._log_function = None
        self._initialize()

    def __call__(self, *args, **kwargs) -> Union[ca.SX, ca.MX, ca.DM]:
        """Calling method"""
        log = kwargs.get('log')
        if log is None:
            log = False

        if log:
            return self._log_function(*args)

    @abstractmethod
    def _initialize(self) -> None:
        """

        :return:
        """
        pass


class Gaussian(Distribution):
    """"""
    def _initialize(self) -> None:
        """

        :return:
        """
        # TODO: Multivariate extension
        y = ca.SX.sym('y')
        mu = ca.SX.sym('mu')
        var = ca.SX.sym('var')

        log_pdf = -(y - mu) ** 2. / (2. * var) - ca.log(2. * ca.pi * var) / 2.

        self._log_function = ca.Function('log_pdf',
                                         [y, mu, var],
                                         [log_pdf])


class Laplace(Distribution):
    """"""
    def _initialize(self) -> None:
        """

        :return:
        """
        # TODO: Multivariate extension
        y = ca.SX.sym('y')
        mu = ca.SX.sym('mu')
        var = ca.SX.sym('var')

        b = ca.sqrt(var / 2.)
        log_pdf = -ca.fabs(y - mu) / b - ca.log(2. * b)

        self._log_function = ca.Function('log_pdf',
                                         [y, mu, var],
                                         [log_pdf])


class StudentsT(Distribution):
    """"""
    def _initialize(self) -> None:
        """

        :return:
        """
        # TODO: Multivariate extension
        y = ca.SX.sym('y')
        mu = ca.SX.sym('mu')
        var = ca.SX.sym('var')
        nu = ca.SX.sym('nu')

        log_Z = gammaln((nu + 1.) / 2.) - gammaln(nu / 2.) - ca.log(var * (nu - 2.) * ca.pi) / 2.
        log_pdf = log_Z - (nu + 1.) / 2. * ca.log(1. + (y - mu) ** 2. / (var * (nu - 2.)))

        self._log_function = ca.Function('log_pdf',
                                         [y, mu, var, nu],
                                         [log_pdf])


class Prior(metaclass=ABCMeta):
    """"""
    def __init__(self, name: str) -> None:
        """Constructor method"""
        self._name = name
        self._pdf = None

    def __call__(self, *args, **kwargs) -> Union[ca.SX, ca.MX, ca.DM]:
        """Calling method"""
        x = args[0]
        p = self._get_parameter_values()
        return self._pdf(x, *p, **kwargs)

    @staticmethod
    def _check_dimensionality(parameter: Any, name: str) -> Numeric:
        """

        :param parameter:
        :param name:
        :return:
        """
        if parameter is not None:
            if not isinstance(parameter, (int, float)):
                if isinstance(parameter, (list, tuple, np.ndarray)):
                    if len(parameter) == 1:
                        parameter = parameter[0]
                    else:
                        raise ValueError(f"Dimension mismatch. Expected 1, got {len(parameter)}.")
                else:
                    raise TypeError(f"Unexpected type '{type(parameter).__name__}' for keyword argument '{name}'")

        return parameter

    @abstractmethod
    def _get_parameter_values(self) -> (Union[Numeric, Sequence[Numeric]], ...):
        """

        :return:
        """
        pass

    @property
    def name(self) -> str:
        """

        :return:
        """
        return self._name

    @staticmethod
    def gaussian(
            mean: Optional[Union[Numeric, Sequence[Numeric]]] = None,
            variance: Optional[Union[Numeric, Sequence[Numeric]]] = None
    ) -> Pr:
        """

        :param mean:
        :param variance:
        :return:
        """
        return GaussianPrior(mean=mean, variance=variance)

    @staticmethod
    def laplace(
            mean: Optional[Union[Numeric, Sequence[Numeric]]] = None,
            variance: Optional[Union[Numeric, Sequence[Numeric]]] = None
    ) -> Pr:
        """

        :param mean:
        :param variance:
        :return:
        """
        return LaplacePrior(mean=mean, variance=variance)

    @staticmethod
    def students_t(
            mean: Optional[Union[Numeric, Sequence[Numeric]]] = None,
            variance: Optional[Union[Numeric, Sequence[Numeric]]] = None,
            nu: Optional[Union[Numeric, Sequence[Numeric]]] = None
    ) -> Pr:
        """

        :param mean:
        :param variance:
        :param nu:
        :return:
        """
        return StudentsTPrior(mean=mean, variance=variance, nu=nu)

    @staticmethod
    def delta() -> Pr:
        """

        :return:
        """
        return DeltaPrior()


class _MeanVariancePrior(Prior):
    """"""
    def __init__(
            self,
            name: str,
            mean: Optional[Union[Numeric, Sequence[Numeric]]] = None,
            variance: Optional[Union[Numeric, Sequence[Numeric]]] = None
    ) -> None:
        """Constructor method"""
        super().__init__(name)

        self._mean = self._check_dimensionality(mean, 'mean')
        self._variance = self._check_dimensionality(variance, 'variance')

    def _get_parameter_values(self) -> (Union[Numeric, Sequence[Numeric]], ...):
        """

        :return:
        """
        return self._mean, self._variance

    @property
    def mean(self) -> Optional[Numeric]:
        """

        :return:
        """
        return self._mean

    @mean.setter
    def mean(self, value: Numeric) -> None:
        self._mean = self._check_dimensionality(value, 'mean')

    @property
    def variance(self) -> Optional[Numeric]:
        """

        :return:
        """
        return self._variance

    @variance.setter
    def variance(self, value: Numeric) -> None:
        self._variance = self._check_dimensionality(value, 'variance')


class GaussianPrior(_MeanVariancePrior):
    """"""
    def __init__(
            self,
            mean: Optional[Union[Numeric, Sequence[Numeric]]] = None,
            variance: Optional[Union[Numeric, Sequence[Numeric]]] = None
    ) -> None:
        """Constructor method"""
        super().__init__('Gaussian', mean=mean, variance=variance)

        self._pdf = Gaussian()


class LaplacePrior(_MeanVariancePrior):
    """"""
    def __init__(
            self,
            mean: Optional[Union[Numeric, Sequence[Numeric]]] = None,
            variance: Optional[Union[Numeric, Sequence[Numeric]]] = None
    ) -> None:
        """Constructor method"""
        super().__init__('Laplace', mean=mean, variance=variance)

        self._pdf = Laplace()


class StudentsTPrior(_MeanVariancePrior):
    """"""
    def __init__(
            self,
            mean: Optional[Union[Numeric, Sequence[Numeric]]] = None,
            variance: Optional[Union[Numeric, Sequence[Numeric]]] = None,
            nu: Optional[Union[Numeric, Sequence[Numeric]]] = None
    ) -> None:
        """Constructor method"""
        super().__init__('Students_T', mean=mean, variance=variance)

        self._nu = self._check_dimensionality(nu, 'nu')
        self._pdf = StudentsT()

    def _get_parameter_values(self) -> (Union[Numeric, Sequence[Numeric]], ...):
        """

        :return:
        """
        return super()._get_parameter_values(), self._nu

    @property
    def nu(self) -> Optional[Numeric]:
        """

        :return:
        """
        return self._nu

    @nu.setter
    def nu(self, value: Numeric) -> None:
        self._nu = self._check_dimensionality(value, 'nu')


class DeltaPrior(Prior):
    """"""
    def __init__(self):
        """Constructor method"""
        super().__init__('Delta')

    def _get_parameter_values(self) -> (Union[Numeric, Sequence[Numeric]], ...):
        """

        :return:
        """
        pass


__all__ = [
    'Prior',
    'GaussianPrior',
    'LaplacePrior',
    'StudentsTPrior',
    'DeltaPrior'
]
