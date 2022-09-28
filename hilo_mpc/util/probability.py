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
from scipy.special import gamma, gammaln


Numeric = Union[int, float]
Pr = TypeVar('Pr', bound='Prior')


class Distribution(metaclass=ABCMeta):
    """"""
    def __init__(self):
        """Constructor method"""
        self._pdf_function = None
        self._log_pdf_function = None
        self._cdf_function = None
        self._initialize()

    def __call__(self, *args, **kwargs) -> Union[ca.SX, ca.MX, ca.DM]:
        """Calling method"""
        which = kwargs.get('which')
        if which is None:
            which = 'pdf'

        log = kwargs.get('log')
        if log is None:
            log = False

        if which == 'pdf':
            if log:
                return self.probability_density_function(*args, log=True)
            else:
                return self.probability_density_function(*args, log=False)
        elif which == 'cdf':
            return self.cumulative_distribution_function(*args)
        else:
            raise ValueError(f"Key word 'which={which}' not recognized")

    @abstractmethod
    def _initialize(self) -> None:
        """

        :return:
        """
        pass

    def probability_density_function(self, *args, log: bool = False) -> Union[ca.SX, ca.MX, ca.DM]:
        """

        :param args:
        :param log:
        :return:
        """
        if log:
            return self._log_pdf_function(*args)
        else:
            return self._pdf_function(*args)

    pdf = probability_density_function

    def cumulative_distribution_function(self, *args) -> Union[ca.SX, ca.MX, ca.DM]:
        """

        :param args:
        :return:
        """
        return self._cdf_function(*args)

    cdf = cumulative_distribution_function


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

        pdf = 1. / ca.sqrt(2. * ca.pi * var) * ca.exp(-.5 * (y - mu) ** 2. / var)
        log_pdf = -(y - mu) ** 2. / (2. * var) - ca.log(2. * ca.pi * var) / 2.
        cdf = (1. + ca.erf((y - mu) / ca.sqrt(2. * var))) / 2.

        self._pdf_function = ca.Function('pdf', [y, mu, var], [pdf])
        self._log_pdf_function = ca.Function('log_pdf', [y, mu, var], [log_pdf])
        self._cdf_function = ca.Function('cdf', [y, mu, var], [cdf])


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
        pdf = ca.exp(-ca.fabs(y - mu) / b) / (2. * b)
        log_pdf = -ca.fabs(y - mu) / b - ca.log(2. * b)
        cdf = .5 + .5 * ca.sign(y - mu) * (1. - ca.exp(-ca.fabs(y - mu) / b))

        self._pdf_function = ca.Function('pdf', [y, mu, var], [pdf])
        self._log_pdf_function = ca.Function('log_pdf', [y, mu, var], [log_pdf])
        self._cdf_function = ca.Function('cdf', [y, mu, var], [cdf])


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
        gamma_1 = ca.SX.sym('gamma_1')
        gamma_2 = ca.SX.sym('gamma_2')
        gammaln_1 = ca.SX.sym('gammaln_1')
        gammaln_2 = ca.SX.sym('gammaln_2')

        Z = gamma_1 / (ca.sqrt(var * (nu - 2.) * ca.pi) * gamma_2)
        pdf = Z * (1. + (y - mu) ** 2. / (var * (nu - 2.)))
        log_Z = gammaln_1 - gammaln_2 - ca.log(var * (nu - 2.) * ca.pi) / 2.
        log_pdf = log_Z - (nu + 1.) / 2. * ca.log(1. + (y - mu) ** 2. / (var * (nu - 2.)))

        self._pdf_function = ca.Function('pdf', [y, mu, var, nu, gamma_1, gamma_2], [pdf])
        self._log_pdf_function = ca.Function('log_pdf', [y, mu, var, nu, gammaln_1, gammaln_2], [log_pdf])

    def probability_density_function(self, *args, log: bool = False) -> Union[ca.SX, ca.MX, ca.DM]:
        """

        :param args:
        :param log:
        :return:
        """
        nu = args[-1]
        if log:
            gammaln_1 = gammaln((nu + 1.) / 2.)
            gammaln_2 = gammaln(nu / 2.)
            return self._log_pdf_function(*args, gammaln_1, gammaln_2)
        else:
            gamma_1 = gamma((nu + 1.) / 2.)
            gamma_2 = gamma(nu / 2.)
            return self._pdf_function(*args, gamma_1, gamma_2)


class Gamma(Distribution):
    """"""
    def _initialize(self) -> None:
        """

        :return:
        """
        y = ca.SX.sym('y')
        alpha = ca.SX.sym('alpha')
        beta = ca.SX.sym('beta')

        pdf = beta ** alpha / gamma(alpha) * y ** (alpha - 1) * ca.exp(-beta * y)

        self._pdf_function = ca.Function('pdf', [y, alpha, beta], [pdf])


class Prior(metaclass=ABCMeta):
    """"""
    def __init__(self, name: str) -> None:
        """Constructor method"""
        self._name = name
        self._distribution = None

    def __call__(self, *args, **kwargs) -> Union[ca.SX, ca.MX, ca.DM]:
        """Calling method"""
        x = args[0]
        p = self._get_parameter_values()
        return self._distribution(x, *p, **kwargs)

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

    @staticmethod
    def gamma(
            shape: Optional[Union[Numeric, Sequence[Numeric]]] = None,
            rate: Optional[Union[Numeric, Sequence[Numeric]]] = None,
            scale: Optional[Union[Numeric, Sequence[Numeric]]] = None
    ) -> Pr:
        """

        :return:
        """
        return GammaPrior(shape=shape, rate=rate, scale=scale)

    def probability_density_function(self, *args, log: bool = False) -> Union[ca.SX, ca.MX, ca.DM]:
        """

        :param args:
        :param log:
        :return:
        """
        p = self._get_parameter_values()
        return self._distribution.probability_density_function(*args, *p, log=log)

    pdf = probability_density_function

    def cumulative_distribution_function(self, *args) -> Union[ca.SX, ca.MX, ca.DM]:
        """

        :param args:
        :return:
        """
        p = self._get_parameter_values()
        return self._distribution.cumulative_distribution_function(*args, *p)

    cdf = cumulative_distribution_function


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


class GammaPrior(Prior):
    """"""
    def __init__(
            self,
            shape: Optional[Union[Numeric, Sequence[Numeric]]] = None,
            rate: Optional[Union[Numeric, Sequence[Numeric]]] = None,
            scale: Optional[Union[Numeric, Sequence[Numeric]]] = None
    ) -> None:
        """Constructor method"""
        super().__init__('Gamma')

        self._shape = shape
        if rate is not None and scale is not None:
            raise ValueError("Please only supply 'rate' or 'scale' for the Gamma prior, not both")
        elif rate is None and scale is not None:
            rate = 1. / scale
        self._rate = rate

    def _get_parameter_values(self) -> (Union[Numeric, Sequence[Numeric]], ...):
        """

        :return:
        """
        return self._shape, self._rate

    @property
    def shape(self) -> Optional[Numeric]:
        """

        :return:
        """
        return self._shape

    @shape.setter
    def shape(self, value: Numeric) -> None:
        self._shape = self._check_dimensionality(value, 'shape')

    @property
    def rate(self) -> Optional[Numeric]:
        """

        :return:
        """
        return self._rate

    @rate.setter
    def rate(self, value: Numeric) -> None:
        self._rate = self._check_dimensionality(value, 'rate')

    @property
    def scale(self) -> Optional[Numeric]:
        """

        :return:
        """
        if self._rate is not None:
            return 1. / self._rate
        return None

    @scale.setter
    def scale(self, value: Numeric) -> None:
        self._rate = self._check_dimensionality(1. / value, 'rate')

    @property
    def mean(self) -> Optional[Numeric]:
        """

        :return:
        """
        if self._shape is not None and self._rate is not None:
            return self._shape / self._rate
        return None

    @property
    def variance(self) -> Optional[Numeric]:
        """

        :return:
        """
        if self._shape is not None and self._rate is not None:
            return self._shape / self._rate ** 2
        return None


__all__ = [
    'Prior',
    'GaussianPrior',
    'LaplacePrior',
    'StudentsTPrior',
    'DeltaPrior',
    'GammaPrior'
]
