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
from itertools import product
from typing import Dict, List, Optional, Sequence, Tuple, TypeVar, Union
import warnings

import casadi as ca
import numpy as np
from scipy.special import factorial, gamma as gamma_fun

from ....util.machine_learning import Parameter, Hyperparameter
from ....util.util import is_list_like


IntArray = Union[int, Sequence[int]]
Numeric = Union[int, float]
NumArray = Union[Numeric, Sequence[Numeric], np.ndarray]
Bounds = Dict[str, Union[str, Tuple[Numeric, Numeric]]]
Cov = TypeVar('Cov', bound='Kernel')
Array = TypeVar('Array', ca.SX, ca.MX, np.ndarray)
Param = TypeVar('Param', bound=Parameter)


class Kernel(metaclass=ABCMeta):
    """
    Kernel base class

    This base class implements the basic methods that are used for all basic kernels, e.g. dot product kernels and
    radial basis function kernels like the squared exponential. It also implements the methods for the composition of
    kernels with the addition, multiplication and exponentiation operation.

    :Note: The Kernel classes return a CasADi function for the respective inputs when called. Each basic Kernel class
        implements its own scalar covariance function. This base class implements the call method where the respective
        scalar covariance functions are generalized to the form of covariance matrices which will be returned as CasADi
        SX function. The SX symbol was chosen in accordance with the CasADi docs which recommend that low-level
        expressions are faster computed for SX symbols.

    :param active_dims:
    :type active_dims: int, list of int, optional
    """
    def __init__(self, active_dims: Optional[IntArray] = None):
        """Constructor method"""
        self.active_dims = active_dims

    def __add__(self, other: Cov) -> 'Sum':
        """Addition method"""
        return Sum(self, other)

    def __mul__(self, other: Cov) -> 'Product':
        """Multiplication method"""
        return Product(self, other)

    def __pow__(self, power: Numeric, modulo: Optional[int] = None) -> 'Power':
        """Power method"""
        return Power(self, power)

    def __radd__(self, other: Cov) -> 'Sum':
        """Addition method (from the right side)"""
        return Sum(other, self)

    def __rmul__(self, other: Cov) -> 'Product':
        """Multiplication method (from the right side)"""
        return Product(other, self)

    def __str__(self) -> str:
        """String representation method"""
        message = f"{self.__class__.__name__} with: \n"
        for attribute, value in self.__dict__.items():
            message += f"\t {attribute}: {value} \n"
        return message

    def __call__(self, X: Array, X_bar: Optional[Array] = None) -> Array:
        """Calling method"""
        X_val = X
        X_bar_val = X_bar

        is_symbolic = False
        if isinstance(X, (ca.SX, ca.MX)):
            if X_bar is not None:
                if not isinstance(X_bar, (ca.SX, ca.MX)):
                    raise ValueError("X and X_bar need to have the same type")
            is_symbolic = True
        else:
            if X_bar is not None:
                if isinstance(X_bar, (ca.SX, ca.MX)):
                    raise ValueError("X and X_bar need to have the same type")

        X, X_bar, X_is_X_bar = _clean_input_matrices(X, X_bar)

        dimension_input_space = X.rows()
        if self.active_dims is None:
            active_dims = np.arange(dimension_input_space, dtype=np.int_)
        else:
            active_dims = np.atleast_1d(np.asarray(self.active_dims, dtype=np.int_))
        x = ca.SX.sym('x', dimension_input_space, 1)
        x_bar = ca.SX.sym('x_bar', dimension_input_space, 1)

        covariance_function = self.get_covariance_function(x, x_bar, active_dims)
        covariance_matrix = self.get_covariance_matrix(covariance_function, X, X_bar)

        if is_symbolic:
            hyperparameters = {parameter.name: parameter.SX for parameter in self.hyperparameters}
        else:
            hyperparameters = {parameter.name: parameter.log / 2. if 'variance' in parameter.name else parameter.log for
                               parameter in self.hyperparameters}

        if X_is_X_bar:
            K = covariance_matrix(X=X_val, X_bar=X_val, **hyperparameters)['covariance']
        else:
            K = covariance_matrix(X=X_val, X_bar=X_bar_val, **hyperparameters)['covariance']

        if is_symbolic:
            return K
        else:
            return K.full()

    @property
    def hyperparameters(self) -> List[Param]:
        """
        List of all hyperparameters in the kernel.

        This attribute can be easily used to access specific attributes of all hyperparameters in generator expressions
        or list comprehensions.
        """
        hyperparameters = []
        for attribute in self.__dict__.values():
            if isinstance(attribute, Parameter):
                hyperparameters.append(attribute)
        return hyperparameters

    @property
    def hyperparameter_names(self) -> List[str]:
        """

        :return:
        """
        names = []
        for attribute in self.__dict__.values():
            if isinstance(attribute, Parameter):
                names.append(attribute.name)
        return names

    @abstractmethod
    def get_covariance_function(self, x: ca.SX, x_bar: ca.SX, active_dims: np.ndarray) -> ca.Function:
        """

        :param x:
        :param x_bar:
        :param active_dims:
        :return:
        """
        pass

    def get_covariance_matrix(self, covariance_function: ca.Function, X: ca.SX, X_bar: ca.SX) -> ca.Function:
        """

        :param covariance_function:
        :param X:
        :param X_bar:
        :return:
        """
        observations_in_X = X.columns()
        observations_in_X_bar = X_bar.columns()
        K = ca.SX.sym('K', observations_in_X, observations_in_X_bar)
        hyperparameter_symbols = {hyperparameter.name: hyperparameter.SX for hyperparameter in self.hyperparameters}
        hyperparameter_names = [hyperparameter.name for hyperparameter in self.hyperparameters]

        # TODO: Could maybe be faster with map?
        for i, j in product(range(observations_in_X), range(observations_in_X_bar)):
            K[i, j] = covariance_function(x=X[:, i], x_bar=X_bar[:, j], **hyperparameter_symbols)['covariance']

        covariance_matrix = ca.Function(
            'K',
            [X, X_bar, *hyperparameter_symbols.values()],
            [K],
            ['X', 'X_bar', *hyperparameter_names],
            ['covariance']
        )

        return covariance_matrix

    def is_bounded(self) -> bool:  # pragma: no cover
        """

        :return:
        """
        for parameter in self.hyperparameters:
            bounds = parameter.log_bounds
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
    def constant(bias: Numeric = 1., bounds: Optional[Bounds] = None) -> Cov:
        """

        :param bias:
        :param bounds:
        :return:
        """
        return ConstantKernel(bias=bias, bounds=bounds)

    @staticmethod
    def squared_exponential(
            active_dims: Optional[IntArray] = None,
            signal_variance: Numeric = 1.,
            length_scales: NumArray = 1.,
            ard: bool = False,
            bounds: Optional[Bounds] = None
    ) -> Cov:
        """

        :param active_dims:
        :param signal_variance:
        :param length_scales:
        :param ard:
        :param bounds:
        :return:
        """
        return SquaredExponentialKernel(active_dims=active_dims, signal_variance=signal_variance,
                                        length_scales=length_scales, ard=ard, bounds=bounds)

    @staticmethod
    def exponential(
            active_dims: Optional[IntArray] = None,
            signal_variance: Numeric = 1.,
            length_scales: NumArray = 1.,
            ard: bool = False,
            bounds: Optional[Bounds] = None
    ) -> Cov:
        """

        :param active_dims:
        :param signal_variance:
        :param length_scales:
        :param ard:
        :param bounds:
        :return:
        """
        return ExponentialKernel(active_dims=active_dims, signal_variance=signal_variance, length_scales=length_scales,
                                 ard=ard, bounds=bounds)

    @staticmethod
    def matern_32(
            active_dims: Optional[IntArray] = None,
            signal_variance: Numeric = 1.,
            length_scales: NumArray = 1.,
            ard: bool = False,
            bounds: Optional[Bounds] = None
    ) -> Cov:
        """

        :param active_dims:
        :param signal_variance:
        :param length_scales:
        :param ard:
        :param bounds:
        :return:
        """
        return Matern32Kernel(active_dims=active_dims, signal_variance=signal_variance, length_scales=length_scales,
                              ard=ard, bounds=bounds)

    @staticmethod
    def matern_52(
            active_dims: Optional[IntArray] = None,
            signal_variance: Numeric = 1.,
            length_scales: NumArray = 1.,
            ard: bool = False,
            bounds: Optional[Bounds] = None
    ) -> Cov:
        """

        :param active_dims:
        :param signal_variance:
        :param length_scales:
        :param ard:
        :param bounds:
        :return:
        """
        return Matern52Kernel(active_dims=active_dims, signal_variance=signal_variance, length_scales=length_scales,
                              ard=ard, bounds=bounds)

    @staticmethod
    def rational_quadratic(
            active_dims: Optional[IntArray] = None,
            signal_variance: Numeric = 1.,
            length_scales: NumArray = 1.,
            alpha: Numeric = 1.,
            ard: bool = False,
            bounds: Optional[Bounds] = None
    ) -> Cov:
        """

        :param active_dims:
        :param signal_variance:
        :param length_scales:
        :param alpha:
        :param ard:
        :param bounds:
        :return:
        """
        return RationalQuadraticKernel(active_dims=active_dims, signal_variance=signal_variance,
                                       length_scales=length_scales, alpha=alpha, ard=ard, bounds=bounds)

    @staticmethod
    def piecewise_polynomial(
            degree: int,
            active_dims: Optional[IntArray] = None,
            signal_variance: Numeric = 1.,
            length_scales: NumArray = 1.,
            ard: bool = False,
            bounds: Optional[Bounds] = None
    ) -> Cov:
        """

        :param degree:
        :param active_dims:
        :param signal_variance:
        :param length_scales:
        :param ard:
        :param bounds:
        :return:
        """
        return PiecewisePolynomialKernel(degree, active_dims=active_dims, signal_variance=signal_variance,
                                         length_scales=length_scales, ard=ard, bounds=bounds)

    @staticmethod
    def polynomial(
            degree: int,
            active_dims: Optional[IntArray] = None,
            signal_variance: Numeric = 1.,
            offset: Numeric = 1.,
            bounds: Optional[Bounds] = None
    ) -> Cov:
        """

        :param degree:
        :param active_dims:
        :param signal_variance:
        :param offset:
        :param bounds:
        :return:
        """
        return PolynomialKernel(degree, active_dims=active_dims, signal_variance=signal_variance, offset=offset,
                                bounds=bounds)

    @staticmethod
    def linear(
            active_dims: Optional[IntArray] = None,
            signal_variance: Numeric = 1.,
            bounds: Optional[Bounds] = None
    ) -> Cov:
        """

        :param active_dims:
        :param signal_variance:
        :param bounds:
        :return:
        """
        return LinearKernel(active_dims=active_dims, signal_variance=signal_variance, bounds=bounds)

    @staticmethod
    def neural_network(
            active_dims: Optional[IntArray] = None,
            signal_variance: Numeric = 1.,
            weight_variance: Numeric = 1.,
            bounds: Optional[Bounds] = None
    ) -> Cov:
        """

        :param active_dims:
        :param signal_variance:
        :param weight_variance:
        :param bounds:
        :return:
        """
        return NeuralNetworkKernel(active_dims=active_dims, signal_variance=signal_variance,
                                   weight_variance=weight_variance, bounds=bounds)

    @staticmethod
    def periodic(
            active_dims: Optional[IntArray] = None,
            signal_variance: Numeric = 1.,
            length_scales: NumArray = 1.,
            period: Numeric = 1.,
            bounds: Optional[Bounds] = None
    ) -> Cov:
        """

        :param active_dims:
        :param signal_variance:
        :param length_scales:
        :param period:
        :param bounds:
        :return:
        """
        return PeriodicKernel(active_dims=active_dims, signal_variance=signal_variance, length_scales=length_scales,
                              period=period, bounds=bounds)


class ConstantKernel(Kernel):
    """
    Constant covariance function

    :param bias:
    :type bias:
    :param bounds:
    :type bounds:
    """
    acronym = "Const"

    def __init__(
            self,
            bias: Numeric = 1.,
            bounds: Optional[Bounds] = None
    ) -> None:
        """Constructor method"""
        super().__init__()

        hyper_kwargs = {}
        if bounds is not None:  # pragma: no cover
            bias_bounds = bounds.get('bias')
            if bias_bounds is not None:
                if bias_bounds == 'fixed':
                    hyper_kwargs['fixed'] = True
                else:
                    hyper_kwargs['bounds'] = bias_bounds
        self.bias = Hyperparameter(f'{self.acronym}.bias', value=bias, **hyper_kwargs)

    def get_covariance_function(self, x: ca.SX, x_bar: ca.SX, active_dims: np.ndarray) -> ca.Function:
        """
        Covariance function of constant as follows:
            bias^2

        :param x:
        :param x_bar:
        :param active_dims:
        :return:
        """
        log_bias = self.bias.SX

        covariance_function = ca.Function(
            'covariance',
            [x, x_bar, log_bias],
            [ca.exp(2 * log_bias)],
            ['x', 'x_bar', self.bias.name],
            ['covariance']
        )

        return covariance_function


class StationaryKernel(Kernel, metaclass=ABCMeta):
    """
    Base for stationary kernels

    In the isotropic form the characteristic length scale is the same for every input dimensions. By parameterization
    each input dimension can be given a length scales, thus enabling automatic relevance detection during training.

    :param active_dims:
    :type active_dims:
    :param length_scales:
    :type length_scales:
    :param ard:
    :type ard:
    :param bounds:
    :type bounds:
    """
    acronym = "Stat"

    def __init__(
            self,
            active_dims: Optional[IntArray] = None,
            length_scales: NumArray = 1.,
            ard: bool = False,
            bounds: Optional[Bounds] = None
    ) -> None:
        """Constructor method"""
        super().__init__(active_dims=active_dims)

        hyper_kwargs = {}
        if bounds is not None:  # pragma: no cover
            length_scale_bounds = bounds.get('length_scales')
            if length_scale_bounds is not None:
                if length_scale_bounds == 'fixed':
                    hyper_kwargs['fixed'] = True
                else:
                    hyper_kwargs['bounds'] = length_scale_bounds

        if ard and active_dims is None:
            raise ValueError("The key word 'ard' can only be set to True if the key word 'active_dims' was supplied")

        if active_dims is not None and is_list_like(length_scales):
            if len(active_dims) != len(length_scales):
                raise ValueError(f"Dimension mismatch between 'active_dims' ({len(active_dims)}) and the number of "
                                 f"length_scales ({len(length_scales)})")

        if not is_list_like(length_scales) and ard:
            length_scales = len(active_dims) * [length_scales]

        self.length_scales = Hyperparameter(f'{self.acronym}.length_scales', value=length_scales, **hyper_kwargs)

    def get_parameterized_length_scales(self, dimension_input_space: int, log_length_scales: ca.SX) -> ca.Function:
        """
        The parameterized notation allows for automatic relevance detection if the length scales are given as vector
        (the kernel is considered anisotropic).

        :param dimension_input_space:
        :param log_length_scales:
        :return:
        """
        if self.is_isotropic():
            M = ca.Function('M',
                            [log_length_scales],
                            [ca.exp(-2 * log_length_scales) * ca.SX.eye(dimension_input_space)])
        elif log_length_scales.numel() == dimension_input_space:
            M = ca.Function('M', [log_length_scales], [ca.diag(ca.exp(-2 * log_length_scales))])
        else:
            raise ValueError("Length scales vector dimension does not equal input space dimension.")
        return M

    def is_isotropic(self) -> bool:
        """

        :return:
        """
        return self.length_scales.is_scalar()


class GammaExponentialKernel(StationaryKernel):
    """
    Gamma-exponential covariance function

    :param active_dims:
    :type active_dims:
    :param signal_variance:
    :type signal_variance:
    :param alpha:
    :type alpha:
    :param gamma:
    :type gamma:
    :param length_scales:
    :type length_scales:
    :param ard:
    :type ard:
    :param bounds:
    :type bounds:
    """
    acronym = "GE"

    def __init__(
            self,
            active_dims: Optional[IntArray] = None,
            signal_variance: Numeric = 1.,
            alpha: Optional[Numeric] = None,
            gamma: Numeric = 1.,
            length_scales: NumArray = 1.,
            ard: bool = False,
            bounds: Optional[Bounds] = None
    ) -> None:
        """Constructor method"""
        super().__init__(active_dims=active_dims, length_scales=length_scales, ard=ard, bounds=bounds)

        if bounds is not None:  # pragma: no cover
            variance_bounds = bounds.get('signal_variance')
            gamma_bounds = bounds.get('gamma')
            alpha_bounds = bounds.get('alpha')
        else:  # pragma: no cover
            variance_bounds, gamma_bounds, alpha_bounds = None, None, None

        hyper_kwargs = {}
        if variance_bounds is not None:  # pragma: no cover
            if variance_bounds == 'fixed':
                hyper_kwargs['fixed'] = True
            else:
                hyper_kwargs['bounds'] = variance_bounds
        self.signal_variance = Hyperparameter(f'{self.acronym}.signal_variance', value=signal_variance, **hyper_kwargs)

        hyper_kwargs = {}
        if gamma == 2.:
            raise ValueError("Value of the hyperparameter 'gamma' is set to 2. Use the squared exponential kernel "
                             "instead.")
        gamma /= 2 - gamma
        if gamma_bounds is not None:  # pragma: no cover
            if gamma_bounds == 'fixed':
                if gamma <= 0. or gamma > 2.:
                    raise ValueError("Value of the hyperparameter 'gamma' needs to be between 0 and 2")
                hyper_kwargs['fixed'] = True
            else:
                gamma_bounds = list(gamma_bounds)
                if gamma_bounds[0] < 0.:
                    warnings.warn("Value of hyperparameter 'gamma' needs to be bigger than 0. Switching lower bound to "
                                  "0...")
                    gamma_bounds[0] = 0.
                if gamma_bounds[1] > 2.:
                    warnings.warn("Value of hyperparameter 'gamma' needs to be smaller than 2. Switching upper bound to"
                                  " 2...")
                    gamma_bounds[1] = 2.
                gamma_bounds[0] /= 2 - gamma_bounds[0]
                gamma_bounds[1] /= 2 - gamma_bounds[1]
                hyper_kwargs['bounds'] = tuple(gamma_bounds)
        self.gamma = Hyperparameter(f'{self.acronym}.gamma', value=gamma, **hyper_kwargs)

        if alpha is not None:
            hyper_kwargs = {}
            if alpha_bounds is not None:  # pragma: no cover
                if alpha_bounds == 'fixed':
                    hyper_kwargs['fixed'] = True
                else:
                    hyper_kwargs['bounds'] = alpha_bounds
            self.alpha = Hyperparameter(f'{self.acronym}.alpha', value=alpha, **hyper_kwargs)
        else:
            self.alpha = 1.

    def get_covariance_function(self, x: ca.SX, x_bar: ca.SX, active_dims: np.ndarray) -> ca.Function:
        """
        Covariance function of gamma exponential as follows:

        .. math::

            \sigma^2 \cdot \exp(-1/2 \cdot ((x-x_{bar})^T M (x-x_{bar}))^{\gamma-1})

        where :math:`M` is the parameterized matrix of length scales

        :param x:
        :param x_bar:
        :param active_dims:
        :return:
        """
        log_std = self.signal_variance.SX
        log_length_scales = self.length_scales.SX

        log_gamma = self.gamma
        if isinstance(log_gamma, Parameter):
            gamma_arg_name = [log_gamma.name]
            log_gamma = log_gamma.SX
            gamma_arg = [log_gamma]
        else:
            log_gamma = ca.DM(log_gamma)
            log_gamma = ca.log(log_gamma / (2 - log_gamma))
            gamma_arg = []
            gamma_arg_name = []
        p = 2 / (1 - ca.exp(-log_gamma))

        log_alpha = self.alpha
        if isinstance(log_alpha, Parameter):
            alpha_arg_name = [log_alpha.name]
            log_alpha = log_alpha.SX
            alpha_arg = [log_alpha]
        else:
            log_alpha = ca.log(log_alpha)
            alpha_arg = []
            alpha_arg_name = []

        M = self.get_parameterized_length_scales(active_dims.size, log_length_scales)
        d2 = _mahalanobis_distance_squared(x[active_dims], x_bar[active_dims], M(log_length_scales))

        covariance_function = ca.Function(
            'covariance',
            [x, x_bar, log_std, log_length_scales] + gamma_arg + alpha_arg,
            [ca.exp(2 * log_std - ca.exp(log_alpha) * d2 ** (p / 2))],
            ['x', 'x_bar', self.signal_variance.name, self.length_scales.name] + gamma_arg_name + alpha_arg_name,
            ['covariance']
        )

        return covariance_function


class SquaredExponentialKernel(GammaExponentialKernel):
    """
    Squared exponential covariance function

    :param active_dims:
    :type active_dims:
    :param signal_variance:
    :type signal_variance:
    :param length_scales:
    :type length_scales:
    :param ard:
    :type ard:
    :param bounds:
    :type bounds:
    """
    acronym = "SE"

    def __init__(
            self,
            active_dims: Optional[IntArray] = None,
            signal_variance: Numeric = 1.,
            length_scales: NumArray = 1.,
            ard: bool = False,
            bounds: Optional[Bounds] = None
    ) -> None:
        """Constructor method"""
        super().__init__(active_dims=active_dims, signal_variance=signal_variance, length_scales=length_scales, ard=ard,
                         bounds=bounds)

        self.gamma = 2.
        self.alpha = .5


class MaternKernel(StationaryKernel):
    """
    Base for Matérn class of covariance functions

    :param p:
    :type p:
    :param active_dims:
    :type active_dims:
    :param signal_variance:
    :type signal_variance:
    :param length_scales:
    :type length_scales:
    :param ard:
    :type ard:
    :param bounds:
    :type bounds:
    """
    acronym = "Matern"

    def __init__(
            self,
            p: int,
            active_dims: Optional[IntArray] = None,
            signal_variance: Numeric = 1.,
            length_scales: NumArray = 1.,
            ard: bool = False,
            bounds: Optional[Bounds] = None
    ) -> None:
        """Constructor method"""
        super().__init__(active_dims=active_dims, length_scales=length_scales, ard=ard, bounds=bounds)

        if bounds is not None:  # pragma: no cover
            variance_bounds = bounds.get('signal_variance')
        else:  # pragma: no cover
            variance_bounds = None

        hyper_kwargs = {}
        if variance_bounds is not None:  # pragma: no cover
            if variance_bounds == 'fixed':
                hyper_kwargs['fixed'] = True
            else:
                hyper_kwargs['bounds'] = variance_bounds
        self.signal_variance = Hyperparameter(f'{self.acronym}.signal_variance', value=signal_variance, **hyper_kwargs)

        self._p = p

    def get_covariance_function(self, x: ca.SX, x_bar: ca.SX, active_dims: np.ndarray) -> ca.Function:
        """

        :param x:
        :param x_bar:
        :param active_dims:
        :return:
        """
        nu = self._p + .5

        log_std = self.signal_variance.SX
        log_length_scales = self.length_scales.SX

        M = self.get_parameterized_length_scales(active_dims.size, log_length_scales)
        d2 = _mahalanobis_distance_squared(x[active_dims], x_bar[active_dims], M(log_length_scales))

        if self._p > 1:
            gamma_p_plus_1 = gamma_fun(self._p + 1)
            gamma_2p_plus_1 = gamma_fun(2 * self._p + 1)
            poly1d = [gamma_p_plus_1 / gamma_2p_plus_1 * factorial(self._p + k) / (
                        factorial(k) * factorial(self._p - k)) * 2. ** (self._p - k) for k in range(self._p - 1)]
        else:
            poly1d = []
        if self._p == 0:
            poly1d.append(0.)
        elif self._p >= 1.:
            poly1d.append(1.)
        for k in range(len(poly1d) - 1, 0, -1):
            poly1d[k - 1] /= poly1d[k]

        d = ca.sqrt(2 * nu * d2)
        f = 1. + d * poly1d[0]
        for k in range(1, len(poly1d)):
            f = 1. + d * poly1d[k] * f

        covariance_function = ca.Function(
            'covariance',
            [x, x_bar, log_std, log_length_scales],
            [ca.exp(2 * log_std - d) * f],
            ['x', 'x_bar', self.signal_variance.name, self.length_scales.name],
            ['covariance']
        )

        return covariance_function


class ExponentialKernel(MaternKernel):
    """
    Exponential covariance function

    :param active_dims:
    :type active_dims:
    :param signal_variance:
    :type signal_variance:
    :param length_scales:
    :type length_scales:
    :param ard:
    :type ard:
    :param bounds:
    :type bounds:
    """
    acronym = "E"

    def __init__(
            self,
            active_dims: Optional[IntArray] = None,
            signal_variance: Numeric = 1.,
            length_scales: NumArray = 1.,
            ard: bool = False,
            bounds: Optional[Bounds] = None
    ) -> None:
        """Constructor method"""
        super().__init__(0, active_dims=active_dims, signal_variance=signal_variance, length_scales=length_scales,
                         ard=ard, bounds=bounds)


class Matern32Kernel(MaternKernel):
    """
    Matérn covariance function where :math:`{\\nu}` = 3/2

    :param active_dims:
    :type active_dims:
    :param signal_variance:
    :type signal_variance:
    :param length_scales:
    :type length_scales:
    :param ard:
    :type ard:
    :param bounds:
    :type bounds:
    """
    acronym = "M32"

    def __init__(
            self,
            active_dims: Optional[IntArray] = None,
            signal_variance: Numeric = 1.,
            length_scales: NumArray = 1.,
            ard: bool = False,
            bounds: Optional[Bounds] = None
    ) -> None:
        """Constructor method"""
        super().__init__(1, active_dims=active_dims, signal_variance=signal_variance, length_scales=length_scales,
                         ard=ard, bounds=bounds)


class Matern52Kernel(MaternKernel):
    """
    Matérn covariance function where :math:`{\\nu}` = 5/2

    :param active_dims:
    :type active_dims:
    :param signal_variance:
    :type signal_variance:
    :param length_scales:
    :type length_scales:
    :param ard:
    :type ard:
    :param bounds:
    :type bounds:
    """
    acronym = "M52"

    def __init__(
            self,
            active_dims: Optional[IntArray] = None,
            signal_variance: Numeric = 1.,
            length_scales: NumArray = 1.,
            ard: bool = False,
            bounds: Optional[Bounds] = None
    ) -> None:
        """Constructor method"""
        super().__init__(2, active_dims=active_dims, signal_variance=signal_variance, length_scales=length_scales,
                         ard=ard, bounds=bounds)


class RationalQuadraticKernel(StationaryKernel):
    """
    Rational quadratic covariance function

    :param active_dims:
    :type active_dims:
    :param signal_variance:
    :type signal_variance:
    :param length_scales:
    :type length_scales:
    :param alpha:
    :type alpha:
    :param ard:
    :type ard:
    :param bounds:
    :type bounds:
    """
    acronym = "RQ"

    def __init__(
            self,
            active_dims: Optional[IntArray] = None,
            signal_variance: Numeric = 1.,
            length_scales: NumArray = 1.,
            alpha: Numeric = 1.,
            ard: bool = False,
            bounds: Optional[Bounds] = None
    ) -> None:
        """Constructor method"""
        super().__init__(active_dims=active_dims, length_scales=length_scales, ard=ard, bounds=bounds)

        if bounds is not None:  # pragma: no cover
            variance_bounds = bounds.get('signal_variance')
            alpha_bounds = bounds.get('alpha')
        else:  # pragma: no cover
            variance_bounds, alpha_bounds = None, None

        hyper_kwargs = {}
        if variance_bounds is not None:  # pragma: no cover
            if variance_bounds == 'fixed':
                hyper_kwargs['fixed'] = True
            else:
                hyper_kwargs['bounds'] = variance_bounds
        self.signal_variance = Hyperparameter(f'{self.acronym}.signal_variance', value=signal_variance, **hyper_kwargs)

        hyper_kwargs = {}
        if alpha_bounds is not None:  # pragma: no cover
            if alpha_bounds == 'fixed':
                hyper_kwargs['fixed'] = True
            else:
                hyper_kwargs['bounds'] = alpha_bounds
        self.alpha = Hyperparameter(f'{self.acronym}.alpha', value=alpha, **hyper_kwargs)

    def get_covariance_function(self, x: ca.SX, x_bar: ca.SX, active_dims: np.ndarray) -> ca.Function:
        """
        Covariance function of rational quadratic as follows:

        .. math::

            \sigma \cdot [1+1/(2\cdot\\alpha)\cdot(x-x_{bar})^T M (x-x_{bar})]^{-\\alpha}

        where :math:`M` is the parameterized matrix of length scales

        :param x:
        :param x_bar:
        :param active_dims:
        :return:
        """
        log_std = self.signal_variance.SX
        log_length_scales = self.length_scales.SX
        log_alpha = self.alpha.SX

        M = self.get_parameterized_length_scales(active_dims.size, log_length_scales)
        d2 = _mahalanobis_distance_squared(x[active_dims], x_bar[active_dims], M(log_length_scales))

        covariance_function = ca.Function(
            'covariance',
            [x, x_bar, log_std, log_length_scales, log_alpha],
            [ca.exp(2 * log_std) * (1 + .5 * d2 / ca.exp(log_alpha)) ** (-ca.exp(log_alpha))],
            # [ca.exp(2 * log_std - ca.exp(log_alpha) * ca.log(1 + .5 * d2 / ca.exp(log_alpha)))],
            ['x', 'x_bar', self.signal_variance.name, self.length_scales.name, self.alpha.name],
            ['covariance']
        )

        return covariance_function


class PiecewisePolynomialKernel(StationaryKernel):
    """
    Piecewise polynomial covariance function with compact support

    :param degree:
    :type degree:
    :param active_dims:
    :type active_dims:
    :param signal_variance:
    :type signal_variance:
    :param length_scales:
    :type length_scales:
    :param ard:
    :type ard:
    :param bounds:
    :type bounds:
    """
    acronym = "PP"

    def __init__(
            self,
            degree: int,
            active_dims: Optional[IntArray] = None,
            signal_variance: Numeric = 1.,
            length_scales: NumArray = 1.,
            ard: bool = False,
            bounds: Optional[Bounds] = None
    ) -> None:
        """Constructor method"""
        if degree not in [0, 1, 2, 3]:
            raise ValueError("The parameter 'degree' has to be one of the following integers: 0, 1, 2, 3")

        super().__init__(active_dims=active_dims, length_scales=length_scales, ard=ard, bounds=bounds)

        if bounds is not None:  # pragma: no cover
            variance_bounds = bounds.get('signal_variance')
        else:  # pragma: no cover
            variance_bounds = None

        hyper_kwargs = {}
        if variance_bounds is not None:  # pragma: no cover
            if variance_bounds == 'fixed':
                hyper_kwargs['fixed'] = True
            else:
                hyper_kwargs['bounds'] = variance_bounds
        self.signal_variance = Hyperparameter(f'{self.acronym}.signal_variance', value=signal_variance, **hyper_kwargs)

        self._q = degree

    @property
    def degree(self) -> int:
        """

        :return:
        """
        return self._q

    @degree.setter
    def degree(self, value: int) -> None:
        if value not in [0, 1, 2, 3]:
            raise ValueError("The property 'degree' has to be one of the following integers: 0, 1, 2, 3")
        self._q = value

    def get_covariance_function(self, x: ca.SX, x_bar: ca.SX, active_dims: np.ndarray) -> ca.Function:
        """

        :param x:
        :param x_bar:
        :param active_dims:
        :return:
        """
        q = self._q
        j = ca.floor(active_dims.size / 2) + q + 1

        log_std = self.signal_variance.SX
        log_length_scales = self.length_scales.SX

        M = self.get_parameterized_length_scales(active_dims.size, log_length_scales)
        d2 = _mahalanobis_distance_squared(x[active_dims], x_bar[active_dims], M(log_length_scales))
        d = ca.sqrt(d2)

        if q == 0:
            f = 1.
        elif q == 1:
            f = (j + 1) * d + 1
        elif q == 2:
            f = (j ** 2 + 4 * j + 3) / 3 * d2 + (j + 2) * d + 1
        elif q == 3:
            f = (j ** 3 + 9 * j ** 2 + 23 * j + 15) / 15 * d ** 3 + (6 * j ** 2 + 36 * j + 45) / 15 * d2 + (
                        j + 3) * d + 1
        else:
            raise RuntimeError("The parameter 'q' has to be one of the following integers: 0, 1, 2, 3")

        compact_support = (d < 1.) * ca.fmax(1. - d, 0.) ** (j + q)

        covariance_function = ca.Function(
            'covariance',
            [x, x_bar, log_std, log_length_scales],
            [ca.exp(2 * log_std) * compact_support * f],
            ['x', 'x_bar', self.signal_variance.name, self.length_scales.name],
            ['covariance']
        )

        return covariance_function


class DotProductKernel(Kernel, metaclass=ABCMeta):
    """
    Base for all dot product kernels

    :param active_dims:
    :type active_dims:
    :param signal_variance:
    :type signal_variance:
    :param offset:
    :type offset:
    :param bounds:
    :type bounds:
    """
    acronym = "Dot"

    def __init__(
            self,
            active_dims: Optional[IntArray] = None,
            signal_variance: Numeric = 1.,
            offset: Numeric = 1.,
            bounds: Optional[Bounds] = None
    ) -> None:
        """Constructor method"""
        super().__init__(active_dims=active_dims)

        if bounds is not None:  # pragma: no cover
            variance_bounds = bounds.get('signal_variance')
            offset_bounds = bounds.get('offset')
        else:  # pragma: no cover
            variance_bounds, offset_bounds = None, None

        hyper_kwargs = {}
        if variance_bounds is not None:  # pragma: no cover
            if variance_bounds == 'fixed':
                hyper_kwargs['fixed'] = True
            else:
                hyper_kwargs['bounds'] = variance_bounds
        self.signal_variance = Hyperparameter(f'{self.acronym}.signal_variance', value=signal_variance, **hyper_kwargs)

        hyper_kwargs = {}
        if offset_bounds is not None:  # pragma: no cover
            if offset_bounds == 'fixed':
                hyper_kwargs['fixed'] = True
            else:
                hyper_kwargs['bounds'] = offset_bounds
        self.offset = Hyperparameter(f'{self.acronym}.offset', value=offset, **hyper_kwargs)


class PolynomialKernel(DotProductKernel):
    """
    Polynomial covariance function

    :param degree:
    :type degree:
    :param active_dims:
    :type active_dims:
    :param signal_variance:
    :type signal_variance:
    :param offset:
    :type offset:
    :param bounds:
    :type bounds:
    """
    acronym = "Poly"

    def __init__(
            self,
            degree: int,
            active_dims: Optional[IntArray] = None,
            signal_variance: Numeric = 1.,
            offset: Numeric = 1.,
            bounds: Optional[Bounds] = None
    ) -> None:
        """Constructor method"""
        super().__init__(active_dims=active_dims, signal_variance=signal_variance, offset=offset, bounds=bounds)

        self._p = degree

    @property
    def degree(self) -> int:
        """

        :return:
        """
        return self._p

    @degree.setter
    def degree(self, value: int) -> None:
        self._p = value

    def get_covariance_function(self, x: ca.SX, x_bar: ca.SX, active_dims: np.ndarray) -> ca.Function:
        """
        Covariance function of polynomial as follows:
            (sigma_bias^2 + sigma_signal^2 * (x dot x_bar))^p

        :param x:
        :param x_bar:
        :param active_dims:
        :return:
        """
        p = self._p

        log_std = self.signal_variance.SX

        log_offset = self.offset
        if isinstance(log_offset, Parameter):
            offset_arg_name = [log_offset.name]
            log_offset = log_offset.SX
            offset_arg = [log_offset]
        else:
            log_offset = ca.log(log_offset)
            offset_arg = []
            offset_arg_name = []

        covariance_function = ca.Function(
            'covariance',
            [x, x_bar, log_std] + offset_arg,
            [ca.exp(2 * log_std) * (x[active_dims].T @ x_bar[active_dims] + ca.exp(log_offset)) ** p],
            ['x', 'x_bar', self.signal_variance.name] + offset_arg_name,
            ['covariance']
        )

        return covariance_function


class LinearKernel(PolynomialKernel):
    """
    Linear covariance function

    :param active_dims:
    :type active_dims:
    :param signal_variance:
    :type signal_variance:
    :param bounds:
    :type bounds:
    """
    acronym = "Lin"

    def __init__(
            self,
            active_dims: Optional[IntArray] = None,
            signal_variance: Numeric = 1.,
            bounds: Optional[Bounds] = None
    ) -> None:
        """Constructor method"""
        super().__init__(1, active_dims=active_dims, signal_variance=signal_variance, bounds=bounds)

        self.offset = 0.


class NeuralNetworkKernel(Kernel):
    """
    Neural network covariance function

    :param active_dims:
    :type active_dims:
    :param signal_variance:
    :type signal_variance:
    :param weight_variance:
    :type weight_variance:
    :param bounds:
    :type bounds:
    """
    acronym = "NN"

    def __init__(
            self,
            active_dims: Optional[IntArray] = None,
            signal_variance: Numeric = 1.,
            weight_variance: Numeric = 1.,
            bounds: Optional[Bounds] = None
    ) -> None:
        """Constructor method"""
        super().__init__(active_dims=active_dims)

        if bounds is not None:  # pragma: no cover
            signal_variance_bounds = bounds.get('signal_variance')
            weight_variance_bounds = bounds.get('weight_variance')
        else:  # pragma: no cover
            signal_variance_bounds, weight_variance_bounds = None, None

        hyper_kwargs = {}
        if signal_variance_bounds is not None:  # pragma: no cover
            if signal_variance_bounds == 'fixed':
                hyper_kwargs['fixed'] = True
            else:
                hyper_kwargs['bounds'] = signal_variance_bounds
        self.signal_variance = Hyperparameter(f'{self.acronym}.signal_variance', value=signal_variance, **hyper_kwargs)

        hyper_kwargs = {}
        if weight_variance_bounds is not None:  # pragma: no cover
            if weight_variance_bounds == 'fixed':
                hyper_kwargs['fixed'] = True
            else:
                hyper_kwargs['bounds'] = weight_variance_bounds
        self.weight_variance = Hyperparameter(f'{self.acronym}.weight_variance', value=weight_variance, **hyper_kwargs)

    def get_covariance_function(self, x: ca.SX, x_bar: ca.SX, active_dims: np.ndarray) -> ca.Function:
        """

        :param x:
        :param x_bar:
        :param active_dims:
        :return:
        """
        log_std = self.signal_variance.SX
        log_weight = self.weight_variance.SX

        num = 1. + x[active_dims].T @ x_bar[active_dims]
        den1 = ca.sqrt(ca.exp(2 * log_weight) + 1. + x[active_dims].T @ x[active_dims])
        den2 = ca.sqrt(ca.exp(2 * log_weight) + 1. + x_bar[active_dims].T @ x_bar[active_dims])

        covariance_function = ca.Function(
            'covariance',
            [x, x_bar, log_std, log_weight],
            [ca.exp(2 * log_std) * ca.asin(num / (den1 * den2))],
            ['x', 'x_bar', self.signal_variance.name, self.weight_variance.name],
            ['covariance']
        )

        return covariance_function


class PeriodicKernel(Kernel):
    """
    Periodic covariance function

    :param active_dims:
    :type active_dims:
    :param signal_variance:
    :type signal_variance:
    :param length_scales:
    :type length_scales:
    :param period:
    :type period:
    :param bounds:
    :type bounds:
    """
    acronym = "Periodic"

    def __init__(
            self,
            active_dims: Optional[IntArray] = None,
            signal_variance: Numeric = 1.,
            length_scales: Numeric = 1.,
            period: Numeric = 1,
            bounds: Optional[Bounds] = None
    ) -> None:
        """Constructor method"""
        super().__init__(active_dims=active_dims)

        if bounds is not None:  # pragma: no cover
            signal_variance_bounds = bounds.get('signal_variance')
            length_scales_bounds = bounds.get('length_scales')
            period_bounds = bounds.get('period')
        else:  # pragma: no cover
            signal_variance_bounds, length_scales_bounds, period_bounds = None, None, None

        hyper_kwargs = {}
        if signal_variance_bounds is not None:  # pragma: no cover
            if signal_variance_bounds == 'fixed':
                hyper_kwargs['fixed'] = True
            else:
                hyper_kwargs['bounds'] = signal_variance_bounds
        self.signal_variance = Hyperparameter(f'{self.acronym}.signal_variance', value=signal_variance, **hyper_kwargs)

        hyper_kwargs = {}
        if length_scales_bounds is not None:  # pragma: no cover
            if length_scales_bounds == 'fixed':
                hyper_kwargs['fixed'] = True
            else:
                hyper_kwargs['bounds'] = length_scales_bounds
        self.length_scales = Hyperparameter(f'{self.acronym}.length_scales', value=length_scales, **hyper_kwargs)

        hyper_kwargs = {}
        if period_bounds is not None:  # pragma: no cover
            if period_bounds == 'fixed':
                hyper_kwargs['fixed'] = True
            else:
                hyper_kwargs['bounds'] = period_bounds
        self.period = Hyperparameter(f'{self.acronym}.period', value=period, **hyper_kwargs)

    def get_covariance_function(self, x: ca.SX, x_bar: ca.SX, active_dims: np.ndarray) -> ca.Function:
        """
        Covariance function of exponential sine as follows:

        .. math::

            \sigma^2 \cdot \exp\\left(-1/2 \cdot \\frac{\sin(x-x_{bar})^T}{p} M \\frac{\sin(x-x_{bar})}{p}\\right)

        where :math:`M` is the parameterized matrix of length scales

        :param x:
        :param x_bar:
        :param active_dims:
        :return:
        """
        log_std = self.signal_variance.SX
        log_length_scales = self.length_scales.SX
        log_period = self.period.SX

        arg = ca.sin(ca.pi * (x[active_dims] - x_bar[active_dims]) / ca.exp(log_period)) / ca.exp(log_length_scales)

        covariance_function = ca.Function(
            'covariance',
            [x, x_bar, log_std, log_length_scales, log_period],
            [ca.exp(2 * log_std - 2 * arg ** 2)],
            ['x', 'x_bar', self.signal_variance.name, self.length_scales.name, self.period.name],
            ['covariance']
        )

        return covariance_function


class KernelOperator(Kernel, metaclass=ABCMeta):
    """
    Kernel operator base class

    There are three basic operations to composite kernels which preserve their properties as valid covariance matrices.
    These are addition, multiplication and exponentiation.

    :param kernel_1:
    :param kernel_2:
    """
    def __init__(self, kernel_1: Cov, kernel_2: Optional[Cov] = None) -> None:
        """Constructor method"""
        super().__init__()

        self.kernel_1 = copy.deepcopy(kernel_1)
        if kernel_2 is not None:
            self.kernel_2 = copy.deepcopy(kernel_2)
        else:
            self.kernel_2 = kernel_2
        self.disambiguate_hyperparameter_names()

    @property
    def hyperparameters(self) -> List[Param]:
        """
        List of all hyperparameters in the kernel.

        This attribute can be easily used to access specific attributes of all hyperparameters in generator expressions
        or list comprehensions.
        """
        if self.kernel_2 is not None:
            return self.kernel_1.hyperparameters + self.kernel_2.hyperparameters
        else:
            return self.kernel_1.hyperparameters

    @property
    def hyperparameter_names(self) -> List[str]:
        """

        :return:
        """
        if self.kernel_2 is not None:
            return self.kernel_1.hyperparameter_names + self.kernel_2.hyperparameter_names
        else:
            return self.kernel_1.hyperparameter_names

    def disambiguate_hyperparameter_names(self) -> None:
        """

        :return:
        """
        if self.kernel_2 is not None:
            kernel_1_has_acronym = hasattr(self.kernel_1, 'acronym')
            kernel_2_has_acronym = hasattr(self.kernel_2, 'acronym')
            if kernel_1_has_acronym and kernel_2_has_acronym:
                if self.kernel_1.acronym == self.kernel_2.acronym:
                    for parameter in self.kernel_1.hyperparameters:
                        old_name = parameter.name
                        if '.' in old_name:
                            new_name = self.kernel_1.acronym + '_1.' + old_name.split('.')[1]
                        else:
                            new_name = self.kernel_1.acronym + '_1.' + old_name
                        parameter.name = new_name
                    for parameter in self.kernel_2.hyperparameters:
                        old_name = parameter.name
                        if '.' in old_name:
                            new_name = self.kernel_2.acronym + '_2.' + old_name.split('.')[1]
                        else:
                            new_name = self.kernel_2.acronym + '_2.' + old_name
                        parameter.name = new_name
            elif kernel_1_has_acronym:
                kernel_2_acronyms = dict.fromkeys([name.split('.')[0] for name in self.kernel_2.hyperparameter_names])
                has_kernel_1_acronym = [self.kernel_1.acronym in name for name in kernel_2_acronyms]
                ct = 1
                for k, val in enumerate(kernel_2_acronyms.keys()):
                    if has_kernel_1_acronym[k]:
                        ct += 1
                        kernel_2_acronyms[val] = ct
                for parameter in self.kernel_1.hyperparameters:
                    old_name = parameter.name
                    if '.' in old_name:
                        new_name = self.kernel_1.acronym + '_1.' + old_name.split('.')[1]
                    else:
                        new_name = self.kernel_1.acronym + '_1.' + old_name
                    parameter.name = new_name
                for parameter in self.kernel_2.hyperparameters:
                    args = parameter.name.split('.')
                    old_acronym = args[0]
                    val = kernel_2_acronyms.get(old_acronym)
                    if val is not None:
                        new_acronym = old_acronym.split('_')[0] + '_' + str(val)
                        parameter.name = new_acronym + '.' + args[1]
            elif kernel_2_has_acronym:
                kernel_1_acronyms = set([name.split('.')[0] for name in self.kernel_1.hyperparameter_names])
                has_kernel_2_acronym = [self.kernel_2.acronym in name for name in kernel_1_acronyms]
                new_index = has_kernel_2_acronym.count(True) + 1
                for parameter in self.kernel_2.hyperparameters:
                    old_name = parameter.name
                    if '.' in old_name:
                        new_name = self.kernel_2.acronym + '_' + str(new_index) + '.' + old_name.split('.')[1]
                    else:
                        new_name = self.kernel_2.acronym + '_' + str(new_index) + '.' + old_name
                    parameter.name = new_name
            else:
                kernel_1_acronyms = set([name.split('.')[0] for name in self.kernel_1.hyperparameter_names])
                kernel_1_acronyms = [name.split('_')[0] for name in kernel_1_acronyms]
                kernel_2_acronyms = dict.fromkeys([name.split('.')[0] for name in self.kernel_2.hyperparameter_names])
                kernel_2_counter = {}
                for k, val in enumerate(kernel_2_acronyms.keys()):
                    args = val.split('_')
                    acronym = args[0]
                    count = kernel_1_acronyms.count(acronym)
                    kernel_2_count = kernel_2_counter.get(acronym)
                    if kernel_2_count is not None:
                        count += kernel_2_count
                        kernel_2_counter[acronym] += 1
                    else:
                        kernel_2_counter[acronym] = 1
                    if count != 0:
                        kernel_2_acronyms[val] = count + 1
                for parameter in self.kernel_2.hyperparameters:
                    args = parameter.name.split('.')
                    old_acronym = args[0]
                    val = kernel_2_acronyms.get(old_acronym)
                    if val is not None:
                        new_acronym = old_acronym.split('_')[0] + '_' + str(val)
                        parameter.name = new_acronym + '.' + args[1]


class Scale(KernelOperator):
    """"""
    def __init__(self, kernel: Cov, scale: Numeric) -> None:
        super().__init__(kernel, kernel_2=None)

        self.scale = scale


class Sum(KernelOperator):
    """
    Sum operator for covariance functions

    :param kernel_1:
    :type kernel_1:
    :param kernel_2:
    :type kernel_2:
    """
    def get_covariance_function(self, x: ca.SX, x_bar: ca.SX, active_dims: np.ndarray) -> ca.Function:
        """

        :param x:
        :param x_bar:
        :param active_dims:
        :return:
        """
        K_1 = self.kernel_1(x, x_bar)
        K_2 = self.kernel_2(x, x_bar)

        hyperparameters = [parameter.SX for parameter in self.hyperparameters]
        hyperparameter_names = [parameter.name for parameter in self.hyperparameters]

        K_sum = ca.Function(
            'covariance',
            [x, x_bar, *hyperparameters],
            [K_1 + K_2],
            ['x', 'x_bar', *hyperparameter_names],
            ['covariance']
        )

        return K_sum


class Product(KernelOperator):
    """
    Product operator for covariance functions

    :param kernel_1:
    :type kernel_1:
    :param kernel_2:
    :type kernel_2:
    """
    def get_covariance_function(self, x: ca.SX, x_bar: ca.SX, active_dims: np.ndarray) -> ca.Function:
        """

        :param x:
        :param x_bar:
        :param active_dims:
        :return:
        """
        K_1 = self.kernel_1(x, x_bar)
        K_2 = self.kernel_2(x, x_bar)

        hyperparameters = [parameter.SX for parameter in self.hyperparameters]
        hyperparameter_names = [parameter.name for parameter in self.hyperparameters]

        K_prod = ca.Function(
            'covariance',
            [x, x_bar, *hyperparameters],
            [K_1 * K_2],
            ['x', 'x_bar', *hyperparameter_names],
            ['covariance']
        )

        return K_prod


class Power(KernelOperator):
    """
    Power operator for covariance functions

    :param kernel:
    :type kernel:
    :param power:
    :type power:
    """
    def __init__(self, kernel: Cov, power: Numeric) -> None:
        super().__init__(kernel, kernel_2=None)

        self.power = power

    def get_covariance_function(self, x: ca.SX, x_bar: ca.SX, active_dims: np.ndarray) -> ca.Function:
        """

        :param x:
        :param x_bar:
        :param active_dims:
        :return:
        """
        K = self.kernel_1(x)
        power = self.power

        hyperparameters = [parameter.SX for parameter in self.hyperparameters]
        hyperparameter_names = [parameter.name for parameter in self.hyperparameters]

        K_pow = ca.Function(
            'covariance',
            [x, x_bar, *hyperparameters],
            [K ** power],
            ['x', 'x_bar', *hyperparameter_names],
            ['covariance']
        )

        return K_pow


class Warp(KernelOperator):
    """"""


def _clean_input_matrices(X: Array, X_bar: Optional[Array] = None) -> (ca.SX, ca.SX, bool):
    """
    Arbitrary array-like input matrix/matrices are converted to CasADi SX symbols. In the case of a single input
    matrix the resulting SX symbols are disambiguated, meaning they have unique CasADi identifiers.

    :param X:
    :param X_bar:
    :return:
    """
    if not isinstance(X, (ca.SX, ca.MX, ca.DM)):
        X = np.atleast_2d(X)
        X = ca.SX.sym('X', *X.shape)

    X_is_X_bar = False
    if X_bar is None or X_bar is X:
        X_bar = ca.SX.sym('X_bar', *X.shape)
        X_is_X_bar = True

    if not isinstance(X_bar, (ca.SX, ca.MX, ca.DM)):
        X_bar = np.atleast_2d(X_bar)
        X_bar = ca.SX.sym('X_bar', *X_bar.shape)

    assert X.shape[0] == X_bar.shape[0], "X and X_bar do not have the same input space dimensions"

    return X, X_bar, X_is_X_bar


def _mahalanobis_distance_squared(x, y, S):
    """

    :param x:
    :param y:
    :param S:
    :return:
    """
    x_minus_y = x - y
    return x_minus_y.T @ S @ x_minus_y


__all__ = [
    'Kernel',
    'ConstantKernel',
    # 'GammaExponentialKernel',
    'SquaredExponentialKernel',
    'MaternKernel',
    'ExponentialKernel',
    'Matern32Kernel',
    'Matern52Kernel',
    'RationalQuadraticKernel',
    'PiecewisePolynomialKernel',
    'DotProductKernel',
    'PolynomialKernel',
    'LinearKernel',
    'NeuralNetworkKernel',
    'PeriodicKernel'
]
