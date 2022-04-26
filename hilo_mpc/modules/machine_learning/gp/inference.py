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
from typing import Dict, Optional, TypeVar, Union
import warnings

import casadi as ca
import numpy as np

from .likelihood import Likelihood
from .mean import Mean
from .kernel import Kernel


Numeric = Union[int, float]
Symbolic = TypeVar('Symbolic', ca.SX, ca.MX)
Array = TypeVar('Array', ca.SX, ca.MX, np.ndarray)
Lik = TypeVar('Lik', bound=Likelihood)
Mu = TypeVar('Mu', bound=Mean)
Cov = TypeVar('Cov', bound=Kernel)


class Inference(metaclass=ABCMeta):
    """"""
    def __init__(self):
        """Constructor method"""
        self._posterior = {
            'mean': ca.SX(),
            'var': ca.SX(),
            'log_marginal_likelihood': ca.SX()
        }

    def __call__(self, *args, **kwargs) -> Optional[Dict[str, Symbolic]]:
        """Calling method"""
        if args and kwargs:
            raise TypeError("Call to Inference can only accept positional or keyword arguments, not both")

        if args:
            X = args[0]
            y = args[1]
            x_test = args[2]  # test input
            noise_variance = args[3]
            likelihood = args[4]
            mean = args[5]
            kernel = args[6]
        elif kwargs:
            X = kwargs.get('X')
            y = kwargs.get('y')
            x_test = kwargs.get('x_test')
            noise_variance = kwargs.get('noise_variance')
            likelihood = kwargs.get('likelihood')
            mean = kwargs.get('mean')
            kernel = kwargs.get('kernel')

            if X is None:
                raise ValueError("No vector X was supplied")
            if y is None:
                raise ValueError("No vector y was supplied")
            if x_test is None:
                raise ValueError("No test input was supplied")
            if noise_variance is None:
                raise ValueError("No noise variance was supplied")
            if likelihood is None:
                raise ValueError("No likelihood was supplied")
            if mean is None:
                raise ValueError("No mean function was supplied")
            if kernel is None:
                raise ValueError("No covariance function (kernel) was supplied")
        else:
            warnings.warn("No arguments supplied")
            return

        self.get_posterior(X, y, x_test, noise_variance, likelihood, mean, kernel)
        return self._posterior

    @property
    def posterior(self) -> Dict[str, Symbolic]:
        """

        :return:
        """
        return self._posterior

    @abstractmethod
    def get_posterior(
            self,
            X: Array,
            y: Array,
            x_test: Array,
            noise_variance: Union[Symbolic, Numeric],
            likelihood: Lik,
            mean: Mu,
            kernel: Cov
    ) -> None:
        """

        :param X:
        :param y:
        :param x_test:
        :param noise_variance:
        :param likelihood:
        :param mean:
        :param kernel:
        :return:
        """
        pass

    @staticmethod
    def exact():
        """

        :return:
        """
        return ExactInference()

    @staticmethod
    def laplace():
        """

        :return:
        """
        return Laplace()

    @staticmethod
    def expectation_propagation():
        """

        :return:
        """
        return ExpectationPropagation()

    @staticmethod
    def variational_bayes():
        """

        :return:
        """
        return VariationalBayes()

    @staticmethod
    def kullback_leibler():
        """

        :return:
        """
        return KullbackLeibler()


class ExactInference(Inference):
    """"""
    def get_posterior(
            self,
            X: Array,
            y: Array,
            x_test: Array,
            noise_variance: Union[Symbolic, Numeric],
            likelihood: Lik,
            mean: Mu,
            kernel: Cov
    ) -> None:
        """

        :param X:
        :param y:
        :param x_test:
        :param noise_variance:
        :param likelihood:
        :param mean:
        :param kernel:
        :return:
        """
        if likelihood.name != "Gaussian":
            raise ValueError("Exact inference is only applicable with Gaussian likelihood. Choose a different inference"
                             " method in order to use other likelihoods.")

        n, D = X.shape

        noise_variance = ca.exp(2 * noise_variance)

        K = kernel(X, X)
        prior_mu = mean(X)

        # NOTE: CasADi's Cholesky decomposition returns an upper triangular matrix, but we need a lower triangular
        #  matrix, so we transpose 'L' in the calculation of 'alpha'
        L = ca.chol(noise_variance * ca.SX.eye(D) + K)
        y_minus_prior = y - prior_mu
        alpha = ca.solve(L, ca.solve(L.T, y_minus_prior.T))  # \alpha = L^T\(L\y) (see Rasmussen p.19)

        log_marginal_likelihood = -.5 * y_minus_prior @ alpha - ca.sum1(ca.log(ca.diag(L))) - D / 2 * ca.log(2 * ca.pi)

        K = kernel(X, x_test)
        mu = mean(x_test) + K.T @ alpha

        v = ca.solve(L.T, K)  # v = L\k_* (see Rasmussen p.19)
        K = kernel(x_test, x_test)
        var = K - v.T @ v

        self._posterior['mean'] = mu
        self._posterior['var'] = var
        self._posterior['log_marginal_likelihood'] = log_marginal_likelihood


class Laplace(Inference):
    """"""
    def __init__(self):
        """Constructor method"""
        super().__init__()

        raise NotImplementedError("Laplace inference not yet implemented")


class ExpectationPropagation(Inference):
    """"""
    def __init__(self):
        """Constructor method"""
        super().__init__()

        raise NotImplementedError("Expectation propagation inference not yet implemented")


class VariationalBayes(Inference):
    """"""
    def __init__(self):
        """Constructor method"""
        super().__init__()

        raise NotImplementedError("Variational Bayes inference not yet implemented")


class KullbackLeibler(Inference):
    """"""
    def __init__(self):
        """Constructor method"""
        super().__init__()

        raise NotImplementedError("Kullback-Leibler inference not yet implemented")
