#
#   This file is part of HILO-MPC
#
#   HILO-MPC is toolbox for easy, flexible and fast development of machine-learning supported
#   optimal control and estimation problems
#
#   Copyright (c) 2021 Johannes Pohlodek, Bruno Morabito, Rofl Findeisen
#                      All rights reserved
#
#   HILO-MPC is free software: you can redistribute it and/or modify
#   it under the terms of the GNU Lesser General Public License as
#   published by the Free Software Foundation, either version 3
#   of the License, or (at your option) any later version.
#
#   HILO-MPC is distributed in the hope that it will be useful,
#   but WITHOUT ANY WARRANTY; without even the implied warranty of
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#   GNU Lesser General Public License for more details.
#
#   You should have received a copy of the GNU General Public License
#   along with HILO-MPC. If not, see <http://www.gnu.org/licenses/>.

from laplace import Laplace, ParametricLaplace
import torch
from torch.optim import Adam

from ...util.machine_learning import net_to_casadi_graph


class _LaplaceWrapper:
    """"""
    def __init__(self, net, **kwargs):
        """Constructor method"""
        likelihood = kwargs.get('likelihood')
        if likelihood is None:
            likelihood = 'regression'

        subset_of_weights = kwargs.get('subset_of_weights')
        if subset_of_weights is None:
            subset_of_weights = 'all'

        hessian_structure = kwargs.get('hessian_structure')
        if hessian_structure is None:
            hessian_structure = 'full'

        hyp_lr = kwargs.get('hyperparameter_learning_rate')
        if hyp_lr is None:
            hyp_lr = .1
        log_prior, log_sigma = torch.ones(1, requires_grad=True), torch.ones(1, requires_grad=True)
        self._hyper_optimizer = Adam([log_prior, log_sigma], lr=hyp_lr)
        self._log_prior = log_prior
        self._log_sigma = log_sigma

        hyp_steps = kwargs.get('hyperparameter_steps')
        if hyp_steps is None:
            hyp_steps = 100
        self._hyp_steps = hyp_steps

        self._module = Laplace(net.module, likelihood, subset_of_weights=subset_of_weights,
                               hessian_structure=hessian_structure)
        self._net = net

    @property
    def module(self) -> ParametricLaplace:
        """

        :return:
        """
        return self._module

    def train(
            self,
            data,
            validation_data,
            batch_size,
            epochs,
            verbose,
            patience,
            shuffle
    ):
        """

        :param data:
        :param validation_data:
        :param batch_size:
        :param epochs:
        :param verbose:
        :param patience:
        :param shuffle:
        :return:
        """
        self._net.train(data, validation_data, batch_size, epochs, verbose, patience, shuffle)
        self._module.fit(self._net.train_loader)

        for _ in range(self._hyp_steps):
            self._hyper_optimizer.zero_grad()
            neg_marg_lik = -self._module.log_marginal_likelihood(prior_precision=self._log_prior.exp(),
                                                                 sigma_noise=self._log_sigma.exp())
            neg_marg_lik.backward()
            self._hyper_optimizer.step()

    def evaluate(self, data=None, batch_size=None, epoch=1, verbose=1):
        """

        :param data:
        :param batch_size:
        :param epoch:
        :param verbose:
        :return:
        """
        self._net.evaluate(data=data, batch_size=batch_size, epoch=epoch, verbose=verbose)

    def build_graph(self, x, layers, input_scaling=None, output_scaling=None):
        """

        :param x:
        :param layers:
        :param input_scaling:
        :param output_scaling:
        :return:
        """
        f_mu = net_to_casadi_graph(self, x, layers, input_scaling=input_scaling)

    def get_weights_and_bias(self):
        """

        :return:
        """
        return self._net.get_weights_and_bias()


def get_wrapper(net, **kwargs):
    """

    :param net:
    :param kwargs:
    :return:
    """
    return _LaplaceWrapper(net, **kwargs)


__all__ = [
    'get_wrapper'
]
