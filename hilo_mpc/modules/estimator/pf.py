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

from typing import Callable, Optional
import warnings

import casadi as ca
import numpy as np
from scipy.stats import norm

from .base import _Estimator
from ..dynamic_model.dynamic_model import Model
from ...util.util import convert


class ParticleFilter(_Estimator):
    """
    Particle filter (PF) class for state estimation

    :param model:
    :param id: The identifier of the PF object. If no identifier is given, a random one will be generated.
    :param name: The name of the PF object. By default the PF object has no name.
    :param plot_backend: Plotting library that is used to visualize estimated data. At the moment only
        `Matplotlib <https://matplotlib.org/>`_ and `Bokeh <https://bokeh.org/>`_ are supported. By default no plotting
        library is selected, i.e. no plots can be generated.
    :param variant:
    :param roughening:
    :param prior_editing:
    :param kwargs:
    """
    def __init__(
            self,
            model: Model,
            id: Optional[str] = None,
            name: Optional[str] = None,
            plot_backend: Optional[str] = None,
            variant: Optional[str] = None,
            roughening: bool = False,
            prior_editing: bool = False,
            **kwargs
    ):
        """Constructor method"""
        if model.is_linear():
            warnings.warn("The supplied model is linear. For better efficiency use an observer targeted at the "
                          "estimation of linear systems.")

        super().__init__(model, id=id, name=name, plot_backend=plot_backend)

        self._variant = variant
        self._roughening = roughening
        self._prior_editing = prior_editing
        if self._roughening or self._prior_editing:
            K = kwargs.get('K')
            if K is None:
                K = .2
            self._roughening_tuning_param = K

        self._setup_normpdf()
        self._sample_size = 15
        self._pdf = lhsnorm
        self._transpose_pdf = None

    def _update_type(self) -> None:
        """

        :return:
        """
        self._type = 'particle filter'

    def _setup_normpdf(self):
        """

        :return:
        """
        x = ca.SX.sym('x')
        mu = ca.SX.sym('mu')
        sigma = ca.SX.sym('sigma')

        y = ca.exp(-.5 * ((x - mu) / sigma) ** 2) / (ca.sqrt(2 * ca.pi) * sigma)

        self._normpdf = ca.Function('normpdf', [x, mu, sigma], [y])

    def _propagate_particles(self, n_samples):
        """

        :param n_samples:
        :return:
        """
        n_x = self._model.n_x
        n_y = self._model.n_y
        n_u = self._model.n_u
        n_p = self._model.n_p

        X = ca.MX.sym('X', n_x, n_samples)
        u = ca.MX.sym('u', n_u)
        p = ca.MX.sym('p', n_p)
        up = ca.vertcat(u, p)

        self._solution.setup('X', X={
            'values_or_names': [X.name() + '_' + str(k) for k in range(X.numel())],
            'description': X.numel() * [''],
            'labels': X.numel() * [''],
            'units': X.numel() * [''],
            'shape': (X.numel(), 0),
            'data_format': ca.DM
        })

        sol = self._model(x0=X, p=up)
        X_prop = sol['xf']
        if n_y == 0:
            warnings.warn(f"The model has no measurement equations, I am assuming measurements of all states "
                          f"{self._model.dynamical_state_names} are available.")
            Y = X_prop
        else:
            Y = sol['yf']

        w = ca.MX.sym('w', n_x, n_samples)
        v = ca.MX.sym('v', n_y, n_samples)

        # NOTE: At the moment only additive noise is supported
        X_prop += w
        Y += v

        self._predict_function = ca.Function('propagation_step',
                                             [X, up, w, v],
                                             [X_prop, Y],
                                             ['X', 'p', 'w', 'v'],
                                             ['X_prop', 'Y'])

    def _evaluate_likelihood(self, n_samples):
        """"""
        n_y = self._model.n_y

        y = ca.SX.sym('y', n_y)
        Y = ca.SX.sym('Y', n_y, n_samples)
        R = ca.SX.sym('R', (n_y, n_y))

        q = self._normpdf(Y, y, ca.sqrt(R))  # Since R is usually a diagonal matrix, we can use ca.sqrt() here. We need
        # to change this, if we have non-diagonal matrices, i.e. covariance entries.
        q /= ca.sum2(q)

        self._update_function = ca.Function('likelihood',
                                            [y, Y, R],
                                            [q],
                                            ['y', 'Y', 'R'],
                                            ['q'])

    def _initial_sample(self) -> None:
        """

        :return:
        """
        P0 = ca.reshape(self._solution.get_by_id('P:f'), self._n_x, self._n_x)
        X = self._pdf(self._solution.get_by_id('x:0').full().flatten(), P0, self._sample_size)
        if self._transpose_pdf:
            X = X.T
        elif self._transpose_pdf is None:
            if X.shape != (2, self._sample_size):
                X = X.T
                if X.shape != (2, self._sample_size):
                    raise ValueError(f"Dimension mismatch. Expected dimension 2x{self._sample_size}, got "
                                     f"{X.shape[1]}x{X.shape[0]}.")
                self._transpose_pdf = True
            else:
                self._transpose_pdf = False
        X = convert(X, ca.DM)
        self._solution.set('X', X[:])

    @property
    def probability_density_function(self) -> Callable[[np.ndarray, np.ndarray, int], np.ndarray]:
        """

        :return:
        """
        return self._pdf

    @probability_density_function.setter
    def probability_density_function(self, pdf: Callable[[np.ndarray, np.ndarray, int], np.ndarray]) -> None:
        """

        :param pdf:
        :return:
        """
        if not callable(pdf):
            raise ValueError(f"Probability density function of the {self.type} needs to be callable.")
        annotations = pdf.__annotations__
        run_function = True
        if annotations:
            if len(annotations) == 4 and 'return' in annotations:
                types = [np.ndarray, np.ndarray, int, np.ndarray]
                args = ["mean", "covariance", "sample size"]
                for k, type_ in enumerate(annotations.values()):
                    if type_ is not types[k]:
                        if k == 0:
                            no = "1st"
                        elif k == 1:
                            no = "2nd"
                        else:
                            no = "3rd"
                        if k < 3:
                            raise TypeError(f"The {no} argument to the probability density function (pdf) needs to be "
                                            f"the '{args[k]}' with type {types[k].__name__}.")
                        else:
                            raise TypeError(f"The return value of the probability density function (pdf) needs to be a"
                                            f" 'random sample' with type {types[k].__name__}.")
                run_function = False
        if run_function:
            try:
                X = pdf(np.zeros(self._n_x), np.eye(self._n_x), self._sample_size)
                if X.shape != (self._n_x, self._sample_size):
                    if X.shape != (self._sample_size, self._n_x):
                        raise ValueError(f"Dimension mismatch. Expected dimension {self._n_x}x{self._sample_size}, got "
                                         f"{X.shape[0]}x{X.shape[1]}.")
                    self._transpose_pdf = True
                else:
                    self._transpose_pdf = False
            except Exception as err:
                raise RuntimeError(f"The following exception was raised\n"
                                   f"   {type(err).__name__}: '{err.args[0]}'.\nPlease make sure that the "
                                   f"supplied probability density function (pdf) has the following arguments\n"
                                   f"   mu - mean of the pdf (type: numpy.ndarray),\n"
                                   f"   sigma - covariance of the mean (type: numpy.ndarray),\n"
                                   f"   n - sample size (type: int),\n"
                                   f"and the following return value\n"
                                   f"   X - random sample (type: numpy.ndarray).")

        self._pdf = pdf

    pdf = probability_density_function

    @property
    def variant(self) -> str:
        """

        :return:
        """
        return self._variant

    @variant.setter
    def variant(self, variant):
        self._variant = variant

    @property
    def sample_size(self):
        """

        :return:
        """
        return self._sample_size

    @sample_size.setter
    def sample_size(self, sample_size):
        self._sample_size = sample_size

    n_samples = sample_size

    def setup(self, **kwargs) -> None:
        """

        :param kwargs:
        :return:
        """
        self._solution.setup(self._model.solution)

        n_s = kwargs.get('n_samples')
        if n_s is None:
            n_s = self._sample_size
        else:
            self._sample_size = n_s

        self._propagate_particles(n_s)
        self._evaluate_likelihood(n_s)

        n_x = self._model.n_x
        n_y = self._model.n_y
        n_u = self._model.n_u
        n_p = self._model.n_p
        n_P = n_x * n_x

        self._solution.setup('P', P={
            'values_or_names': ['P_' + str(k) for k in range(n_P)],
            'description': n_P * [''],
            'labels': n_P * [''],
            'units': n_P * [''],
            'shape': (n_P, 0),
            'data_format': ca.DM
        })

        X = ca.MX.sym('X', n_x, n_s)
        y = ca.MX.sym('y', n_y)
        u = ca.MX.sym('u', n_u)
        p = ca.MX.sym('p', n_p)
        up = ca.vertcat(u, p)
        w = ca.MX.sym('w', n_x, n_s)
        v = ca.MX.sym('v', n_y, n_s)
        Q = ca.MX.sym('Q', (n_x, n_x))
        R = ca.MX.sym('R', (n_y, n_y))

        prediction = self._predict_function(X=X, p=up, w=w, v=v)
        X_prop = prediction['X_prop']
        Y = prediction['Y']
        update = self._update_function(y=y, Y=Y, R=R)

        self._function = ca.Function('function',
                                     [X, y, up, w, v, R],
                                     [X_prop, Y, update['q']],
                                     ['X', 'y', 'p', 'w', 'v', 'R'],
                                     ['X_prop', 'Y', 'q'])

        self._n_x = n_x
        self._n_y = n_y
        # self._n_z = n_z
        self._n_u = n_u
        self._n_p = n_p
        # self._n_p_est = n_p_est

        self._process_noise_covariance = ca.DM.zeros(Q.shape)
        self._measurement_noise_covariance = ca.DM.zeros(R.shape)

    def estimate(self, *args, **kwargs):
        """

        :param args:
        :param kwargs:
        :return:
        """
        # TODO: Adjust according to Model.simulate
        self._check_setup()

        kwargs['skip'] = ['x']

        args = self._process_inputs(**kwargs)
        tf = args.pop('t0')
        steps = args.pop('steps')
        if self._solution.get_by_id('X').is_empty():
            self._initial_sample()
        args['X'] = ca.reshape(self._solution.get_by_id('X:f'), self._n_x, self._sample_size)

        if steps > 1:
            raise NotImplementedError("Particle filter is not yet implemented with steps > 1.")
        else:
            R = self._measurement_noise_covariance
            w = self._pdf(np.zeros(self._n_x), self._process_noise_covariance.full(), self._sample_size)
            # v = self._pdf(np.zeros(self._n_y), self._measurement_noise_covariance.full(), self._sample_size)
            v = ca.sqrt(R) @ np.random.randn(self._n_y, self._sample_size)  # Only possible since R is usually a
            # diagonal matrix (see self._evaluate_likelihood)
            if self._transpose_pdf:
                w = w.T
                # v = v.T
            args['w'] = w
            args['v'] = v
            args['R'] = R

            result = self._function(**args)
            Y = result['Y']

            # Prior editing
            if self._prior_editing:
                mag = args['y'] - Y
                need_roughening = ca.fabs(mag) > 6 * ca.sqrt(R)
                n_r = int(ca.sum2(need_roughening))
                while n_r > 0:
                    # TODO: Check out prior editing in more detail. Should we only use the X that are indexed by
                    #  need_roughening for the calculation of dx? Do we roughen already roughened X, or do we just
                    #  replace the ones that were improved by the roughening and keep the other ones at their original
                    #  value?
                    dx = np.max(args['X'], axis=1) - np.min(args['X'], axis=1)
                    dx = self._pdf(np.zeros(self._n_x), self._roughening_tuning_param * np.diag(dx) *
                                   n_r ** (-1 / self._n_x), n_r)
                    if self._transpose_pdf:
                        dx = dx.T
                    mask = np.where(need_roughening == 1)[1]
                    args['X'][:, mask] += dx

                    result = self._function(**args)

                    Y = result['Y']
                    mag = args['y'] - Y
                    need_roughening = ca.fabs(mag) > 6 * ca.sqrt(R)
                    n_r = int(ca.sum2(need_roughening))

            X = result['X_prop']
            q = result['q']

            # Resample (Survival of the fittest)
            ind = np.random.choice(self._sample_size, size=self._sample_size, replace=True, p=q.full().flatten())
            X = X[:, ind]
            Y = Y[:, ind]

            # Roughening
            if self._roughening:
                dx = np.max(X, axis=1) - np.min(X, axis=1)
                dx = self._pdf(np.zeros(self._n_x), self._roughening_tuning_param * np.diag(dx) *
                               self._sample_size ** (-1 / self._n_x), self._sample_size)
                if self._transpose_pdf:
                    dx = dx.T
                X += dx

            x = ca.sum2(X) / self._sample_size
            y = ca.sum2(Y) / self._sample_size
            P = convert(np.cov(X), ca.DM)
        self._solution.update(t=tf, x=x, X=X[:], P=P[:], y=y)


def lhsnorm(mu, sigma, n):
    """

    :param mu:
    :param sigma:
    :param n:
    :return:
    """
    # TODO: Migrate to somewhere in util
    n_m = mu.size
    z = np.random.multivariate_normal(mu, sigma, size=n)
    x = np.zeros_like(z, dtype=z.dtype)

    idz = np.argsort(z, axis=0)
    for k in range(n_m):
        x[idz[:, k], k] = np.linspace(1, n, n)
    x -= np.random.rand(*x.shape)
    x /= n

    for k in range(n_m):
        x[:, k] = norm.ppf(x[:, k], loc=mu[k], scale=np.sqrt(sigma[k, k]))

    return x


__all__ = [
    'ParticleFilter'
]
