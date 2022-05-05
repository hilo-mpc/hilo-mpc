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
from typing import Optional

import casadi as ca

from ..base import Base, Vector, TimeSeries
from ..dynamic_model.dynamic_model import Model
from ...util.util import convert


class Estimator(metaclass=ABCMeta):
    """"""
    _type = ''

    def __init__(self, *args, **kwargs):
        """Constructor method"""
        super().__init__(*args, **kwargs)
        self._update_type()

    @abstractmethod
    def _update_type(self) -> None:
        """

        :return:
        """
        pass

    @property
    def type(self) -> str:
        """

        :return:
        """
        return self._type


class _Estimator(Estimator, Base, metaclass=ABCMeta):
    """
    Base class for all estimators

    :param model:
    :param id: The identifier of the estimator. If no identifier is given, a random one will be generated.
    :param name: The name of the model. By default, the model has no name.
    :param plot_backend: Plotting library that is used to visualize estimated data. At the moment only Matplotlib and
        Bokeh are supported. By default, no plotting library is selected, i.e. no plots can be generated.
    """
    def __init__(
            self,
            model: Model,
            id: Optional[str] = None,
            name: Optional[str] = None,
            plot_backend: Optional[str] = None
    ) -> None:
        """Constructor method"""
        super().__init__(id=id, name=name)
        if self._id is None:
            self._create_id()

        if not model.is_setup():
            raise RuntimeError(f"Model is not set up. Run Model.setup() before passing it to the {self.type}.")

        self._model = model.copy(setup=True)
        self._error_covariance = None
        self._process_noise_covariance = None
        self._measurement_noise_covariance = None

        self._solution = TimeSeries(plot_backend, parent=self)

        self._n_x = 0
        self._n_y = 0
        self._n_z = 0
        self._n_u = 0
        self._n_p = 0
        self._n_p_est = 0

    def _set_process_noise(self, var):
        """

        :param var:
        :return:
        """
        Q = convert(var, ca.DM)
        if not Q.is_square():
            Q = ca.diag(Q)
        if Q.shape != (self._n_x, self._n_x):
            raise ValueError(f"Dimension mismatch. Supplied dimension is {Q.shape[0]}x{Q.shape[1]}, but required "
                             f"dimension is {self._n_x}x{self._n_x}")
        self._process_noise_covariance = Q

    def _set_measurement_noise(self, var):
        """

        :param var:
        :return:
        """
        R = convert(var, ca.DM)
        if not R.is_square():
            R = ca.diag(R)
        if R.shape != (self._n_y, self._n_y):
            raise ValueError(f"Dimension mismatch. Supplied dimension is {R.shape[0]}x{R.shape[1]}, but required "
                             f"dimension is {self._n_y}x{self._n_y}")
        self._measurement_noise_covariance = R

    def _set_error_covariance(self, P0):
        """

        :param P0:
        :return:
        """
        if P0 is None:
            P0 = ca.DM.eye(self._n_x)
        else:
            P0 = convert(P0, ca.DM)
            if not P0.is_square():
                P0 = ca.diag(P0)
            if P0.shape != (self._n_x, self._n_x):
                raise ValueError(f"Dimension mismatch. Supplied dimension is {P0.shape[0]}x{P0.shape[1]}, but required"
                                 f" dimension is {self._n_x}x{self._n_x}")
        self._solution.set('P', P0[:])

    def _check_setup(self):
        """

        :return:
        """
        if self._function is None:
            # NOTE: Not sure if we want to throw an error here
            type_ = self.type.split()
            type_[0] = type_[0].capitalize()
            msg = f"{' '.join(type_)} is not set up. Run {self.__class__.__name__}.setup() before running simulations."
            raise RuntimeError(msg)

        if self.n_x > 0 and self._solution.get_by_id('x').is_empty():
            raise RuntimeError(f"No initial guess for the states found. Please set initial guess before running the "
                               f"{self.type}!")

        if self._n_z > 0 and self._solution.get_by_id('z').is_empty():
            raise RuntimeError(f"No initial guess for the algebraic variables found. Please set initial guess before "
                               f"running the {self.type}!")

    def _process_inputs(self, **kwargs):
        """

        :param kwargs:
        :return:
        """
        dt = self._solution.dt
        if dt is not None:
            tf = kwargs.get('tf')
            if tf is None:
                steps = kwargs.get('steps', 1)
                tf = steps * dt
                # NOTE: Don't know if this is necessary
                tf = ca.linspace(dt, tf, steps).T
            else:
                steps = int(tf / dt)

                if self._solution['t'].is_empty():
                    self._solution.add('t', 0.)
        else:
            raise NotImplementedError("Support for grids is not yet implemented in the Kalman filter.")

        y = kwargs.get('y')
        u = kwargs.get('u')
        p = kwargs.get('p')
        if y is not None:
            if not isinstance(y, Vector):
                y = convert(y, ca.DM)
            if u is not None:
                if not isinstance(u, Vector):
                    u = convert(u, ca.DM)
            if p is not None:
                if not isinstance(p, Vector):
                    p = convert(p, ca.DM)
            if steps is not None:
                if y.size2() != 1 and steps != y.size2():
                    raise IndexError(f"Dimension mismatch for variable y. Supplied dimension is {y.size2()}, but "
                                     f"required dimension is {steps}.")
                elif y.size2() == 1 and steps != 1:
                    if isinstance(y, Vector):
                        y.repmat(steps, axis=1)
                    else:
                        y = ca.repmat(y, 1, steps)
            else:
                steps = y.size2()
            if u is not None:
                if u.size2() != 1 and steps != u.size2():
                    raise IndexError(f"Dimension mismatch for variable u. Supplied dimension is {u.size2()}, but "
                                     f"required dimension is {steps}.")
                elif u.size2() == 1 and steps != 1:
                    if isinstance(u, Vector):
                        u.repmat(steps, axis=1)
                    else:
                        u = ca.repmat(u, 1, steps)
                self._solution.add('u', u)
            if p is not None:
                if p.size2() != 1 and steps != p.size2():
                    raise IndexError(f"Dimension mismatch for variable p. Supplied dimension is {p.size2()}, but "
                                     f"required dimension is {steps}.")
                elif p.size2() == 1 and steps != 1:
                    if isinstance(p, Vector):
                        p.repmat(steps, axis=1)
                    else:
                        p = ca.repmat(p, 1, steps)
                self._solution.add('p', p)
        else:
            raise RuntimeError("No measurement data supplied.")

        skip = kwargs.get('skip')
        args = self._solution.get_function_args(steps=steps, skip=skip)
        args['t0'] += tf
        args['steps'] = steps
        # TODO: Add to solution like u and p?
        args['y'] = y

        return args

    @property
    def error_covariance(self):
        """

        :return:
        """
        if self._solution.is_set_up() and not self._solution.get_by_id('P').is_empty():
            return ca.reshape(self._solution['Pf'], self._n_x, self._n_x)
        return None

    P = error_covariance

    @property
    def process_noise_covariance(self):
        """

        :return:
        """
        return self._process_noise_covariance

    @process_noise_covariance.setter
    def process_noise_covariance(self, cov):
        self._set_process_noise(cov)

    Q = process_noise_covariance

    @property
    def measurement_noise_covariance(self):
        """

        :return:
        """
        return self._measurement_noise_covariance

    @measurement_noise_covariance.setter
    def measurement_noise_covariance(self, cov):
        self._set_measurement_noise(cov)

    R = measurement_noise_covariance

    @property
    def n_x(self):
        """

        :return:
        """
        return self._n_x

    @property
    def n_y(self):
        """

        :return:
        """
        return self._n_y

    @property
    def n_z(self):
        """

        :return:
        """
        return self._n_z

    @property
    def n_u(self):
        """

        :return:
        """
        return self._n_u

    @property
    def n_p(self):
        """

        :return:
        """
        return self._n_p

    @property
    def n_p_est(self):
        """

        :return:
        """
        return self._n_p_est

    @property
    def solution(self):
        """

        :return:
        """
        return self._solution

    def set_initial_guess(self, x0, t0=0., z0=None, P0=None):
        """

        :param x0:
        :param t0:
        :param z0:
        :param P0:
        :return:
        """
        self._model.set_initial_conditions(x0, t0=t0, z0=z0)
        self._solution.set('t', self._model.solution.get_by_id('t:0'))
        if not self._model.x.is_empty():
            self._solution.set('x', self._model.solution.get_by_id('x:0'))
        if not self._model.z.is_empty():
            self._solution.set('z', self._model.solution.get_by_id('z:0'))
        if not self._model.y.is_empty():
            self._solution.set('y', self._model.solution.get_by_id('y:0'))

        self._set_error_covariance(P0)

    def set_initial_parameter_values(self, p):
        """

        :param p:
        :return:
        """
        self._model.set_initial_parameter_values(p)
        if not self._model.p.is_empty():
            self._solution.add('p', p)

    @abstractmethod
    def setup(self, **kwargs):
        """

        :param kwargs:
        :return:
        """
        pass
