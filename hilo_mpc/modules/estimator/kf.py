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
from typing import Optional, Union
import warnings

import casadi as ca
import numpy as np

from .base import _Estimator
from ..dynamic_model.dynamic_model import Model


class _KalmanFilter(_Estimator, metaclass=ABCMeta):
    """
    Kalman filter base class

    :param model:
    :param id: The identifier of the KF object. If no identifier is given, a random one will be generated.
    :param name: The name of the KF object. By default the KF object has no name.
    :param plot_backend: Plotting library that is used to visualize estimated data. At the moment only
        `Matplotlib <https://matplotlib.org/>`_ and `Bokeh <https://bokeh.org/>`_ are supported. By default, no plotting
        library is selected, i.e. no plots can be generated.
    :param square_root_form: Not used at the moment (will be implemented in the future)
    """
    def __init__(
            self,
            model: Model,
            id: Optional[str] = None,
            name: Optional[str] = None,
            plot_backend: Optional[str] = None,
            square_root_form: bool = True
    ) -> None:
        """Constructor method"""
        super().__init__(model, id=id, name=name, plot_backend=plot_backend)

        self._square_root_form = square_root_form
        self._predict_function = None
        self._update_function = None
        self._is_linearized = False

    @abstractmethod
    def _setup_parameters(self) -> None:
        """

        :return:
        """
        pass

    def _setup_predict(self, *args: Optional[ca.Function]) -> None:
        """

        :param args:
        :return:
        """
        state_matrix = args[0]

        n_x = self._model.n_x
        n_u = self._model.n_u
        n_p = self._model.n_p

        x = ca.MX.sym('x', n_x)
        u = ca.MX.sym('u', n_u)
        p = ca.MX.sym('p', n_p)

        P = ca.SX.sym('P', (n_x, n_x))
        Q = ca.SX.sym('Q', (n_x, n_x))
        if self._is_linearized:
            # NOTE: For now we ignore time-variant systems (i.e., [] instead of t)
            F = state_matrix(self._model.x, [], self._model.u, self._model.p, self._solution.dt, [])
        else:
            # NOTE: For now we ignore time-variant systems (i.e., [] instead of t)
            F = state_matrix(self._model.p, self._solution.dt, [])
        if self._model.discrete:
            Pdot = F @ P @ F.T + Q
        else:
            Pdot = F @ P + P @ F.T + Q

        model = self._model.copy(setup=False)

        model.add_dynamical_states(P[:])
        model.add_dynamical_equations(Pdot[:])
        model.add_parameters(Q[:])
        if self._solution.dt is not None:
            model.setup(dt=self._solution.dt)
        elif self._solution.grid is not None:
            model.setup(grid=self._solution.grid)
        else:
            model.setup()

        self._solution.setup('P', P={
            'values_or_names': [k.name() for k in P.elements()],
            'description': P.numel() * [''],
            'labels': P.numel() * [''],
            'units': P.numel() * [''],
            'shape': (P.numel(), 0),
            'data_format': ca.DM
        })

        P = ca.MX.sym('P', P.shape)
        Q = ca.MX.sym('Q', Q.shape)

        sol = model(x0=ca.vertcat(x, P[:]), p=ca.vertcat(u, p, Q[:]))
        result = sol['xf']
        x_pred = result[:n_x]
        P_pred = ca.reshape(result[n_x:], n_x, n_x)

        self._predict_function = ca.Function('prediction_step',
                                             [ca.horzcat(x, P), ca.vertcat(u, p), Q],
                                             [ca.horzcat(x_pred, P_pred)],
                                             ['x0', 'p', 'Q'],
                                             ['x'])

    def _setup_update(self, *args: Optional[ca.Function]) -> None:
        """

        :param args:
        :return:
        """
        output_matrix = args[0]

        n_x = self._model.n_x
        n_y = self._model.n_y
        n_u = self._model.n_u
        n_p = self._model.n_p

        x = ca.MX.sym('x', n_x)
        u = ca.MX.sym('u', n_u)
        p = ca.MX.sym('p', n_p)
        up = ca.vertcat(u, p)
        P = ca.MX.sym('P', (n_x, n_x))

        if n_y == 0:
            y_pred = x
            n_y = n_x
        else:
            sol = self._model(x0=x, p=up, which='meas_function')
            y_pred = sol['yf']
        y = ca.MX.sym('y', n_y)

        if self._is_linearized:
            # NOTE: For now we ignore time-variant systems (i.e., [] instead of t)
            H = output_matrix(x, [], u, p, self._solution.dt, [])
        else:
            # NOTE: For now we ignore time-variant systems (i.e., [] instead of t)
            H = output_matrix(p, self._solution.dt, [])
        R = ca.MX.sym('R', (n_y, n_y))
        P_xy = P @ H.T
        P_yy = H @ P @ H.T + R
        # NOTE: Since matrices cannot really be divided (the "/"-operator divides matrices element-wise), a linear
        #  system of equations needs to be solved (solve() in CasADi, which corresponds to the "\"-operator in MATLAB)
        # K = P_xy*P_yy^-1
        # K*P_yy = P_xy
        # P_yy^T*K^T = P_xy^T
        # K^T = P_yy^T\P_xy^T
        K = ca.solve(P_yy.T, P_xy.T).T

        x_up = x + K @ (y - y_pred)
        P_up = P - K @ P_yy @ K.T

        self._update_function = ca.Function('update_step',
                                            [ca.horzcat(x, P), y, up, R],
                                            [ca.horzcat(x_up, P_up), y_pred],
                                            ['x0', 'y', 'p', 'R'],
                                            ['x', 'y_pred'])

    def check_consistency(self) -> None:
        """

        :return:
        """
        super().check_consistency()

        if self._predict_function is not None:
            if self._predict_function.has_free():
                free_vars = ", ".join(self._predict_function.get_free())
                msg = f"Instance {self.__class__.__name__} has the following free variables/parameters: {free_vars}"
                raise RuntimeError(msg)

        if self._update_function is not None:
            if self._update_function.has_free():
                free_vars = ", ".join(self._update_function.get_free())
                msg = f"Instance {self.__class__.__name__} has the following free variables/parameters: {free_vars}"
                raise RuntimeError(msg)

    def setup(self, **kwargs) -> None:
        """

        :param kwargs:
        :return:
        """
        self._solution.setup(self._model.solution)

        if not self._model.is_linear() and self.type == 'extended Kalman filter':
            model = self._model.linearize()
            self._is_linearized = True
        else:
            model = self._model.copy(setup=False)
            self._is_linearized = False

        n_y = model.n_y
        if n_y == 0:
            warnings.warn(f"The model has no measurement equations, I am assuming measurements of all states "
                          f"{self._model.dynamical_state_names} are available.")
            model.set_measurement_equations(model.dynamical_states)

        if self.type != 'unscented Kalman filter':
            arg_in = [model.p, model.dt, model.t]
            if self._is_linearized:
                arg_in = [model.x_eq, model.z_eq, model.u_eq] + arg_in
            state_matrix = ca.Function('state_matrix', arg_in, [model.A])
            output_matrix = ca.Function('output_matrix', arg_in, [model.C])
            predict_args = (state_matrix,)
            update_args = (output_matrix,)
        else:
            predict_args = tuple()
            update_args = tuple()

        n_x = self._model.n_x
        n_y = self._model.n_y
        n_u = self._model.n_u
        n_p = self._model.n_p

        x = ca.MX.sym('x', n_x)
        y = ca.MX.sym('y', n_y)
        u = ca.MX.sym('u', n_u)
        p = ca.MX.sym('p', n_p)
        up = ca.vertcat(u, p)
        P = ca.MX.sym('P', (n_x, n_x))
        Q = ca.MX.sym('Q', (n_x, n_x))
        R = ca.MX.sym('R', (n_y, n_y))

        self._setup_parameters()
        self._setup_update(*update_args)
        self._setup_predict(*predict_args)

        prediction = self._predict_function(ca.horzcat(x, P), up, Q)
        update, y_pred = self._update_function(prediction, y, up, R)

        self._function = ca.Function('function',
                                     [ca.horzcat(x, P), y, up, Q, R],
                                     [update, y_pred],
                                     ['x0', 'y', 'p', 'Q', 'R'],
                                     ['x', 'y_pred'])

        self.check_consistency()

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

        args = self._process_inputs(**kwargs)
        tf = args.pop('t0')
        steps = args.pop('steps')
        # TODO: Maybe create a TimeSeries specifically tailored to estimation, where get_function_args(.) returns last
        #  error covariance as well
        args['x0'] = ca.horzcat(args['x0'], ca.reshape(self._solution.get_by_id('P:f'), self._n_x, self._n_x))

        if steps > 1:
            Q = ca.repmat(self._process_noise_covariance, 1, steps)
            R = ca.repmat(self._measurement_noise_covariance, 1, steps)
            args['Q'] = Q
            args['R'] = R
            function = self._function.mapaccum(steps)
            result = function(**args)
        else:
            args['Q'] = self._process_noise_covariance
            args['R'] = self._measurement_noise_covariance
            result = self._function(**args)
        self._solution.update(t=tf, x=result['x'][:, 0], P=result['x'][:, 1:][:], y=result['y_pred'])

    def predict(self, *args, **kwargs):
        """

        :param args:
        :param kwargs:
        :return:
        """
        return self._predict_function(*args, **kwargs)

    def update(self, *args, **kwargs):
        """

        :param args:
        :param kwargs:
        :return:
        """
        return self._update_function(*args, **kwargs)


class KalmanFilter(_KalmanFilter):
    """
    Kalman filter (KF) class for state estimation (parameter estimation will follow soon)

    :param model:
    :param id: The identifier of the KF object. If no identifier is given, a random one will be generated.
    :param name: The name of the KF object. By default the KF object has no name.
    :param plot_backend: Plotting library that is used to visualize estimated data. At the moment only
        `Matplotlib <https://matplotlib.org/>`_ and `Bokeh <https://bokeh.org/>`_ are supported. By default no plotting
        library is selected, i.e. no plots can be generated.
    :param square_root_form: Not used at the moment (will be implemented in the future)
    """
    def __init__(
            self,
            model: Model,
            id: Optional[str] = None,
            name: Optional[str] = None,
            plot_backend: Optional[str] = None,
            square_root_form: bool = True
    ) -> None:
        """Constructor method"""
        if not model.is_linear():
            raise ValueError("The supplied model is nonlinear. Please use an estimator targeted at the estimation of "
                             "nonlinear systems.")

        super().__init__(model, id=id, name=name, plot_backend=plot_backend, square_root_form=square_root_form)

    def _update_type(self) -> None:
        """

        :return:
        """
        self._type = 'Kalman filter'

    def _setup_parameters(self) -> None:
        """

        :return:
        """
        pass


class ExtendedKalmanFilter(_KalmanFilter):
    """
    Extended Kalman filter (EKF) class for state estimation (parameter estimation will follow soon)

    :param model:
    :param id: The identifier of the EKF object. If no identifier is given, a random one will be generated.
    :param name: The name of the EKF object. By default the EKF object has no name.
    :param plot_backend: Plotting library that is used to visualize estimated data. At the moment only
        `Matplotlib <https://matplotlib.org/>`_ and `Bokeh <https://bokeh.org/>`_ are supported. By default no plotting
        library is selected, i.e. no plots can be generated.
    :param square_root_form: Not used at the moment (will be implemented in the future)
    :note: The same methods and properties as for the :py:class:`Kalman filter <.KalmanFilter>` apply
    """
    def __init__(
            self,
            model: Model,
            id: Optional[str] = None,
            name: Optional[str] = None,
            plot_backend: Optional[str] = None,
            square_root_form: bool = True
    ) -> None:
        """Constructor method"""
        if model.is_linear():
            warnings.warn("The supplied model is linear. For better efficiency use an observer targeted at the "
                          "estimation of linear systems.")

        super().__init__(model, id=id, name=name, plot_backend=plot_backend, square_root_form=square_root_form)

    def _update_type(self) -> None:
        """

        :return:
        """
        self._type = "extended Kalman filter"

    def _setup_parameters(self) -> None:
        """

        :return:
        """
        pass


class UnscentedKalmanFilter(_KalmanFilter):
    """
    Unscented Kalman filter (UKF) class for state estimation (parameter estimation will follow soon)

    :param model:
    :param id: The identifier of the UKF object. If no identifier is given, a random one will be generated.
    :param name: The name of the UKF object. By default the UKF object has no name.
    :param alpha:
    :param beta:
    :param kappa:
    :param plot_backend: Plotting library that is used to visualize estimated data. At the moment only
        `Matplotlib <https://matplotlib.org/>`_ and `Bokeh <https://bokeh.org/>`_ are supported. By default no plotting
        library is selected, i.e. no plots can be generated.
    :param square_root_form: Not used at the moment (will be implemented in the future)
    """
    def __init__(
            self,
            model: Model,
            id: Optional[str] = None,
            name: Optional[str] = None,
            alpha: Union[int, float] = None,
            beta: Union[int, float] = None,
            kappa: Union[int, float] = None,
            plot_backend: Optional[str] = None,
            square_root_form: bool = True
    ) -> None:
        """Constructor method"""
        if model.is_linear():
            warnings.warn("The supplied model is linear. For better efficiency use an observer targeted at the "
                          "estimation of linear systems.")

        super().__init__(model, id=id, name=name, plot_backend=plot_backend, square_root_form=square_root_form)

        if alpha is None:
            alpha = .001
        self._check_parameter_bounds('alpha', alpha)
        if beta is None:
            beta = 2.
        self._check_parameter_bounds('beta', beta)
        if kappa is None:
            kappa = 0.
        self._check_parameter_bounds('kappa', kappa)
        self._lambda = None
        self._gamma = None

        self._weights = None
        self._sqrt = None

    def _update_type(self) -> None:
        """

        :return:
        """
        self._type = "unscented Kalman filter"

    def _check_parameter_bounds(self, param: str, value: Union[int, float]) -> None:
        """

        :param param:
        :param value:
        :return:
        """
        if param == 'alpha':
            if value <= 0. or value > 1.:
                raise ValueError(f"The parameter alpha needs to lie in the interval (0, 1]. Supplied alpha is {value}.")
            self._alpha = value
        elif param == 'beta':
            self._beta = value
        elif param == 'kappa':
            if value < 0:
                raise ValueError(f"The parameter kappa needs to be greater or equal to 0. Supplied kappa is {value}.")
            self._kappa = value

    def _setup_parameters(self) -> None:
        """

        :return:
        """
        n_x = self._model.n_x

        self._lambda = self._alpha ** 2 * (n_x + self._kappa) - n_x
        self._gamma = np.sqrt(n_x + self._lambda)

        weights = np.zeros((2, 2 * n_x + 1))
        weights[0, 0] = self._lambda / (n_x + self._lambda)
        weights[1, 0] = self._lambda / (n_x + self._lambda) + 1 - self._alpha ** 2 + self._beta
        weights[:, 1:] = 1 / (2 * (n_x + self._lambda))
        self._weights = weights

        a = ca.SX.sym('a', (n_x, n_x))
        self._sqrt = ca.Function('sqrt', [a], [ca.chol(a)])

    def _setup_predict(self, *args: Optional[ca.Function]) -> None:
        """

        :param args:
        :return:
        """
        n_x = self._model.n_x
        n_u = self._model.n_u
        n_p = self._model.n_p

        x = ca.MX.sym('x', n_x)
        u = ca.MX.sym('u', n_u)
        p = ca.MX.sym('p', n_p)
        up = ca.vertcat(u, p)

        P = ca.MX.sym('P', (n_x, n_x))
        Q = ca.MX.sym('Q', (n_x, n_x))
        P_sqrt = self._sqrt(P)
        X = [x]
        for k in range(n_x):
            X.append(x + self._gamma * P_sqrt[:, k])
        for k in range(n_x):
            X.append(x - self._gamma * P_sqrt[:, k])
        X = ca.horzcat(*X)

        self._solution.setup('P', P={
            'values_or_names': [P.name() + '_' + str(k) for k in range(P.numel())],
            'description': P.numel() * [''],
            'labels': P.numel() * [''],
            'units': P.numel() * [''],
            'shape': (P.numel(), 0),
            'data_format': ca.DM
        })

        sol = self._model(x0=X, p=up)
        X = sol['xf']

        x_pred = ca.MX(0)
        for k in range(2 * n_x + 1):
            x_pred += self._weights[0, k] * X[:, k]

        P_pred = Q
        for k in range(2 * n_x + 1):
            P_pred += self._weights[1, k] * (X[:, k] - x_pred) @ (X[:, k] - x_pred).T

        self._predict_function = ca.Function('prediction_step',
                                             [ca.horzcat(x, P), up, Q],
                                             [ca.horzcat(x_pred, P_pred, X)],
                                             ['x0', 'p', 'Q'],
                                             ['x'])

    def _setup_update(self, *args: Optional[ca.Function]) -> None:
        """

        :param args:
        :return:
        """
        n_x = self._model.n_x
        n_y = self._model.n_y
        n_u = self._model.n_u
        n_p = self._model.n_p

        x = ca.MX.sym('x', n_x)
        X = ca.MX.sym('X', n_x, 2 * n_x + 1)
        y = ca.MX.sym('y', n_y)
        u = ca.MX.sym('u', n_u)
        p = ca.MX.sym('p', n_p)
        up = ca.vertcat(u, p)
        P = ca.MX.sym('P', (n_x, n_x))

        if n_y == 0:
            warnings.warn(f"The model has no measurement equations, I am assuming measurements of all states "
                          f"{self._model.dynamical_state_names} are available.")
            Y = X
        else:
            sol = self._model(x0=X, p=up, which='meas_function')
            Y = sol['yf']

        y_pred = ca.MX(0)
        for k in range(2 * n_x + 1):
            y_pred += self._weights[0, k] * Y[:, k]

        R = ca.MX.sym('R', (n_y, n_y))
        P_xy = ca.MX(0)
        P_yy = R
        for k in range(2 * n_x + 1):
            P_xy += self._weights[1, k] * (X[:, k] - x) @ (Y[:, k] - y_pred).T
            P_yy += self._weights[1, k] * (Y[:, k] - y_pred) @ (Y[:, k] - y_pred).T

        # NOTE: See NOTE in KalmanFilter.setup()
        K = ca.solve(P_yy.T, P_xy.T).T

        x_up = x + K @ (y - y_pred)
        P_up = P - K @ P_yy @ K.T

        self._update_function = ca.Function('update_step',
                                            [ca.horzcat(x, P, X), y, up, R],
                                            [ca.horzcat(x_up, P_up), y_pred],
                                            ['x0', 'y', 'p', 'R'],
                                            ['x', 'y_pred'])

    @property
    def alpha(self) -> float:
        """

        :return:
        """
        return float(self._alpha)

    @alpha.setter
    def alpha(self, arg: Union[int, float]) -> None:
        self._check_parameter_bounds('alpha', arg)

    @property
    def beta(self) -> float:
        """

        :return:
        """
        return float(self._beta)

    @beta.setter
    def beta(self, arg: Union[int, float]) -> None:
        self._check_parameter_bounds('beta', arg)

    @property
    def kappa(self) -> float:
        """

        :return:
        """
        return float(self._kappa)

    @kappa.setter
    def kappa(self, arg: Union[int, float]) -> None:
        self._check_parameter_bounds('kappa', arg)


__all__ = [
    'KalmanFilter',
    'ExtendedKalmanFilter',
    'UnscentedKalmanFilter'
]
