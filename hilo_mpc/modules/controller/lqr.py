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

from typing import Optional

import casadi as ca

from .base import Controller
from ..base import Base, TimeSeries
from ..dynamic_model.dynamic_model import Model
from ...util.util import convert, is_real, is_symmetric, is_psd, is_pd


class LinearQuadraticRegulator(Controller, Base):
    """
    Class for linear-quadratic regulator (LQR)

    :param model:
    :type model:
    :param id:
    :type id: str, optional
    :param name:
    :type name: str, optional
    :param discrete:
    :type discrete: bool
    :param plot_backend:
    :type plot_backend: str, optional
    """
    def __init__(
            self,
            model: Model,
            id: Optional[str] = None,
            name: Optional[str] = None,
            discrete: bool = True,
            plot_backend: Optional[str] = None
    ) -> None:
        # TODO: Check for empty models
        # NOTE: I don't know how much sense the first if-condition makes, since it's already clear from the typing
        #  information, that we need a Model object.
        if not isinstance(model, Model):
            raise TypeError("The model must be an object of the Model class.")
        if not model.is_setup():
            raise RuntimeError("Model is not set up. Run Model.setup() before passing it to the controller.")
        if discrete and model.continuous:
            raise RuntimeError("The model used for the LQR needs to be discrete. Use Model.discretize() to obtain a "
                               "discrete model.")
        if not discrete and model.discrete:
            raise RuntimeError("The model used for the LQR needs to be continuous.")
        if not model.is_linear():
            raise RuntimeError("The model used for the LQR needs to be linear. Use Model.linearize() to obtain a "
                               "linearized model.")
        if model.is_autonomous():
            raise RuntimeError("The model used for the LQR is autonomous.")

        super().__init__(id=id, name=name)
        if self._id is None:
            self._create_id()

        self._model = model.copy(setup=True)
        self._Q = None
        self._R = None
        self._N = None
        self._K = None
        self._n_x = 0
        self._n_u = 0
        self._n_p = 0
        self._horizon = None

        self._solution = TimeSeries(plot_backend, parent=self)

    def _update_type(self) -> None:
        """

        :return:
        """
        self._type = 'LQR'

    @property
    def Q(self) -> Optional[ca.DM]:
        """

        :return:
        """
        return self._Q

    @Q.setter
    def Q(self, arg):
        if not is_real(arg):
            raise ValueError("LQR matrix Q needs to be real-valued")
        Q = convert(arg, ca.DM)
        if not Q.is_square():
            Q = ca.diag(Q)
        if Q.shape != (self._n_x, self._n_x):
            raise ValueError(f"Dimension mismatch. Supplied dimension is {Q.shape[0]}x{Q.shape[1]}, but required "
                             f"dimension is {self._n_x}x{self._n_x}")
        if not is_symmetric(Q):
            raise ValueError("LQR matrix Q needs to be symmetric")
        if not is_psd(Q):
            raise ValueError("LQR matrix Q needs to be positive semidefinite")
        self._Q = Q
        if self._K is not None:
            self._K = None

    @property
    def R(self) -> Optional[ca.DM]:
        """

        :return:
        """
        return self._R

    @R.setter
    def R(self, arg):
        if not is_real(arg):
            raise ValueError("LQR matrix R needs to be real-valued")
        R = convert(arg, ca.DM)
        if not R.is_square():
            R = ca.diag(R)
        if R.shape != (self._n_u, self._n_u):
            raise ValueError(f"Dimension mismatch. Supplied dimension is {R.shape[0]}x{R.shape[1]}, but required "
                             f"dimension is {self._n_u}x{self._n_u}")
        if not is_symmetric(R):
            raise ValueError("LQR matrix R needs to be symmetric")
        if not is_pd(R):
            raise ValueError("LQR matrix R needs to be positive definite")
        self._R = R
        if self._K is not None:
            self._K = None

    @property
    def N(self) -> Optional[ca.DM]:
        """

        :return:
        """
        return self._N

    @property
    def feedback_gain(self) -> Optional[ca.DM]:
        """

        :return:
        """
        return self._K

    K = feedback_gain

    @property
    def n_x(self) -> int:
        """

        :return:
        """
        return self._n_x

    @property
    def n_u(self) -> int:
        """

        :return:
        """
        return self._n_u

    @property
    def n_p(self) -> int:
        """

        :return:
        """
        return self._n_p

    @property
    def horizon(self) -> Optional[int]:
        """

        :return:
        """
        return self._horizon

    @horizon.setter
    def horizon(self, arg: int) -> None:
        self._horizon = arg

    def setup(self, **kwargs) -> None:
        """

        :param kwargs:
        :return:
        """
        self._n_x = self._model.n_x
        self._n_u = self._model.n_u
        self._n_p = self._model.n_p

        x = ca.SX.sym('x', self._n_x)
        p = ca.SX.sym('p', self._n_p)

        # NOTE: For now only discrete formulations are supported
        if self._horizon is None:
            # Infinite horizon LQR
            raise NotImplementedError("Infinite horizon LQR will be implemented in future releases")
        else:
            # Finite horizon LQR
            # Model matrices
            state_matrix = ca.Function('state_matrix', [self._model.p, self._model.dt, self._model.t], [self._model.A])
            input_matrix = ca.Function('input_matrix', [self._model.p, self._model.dt, self._model.t], [self._model.B])
            # NOTE: For now we ignore time-variant systems (i.e., [] instead of t)
            A = state_matrix(p, self._model.solution.dt, [])  # switch to self._solution if it is used
            B = input_matrix(p, self._model.solution.dt, [])  # switch to self._solution if it is used

            # LQR matrices
            P = ca.SX.sym('P', self._n_x, self._n_x)
            Q = ca.SX.sym('Q', self._n_x, self._n_x)
            R = ca.SX.sym('R', self._n_u, self._n_u)
            N = ca.SX.sym('N', self._n_x, self._n_u)

            # Solve dynamic Riccati equation
            Pk = P
            for k in range(self._horizon):
                APBN = A.T @ Pk @ B + N
                RBPB = R + B.T @ Pk @ B
                Pk = A.T @ Pk @ A - ca.solve(RBPB.T, APBN.T).T @ (B.T @ Pk @ A + N.T) + Q

            # Optimal control input
            K = ca.solve(R + B.T @ Pk @ B, B.T @ Pk @ A + N.T)
            u = -K @ x

            self._function = ca.Function('function',
                                         [x, p, P, Q, R, N],
                                         [u, K],
                                         ['x', 'p', 'P', 'Q', 'R', 'N'],
                                         ['u', 'K'])

            self.check_consistency()

            self._Q = None
            self._R = None
            self._N = ca.DM.zeros(N.shape)
            self._K = None

            # TODO: Set up and use solution?

    def call(self, *args, **kwargs):
        """

        :param args:
        :param kwargs:
        :return:
        """
        # TODO: Rename to optimize, so that the syntax is close to the LMPC class
        if self._function is None:
            raise RuntimeError("LQR is not set up. Run LQR.setup(...) before calling the LQR.")
        if self._Q is None:
            raise RuntimeError("Matrix Q is not set properly. To ensure that a unique solution exists, the matrix Q "
                               "needs to be symmetric, real-valued and positive semidefinite.")
        if self._R is None:
            raise RuntimeError("Matrix R is not set properly. To ensure that a unique solutions exists, the matrix R "
                               "needs to be symmetric, real-valued and positive definite.")

        x = kwargs.get('x')
        if x is not None:
            x = convert(x, ca.DM, axis=1)
        else:
            # TODO: Once reference/trajectory tracking and path following are supported we could revert back to allowing
            #  this (and consecutively initializing the states with 0)
            raise ValueError("No state information was supplied to the LQR!")

        p = kwargs.get('p')
        if p is not None:
            p = convert(p, ca.DM, axis=1)
        else:
            p = ca.DM.zeros(1, self._n_p)

        args = {
            'x': x,
            'p': p,
            'P': self._Q,
            'Q': self._Q,
            'R': self._R,
            'N': self._N
        }

        result = self._function(**args)
        if self._K is None:
            self._K = result['K']

        return result['u']


__all__ = [
    'LinearQuadraticRegulator'
]
