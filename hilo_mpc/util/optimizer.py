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

from typing import Optional, TypeVar, Union

import casadi as ca
import numpy as np
from scipy.optimize import minimize


Numeric = Union[int, float]
Symbolic = TypeVar('Symbolic', ca.SX, ca.MX)


class IpoptDebugger(ca.Callback):
    """"""
    def __init__(self, name, n_v, n_g, n_p, iter_step, ind_x, ind_u, opts=None):
        """Constructor method"""
        super().__init__()

        self.n_v = n_v
        self.n_g = n_g
        self.np = n_p
        self.ind_x = ind_x
        self.ind_u = ind_u
        self.x_sols = []
        self.f_sols = []
        self.g_sols = []
        self.lam_x_sols = []
        self.lam_g_sols = []
        self.lam_p_sols = []
        self.iter = []
        self.iter_step = iter_step

        # Initialize internal objects
        if opts is None:
            opts = {}
        self.construct(name, opts)

    def get_n_in(self):
        """

        :return:
        """
        return ca.nlpsol_n_out()

    def get_n_out(self):
        """

        :return:
        """
        return 1

    def get_name_in(self, i):
        """

        :param i:
        :return:
        """
        return ca.nlpsol_out(i)

    def get_name_out(self, i):
        """

        :param i:
        :return:
        """
        return "ret"

    def get_sparsity_in(self, i):
        """

        :param i:
        :return:
        """
        n = ca.nlpsol_out(i)
        if n == 'f':
            return ca.Sparsity.scalar()
        elif n in ('x', 'lam_x'):
            return ca.Sparsity.dense(self.n_v)
        elif n in ('g', 'lam_g'):
            return ca.Sparsity.dense(self.n_g)
        else:
            return ca.Sparsity(0, 0)

    def eval(self, arg):
        """

        :param arg:
        :return:
        """
        # Create dictionary
        if not self.iter:
            self.iter = [0]
        else:
            self.iter.append(self.iter[-1] + self.iter_step)
        darg = {}
        for (i, s) in enumerate(ca.nlpsol_out()): darg[s] = arg[i]
        x_sol = [float(elem) for elem in darg['x'].full()]

        self.x_sols.append(x_sol)
        self.f_sols.append(darg['f'].full()[0][0])
        self.g_sols.append(darg['g'].full())
        self.lam_x_sols.append(darg['lam_x'].full())
        self.lam_g_sols.append(darg['lam_g'].full())
        self.lam_p_sols.append(darg['lam_p'].full())

        return [0]

    def reset_solution(self):
        """

        :return:
        """
        self.x_sols = []
        self.f_sols = []
        self.g_sols = []
        self.lam_x_sols = []
        self.lam_g_sols = []
        self.lam_p_sols = []


class SciPyOptimizer:
    """
    Wrapper class for SciPy's optimizations

    :param name:
    :type name: str
    :param solver:
    :type solver: str
    :param problem:
    :type problem:
    :param options:
    :type options:
    """
    def __init__(
            self,
            name: str,
            solver: str,
            problem: dict[str, Symbolic],
            options: Optional[dict[str, Union[str, Numeric]]] = None
    ) -> None:
        """Constructor method"""
        self._name = name
        self._solver = solver

        self._function = None
        self._jacobian = None
        self._hessian = None

        self._get_free = []
        self._has_free = False

        self._n_x = 0
        self._n_p = 0
        self._n_f = 0

        self._parse_problem(problem)

        if options is None:
            options = {}
        self._options = options

        self._stats = {}

    def __call__(self, *args, **kwargs):
        """Calling method"""
        # TODO: Dimension checks
        # TODO: Typing hints
        if args and kwargs:
            raise TypeError("Instance of SciPyOptimizer can only accept positional or keyword arguments, not both")

        if args:
            if len(args) == 1:
                x0 = args[0]
                p = None
                lbx = None
                ubx = None
            elif len(args) == 2:
                x0 = args[0]
                p = args[1]
                lbx = None
                ubx = None
            elif len(args) == 3:
                x0 = args[0]
                p = args[1]
                lbx = args[2]
                ubx = None
            elif len(args) == 4:
                x0 = args[0]
                p = args[1]
                lbx = args[2]
                ubx = args[3]
            else:
                raise ValueError()

        if kwargs:
            x0 = kwargs.get('x0')
            p = kwargs.get('p')
            lbx = kwargs.get('lbx')
            ubx = kwargs.get('ubx')

        if not args and not kwargs:
            x0 = np.zeros((self._n_x, 1))
            if self._n_p > 0:
                p = np.zeros((self._n_p, 1))
            else:
                p = None

        min_kwargs = {}
        if p is not None:
            min_kwargs['args'] = (p, )
        min_kwargs['method'] = self._solver
        if self._solver in ['CG', 'BFGS', 'Newton-CG', 'L-BFGS-B']:
            min_kwargs['jac'] = self._jacobian
        if self._solver in ['Newton-CG']:
            min_kwargs['hess'] = self._hessian
        if self._solver in ['L-BFGS-B']:
            bounds = []
            for k in range(self._n_x):
                lbk = lbx[k] if lbx is not None else None
                ubk = ubx[k] if ubx is not None else None
                bounds.append((lbk, ubk))
            min_kwargs['bounds'] = bounds
        min_kwargs.update(self._options)

        sol = minimize(self._function, x0, **min_kwargs)

        self._stats['status'] = sol.status
        self._stats['success'] = sol.success
        self._stats['message'] = sol.message

        return {'f': sol.fun, 'x': sol.x}

    def _parse_problem(self, problem: dict[str, Symbolic]) -> None:
        """

        :param problem:
        :return:
        """
        # TODO: Add support for constraints
        # TODO: Add support for multi-objective optimization
        f = problem.get('f')
        x = problem.get('x')
        p = problem.get('p')

        if f is None:
            raise ValueError("No objective function supplied")
        if x is None:
            raise ValueError("No decision variables supplied")
        if p is None:
            p = f.sym('', 0)

        self._n_x = x.shape[0]
        self._n_p = p.shape[0]
        self._n_f = f.shape[0]

        function = ca.Function('function', [x, p], [f])
        function_free = function.get_free()

        def fun(w, args):
            """

            :param w:
            :param args:
            :return:
            """
            return function(w, args).full().flatten()

        self._function = fun

        if self._solver in ['CG', 'BFGS', 'Newton-CG', 'L-BFGS-B']:
            jacobian = ca.Function('jacobian', [x, p], [ca.jacobian(f, x)])
            jacobian_free = jacobian.get_free()

            def jac(w, args):
                """

                :param w:
                :param args:
                :return:
                """
                return jacobian(w, args).full().flatten()

            self._jacobian = jac
        else:
            jacobian_free = []

        if self._solver in ['Newton-CG']:
            hessian = ca.Function('hessian', [x, p], [ca.hessian(f, x)[0]])
            hessian_free = hessian.get_free()

            def hess(w, args):
                """

                :param w:
                :param args:
                :return:
                """
                return hessian(w, args).full()

            self._hessian = hess
        else:
            hessian_free = []

        self._get_free = list(set(function_free + jacobian_free + hessian_free))
        if self._get_free:
            self._has_free = True

    def get_free(self) -> list[str]:
        """

        :return:
        """
        return self._get_free

    def has_free(self) -> bool:
        """

        :return:
        """
        return self._has_free

    def stats(self) -> dict[str, Union[bool, int, str]]:
        """

        :return:
        """
        return self._stats
