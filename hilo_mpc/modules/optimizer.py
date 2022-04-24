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
from copy import deepcopy
from typing import Optional, Sequence, TypeVar, Union
import warnings

import casadi as ca
import numpy as np

from .base import Base, Vector, Problem, OptimizationSeries
from ..util.util import dump_clean, generate_c_code, who_am_i, JIT


Numeric = Union[int, float]
NumArray = Union[Sequence[Numeric], np.ndarray]
ArrayLike = Union[list, tuple, dict, NumArray, ca.DM, Vector]
Symbolic = TypeVar('Symbolic', ca.SX, ca.MX)


class Optimizer(Base, metaclass=ABCMeta):
    """
    Base class for all optimizers

    :param id:
    :type id: str, optional
    :param name:
    :type name: str, optional
    :param solver:
    :type solver: str, optional
    :param solver_options:
    :type solver_options:
    :param plot_backend:
    :type plot_backend:
    """
    # TODO: Getters and setter for boxed constraints, inequality constraints and equality constraints
    # TODO: Remove methods
    # TODO: Add methods
    # TODO: Add solver check (similar to the one in the Model class)
    # TODO: Add parsing of optimization problem
    def __init__(
            self,
            id: Optional[str] = None,
            name: Optional[str] = None,
            solver: Optional[str] = None,
            solver_options: Optional[dict[str, Union[str, Numeric]]] = None,
            plot_backend: Optional[str] = None
    ) -> None:
        """Constructor method"""
        super().__init__(id=id, name=name)
        if self.id is None:
            self._create_id()

        prefix = self.name + '_' if self.name is not None else ''
        suffix = '_' + self._id.rsplit('_', 1)[-1]

        self._x = Vector(ca.SX, id='variables' + suffix, name=prefix + 'variables', parent=self)
        self._lbx = Vector(ca.DM, values_or_names=[], id='variables_lower_bound' + suffix,
                           name=prefix + 'variables_lower_bound', parent=self)
        self._ubx = Vector(ca.DM, values_or_names=[], id='variables_upper_bound' + suffix,
                           name=prefix + 'variables_upper_bound', parent=self)
        self._p = Vector(ca.SX, id='parameters' + suffix, name=prefix + 'parameters', parent=self)
        self._problem = Problem(parent=self)
        self._update_dimensions()

        if solver is None:
            solver = 'ipopt'
        if solver_options is None:
            solver_options = {}
        self._solver = solver
        self._solver_opts = solver_options

        self._is_linear = None

        hist = {}
        self._solution = OptimizationSeries(plot_backend, parent=self, **hist)

    def __getstate__(self) -> dict:
        # TODO: Test this
        state = self.__dict__.copy()
        # Remove unpicklable entries
        # del state['']
        # or make them picklable
        # state[''] = []
        return state

    def __setstate__(self, state: dict) -> None:
        # TODO: Test this
        self.__dict__.update(state)
        for attr in ['_x', '_p', '_problem']:  # TODO: Add constraints and objective
            pointer = getattr(self, attr)
            pointer.parent = self
        if not hasattr(self, 'name'):
            self.name = None

    def _check_bounds(self) -> None:
        """

        :return:
        """
        lbx, ubx = self.bounds
        diff = ubx - lbx
        mask = diff < 0
        indices = np.flatnonzero(mask)
        if indices.size > 0:
            warnings.warn(f"Inconsistencies in the variable bounds. Bounds flipped at the inconsistent positions. The "
                          f"lower bound was bigger than the upper bound at the following indices: "
                          f"{', '.join(str(k) for k in indices)}")
            lbx[indices], ubx[indices] = ubx[indices], lbx[indices]

    def _check_linearity(self) -> None:
        """

        :return:
        """
        is_linear = True
        for eq in ['obj', 'cons']:
            for var in ['_x', '_p']:
                if hasattr(self, var):
                    is_linear = ca.is_linear(getattr(self._problem, eq), getattr(self, var).values)
                if not is_linear:
                    break
            if not is_linear:
                break

        self._is_linear = is_linear

    def _update_dimensions(self) -> None:
        """

        :return:
        """
        self._n_x = self._x.size1()
        self._n_p = self._p.size1()
        self._n_o = self._problem.obj.size1()
        self._n_c = self._problem.cons.size1()

    @abstractmethod
    def _update_solver(self) -> None:
        """

        :return:
        """
        pass

    @property
    def decision_variables(self) -> Symbolic:
        """

        :return:
        """
        return self._x.values

    optimization_variables = decision_variables
    x = decision_variables

    # @decision_variables.setter
    def set_decision_variables(
            self,
            *args: str,
            lower_bound: Optional[Numeric] = None,
            upper_bound: Optional[Numeric] = None,
            description: Optional[Union[str, Sequence[str]]] = None,
            labels: Optional[Union[str, Sequence[str]]] = None,
            units: Optional[Union[str, Sequence[str]]] = None,
            **kwargs: str
    ) -> Symbolic:
        """

        :param args:
        :param lower_bound:
        :param upper_bound:
        :param description:
        :param labels:
        :param units:
        :param kwargs:
        :return:
        """
        if lower_bound is None:
            lower_bound = -ca.inf

        if upper_bound is None:
            upper_bound = ca.inf

        if description is None:
            kwargs['description'] = ''
        else:
            kwargs['description'] = description

        if labels is None:
            kwargs['labels'] = ''
        else:
            kwargs['labels'] = labels

        if units is None:
            kwargs['units'] = ''
        else:
            kwargs['units'] = units

        if len(args) == 1:
            if isinstance(args[0], int) and 'name' not in kwargs:
                kwargs['name'] = 'x'
            self._set('x', args[0], **kwargs)
        elif len(args) == 2:
            if isinstance(args[0], str) and isinstance(args[1], int):
                kwargs['n_dim'] = args[1]
                self._set('x', args[0], **kwargs)
            elif isinstance(args[0], int) and isinstance(args[1], str):
                kwargs['n_dim'] = args[0]
                self._set('x', args[1], **kwargs)
            else:
                self._set('x', args, **kwargs)
        elif len(args) > 2:
            self._set('x', args, **kwargs)
        else:
            self._set('x', None, **kwargs)

        self.bounds = (lower_bound, upper_bound)

        return self._x.values

    set_optimization_variables = set_decision_variables
    set_variables = set_decision_variables

    @property
    def lower_bound(self) -> ca.DM:
        """

        :return:
        """
        return self._lbx.values

    @lower_bound.setter
    def lower_bound(self, arg: Numeric) -> None:
        self._set('lbx', arg, shape=self._x.shape)

    lbx = lower_bound

    @property
    def upper_bound(self) -> ca.DM:
        """

        :return:
        """
        return self._ubx.values

    @upper_bound.setter
    def upper_bound(self, arg: Numeric) -> None:
        self._set('ubx', arg, shape=self._x.shape)

    ubx = upper_bound

    @property
    def bounds(self) -> (ca.DM, ca.DM):
        """

        :return:
        """
        return self._lbx.values, self._ubx.values

    @bounds.setter
    def bounds(self, args: Union[Numeric, (Numeric, Numeric)]) -> None:
        self.set_bounds(*args)

    def set_bounds(self, *args: Numeric, **kwargs: dict[str, Numeric]) -> None:
        """

        :param args:
        :param kwargs:
        :return:
        """
        if args and kwargs:
            raise TypeError("Optimizer.set_bounds() can only accept positional or keyword arguments, not both")

        lbx = self._lbx.values
        ubx = self._ubx.values

        if args:
            if len(args) == 1:
                lbx = args[0]
                ubx = args[0]
            elif len(args) == 2:
                lbx = args[0]
                ubx = args[1]
            else:
                raise ValueError(f"Too many input arguments for setting the boundaries. Only 2 arguments are expected, "
                                 f"got {len(args)}")

        if kwargs:
            lbx = kwargs.get('lower_bound', lbx)
            ubx = kwargs.get('upper_bound', ubx)

        self.lower_bound = lbx
        self.upper_bound = ubx
        self._check_bounds()

    @property
    def n_x(self) -> int:
        """

        :return:
        """
        return self._n_x

    @property
    def initial_guess(self) -> Optional[ca.DM]:
        """

        :return:
        """
        if 'x' not in self._solution or self._solution.get_by_id('x').is_empty():
            return None
        return self._solution.get_by_id('x:0')

    @initial_guess.setter
    def initial_guess(self, arg: Union[Numeric, ArrayLike]) -> None:
        self.set_initial_guess(arg)

    x0 = initial_guess

    def set_initial_guess(self, x0: Union[Numeric, ArrayLike]) -> None:
        """

        :param x0:
        :return:
        """
        if not self._solution.is_set_up():
            raise RuntimeError("Optimizer is not set up. Run Optimizer.setup() before setting the initial guess.")

        if not self._x.is_empty():
            # TODO: Maybe rewrite similar to Model.set_initial_conditions()
            self._solution.set('x', x0)
        else:
            # TODO: Raise an error here?
            warnings.warn("Initial guess cannot be set, since no decision variables are defined for the optimization "
                          "problem.")

    @property
    def parameters(self) -> Symbolic:
        """

        :return:
        """
        return self._p.values

    p = parameters

    # @parameters.setter
    def set_parameters(
            self,
            *args: str,
            description: Optional[Union[str, Sequence[str]]] = None,
            labels: Optional[Union[str, Sequence[str]]] = None,
            units: Optional[Union[str, Sequence[str]]] = None,
            **kwargs: str
    ) -> Symbolic:
        """

        :param args:
        :param description:
        :param labels:
        :param units:
        :param kwargs:
        :return:
        """
        if description is None:
            kwargs['description'] = ''
        else:
            kwargs['description'] = description

        if labels is None:
            kwargs['labels'] = ''
        else:
            kwargs['labels'] = labels

        if units is None:
            kwargs['units'] = ''
        else:
            kwargs['units'] = units

        if len(args) == 1:
            if isinstance(args[0], int) and 'name' not in kwargs:
                kwargs['name'] = 'p'
            self._set('p', args[0], **kwargs)
        elif len(args) == 2:
            if isinstance(args[0], str) and isinstance(args[1], int):
                kwargs['n_dim'] = args[1]
                self._set('p', args[0], **kwargs)
            elif isinstance(args[0], int) and isinstance(args[1], str):
                kwargs['n_dim'] = args[0]
                self._set('p', args[1], **kwargs)
            else:
                self._set('p', args, **kwargs)
        elif len(args) > 2:
            self._set('p', args, **kwargs)
        else:
            self._set('p', None, **kwargs)

        return self._p.values

    @property
    def n_p(self) -> int:
        """

        :return:
        """
        return self._n_p

    def set_parameter_values(self, p: Union[Numeric, ArrayLike]) -> None:
        """

        :param p:
        :return:
        """
        if not self._p.is_empty():
            self._solution.add('p', p)
        else:
            warnings.warn(f"The optimization problem {self.name} doesn't have any parameters")

    @property
    def objective(self) -> Symbolic:
        """

        :return:
        """
        return self._problem.obj

    @objective.setter
    def objective(self, arg: Symbolic) -> None:
        self.set_objective(arg)

    obj = objective

    def set_objective(self, arg: Symbolic) -> None:
        """

        :param arg:
        :return:
        """
        if isinstance(arg, int):
            raise TypeError(f"Wrong type of argument for function {who_am_i()}")
        else:
            self._set('obj', arg)

    @property
    def n_o(self) -> int:
        """

        :return:
        """
        return self._n_o

    @property
    def sense(self) -> str:
        """

        :return:
        """
        return self._problem.sense

    @sense.setter
    def sense(self, arg: str) -> None:
        self._problem.sense = arg

    @property
    def constraints(self) -> Symbolic:
        """

        :return:
        """
        return self._problem.cons

    cons = constraints

    # @cons.setter
    def set_constraints(self, *args, **kwargs) -> None:
        """

        :param args:
        :param kwargs:
        :return:
        """
        # TODO: Support for constraints in string format
        # TODO: Think of other ways to pass information
        self._set('cons', args, x=self._x.values, p=self._p.values)

    @property
    def n_c(self) -> int:
        """

        :return:
        """
        return self._n_c

    @property
    def solver(self) -> str:
        """

        :return:
        """
        return self._solver

    @solver.setter
    def solver(self, arg: str) -> None:
        if arg != self._solver:
            if isinstance(arg, str):
                if self.check_solver(arg):
                    self._solver = arg
                    self._solver_opts = {}
                    if self._display:
                        print("Solver options have been reset")
                else:
                    if self._display:
                        print("Solver not updated")
            else:
                raise TypeError("Solver type must be a string")

    @property
    def options(self) -> str:
        """

        :return:
        """
        return dump_clean(self._solver_opts)

    @property
    def solution(self) -> OptimizationSeries:
        """

        :return:
        """
        return self._solution

    def check_consistency(self) -> None:
        """

        :return:
        """
        if self._function is not None:
            if self._function.has_free():
                free_vars = ", ".join(self._function.get_free())
                msg = f"The optimization problem has the following free variables/parameters: {free_vars}"
                raise RuntimeError(msg)

        self._check_bounds()

    @abstractmethod
    def check_solver(self, solver: str) -> None:
        """

        :param solver:
        :return:
        """
        pass

    def setup(self, **kwargs) -> None:
        """

        :param kwargs:
        :return:
        """
        interface = kwargs.get('interface', None)
        if interface is None:
            interface = ca.nlpsol

        opts = kwargs.get('options', None)
        if opts is not None:
            options = deepcopy(opts)
        else:
            options = {}
        options.update(self._solver_opts)

        solver = self._problem.to_solver('solver', interface, options=options)

        use_c_code = kwargs.get('c_code', False)
        if use_c_code:
            gen_path, gen_name, gen_opts = self._generator(**kwargs)
            if gen_path is not None:
                self._c_name = generate_c_code(solver, gen_path, gen_name, opts=gen_opts)
                if self._compiler_opts['method'] in JIT and self._compiler_opts['compiler'] == 'shell':
                    self._compiler_opts['path'] = gen_path
                self._function = 'solver'
                super().setup()
            else:
                self._function = solver
        else:
            self._function = solver

        self.check_consistency()

        names = []
        vector = {}
        if self._x.names:
            names += ['x']
            vector['x'] = {
                'values_or_names': self._x.names,
                'description': self._x.description,
                'labels': self._x.labels,
                'units': self._x.units,
                'shape': (self._n_x, 0),
                'data_format': ca.DM
            }
        if self._p.names:
            names += ['p']
            vector['p'] = {
                'values_or_names': self._p.names,
                'description': self._p.description,
                'labels': self._p.labels,
                'units': self._p.units,
                'shape': (self._n_p, 0),
                'data_format': ca.DM
            }
        self._solution.setup(*names, **vector)

    def solve(self, *args, **kwargs):
        """

        :param args:
        :param kwargs:
        :return:
        """
        # TODO: Store return_status and success (accessible via self._function.stats())
        if self._function is None:
            # NOTE: Not sure if we want to throw an error here
            raise RuntimeError("Optimizer is not set up. Run the method setup() before solving the problem.")

        # TODO: Need to rethink structure here
        warm_start = kwargs.get('warm_start')
        if warm_start is None:
            warm_start = False
        if warm_start:
            x0 = self._solution.get_by_id('x')
            if x0.size2() <= 1:
                warnings.warn("Warm start is not applicable, since no previous solution exists")
                if x0.is_empty():
                    self.set_initial_guess(x0=0.)
        else:
            if self._solution.get_by_id('x').is_empty():
                warnings.warn(f"No initial guess supplied. Using 0 as initial guess. Execute "
                              f"{self.__class__.__name__}.set_initial_guess(x0) to set a different initial guess.")
                self.set_initial_guess(self._n_x * [0.])

        p = kwargs.get('p')
        if p is not None:
            self._solution.add('p', p)
        elif self._solution.get_by_id('p').is_empty():
            warnings.warn(f"No parameter values supplied. Setting them to 0. Execute "
                          f"{self.__class__.__name__}.set_parameter_values(p) or supply them as a keyword argument to "
                          f"{self.__class__.__name__}.solve(...) to set different parameter values.")
            self.set_parameter_values(self._n_p * [0.])

        args = self._solution.get_function_args()
        args['lbx'] = self._lbx.values
        args['ubx'] = self._ubx.values
        # TODO: Niceify (don't access 'protected' attributes of self._problem here)
        args['lbg'] = self._problem._lbg
        args['ubg'] = self._problem._ubg

        result = self._function(**args)
        self._solution.update(**result)

    def stats(self) -> dict:
        """

        :return:
        """
        return self._function.stats()


class LinearProgram(Optimizer):
    """
    Linear programming problem

    :param id:
    :type id: str, optional
    :param name:
    :type name: str, optional
    :param solver:
    :type solver: str, optional
    :param solver_options:
    :type solver_options:
    :param plot_backend:
    :type plot_backend:
    """
    def __init__(
            self,
            id: Optional[str] = None,
            name: Optional[str] = None,
            solver: Optional[str] = None,
            solver_options: Optional[dict[str, Union[str, Numeric]]] = None,
            plot_backend: Optional[str] = None
    ) -> None:
        if solver is None:
            solver = 'clp'
        super().__init__(id=id, name=name, solver=solver, solver_options=solver_options, plot_backend=plot_backend)

    def _update_solver(self) -> None:
        """

        :return:
        """
        # TODO: Handle options in case of solver switches
        if isinstance(self._solver, str):
            if not ca.has_conic(self._solver) and not self._solver == 'ampl':
                if self._display:
                    print(f"Solver '{self._solver}' is either not available on your system or is not a suitable solver."
                          f"Switching to 'clp'...")
                self._solver = 'clp'
        else:
            self._solver = 'clp'

    def check_solver(self, solver: str) -> bool:
        """

        :param solver:
        :return:
        """
        if isinstance(solver, str):
            if ca.has_conic(solver):
                return True
            else:
                return False
        else:
            raise TypeError("Solver type must be a string")

    def setup(self, **kwargs):
        """

        :param kwargs:
        :return:
        """
        self._check_linearity()
        if not self._is_linear:
            raise RuntimeError("Linear programming was chosen as the optimizer, but the supplied problem is not linear")

        super().setup(interface=ca.qpsol, **kwargs)


LP = LinearProgram


class QuadraticProgram(LinearProgram):
    """"""


QP = QuadraticProgram
