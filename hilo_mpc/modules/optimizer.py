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

from .base import Base, Vector, Problem, OptimizationSeries, TimeSeries
from .dynamic_model.dynamic_model import Model
from ..util.optimizer import SciPyOptimizer as scisol
from ..util.util import check_and_wrap_to_list, check_if_list_of_string, dump_clean, generate_c_code, who_am_i, JIT


Numeric = Union[int, float]
NumArray = Union[Sequence[Numeric], np.ndarray]
ArrayLike = Union[list, tuple, dict, NumArray, ca.DM, Vector]
Symbolic = TypeVar('Symbolic', ca.SX, ca.MX)


SCIPY_OPTIMIZERS = ['CG', 'BFGS', 'Newton-CG', 'Nelder-Mead', 'L-BFGS-B', 'Powell']


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
        """Constructor method"""
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


class QuadraticProgram(LinearProgram):
    """"""


class NonlinearProgram(Optimizer):
    """
    Nonlinear programming problem

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
        """Constructor method"""
        if solver is None:
            solver = 'ipopt'
        super().__init__(id=id, name=name, solver=solver, solver_options=solver_options, plot_backend=plot_backend)

    def _update_solver(self) -> None:
        """

        :return:
        """
        # TODO: Handle options in case of solver switches
        if isinstance(self._solver, str):
            if not ca.has_nlpsol(self._solver) and self._solver not in SCIPY_OPTIMIZERS and not self._solver == 'ampl':
                if self._display:
                    print(f"Solver '{self._solver}' is either not available on your system or is not a suitable solver."
                          f" Switching to 'ipopt'...")
                self._solver = 'ipopt'
        else:
            self._solver = 'ipopt'

    def check_solver(self, solver: str) -> bool:
        """

        :param solver:
        :return:
        """
        if isinstance(solver, str):
            if ca.has_nlpsol(solver) or solver in SCIPY_OPTIMIZERS:
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
        if self._is_linear:
            warnings.warn("Nonlinear programming was chosen as the optimizer, but the supplied problem is linear. "
                          "Switching to linear programming for this problem might result in a better performance.")

        if self._solver not in SCIPY_OPTIMIZERS:
            super().setup(interface=ca.nlpsol, **kwargs)
        else:
            super().setup(interface=scisol, **kwargs)


class DynamicOptimization(Base):
    """Base class for all MPC and MHE"""
    def __init__(self, model, id=None, name=None, plot_backend=None, stats=False, use_sx=True):
        """Constructor method"""
        super().__init__(id=id, name=name)
        if not isinstance(model, Model):
            raise TypeError('The model must be an object of the Model class.')

        if not model.is_setup():
            raise RuntimeError("Model is not set up. Run Model.setup() before passing it to the controller.")

        self._model = model.copy(setup=False, use_sx=use_sx)
        self._model_orig = model.copy(setup=False, use_sx=use_sx)
        self._sampling_interval = model.solution.dt

        # Solver status
        self._solver_status = 'init'

        # Flags
        self._nlp_setup_done = False
        self._box_constraints_is_set = False
        self._initial_guess_is_set = False
        self._nlp_options_is_set = False
        self._solver_options_is_set = False
        self._nlp_solver_is_set = False
        self._custom_constraint_flag = False
        self._custom_constraint_is_soft_flag = False
        self._change_mode_term_flat = False
        self._scaling_is_set = False
        self._time_varying_parameters_is_set = False
        self._sampling_time_is_set = False

        self._horizon = None
        # Options
        self.int_opts = []
        self.integration_method = []
        self.collocation_polynomials = []
        self.collocation_polynomials_degree = []
        self.collocation_polynomials_fin_elements = []
        self._nlp_opts = {}
        self._nlp_solution = None

        # Time varying parameters settings
        self._n_tvp = 0
        self._time_varying_parameters_horizon = ca.DM.zeros((0, 0))

        self._solution = TimeSeries(plot_backend, parent=self)

        self._solver_name_list_qp = ['qpoases', 'cplex', 'gurobi', 'oopq', 'sqic', 'nlp']
        self._solver_name_list_nlp = ['ipopt', 'bonmin', 'knitro', 'snopt', 'worhp', 'scpgen', 'sqpmethod', 'blocksqp',
                                      'AmplInterface']

        self._n_iterations = 0

        # Here flag for statistics
        self._stats = stats

    def _rearrange_parameters(self, tvp, cp):
        """
        Takes time-varying parameters, and not time-varying parameters and rearranges them in the correct order of the
        model parameter vector

        :param tvp:
        :param cp:
        :return:
        """
        if self._n_tvp != 0:
            p = ca.MX.sym('p', 0)
            p_names = self._model.parameter_names
            ii_tvp = 0
            ii_cp = 0
            for name in p_names:
                if name in self._time_varying_parameters:
                    p = ca.vertcat(p, tvp[ii_tvp])
                    ii_tvp += 1
                else:
                    p = ca.vertcat(p, cp[ii_cp])
                    ii_cp += 1
        else:
            p = cp

        return p

    def _print_message(self):
        """

        :return:
        """
        # TODO create solver statuses that are used by HILO such that we can use the same for different solvers
        if self._nlp_options['print_level'] == 0:
            pass
        if self._nlp_options['print_level'] == 1:
            if self._solver_name == 'ipopt':
                refer_ipopt = f"Please refer to ipopt documentation. For more info pass ipopt.print_level:5 to the " \
                              f"solver_options in the {self.__class__.__name__}.setup()"

                if self._solver_status_code == 1:
                    print(f"{self.__class__.__name__} found an optimal solution.")

                elif self._solver_status_code == 2:
                    print(f"{self.__class__.__name__} found an acceptable solution.")

                else:
                    print(f"Ups... {self.__class__.__name__} had some problems. The ipopt error message is: "
                          f"{self._solver_status}. " + refer_ipopt)

    def _populate_solution(self):
        """

        :return:
        """
        names = ['t']
        vector = {
            't': {
                'values_or_names': self._model._t.names,
                'description': self._model._t.description,
                'labels': self._model._t.labels,
                'units': self._model._t.units,
                'shape': (1, 0),
                'data_format': ca.DM
            }
        }
        if self._model_orig._x.names:
            names += ['x']
            vector['x'] = {
                'values_or_names': self._model_orig._x.names,
                'description': self._model_orig._x.description,
                'labels': self._model_orig._x.labels,
                'units': self._model_orig._x.units,
                'shape': (self._model_orig._n_x, 0),
                'data_format': ca.DM
            }
        if self._model_orig._y.names:
            names += ['y']
            vector['y'] = {
                'values_or_names': self._model_orig._y.names,
                'description': self._model_orig._y.description,
                'labels': self._model_orig._y.labels,
                'units': self._model_orig._y.units,
                'shape': (self._model_orig._n_y, 0),
                'data_format': ca.DM
            }
        if self._model_orig._z.names:
            names += ['z']
            vector['z'] = {
                'values_or_names': self._model_orig._z.names,
                'description': self._model_orig._z.description,
                'labels': self._model_orig._z.labels,
                'units': self._model_orig._z.units,
                'shape': (self._model_orig._n_z, 0),
                'data_format': ca.DM
            }
        if self._model_orig._u.names:
            names += ['u']
            vector['u'] = {
                'values_or_names': self._model_orig._u.names,
                'description': self._model_orig._u.description,
                'labels': self._model_orig._u.labels,
                'units': self._model_orig._u.units,
                'shape': (self._model_orig._n_u, 0),
                'data_format': ca.DM
            }
        if self._model_orig._p.names:
            names += ['p']
            vector['p'] = {
                'values_or_names': self._model_orig._p.names,
                'description': self._model_orig._p.description,
                'labels': self._model_orig._p.labels,
                'units': self._model_orig._p.units,
                'shape': (self._model_orig._n_p, 0),
                'data_format': ca.DM
            }
        if self._model_orig._n_q > 0:
            names += ['q']
            vector['q'] = {
                'values_or_names': 'q',
                'shape': (self._model_orig._n_q, 0),
                'data_format': ca.DM
            }

        if hasattr(self, 'type') and self.type == 'NMPC':
            names += [f'thetapfo']
            vector[f'thetapfo'] = {
                'values_or_names': [f'thetapfo{i+1}' for i in range(self.n_of_path_vars)],
                'description': [f'theta p.f. num. {i+1}' for i in range(self.n_of_path_vars)],
                'shape': (self.n_of_path_vars, 0),
                'data_format': ca.DM
            }

        if self._stats:
            names += ['extime']
            vector['extime'] = {
                'values_or_names': 'extime',
                'units': 'seconds',
                'shape': (1, 0),
                'data_format': ca.DM
            }
            names += ['niterations']
            vector['niterations'] = {
                'values_or_names': 'niterations',
                'shape': (1, 0),
                'data_format': ca.DM
            }
            names += ['solvstatus']
            vector['solvstatus'] = {
                'values_or_names': 'solvstatus',
                'shape': (1, 0),
                'data_format': ca.DM
            }
        self._solution.setup(*names, **vector)

    def _solver_status_wrapper(self):
        """

        :return:
        """
        # TODO: consider other outputs
        if self._solver_name == 'ipopt':
            self._solver_status = self._solver.stats()['return_status'].lower()
            if self._solver_status == 'solve_succeeded':
                self._solver_status_code = 1
            elif self._solver_status == 'solved_to_acceptable_level':
                self._solver_status_code = 2
            elif self._solver_status == 'infeasible_problem_detected':
                self._solver_status_code = 3
            elif self._solver_status == 'restoration_failed':
                self._solver_status_code = 4
            elif self._solver_status == 'maximum_iterations_exceeded':
                self._solver_status_code = 5
            else:
                self._solver_status_code = -1

    def _get_tvp_parameters_values(self):
        """
        Given the current iteration, returns the value for every sampling time of all time-varying parameter

        :return:
        """
        ci = self._n_iterations

        # Shift the horizon of tvp one step back
        if ci > 0:
            self._time_varying_parameters_horizon[:, 0:-1] = self._time_varying_parameters_horizon[:, 1:]
            for k, name in enumerate(self._time_varying_parameters):
                value = self._time_varying_parameters_values[name]

                if ci + self.horizon > len(value):
                    warnings.warn("The prediction horizon is predicting outside the values of the time varying "
                                  "parameters. I am now taking looping back the values and start from there. "
                                  "See documentation for info.")

                    n_of_overtakes = int(np.floor(ci / self.horizon))

                    self._time_varying_parameters_horizon[k, -1] = value[
                        ci - self.horizon * n_of_overtakes]
                else:
                    self._time_varying_parameters_horizon[k, -1] = value[ci + self.horizon - 1]
        else:
            self._time_varying_parameters_horizon = ca.DM.zeros((self._n_tvp), self.prediction_horizon)
            tvp_counter = 0
            for key, value in self._time_varying_parameters_values.items():
                if len(value) < self.prediction_horizon:
                    raise TypeError(f"When passing time-varying parameters, you need to pass a number of values at "
                                    f"least as long as the prediction horizon. The parameter {key} has {len(value)} "
                                    f"values but the MPC has a prediction horizon length of {self._prediction_horizon}."
                                    )

                value = self._time_varying_parameters_values[key]
                self._time_varying_parameters_horizon[tvp_counter, :] = value[0:self._prediction_horizon]
                tvp_counter += 1

    def reset_solution(self) -> None:
        """

        :return:
        """
        if not self._solution.is_empty():
            for k in self._solution:
                self._solution.remove(k, slice(0, None))

    def set_stage_constraints(self, stage_constraint=None, lb=None, ub=None, is_soft=False, max_violation=ca.inf,
                              weight=None, name='stage_constraint'):
        """
        Allows to add a (nonlinear) stage constraint.

        :param stage_constraint: SX expression. It has to contain variables of the model
        :param lb: Vector or list float,integer or casadi.DM. Lower bound on the constraint
        :param ub: Vector or list float,integer or casadi.DM. Upper bound on the constraint
        :param is_soft: bool: if True soft constraints are used.
        :param max_violation: (optional) Vector or list float,integer or casadi.DM. Maximum violation if constraint is
            soft. If None, there is no limit on the violation of the soft constraints
        :param weight: (optional) matrix of appropriate dimension. If is_soft=True it will be used to weight the soft
            constraint in the objective function using a quadratic cost.
        :param name:
        :return:
        """
        # TODO check if all the variables used in the function are in the model
        # TODO allow to pass either only the lower or the upper bound
        self.stage_constraint.constraint = stage_constraint
        self.stage_constraint.lb = lb
        self.stage_constraint.ub = ub
        self.stage_constraint.is_soft = is_soft
        self.stage_constraint.max_violation = max_violation
        self.stage_constraint.weight = weight
        self.stage_constraint.name = name

    def set_custom_constraints_function(self, fun=None, lb=None, ub=None, soft=False, max_violation=ca.inf):

        """
        Set a custom function that will be added as last position in the nonlinear constraints vector:
                                            lb <= fun <= ub
        This must take the entire optimization vector z and the indices of states and inputs as
                                            fun(v, x_ind, u_ind)

        :param fun: python function
        :param lb: lower bound
        :param ub: upper bound
        :param soft:
        :param max_violation:
        :return:
        """
        if lb is None:
            lb = [-ca.inf]
        lb = check_and_wrap_to_list(lb)
        if ub is None:
            ub = [ca.inf]
        ub = check_and_wrap_to_list(ub)

        self._custom_constraint_fun = fun
        self._custom_constraint_fun_lb = lb
        self._custom_constraint_fun_ub = ub
        self._custom_constraint_size = len(lb)
        self._custom_constraint_flag = True
        self._custom_constraint_is_soft_flag = soft
        self._custom_constraint_maximum_violation = max_violation

    def set_box_constraints(self, x_ub=None, x_lb=None, u_ub=None, u_lb=None, y_ub=None, y_lb=None, z_ub=None,
                            z_lb=None):
        """
        Set box constraints to the model's variables. These look like

        .. math::
                                    x_{lb} \leq x \leq x_{ub}

        :param x_ub: upper bound on states.
        :type x_ub: list, numpy array or CasADi DM array
        :param x_lb: lower bound on  states
        :type x_lb: list, numpy array or CasADi DM array
        :param u_ub: upper bound on inputs
        :type u_ub: list, numpy array or CasADi DM array
        :param u_lb: lower bound on inputs
        :type u_lb: list, numpy array or CasADi DM array
        :param y_ub: upper bound on measurements
        :type y_ub: list, numpy array or CasADi DM array
        :param y_lb: lower bound on measurements
        :type y_lb: list, numpy array or CasADi DM array
        :param z_ub: upper bound on algebraic states
        :type z_ub: list, numpy array or CasADi DM array
        :param z_lb: lower bound on algebraic states
        :type z_lb: list, numpy array or CasADi DM array
        :return:
        """
        if x_ub is not None:
            x_ub = deepcopy(x_ub)
            x_ub = check_and_wrap_to_list(x_ub)
            if len(x_lb) != self._n_x:
                raise TypeError(f"The model has {self._n_x} states. You need to pass the same number of bounds.")
            self._x_ub = x_ub
        else:
            self._x_ub = [ca.inf for i in range(self._model.n_x)]

        if x_lb is not None:
            x_lb = deepcopy(x_lb)
            x_lb = check_and_wrap_to_list(x_lb)
            if len(x_lb) != self._n_x:
                raise TypeError(f"The model has {self._n_x} states. You need to pass the same number of bounds.")
            self._x_lb = x_lb
        else:
            self._x_lb = [-ca.inf for i in range(self._model.n_x)]

        # Input constraints
        if u_ub is not None:
            u_ub = deepcopy(u_ub)
            u_ub = check_and_wrap_to_list(u_ub)
            if len(u_ub) != self._n_u:
                raise TypeError(f"The model has {self._n_u} inputs. You need to pass the same number of bounds.")
            self._u_ub = u_ub
        else:
            self._u_ub = [ca.inf for i in range(self._model.n_u)]

        if u_lb is not None:
            u_lb = deepcopy(u_lb)
            u_lb = check_and_wrap_to_list(u_lb)
            if len(u_lb) != self._n_u:
                raise TypeError(f"The model has {self._n_u} inputs. You need to pass the same number of bounds.")
            self._u_lb = u_lb
        else:
            self._u_lb = [-ca.inf for i in range(self._model.n_u)]

        # Algebraic constraints
        if z_ub is not None:
            z_ub = deepcopy(z_ub)
            z_ub = check_and_wrap_to_list(z_ub)
            if len(z_ub) != self._n_z:
                raise TypeError(f"The model has {self._n_z} algebraic states. You need to pass the same number of "
                                f"bounds.")
        else:
            self._z_ub = [ca.inf for i in range(self._model.n_z)]

        if z_lb is not None:
            z_lb = deepcopy(z_lb)
            z_lb = check_and_wrap_to_list(z_lb)
            if len(z_lb) != self._n_z:
                raise TypeError(f"The model has {self._n_z} algebraic states. You need to pass the same number of "
                                f"bounds.")
        else:
            self._z_lb = [-ca.inf for i in range(self._model.n_z)]

        if y_lb is not None or y_ub is not None:
            # Measurement box constraints can be added by an extra stage and terminal constraint (possibly nonlinear)
            self.set_stage_constraints(stage_constraint=self._model.meas, ub=deepcopy(y_ub), lb=deepcopy(y_lb),
                                       name='measurement_constraint')
            self.set_terminal_constraints(terminal_constraint=self._model.meas, ub=deepcopy(y_ub), lb=deepcopy(y_lb),
                                          name='measurement_constraint')

        self._box_constraints_is_set = True

    def set_initial_guess(self, x_guess=None, u_guess=None, z_guess=None):
        """
        Sets initial guess for the optimizer when no other information of the states or inputs are available.
        :param x_guess: optimal state guesss
        :type x_guess: list, numpy array, float or casADi DM array
        :param u_guess: optimal input guess
        :type u_guess: list, numpy array, float or casADi DM array
        :param z_guess: optimal algebraic states guess
        :type z_guess: list, numpy array, float or casADi DM array
        :return:
        """
        # TODO: make this nicer
        if x_guess is not None:
            self._x_guess = check_and_wrap_to_list(deepcopy(x_guess))

            if len(self._x_guess) != self._model_orig.n_x:
                raise ValueError(f"x_guess dimension and model state dimension do not match. Model state has dimension "
                                 f"{self._model_orig.n_x} while x_guess has dimension {len(self._x_guess)}")
        else:
            self._x_guess = self._model_orig.n_x * [0]

        if u_guess is not None:
            self._u_guess = check_and_wrap_to_list(deepcopy(u_guess))

            if len(self._u_guess) != self._model_orig.n_u:
                raise ValueError(f"u_guess dimension and model u dimension do not match. Model u_guess has dimension "
                                 f"{len(self._u_guess)} while the model input has dimension {self._model.n_u}")
        else:
            self._u_guess = self._model_orig.n_u * [0]

        if z_guess is not None:
            self._z_guess = check_and_wrap_to_list(deepcopy(z_guess))

            if len(self._z_guess) != self._model_orig.n_z:
                raise ValueError(f"z_guess dimension and model z dimension do not match. Model z has dimension "
                                 f"{self._model_orig.n_z} while z_guess {len(self._z_guess)}")
        else:
            self._z_guess = self._model_orig.n_z * [0]

        self._initial_guess_is_set = True

    def set_solver_opts(self, *args, **kwargs):
        """

        Sets the nonlinear programming settings. This can be a dictionary or key-valued arguments
        These correspond to the option of the used nlp solver, for example ipopt or bonmin
        :param args:
        :param kwargs:
        :return:
        """
        if len(args) != 0:
            if isinstance(args[0], dict):
                self._nlp_opts = args[0]
        else:
            if len(kwargs) != 0:
                self._nlp_opts = kwargs[0]
            else:
                self._nlp_opts = {}

        if self._nlp_options_is_set and self._nlp_options['print_level'] == 1:
            if self._nlp_opts.get('ipopt.print_level') is None:
                self._nlp_opts['ipopt.suppress_all_output'] = 'yes'

            if self._nlp_opts.get('print_time') is None:
                self._nlp_opts['print_time'] = False
        else:
            self._nlp_opts['ipopt.suppress_all_output'] = 'yes'
            self._nlp_opts['print_time'] = False

        self._solver_options_is_set = True

    def set_nlp_solver(self, solver):
        """
        Set the solver for the MPC. Possible options are
        For nonlinear programming problems (e.g. nonlinear MPC)
        - 'ipopt', 'bonmin', 'knitro', 'snopt', 'worhp', 'scpgen', 'sqpmethod', 'blocksqp', 'AmplInterface'
        for quadratic programmin problems (e.g. linear MPC)
        - 'qpoases', 'cplex', 'gurobi', 'oopq', 'sqic', 'nlp
        For more info refer to the CasADi documentation https://web.casadi.org/docs/.

        :param solver:
        :return:
        """
        self._solver_name_list = self._solver_name_list_qp + self._solver_name_list_nlp
        self._solver_name = solver
        self._nlp_solver_is_set = True

    def set_nlp_options(self, *args, **kwargs):
        """
        Sets the options that modify how the MPC problem is set

        :param args:
        :param kwargs:
        :return:
        """
        # TODO: when multiple-shooting and irk are implemented/tested add them to the list
        possible_choices = {}
        possible_choices['integration_method'] = ['collocation', 'rk4', 'erk', 'discrete', 'idas', 'cvodes']  # 'irk'
        possible_choices['solver'] = self._solver_name_list_nlp
        possible_choices['collocation_points'] = ['radau', 'legendre']
        possible_choices['objective_function'] = ['discrete', 'continuous']
        possible_choices['warm_start'] = [True, False]
        possible_choices['degree'] = None
        possible_choices['print_level'] = [0, 1]
        possible_choices['ipopt_debugger'] = [True, False]

        option_list = list(possible_choices.keys())

        default_opts = {
            'integration_method': 'collocation',
            'collocation_points': 'radau',
            'degree': 3,
            'print_level': 1,
            'warm_start': True,
            'solver': 'ipopt',
            'ipopt_debugger': False
        }

        if self._model.discrete:
            default_opts.update({'objective_function': 'discrete'})
        else:
            default_opts.update({'objective_function': 'continuous'})

        opts = {}
        if len(args) != 0:
            if isinstance(args[0], dict):
                opts = args[0]
        else:
            if len(kwargs) != 0:
                opts = kwargs

        for key, value in opts.items():
            if key not in option_list:
                raise ValueError(f"The option named {key} does not exist. Possible options are {option_list}.")
            if possible_choices[key] is not None and value not in possible_choices[key]:
                raise ValueError(f"The option {key} is set to value {value} but the only allowed values are "
                                 f"{possible_choices[key]}.")
            else:
                default_opts[key] = value

        if default_opts.get('integration_method') != 'discrete' and self._model.discrete is True:
            warnings.warn(f"The integration method is set to {default_opts.get('integration_method')} but I notice that"
                          f" the model is in discrete time. I am overwriting and using discrete mode.")
            default_opts['integration_method'] = 'discrete'

        # Integration methods. Those are necessary for the RungeKutta class
        if default_opts['integration_method'] == 'rk4':
            default_opts['class'] = 'explicit'
            default_opts['method'] = 'rk'
            default_opts['order'] = 4
            default_opts['category'] = 'runge-kutta'
        elif default_opts['integration_method'] == 'erk':
            default_opts['class'] = 'explicit'
            default_opts['method'] = 'rk'
            default_opts['category'] = 'runge-kutta'
        elif default_opts['integration_method'] == 'irk':
            default_opts['class'] = 'implicit'
            default_opts['method'] = 'rk'
            default_opts['category'] = 'runge-kutta'
        elif default_opts['integration_method'] == 'collocation':
            default_opts['class'] = 'implicit'
            default_opts['method'] = 'collocation'
            default_opts['category'] = 'runge-kutta'

        if not self._nlp_solver_is_set:
            self.set_nlp_solver(default_opts['solver'])

        self._nlp_options_is_set = True
        self._nlp_options = default_opts

    def set_scaling(self, x_scaling=None, u_scaling=None, y_scaling=None):
        """
        Pass the scaling factors of states, inputs and outputs.
        This is important for system with where the optimization variable can have a large difference of order of magnitude.
        The scaling factors divide the respective variables.

        :param x_scaling: scaling factors for the states
        :type x_scaling: list
        :param u_scaling: scaling of scaling factors for inputs
        :type u_scaling: list
        :param y_scaling: scaling of scaling factors for outputs
        :type y_scaling: list
        :return:
        """

        if x_scaling is None:
            self._x_scaling = [1 for i in range(self._model.n_x)]
        else:
            self._x_scaling = check_and_wrap_to_list(x_scaling)

        if u_scaling is None:
            self._u_scaling = [1 for i in range(self._model.n_u)]
        else:
            self._u_scaling = check_and_wrap_to_list(u_scaling)

        if y_scaling is None:
            self._y_scaling = [1 for i in range(self._model.n_y)]
        else:
            self._y_scaling = check_and_wrap_to_list(y_scaling)

        self._scaling_is_set = True

    def set_sampling_interval(self, dt=None):
        """

        :param dt:
        :return:
        """
        if dt is not None:
            if isinstance(dt, float) or isinstance(dt, int):
                self._sampling_interval = dt
            else:
                raise TypeError("Sampling interval must be a float.")

    def set_time_varying_parameters(self, names=None, values=None):
        """
        Sets the time-varying parameters values for then MPC.
        Time varying parameters are model parameters that change in time. Hence, when predicting the system dynamics,
        their value must be supplied to the model at the every sampling time.

        :param names: list of strings with time varying parameter names
        :type names: list of strings
        :param values: values of the time varying parameters. You need to pass at least a number of values as long as
            the prediction horizon. Note: if the prediction goes after the last supplied value, the values will be
            repeated!
        :type values: dict (optional)
        :return: None
        """

        if names is None:
            self._time_varying_parameters = []
        else:
            if not check_if_list_of_string(names):
                raise ValueError('Tvp must be a list of strings with the paramers name that are time varying')

            for k, tvp in enumerate(names):
                if tvp not in self._model.parameter_names:
                    raise ValueError(f"I could not find the parameter {tvp} in the model. The models parameters are "
                                     f"{self._model.parameter_names}.")
                else:
                    self._time_varying_parameters_ind.append(k)
            self._n_tvp = len(names)
            self._time_varying_parameters = names
        self._time_varying_parameters_is_set = True

        self._time_varying_parameters_values = values

        if values is not None:
            if not isinstance(values, dict):
                raise TypeError("The values parameter must be a dictionary.")
            # Check that the dictionary contains as key the names of the tvp:
            for key in values.keys():
                if key not in names:
                    raise ValueError(f"The key {key} is not in the name vector: {names}. You need to pass a dictionary "
                                     f"where the keys are the name of the time varying parameters.")

    def plot_iterations(self, plot_states=False, **kwargs):
        """
        This plots the states and constraints for every mpc iteration. Useful to debug.
        At the moment it is working only with Bokeh.
        Warning: if the optimizer computes a large number of iterations the plots might take a bit of time to show up.

        :param plot_states: if True, also the states are plotted
        :param kwargs: If you pass 'plot_last' only the last iteration is plotted. This is to avoid very busy figures.
        :return:
        """
        # TODO: expand this to matplotlib
        # TODO: allow usage in jupyter notebook
        # TODO: expand this to MHE
        import itertools

        from bokeh.layouts import column
        from bokeh.models import BoxAnnotation
        from bokeh.plotting import figure, show, gridplot
        from bokeh.palettes import inferno  # select a palette

        if not hasattr(self, 'debugger'):
            raise ValueError("You need to activate the debugger. Pass 'options={'ipopt_debugger': True}' in the setup "
                             "of the NMPC.")

        plot_last = kwargs.get('plot_last', False)
        numLines = len(self.debugger.g_sols)

        pg = figure(width=2000, height=1200, title='Nonlinear constraints')
        x = np.arange(0, self._g.shape[0])
        colors_g = itertools.cycle(inferno(numLines))
        if plot_last:
            y = self.debugger.g_sols[-1]
            pg.scatter(x, y.squeeze(), color=next(colors_g), legend_label=f'iter. {numLines}', line_width=2)
        else:
            for m in range(numLines):
                y = self.debugger.g_sols[m]
                pg.scatter(x, y.squeeze(), color=next(colors_g), legend_label=f'iter. {m}', line_width=2)

        # Add sections saying what is what
        for i in range(len(self._g_indices['dynamics_collocation'])):
            low_box = BoxAnnotation(left=self._g_indices['dynamics_collocation'][i][0],
                                    right=self._g_indices['dynamics_collocation'][i][1],
                                    fill_alpha=0.1, fill_color='red')
            pg.add_layout(low_box)

        for i in range(len(self._g_indices['dynamics_multiple_shooting'])):
            low_box = BoxAnnotation(left=self._g_indices['dynamics_multiple_shooting'][i][0],
                                    right=self._g_indices['dynamics_multiple_shooting'][i][1],
                                    fill_alpha=0.1, fill_color='blue')
            pg.add_layout(low_box)

        for i in range(len(self._g_indices['nonlin_stag_const'])):
            low_box = BoxAnnotation(left=self._g_indices['nonlin_stag_const'][i][0],
                                    right=self._g_indices['nonlin_stag_const'][i][1],
                                    fill_alpha=0.1, fill_color='green')
            pg.add_layout(low_box)

        for i in range(len(self._g_indices['nonlin_term_const'])):
            low_box = BoxAnnotation(left=self._g_indices['nonlin_term_const'][i][0],
                                    right=self._g_indices['nonlin_term_const'][i][1],
                                    fill_alpha=0.1, fill_color='orange')
            pg.add_layout(low_box)

        plagg = figure(width=2000, height=1200, title='Lagrange multipliers nonlinear constraints')
        x = np.arange(0, self._g.shape[0])
        colors_lagg = itertools.cycle(inferno(numLines))
        if plot_last:
            y = self.debugger.lam_g_sols[-1]
            plagg.line(x, y.squeeze(), color=next(colors_lagg), legend_label=f'iter. {numLines}')
        else:
            for m in range(numLines):
                y = self.debugger.lam_g_sols[m]
                plagg.line(x, y.squeeze(), color=next(colors_lagg), legend_label=f'iter. {m}')
        show(column([pg, plagg]))

        if plot_states:

            px = []
            for name in self._model.dynamical_state_names:
                p = figure(width=800, height=800, title=name)
                px.append(p)
            colors_x = itertools.cycle(inferno(numLines))
            if plot_last:
                color = next(colors_x)
                x_pred = np.zeros((self._model.n_x, self.prediction_horizon + 1))
                u_pred = np.zeros((self._model.n_u, self.control_horizon))
                dt_pred = np.zeros((self.prediction_horizon))

                for ii in range(self.prediction_horizon + 1):
                    x_pred[:, ii] = np.array(self.debugger.x_sols[-1])[self._x_ind[ii]] * self._x_scaling
                for ii in range(self.control_horizon):
                    u_pred[:, ii] = np.array(self.debugger.x_sols[-1])[self._u_ind[ii]] * self._u_scaling
                if len(self._dt_ind) > 0:
                    for ii in range(self.prediction_horizon):
                        dt_pred[ii] = np.array(self.debugger.x_sols[-1])[self._dt_ind[ii]]
                else:
                    dt_pred = np.arange(0, (self.prediction_horizon + 1) * self._sampling_interval,
                                        self._sampling_interval)

                for k in range(self._n_x):
                    px[k].line(dt_pred, x_pred[k, :], color=color, legend_label=f'iter. {numLines}')
            else:
                for m in range(numLines):
                    color = next(colors_x)
                    x_pred = np.zeros((self._model.n_x, self.prediction_horizon + 1))
                    u_pred = np.zeros((self._model.n_u, self.control_horizon))
                    dt_pred = np.zeros((self.prediction_horizon))

                    for ii in range(self.prediction_horizon + 1):
                        x_pred[:, ii] = np.array(self.debugger.x_sols[m])[self._x_ind[ii]] * self._x_scaling
                    for ii in range(self.control_horizon):
                        u_pred[:, ii] = np.array(self.debugger.x_sols[m])[self._u_ind[ii]] * self._u_scaling
                    if len(self._dt_ind) > 0:
                        for ii in range(self.prediction_horizon):
                            dt_pred[ii] = np.array(self.debugger.x_sols[m])[self._dt_ind[ii]]
                    else:
                        dt_pred = np.arange(0, (self.prediction_horizon + 1) * self._sampling_interval,
                                            self._sampling_interval)

                    for k in range(self._n_x):
                        px[k].line(dt_pred, x_pred[k, :], color=color, legend_label=f'iter. {m}')

            show(gridplot(px, ncols=3))

    @property
    def sampling_interval(self):
        """

        :return:
        """
        return self._sampling_interval

    @property
    def initial_time(self):
        """

        :return:
        """
        return self._time

    @initial_time.setter
    def initial_time(self, arg):
        self._time = arg

    @property
    def obj_function_stage_cost(self):
        """

        :return:
        """
        return self._lag_term

    @property
    def current_time(self):
        """

        :return:
        """
        return self._time

    @property
    def horizon(self):
        """

        :return:
        """
        return self._horizon

    @horizon.setter
    def horizon(self, arg):
        if isinstance(arg, int) and arg >= 1:
            self._horizon = arg
            self._prediction_horizon = arg
            self._control_horizon = arg
            self._prediction_horizon_is_set = True
            self._control_horizon_is_set = True
        else:
            raise ValueError("The horizon must be a positive nonzero integer")

    @property
    def solution(self):
        """

        :return:
        """
        return self._solution

    @property
    def n_iterations(self):
        """

        :return:
        """
        return self._n_iterations

    def is_setup(self):
        """

        :return:
        """
        # TODO: Could we use self._function instead of self._solver? Then we would not need to overwrite this function.
        return hasattr(self, '_solver')


__all__ = [
    'LinearProgram',
    'QuadraticProgram',
    'NonlinearProgram'
]
