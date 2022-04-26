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

from copy import deepcopy
import platform
from typing import Optional, Sequence, TypeVar, Union
import warnings

import casadi as ca
import numpy as np

from ..base import Base, Vector, Equations, RightHandSide, TimeSeries
from ..machine_learning.base import LearningBase
from ...util.data import DataSet, DataGenerator
from ...util.modeling import GenericCost, QuadraticCost, continuous2discrete
from ...util.parsing import parse_dynamic_equations
from ...util.util import check_if_list_of_string, convert, dump_clean, generate_c_code, is_iterable, is_list_like, \
    is_square, who_am_i, JIT


Symbolic = TypeVar('Symbolic', ca.SX, ca.MX)
Mod = TypeVar('Mod', bound='_Model')
Numeric = Union[int, float]
NumArray = Union[Sequence[Numeric], np.ndarray]
ArrayLike = Union[list, tuple, dict, NumArray, ca.DM, Vector]


class _Model(Base):
    """"""
    def __init__(
            self,
            id: Optional[str] = None,
            name: Optional[str] = None,
            discrete: bool = False,
            solver: Optional[str] = None,
            solver_options: Optional[dict] = None,
            time_unit: str = "h",
            use_sx: bool = True
    ) -> None:
        """Constructor method"""
        super().__init__(id=id, name=name)
        if self._id is None:
            self._create_id()

        prefix = self.name + '_' if self.name is not None else ''
        suffix = '_' + self._id

        if solver is None:
            if not discrete:
                solver = 'cvodes'
        elif discrete:
            solver = None
        if solver_options is None:
            solver_options = {}
        self._solver = solver
        self._solver_opts = solver_options

        self._integrator_function = None
        self._meas_function = None
        self._is_linear = None
        self._is_linearized = None
        self._f0 = None
        self._g0 = None
        self._linearization_about_trajectory = None
        self._is_time_variant = None
        self._state_space_rep = {'A': None, 'B': None, 'C': None, 'D': None, 'M': None}
        self._state_space_changed = False
        self._use_sx = use_sx

        self._empty_model(time_unit, discrete, prefix, suffix)
        self._update_dimensions()

    def __add__(self, other):
        """Addition method"""
        model = self.copy()
        model += other
        return model

    def __iadd__(self, other):
        """Incremental addition method"""
        self._append_learned(other)
        return self

    def __sub__(self, other):
        """Subtraction method"""
        model = self.copy()
        model -= other
        return model

    def __isub__(self, other):
        """Incremental subtraction method"""
        self._append_learned(other, sign='-')
        return self

    def __mul__(self, other):
        """Multiplication method"""
        model = self.copy()
        model *= other
        return model

    def __imul__(self, other):
        """Incremental multiplication method"""
        self._append_learned(other, sign='*')
        return self

    def __truediv__(self, other):
        """True division method"""
        model = self.copy()
        model /= other
        return model

    def __itruediv__(self, other):
        """Incremental true division method"""
        self._append_learned(other, sign='/')
        return self

    def __getstate__(self) -> dict:
        """State getter method"""
        # TODO: Test this
        state = self.__dict__.copy()
        # Remove unpicklable entries
        # del state['']
        # or make them picklable
        # state[''] = []
        return state

    def __setstate__(self, state: dict) -> None:
        """State setter method"""
        # TODO: Test this
        self.__dict__.update(state)
        for attr in ['_dt', '_t', '_x', '_y', '_z', '_u', '_p', '_rhs', '_quad', '_x_eq', '_u_eq', '_x_col', '_z_col']:
            pointer = getattr(self, attr)
            pointer.parent = self
        if not hasattr(self, 'name'):
            self.name = None

    def __repr__(self) -> str:
        """Representation method"""
        args = ""
        if self._id is not None:
            args += f"id='{self._id}'"
        if self.name is not None:
            args += f", name='{self.name}'"
        args += f", discrete={self._rhs.discrete}"
        if self._solver is not None:
            args += f", solver='{self._solver}'"
        if self._solver_opts is not None and self._solver_opts:
            args += f", solver_options={self._solver_opts}"
        args += f", time_unit='{self._t.units[0]}'"
        args += f", use_sx={self._use_sx}"
        return f"{type(self).__name__}({args})"

    def __iter__(self):
        """Item iteration method"""
        yield from self._to_dict().items()

    def _empty_model(self, time_unit, discrete, prefix, suffix):
        """

        :param time_unit:
        :param discrete:
        :param prefix:
        :param suffix:
        :return:
        """
        fx = ca.SX if self._use_sx else ca.MX
        self._dt = Vector(fx, values_or_names='dt', description="time...", labels="time", units=time_unit,
                          id='sampling_time' + suffix, name=prefix + 'sampling_time', parent=self)
        self._t = Vector(fx, values_or_names='t', description="time...", labels="time", units=time_unit,
                         id='time' + suffix, name=prefix + 'time', parent=self)
        self._time_unit = time_unit
        self._x = Vector(fx, shape=(0, 1), id='dynamical_states' + suffix, name=prefix + 'dynamical_states', parent=self
                         )
        self._x_eq = Vector(fx, id='equilibrium_point_dynamical_states' + suffix,
                            name=prefix + 'equilibrium_point_dynamical_states', parent=self)
        self._x_col = Vector(fx, id='collocation_points_dynamical_states' + suffix,
                             name=prefix + 'collocation_points_dynamical_states', parent=self)
        self._y = Vector(fx, shape=(0, 1), id='measurements' + suffix, name=prefix + 'measurements', parent=self)
        self._z = Vector(fx, shape=(0, 1), id='algebraic_states' + suffix, name=prefix + 'algebraic_states', parent=self
                         )
        self._z_eq = Vector(fx, id='equilibrium_point_algebraic_states' + suffix,
                            name=prefix + 'equilibrium_point_algebraic_states', parent=self)
        self._z_col = Vector(fx, id='collocation_points_algebraic_states' + suffix,
                             name=prefix + 'collocation_points_algebraic_states', parent=self)
        self._u = Vector(fx, shape=(0, 1), id='inputs' + suffix, name=prefix + 'inputs', parent=self)
        self._u_eq = Vector(fx, id='equilibrium_point_inputs' + suffix, name=prefix + 'equilibrium_point_inputs',
                            parent=self)
        self._p = Vector(fx, shape=(0, 1), id='parameters' + suffix, name=prefix + 'parameters', parent=self)
        self._rhs = RightHandSide(discrete=discrete, use_sx=self._use_sx, parent=self)
        self._quad = fx()

    def _check_linearity(self) -> None:
        """

        :return:
        """
        is_linear = True
        for eq in ['ode', 'alg', 'meas']:
            for var in ['_x', '_y', '_z', '_u']:  # '_p'
                if hasattr(self, var):
                    is_linear = ca.is_linear(getattr(self._rhs, eq), getattr(self, var).values)
                if not is_linear:
                    break
            if not is_linear:
                break

        self._is_linear = is_linear
        if self._is_linearized and not self._is_linear:
            self._is_linearized = False

    def _check_time_variance(self) -> None:
        """

        :return:
        """
        self._is_time_variant = self._rhs.is_time_variant()

    def _update_dimensions(self) -> None:
        """

        :return:
        """
        self._n_x = self._x.size1()
        self._n_y = self._y.size1()
        self._n_z = self._z.size1()
        self._n_u = self._u.size1()
        self._n_p = self._p.size1()
        if isinstance(self._quad, (ca.SX, ca.MX)):
            self._n_q = self._quad.size1()
        elif isinstance(self._quad, ca.Function):
            self._n_q = self._quad.numel_out()
        elif isinstance(self._quad, (GenericCost, QuadraticCost)):
            # TODO: Maybe we could add a __len__ dunder method to the cost classes, that would wrap this behavior
            self._n_q = self._quad.cost.size1()
        else:
            self._n_q = 0

    def _update_solver(self):
        """

        :return:
        """
        # TODO: Handle options in case of solver switches
        # TODO: Solver could also be the interface to a solver
        if not self.discrete:
            if isinstance(self._solver, str):
                if ca.has_integrator(self._solver):
                    if self._solver == 'cvodes' and (not self._rhs.alg.is_empty() or not self._z.is_empty()):
                        if self._display:
                            print(f"Solver '{self._solver}' is not suitable for DAE systems. Switching to 'idas'...")
                        self._solver = 'idas'
                else:
                    if self._rhs.alg.is_empty() and self._z.is_empty():
                        if self._display:
                            print(f"Solver '{self._solver}' is not available on your system. Switching to 'cvodes'...")
                        self._solver = 'cvodes'
                    else:
                        if self._display:
                            print(f"Solver '{self._solver}' is not available on your system. Switching to 'idas'...")
                        self._solver = 'idas'
            else:
                if self._rhs.alg.is_empty() and self._z.is_empty():
                    self._solver = 'cvodes'
                else:
                    self._solver = 'idas'
        else:
            self._solver = None

    def _parse_equations(self, equations):
        """

        :param equations:
        :return:
        """
        var = {
            'discrete': self._rhs.discrete,
            'use_sx': self._use_sx,
            'dt': self._dt.values.elements(),
            't': self._t.values.elements()
        }
        if not self._x.is_empty():
            var['x'] = self._x.values.elements()
        if not self._y.is_empty():
            var['y'] = self._y.values.elements()
        if not self._z.is_empty():
            var['z'] = self._z.values.elements()
        if not self._u.is_empty():
            var['u'] = self._u.values.elements()
        if not self._p.is_empty():
            var['p'] = self._p.values.elements()

        dae = parse_dynamic_equations(equations, **var)

        description = dae.pop('description')
        labels = dae.pop('labels')
        units = dae.pop('units')

        vec = dae.pop('x')
        if self._x.is_empty():
            self._set('x', vec, description=description['x'], labels=labels['x'], units=units['x'])
        else:
            if vec not in self._x:
                for x in vec:
                    if x not in self._x:
                        self._add('x', x, None)
        vec = dae.pop('y')
        if self._y.is_empty():
            self._set('y', vec, description=description['y'], labels=labels['y'], units=units['y'])
        else:
            if vec not in self._y:
                for y in vec:
                    if y not in self._y:
                        self._add('y', y, None)
        vec = dae.pop('z')
        if self._z.is_empty():
            self._set('z', vec, description=description['z'], labels=labels['z'], units=units['z'])
        else:
            if vec not in self._z:
                for z in vec:
                    if z not in self._z:
                        self._add('z', z, None)
        vec = dae.pop('u')
        if self._u.is_empty():
            self._set('u', vec, description=description['u'], labels=labels['u'], units=units['u'])
        else:
            if vec not in self._u:
                for u in vec:
                    if u not in self._u:
                        self._add('u', u, None)
        vec = dae.pop('p')
        if self._p.is_empty():
            self._set('p', vec, description=description['p'], labels=labels['p'], units=units['p'])
        else:
            if vec not in self._p:
                for p in vec:
                    if p not in self._p:
                        self._add('p', p, None)

        quad = dae.pop('quad')
        if quad:
            self._quad = quad[0]

        if self._rhs.is_empty():
            self._rhs.set(dae)
        else:
            self._rhs.add(dae)
        self._update_solver()

    def _unpack_state_space(self):
        """

        :return:
        """
        n_x = self._n_x
        n_z = self._n_z
        n_u = self._n_u
        n_y = self._n_y

        M = self._state_space_rep['M']
        if M is not None:
            # NOTE: This check is redundant, since it will already be checked during setting of the matrix whether E is
            #  a square matrix
            if not is_square(M):
                raise ValueError("Supplied mass matrix (M) needs to be a square matrix. Either supply a square matrix "
                                 "or the entries of the diagonal as an array-like.")
            m = np.diag(M.copy())
            m_x = np.flatnonzero(m)
            m_z = np.flatnonzero(m == 0)

            M = M.copy()
            M[m_x, m_x] /= m[m_x]  # Normalize
            M_inv = M.copy()
            M_inv[m_x, m_x] = 0
            M_inv[m_z, m_z] = 1

            n_x = m_x.size
            n_z = m_z.size
            if 0 < self._n_x != n_x:
                raise ValueError(f"Dimension mismatch. Supplied mass matrix (M) has {n_x} non-zero elements on its "
                                 f"diagonal (i.e. dynamical states), but the number of set dynamical states is "
                                 f"{self._n_x}.")
            if 0 < self._n_z != n_z:
                raise ValueError(f"Dimension mismatch. Supplied mass matrix (M) has {n_z} zero elements on its diagonal"
                                 f" (i.e. algebraic states), but the number of set algebraic states is {self._n_z}.")

        a = self._state_space_rep['A']
        if a is not None:
            if n_x > 0 and n_z > 0:
                if a.shape != (n_x + n_z, n_x + n_z):
                    raise ValueError(f"Dimension mismatch in state matrix (A). Supplied dimension is "
                                     f"{a.size1()}x{a.size2()}, but required dimension is {n_x + n_z}x{n_x + n_z}.")
                if not self._x.is_empty():
                    x = self._x.values
                else:
                    x = self._x.values.sym('x', n_x)
                if not self._z.is_empty():
                    z = self._z.values
                else:
                    z = self._z.values.sym('z', n_z)
            elif n_x > 0:
                if a.shape != (n_x, n_x):
                    raise ValueError(f"Dimension mismatch in state matrix (A). Supplied dimension is "
                                     f"{a.size1()}x{a.size2()}, but required dimension is {n_x}x{n_x}.")
                if not self._x.is_empty():
                    x = self._x.values
                else:
                    # NOTE: This could happen, if mass matrix (M) was an identity matrix, i.e. no algebraic states
                    x = self._x.values.sym('x', n_x)
                z = self._z.values.sym('z', n_z)  # n_z is 0 here
            elif n_z > 0:
                raise RuntimeError("Only algebraic equations were supplied for the DAE system. ODE's are still missing."
                                   )
            else:
                n_x = a.size1()
                x = self._x.values.sym('x', n_x)
                z = self._z.values.sym('z', n_z)  # n_z is 0 here
        else:
            a = ca.DM.zeros(n_x + n_z, n_x + n_z)
            x = self._x.values
            z = self._z.values
        xz = ca.vertcat(x, z)
        if M is None:
            M = np.diag(n_x * [1] + n_z * [0])
            M_inv = np.diag(n_x * [0] + n_z * [1])
            m = np.diag(M)
            m_x = np.arange(n_x)
            m_z = np.arange(n_z) + n_x

        b = self._state_space_rep['B']
        if b is not None:
            if n_u > 0:
                u = self._u.values
            else:
                n_u = b.size2()
                u = self._u.values.sym('u', n_u)
            if b.shape != (n_x + n_z, n_u):
                raise ValueError(f"Dimension mismatch in input matrix (B). Supplied dimension is "
                                 f"{b.size1()}x{b.size2()}, but required dimension is {n_x + n_z}x{n_u}.")
        else:
            b = ca.DM.zeros(n_x + n_z, n_u)
            u = self._u.values

        c = self._state_space_rep['C']
        if c is not None:
            if n_y > 0:
                y = self._y.values
            else:
                n_y = c.size1()
                y = self._y.values.sym('y', n_y)
            if c.shape != (n_y, n_x + n_z):
                raise ValueError(f"Dimension mismatch in output matrix (C). Supplied dimension is "
                                 f"{c.size1()}x{c.size2()}, but required dimension is {n_y}x{n_x + n_z}.")
        else:
            c = ca.DM.zeros(n_y, n_x + n_z)
            y = self._y.values

        d = self._state_space_rep['D']
        if d is not None:
            if d.shape != (n_y, n_u):
                raise ValueError(f"Dimension mismatch in feedthrough matrix (D). Supplied dimension is "
                                 f"{d.size1()}x{d.size2()}, but required dimension is {n_y}x{n_u}.")
        else:
            d = ca.DM.zeros(n_y, n_u)

        fg = a @ xz + b @ u
        f = (M[m_x, :] @ fg) / m[m_x]  # scale back
        g = M_inv[m_z, :] @ fg
        h = c @ xz + d @ u

        function = ca.Function('function', [x, z, u, self._p.values, self._dt.values, self._t.values], [f, g, h])
        p = ca.SX.get_free(function)

        return f, g, h, x, z, y, u, p

    def _append_learned(self, learned: Union[Equations, LearningBase, Sequence[LearningBase]], sign: str = '+') -> None:
        """

        :param learned:
        :param sign:
        :return:
        """
        if isinstance(learned, LearningBase):
            # if self._slice is None:
            if self._n_x + self._n_z != learned.n_labels:
                raise ValueError(f"Dimension mismatch. Supplied dimension is {learned.n_labels}x1, but required "
                                 f"dimension is {self._n_x + self._n_z}x1.")
            # else:
            #     if len(self._slice) != learned.n_labels:
            #         raise
            dyn_eq = ca.vertcat(self._rhs.ode, self._rhs.alg)
            params = []
            for k, label in enumerate(learned.labels):
                param = self._p.values.sym(label)
                if sign == '+':
                    dyn_eq[k] += param
                elif sign == '-':
                    dyn_eq[k] -= param
                elif sign == '*':
                    dyn_eq[k] *= param
                elif sign == '/':
                    dyn_eq[k] /= param
                params.append(param)
        elif is_iterable(learned) and not isinstance(learned, str):
            if self._n_x + self._n_z != len(learned):
                raise ValueError(f"Dimension mismatch. Supplied dimension is {len(learned)}x1, but required dimension "
                                 f"is {self._n_x + self._n_z}x1.")
            dyn_eq = ca.vertcat(self._rhs.ode, self._rhs.alg)
            params = []
            for k, learn in enumerate(learned):
                if isinstance(learn, LearningBase):
                    label = learn.labels
                    if len(label) > 1:
                        raise ValueError(f"Dimension mismatch. Expected dimension 1x1, got dimension {len(label)}x1.")
                    param = self._p.values.sym(label[0])
                    params.append(param)
                elif learn is None or learn == 0.:
                    param = self._p.values.zeros(1)
                else:
                    raise ValueError(f"Unsupported type {type(learn).__name__} for operations on the model instance.")
                if sign == '+':
                    dyn_eq[k] += param
                elif sign == '-':
                    dyn_eq[k] -= param
                elif sign == '*':
                    dyn_eq[k] *= param
                elif sign == '/':
                    dyn_eq[k] /= param
            learned = [learn for learn in learned if isinstance(learn, LearningBase)]
        elif isinstance(learned, ca.Function):
            n_out = learned.numel_out()
            if self._n_x + self._n_z != n_out:
                raise ValueError(f"Dimension mismatch. Supplied dimension is {n_out}x1, but required dimension is "
                                 f"{self._n_x + self._n_z}x1.")
            dyn_eq = ca.vertcat(self._rhs.ode, self._rhs.alg)
            labels = []
            params = []
            for k in range(n_out):
                labels.append('p' + str(k))
                param = self._p.values.sym(labels[-1])
                if sign == '+':
                    dyn_eq[k] += param
                elif sign == '-':
                    dyn_eq[k] -= param
                elif sign == '*':
                    dyn_eq[k] *= param
                elif sign == '/':
                    dyn_eq[k] /= param
                params.append(param)
            learned = {'labels': labels, 'function': learned}
        else:
            raise TypeError(f"Wrong type {type(learned).__name__}")

        ode = dyn_eq[:self._n_x]
        alg = dyn_eq[self._n_x:]
        if self._n_p == 0:
            self.set_parameters(params)
        else:
            self.add_parameters(params)
        kwargs = {}
        if not ode.is_empty():
            kwargs['ode'] = ode
        if not alg.is_empty():
            kwargs['alg'] = alg
        self.set_equations(**kwargs)

        self.substitute_from(learned)

    def _to_dict(self, use_sx=None):
        """

        :param use_sx:
        :return:
        """
        if use_sx is None:
            use_sx = self._use_sx

        args = [self._quad, self._dt.values, self._t.values, self._x.values, self._z.values, self._u.values,
                self._p.values, self._x_eq.values, self._z_eq.values, self._u_eq.values]
        y = self._y.values
        if not use_sx and self._use_sx is not use_sx:
            # NOTE: At the moment it is assumed, that the quadrature function is empty
            if not self._quad.is_empty():
                raise NotImplementedError
            args[0] = ca.MX()
            if not y.is_empty():
                y = ca.vertcat(*[ca.MX.sym(var.name()) for var in y.elements()])
            else:
                y = ca.MX()
        variables, equations, _ = self._rhs.check_quadrature_function(*args)

        res = {
            'dt': variables[0],
            't': variables[1],
            'x': variables[2],
            'y': y,
            'z': variables[3],
            'u': variables[4],
            'p': variables[5],
            'ode': equations[0],
            'alg': equations[1],
            'meas': equations[2],
            'quad': equations[3]
        }

        x_eq = variables[6]
        z_eq = variables[7]
        u_eq = variables[8]
        if not x_eq.is_empty():
            res['x_eq'] = x_eq
        if not z_eq.is_empty():
            res['z_eq'] = z_eq
        if not u_eq.is_empty():
            res['u_eq'] = u_eq

        if not use_sx:
            if not self._x_col.is_empty():
                res['X'] = self._x_col.values
            if not self._z_col.is_empty():
                res['Z'] = self._z_col.values

        return res

    def _from_dict(self, arg, kwargs):
        """
        Create model from dict

        :param arg:
        :param kwargs:
        :return:
        """
        self._dt.set(arg['dt'], **kwargs['dt'])
        self._t.set(arg['t'], **kwargs['t'])
        self._time_unit = self._t.units[0]  # should always be a list of one entry
        self.set_dynamical_states(arg['x'], **kwargs['x'])
        self.set_dynamical_equations(arg['ode'])
        self.set_algebraic_states(arg['z'], **kwargs['z'])
        self.set_algebraic_equations(arg['alg'])
        self.set_parameters(arg['p'], **kwargs['p'])
        self.set_measurements(arg['y'], **kwargs['y'])
        self.set_measurement_equations(arg['meas'])
        self.set_inputs(arg['u'], **kwargs['u'])
        self.set_quadrature_function(arg['quad'])
        if 'display' in arg:
            self.display = arg['display']
        if 'linearized' in arg:
            self._is_linearized = arg['linearized']
            trj = kwargs.get('linearized')
            if trj is not None and trj == 'trajectory':
                self._linearization_about_trajectory = True
        if 'f0' in arg:
            self._f0 = arg['f0']
        if 'g0' in arg:
            self._g0 = arg['g0']
        # self.discrete = arg['discrete']
        # self.set_quadrature_functions(arg['quad'])
        if 'x_eq' in arg:
            self._x_eq.set(arg['x_eq'], **kwargs['x_eq'])
        if 'z_eq' in arg:
            self._z_eq.set(arg['z_eq'], **kwargs['z_eq'])
        if 'u_eq' in arg:
            self._u_eq.set(arg['u_eq'], **kwargs['u_eq'])
        if 'X' in arg:
            self._x_col.set(arg['X'], **kwargs['X'])
        if 'Z' in arg:
            self._z_col.set(arg['Z'], **kwargs['Z'])

    @property
    def time(self) -> Symbolic:
        """
        Time variable of the model

        :returns: Time variable of the model as a symbolic expression
        :rtype: :py:class:`casadi.casadi.SX` | :py:class:`casadi.casadi.MX`
        :alias: :py:attr:`~hilo_mpc.core.model.Model.t`
        """
        return self._t.values

    t = time

    @property
    def sampling_time(self) -> Symbolic:
        """
        Sampling time variable of the model

        :return: Sampling time variable of the model as a symbolic expression
        :rtype: :py:class:`casadi.casadi.SX` | :py:class:`casadi.casadi.MX`
        :alias: :py:attr:`~hilo_mpc.core.model.Model.dt`
        """
        return self._dt.values

    dt = sampling_time

    @property
    def time_unit(self) -> str:
        """
        Time unit of the model

        :return:
        """
        return self._time_unit

    @property
    def dynamical_states(self) -> Symbolic:
        """
        Differential state vector of the model

        :return: Differential state vector of the model as a symbolic expression
        :rtype: :py:class:`casadi.casadi.SX` | :py:class:`casadi.casadi.MX`
        :alias: :py:attr:`~hilo_mpc.core.model.Model.x`
        """
        return self._x.values

    # @dynamical_states.setter
    def set_dynamical_states(
            self,
            *args: str,
            description: Optional[Union[str, Sequence[str]]] = None,
            labels: Optional[Union[str, Sequence[str]]] = None,
            units: Optional[Union[str, Sequence[str]]] = None,
            **kwargs: str
    ) -> Symbolic:
        """
        Sets the dynamical state vector of the model

        :param args: Possible combinations of how to supply arguments to this method can be taken from the table
        :type args:
        :param description: Description of the single elements of the dynamical state vector. If only a single
            description is supplied, but the vector has a dimension higher than 1, it is assumed that all the elements
            of the vector have the same description. This is just for convenience, since the information stored in the
            description will not be used anywhere else.
        :type description: str, sequence of str, optional
        :param labels: Labels of the single elements of the dynamical state vector. If only a single label is
            supplied, but the vector has a dimension higher than 1, it is assumed that all the elements of the vector
            have the same label. At the moment the label information is only used for automatically generating the
            y-axis labels for the plotting functionality.
        :type labels: str, sequence of str, optional
        :param units: Units of the single elements of the dynamical state vector. If only a single unit is supplied,
            but the vector has a dimension higher than 1, it is assumed that all the elements of the vector have the
            same unit. At the moment the unit information is only used for automatically generating the y-axis labels
            for the plotting functionality.
        :type units: str, sequence of str, optional
        :param kwargs: Additional keyword arguments, see table
        :type kwargs: str, optional
        :return: Differential state vector of the model as a symbolic expression
        :rtype: :py:class:`casadi.casadi.SX` | :py:class:`casadi.casadi.MX`

        ============================================= ==============================
        args                                          kwargs
        ============================================= ==============================
        list of each state variable (sequence of str) --
        dimension of state vector (int)               **name** of state vector (str)
        ============================================= ==============================
        """
        # TODO: Use overload here (from typing library) see https://stackoverflow.com/a/45869067
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

        if not self._state_space_changed:
            self._state_space_changed = True

        return self._x.values

    x = dynamical_states

    @property
    def x_eq(self) -> Symbolic:
        """

        :return:
        """
        return self._x_eq.values

    @property
    def dynamical_state_names(self) -> list[str]:
        """

        :return:
        """
        return self._x.names

    @property
    def dynamical_state_units(self) -> list[str]:
        """

        :return:
        """
        return self._x.units

    @property
    def dynamical_state_labels(self) -> list[str]:
        """

        :return:
        """
        return self._x.labels

    @property
    def dynamical_state_description(self) -> list[str]:
        """

        :return:
        """
        return self._x.description

    @property
    def n_x(self) -> int:
        """

        :return:
        """
        return self._n_x

    @property
    def measurements(self) -> Symbolic:
        """

        :return:
        """
        return self._y.values

    # @measurements.setter
    def set_measurements(self, *args: str, **kwargs: str) -> Symbolic:
        """

        :param args:
        :param kwargs:
        :return:
        """
        # TODO: Use overload here (from typing library) see https://stackoverflow.com/a/45869067
        keep_refs = kwargs.get('keep_refs')
        if keep_refs is None:
            keep_refs = False

        if 'description' not in kwargs:
            kwargs['description'] = ''

        if 'labels' not in kwargs:
            kwargs['labels'] = ''

        if 'units' not in kwargs:
            kwargs['units'] = ''

        if len(args) == 1:
            if not keep_refs or not isinstance(args[0], Vector):
                if isinstance(args[0], int) and 'name' not in kwargs:
                    kwargs['name'] = 'y'
                self._set('y', args[0], **kwargs)
            else:
                # TODO: SX/MX check
                self._y = args[0]
                self._update_dimensions()
        elif len(args) == 2:
            if isinstance(args[0], str) and isinstance(args[1], int):
                kwargs['n_dim'] = args[1]
                self._set('y', args[0], **kwargs)
            elif isinstance(args[0], int) and isinstance(args[1], str):
                kwargs['n_dim'] = args[0]
                self._set('y', args[1], **kwargs)
            else:
                self._set('y', args, **kwargs)
        elif len(args) > 2:
            self._set('y', args, **kwargs)
        else:
            self._set('y', None, **kwargs)

        if not self._state_space_changed:
            self._state_space_changed = True

        return self._y.values

    y = measurements

    @property
    def measurement_names(self) -> list[str]:
        """

        :return:
        """
        return self._y.names

    @property
    def measurement_units(self) -> list[str]:
        """

        :return:
        """
        return self._y.units

    @property
    def measurement_labels(self) -> list[str]:
        """

        :return:
        """
        return self._y.labels

    @property
    def measurement_description(self) -> list[str]:
        """

        :return:
        """
        return self._y.description

    @property
    def n_y(self) -> int:
        """

        :return:
        """
        return self._n_y

    @property
    def algebraic_states(self) -> Symbolic:
        """

        :return:
        """
        return self._z.values

    # @algebraic_states.setter
    def set_algebraic_states(self, *args: str, **kwargs: str) -> Symbolic:
        """

        :param args:
        :param kwargs:
        :return:
        """
        # TODO: Use overload here (from typing library) see https://stackoverflow.com/a/45869067
        keep_refs = kwargs.get('keep_refs')
        if keep_refs is None:
            keep_refs = False

        if 'description' not in kwargs:
            kwargs['description'] = ''

        if 'labels' not in kwargs:
            kwargs['labels'] = ''

        if 'units' not in kwargs:
            kwargs['units'] = ''

        if len(args) == 1:
            if not keep_refs or not isinstance(args[0], Vector):
                if isinstance(args[0], int) and 'name' not in kwargs:
                    kwargs['name'] = 'z'
                self._set('z', args[0], **kwargs)
            else:
                # TODO: SX/MX check
                self._z = args[0]
                self._update_dimensions()
        elif len(args) == 2:
            if isinstance(args[0], str) and isinstance(args[1], int):
                kwargs['n_dim'] = args[1]
                self._set('z', args[0], **kwargs)
            elif isinstance(args[0], int) and isinstance(args[1], str):
                kwargs['n_dim'] = args[0]
                self._set('z', args[1], **kwargs)
            else:
                self._set('z', args, **kwargs)
        elif len(args) > 2:
            self._set('z', args, **kwargs)
        else:
            self._set('z', None, **kwargs)

        self._update_solver()

        if not self._state_space_changed:
            self._state_space_changed = True

        return self._z.values

    z = algebraic_states

    @property
    def z_eq(self) -> Symbolic:
        """

        :return:
        """
        return self._z_eq.values

    @property
    def algebraic_state_names(self) -> list[str]:
        """

        :return:
        """
        return self._z.names

    @property
    def algebraic_state_units(self) -> list[str]:
        """

        :return:
        """
        return self._z.units

    @property
    def algebraic_state_labels(self) -> list[str]:
        """

        :return:
        """
        return self._z.labels

    @property
    def algebraic_state_description(self) -> list[str]:
        """

        :return:
        """
        return self._z.description

    @property
    def n_z(self) -> int:
        """

        :return:
        """
        return self._n_z

    @property
    def inputs(self) -> Symbolic:
        """

        :return:
        """
        return self._u.values

    # @inputs.setter
    def set_inputs(self, *args: str, **kwargs: str) -> Symbolic:
        """

        :param args:
        :param kwargs:
        :return:
        """
        # TODO: Use overload here (from typing library) see https://stackoverflow.com/a/45869067
        keep_refs = kwargs.get('keep_refs')
        if keep_refs is None:
            keep_refs = False

        if 'description' not in kwargs:
            kwargs['description'] = ''

        if 'labels' not in kwargs:
            kwargs['labels'] = ''

        if 'units' not in kwargs:
            kwargs['units'] = ''

        if len(args) == 1:
            if not keep_refs or not isinstance(args[0], Vector):
                if isinstance(args[0], int) and 'name' not in kwargs:
                    kwargs['name'] = 'u'
                self._set('u', args[0], **kwargs)
            else:
                # TODO: SX/MX check
                self._u = args[0]
                self._update_dimensions()
        elif len(args) == 2:
            if isinstance(args[0], str) and isinstance(args[1], int):
                kwargs['n_dim'] = args[1]
                self._set('u', args[0], **kwargs)
            elif isinstance(args[0], int) and isinstance(args[1], str):
                kwargs['n_dim'] = args[0]
                self._set('u', args[1], **kwargs)
            else:
                self._set('u', args, **kwargs)
        elif len(args) > 2:
            self._set('u', args, **kwargs)
        else:
            self._set('u', None, **kwargs)

        if not self._state_space_changed:
            self._state_space_changed = True

        return self._u.values

    u = inputs

    @property
    def u_eq(self) -> Symbolic:
        """

        :return:
        """
        return self._u_eq.values

    @property
    def input_names(self) -> list[str]:
        """

        :return:
        """
        return self._u.names

    @property
    def input_units(self) -> list[str]:
        """

        :return:
        """
        return self._u.units

    @property
    def input_labels(self) -> list[str]:
        """

        :return:
        """
        return self._u.labels

    @property
    def input_description(self) -> list[str]:
        """

        :return:
        """
        return self._u.description

    @property
    def n_u(self) -> int:
        """

        :return:
        """
        return self._n_u

    @property
    def parameters(self) -> Symbolic:
        """

        :return:
        """
        return self._p.values

    # @parameters.setter
    def set_parameters(self, *args: str, **kwargs: str) -> Symbolic:
        """

        :param args:
        :param kwargs:
        :return:
        """
        # TODO: Use overload here (from typing library) see https://stackoverflow.com/a/45869067
        keep_refs = kwargs.get('keep_refs')
        if keep_refs is None:
            keep_refs = False

        if 'description' not in kwargs:
            kwargs['description'] = ''

        if 'labels' not in kwargs:
            kwargs['labels'] = ''

        if 'units' not in kwargs:
            kwargs['units'] = ''

        if len(args) == 1:
            if not keep_refs or not isinstance(args[0], Vector):
                if isinstance(args[0], int) and 'name' not in kwargs:
                    kwargs['name'] = 'p'
                self._set('p', args[0], **kwargs)
            else:
                # TODO: SX/MX check
                self._p = args[0]
                self._update_dimensions()
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

        if not self._state_space_changed:
            self._state_space_changed = True

        return self._p.values

    p = parameters

    @property
    def parameter_names(self) -> list[str]:
        """

        :return:
        """
        return self._p.names

    @property
    def parameter_units(self) -> list[str]:
        """

        :return:
        """
        return self._p.units

    @property
    def parameter_labels(self) -> list[str]:
        """

        :return:
        """
        return self._p.labels

    @property
    def parameter_description(self) -> list[str]:
        """

        :return:
        """
        return self._p.description

    @property
    def n_p(self) -> int:
        """

        :return:
        """
        return self._n_p

    @property
    def dynamical_equations(self) -> Symbolic:
        """

        :return:
        """
        return self._rhs.ode

    # @dynamical_equations.setter
    def set_dynamical_equations(self, arg, delimiter=None, check_properties=False) -> None:
        """

        :param arg:
        :param delimiter:
        :param check_properties:
        :return:
        """
        if isinstance(arg, int):
            raise TypeError(f"Wrong type of arguments for function {who_am_i()}")
        elif isinstance(arg, str):
            # TODO: What if string is a path to a file
            if delimiter is None:
                if platform.system() == 'Linux':
                    delimiter = '\n'
                elif platform.system() == 'Windows':
                    delimiter = '\r\n'
                else:
                    msg = f"Parsing from string not supported for operating system {platform.system()}"
                    raise NotImplementedError(msg)
            equations = arg.split(delimiter)
            if self._n_x > len(equations):
                raise ValueError(f"Dimension mismatch between supplied dynamical equations ({len(equations)}) and "
                                 f"dynamical states ({self._n_x})")
            if not self._rhs.discrete:
                equations = [f'd{self._x.names[k]}(t)/dt=' + val for k, val in enumerate(equations)]
            else:
                equations = [f'{self._x.names[k]}(k+1)=' + val for k, val in enumerate(equations)]
            self._parse_equations(equations)
        elif not isinstance(arg, (ca.SX, ca.MX)) and check_if_list_of_string(arg):
            if self._n_x > len(arg):
                raise ValueError(f"Dimension mismatch between supplied dynamical equations ({len(arg)}) and dynamical "
                                 f"states ({self._n_x})")
            if not self._rhs.discrete:
                equations = [f'd{self._x.names[k]}(t)/dt=' + val for k, val in enumerate(arg)]
            else:
                equations = [f'{self._x.names[k]}(k+1)=' + val for k, val in enumerate(arg)]
            self._parse_equations(equations)
        else:
            self._set('ode', arg)

        if check_properties:
            self._check_linearity()
            self._check_time_variance()

        if not self._state_space_changed:
            self._state_space_changed = True

    ode = dynamical_equations

    @property
    def measurement_equations(self) -> Symbolic:
        """

        :return:
        """
        return self._rhs.meas

    # @measurement_equations.setter
    def set_measurement_equations(self, arg, delimiter=None, check_properties=False) -> None:
        """

        :param arg:
        :param delimiter:
        :param check_properties:
        :return:
        """
        if isinstance(arg, int):
            raise TypeError(f"Wrong type of arguments for function {who_am_i()}")
        elif isinstance(arg, str):
            # TODO: What if string is a path to a file
            if delimiter is None:
                if platform.system() == 'Linux':
                    delimiter = '\n'
                elif platform.system() == 'Windows':
                    delimiter = '\r\n'
                else:
                    msg = f"Parsing from string not supported for operating system {platform.system()}"
                    raise NotImplementedError(msg)
            equations = arg.split(delimiter)
            if self._n_y > len(equations):
                raise ValueError(f"Dimension mismatch between supplied measurement equations ({len(equations)}) and "
                                 f"measurements ({self._n_y})")
            if self._y.is_empty():
                self.set_measurements('y', len(equations))
            equations = [f'{self._y.names[k]}(k)=' + val for k, val in enumerate(equations)]
            self._parse_equations(equations)
        elif not isinstance(arg, (ca.SX, ca.MX)) and check_if_list_of_string(arg):
            if self._n_y > len(arg):
                raise ValueError(f"Dimension mismatch between supplied measurement equations ({len(arg)}) and "
                                 f"measurements ({self._n_y})")
            if self._y.is_empty():
                self.set_measurements('y', len(arg))
            equations = [f'{self._y.names[k]}(k)=' + val for k, val in enumerate(arg)]
            self._parse_equations(equations)
        else:
            self._set('meas', arg)

        if check_properties:
            self._check_linearity()
            self._check_time_variance()

        if not self._state_space_changed:
            self._state_space_changed = True

        if self._y.is_empty():
            self.set_measurements('y', self._rhs.meas.size1())

    meas = measurement_equations

    @property
    def algebraic_equations(self) -> Symbolic:
        """

        :return:
        """
        return self._rhs.alg

    # @algebraic_equations.setter
    def set_algebraic_equations(self, arg, delimiter=None, check_properties=False) -> None:
        """

        :param arg:
        :param delimiter:
        :param check_properties:
        :return:
        """
        if isinstance(arg, int):
            raise TypeError(f"Wrong type of arguments for function {who_am_i()}")
        elif isinstance(arg, str):
            # TODO: What if string is a path to a file
            if delimiter is None:
                if platform.system() == 'Linux':
                    delimiter = '\n'
                elif platform.system() == 'Windows':
                    delimiter = '\r\n'
                else:
                    msg = f"Parsing from string not supported for operating system {platform.system()}"
                    raise NotImplementedError(msg)
            equations = arg.split(delimiter)
            if self._n_z > len(equations):
                raise ValueError(f"Dimension mismatch between supplied algebraic equations ({len(equations)}) and "
                                 f"algebraic states ({self._n_z})")
            equations = [f'0=' + val for val in equations]
            self._parse_equations(equations)
        elif not isinstance(arg, (ca.SX, ca.MX)) and check_if_list_of_string(arg):
            if self._n_z > len(arg):
                raise ValueError(f"Dimension mismatch between supplied algebraic equations ({len(arg)}) and algebraic "
                                 f"states ({self._n_z})")
            equations = [f'0=' + val for val in arg]
            self._parse_equations(equations)
        else:
            self._set('alg', arg)
            self._update_solver()

        if check_properties:
            self._check_linearity()
            self._check_time_variance()

        if not self._state_space_changed:
            self._state_space_changed = True

    alg = algebraic_equations

    @property
    def quadrature_function(self):
        """

        :return:
        """
        return self._quad

    # @quadrature_function.setter
    def set_quadrature_function(self, arg):
        """

        :param arg:
        :return:
        """
        if isinstance(arg, int):
            raise TypeError(f"Wrong type of arguments for function {who_am_i()}")
        elif isinstance(arg, str):
            if not self._rhs.discrete:
                equations = [f'int={arg}']
            else:
                equations = [f'sum={arg}']
            self._parse_equations(equations)
        elif isinstance(arg, ca.Function):
            all_the_names = set(self._x.names + self._y.names + self._z.names + self._u.names + self._p.names)
            if len(all_the_names.intersection(arg.name_in())) == arg.n_in():
                self._quad = arg
            else:
                not_in_model = [k for k in arg.name_in() if k not in all_the_names]
                msg = f"The following arguments to the supplied function are not variables of the model: " \
                      f"{', '.join(not_in_model)}"
                raise ValueError(msg)  # TODO: Which error makes the most sense here? RuntimeError?
        elif isinstance(arg, (GenericCost, QuadraticCost)):
            # TODO: We probably need some further checks here
            self._quad = arg
        else:
            self._set('quad', arg)
        self._update_dimensions()
        # self._n_q = self._quad.size1()
        # self.check_consistency()

    quad = quadrature_function

    @property
    def n_q(self):
        """

        :return:
        """
        return self._n_q

    def set_equations(self, equations=None, **kwargs):
        """

        :param equations:
        :param kwargs:
        :return:
        """
        # TODO: See add_equations()
        # TODO: What about keep_refs?
        ode = kwargs.get('ode', None)
        alg = kwargs.get('alg', None)
        quad = kwargs.get('quad', None)
        meas = kwargs.get('meas', None)
        if equations is not None:
            if isinstance(equations, (dict, RightHandSide)):
                self._set('rhs', equations)
                if 'alg' in equations:
                    self._update_solver()
            elif isinstance(equations, (list, tuple)):
                if all(isinstance(k, str) for k in equations):
                    self._parse_equations(equations)
                else:
                    # What now?
                    msg = "Input not recognized. No changes applied."
                    warnings.warn(msg)
            elif isinstance(equations, str):
                # TODO: What if string is a path to a file
                if platform.system() == 'Linux':
                    self._parse_equations(equations.split('\n'))
                elif platform.system() == 'Windows':
                    self._parse_equations(equations.split('\n'))
                else:
                    msg = f"Parsing from string not supported for operating system {platform.system()}"
                    warnings.warn(msg)
                if not self._state_space_changed:
                    self._state_space_changed = True
        else:
            if ode is not None:
                self.set_dynamical_equations(ode, check_properties=False)
            if alg is not None:
                self.set_algebraic_equations(alg, check_properties=False)
            if quad is not None:
                self.set_quadrature_function(quad)
            if meas is not None:
                self.set_measurement_equations(meas, check_properties=False)

    @property
    def state_matrix(self):
        """

        :return:
        """
        if not self._rhs.is_empty():
            if self._is_linear is None:
                self._check_linearity()
            if self._is_linear:
                if self._state_space_changed or self._rhs.matrix_notation is None:
                    self._rhs.generate_matrix_notation(ca.vertcat(self._x.values, self._z.values), self._u.values)
                    self._state_space_changed = False
                return self._rhs.matrix_notation['A']
        elif self._state_space_rep['A'] is not None:
            return self._state_space_rep['A']
        return None

    @state_matrix.setter
    def state_matrix(self, a):
        a = convert(a, ca.SX)
        if not a.is_square():
            raise ValueError(f"The state matrix needs to be a square matrix. Supplied matrix has dimensions "
                             f"{a.size1()}x{a.size2()}.")
        self._state_space_rep['A'] = a
        if not self._state_space_changed:
            self._state_space_changed = True

    system_matrix = state_matrix

    A = state_matrix

    @property
    def input_matrix(self):
        """

        :return:
        """
        if not self._rhs.is_empty():
            if self._is_linear is None:
                self._check_linearity()
            if self._is_linear:
                if self._state_space_changed or self._rhs.matrix_notation is None:
                    self._rhs.generate_matrix_notation(ca.vertcat(self._x.values, self._z.values), self._u.values)
                    self._state_space_changed = False
                return self._rhs.matrix_notation['B']
        elif self._state_space_rep['B'] is not None:
            return self._state_space_rep['B']
        return None

    @input_matrix.setter
    def input_matrix(self, b):
        b = convert(b, ca.SX)
        self._state_space_rep['B'] = b
        if not self._state_space_changed:
            self._state_space_changed = True

    B = input_matrix

    @property
    def output_matrix(self):
        """

        :return:
        """
        if not self._rhs.is_empty():
            if self._is_linear is None:
                self._check_linearity()
            if self._is_linear:
                if self._state_space_changed or self._rhs.matrix_notation is None:
                    self._rhs.generate_matrix_notation(ca.vertcat(self._x.values, self._z.values), self._u.values)
                    self._state_space_changed = False
                return self._rhs.matrix_notation['C']
        elif self._state_space_rep['C'] is not None:
            return self._state_space_rep['C']
        return None

    @output_matrix.setter
    def output_matrix(self, c):
        c = convert(c, ca.SX)
        self._state_space_rep['C'] = c
        if not self._state_space_changed:
            self._state_space_changed = True

    C = output_matrix

    @property
    def feedthrough_matrix(self):
        """

        :return:
        """
        if not self._rhs.is_empty():
            if self._is_linear is None:
                self._check_linearity()
            if self._is_linear:
                if self._state_space_changed or self._rhs.matrix_notation is None:
                    self._rhs.generate_matrix_notation(ca.vertcat(self._x.values, self._z.values), self._u.values)
                    self._state_space_changed = False
                return self._rhs.matrix_notation['D']
        elif self._state_space_rep['D'] is not None:
            return self._state_space_rep['D']
        return None

    @feedthrough_matrix.setter
    def feedthrough_matrix(self, d):
        d = convert(d, ca.SX)
        self._state_space_rep['D'] = d
        if not self._state_space_changed:
            self._state_space_changed = True

    feedforward_matrix = feedthrough_matrix

    D = feedthrough_matrix

    @property
    def mass_matrix(self):
        """

        :return:
        """
        if not self._rhs.is_empty():
            if self._is_linear is None:
                self._check_linearity()
            if self._is_linear:
                return np.diag(self._n_x * [1] + self._n_z * [0])
        elif self._state_space_rep['M'] is not None:
            return self._state_space_rep['M']
        return None

    @mass_matrix.setter
    def mass_matrix(self, m):
        m = convert(m, np.ndarray)
        if 1 in m.shape or m.ndim == 1:
            if m.ndim == 2:
                m = m.flatten()
            m = np.diag(m)
        if not is_square(m):
            raise ValueError(f"The mass matrix (M) needs to be a square matrix. Supplied matrix has dimensions "
                             f"{m.shape[0]}x{m.shape[1]}.")
        self._state_space_rep['M'] = m
        if not self._state_space_changed:
            self._state_space_changed = True

    M = mass_matrix

    @property
    def discrete(self):
        """

        :return:
        """
        return self._rhs.discrete

    @discrete.setter
    def discrete(self, boolean):
        self._rhs.discrete = boolean

    @property
    def continuous(self):
        """

        :return:
        """
        return self._rhs.continuous

    @continuous.setter
    def continuous(self, boolean):
        self._rhs.continuous = boolean

    @property
    def solver(self):
        """

        :return:
        """
        return self._solver

    @solver.setter
    def solver(self, arg):
        # TODO: Check which solver options can be used for multiple solvers and keep them
        # TODO: Solver could also be the interface to a solver
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
    def options(self):
        """

        :return:
        """
        return dump_clean(self._solver_opts)

    def add_dynamical_states(self, states, position=None):
        """

        :param states:
        :param position:
        :return:
        """
        # TODO: Add support for description, labels and units
        self._add('x', states, position)
        if not self._state_space_changed:
            self._state_space_changed = True

    def add_measurements(self, measurements, position=None):
        """

        :param measurements:
        :param position:
        :return:
        """
        # TODO: Add support for description, labels and units
        self._add('y', measurements, position)
        if not self._state_space_changed:
            self._state_space_changed = True

    def add_algebraic_states(self, states, position=None):
        """

        :param states:
        :param position:
        :return:
        """
        # TODO: Add support for description, labels and units
        # TODO: Add self._update_solver
        self._add('z', states, position)
        if not self._state_space_changed:
            self._state_space_changed = True

    def add_inputs(self, inputs, position=None):
        """

        :param inputs:
        :param position:
        :return:
        """
        # TODO: Add support for description, labels and units
        self._add('u', inputs, position)
        if not self._state_space_changed:
            self._state_space_changed = True

    def add_parameters(self, parameters, position=None, **kwargs):
        """

        :param parameters:
        :param position:
        :param kwargs:
        :return:
        """
        # TODO: Add support for description, labels and units
        self._add('p', parameters, position, **kwargs)
        if not self._state_space_changed:
            self._state_space_changed = True

    def add_dynamical_equations(self, equations, position=None, check_properties=False):
        """

        :param equations:
        :param position:
        :param check_properties:
        :return:
        """
        self._add('ode', equations, position)
        if check_properties:
            self._check_linearity()
            self._check_time_variance()
        if not self._state_space_changed:
            self._state_space_changed = True

    def add_algebraic_equations(self, equations, position=None, check_properties=False):
        """

        :param equations:
        :param position:
        :param check_properties:
        :return:
        """
        # TODO: Add self._update_solver
        self._add('alg', equations, position)
        if check_properties:
            self._check_linearity()
            self._check_time_variance()
        if not self._state_space_changed:
            self._state_space_changed = True

    def add_quadrature_functions(self, functions, position=None):
        """

        :param functions:
        :param position:
        :return:
        """
        self._add('quad', functions, position)

    def add_measurement_equations(self, equations, position=None, check_properties=False):
        """

        :param equations:
        :param position:
        :param check_properties:
        :return:
        """
        self._add('meas', equations, position)
        if check_properties:
            self._check_linearity()
            self._check_time_variance()
        if not self._state_space_changed:
            self._state_space_changed = True

    def add_equations(self, equations=None, **kwargs):
        """

        :param equations:
        :param kwargs:
        :return:
        """
        # NOTE: Right now, if the 'equations' argument is given, the arguments 'ode', 'alg', 'quad' and 'meas' will be
        # overwritten.
        # TODO: Arguments that are not part of the 'equations' dict should not be overwritten (see NOTE above)
        if equations is not None:
            if isinstance(equations, RightHandSide):
                ode = equations.ode
                alg = equations.alg
                quad = None
                meas = equations.meas
            elif isinstance(equations, dict):
                ode = equations.get('ode', None)
                alg = equations.get('alg', None)
                quad = equations.get('quad', None)
                meas = equations.get('meas', None)
        else:
            ode = kwargs.get('ode', None)
            alg = kwargs.get('alg', None)
            quad = kwargs.get('quad', None)
            meas = kwargs.get('meas', None)

        if ode is not None:
            self.add_dynamical_equations(ode, check_properties=False)
        if alg is not None:
            self.add_algebraic_equations(alg, check_properties=False)
        if quad is not None:
            self.add_quadrature_functions(quad)
        if meas is not None:
            self.add_measurement_equations(meas, check_properties=False)

    def add_options(self, opts=None, keys=None, values=None):
        """

        :param opts:
        :param keys:
        :param values:
        :return:
        """
        # TODO: Check solver options at the end and omit options that are not used for the current solver
        if opts is not None:
            self._solver_opts.update(deepcopy(opts))

        if keys is not None and values is not None:
            if isinstance(keys, (list, tuple)):
                if isinstance(values, (list, tuple)):
                    assert len(keys) == len(values)
                    for k, key in enumerate(keys):
                        if isinstance(values[k], (dict, list, set)):
                            self._solver_opts[key] = deepcopy(values[k])
                        else:
                            self._solver_opts[key] = values[k]
                else:
                    for key in keys:
                        self._solver_opts[key] = values
            else:
                if isinstance(values, (dict, list, set)):
                    self._solver_opts[keys] = deepcopy(values)
                else:
                    self._solver_opts[keys] = values

    def check_consistency(self):
        """

        :return:
        """
        super().check_consistency()

        # TODO: Check size2() as well (it should not be higher than 1)
        if self._x.size1() != self._rhs.ode.size1():
            msg = "Dimension mismatch between the dynamical states and the dynamical equations"
            raise RuntimeError(msg)
        if self._y.size1() != self._rhs.meas.size1():
            msg = "Dimension mismatch between the measurements and the measurement equations"
            raise RuntimeError(msg)
        if self._z.size1() != self._rhs.alg.size1():
            msg = "Dimension mismatch between the algebraic states and the algebraic equations"
            raise RuntimeError(msg)

    def check_solver(self, solver):
        """

        :param solver:
        :return:
        """
        # TODO: Solver could also be the interface to a solver
        if not self.discrete:
            if isinstance(solver, str):
                if ca.has_integrator(solver):
                    if (solver == 'cvodes' or solver == 'rk') and (not self._rhs.alg.is_empty()
                                                                   or not self._z.is_empty()):
                        if self._display:
                            msg = f"Solver '{solver}' is not suitable for DAE systems. Use 'idas' or 'collocation' " \
                                  f"instead"
                            warnings.warn(msg)
                        return False
                    else:
                        return True
                else:
                    if self._rhs.alg.is_empty() and self._z.is_empty():
                        if self._display:
                            msg = f"Solver '{solver}' is not available on your system. Use 'cvodes', 'rk' or " \
                                  f"'collocation' instead"
                            warnings.warn(msg)
                        return False
                    else:
                        if self._display:
                            msg = f"Solver '{solver}' is not available on your system. Use 'idas' or 'collocation' " \
                                  f"instead"
                            warnings.warn(msg)
                        return False
            else:
                raise TypeError("Solver type must be a string")
        else:
            print("Model is discrete. No solver is required.")

    def copy(self, name=None, setup=False, **kwargs):
        """

        :param name:
        :param setup:
        :param kwargs:
        :return:
        """
        use_sx = kwargs.get('use_sx')
        if use_sx is None:
            use_sx = self._use_sx

        problem = self._to_dict(use_sx=use_sx)
        if self._rhs.is_empty():
            f, g, h, x, z, y, u, free_vars = self._unpack_state_space()
            problem['x'] = x
            problem['y'] = y
            problem['z'] = z
            problem['u'] = u
            problem['p'] = ca.vertcat(problem['p'], free_vars)
            problem['ode'] = f
            problem['alg'] = g
            problem['meas'] = h
        options = {
            'dt': {
                'description': self._dt.description,
                'labels': self._dt.labels,
                'units': self._dt.units
            },
            't': {
                'description': self._t.description,
                'labels': self._t.labels,
                'units': self._t.units
            },
            'x': {
                'description': self._x.description,
                'labels': self._x.labels,
                'units': self._x.units,
            },
            'y': {
                'description': self._y.description,
                'labels': self._y.labels,
                'units': self._y.units
            },
            'z': {
                'description': self._z.description,
                'labels': self._z.labels,
                'units': self._z.units
            },
            'u': {
                'description': self._u.description,
                'labels': self._u.labels,
                'units': self._u.units
            },
            'p': {
                'description': self._p.description,
                'labels': self._p.labels,
                'units': self._p.units
            }
        }
        if 'x_eq' in problem:
            options['x_eq'] = {
                'description': self._x_eq.description,
                'labels': self._x_eq.labels,
                'units': self._x_eq.units,
            }
        if 'z_eq' in problem:
            options['z_eq'] = {
                'description': self._z_eq.description,
                'labels': self._z_eq.labels,
                'units': self._z_eq.units,
            }
        if 'u_eq' in problem:
            options['u_eq'] = {
                'description': self._u_eq.description,
                'labels': self._u_eq.labels,
                'units': self._u_eq.units
            }
        if 'X' in problem:
            options['X'] = {
                'description': self._x_col.description,
                'labels': self._x_col.labels,
                'units': self._x_col.units,
            }
        if 'Z' in problem:
            options['Z'] = {
                'description': self._z_col.description,
                'labels': self._z_col.labels,
                'units': self._z_col.units
            }

        if name is None and self.name is not None:
            name = self.name
        model = _Model(name=name, discrete=self._rhs.discrete, solver=self._solver, solver_options=self._solver_opts,
                       use_sx=use_sx)

        # TODO: What about other properties?
        problem['display'] = self._display
        problem['linearized'] = self._is_linearized
        if self._f0 is not None:
            problem['f0'] = self._f0
        if self._g0 is not None:
            problem['g0'] = self._g0

        model._from_dict(problem, options)

        if setup:
            if not model._rhs.is_empty():
                model.setup()
            else:
                warnings.warn("Model to be copied has no right-hand side, so the new model cannot be set up.")
        else:
            if not model._rhs.is_empty():
                if model.is_linear():
                    model._rhs.generate_matrix_notation(ca.vertcat(model.x, model.z), model.u)

        return model

    def discretize(self, method, inplace=False, **kwargs):
        """

        :param method:
        :param inplace:
        :param kwargs:
        :return:
        """
        # TODO: Add zero order hold for linear systems
        # TODO: What about keep_refs?
        if self._rhs.discrete:
            print("Model is already discrete. Nothing to be done.")
            if not inplace:
                return self
            else:
                return

        self.check_consistency()

        problem = self._to_dict()
        if self._rhs.is_empty():
            f, g, h, x, z, y, u, free_vars = self._unpack_state_space()
            problem['x'] = x
            problem['y'] = y
            problem['z'] = z
            problem['u'] = u
            problem['p'] = ca.vertcat(problem['p'], free_vars)
            problem['ode'] = f
            problem['alg'] = g
            problem['meas'] = h
        # TODO: What about mixtures of SX and MX?
        use_sx = self._use_sx

        if method == 'rk4':
            kwargs['class'] = 'explicit'
            kwargs['method'] = 'rk'
            kwargs['order'] = 4
            category = 'runge-kutta'
        elif method == 'erk':
            kwargs['class'] = 'explicit'
            kwargs['method'] = 'rk'
            category = 'runge-kutta'
        elif method == 'irk':
            kwargs['class'] = 'implicit'
            kwargs['method'] = 'rk'
            category = 'runge-kutta'
        elif method == 'collocation':
            kwargs['class'] = 'implicit'
            kwargs['method'] = 'collocation'
            category = 'runge-kutta'
        else:
            raise NotImplementedError(f"The discretization method {method} is not implemented")

        options = {
            'dt': {
                'description': self._dt.description,
                'labels': self._dt.labels,
                'units': self._dt.units
            },
            't': {
                'description': self._t.description,
                'labels': self._t.labels,
                'units': self._t.units
            },
            'x': {
                'description': self._x.description,
                'labels': self._x.labels,
                'units': self._x.units,
            },
            'y': {
                'description': self._y.description,
                'labels': self._y.labels,
                'units': self._y.units
            },
            'z': {
                'description': self._z.description,
                'labels': self._z.labels,
                'units': self._z.units
            },
            'u': {
                'description': self._u.description,
                'labels': self._u.labels,
                'units': self._u.units
            },
            'p': {
                'description': self._p.description,
                'labels': self._p.labels,
                'units': self._p.units
            }
        }
        if 'x_eq' in problem:
            options['x_eq'] = {
                'description': self._x_eq.description,
                'labels': self._x_eq.labels,
                'units': self._x_eq.units,
            }
        if 'z_eq' in problem:
            options['z_eq'] = {
                'description': self._z_eq.description,
                'labels': self._z_eq.labels,
                'units': self._z_eq.units,
            }
        if 'u_eq' in problem:
            options['u_eq'] = {
                'description': self._u_eq.description,
                'labels': self._u_eq.labels,
                'units': self._u_eq.units
            }

        continuous2discrete(problem, category=category, **kwargs)
        if kwargs['class'] == 'explicit' and kwargs['method'] == 'rk' and self._n_z > 0:
            # TODO: Ignore variable t here if system is not time-variant, otherwise we need to supply a value for t as
            #  well when calling the corresponding MXFunction (during Model.setup(), in self._rhs.to_function())
            dt = problem['dt']
            t = problem['t']
            x = problem['x']
            y = problem['y']
            z = problem['z']
            u = problem['u']
            p = problem['p']
            Z = problem['discretization_points']
            ode = problem['ode']
            alg = problem['alg']
            meas = problem['meas']
            quad = problem['quad']

            # q = root_finder([Z, dt, t, x, u, p], [ode, alg])

            ode = ca.Function('ode', [dt, t, x, Z, u, p], [ode])
            _alg = ca.Function('alg', [Z, dt, t, x, u, p], [alg])
            _alg = ca.rootfinder('alg', 'newton', _alg)
            alg = ca.Function('alg', [z, dt, t, x, u, p], [self._rhs.alg])
            alg = ca.rootfinder('alg', 'newton', alg)
            meas = ca.Function('meas', [dt, t, x, z, u, p], [meas])
            quad = ca.Function('quad', [dt, t, x, Z, u, p], [quad])

            if use_sx:
                dt = ca.MX.sym('dt')
                t = ca.MX.sym('t')
                # NOTE: ca.vertcat() would return a ca.Sparsity object, if the corresponding lists were empty. Since we
                #  don't want do deal with ca.Sparsity at the moment during model setup, we do this workaround
                x = [ca.MX.sym(k) for k in self._x.names]
                if x:
                    x = ca.vertcat(*x)
                else:
                    x = ca.MX()
                y = [ca.MX.sym(k) for k in self._y.names]
                if y:
                    y = ca.vertcat(*y)
                else:
                    y = ca.MX()
                z = [ca.MX.sym(k) for k in self._z.names]
                if z:
                    z = ca.vertcat(*z)
                else:
                    z = ca.MX()
                u = [ca.MX.sym(k) for k in self._u.names]
                if u:
                    u = ca.vertcat(*u)
                else:
                    u = ca.MX()
                p = [ca.MX.sym(k) for k in self._p.names]
                if p:
                    p = ca.vertcat(*p)
                else:
                    p = ca.MX()
                Z = [ca.MX.sym(k.name()) for k in Z.elements()]
                if Z:
                    Z = ca.vertcat(*Z)
                else:
                    Z = ca.MX()
            use_sx = False

            order = kwargs['order'] if 'order' in kwargs else 1
            v = _alg(Z, dt, t, x, u, p)
            xf = ode(dt, t, x, v, u, p)
            qf = quad(dt, t, x, v, u, p)
            zf = alg(z, dt, t, xf, u, p)
            yf = meas(dt, t, xf, zf, u, p)

            options['Z'] = {
                'description': order * self._z.description,
                'labels': order * self._z.labels,
                'units': order * self._z.units
            }
            options['degree'] = order

            del problem['discretization_points']
            problem['dt'] = dt
            problem['t'] = t
            problem['x'] = x
            problem['y'] = y
            problem['z'] = z
            problem['u'] = u
            problem['p'] = p
            problem['ode'] = xf
            problem['alg'] = zf
            problem['meas'] = yf
            problem['quad'] = qf
            problem['Z'] = Z
        elif kwargs['class'] == 'implicit' and kwargs['method'] == 'collocation':
            dt = problem['dt']
            t = problem['t']
            x = problem['x']
            y = problem['y']
            z = problem['z']
            u = problem['u']
            p = problem['p']
            X = ca.vertcat(*problem['collocation_points_ode'])
            Z = ca.vertcat(*problem['collocation_points_alg'])
            col_eq = problem['collocation_equations']
            ode = problem['ode']
            alg = problem['alg']
            meas = problem['meas']
            quad = problem['quad']

            col_eq = ca.Function('col_eq', [ca.vertcat(X, Z), dt, t, x, z, u, p], [col_eq])  # We probably don't need
            # 'z' here
            col_eq = ca.rootfinder('col_eq', 'newton', col_eq)
            function = ca.Function('function', [ca.vertcat(X, Z), dt, t, x, z, u, p], [ode, alg, quad, meas])

            if use_sx:
                dt = ca.MX.sym('dt')
                t = ca.MX.sym('t')
                # NOTE: ca.vertcat() would return a ca.Sparsity object, if the corresponding lists were empty. Since we
                #  don't want to deal with ca.Sparsity at the moment during model setup, we do this workaround
                x = [ca.MX.sym(k) for k in self._x.names]
                if x:
                    x = ca.vertcat(*x)
                else:
                    x = ca.MX()
                y = [ca.MX.sym(k) for k in self._y.names]
                if y:
                    y = ca.vertcat(*y)
                else:
                    y = ca.MX()
                z = [ca.MX.sym(k) for k in self._z.names]
                if z:
                    z = ca.vertcat(*z)
                else:
                    z = ca.MX()
                u = [ca.MX.sym(k) for k in self._u.names]
                if u:
                    u = ca.vertcat(*u)
                else:
                    u = ca.MX()
                p = [ca.MX.sym(k) for k in self._p.names]
                if p:
                    p = ca.vertcat(*p)
                else:
                    p = ca.MX()
                X = [ca.MX.sym(k.name()) for k in X.elements()]
                if X:
                    X = ca.vertcat(*X)
                else:
                    X = ca.MX()
                Z = [ca.MX.sym(k.name()) for k in Z.elements()]
                if Z:
                    Z = ca.vertcat(*Z)
                else:
                    Z = ca.MX()
            use_sx = False

            degree = kwargs['degree'] if 'degree' in kwargs else 2
            v = col_eq(ca.vertcat(X, Z), dt, t, x, z, u, p)
            xf, zf, qf, yf = function(v, dt, t, x, z, u, p)

            options['X'] = {
                'description': degree * self._x.description,
                'labels': degree * self._x.labels,
                'units': degree * self._x.units
            }
            options['Z'] = {
                'description': degree * self._z.description,
                'labels': degree * self._z.labels,
                'units': degree * self._z.units
            }

            del problem['collocation_points_ode']
            del problem['collocation_points_alg']
            del problem['collocation_equations']
            problem['dt'] = dt
            problem['t'] = t
            problem['x'] = x
            problem['y'] = y
            problem['z'] = z
            problem['u'] = u
            problem['p'] = p
            problem['ode'] = xf
            problem['alg'] = zf
            problem['meas'] = yf
            problem['quad'] = qf
            problem['X'] = X
            problem['Z'] = Z

        if not inplace:
            # TODO: Remove solver options that are not used for discrete models
            # TODO: Use Model.copy here
            if self.name is not None:
                model = _Model(name=self.name + '_discretized', discrete=True, solver_options=self._solver_opts,
                               use_sx=use_sx)
            else:
                model = _Model(name=self.name, discrete=True, solver_options=self._solver_opts, use_sx=use_sx)

            problem['display'] = self._display
            problem['linearized'] = self._is_linearized
            if self._f0 is not None:
                problem['f0'] = self._f0
            if self._g0 is not None:
                problem['g0'] = self._g0

            model._from_dict(problem, options)
            return model
        else:
            self.discrete = True
            self._update_solver()
            if self._function is not None:
                warnings.warn("The model needs to be set up again in order to run simulations. If you want to prevent "
                              "this warning message from appearing, run Model.discretize(...) before Model.setup().")
                self._function = None
            if self._integrator_function is not None:
                self._integrator_function = None
            if self._meas_function is not None:
                self._meas_function = None
            if use_sx:
                if self._rhs.is_empty():
                    self.set_dynamical_states(problem['x'])
                    self.set_measurements(problem['y'])
                    self.set_algebraic_states(problem['z'])
                    self.set_inputs(problem['u'])
                    self.set_parameters(problem['p'])
                equations = {
                    'ode': problem['ode'],
                    'alg': problem['alg'],
                    'quad': problem['quad'],
                    'meas': problem['meas']
                }
                self.set_equations(**equations)
            else:
                # TODO: Add _discretized suffix?
                prefix = self.name + '_' if self.name is not None else ''
                suffix = '_' + self._id
                self._empty_model(False, "", True, prefix, suffix)
                self._from_dict(problem, options)

    def is_autonomous(self) -> bool:
        """

        :return:
        """
        return self._n_u == 0

    def is_linear(self) -> bool:
        """

        :return:
        """
        self._check_linearity()
        return self._is_linear

    def is_linearized(self) -> bool:
        """

        :return:
        """
        return self._is_linearized

    def is_time_variant(self):
        """

        :return:
        """
        self._check_time_variance()
        return self._is_time_variant

    def linearize(self, name: str = None, trajectory: Optional[dict] = None) -> Mod:
        """

        :param name:
        :param trajectory:
        :return:
        """
        # TODO: What about nonlinear quadrature functions?
        if not self._rhs.is_empty():
            self._check_linearity()
            if self._is_linearized:
                print("Model is already linearized. Nothing to be done.")
                return self
            if self._is_linear:
                print("Model is already linear. Linearization is not necessary. Nothing to be done.")
                return self
        else:
            if any(val is not None for val in self._state_space_rep.values()):
                print("Model is already linear. Linearization is not necessary. Nothing to be done.")
                return self
            else:
                print("Model is empty. Nothing to be done.")
                return self

        problem = self._to_dict()
        dt = problem['dt']
        t = problem['t']
        x = problem['x']
        # y = problem['y']
        z = problem['z']
        u = problem['u']
        p = problem['p']
        equations = [problem['ode'], problem['alg'], problem['meas'], problem['quad']]
        options = {
            'dt': {
                'description': self._dt.description,
                'labels': self._dt.labels,
                'units': self._dt.units
            },
            't': {
                'description': self._t.description,
                'labels': self._t.labels,
                'units': self._t.units
            },
            'x': {
                'description': self._x.description,
                'labels': self._x.labels,
                'units': self._x.units,
            },
            'y': {
                'description': self._y.description,
                'labels': self._y.labels,
                'units': self._y.units
            },
            'z': {
                'description': self._z.description,
                'labels': self._z.labels,
                'units': self._z.units
            },
            'u': {
                'description': self._u.description,
                'labels': self._u.labels,
                'units': self._u.units
            },
            'p': {
                'description': self._p.description,
                'labels': self._p.labels,
                'units': self._p.units
            }
        }

        x_s = None
        z_s = None
        u_s = None
        if trajectory is not None:
            # TODO: Add support for trajectories in CasADi variables
            trajectories = []
            x_trajectory = trajectory.get('x')
            u_trajectory = trajectory.get('u')
            if isinstance(x_trajectory, (list, tuple)):
                trajectories.extend(x_trajectory)
            elif x_trajectory is not None:
                trajectories.append(x_trajectory)
            if len(trajectories) != self._n_x:
                raise ValueError(f"Dimension mismatch for dynamical state trajectory. Expected trajectory of dimension "
                                 f"{self._n_x}, got {len(trajectories)}.")
            if isinstance(u_trajectory, (list, tuple)):
                trajectories.extend(u_trajectory)
            elif u_trajectory is not None:
                trajectories.append(u_trajectory)
            if len(trajectories) - self._n_x != self._n_u:
                raise ValueError(f"Dimension mismatch for input trajectory. Expected trajectory of dimension "
                                 f"{self._n_u}, got {len(trajectories) - self._n_x}.")
            trajectories = [f'trj_{key}(k)=' + val for key, val in enumerate(trajectories)]
            var = {
                'discrete': self._rhs.discrete,
                'use_sx': self._use_sx,
                'dt': [dt],
                't': [t]
            }
            trj = parse_dynamic_equations(trajectories, **var)['meas']
            x_s = ca.vertcat(*trj[:self._n_x])
            u_s = ca.vertcat(*trj[self._n_x:])

        if self._use_sx:
            if self._n_x > 0:
                dx = ca.vertcat(*[ca.SX.sym('d' + var.name()) for var in x.elements()])
                if x_s is None:
                    x_s = ca.vertcat(*[ca.SX.sym(var.name() + '_eq') for var in x.elements()])
            else:
                dx = ca.SX()
                if x_s is None:
                    x_s = ca.SX()
            if self._n_z > 0:
                dz = ca.vertcat(*[ca.SX.sym('d' + var.name()) for var in z.elements()])
                if z_s is None:
                    z_s = ca.vertcat(*[ca.SX.sym(var.name() + '_eq') for var in z.elements()])
            else:
                dz = ca.SX()
                if z_s is None:
                    z_s = ca.SX()
            if self._n_u > 0:
                du = ca.vertcat(*[ca.SX.sym('d' + var.name()) for var in u.elements()])
                if u_s is None:
                    u_s = ca.vertcat(*[ca.SX.sym(var.name() + '_eq') for var in u.elements()])
            else:
                du = ca.SX()
                if u_s is None:
                    u_s = ca.SX()
        else:
            if self._n_x > 0:
                dx = ca.vertcat(*[ca.MX.sym('d' + var.name()) for var in x.elements()])
                if x_s is None:
                    x_s = ca.vertcat(*[ca.MX.sym(var.name() + '_eq') for var in x.elements()])
            else:
                dx = ca.MX()
                if x_s is None:
                    x_s = ca.MX()
            if self._n_z > 0:
                dz = ca.vertcat(*[ca.MX.sym('d' + var.name()) for var in z.elements()])
                if z_s is None:
                    z_s = ca.vertcat(*[ca.MX.sym(var.name() + '_eq') for var in z.elements()])
            else:
                dz = ca.MX()
                if z_s is None:
                    z_s = ca.MX()
            if self._n_u > 0:
                du = ca.vertcat(*[ca.MX.sym('d' + var.name()) for var in u.elements()])
                if u_s is None:
                    u_s = ca.vertcat(*[ca.MX.sym(var.name() + '_eq') for var in u.elements()])
            else:
                du = ca.MX()
                if u_s is None:
                    u_s = ca.MX()
        options['x_eq'] = {
            'description': self._x.description,
            'labels': self._x.labels,
            'units': self._x.units,
        }
        options['z_eq'] = {
            'description': self._z.description,
            'labels': self._z.labels,
            'units': self._z.units,
        }
        options['u_eq'] = {
            'description': self._u.description,
            'labels': self._u.labels,
            'units': self._u.units
        }

        xzu = ca.vertcat(x, z, u)
        xzu_s = ca.vertcat(x_s, z_s, u_s)
        for k, eq in enumerate(equations):
            if eq.is_empty():
                continue

            if not dx.is_empty():
                dyn_state_derivative = ca.jtimes(eq, x, dx)
            else:
                if self._use_sx:
                    dyn_state_derivative = ca.SX.zeros(eq.shape)
                else:
                    dyn_state_derivative = ca.MX.zeros(eq.shape)

            if not dz.is_empty():
                alg_state_derivative = ca.jtimes(eq, z, dz)
            else:
                if self._use_sx:
                    alg_state_derivative = ca.SX.zeros(eq.shape)
                else:
                    alg_state_derivative = ca.MX.zeros(eq.shape)

            if not du.is_empty():
                input_derivative = ca.jtimes(eq, u, du)
            else:
                if self._use_sx:
                    input_derivative = ca.SX.zeros(eq.shape)
                else:
                    input_derivative = ca.MX.zeros(eq.shape)

            taylor = ca.substitute(dyn_state_derivative + alg_state_derivative + input_derivative, xzu, xzu_s)

            if k == 0:
                if self._rhs.discrete:
                    eq -= x
                problem['f0'] = ca.substitute(eq, xzu, xzu_s)
                problem['ode'] = taylor
            elif k == 1:
                problem['g0'] = ca.substitute(eq, xzu, xzu_s)
                problem['alg'] = taylor
            elif k == 2:
                problem['meas'] = taylor
            elif k == 3:
                # TODO: See above (quadrature)
                pass

        if name is None and self.name is not None:
            name = self.name + '_linearized'
        model = _Model(name=name, discrete=self._rhs.discrete, solver=self._solver, solver_options=self._solver_opts,
                       use_sx=self._use_sx)

        problem['x'] = dx
        problem['z'] = dz
        problem['u'] = du
        if trajectory is None:
            problem['x_eq'] = x_s
            problem['z_eq'] = z_s
            problem['u_eq'] = u_s

        # TODO: What about other properties?
        problem['display'] = self._display
        problem['linearized'] = True

        if trajectory is None:
            if 'f0' in problem:
                problem['f0'] = ca.Function('f0', [dt, t, x_s, z_s, u_s, p], [problem['f0']])
            if 'g0' in problem:
                problem['g0'] = ca.Function('g0', [dt, t, x_s, z_s, u_s, p], [problem['g0']])
        else:
            options['linearized'] = 'trajectory'
        model._from_dict(problem, options)

        return model

    def remove_dynamical_states(self, indices_or_names):
        """

        :param indices_or_names:
        :return:
        """
        self._remove('x', indices_or_names)
        if not self._state_space_changed:
            self._state_space_changed = True

    def remove_measurements(self, indices_or_names):
        """

        :param indices_or_names:
        :return:
        """
        self._remove('y', indices_or_names)
        if not self._state_space_changed:
            self._state_space_changed = True

    def remove_algebraic_states(self, indices_or_names):
        """

        :param indices_or_names:
        :return:
        """
        self._remove('z', indices_or_names)
        self._update_solver()
        if not self._state_space_changed:
            self._state_space_changed = True

    def remove_inputs(self, indices_or_names):
        """

        :param indices_or_names:
        :return:
        """
        self._remove('u', indices_or_names)
        if not self._state_space_changed:
            self._state_space_changed = True

    def remove_parameters(self, indices_or_names):
        """

        :param indices_or_names:
        :return:
        """
        self._remove('p', indices_or_names)
        if not self._state_space_changed:
            self._state_space_changed = True

    def remove_dynamical_equations(self, indices, check_properties=False):
        """

        :param indices:
        :param check_properties:
        :return:
        """
        self._remove('ode', indices)
        if check_properties:
            self._check_linearity()
            self._check_time_variance()
        if not self._state_space_changed:
            self._state_space_changed = True

    def remove_algebraic_equations(self, indices, check_properties=False):
        """

        :param indices:
        :param check_properties:
        :return:
        """
        self._remove('alg', indices)
        self._update_solver()
        if check_properties:
            self._check_linearity()
            self._check_time_variance()
        if not self._state_space_changed:
            self._state_space_changed = True

    def remove_quadrature_functions(self, indices):
        """

        :param indices:
        :return:
        """
        self._remove('quad', indices)

    def remove_measurement_equations(self, indices, check_properties=False):
        """

        :param indices:
        :param check_properties:
        :return:
        """
        self._remove('meas', indices)
        if check_properties:
            self._check_linearity()
            self._check_time_variance()
        if not self._state_space_changed:
            self._state_space_changed = True

    def remove_equations(self, ode=None, alg=None, quad=None, meas=None):
        """

        :param ode:
        :param alg:
        :param quad:
        :param meas:
        :return:
        """
        if ode is not None:
            self.remove_dynamical_equations(ode, check_properties=False)
        if alg is not None:
            self.remove_algebraic_equations(alg, check_properties=False)
        if quad is not None:
            self.remove_quadrature_functions(quad)
        if meas is not None:
            self.remove_measurement_equations(meas, check_properties=False)

    def remove_options(self, keys):
        """

        :param keys:
        :return:
        """
        if isinstance(keys, (list, tuple)):
            for key in keys:
                # NOTE: This should work, otherwise use self._solver_opts.pop(key, None)
                # It could be possible that 'key' is present during the if-condition, but not at the 'del' statement
                if key in self._solver_opts:
                    del self._solver_opts[key]
        elif isinstance(keys, str):
            # NOTE: See first if-condition
            if keys in self._solver_opts:
                del self._solver_opts[keys]
        else:
            msg = f"Wrong type '{type(keys)}' of attribute for function {who_am_i()}"
            warnings.warn(msg)

    def reset(self):
        """

        :return:
        """
        # TODO: What else to reset? Maybe solution?
        self._function = None
        self._integrator_function = None
        self._meas_function = None

    def setup(self, **kwargs):
        """

        :param kwargs:
        :return:
        """
        # TODO: Add keyword argument 'input_function' to allow continuous input signals via PID or LQR
        # TODO: Add support for time grid
        # TODO: Check time-dependent odes
        opts = kwargs.get('opts')
        if opts is not None:
            options = deepcopy(opts)
        else:
            options = {}
        options.update(self._solver_opts)

        dt = kwargs.get('dt')
        if dt is not None:
            t0 = kwargs.get('t0', 0.)
            if t0 != 0.:
                # TODO: add tf (tf = t0 + dt)?
                options['t0'] = t0
            options['tf'] = dt
        else:
            options['grid'] = kwargs.get('grid')

        if self._rhs.is_empty():
            f, g, h, x, z, y, u, free_vars = self._unpack_state_space()

            # TODO: Is this sufficient enough?
            if not self._rhs.discrete and ca.depends_on(f, self._dt.values):
                print(f"Switching model '{self.name}' to discrete, since discrete time dynamics were supplied for the "
                      f"state equations.")
                self._rhs.discrete = True
            elif self._rhs.discrete and not ca.depends_on(f, self._dt.values):
                warnings.warn("The model was initialized as a discrete model, but no sampling time was supplied. "
                              "Use Model.dt to extract the sampling time.")

            if not x.is_empty() and self._x.is_empty():
                self.set_dynamical_states(x)
            if not z.is_empty() and self._z.is_empty():
                self.set_algebraic_states(z)
            if not y.is_empty() and self._y.is_empty():
                self.set_measurements(y)
            if not u.is_empty() and self._u.is_empty():
                self.set_inputs(u)
            if free_vars:
                if self._p.is_empty():
                    self.set_parameters(free_vars)
                else:
                    self.add_parameters(free_vars)
            self.set_equations(ode=f, alg=g, meas=h)
        else:
            if not self._rhs.discrete and ca.depends_on(self._rhs.ode, self._dt.values):
                print(f"Switching model '{self.name}' to discrete, since discrete time dynamics were supplied for the "
                      f"state equations.")
                self._rhs.discrete = True
            elif self._rhs.discrete and not ca.depends_on(self._rhs.ode, self._dt.values):
                warnings.warn("The model was initialized as a discrete model, but no sampling time was implemented in "
                              "the model. Assuming that the sampling time 'dt' is already integrated numerically.")

        self._check_linearity()
        self._check_time_variance()

        generate_matrix_notation = kwargs.get('generate_matrix_notation')
        if generate_matrix_notation is None:
            generate_matrix_notation = True
        if generate_matrix_notation:
            if self._rhs.matrix_notation is None or self._state_space_changed:
                generate_matrix_notation = self._is_linear and generate_matrix_notation
            else:
                generate_matrix_notation = False

        col_points = {}
        if not self._x_col.is_empty():
            col_points['X'] = self._x_col.values
        if not self._z_col.is_empty():
            col_points['Z'] = self._z_col.values

        if isinstance(self._quad, (GenericCost, QuadraticCost)):
            quad = self._quad.cost
        else:
            quad = self._quad

        steady_state = None
        if self._is_linearized:
            steady_state = (self._x_eq.values, self._z_eq.values, self._u_eq.values)

        function, integrator, meas = self._rhs.to_function('integrator', quadrature=quad,
                                                           external_parameters=steady_state, opts=options,
                                                           generate_matrix_notation=generate_matrix_notation,
                                                           **col_points)

        use_c_code = kwargs.get('c_code', False)
        if use_c_code:
            gen_path, gen_name, gen_opts = self._generator(**kwargs)
            if gen_path is not None:
                self._c_name = generate_c_code(function, gen_path, gen_name, opts=gen_opts)
                if self._compiler_opts['method'] in JIT and self._compiler_opts['compiler'] == 'shell':
                    self._compiler_opts['path'] = gen_path
                if not self._rhs.discrete:
                    self._function = 'integrator_continuous'
                else:
                    self._function = 'integrator_discrete'
                super().setup()
            else:
                self._function = function
        else:
            self._function = function
        self._integrator_function = integrator
        self._meas_function = meas

        self.check_consistency()

    def substitute(self, *args, **kwargs):
        """

        :param args:
        :param kwargs:
        :return:
        """
        if args and kwargs:
            raise TypeError("Model.substitute() can only accept positional or keyword arguments, not both")

        if len(args) == 2:
            pairs = [(args[0], args[1])]
        elif len(args) > 2:
            raise TypeError("No more than 2 positional arguments allowed")
        else:
            pairs = kwargs.items()

        model_changed = False
        for key, val in pairs:
            if key in ['x', 'states']:
                if isinstance(val, (ca.SX, ca.MX)):
                    self._rhs.substitute(self._x, val)
            elif key in ['p', 'param', 'params', 'parameter', 'parameters']:
                if isinstance(val, dict):
                    # TODO: Support for symbolic parameters
                    self._rhs.substitute(self._p, **val)
                    self._p.remove(index=self._p.index(val.keys()))
                model_changed = True

        if model_changed:
            if self._function is not None:
                warnings.warn("The model needs to be set up again in order to run simulations. If you want to prevent "
                              "this warning message from appearing, run Model.substitute(...) before Model.setup().")
                self._function = None
            if self._integrator_function is not None:
                self._integrator_function = None
            if self._meas_function is not None:
                self._meas_function = None

            self._check_linearity()
            self._check_time_variance()
            if not self._state_space_changed:
                self._state_space_changed = True

    def substitute_from(self, obj):
        """

        :param obj:
        :return:
        """
        # TODO: Error handling
        if isinstance(obj, LearningBase):
            kwargs = {
                'x': {},
                'y': {},
                'z': {},
                'u': {},
                'p': {}
            }
            predict = getattr(obj, 'predict')
            all_the_names = self._x.names + self._y.names + self._z.names + self._u.names + self._p.names
            all_the_variables = ca.vertcat(self._x.values, self._y.values, self._z.values, self._u.values,
                                           self._p.values)
            use_sx = isinstance(all_the_variables, ca.SX)
            # all_the_indices = [k for k, val in enumerate(all_the_names) if val in obj.features]
            all_the_indices = [all_the_names.index(val) for val in obj.features]
            if hasattr(obj, 'features') and hasattr(obj, 'labels') and callable(predict):
                # TODO: Check the following if-else statement
                if use_sx:
                    out = predict(all_the_variables[all_the_indices])
                else:
                    out = predict(ca.vertcat(*[all_the_variables[i] for i in all_the_indices]))
                # TODO: Maybe add a parameter return_variance (defaults to False) to the predict method of the GPR
                #  instead?
                if is_list_like(out):
                    out = out[0]
                for k, label in enumerate(obj.labels):
                    if label in self._x:
                        kwargs['x'][label] = out[k]
                    elif label in self._y:
                        kwargs['y'][label] = out[k]
                    elif label in self._z:
                        kwargs['z'][label] = out[k]
                    elif label in self._u:
                        kwargs['u'][label] = out[k]
                    elif label in self._p:
                        kwargs['p'][label] = out[k]
            self.substitute(**kwargs)
        elif isinstance(obj, (list, set, tuple)):
            for learned in obj:
                self.substitute_from(learned)
        elif isinstance(obj, dict):
            kwargs = {
                'x': {},
                'y': {},
                'z': {},
                'u': {},
                'p': {}
            }
            all_the_names = self._x.names + self._y.names + self._z.names + self._u.names + self._p.names
            all_the_variables = ca.vertcat(self._x.values, self._y.values, self._z.values, self._u.values,
                                           self._p.values)
            use_sx = isinstance(all_the_variables, ca.SX)
            function = obj.get('function')
            if function is not None and isinstance(function, ca.Function):
                # all_the_indices = [k for k, val in enumerate(all_the_names) if val in function.name_in()]
                all_the_indices = [all_the_names.index(val) for val in function.name_in()]
                if function.numel_in() == function.n_in():
                    if use_sx:
                        out = function(*all_the_variables[all_the_indices].elements())
                    else:
                        out = function(*[all_the_variables[k] for k in all_the_indices])
                else:
                    out = function(all_the_variables[all_the_indices])
                labels = obj.get('labels')
                if labels is not None:
                    for k, label in enumerate(labels):
                        if label in self._x:
                            kwargs['x'][label] = out[k]
                        elif label in self._y:
                            kwargs['y'][label] = out[k]
                        elif label in self._z:
                            kwargs['z'][label] = out[k]
                        elif label in self._u:
                            kwargs['u'][label] = out[k]
                        elif label in self._p:
                            kwargs['p'][label] = out[k]
                    self.substitute(**kwargs)
        else:
            warnings.warn(f"Input type {type(obj).__name__} to Model.substitute_from() not supported")

    def scale(
            self,
            value: Union[Numeric, NumArray],
            id: Optional[str] = None,
            name: Optional[str] = None
    ) -> None:
        """

        :param value:
        :param id:
        :param name:
        :return:
        """
        if id is None and name is None:
            # TODO: Raise error here?
            warnings.warn("No identifier and no name supplied. No changes applied...")
            return

        if id is not None and name is not None:
            # TODO: Raise error here?
            warnings.warn("Both keyword arguments 'id' and 'name' were supplied. This is not allowed. No changes "
                          "applied...")
            return

        if id is not None:
            if id == 'x':
                var = self._x
                names = self._x.names
                values = value
                eq = 'ode'
            elif id == 'y':
                var = self._y
                names = self._y.names
                values = value
                eq = 'meas'
            elif id == 'z':
                var = self._z
                names = self._z.names
                values = value
                eq = 'alg'
            elif id == 'u':
                var = self._u
                names = self._u.names
                values = value
                eq = None
            elif id == 'p':
                var = self._p
                names = self._p.names
                values = value
                eq = None
            else:
                # TODO: Raise error here?
                warnings.warn(f"Unrecognized identifier: {id}\nNo changes applied...")
                return

        if name is not None:
            if name in self._x.names:
                var = self._x
                names = [name]
                index = self._x.index(names)
                values = ca.DM.ones(self._x.shape)
                values[index] = value
                eq = 'ode'
            elif name in self._y.names:
                var = self._y
                names = [name]
                index = self._y.index(names)
                values = ca.DM.ones(self._y.shape)
                values[index] = value
                eq = 'meas'
            elif name in self._z.names:
                var = self._z
                names = [name]
                index = self._z.index(names)
                values = ca.DM.ones(self._z.shape)
                values[index] = value
                eq = 'alg'
            elif name in self._u.names:
                var = self._u
                names = [name]
                index = self._u.index(names)
                values = ca.DM.ones(self._u.shape)
                values[index] = value
                eq = None
            elif name in self._p.names:
                var = self._p
                names = [name]
                index = self._p.index(names)
                values = ca.DM.ones(self._p.shape)
                values[index] = value
                eq = None
            else:
                # TODO: Raise error here?
                warnings.warn(f"Unrecognized name: {name}\nNo changes applied...")
                return

        self._rhs.scale(var, names, values, eq=eq)


class Model(_Model):
    """
    Class representing the dynamic model of a system over time

    :param id: The identifier of the model object. If no identifier is given, a random one will be generated.
    :type id: str, optional
    :param name: The name of the model object. By default the model object has no name.
    :type name: str, optional
    :param discrete: Whether or not the dynamics of the model are discrete, defaults to False (i.e. the model has
        continuous-time dynamics)
    :type discrete: bool, optional
    :param solver: Name of the solver to be used for continuous-time integration, defaults to 'cvodes' if **discrete**
        is set to False. If **discrete** is set to True the argument **solver** will not be used.
    :type solver: str, optional
    :param solver_options: Options to be passed to the selected **solver**. Only used when the model has continuous-time
        dynamics. By default no options will be set.
    :type solver_options: dict, optional
    :param time_unit: The time unit of the model. Common time units are seconds ('s'), hours ('h') or days ('d'). Will
        only be used for plotting and as an information storage. Defaults to 'h'.
    :type time_unit: str
    :param plot_backend: Plotting library that is used to visualize simulated data. At the moment only
        `Matplotlib <https://matplotlib.org/>`_ and `Bokeh <https://bokeh.org/>`_ are supported. By default no plotting
        library is selected, i.e. no plots can be generated.
    :type plot_backend: str, optional
    """
    def __init__(
            self,
            id: Optional[str] = None,
            name: Optional[str] = None,
            discrete: bool = False,
            solver: Optional[str] = None,
            solver_options: Optional[dict] = None,
            time_unit: str = "h",
            plot_backend: Optional[str] = None
    ) -> None:
        """Constructor method"""
        super().__init__(id=id, name=name, discrete=discrete, solver=solver, solver_options=solver_options,
                         time_unit=time_unit, use_sx=True)

        self._solution = TimeSeries(plot_backend, parent=self)
        self._steady_state = TimeSeries(plot_backend, parent=self)
        # TODO: Figure out if we really need self._collocation_points as a TimeSeries class, since the only occurrences
        #  at the moment just involve the degree.
        self._collocation_points = TimeSeries(plot_backend, parent=self)

        self._dxdp_nnz = 0
        self._dydp_nnz = 0
        self._dzdp_nnz = 0

    def __repr__(self) -> str:
        args = ""
        if self._id is not None:
            args += f"id='{self._id}'"
        if self.name is not None:
            args += f", name='{self.name}'"
        args += f", discrete={self._rhs.discrete}"
        if self._solver is not None:
            args += f", solver='{self._solver}'"
        if self._solver_opts is not None and self._solver_opts:
            args += f", solver_options={self._solver_opts}"
        if self._solution.plot_backend is not None:
            args += f", plot_backend='{self._solution.plot_backend}'"
        return f"{type(self).__name__}({args})"

    def __str__(self) -> str:
        str_repr = f"\nModel {self.name}\n\n"
        add_newline = False

        for k, ode in enumerate(self._rhs.ode.elements()):
            if k == 0:
                str_repr += "# ODEs\n"
                add_newline = True
            str_repr += f"d{self._x.names[k]}/dt = {ode}\n"
        if add_newline:
            str_repr += "\n"
            add_newline = False

        for k, alg in enumerate(self._rhs.alg.elements()):
            if k == 0:
                str_repr += "# ALGs\n"
                add_newline = True
            str_repr += f"0 = {alg}\n"
        if add_newline:
            str_repr += "\n"
            add_newline = False

        for k, meas in enumerate(self._rhs.meas.elements()):
            if k == 0:
                str_repr += "# MEAS\n"
            str_repr += f"{self._y.names[k]} = {meas}\n"

        return str_repr

    @property
    def initial_time(self) -> Optional[ca.DM]:
        """

        :return:
        :rtype: DM
        """
        if 't' not in self._solution or self._solution.get_by_id('t').is_empty():
            # TODO: Put warnings here, to inform user?
            return None
        return self._solution.get_by_id('t:0')

    t0 = initial_time

    @property
    def initial_dynamical_states(self) -> Optional[ca.DM]:
        """

        :return:
        """
        if 'x' not in self._solution or self._solution.get_by_id('x').is_empty():
            # TODO: Put warnings here, to inform user?
            return None
        return self._solution.get_by_id('x:0')

    x0 = initial_dynamical_states

    @property
    def initial_algebraic_states(self) -> Optional[ca.DM]:
        """

        :return:
        """
        if 'z' not in self._solution or self._solution.get_by_id('z').is_empty():
            # TODO: Put warnings here, to inform user?
            return None
        return self._solution.get_by_id('z:0')

    z0 = initial_algebraic_states

    def set_initial_conditions(
            self,
            x0: Union[Numeric, ArrayLike],
            t0: Union[int, float] = 0.,
            z0: Optional[Union[Numeric, ArrayLike]] = None
    ) -> None:
        """

        :param x0:
        :param t0:
        :param z0:
        :return:
        """
        # TODO: Check consistency of solution object?
        # TODO: See set_equilibrium_point (see TODO above)
        if not self._solution.is_set_up():
            raise RuntimeError("Model is not set up. Run Model.setup() before setting the initial conditions.")

        # TODO: Support for time grids
        self._solution.set('t', t0)

        if not self._x.is_empty():
            # NOTE: 'warnings.catch_warnings' is not thread-safe. Don't know if that needs to concern us.
            # TODO: We could also rewrite TimeSeries.set so that it doesn't care if it's already populated and handle
            #  the processing entirely here (if Vector is not empty)
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter('always')
                self._solution.set('x', x0)
                # self._solution.set('x0_noise', 0.)
                self._solution.update(x_noise=ca.DM.zeros(self._n_x))
            match = [k for k in w if k.category == UserWarning and str(k.message) ==
                     "Vector is not empty. No changes applied."]
            if match:
                warnings.warn("The model has already been simulated. Please reset the solution of the model in order to"
                              " record a new solution. No changes applied.")
        else:
            # TODO: Raise an error here?
            warnings.warn("Initial dynamical states cannot be set, since no dynamical states are defined for the model."
                          )

        if not self._z.is_empty():
            if z0 is None:
                raise ValueError("The initial values for the algebraic states also have to be set in order to simulate "
                                 "the model.")
            # NOTE: 'warnings.catch_warnings' is not thread-safe. Don't know if that needs to concern us.
            # TODO: We could also rewrite TimeSeries.set so that it doesn't care if it's already populated and handle
            #  the processing entirely here (if Vector is not empty)
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter('always')
                self._solution.set('z', z0)
                # self._solution.set('z0_noise', 0.)
                self._solution.update(z_noise=ca.DM.zeros(self._n_z))
            match = [k for k in w if k.category == UserWarning and str(k.message) ==
                     "Vector is not empty. No changes applied."]
            if match:
                warnings.warn("The model has already been simulated. Please reset the solution of the model in order to"
                              " record a new solution. No changes applied.")
        elif z0 is not None:
            warnings.warn("Initial algebraic states cannot be set, since no algebraic states are defined for the model."
                          )

        if not self._y.is_empty():
            if self._dydp_nnz > 0 and self._solution.get_by_id('p').is_empty():
                raise RuntimeError("Please set the values for the parameters by executing the "
                                   "'set_initial_parameter_values' method before setting the initial conditions.")

            to_skip = ['t', 'u']
            if self._dydp_nnz == 0:
                to_skip.append('p')
            args = self._solution.get_function_args(skip=to_skip)
            if self._dydp_nnz == 0 and self._n_p > 0:
                args['p'] = ca.DM.zeros(self._n_p)

            # NOTE: dt is subtracted for time-variant systems due to the way the function for the measurement
            #  equation is generated
            # TODO: What about time grids?
            dt = self._solution.dt
            if dt is not None:
                if 'p' in args:
                    if not self._u.is_empty():
                        args['p'] = ca.vertcat(ca.DM.zeros(self._n_u), args['p'])
                    if self._is_time_variant:
                        args['p'] = ca.vertcat(args['p'], t0 - dt)
                else:
                    if not self._u.is_empty():
                        args['p'] = ca.DM.zeros(self._n_u)
                        if self._is_time_variant:
                            args['p'] = ca.vertcat(args['p'], t0 - dt)
                    else:
                        if self._is_time_variant:
                            args['p'] = t0 - dt
            else:
                grid = self._solution.grid
                raise NotImplementedError("Grids are not yet supported for calculating the initial measurements.")

            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter('always')
                self._solution.set('y', self._meas_function(**args)['yf'])
                # TODO: Make the following line work (see Series.set)
                # self._solution.set('y0_noise', 0.)
                self._solution.update(y_noise=ca.DM.zeros(self._n_y))
            match = [k for k in w if k.category == UserWarning and str(k.message) ==
                     "Vector is not empty. No changes applied."]
            if match:
                warnings.warn("The model has already been simulated. Please reset the solution of the model in order to"
                              " record a new solution. No changes applied.")

    def set_initial_parameter_values(self, p: Union[Numeric, ArrayLike]) -> None:
        """

        :param p:
        :return:
        """
        # TODO: See set_equilibrium_point
        if not self._p.is_empty():
            self._solution.add('p', p)
        else:
            warnings.warn(f"The model {self.name} doesn't have any parameters")

    def set_equilibrium_point(
            self,
            x_eq: Optional[Union[Numeric, ArrayLike]] = None,
            z_eq: Optional[Union[Numeric, ArrayLike]] = None,
            u_eq: Optional[Union[Numeric, ArrayLike]] = None,
            t0: Union[int, float] = 0.,
            tol: float = 1e-8
    ) -> None:
        """

        :param x_eq:
        :param z_eq:
        :param u_eq:
        :param t0:
        :param tol:
        :return:
        """
        if not self._is_linearized:
            print("Model is not linearized. Supplying an equilibrium point is not necessary. No changes applied.")
            return

        if x_eq is None and self._n_x > 0:
            raise ValueError("Dynamical state information is missing from the equilibrium point.")
        if z_eq is None and self._n_z > 0:
            raise ValueError("Algebraic state information is missing from the equilibrium point.")
        if u_eq is None and self._n_u > 0:
            raise ValueError("Input information is missing from the equilibrium point.")

        x_eq = convert(x_eq, ca.DM)
        z_eq = convert(z_eq, ca.DM)
        u_eq = convert(u_eq, ca.DM)

        if x_eq.size1() != self._n_x:
            raise ValueError(f"Dimension mismatch for the dynamical state information of the equilibrium point. "
                             f"Got {x_eq.size1()}, expected {self._n_x}.")
        if z_eq.size1() != self._n_z:
            raise ValueError(f"Dimension mismatch for the algebraic state information of the equilibrium point. "
                             f"Got {z_eq.size1()}, expected {self._n_z}.")
        if u_eq.size1() != self._n_u:
            raise ValueError(f"Dimension mismatch for the input information of the equilibrium point. "
                             f"Got {u_eq.size1()}, expected {self._n_u}.")

        if (self._dxdp_nnz > 0 or self._dzdp_nnz > 0) and self._solution.get_by_id('p').is_empty():
            # NOTE: The symbolic linearization should also contain the parameters of the original nonlinear model, i.e.,
            #  if the symbolic linearized model depends on parameters, so does the original nonlinear model, which we
            #  will use to check the equilibrium point.
            # TODO: Or maybe have a link to original Model object and execute that instead of storing a function in
            #  self._steady_state?
            raise RuntimeError("Please set the values for the parameters by executing the "
                               "'set_initial_parameter_values' method before setting the equilibrium point.")

        dt = self._solution.dt
        if self._n_p > 0:
            p = self._solution.get_by_id('p:f')
        else:
            p = []

        if self._f0 is not None:
            f0 = self._f0(dt, t0, x_eq, z_eq, u_eq, p)
            if not ca.logic_all(ca.le(f0.fabs(), tol)):
                raise ValueError(f"Supplied values are not an equilibrium point. Maximum error: "
                                 f"{float(ca.mmax(f0.fabs())):.5f}")
        if self._g0 is not None:
            g0 = self._g0(dt, t0, x_eq, z_eq, u_eq, p)
            if not ca.logic_all(ca.le(g0.fabs(), tol)):
                raise ValueError(f"Supplied values are not an equilibrium point. Maximum error: "
                                 f"{float(ca.mmax(g0.fabs())):.5f}")
        self._steady_state.set('t', t0)
        if not self._x_eq.is_empty():
            self._steady_state.add('x', x_eq)
        if not self._z_eq.is_empty():
            self._steady_state.add('z', z_eq)
        if not self._u_eq.is_empty():
            self._steady_state.add('u', u_eq)

    @property
    def solution(self):
        """

        :return:
        """
        return self._solution

    def copy(self, name=None, setup=False, **kwargs):
        """

        :param name:
        :param setup:
        :param kwargs:
        :return:
        """
        dt = kwargs.pop('dt', None)
        if dt is None:
            grid = kwargs.pop('grid', None)
            if grid is None:
                if self.is_setup():
                    dt = self._solution.dt
                    if dt is None:
                        grid = self._solution.grid

        _model = super().copy(name=name, setup=False, **kwargs)
        model = Model(plot_backend=self._solution.plot_backend)
        model.__setstate__(_model.__getstate__())

        if setup:
            if not self._rhs.is_empty():
                if dt is not None:
                    model.setup(dt=dt)
                elif grid is not None:
                    model.setup(grid=grid)
                else:
                    model.setup()
            else:
                warnings.warn("Model to be copied has no right-hand side, so the new model cannot be set up.")
        else:
            if not self._rhs.is_empty():
                if self.is_linear():
                    self._rhs.generate_matrix_notation(ca.vertcat(self._x.values, self._z.values), self._u.values)

        return model

    def discretize(
            self,
            method: str,
            order: Optional[int] = None,
            butcher_tableau: Optional[str] = None,
            degree: Optional[int] = None,
            polynomial_type: Optional[str] = None,
            inplace: bool = False
    ) -> Optional['Model']:
        """
        Discretize a continuous-time model. Implemented discretization methods are explicit Runge-Kutta and collocation
        schemes.

        :note: The :meth:`~hilo_mpc.core.model.Model.discretize` method doesn't require a numerical sampling time
            'dt', since the discretization is executed symbolically.

        :note: If the model is already discrete, an according message will be displayed and the discretization attempt
            will be aborted.

        :param method: The method used for discretization. At the moment only explicit Runge-Kutta methods and
            collocation schemes are supported. Accepted values are 'rk4' for the most widely known 4th order
            Runge-Kutta method, 'erk' for general order explicit Runge-Kutta methods and 'collocation' for the implicit
            collocation methods.
        :type method: str
        :param order: Order of the explicit Runge-Kutta method 'erk'. Supported orders are 1 to 4. Will be ignored if
            'rk4' or 'collocation' is selected. Defaults to 1 for 'erk'.
        :type order: int, optional
        :param butcher_tableau: The Butcher tableau used in the discretization with the explicit Runge-Kutta method
            'erk'. Will be ignored if 'rk4' or 'collocation' is selected. The default value of the Butcher tableau
            depends on the order of the explicit Runge-Kutta method. For more information on the Butcher tableau refer
            to the :ref:`Modules <modelling_module>` section.
        :type butcher_tableau: str, optional
        :param degree: Degree of the interpolating polynomial used in the collocation method. Will be ignored if 'erk'
            or 'rk4' is selected. Defaults to 2 for 'collocation'.
        :type degree: int, optional
        :param polynomial_type: Polynomial used for calculating the collocation points, i.e. the roots of the
            polynomial. At the moment only Gauss-Legendre polynomials and Gauss-Radau polynomials are supported. Set
            **polynomial_type** to 'legendre' for Gauss-Legendre polynomials and to 'radau' for Gauss-Radau polynomials.
            Will be ignored if 'erk' or 'rk4' is selected. Defaults to 'radau' for 'collocation'.
        :type polynomial_type: str, optional
        :param inplace: If True, do discretization on the current :class:`~hilo_mpc.core.model.Model` instance and
            return None, otherwise a new discretized :class:`~hilo_mpc.core.model.Model` instance will be returned.
            Defaults to False.
        :type inplace: bool
        :return: If argument **inplace** was set to True a new :class:`~hilo_mpc.core.model.Model` instance will be
            returned, otherwise None
        """
        kwargs = {}
        if method == 'collocation':
            if degree is None:
                degree = 2
            kwargs['degree'] = degree
            kwargs['collocation_points'] = polynomial_type
        elif method == 'erk':
            if order is None:
                order = 1
            kwargs['order'] = order
            kwargs['butcher_tableau'] = butcher_tableau

        _model = super().discretize(method, inplace=inplace, **kwargs)
        if _model is not None:  # Basically means 'inplace' parameter was set to False
            model = Model(plot_backend=self._solution.plot_backend)
            model.__setstate__(_model.__getstate__())
            if method == 'collocation':
                model._collocation_points.degree = degree

            return model
        if method == 'collocation':
            self._collocation_points.degree = degree

    def linearize(self, name: str = None, trajectory: Optional[dict] = None) -> 'Model':
        """

        :param name:
        :param trajectory:
        :return:
        """
        _model = super().linearize(name=name, trajectory=trajectory)
        model = Model(plot_backend=self._solution.plot_backend)
        model.__setstate__(_model.__getstate__())

        if not model.is_linear():
            raise RuntimeError("Something went wrong. Linearized model is not linear.")

        return model

    def reset_solution(self, keep_initial_conditions: bool = True, keep_equilibrium_point: bool = True) -> None:
        """

        :param keep_initial_conditions:
        :param keep_equilibrium_point:
        :return:
        """
        if not self._solution.is_empty():
            if keep_initial_conditions:
                index = slice(1, None)
            else:
                index = slice(0, None)

            # NOTE: To completely remove references and bounds (maybe there's a better way)
            self._solution.remove('x', slice(0, None), skip=['data', 'noise'])

            self._solution.remove('t', index)
            if self._n_x > 0:
                self._solution.remove('x', index)
            if self._n_y > 0:
                self._solution.remove('y', index)
            if self._n_z > 0:
                self._solution.remove('z', index)
            if self._n_u > 0:
                self._solution.remove('u', slice(0, None))
            if self._n_p > 0:
                self._solution.remove('p', index)

        if not self._steady_state.is_empty():
            index = None
            if keep_equilibrium_point:
                index = slice(1, None)
            self._steady_state.remove('t', index)
            if self._n_x > 0:
                self._steady_state.remove('x', index)
            if self._n_z > 0:
                self._steady_state.remove('z', index)
            if self._n_u > 0:
                # NOTE: We don't want to clear the complete array here, since this is supposed to be a parameter.
                self._steady_state.remove('u', index)

    def setup(self, **kwargs):
        """

        :param kwargs:
        :return:
        """
        # TODO: Add support for time grid
        # TODO: Check time-dependent odes
        self._solution.clear()

        dt = kwargs.get('dt')
        if dt is None:
            grid = kwargs.get('grid')
            if grid is None:
                warnings.warn(f"Sampling time 'dt' was not supplied. Assuming a sampling time of "
                              f"'dt=1{self._t.units[0]}'")
                dt = 1.
                kwargs['dt'] = dt
        else:
            grid = None

        super().setup(**kwargs)

        if self._n_x > 0 and self._n_p > 0:
            self._dxdp_nnz = ca.jacobian(self._rhs.ode, self._p.values).nnz()
        if self._n_y > 0 and self._n_p > 0:
            self._dydp_nnz = ca.jacobian(self._rhs.meas, self._p.values).nnz()
        if self._n_z > 0 and self._n_p > 0:
            self._dzdp_nnz = ca.jacobian(self._rhs.alg, self._p.values).nnz()

        if dt is not None:
            names = ['dt']
            vector = {
                'dt': dt
            }
        else:
            names = ['grid']
            vector = {
                'grid': grid
            }
        names += ['t']
        vector['t'] = {
            'values_or_names': self._t.names,
            'description': self._t.description,
            'labels': self._t.labels,
            'units': self._t.units,
            'shape': (1, 0),
            'data_format': ca.DM
        }
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
        if self._y.names:
            names += ['y']
            vector['y'] = {
                'values_or_names': self._y.names,
                'description': self._y.description,
                'labels': self._y.labels,
                'units': self._y.units,
                'shape': (self._n_y, 0),
                'data_format': ca.DM
            }
        if self._z.names:
            names += ['z']
            vector['z'] = {
                'values_or_names': self._z.names,
                'description': self._z.description,
                'labels': self._z.labels,
                'units': self._z.units,
                'shape': (self._n_z, 0),
                'data_format': ca.DM
            }
        if self._u.names:
            names += ['u']
            vector['u'] = {
                'values_or_names': self._u.names,
                'description': self._u.description,
                'labels': self._u.labels,
                'units': self._u.units,
                'shape': (self._n_u, 0),
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
        if self._n_q > 0:
            names += ['q']
            vector['q'] = {
                'values_or_names': 'q',
                'shape': (self._n_q, 0),
                'data_format': ca.DM
            }
        self._solution.setup(*names, **vector)

        if self._is_linearized:
            if dt is not None:
                names = ['dt']
                vector = {
                    'dt': dt
                }
            else:
                names = ['grid']
                vector = {
                    'grid': grid
                }
            names += ['t']
            vector['t'] = {
                'values_or_names': self._t.names,
                'description': self._t.description,
                'labels': self._t.labels,
                'units': self._t.units,
                'shape': (1, 0),
                'data_format': ca.DM
            }
            if not self._x_eq.is_empty():
                names += ['x']
                vector['x'] = {
                    'values_or_names': self._x_eq.names,
                    'description': self._x_eq.description,
                    'labels': self._x_eq.labels,
                    'units': self._x_eq.units,
                    'shape': (self._n_x, 0),  # x_eq and x should have the same dimensions
                    'data_format': ca.DM
                }
            if not self._z_eq.is_empty():
                names += ['z']
                vector['z'] = {
                    'values_or_names': self._z_eq.names,
                    'description': self._z_eq.description,
                    'labels': self._z_eq.labels,
                    'units': self._z_eq.units,
                    'shape': (self._n_z, 0),  # z_eq and z should have the same dimensions
                    'data_format': ca.DM
                }
            if not self._u_eq.is_empty():
                names += ['u']
                vector['u'] = {
                    'values_or_names': self._u_eq.names,
                    'description': self._u_eq.description,
                    'labels': self._u_eq.labels,
                    'units': self._u_eq.units,
                    'shape': (self._n_u, 0),  # u_eq and u should have the same dimensions
                    'data_format': ca.DM
                }
            self._steady_state.setup(*names, **vector)

        names = []
        vector = {}
        if self._x_col.names:
            names += ['x']
            vector['x'] = {
                'values_or_names': self._x_col.names,
                'description': self._x_col.description,
                'labels': self._x_col.labels,
                'units': self._x_col.units,
                'shape': (self._x_col.size1(), 0),
                'data_format': ca.DM
            }
        if self._z_col.names:
            names += ['z']
            vector['z'] = {
                'values_or_names': self._z_col.names,
                'description': self._z_col.description,
                'labels': self._z_col.labels,
                'units': self._z_col.units,
                'shape': (self._z_col.size1(), 0),
                'data_format': ca.DM
            }
        self._collocation_points.setup(*names, **vector)

    def simulate(self, *args, **kwargs):
        """

        :param args:
        :param kwargs:
        :return:
        """
        # TODO: Throw error when SX and MX are mixed
        # TODO: What if time grid is chosen?
        # TODO: Deal with time span as an argument
        # TODO: Deal with time-dependent DAEs
        # TODO: Deal with sporadic measurement (not equidistant).
        # NOTE: Sporadic measurement time points should be multiples of dt.
        if self._function is None:
            # NOTE: Not sure if we want to throw an error here
            raise RuntimeError("Model is not set up. Run Model.setup() before running simulations.")

        if self._solution.get_by_id('x').is_empty():
            raise RuntimeError("No initial dynamical states found. Please set initial conditions before simulating the "
                               "model!")

        if self._n_z > 0 and self._solution.get_by_id('z').is_empty():
            raise RuntimeError("No initial algebraic states found. Please set initial conditions before simulating the "
                               "model!")

        dt = self._solution.dt
        tf = kwargs.get('tf')
        if tf is None:
            steps = kwargs.get('steps', 1)
        else:
            steps = int(tf / dt)
        u = kwargs.get('u')
        p = kwargs.get('p')
        if u is not None:
            u = convert(u, ca.DM, axis=1)
            if u.size1() != self._n_u:
                raise ValueError(f"Dimension mismatch. Supplied dimension for the inputs 'u' is "
                                 f"{u.size1()}x{u.size2()}, but required dimension is {self._n_u}x{steps}.")
            if steps > 1:
                if u.size2() == 1:
                    u = ca.repmat(u, 1, steps)
                elif u.size2() != steps:
                    raise ValueError(f"Dimension mismatch. Supplied dimension for the inputs 'u' is "
                                     f"{u.size1()}x{u.size2()}, but required dimension is {self._n_u}x{steps}.")
            else:
                if u.size2() != steps:
                    steps = u.size2()
            self._solution.add('u', u)
        else:
            # NOTE: Instead of checking whether 'u' is contained in self._solution, we could also check if
            #  self._n_u > 0, since we only get here if the model is set up
            if 'u' in self._solution:
                raise RuntimeError("No input 'u' to the system was supplied")
        if p is not None:
            p = convert(p, ca.DM, axis=1)
            if p.size1() != self._n_p:
                raise ValueError(f"Dimension mismatch. Supplied dimension for the parameters 'p' is "
                                 f"{p.size1()}x{p.size2()}, but required dimension is {self._n_p}x{steps}.")
            if steps > 1:
                if p.size2() == 1:
                    p = ca.repmat(p, 1, steps)
                elif p.size2() != steps:
                    raise ValueError(f"Dimension mismatch. Supplied dimension for the parameters 'p' is "
                                     f"{p.size1()}x{p.size2()}, but required dimension is {self._n_p}x{steps}.")
            else:
                if p.size2() != steps:
                    steps = p.size2()
            self._solution.add('p', p)
        else:
            # NOTE: Instead of checking whether 'p' is contained in self._solution, we could also check if
            #  self._n_p > 0, since we only get here if the model is set up
            if 'p' in self._solution:
                p = self._solution.get_by_id('p')
                if p.is_empty():
                    raise RuntimeError("No parameter 'p' of the system was supplied")
                if steps > 1:
                    p = ca.repmat(p[:, -1], 1, steps - p.size2())
                    self._solution.add('p', p)
                else:
                    self._solution.add('p', p[:, -1])

        if dt is not None:
            if tf is None:
                tf = steps * dt
                # NOTE: Don't know if this is necessary
                tf = ca.linspace(dt, tf, steps).T

            if self._solution['t'].is_empty():
                self._solution.add('t', 0.)
        else:
            steps = 1
            grid = self._solution.grid
            tf = grid[1:].reshape(1, -1)

            if self._solution['t'].is_empty():
                self._solution.add('t', grid[0])
            else:
                if self._solution['t:f'] != grid[0]:
                    self._solution.add('t', grid[0])

        # TODO: Add to_skip from kwargs
        args = self._solution.get_function_args(steps=steps)
        t0 = args.pop('t0')

        if self._is_time_variant:
            if steps > 1:
                t_args = ca.horzcat(t0, tf[:, :-1])
            else:
                t_args = t0
            if 'p' in args:
                args['p'] = ca.vertcat(args['p'], t_args)
            else:
                args['p'] = t_args
        tf += t0

        if self._is_linearized and not self._linearization_about_trajectory:
            x_eq_is_required = self._n_x > 0 and self._steady_state.get_by_id('x').is_empty()
            z_eq_is_required = self._n_z > 0 and self._steady_state.get_by_id('z').is_empty()
            u_eq_is_required = self._n_u > 0 and self._steady_state.get_by_id('u').is_empty()
            if x_eq_is_required or z_eq_is_required or u_eq_is_required:
                raise RuntimeError("Model is linearized, but no equilibrium point was set. Please set equilibrium point"
                                   " before simulating the model!")

            x_eq = kwargs.get('x_eq')
            z_eq = kwargs.get('z_eq')
            u_eq = kwargs.get('u_eq')
            check_eq = False
            if x_eq is not None:
                x_eq = convert(x_eq, ca.DM, axis=1)
                if x_eq.size1() != self._n_x:
                    raise ValueError(f"Dimension mismatch. Supplied dimension for the equilibrium point 'x_eq' is "
                                     f"{x_eq.size1()}x{x_eq.size2()}, but required dimension is {self._n_x}x{steps}.")
                if steps > 1:
                    if x_eq.size2() == 1:
                        x_eq = ca.repmat(x_eq, 1, steps)
                    elif x_eq.size2() != steps:
                        raise ValueError(f"Dimension mismatch. Supplied dimension for the equilibrium point 'x_eq' is "
                                         f"{x_eq.size1()}x{x_eq.size2()}, but required dimension is "
                                         f"{self._n_x}x{steps}.")
                    else:
                        check_eq = True
            else:
                # NOTE: Instead of checking whether 'x' is contained in self._steady_state, we could also check if
                #  self._n_x > 0, since we only get here if the model is set up
                if 'x' in self._steady_state:
                    x_eq = self._steady_state.get_by_id('x')
                    if x_eq.is_empty():
                        # NOTE: We should not get here
                        raise RuntimeError("No parameter 'x_eq' of the system was supplied")
                    if steps > 1:
                        x_eq = ca.repmat(x_eq[:, -1], 1, steps - x_eq.size2() + 1)
                        self._steady_state.add('x', x_eq[:, :-1])
                    else:
                        self._steady_state.add('x', x_eq[:, -1])
                else:
                    x_eq = ca.DM()
            if z_eq is not None:
                z_eq = convert(z_eq, ca.DM, axis=1)
                if z_eq.size1() != self._n_z:
                    raise ValueError(f"Dimension mismatch. Supplied dimension for the equilibrium point 'z_eq' is "
                                     f"{z_eq.size1()}x{z_eq.size2()}, but required dimension is {self._n_z}x{steps}.")
                if steps > 1:
                    if z_eq.size2() == 1:
                        z_eq = ca.repmat(z_eq, 1, steps)
                    elif z_eq.size2() != steps:
                        raise ValueError(f"Dimension mismatch. Supplied dimension for the equilibrium point 'z_eq' is "
                                         f"{z_eq.size1()}x{z_eq.size2()}, but required dimension is "
                                         f"{self._n_z}x{steps}.")
                    else:
                        check_eq = True
            else:
                # NOTE: Instead of checking whether 'z' is contained in self._steady_state, we could also check if
                #  self._n_z > 0, since we only get here if the model is set up
                if 'z' in self._steady_state:
                    z_eq = self._steady_state.get_by_id('z')
                    if z_eq.is_empty():
                        # NOTE: We should not get here
                        raise RuntimeError("No parameter 'z_eq' of the system was supplied")
                    if steps > 1:
                        z_eq = ca.repmat(z_eq[:, -1], 1, steps - z_eq.size2() + 1)
                        self._steady_state.add('z', z_eq[:, :-1])
                    else:
                        self._steady_state.add('z', z_eq[:, -1])
                else:
                    z_eq = ca.DM()
            if u_eq is not None:
                u_eq = convert(u_eq, ca.DM, axis=1)
                if u_eq.size1() != self._n_u:
                    raise ValueError(f"Dimension mismatch. Supplied dimension for the equilibrium point 'u_eq' is "
                                     f"{u_eq.size1()}x{u_eq.size2()}, but required dimension is {self._n_u}x{steps}.")
                if steps > 1:
                    if u_eq.size2() == 1:
                        u_eq = ca.repmat(u_eq, 1, steps)
                    elif u_eq.size2() != steps:
                        raise ValueError(f"Dimension mismatch. Supplied dimension for the equilibrium point 'u_eq' is "
                                         f"{u_eq.size1()}x{u_eq.size2()}, but required dimension is "
                                         f"{self._n_u}x{steps}.")
                    else:
                        check_eq = True
            else:
                # NOTE: Instead of checking whether 'u' is contained in self._steady_state, we could also check if
                #  self._n_u > 0, since we only get here if the model is set up
                if 'u' in self._steady_state:
                    u_eq = self._steady_state.get_by_id('u')
                    if u_eq.is_empty():
                        # NOTE: We should not get here
                        raise RuntimeError("No parameter 'u_eq' of the system was supplied")
                    if steps > 1:
                        u_eq = ca.repmat(u_eq[:, -1], 1, steps - u_eq.size2() + 1)
                        self._steady_state.add('u', u_eq[:, :-1])
                    else:
                        self._steady_state.add('u', u_eq[:, -1])
                else:
                    u_eq = ca.DM()
            self._steady_state.add('t', tf)

            if check_eq:
                tol = kwargs.get('tol')
                if tol is None:
                    tol = 1e-8
                if self._f0 is not None:
                    f0 = self._f0(dt, t0, x_eq, z_eq, u_eq, args['p'][self._n_u:self._n_u + self._n_p, :])
                    if not ca.logic_all(ca.le(f0.fabs(), tol)):
                        raise ValueError(f"Supplied values are not an equilibrium point. Maximum error: "
                                         f"{float(ca.mmax(f0.fabs())):.5f}")
                if self._g0 is not None:
                    g0 = self._g0(dt, t0, x_eq, z_eq, u_eq, args['p'][self._n_u:self._n_u + self._n_p, :])
                    if not ca.logic_all(ca.le(g0.fabs(), tol)):
                        raise ValueError(f"Supplied values are not an equilibrium point. Maximum error: "
                                         f"{float(ca.mmax(g0.fabs())):.5f}")
            args['p'] = ca.vertcat(args['p'], x_eq, z_eq, u_eq)

        # NOTE: More often than not the rootfinder won't work with the latest values for the algebraic states stored
        #  in the solution object as initial conditions. Here, we enable to set the initial conditions of the rootfinder
        #  arbitrarily by the user.
        z = kwargs.get('z')
        if z is not None:
            z = convert(z, ca.DM, axis=1)
            if z.size1() != self._n_z:
                raise ValueError(f"Dimension mismatch. Supplied dimension for the algebraic states 'z' is "
                                 f"{z.size1()}x{z.size2()}, but required dimension is {self._n_z}x{steps}.")
            if steps > 1:
                if z.size2() == 1:
                    z = ca.repmat(z, 1, steps)
                elif z.size2() != steps:
                    raise ValueError(f"Dimension mismatch. Supplied dimension for the algebraic states 'z' is "
                                     f"{z.size1()}x{z.size2()}, but required dimension is {self._n_z}x{steps}.")
            args['z0'] = z

        if not self._x_col.is_empty():
            x_col = kwargs.get('x_col')
            if x_col is not None:
                x_col = convert(x_col, ca.DM, axis=1)
                if x_col.size1() != self._x_col.size1():
                    raise ValueError(f"Dimension mismatch. Supplied dimension for the dynamical states at the "
                                     f"collocation points 'x_col' is {x_col.size1()}x{x_col.size2()}, but required "
                                     f"dimension is {self._x_col.size1()}x{steps}.")
                if steps > 1:
                    if x_col.size2() == 1:
                        x_col = ca.repmat(x_col, 1, steps)
                    elif x_col.size2() != steps:
                        raise ValueError(f"Dimension mismatch. Supplied dimension for the dynamical states at the "
                                         f"collocation points 'x_col' is {x_col.size1()}x{x_col.size2()}, but required "
                                         f"dimension is {self._x_col.size1()}x{steps}.")
                args['x_col'] = x_col
            else:
                args['x_col'] = ca.repmat(args['x0'], self._collocation_points.degree, steps)

        if not self._z_col.is_empty():
            z_col = kwargs.get('z_col')
            if z_col is not None:
                z_col = convert(z_col, ca.DM, axis=1)
                if z_col.size1() != self._z_col.size1():
                    raise ValueError(f"Dimension mismatch. Supplied dimension for the algebraic states at the "
                                     f"collocation points 'z_col' is {z_col.size1()}x{z_col.size2()}, but required "
                                     f"dimension is {self._z_col.size1()}x{steps}.")
                if steps > 1:
                    if z_col.size2() == 1:
                        z_col = ca.repmat(z_col, 1, steps)
                    elif z_col.size2() != steps:
                        raise ValueError(f"Dimension mismatch. Supplied dimension for the algebraic states at the "
                                         f"collocation points 'z_col' is {z_col.size1()}x{z_col.size2()}, but required "
                                         f"dimension is {self._z_col.size1()}x{steps}.")
                args['z_col'] = z_col
            else:
                args['z_col'] = ca.repmat(args['z0'], self._collocation_points.degree)

        if steps > 1:
            function = self._function.mapaccum(steps)
            result = function(**args)
        else:
            result = self._function(**args)
        result['t'] = tf
        if self._n_x > 0:
            result['x'] = result.pop('xf')
        if self._n_z > 0:
            result['z'] = result.pop('zf')
        if self._n_y > 0:
            result['y'] = result.pop('yf')
        if self._n_q > 0:
            result['q'] = result.pop('qf')
        self._solution.update(**result)

    def generate_data(
            self,
            signal_type: str,
            *args,
            output: str = 'absolute',
            shift: int = 0,
            add_noise: Optional[Union[dict[str, Numeric], dict[str, dict[str, Numeric]]]] = None,
            n_samples: Optional[int] = None,
            steps: Optional[int] = None,
            bounds: Optional[dict[str, dict[str, Numeric]]] = None,
            mean: Optional[dict[str, Numeric]] = None,
            variance: Optional[dict[str, Numeric]] = None,
            seed: Optional[int] = None,
            chirps: Optional[dict[str, dict[str, Union[str, Numeric, Sequence[str], Sequence[Numeric]]]]] = None,
            skip: Optional[Union[Sequence[str], Sequence[int]]] = None,
            **kwargs
    ) -> DataSet:
        """

        :param signal_type:
        :param args:
        :param output:
        :param shift:
        :param add_noise:
        :param n_samples:
        :param steps:
        :param bounds:
        :param mean:
        :param variance:
        :param seed:
        :param chirps:
        :param skip:
        :param kwargs:
        :return:
        """
        # TODO: Support for algebraic states and parameters
        data_generator = DataGenerator(self, **kwargs)
        signal_type = signal_type.replace(' ', '_').lower()
        if signal_type in ['random_uniform', 'random_normal']:
            if n_samples is None:
                raise ValueError(f"Generating data using the {signal_type.replace('_', ' ')} signal requires "
                                 f"information about the number of samples by supplying the keyword 'n_samples'")
            if steps is None:
                raise ValueError(f"Generating data using the {signal_type.replace('_', ' ')} signal requires "
                                 f"information about the number of steps by supplying the keyword 'steps'")

            lbs = []
            ubs = []
            for k in self._u.names:
                name = bounds.get(k)
                lb = None
                ub = None
                if name is not None:
                    lb = name.get('lb')
                    if lb is None:
                        lb = name.get('lower_bound')
                    ub = name.get('ub')
                    if ub is None:
                        ub = name.get('upper_bound')
                if lb is None:
                    lb = 0.
                if ub is None:
                    ub = 1.
                lbs.append(lb)
                ubs.append(ub)

            if signal_type == 'random_uniform':
                data_generator.random_uniform(n_samples, steps, lbs, ubs, seed=seed)
            else:  # signal_type == 'random_normal'
                mus = []
                sigmas = []
                for k in self._u.names:
                    mu = None
                    if mean is not None:
                        mu = mean.get(k)
                    mus.append(mu)
                    sigma = None
                    if variance is not None:
                        sigma = variance.get(k)
                    sigmas.append(sigma)

                for k in range(self._n_u):
                    if mus[k] is None:
                        mus[k] = lbs[k] + (ubs[k] - lbs[k]) / 2
                    if sigmas[k] is None:
                        sigmas[k] = (1. / 3. * (ubs[k] - lbs[k]) / 2) ** 2  # 99.7% lie between lb and ub

                data_generator.random_normal(n_samples, steps, mus, sigmas, seed=seed)
        elif signal_type == 'chirp':
            types = []
            amplitudes = []
            lengths = []
            means = []
            chirp_rates = []
            initial_phases = []
            initial_frequencies = []
            initial_frequency_ratios = []
            for k in self._u.names:
                chirp = chirps.get(k)
                if chirp is not None:
                    type_ = chirp.get('type')
                    if type_ is None:
                        type_ = 'linear'
                    types.append(type_)

                    amplitude = chirp.get('amplitude')
                    if amplitude is None:
                        amplitude = 1.
                    amplitudes.append(amplitude)

                    length = chirp.get('length')
                    if length is None:
                        raise ValueError(f"No length(s) supplied for input '{k}'")
                    lengths.append(length)

                    mean = chirp.get('mean')
                    if mean is None:
                        mean = 0.
                    means.append(mean)

                    chirp_rate = chirp.get('chirp_rate')
                    if chirp_rate is None:
                        raise ValueError(f"No chirp rate(s) supplied for input '{k}'")
                    chirp_rates.append(chirp_rate)

                    initial_phases.append(chirp.get('initial_phase'))
                    initial_frequencies.append(chirp.get('initial_frequency'))
                    initial_frequency_ratios.append(chirp.get('initial_frequency_ratio'))
                else:
                    raise ValueError(f"No chirp signals supplied for input'{k}'")

            data_generator.chirp(types, amplitudes, lengths, means, chirp_rates, initial_phase=initial_phases,
                                 initial_frequency=initial_frequencies,
                                 initial_frequency_ratio=initial_frequency_ratios)
        elif signal_type == 'closed_loop':
            if not args:
                raise ValueError(f"No controller supplied for {signal_type.replace('_', ' ')} setup")
            if steps is None:
                raise ValueError(f"Generating data using the {signal_type.replace('_', ' ')} setup requires "
                                 f"information about the number of steps by supplying the keyword 'steps'")

            data_generator.closed_loop(*args, steps)

        if add_noise is not None and 'seed' not in add_noise and seed is not None:
            add_noise = add_noise.copy()
            add_noise['seed'] = seed

        if skip is not None:
            if all(isinstance(k, str) for k in skip):
                skip = self._x.index(skip)
            elif any(isinstance(k, str) for k in skip):
                raise ValueError("Mixed iterables for the keyword argument 'skip' are not allowed. Either use only "
                                 "strings or only indices (integers) in your iterable.")
        data_generator.run(output, skip=skip, shift=shift, add_noise=add_noise)

        return data_generator.data
