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
import platform
from typing import Optional, Sequence, TypeVar, Union
import warnings

import casadi as ca
import numpy as np

from ..base import Base, Vector, Equations, RightHandSide
from ..machine_learning.base import LearningBase
from ...util.dynamic_model import GenericCost, QuadraticCost, continuous2discrete
from ...util.parsing import parse_dynamic_equations
from ...util.util import check_if_list_of_string, convert, dump_clean, generate_c_code, is_iterable, is_list_like, \
    is_square, who_am_i, JIT


Symbolic = TypeVar('Symbolic', ca.SX, ca.MX)
Mod = TypeVar('Mod', bound='_Model')
Numeric = Union[int, float]
NumArray = Union[Sequence[Numeric], np.ndarray]


class _Model(Base, metaclass=ABCMeta):
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

    @abstractmethod
    def simulate(self, *args, **kwargs):
        """

        :param args:
        :param kwargs:
        :return:
        """
        pass

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
