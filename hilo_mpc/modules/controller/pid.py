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

from typing import Optional, Union

import casadi as ca
import numpy as np

from .base import Controller
from ..base import Base, TimeSeries
from ...util.util import convert


Numeric = Union[int, float]


class PID(Controller, Base):
    """
    Class for PID controller

    :param n_inputs:
    :type n_inputs: int
    :param n_outputs:
    :type n_outputs: int
    :param id:
    :type id: str, optional
    :param name:
    :type name: str, optional
    :param k_p:
    :type k_p:
    :param t_i:
    :type t_i:
    :param t_d:
    :type t_d:
    :param proportional_on_process_value:
    :type proportional_on_process_value: bool
    :param derivative_on_process_value:
    :type derivative_on_process_value: bool
    :param plot_backend:
    :type plot_backend: str, optional
    """
    def __init__(
            self,
            n_inputs: int,
            n_outputs: int,
            id: Optional[str] = None,
            name: Optional[str] = None,
            k_p: Union[Numeric, np.ndarray] = 1.,
            t_i: Union[Numeric, np.ndarray] = np.inf,
            t_d: Union[Numeric, np.ndarray] = 0.,
            proportional_on_process_value: bool = False,
            derivative_on_process_value: bool = False,
            plot_backend: Optional[str] = None
    ) -> None:
        """Constructor method"""
        super().__init__(id=id, name=name)
        if self._id is None:
            self._create_id()

        self._n_inputs = n_inputs
        self._n_outputs = n_outputs
        self._set_tuning_parameter('k_p', k_p)
        self._set_tuning_parameter('t_i', t_i)
        self._set_tuning_parameter('t_d', t_d)
        self._proportional_on_process_value = proportional_on_process_value
        self._derivative_on_process_value = derivative_on_process_value
        self._p_band = False
        self._anti_windup = None
        self._set_point = ca.DM.zeros(n_inputs)

        self._solution = TimeSeries(plot_backend, parent=self)

    def _update_type(self) -> None:
        """

        :return:
        """
        self._type = 'PID'

    def _set_tuning_parameter(self, which: str, value: Union[Numeric, np.ndarray]) -> None:
        """

        :param which:
        :param value:
        :return:
        """
        # TODO: Dimensions
        if which == 'k_p':
            self._k_p = value
        elif which == 't_i':
            self._t_i = value
        elif which == 't_d':
            self._t_d = value

    def _append_process_value(self, process_value: ca.DM) -> None:
        """

        :param process_value:
        :return:
        """
        if self._solution.get_by_id('x').is_empty():
            process_value_vector = ca.DM.zeros(3, self._n_inputs)
        else:
            process_value_vector = ca.reshape(self._solution.get_by_id('x:f'), 3, self._n_inputs)
            process_value_vector[:-1, :] = process_value_vector[1:, :]
        process_value_vector[-1, :] = process_value
        self._solution.add('x', process_value_vector[:])

    def _append_set_point(self) -> None:
        """

        :return:
        """
        if self._solution.get_by_id('p').is_empty():
            set_point_vector = ca.DM.zeros(3, self._n_inputs)
        else:
            params = self._solution.get_by_id('p:f')
            set_point_vector = ca.reshape(params[:3 * self._n_inputs], 3, self._n_inputs)
            set_point_vector[:-1, :] = set_point_vector[1:, :]
        set_point_vector[-1, :] = self._set_point
        params = ca.vertcat(set_point_vector[:], self._k_p, self._t_i, self._t_d)
        self._solution.add('p', params)

    def _initialize_controller_output(self) -> None:
        """

        :return:
        """
        u = ca.DM.zeros(self._n_outputs)
        self._solution.set('u', u)

    @property
    def proportional_gain(self) -> Numeric:
        """

        :return:
        """
        return self._k_p

    @proportional_gain.setter
    def proportional_gain(self, k_p: Numeric) -> None:
        self._k_p = k_p

    k_p = proportional_gain

    @property
    def integral_time(self) -> Numeric:
        """

        :return:
        """
        return self._t_i

    @integral_time.setter
    def integral_time(self, t_i: Numeric) -> None:
        self._t_i = t_i

    t_i = integral_time

    @property
    def integral_gain(self) -> Numeric:
        """

        :return:
        """
        return self._k_p / self._t_i

    k_i = integral_gain

    @property
    def derivative_time(self) -> Numeric:
        """

        :return:
        """
        return self._t_d

    @derivative_time.setter
    def derivative_time(self, t_d: Numeric) -> None:
        self._t_d = t_d

    t_d = derivative_time

    @property
    def derivative_gain(self) -> Numeric:
        """

        :return:
        """
        return self._k_p * self._t_d

    k_d = derivative_gain

    @property
    def set_point(self) -> ca.DM:
        """

        :return:
        """
        return self._set_point

    @set_point.setter
    def set_point(self, set_point: Union[Numeric, np.ndarray]) -> None:
        set_point = convert(set_point, ca.DM, axis=1)
        if self._n_inputs > 1:
            if set_point.size2() == 1:
                set_point = ca.repmat(set_point, 1, self._n_inputs)
            elif set_point.size2() != self._n_inputs:
                raise ValueError(f"Dimension mismatch. Supplied dimension for the set point is "
                                 f"{set_point.size1()}x{set_point.size2()}, but required dimension is "
                                 f"{self._n_inputs}x1.")
        self._set_point = set_point

    s_p = set_point

    def setup(self, **kwargs) -> None:
        """

        :param kwargs:
        :return:
        """
        dt = kwargs.get('dt')
        if dt is None:
            dt = 1.

        process_value = ca.SX.sym('pv', 3, self._n_inputs)
        set_point = ca.SX.sym('sp', 3, self._n_inputs)
        error = set_point - process_value
        output = ca.SX.sym('u', self._n_outputs)
        delta_output = 0.

        # TODO: Dimensions
        k_p = ca.SX.sym('k_p')
        t_i = ca.SX.sym('t_i')
        t_d = ca.SX.sym('t_d')

        if self._p_band:
            raise NotImplementedError("The use of the proportional band has not been implemented yet.")
        else:
            if self._proportional_on_process_value:
                delta_output -= process_value[-1, :] - process_value[-2, :]
            else:
                delta_output += error[-1, :] - error[-2, :]

        delta_output += dt / t_i * error[-1, :]
        # TODO: Add anti-windup

        if self._derivative_on_process_value:
            delta_output -= t_d / dt * (process_value[-1, :] - 2 * process_value[-2, :] + process_value[-3, :])
        else:
            delta_output += t_d / dt * (error[-1, :] - 2 * error[-2, :] + error[-3, :])

        # TODO: Add anti-windup
        parameters = ca.vertcat(set_point[:], k_p, t_i, t_d)
        self._function = ca.Function('function',
                                     [process_value, ca.vertcat(output, parameters)],
                                     [error, output + k_p * delta_output],
                                     ['x0', 'p'],
                                     ['error', 'yf'])

        self.check_consistency()

        n_pv = process_value.numel()
        n_y = output.numel()
        n_sp = set_point.numel()
        n_k_p = k_p.numel()
        n_t_i = t_i.numel()
        n_t_d = t_d.numel()
        n_p = n_sp + n_k_p + n_t_i + n_t_d  # same as parameters.numel()

        names = ['dt', 'x', 'u', 'p']  # 'y'
        vector = {
            'dt': dt,
            'x': {
                'values_or_names': [pv.name() for pv in process_value.elements()],
                'description': n_pv * [''],
                'labels': n_pv * ['process value'],
                'units': n_pv * [''],
                'shape': (n_pv, 0),
                'data_format': ca.DM
            },
            'u': {
                'values_or_names': [out.name() for out in output.elements()],
                'description': n_y * [''],
                'labels': n_y * ['controller output'],
                'units': n_y * [''],
                'shape': (n_y, 0),
                'data_format': ca.DM
            },
            'p': {
                'values_or_names': [par.name() for par in parameters.elements()],
                'description': n_p * [''],
                'labels': n_sp * ['set point'] + n_k_p * ['K_P'] + n_t_i * ['T_I'] + n_t_d * ['T_D'],
                'units': n_p * [''],
                'shape': (n_p, 0),
                'data_format': ca.DM
            }
        }
        self._solution.setup(*names, **vector)

    def call(self, *args, **kwargs):
        """

        :param args:
        :param kwargs:
        :return:
        """
        if self._function is None:
            type_ = 'P'
            if self._t_i != np.inf:
                type_ += 'I'
            if self._t_d != 0.:
                type_ += 'D'
            raise RuntimeError(f"{type_} controller is not set up. Run PID.setup() before calling the {type_} "
                               f"controller")

        pv = kwargs.get('pv')
        if pv is not None:
            pv = convert(pv, ca.DM, axis=1)
        else:
            pv = ca.DM.zeros(1, self._n_inputs)
        self._append_process_value(pv)
        self._append_set_point()
        if self._solution.get_by_id('u').is_empty():
            self._initialize_controller_output()

        args = self._solution.get_function_args()
        result = self._function(**args)
        self._solution.update(u=result['yf'])

        return result['yf']


__all__ = [
    'PID'
]
