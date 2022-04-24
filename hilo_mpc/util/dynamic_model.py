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

# TODO: Typing hints

import casadi as ca


class GenericCost:
    """Class for generic cost functions"""
    def __init__(self, model, use_sx=True):
        """Constructor method"""
        self._cost = 0
        self._is_set = False
        self._model = model
        self._use_sx = use_sx

        # Number of time-varying references. Use for traj. tracking and path following
        self._n_tv_ref = 0

    def __repr__(self):
        """Representation method"""
        pass

    @property
    def cost(self):
        """

        :return:
        """
        return self._cost

    @cost.setter
    def cost(self, arg):
        # TODO check if the arg has the same SX variables that appear in the model
        if not isinstance(arg, ca.SX) and not isinstance(arg, ca.MX):
            raise TypeError('The cost function must be a casadi SX/MX expression.')

        if not arg.shape[0] == arg.shape[0] == 1:
            raise TypeError('The cost must be a one-dimensional function.')

        # Substitute the measurement equation if the user imputed it
        for i in range(self._model.n_y):
            arg = ca.substitute(arg, self._model._y[i], self._model.meas[i])

        self._is_set = True
        self._cost += arg

    @property
    def n_tv_ref(self):
        """
        Number of time-varying references

        :return:
        """
        return self._n_tv_ref
