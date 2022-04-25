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


class Controller(metaclass=ABCMeta):
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
