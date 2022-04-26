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

from sklearn.preprocessing import StandardScaler


class _SklearnWrapper:
    """"""
    def __init__(self, module, **kwargs):
        """Constructor method"""
        self._module = module

    def __call__(self, *args, **kwargs):
        """Calling method"""
        return self._module(*args, **kwargs)


MODULES = {
    'StandardScaler': StandardScaler
}


def get_wrapper(kind, *args, **kwargs):
    """

    :param kind:
    :param args:
    :param kwargs:
    :return:
    """
    module = MODULES[kind]
    return _SklearnWrapper(module, **kwargs)


__all__ = [
    'get_wrapper'
]
