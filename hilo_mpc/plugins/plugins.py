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

_BOKEH_VERSION = ''
_MATPLOTLIB_VERSION = ''


class Manager:
    """
    Base class for managers

    :param backend:
    :type backend:
    """
    # TODO: Typing hints
    def __init__(self, backend):
        """Constructor method"""
        self._backend = backend

    @property
    def backend(self):
        """

        :return:
        """
        return self._backend

    @backend.setter
    def backend(self, arg):
        self._backend = arg


class PlotManager(Manager):
    """"""
    def plot(self, data, kind='line', **kwargs):
        """

        :param data:
        :param kind:
        :param kwargs:
        :return:
        """
        if isinstance(self._backend, str):
            plot_backend = _get_plot_backend(self._backend)
        else:
            if not hasattr(self._backend, 'plot'):
                raise AttributeError(f"Supplied backend '{self._backend}' has no attribute 'plot'")
            plot_backend = self._backend
        return plot_backend.plot(data, kind, **kwargs)


def _get_plot_backend(backend):
    """

    :param backend:
    :return:
    """
    if backend == 'matplotlib':
        try:
            import matplotlib.pyplot as plt
        except ModuleNotFoundError:
            raise ModuleNotFoundError("Backend 'matplotlib' is not installed")
        import hilo_mpc.plugins.matplotlib as module
    elif backend == 'bokeh':
        try:
            import bokeh.plotting as plt
        except ModuleNotFoundError:
            raise ModuleNotFoundError("Backend 'bokeh' is not installed")
        import hilo_mpc.plugins.bokeh as module
    else:
        raise ValueError(f"Backend '{backend}' not recognized")

    return module
