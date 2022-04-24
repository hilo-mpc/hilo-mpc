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

from distutils.version import StrictVersion
import os


_PYTORCH_VERSION = '1.2.0'
_TENSORFLOW_VERSION = '2.3.0'
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


class LearningManager(Manager):
    """"""
    def setup(self, kind, *args, **kwargs):
        """

        :param kind:
        :param args:
        :param kwargs:
        :return:
        """
        if isinstance(self._backend, str):
            backend = _get_learning_backend(self._backend)
        else:
            backend = self._backend
        return backend.setup(kind, *args, **kwargs)


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


def _get_learning_backend(backend):
    """

    :param backend:
    :return:
    """
    if backend == 'pytorch':
        try:
            import torch
        except ModuleNotFoundError:
            raise ModuleNotFoundError("Backend 'PyTorch' is not installed")
        if StrictVersion(torch.__version__.split('+')[0]) < StrictVersion(_PYTORCH_VERSION):
            raise RuntimeError(f"Backend 'PyTorch' is installed with version '{torch.__version__}', but version "
                               f"'{_PYTORCH_VERSION}' or higher is required")
        import hilo_mpc.plugins.pytorch as module
    elif backend == 'tensorflow':
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        try:
            import tensorflow
        except ModuleNotFoundError:
            raise ModuleNotFoundError("Backend 'TensorFlow' is not installed")
        if StrictVersion(tensorflow.__version__) < StrictVersion(_TENSORFLOW_VERSION):
            raise RuntimeError(f"Backend 'TensorFlow' is installed with version '{tensorflow.__version__}', but version"
                               f" '{_TENSORFLOW_VERSION}' or higher is required")
        import hilo_mpc.plugins.tensorflow as module
    elif backend in ['sklearn', 'scikit-learn']:
        try:
            import sklearn
        except ModuleNotFoundError:
            raise ModuleNotFoundError("Backend 'Scikit-learn' is not installed")
        import hilo_mpc.plugins.sklearn as module
    else:
        raise ValueError(f"Backend '{backend}' not recognized")
    return module


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
