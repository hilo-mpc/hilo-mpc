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
_TENSORFLOW_VERSION = ('2.3.0', '2.8.0')
_SCIKIT_LEARN_VERSION = '0.19.2'
_TENSORBOARD_VERSION = '2.3.0'
_BOKEH_VERSION = '2.3.0'
_MATPLOTLIB_VERSION = '3.0.0'
_PANDAS_VERSION = '1.0.0'


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


class LearningVisualizationManager(Manager):
    """"""
    def setup(self, *args, **kwargs):
        """

        :param args:
        :param kwargs:
        :return:
        """
        if isinstance(self._backend, str):
            backend = _get_learning_visualization_backend(self._backend)
        else:
            backend = self._backend
        return backend.setup(*args, **kwargs)


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
        if StrictVersion(tensorflow.__version__) < StrictVersion(_TENSORFLOW_VERSION[0]) or \
                StrictVersion(tensorflow.__version__) >= StrictVersion(_TENSORFLOW_VERSION[1]):
            raise RuntimeError(f"Backend 'TensorFlow' is installed with version '{tensorflow.__version__}', but version"
                               f" needs to be higher or equal to '{_TENSORFLOW_VERSION[0]}' and lower than "
                               f"'{_TENSORFLOW_VERSION[1]}'")
        import hilo_mpc.plugins.tensorflow as module
    elif backend in ['sklearn', 'scikit-learn']:
        try:
            import sklearn
        except ModuleNotFoundError:
            raise ModuleNotFoundError("Backend 'scikit-learn' is not installed")
        if StrictVersion(sklearn.__version__) < StrictVersion(_SCIKIT_LEARN_VERSION):
            raise RuntimeError(f"Backend 'scikit-learn' is installed with version '{sklearn.__version__}', but version"
                               f" '{_SCIKIT_LEARN_VERSION}' or higher is required")
        import hilo_mpc.plugins.sklearn as module
    else:
        raise ValueError(f"Backend '{backend}' not recognized")
    return module


def _get_learning_visualization_backend(backend):
    """

    :param backend:
    :return:
    """
    if backend == 'tensorboard':
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        try:
            import tensorboard
        except ModuleNotFoundError:
            raise ModuleNotFoundError("Backend 'Tensorboard' is not installed")
        import hilo_mpc.plugins.tensorboard as module
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
            import matplotlib
        except ModuleNotFoundError:
            raise ModuleNotFoundError("Backend 'Matplotlib' is not installed")
        if StrictVersion(matplotlib._get_version()) < StrictVersion(_MATPLOTLIB_VERSION):
            raise RuntimeError(f"Backend 'Matplotlib' is installed with version '{matplotlib._get_version()}', but "
                               f"version '{_MATPLOTLIB_VERSION}' or higher is required")
        import hilo_mpc.plugins.matplotlib as module
    elif backend == 'bokeh':
        try:
            import bokeh
        except ModuleNotFoundError:
            raise ModuleNotFoundError("Backend 'Bokeh' is not installed")
        if StrictVersion(bokeh.__version__) < StrictVersion(_BOKEH_VERSION):
            raise RuntimeError(f"Backend 'Bokeh' is installed with version '{bokeh.__version__}', but version "
                               f"'{_BOKEH_VERSION}' or higher is required")
        import hilo_mpc.plugins.bokeh as module
    else:
        raise ValueError(f"Backend '{backend}' not recognized")

    return module


def check_version(library: str) -> None:
    """

    :param library:
    :return:
    """
    if library == 'pandas':
        try:
            import pandas
        except ModuleNotFoundError:
            raise ModuleNotFoundError("Plugin 'pandas' is not installed")
        if StrictVersion(pandas.__version__) < StrictVersion(_PANDAS_VERSION):
            raise RuntimeError(f"Plugin 'pandas' is installed with version '{pandas.__version__}', but version "
                               f"{_PANDAS_VERSION}' or higher is required")
