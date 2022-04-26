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

from typing import Optional
import warnings

import casadi as ca

from ..base import Base, Equations
from ...plugins.plugins import LearningManager


class LearningBase(Base):
    """
    Base class for all machine learning classes

    :param features:
    :param labels:
    :param id:
    :param name:
    """
    def __init__(
            self,
            features: list[str],
            labels: list[str],
            id: Optional[str] = None,
            name: Optional[str] = None
    ) -> None:
        """Constructor method"""
        super().__init__(id=id, name=name)

        self._features = features
        self._n_features = len(features)
        self._labels = labels
        self._n_labels = len(labels)
        self._backend = None

    __array_ufunc__ = None  # https://stackoverflow.com/questions/38229953/array-and-rmul-operator-in-python-numpy

    def __matmul__(self, other):
        """Matrix multiplication method"""
        features = [ca.SX.sym(k) for k in self._features]
        out = self._function(features=ca.vertcat(*features))
        labels = out['labels']
        # NOTE: @-operator doesn't seem to work because we set __array_ufunc__ to None
        equations = Equations(ca.SX, {'matmul': ca.mtimes(labels, other)})
        return equations.to_function(f'{type(self).__name__}_matmul', *features)

    def __rmatmul__(self, other):
        """Matrix multiplication method (from the right side)"""
        features = [ca.SX.sym(k) for k in self._features]
        out = self._function(features=ca.vertcat(*features))
        labels = out['labels']
        # NOTE: @-operator doesn't seem to work because we set __array_ufunc__ to None
        equations = Equations(ca.SX, {'matmul': ca.mtimes(other, labels)})
        return equations.to_function(f'{type(self).__name__}_matmul', *features)

    def _set_backend(self, train_backend):
        """

        :param train_backend:
        :return:
        """
        if isinstance(train_backend, str):
            backend = train_backend.lower()
            available_backends = ['pytorch', 'tensorflow']
            if backend in available_backends:
                self._backend = LearningManager(backend)
            else:
                # TODO: Throw an error here?
                warnings.warn(f"Supplied backend '{train_backend}' for training neural networks is not supported. "
                              f"Either use your own backend or choose one of the following: "
                              f"{','.join(available_backends)}")
        else:
            self._backend = train_backend

    @property
    def features(self) -> list[str]:
        """

        :return:
        """
        return self._features

    @property
    def n_features(self) -> int:
        """

        :return:
        """
        return self._n_features

    @property
    def labels(self) -> list[str]:
        """

        :return:
        """
        return self._labels

    @property
    def n_labels(self) -> int:
        """

        :return:
        """
        return self._n_labels

    def update(self, *args):
        """

        :param args:
        :return:
        """
        pass
