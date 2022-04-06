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

import shutil
import tempfile


class Session:
    """"""
    def __init__(self) -> None:
        """Constructor method"""
        self._temp_dir = None

    def __enter__(self) -> 'TempDir':
        """Method for entering runtime context"""
        self._temp_dir = TempDir()
        return self._temp_dir

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Method for exiting runtime context"""
        shutil.rmtree(self._temp_dir.path)
        self._temp_dir.close()
        self._temp_dir = None


class TempDir:
    """"""
    def __init__(self) -> None:
        """Constructor method"""
        self._path = tempfile.mkdtemp()

    @property
    def path(self) -> str:
        """
        Path to temporarily created directory

        :return: Path to temporarily created directory
        :rtype: str
        """
        return self._path + '/'

    def close(self) -> None:
        """

        :return:
        """
        self._path = None
