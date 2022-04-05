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

from typing import Optional
import uuid


class Object:
    """
    Defines common behavior of various objects in HILO-MPC.

    :param id: Identifier of the object
    :type id: str, optional
    :param name: Name of the object
    :type name: str, optional
    """
    def __init__(self, id: Optional[str] = None, name: Optional[str] = None) -> None:
        """Constructor method"""
        self._id = id
        self.name = name

    def __repr__(self) -> str:
        """Representation method"""
        args = ""
        if self._id is not None:
            args += f"id='{self._id}'"
        if self.name is not None:
            args += f", name='{self.name}'"
        return f"{self.__class__.__name__}({args})"

    def __str__(self) -> str:
        """String representation method"""
        return f"{self.__class__.__name__}: {self.name} ({self._id})"

    def _create_id(self) -> None:
        """
        Generates a unique identifier for the object.

        :return:
        """
        self._id = str(uuid.uuid4())

    @property
    def id(self) -> str:
        """
        Identifier of the object

        :return: Identifier of the object
        :rtype: str
        """
        return self._id

    @id.setter
    def id(self, value: str) -> None:
        if value != self._id:
            if not isinstance(value, str):
                raise TypeError(f"{self.__class__.__name__} ID must be a string")
            else:
                self._id = value
