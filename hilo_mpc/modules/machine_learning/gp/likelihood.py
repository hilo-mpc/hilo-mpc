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

class Likelihood:
    """"""
    def __init__(self, name: str) -> None:
        """Constructor method"""
        self._name = name

    @property
    def name(self) -> str:
        """

        :return:
        """
        return self._name

    @staticmethod
    def gaussian():
        """

        :return:
        """
        return Gaussian()

    @staticmethod
    def logistic():
        """

        :return:
        """
        return Logistic()

    @staticmethod
    def laplacian():
        """

        :return:
        """
        return Laplacian()

    @staticmethod
    def students_t():
        """

        :return:
        """
        return StudentsT()


class Gaussian(Likelihood):
    """"""
    def __init__(self):
        """Constructor method"""
        super().__init__('Gaussian')


class Logistic(Likelihood):
    """"""
    def __init__(self):
        """Constructor method"""
        super().__init__('Logistic')

        raise NotImplementedError("Logistic likelihood not yet implemented")


class Laplacian(Likelihood):
    """"""
    def __init__(self):
        """Constructor method"""
        super().__init__('Laplacian')

        raise NotImplementedError("Laplacian likelihood not yet implemented")


class StudentsT(Likelihood):
    """"""
    def __init__(self):
        """Constructor method"""
        super().__init__('Students_T')

        raise NotImplementedError("Student's t likelihood not yet implemented")
