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

import os
import subprocess
from typing import Optional


UNIX_COMPILERS = ['gcc', 'g++']


def compile_so(path_to_file: str, compiler: str, output: Optional[str] = None) -> str:
    """

    :param path_to_file:
    :param compiler:
    :param output:
    :return:
    """
    if os.path.exists(path_to_file):
        command = [compiler, '-fPIC', '-shared', path_to_file, '-o']
        if output is not None:
            so_path = output
        else:
            so_path = path_to_file.rsplit('.', 1)
            so_path[-1] = 'so'
            so_path = '.'.join(so_path)
        command.append(so_path)
        if subprocess.call(command) == 0:
            return so_path
        else:
            raise RuntimeError("Could not compile library")
    else:
        raise FileNotFoundError(f"File {path_to_file} does not exist")


def find_compiler(compiler: str) -> bool:
    """

    :param compiler:
    :return:
    """
    if subprocess.call([compiler, '--version'], stdout=subprocess.DEVNULL) == 0:
        return True
    else:
        return False
