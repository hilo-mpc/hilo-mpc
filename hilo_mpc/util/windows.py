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

import os
from subprocess import Popen, PIPE, STDOUT
import sys
from typing import Optional


LETTERS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
WINDOWS_COMPILERS = ['cl']


def compile_dll(path_to_file: str, output: Optional[str] = None) -> Optional[str]:
    """

    :param path_to_file:
    :param output:
    :return:
    """
    if os.path.isfile(path_to_file):
        vcvars = get_vcvars()
        if vcvars is not None:
            cl = f'cl.exe /LD {path_to_file}'
            if output is not None:
                cl += f' /F {output}'
            commands = ' && '.join((vcvars, cl, 'exit 0'))
            commands = f'({commands}) || exit 1\r\n'
            out = execute_commands(commands, encoding='latin1')
            if out is not None:
                if output is not None:
                    dll_path = output
                else:
                    dll_path = path_to_file.rsplit('.', 1)
                    dll_path[-1] = 'dll'
                return '.'.join(dll_path)
            else:
                return None
        else:
            return None
    else:
        raise FileNotFoundError(f"File {path_to_file} does not exist")


def execute_commands(commands: str, encoding: Optional[str] = None, errors: str = 'strict') -> str:
    """

    :param commands:
    :param encoding:
    :param errors:
    :return:
    """
    text_mode = (encoding is None)
    with Popen('cmd.exe', stdin=PIPE, stdout=PIPE, stderr=STDOUT, universal_newlines=text_mode) as process:
        if not text_mode:
            commands = commands.encode(encoding, errors)
        out, _ = process.communicate(commands)
    # Somehow err (2. output of communicate()) can be not empty although everything worked and is also not really
    # storing errors
    # NOTE: Workaround by using 'exit 0' and 'exit 1' in 'cmd'
    if process.returncode != 0:
        if not text_mode:
            out = out.decode(encoding, errors)
        raise RuntimeError(f"Could not execute commands"
                           f"{out}"
                           f"Return value: {process.returncode}")
    return out if text_mode else out.decode(encoding, errors)


def find_files(name: str) -> (list[str], list[str]):
    """

    :param name:
    :return:
    """
    loc = []
    date = []
    drives = get_hard_drives()
    for drive in drives:
        for root, dirs, files in os.walk(drive):
            for file in files:
                if file == name:
                    path_to_file = root + '\\' + file
                    loc.append(path_to_file)
                    date.append(os.path.getctime(path_to_file))
    if len(loc) != len(date):
        raise ValueError("List 'loc' and 'list' date must be of same length")
    return loc, date


def get_hard_drives() -> list[str]:
    """

    :return:
    """
    return [f'{drive}:\\' for drive in LETTERS if os.path.exists(f'{drive}:\\')]


def get_vcvars() -> str:
    """

    :return:
    """
    files, dates = find_files('vcvarsall.bat')
    arch = '64bit' if sys.maxsize > 2 ** 32 else '32bit'
    if files:
        index_max = max(range(len(dates)), key=dates.__getitem__)
        chosen = files[index_max]
        if ' ' in chosen:
            if arch == '64bit':
                return f'"{chosen}" x64'
            else:
                return f'"{chosen}" x32'
        else:
            if arch == '64bit':
                return f'{chosen} x64'
            else:
                return f'{chosen} x32'
    else:
        raise RuntimeError("Compiler not found")
