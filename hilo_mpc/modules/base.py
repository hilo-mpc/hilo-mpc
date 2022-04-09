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

import platform
from typing import Optional
import warnings

import casadi as ca
import numpy as np

from .object import Object
from ..util.util import setup_warning, check_compiler, convert, dump_clean, lower_case, AOT, JIT

if platform.system() == 'Linux':
    from hilo_mpc.util.unix import compile_so
elif platform.system() == 'Windows':
    from hilo_mpc.util.windows import compile_dll


class Base(Object):
    """
    Base class for all building blocks

    :param id: Identifier of the object
    :type id: str, optional
    :param name: Name of the object
    :type name: str, optional
    :param display: Whether to display certain messages, defaults to True
    :type display: bool
    """
    def __init__(self, id: Optional[str] = None, name: Optional[str] = None, display: bool = True) -> None:
        """Constructor method"""
        super().__init__(id=id, name=name)
        self._function = None
        self._display = display

        self._c_name = None
        self._compiler_opts = {}
        method, compiler, extra = check_compiler('jit', 'clang')
        if method is not None and compiler is not None:
            self._compiler_opts['method'] = method
            self._compiler_opts['compiler'] = compiler
            if extra is not None:
                self._compiler_opts['extra'] = extra

    def __call__(self, *args, **kwargs):
        """Calling method"""
        # TODO: Typing hints
        which = kwargs.pop('which', None)
        if which is None:
            which = '_function'
        if not which.startswith('_'):
            which = '_' + which
        func = getattr(self, which)
        return func(*args, **kwargs)

    def __repr__(self) -> str:
        """Representation method"""
        args = ""
        if self._id is not None:
            args += f"id='{self._id}'"
        if self.name is not None:
            args += f", name='{self.name}'"
        args += f", display={self._display}"
        return f"{self.__class__.__name__}({args})"

    @setup_warning
    def _set(self, attr, args, **kwargs):
        """

        :param attr:
        :param args:
        :param kwargs:
        :return:
        """
        # TODO: Typing hints
        if attr not in ['ode', 'alg', 'meas', 'obj', 'cons']:
            pointer = getattr(self, '_' + attr)
        else:
            if hasattr(self, '_rhs'):
                pointer = self._rhs
            elif hasattr(self, '_problem'):
                pointer = self._problem
            else:
                pointer = None

        if isinstance(pointer, (ca.SX, ca.MX, ca.DM)):
            setattr(self, '_' + attr, convert(args, type(pointer), name=attr))
        else:
            if type(pointer).__module__ != 'hilo_mpc.modules.base':
                raise TypeError(f"Wrong type {type(pointer).__name__} for supplied attribute")
            if isinstance(pointer, Vector):
                if args is not None:
                    pointer.set(args, **kwargs)
                else:
                    pointer.set(attr, **kwargs)
            elif isinstance(pointer, RightHandSide):
                if isinstance(args, (dict, RightHandSide)):
                    pointer.set(args, **kwargs)
                else:
                    pointer.set({attr: args}, **kwargs)
            elif isinstance(pointer, Problem):
                if isinstance(args, (dict, Problem)):
                    pointer.set(args, **kwargs)
                else:
                    pointer.set({attr: args}, **kwargs)
            else:
                raise TypeError(f"Wrong type '{type(attr).__name__}' for supplied attribute")

    @setup_warning
    def _add(self, attr, args, position, **kwargs):
        """

        :param attr:
        :param args:
        :param position:
        :param kwargs:
        :return:
        """
        # TODO: Typing hints
        if attr not in ['ode', 'alg', 'meas', 'obj', 'cons']:
            pointer = getattr(self, '_' + attr)
        else:
            if hasattr(self, '_rhs'):
                pointer = self._rhs
            elif hasattr(self, '_problem'):
                pointer = self._problem
            else:
                pointer = None

        if isinstance(pointer, (ca.SX, ca.MX, ca.DM)):
            # setattr(self, '_' + attr, convert(args, type(pointer), name=attr))
            raise NotImplementedError
        else:
            if type(pointer).__module__ != 'hilo_mpc.modules.base':
                raise TypeError(f"Wrong type {type(pointer).__name__} for supplied attribute")
            if isinstance(pointer, Vector):
                if args is not None:
                    pointer.add(args, **kwargs)
                else:
                    pointer.add(attr, **kwargs)
            elif isinstance(pointer, RightHandSide):
                if isinstance(args, (dict, RightHandSide)):
                    pointer.add(args, **kwargs)
                else:
                    pointer.add({attr: args}, **kwargs)
            elif isinstance(pointer, Problem):
                if isinstance(args, (dict, Problem)):
                    pointer.add(args, **kwargs)
                else:
                    pointer.add({attr: args}, **kwargs)
            else:
                raise TypeError(f"Wrong type '{type(attr).__name__}' for supplied attribute")

    @setup_warning
    def _remove(self, attr, args):
        """

        :param attr:
        :param args:
        :return:
        """
        # TODO: Typing hints
        if attr not in ['ode', 'alg', 'meas', 'obj', 'cons']:
            pointer = getattr(self, '_' + attr)
        else:
            if hasattr(self, '_rhs'):
                pointer = self._rhs
            elif hasattr(self, '_problem'):
                pointer = self._problem
            else:
                pointer = None

        if isinstance(args, (int, slice)):
            del pointer[args]
        elif isinstance(args, (list, tuple)) and all(isinstance(arg, int) for arg in args):
            # TODO: Do we need to loop in reversed order?
            for arg in args:
                del pointer[arg]
        else:
            pointer.remove(args)

    def _generator(self, **kwargs):
        """

        :param kwargs:
        :return:
        """
        generator = kwargs.get('generator')
        if generator is not None:
            if not isinstance(generator, dict):
                gen_path = None
                gen_name = None
                gen_opts = None
            else:
                gen_path = generator.get('path')
                gen_name = generator.get('name')
                gen_opts = generator.get('opts')
        else:
            gen_path = kwargs.get('gen_path')
            gen_name = kwargs.get('gen_name')
            gen_opts = kwargs.get('gen_opts')

        if gen_path is None:
            session = kwargs.get('session')
            if session is not None:
                gen_path = session.path

        if gen_name is None:
            if self.name is None:
                gen_name = lower_case(self.__class__.__name__)
            else:
                gen_name = self.name.replace(' ', '_')

        if gen_opts is not None:
            if platform.system() == 'Linux':
                if 'cpp' in gen_opts:
                    if gen_opts['cpp'] and self._compiler_opts['compiler'] == 'gcc':
                        warnings.warn("The option 'cpp' for the CodeGenerator can not be set to true, when the compiler"
                                      " 'gcc' is set. Please use a compiler that is able to compile C++ code, "
                                      "e.g., 'g++'. Setting option 'cpp' to False.")
                        gen_opts['cpp'] = False
                    elif not gen_opts['cpp'] and self._compiler_opts['compiler'] == 'g++':
                        warnings.warn("The option 'cpp' for the CodeGenerator can not be set to false, when the "
                                      "compiler 'g++' is set. Please use a compiler that is able to compile C code, "
                                      "e.g., 'gcc'. Setting option 'cpp' to True.")
                        gen_opts['cpp'] = True
                else:
                    if self._compiler_opts['compiler'] == 'g++':
                        warnings.warn("The option 'cpp' for the CodeGenerator is not provided, but the compiler 'g++' "
                                      "is set. Please use a compiler that is able to compile C code, e.g., 'gcc'. "
                                      "Setting option 'cpp' to True.")
                        gen_opts['cpp'] = True

        return gen_path, gen_name, gen_opts

    @property
    def compiler(self) -> str:
        """

        :return:
        """
        return dump_clean(self._compiler_opts)

    # @compiler.setter
    def set_compiler(self, method: str, compiler: str) -> None:
        """

        :param method:
        :param compiler:
        :return:
        """
        old_method = method
        old_compiler = compiler
        method, compiler, extra = check_compiler(method, compiler)
        if method != self._compiler_opts['method'] or compiler != self._compiler_opts['compiler']:
            if method is not None and compiler is not None:
                self._compiler_opts['method'] = method
                self._compiler_opts['compiler'] = compiler
                if extra is not None:
                    self._compiler_opts['extra'] = extra
            else:
                warnings.warn(f"Compiler '{old_compiler}' could not be set for method '{old_method}'. No changes "
                              f"applied.")

    @property
    def display(self) -> bool:
        """

        :return:
        """
        return self._display

    @display.setter
    def display(self, value: bool) -> None:
        if value is not self._display:
            self._display = value

    def check_consistency(self) -> None:
        """

        :return:
        """
        if self._function is not None:
            if self._function.has_free():
                free_vars = ", ".join(self._function.get_free())
                msg = f"Instance {self.__class__.__name__} has the following free variables/parameters: {free_vars}"
                raise RuntimeError(msg)

    def setup(self, **kwargs):
        """

        :param kwargs:
        :return:
        """
        # TODO: Typing hints
        # TODO: Support for user-defined output
        if self._compiler_opts['method'] in JIT:
            importer_opts = {}
            if self._compiler_opts['compiler'] == 'shell':
                path = self._compiler_opts.pop('path')
                # TODO: Cleanup should be True if a user-defined path is given
                # Cleanup set to false, to prevent warning messages that temporary files are not found. Since closing
                # the session will remove the temporary path, CasADi cannot find the temporary files anymore.
                importer_opts['cleanup'] = False
                importer_opts['name'] = path + 'tmp_casadi_compiler_shell'
                if platform.system() == 'Windows':
                    vcvars = self._compiler_opts.get('extra', None)
                    if vcvars is not None:
                        compiler = ' && '.join((vcvars, 'cl.exe'))
                        linker = ' && '.join((vcvars, 'link.exe'))
                        importer_opts['compiler'] = compiler
                        importer_opts['linker'] = linker
                    else:
                        warnings.warn("Could not find a compiler. Make sure Microsoft Visual Studio is installed "
                                      "(other compilers are not yet supported).")
                        return
            C = ca.Importer(self._c_name, self._compiler_opts['compiler'], importer_opts)
            self._function = ca.external(self._function, C)
        elif self._compiler_opts['method'] in AOT:
            if platform.system() == 'Linux':
                library = compile_so(self._c_name, self._compiler_opts['compiler'])
            elif platform.system() == 'Windows':
                library = compile_dll(self._c_name)
            else:
                library = None
            if library is not None:
                self._function = ca.external(self._function, library)
        else:
            warnings.warn("Unknown method for C/C++ code generation")

    def is_setup(self) -> bool:
        """

        :return:
        """
        if self._function is not None:
            return True
        else:
            return False


class Container(Object):
    """"""
    # TODO: Support for pandas Series and DataFrame?
    # TODO: Typing hints
    def __init__(self, data_format, values=None, shape=None, id=None, name=None, parent=None):
        """Constructor method"""
        super().__init__(id=id, name=name)
        self._parent = parent

        if data_format in [ca.SX, ca.MX, ca.DM]:
            self._fx = data_format
        elif isinstance(data_format, str):
            if data_format in ['sx', 'SX']:
                self._fx = ca.SX
            elif data_format in ['mx', 'MX']:
                self._fx = ca.MX
            elif data_format in ['dm', 'DM']:
                self._fx = ca.DM
            elif data_format in ['list', 'tuple', 'ndarray', 'Series', 'DataFrame']:
                self._fx = ca.DM
            else:
                raise ValueError(f"Format {data_format} not recognized")
        elif isinstance(data_format, (list, tuple)):
            self._fx = ca.DM
        elif isinstance(data_format, np.ndarray):
            self._fx = ca.DM
        else:
            raise ValueError(f"Type {type(data_format)} not recognized")

        self._axis = 0
        self._values = convert(values if self._fx is not ca.DM else None, self._fx, shape=shape)
        self._shape = shape
        self._update_shape(shape)

        if self._id is None:
            self._create_id()

    # def __str__(self) -> str:
    #     """String representation method"""
    #     return ""

    def __repr__(self) -> str:
        """Representation method"""
        args = f"'{self._fx.__name__}'"
        if self._values is not None:
            elements = ", ".join([f"'{element.name()}'" for element in self._values.elements()])
            args += f", values=[{elements}]"
        if self._shape is not None:
            args += f", shape=({', '.join([str(dim) for dim in self._shape])})"
        if self._id is not None:
            args += f", id='{self._id}'"
        if self.name is not None:
            args += f", name='{self.name}'"
        if self._parent is not None:
            args += f", parent={repr(self._parent)}"
        return f"{self.__class__.__name__}({args})"

    def __len__(self) -> int:
        """Length method"""
        return self.size1()

    def __getitem__(self, item):
        """Item getter method"""
        return self._values[item]

    def __setitem__(self, key, value):
        """Item setter method"""
        self._values[key] = value

    def __delitem__(self, key):
        """Item deletion method"""
        # NOTE: Right now, only working for vectors not matrices
        # NOTE: del is not able to work with lists of indices and SX_remove is not able to deal with slices
        # TODO: Is it possible to deal with matrices?
        if isinstance(key, (list, tuple)):
            to_remove = key
        elif isinstance(key, slice):
            if key.stop is not None:
                to_remove = list(range(key.stop)[key])
            else:
                dim = getattr(self, f'size{self._axis + 1}')
                to_remove = list(range(dim())[key])
        elif key == -1:
            dim = getattr(self, f'size{self._axis + 1}')
            to_remove = [dim() - 1]
        else:
            to_remove = [key]

        if self._axis == 0:
            if self._fx == ca.MX:
                # NOTE: Apparently the remove method doesn't exist for MX variables
                self._values = ca.vertcat(*[self._values[k, :] for k in range(self._shape[0]) if k not in to_remove])
            else:
                self._values.remove(to_remove, [])
        elif self._axis == 1:
            if self._fx == ca.MX:
                # TODO: Test this
                self._values = ca.horzcat(*[self._values[:, k] for k in range(self._shape[1]) if k not in to_remove])
            else:
                self._values.remove([], to_remove)
        else:
            raise ValueError(f"Axis attribute (axis={self._axis}) out of bounds")
        self._update_shape(None)
        self._update_parent()

    def __iter__(self):
        """Item iteration method"""
        yield from self._values.elements()

    def __reversed__(self):
        """Reverse item iteration method"""
        yield from self._values.elements()[::-1]

    def _update_shape(self, shape):
        """

        :param shape:
        :return:
        """
        if shape is not None:
            if self._fx is ca.DM:
                self._values.reshape(shape)
                self._shape = self._values.shape
            else:
                if shape != self._values.shape:
                    raise ValueError(f"Shape dimensions don't match: {shape} and {self._values.shape}")
                else:
                    pass
        else:
            self._shape = self._values.shape

    def _update_parent(self):
        """

        :return:
        """
        if self._parent is not None:
            self._parent._update_dimensions()

    @property
    def parent(self):
        """

        :return:
        """
        return self._parent

    @parent.setter
    def parent(self, parent):
        if self._parent != parent:
            self._parent = parent

    @property
    def values(self):
        """

        :return:
        """
        return self._values

    @property
    def shape(self):
        """

        :return:
        """
        return self._shape

    def is_constant(self, *args):
        """

        :param args:
        :return:
        """
        return self._values.is_constant(*args)

    def is_empty(self, *args):
        """

        :param args:
        :return:
        """
        return self._values.is_empty(*args)

    def is_scalar(self, *args):
        """

        :param args:
        :return:
        """
        return self._values.is_scalar(*args)

    def size(self, *args):
        """

        :param args:
        :return:
        """
        return self._values.size(*args)

    def size1(self, *args):
        """

        :param args:
        :return:
        """
        return self._values.size1(*args)

    def size2(self, *args):
        """

        :param args:
        :return:
        """
        return self._values.size2(*args)


class Vector(Container):
    """"""


class Equations:
    """"""


class RightHandSide(Equations):
    """"""


class Problem(Equations):
    """"""


class Series(Object):
    """"""


class TimeSeries(Series):
    """"""


class OptimizationSeries(Series):
    """"""
