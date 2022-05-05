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

from abc import ABCMeta, abstractmethod
from collections.abc import KeysView
from copy import copy
import platform
from typing import Any, Optional, Sequence, Union
import warnings

import casadi as ca
import numpy as np

from .object import Object
from ..plugins.plugins import PlotManager
from ..util.io import save_mat
from ..util.plotting import get_plot_backend
from ..util.util import setup_warning, check_compiler, check_if_list_of_type, convert, dump_clean, is_list_like,\
    lower_case, who_am_i, _split_expression, AOT, JIT

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
    # TODO: Add __str__
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
    # TODO: Add __str__
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
    # TODO: Add duplicate check for names
    # TODO: Typing hints
    # TODO: Add __str__
    def __init__(
            self,
            data_format,
            values_or_names=None,
            description=None,
            labels=None,
            units=None,
            shape=None,
            id=None,
            name=None,
            parent=None
    ):
        """Constructor method"""
        super().__init__(data_format, values=values_or_names, shape=shape, id=id, name=name, parent=parent)

        if self._fx is ca.DM:
            if isinstance(values_or_names, str):
                self._names = [values_or_names]
            elif isinstance(values_or_names, (list, tuple)):
                self._names = values_or_names
            else:
                raise TypeError(f"Wrong type {type(values_or_names)} for argument 'values_or_names'")
        else:
            self._update_names()

        self._update_description(description)
        self._update_labels(labels)
        self._update_units(units)

    def __repr__(self):
        """Representation method"""
        args = f"'{self._fx.__name__}'"
        if any(self._names):
            elements = ", ".join([f"'{name}'" for name in self._names])
            args += f", values_or_names=[{elements}]"
        if any(self._description):
            elements = ", ".join([f"'{description}'" for description in self._description])
            args += f", description=[{elements}]"
        if any(self._labels):
            elements = ", ".join([f"'{label}'" for label in self._labels])
            args += f", labels=[{elements}]"
        if any(self._units):
            elements = ", ".join([f"'{unit}'" for unit in self._units])
            args += f", units=[{elements}]"
        if self._shape is not None:
            args += f", shape=({', '.join([str(dim) for dim in self._shape])})"
        if self._id is not None:
            args += f", id='{self._id}'"
        if self.name is not None:
            args += f", name='{self.name}'"
        if self._parent is not None:
            args += f", parent={repr(self._parent)}"
        return f"{self.__class__.__name__}({args})"

    def __setitem__(self, key, value):
        """Item setter method"""
        super().__setitem__(key, value)
        if value.is_symbolic():
            warnings.warn(f"It is not recommended to update the values of '{self.name}' via indexing using symbolic "
                          f"values. Use the update method instead. Make sure to update all required attributes as well."
                          )
            self._update_names()

    def __delitem__(self, key):
        """Item deletion method"""
        super().__delitem__(key)
        self._update_names()
        if isinstance(key, (list, tuple)):
            for k in reversed(key):
                if self._axis == 0:
                    if self._fx is ca.DM:
                        # NOTE: The names won't be adjusted by self._update_names(), if the data format of the vector
                        #  is DM.
                        del self._names[k]
                    del self._description[k]
                    del self._labels[k]
                    del self._units[k]
        else:
            if self._axis == 0:
                if self._fx is ca.DM:
                    # NOTE: The names won't be adjusted by self._update_names(), if the data format of the vector is DM.
                    del self._names[key]
                del self._description[key]
                del self._labels[key]
                del self._units[key]

    def __contains__(self, item):
        """Item check method"""
        if isinstance(item, ca.SX):
            if item.is_scalar():
                if item.name() in self._names:
                    return True
            else:
                return self.__contains__(item.elements())
        elif isinstance(item, str):
            if item in self._names:
                return True
        elif isinstance(item, (list, tuple)):
            return all(self.__contains__(k) for k in item)
        return False

    def _update_names(self):
        """

        :return:
        """
        if self._fx is ca.SX:
            self._names = [val.name() for val in self._values.elements()]
        elif self._fx is ca.MX:
            if 'Vertcat' in self._values.class_name():
                if self._values.n_primitives() == self._values.size1():
                    self._names = [self._values[k].name() for k in range(self._values.size1())]
                else:
                    # NOTE: Should be working for collocation points right now. Other use cases still need to be tested
                    #  -> TODO
                    self._names = [prim.name() + '_' + str(k) for prim in self._values.primitives() for k in
                                   range(prim.size1())]
            else:
                if self.size1() > 1:
                    self._names = [self._values.name() + '_' + str(k) for k in range(self._values.size1())]
                elif self.size1() == 1:
                    self._names = [self._values.name()]
                else:
                    self._names = []

    def _update_description(self, description):
        """

        :param description:
        :return:
        """
        if description is not None:
            if isinstance(description, (list, tuple)):
                if len(description) != len(self._names):
                    raise ValueError(f"Dimension mismatch while trying to add description information. Expected "
                                     f"iterable of dimension {len(self._names)}, got iterable of dimension "
                                     f"{len(description)}.")
                # NOTE: The argument 'description' should be a list of strings, so deepcopy is not necessary here
                if isinstance(description, tuple):
                    self._description = list(description)
                else:
                    self._description = copy(description)
            elif isinstance(description, str):
                self._description = len(self._names) * [description]
            else:
                raise TypeError("Wrong type {} for 'description' argument".format(type(description)))
        else:
            self._description = len(self._names) * [""]

    def _update_labels(self, labels):
        """

        :param labels:
        :return:
        """
        if labels is not None:
            if isinstance(labels, (list, tuple)):
                if len(labels) != len(self._names):
                    raise ValueError(f"Dimension mismatch while trying to add label information. Expected iterable of "
                                     f"dimension {len(self._names)}, got iterable of dimension {len(labels)}.")
                # NOTE: The argument 'labels' should be a list of strings, so deepcopy is not necessary here
                if isinstance(labels, tuple):
                    self._labels = list(labels)
                else:
                    self._labels = copy(labels)
            elif isinstance(labels, str):
                self._labels = len(self._names) * [labels]
            else:
                raise TypeError("Wrong type {} for 'labels' argument".format(type(labels)))
        else:
            self._labels = len(self._names) * [""]

    def _update_units(self, units):
        """

        :param units:
        :return:
        """
        if units is not None:
            if isinstance(units, (list, tuple)):
                if len(units) != len(self._names):
                    raise ValueError(f"Dimension mismatch while trying to add unit information. Expected iterable of "
                                     f"dimension {len(self._names)}, got iterable of dimension {len(units)}.")
                # NOTE: The argument 'units' should be a list of strings, so deepcopy is not necessary here
                if isinstance(units, tuple):
                    self._units = list(units)
                else:
                    self._units = copy(units)
            elif isinstance(units, str):
                self._units = len(self._names) * [units]
            else:
                raise TypeError("Wrong type {} for 'units' argument".format(type(units)))
        else:
            self._units = len(self._names) * [""]

    @property
    def names(self):
        """

        :return:
        """
        return self._names

    @property
    def description(self):
        """

        :return:
        """
        return self._description

    @property
    def labels(self):
        """

        :return:
        """
        return self._labels

    @property
    def units(self):
        """

        :return:
        """
        return self._units

    def add(self, obj, axis=0, description=None, labels=None, units=None):
        """

        :param obj:
        :param axis:
        :param description:
        :param labels:
        :param units:
        :return:
        """
        # TODO: Implement skip functionality
        if isinstance(obj, Vector):
            if axis == 0:
                self._values = ca.vertcat(self._values, obj.values)
                self._update_names()
                # TODO: Handle description, labels and units keywords if not None
                self._update_description(self._description + obj.description)
                self._update_labels(self._labels + obj.labels)
                self._update_units(self._units + obj.units)
            elif axis == 1:
                self._values = ca.horzcat(self._values, obj.values)
            else:
                raise IndexError("Argument 'axis' is out of bounds for array of dimension 2")
        else:
            other = convert(obj, self._fx)
            if axis == 0:
                self._values = ca.vertcat(self._values, other)
                self._update_names()
            elif axis == 1:
                self._values = ca.horzcat(self._values, other)
            else:
                raise IndexError("Argument 'axis' is out of bounds for array of dimension 2")
            n1 = other.size1()

            if units is not None:
                if isinstance(units, list):
                    if len(units) != len(n1):
                        raise ValueError(
                            f"Dimension mismatch while trying to update unit information. Expected iterable of "
                            f"dimension {len(n1)}, got iterable of dimension {len(units)}.")
                    if axis == 0:
                        self._update_units(self._units + units)
                    elif axis == 1:
                        # TODO: Compare units and throw a warning if they are not consistent
                        pass
                    else:
                        raise IndexError("Argument 'axis' is out of bounds for array of dimension 2")
                elif isinstance(units, str):
                    if axis == 0:
                        self._update_units(self._units + n1 * [units])
                    elif axis == 1:
                        # TODO: Compare units and throw a warning if they are not consistent
                        pass
                    else:
                        raise IndexError("Argument 'axis' is out of bounds for array of dimension 2")
                else:
                    raise TypeError("Wrong type of arguments for function {}".format(who_am_i()))
            else:
                if axis == 0:
                    self._update_units(self._units + n1 * [''])
                elif axis == 1:
                    # TODO: Compare units and throw a warning if they are not consistent
                    pass
                else:
                    raise IndexError("Argument 'axis' is out of bounds for array of dimension 2")

            if description is not None:
                if isinstance(description, list):
                    if len(description) != len(n1):
                        raise ValueError(f"Dimension mismatch while trying to update description information. Expected "
                                         f"iterable of dimension {len(n1)}, got iterable of dimension "
                                         f"{len(description)}.")
                    if axis == 0:
                        self._update_description(self._description + description)
                    elif axis == 1:
                        # TODO: Compare units and throw a warning if they are not consistent
                        pass
                    else:
                        raise IndexError("Argument 'axis' is out of bounds for array of dimension 2")
                elif isinstance(description, str):
                    if axis == 0:
                        self._update_description(self._description + n1 * [description])
                    elif axis == 1:
                        # TODO: Compare units and throw a warning if they are not consistent
                        pass
                    else:
                        raise IndexError("Argument 'axis' is out of bounds for array of dimension 2")
                else:
                    raise TypeError("Wrong type of arguments for function {}".format(who_am_i()))
            else:
                if axis == 0:
                    self._update_description(self._description + n1 * [''])
                elif axis == 1:
                    # TODO: Compare units and throw a warning if they are not consistent
                    pass
                else:
                    raise IndexError("Argument 'axis' is out of bounds for array of dimension 2")

            if labels is not None:
                if isinstance(labels, list):
                    if len(labels) != len(n1):
                        raise ValueError(f"Dimension mismatch while trying to update label information. Expected "
                                         f"iterable of dimension {len(n1)}, got iterable of dimension {len(labels)}.")
                    if axis == 0:
                        self._update_labels(self._labels + labels)
                    elif axis == 1:
                        # TODO: Compare units and throw a warning if they are not consistent
                        pass
                    else:
                        raise IndexError("Argument 'axis' is out of bounds for array of dimension 2")
                elif isinstance(labels, str):
                    if axis == 0:
                        self._update_labels(self._labels + n1 * [labels])
                    elif axis == 1:
                        # TODO: Compare units and throw a warning if they are not consistent
                        pass
                    else:
                        raise IndexError("Argument 'axis' is out of bounds for array of dimension 2")
                else:
                    raise TypeError("Wrong type of arguments for function {}".format(who_am_i()))
            else:
                if axis == 0:
                    self._update_labels(self._labels + n1 * [''])
                elif axis == 1:
                    # TODO: Compare units and throw a warning if they are not consistent
                    pass
                else:
                    raise IndexError("Argument 'axis' is out of bounds for array of dimension 2")

        self._update_shape(None)
        self._update_parent()

    def clear(self) -> None:
        """

        :return:
        """
        self._values = self._fx()
        if self._fx is ca.DM:
            self._update_shape(None)
        else:  # SX or MX
            self._update_names()
            self._update_description(None)
            self._update_labels(None)
            self._update_units(None)
            self._update_shape(None)
            self._update_parent()

    def get_by_name(self, name):
        """

        :param name:
        :return:
        """
        index = self.index(name)
        if self._fx is ca.MX:
            # NOTE: Slicing or list access makes MX variables non-symbolic, so they are no longer valid inputs for
            #  functions (check 'MX.is_symbolic' and 'MX.is_valid_input' methods). Workaround by concatenating the
            #  single entries.
            return ca.vertcat(*[self._values[k] for k in index])
        else:
            return self._values[index, :]

    def index(self, names):
        """

        :param names:
        :return:
        """
        if isinstance(names, str):
            return [k for k, v in enumerate(self._names) if v == names]
        elif isinstance(names, (list, tuple, KeysView)):
            return [self._names.index(name) for name in names if name in self._names]
        else:
            raise TypeError(f"Wrong type {type(names)} for argument 'names'")

    def pop(self, index=-1):
        """

        :param index:
        :return:
        """
        if not isinstance(index, (list, tuple)):
            index = [index]
        stack = [self._values[k] for k in index]
        self.__delitem__(index)
        return ca.vertcat(*stack)

    def remove(self, index: Union[int, Sequence[int], slice], axis: Optional[int] = None) -> None:
        """

        :param index:
        :param axis:
        :return:
        """
        old_axis = self._axis
        if axis is not None:
            self._axis = axis

        self.__delitem__(index)

        self._axis = old_axis

    def set(self, obj, description=None, labels=None, units=None, **kwargs):
        """

        :param obj:
        :param description:
        :param labels:
        :param units:
        :param kwargs:
        :return:
        """
        # TODO: Implement skip functionality
        if isinstance(obj, Vector):
            self._values = copy(obj.values)
            self._update_names()
            if description is not None:
                self._update_description(description)
            else:
                self._update_description(obj.description)
            if labels is not None:
                self._update_labels(labels)
            else:
                self._update_labels(obj.labels)
            if units is not None:
                self._update_units(units)
            else:
                self._update_units(obj.units)
        elif isinstance(obj, self._fx):
            self._values = copy(obj)
            self._update_names()
            if description is not None:
                self._update_description(description)
            if labels is not None:
                self._update_labels(labels)
            if units is not None:
                self._update_units(units)
        elif isinstance(obj, (list, tuple)):
            self._values = convert(obj, self._fx, **kwargs)
            self._update_names()
            if description is not None:
                self._update_description(description)
            if labels is not None:
                self._update_labels(labels)
            if units is not None:
                self._update_units(units)
        elif isinstance(obj, np.ndarray):
            self._values = convert(obj, self._fx, **kwargs)
            self._update_names()
            if description is not None:
                self._update_description(description)
            if labels is not None:
                self._update_labels(labels)
            if units is not None:
                self._update_units(units)
        elif isinstance(obj, (str, int, float)):
            self._values = convert(obj, self._fx, **kwargs)
            self._update_names()
            if description is not None:
                self._update_description(description)
            if labels is not None:
                self._update_labels(labels)
            if units is not None:
                self._update_units(units)
        else:
            raise TypeError("Wrong type of arguments for function {}".format(who_am_i()))
        self._update_shape(None)
        self._update_parent()


class Equations:
    """"""
    # TODO: Typing hints
    # TODO: Add __repr__ & __str__
    # TODO: Inherit from Object
    # TODO: Switch between SX and MX
    def __init__(self, data_format, expression=None):
        """Constructor method"""
        if data_format in [ca.SX, ca.MX, ca.DM]:
            self._fx = data_format
        elif isinstance(data_format, str):
            if data_format.lower() == 'sx':
                self._fx = ca.SX
            elif data_format.lower() == 'mx':
                self._fx = ca.MX
            elif data_format.lower() == 'dm':
                self._fx = ca.DM
            else:
                raise ValueError(f"Format {data_format} not recognized")
        else:
            raise TypeError(f"Type {type(data_format)} not supported")

        self._equations = []
        if expression is None:
            expression = {}
        for key, val in expression.items():
            self._equations.append(key)
            if not isinstance(val, self._fx):
                setattr(self, '_' + key, convert(val, self._fx))
            else:
                setattr(self, '_' + key, val)

    def __contains__(self, item):
        """Item check method"""
        if hasattr(self, '_' + item):
            return True
        else:
            return False

    def _add(self, attr, obj, **kwargs):
        """

        :param attr:
        :param obj:
        :param kwargs:
        :return:
        """
        pointer = getattr(self, '_' + attr)
        if isinstance(obj, Vector):
            setattr(self, '_' + attr, ca.vertcat(pointer, copy(obj.values)))
        elif isinstance(obj, self._fx):
            setattr(self, '_' + attr, ca.vertcat(pointer, copy(obj)))
        elif isinstance(obj, (list, tuple)):
            setattr(self, '_' + attr, ca.vertcat(pointer, convert(obj, self._fx, name=attr)))
        else:
            raise TypeError(f"Wrong type of arguments for function {who_am_i()}")

    def _set(self, attr, obj, **kwargs):
        """

        :param attr:
        :param obj:
        :param kwargs:
        :return:
        """
        if isinstance(obj, Vector):
            setattr(self, '_' + attr, copy(obj.values))
        elif isinstance(obj, self._fx):
            setattr(self, '_' + attr, copy(obj))
        elif isinstance(obj, (list, tuple)):
            _fx = kwargs.get('_fx', None)  # for lbg and ubg of Problem class
            setattr(self, '_' + attr, convert(obj, _fx or self._fx, name=attr))
        else:
            raise TypeError(f"Wrong type of arguments for function {who_am_i()}")

    def _substitute(self, keys, values):
        """

        :param keys:
        :param values:
        :return:
        """
        for attr in self._equations:
            eq = getattr(self, '_' + attr)
            if not eq.is_empty():
                self._set(attr, ca.substitute(eq, keys, values))

    def _switch_data_type(self, *args, reverse=False):
        """

        :param args:
        :param reverse:
        :return:
        """
        equations = [getattr(self, '_' + attr) for attr in self._equations]
        transformer = ca.Function('transformer', [*args], equations)

        new_args = []
        for arg in args:
            if not arg.is_empty():
                new_arg = ca.vertcat(*[ca.MX.sym(var.name()) for var in arg.elements()])
            else:
                new_arg = ca.MX()
            new_args.append(new_arg)

        equations = transformer(*new_args)

        return new_args, equations

    def sym(self, *args):
        """

        :param args:
        :return:
        """
        return self._fx.sym(*args)

    def to_function(self, name, *args, **kwargs):
        """

        :param name:
        :param args:
        :param kwargs:
        :return:
        """
        arg_names = [arg.name() for arg in args]
        return ca.Function(name,
                           args,
                           [getattr(self, '_' + key) for key in self._equations],
                           arg_names,
                           ['out' + str(k) for k in range(len(self._equations))])


class RightHandSide(Equations):
    """"""
    # TODO: Typing hints
    # TODO: Add __repr__ & __str__
    def __init__(self, equations=None, discrete=False, use_sx=True, parent=None):
        """Constructor method"""
        expression = {}
        if isinstance(equations, dict):
            ode = equations.get('ode', None)
            alg = equations.get('alg', None)
            meas = equations.get('meas', None)
            expression['ode'] = ode
            expression['alg'] = alg
            expression['meas'] = meas
        else:
            expression['ode'] = []
            expression['alg'] = []
            expression['meas'] = []

        if use_sx:
            super().__init__(ca.SX, expression=expression)
        else:
            super().__init__(ca.MX, expression=expression)

        self._matrix_notation = None
        self._discrete = discrete
        self._use_sx = use_sx
        self._parent = parent

    def _generate_matrix_notation(self, states, inputs):
        """

        :param states:
        :param inputs:
        :return:
        """
        matrix_notation = {}
        if not self._ode.is_empty():
            equations = ca.vertcat(self._ode, self._alg)
            matrix_notation['A'] = ca.jacobian(equations, states)
            matrix_notation['B'] = ca.jacobian(equations, inputs)
        if not self._meas.is_empty():
            matrix_notation['C'] = ca.jacobian(self._meas, states)
            matrix_notation['D'] = ca.jacobian(self._meas, inputs)
        if not matrix_notation:
            matrix_notation = None
        self._matrix_notation = matrix_notation

    def _is_time_variant(self, t):
        """

        :param t:
        :return:
        """
        return ca.depends_on(self._ode, t), ca.depends_on(self._alg, t), ca.depends_on(self._meas, t)

    @property
    def ode(self):
        """

        :return:
        """
        return self._ode

    # @ode.setter
    def set_ode(self, obj, **kwargs):
        """

        :param obj:
        :param kwargs:
        :return:
        """
        self._set('ode', obj, **kwargs)

    @property
    def alg(self):
        """

        :return:
        """
        return self._alg

    # @alg.setter
    def set_alg(self, obj, **kwargs):
        """

        :param obj:
        :param kwargs:
        :return:
        """
        self._set('alg', obj, **kwargs)

    @property
    def meas(self):
        """

        :return:
        """
        return self._meas

    # @meas.setter
    def set_meas(self, obj, **kwargs):
        """

        :param obj:
        :param kwargs:
        :return:
        """
        self._set('meas', obj, **kwargs)

    @property
    def matrix_notation(self):
        """

        :return:
        """
        return self._matrix_notation

    @property
    def discrete(self):
        """

        :return:
        """
        return self._discrete

    @discrete.setter
    def discrete(self, boolean):
        if boolean is not self._discrete:
            self._discrete = boolean

    @property
    def continuous(self):
        """

        :return:
        """
        return not self._discrete

    @continuous.setter
    def continuous(self, boolean):
        if boolean is self._discrete:
            self._discrete = not boolean

    @property
    def parent(self):
        """

        :return:
        """
        return self._parent

    @parent.setter
    def parent(self, arg):
        if self._parent != arg:
            self._parent = arg

    def add(self, obj, **kwargs):
        """

        :param obj:
        :param kwargs:
        :return:
        """
        if isinstance(obj, RightHandSide):
            self._ode = ca.vertcat(self._ode, obj.ode)
            self._alg = ca.vertcat(self._alg, obj.alg)
            self._meas = ca.vertcat(self._meas, obj.meas)
        elif isinstance(obj, dict):
            for attr in ['ode', 'alg', 'meas']:
                value = obj.get(attr)
                if value is not None:
                    self._add(attr, value, **kwargs)
        else:
            raise TypeError("Wrong type of arguments for function {}".format(who_am_i()))

    def check_quadrature_function(self, quad, dt, t, x, z, u, p, *p_ext):
        """

        :param quad:
        :param dt:
        :param t:
        :param x:
        :param z:
        :param u:
        :param p:
        :param p_ext:
        :return:
        """
        # TODO: Quadrature function could also be time-variant
        time_variant = self._is_time_variant(t)

        variables = [dt, t, x, z, u, p]
        if p_ext:
            variables.extend(p_ext)
        ode = self._ode
        alg = self._alg
        meas = self._meas

        eval_sx = True
        if isinstance(quad, ca.Function):
            # TODO: Is there a better way to check whether eval_sx is defined for a specific function, i.e., if we can
            #  use SX variables as arguments to the function?
            try:
                quad = quad.expand()
            except RuntimeError:
                eval_sx = False

            name_in = quad.name_in()
            name_out = quad.name_out()

            if eval_sx:
                all_the_variables = ca.vertcat(*variables)
                all_the_names = [var.name() for var in all_the_variables.elements()]
            else:
                sx_vars = check_if_list_of_type(variables, ca.SX)
                if sx_vars and self._fx is ca.SX:
                    all_the_names = [el.name() for var in variables for el in var.elements()]
                    if not any(time_variant):
                        del variables[1]
                        del all_the_names[1]
                    if not self._discrete:
                        del variables[0]
                        del all_the_names[0]

                    variables, equations = self._switch_data_type(*variables)
                    all_the_variables = ca.vertcat(*variables)
                    ode, alg, meas = equations

                    if not t.is_empty():
                        t = ca.MX.sym('t')
                    else:
                        t = ca.MX()
                    if not self._discrete:
                        if not dt.is_empty():
                            dt = ca.MX.sym('dt')
                        else:
                            dt = ca.MX()
                        variables = [dt, t] + variables
                    else:
                        variables.insert(1, t)
                elif not sx_vars and self._fx is ca.MX:
                    all_the_variables = ca.vertcat(*variables)
                    all_the_names = [all_the_variables[k].name() for k in range(all_the_variables.size1())]
                else:
                    if sx_vars:
                        raise RuntimeError("Data format mismatch. Variables are given in the SX format, but equations"
                                           " are in the MX format.")
                    else:
                        raise RuntimeError("Data format mismatch. Variables are given in the MX format, but equations"
                                           " are in the SX format.")

            vars_in = {var: all_the_variables[all_the_names.index(var)] for var in name_in}

            quad = quad(**vars_in)
            quad = ca.vertcat(*[quad[var] for var in name_out])
        elif isinstance(quad, ca.MX) and quad.is_empty():
            sx_vars = check_if_list_of_type(variables, ca.SX)
            if sx_vars and self._fx is ca.SX:
                all_the_names = [el.name() for var in variables for el in var.elements()]
                if not any(time_variant):
                    del variables[1]
                    del all_the_names[1]
                if not self._discrete:
                    del variables[0]
                    del all_the_names[0]

                variables, equations = self._switch_data_type(*variables)
                ode, alg, meas = equations

                if not t.is_empty():
                    t = ca.MX.sym('t')
                else:
                    t = ca.MX()
                if not self._discrete:
                    if not dt.is_empty():
                        dt = ca.MX.sym('dt')
                    else:
                        dt = ca.MX()
                    variables = [dt, t] + variables
                else:
                    variables.insert(1, t)

        return variables, (ode, alg, meas, quad), time_variant

    def generate_matrix_notation(self, states, inputs):
        """

        :param states:
        :param inputs:
        :return:
        """
        # TODO: Add more flexibility processing the arguments (e.g., check for Vector vs. SX, ...)
        self._generate_matrix_notation(states, inputs)

    def set(self, obj, **kwargs):
        """

        :param obj:
        :param kwargs:
        :return:
        """
        if isinstance(obj, RightHandSide):
            self._ode = obj.ode
            self._alg = obj.alg
            self._meas = obj.meas
        elif isinstance(obj, dict):
            for attr in ['ode', 'alg', 'meas']:
                value = obj.get(attr, None)
                if value is not None:
                    self._set(attr, value, **kwargs)
        else:
            raise TypeError("Wrong type of arguments for function {}".format(who_am_i()))

    def substitute(self, obj, *args, **kwargs):
        """

        :param obj:
        :param args:
        :param kwargs:
        :return:
        """
        if len(args) == 1:
            keys = obj.values
            values = convert(args[0], self._fx)
            if not values.is_empty():
                if not self._ode.is_empty():
                    self._set('ode', ca.substitute(self._ode, keys, values))

        if kwargs:
            keys = obj.get_by_name(kwargs.keys())
            values = convert(kwargs.values(), self._fx)
            if not values.is_empty():
                self._substitute(keys, values)

    def scale(self, var, names, value, eq=None):
        """

        :param var:
        :param names:
        :param value:
        :param eq:
        :return:
        """
        if not is_list_like(names):
            names = [names]
        if not is_list_like(value):
            value = [value]
        else:
            if hasattr(value, 'ndim') and value.ndim > 1:
                value = value.flatten()

        if var.size1() != len(value) != len(names):
            raise ValueError(f"Dimension mismatch. Expected {var.size1()}x{var.size2()}, got {len(value)}x1.")

        scale_kwargs = {f'{name}': var[k] * value[k] for k, name in enumerate(names)}
        self.substitute(var, **scale_kwargs)

        if eq is not None:
            index = var.index(names)
            if not index:
                index = slice(0, var.size1())
            attr = getattr(self, '_' + eq)
            attr[index] /= value
            self._set(eq, attr)

    def to_function(self, name, *args, **kwargs):
        """

        :param name:
        :param args:
        :param kwargs:
        :return:
        """
        # TODO: Maybe prioritize Vector variables and solver from kwargs
        if self._parent is not None:
            dt = self._parent.dt
            t = self._parent.t
            x = self._parent.x
            z = self._parent.z
            u = self._parent.u
            p = self._parent.p
            solver = self._parent.solver
        else:
            dt = kwargs.get('dt')
            t = kwargs.get('t')
            x = kwargs.get('x')
            z = kwargs.get('z')
            u = kwargs.get('u')
            p = kwargs.get('p')
            solver = kwargs.get('solver')

            # TODO: Switch to self._fx()?
            if dt is None:
                dt = ca.SX()
            if t is None:
                t = ca.SX()
            if x is None:
                x = ca.SX()
            if z is None:
                z = ca.SX()
            if u is None:
                u = ca.SX()
            if p is None:
                p = ca.SX()

        append_dynamics_framework = kwargs.get('append_dynamics_framework', True)

        generate_matrix_notation = kwargs.get('generate_matrix_notation', False)
        if generate_matrix_notation:
            self._generate_matrix_notation(ca.vertcat(x, z), u)

        options = kwargs.get('opts', None)
        if options is None:
            options = {}

        external_parameters = kwargs.get('external_parameters')
        if external_parameters is None:
            external_parameters = []

        quad = kwargs.get('quadrature')
        if quad is not None:
            variables, equations, time_variant = self.check_quadrature_function(quad, dt, t, x, z, u, p,
                                                                                *external_parameters)
            dt = variables[0]
            t = variables[1]
            x = variables[2]
            z = variables[3]
            u = variables[4]
            p = variables[5]
            external_parameters = []
            if len(variables) > 6:
                external_parameters.append(variables[6])
            if len(variables) > 7:
                external_parameters.append(variables[7])
            if len(variables) > 8:
                external_parameters.append(variables[8])
            ode, alg, meas, quad = equations
        else:
            time_variant = self._is_time_variant(t)

            ode = self._ode
            alg = self._alg
            meas = self._meas
            quad = self._fx()

        if not self._discrete:
            if append_dynamics_framework:
                name += '_continuous'

            if any(time_variant):
                t0 = self._fx.sym('t0')

            dae = {}
            if any(time_variant) and not t.is_empty():
                dae['t'] = t
            if not x.is_empty():
                dae['x'] = x
            if not z.is_empty():
                dae['z'] = z
            parameter_args = []
            if not u.is_empty():
                parameter_args.append(u)
            if not p.is_empty():
                parameter_args.append(p)
            if any(time_variant):
                parameter_args.append(t0)
            for p_ext in external_parameters:
                if not p_ext.is_empty():
                    parameter_args.append(p_ext)
            dae['p'] = ca.vertcat(*parameter_args)
            if not ode.is_empty():
                if time_variant[0]:
                    dae['ode'] = ca.substitute(ode, t, t + t0)
                else:
                    dae['ode'] = ode
            if not alg.is_empty():
                if time_variant[1]:
                    dae['alg'] = ca.substitute(alg, t, t + t0)
                else:
                    dae['alg'] = alg
            if not quad.is_empty():
                dae['quad'] = quad

            integrator = None
            meas_fun = None
            function = ca.integrator(name, solver, dae, options)

            if not meas.is_empty():
                if any(time_variant) and not t.is_empty():
                    del dae['t']
                if not x.is_empty():
                    dae['x0'] = dae.pop('x')
                if not z.is_empty():
                    dae['z0'] = dae.pop('z')
                if not ode.is_empty():
                    del dae['ode']
                if not alg.is_empty():
                    del dae['alg']
                if not quad.is_empty():
                    del dae['quad']
                if any(time_variant):
                    dae['yf'] = ca.substitute(meas, t, t0 + options['tf'])
                else:
                    dae['yf'] = meas
                meas_fun = ca.Function('meas_fun', dae, ca.integrator_in(), ca.integrator_out() + ['yf'])

                x0 = ca.MX.sym('x0', x.shape)
                z0 = ca.MX.sym('z0', z.shape)
                if 'p' in dae:
                    up = ca.MX.sym('p', dae['p'].shape)
                else:
                    up = ca.MX()

                integrator = function
                result = integrator(x0=x0, z0=z0, p=up)
                xf = result['xf']
                zf = result['zf']
                qf = result['qf']
                result = meas_fun(x0=xf, z0=zf, p=up)
                y = result['yf']

                if not x.is_empty():
                    dae['x0'] = x0
                if not z.is_empty():
                    dae['z0'] = z0
                if not up.is_empty():
                    dae['p'] = up
                if not ode.is_empty():
                    dae['xf'] = xf
                dae['yf'] = y
                if not alg.is_empty():
                    dae['zf'] = zf
                if not quad.is_empty():
                    dae['qf'] = qf

                function = ca.Function(name, dae, ca.integrator_in(), ca.integrator_out() + ['yf'])
        else:
            # TODO: Deal with initial time 't0' in options
            if append_dynamics_framework:
                if self._parent is not None:
                    if hasattr(self._parent, 'solver'):
                        name += '_discrete'
                else:
                    name += '_discrete'

            X = kwargs.get('X')
            Z = kwargs.get('Z')
            if X is None:
                X = self._fx()
            if Z is None:
                Z = self._fx()

            dae = {}
            if not x.is_empty():
                dae['x0'] = x
            if not z.is_empty():
                dae['z0'] = z
            parameter_args = []
            if not u.is_empty():
                parameter_args.append(u)
            if not p.is_empty():
                parameter_args.append(p)
            if any(time_variant):
                parameter_args.append(t)
            for p_ext in external_parameters:
                if not p_ext.is_empty():
                    parameter_args.append(p_ext)
            dae['p'] = ca.vertcat(*parameter_args)
            col_in = []
            if not X.is_empty():
                dae['x_col'] = X
                col_in.append('x_col')
            if not Z.is_empty():
                dae['z_col'] = Z
                col_in.append('z_col')

            if 'tf' in options:
                # NOTE: Somehow, if we have explicit Runge-Kutta on a DAE or collocation, we need to substitute 't' as
                #  well for the time-variant cases, since the function will have 't' as a free variable although none
                #  of the equations depend on 't'
                if isinstance(t, ca.MX) and not any(time_variant):
                    to_substitute = ca.vertcat(dt, t)
                    substitute = [options['tf'], 0.]
                else:
                    to_substitute = dt
                    substitute = options['tf']

                if not ode.is_empty():
                    dae['xf'] = ca.substitute(ode, to_substitute, substitute)
                if not alg.is_empty():
                    dae['zf'] = ca.substitute(alg, to_substitute, substitute)
                if not quad.is_empty():
                    dae['qf'] = ca.substitute(quad, to_substitute, substitute)

                function = ca.Function(name, dae, ca.integrator_in() + col_in, ca.integrator_out())
            elif 'grid' in options:
                dae['dt'] = dt
                col_in = ['dt'] + col_in

                if isinstance(t, ca.MX) and not any(time_variant):
                    to_substitute = t
                    substitute = [0.]
                    if not ode.is_empty():
                        dae['xf'] = ca.substitute(ode, t, 0.)
                    if not alg.is_empty():
                        dae['zf'] = ca.substitute(alg, t, 0.)
                    if not quad.is_empty():
                        dae['qf'] = ca.substitute(quad, t, 0.)
                else:
                    to_substitute = self._fx()
                    substitute = []
                    if not ode.is_empty():
                        dae['xf'] = ode
                    if not alg.is_empty():
                        dae['zf'] = alg
                    if not quad.is_empty():
                        dae['qf'] = quad

                function = ca.Function(name, dae, ca.integrator_in() + col_in, ca.integrator_out())
                diff = ca.diff(options['grid'])

                if not ode.is_empty():
                    del dae['xf']
                if not alg.is_empty():
                    del dae['zf']
                if not quad.is_empty():
                    del dae['qf']

                function = function.mapaccum(diff.size1())
                dae['dt'] = diff
                result = function(**dae)

                if not ode.is_empty():
                    dae['xf'] = result['xf']
                if not alg.is_empty():
                    dae['zf'] = result['zf']
                if not quad.is_empty():
                    dae['qf'] = result['qf']

                col_in = col_in[1:]
                del dae['dt']
                function = ca.Function(name, dae, ca.integrator_in() + col_in, ca.integrator_out())
            else:
                raise KeyError("Options dictionary incomplete. Either 'tf' or 'grid' is missing.")
            integrator = None
            meas_fun = None

            if not meas.is_empty():
                if not ode.is_empty():
                    del dae['xf']
                if not alg.is_empty():
                    del dae['zf']
                if not quad.is_empty():
                    del dae['qf']
                if not to_substitute.is_empty() and substitute:
                    dae['yf'] = ca.substitute(meas, to_substitute, substitute)
                else:
                    dae['yf'] = meas
                # TODO: Do we need col_in here as well?
                meas_fun = ca.Function('meas', dae, ca.integrator_in(), ca.integrator_out() + ['yf'])
                del dae['yf']

                integrator = function
                result = integrator(**dae)
                xf = result['xf']
                zf = result['zf']
                qf = result['qf']
                result = meas_fun(x0=xf, z0=zf, p=dae.get('p', self._fx()))
                y = result['yf']

                if not ode.is_empty():
                    dae['xf'] = xf
                dae['yf'] = y
                if not alg.is_empty():
                    dae['zf'] = zf
                if not quad.is_empty():
                    dae['qf'] = qf

                # TODO: Do we need col_in here as well?
                function = ca.Function(name, dae, ca.integrator_in(), ca.integrator_out() + ['yf'])

        return function, integrator, meas_fun

    def is_empty(self, *args):
        """

        :param args:
        :return:
        """
        return self._ode.is_empty(*args) and self._alg.is_empty(*args) and self._meas.is_empty(*args)

    def is_time_variant(self, t=None):
        """

        :param t:
        :return:
        """
        if self._parent is not None and not self._parent.t.is_empty():
            if t is not None:
                warnings.warn("Setting the keyword parameter 't' while having a parent object is not supported. Using "
                              "parent object.")
            return any(self._is_time_variant(self._parent.t))
        elif t is not None and isinstance(t, self._fx):
            return any(self._is_time_variant(t))
        return None


class Problem(Equations):
    """"""
    # TODO: Typing hints
    # TODO: Add __repr__ & __str__
    def __init__(self, equations=None, sense=None, parent=None):
        """Constructor method"""
        expression = {}
        if isinstance(equations, dict):
            obj = equations.get('obj', None)
            cons = equations.get('cons', None)
            expression['obj'] = obj
            expression['cons'] = cons
        else:
            expression['obj'] = []
            expression['cons'] = []
            self._lbg = ca.DM()
            self._ubg = ca.DM()

        super().__init__(ca.SX, expression=expression)

        if isinstance(sense, str):
            if sense.lower() in ['min', 'minimise', 'minimize', '1']:
                self._sense = 'min'
            elif sense.lower() in ['max', 'maximise', 'maximize', '-1']:
                self._sense = 'max'
            else:
                raise ValueError(f"Optimization direction {sense} not understood")
        elif isinstance(sense, int):
            if sense == 1:
                self._sense = 'min'
            elif sense == -1:
                self._sense = 'max'
            else:
                raise ValueError(f"Optimization direction {sense} not understood")
        elif sense is not None:
            raise TypeError(f"Wrong type '{type(sense)}' for input argument 'sense'")
        else:
            self._sense = 'min'

        self._parent = parent

    def _process_constraints(self, constraints, **kwargs):
        """

        :param constraints:
        :param kwargs:
        :return:
        """
        # TODO: Add support for dynamic constraints
        g = []
        lbg = []
        ubg = []

        x = kwargs.get('x', None)
        p = kwargs.get('p', None)
        if x is None:
            x = self._fx()
        if p is None:
            p = self._fx()
        fun = None

        if isinstance(constraints, (list, tuple)):
            if all(isinstance(constraint, self._fx) for constraint in constraints):
                fun = ca.Function('fun', [x, p], constraints)
        else:
            if isinstance(constraints, self._fx):
                fun = ca.Function('fun', [x, p], [constraints])

        if fun is not None:
            split_expr = _split_expression(fun, ca.vertcat(x, p), '<', '<=', '==', '>=', '>')
            for expr in split_expr:
                if len(expr) == 3:
                    g.append(expr[1])
                    lbg.append(expr[0])
                    ubg.append(expr[2])
                elif len(expr) == 2:
                    if isinstance(expr[0], self._fx):
                        if isinstance(expr[1], self._fx):
                            raise RuntimeError(f"Something went wrong while parsing the input arguments for function "
                                               f"{who_am_i()} of class {self.__class__.__name__}. Inform the "
                                               f"maintainer.")
                        g.append(expr[0])
                        lbg.append(-ca.inf)
                        ubg.append(expr[1])
                    elif isinstance(expr[1], self._fx):
                        g.append(expr[1])
                        lbg.append(expr[0])
                        ubg.append(ca.inf)
                    else:
                        raise RuntimeError(f"Something went wrong while parsing the input arguments for function "
                                           f"{who_am_i()} of class {self.__class__.__name__}. Inform the maintainer.")
                else:
                    index = split_expr.index(expr)
                    raise ValueError(f"Ambiguous constraint '{constraints[index]}'. Following forms are allowed: "
                                     f"'lbg <= g <= ubg', 'lbg <= g', 'g <= ubg'")

        return g, lbg, ubg

    @property
    def objective(self):
        """

        :return:
        """
        return self._obj

    obj = objective

    def get_objective(self):
        """

        :return:
        """
        return self._obj

    # @objective.setter
    def set_objective(self, obj, **kwargs):
        """

        :param obj:
        :param kwargs:
        :return:
        """
        self._set('obj', obj, **kwargs)

    @property
    def sense(self):
        """

        :return:
        """
        return self._sense

    @sense.setter
    def sense(self, arg):
        arg = arg.lower().strip()
        if arg in ['min', 'minimize', 'minimise'] and self._sense == 'max':
            self._sense = 'min'
        elif arg in ['max', 'maximize', 'maximise'] and self._sense == 'min':
            self._sense = 'max'
        elif arg not in ['min', 'minimize', 'minimise', 'max', 'maximize', 'maximise']:
            print("Argument not recognized. Ignoring it.")

    @property
    def constraints(self):
        """

        :return:
        """
        # TODO: Niceify output
        return self._cons

    cons = constraints

    def set(self, obj, **kwargs):
        """

        :param obj:
        :param kwargs:
        :return:
        """
        if isinstance(obj, Problem):
            self._obj = obj.obj
            self._cons = obj.cons
        elif isinstance(obj, dict):
            for attr in ['obj', 'cons']:
                value = obj.get(attr, None)
                if value is not None:
                    if attr == 'cons':
                        g, lbg, ubg = self._process_constraints(value, **kwargs)
                        self._set(attr, g, **kwargs)
                        self._set('lbg', lbg, _fx=ca.DM)
                        self._set('ubg', ubg, _fx=ca.DM)
                    else:
                        self._set(attr, value, **kwargs)
        else:
            raise TypeError(f"Wrong type '{type(obj).__name__}' of argument for function {who_am_i()}")

    def to_solver(self, name, interface, **kwargs):
        """

        :param name:
        :param interface:
        :param kwargs:
        :return:
        """
        if self._parent is not None:
            x = self._parent.x
            p = self._parent.p
            solver = self._parent.solver
        else:
            x = kwargs.get('x', None)
            p = kwargs.get('p', None)
            solver = kwargs.get('solver', None)

            if x is None:
                x = self._fx()
            if p is None:
                p = self._fx()

        options = kwargs.get('options', None)
        if options is None:
            options = {}

        problem = {}
        if not x.is_empty():
            problem['x'] = x
        if not p.is_empty():
            problem['p'] = p
        if not self._obj.is_empty():
            if self._sense == 'max':
                problem['f'] = -self._obj
            else:
                problem['f'] = self._obj
        if not self._cons.is_empty():
            problem['g'] = self._cons

        solver = interface(name, solver, problem, options)

        return solver


class Series(Object, metaclass=ABCMeta):
    """"""
    # TODO: Typing hints
    def __init__(
            self,
            backend: Optional[Union[str, PlotManager]] = None,
            id: Optional[str] = None,
            name: Optional[str] = None,
            parent: Optional[Any] = None
    ) -> None:
        """Constructor method"""
        super().__init__(id=id, name=name)

        if backend is None:
            backend = get_plot_backend()
        if backend is None:
            self._plot_manager = None
            warnings.warn("Plots are disabled, since no backend was selected.")
        elif isinstance(backend, str):
            self._plot_manager = PlotManager(backend)
        elif isinstance(backend, PlotManager):
            self._plot_manager = backend
        else:
            self._plot_manager = None
            warnings.warn("Backend for plots not recognized. Plots are disabled.")
        self._parent = parent
        self._data = {}
        self._reference = {}
        self._lower_bound = {}
        self._upper_bound = {}
        self._noise = {}

        if self._id is None:
            self._create_id()

        self._names = []
        self._abscissa = None

        self._n_samples = 0

    def __del__(self):
        """Deletion method"""
        self._data = {}
        self._reference = {}
        self._lower_bound = {}
        self._upper_bound = {}
        self._noise = {}
        self._names = []

    def __getitem__(self, item: str):
        """Item getter method"""
        if item in self._names:
            return self._get_by_name(item)
        else:
            return self.get_by_id(item)

    def __delitem__(self, key):
        """Item deletion method"""
        index = slice(0, None)
        self.remove(key, index)

    def __iter__(self):
        """Item iteration method"""
        yield from self._data

    def __contains__(self, item):
        """Item check method"""
        if item in self._data:
            return True
        return False

    def _check_samples(self) -> None:
        """

        :return:
        """
        n_samples = 0
        for value in self._data.values():
            n_values = value.shape[1]
            if n_values != 0:
                if n_samples == 0:
                    n_samples = n_values
                elif n_samples != n_values:
                    raise ValueError(f"Dimension mismatch in the items of the {self.__class__.__name__} object")
        self._n_samples = n_samples

    @abstractmethod
    def _update_dimensions(self) -> None:
        """

        :return:
        """
        pass

    def _update_kwargs(self, other, kwargs):
        """

        :param other:
        :param kwargs:
        :return:
        """
        pass

    @property
    def plot_backend(self):
        """

        :return:
        """
        if self._plot_manager is None:
            return None
        return self._plot_manager.backend

    @plot_backend.setter
    def plot_backend(self, backend):
        if self._plot_manager is None:
            if backend is not None:
                self._plot_manager = PlotManager(backend)
        elif self._plot_manager is not None:
            if backend is None:
                self._plot_manager = None
            else:
                self._plot_manager.backend = backend

    @property
    def n_samples(self) -> int:
        """

        :return:
        """
        self._check_samples()
        return self._n_samples

    def add(self, arg, value):
        """

        :param arg:
        :param value:
        :return:
        """
        # TODO: Add pandas.Series and pandas.DataFrame
        allowed_types = (int, float, list, tuple, np.ndarray, ca.DM, Vector)
        to_be_listed = (int, float)
        if isinstance(value, Vector):
            if not value.is_constant():
                raise TypeError("Input arguments of class Vector containing symbolic expressions are not yet "
                                "supported.")
        if isinstance(value, dict):
            old_value = value
            value = len(old_value) * [None]
            for key, val in old_value.items():
                index = self._data[arg].index(key)[0]
                value[index] = val

        # NOTE: Ignore information about initial or final value
        arg = arg.split(':')
        arg = arg[0]

        arg = arg.rsplit('_')
        if len(arg) == 1:
            index = ''
        elif len(arg) == 2:
            index = arg[1]
        else:
            raise ValueError(f"Unsupported key {'_'.join(arg)}")
        arg = arg[0]

        if arg in self._data and not index:
            if isinstance(value, allowed_types):
                if isinstance(value, to_be_listed):
                    self._data[arg].add([value], axis=1)
                else:
                    self._data[arg].add(value, axis=1)
            else:
                raise TypeError(f"Expected argument 'value' of type {[type(k).__name__ for k in allowed_types]}, "
                                f"got {type(value)} instead.")
        elif index in ['ref', 'reference']:
            if isinstance(value, allowed_types):
                if isinstance(value, to_be_listed):
                    self._reference[arg].add([value], axis=1)
                else:
                    self._reference[arg].add(value, axis=1)
            else:
                raise TypeError(f"Expected argument 'value' of type {[type(k).__name__ for k in allowed_types]}, "
                                f"got {type(value)} instead.")
        elif index in ['lb', 'lower_bound']:
            if isinstance(value, allowed_types):
                if isinstance(value, to_be_listed):
                    self._lower_bound[arg].add([value], axis=1)
                else:
                    self._lower_bound[arg].add(value, axis=1)
            else:
                raise TypeError(f"Expected argument 'value' of type {[type(k).__name__ for k in allowed_types]}, "
                                f"got {type(value)} instead.")
        elif index in ['ub', 'upper_bound']:
            if isinstance(value, allowed_types):
                if isinstance(value, to_be_listed):
                    self._upper_bound[arg].add([value], axis=1)
                else:
                    self._upper_bound[arg].add(value, axis=1)
            else:
                raise TypeError(f"Expected argument 'value' of type {[type(k).__name__ for k in allowed_types]}, "
                                f"got {type(value)} instead.")
        elif index == 'noise':
            if isinstance(value, allowed_types):
                if isinstance(value, to_be_listed):
                    self._noise[arg].add([value], axis=1)
                else:
                    self._noise[arg].add(value, axis=1)
            else:
                raise TypeError(f"Expected argument 'value' of type {[type(k).__name__ for k in allowed_types]}, "
                                f"got {type(value)} instead.")
        else:
            if not self._data:
                raise KeyError("Data container is empty")

    def append(self, other):
        """

        :param other:
        :return:
        """
        raise NotImplementedError("The append functionality will be implemented in future releases")

    def clear(self):
        """

        :return:
        """
        self.__del__()

    def copy(self):
        """

        :return:
        """
        if self.name is not None:
            new = self.__class__(backend=self._plot_manager.backend, name='copy_of_' + self.name)
        else:
            new = self.__class__(backend=self._plot_manager.backend)

        kwargs = {key: {
            'data_format': ca.DM,
            'description': value.description,
            'labels': value.labels,
            'units': value.units,
            'shape': (value.shape[0], 0),
            'values_or_names': value.names
        } for key, value in self.items()}
        new.setup(*kwargs.keys(), **kwargs)

        for key, value in self._data.items():
            if not value.is_empty():
                new.set(key, value.values)
        for key, value in self._reference.items():
            if not value.is_empty():
                new.set(key + '_ref', value.values)
        for key, value in self._lower_bound.items():
            if not value.is_empty():
                new.set(key + '_lb', value.values)
        for key, value in self._upper_bound.items():
            if not value.is_empty():
                new.set(key + '_ub', value.values)
        for key, value in self._noise.items():
            if not value.is_empty():
                new.set(key + '_noise', value.values)

        return new

    def get_by_id(self, arg):
        """

        :param arg:
        :return:
        """
        arg = arg.split(':')
        if len(arg) == 1:
            bc = ''
        elif len(arg) == 2:
            bc = arg[1]
            if bc not in ['0', 'f']:
                raise ValueError(f"Unsupported appendage '{bc}'. Use '0' to indicate initial values and 'f' to indicate"
                                 f" final values.")
        else:
            raise ValueError(f"Unsupported key {'_'.join(arg)}")
        arg = arg[0]

        arg = arg.rsplit('_')
        if len(arg) == 1:
            index = ''
        elif len(arg) == 2:
            index = arg[1]
        else:
            raise ValueError(f"Unsupported key {'_'.join(arg)}")
        arg = arg[0]

        if bc == '0' and arg in self._data and not index:
            return self._data[arg][:, 0]
        elif bc == 'f' and arg in self._data and not index:
            return self._data[arg][:, -1]
        elif arg in self._data and not bc and not index:
            # NOTE: Used the method Vector.values here, to get consistent output (all is either SX, MX, or DM)
            return self._data[arg].values
        elif index in ['ref', 'reference']:
            return self._reference[arg].values
        elif index in ['lb', 'lower_bound']:
            return self._lower_bound[arg].values
        elif index in ['ub', 'upper_bound']:
            return self._upper_bound[arg].values
        elif index == 'noise':
            return self._noise[arg].values
        elif index == 'noisy':
            return self._data[arg].values + self._noise[arg].values
        return None

    def _get_by_name(self, arg: str) -> Optional[ca.DM]:
        """

        :param arg:
        :return:
        """
        # TODO: Add support for appendices (ref/reference, lb/lower bound, ub/upper bound, noise, noisy)
        arg = arg.split(':')
        if len(arg) == 1:
            k = None
        elif len(arg) == 2:
            bc = arg[1]
            if bc not in ['0', 'f']:
                raise ValueError(f"Unsupported appendage '{bc}'. Use '0' to indicate initial values and 'f' to indicate"
                                 f" final values.")
            k = 0 if bc == '0' else -1
        else:
            raise ValueError(f"Unsupported key {'_'.join(arg)}")
        arg = arg[0]

        if arg.endswith(('noisy', 'noise', 'ref', 'reference', 'lb', 'lower_bound', 'ub', 'upper_bound')):
            arg, container = arg.rsplit('_', 1)
        else:
            container = None

        if container is None or container == 'noisy':
            for value in self._data.values():
                if arg in value:
                    if not value.is_empty():
                        if k is None:
                            k = slice(0, value.shape[1])
                        index = value.index(arg)
                        if container == 'noisy':
                            id_ = self.get_id(arg)
                            return value[index, k] + self._noise[id_][index, k]
                        else:
                            return value[index, k]
                    else:
                        return value.values
        return None

    def get_by_name(self, *args: Sequence[str]) -> Optional[Union[ca.DM, list[Optional[ca.DM]]]]:
        """

        :param args:
        :return:
        """
        if len(args) == 1:
            return self._get_by_name(args[0])
        return [self._get_by_name(arg) for arg in args]

    def get_id(self, name: str) -> str:
        """

        :param name:
        :return:
        """
        for key, value in self._data.items():
            if name in value:
                return key

    def get_ref_by_name(self, name):
        """

        :param name:
        :return:
        """
        # TODO: Update according to get_by_name()?
        for value in self._reference.values():
            if name in value:
                if not value.is_empty():
                    index = value.index(name)
                    val = value[index, :]
                    if not np.isnan(val).all():
                        return val
                else:
                    return value.values
        return None

    def get_lb_by_name(self, name):
        """

        :param name:
        :return:
        """
        # TODO: Update according to get_by_name()?
        for value in self._lower_bound.values():
            if name in value:
                if not value.is_empty():
                    index = value.index(name)
                    val = value[index, :]
                    if not np.isnan(val).all():
                        return val
                else:
                    return value.values
        return None

    def get_ub_by_name(self, name):
        """

        :param name:
        :return:
        """
        # TODO: Update according to get_by_name()?
        for value in self._upper_bound.values():
            if name in value:
                if not value.is_empty():
                    index = value.index(name)
                    val = value[index, :]
                    if not np.isnan(val).all():
                        return val
                else:
                    return value.values
        return None

    def get_description(self, arg):
        """

        :param arg:
        :return:
        """
        if arg in self._names:
            for value in self._data.values():
                if arg in value:
                    index = value.index(arg)
                    # NOTE: index should always be a list of length 1
                    return value.description[index[0]]
        else:
            return self._data[arg].description

    def get_function_args(self, **kwargs):
        """

        :param kwargs:
        :return:
        """
        args = {}
        if 'x' in self._data:
            args['x0'] = self._data['x'][:, -1]
        if 'p' in self._data:
            args['p'] = self._data['p'][:, -1]
        return args

    def get_names(self, *args, **kwargs):
        """

        :param args:
        :param kwargs:
        :return:
        """
        if not args:
            return self._names
        else:
            names = []
            ignore = kwargs.get('ignore', None)
            if ignore is None:
                ignore = False
            for arg in args:
                if ignore:
                    if arg in self:
                        names.extend(self._data[arg].names)
                else:
                    names.extend(self._data[arg].names)
            return names

    def get_labels(self, arg):
        """

        :param arg:
        :return:
        """
        if arg in self._names:
            for value in self._data.values():
                if arg in value:
                    index = value.index(arg)
                    # NOTE: index should always be a list of length 1
                    return value.labels[index[0]]
        else:
            return self._data[arg].labels

    def get_units(self, arg):
        """

        :param arg:
        :return:
        """
        if arg in self._names:
            for value in self._data.values():
                if arg in value:
                    index = value.index(arg)
                    # NOTE: index should always be a list of length 1
                    return value.units[index[0]]
        else:
            return self._data[arg].units

    def is_empty(self) -> bool:
        """

        :return:
        """
        return all(data.is_empty() for data in self._data.values())

    def is_set_up(self) -> bool:
        """

        :return:
        """
        return bool(self._data)

    def items(self):
        """

        :return:
        """
        yield from self._data.items()

    def make_some_noise(self, *args, distribution='normal', inplace=True, seed=None, **kwargs):
        """

        :param args:
        :param distribution:
        :param inplace:
        :param seed:
        :param kwargs:
        :return:
        """
        # TODO: Add support for changing variance (over time)?
        if seed is not None:
            np.random.seed(seed)

        if not is_list_like(distribution):
            distribution = len(args) * [distribution]

        first_or_last = []
        for arg in args:
            arg = arg.split(':')
            if len(arg) == 2:
                bc = arg[1]
                if bc not in ['0', 'f']:
                    raise ValueError(f"Unsupported appendage '{bc}'. Use '0' to indicate initial values and 'f' to "
                                     f"indicate final values.")
                else:
                    first_or_last.append(True)
            else:
                first_or_last.append(False)
        if not all(first_or_last):
            if any(first_or_last):
                # TODO: Not a very informative error message. The actual problem is mixing variable identifiers
                #  (x, y, z, u, p) with true variable names.
                raise ValueError("Mixing of first and last values and vectors not allowed")
            sol = None
            if not inplace:
                sol = self.get_by_name(*args)
        else:
            # TODO: Should we also distinguish between first and last
            sol = [self.get_by_id(arg) for arg in args]
            # NOTE: arg[:-2] because the appendages '0' (via ':0') and 'f' (via ':f') take a way 2 characters
            args = tuple(arg[:-2] for arg in args)

        noise_to_add = {key: np.zeros_like(value.values) if not all(first_or_last) else np.zeros((value.size1())) for
                        key, value in self._data.items()}

        for k, arg in enumerate(args):
            if distribution[k] == 'normal':
                if not all(first_or_last):
                    if inplace:
                        id_ = self.get_id(arg)
                        index = self._data[id_].index(arg)
                        shape = self._data[id_][index, :].shape
                        _, std = _process_noise_inputs_normal(arg, shape, **kwargs)
                        noise = std * np.random.randn(*shape)
                        noise_to_add[id_][index, :] = noise
                    else:
                        shape = sol[k].shape
                        _, std = _process_noise_inputs_normal(arg, shape, **kwargs)
                        noise = std * np.random.randn(*shape)
                        sol[k] += noise
                else:
                    shape = sol[k].shape
                    _, std = _process_noise_inputs_normal(arg, shape, **kwargs)
                    noise = std * np.random.randn(*shape)
                    sol[k] += noise
                    if inplace:
                        noise_to_add[arg] = noise

        if inplace:
            for key, value in noise_to_add.items():
                self.add(key + '_noise', value)

        return sol

    def merge(self, other):
        """

        :param other:
        :return:
        """
        # TODO: Compare sampling time and grid (-> throw error?)
        if self.is_empty():
            if self._data:
                raise NotImplementedError(f"Merging has not been implemented yet for the case where the "
                                          f"{self.__class__.__name__} object has already been set up.")
            else:
                kwargs = {key: {
                    'data_format': ca.DM,
                    'description': value.description,
                    'labels': value.labels,
                    'units': value.units,
                    'shape': (value.shape[0], 0),
                    'values_or_names': value.names
                } for key, value in other.items()}
                self._update_kwargs(other, kwargs)
                self.setup(*kwargs.keys(), **kwargs)
        else:
            raise NotImplementedError(f"Merging has not been implemented yet for the case where the "
                                      f"{self.__class__.__name__} object has already been populated.")

    def plot(self, *args, **kwargs):
        """

        :param args:
        :param kwargs:
        :return:
        """
        # TODO: Too convoluted. Maybe create a SeriesPlotter class that is called here?
        # TODO: Add support for labels if labels or units are lists
        # TODO: Add support for user-supplied legend labels
        # TODO: Add support for user-supplied colors
        ext_data = kwargs.pop('data', None)
        ext_x_data = kwargs.pop('x_data', None)
        ext_y_data = kwargs.pop('y_data', None)
        kind = kwargs.pop('kind', None)
        marker = kwargs.pop('marker', None)
        marker_size = kwargs.pop('marker_size', None)
        shape = kwargs.pop('shape', None)

        if not args:
            abscissas = self._data[self._abscissa].names
            args = [(abscissa, var) for abscissa in abscissas for var in self._names if var not in abscissas]
        n = len(args)

        ref_data = {}
        lb_data = {}
        ub_data = {}
        ext_data_suffix = kwargs.pop('data_suffix', None)
        if ext_data_suffix is None:
            ext_data_suffix = n * ['ext']
        elif isinstance(ext_data_suffix, str):
            ext_data_suffix = n * [ext_data_suffix]
        ext_data_skip = kwargs.pop('data_skip', None)
        if ext_data_skip is None:
            ext_data_skip = []
        elif isinstance(ext_data_skip, int):
            ext_data_skip = [ext_data_skip]
        for k, arg in enumerate(args):
            for ref in self._reference.values():
                if not ref.is_empty():
                    for idx, val in enumerate(ref.names):
                        if val in arg[1:] and ref[idx, :].is_regular():
                            ref_data[val + '_ref'] = {
                                'data': ref[idx, :],
                                'kind': 'dashed',
                                'color': 'k',
                                'subplot': k
                            }
            for lb in self._lower_bound.values():
                if not lb.is_empty():
                    for idx, val in enumerate(lb.names):
                        if val in arg[1:] and lb[idx, :].is_regular():
                            lb_data[val + '_lb'] = {
                                'data': lb[idx, :],
                                'kind': 'dashed',
                                'subplot': k
                            }
            for ub in self._upper_bound.values():
                if not ub.is_empty():
                    for idx, val in enumerate(ub.names):
                        if val in arg[1:] and ub[idx, :].is_regular():
                            ub_data[val + '_ub'] = {
                                'data': ub[idx, :],
                                'kind': 'dashed',
                                'subplot': k
                            }
            if ext_data is not None and isinstance(ext_data, Series) and k not in ext_data_skip:
                if ext_x_data is None:
                    ext_x_data = []
                if ext_y_data is None:
                    ext_y_data = []
                x = ext_data.get_by_name(arg[0])
                for _arg in arg[1:]:
                    idx = ext_data.get_id(_arg)
                    if idx is not None:
                        # TODO: This is a workaround for when noisy data is to be plotted, since noise does not
                        #  necessarily exist in the supplied Series object. There is probably a better way to deal with
                        #  this.
                        val = ext_data.get_by_name(_arg)
                        if not val.is_empty():
                            if idx in ['u', 'p']:
                                n_y = val.size2()
                                n_x = x.size2()
                                if n_y != n_x:
                                    # NOTE: u should not have more entries than t
                                    val_f = val[0, -1]
                                    val = ca.horzcat(val, val_f * ca.DM.ones(1, n_x - n_y))
                            ext_x_data.append({
                                'data': x,
                                'subplot': k,
                                'label': _arg + '_' + ext_data_suffix[k]
                            })
                            ext_y_data.append({
                                'data': val,
                                'subplot': k,
                                'label': _arg + '_' + ext_data_suffix[k],
                                'kind': 'line' if idx != 'u' and idx != 'p' else 'step'
                            })
                        ref = ext_data.get_ref_by_name(_arg)
                        if not ref.is_empty():
                            ext_x_data.append({
                                'data': x,
                                'subplot': k,
                                'label': _arg + '_ref_' + ext_data_suffix[k]
                            })
                            ext_y_data.append({
                                'data': ref,
                                'subplot': k,
                                'label': _arg + '_ref_' + ext_data_suffix[k],
                                'kind': 'dashed'
                            })
                        lb = ext_data.get_lb_by_name(_arg)
                        if not lb.is_empty():
                            ext_x_data.append({
                                'data': x,
                                'subplot': k,
                                'label': _arg + '_lb_' + ext_data_suffix[k]
                            })
                            ext_y_data.append({
                                'data': lb,
                                'subplot': k,
                                'label': _arg + '_lb_' + ext_data_suffix[k],
                                'kind': 'dashed'
                            })
                        ub = ext_data.get_ub_by_name(_arg)
                        if not ub.is_empty():
                            ext_x_data.append({
                                'data': x,
                                'subplot': k,
                                'label': _arg + '_ub_' + ext_data_suffix[k]
                            })
                            ext_y_data.append({
                                'data': ub,
                                'subplot': k,
                                'label': _arg + '_ub_' + ext_data_suffix[k],
                                'kind': 'dashed'
                            })

        if ext_x_data is not None:
            n_x = len(ext_x_data)
        else:
            n_x = 0
        if ext_y_data is not None:
            n_y = len(ext_y_data)
        else:
            n_y = 0

        if n_x > 0:
            if n_y > 0:
                if n_x > n_y or (n_x > 1 and n_x != n_y):
                    raise ValueError(f"{n_x} datasets for the x-axis supplied, but only {n_y} datasets for the "
                                     f"y-axis.\nTo plot external data, make sure that the number of datasets for "
                                     f"the x-axis is either equal to the number of datasets for the y-axis or equal"
                                     f" to 1.\nAlso make sure, that the dictionaries for the datasets for the x-axis "
                                     f"and the dictionaries for the datasets for the y-axis share the corresponding "
                                     f"labels, when both have the same amount of datasets.\nIf no dataset for the "
                                     f"x-axis is supplied, the stored dataset will be used.")
            else:
                warnings.warn("Datasets for the x-axis were supplied, but no dataset for the y-axis. Ignoring supplied "
                              "external datasets.")

        data = {k: {'x': {}, 'y': {}} for k in range(n)}
        xlabel = n * [None]
        ylabel = n * [None]
        legend = n * [False]
        if kind is not None:
            kind_supplied = True
        else:
            kind_supplied = False
            kind = n * [None]
        if marker is not None:
            marker_supplied = True
        else:
            marker_supplied = False
            marker = n * [None]
        if marker_size is not None:
            marker_size_supplied = True
        else:
            marker_size_supplied = False
            marker_size = n * [None]
        # color = n * [None]

        def update_marker(
                subplot: int,
                m: Optional[str] = None,
                ms: Optional[int] = None,
                m_container: Optional[list[str]] = None,
                ms_container: Optional[list[int]] = None
        ) -> None:
            """

            :param subplot:
            :param m:
            :param ms:
            :param m_container:
            :param ms_container:
            :return:
            """
            if m is not None:
                if m_container is not None:
                    m_container.append(m)
                    ms_container.append(ms)
                else:
                    if isinstance(marker[subplot], list):
                        marker[subplot].append(m)
                    else:
                        if marker[subplot] is not None:
                            marker[subplot] = [marker[subplot], m]
                        else:
                            marker[subplot] = [m]
                    if isinstance(marker_size[subplot], list):
                        marker_size[subplot].append(ms)
                    else:
                        if marker_size[subplot] is not None:
                            marker_size[subplot] = [marker_size[subplot], ms]
                        else:
                            marker_size[subplot] = [ms]

        duplicate_y_counter = 0
        step_mode = None
        for sub, arg in enumerate(args):
            x = arg[0]
            y = arg[1:]

            if not isinstance(x, str):
                raise TypeError(f"Wrong type {type(x).__name__} for argument 'x'")
            if not isinstance(y, list):
                if isinstance(y, (tuple, set)):
                    y = list(y)
                elif isinstance(y, np.ndarray):
                    y = y.tolist()
                else:
                    y = [y]
            if not all(isinstance(k, str) for k in y):
                raise TypeError(f"Wrong type {(type(k).__name__ for k in y)} for argument 'y'")

            label = self.get_labels(x)
            if not label:
                label = x
            units = self.get_units(x)
            x = self.get_by_name(x)
            if x is not None:
                x = x.full().flatten()
                if label:
                    xlabel[sub] = f"{label}"
                    if units:
                        xlabel[sub] += f" in [{units}]"

            ext_data = {}
            if n_y > 0:
                keys = []
                for ext in ext_y_data:
                    if 'subplot' in ext and ext['subplot'] == sub:
                        key = ext['label']
                        keys.append(key)
                        ext_data[key] = ext
                    if 'kind' in ext and isinstance(kind, str):
                        kind = n * [kind]
                    if 'marker' in ext and isinstance(marker, str):
                        marker = n * [marker]
                    if 'marker_size' in ext and isinstance(marker_size, int):
                        marker_size = n * [marker_size]

                if ext_data:
                    y.append(ext_data)

                ext_data = {}
                if n_x > 0:
                    ext_data = {k.get('label'): k.get('data') for k in ext_x_data if
                                k.get('label') in keys and k.get('subplot') == sub}

            if ref_data or lb_data or ub_data:
                add_data = {}
                for key, val in ref_data.items():
                    if 'subplot' in val and val['subplot'] == sub:
                        add_data[key] = val
                for key, val in lb_data.items():
                    if 'subplot' in val and val['subplot'] == sub:
                        add_data[key] = val
                for key, val in ub_data.items():
                    if 'subplot' in val and val['subplot'] == sub:
                        add_data[key] = val
                if add_data:
                    y.append(add_data)

            kind_k = []
            marker_k = []
            marker_size_k = []
            # color_k = []
            n_yy = len(y)
            for k in y:
                if isinstance(k, str) and k.endswith('_noisy'):
                    k, appendix = k.rsplit('_', 1)
                    appendix = '_' + appendix
                else:
                    appendix = ''
                if isinstance(k, str):
                    yk = self.get_by_name(k + appendix)
                    if yk is not None:
                        if yk.size1() == 1:
                            if ('u' in self._data and k in self._data['u']) or (
                                    'p' in self._data and k in self._data['p']):
                                n_up = yk.size2()
                                n_xx = x.size
                                if n_up != n_xx:
                                    # NOTE: u should not have more entries than t
                                    if n_up > 0:
                                        ykf = yk[0, -1]
                                        yk = ca.horzcat(yk, ykf * ca.DM.ones(1, n_xx - n_up))
                                    else:
                                        yk = ca.horzcat(yk, ca.DM.nan(1, n_xx))
                                if not kind_supplied:
                                    kind_k.append('step')
                            else:
                                if not kind_supplied:
                                    kind_k.append('line')
                            data[sub]['x'][k + appendix] = x
                            data[sub]['y'][k + appendix] = yk.full().flatten()
                        elif yk.size1() == 0:
                            if not kind_supplied:
                                if ('u' in self._data and k in self._data['u']) or (
                                        'p' in self._data and k in self._data['p']):
                                    kind_k.append('step')
                                else:
                                    kind_k.append('line')
                            n_xx = x.size
                            data[sub]['x'][k + appendix] = x
                            data[sub]['y'][k + appendix] = np.full(n_xx, np.nan)
                        else:
                            # NOTE: Duplicate y names. Preliminary! Kind of a dirty hack. Right now only used in GP
                            #  regression plots.
                            ykk = yk[duplicate_y_counter, :]
                            duplicate_y_counter += 1
                            if ykk.size1() == 1:
                                if ('u' in self._data and k in self._data['u']) or (
                                        'p' in self._data and k in self._data['p']):
                                    n_up = ykk.size2()
                                    n_xx = x.size
                                    if n_up != n_xx:
                                        # NOTE: u should not have more entries than t
                                        ykkf = ykk[0, -1]
                                        ykk = ca.horzcat(ykk, ykkf * ca.DM.ones(1, n_xx - n_up))
                                    if not kind_supplied:
                                        kind_k.append('step')
                                else:
                                    if not kind_supplied:
                                        kind_k.append('line')
                                data[sub]['x'][k + appendix] = x
                                data[sub]['y'][k + appendix] = ykk.full().flatten()
                            elif ykk.size1() == 0:
                                if not kind_supplied:
                                    if ('u' in self._data and k in self._data['u']) or (
                                            'p' in self._data and k in self._data['p']):
                                        kind_k.append('step')
                                    else:
                                        kind_k.append('line')
                                n_xx = x.size
                                data[sub]['x'][k + appendix] = x
                                data[sub]['y'][k + appendix] = np.full(n_xx, np.nan)
                            else:
                                raise RuntimeError("How did I get here?")

                        label = self.get_labels(k)
                        if label:
                            if ylabel[sub] is None:
                                ylabel[sub] = f"{label}"
                                units = self.get_units(k)
                                if units:
                                    ylabel[sub] += f" in [{units}]"
                            else:
                                units = self.get_units(k)
                                if units:
                                    label += f" in [{units}]"
                                if label != ylabel[sub]:
                                    ylabel[sub] = [ylabel[sub], label]
                        elif n_yy == 1:
                            # NOTE: This is only accessed when we have one object to plot
                            ylabel[sub] = k
                            units = self.get_units(k)
                            if units:
                                ylabel[sub] += f" in [{units}]"
                elif isinstance(k, dict):
                    # NOTE: This is for external data
                    # TODO: How to deal with xlabel and ylabel here?
                    for key, value in k.items():
                        if not kind_supplied:
                            if 'kind' in value:
                                kind_k.append(value['kind'])
                                if value['kind'] == 'scatter':
                                    update_marker(0, m=value.get('marker'), ms=value.get('marker_size'),
                                                  m_container=marker_k, ms_container=marker_size_k)
                            else:
                                kind_k.append('line')
                        else:
                            if isinstance(kind[sub], list):
                                if 'kind' in value:
                                    kind[sub].append(value['kind'])
                                    if value['kind'] == 'scatter':
                                        update_marker(sub, m=value['marker'], ms=value['marker_size'])
                                else:
                                    kind[sub].append('line')
                            else:
                                if kind[sub] is not None:
                                    if 'kind' in value:
                                        kind[sub] = [kind[sub], value['kind']]
                                        if value['kind'] == 'scatter':
                                            update_marker(sub, m=value['marker'], ms=value['marker_size'])
                                    else:
                                        kind[sub] = [kind[sub], 'line']
                                else:
                                    if 'kind' in value:
                                        kind[sub] = [value['kind']]
                                        if value['kind'] == 'scatter':
                                            update_marker(sub, m=value['marker'], ms=value['marker_size'])
                                    else:
                                        kind[sub] = ['line']
                        y_data = _clean_external_data(value['data'])
                        if key in ext_data:
                            x_data = ext_data[key]
                            x_data = _clean_external_data(x_data)
                            data[sub]['x'][key] = x_data
                        else:
                            data[sub]['x'][key] = x
                        data[sub]['y'][key] = y_data

            if n_yy > 1:
                legend[sub] = True
            if not kind_supplied and kind[sub] is None:
                kind[sub] = kind_k
            if 'step' in kind_k and step_mode is None:
                if self._plot_manager.backend == 'bokeh':
                    step_mode = 'after'
                elif self._plot_manager.backend == 'matplotlib':
                    step_mode = 'post'
            if not marker_supplied and marker[sub] is None:
                marker[sub] = marker_k
            if not marker_size_supplied and marker_size[sub] is None:
                marker_size[sub] = marker_size_k

        if not kind_supplied:
            # TODO: Something with collections.Counter (see https://stackoverflow.com/a/9623147)
            # raise NotImplementedError
            pass
        if step_mode is not None:
            kwargs['step_mode'] = step_mode
        if (is_list_like(marker) and any(m is not None for m in marker)) or isinstance(marker, str):
            kwargs['marker'] = marker
        if (is_list_like(marker_size) and any(ms is not None for ms in marker_size)) or isinstance(marker_size, int):
            kwargs['marker_size'] = marker_size

        if n > 1 and 'subplots' not in kwargs:
            kwargs['subplots'] = True
            if shape is not None:
                kwargs['layout'] = shape
        if 'xlabel' not in kwargs:
            kwargs['xlabel'] = xlabel
        if 'ylabel' not in kwargs:
            kwargs['ylabel'] = ylabel
        if 'legend' not in kwargs:
            kwargs['legend'] = legend
        if 'title' not in kwargs:
            if n > 1:
                kwargs['title'] = n * ["Such a nice plot"]
            else:
                kwargs['title'] = "Such a nice plot"
        if 'fill_between' in kwargs:
            fill_between = kwargs['fill_between']
            if isinstance(fill_between, (list, set, tuple)):
                for filler in fill_between:
                    x_fill = filler.get('x')
                    if x_fill is None:
                        raise KeyError("The fill_between keyword argument was supplied without x-data")
                    elif isinstance(x_fill, str):
                        x = self.get_by_name(x_fill).full().flatten()
                    else:
                        x = _clean_external_data(x_fill)
                    filler['x'] = x
            else:
                x_fill = fill_between.get('x')
                if x_fill is None:
                    raise KeyError("The fill_between keyword argument was supplied without x-data")
                elif isinstance(x_fill, str):
                    x = self.get_by_name(x_fill).full().flatten()
                else:
                    x = _clean_external_data(x_fill)
                fill_between['x'] = x
        if 'interactive' in kwargs and kwargs['interactive'] and self._plot_manager.backend == 'latex':
            warnings.warn("Interactive plotting using the backend 'latex' is not supported. Switching 'interactive' to"
                          " False...")
            kwargs['interactive'] = False

        return self._plot_manager.plot(data, kind=kind, **kwargs)

    def remove(
            self,
            arg: str,
            index: Union[int, Sequence[int], slice],
            skip: Optional[Union[str, Sequence[str]]] = None
    ) -> None:
        """

        :param arg:
        :param index:
        :param skip:
        :return:
        """
        if skip is not None:
            if isinstance(skip, str):
                skip = [skip]
        else:
            skip = []

        if 'data' not in skip:
            if arg in self._data:
                if not self._data[arg].is_empty():
                    self._data[arg].remove(index, axis=1)
            else:
                if self._data:
                    raise KeyError(f"Argument '{arg}' not found in data container.")
                else:
                    raise KeyError("Data container is empty")
        if 'ref' not in skip and 'reference' not in skip:
            if arg in self._reference:
                if not self._reference[arg].is_empty():
                    self._reference[arg].remove(index, axis=1)
        if 'lb' not in skip and 'lower_bound' not in skip:
            if arg in self._lower_bound:
                if not self._lower_bound[arg].is_empty():
                    self._lower_bound[arg].remove(index, axis=1)
        if 'ub' not in skip and 'upper_bound' not in skip:
            if arg in self._upper_bound:
                if not self._upper_bound[arg].is_empty():
                    self._upper_bound[arg].remove(index, axis=1)
        if 'noise' not in skip:
            if arg in self._noise:
                if not self._noise[arg].is_empty():
                    self._noise[arg].remove(index, axis=1)

    def set(self, arg, value):
        """

        :param arg:
        :param value:
        :return:
        """
        # TODO: Add pandas.Series and pandas.DataFrame
        allowed_types = (int, float, list, tuple, np.ndarray, ca.DM, Vector)
        to_be_listed = (int, float)

        arg = arg.split(':')
        if len(arg) == 1:
            ic = ''
        elif len(arg) == 2:
            ic = arg[1]
            if ic not in ['0', 'f']:
                raise ValueError(f"Unsupported appendage '{ic}'. Use '0' to indicate initial values.")
            if ic != '0':
                warnings.warn("You supplied the appendage 'f'. This is not supported for the set method.")
        else:
            raise ValueError(f"Unsupported key {'_'.join(arg)}")
        arg = arg[0]

        if isinstance(value, Vector):
            if not value.is_constant():
                raise TypeError("Input arguments of class Vector containing symbolic expressions are not yet "
                                "supported.")

        if isinstance(value, dict):
            old_value = value
            value = len(old_value) * [None]
            for key, val in old_value.items():
                index = self._data[arg].index(key)[0]
                value[index] = val

        arg = arg.rsplit('_')
        if len(arg) == 1:
            index = ''
            arg = arg[0]
        elif len(arg) == 2:
            index = arg[1]
            arg = arg[0]
        else:
            raise ValueError(f"Unsupported key {'_'.join(arg)}")

        # TODO: Maybe we can combine the first two conditions
        if arg in self._data and not ic and not index:
            if isinstance(value, allowed_types):
                if isinstance(value, to_be_listed):
                    self._data[arg].set([value])
                else:
                    self._data[arg].set(value)
            else:
                raise TypeError("Expected argument 'value' of type {}, got {} instead."
                                .format([type(k) for k in allowed_types], type(value)))
        elif arg in self._data and ic == '0' and not index:
            if isinstance(value, allowed_types):
                if self._data[arg].is_empty() or self._data[arg].size2() == 1:
                    if isinstance(value, to_be_listed):
                        self._data[arg].set([value])
                    else:
                        self._data[arg].set(value)
                else:
                    # TODO: Update initial value in this case
                    # NOTE: I'm not sure whether it could make sense to be able to do this. Also the same could be
                    #  done for the final value or for any other value actually
                    warnings.warn("Vector is not empty. No changes applied.", UserWarning)
            else:
                raise TypeError(f"Expected argument 'value' of type {[type(k) for k in allowed_types]}, "
                                f"got {type(value)} instead.")
        elif index in ['ref', 'reference']:
            if isinstance(value, allowed_types):
                if isinstance(value, to_be_listed):
                    self._reference[arg].set([value], axis=1)
                else:
                    self._reference[arg].set(value, axis=1)
            else:
                raise TypeError(f"Expected argument 'value' of type {[type(k).__name__ for k in allowed_types]}, "
                                f"got {type(value)} instead.")
        elif index in ['lb', 'lower_bound']:
            if isinstance(value, allowed_types):
                if isinstance(value, to_be_listed):
                    self._lower_bound[arg].set([value], axis=1)
                else:
                    self._lower_bound[arg].set(value, axis=1)
            else:
                raise TypeError(f"Expected argument 'value' of type {[type(k).__name__ for k in allowed_types]}, "
                                f"got {type(value)} instead.")
        elif index in ['ub', 'upper_bound']:
            if isinstance(value, allowed_types):
                if isinstance(value, to_be_listed):
                    self._upper_bound[arg].set([value], axis=1)
                else:
                    self._upper_bound[arg].set(value, axis=1)
            else:
                raise TypeError(f"Expected argument 'value' of type {[type(k).__name__ for k in allowed_types]}, "
                                f"got {type(value)} instead.")
        elif index == 'noise':
            if isinstance(value, allowed_types):
                if isinstance(value, to_be_listed):
                    self._noise[arg].set([value], axis=1)
                else:
                    self._noise[arg].set(value, axis=1)
            else:
                raise TypeError(f"Expected argument 'value' of type {[type(k).__name__ for k in allowed_types]}, "
                                f"got {type(value)} instead.")
        else:
            if self._data:
                raise KeyError("Argument '{}' not found in data container.")
            else:
                raise KeyError("Data container is empty")

    def setup(self, *args, **kwargs):
        """

        :param args:
        :param kwargs:
        :return:
        """
        if len(args) == 1 and isinstance(args[0], self.__class__):
            self.merge(args[0])
        else:
            for arg in args:
                self._data[arg] = Vector(**kwargs[arg], parent=self)
                self._reference[arg] = Vector(**kwargs[arg], parent=self)
                self._lower_bound[arg] = Vector(**kwargs[arg], parent=self)
                self._upper_bound[arg] = Vector(**kwargs[arg], parent=self)
                self._noise[arg] = Vector(**kwargs[arg], parent=self)
                self._names.extend(self._data[arg].names)

    def to_dict(self, *args, **kwargs):
        """

        :param args:
        :param kwargs:
        :return:
        """
        subplots = kwargs.get('subplots', False)
        if subplots:
            plot = 0
        suffix = kwargs.get('suffix', None)
        if suffix is not None:
            suffix = '_' + suffix
        else:
            suffix = ''
        index = kwargs.get('index', None)

        data = {}
        duplicate_counter = {}
        for arg in args:
            current_counter = ''
            if args.count(arg) > 1:
                if arg not in duplicate_counter:
                    duplicate_counter[arg] = 1
                else:
                    duplicate_counter[arg] += 1
                current_counter += f'_{duplicate_counter[arg]}'

            if arg.endswith('_noisy'):
                arg, container = arg.rsplit('_', 1)
                container = '_' + container
            else:
                container = ''

            if arg in self._names:
                arg_data = self.get_by_name(arg + container).full()
                if index is not None:
                    arg_data = arg_data[:, index]
                if subplots:
                    data[arg + current_counter + container + suffix] = {'data': arg_data}
                    data[arg + current_counter + container + suffix]['subplot'] = plot
                    plot += 1
                else:
                    data[arg + current_counter + container + suffix] = arg_data
            elif arg in self._data:
                for name in self._data[arg].names:
                    arg_data = self.get_by_name(name).full()
                    if index is not None:
                        arg_data = arg_data[:, index]
                    if subplots:
                        data[name + current_counter + container + suffix] = {'data': arg_data}
                        data[name + current_counter + container + suffix]['subplot'] = plot
                        plot += 1
                    else:
                        data[name + current_counter + container + suffix] = arg_data
            else:
                msg = f"Argument '{arg}' not recognized"
                raise ValueError(msg)

        return data

    def to_mat(self, *args, **kwargs):
        """
        Saves the solution to .mat file. Must take as an argument a string. Pass to the keyword argument 'file_name'
        the name of the file. For example model.solution.to_mat('x','file_name'="results/file.mat")

        :param args:
        :param kwargs:
        :return:
        """
        file_name = kwargs.pop("file_name", "data.mat")
        data = self.to_dict(*args, **kwargs)
        save_mat(file_name, data)

    def update(self, **kwargs):
        """

        :param kwargs:
        :return:
        """
        for arg, value in kwargs.items():
            self.add(arg, value)


class TimeSeries(Series):
    """"""
    # TODO: Typing hints
    def __init__(self, backend=None, id=None, name=None, parent=None, **kwargs):
        """Constructor method"""
        super().__init__(backend=backend, id=id, name=name, parent=parent)

        self._n_x = 0
        self._n_y = 0
        self._n_z = 0
        self._n_u = 0
        self._n_p = 0
        self._n_t = 0

        self._dt = None
        self._grid = None

        self._abscissa = 't'

    def __del__(self):
        """Deletion method"""
        super().__del__()
        self._dt = None
        self._grid = None

    def _check_samples(self) -> None:
        """

        :return:
        """
        if 't' in self._data:
            self._n_samples = self._data['t'].shape[1]
        elif 'x' in self._data:
            self._n_samples = self._data['x'].shape[1]
        elif 'y' in self._data:
            self._n_samples = self._data['y'].shape[1]
        elif 'z' in self._data:
            self._n_samples = self._data['z'].shape[1]

    def _update_dimensions(self):
        """

        :return:
        """
        for k in ['x', 'y', 'z', 'u', 'p', 't']:
            if k in self._data:
                setattr(self, '_n_' + k, self._data[k].size1())

    def _update_kwargs(self, other, kwargs):
        """

        :param other:
        :param kwargs:
        :return:
        """
        other_has_dt = hasattr(other, 'dt')
        other_has_grid = hasattr(other, 'grid')

        if self._dt is not None:
            if other_has_dt and other.dt is not None and self._dt != other.dt:
                warnings.warn(f"Sampling time 'dt' is different between {self.name} (dt={self._dt}) and {other.name} "
                              f"(dt={other.dt}). Choosing dt={self._dt}.")
            elif not other_has_dt:
                warnings.warn(f"{other.name} doesn't have the attribute sampling time 'dt'. Choosing dt={self._dt} from"
                              f" {self.name}.")
            elif other.dt is None:
                warnings.warn(f"Sampling time 'dt' is not set for {other.name}. Choosing dt={self._dt} from "
                              f"{self.name}.")
            kwargs['dt'] = self._dt
        elif self._grid is not None:
            if other_has_grid and other.grid is not None and not ca.is_equal(self._grid, other.grid):
                warnings.warn(f"Grid is different between {self.name} and {other.name}."
                              f"\n{self.name}: {self._grid}\n{other.name}: {other.grid}\n"
                              f"Choosing grid from {self.name}.")
            elif not other_has_grid:
                warnings.warn(f"{other.name} doesn't have the attribute grid. Choosing grid from {self.name}.")
            elif other.grid is None:
                warnings.warn(f"Grid is not set for {other.name}. Choosing grid from {self.name}.")
            kwargs['grid'] = self._grid
        elif other_has_dt and other.dt is not None:
            kwargs['dt'] = other.dt
        elif other_has_grid and other.grid is not None:
            kwargs['grid'] = other.grid

    @property
    def dt(self) -> Optional[Union[int, float]]:
        """

        :return:
        """
        return self._dt

    @property
    def grid(self) -> Optional[ca.DM]:
        """

        :return:
        """
        return self._grid

    def copy(self):
        """

        :return:
        """
        new = super().copy()
        if self._dt is not None:
            new.setup('dt', dt=self._dt)
        elif self._grid is not None:
            new.setup('grid', grid=self._grid)
        return new

    def get_function_args(self, **kwargs):
        """

        :param kwargs:
        :return:
        """
        steps = kwargs.get('steps', 1)
        skip = kwargs.get('skip', None)
        if skip is not None:
            if not isinstance(skip, (list, tuple)):
                skip = [skip]
        else:
            skip = []

        if steps == 1:
            args = {}
            if 't' in self._data and 't' not in skip:
                args['t0'] = self._data['t'][:, -1]
            if 'x' in self._data and 'x' not in skip:
                args['x0'] = self._data['x'][:, -1]
            if 'z' in self._data and 'z' not in skip:
                args['z0'] = self._data['z'][:, -1]
            if 'u' in self._data and 'u' not in skip:
                if 'p' in self._data and 'p' not in skip:
                    args['p'] = ca.vertcat(self._data['u'][:, -1], self._data['p'][:, -1])
                else:
                    args['p'] = self._data['u'][:, -1]
            else:
                if 'p' in self._data and 'p' not in skip:
                    args['p'] = self._data['p'][:, -1]
            return args
        elif steps > 1:
            args = {}
            if 't' in self._data and 't' not in skip:
                args['t0'] = self._data['t'][:, -1]
            if 'x' in self._data and 'x' not in skip:
                args['x0'] = self._data['x'][:, -1]
            if 'z' in self._data and 'z' not in skip:
                args['z0'] = ca.repmat(self._data['z'][:, -1], 1, steps)
            if 'u' in self._data and 'u' not in skip:
                if 'p' in self._data and 'p' not in skip:
                    args['p'] = ca.vertcat(self._data['u'][:, -steps:], self._data['p'][:, -steps:])
                else:
                    args['p'] = self._data['u'][:, -steps:]
            else:
                if 'p' in self._data and 'p' not in skip:
                    args['p'] = self._data['p'][:, -steps:]
            return args
        else:
            raise ValueError("The 'steps' argument has to be greater than 0.")

    def setup(self, *args, **kwargs):
        """

        :param args:
        :param kwargs:
        :return:
        """
        if 'dt' in args:
            self._dt = kwargs.pop('dt')
        elif 'grid' in args:
            self._grid = convert(kwargs.pop('grid'), ca.DM)
        args = tuple(k for k in args if k not in ['dt', 'grid'])
        super().setup(*args, **kwargs)


class OptimizationSeries(Series):
    """"""
    # TODO: Typing hints
    # TODO: Add objective value to data (and maybe g and lam_g / lam_x too)
    def __init__(self, backend=None, id=None, name=None, parent=None, **kwargs):
        """Constructor method"""
        super().__init__(backend=backend, id=id, name=name, parent=parent)

        self._n_x = 0
        self._n_p = 0

    def _update_dimensions(self):
        """

        :return:
        """
        for k in ['x', 'p']:
            if k in self._data:
                setattr(self, '_n_' + k, self._data[k].size1())


def _clean_external_data(array):
    """

    :param array:
    :return:
    """
    if isinstance(array, ca.DM):
        if not array.is_row():
            if array.is_column():
                array = array.T
            else:
                # Multiple rows -> We need to append multiple data lines (append index to name)
                raise NotImplementedError
        array = array.full().flatten()
    elif isinstance(array, (list, set, tuple)):
        # TODO: What if list of lists is supplied?
        if any(isinstance(k, (list, set, tuple)) for k in array):
            raise NotImplementedError
        array = np.array(array)
    elif isinstance(array, np.ndarray):
        # TODO: What about multi-dimensional arrays
        if array.ndim > 1:
            if 1 in array.shape:
                array = array.flatten()
            else:
                # Multiple rows -> We need to append multiple data lines (append index to name)
                raise NotImplementedError
    else:
        raise TypeError(f"Wrong type '{type(array).__name__}' for 'data' values.")

    return array


def _process_noise_inputs_normal(arg, shape, **kwargs):
    """

    :param arg:
    :param shape:
    :param kwargs:
    :return:
    """
    mean = kwargs.get('mean')
    if mean is None:
        mean = kwargs.get('mu')
    if isinstance(mean, dict):
        mean = mean.get(arg)
    if mean is None:
        mean = 0.

    std = kwargs.get('std')
    if std is None:
        std = kwargs.get('standard_deviation')
    if isinstance(std, dict):
        std = std.get(arg)
    if std is None:
        var = kwargs.get('var')
        if var is None:
            var = kwargs.get('variance')
        if isinstance(var, dict):
            var = var.get(arg)
        if var is None:
            std = 1.
        else:
            std = np.sqrt(var)

    n_dim = len(shape)
    if n_dim > 2:
        raise RuntimeError("Tensors are not supported at the moment")
    if n_dim == 1:
        raise RuntimeError("Dimension mismatch. Expected shape of length 2, got 1.")

    if std.ndim == 1:
        if n_dim == 2 and std.size != shape[0] * shape[1]:
            expected_size = shape[0] * shape[1] if n_dim == 2 else shape[0]
            raise ValueError(f"Dimension mismatch for supplied standard deviation/variance. Expected array of size "
                             f"{expected_size}, got {std.size}.")
        std = np.reshape(std, shape)
    elif std.ndim == 2:
        if n_dim == 2 and shape != std.shape:
            if shape[::-1] != std.shape:
                raise ValueError(f"Dimension mismatch for supplied standard deviation/variance. Expected "
                                 f"{shape[0]}x{shape[1]}, got {std.shape[0]}x{std.shape[1]}.")
            std = np.reshape(std, shape)
    elif std.ndim > 2:
        raise ValueError("Tensors are not supported at the moment")
    elif isinstance(std, (int, float)):
        std = np.reshape(std, shape)
    else:
        raise TypeError(f"Expected array-like argument, got {type(std).__name__}.")

    return mean, std
