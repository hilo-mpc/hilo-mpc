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

from collections.abc import KeysView
from copy import copy
import platform
from typing import Optional, Sequence
import warnings

import casadi as ca
import numpy as np

from .object import Object
from ..util.util import setup_warning, check_compiler, convert, dump_clean, lower_case, who_am_i, AOT, JIT

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
    # TODO: Add duplicate check for names
    # TODO: Typing hints
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
                    del self._description[k]
                    del self._labels[k]
                    del self._units[k]
        else:
            if self._axis == 0:
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

    def clear(self, axis: Optional[int] = None, index: Optional[Sequence[int]] = None) -> None:
        """

        :param axis:
        :param index:
        :return:
        """
        if axis is not None:
            self._axis = axis

        if index is None:
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
        else:
            self.__delitem__(index)

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

    def remove(self, values):
        """

        :param values:
        :return:
        """
        indices = self.index(values)
        self.__delitem__(indices)

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
