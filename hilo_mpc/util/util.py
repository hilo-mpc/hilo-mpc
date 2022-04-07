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

from collections import ValuesView
import functools
import platform
import sys
from typing import Any, Callable, Optional, TypeVar, cast
import warnings

import casadi as ca
import numpy as np

if platform.system() == 'Windows':
    from .windows import get_vcvars, WINDOWS_COMPILERS
elif platform.system() == 'Linux':
    from .unix import find_compiler, UNIX_COMPILERS


Function = TypeVar('Function', bound=Callable[..., Any])


JIT = ['jit', 'just-in-time']
AOT = ['aot', 'ahead-of-time']
TYPES = {
    ca.SX: ca.SX,
    ca.MX: ca.MX,
    ca.DM: ca.DM,
    np.ndarray: np.array
}


def setup_warning(function: Function) -> Function:
    """

    :param function:
    :return:
    """
    @functools.wraps(function)
    def wrapper(*args, **kwargs):
        """

        :param args:
        :param kwargs:
        :return:
        """
        if args[0].is_setup():
            warnings.warn(f"{args[0].__class__.__name__} {args[0].name} was already set up. Please run the setup "
                          f"method again to apply the changes and prevent strange behavior.")
        return function(*args, **kwargs)
    return cast(Function, wrapper)


def _get_shape(**kwargs):
    """

    :param kwargs:
    :return:
    """
    n_dim = kwargs.get('n_dim', None)
    shape = kwargs.get('shape', None)
    if n_dim is None and shape is None:
        shape = (1, 1)
    elif n_dim is not None:
        if shape is None:
            axis = kwargs.get('axis', None)
            if axis is None or axis == 0:
                shape = (n_dim, 1)
            elif axis == 1:
                shape = (1, n_dim)
            elif axis > 1:
                raise IndexError("Argument 'axis' is out of bounds for array of dimension 2")
            else:
                raise TypeError("Expected argument 'axis' of type {}, got {} instead".format(type(int), type(axis)))
        else:
            if isinstance(shape, (int, tuple)):
                if isinstance(shape, int):
                    axis = kwargs.get('axis', None)
                    if axis is None or axis == 0:
                        shape = (shape, 1)
                    elif axis == 1:
                        shape = (1, shape)
                    elif axis > 1:
                        raise IndexError("Argument 'axis' is out of bounds for array of dimension 2")
                    else:
                        raise TypeError("Expected argument 'axis' of type {}, got {} instead".format(type(int),
                                                                                                     type(axis)))
                elif isinstance(shape, tuple):
                    if len(shape) > 2:
                        raise IndexError("Argument 'shape' is out of bounds for array of dimension 2")
                warnings.warn("Argument 'ndim' ignored, since argument 'shape' was supplied.")
            else:
                raise TypeError("Expected argument 'shape' of type {} or {}, got {} instead"
                                .format(type(int), type(tuple), type(shape)))

    return shape


def check_compiler(method: str, compiler: str) -> (Optional[str], Optional[str], Optional[str]):
    """

    :param method:
    :param compiler:
    :return:
    """
    method = lower_case(method)
    compiler = lower_case(compiler)
    if method in JIT:
        if ca.Importer.has_plugin(compiler):
            if compiler == 'shell' and platform.system() == 'Windows':
                vcvars = get_vcvars()
                return method, compiler, vcvars
            else:
                return method, compiler, None
    elif method in AOT:
        if platform.system() == 'Linux' and compiler in UNIX_COMPILERS:
            if find_compiler(compiler):
                return method, compiler, None
        elif platform.system() == 'Windows' and compiler in WINDOWS_COMPILERS:
            vcvars = get_vcvars()
            if vcvars:
                return method, compiler, vcvars
    return None, None, None


def convert(obj, _type, **kwargs):
    """

    :param obj:
    :param _type:
    :param kwargs:
    :return:
    """
    # TODO: Convoluted. Simplify if possible.
    # TODO: Add support for dict?
    # TODO: Is there a nice way to convert SX to MX and vice versa?
    fx = TYPES[_type]
    if obj is None:
        return convert(fx(), _type, **kwargs)
    if isinstance(obj, _type):
        shape = _get_shape(**kwargs)
        # TODO: What if size is to be reduced to (1, 1)? Should throw an error. Would do nothing here.
        if shape != (1, 1):
            if _type in [ca.SX, ca.MX, ca.DM]:
                if obj.is_scalar():
                    return ca.repmat(obj, shape)
                elif obj.is_column() or obj.is_row():
                    if max(obj.shape) != max(shape):
                        raise IndexError("Dimension mismatch. Cannot reduce or increase size of the shape.")
                    if obj.shape == shape[::-1]:
                        return obj.T
                    else:
                        raise Exception("Don't know how I got here. Please inform the maintainer.")
                elif obj.is_empty():
                    return obj.reshape(shape)
            else:
                if obj.size == 1:
                    return np.tile(obj, shape)
                elif 1 in obj.shape:
                    if max(obj.shape) != max(shape):
                        raise IndexError("Dimension mismatch. Cannot reduce or increase size of the shape.")
                    if obj.shape == shape[::-1]:
                        return obj.T
                    else:
                        raise Exception("Don't know how I got here. Please inform the maintainer.")
        return obj
    elif isinstance(obj, (list, tuple, np.ndarray, ca.DM)):
        # TODO: Add support for axis argument
        if isinstance(obj, (np.ndarray, ca.DM)):
            return convert(fx(obj), _type, **kwargs)
        elif not obj:
            return convert(fx(), _type, **kwargs)
        else:
            types = [isinstance(k, (_type, str, int, float)) for k in obj]
            not_supported = [k for k, val in enumerate(types) if not val]
            for k in not_supported:
                if hasattr(obj[k], 'values'):
                    obj[k] = obj[k].values
                    types[k] = True
            if all(types):
                if all(isinstance(k, _type) for k in obj):
                    return convert(ca.vertcat(*obj), _type, **kwargs)
                elif all(isinstance(k, str) for k in obj):
                    return convert([fx.sym(parse(k)) for k in obj], _type, **kwargs)
                elif all(isinstance(k, float) for k in obj):
                    return convert(fx(obj), _type, **kwargs)
                elif all(isinstance(k, int) for k in obj):
                    if isinstance(obj, tuple):
                        if len(obj) > 2:
                            raise IndexError("Shape is out of bounds for array of dimension 2")
                    elif isinstance(obj, list):
                        return convert(fx(obj), _type, **kwargs)
                    else:
                        raise TypeError("Expected argument of type {}, got {} instead".format(type(tuple), type(obj)))
                    name = kwargs.get('name', None)
                    if name is not None:
                        if isinstance(name, str):
                            return convert(fx.sym(name, obj), _type, **kwargs)
                        else:
                            raise TypeError("Expected argument 'name' of type {}, got {} instead"
                                            .format(type(str), type(name)))
                    else:
                        raise KeyError("Additional argument 'name' is missing")
                else:
                    # Mix of strings and SX/MX and possibly floats
                    res = []
                    for k in obj:
                        if isinstance(k, _type):
                            res.append(k)
                        else:
                            if isinstance(k, str):
                                res.append(fx.sym(k))
                            elif isinstance(k, (int, float)):
                                res.append(fx(k))
                            else:
                                raise TypeError(f"Wrong type of arguments for supplied list: {type(k).__name__}")
                    return convert(res, _type, **kwargs)
            else:
                raise TypeError("Wrong type of arguments for function {}".format(whoami()))
    elif isinstance(obj, str):
        shape = _get_shape(**kwargs)
        return fx.sym(parse(obj), shape)
    elif isinstance(obj, int):
        if _type in [ca.DM, np.ndarray]:
            return convert(fx(obj), _type, **kwargs)
        else:
            axis = kwargs.get('axis', None)
            if axis is None or axis == 0:
                shape = (obj, 1)
            elif axis == 1:
                shape = (1, obj)
            elif axis > 1:
                raise IndexError("Argument 'axis' is out of bounds for array of dimension 2")
            else:
                raise TypeError("Expected argument 'axis' of type {}, got {} instead".format(type(int), type(axis)))
            name = kwargs.get('name', None)
            if name is not None:
                if isinstance(name, str):
                    return fx.sym(name, shape)
                else:
                    raise TypeError("Expected argument 'name' of type {}, got {} instead".format(type(str), type(name)))
            else:
                raise KeyError("Additional argument 'name' is missing")
    elif isinstance(obj, float):
        return convert(fx(obj), _type, **kwargs)
    elif isinstance(obj, ValuesView):
        return convert(list(obj), _type, **kwargs)
    else:
        raise TypeError("Wrong type of arguments for function {}".format(whoami()))


def dump_clean(obj):
    """

    :param obj:
    :return:
    """
    # FIXME: Strings also have the attribute __iter__
    if isinstance(obj, dict):
        for k, v in obj.items():
            if hasattr(v, '__iter__'):
                print(k)
                dump_clean(v)
            else:
                print("{} : {}".format(k, v))
    elif isinstance(obj, list):
        for v in obj:
            if hasattr(v, '__iter__'):
                dump_clean(v)
            else:
                print(v)
    else:
        print(obj)


def lower_case(obj: Any) -> Any:
    """

    :param obj:
    :return:
    """
    if isinstance(obj, dict):
        return {key.lower(): lower_case(val) for key, val in obj.items()}
    elif isinstance(obj, (list, set, tuple)):
        return type(obj)(lower_case(k) for k in obj)
    elif isinstance(obj, str):
        return obj.lower()
    else:
        return obj


def parse(string):
    """

    :param string:
    :return:
    """
    # TODO: Parse string, e.g. if equations are given in strings (could be useful later for writing equations in GUI)
    # Check if any of the strings in 'OPERATORS' is in 'string'
    return string


def whoami():
    """

    :return:
    """
    return sys._getframe(1).f_code.co_name
