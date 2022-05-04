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
from scipy.sparse import issparse
from scipy.sparse.linalg import norm as sparse_norm

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


def _convert_operators(*args):
    """

    :param args:
    :return:
    """
    ret = set()
    for arg in args:
        if arg == '<' or arg == '>':
            ret.add(ca.OP_LT)
        elif arg == '<=' or arg == '>=':
            ret.add(ca.OP_LE)
        elif arg == '==':
            ret.add(ca.OP_EQ)
        elif arg == '+':
            ret.add(ca.OP_ADD)
        elif arg == '-':
            ret.add(ca.OP_SUB)
        elif arg == '*':
            ret.add(ca.OP_MUL)
        elif arg == '/':
            ret.add(ca.OP_DIV)

    return tuple(ret)


def _get_shape(**kwargs):
    """

    :param kwargs:
    :return:
    """
    # TODO: Typing hints
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


def _split_expression(fun, var, *args):
    """

    :param fun:
    :param var:
    :param args:
    :return:
    """
    work = fun.sz_w() * ['']
    n_var = var.numel()
    split_ops = _convert_operators(*args)
    split_expr = []
    for k in range(fun.n_instructions()):
        op = fun.instruction_id(k)
        o = fun.instruction_output(k)
        o0 = -1
        o1 = -1
        if o:
            o0 = o[0]
        if len(o) > 1:
            o1 = o[1]
        i = fun.instruction_input(k)
        i0 = -1
        i1 = -1
        if i:
            i0 = i[0]
        if len(i) > 1:
            i1 = i[1]
        if op in split_ops:
            work[o0] = [work[i0], work[i1]]
        elif op == ca.OP_CONST:
            work[o0] = fun.instruction_constant(k)
        elif op == ca.OP_INPUT:
            work[o0] = var[i0 * n_var + i1]
        elif op == ca.OP_OUTPUT:
            # work[o1] = work[i0]
            split_expr.append(work[i0])
        elif op == ca.OP_ADD:
            work[o0] = work[i0] + work[i1]
        elif op == ca.OP_SUB:
            work[o0] = work[i0] - work[i1]
        elif op == ca.OP_MUL:
            work[o0] = work[i0] * work[i1]
        elif op == ca.OP_DIV:
            work[o0] = work[i0] / work[i1]
        else:
            raise ValueError(f"Operation with id '{op}' is unknown or not implemented")

    return split_expr


def check_and_wrap_to_list(arg):
    """
    Check if the input is an np.array or a list.

    :param arg:
    :return:
    """
    if isinstance(arg, np.ndarray):
        arg_ = arg.squeeze().tolist()
        if isinstance(arg_, int) or isinstance(arg_, float):
            # if arg is a np,array with a single value the .tolist() method returns a integer or float.
            arg_ = [arg]
    elif isinstance(arg, list):
        arg_ = arg
    elif isinstance(arg, float) or isinstance(arg, int) or isinstance(arg, str):
        arg_ = [arg]
    elif isinstance(arg, ca.DM):
        arg_ = np.array(arg).squeeze(-1).tolist()
    else:
        raise TypeError(f"Type {type(arg)} not supported. Must be list,float,ca.DM or numpy array ")

    return arg_


def check_and_wrap_to_DM(arg):
    """

    :param arg:
    :return:
    """
    if isinstance(arg, np.ndarray):
        arg_ = ca.DM(arg.squeeze())
    elif isinstance(arg, list) or isinstance(arg, float) or isinstance(arg, int):
        arg_ = ca.DM(arg)
    elif isinstance(arg, ca.DM):
        arg_ = arg
    else:
        raise TypeError(f"Type {type(arg)} not supported. Must be list,float,ca.DM or numpy array ")
    return arg_


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


def check_if_has_duplicates(vector):
    """
    Checks if list contains duplicates

    :param vector:
    :return:
    """
    if not isinstance(vector, list):
        raise TypeError("Vector must be a list.")

    if len(vector) == len(set(vector)):
        return False
    else:
        return True


def check_if_list_of_type(a, types):
    """

    :param a:
    :param types:
    :return:
    """
    # TODO: Typing hints
    if not isinstance(a, list):
        return False
    else:
        return all(isinstance(item, types) for item in a)


def check_if_list_of_none(a):
    """
    Check if a list contains only None

    :param a:
    :return:
    """
    return all(item is None for item in a)


def check_if_list_of_string(a):
    """
    Check if a list contains only strings

    :param a:
    :return:
    """
    return all(isinstance(item, str) for item in a)


def check_if_square(arg):
    """

    :param arg: DM or NP matrix
    :return: True if matrix is square, False if not
    """
    if len(arg.shape) > 2:
        raise ValueError("The input can have at most two dimensions.")

    if len(arg.shape) == 1:
        return False
    if len(arg.shape) == 2:
        if arg.shape[0] == arg.shape[1]:
            return True
        else:
            return False


def convert(obj, _type, **kwargs):
    """

    :param obj:
    :param _type:
    :param kwargs:
    :return:
    """
    # TODO: Typing hints
    # TODO: Convoluted. Simplify if possible.
    # TODO: Add support for dict?
    # TODO: Is there a nice way to convert SX to MX and vice versa?
    # TODO: Return success flag and error message for processing outside of convert
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
                elif obj.ndim == 1:
                    if len(shape) == 2 and shape[0] == shape[1] and shape[0] in obj.shape:
                        obj = np.diag(obj)
                elif 1 in obj.shape:
                    if max(obj.shape) != max(shape):
                        raise IndexError("Dimension mismatch. Cannot reduce or increase size of the shape.")
                    if obj.shape == shape[::-1]:
                        return obj.T
                    elif obj.ndim == 2:
                        if len(shape) == 2 and shape[0] == shape[1] and shape[0] in obj.shape:
                            obj = np.diag(obj.flatten())
                    else:
                        raise Exception("Don't know how I got here. Please inform the maintainer.")
        elif _type is np.ndarray and 'shape' in kwargs and obj.shape != shape:
            # TODO: Catch exceptions to generate meaningful error messages
            obj = np.reshape(obj, shape)
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
                raise TypeError("Wrong type of arguments for function {}".format(who_am_i()))
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
        raise TypeError("Wrong type of arguments for function {}".format(who_am_i()))


def dump_clean(obj):
    """

    :param obj:
    :return:
    """
    # TODO: Typing hints
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


def generate_c_code(functions, path, name, opts=None):
    """

    :param functions:
    :param path:
    :param name:
    :param opts:
    :return:
    """
    if functions:
        if opts is None:
            opts = {}
        generator = ca.CodeGenerator(name, opts)
        if isinstance(functions, list):
            for f in functions:
                generator.add(f)
        else:
            generator.add(functions)
        c_name = generator.generate(path)
        return c_name
    else:
        return None


def is_array_like(obj):
    """

    :param obj:
    :return:
    """
    # TODO: Typing hints
    return is_list_like(obj) and hasattr(obj, 'dtype')


def is_diagonal(obj):
    """

    :param obj:
    :return:
    """
    diag = np.zeros(obj.shape)
    np.fill_diagonal(diag, obj.diagonal())
    return np.all(obj == diag)


def is_integer(n):
    """

    :param n:
    :return:
    """
    # TODO: Typing hints
    if isinstance(n, int):
        return True
    if isinstance(n, float):
        return n.is_integer()
    return False


def is_iterable(obj) -> bool:
    """

    :param obj:
    :return:
    """
    try:
        iter(obj)
    except TypeError:
        return False
    return True


def is_list_like(obj):
    """

    :param obj:
    :return:
    """
    # TODO: Typing hints
    return isinstance(obj, (list, tuple, set, np.ndarray))


def is_pd(x):
    """
    https://stackoverflow.com/a/63911811

    :param x:
    :return:
    """
    try:
        np.linalg.cholesky(x)
    except np.linalg.LinAlgError:
        return False
    return True


def is_psd(x, tol=1e-14):
    """
    https://stackoverflow.com/a/63911811

    :param x:
    :param tol:
    :return:
    """
    try:
        regularized_x = x + np.eye(x.shape[0]) * tol
        np.linalg.cholesky(regularized_x)
    except np.linalg.LinAlgError:
        return False
    return True


def is_real(x):
    """

    :param x:
    :return:
    """
    if not is_list_like(x):
        return np.isreal(x)
    else:
        return np.isreal(x).all()


def is_square(array):
    """

    :param array:
    :return:
    """
    if array.ndim == 1:
        return False
    return all([len(row) == len(array) for row in array])


def is_symmetric(x, tol=1e-8):
    """
    https://stackoverflow.com/a/67406237

    :param x:
    :param tol:
    :return:
    """
    if issparse(x):
        # TODO: Test this
        return sparse_norm(x - x.T, ca.inf) < tol
    else:
        return np.linalg.norm(x - x.T, ca.inf) < tol


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
    # TODO: Typing hints
    # TODO: Parse string, e.g. if equations are given in strings (could be useful later for writing equations in GUI)
    # Check if any of the strings in 'OPERATORS' is in 'string'
    return string


def random_state(state=None):
    """

    :param state:
    :return:
    """
    # TODO: Typing hints
    if is_integer(state) or is_array_like(state):
        return np.random.RandomState(state)
    elif isinstance(state, np.random.RandomState):
        return state
    elif state is None:
        return np.random
    else:
        raise ValueError("Argument 'random_state' must be an integer, array-like, a NumPy RandomState, or None")


def scale_vector(vector, scaler):
    """

    :param vector: NP array, list or ca.DM vector
    :param scaler:
    :return:
    """

    vector = check_and_wrap_to_list(vector)
    scaler = check_and_wrap_to_list(scaler)

    if len(vector) != len(scaler):
        raise ValueError(f"The length of scaler and vector must be equal. The vector has length {len(vector)} while the"
                         f" scaler {len(scaler)}")
    s_vector = vector.copy()
    for k, i in enumerate(vector):
        s_vector[k] = i / scaler[k]

    return s_vector


def who_am_i() -> str:
    """

    :return:
    """
    return sys._getframe(1).f_code.co_name
