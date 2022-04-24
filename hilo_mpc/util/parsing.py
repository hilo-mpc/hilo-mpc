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

import ast
import re
from typing import Optional, Sequence, TypeVar

import casadi as ca


Symbolic = TypeVar('Symbolic', ca.SX, ca.MX)


FUNCTIONS = {
    'sqrt': ca.sqrt,
    'exp': ca.exp,
    'log': ca.log,
    'log10': ca.log10,
    'sign': ca.sign,
    'abs': ca.fabs,
    'min': ca.fmin,
    'max': ca.fmax,
    'sin': ca.sin,
    'cos': ca.cos,
    'tan': ca.tan,
    'arcsin': ca.asin,
    'arccos': ca.acos,
    'arctan': ca.atan,
    'arctan2': ca.atan2,
    'sinh': ca.sinh,
    'cosh': ca.cosh,
    'tanh': ca.tanh,
    'arsinh': ca.asinh,
    'arcosh': ca.acosh,
    'artanh': ca.atanh
}


class Parser:
    """"""
    fx = ca.SX

    @classmethod
    def _get_op(cls, node):
        """

        :param node:
        :return:
        """
        op_type = type(node.op)

        return {
            ast.Add: ca.plus,
            ast.Sub: ca.minus,
            ast.Mult: ca.times,
            ast.Div: ca.rdivide,
            ast.Pow: ca.power,
            # ast.BitXor: ca.power,  # Doesn't make sense right now (see https://github.com/casadi/casadi/issues/586)
            ast.USub: neg
        }[op_type]

    @classmethod
    def _get_op_fun(cls, node):
        """

        :param node:
        :return:
        """
        return FUNCTIONS[node.func.id]

    @classmethod
    def _num_op(cls, node, subs, free):
        """

        :param node:
        :param subs:
        :param free:
        :return:
        """
        return node.n

    @classmethod
    def _bin_op(cls, node, subs, free):
        """

        :param node:
        :param subs:
        :param free:
        :return:
        """
        op = cls._get_op(node)
        left_node = cls.eval(node.left, subs, free)
        right_node = cls.eval(node.right, subs, free)
        return op(left_node, right_node)

    @classmethod
    def _unary_op(cls, node, subs, free):
        """

        :param node:
        :param subs:
        :param free:
        :return:
        """
        op = cls._get_op(node)
        return op(cls.eval(node.operand, subs, free))

    @classmethod
    def _subs_op(cls, node, subs, free):
        """

        :param node:
        :param subs:
        :return:
        """
        n_id = node.id
        if n_id in subs:
            return subs[n_id]
        else:
            if free is not None:
                value = cls.fx.sym(n_id)
                if isinstance(free, dict):
                    free[n_id] = value
                elif isinstance(free, (list, set, tuple)):
                    free.append(value)
                else:
                    msg = "Type {} for argument 'free' not supported".format(type(free))
                    raise TypeError(msg)
                subs[n_id] = value
                return value
            else:
                raise TypeError(node)

    @classmethod
    def _call_op(cls, node, subs, free):
        """

        :param node:
        :param subs:
        :param free:
        :return:
        """
        if node.func.id in FUNCTIONS:
            args = []
            for arg in node.args:
                args.append(cls.eval(arg, subs, free))
            fun = cls._get_op_fun(node)
            return fun(*args)
        else:
            # NOTE: Everything that is dependent on the time and not part of subs will be treated as a function
            # (like e.g. u(t)). We could also raise an error here.
            return cls.eval(node.func, subs, free)

    @classmethod
    def _const_op(cls, node, subs, free):
        """

        :param node:
        :param subs:
        :param free:
        :return:
        """
        return node.value

    @classmethod
    def eval(cls, node, subs, free):
        """

        :param node:
        :param subs:
        :param free:
        :return:
        """
        node_type = type(node)

        return {
            ast.Num: cls._num_op,
            ast.BinOp: cls._bin_op,
            ast.UnaryOp: cls._unary_op,
            ast.Name: cls._subs_op,
            ast.Call: cls._call_op,
            ast.Constant: cls._const_op
        }[node_type](node, subs, free)

    @classmethod
    def eval_expression(cls, expr, subs=None, free=None):
        """

        :param expr:
        :param subs:
        :param free:
        :return:
        """
        if subs is None:
            subs = {}

        expr_tree = ast.parse(expr, mode='eval').body
        return cls.eval(expr_tree, subs, free)


def neg(operand):
    """

    :param operand:
    :return:
    """
    return operand.__neg__()


def is_numeric_2(string):
    """

    :param string:
    :return:
    """
    try:
        value = float(string)
    except ValueError:
        return None
    else:
        return value


def parse_dynamic_equations(
        equations: Sequence[str],
        discrete: bool = False,
        replace_bitwise_xor: bool = True,
        use_sx: bool = True,
        **kwargs: Optional[Sequence[Symbolic]]
) -> dict[str, list[Symbolic]]:
    """

    :param equations:
    :param discrete:
    :param replace_bitwise_xor:
    :param use_sx:
    :param kwargs:
    :return:
    """
    # NOTE: This is a work in progress, that will be updated as more and more examples become available
    # TODO: Multi-mode parsing
    # TODO: Catch not working names, like e.g. lambda
    dt = kwargs.get('dt')
    t = kwargs.get('t')
    x = kwargs.get('x')
    y = kwargs.get('y')
    z = kwargs.get('z')
    u = kwargs.get('u')
    p = kwargs.get('p')

    to_substitute = {}
    if dt is not None:
        to_substitute.update({k.name(): k for k in dt})
    if t is not None:
        to_substitute.update({k.name(): k for k in t})
    if x is not None:
        to_substitute.update({k.name(): k for k in x})
    else:
        x = []
    if y is not None:
        to_substitute.update({k.name(): k for k in y})
    else:
        y = []
    if z is not None:
        to_substitute.update({k.name(): k for k in z})
    else:
        z = []
    if u is not None:
        to_substitute.update({k.name(): k for k in u})
    else:
        u = []
    if p is not None:
        to_substitute.update({k.name(): k for k in p})
    else:
        p = []

    description = {}
    labels = {}
    units = {}

    model = {
        'x': x,
        'y': y,
        'z': z,
        'u': u,
        'p': p,
        'ode': [],
        'alg': [],
        'quad': [],
        'meas': [],
        'const': {}
    }

    if use_sx:
        fx = ca.SX
        Parser.fx = ca.SX
    else:
        fx = ca.MX
        Parser.fx = ca.MX

    to_replace = {}
    odes = []
    algs = []
    quad = []
    meas = []

    ivp = '...'

    for k, equation in enumerate(equations):
        if '|' in equation:
            var, prop = equation.split('|')
            if ':' in prop:
                var = var.lstrip()
                key, val = prop.split(':')
                if key == 'description':
                    description[var] = val.lstrip()
                elif key == 'label':
                    labels[var] = val.lstrip()
                elif key == 'unit':
                    units[var] = val.lstrip()
            continue

        eq = equation.replace(' ', '')
        ivp = ivp.replace('...', eq)
        if ivp.endswith('...'):
            continue
        elif ivp.startswith('#') or '=' not in ivp:
            ivp = '...'
            continue
        if replace_bitwise_xor:
            ivp = ivp.replace('^', '**')
        else:
            raise NotImplementedError("Bitwise operations are not yet supported for parsing")

        lhs, rhs = ivp.split('=')

        if not discrete:
            # Check if equation is a quadrature function
            # TODO: Raise warning when 'sum' is used?
            if lhs == 'int':
                quad.append(rhs)
                continue

            # Check if equation is a differential equation
            # Check if left-hand side is of form dx(t)/dt
            match = re.search(r'd(.+)/dt', lhs)
            if match is None:
                # If there is not match, check if left-hand side is of form d/dt(x(t))
                match = re.search(r'd/dt\((.+)\)', lhs)
                if match is not None:
                    state = re.sub(r'\(.*?\)', '', match.groups()[0])
                    if state not in to_substitute:
                        value = fx.sym(state)
                        to_substitute[state] = value
                        model['x'].append(value)
                    # In case we have some nested expressions using d/dt(x(t))
                    to_replace[lhs] = f'({rhs})'
                    odes.append(rhs)
                else:
                    # No match, i.e. no differential equation
                    if lhs == '0':
                        # Algebraic equation 0 = g(x,z,u,t) (implicit form)
                        algs.append(rhs)
                    else:
                        value = is_numeric_2(rhs)
                        if value is None:
                            # Either algebraic variable or measurement
                            match = re.search(r'(.*)\(t\)', lhs)
                            if match is None:
                                # No algebraic variable
                                match = re.search(r'(.*)\(k\)', lhs)
                                if match is None:
                                    # No measurement, just an auxiliary variable that needs to be replaced
                                    # TODO: Process data even more, see DaeBuilder ddef (dependent parameters)
                                    to_replace[lhs] = f'({rhs})'
                                else:
                                    # Measurement y(k) = ...
                                    if match.groups()[0] not in to_substitute:
                                        value = fx.sym(match.groups()[0])
                                        to_substitute[match.groups()[0]] = value
                                        model['y'].append(value)
                                    meas.append(rhs)
                            else:
                                # Algebraic variable z(t) = ... (explicit form)
                                if match.groups()[0] not in to_substitute:
                                    value = fx.sym(match.groups()[0])
                                    to_substitute[match.groups()[0]] = value
                                    model['z'].append(value)
                                algs.append(rhs + '-' + lhs)
                        else:
                            # Constants
                            to_substitute[lhs] = value
                            model['const'][lhs] = value
            else:
                state = re.sub(r'\(.*?\)', '', match.groups()[0])
                if state not in to_substitute:
                    value = fx.sym(state)
                    to_substitute[state] = value
                    model['x'].append(value)
                # In case we have some nested expressions using d(x(t))/dt
                to_replace[lhs] = f'({rhs})'
                odes.append(rhs)
        else:
            if lhs == 'sum':
                # TODO: Raise warning when 'int' is used?
                quad.append(rhs)
                continue

            match = re.search(r'(.*)\(k\+1\)', lhs)
            if match is not None:
                # Difference equation
                if match.groups()[0] not in to_substitute:
                    value = fx.sym(match.groups()[0])
                    to_substitute[match.groups()[0]] = value
                    model['x'].append(value)
                # In case we have some nested expressions using x(k+1)
                to_replace[lhs] = f'({rhs})'
                odes.append(rhs)
            else:
                # No match, i.e. no difference equation
                if lhs == '0':
                    # Algebraic equation 0 = g(x(k),z(k),u(k),t(k)) (implicit form)
                    algs.append(rhs)
                else:
                    value = is_numeric_2(rhs)
                    if value is None:
                        # Check if measurement (algebraic variables are not possible here, since they would be
                        # indistinguishable from the measurements: y(k) vs z(k))
                        match = re.search(r'(.*)\(k\)', lhs)
                        if match is None:
                            # No measurement, just an auxiliary variable that needs to be replaced
                            # TODO: Process data even more, see DaeBuilder ddef (dependent parameters)
                            to_replace[lhs] = f'({rhs})'
                        else:
                            # Measurement y(k) = ...
                            if match.groups()[0] not in to_substitute:
                                value = fx.sym(match.groups()[0])
                                to_substitute[match.groups()[0]] = value
                                model['y'].append(value)
                            meas.append(rhs)

        ivp = '...'

    to_replace = dict((re.escape(k), v) for k, v in sorted(to_replace.items(), key=lambda item: item[0], reverse=True))
    pattern = re.compile('|'.join(to_replace.keys()))

    def replace_nested(m) -> str:
        """

        :param m:
        :return:
        """
        key = m.group(0)
        end_pos = m.regs[0][-1]
        if end_pos == len(m.string):
            # NOTE: This could be any character as long as it's inside the list of the next if-condition
            next_char = '+'
        else:
            next_char = m.string[end_pos]
        if next_char in ['+', '-', '*', '/', ')']:
            val = nested.get(re.escape(key), None)
            if val is not None:
                return pattern.sub(replace_nested, val)
        return key

    def check_right_hand_side():
        """

        :return:
        """
        # NOTE: The following pattern doesn't include algebraic variables and inputs that have an underscore in their
        #  name
        # pattern = re.compile(r'([a-zA-Z]+)\(([kt0-9+\-]{1,3})\)')
        # NOTE: r'([.+])\(... could match the whole equation or parts of it depending on where it finds the last
        #  brackets containing 'k' or 't'
        pattern = re.compile(r'([a-zA-Z0-9_]+)\(([kt0-9+\-]{1,3})\)')
        matches = pattern.findall(eq)
        for match in matches:
            if match[0] not in FUNCTIONS and match[0] not in to_substitute:
                if not discrete:
                    value = fx.sym(match[0])
                    to_substitute[match[0]] = value
                    if match[1] == 't':
                        model['z'].append(value)
                    else:
                        # TODO: If necessary, deal with different time points for u, i.e. k, k-1, k-2,...
                        model['u'].append(value)
                else:
                    # TODO: How to distinguish between algebraic variables and input?
                    #  Do 'difference algebraic equations' even make sense?
                    pass

    nested = {}
    for ode in odes:
        # TODO: Maybe set variable that is to be replaced to 'ode' or 'alg' and then grab the slice, which is already a
        #  SX variable, inside the Parser, so we don't have to parse this string again
        if pattern.pattern:
            nested.update(to_replace)
            eq = pattern.sub(replace_nested, ode)
        else:
            # No pattern that needs to be replaced
            eq = ode
        check_right_hand_side()
        model['ode'].append(Parser.eval_expression(eq, subs=to_substitute, free=model['p']))

    for alg in algs:
        if pattern.pattern:
            nested.update(to_replace)
            eq = pattern.sub(replace_nested, alg)
        else:
            eq = alg
        check_right_hand_side()
        model['alg'].append(Parser.eval_expression(eq, subs=to_substitute, free=model['p']))

    for q in quad:
        if pattern.pattern:
            nested.update(to_replace)
            eq = pattern.sub(replace_nested, q)
        else:
            eq = q
        check_right_hand_side()
        model['quad'].append(Parser.eval_expression(eq, subs=to_substitute, free=model['p']))

    for mea in meas:
        if pattern.pattern:
            nested.update(to_replace)
            eq = pattern.sub(replace_nested, mea)
        else:
            eq = mea
        check_right_hand_side()
        model['meas'].append(Parser.eval_expression(eq, subs=to_substitute, free=model['p']))

    model['description'] = {
        'x': [description[xk.name()] if xk.name() in description else '' for xk in x],
        'y': [description[yk.name()] if yk.name() in description else '' for yk in y],
        'z': [description[zk.name()] if zk.name() in description else '' for zk in z],
        'u': [description[uk.name()] if uk.name() in description else '' for uk in u],
        'p': [description[pk.name()] if pk.name() in description else '' for pk in p]
    }
    model['labels'] = {
        'x': [labels[xk.name()] if xk.name() in labels else '' for xk in x],
        'y': [labels[yk.name()] if yk.name() in labels else '' for yk in y],
        'z': [labels[zk.name()] if zk.name() in labels else '' for zk in z],
        'u': [labels[uk.name()] if uk.name() in labels else '' for uk in u],
        'p': [labels[pk.name()] if pk.name() in labels else '' for pk in p]
    }
    model['units'] = {
        'x': [units[xk.name()] if xk.name() in units else '' for xk in x],
        'y': [units[yk.name()] if yk.name() in units else '' for yk in y],
        'z': [units[zk.name()] if zk.name() in units else '' for zk in z],
        'u': [units[uk.name()] if uk.name() in units else '' for uk in u],
        'p': [units[pk.name()] if pk.name() in units else '' for pk in p]
    }

    return model
