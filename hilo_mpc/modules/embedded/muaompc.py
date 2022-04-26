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

from importlib import import_module, invalidate_caches
import os
import shutil
from subprocess import call
import sys
from types import ModuleType

import casadi as ca
import numpy as np

from muaompc import ldt


mixed_box_constraints_reg_orig = """variable u[0:N-1](m);
auxs x[0:N](n);
parameters x0(n);
minimize sum(quad(x[i],Q)+quad(u[i], R), i=0:N-1)+quad(x[N],P);
subject to x[i+1] = A*x[i]+B*u[i], i=0:N-1;
u_lb <= u[i] <= u_ub, i=0:N-1;
x_lb <= x[i] <= x_ub, i=0:N-1;
x[0]=x0;
"""


CTL_SOLVER = None
HILO_MPC = None


def solver(x0):
    """

    :param x0:
    :return:
    """
    global CTL_SOLVER, HILO_MPC
    x0_shape = CTL_SOLVER.parameters.x0.shape
    n_u = HILO_MPC._n_u

    CTL_SOLVER.parameters.x0[:] = np.array(x0).reshape(x0_shape)
    CTL_SOLVER.solve_problem()
    return CTL_SOLVER.u_opt[0:n_u]


def set_ctl_solver(ctl):
    """

    :param ctl:
    :return:
    """
    global CTL_SOLVER
    CTL_SOLVER = ctl


def set_hilo_mpc(mpc):
    """

    :param mpc:
    :return:
    """
    global HILO_MPC
    HILO_MPC = mpc


def setup_solver(mpc):
    """

    :param mpc:
    :return:
    """
    dest_dir = 'muaompc_codegen'
    mod_name = 'hiloext'
    prb_name = 'muaompc'
    os.makedirs(dest_dir, exist_ok=True)
    prb_fname = os.path.join(dest_dir, prb_name+'.prb')
    basedir = shutil.os.path.abspath(shutil.os.curdir)
    codegendir = os.path.join(basedir, dest_dir, '%s_%s' % (
        mod_name, prb_name))
    is_new_prb = write_prb_file(prb_fname,
                                mixed_box_constraints_reg_orig)
    datmod = gen_data_module(mpc)

    # Generate first code for microcontroller
    mpcprb = ldt.setup_mpc_problem(
        prb_fname, prefix=mod_name, destdir=dest_dir,
        numeric='float32')  # single precision float for microcontrollers
    ldt.generate_mpc_data(mpcprb, datmod, safe_mode=False, muc=True)

    # Generate Cython interface
    mpcprb = ldt.setup_mpc_problem(
        prb_fname, prefix=mod_name, destdir=dest_dir,
        numeric='float64')  # Cython interface uses double precision float
    if is_new_prb:
        compile_controller(mod_name, basedir, codegendir)
        m = "I must stop the execution of the program. Restart your program "
        m += " to correctly load the generated muaompc interface module. "
        m += "muaompc's Python interface was installed successfully.\n"
        m += "After restarting your program, you should not see this message."
        # It is not trivial to _universally_ load a module installed on the fly
        raise ModuleNotFoundError(m)
    ldt.generate_mpc_data(mpcprb, datmod, safe_mode=False, muc=False)
    ctl = init_controller(datmod, mod_name, basedir, codegendir)
    config_controller(ctl, mpc._nlp_opts)
    set_hilo_mpc(mpc)
    set_ctl_solver(ctl)
    return solver


def compile_controller(mod_name, basedir, codegendir):
    """

    :param mod_name:
    :param basedir:
    :param codegendir:
    :return:
    """
    python = sys.executable
    shutil.os.chdir(codegendir)
    call([python, mod_name + 'setup.py', 'install'])
    invalidate_caches()
    shutil.os.chdir(basedir)


def config_controller(ctl, opts):
    """

    :param ctl:
    :param opts:
    :return:
    """
    if opts.get('in_iter') is not None:
        ctl.conf.in_iter = opts['in_iter']
    if opts.get('ex_iter') is not None:
        ctl.conf.ex_iter = opts['ex_iter']
    if opts.get('warm_start') is not None:
        ctl.conf.warm_start = opts['warm_start']


def init_controller(dat_mod, mod_name, basedir, codegendir):
    """

    :param dat_mod:
    :param mod_name:
    :param basedir:
    :param codegendir:
    :return:
    """
    ctlmod = import_module('.'+mod_name+'ctl', package=mod_name)
    data_name = dat_mod.__name__
    shutil.os.chdir(codegendir)
    ctl = ctlmod.Ctl('data/%s/%s%s.json' % (data_name, mod_name, data_name))
    shutil.os.chdir(basedir)
    return ctl


def write_prb_file(fname, prb):
    """

    :param fname:
    :param prb:
    :return:
    """
    try:
        f = open(fname, 'r')
    except FileNotFoundError:
        write_new_prb = True
    else:
        if f.read() == prb:  # content of file is identical
            write_new_prb = False  # do not overwrite prb file
        else:
            write_new_prb = True
        f.close()

    if write_new_prb:
        with open(fname, 'w') as f:
            f.write(prb)

    return write_new_prb


def gen_data_module(mpc):
    """

    :param mpc:
    :return:
    """
    model = mpc._model
    #R = np.array(mpc.R).reshape(m,m)
    dat = ModuleType('dat')
    dat.A = np.array(ca.DM(model.A))
    dat.B = np.array(ca.DM(model.B))
    dat.Q = np.array(ca.DM(mpc.Q))
    try:
        dat.P = np.array(ca.DM(mpc.P))
    except AttributeError:
        dat.P = np.array(ca.DM(mpc.Q))
    dat.R = np.array(ca.DM(mpc.R))
    dat.n, dat.m = dat.B.shape
    dat.u_lb = np.array(ca.DM(mpc._u_lb)).reshape(dat.m, 1)
    dat.u_ub = np.array(ca.DM(mpc._u_ub)).reshape(dat.m, 1)
    dat.x_lb = np.array(ca.DM(mpc._x_lb)).reshape(dat.n, 1)
    dat.x_ub = np.array(ca.DM(mpc._x_ub)).reshape(dat.n, 1)
    dat.N = mpc.horizon

    return dat
