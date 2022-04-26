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

from ..modules.dynamic_model.dynamic_model import Model
from ..util.plotting import get_plot_backend


def cstr_schaffner_and_zeitz():
    """

    :return:
    """
    # Load plot backend
    plot_backend = get_plot_backend()

    # Initialize empty model
    model = Model(name='CSTR', plot_backend=plot_backend)

    # Define model equations
    equations = """
    # ODEs
    dx_1/dt = -a_1*x_1(t) + b_1*r
    dx_2/dt = -a_2*x_2(t) + b_2*r + g*u(k)

    # Measurement
    y(k) = x_2(t)

    # Algebraic equations
    r = (1 - x_1(t))*exp(-E/(1 + x_2(t)))
    """
    model.set_equations(equations=equations)

    return model


def cstr_seborg():
    """

    :return:
    """
    # Load plot backend
    plot_backend = get_plot_backend()

    # Initialize empty model
    model = Model(name='CSTR', plot_backend=plot_backend)

    # Define model equations
    equations = """
    # ODEs
    dC_A/dt = q_0/V*(C_Af - C_A(t)) - k_0*exp(-E/(R*T(t)))*C_A(t)
    dT/dt = q_0/V*(T_f - T(t)) - DeltaH_r*k_0/(rho*C_p)*exp(-E/(R*T(t)))*C_A(t) + UA/(V*rho*C_p)*(T_c(t) - T(t))
    dT_c/dt = (T_cr(k) - T_c(t))/tau

    # Measurements
    y(k) = C_A(t)

    # Constants
    R = 8.314

    # General
    C_A|description: concentration of substrate A
    C_A|label: concentration
    C_A|unit: mol/L
    T|description: tank temperature
    T|label: temperature
    T|unit: K
    T_c|description: coolant temperature
    T_c|label: temperature
    T_c|unit: K
    T_cr|description: coolant temperature reference
    T_cr|label: temperature
    T_cr|unit: K
    # ...parameters
    """
    model.set_equations(equations=equations)

    return model


__all__ = [
    'cstr_schaffner_and_zeitz',
    'cstr_seborg'
]
