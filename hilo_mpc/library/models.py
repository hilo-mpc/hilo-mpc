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

import casadi as ca

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


def ecoli_D1210_conti(model='simple'):
    """
    Returns a hilo-mpc Model object.
    The model is an adaptation to a continuous batch process from the fedbach process proposed in

    Lee,  J.;  Ramirez,  W.  F.
    Mathematical  modelling  of  induced  foreign protein  production  by  recombinant  bacteria.
    Biotechnol.  Bioeng. 1992,39, 635-646.

    If model ='simple' the model does not have the two inducer states and the reactions rates are left as parameters
    If model= 'complex' the model has the two inducer states and the reaction rates are the same as in Lee et al.

    :return:
    """
    # Load plot backend
    plot_backend = get_plot_backend()

    if model == 'complex':
        model = Model(plot_backend=plot_backend, name='ecoli_D1210_complex')
        x = model.set_dynamical_states('X', 'S', 'P', 'I', 'ISF', 'IRF')
        u = model.set_inputs('DS', 'DI')
        p = model.set_parameters('Sf', 'If')
        model.set_measurements('mu', 'Rs', 'Rfp')

        # Unwrap states
        X = x[0]
        S = x[1]
        P = x[2]
        I = x[3]
        ISF = x[4]
        IRF = x[5]

        # Unwrap inputs
        u1 = u[0]
        u2 = u[1]

        # Unwrap parameters
        Sf = p[0]
        If = p[1]

        # Reaction rates
        phi = 0.407 * S / (0.108 + S + (S ** 2) / 14814.0)
        mu = phi * (ISF + (0.22 * IRF) / (0.22 + I))
        Rs = 2 * mu
        Rfp = phi * (0.0005 + I) / (0.022 + I)
        k1 = 0.09 * I / (0.034 + I)
        k2 = 0.09 * I / (0.034 + I)

        D = u1 + u2

        dX = mu * X - D * X
        dS = - Rs * X - D * S + u1 * Sf
        dP = Rfp * X - D * P
        dI = - D * I + u2 * If
        dISF = -k1 * ISF
        dIRF = k2 * (1 - IRF)

        model.set_dynamical_equations([dX, dS, dP, dI, dISF, dIRF])
        model.set_measurement_equations([mu, Rs, Rfp])

        return model
    elif model == 'simple':
        model = Model(plot_backend=plot_backend, name='ecoli_D1210_conti_simple')
        x = model.set_dynamical_states('X', 'S', 'P', 'I')
        u = model.set_inputs('DS', 'DI')
        p = model.set_parameters('Sf', 'If', 'mu', 'Rs', 'Rfp')

        # Unwrap states
        X = x[0]
        S = x[1]
        P = x[2]
        I = x[3]

        # Unwrap inputs
        DS = u[0]
        DI = u[1]

        # Unwrap parameters
        Sf = p[0]
        If = p[1]

        # Unknown reaction rates
        mu = p[2]
        Rs = p[3]
        Rfp = p[4]

        D_tot = DS + DI

        dX = mu * X - D_tot * X
        dS = - Rs * X - D_tot * S + DS * Sf
        dP = Rfp * X - D_tot * P
        dI = - D_tot * I + DI * If
        model.set_dynamical_equations([dX, dS, dP, dI])

        return model


def ecoli_D1210_fedbatch():
    """
    Returns a hilo-mpc Model object of fedbatch bioreactor containing genetically modified Escherichia Coli D1210
    The model is taken from

    Lee,  J.;  Ramirez,  W.  F.
    Mathematical  modelling  of  induced  foreign protein  production  by  recombinant  bacteria.
    Biotechnol.  Bioeng.1992,39, 635-646.

    :return:
    """
    # Load plot backend
    plot_backend = get_plot_backend()

    model = Model(plot_backend=plot_backend, name='ecoli_D1210_fedbatch_complex')
    x = model.set_dynamical_states('X', 'S', 'P', 'I', 'ISF', 'IRF', 'V')
    u = model.set_inputs('FeedS', 'FeedI')

    Sf = 100
    If = 4

    # Unwrap states
    X = x[0]
    S = x[1]
    P = x[2]
    I = x[3]
    ISF = x[4]
    IRF = x[5]
    V = x[6]

    # Unwrap inputs
    u1 = u[0]
    u2 = u[1]

    # Reaction rates
    phi = 0.407 * S / (0.108 + S + (S ** 2) / 14814.0)
    mu = phi * (ISF + (0.22 * IRF) / (0.22 + I))
    Rs = 2 * mu
    Rfp = phi * (0.0005 + I) / (0.022 + I)
    k1 = 0.09 * I / (0.034 + I)
    k2 = 0.09 * I / (0.034 + I)

    D = (u1 + u2) / V

    dX = mu * X - D * X
    dS = - Rs * X - D * S + u1 * Sf / V
    dP = Rfp * X - D * P
    dI = - D * I + u2 * If / V
    dISF = -k1 * ISF
    dIRF = k2 * (1 - IRF)
    dV = u1 + u2

    model.set_dynamical_equations([dX, dS, dP, dI, dISF, dIRF, dV])

    return model


def scerevisiae_SEY2102_fedbatch():
    """
    Returns a hilo-mpc Model object of fedbatch bioreactor containing Saccharomyces Cerevisiae SEY2102. The model is
    taken from
    Park, Seujeung, and W. Fred Ramirez.
    "Dynamics of foreign protein secretion from Saccharomyces cerevisiae."
    Biotechnology and bioengineering 33.3 (1989): 272-281.

    :return:
    """
    # Load plot backend
    plot_backend = get_plot_backend()

    model = Model(plot_backend=plot_backend, name='scerevisae_SEY2102_fedbatch')
    x = model.set_dynamical_states('bio', 's', 'pt', 'pm', 'v')
    F = model.set_inputs('F')
    # s0 = model.set_parameters('s0')
    s0 = 20  # g/l
    bio = x[0]
    s = x[1]
    pt = x[2]
    pm = x[3]
    V = x[4]

    mu = (21.87 * s) / ((s + 0.4) * (s + 62.5))
    fp = (s * ca.exp(-5 * s)) / (s + 0.1)
    phi = 4.75 * mu / (0.12 + mu)

    D = F / V

    dbio = mu * bio - D * bio
    ds = -7.3 * mu * bio - D * (s - s0)
    dpt = fp * bio - D * pt
    dpm = phi * (pt - pm) - D * pm
    dv = F

    model.set_dynamical_equations([dbio, ds, dpt, dpm, dv])
    model.set_measurements('bio', 's', 'pt', 'pm', 'v')
    model.set_measurement_equations(model.x)

    return model


__all__ = [
    'cstr_schaffner_and_zeitz',
    'cstr_seborg',
    'ecoli_D1210_conti',
    'ecoli_D1210_fedbatch',
    'scerevisiae_SEY2102_fedbatch'
]
