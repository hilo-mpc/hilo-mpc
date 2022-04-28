#
#   This file is part of HILO-MPC
#
#   HILO-MPC is toolbox for easy, flexible and fast development of machine-learning supported
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
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#   GNU Lesser General Public License for more details.
#
#   You should have received a copy of the GNU General Public License
#   along with HILO-MPC.  If not, see <http://www.gnu.org/licenses/>.

import pathlib
import unittest
from unittest import skip

import casadi as ca
import numpy as np
import pandas as pd
import time

from hilo_mpc import Model, MHE

from bokeh.io import output_file, show
from bokeh.plotting import figure
from bokeh.layouts import gridplot
import numpy as np


class MyTestCase(unittest.TestCase):
    def test_1(self):
        """
        Runs MHE, with a very symple system. one state one input, no state and measurement noise, not model mismatch
        :return:
        """
        model = Model(plot_backend='bokeh')
        x = model.set_dynamical_states(['x1'])
        u = model.set_inputs(['u'])
        model.set_measurement_equations(x)
        model.set_measurements(['y1'])
        sampling_interval = 0.5
        n_steps = 1

        # Unwrap states
        X = x[0]
        # unwrap inputs

        dX = -X - u * X

        model.set_equations(ode=dX)

        model.setup(dt=sampling_interval)

        # Initial conditions
        X0 = 5

        # Model initial conditions
        model.set_initial_conditions(x0=X0)

        # Moving Horizon Estimator
        x_lb = [0]
        x_ub = [6]

        mhe = MHE(model)
        # mhe.has_state_noise = True
        mhe.horizon = 10
        mhe.quad_stage_cost.add_measurements(weights=[10])
        mhe.quad_arrival_cost.add_states(weights=[10], guess=X0)
        mhe.set_box_constraints(x_lb=x_lb, x_ub=x_ub)
        mhe.setup()

        for step in range(n_steps):
            model.simulate(u=0.4)
            y_meas = model.solution['y'][-2]
            mhe.add_measurements(y_meas, u_meas=0.4)
            x0, p0 = mhe.estimate()
            if x0 is not None:
                print(f"Estimated: {x0}, real: {model.solution['x'][-2]}")

    def test_2(self):
        """
        Runs MHE, with a very symple system. one state and one  parameter, no state and measurement noise, not model mismatch
        :return:
        """
        model = Model(plot_backend='bokeh')
        x = model.set_dynamical_states(['x0'])
        p = model.set_parameters(['k1'])
        model.set_measurement_equations(x)
        model.set_measurements(['y1'])
        sampling_interval = 0.5
        n_steps = 2

        # Unwrap states
        X = x[0]
        # Unwrap parameters
        k1 = p[0]

        dX = 3 - X - k1 * X

        model.set_dynamical_equations([dX])

        model.setup(dt=sampling_interval)

        # Initial conditions
        X0 = 5

        # Model initial conditions
        x0 = np.array([X0])
        p0_real = [1]
        model.set_initial_conditions(x0=x0)
        sol = model.solution

        # Moving Horizon Estimator
        x_lb = [0]
        x_ub = [6]
        p_ub = [5]
        p_lb = [0]

        mhe = MHE(model)
        # mhe.has_state_noise = True
        mhe.horizon = 10
        mhe.quad_arrival_cost.add_states(weights=[10], guess=x0)
        mhe.quad_arrival_cost.add_parameters(weights=[10], guess=p0_real)
        mhe.quad_stage_cost.add_measurements(weights=[10])
        # mhe.quad_stage_cost.add_state_noise(weights=[10, 10])
        mhe.set_box_constraints(x_lb=x_lb, x_ub=x_ub, p_lb=p_lb, p_ub=p_ub)  # , w_lb=[0, 0], w_ub=[0, 0])
        mhe.setup()

        for step in range(n_steps):
            model.simulate(p=p0_real)
            y_meas = model.solution['y'][-2]
            mhe.add_measurements(y_meas)
            x0, p0 = mhe.estimate()
            if x0 is not None:
                print(f"Estimated: {x0}, real: {model.solution['x'][-2]}")
                print(f"Estimated: {p0}, real: {p0_real}")

    def test_3(self):
        """
        Runs MHE, with a very symple system. one state and one  parameter, state noise
        :return:
        """
        model = Model(plot_backend='bokeh')
        x = model.set_dynamical_states(['x0'])
        p = model.set_parameters(['k1'])
        model.set_measurement_equations(x)
        model.set_measurements(['y1'])
        sampling_interval = 0.5
        n_steps = 2

        # Unwrap states
        X = x[0]
        # Unwrap parameters
        k1 = p[0]

        dX = 3 - X - k1 * X

        model.set_dynamical_equations([dX])

        model.setup(dt=sampling_interval)

        # Initial conditions
        X0 = 5

        # Model initial conditions
        x0 = np.array([X0])
        p0_real = [1]
        model.set_initial_conditions(x0=x0)

        # Moving Horizon Estimator
        x_lb = [0]
        x_ub = [6]
        p_ub = [5]
        p_lb = [0]

        mhe = MHE(model)
        mhe.has_state_noise = True
        mhe.horizon = 10
        mhe.quad_arrival_cost.add_states(weights=[10], guess=x0)
        mhe.quad_arrival_cost.add_parameters(weights=[10], guess=p0_real)
        mhe.quad_stage_cost.add_measurements(weights=[10])
        mhe.quad_stage_cost.add_state_noise(weights=[10])
        mhe.set_box_constraints(x_lb=x_lb, x_ub=x_ub, p_lb=p_lb, p_ub=p_ub, w_lb=[0], w_ub=[0.1])
        mhe.setup()

        for step in range(n_steps):
            model.simulate(p=p0_real)
            y_meas = model.solution['y'][-2]
            mhe.add_measurements(y_meas)
            x0, p0 = mhe.estimate()
            if x0 is not None:
                print(f"Estimated: {x0}, real: {model.solution['x'][-2]}")
                print(f"Estimated: {p0}, real: {p0_real}")

    def test_4(self):
        """
        Runs MHE, with symple system. 2 states and 2  parameters, state noise
        :return:
        """
        model = Model(plot_backend='bokeh')
        x = model.set_dynamical_states(['x0', 'x1'])
        p = model.set_parameters(['k1', 'k2'])
        model.set_measurement_equations(x)
        model.set_measurements(['y1', 'y2'])
        sampling_interval = 0.5
        n_steps = 2

        # Unwrap states
        X = x[0]
        S = x[1]
        # Unwrap parameters
        k1 = p[0]
        k2 = p[1]

        dX = 3 - X - k1 * X
        dS = 3 - S - k1 * S

        model.set_dynamical_equations([dX, dS])

        model.setup(dt=sampling_interval)

        # Initial conditions
        X0 = 5
        S0 = 3
        # Model initial conditions
        x0 = np.array([X0, S0])
        p0_real = [1, 0.5]
        model.set_initial_conditions(x0=x0)

        # Moving Horizon Estimator
        x_lb = [0, 0]
        x_ub = [6, 5]
        p_ub = [2, 2]
        p_lb = [0, 0]

        mhe = MHE(model)
        mhe.has_state_noise = True
        mhe.horizon = 10
        mhe.quad_arrival_cost.add_states(weights=[10, 10], guess=x0)
        mhe.quad_arrival_cost.add_parameters(weights=[10, 10], guess=p0_real)
        mhe.quad_stage_cost.add_measurements(weights=[10, 10])
        mhe.quad_stage_cost.add_state_noise(weights=[10, 10])
        mhe.set_box_constraints(x_lb=x_lb, x_ub=x_ub, p_lb=p_lb, p_ub=p_ub, w_lb=[0, 0], w_ub=[0, 0])
        mhe.setup()

        for step in range(n_steps):
            model.simulate(p=p0_real)
            y_meas = model.solution['y'][:,-2]
            mhe.add_measurements(y_meas)
            x0, p0 = mhe.estimate()
            if x0 is not None:
                print(f"Estimated: {x0}, real: {model.solution['x'][-2]}")
                print(f"Estimated: {p0}, real: {p0_real}")
                pass

    def test_5(self):
        """
        Runs MHE, with symple system. 2 states and 2  parameters, state noise. Test scaling
        :return:
        """
        model = Model(plot_backend='bokeh')
        x = model.set_dynamical_states(['x0', 'x1'])
        p = model.set_parameters(['k1', 'k2'])
        model.set_measurement_equations(x)
        model.set_measurements(['y1', 'y2'])
        sampling_interval = 0.5
        n_steps = 2

        # Unwrap states
        X = x[0]
        S = x[1]
        # Unwrap parameters
        k1 = p[0]
        k2 = p[1]

        dX = 3 - X - k1 * X
        dS = 3 - S - k2 * S

        model.set_dynamical_equations([dX, dS])

        model.setup(dt=sampling_interval)

        # Initial conditions
        X0 = 5
        S0 = 3
        # Model initial conditions
        x0 = np.array([X0, S0])
        p0_real = [1, 0.5]
        model.set_initial_conditions(x0=x0)

        # Moving Horizon Estimator
        x_lb = [0, 0]
        x_ub = [6, 5]
        p_ub = [2, 2]
        p_lb = [0, 0]

        mhe = MHE(model)
        mhe.has_state_noise = True
        mhe.horizon = 10
        mhe.quad_arrival_cost.add_states(weights=[10, 10], guess=x0)
        mhe.quad_arrival_cost.add_parameters(weights=[10, 10], guess=p0_real)
        mhe.quad_stage_cost.add_measurements(weights=[10, 10])
        mhe.quad_stage_cost.add_state_noise(weights=[10, 10])
        mhe.set_scaling(x_scaling=[10] * 2, p_scaling=[10, 10])
        mhe.set_box_constraints(x_lb=x_lb, x_ub=x_ub, p_lb=p_lb, p_ub=p_ub, w_lb=[0, 0], w_ub=[0, 0])
        mhe.setup()

        for step in range(n_steps):
            model.simulate(p=p0_real)
            y_meas = model.solution['y'][:, -2]
            mhe.add_measurements(y_meas)
            x0, p0 = mhe.estimate()
            if x0 is not None:
                print(f"Estimated: {x0}, real: {model.solution['x'][:, -2]}")
                print(f"Estimated: {p0}, real: {p0_real}")
                pass

    @skip("Deprecated")
    def test_bio_MHE(self):
        #TODO add ecoli_D1210_fedbatch_plant to the models
        from hilo_mpc.library.models import ecoli_D1210_fedbatch

        model = ecoli_D1210_fedbatch()
        model.set_measurement_equations(model.x)
        model.set_measurements([f'y_{i}' for i in model.dynamical_state_names])

        # Initial conditions
        X0 = 0.1
        S0 = 40.
        P0 = 0.
        I0 = 0.
        V0 = 1.
        ISF0 = 0
        IRF0 = 0
        # Model initial conditions
        x0 = [X0, S0, P0, I0, ISF0, IRF0, V0]
        p0 = [10, 5]
        x_lb = [0, 0, 0, 0, 0, 0, 0]
        x_ub = [10, 100, 100, 100, 10, 10, 15]
        p_ub = [20, 10]
        p_lb = [0, 0]

        # MPC sampling interval
        sampling_interval = 0.5
        model.setup()
        model.set_initial_conditions(x0=x0)
        n_steps = 2
        # Number of steps
        mhe = MHE(model)
        mhe.horizon = 20
        # mhe.has_state_noise = True
        mhe.quad_arrival_cost.add_states(weights=[10] * 7, guess=x0)
        mhe.quad_arrival_cost.add_parameters(weights=[10] * 2, guess=p0)
        mhe.quad_stage_cost.add_measurements(weights=[10] * 7)
        # mhe.quad_stage_cost.add_state_noise(weights=[10] * 7)
        mhe.set_box_constraints(x_lb=x_lb, x_ub=x_ub, p_lb=p_lb, p_ub=p_ub,
                                w_lb=[-0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1],
                                w_ub=[0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
        mhe.set_initial_guess(x_guess=x0, w_guess=[0, 0, 0, 0, 0, 0, 0], p_guess=p0)

        mhe.setup()
        u = [0.1, 0.1]
        for step in range(n_steps):
            model.simulate(p=p0, u=u)
            y_meas = model.solution['y'][:, -2]
            mhe.add_measurements(y_meas, u_meas=u)
            x0_est, p0_est = mhe.estimate()
            if x0_est is not None:
                print(f"Estimated: {x0_est}, real: {model.solution['x'][:, -2]}")
                print(f"Estimated: {p0_est}, real: {p0}")
        #
        # mhe.solution.plot(
        #     ('t', 'X'),
        #     ('t', 'Sf'),
        #     ('t', 'If'),
        #     subplots=True,
        #     title=['X0', 'Sf', 'IF'],
        #     xlabel=None,
        #     legend=False)

    def test_chemical_reaction(self):
        """
        Test rk4 method
        :return:
        """
        import time

        from hilo_mpc import Model, MHE

        from bokeh.io import output_file, show
        from bokeh.plotting import figure
        from bokeh.layouts import gridplot
        import numpy as np

        # Create model
        model = Model(plot_backend='bokeh')
        x = model.set_dynamical_states(['Ca', 'Cb', 'Cc'], units=['mol/l', 'mol/l', 'mol/l'],
                                       short_description=['Ca', 'Cb', 'Cc'])
        model.set_measurements(['P'], units=['atm'], short_description=['Pressure'])

        # Unwrap states
        Ca = x[0]
        Cb = x[1]
        Cc = x[2]

        # Known Parameters
        k1 = 0.5
        k_1 = 0.05
        k2 = 0.2
        k_2 = 0.01
        dt = 0.25
        RT = 32.84  # L atm/ (mol)

        dCa = - (k1 * Ca - k_1 * Cb * Cc)
        dCb = k1 * Ca - k_1 * Cb * Cc - 2 * (k2 * Cb ** 2 - k_2 * Cc)
        dCc = k1 * Ca - k_1 * Cb * Cc + 1 * (k2 * Cb ** 2 - k_2 * Cc)

        model.set_measurement_equations(RT * (Ca + Cb + Cc))
        model.set_dynamical_equations([dCa, dCb, dCc])

        model.setup(dt=dt)

        # Initial conditions
        x0_real = [0.5, 0.05, 0]
        x0_est = [1, 0, 4]

        model.set_initial_conditions(x0=x0_real)

        n_steps = 2

        # Setup the MHE
        mhe = MHE(model)
        mhe.quad_arrival_cost.add_states(weights=[1 / (0.5 ** 2), 1 / (0.5 ** 2), 1 / (0.5 ** 2)], guess=x0_est)
        mhe.quad_stage_cost.add_measurements(weights=[1 / (0.25 ** 2)])
        mhe.quad_stage_cost.add_state_noise(weights=[1 / (0.001 ** 2), 1 / (0.001 ** 2), 1 / (0.001 ** 2)])
        mhe.set_box_constraints(x_lb=[0, 0, 0])
        mhe.horizon = 20
        mhe.setup(options={'print_level': 0})

        # Run the simulation
        for i in range(n_steps):
            model.simulate()
            mhe.add_measurements(y_meas=model.solution['y'][:, -2])
            x_est, _ = mhe.estimate()

        # p_tot = []
        # p = figure(background_fill_color="#fafafa")
        # p.scatter(x=np.array(mhe.solution['t']).squeeze(), y=np.array(mhe.solution['Ca']).squeeze(),
        #           legend_label='Estimated')
        # p.line(x=np.array(model.solution['t']).squeeze(), y=np.array(model.solution['Ca']).squeeze(),
        #        legend_label='Real')
        # p_tot.append(p)
        #
        # p = figure(background_fill_color="#fafafa")
        # p.scatter(x=np.array(mhe.solution['t']).squeeze(), y=np.array(mhe.solution['Cb']).squeeze(),
        #           legend_label='Estimated')
        # p.line(x=np.array(model.solution['t']).squeeze(), y=np.array(model.solution['Cb']).squeeze(),
        #        legend_label='Real')
        # p_tot.append(p)
        #
        # p = figure(background_fill_color="#fafafa")
        # p.scatter(x=np.array(mhe.solution['t']).squeeze(), y=np.array(mhe.solution['Cc']).squeeze(),
        #           legend_label='Estimated')
        # p.line(x=np.array(model.solution['t']).squeeze(), y=np.array(model.solution['Cc']).squeeze(),
        #        legend_label='Real')
        # p_tot.append(p)
        #
        # show(gridplot(p_tot, ncols=3))

    def test_chemical_reaction_2(self):
        """
        Test multiple shooting
        :return:
        """
        import time

        from hilo_mpc import Model, MHE

        from bokeh.io import output_file, show
        from bokeh.plotting import figure
        from bokeh.layouts import gridplot
        import numpy as np

        # Create model
        model = Model(plot_backend='bokeh')
        x = model.set_dynamical_states(['Ca', 'Cb', 'Cc'], units=['mol/l', 'mol/l', 'mol/l'],
                                       short_description=['Ca', 'Cb', 'Cc'])
        model.set_measurements(['P'], units=['atm'], short_description=['Pressure'])

        # Unwrap states
        Ca = x[0]
        Cb = x[1]
        Cc = x[2]

        # Known Parameters
        k1 = 0.5
        k_1 = 0.05
        k2 = 0.2
        k_2 = 0.01
        dt = 0.25
        RT = 32.84  # L atm/ (mol)

        dCa = - (k1 * Ca - k_1 * Cb * Cc)
        dCb = k1 * Ca - k_1 * Cb * Cc - 2 * (k2 * Cb ** 2 - k_2 * Cc)
        dCc = k1 * Ca - k_1 * Cb * Cc + 1 * (k2 * Cb ** 2 - k_2 * Cc)

        model.set_measurement_equations(RT * (Ca + Cb + Cc))
        model.set_dynamical_equations([dCa, dCb, dCc])

        model.setup(dt=dt)

        # Initial conditions
        x0_real = [0.5, 0.05, 0]
        x0_est = [1, 0, 4]

        model.set_initial_conditions(x0=x0_real)

        n_steps = 2

        # Setup the MHE
        mhe = MHE(model)
        mhe.quad_arrival_cost.add_states(weights=[1 / (0.5 ** 2), 1 / (0.5 ** 2), 1 / (0.5 ** 2)], guess=x0_est)
        mhe.quad_stage_cost.add_measurements(weights=[1 / (0.25 ** 2)])
        mhe.quad_stage_cost.add_state_noise(weights=[1 / (0.001 ** 2), 1 / (0.001 ** 2), 1 / (0.001 ** 2)])
        mhe.set_box_constraints(x_lb=[0, 0, 0])
        mhe.horizon = 20
        mhe.set_nlp_options({'integration_method': 'multiple_shooting'})
        mhe.setup(options={'print_level': 0})

        # Run the simulation
        for i in range(n_steps):
            model.simulate()
            mhe.add_measurements(y_meas=model.solution['y'][:, -2])
            x_est, _ = mhe.estimate()

        # p_tot = []
        # p = figure(background_fill_color="#fafafa")
        # p.scatter(x=np.array(mhe.solution['t']).squeeze(), y=np.array(mhe.solution['Ca']).squeeze(),
        #           legend_label='Estimated')
        # p.line(x=np.array(model.solution['t']).squeeze(), y=np.array(model.solution['Ca']).squeeze(),
        #        legend_label='Real')
        # p_tot.append(p)
        #
        # p = figure(background_fill_color="#fafafa")
        # p.scatter(x=np.array(mhe.solution['t']).squeeze(), y=np.array(mhe.solution['Cb']).squeeze(),
        #           legend_label='Estimated')
        # p.line(x=np.array(model.solution['t']).squeeze(), y=np.array(model.solution['Cb']).squeeze(),
        #        legend_label='Real')
        # p_tot.append(p)
        #
        # p = figure(background_fill_color="#fafafa")
        # p.scatter(x=np.array(mhe.solution['t']).squeeze(), y=np.array(mhe.solution['Cc']).squeeze(),
        #           legend_label='Estimated')
        # p.line(x=np.array(model.solution['t']).squeeze(), y=np.array(model.solution['Cc']).squeeze(),
        #        legend_label='Real')
        # p_tot.append(p)
        #
        # show(gridplot(p_tot, ncols=3))

    def test_chemical_reaction_3(self):
        """
        Test discrete system
        :return:
        """


        # Create model
        model = Model(plot_backend='bokeh')
        x = model.set_dynamical_states(['Ca', 'Cb', 'Cc'], units=['mol/l', 'mol/l', 'mol/l'],
                                       short_description=['Ca', 'Cb', 'Cc'])
        model.set_measurements(['P'], units=['atm'], short_description=['Pressure'])

        # Unwrap states
        Ca = x[0]
        Cb = x[1]
        Cc = x[2]

        # Known Parameters
        k1 = 0.5
        k_1 = 0.05
        k2 = 0.2
        k_2 = 0.01
        dt = 0.25
        RT = 32.84  # L atm/ (mol)

        dCa = - (k1 * Ca - k_1 * Cb * Cc)
        dCb = k1 * Ca - k_1 * Cb * Cc - 2 * (k2 * Cb ** 2 - k_2 * Cc)
        dCc = k1 * Ca - k_1 * Cb * Cc + 1 * (k2 * Cb ** 2 - k_2 * Cc)

        model.set_measurement_equations(RT * (Ca + Cb + Cc))
        model.set_dynamical_equations([dCa, dCb, dCc])

        model_disc = model.discretize('rk4')
        model.setup(dt=dt)

        # Initial conditions
        x0_real = [0.5, 0.05, 0]
        x0_est = [1, 0, 4]

        model.set_initial_conditions(x0=x0_real)

        n_steps = 2

        # Setup the MHE
        model_disc.setup(dt=dt)
        mhe = MHE(model_disc)
        mhe.quad_arrival_cost.add_states(weights=[1 / (0.5 ** 2), 1 / (0.5 ** 2), 1 / (0.5 ** 2)], guess=x0_est)
        mhe.quad_stage_cost.add_measurements(weights=[1 / (0.25 ** 2)])
        mhe.quad_stage_cost.add_state_noise(weights=[1 / (0.001 ** 2), 1 / (0.001 ** 2), 1 / (0.001 ** 2)])
        mhe.set_box_constraints(x_lb=[0, 0, 0])
        mhe.horizon = 20
        mhe.set_nlp_options({'integration_method': 'multiple_shooting'})
        mhe.setup(options={'print_level': 0})

        # Run the simulation
        for i in range(n_steps):
            model.simulate()
            mhe.add_measurements(y_meas=model.solution['y'][:, -2])
            x_est, _ = mhe.estimate()

        # p_tot = []
        # p = figure(background_fill_color="#fafafa")
        # p.scatter(x=np.array(mhe.solution['t']).squeeze(), y=np.array(mhe.solution['Ca']).squeeze(),
        #           legend_label='Estimated')
        # p.line(x=np.array(model.solution['t']).squeeze(), y=np.array(model.solution['Ca']).squeeze(),
        #        legend_label='Real')
        # p_tot.append(p)
        #
        # p = figure(background_fill_color="#fafafa")
        # p.scatter(x=np.array(mhe.solution['t']).squeeze(), y=np.array(mhe.solution['Cb']).squeeze(),
        #           legend_label='Estimated')
        # p.line(x=np.array(model.solution['t']).squeeze(), y=np.array(model.solution['Cb']).squeeze(),
        #        legend_label='Real')
        # p_tot.append(p)
        #
        # p = figure(background_fill_color="#fafafa")
        # p.scatter(x=np.array(mhe.solution['t']).squeeze(), y=np.array(mhe.solution['Cc']).squeeze(),
        #           legend_label='Estimated')
        # p.line(x=np.array(model.solution['t']).squeeze(), y=np.array(model.solution['Cc']).squeeze(),
        #        legend_label='Real')
        # p_tot.append(p)
        #
        # show(gridplot(p_tot, ncols=3))

    def test_chemical_reaction_4_stage_const(self):
        """
        Test nonlinear stage constraints
        :return:
        """


        # Create model
        model = Model(plot_backend='bokeh')
        x = model.set_dynamical_states(['Ca', 'Cb', 'Cc'], units=['mol/l', 'mol/l', 'mol/l'],
                                       short_description=['Ca', 'Cb', 'Cc'])
        model.set_measurements(['P'], units=['atm'], short_description=['Pressure'])

        # Unwrap states
        Ca = x[0]
        Cb = x[1]
        Cc = x[2]

        # Known Parameters
        k1 = 0.5
        k_1 = 0.05
        k2 = 0.2
        k_2 = 0.01
        dt = 0.25
        RT = 32.84  # L atm/ (mol)

        dCa = - (k1 * Ca - k_1 * Cb * Cc)
        dCb = k1 * Ca - k_1 * Cb * Cc - 2 * (k2 * Cb ** 2 - k_2 * Cc)
        dCc = k1 * Ca - k_1 * Cb * Cc + 1 * (k2 * Cb ** 2 - k_2 * Cc)

        model.set_measurement_equations(RT * (Ca + Cb + Cc))
        model.set_dynamical_equations([dCa, dCb, dCc])

        model.setup(dt=dt)

        # Initial conditions
        x0_real = [0.5, 0.05, 0]
        x0_est = [1, 0, 4]

        model.set_initial_conditions(x0=x0_real)

        n_steps = 2

        # Setup the MHE
        mhe = MHE(model)
        mhe.quad_arrival_cost.add_states(weights=[1 / (0.5 ** 2), 1 / (0.5 ** 2), 1 / (0.5 ** 2)], guess=x0_est)
        mhe.quad_stage_cost.add_measurements(weights=[1 / (0.25 ** 2)])
        mhe.quad_stage_cost.add_state_noise(weights=[1 / (0.001 ** 2), 1 / (0.001 ** 2), 1 / (0.001 ** 2)])
        mhe.set_box_constraints(x_lb=[0, 0, 0])
        mhe.horizon = 20
        mhe.stage_constraint.constraint = model.x[0]
        mhe.stage_constraint.lb = 0.1
        mhe.stage_constraint.ub = 5
        mhe.setup(options={'print_level': 0})

        # Run the simulation
        for i in range(n_steps):
            model.simulate()
            mhe.add_measurements(y_meas=model.solution['y'][:, -2])
            x_est, _ = mhe.estimate()

        # p_tot = []
        # p = figure(background_fill_color="#fafafa")
        # p.scatter(x=np.array(mhe.solution['t']).squeeze(), y=np.array(mhe.solution['Ca']).squeeze(),
        #           legend_label='Estimated')
        # p.line(x=np.array(model.solution['t']).squeeze(), y=np.array(model.solution['Ca']).squeeze(),
        #        legend_label='Real')
        # p_tot.append(p)
        #
        # p = figure(background_fill_color="#fafafa")
        # p.scatter(x=np.array(mhe.solution['t']).squeeze(), y=np.array(mhe.solution['Cb']).squeeze(),
        #           legend_label='Estimated')
        # p.line(x=np.array(model.solution['t']).squeeze(), y=np.array(model.solution['Cb']).squeeze(),
        #        legend_label='Real')
        # p_tot.append(p)
        #
        # p = figure(background_fill_color="#fafafa")
        # p.scatter(x=np.array(mhe.solution['t']).squeeze(), y=np.array(mhe.solution['Cc']).squeeze(),
        #           legend_label='Estimated')
        # p.line(x=np.array(model.solution['t']).squeeze(), y=np.array(model.solution['Cc']).squeeze(),
        #        legend_label='Real')
        # p_tot.append(p)
        #
        # show(gridplot(p_tot, ncols=3))

class TestTimeVariantSys(unittest.TestCase):

    def setUp(self) -> None:
        model = Model(plot_backend='bokeh')
        # Constants
        k = 1e-5

        # States (internal temperature)
        T = model.set_dynamical_states('T')
        model.set_measurements(['yT'])
        model.set_measurement_equations([T])
        # Inputs
        F = 0
        # time
        t = model.time
        # Extenral temperature
        T_ex = ca.sin(2 * ca.pi * t / (3600 * 24)) * 5 + 10
        # ODE
        dT = F - k * (T - T_ex)

        model.set_equations(ode=[dT])

        # Initial conditions
        x0 = 14
        u0 = 0.

        # Create model and run simulation
        dt = 900  # seconds
        model.setup(dt=dt)

        self.model = model
        self.dt = dt
        self.x0 = x0


    def test_heat_exchange(self):
        model = self.model
        x0_real = self.x0
        dt = self.dt
        t0 = 0

        model.setup(dt=dt)

        # Initial conditions
        x0_est = 22

        model.set_initial_conditions(x0=x0_real)

        n_steps = 2

        # Setup the MHE
        mhe = MHE(model)
        mhe.quad_arrival_cost.add_states(weights=1, guess=x0_est)
        mhe.quad_stage_cost.add_measurements(weights=2)
        mhe.quad_stage_cost.add_state_noise(weights=20)
        mhe.initial_time = t0
        mhe.horizon = 20
        mhe.setup()

        # Run the simulation
        for i in range(n_steps):
            model.simulate()
            mhe.add_measurements(y_meas=model.solution['y'][:, -2]+2*(np.random.rand(1)-0.5))
            x_est, _ = mhe.estimate()

        # p_tot = []
        # p = figure(background_fill_color="#fafafa")
        # p.scatter(x=np.array(mhe.solution['t']).squeeze(), y=np.array(mhe.solution['T']).squeeze(),
        #           legend_label='Estimated')
        # p.line(x=np.array(model.solution['t']).squeeze(), y=np.array(model.solution['T']).squeeze(),
        #        legend_label='Real')
        # p_tot.append(p)
        #
        # show(gridplot(p_tot, ncols=3))

if __name__ == '__main__':
    unittest.main()
