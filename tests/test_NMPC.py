from unittest import TestCase, skip

import casadi as ca
import numpy as np

from hilo_mpc import NMPC, Model, SimpleControlLoop


class TestNMPC(TestCase):

    def setUp(self) -> None:

        model = Model(plot_backend='bokeh')
        # Constants
        M = 5.
        m = 1.
        l = 1.
        h = .5
        g = 9.81

        # States and algebraic variables
        x = model.set_dynamical_states(['x', 'v', 'theta', 'omega'])
        model.set_measurements(['yx', 'yv', 'ytheta', 'tomega'])
        model.set_measurement_equations([x[0], x[1], x[2], x[3]])
        # y = model.set_algebraic_variables(['y'])
        v = x[1]
        theta = x[2]
        omega = x[3]
        # Inputs
        F = model.set_inputs('F')

        # ODE
        dx = v
        dv = 1. / (M + m - m * ca.cos(theta)) * (m * g * ca.sin(theta) - m * l * ca.sin(theta) * omega ** 2 + F)
        dtheta = omega
        domega = 1. / l * (dv * ca.cos(theta) + g * ca.sin(theta))

        model.set_equations(ode=[dx, dv, dtheta, domega])

        # Initial conditions
        x0 = [2.5, 0., 0.1, 0.]
        z0 = ca.sqrt(3.) / 2.
        u0 = 0.

        # Create model and run simulation
        dt = .1
        model.setup(dt=dt)

        self.model = model
        self.dt = dt
        self.x0 = x0
        self.u0 = u0

        # may_term = (model.x[3]-ca.pi)*100*(model.x[3]-ca.pi)
        # lag_term = (model.x[3]-ca.pi)*10*(model.x[3]-ca.pi) #+ model.u*0.01*model.u

    def test_closed_loop_b(self):
        " Test normal nonlinear MPC for using a pendulum model. This test checks the normal problem setup"
        x0 = self.x0
        u0 = self.u0
        model = self.model
        nmpc = NMPC(model)
        nmpc.quad_stage_cost.add_states(names=['v', 'theta'], ref=[0, 0], weights=[10, 5])
        nmpc.quad_stage_cost.add_inputs(names='F', weights=0.1)
        nmpc.horizon = 25
        nmpc.set_box_constraints(x_ub=[5, 10, 10, 10], x_lb=[-5, -10, -10, -10])
        nmpc.set_initial_guess(x_guess=x0, u_guess=u0)
        nmpc.setup()

        n_steps = 10
        model.set_initial_conditions(x0=x0)
        scl = SimpleControlLoop(model, nmpc)
        scl.run(steps=n_steps)
        # scl.plot()

    def test_closed_loop_c(self):
        " Test normal nonlinear MPC for using a pendulum model. This test checks the normal problem setup"
        x0 = self.x0
        u0 = self.u0
        model = self.model
        nmpc = NMPC(model)
        nmpc.quad_stage_cost.add_states(names=['v', 'theta'], ref=[0, 0], weights=[10, 5])
        nmpc.quad_stage_cost.add_inputs(names='F', weights=0.1)
        nmpc.prediction_horizon = 25
        nmpc.control_horizon = 25
        nmpc.set_box_constraints(x_ub=[5, 10, 10, 10], x_lb=[-5, -10, -10, -10])
        nmpc.set_initial_guess(x_guess=x0, u_guess=u0)
        nmpc.setup()

        n_steps = 1
        model.set_initial_conditions(x0=x0)
        scl = SimpleControlLoop(model, nmpc)
        scl.run(steps=n_steps)
        # scl.plot()

    def test_optimize_initial_conditions(self):
        " In this test the initialization of the initial conditions for the MPC is tested."
        x0 = self.x0
        u0 = self.u0
        model = self.model
        nmpc = NMPC(model)
        nmpc.quad_stage_cost.add_states(names=['v', 'theta'], ref=[0, 0], weights=[10, 5])
        nmpc.quad_stage_cost.add_inputs(names='F', weights=0.1)
        nmpc.horizon = 25
        nmpc.set_box_constraints(x_ub=[5, 10, 10, 10], x_lb=[-5, -10, -10, -10])
        nmpc.set_initial_guess(x_guess=x0, u_guess=u0)
        nmpc.setup()

        model.set_initial_conditions(x0=x0)

        _ = nmpc.optimize(x0, fix_x0=False, x0_lb=[-5, -10, -10, -10], x0_ub=[5, 10, 10, 10])
        x_pred, u_pred, _ = nmpc.return_prediction()

    def test_multi_run(self):
        """
        Test if the multirung approach, where the MPC is computed multimple time at the same time, with different intial
        guesses for the optimizer
        PASSED - still is not bullet proof because initial guess could be outside constraints or oen can have problems with the ODE solver
        """
        x0 = self.x0
        u0 = self.u0
        model = self.model
        nmpc = NMPC(model)
        nmpc.quad_stage_cost.add_states(names='theta', ref=ca.pi, weights=10)
        nmpc.horizon = 10
        nmpc.set_box_constraints(x_ub=[3, 0.5, 10, 10], x_lb=[2, 0, 0, 0])
        nmpc.set_initial_guess(x_guess=x0, u_guess=u0)
        nmpc.setup()

        n_steps = 10
        # Initial conditions
        model.set_initial_conditions(x0=x0)
        sol = model.solution

        for step in range(n_steps):
            u = nmpc.optimize(x0, runs=3, pert_factor=0.1)
            model.simulate(u=u)
            x0 = sol['x:f']

    def test_scaling(self):
        """
        Test the scaling of states and inputs for the MPC - nominal case
        :return:
        """
        x0 = self.x0
        u0 = self.u0
        model = self.model
        nmpc = NMPC(model)
        nmpc.quad_stage_cost.add_states(names='theta', ref=ca.pi, weights=10)

        nmpc.quad_stage_cost.add_inputs(names=['F'], weights=np.array([[0.0001]]))
        nmpc.set_scaling(x_scaling=[10, 10, 10, 10], u_scaling=[10])
        nmpc.horizon = 10
        nmpc.set_box_constraints(x_ub=[3, 0.5, 10, 10], x_lb=[2, -0.5, -10, -10])
        nmpc.set_initial_guess(x_guess=x0, u_guess=u0)
        nmpc.setup()

        n_steps = 10
        model.set_initial_conditions(x0=x0)
        sol = model.solution

        for step in range(n_steps):
            u = nmpc.optimize(x0)
            model.simulate(u=u)
            x0 = sol['x:f']
        # model.solution.plot()
        # sol.plot(
        #     ('t', 'x'),
        #     ('t', 'v'),
        #     ('t', 'theta'),
        #     ('t', 'omega'),
        #     subplots=True,
        #     title=sol.get_names('x'),
        #     xlabel=None,
        #     legend=False)

    def test_matrix_in_cost(self):
        """
        Test if giving a matrix in the cost works
        :return:
        """
        x0 = self.x0
        u0 = self.u0
        model = self.model
        nmpc = NMPC(model)  # (x - xr).T Q (x - xr ) + (u - ur).T R (u - ur)
        nmpc.quad_stage_cost.add_states(names='theta', ref=ca.pi, weights=np.array([[10]]))
        nmpc.quad_stage_cost.add_inputs(names=['F'], weights=np.array([[0.0001]]))
        nmpc.horizon = 10
        nmpc.set_box_constraints(x_ub=[3, 0.5, 10, 10], x_lb=[2, -0.5, -10, -10])
        nmpc.set_initial_guess(x_guess=x0, u_guess=u0)
        nmpc.setup()

        n_steps = 1
        model.set_initial_conditions(x0=x0)
        sol = model.solution

        for step in range(n_steps):
            u = nmpc.optimize(x0)
            model.simulate(u=u)
            x0 = sol['x:f']

        # sol.plot(
        #     ('t', 'x'),
        #     ('t', 'v'),
        #     ('t', 'theta'),
        #     ('t', 'omega'),
        #     subplots=True,
        #     title=sol.get_names('x'),
        #     xlabel=None,
        #     legend=False)

    def test_generic_cost(self):
        x0 = self.x0
        u0 = self.u0
        model = self.model
        nmpc = NMPC(model)
        nmpc.stage_cost.cost = 10 * (model.x[2] - ca.pi) ** 2
        nmpc.terminal_cost.cost = 10 * (model.x[2] - ca.pi) ** 2
        nmpc.horizon = 10
        nmpc.set_box_constraints(x_ub=[3, 0.5, 10, 10], x_lb=[2, -0.5, -10, -10])
        nmpc.set_initial_guess(x_guess=x0, u_guess=u0)
        nmpc.setup()

        n_steps = 1
        model.set_initial_conditions(x0=x0)
        sol = model.solution

        for step in range(n_steps):
            u = nmpc.optimize(x0)
            model.simulate(u=u)
            x0 = sol['x:f']

        # sol.plot(
        #     ('t', 'x'),
        #     ('t', 'v'),
        #     ('t', 'theta'),
        #     ('t', 'omega'),
        #     subplots=True,
        #     title=sol.get_names('x'),
        #     xlabel=None,
        #     legend=False)

    def test_terminal_cost(self):
        x0 = self.x0
        u0 = self.u0
        model = self.model
        nmpc = NMPC(model)
        nmpc.quad_terminal_cost.add_states(names='theta', ref=ca.pi, weights=10)
        nmpc.horizon = 10
        nmpc.set_box_constraints(x_ub=[3, 0.5, 10, 10], x_lb=[2, -0.5, -10, -10])
        nmpc.set_initial_guess(x_guess=x0, u_guess=u0)
        nmpc.setup()

        n_steps = 1
        model.set_initial_conditions(x0=x0)
        sol = model.solution

        for step in range(n_steps):
            u = nmpc.optimize(x0)
            model.simulate(u=u)
            x0 = sol['x:f']

        # sol.plot(
        #     ('t', 'x'),
        #     ('t', 'v'),
        #     ('t', 'theta'),
        #     ('t', 'omega'),
        #     subplots=True,
        #     title=sol.get_names('x'),
        #     xlabel=None,
        #     legend=False)

    def test_output_feedback(self):
        x0 = self.x0
        u0 = self.u0
        model = self.model
        nmpc = NMPC(model)
        nmpc.quad_stage_cost.add_measurements(names='ytheta', ref=ca.pi, weights=10)
        nmpc.horizon = 10
        nmpc.set_box_constraints(x_ub=[3, 0.5, 10, 10], x_lb=[2, -0.5, -10, -10])
        nmpc.set_initial_guess(x_guess=x0, u_guess=u0)
        nmpc.setup()

        n_steps = 1
        model.set_initial_conditions(x0=x0)
        sol = model.solution

        for step in range(n_steps):
            u = nmpc.optimize(x0)
            model.simulate(u=u)
            x0 = sol['x:f']

        # sol.plot(
        #     ('t', 'x'),
        #     ('t', 'v'),
        #     ('t', 'theta'),
        #     ('t', 'omega'),
        #     subplots=True,
        #     title=sol.get_names('x'),
        #     xlabel=None,
        #     legend=False)

    def test_output_feedback_2(self):
        """
        Check if output feedback works in the stage_cost
        :return:
        """
        x0 = self.x0
        u0 = self.u0
        model = self.model
        nmpc = NMPC(model)
        nmpc.stage_cost.cost = (model.y[2] - ca.pi) ** 2 * 10
        nmpc.horizon = 10
        nmpc.set_box_constraints(x_ub=[3, 0.5, 10, 10], x_lb=[2, -0.5, -10, -10])
        nmpc.set_initial_guess(x_guess=x0, u_guess=u0)
        nmpc.setup()

        n_steps = 1
        model.set_initial_conditions(x0=x0)
        sol = model.solution

        for step in range(n_steps):
            u = nmpc.optimize(x0)
            model.simulate(u=u)
            x0 = sol['x:f']

        # sol.plot(
        #     ('t', 'x'),
        #     ('t', 'v'),
        #     ('t', 'theta'),
        #     ('t', 'omega'),
        #     subplots=True,
        #     title=sol.get_names('x'),
        #     xlabel=None,
        #     legend=False)

    def test_output_constraints(self):
        """
        Test the output box constraints
        :return:
        """
        x0 = self.x0
        u0 = self.u0
        model = self.model
        nmpc = NMPC(model)
        nmpc.quad_stage_cost.add_states(names='theta', ref=ca.pi, weights=10)
        nmpc.horizon = 10
        nmpc.set_box_constraints(y_ub=[3, 0.5, 10, 10], y_lb=[2, -0.5, -10, -10])
        nmpc.set_initial_guess(x_guess=x0, u_guess=u0)
        nmpc.setup()

        n_steps = 1
        model.set_initial_conditions(x0=x0)
        sol = model.solution

        for step in range(n_steps):
            u = nmpc.optimize(x0)
            model.simulate(u=u, steps=1)
            x0 = sol['x:f']

        # sol.plot(
        #     ('t', 'x'),
        #     ('t', 'v'),
        #     ('t', 'theta'),
        #     ('t', 'omega'),
        #     subplots=True,
        #     title=sol.get_names('x'),
        #     xlabel=None,
        #     legend=False)

    def test_input_change_cost(self):
        " Test normal nonlinear MPC for using a pendulum model. This test checks the normal problem setup"
        x0 = self.x0
        u0 = self.u0
        model = self.model
        nmpc = NMPC(model)
        nmpc.quad_stage_cost.add_states(names=['v', 'theta'], ref=[0, 0], weights=[10, 5])
        nmpc.quad_stage_cost.add_inputs(names='F', weights=0.1)
        # nmpc.quad_stage_cost.add_inputs_change(names='F', weights=2)
        nmpc.horizon = 10
        nmpc.set_box_constraints(x_ub=[5, 10, 10, 10], x_lb=[-5, -10, -10, -10])
        nmpc.set_initial_guess(x_guess=x0, u_guess=u0)
        nmpc.setup()

        n_steps = 1
        model.set_initial_conditions(x0=x0)
        sol = model.solution

        for step in range(n_steps):
            u = nmpc.optimize(x0)
            model.simulate(u=u, steps=1)
            x0 = sol['x:f']

        # sol.plot(
        #     ('t', 'x'),
        #     ('t', 'v'),
        #     ('t', 'theta'),
        #     ('t', 'omega'),
        #     ('t', 'F'),
        #     subplots=True,
        #     title=sol.get_names('x').extend(sol.get_names('u')),
        #     xlabel=None,
        #     legend=False)

    def test_QRP_matrix(self):

        model = self.model
        nmpc = NMPC(model)
        nmpc.quad_stage_cost.Q = np.array([[1, 1], [1, 1]])
        nmpc.quad_stage_cost.R = np.array([[1, 1], [1, 1]])
        nmpc.quad_terminal_cost.P = np.array([[1, 1], [1, 1]])

        # print(nmpc.quad_stage_cost.Q)
        # print(nmpc.quad_stage_cost.R)
        # print(nmpc.quad_terminal_cost.P)

    def test_multipleshooting_idas(self):
        " Test normal nonlinear MPC for using a pendulum model. This test checks the normal problem setup"
        x0 = self.x0
        u0 = self.u0
        model = self.model
        nmpc = NMPC(model)
        nmpc.quad_stage_cost.add_states(names=['v', 'theta'], ref=[0, 0], weights=[10, 5])
        nmpc.quad_stage_cost.add_inputs(names='F', weights=0.1)
        nmpc.horizon = 25
        nmpc.set_box_constraints(x_ub=[5, 10, 10, 10], x_lb=[-5, -10, -10, -10])
        nmpc.set_initial_guess(x_guess=x0, u_guess=u0)
        nmpc.setup(options={'integration_method': 'idas'})

        n_steps = 1
        model.set_initial_conditions(x0=x0)

        ss = SimpleControlLoop(model, nmpc)
        ss.run(n_steps)
        # ss.plot()

    def text_check_wellposeness_no_horizon(self):
        x0 = self.x0
        u0 = self.u0
        model = self.model
        nmpc = NMPC(model)
        nmpc.quad_stage_cost.add_states(names=['v', 'theta'], ref=[0, 0], weights=[10, 5])
        nmpc.quad_stage_cost.add_inputs(names='F', weights=0.1)
        nmpc.set_box_constraints(x_ub=[5, 10, 10, 10], x_lb=[-5, -10, -10, -10])
        nmpc.set_initial_guess(x_guess=x0, u_guess=u0)
        self.assertRaises(ValueError, nmpc.setup)

    def text_check_wellposeness_no_objective(self):
        x0 = self.x0
        u0 = self.u0
        model = self.model
        nmpc = NMPC(model)
        nmpc.horizon = 10
        nmpc.set_box_constraints(x_ub=[5, 10, 10, 10], x_lb=[-5, -10, -10, -10])
        nmpc.set_initial_guess(x_guess=x0, u_guess=u0)
        self.assertRaises(ValueError, nmpc.setup)


class TestNMPCConstraints(TestCase):
    def setUp(self):

        model = Model(plot_backend='bokeh')
        # Constants
        M = 5.
        m = 1.
        l = 1.
        h = .5
        g = 9.81

        # States and algebraic variables
        x = model.set_dynamical_states(['x', 'v', 'theta', 'omega', 'dummy'])
        # y = model.set_algebraic_variables(['y'])
        v = x[1]
        theta = x[2]
        omega = x[3]
        dummy = x[4]
        # Inputs
        F = model.set_inputs(['F', 'u_dummy'])
        u_dummy = F[1]
        # ODEs
        dd = ca.SX.sym('dd', 5)
        dd[0] = v
        dd[1] = 1. / (M + m - m * ca.cos(theta)) * (m * g * ca.sin(theta) - m * l * ca.sin(theta) * omega ** 2 + F[0])
        dd[2] = omega
        dd[3] = 1. / l * (dd[1] * ca.cos(theta) + g * ca.sin(theta))
        dd[4] = u_dummy

        # Algebraic equations
        # dy = h + l * ca.cos(theta) - y

        model.set_equations(ode=dd)

        # Initial conditions
        x0 = [2.5, 0., 1.5, 0., 0.]
        z0 = ca.sqrt(3.) / 2.
        u0 = [0., 0.]

        # Create model and run simulation
        dt = .1
        model.setup(dt=dt)

        self.model = model
        self.dt = dt
        self.x0 = x0
        self.u0 = u0

        def format_figure(p):
            p.yaxis.axis_label_text_font_size = "18pt"
            p.yaxis.major_label_text_font_size = "18pt"
            p.xaxis.major_label_text_font_size = "18pt"
            p.xaxis.axis_label_text_font_size = "18pt"
            return p

        self.format_figure = format_figure
        # may_term = (model.x[3]-ca.pi)*100*(model.x[3]-ca.pi)
        # lag_term = (model.x[3]-ca.pi)*10*(model.x[3]-ca.pi) #+ model.u*0.01*model.u

    def test_custom_constraint_function(self):
        x0 = self.x0
        u0 = self.u0
        dt = self.dt
        model = self.model

        def custom_fun(v, x_ind, u_ind, sampling_time):
            integral_ = 0
            for k in range(len(x_ind) - 1):
                integral_ += (v[x_ind[k][4]] + v[x_ind[k + 1][4]]) / 2 * sampling_time

            return integral_

        x_ub = [3, 0.5, 10, 10, 10000]
        x_lb = [2, -0.5, -10, -10, 0]

        nmpc = NMPC(model)
        nmpc.quad_stage_cost.add_states(names=['theta', 'dummy'], ref=[ca.pi, 10], weights=[ca.pi, 10])
        nmpc.horizon = 10
        nmpc.set_box_constraints(x_ub=x_ub, x_lb=x_lb)
        nmpc.set_initial_guess(x_guess=x0, u_guess=u0)
        c_fun = lambda v, x_ind, u_ind: custom_fun(v, x_ind, u_ind, dt)
        upper_bound_integral = 4
        nmpc.set_custom_constraints_function(c_fun, ub=upper_bound_integral, lb=0)
        nmpc.setup()
        nmpc.optimize(x0)
        x_opt, _, _ = nmpc.return_prediction()
        # nmpc.plot_mpc_prediction(extras={'x': x_opt[0, :]}, extras_names=['meas'], format_figure=format_figure)
        integral = 0
        for i in range(x_opt[4].shape[0] - 1):
            integral += (x_opt[4][i] + x_opt[4][i + 1]) / 2 * 0.1

        epsilon = 1e-3
        self.assertTrue(integral - upper_bound_integral < epsilon)

    def test_terminal_constraint(self):
        """
        check terminal constraint
        :return:
        """
        x0 = self.x0
        u0 = self.u0
        dt = self.dt
        model = self.model
        nmpc = NMPC(model)
        nmpc.quad_stage_cost.add_states(names=['theta', 'dummy'], ref=[ca.pi, 10], weights=[ca.pi, 10])
        nmpc.quad_terminal_cost.add_states(names=['theta', 'dummy'], ref=[ca.pi, 10], weights=[ca.pi, 10])
        nmpc.horizon = 10
        nmpc.set_box_constraints(x_ub=[3, 0.5, 10, 10, 100], x_lb=[2, -0.5, -10, -10, -100])
        nmpc.set_initial_guess(x_guess=x0, u_guess=u0)
        nmpc.set_terminal_constraints(model.x[4], lb=[4], ub=[5])
        nmpc.setup()
        nmpc.optimize(x0)
        # nmpc.plot_mpc_prediction()

    def test_terminal_constraint_1(self):
        """
        check terminal constraint with scaling
        :return:
        """
        x0 = self.x0
        u0 = self.u0
        dt = self.dt
        model = self.model
        nmpc = NMPC(model)
        nmpc.quad_stage_cost.add_states(names=['theta', 'dummy'], ref=[ca.pi, 10], weights=[ca.pi, 10])
        nmpc.quad_terminal_cost.add_states(names=['theta', 'dummy'], ref=[ca.pi, 10], weights=[ca.pi, 10])
        nmpc.horizon = 10
        nmpc.set_box_constraints(x_ub=[3, 0.5, 10, 10, 100], x_lb=[2, -0.5, -10, -10, -100])
        nmpc.set_initial_guess(x_guess=x0, u_guess=u0)
        nmpc.set_terminal_constraints(model.x[4], lb=[4], ub=[5])
        nmpc.set_scaling(x_scaling=[10, 10, 10, 10, 10], u_scaling=[10., 10])

        nmpc.setup()
        nmpc.optimize(x0)
        # nmpc.plot_mpc_prediction()

    def test_soft_terminal_constraint_1(self):
        """
        check terminal constraint with scaling
        :return:
        """
        x0 = self.x0
        u0 = self.u0
        dt = self.dt
        model = self.model
        nmpc = NMPC(model)
        nmpc.quad_stage_cost.add_states(names=['theta', 'dummy'], ref=[ca.pi, 10], weights=[ca.pi, 10])
        nmpc.quad_terminal_cost.add_states(names=['theta', 'dummy'], ref=[ca.pi, 10], weights=[ca.pi, 10])
        nmpc.horizon = 10
        nmpc.set_box_constraints(x_ub=[3, 0.5, 10, 10, 100], x_lb=[2, -0.5, -10, -10, -100])
        nmpc.set_initial_guess(x_guess=x0, u_guess=u0)
        nmpc.set_terminal_constraints(ca.vertcat(model.x[3], model.x[4]), lb=[3, 4], ub=[4, 5], is_soft=True)
        nmpc.set_scaling(x_scaling=[10, 10, 10, 10, 10], u_scaling=[10., 10])

        nmpc.setup()
        nmpc.optimize(x0)
        # nmpc.plot_mpc_prediction()

    def test_stage_constraint(self):
        """ Test stage constraints with scaling"""
        x0 = self.x0
        u0 = self.u0
        model = self.model
        nmpc = NMPC(model)
        nmpc.quad_stage_cost.add_states(names=['theta', 'dummy'], ref=[ca.pi, 10], weights=[ca.pi, 10])
        nmpc.quad_terminal_cost.add_states(names=['theta', 'dummy'], ref=[ca.pi, 10], weights=[ca.pi, 10])
        nmpc.horizon = 10
        nmpc.set_box_constraints(x_ub=[3, 0.5, 10, 10, 100], x_lb=[2, -0.5, -10, -10, -100])
        nmpc.set_initial_guess(x_guess=x0, u_guess=u0)
        nmpc.set_stage_constraints(model.x[4] * model.u[0], lb=[0], ub=[5])
        nmpc.set_scaling(x_scaling=[10, 10, 10, 10, 10], u_scaling=[10., 10])

        nmpc.setup()
        nmpc.optimize(x0)

    def test_soft_stage_constraint(self):
        """ Test soft stage constraints with"""

        x0 = self.x0
        u0 = self.u0
        model = self.model
        nmpc = NMPC(model)
        nmpc.quad_stage_cost.add_states(names=['theta', 'dummy'], ref=[ca.pi, 10], weights=[ca.pi, 10])
        nmpc.quad_terminal_cost.add_states(names=['theta', 'dummy'], ref=[ca.pi, 10], weights=[ca.pi, 10])
        nmpc.horizon = 10
        nmpc.set_box_constraints(x_ub=[3, 0.5, 10, 10, 100], x_lb=[2, -0.5, -10, -10, -100])
        nmpc.set_initial_guess(x_guess=x0, u_guess=u0)
        nmpc.set_stage_constraints(model.x[4], lb=[0], ub=[5], is_soft=True)
        nmpc.set_scaling(x_scaling=[10, 10, 10, 10, 10], u_scaling=[10., 10])

        nmpc.setup()
        nmpc.optimize(x0)
        # nmpc.plot_prediction(format_figure=self.format_figure)

    def test_soft_stage_constraint_cl(self):
        """ Test soft stage constraints with"""

        x0 = self.x0
        u0 = self.u0
        model = self.model
        model.set_initial_conditions(x0=x0)
        nmpc = NMPC(model)
        nmpc.quad_stage_cost.add_states(names=['theta', 'dummy'], ref=[ca.pi, 10], weights=[ca.pi, 10])
        nmpc.quad_terminal_cost.add_states(names=['theta', 'dummy'], ref=[ca.pi, 10], weights=[ca.pi, 10])
        nmpc.horizon = 10
        nmpc.set_box_constraints(x_ub=[3, 0.5, 10, 10, 100], x_lb=[2, -0.5, -10, -10, -100])
        nmpc.set_initial_guess(x_guess=x0, u_guess=u0)
        nmpc.set_stage_constraints(model.x[4], lb=[0], ub=[5], is_soft=True)
        nmpc.set_scaling(x_scaling=[10, 10, 10, 10, 10], u_scaling=[10., 10])
        nmpc.setup()
        n_steps = 1
        sol = model.solution
        for step in range(n_steps):
            u = nmpc.optimize(x0)
            model.simulate(u=u, steps=1)
            x0 = sol['x:f']

        # sol.plot(
        #     ('t', 'x'),
        #     ('t', 'v'),
        #     ('t', 'theta'),
        #     ('t', 'omega'),
        #     ('t', 'dummy'),
        #     subplots=True,
        #     title=sol.get_names('x'),
        #     xlabel=None,
        #     legend=False)
        # nmpc.plot_mpc_prediction(format_figure=self.format_figure)

    def test_time_varyng_constreaints(self):
        model = Model(plot_backend='bokeh')
        # Constants
        M = 5.

        # States and algebraic variables
        xx = model.set_dynamical_states(['x', 'vx', 'y', 'vy'])
        model.set_measurements(['y_x', 'y_vx', 'y_y', 'y_vy'])
        model.set_measurement_equations([xx[0], xx[1], xx[2], xx[3]])
        x = xx[0]
        vx = xx[1]
        y = xx[2]
        vy = xx[3]
        # Inputs
        F = model.set_inputs(['Fx', 'Fy'])
        Fx = F[0]
        Fy = F[1]
        # ODEs
        dd = ca.SX.sym('dd', 4)
        dd[0] = vx
        dd[1] = Fx / M
        dd[2] = vy
        dd[3] = Fy / M

        # time
        t = model.time
        model.set_equations(ode=dd)

        # Initial conditions
        x0 = [0, 0, 1.5, 0]
        u0 = [0., 0.]

        # Create model and run simulation
        dt = 0.1
        model.setup(dt=dt)

        nmpc = NMPC(model)
        nmpc.horizon = 100
        nmpc.set_initial_guess(x_guess=x0, u_guess=u0)
        nmpc.quad_stage_cost.add_states(names=['x', 'y'], ref=[0, 1.5], weights=[10, 10])
        nmpc.quad_terminal_cost.add_states(names=['x', 'y'], ref=[0, 1.5], weights=[10, 10])
        nmpc.set_stage_constraints(stage_constraint=ca.vertcat(y - ca.sin(t * 10), y - ca.sin(t * 10)), lb=[-ca.inf, 0],
                                   ub=[2, ca.inf])
        nmpc.setup()

        model.set_initial_conditions(x0=x0)

        _ = nmpc.optimize(x0)

        # nmpc.plot_prediction()


class TestTrajectoryPathFollowingMPC(TestCase):
    def setUp(self):
        model = Model(plot_backend='bokeh')
        # Constants
        M = 5.

        # States and algebraic variables
        xx = model.set_dynamical_states(['x', 'vx', 'y', 'vy'])
        model.set_measurements(['y_x', 'y_vx', 'y_y', 'y_vy'])
        model.set_measurement_equations([xx[0], xx[1], xx[2], xx[3]])
        x = xx[0]
        vx = xx[1]
        y = xx[2]
        vy = xx[3]
        # Inputs
        F = model.set_inputs(['Fx', 'Fy'])
        Fx = F[0]
        Fy = F[1]
        # ODEs
        dd1 = vx
        dd2 = Fx / M
        dd3 = vy
        dd4 = Fy / M

        model.set_dynamical_equations([dd1, dd2, dd3, dd4])

        # Initial conditions
        x0 = [0, 0, 0, 0]
        u0 = [0., 0.]

        # Create model and run simulation
        dt = 0.1
        model.setup(dt=dt)

        self.model = model
        self.dt = dt
        self.x0 = x0
        self.u0 = u0

        def format_figure(p):
            p.yaxis.axis_label_text_font_size = "18pt"
            p.yaxis.major_label_text_font_size = "18pt"
            p.xaxis.major_label_text_font_size = "18pt"
            p.xaxis.axis_label_text_font_size = "18pt"
            return p

        self.format_figure = format_figure

    def test_pf_v2(self):
        x0 = self.x0
        u0 = self.u0
        model = self.model
        nmpc = NMPC(model)
        theta = nmpc.create_path_variable()

        nmpc.quad_stage_cost.add_states(names=['x', 'y'], weights=[10, 10],
                                        ref=ca.vertcat(ca.sin(theta), ca.sin(2 * theta)), path_following=True)
        nmpc.quad_terminal_cost.add_states(names=['x', 'y'], weights=[10, 10],
                                           ref=ca.vertcat(ca.sin(theta), ca.sin(2 * theta)), path_following=True)
        nmpc.horizon = 10
        nmpc.set_initial_guess(x_guess=x0, u_guess=u0)
        nmpc.setup()

        n_steps = 1
        model.set_initial_conditions(x0=x0)
        sol = model.solution

        def path(theta):
            return np.sin(theta), np.sin(2 * theta)

        x_path = []
        y_path = []
        for t in range(1000):
            x_p, y_p = path(t / 100)
            x_path.append(x_p)
            y_path.append(y_p)
        #
        # p = figure()
        # p.line(x=x_path, y=y_path, line_color='red', line_dash='dashed')
        # p = self.format_figure(p)
        # p.yaxis.axis_label = "y [m]"
        # p.xaxis.axis_label = "x [m]"
        # show(p)

        for step in range(n_steps):
            u = nmpc.optimize(x0)
            x_pred, _, _ = nmpc.return_prediction()
            model.simulate(u=u)
            x0 = sol['x:f']

        # p = figure()
        # p.line(x=np.array(model.solution['x']).squeeze(), y=np.array(model.solution['y']).squeeze())
        # p.line(x=x_pred[0, :].squeeze(), y=x_pred[2, :].squeeze().squeeze(), line_color='green')
        # p.line(x=x_path, y=y_path, line_color='red', line_dash='dashed')
        # p = self.format_figure(p)
        # p.yaxis.axis_label = "y [m]"
        # p.xaxis.axis_label = "x [m]"
        # show(p)

    def test_pf_v3(self):
        """
        Test adding multiple time a path
        :return:
        """
        x0 = self.x0
        u0 = self.u0
        model = self.model
        nmpc = NMPC(model)
        theta = nmpc.create_path_variable()

        ca.vertcat(ca.sin(theta), ca.sin(2 * theta))
        nmpc.quad_stage_cost.add_states(names=['x'], weights=[10],
                                        ref=ca.sin(theta), path_following=True)
        nmpc.quad_stage_cost.add_states(names=['y'], weights=[10],
                                        ref=ca.sin(2 * theta), path_following=True)
        nmpc.quad_terminal_cost.add_states(names=['x', 'y'], weights=[10, 10],
                                           ref=ca.vertcat(ca.sin(theta), ca.sin(2 * theta)), path_following=True)
        nmpc.horizon = 10
        nmpc.set_initial_guess(x_guess=x0, u_guess=u0)
        nmpc.setup()
        n_steps = 1
        model.set_initial_conditions(x0=x0)
        sol = model.solution

        def path(theta):
            return np.sin(theta), np.sin(2 * theta)

        x_path = []
        y_path = []
        for t in range(1000):
            x_p, y_p = path(t / 100)
            x_path.append(x_p)
            y_path.append(y_p)

        for step in range(n_steps):
            u = nmpc.optimize(x0)
            x_pred, _, _ = nmpc.return_prediction()
            model.simulate(u=u)
            x0 = sol['x:f']
            # p = figure()
            # p.line(x=np.array(model.solution['x']).squeeze(), y=np.array(model.solution['y']).squeeze())
            # p.line(x=x_pred[0, :].squeeze(), y=x_pred[2, :].squeeze().squeeze(), line_color='green')
            # p.line(x=x_path, y=y_path, line_color='red', line_dash='dashed')
            # show(p)

        # p = figure()
        # p.line(x=np.array(model.solution['x']).squeeze(), y=np.array(model.solution['y']).squeeze())
        # p.line(x=x_pred[0, :].squeeze(), y=x_pred[2, :].squeeze().squeeze(), line_color='green')
        # p.line(x=x_path, y=y_path, line_color='red', line_dash='dashed')
        # show(p)

    def test_pf_v4(self):
        """
        test path following plus reference
        :return:
        """
        x0 = self.x0
        u0 = self.u0
        model = self.model
        nmpc = NMPC(model)
        theta = nmpc.create_path_variable()

        ca.vertcat(ca.sin(theta), ca.sin(2 * theta))
        nmpc.quad_stage_cost.add_states(names=['x'], weights=[10],
                                        ref=ca.sin(theta), path_following=True)
        nmpc.quad_stage_cost.add_states(names=['y'], weights=[10],
                                        ref=[1])
        nmpc.quad_terminal_cost.add_states(names=['x'], weights=[10],
                                           ref=ca.sin(theta), path_following=True)
        nmpc.horizon = 10
        nmpc.set_initial_guess(x_guess=x0, u_guess=u0)
        nmpc.setup()

        n_steps = 1
        model.set_initial_conditions(x0=x0)
        sol = model.solution

        def path(theta):
            return np.sin(theta), np.sin(2 * theta)

        x_path = []
        y_path = []
        for t in range(1000):
            x_p, y_p = path(t / 100)
            x_path.append(x_p)
            y_path.append(y_p)

        for step in range(n_steps):
            u = nmpc.optimize(x0)
            x_pred, _, _ = nmpc.return_prediction()
            model.simulate(u=u)
            x0 = sol['x:f']
            # p = figure()
            # p.line(x=np.array(model.solution['x']).squeeze(), y=np.array(model.solution['y']).squeeze())
            # p.line(x=x_pred[0, :].squeeze(), y=x_pred[2, :].squeeze().squeeze(), line_color='green')
            # p.line(x=x_path, y=y_path, line_color='red', line_dash='dashed')
            # show(p)

        # p = figure()
        # p.line(x=np.array(model.solution['x']).squeeze(), y=np.array(model.solution['y']).squeeze())
        # p.line(x=x_pred[0, :].squeeze(), y=x_pred[2, :].squeeze().squeeze(), line_color='green')
        # p.line(x=x_path, y=y_path, line_color='red', line_dash='dashed')
        # show(p)

    def test_pf_v5(self):
        """
        test conflicting path following plus reference cost
        :return:
        """

        """
        Test if giving a matrix in the cost works
        :return:
        """
        x0 = self.x0
        u0 = self.u0
        model = self.model
        nmpc = NMPC(model)
        # nmpc.set_varying_reference(path_following=True, ub_vel=3, lb_vel=0.4)
        theta = nmpc.create_path_variable()

        nmpc.quad_stage_cost.add_states(names=['x', 'y'], weights=[10, 10],
                                        ref=ca.vertcat(ca.sin(theta), ca.sin(2 * theta)), path_following=True)

        nmpc.quad_stage_cost.add_states(names=['y'], weights=[1],
                                        ref=[1])

        nmpc.quad_terminal_cost.add_states(names=['x'], weights=[10],
                                           ref=ca.sin(theta), path_following=True)

        nmpc.quad_stage_cost.add_states(names=['y'], weights=[1],
                                        ref=[1])
        nmpc.horizon = 10
        nmpc.set_initial_guess(x_guess=x0, u_guess=u0)
        nmpc.setup()

        n_steps = 1
        model.set_initial_conditions(x0=x0)
        sol = model.solution

        def path(theta):
            return np.sin(theta), np.sin(2 * theta)

        x_path = []
        y_path = []
        for t in range(1000):
            x_p, y_p = path(t / 100)
            x_path.append(x_p)
            y_path.append(y_p)

        for step in range(n_steps):
            u = nmpc.optimize(x0)
            x_pred, _, _ = nmpc.return_prediction()
            model.simulate(u=u)
            x0 = sol['x:f']
            # p = figure()
            # p.line(x=np.array(model.solution['x']).squeeze(), y=np.array(model.solution['y']).squeeze())
            # p.line(x=x_pred[0, :].squeeze(), y=x_pred[2, :].squeeze().squeeze(), line_color='green')
            # p.line(x=x_path, y=y_path, line_color='red', line_dash='dashed')
            # show(p)

        # p = figure()
        # p.line(x=np.array(model.solution['x']).squeeze(), y=np.array(model.solution['y']).squeeze())
        # p.line(x=x_pred[0, :].squeeze(), y=x_pred[2, :].squeeze().squeeze(), line_color='green')
        # p.line(x=x_path, y=y_path, line_color='red', line_dash='dashed')
        # show(p)

    def test_pf_v6(self):
        """
        test trajectory tracking
        :return:
        """
        x0 = self.x0
        u0 = self.u0
        model = self.model
        nmpc = NMPC(model)
        # nmpc.set_varying_reference(path_following=True, ub_vel=3, lb_vel=0.4)
        time = nmpc.get_time_variable()

        nmpc.quad_stage_cost.add_states(names=['x', 'y'], weights=[10, 10],
                                        ref=ca.vertcat(ca.sin(time), ca.sin(2 * time)), trajectory_tracking=True)

        nmpc.quad_terminal_cost.add_states(names=['x', 'y'], weights=[10, 10],
                                           ref=ca.vertcat(ca.sin(time), ca.sin(2 * time)), trajectory_tracking=True)

        nmpc.horizon = 10
        nmpc.set_initial_guess(x_guess=x0, u_guess=u0)
        nmpc.setup(options={'objective_function': 'discrete'})

        n_steps = 1
        model.set_initial_conditions(x0=x0)
        sol = model.solution

        def path(theta):
            return np.sin(theta), np.sin(2 * theta)

        x_path = []
        y_path = []
        for t in range(1000):
            x_p, y_p = path(t / 100)
            x_path.append(x_p)
            y_path.append(y_p)

        for step in range(n_steps):
            u = nmpc.optimize(x0)
            x_pred, _, _ = nmpc.return_prediction()
            model.simulate(u=u)
            x0 = sol['x:f']
            # p = figure()
            # p.line(x=np.array(model.solution['x']).squeeze(), y=np.array(model.solution['y']).squeeze())
            # p.line(x=x_pred[0, :].squeeze(), y=x_pred[2, :].squeeze().squeeze(), line_color='green')
            # p.line(x=x_path, y=y_path, line_color='red', line_dash='dashed')
            # show(p)

        # p = figure()
        # p.line(x=np.array(model.solution['x']).squeeze(), y=np.array(model.solution['y']).squeeze())
        # p.line(x=x_pred[0, :].squeeze(), y=x_pred[2, :].squeeze().squeeze(), line_color='green')
        # p.line(x=x_path, y=y_path, line_color='red', line_dash='dashed')
        # show(p)

    def test_pf_v7(self):
        """
        test trajectory tracking with path following together
        :return:
        """
        x0 = self.x0
        u0 = self.u0
        model = self.model
        nmpc = NMPC(model)
        # nmpc.set_varying_reference(path_following=True, ub_vel=3, lb_vel=0.4)
        time = nmpc.get_time_variable()
        theta = nmpc.create_path_variable()
        nmpc.quad_stage_cost.add_states(names=['x'], weights=[10],
                                        ref=ca.sin(theta), path_following=True)
        nmpc.quad_terminal_cost.add_states(names=['x'], weights=[10],
                                           ref=ca.sin(theta), path_following=True)

        nmpc.quad_stage_cost.add_states(names=['y'], weights=[100],
                                        ref=ca.sin(2 * time), trajectory_tracking=True)
        nmpc.quad_terminal_cost.add_states(names=['y'], weights=[100],
                                           ref=ca.sin(2 * time), trajectory_tracking=True)

        nmpc.horizon = 10
        nmpc.set_initial_guess(x_guess=x0, u_guess=u0)
        nmpc.setup()

        n_steps = 1
        model.set_initial_conditions(x0=x0)
        sol = model.solution

        def path(theta):
            return np.sin(theta), np.sin(2 * theta)

        x_path = []
        y_path = []
        for t in range(1000):
            x_p, y_p = path(t / 100)
            x_path.append(x_p)
            y_path.append(y_p)

        for step in range(n_steps):
            u = nmpc.optimize(x0)
            x_pred, _, _ = nmpc.return_prediction()
            model.simulate(u=u)
            x0 = sol['x:f']
            # p = figure()
            # p.line(x=np.array(model.solution['x']).squeeze(), y=np.array(model.solution['y']).squeeze())
            # p.line(x=x_pred[0, :].squeeze(), y=x_pred[2, :].squeeze().squeeze(), line_color='green')
            # p.line(x=x_path, y=y_path, line_color='red', line_dash='dashed')
            # show(p)
        #
        # p = figure()
        # p.line(x=np.array(model.solution['x']).squeeze(), y=np.array(model.solution['y']).squeeze())
        # p.line(x=x_pred[0, :].squeeze(), y=x_pred[2, :].squeeze().squeeze(), line_color='green')
        # p.line(x=x_path, y=y_path, line_color='red', line_dash='dashed')
        # show(p)

    def test_tt_v8(self):
        """
        test trajectory tracking with outputs
        :return:
        """
        x0 = self.x0
        u0 = self.u0
        model = self.model
        nmpc = NMPC(model)
        time = nmpc.get_time_variable()

        nmpc.quad_stage_cost.add_measurements(names=['y_x', 'y_y'], weights=[10, 10],
                                              ref=ca.vertcat(ca.sin(time), ca.sin(2 * time)),
                                              trajectory_tracking=True)

        nmpc.quad_terminal_cost.add_measurements(names=['y_x', 'y_y'], weights=[10, 10],
                                                 ref=ca.vertcat(ca.sin(time), ca.sin(2 * time)),
                                                 trajectory_tracking=True)

        nmpc.horizon = 10
        nmpc.set_initial_guess(x_guess=x0, u_guess=u0)
        nmpc.setup()
        print(nmpc)

        n_steps = 1
        model.set_initial_conditions(x0=x0)
        sol = model.solution

        def path(theta):
            return np.sin(theta), np.sin(2 * theta)

        x_path = []
        y_path = []
        for t in range(1000):
            x_p, y_p = path(t / 100)
            x_path.append(x_p)
            y_path.append(y_p)

        for step in range(n_steps):
            u = nmpc.optimize(x0)
            x_pred, _, _ = nmpc.return_prediction()
            model.simulate(u=u)
            x0 = sol['x:f']
            # p = figure()
            # p.line(x=np.array(model.solution['x']).squeeze(), y=np.array(model.solution['y']).squeeze())
            # p.line(x=x_pred[0, :].squeeze(), y=x_pred[2, :].squeeze().squeeze(), line_color='green')
            # p.line(x=x_path, y=y_path, line_color='red', line_dash='dashed')
            # show(p)

        # p = figure()
        # p.line(x=np.array(model.solution['x']).squeeze(), y=np.array(model.solution['y']).squeeze())
        # p.line(x=x_pred[0, :].squeeze(), y=x_pred[2, :].squeeze().squeeze(), line_color='green')
        # p.line(x=x_path, y=y_path, line_color='red', line_dash='dashed')
        # show(p)

    def test_tt_v9(self):
        """
        test trajectory tracking with outputs and weight on inputs
        :return:
        """
        x0 = self.x0
        u0 = self.u0
        model = self.model
        nmpc = NMPC(model)
        time = nmpc.get_time_variable()

        nmpc.quad_stage_cost.add_measurements(names=['y_x', 'y_y'], weights=[10, 10],
                                              ref=[ca.sin(time), ca.sin(2 * time)],
                                              trajectory_tracking=True)

        nmpc.quad_terminal_cost.add_measurements(names=['y_x', 'y_y'], weights=[10, 10],
                                                 ref=[ca.sin(time), ca.sin(2 * time)],
                                                 trajectory_tracking=True)

        nmpc.quad_stage_cost.add_inputs(names=['Fx', 'Fy'], weights=[0.01, 0.01])

        nmpc.horizon = 10
        nmpc.set_initial_guess(x_guess=x0, u_guess=u0)
        nmpc.setup()

        n_steps = 1
        model.set_initial_conditions(x0=x0)
        sol = model.solution

        def path(theta):
            return np.sin(theta), np.sin(2 * theta)

        x_path = []
        y_path = []
        for t in range(1000):
            x_p, y_p = path(t / 100)
            x_path.append(x_p)
            y_path.append(y_p)

        for step in range(n_steps):
            u = nmpc.optimize(x0)
            x_pred, _, _ = nmpc.return_prediction()
            model.simulate(u=u)
            x0 = sol['x:f']
            # p = figure()
            # p.line(x=np.array(model.solution['x']).squeeze(), y=np.array(model.solution['y']).squeeze())
            # p.line(x=x_pred[0, :].squeeze(), y=x_pred[2, :].squeeze().squeeze(), line_color='green')
            # p.line(x=x_path, y=y_path, line_color='red', line_dash='dashed')
            # show(p)

        # p = figure()
        # p.line(x=np.array(model.solution['x']).squeeze(), y=np.array(model.solution['y']).squeeze())
        # p.line(x=x_pred[0, :].squeeze(), y=x_pred[2, :].squeeze().squeeze(), line_color='green')
        # p.line(x=x_path, y=y_path, line_color='red', line_dash='dashed')
        # show(p)

    def test_pf_v10(self):
        """
        test giving reference input to path variable
        :return:
        """
        x0 = self.x0
        u0 = self.u0
        model = self.model
        nmpc = NMPC(model)
        # nmpc.set_varying_reference(path_following=True, ub_vel=3, lb_vel=0.4)
        # u_pf_lb=0.0001, u_pf_ub=1, u_pf_ref=None, u_pf_weight=10
        theta = nmpc.create_path_variable(u_pf_ref=0.1, u_pf_weight=100)

        nmpc.quad_stage_cost.add_states(names=['x', 'y'], weights=[10, 10],
                                        ref=ca.vertcat(ca.sin(theta), ca.sin(2 * theta)), path_following=True)

        nmpc.horizon = 10
        nmpc.set_initial_guess(x_guess=x0, u_guess=u0)
        nmpc.setup()

        n_steps = 1
        model.set_initial_conditions(x0=x0)
        sol = model.solution

        def path(theta):
            return np.sin(theta), np.sin(2 * theta)

        x_path = []
        y_path = []
        for t in range(1000):
            x_p, y_p = path(t / 100)
            x_path.append(x_p)
            y_path.append(y_p)

        for step in range(n_steps):
            u = nmpc.optimize(x0)
            x_pred, _, _ = nmpc.return_prediction()
            model.simulate(u=u)
            x0 = sol['x:f']
            # p = figure()
            # p.line(x=np.array(model.solution['x']).squeeze(), y=np.array(model.solution['y']).squeeze())
            # p.line(x=x_pred[0, :].squeeze(), y=x_pred[2, :].squeeze().squeeze(), line_color='green')
            # p.line(x=x_path, y=y_path, line_color='red', line_dash='dashed')
            # show(p)

        # p = figure()
        # p.line(x=np.array(model.solution['x']).squeeze(), y=np.array(model.solution['y']).squeeze())
        # p.line(x=x_pred[0, :].squeeze(), y=x_pred[2, :].squeeze().squeeze(), line_color='green')
        # p.line(x=x_path, y=y_path, line_color='red', line_dash='dashed')
        # show(p)

    def test_pf_v11(self):
        """
        Test having more than a path variable
        :return:
        """
        x0 = self.x0
        u0 = self.u0
        model = self.model
        nmpc = NMPC(model)

        theta1 = nmpc.create_path_variable(name='theta1')
        theta2 = nmpc.create_path_variable(name='theta2')

        nmpc.quad_stage_cost.add_states(names=['x', 'y'], weights=[10, 10],
                                        ref=ca.vertcat(ca.sin(theta1), ca.sin(2 * theta2)), path_following=True)

        nmpc.quad_terminal_cost.add_states(names=['x', 'y'], weights=[10, 10],
                                           ref=ca.vertcat(ca.sin(theta1), ca.sin(2 * theta2)), path_following=True)
        nmpc.horizon = 10
        nmpc.set_initial_guess(x_guess=x0, u_guess=u0)
        nmpc.setup()

        n_steps = 1
        model.set_initial_conditions(x0=x0)
        sol = model.solution

        def path(theta):
            return np.sin(theta), np.sin(2 * theta)

        x_path = []
        y_path = []
        for t in range(1000):
            x_p, y_p = path(t / 100)
            x_path.append(x_p)
            y_path.append(y_p)

        # p = figure()
        # p.line(x=x_path, y=y_path, line_color='red', line_dash='dashed')
        # p = self.format_figure(p)
        # p.yaxis.axis_label = "y [m]"
        # p.xaxis.axis_label = "x [m]"
        # show(p)

        scl = SimpleControlLoop(model, nmpc)
        scl.run(steps=n_steps)
        # scl.plot()
        # p = figure()
        # p.line(x=np.array(model.solution['x']).squeeze(), y=np.array(model.solution['y']).squeeze())
        # p.line(x=x_pred[0, :].squeeze(), y=x_pred[2, :].squeeze().squeeze(), line_color='green')
        # p.line(x=x_path, y=y_path, line_color='red', line_dash='dashed')
        # show(p)

        # p = figure()
        # p.line(x=np.array(model.solution['x']).squeeze(), y=np.array(model.solution['y']).squeeze())
        # p.line(x=x_pred[0, :].squeeze(), y=x_pred[2, :].squeeze().squeeze(), line_color='green')
        # p.line(x=x_path, y=y_path, line_color='red', line_dash='dashed')
        # p = self.format_figure(p)
        # p.yaxis.axis_label = "y [m]"
        # p.xaxis.axis_label = "x [m]"
        # show(p)

    def test_pf_v12(self):
        """
        Test path following with SimpleControlLoop
        :return:
        """
        x0 = self.x0
        u0 = self.u0
        model = self.model
        nmpc = NMPC(model)
        theta = nmpc.create_path_variable()

        nmpc.quad_stage_cost.add_states(names=['x', 'y'], weights=[10, 10],
                                        ref=ca.vertcat(ca.sin(theta), ca.sin(2 * theta)), path_following=True)
        nmpc.quad_terminal_cost.add_states(names=['x', 'y'], weights=[10, 10],
                                           ref=ca.vertcat(ca.sin(theta), ca.sin(2 * theta)), path_following=True)
        nmpc.horizon = 10
        nmpc.set_initial_guess(x_guess=x0, u_guess=u0)
        nmpc.setup()

        n_steps = 1
        model.set_initial_conditions(x0=x0)
        sol = model.solution

        scl = SimpleControlLoop(model, nmpc)
        scl.run(steps=n_steps)
        # scl.plot()

    def test_tt_v12(self):
        """
        test trajectory tracking - pass data directly and not the function
        :return:
        """
        x0 = self.x0
        u0 = self.u0
        model = self.model
        nmpc = NMPC(model)
        # nmpc.set_varying_reference(path_following=True, ub_vel=3, lb_vel=0.4)
        time = nmpc.get_time_variable()

        nmpc.quad_stage_cost.add_states(names=['x', 'y'], weights=[10, 10], trajectory_tracking=True)

        nmpc.quad_terminal_cost.add_states(names=['x', 'y'], weights=[10, 10], trajectory_tracking=True)

        nmpc.horizon = 10
        nmpc.set_initial_guess(x_guess=x0, u_guess=u0)
        nmpc.setup(options={'objective_function': 'discrete'})

        n_steps = 1
        model.set_initial_conditions(x0=x0)
        sol = model.solution

        def path(theta):
            return np.sin(theta), np.sin(2 * theta)

        x_traj = []
        y_traj = []
        t0 = 0
        for t in range(1000):
            x_p, y_p = path(t0 + self.dt)
            x_traj.append(x_p)
            y_traj.append(y_p)
            t0 += self.dt

        x_path = []
        y_path = []
        for t in range(1000):
            x_p, y_p = path(t / 100)
            x_path.append(x_p)
            y_path.append(y_p)

        for step in range(n_steps):
            u = nmpc.optimize(x0, ref_sc={'x': x_traj, 'y': y_traj},
                              ref_tc={'x': x_traj, 'y': y_traj})
            x_pred, _, _ = nmpc.return_prediction()
            model.simulate(u=u)
            x0 = sol['x:f']
            # p = figure()
            # p.line(x=np.array(model.solution['x']).squeeze(), y=np.array(model.solution['y']).squeeze())
            # p.line(x=x_pred[0, :].squeeze(), y=x_pred[2, :].squeeze().squeeze(), line_color='green')
            # p.line(x=x_path, y=y_path, line_color='red', line_dash='dashed')
            # show(p)

        # p = figure()
        # p.line(x=np.array(model.solution['x']).squeeze(), y=np.array(model.solution['y']).squeeze())
        # p.line(x=x_pred[0, :].squeeze(), y=x_pred[2, :].squeeze().squeeze(), line_color='green')
        # p.line(x=x_path, y=y_path, line_color='red', line_dash='dashed')
        # show(p)

    def test_tt_v13(self):
        """
        test trajectory tracking - test input trajectory
        :return:
        """
        x0 = self.x0
        u0 = self.u0
        model = self.model
        nmpc = NMPC(model)
        # nmpc.set_varying_reference(path_following=True, ub_vel=3, lb_vel=0.4)
        time = nmpc.get_time_variable()

        nmpc.quad_stage_cost.add_inputs(names=['Fx', 'Fy'], weights=[10, 10], trajectory_tracking=True)

        nmpc.horizon = 10
        nmpc.set_initial_guess(x_guess=x0, u_guess=u0)
        nmpc.setup(options={'objective_function': 'discrete'})

        n_steps = 1
        model.set_initial_conditions(x0=x0)
        sol = model.solution

        x_traj = []
        y_traj = []
        t0 = 0
        for t in range(1000):
            x_traj.append(0.1)
            y_traj.append(0.1)
            t0 += self.dt

        for step in range(n_steps):
            u = nmpc.optimize(x0, ref_sc={'Fx': x_traj, 'Fy': y_traj},
                              ref_tc={'Fx': x_traj, 'Fy': y_traj})
            x_pred, _, _ = nmpc.return_prediction()
            model.simulate(u=u)
            x0 = sol['x:f']
            # p = figure()
            # p.line(x=np.array(model.solution['x']).squeeze(), y=np.array(model.solution['y']).squeeze())
            # p.line(x=x_pred[0, :].squeeze(), y=x_pred[2, :].squeeze().squeeze(), line_color='green')
            # p.line(x=x_path, y=y_path, line_color='red', line_dash='dashed')
            # show(p)

        # model.solution.plot()
        # p = figure()
        # p.line(x=np.array(model.solution['x']).squeeze(), y=np.array(model.solution['y']).squeeze())
        # p.line(x=x_pred[0, :].squeeze(), y=x_pred[2, :].squeeze().squeeze(), line_color='green')
        # p.line(x=x_path, y=y_path, line_color='red', line_dash='dashed')
        # show(p)

    def test_tt_v14(self):
        """
        test trajectory tracking - pass data directly and not the function, try multiple trajectories for the
        stage cost
        :return:
        """
        x0 = self.x0
        u0 = self.u0
        model = self.model
        nmpc = NMPC(model)
        # nmpc.set_varying_reference(path_following=True, ub_vel=3, lb_vel=0.4)
        time = nmpc.get_time_variable()

        nmpc.quad_stage_cost.add_states(names=['x'], weights=[10], trajectory_tracking=True)
        nmpc.quad_stage_cost.add_states(names=['y'], weights=[10], trajectory_tracking=True)

        nmpc.quad_terminal_cost.add_states(names=['x', 'y'], weights=[10, 10], trajectory_tracking=True)

        nmpc.horizon = 10
        nmpc.set_initial_guess(x_guess=x0, u_guess=u0)
        nmpc.setup(options={'objective_function': 'discrete'})

        n_steps = 1
        model.set_initial_conditions(x0=x0)
        sol = model.solution

        def path(theta):
            return np.sin(theta), np.sin(2 * theta)

        x_traj = []
        y_traj = []
        t0 = 0
        for t in range(1000):
            x_p, y_p = path(t0 + self.dt)
            x_traj.append(x_p)
            y_traj.append(y_p)
            t0 += self.dt

        x_path = []
        y_path = []
        for t in range(1000):
            x_p, y_p = path(t / 100)
            x_path.append(x_p)
            y_path.append(y_p)

        for step in range(n_steps):
            u = nmpc.optimize(x0, ref_sc={'x': x_traj, 'y': y_traj},
                              ref_tc={'x': x_traj, 'y': y_traj})

            x_pred, _, _ = nmpc.return_prediction()
            model.simulate(u=u)
            x0 = sol['x:f']
            # p = figure()
            # p.line(x=np.array(model.solution['x']).squeeze(), y=np.array(model.solution['y']).squeeze())
            # p.line(x=x_pred[0, :].squeeze(), y=x_pred[2, :].squeeze().squeeze(), line_color='green')
            # p.line(x=x_path, y=y_path, line_color='red', line_dash='dashed')
            # show(p)

        # p = figure()
        # p.line(x=np.array(model.solution['x']).squeeze(), y=np.array(model.solution['y']).squeeze())
        # p.line(x=x_pred[0, :].squeeze(), y=x_pred[2, :].squeeze().squeeze(), line_color='green')
        # p.line(x=x_path, y=y_path, line_color='red', line_dash='dashed')
        # show(p)

    def test_tt_v15(self):
        """
        test trajectory tracking - test wrong keys in dictionary
        stage cost
        :return:
        """
        x0 = self.x0
        u0 = self.u0
        model = self.model
        nmpc = NMPC(model)
        # nmpc.set_varying_reference(path_following=True, ub_vel=3, lb_vel=0.4)
        time = nmpc.get_time_variable()

        nmpc.quad_stage_cost.add_states(names=['x'], weights=[10], trajectory_tracking=True)
        nmpc.quad_stage_cost.add_states(names=['y'], weights=[10], trajectory_tracking=True)

        nmpc.quad_terminal_cost.add_states(names=['x', 'y'], weights=[10, 10], trajectory_tracking=True)

        nmpc.horizon = 10
        nmpc.set_initial_guess(x_guess=x0, u_guess=u0)
        nmpc.setup(options={'objective_function': 'discrete'})

        n_steps = 1
        model.set_initial_conditions(x0=x0)
        sol = model.solution

        def path(theta):
            return np.sin(theta), np.sin(2 * theta)

        x_traj = []
        y_traj = []
        t0 = 0
        for t in range(1000):
            x_p, y_p = path(t0 + self.dt)
            x_traj.append(x_p)
            y_traj.append(y_p)
            t0 += self.dt

        x_path = []
        y_path = []
        for t in range(1000):
            x_p, y_p = path(t / 100)
            x_path.append(x_p)
            y_path.append(y_p)

        self.assertRaises(ValueError, nmpc.optimize, x0, ref_sc={'xx': x_traj, 'ty': y_traj},
                          ref_tc={'x': x_traj, 'y': y_traj})

    def test_tt_v16(self):
        """
        test trajectory tracking with SimpleControlLoop class
        :return:
        """
        x0 = self.x0
        u0 = self.u0
        model = self.model
        nmpc = NMPC(model)
        time = nmpc.get_time_variable()

        nmpc.quad_stage_cost.add_states(names=['x', 'y'], weights=[10, 10],
                                        ref=[ca.sin(time), ca.sin(2 * time)],
                                        trajectory_tracking=True)

        nmpc.quad_terminal_cost.add_states(names=['x', 'y'], weights=[10, 10],
                                           ref=[ca.sin(time), ca.sin(2 * time)],
                                           trajectory_tracking=True)

        nmpc.horizon = 10
        nmpc.set_initial_guess(x_guess=x0, u_guess=u0)
        nmpc.setup()

        n_steps = 1
        model.set_initial_conditions(x0=x0)

        scl = SimpleControlLoop(model, nmpc)
        scl.run(steps=n_steps)
        # scl.plot()

    def test_tt_v17(self):
        """
        test trajectory tracking with SimpleControlLoop class
        :return:
        """
        x0 = self.x0
        u0 = self.u0
        model = self.model
        nmpc = NMPC(model)

        nmpc.quad_stage_cost.add_states(names=['x', 'y'], weights=[10, 10],
                                        trajectory_tracking=True)

        nmpc.quad_terminal_cost.add_states(names=['x', 'y'], weights=[10, 10],
                                           trajectory_tracking=True)

        nmpc.horizon = 10
        nmpc.set_initial_guess(x_guess=x0, u_guess=u0)
        nmpc.setup(options={'objective_function': 'discrete'})

        n_steps = 1
        model.set_initial_conditions(x0=x0)

        def path(theta):
            return np.sin(theta), np.sin(2 * theta)

        x_traj = []
        y_traj = []
        t0 = 0
        for t in range(1000):
            x_p, y_p = path(t0)
            x_traj.append(x_p)
            y_traj.append(y_p)
            t0 += self.dt

        scl = SimpleControlLoop(model, nmpc)
        scl.run(steps=n_steps, ref_sc={'x': x_traj, 'y': y_traj}, ref_tc={'x': x_traj, 'y': y_traj})
        # scl.plot()

    def test_tt_v18(self):
        """
        test fail trajectory tracking wrong reference type
        :return:
        """
        x0 = self.x0
        u0 = self.u0
        model = self.model
        nmpc = NMPC(model)

        nmpc.quad_stage_cost.add_states(names=['x', 'y'], weights=[10, 10],
                                        trajectory_tracking=True)

        nmpc.quad_terminal_cost.add_states(names=['x', 'y'], weights=[10, 10],
                                           trajectory_tracking=True)

        nmpc.horizon = 10
        nmpc.set_initial_guess(x_guess=x0, u_guess=u0)
        nmpc.setup(options={'objective_function': 'discrete'})

        n_steps = 1
        model.set_initial_conditions(x0=x0)


        self.assertRaises(TypeError, nmpc.optimize,x0=x0, ref_sc=[1,1], ref_tc=[0,1])

    def test_tt_v19(self):
        """
        test fail trajectory tracking no discrete obj_fun
        :return:
        """
        x0 = self.x0
        u0 = self.u0
        model = self.model
        nmpc = NMPC(model)

        nmpc.quad_stage_cost.add_states(names=['x', 'y'], weights=[10, 10],
                                        trajectory_tracking=True)

        nmpc.quad_terminal_cost.add_states(names=['x', 'y'], weights=[10, 10],
                                           trajectory_tracking=True)

        nmpc.horizon = 10
        nmpc.set_initial_guess(x_guess=x0, u_guess=u0)
        nmpc.setup()

        model.set_initial_conditions(x0=x0)


        def path(theta):
            return np.sin(theta), np.sin(2 * theta)

        x_traj = []
        y_traj = []
        t0 = 0
        for t in range(1000):
            x_p, y_p = path(t0)
            x_traj.append(x_p)
            y_traj.append(y_p)
            t0 += self.dt

        self.assertRaises(AssertionError, nmpc.optimize,x0=x0,  ref_sc={'x': x_traj, 'y': y_traj}, ref_tc={'x': x_traj, 'y': y_traj})

    def test_tt_v20(self):
        """
        test fail trajectory tracking no discrete obj_fun
        :return:
        """
        x0 = self.x0
        u0 = self.u0
        model = self.model
        nmpc = NMPC(model)

        nmpc.quad_stage_cost.add_states(names=['x', 'y'], weights=[10, 10],
                                        trajectory_tracking=True)

        nmpc.quad_terminal_cost.add_states(names=['x', 'y'], weights=[10, 10],
                                           trajectory_tracking=True)

        nmpc.horizon = 10
        nmpc.set_initial_guess(x_guess=x0, u_guess=u0)
        nmpc.setup()

        model.set_initial_conditions(x0=x0)


        def path(theta):
            return np.sin(theta), np.sin(2 * theta)

        x_traj = []
        y_traj = []
        t0 = 0
        for t in range(5):
            x_p, y_p = path(t0)
            x_traj.append(x_p)
            y_traj.append(y_p)
            t0 += self.dt

        self.assertRaises(ValueError, nmpc.optimize,x0=x0,  ref_sc={'x': x_traj, 'y': y_traj}, ref_tc={'x': x_traj, 'y': y_traj})

    def test_vr_1(self):
        """
        Test change in reference setpoint online
        :return:
        """
        """
        Test having more than a path variable
        :return:
        """
        x0 = self.x0
        u0 = self.u0
        model = self.model
        nmpc = NMPC(model)

        nmpc.quad_stage_cost.add_states(names=['x', 'y'], weights=[10, 10], trajectory_tracking=True)
        nmpc.quad_terminal_cost.add_states(names=['x', 'y'], weights=[10, 10], trajectory_tracking=True)

        nmpc.horizon = 10
        nmpc.set_initial_guess(x_guess=x0, u_guess=u0)
        nmpc.setup()

        model.set_initial_conditions(x0=x0)
        sol = model.solution

        ss = SimpleControlLoop(model, nmpc)
        ss.run(2, ref_sc={'x': 1, 'y': 2},
               ref_tc={'x': 1, 'y': 2})

        ss.run(2, ref_sc={'x': 2, 'y': 1},
               ref_tc={'x': 2, 'y': 1})

        # ss.plot()


class TestNMPCOptions(TestCase):
    def setUp(self):
        model = Model(plot_backend='bokeh')
        # Constants
        M = 5.

        # States and algebraic variables
        xx = model.set_dynamical_states(['x', 'vx', 'y', 'vy'])
        model.set_measurements(['y_x', 'y_vx', 'y_y', 'y_vy'])
        model.set_measurement_equations([xx[0], xx[1], xx[2], xx[3]])
        x = xx[0]
        vx = xx[1]
        y = xx[2]
        vy = xx[3]
        # Inputs
        F = model.set_inputs(['Fx', 'Fy'])
        Fx = F[0]
        Fy = F[1]
        # ODEs
        dd = ca.SX.sym('dd', 4)
        dd[0] = vx
        dd[1] = Fx / M
        dd[2] = vy
        dd[3] = Fy / M

        # Algebraic equations
        # dy = h + l * ca.cos(theta) - y

        model.set_equations(ode=dd)

        # Initial conditions
        x0 = [0, 0, 0, 0]
        u0 = [0., 0.]

        # Create model and run simulation
        dt = 0.1
        model.setup(dt=dt)

        self.model = model
        self.dt = dt
        self.x0 = x0
        self.u0 = u0

        def format_figure(p):
            p.yaxis.axis_label_text_font_size = "18pt"
            p.yaxis.major_label_text_font_size = "18pt"
            p.xaxis.major_label_text_font_size = "18pt"
            p.xaxis.axis_label_text_font_size = "18pt"
            return p

        self.format_figure = format_figure

    @skip('not really sure what this tests does')
    def test_pf_v2(self):
        "CHeck what happens if I put wrong options"

        x0 = self.x0
        u0 = self.u0
        model = self.model
        nmpc = NMPC(model)
        theta = nmpc.create_path_variable()

        nmpc.quad_stage_cost.add_states(names=['x', 'y'], weights=[10, 10],
                                        ref=ca.vertcat(ca.sin(theta), ca.sin(2 * theta)), path_following=True)
        nmpc.quad_terminal_cost.add_states(names=['x', 'y'], weights=[10, 10],
                                           ref=ca.vertcat(ca.sin(theta), ca.sin(2 * theta)), path_following=True)
        nmpc.horizon = 10
        nmpc.set_initial_guess(x_guess=x0, u_guess=u0)
        with self.assertRaises(ValueError) as cm:
            nmpc.setup()

        print(cm.exception)


class TestDAE(TestCase):
    def setUp(self) -> None:
        model = Model(plot_backend='bokeh')
        # Constants
        M = 5.
        m = 1.
        l = 1.
        h = .5
        g = 9.81

        # States and algebraic variables
        x = model.set_dynamical_states(['x', 'v', 'theta', 'omega'])
        model.set_measurements(['yx', 'yv', 'ytheta', 'tomega'])
        model.set_measurement_equations([x[0], x[1], x[2], x[3]])
        y = model.set_algebraic_states(['y'])
        v = x[1]
        theta = x[2]
        omega = x[3]
        # Inputs
        F = model.set_inputs('F')

        # ODEs
        dd = ca.SX.sym('dd', 4)
        dd[0] = v
        dd[1] = 1. / (M + m - m * ca.cos(theta)) * (m * g * ca.sin(theta) - m * l * ca.sin(theta) * omega ** 2 + F)
        dd[2] = omega
        dd[3] = 1. / l * (dd[1] * ca.cos(theta) + g * ca.sin(theta))

        # Algebraic equations
        dy = h + l * ca.cos(theta) - y

        model.set_dynamical_equations(dd)
        model.set_algebraic_equations(dy)
        # Initial conditions
        x0 = [2.5, 0., 0.1, 0.]
        z0 = h + l * ca.cos(x0[2]) - h
        u0 = 0.

        # Create model and run simulation
        dt = .1
        model.setup(dt=dt)

        self.model = model
        self.dt = dt
        self.x0 = x0
        self.u0 = u0
        self.z0 = z0

        # may_term = (model.x[3]-ca.pi)*100*(model.x[3]-ca.pi)
        # lag_term = (model.x[3]-ca.pi)*10*(model.x[3]-ca.pi) #+ model.u*0.01*model.u

    def test_closed_loop_collocation(self):
        " Test normal nonlinear MPC for using a inverted pendulum model with DAE. This test checks the normal problem setup"
        x0 = self.x0
        u0 = self.u0
        z0 = self.z0

        model = self.model
        nmpc = NMPC(model)
        nmpc.quad_stage_cost.add_states(names=['v', 'theta'], ref=[0, 0], weights=[10, 5])
        nmpc.quad_stage_cost.add_inputs(names='F', weights=0.1)
        nmpc.horizon = 25
        nmpc.set_box_constraints(x_ub=[5, 10, 10, 10], x_lb=[-5, -10, -10, -10])
        nmpc.set_initial_guess(x_guess=x0, u_guess=u0)
        nmpc.setup()

        n_steps = 1
        model.set_initial_conditions(x0=x0, z0=z0)
        sol = model.solution

        for step in range(n_steps):
            u = nmpc.optimize(x0)
            model.simulate(u=u, steps=1)
            x0 = sol['x:f']

        # sol.plot(
        #     ('t', 'x'),
        #     ('t', 'v'),
        #     ('t', 'theta'),
        #     ('t', 'omega'),
        #     subplots=True,
        #     title=sol.get_names('x'),
        #     xlabel=None,
        #     legend=False)
        # nmpc.plot_mpc_prediction()

    def test_closed_loop_rk(self):
        " Test normal nonlinear MPC for using a inverted pendulum model with DAE. This test checks the normal problem setup"
        x0 = self.x0
        u0 = self.u0
        z0 = self.z0

        model = self.model
        nmpc = NMPC(model)
        nmpc.quad_stage_cost.add_states(names=['v', 'theta'], ref=[0, 0], weights=[10, 5])
        nmpc.quad_stage_cost.add_inputs(names='F', weights=0.1)
        nmpc.horizon = 25
        nmpc.set_box_constraints(x_ub=[5, 10, 10, 10], x_lb=[-5, -10, -10, -10], z_lb=-100,z_ub=100)
        nmpc.set_initial_guess(x_guess=x0, u_guess=u0)
        nmpc.set_nlp_options({'integration_method': 'rk4'})
        nmpc.setup()

        n_steps = 1
        model.set_initial_conditions(x0=x0, z0=z0)
        sol = model.solution

        for step in range(n_steps):
            u = nmpc.optimize(x0)
            model.simulate(u=u, steps=1)
            x0 = sol['x:f']

        # sol.plot(
        #     ('t', 'x'),
        #     ('t', 'v'),
        #     ('t', 'theta'),
        #     ('t', 'omega'),
        #     subplots=True,
        #     title=sol.get_names('x'),
        #     xlabel=None,
        #     legend=False)
        # nmpc.plot_mpc_prediction()


class TestTvp(TestCase):

    def setUp(self) -> None:
        model = Model(plot_backend='bokeh')
        # Constants
        M = 5.
        m = 1.
        l = 1.
        h = .5
        g = 9.81

        # States and algebraic variables
        x = model.set_dynamical_states(['x', 'v', 'theta', 'omega'])
        model.set_measurements(['yx', 'yv', 'ytheta', 'tomega'])
        model.set_measurement_equations([x[0], x[1], x[2], x[3]])
        d = model.set_parameters(['d_tvp', 'd_const'])  # disturbance
        v = x[1]
        theta = x[2]
        omega = x[3]
        # Inputs
        F = model.set_inputs('F')

        # ODEs

        dx = v
        dv = 1. / (M + m - m * ca.cos(theta)) * (m * g * ca.sin(theta) - m * l * ca.sin(theta) * omega ** 2 + F)
        dtheta = omega + d[0]
        domega = 1. / l * (dv * ca.cos(theta) + g * ca.sin(theta)) + d[1]

        # Algebraic equations
        # dy = h + l * ca.cos(theta) - y

        model.set_equations(ode=[dx, dv, dtheta, domega])

        # Initial conditions
        x0 = [2.5, 0., 0.1, 0.]
        u0 = 0.

        # Create model and run simulation
        dt = .1
        model.setup(dt=dt)

        self.model = model
        self.dt = dt
        self.x0 = x0
        self.u0 = u0

    def test_tvp(self) -> None:
        """
        Test what happens if we pass tvp to the set_time_varying parameters method
        :return:
        """
        x0 = self.x0
        u0 = self.u0
        model = self.model
        nmpc = NMPC(model)
        nmpc.quad_stage_cost.add_states(names=['v', 'theta'], ref=[0, 0], weights=[10, 5])
        nmpc.quad_stage_cost.add_inputs(names='F', weights=0.1)
        nmpc.horizon = 25
        nmpc.set_box_constraints(x_ub=[5, 10, 10, 10], x_lb=[-5, -10, -10, -10])
        nmpc.set_initial_guess(x_guess=x0, u_guess=u0)
        nmpc.set_time_varying_parameters(names=['d_tvp'], values={'d_tvp': np.sin(np.arange(0, 5, self.dt)).tolist()})
        nmpc.setup()

        n_steps = 2
        model.set_initial_conditions(x0=x0)
        sol = model.solution

        for step in range(n_steps):
            u = nmpc.optimize(x0, cp=0.1)
            model.simulate(u=u, p=[np.sin(self.dt * step), 0.1])
            x0 = sol['x:f']

        # sol.plot(
        #     ('t', 'x'),
        #     ('t', 'v'),
        #     ('t', 'theta'),
        #     ('t', 'omega'),
        #     subplots=True,
        #     title=sol.get_names('x'),
        #     xlabel=None,
        #     legend=False)

    def test_tvp_2(self) -> None:
        """
        Test what happens if the tvp are passed to the optimize()
        :return:
        """
        x0 = self.x0
        u0 = self.u0
        model = self.model
        nmpc = NMPC(model)
        nmpc.quad_stage_cost.add_states(names=['v', 'theta'], ref=[0, 0], weights=[10, 5])
        nmpc.quad_stage_cost.add_inputs(names='F', weights=0.1)
        nmpc.horizon = 25
        nmpc.set_box_constraints(x_ub=[5, 10, 10, 10], x_lb=[-5, -10, -10, -10])
        nmpc.set_initial_guess(x_guess=x0, u_guess=u0)
        nmpc.set_time_varying_parameters(names=['d_tvp'])
        nmpc.setup()

        n_steps = 1
        model.set_initial_conditions(x0=x0)
        sol = model.solution

        for step in range(n_steps):
            u = nmpc.optimize(x0, cp=0.1, tvp={'d_tvp': np.sin(
                np.arange(step * self.dt, step * self.dt + self.dt * nmpc.horizon, self.dt)).tolist()})
            model.simulate(u=u, p=[np.sin(self.dt * step), 0.1])
            x0 = sol['x:f']

        # sol.plot(
        #     ('t', 'x'),
        #     ('t', 'v'),
        #     ('t', 'theta'),
        #     ('t', 'omega'),
        #     subplots=True,
        #     title=sol.get_names('x'),
        #     xlabel=None,
        #     legend=False)

    def test_tvp_3(self) -> None:
        """
        Test what happens if we pass tvp to the set_time_varying parameters method but they are shorter than the
        prediction horizon
        :return:
        """
        x0 = self.x0
        u0 = self.u0
        model = self.model
        nmpc = NMPC(model)
        nmpc.quad_stage_cost.add_states(names=['v', 'theta'], ref=[0, 0], weights=[10, 5])
        nmpc.quad_stage_cost.add_inputs(names='F', weights=0.1)
        nmpc.horizon = 25
        nmpc.set_box_constraints(x_ub=[5, 10, 10, 10], x_lb=[-5, -10, -10, -10])
        nmpc.set_initial_guess(x_guess=x0, u_guess=u0)
        nmpc.set_time_varying_parameters(names=['d_tvp'], values={'d_tvp': np.sin(np.arange(0, 1, self.dt)).tolist()})
        self.assertRaises(TypeError, nmpc.setup)


class TestChangeInputWeight(TestCase):
    def setUp(self):
        model = Model(plot_backend='bokeh')
        # Constants
        M = 5.

        # States and algebraic variables
        xx = model.set_dynamical_states(['x', 'vx', 'y', 'vy'])
        model.set_measurements(['y_x', 'y_vx', 'y_y', 'y_vy'])
        model.set_measurement_equations([xx[0], xx[1], xx[2], xx[3]])
        x = xx[0]
        vx = xx[1]
        y = xx[2]
        vy = xx[3]
        # Inputs
        F = model.set_inputs(['Fx', 'Fy'])
        Fx = F[0]
        Fy = F[1]
        # ODEs
        dd = ca.SX.sym('dd', 4)
        dd[0] = vx
        dd[1] = Fx / M
        dd[2] = vy
        dd[3] = Fy / M

        # Algebraic equations
        # dy = h + l * ca.cos(theta) - y

        model.set_equations(ode=dd)

        # Initial conditions
        x0 = [0, 0, 0, 0]
        u0 = [0., 0.]

        # Create model and run simulation
        dt = 0.1
        model.setup(dt=dt)

        self.model = model
        self.dt = dt
        self.x0 = x0
        self.u0 = u0

        def format_figure(p):
            p.yaxis.axis_label_text_font_size = "18pt"
            p.yaxis.major_label_text_font_size = "18pt"
            p.xaxis.major_label_text_font_size = "18pt"
            p.xaxis.axis_label_text_font_size = "18pt"
            return p

        self.format_figure = format_figure

    def test_ciw_v1(self):
        """
        Add one input change cost
        :return:
        """

        x0 = self.x0
        u0 = self.u0
        model = self.model
        nmpc = NMPC(model)
        # nmpc.set_varying_reference(path_following=True, ub_vel=3, lb_vel=0.4)
        theta = nmpc.create_path_variable()

        nmpc.quad_stage_cost.add_states(names=['x', 'y'], weights=[10, 10],
                                        ref=ca.vertcat(ca.sin(theta), ca.sin(2 * theta)), path_following=True)
        nmpc.quad_terminal_cost.add_states(names=['x', 'y'], weights=[10, 10],
                                           ref=ca.vertcat(ca.sin(theta), ca.sin(2 * theta)), path_following=True)

        nmpc.quad_stage_cost.add_inputs_change(names=['Fx'], weights=[10])

        nmpc.horizon = 10
        nmpc.set_initial_guess(x_guess=x0, u_guess=u0)
        nmpc.setup()

        n_steps = 1
        model.set_initial_conditions(x0=x0)
        sol = model.solution

        def path(theta):
            return np.sin(theta), np.sin(2 * theta)

        x_path = []
        y_path = []
        for t in range(1000):
            x_p, y_p = path(t / 100)
            x_path.append(x_p)
            y_path.append(y_p)

        for step in range(n_steps):
            u = nmpc.optimize(x0)
            x_pred, _, _ = nmpc.return_prediction()
            model.simulate(u=u)
            x0 = sol['x:f']

        # p = figure()
        # p.line(x=np.array(model.solution['x']).squeeze(), y=np.array(model.solution['y']).squeeze())
        # p.line(x=x_pred[0, :].squeeze(), y=x_pred[2, :].squeeze().squeeze(), line_color='green')
        # p.line(x=x_path, y=y_path, line_color='red', line_dash='dashed')
        # show(p)

    def test_ciw_v2(self):
        """
        Add two input change costs separatelly
        :return:
        """

        x0 = self.x0
        u0 = self.u0
        model = self.model
        nmpc = NMPC(model)
        # nmpc.set_varying_reference(path_following=True, ub_vel=3, lb_vel=0.4)
        theta = nmpc.create_path_variable()

        nmpc.quad_stage_cost.add_states(names=['x', 'y'], weights=[10, 10],
                                        ref=ca.vertcat(ca.sin(theta), ca.sin(2 * theta)), path_following=True)
        nmpc.quad_terminal_cost.add_states(names=['x', 'y'], weights=[10, 10],
                                           ref=ca.vertcat(ca.sin(theta), ca.sin(2 * theta)), path_following=True)

        nmpc.quad_stage_cost.add_inputs_change(names=['Fx'], weights=[10])
        nmpc.quad_stage_cost.add_inputs_change(names=['Fy'], weights=[10])

        nmpc.horizon = 10
        nmpc.set_initial_guess(x_guess=x0, u_guess=u0)
        nmpc.setup()

        n_steps = 1
        model.set_initial_conditions(x0=x0)
        sol = model.solution

        def path(theta):
            return np.sin(theta), np.sin(2 * theta)

        x_path = []
        y_path = []
        for t in range(1000):
            x_p, y_p = path(t / 100)
            x_path.append(x_p)
            y_path.append(y_p)

        for step in range(n_steps):
            u = nmpc.optimize(x0)
            x_pred, _, _ = nmpc.return_prediction()
            model.simulate(u=u)
            x0 = sol['x:f']

        # p = figure()
        # p.line(x=np.array(model.solution['x']).squeeze(), y=np.array(model.solution['y']).squeeze())
        # p.line(x=x_pred[0, :].squeeze(), y=x_pred[2, :].squeeze().squeeze(), line_color='green')
        # p.line(x=x_path, y=y_path, line_color='red', line_dash='dashed')
        # show(p)

    def test_ciw_v3(self):
        """
        Add two input change costs together
        :return:
        """

        x0 = self.x0
        u0 = self.u0
        model = self.model
        nmpc = NMPC(model)
        # nmpc.set_varying_reference(path_following=True, ub_vel=3, lb_vel=0.4)
        theta = nmpc.create_path_variable()

        nmpc.quad_stage_cost.add_states(names=['x', 'y'], weights=[10, 10],
                                        ref=ca.vertcat(ca.sin(theta), ca.sin(2 * theta)), path_following=True)
        nmpc.quad_terminal_cost.add_states(names=['x', 'y'], weights=[10, 10],
                                           ref=ca.vertcat(ca.sin(theta), ca.sin(2 * theta)), path_following=True)

        nmpc.quad_stage_cost.add_inputs_change(names=['Fx', 'Fy'], weights=[10, 10])

        nmpc.horizon = 10
        nmpc.set_initial_guess(x_guess=x0, u_guess=u0)
        nmpc.setup()

        n_steps = 1
        model.set_initial_conditions(x0=x0)
        sol = model.solution

        def path(theta):
            return np.sin(theta), np.sin(2 * theta)

        x_path = []
        y_path = []
        for t in range(1000):
            x_p, y_p = path(t / 100)
            x_path.append(x_p)
            y_path.append(y_p)

        for step in range(n_steps):
            u = nmpc.optimize(x0)
            x_pred, _, _ = nmpc.return_prediction()
            model.simulate(u=u)
            x0 = sol['x:f']

        # p = figure()
        # p.line(x=np.array(model.solution['x']).squeeze(), y=np.array(model.solution['y']).squeeze())
        # p.line(x=x_pred[0, :].squeeze(), y=x_pred[2, :].squeeze().squeeze(), line_color='green')
        # p.line(x=x_path, y=y_path, line_color='red', line_dash='dashed')
        # show(p)

    def test_ciw_v4(self):
        """
        Add two input change costs separately with inversed order
        :return:
        """

        x0 = self.x0
        u0 = self.u0
        model = self.model
        nmpc = NMPC(model)
        # nmpc.set_varying_reference(path_following=True, ub_vel=3, lb_vel=0.4)
        theta = nmpc.create_path_variable()

        nmpc.quad_stage_cost.add_states(names=['x', 'y'], weights=[10, 10],
                                        ref=ca.vertcat(ca.sin(theta), ca.sin(2 * theta)), path_following=True)
        nmpc.quad_terminal_cost.add_states(names=['x', 'y'], weights=[10, 10],
                                           ref=ca.vertcat(ca.sin(theta), ca.sin(2 * theta)), path_following=True)

        nmpc.quad_stage_cost.add_inputs_change(names=['Fy'], weights=[10])
        nmpc.quad_stage_cost.add_inputs_change(names=['Fx'], weights=[10])

        nmpc.horizon = 10
        nmpc.set_initial_guess(x_guess=x0, u_guess=u0)
        nmpc.setup()

        n_steps = 1
        model.set_initial_conditions(x0=x0)
        sol = model.solution

        def path(theta):
            return np.sin(theta), np.sin(2 * theta)

        x_path = []
        y_path = []
        for t in range(1000):
            x_p, y_p = path(t / 100)
            x_path.append(x_p)
            y_path.append(y_p)

        for step in range(n_steps):
            u = nmpc.optimize(x0)
            x_pred, _, _ = nmpc.return_prediction()
            model.simulate(u=u)
            x0 = sol['x:f']

        # p = figure()
        # p.line(x=np.array(model.solution['x']).squeeze(), y=np.array(model.solution['y']).squeeze())
        # p.line(x=x_pred[0, :].squeeze(), y=x_pred[2, :].squeeze().squeeze(), line_color='green')
        # p.line(x=x_path, y=y_path, line_color='red', line_dash='dashed')
        # show(p)


class TestStringMethod(TestCase):
    def setUp(self):
        model = Model(plot_backend='bokeh')
        # Constants
        M = 5.

        # States and algebraic variables
        xx = model.set_dynamical_states(['x', 'vx', 'y', 'vy'])
        model.set_measurements(['y_x', 'y_vx', 'y_y', 'y_vy'])
        model.set_measurement_equations([xx[0], xx[1], xx[2], xx[3]])
        x = xx[0]
        vx = xx[1]
        y = xx[2]
        vy = xx[3]
        # Inputs
        F = model.set_inputs(['Fx', 'Fy'])
        Fx = F[0]
        Fy = F[1]
        # ODEs
        dd = ca.SX.sym('dd', 4)
        dd[0] = vx
        dd[1] = Fx / M
        dd[2] = vy
        dd[3] = Fy / M

        # Algebraic equations
        # dy = h + l * ca.cos(theta) - y

        model.set_equations(ode=dd)

        # Initial conditions
        x0 = [0, 0, 0, 0]
        u0 = [0., 0.]

        # Create model and run simulation
        dt = 0.1
        model.setup(dt=dt)

        self.model = model
        self.dt = dt
        self.x0 = x0
        self.u0 = u0

        def format_figure(p):
            p.yaxis.axis_label_text_font_size = "18pt"
            p.yaxis.major_label_text_font_size = "18pt"
            p.xaxis.major_label_text_font_size = "18pt"
            p.xaxis.axis_label_text_font_size = "18pt"
            return p

        self.format_figure = format_figure

    def test_1(self):
        x0 = self.x0
        u0 = self.u0
        model = self.model
        nmpc = NMPC(model)
        # nmpc.set_varying_reference(path_following=True, ub_vel=3, lb_vel=0.4)
        theta = nmpc.create_path_variable()

        nmpc.quad_stage_cost.add_states(names=['x', 'y'], weights=[10, 10],
                                        ref=ca.vertcat(ca.sin(theta), ca.sin(2 * theta)), path_following=True)
        nmpc.quad_terminal_cost.add_states(names=['x', 'y'], weights=[10, 10],
                                           ref=ca.vertcat(ca.sin(theta), ca.sin(2 * theta)), path_following=True)
        nmpc.horizon = 10
        nmpc.set_initial_guess(x_guess=x0, u_guess=u0)
        nmpc.setup()
        print(nmpc)


class TestNewIntegration(TestCase):

    def setUp(self) -> None:

        model = Model(plot_backend='bokeh')
        # Constants
        M = 5.
        m = 1.
        l = 1.
        h = .5
        g = 9.81

        # States and algebraic variables
        x = model.set_dynamical_states(['x', 'v', 'theta', 'omega'])
        model.set_measurements(['yx', 'yv', 'ytheta', 'tomega'])
        model.set_measurement_equations([x[0], x[1], x[2], x[3]])
        # y = model.set_algebraic_variables(['y'])
        v = x[1]
        theta = x[2]
        omega = x[3]
        # Inputs
        F = model.set_inputs('F')

        # ODE
        dx = v
        dv = 1. / (M + m - m * ca.cos(theta)) * (m * g * ca.sin(theta) - m * l * ca.sin(theta) * omega ** 2 + F)
        dtheta = omega
        domega = 1. / l * (dv * ca.cos(theta) + g * ca.sin(theta))

        model.set_equations(ode=[dx, dv, dtheta, domega])

        # Initial conditions
        x0 = [2.5, 0., 0.1, 0.]
        z0 = ca.sqrt(3.) / 2.
        u0 = 0.

        # Create model and run simulation
        dt = .1
        model.setup(dt=dt)

        self.model = model
        self.dt = dt
        self.x0 = x0
        self.u0 = u0

        # may_term = (model.x[3]-ca.pi)*100*(model.x[3]-ca.pi)
        # lag_term = (model.x[3]-ca.pi)*10*(model.x[3]-ca.pi) #+ model.u*0.01*model.u

    def test_closed_loop_collocation(self):
        " Test normal nonlinear MPC for using a pendulum model. This test checks the normal problem setup with new RungeKutta class"
        x0 = self.x0
        u0 = self.u0
        model = self.model
        nmpc = NMPC(model)
        nmpc.quad_stage_cost.add_states(names=['v', 'theta'], ref=[0, 0], weights=[10, 5])
        nmpc.quad_stage_cost.add_inputs(names='F', weights=0.1)

        nmpc.horizon = 25
        nmpc.set_box_constraints(x_ub=[5, 10, 10, 10], x_lb=[-5, -10, -10, -10])
        nmpc.set_initial_guess(x_guess=x0, u_guess=u0)
        nmpc.set_nlp_options({'integration_method': 'collocation'})
        nmpc.setup()

        n_steps = 1
        model.set_initial_conditions(x0=x0)
        sol = model.solution

        for step in range(n_steps):
            u = nmpc.optimize(x0)
            model.simulate(u=u, steps=1)
            x0 = sol['x:f']

        # sol.plot(
        #     ('t', 'x'),
        #     ('t', 'v'),
        #     ('t', 'theta'),
        #     ('t', 'omega'),
        #     subplots=True,
        #     title=sol.get_names('x'),
        #     xlabel=None,
        #     legend=False)
        # nmpc.plot_mpc_prediction()

    def test_closed_loop_rk(self):
        " Test normal nonlinear MPC for using a pendulum model. This test checks the normal problem setup with new RungeKutta class"
        x0 = self.x0
        u0 = self.u0
        model = self.model
        nmpc = NMPC(model)
        nmpc.quad_stage_cost.add_states(names=['v', 'theta'], ref=[0, 0], weights=[10, 5])
        nmpc.quad_stage_cost.add_inputs(names='F', weights=0.1)
        nmpc.horizon = 25
        nmpc.set_box_constraints(x_ub=[5, 10, 10, 10], x_lb=[-5, -10, -10, -10])
        nmpc.set_initial_guess(x_guess=x0, u_guess=u0)
        nmpc.set_nlp_options({'integration_method': 'rk4'})
        nmpc.setup()

        n_steps = 1
        model.set_initial_conditions(x0=x0)
        sol = model.solution

        for step in range(n_steps):
            u = nmpc.optimize(x0)
            model.simulate(u=u, steps=1)
            x0 = sol['x:f']

        # sol.plot(
        #     ('t', 'x'),
        #     ('t', 'v'),
        #     ('t', 'theta'),
        #     ('t', 'omega'),
        #     subplots=True,
        #     title=sol.get_names('x'),
        #     xlabel=None,
        #     legend=False)
        # nmpc.plot_mpc_prediction()

    def test_closed_loop_ms(self):
        " Test normal nonlinear MPC for using a pendulum model. This test checks the normal problem setup with new RungeKutta class"
        x0 = self.x0
        u0 = self.u0
        model = self.model
        nmpc = NMPC(model)
        nmpc.quad_stage_cost.add_states(names=['v', 'theta'], ref=[0, 0], weights=[10, 5])
        nmpc.quad_stage_cost.add_inputs(names='F', weights=0.1)
        nmpc.horizon = 25
        nmpc.set_box_constraints(x_ub=[5, 10, 10, 10], x_lb=[-5, -10, -10, -10])
        nmpc.set_initial_guess(x_guess=x0, u_guess=u0)
        nmpc.set_nlp_options({'integration_method': 'idas'})
        nmpc.setup()

        n_steps = 1
        model.set_initial_conditions(x0=x0)
        sol = model.solution

        for step in range(n_steps):
            u = nmpc.optimize(x0)
            model.simulate(u=u, steps=1)
            x0 = sol['x:f']

        # sol.plot(
        #     ('t', 'x'),
        #     ('t', 'v'),
        #     ('t', 'theta'),
        #     ('t', 'omega'),
        #     subplots=True,
        #     title=sol.get_names('x'),
        #     xlabel=None,
        #     legend=False)
        # nmpc.plot_mpc_prediction()

    def test_closed_loop_disc(self):
        """
        Test normal nonlinear MPC for using a discrete pendulum model.
        This test checks the normal problem setup with new RungeKutta class
        """
        x0 = self.x0
        u0 = self.u0
        model = self.model
        model_disc = model.discretize('rk4')
        model_disc.setup(dt=self.dt)
        nmpc = NMPC(model_disc)
        nmpc.quad_stage_cost.add_states(names=['v', 'theta'], ref=[0, 0], weights=[10, 5])
        nmpc.quad_stage_cost.add_inputs(names='F', weights=0.1)
        nmpc.horizon = 25
        nmpc.set_box_constraints(x_ub=[5, 10, 10, 10], x_lb=[-5, -10, -10, -10])
        nmpc.set_initial_guess(x_guess=x0, u_guess=u0)
        nmpc.setup()

        n_steps = 1
        model.set_initial_conditions(x0=x0)
        sol = model.solution

        for step in range(n_steps):
            u = nmpc.optimize(x0)
            model.simulate(u=u, steps=1)
            x0 = sol['x:f']

        # sol.plot(
        #     ('t', 'x'),
        #     ('t', 'v'),
        #     ('t', 'theta'),
        #     ('t', 'omega'),
        #     subplots=True,
        #     title=sol.get_names('x'),
        #     xlabel=None,
        #     legend=False)
        # nmpc.plot_mpc_prediction()


class TestTimeVariantSys(TestCase):
    def setUp(self) -> None:
        model = Model(plot_backend='bokeh')
        # Constants
        k = 0.1

        # States (internal temperature)
        T = model.set_dynamical_states(['T'])
        # Inputs
        F = model.set_inputs('F')
        # time
        t = model.time
        # Extenral temperature
        T_ex = ca.sin(2 * ca.pi * t / (3600 * 24)) * 5
        # ODE
        dT = F - k * (T - T_ex)

        model.set_equations(ode=[dT])

        # Initial conditions
        x0 = 25
        u0 = 0.

        # Create model and run simulation
        dt = 900  # seconds
        model.setup(dt=dt)

        self.model = model
        self.dt = dt
        self.x0 = x0
        self.u0 = u0

        # may_term = (model.x[3]-ca.pi)*100*(model.x[3]-ca.pi)
        # lag_term = (model.x[3]-ca.pi)*10*(model.x[3]-ca.pi) #+ model.u*0.01*model.u

    def test_closed_loop_b(self):
        " Test normal nonlinear MPC for using a pendulum model. This test checks the normal problem setup"
        x0 = self.x0
        u0 = self.u0
        t0 = 0
        model = self.model
        nmpc = NMPC(model)
        nmpc.quad_stage_cost.add_states(names=['T'], ref=[25], weights=[10])
        # nmpc.quad_stage_cost.add_inputs(names='F', weights=0.1)
        nmpc.horizon = 100
        nmpc.set_box_constraints(x_ub=[30], x_lb=[20])
        nmpc.set_initial_guess(x_guess=x0, u_guess=u0)
        nmpc.initial_time = t0
        nmpc.setup()

        n_steps = 1
        model.set_initial_conditions(x0=x0, t0=t0)
        sol = model.solution

        for step in range(n_steps):
            u = nmpc.optimize(x0)
            model.simulate(u=u, steps=1)
            x0 = sol['x:f']

            # nmpc.plot_mpc_prediction()
        # sol.plot(output_file="results/test.html")


class TestMinimumTime(TestCase):

    def setUp(self) -> None:
        model = Model(plot_backend='bokeh')
        # States and algebraic variables
        x = model.set_dynamical_states(['p', 'v'])
        v = x[1]
        # Inputs
        u = model.set_inputs('u')

        # ODE
        dx = v
        dv = u - v

        model.set_equations(ode=[dx, dv])

        # Initial conditions
        x0 = [0, 0]
        u0 = 0.

        # Create model and run simulation
        dt = .1
        model.setup(dt=dt)

        self.model = model
        self.dt = dt
        self.x0 = x0
        self.u0 = u0

        # may_term = (model.x[3]-ca.pi)*100*(model.x[3]-ca.pi)
        # lag_term = (model.x[3]-ca.pi)*10*(model.x[3]-ca.pi) #+ model.u*0.01*model.u

    def test_race_car(self):
        " Test normal nonlinear MPC for using a pendulum model. This test checks the normal problem setup"
        x0 = self.x0
        u0 = self.u0
        model = self.model
        nmpc = NMPC(model)
        nmpc.minimize_final_time(weight=1)
        nmpc.horizon = 100
        nmpc.set_initial_guess(x_guess=x0, u_guess=u0)
        nmpc.set_terminal_constraints(terminal_constraint=model.x[0], lb=1, ub=1)
        nmpc.set_stage_constraints(stage_constraint=model.x[1] - (1 - ca.sin(2 * ca.pi * model.x[0]) / 2), lb=-ca.inf,
                                   ub=0)
        nmpc.set_box_constraints(u_lb=0, u_ub=1)
        nmpc.setup()

        n_steps = 1
        model.set_initial_conditions(x0=x0)
        sol = model.solution

        u = nmpc.optimize(x0)

        # nmpc.plot_prediction()


class TestStats(TestCase):
    def setUp(self) -> None:
        model = Model(plot_backend='bokeh')
        # Constants
        M = 5.
        m = 1.
        l = 1.
        h = .5
        g = 9.81

        # States and algebraic variables
        x = model.set_dynamical_states(['x', 'v', 'theta', 'omega'])
        model.set_measurements(['yx', 'yv', 'ytheta', 'tomega'])
        model.set_measurement_equations([x[0], x[1], x[2], x[3]])
        # y = model.set_algebraic_variables(['y'])
        v = x[1]
        theta = x[2]
        omega = x[3]
        # Inputs
        F = model.set_inputs('F')

        # ODE
        dx = v
        dv = 1. / (M + m - m * ca.cos(theta)) * (m * g * ca.sin(theta) - m * l * ca.sin(theta) * omega ** 2 + F)
        dtheta = omega
        domega = 1. / l * (dv * ca.cos(theta) + g * ca.sin(theta))

        model.set_equations(ode=[dx, dv, dtheta, domega])

        # Initial conditions
        x0 = [2.5, 0., 0.1, 0.]
        z0 = ca.sqrt(3.) / 2.
        u0 = 0.

        # Create model and run simulation
        dt = .1
        model.setup(dt=dt)

        self.model = model
        self.dt = dt
        self.x0 = x0
        self.u0 = u0

        # may_term = (model.x[3]-ca.pi)*100*(model.x[3]-ca.pi)
        # lag_term = (model.x[3]-ca.pi)*10*(model.x[3]-ca.pi) #+ model.u*0.01*model.u

    def test_closed_loop_b(self):
        " Test normal nonlinear MPC for using a pendulum model. This test checks the normal problem setup"
        x0 = self.x0
        u0 = self.u0
        model = self.model
        nmpc = NMPC(model, stats=True)
        nmpc.quad_stage_cost.add_states(names=['v', 'theta'], ref=[0, 0], weights=[10, 5])
        nmpc.quad_stage_cost.add_inputs(names='F', weights=0.1)
        nmpc.horizon = 25
        nmpc.set_box_constraints(x_ub=[5, 10, 10, 10], x_lb=[-5, -10, -10, -10])
        nmpc.set_initial_guess(x_guess=x0, u_guess=u0)
        nmpc.setup()

        n_steps = 1
        model.set_initial_conditions(x0=x0)
        sol = model.solution

        for step in range(n_steps):
            u = nmpc.optimize(x0)
            model.simulate(u=u, steps=1)
            x0 = sol['x:f']
        #
        # nmpc.solution.plot(
        #     ('niterations', 'extime'),
        #     ('niterations', 'solvstatus'),
        #     title='Execution Time',
        #     subplot=True,
        #     legend=False)
        #
        # nmpc.solution.to_mat('t', 'x', 'extime', 'niterations', 'solvstatus', file_name='results/test.mat')


class TestTimeVaryingWeights(TestCase):
    def setUp(self) -> None:
        model = Model(plot_backend='bokeh')
        # Constants
        M = 5.
        m = 1.
        l = 1.
        h = .5
        g = 9.81

        # States and algebraic variables
        x = model.set_dynamical_states(['x', 'v', 'theta', 'omega'])
        model.set_measurements(['yx', 'yv', 'ytheta', 'tomega'])
        model.set_parameters(['weight_v', 'weight_theta'])
        model.set_measurement_equations([x[0], x[1], x[2], x[3]])
        # y = model.set_algebraic_variables(['y'])
        v = x[1]
        theta = x[2]
        omega = x[3]
        # Inputs
        F = model.set_inputs('F')

        # ODE
        dx = v
        dv = 1. / (M + m - m * ca.cos(theta)) * (m * g * ca.sin(theta) - m * l * ca.sin(theta) * omega ** 2 + F)
        dtheta = omega
        domega = 1. / l * (dv * ca.cos(theta) + g * ca.sin(theta))

        model.set_equations(ode=[dx, dv, dtheta, domega])

        # Initial conditions
        x0 = [2.5, 0., 0.1, 0.]
        z0 = ca.sqrt(3.) / 2.
        u0 = 0.

        # Create model and run simulation
        dt = .1
        model.setup(dt=dt)

        self.model = model
        self.dt = dt
        self.x0 = x0
        self.u0 = u0

    def test_time_varying_weights(self):
        x0 = self.x0
        u0 = self.u0
        model = self.model
        nmpc = NMPC(model, stats=True)
        nmpc.stage_cost.cost = model.x[1] * model.p[0] * model.x[1] + model.x[2] * model.p[1] * model.x[2]
        # nmpc.stage_cost.add_states(names=['v', 'theta'], ref=[0, 0], weights=[10, 5])
        nmpc.quad_stage_cost.add_inputs(names='F', weights=0.1)
        nmpc.horizon = 25
        nmpc.set_box_constraints(x_ub=[5, 10, 10, 10], x_lb=[-5, -10, -10, -10])
        nmpc.set_initial_guess(x_guess=x0, u_guess=u0)
        nmpc.setup()

        n_steps = 1
        model.set_initial_conditions(x0=x0)
        sol = model.solution

        for step in range(n_steps):
            u = nmpc.optimize(x0, cp=[10, 5])
            model.simulate(u=u, p=[0, 0])
            x0 = sol['x:f']
        #
        # model.solution.plot()
        #
        # nmpc.solution.to_mat('t', 'x', 'extime', 'niterations', 'solvstatus', file_name='results/test.mat')
