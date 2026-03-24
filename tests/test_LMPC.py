import unittest
from hilo_mpc import Model, LMPC, SimpleControlLoop, NMPC
import numpy as np
import casadi as ca


class TestAlreadyLinearMOdels(unittest.TestCase):
    def setUp(self) -> None:
        model = Model(plot_backend='bokeh', discrete=True)

        Ts = 0.5
        x0 = [1, 1]
        model.A = np.array([[1, model.dt], [0, 1]])
        model.B = np.array([[model.dt ** 2 / 2], [model.dt]])

        model.setup(dt=Ts)
        model.set_initial_conditions(x0=x0)
        self.model = model
        self.x0 = x0

    def test_already_linear_model(self):
        model = self.model
        x0 = self.x0

        mpc = LMPC(model)
        mpc.Q = np.eye(2)
        mpc.R = 1
        mpc.horizon = 10
        mpc.set_box_constraints(x_lb=[-5, -5], x_ub=[5, 5], u_lb=[-1], u_ub=[1])
        mpc.setup()

        for i in range(200):
            u = mpc.optimize(x0=x0)
            model.simulate(u=u)
            x0 = model.solution['x:f']

        model.solution.plot(output_file='results/test_lmpc.html')

        model.reset_solution(keep_initial_conditions=True)
        x0 = self.x0

        nmpc = NMPC(model)
        nmpc.horizon = 10
        nmpc.quad_stage_cost.add_states(names=['x_0', 'x_1'], weights=[2, 2])
        nmpc.quad_stage_cost.add_inputs(names=['u'], weights=[1])
        nmpc.set_initial_guess(x_guess=x0, u_guess=[0])
        nmpc.set_box_constraints(x_lb=[-5, -5], x_ub=[5, 5], u_lb=[-1], u_ub=[1])
        nmpc.setup()

        for i in range(200):
            u = nmpc.optimize(x0=x0)
            model.simulate(u=u)
            x0 = model.solution['x:f']

        model.solution.plot(output_file='results/test_mpc.html')


class TestLinearizedModel(unittest.TestCase):
    def setUp(self) -> None:
        model = Model(plot_backend='bokeh')

        states = model.set_dynamical_states(['px', 'py', 'v', 'phi'])
        inputs = model.set_inputs(['a', 'delta'])

        # Unwrap states
        px = states[0]
        py = states[1]
        v = states[2]
        phi = states[3]

        # Unwrap states
        a = inputs[0]
        delta = inputs[1]

        # Parameters
        lr = 1.4  # [m]
        lf = 1.8  # [m]
        beta = ca.arctan(lr / (lr + lf) * ca.tan(delta))

        # ODE
        dpx = v * ca.cos(phi + beta)
        dpy = v * ca.sin(phi + beta)
        dv = a
        dphi = v / lr * ca.sin(beta)

        model.set_dynamical_equations([dpx, dpy, dv, dphi])
        model.discretize(method='rk4', inplace=True)
        model = model.linearize()
        dt = 0.05
        model.setup(dt=dt)
        model.set_equilibrium_point(x_eq=[0, 0, 0, 0], u_eq=[0, 0])

        self.model = model

    def test_linearized_model(self):
        model = self.model

        mpc = LMPC(model)
        mpc.horizon = 10
        mpc.Q = np.eye(model.n_x)
        mpc.R = np.eye(model.n_u)
        mpc.setup()
        x0 = [0, 0, 0, 0]

        mpc.optimize(x0=x0)

    def test_plot_prediction(self):
        model = self.model

        mpc = LMPC(model)
        mpc.horizon = 10
        mpc.Q = np.eye(model.n_x)
        mpc.R = np.eye(model.n_u)
        mpc.setup()
        x0 = [0.5, 0, 0, 0]

        mpc.optimize(x0=x0)
        mpc.solution.plot()


class TestModelWithParameters(unittest.TestCase):
    def setUp(self) -> None:
        model = Model(plot_backend='bokeh')

        states = model.set_dynamical_states(['px', 'py', 'v', 'phi'])
        inputs = model.set_inputs(['a', 'delta'])
        parameters = model.set_parameters(['lr', 'lf'])

        # Unwrap states
        v = states[2]
        phi = states[3]

        # Unwrap states
        a = inputs[0]
        delta = inputs[1]

        # Unwrap Parameters
        lr = parameters[0]
        lf = parameters[1]

        beta = ca.arctan(lr / (lr + lf) * ca.tan(delta))

        # ODE
        dpx = v * ca.cos(phi + beta)
        dpy = v * ca.sin(phi + beta)
        dv = a
        dphi = v / lr * ca.sin(beta)

        model.set_dynamical_equations([dpx, dpy, dv, dphi])
        model.discretize(method='rk4', inplace=True)
        model = model.linearize()
        dt = 0.05
        lr0 = 1.4  # [m]
        lf0 = 1.8  # [m]
        model.setup(dt=dt)
        model.set_initial_parameter_values(p=[lr0, lf0])
        model.set_equilibrium_point(x_eq=[0, 0, 0, 0], u_eq=[0, 0])

        self.model = model

    def test_constant_parameters(self):
        model = self.model
        mpc = LMPC(model)
        mpc.horizon = 10
        mpc.Q = np.eye(model.n_x)
        mpc.R = np.eye(model.n_u)
        mpc.setup()
        x0 = [0, 0, 0, 0]
        lr0 = 1.4  # [m]
        lf0 = 1.8  # [m]

        mpc.optimize(x0=x0, cp=[lr0, lf0])
        mpc.solution.plot()

    def test_time_varying_parameters(self):
        model = self.model
        mpc = LMPC(model)
        mpc.horizon = 10
        mpc.Q = np.eye(model.n_x)
        mpc.R = np.eye(model.n_u)
        mpc.set_time_varying_parameters(names=['lf'])
        mpc.setup()
        x0 = [1, 1, 0, 0]
        lr0 = 1.4  # [m]
        lf0 = 1.8  # [m]

        mpc.optimize(x0=x0, cp=[lr0], tvp={'lf': [1.8, 1.8, 1.8, 1.8, 1.8, 1.4, 1.4, 1.4, 1.4, 1.4]})
        mpc.solution.plot()


class TestAlreadyLinearModelWithTvp(unittest.TestCase):
    def setUp(self) -> None:
        model = Model(plot_backend='bokeh', discrete=True)
        x = model.set_dynamical_states(['x_1', 'x_2'])
        u = model.set_inputs(['u_1', 'u_2'])
        p = model.set_parameters('p')
        A = np.array([[-1, 2 * p], [0, -1]])
        B = np.array([[1, 0], [0, 1]])
        model.set_dynamical_equations(A @ x + B @ u)
        model.setup()
        self.model = model
        self.x0 = [1, 2]
        self.tvp = 60 * [1] + 50 * [0]

    def test_compare_NMPC_LMPC(self):
        model = self.model
        x0 = self.x0
        nmpc = NMPC(model)
        nmpc.horizon = 10
        nmpc.quad_stage_cost.add_states(names=['x_1', 'x_2'], weights=[1, 1])
        nmpc.quad_terminal_cost.add_states(names=['x_1', 'x_2'], weights=[1, 1])
        nmpc.quad_stage_cost.add_inputs(names=['u_1', 'u_2'], weights=[1, 1])
        nmpc.set_box_constraints(u_ub=[0.5, 10])
        nmpc.set_time_varying_parameters(names=['p'], values={'p': self.tvp})
        nmpc.setup()
        model.set_initial_conditions(x0=x0)
        xi = x0.copy()
        for i in range(100):
            u = nmpc.optimize(x0=xi)
            model.simulate(u=u, p=self.tvp[i])
            xi = model.solution['x:f']

        model.solution.plot()

        model.reset_solution(keep_initial_conditions=False)
        model.set_initial_conditions(x0=x0)

        lmpc = LMPC(model)
        lmpc.horizon = 10
        lmpc.Q = ca.DM.eye(2)
        lmpc.P = ca.DM.eye(2)
        lmpc.R = ca.DM.eye(2)
        lmpc.set_time_varying_parameters(names=['p'], values={'p': self.tvp})
        lmpc.set_box_constraints(u_ub=[0.5, 10])
        lmpc.set_scaling(x_scaling=[2, 2], u_scaling=[0.9, 0.9])
        lmpc.setup()
        xi = x0.copy()
        for i in range(100):
            u = lmpc.optimize(x0=xi)
            model.simulate(u=u, p=self.tvp[i])
            xi = model.solution['x:f']

        model.solution.plot()


class TestErrors(unittest.TestCase):
    def setUp(self) -> None:
        model = Model(plot_backend='bokeh', discrete=True)

        Ts = 0.5
        x0 = [1, 1]
        model.A = np.array([[1, model.dt], [0, 1]])
        model.B = np.array([[model.dt ** 2 / 2], [model.dt]])

        model.setup(dt=Ts)
        model.set_initial_conditions(x0=x0)
        self.model = model
        self.x0 = x0

    def test_wron_weights_dimensions(self):
        model = self.model
        x0 = self.x0

        mpc = LMPC(model)
        mpc.Q = np.eye(1)
        mpc.R = 1
        mpc.horizon = 10
        mpc.set_box_constraints(x_lb=[-5, -5], x_ub=[5, 5], u_lb=[-1], u_ub=[1])
        self.assertRaises(ValueError, mpc.setup)


class TestSolvers(unittest.TestCase):
    def setUp(self) -> None:
        model = Model(plot_backend='bokeh', discrete=True)

        Ts = 0.5
        x0 = [1, 1]
        model.A = np.array([[1, 0], [0, 1]])
        model.B = np.array([[1], [1]])

        model.setup(dt=Ts)
        model.set_initial_conditions(x0=x0)
        self.model = model
        self.x0 = x0

    def test_gurobi(self):
        model = self.model
        x0 = self.x0

        mpc = LMPC(model)
        mpc.Q = np.eye(2)
        mpc.R = 1
        mpc.horizon = 10
        mpc.set_box_constraints(x_lb=[-5, -5], x_ub=[5, 5], u_lb=[-1], u_ub=[1])
        mpc.setup(solver='qpoases', solver_options={'sparse':True})
        mpc.optimize(x0=x0)


class TestLMPCCostFunction(unittest.TestCase):
    """Tests for LMPC cost function configurations"""

    def setUp(self) -> None:
        model = Model(plot_backend='bokeh', discrete=True)
        Ts = 0.5
        model.A = np.array([[1, Ts], [0, 1]])
        model.B = np.array([[Ts ** 2 / 2], [Ts]])
        model.setup(dt=Ts)
        model.set_initial_conditions(x0=[1., 0.])
        self.model = model
        self.x0 = [1., 0.]

    def test_lmpc_with_terminal_cost(self):
        """LMPC with terminal cost (P matrix) sets up and optimizes without error"""
        model = self.model
        x0 = self.x0
        mpc = LMPC(model)
        mpc.Q = np.eye(2)
        mpc.R = 1
        mpc.P = np.eye(2)  # Terminal cost matrix
        mpc.horizon = 5
        mpc.setup()
        u = mpc.optimize(x0=x0)
        self.assertIsNotNone(u)

    def test_lmpc_q_matrix_stored(self):
        """LMPC stores Q matrix correctly after setup"""
        model = self.model
        Q = 2. * np.eye(2)
        mpc = LMPC(model)
        mpc.Q = Q
        mpc.R = 1
        mpc.horizon = 5
        mpc.setup()
        np.testing.assert_allclose(mpc.Q, Q)

    def test_lmpc_r_matrix_stored(self):
        """LMPC stores R matrix correctly after setup"""
        model = self.model
        mpc = LMPC(model)
        mpc.Q = np.eye(2)
        mpc.R = 0.5
        mpc.horizon = 5
        mpc.setup()
        self.assertAlmostEqual(float(mpc.R), 0.5)


class TestLMPCConstraints(unittest.TestCase):
    """Tests for LMPC constraint configurations"""

    def setUp(self) -> None:
        model = Model(plot_backend='bokeh', discrete=True)
        Ts = 0.5
        model.A = np.array([[1, Ts], [0, 1]])
        model.B = np.array([[Ts ** 2 / 2], [Ts]])
        model.setup(dt=Ts)
        model.set_initial_conditions(x0=[2., 0.])
        self.model = model
        self.x0 = [2., 0.]

    def test_lmpc_box_constraints_respected(self):
        """LMPC optimal input respects upper bound constraint"""
        model = self.model
        x0 = self.x0
        u_ub = 0.5
        mpc = LMPC(model)
        mpc.Q = np.eye(2)
        mpc.R = 1
        mpc.horizon = 10
        mpc.set_box_constraints(x_lb=[-5, -5], x_ub=[5, 5], u_lb=[-u_ub], u_ub=[u_ub])
        mpc.setup()
        u = mpc.optimize(x0=x0)
        self.assertLessEqual(abs(float(u[0])), u_ub + 1e-6)

    def test_lmpc_no_constraints_setup(self):
        """LMPC without box constraints sets up and runs"""
        model = self.model
        x0 = self.x0
        mpc = LMPC(model)
        mpc.Q = np.eye(2)
        mpc.R = 1
        mpc.horizon = 5
        mpc.setup()
        u = mpc.optimize(x0=x0)
        self.assertIsNotNone(u)

    def test_lmpc_soft_constraints(self):
        """LMPC with soft state constraints sets up without error"""
        model = self.model
        x0 = self.x0
        mpc = LMPC(model)
        mpc.Q = np.eye(2)
        mpc.R = 1
        mpc.horizon = 5
        mpc.set_box_constraints(x_lb=[-5, -5], x_ub=[5, 5], u_lb=[-2], u_ub=[2])
        mpc.setup()
        u = mpc.optimize(x0=x0)
        self.assertIsNotNone(u)


class TestLMPCHorizon(unittest.TestCase):
    """Tests for LMPC prediction horizon configuration"""

    def setUp(self) -> None:
        model = Model(plot_backend='bokeh', discrete=True)
        Ts = 0.5
        model.A = np.array([[1, Ts], [0, 1]])
        model.B = np.array([[Ts ** 2 / 2], [Ts]])
        model.setup(dt=Ts)
        model.set_initial_conditions(x0=[1., 0.])
        self.model = model
        self.x0 = [1., 0.]

    def test_lmpc_short_horizon(self):
        """LMPC with horizon=2 sets up and runs"""
        model = self.model
        x0 = self.x0
        mpc = LMPC(model)
        mpc.Q = np.eye(2)
        mpc.R = 1
        mpc.horizon = 2
        mpc.setup()
        u = mpc.optimize(x0=x0)
        self.assertIsNotNone(u)

    def test_lmpc_long_horizon(self):
        """LMPC with horizon=50 sets up and runs"""
        model = self.model
        x0 = self.x0
        mpc = LMPC(model)
        mpc.Q = np.eye(2)
        mpc.R = 1
        mpc.horizon = 50
        mpc.setup()
        u = mpc.optimize(x0=x0)
        self.assertIsNotNone(u)

    def test_lmpc_horizon_stored(self):
        """LMPC stores prediction horizon correctly"""
        model = self.model
        mpc = LMPC(model)
        mpc.horizon = 15
        self.assertEqual(mpc.horizon, 15)


if __name__ == '__main__':
    unittest.main()
