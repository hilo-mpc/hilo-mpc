import unittest

import casadi as ca
import numpy as np

from hilo_mpc import SimpleControlLoop, Model, OCP


def _make_pendulum_model(dt=0.1):
    """Helper: cart-pendulum model with two inputs"""
    model = Model(plot_backend='bokeh')
    M, m, l, g = 5., 1., 1., 9.81
    x = model.set_dynamical_states('x', 'v', 'theta', 'omega')
    model.set_measurements('yx', 'yv', 'ytheta', 'tomega')
    model.set_measurement_equations([x[0], x[1], x[2], x[3]])
    v = x[1]
    theta = x[2]
    omega = x[3]
    F = model.set_inputs('F1', 'F2')
    dx = v
    dv = 1. / (M + m - m * ca.cos(theta)) * (m * g * ca.sin(theta) - m * l * ca.sin(theta) * omega ** 2 + F[0])
    dtheta = omega
    domega = 1. / l * (dv * ca.cos(theta) + g * ca.sin(theta)) + F[1]
    model.set_equations(ode=[dx, dv, dtheta, domega])
    model.setup(dt=dt)
    return model


class TestOCPSetup(unittest.TestCase):
    """"""
    def setUp(self) -> None:
        self.model = _make_pendulum_model()
        self.x0 = [2.5, 0., 0.1, 0.]
        self.u0 = [0., 0.]

    def test_ocp_type(self):
        """OCP reports type 'OCP'"""
        ocp = OCP(self.model)
        self.assertEqual(ocp.type, 'OCP')

    def test_ocp_initialization(self):
        """OCP can be instantiated with a model"""
        ocp = OCP(self.model)
        self.assertIsNotNone(ocp)

    def test_ocp_horizon(self):
        """OCP stores horizon length"""
        ocp = OCP(self.model)
        ocp.horizon = 20
        self.assertEqual(ocp.horizon, 20)

    def test_ocp_setup(self):
        """OCP sets up without error when horizon and cost are provided"""
        ocp = OCP(self.model)
        ocp.quad_stage_cost.add_states(names=['v', 'theta'], ref=[0, 0], weights=[10, 5])
        ocp.quad_stage_cost.add_inputs(names=['F1', 'F2'], weights=[0.1, 0.1])
        ocp.horizon = 10
        ocp.set_initial_guess(x_guess=self.x0, u_guess=self.u0)
        ocp.setup()
        self.assertTrue(ocp.is_setup())

    def test_ocp_with_box_constraints(self):
        """OCP setup succeeds with state and input box constraints"""
        ocp = OCP(self.model)
        ocp.quad_stage_cost.add_states(names=['v', 'theta'], ref=[0, 0], weights=[10, 5])
        ocp.quad_stage_cost.add_inputs(names=['F1', 'F2'], weights=[0.1, 0.1])
        ocp.horizon = 10
        ocp.set_box_constraints(x_ub=[5, 10, 10, 10], x_lb=[-5, -10, -10, -10])
        ocp.set_initial_guess(x_guess=self.x0, u_guess=self.u0)
        ocp.setup()
        self.assertTrue(ocp.is_setup())

    def test_ocp_terminal_cost(self):
        """OCP sets up with terminal cost without error"""
        ocp = OCP(self.model)
        ocp.quad_stage_cost.add_states(names=['v', 'theta'], ref=[0, 0], weights=[10, 5])
        ocp.quad_terminal_cost.add_states(names=['v', 'theta'], ref=[0, 0], weights=[10, 5])
        ocp.quad_stage_cost.add_inputs(names=['F1', 'F2'], weights=[0.1, 0.1])
        ocp.horizon = 10
        ocp.set_initial_guess(x_guess=self.x0, u_guess=self.u0)
        ocp.setup()
        self.assertTrue(ocp.is_setup())


class TestOCPSolve(unittest.TestCase):
    """"""
    def setUp(self) -> None:
        self.model = _make_pendulum_model()
        self.x0 = [2.5, 0., 0.1, 0.]
        self.u0 = [0., 0.]

    def test_ocp(self):
        """OCP runs full open-loop optimal trajectory via SimpleControlLoop"""
        model = self.model
        x0 = self.x0
        u0 = self.u0
        ocp = OCP(model)
        ocp.quad_stage_cost.add_states(names=['v', 'theta'], ref=[0, 0], weights=[10, 5])
        ocp.quad_stage_cost.add_inputs(names=['F1', 'F2'], weights=[0.1, 0.1])
        ocp.horizon = 25
        ocp.set_box_constraints(x_ub=[5, 10, 10, 10], x_lb=[-5, -10, -10, -10])
        ocp.set_initial_guess(x_guess=x0, u_guess=u0)
        ocp.setup()

        n_steps = 25
        model.set_initial_conditions(x0=x0)
        scl = SimpleControlLoop(model, ocp)
        scl.run(steps=n_steps)

    def test_ocp_optimize_returns_input(self):
        """OCP.optimize() returns an input vector of correct size"""
        model = self.model
        x0 = self.x0
        u0 = self.u0
        ocp = OCP(model)
        ocp.quad_stage_cost.add_states(names=['v', 'theta'], ref=[0, 0], weights=[10, 5])
        ocp.quad_stage_cost.add_inputs(names=['F1', 'F2'], weights=[0.1, 0.1])
        ocp.horizon = 10
        ocp.set_initial_guess(x_guess=x0, u_guess=u0)
        ocp.setup()
        u = ocp.optimize(x0=x0)
        # u should have shape (n_u,) or (n_u, 1)
        self.assertEqual(len(np.array(u).flatten()), 2)  # 2 inputs (F1, F2)


if __name__ == '__main__':
    unittest.main()
