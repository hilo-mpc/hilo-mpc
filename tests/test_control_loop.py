import unittest

import casadi as ca
import numpy as np

from hilo_mpc import Model, NMPC, LMPC, LQR, KF, SimpleControlLoop


def _make_discrete_integrator(dt=0.5):
    """Helper: discrete-time double integrator via A/B matrices"""
    model = Model(plot_backend='bokeh', discrete=True)
    model.A = np.array([[1., dt], [0., 1.]])
    model.B = np.array([[dt ** 2 / 2.], [dt]])
    model.setup(dt=dt)
    return model


def _make_pendulum(dt=0.1):
    """Helper: continuous cart-pendulum system"""
    model = Model(plot_backend='bokeh')
    M, m, l, g = 5., 1., 1., 9.81
    x = model.set_dynamical_states('x', 'v', 'theta', 'omega')
    model.set_measurements('yx', 'yv', 'ytheta', 'yomega')
    model.set_measurement_equations([x[0], x[1], x[2], x[3]])
    v = x[1]
    theta = x[2]
    omega = x[3]
    F = model.set_inputs('F')
    dx = v
    dv = 1. / (M + m - m * ca.cos(theta)) * (m * g * ca.sin(theta) - m * l * ca.sin(theta) * omega ** 2 + F)
    dtheta = omega
    domega = 1. / l * (dv * ca.cos(theta) + g * ca.sin(theta))
    model.set_equations(ode=[dx, dv, dtheta, domega])
    model.setup(dt=dt)
    return model


class TestSimpleControlLoopSetup(unittest.TestCase):
    """Tests for SimpleControlLoop instantiation"""

    def test_setup_with_lmpc(self):
        """SimpleControlLoop can be created with a discrete model and LMPC"""
        model = _make_discrete_integrator()
        mpc = LMPC(model)
        mpc.Q = np.eye(2)
        mpc.R = np.eye(1)
        mpc.horizon = 5
        mpc.setup()
        model.set_initial_conditions(x0=[1., 0.])
        scl = SimpleControlLoop(model, mpc)
        self.assertIsNotNone(scl)

    def test_setup_with_nmpc(self):
        """SimpleControlLoop can be created with a continuous model and NMPC"""
        model = _make_pendulum()
        x0 = [2.5, 0., 0.1, 0.]
        nmpc = NMPC(model)
        nmpc.quad_stage_cost.add_states(names=['v', 'theta'], ref=[0, 0], weights=[10, 5])
        nmpc.quad_stage_cost.add_inputs(names='F', weights=0.1)
        nmpc.horizon = 10
        nmpc.set_box_constraints(x_ub=[5, 10, 10, 10], x_lb=[-5, -10, -10, -10])
        nmpc.set_initial_guess(x_guess=x0, u_guess=[0.])
        nmpc.setup()
        model.set_initial_conditions(x0=x0)
        scl = SimpleControlLoop(model, nmpc)
        self.assertIsNotNone(scl)

    def test_auto_setup_on_run(self):
        """SimpleControlLoop calls setup on controller if not already set up"""
        model = _make_discrete_integrator()
        mpc = LMPC(model)
        mpc.Q = np.eye(2)
        mpc.R = np.eye(1)
        mpc.horizon = 5
        # Do NOT call mpc.setup() — SimpleControlLoop should handle it
        model.set_initial_conditions(x0=[1., 0.])
        scl = SimpleControlLoop(model, mpc)
        self.assertIsNotNone(scl)


class TestSimpleControlLoopRun(unittest.TestCase):
    """Tests for running SimpleControlLoop"""

    def test_lmpc_run_reduces_state_norm(self):
        """LMPC closed loop drives double integrator toward origin"""
        dt = 0.5
        model = _make_discrete_integrator(dt=dt)
        x0 = [2., 1.]
        mpc = LMPC(model)
        mpc.Q = np.eye(2)
        mpc.R = np.eye(1)
        mpc.horizon = 10
        mpc.set_box_constraints(x_lb=[-10, -10], x_ub=[10, 10], u_lb=[-5], u_ub=[5])
        mpc.setup()
        model.set_initial_conditions(x0=x0)
        scl = SimpleControlLoop(model, mpc)
        scl.run(steps=30)
        final_state = np.array(model.solution['x:f']).flatten()
        # State should be much closer to 0 after 30 steps
        self.assertLess(np.linalg.norm(final_state), np.linalg.norm(x0))

    def test_nmpc_run_completes(self):
        """NMPC closed loop runs for N steps without error"""
        model = _make_pendulum()
        x0 = [2.5, 0., 0.1, 0.]
        nmpc = NMPC(model)
        nmpc.quad_stage_cost.add_states(names=['v', 'theta'], ref=[0, 0], weights=[10, 5])
        nmpc.quad_stage_cost.add_inputs(names='F', weights=0.1)
        nmpc.horizon = 10
        nmpc.set_box_constraints(x_ub=[5, 10, 10, 10], x_lb=[-5, -10, -10, -10])
        nmpc.set_initial_guess(x_guess=x0, u_guess=[0.])
        nmpc.setup()
        model.set_initial_conditions(x0=x0)
        scl = SimpleControlLoop(model, nmpc)
        scl.run(steps=5)

    def test_solution_stored_after_run(self):
        """After run, model solution contains simulation results"""
        model = _make_discrete_integrator()
        x0 = [1., 0.]
        mpc = LMPC(model)
        mpc.Q = np.eye(2)
        mpc.R = np.eye(1)
        mpc.horizon = 5
        mpc.setup()
        model.set_initial_conditions(x0=x0)
        scl = SimpleControlLoop(model, mpc)
        scl.run(steps=5)
        final = model.solution['x:f']
        self.assertIsNotNone(final)

    def test_lmpc_run_multiple_steps_solution_size(self):
        """After N steps, solution contains N+1 time points"""
        model = _make_discrete_integrator()
        x0 = [1., 0.]
        n_steps = 10
        mpc = LMPC(model)
        mpc.Q = np.eye(2)
        mpc.R = np.eye(1)
        mpc.horizon = 5
        mpc.setup()
        model.set_initial_conditions(x0=x0)
        scl = SimpleControlLoop(model, mpc)
        scl.run(steps=n_steps)
        # Solution x should have n_steps+1 columns (including initial)
        x_solution = model.solution.get_by_id('x')
        self.assertEqual(x_solution.shape[1], n_steps + 1)


class TestSimpleControlLoopWithObserver(unittest.TestCase):
    """Tests for SimpleControlLoop with a Kalman Filter observer"""

    def _make_model_with_measurements(self, dt=0.5):
        """Create a model with explicit symbolic states for KF"""
        model = Model(plot_backend='bokeh', discrete=True)
        x = model.set_dynamical_states('x_0', 'x_1')
        u = model.set_inputs('u')
        model.set_measurements('y')
        model.set_measurement_equations([x[0]])
        model.set_dynamical_equations([x[0] + dt * x[1] + 0.5 * dt ** 2 * u[0],
                                       x[1] + dt * u[0]])
        model.setup(dt=dt)
        return model

    def test_setup_with_kf_observer(self):
        """SimpleControlLoop can be set up with a Kalman Filter observer"""
        model = self._make_model_with_measurements()
        model.set_initial_conditions(x0=[1., 0.])

        mpc = LMPC(model)
        mpc.Q = np.eye(2)
        mpc.R = np.eye(1)
        mpc.horizon = 5
        mpc.setup()

        kf = KF(model)
        kf.setup()  # Must call setup() before setting Q and R (dimensions only known after setup)
        kf.Q = 0.01 * np.eye(2)
        kf.R = np.eye(1)

        scl = SimpleControlLoop(model, mpc, kf)
        self.assertIsNotNone(scl)


if __name__ == '__main__':
    unittest.main()
